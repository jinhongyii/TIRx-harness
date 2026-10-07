//! Dense matmul numeric cores and the pluggable `MatmulBackend`.
//!
//! Legacy sources:
//! - `engine-rs/src/runtime/tcgen_ops.rs`: `mma_f32_dot_increasing_k`,
//!   `mma_f32_abt_increasing_k`, `mma_f32_abt_banked_a_increasing_k` and the
//!   `#[cfg(feature = "python")]` / `#[cfg(not(...))]` split inside
//!   `tile_gemm_bf16_f32_ss_cta1` (the backend choice below);
//! - `engine-rs/src/runtime/matrix_ops.rs`: `mma_f64` (here generalised over
//!   `n`; `mma.sync` always passes `n = 8`) and the i64-intermediate integer
//!   accumulate/saturate used by `raw_mma_sync_packed_integer` / sparse.
//!
//! `ProfileTimer` instrumentation was dropped.

use crate::fpenv;
use crate::scalar::{fma_f64, F32RoundingMode};
use crate::types::{OpError, OpResult};

/// One output element: `accumulator = fma(a[i], b[i], accumulator)` for
/// `i = 0, 1, ..., k-1`, every step a single binary32 rounding.
#[inline]
pub fn mma_f32_dot_increasing_k(
    a_values: &[f32],
    b_values: &[f32],
    mut accumulator: f32,
) -> OpResult<f32> {
    if a_values.len() != b_values.len() {
        return Err(OpError::message(format!(
            "MMA dot operands have different K extents: {} and {}",
            a_values.len(),
            b_values.len(),
        )));
    }
    for (&a, &b) in a_values.iter().zip(b_values) {
        accumulator = a.mul_add(b, accumulator);
    }
    Ok(accumulator)
}

/// `D[m,n] = scale * D_in + A[m,k] * B[n,k]^T` with the NumSim f32 contract
/// (see `crate::mma` docs): the initial accumulator is `d_in * scale` (one
/// binary32 multiply, skipped entirely when `input_d` is `None`, giving +0.0),
/// then K is consumed in increasing order with binary32 FMA.
pub fn mma_f32_abt_increasing_k(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f32],
    b_values: &[f32],
    input_d: Option<(&[f32], f32)>,
) -> OpResult<Vec<f32>> {
    let a_len = m
        .checked_mul(k)
        .ok_or_else(|| OpError::message("raw TCGEN A matrix shape overflow"))?;
    let b_len = n
        .checked_mul(k)
        .ok_or_else(|| OpError::message("raw TCGEN B matrix shape overflow"))?;
    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| OpError::message("raw TCGEN output matrix shape overflow"))?;
    if a_values.len() != a_len {
        return Err(OpError::message(format!(
            "raw TCGEN A matrix has {} values, expected {a_len}",
            a_values.len()
        )));
    }
    if b_values.len() != b_len {
        return Err(OpError::message(format!(
            "raw TCGEN B matrix has {} values, expected {b_len}",
            b_values.len()
        )));
    }
    if let Some((values, _)) = input_d {
        if values.len() != output_len {
            return Err(OpError::message(format!(
                "raw TCGEN input D matrix has {} values, expected {output_len}",
                values.len()
            )));
        }
    }

    // NumSim uses one stable logical oracle: every output element accumulates K
    // in increasing order by binary32 FMA. D is the initial accumulator, not a
    // second rounded addition after a matrix product.
    //
    // Store B transposed so the independent columns of one output row are
    // contiguous. Advancing all columns together for each increasing K keeps
    // every element's exact FMA chain unchanged while allowing LLVM to
    // vectorize those independent chains.
    let mut b_transposed = vec![0.0; b_len];
    for col in 0..n {
        for inner in 0..k {
            b_transposed[inner * n + col] = b_values[col * k + inner];
        }
    }
    let mut output = input_d
        .map(|(values, scale)| values.iter().map(|&value| value * scale).collect())
        .unwrap_or_else(|| vec![0.0_f32; output_len]);
    fpenv::fma_f32_abt_increasing_k(m, n, k, a_values, &b_transposed, &mut output)
        .map_err(|error| OpError::message(format!("raw TCGEN MMA SIMD shape error: {error}")))?;
    Ok(output)
}

/// Banked-A variant: `a_values` holds `banks` consecutive `[m, k]` A matrices;
/// bank `b` produces output columns `[b*n/banks, (b+1)*n/banks)` from the
/// matching contiguous `[n/banks, k]` slice of B. Each bank is one
/// `mma_f32_abt_increasing_k` call.
pub fn mma_f32_abt_banked_a_increasing_k(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f32],
    b_values: &[f32],
    input_d: Option<(&[f32], f32)>,
    banks: usize,
) -> OpResult<Vec<f32>> {
    if !matches!(banks, 2 | 4) || n % banks != 0 {
        return Err(OpError::message(format!(
            "raw banked-A MMA requires N divisible by {banks} banks, got {n}"
        )));
    }
    let bank_columns = n / banks;
    let a_bank_len = m
        .checked_mul(k)
        .ok_or_else(|| OpError::message("raw banked-A shape overflow"))?;
    let expected_a_len = a_bank_len
        .checked_mul(banks)
        .ok_or_else(|| OpError::message("raw banked-A shape overflow"))?;
    if a_values.len() != expected_a_len {
        return Err(OpError::message(format!(
            "raw banked A has {} values, expected {expected_a_len}",
            a_values.len()
        )));
    }
    let expected_b_len = n
        .checked_mul(k)
        .ok_or_else(|| OpError::message("raw banked-B shape overflow"))?;
    if b_values.len() != expected_b_len {
        return Err(OpError::message(format!(
            "raw B matrix has {} values, expected {expected_b_len}",
            b_values.len()
        )));
    }
    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| OpError::message("raw banked output shape overflow"))?;
    if let Some((values, _)) = input_d {
        if values.len() != output_len {
            return Err(OpError::message(format!(
                "raw input D matrix has {} values, expected {output_len}",
                values.len()
            )));
        }
    }

    let mut output = vec![0.0_f32; output_len];
    for bank in 0..banks {
        let a_start = bank * a_bank_len;
        let b_bank_len = bank_columns * k;
        let b_start = bank * b_bank_len;
        let input_bank = input_d.map(|(values, scale)| {
            let mut selected = Vec::with_capacity(m * bank_columns);
            for row in 0..m {
                let start = row * n + bank * bank_columns;
                selected.extend_from_slice(&values[start..start + bank_columns]);
            }
            (selected, scale)
        });
        let bank_output = mma_f32_abt_increasing_k(
            m,
            bank_columns,
            k,
            &a_values[a_start..a_start + a_bank_len],
            &b_values[b_start..b_start + b_bank_len],
            input_bank
                .as_ref()
                .map(|(values, scale)| (values.as_slice(), *scale)),
        )?;
        for row in 0..m {
            let source = row * bank_columns;
            let destination = row * n + bank * bank_columns;
            output[destination..destination + bank_columns]
                .copy_from_slice(&bank_output[source..source + bank_columns]);
        }
    }
    Ok(output)
}

/// `D[m,n] = D_in + A[m,k] * B[n,k]^T` in binary64, one increasing-K FMA
/// chain per element starting at `D_in` (or +0.0). `rounding != Nearest`
/// uses `scalar::fma_f64` per step (exactly-rounded rz/rm/rp); `Nearest`
/// uses the vectorized `fpenv::fma_f64_abt_increasing_k` (same chain).
/// Legacy `matrix_ops.rs::mma_f64` (which fixed `n = 8`).
pub fn mma_f64_abt_increasing_k(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f64],
    b_values: &[f64],
    input_d: Option<&[f64]>,
    rounding: F32RoundingMode,
) -> OpResult<Vec<f64>> {
    if a_values.len() != m * k
        || b_values.len() != n * k
        || input_d.is_some_and(|d| d.len() != m * n)
    {
        return Err(OpError::message(
            "raw MMA SIMD shape error: f64 operand shapes do not match",
        ));
    }
    let mut output = input_d
        .map(<[f64]>::to_vec)
        .unwrap_or_else(|| vec![0.0_f64; m * n]);
    if rounding != F32RoundingMode::Nearest {
        for row in 0..m {
            for column in 0..n {
                let accumulator = &mut output[row * n + column];
                for inner in 0..k {
                    *accumulator = fma_f64(
                        a_values[row * k + inner],
                        b_values[column * k + inner],
                        *accumulator,
                        rounding,
                    );
                }
            }
        }
        return Ok(output);
    }
    let mut b_transposed = vec![0.0_f64; b_values.len()];
    for column in 0..n {
        for inner in 0..k {
            b_transposed[inner * n + column] = b_values[column * k + inner];
        }
    }
    fpenv::fma_f64_abt_increasing_k(m, n, k, a_values, &b_transposed, &mut output)
        .map_err(|error| OpError::message(format!("raw MMA SIMD shape error: {error}")))?;
    Ok(output)
}

/// Exact integer `output[m,n] += A[m,k] * B[n,k]^T` in i64 (products of i32).
pub fn mma_i32_abt_i64(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[i32],
    b_values: &[i32],
    output: &mut [i64],
) -> OpResult<()> {
    fpenv::multiply_accumulate_i32_abt(m, n, k, a_values, b_values, output)
        .map_err(|error| OpError::message(format!("raw MMA SIMD shape error: {error}")))
}

/// Narrow the i64 integer accumulator to the s32 destination: `.satfinite`
/// clamps, otherwise two's-complement wraps.
pub fn narrow_i64_accumulator(values: Vec<i64>, saturate: bool) -> Vec<i32> {
    values
        .into_iter()
        .map(|value| {
            if saturate {
                value.clamp(i32::MIN as i64, i32::MAX as i64) as i32
            } else {
                value as i32
            }
        })
        .collect()
}

/// A provider of the plain f32 product `C = A * B^T` (no input D).
///
/// `a` is row-major `[m, k]`, `b` is row-major `[n, k]`, `c` is row-major
/// `[m, n]` and is overwritten. The only legacy call site is the typed
/// canonical BF16 tile GEMM (`tile_gemm_bf16_f32_ss_cta1`, which is only
/// emitted for `accumulate=false`). Every fused-D accumulation (mma.sync,
/// raw tcgen05, banked) must call `mma_f32_abt_increasing_k` directly.
pub trait MatmulBackend {
    fn matmul_f32_abt(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> OpResult<()>;
}

/// Pure-Rust backend: increasing-K binary32 FMA chain from +0.0. This is the
/// legacy non-`python` build and the observed (checker) path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReferenceBackend;

impl MatmulBackend for ReferenceBackend {
    fn matmul_f32_abt(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> OpResult<()> {
        let output = mma_f32_abt_increasing_k(m, n, k, a, b, None)?;
        if c.len() != output.len() {
            return Err(OpError::message(format!(
                "matmul output has {} values, expected {}",
                c.len(),
                output.len()
            )));
        }
        c.copy_from_slice(&output);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ported from legacy tcgen_ops.rs `mod tests` (raw_mma_*).
    #[test]
    fn raw_mma_uses_a_fixed_increasing_k_reduction() {
        let large = 16_777_216.0_f32;
        let output =
            mma_f32_abt_increasing_k(1, 1, 3, &[1.0, 1.0, 1.0], &[large, 1.0, -large], None)
                .unwrap();

        assert_eq!(output, vec![0.0]);
        assert_eq!((large + -large) + 1.0, 1.0);
    }

    #[test]
    fn raw_mma_starts_the_fma_chain_from_scaled_input_d() {
        let large = 16_777_216.0_f32;
        let input_d = [2.0_f32];
        let output = mma_f32_abt_increasing_k(
            1,
            1,
            2,
            &[1.0, 1.0],
            &[large, -large],
            Some((&input_d, 0.5)),
        )
        .unwrap();

        assert_eq!(output, vec![0.0]);
        let product_then_d = (large + -large) + input_d[0] * 0.5;
        assert_eq!(product_then_d, 1.0);
    }

    #[test]
    fn raw_mma_vectorized_columns_are_bitwise_equal_to_scalar_dots() {
        let (m, n, k) = (3_usize, 19_usize, 7_usize);
        let a_values = (0..m * k)
            .map(|index| ((index as f32 - 9.0) * 0.3125).sin())
            .collect::<Vec<_>>();
        let b_values = (0..n * k)
            .map(|index| ((index as f32 + 3.0) * -0.21875).cos())
            .collect::<Vec<_>>();
        let input_d = (0..m * n)
            .map(|index| (index as f32 - 6.0) * 0.0625)
            .collect::<Vec<_>>();
        let scale = -0.75_f32;

        let output =
            mma_f32_abt_increasing_k(m, n, k, &a_values, &b_values, Some((&input_d, scale)))
                .unwrap();
        let mut scalar = vec![0.0_f32; m * n];
        for row in 0..m {
            for col in 0..n {
                scalar[row * n + col] = mma_f32_dot_increasing_k(
                    &a_values[row * k..(row + 1) * k],
                    &b_values[col * k..(col + 1) * k],
                    input_d[row * n + col] * scale,
                )
                .unwrap();
            }
        }

        assert_eq!(
            output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            scalar
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn raw_mma_rejects_malformed_matrix_payloads() {
        let error = mma_f32_abt_increasing_k(2, 1, 2, &[0.0; 3], &[0.0; 2], None).unwrap_err();
        assert!(error
            .to_string()
            .contains("A matrix has 3 values, expected 4"));

        let error =
            mma_f32_abt_increasing_k(1, 2, 1, &[0.0], &[0.0; 2], Some((&[0.0], 1.0))).unwrap_err();
        assert!(error
            .to_string()
            .contains("input D matrix has 1 values, expected 2"));
    }

    // New: contract checks for the remaining cores.
    #[test]
    fn banked_a_matches_per_bank_calls() {
        let (m, n, k, banks) = (2_usize, 8_usize, 3_usize, 2_usize);
        let a: Vec<f32> = (0..banks * m * k)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.11).cos()).collect();
        let d: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.25).collect();
        let output =
            mma_f32_abt_banked_a_increasing_k(m, n, k, &a, &b, Some((&d, 2.0)), banks).unwrap();
        for row in 0..m {
            for col in 0..n {
                let bank = col / (n / banks);
                let a_row = &a[bank * m * k + row * k..bank * m * k + (row + 1) * k];
                let expected = mma_f32_dot_increasing_k(
                    a_row,
                    &b[col * k..(col + 1) * k],
                    d[row * n + col] * 2.0,
                )
                .unwrap();
                assert_eq!(output[row * n + col].to_bits(), expected.to_bits());
            }
        }
        assert!(mma_f32_abt_banked_a_increasing_k(m, 6, k, &a, &b[..18], None, 4).is_err());
    }

    #[test]
    fn f64_core_keeps_one_chain_and_directed_rounding() {
        let big = 2.0_f64.powi(53);
        let out = mma_f64_abt_increasing_k(
            1,
            1,
            3,
            &[1.0, 1.0, 1.0],
            &[big, 1.0, -big],
            None,
            F32RoundingMode::Nearest,
        )
        .unwrap();
        assert_eq!(out, vec![0.0]);
        let up = mma_f64_abt_increasing_k(
            1,
            1,
            1,
            &[1.0],
            &[f64::EPSILON / 4.0],
            Some(&[1.0]),
            F32RoundingMode::Up,
        )
        .unwrap();
        assert_eq!(up[0], 1.0 + f64::EPSILON);
    }

    #[test]
    fn reference_backend_is_the_increasing_k_product() {
        let mut c = vec![f32::NAN; 1];
        ReferenceBackend
            .matmul_f32_abt(
                1,
                1,
                3,
                &[1.0, 1.0, 1.0],
                &[16_777_216.0, 1.0, -16_777_216.0],
                &mut c,
            )
            .unwrap();
        assert_eq!(c[0], 0.0);
        assert!(ReferenceBackend
            .matmul_f32_abt(1, 1, 1, &[1.0], &[1.0], &mut [0.0; 2])
            .is_err());
    }
}
