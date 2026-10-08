//! Host floating-point environment and order-preserving dense FMA kernels.
//!
//! Moved from the legacy `numsim-fp-env` crate. Thread affinity and heap
//! trimming stayed behind: they are runtime/host concerns, not numerics.

use core::ffi::c_int;
use std::error::Error;
use std::fmt;

#[cfg(target_os = "linux")]
// Linux libc ABIs define FE_TONEAREST as zero.
const FE_TONEAREST: c_int = 0;

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn fesetround(round: c_int) -> c_int;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatEnvironmentError {
    UnsupportedPlatform,
    SetRoundingFailed(c_int),
}

impl fmt::Display for FloatEnvironmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => {
                formatter.write_str("floating-point environment setup is unsupported")
            }
            Self::SetRoundingFailed(status) => {
                write!(formatter, "fesetround returned status {status}")
            }
        }
    }
}

impl Error for FloatEnvironmentError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FmaShapeError {
    operand: &'static str,
    actual: usize,
    expected: Option<usize>,
}

impl fmt::Display for FmaShapeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.expected {
            Some(expected) => write!(
                formatter,
                "{} has {} elements, expected {expected}",
                self.operand, self.actual
            ),
            None => write!(formatter, "{} shape overflows usize", self.operand),
        }
    }
}

impl Error for FmaShapeError {}

/// Select IEEE round-to-nearest-even for the calling thread.
#[cfg(target_os = "linux")]
pub fn set_round_to_nearest() -> Result<(), FloatEnvironmentError> {
    // SAFETY: fesetround has no pointer arguments and FE_TONEAREST is a valid
    // Linux libc rounding-mode constant. The floating-point environment is local to
    // the calling worker thread.
    let status = unsafe { fesetround(FE_TONEAREST) };
    if status == 0 {
        Ok(())
    } else {
        Err(FloatEnvironmentError::SetRoundingFailed(status))
    }
}

/// Fail closed when the host has no audited thread-local fenv implementation.
#[cfg(not(target_os = "linux"))]
pub fn set_round_to_nearest() -> Result<(), FloatEnvironmentError> {
    Err(FloatEnvironmentError::UnsupportedPlatform)
}

fn validate_abt_shapes<A, B, O>(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[A],
    b_transposed: &[B],
    output: &[O],
) -> Result<(), FmaShapeError> {
    for (operand, actual, shape) in [
        ("A", a_values.len(), (m, k)),
        ("transposed B", b_transposed.len(), (k, n)),
        ("output", output.len(), (m, n)),
    ] {
        let Some(expected) = shape.0.checked_mul(shape.1) else {
            return Err(FmaShapeError {
                operand,
                actual,
                expected: None,
            });
        };
        if actual != expected {
            return Err(FmaShapeError {
                operand,
                actual,
                expected: Some(expected),
            });
        }
    }
    Ok(())
}

/// `a * b + c` with the pinned host NaN (`scalar::host_fma_f32/f64`). The
/// kernels keep plain `mul_add` (vectorizable); [`repin_nan_outputs`] then
/// recomputes every NaN output with this, so the NaN payload never depends
/// on the operand order LLVM picks for `vfmadd`.
trait PinnedFma: Copy {
    fn pinned(self, b: Self, c: Self) -> Self;
    fn nan(self) -> bool;
}
impl PinnedFma for f32 {
    #[inline(always)]
    fn pinned(self, b: f32, c: f32) -> f32 {
        crate::scalar::host_fma_f32(self, b, c)
    }
    #[inline(always)]
    fn nan(self) -> bool {
        self.is_nan()
    }
}
impl PinnedFma for f64 {
    #[inline(always)]
    fn pinned(self, b: f64, c: f64) -> f64 {
        crate::scalar::host_fma_f64(self, b, c)
    }
    #[inline(always)]
    fn nan(self) -> bool {
        self.is_nan()
    }
}
#[inline(always)]
fn fma_pinned<T: PinnedFma>(a: T, b: T, c: T) -> T {
    a.pinned(b, c)
}

/// SIMD lanes leave a NaN's payload to the instruction's operand order:
/// recompute every NaN output's chain with the pinned scalar FMA from its
/// initial accumulator (a NaN, once produced, stays NaN along the chain).
fn repin_nan_outputs<T: PinnedFma>(
    n: usize,
    k: usize,
    a_values: &[T],
    b_transposed: &[T],
    initial: &[T],
    output: &mut [T],
) {
    // Branch-free NaN scan first (perf): the common all-finite output skips
    // the indexed walk entirely.
    if !output.iter().fold(false, |any, value| any | value.nan()) {
        return;
    }
    for (index, value) in output.iter_mut().enumerate() {
        if !value.nan() {
            continue;
        }
        let (row, column) = (index / n, index % n);
        let mut accumulator = initial[index];
        for inner in 0..k {
            accumulator = fma_pinned(
                a_values[row * k + inner],
                b_transposed[inner * n + column],
                accumulator,
            );
        }
        *value = accumulator;
    }
}

fn fma_f32_abt_increasing_k_scalar(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f32],
    b_transposed: &[f32],
    output: &mut [f32],
) {
    for row in 0..m {
        let a_row = &a_values[row * k..(row + 1) * k];
        let output_row = &mut output[row * n..(row + 1) * n];
        for (inner, &a) in a_row.iter().enumerate() {
            let b_row = &b_transposed[inner * n..(inner + 1) * n];
            for (accumulator, &b) in output_row.iter_mut().zip(b_row) {
                *accumulator = a.mul_add(b, *accumulator);
            }
        }
    }
}

/// One register tile of `R` rows by `C` 16-lane column vectors.
///
/// Every accumulator stays in a register across the whole K loop, so each
/// output element's increasing-K FMA chain is unchanged — only its
/// intermediate values move from memory round-trips into registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fma_tile_avx512<const R: usize, const C: usize>(
    n: usize,
    k: usize,
    a_values: &[f32],
    b_transposed: &[f32],
    output: &mut [f32],
    row: usize,
    column: usize,
) {
    use std::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_setzero_ps, _mm512_storeu_ps,
    };

    const LANES: usize = 16;
    // SAFETY: the caller guarantees `row + R <= m` and `column + C * 16 <= n`
    // over shape-validated slices.
    unsafe {
        let mut accumulators = [[_mm512_setzero_ps(); C]; R];
        for (r, row_accumulators) in accumulators.iter_mut().enumerate() {
            for (c, accumulator) in row_accumulators.iter_mut().enumerate() {
                *accumulator =
                    _mm512_loadu_ps(output.as_ptr().add((row + r) * n + column + c * LANES));
            }
        }
        for inner in 0..k {
            let mut b_lanes = [_mm512_setzero_ps(); C];
            for (c, lanes) in b_lanes.iter_mut().enumerate() {
                *lanes = _mm512_loadu_ps(b_transposed.as_ptr().add(inner * n + column + c * LANES));
            }
            for (r, row_accumulators) in accumulators.iter_mut().enumerate() {
                let a_lanes = _mm512_set1_ps(*a_values.get_unchecked((row + r) * k + inner));
                for (accumulator, &b) in row_accumulators.iter_mut().zip(&b_lanes) {
                    *accumulator = _mm512_fmadd_ps(a_lanes, b, *accumulator);
                }
            }
        }
        for (r, row_accumulators) in accumulators.iter().enumerate() {
            for (c, &accumulator) in row_accumulators.iter().enumerate() {
                _mm512_storeu_ps(
                    output.as_mut_ptr().add((row + r) * n + column + c * LANES),
                    accumulator,
                );
            }
        }
    }
}

/// Scalar columns keep the same register-resident chain: one accumulator per
/// element carried across the whole K loop.
fn fma_scalar_columns(
    n: usize,
    k: usize,
    a_values: &[f32],
    b_transposed: &[f32],
    output: &mut [f32],
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
) {
    for row in rows {
        for column in columns.clone() {
            let mut accumulator = output[row * n + column];
            for inner in 0..k {
                accumulator = a_values[row * k + inner]
                    .mul_add(b_transposed[inner * n + column], accumulator);
            }
            output[row * n + column] = accumulator;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fma_f32_abt_increasing_k_avx512(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f32],
    b_transposed: &[f32],
    output: &mut [f32],
) {
    const LANES: usize = 16;
    const ROW_TILE: usize = 4;
    const COL_VECTORS: usize = 4;
    const COL_TILE: usize = COL_VECTORS * LANES;

    // SAFETY: shape validation proves the slice extents; every tile call stays
    // inside `row + rows <= m` and `column + width <= n`.
    unsafe {
        let mut row = 0;
        while row + ROW_TILE <= m {
            let mut column = 0;
            while column + COL_TILE <= n {
                fma_tile_avx512::<ROW_TILE, COL_VECTORS>(
                    n,
                    k,
                    a_values,
                    b_transposed,
                    output,
                    row,
                    column,
                );
                column += COL_TILE;
            }
            while column + LANES <= n {
                fma_tile_avx512::<ROW_TILE, 1>(n, k, a_values, b_transposed, output, row, column);
                column += LANES;
            }
            if column < n {
                fma_scalar_columns(
                    n,
                    k,
                    a_values,
                    b_transposed,
                    output,
                    row..row + ROW_TILE,
                    column..n,
                );
            }
            row += ROW_TILE;
        }
        while row < m {
            let mut column = 0;
            while column + COL_TILE <= n {
                fma_tile_avx512::<1, COL_VECTORS>(
                    n,
                    k,
                    a_values,
                    b_transposed,
                    output,
                    row,
                    column,
                );
                column += COL_TILE;
            }
            while column + LANES <= n {
                fma_tile_avx512::<1, 1>(n, k, a_values, b_transposed, output, row, column);
                column += LANES;
            }
            if column < n {
                fma_scalar_columns(
                    n,
                    k,
                    a_values,
                    b_transposed,
                    output,
                    row..row + 1,
                    column..n,
                );
            }
            row += 1;
        }
    }
}

/// AVX2 register tile: `R` rows x `C` 8-lane vectors of accumulators held in
/// registers across the whole K loop (as [`fma_tile_avx512`]): each output
/// element's increasing-K FMA chain is unchanged, only the memory round trips
/// of the intermediate sums go away (perf, W4 profile).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_tile_avx2<const R: usize, const C: usize>(
    n: usize,
    k: usize,
    a_values: &[f32],
    b_transposed: &[f32],
    output: &mut [f32],
    row: usize,
    column: usize,
) {
    use std::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    const LANES: usize = 8;
    // SAFETY: the caller guarantees `row + R <= m` and `column + C * 8 <= n`
    // over shape-validated slices.
    unsafe {
        let mut accumulators = [[_mm256_setzero_ps(); C]; R];
        for (r, row_accumulators) in accumulators.iter_mut().enumerate() {
            for (c, accumulator) in row_accumulators.iter_mut().enumerate() {
                *accumulator =
                    _mm256_loadu_ps(output.as_ptr().add((row + r) * n + column + c * LANES));
            }
        }
        for inner in 0..k {
            let mut b_lanes = [_mm256_setzero_ps(); C];
            for (c, lanes) in b_lanes.iter_mut().enumerate() {
                *lanes = _mm256_loadu_ps(b_transposed.as_ptr().add(inner * n + column + c * LANES));
            }
            for (r, row_accumulators) in accumulators.iter_mut().enumerate() {
                let a_lanes = _mm256_set1_ps(*a_values.get_unchecked((row + r) * k + inner));
                for (accumulator, &b) in row_accumulators.iter_mut().zip(&b_lanes) {
                    *accumulator = _mm256_fmadd_ps(a_lanes, b, *accumulator);
                }
            }
        }
        for (r, row_accumulators) in accumulators.iter().enumerate() {
            for (c, &accumulator) in row_accumulators.iter().enumerate() {
                _mm256_storeu_ps(
                    output.as_mut_ptr().add((row + r) * n + column + c * LANES),
                    accumulator,
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_f32_abt_increasing_k_avx2(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f32],
    b_transposed: &[f32],
    output: &mut [f32],
) {
    const LANES: usize = 8;
    const ROW_TILE: usize = 4;
    const COL_VECTORS: usize = 2;
    const COL_TILE: usize = COL_VECTORS * LANES;

    // SAFETY: shape validation proves the slice extents; every tile call stays
    // inside `row + rows <= m` and `column + width <= n`.
    unsafe {
        let mut row = 0;
        while row < m {
            let rows = if row + ROW_TILE <= m { ROW_TILE } else { 1 };
            let mut column = 0;
            while column + COL_TILE <= n {
                if rows == ROW_TILE {
                    fma_tile_avx2::<ROW_TILE, COL_VECTORS>(
                        n,
                        k,
                        a_values,
                        b_transposed,
                        output,
                        row,
                        column,
                    );
                } else {
                    fma_tile_avx2::<1, COL_VECTORS>(
                        n,
                        k,
                        a_values,
                        b_transposed,
                        output,
                        row,
                        column,
                    );
                }
                column += COL_TILE;
            }
            while column + LANES <= n {
                if rows == ROW_TILE {
                    fma_tile_avx2::<ROW_TILE, 1>(n, k, a_values, b_transposed, output, row, column);
                } else {
                    fma_tile_avx2::<1, 1>(n, k, a_values, b_transposed, output, row, column);
                }
                column += LANES;
            }
            if column < n {
                fma_scalar_columns(
                    n,
                    k,
                    a_values,
                    b_transposed,
                    output,
                    row..row + rows,
                    column..n,
                );
            }
            row += rows;
        }
    }
}

/// Apply `output = A * B^T + output` while preserving increasing-K binary32
/// FMA order independently for every output element.
///
/// `b_transposed` is laid out as `[k, n]`. Runtime SIMD dispatch changes only
/// which independent output columns advance together; it never reassociates an
/// individual output element's FMA chain.
pub fn fma_f32_abt_increasing_k(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f32],
    b_transposed: &[f32],
    output: &mut [f32],
) -> Result<(), FmaShapeError> {
    validate_abt_shapes(m, n, k, a_values, b_transposed, output)?;
    let initial = output.to_vec();
    fma_f32_abt_increasing_k_dispatch(m, n, k, a_values, b_transposed, output);
    repin_nan_outputs(n, k, a_values, b_transposed, &initial, output);
    Ok(())
}

fn fma_f32_abt_increasing_k_dispatch(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f32],
    b_transposed: &[f32],
    output: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512 first: every AVX-512F host also reports AVX2, so the
        // narrower path must not shadow the wider one.
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime detection establishes AVX-512F support and the
            // validated slices satisfy the implementation's bounds contract.
            unsafe {
                fma_f32_abt_increasing_k_avx512(m, n, k, a_values, b_transposed, output);
            }
            return;
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: runtime detection establishes AVX2/FMA support and the
            // validated slices satisfy the implementation's bounds contract.
            unsafe {
                fma_f32_abt_increasing_k_avx2(m, n, k, a_values, b_transposed, output);
            }
            return;
        }
    }

    fma_f32_abt_increasing_k_scalar(m, n, k, a_values, b_transposed, output);
}

fn fma_f64_abt_increasing_k_scalar(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f64],
    b_transposed: &[f64],
    output: &mut [f64],
) {
    for row in 0..m {
        let a_row = &a_values[row * k..(row + 1) * k];
        let output_row = &mut output[row * n..(row + 1) * n];
        for (inner, &a) in a_row.iter().enumerate() {
            let b_row = &b_transposed[inner * n..(inner + 1) * n];
            for (accumulator, &b) in output_row.iter_mut().zip(b_row) {
                *accumulator = a.mul_add(b, *accumulator);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fma_f64_abt_increasing_k_avx512(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f64],
    b_transposed: &[f64],
    output: &mut [f64],
) {
    use std::arch::x86_64::{_mm512_fmadd_pd, _mm512_loadu_pd, _mm512_set1_pd, _mm512_storeu_pd};

    for row in 0..m {
        let a_row = &a_values[row * k..(row + 1) * k];
        let output_row = &mut output[row * n..(row + 1) * n];
        for (inner, &a) in a_row.iter().enumerate() {
            let b_row = &b_transposed[inner * n..(inner + 1) * n];
            let mut column = 0;
            // SAFETY: shape validation proves both row slices have n elements,
            // and the loop admits only complete eight-element vectors.
            unsafe {
                let a_lanes = _mm512_set1_pd(a);
                while column + 8 <= n {
                    let b_lanes = _mm512_loadu_pd(b_row.as_ptr().add(column));
                    let accumulators = _mm512_loadu_pd(output_row.as_ptr().add(column));
                    _mm512_storeu_pd(
                        output_row.as_mut_ptr().add(column),
                        _mm512_fmadd_pd(a_lanes, b_lanes, accumulators),
                    );
                    column += 8;
                }
            }
            for (accumulator, &b) in output_row[column..].iter_mut().zip(&b_row[column..]) {
                *accumulator = a.mul_add(b, *accumulator);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_f64_abt_increasing_k_avx2(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f64],
    b_transposed: &[f64],
    output: &mut [f64],
) {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd};

    for row in 0..m {
        let a_row = &a_values[row * k..(row + 1) * k];
        let output_row = &mut output[row * n..(row + 1) * n];
        for (inner, &a) in a_row.iter().enumerate() {
            let b_row = &b_transposed[inner * n..(inner + 1) * n];
            let mut column = 0;
            // SAFETY: shape validation proves both row slices have n elements,
            // and the loop admits only complete four-element vectors.
            unsafe {
                let a_lanes = _mm256_set1_pd(a);
                while column + 4 <= n {
                    let b_lanes = _mm256_loadu_pd(b_row.as_ptr().add(column));
                    let accumulators = _mm256_loadu_pd(output_row.as_ptr().add(column));
                    _mm256_storeu_pd(
                        output_row.as_mut_ptr().add(column),
                        _mm256_fmadd_pd(a_lanes, b_lanes, accumulators),
                    );
                    column += 4;
                }
            }
            for (accumulator, &b) in output_row[column..].iter_mut().zip(&b_row[column..]) {
                *accumulator = a.mul_add(b, *accumulator);
            }
        }
    }
}

/// Apply `output = A * B^T + output` with one binary64 FMA chain per output.
pub fn fma_f64_abt_increasing_k(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f64],
    b_transposed: &[f64],
    output: &mut [f64],
) -> Result<(), FmaShapeError> {
    validate_abt_shapes(m, n, k, a_values, b_transposed, output)?;
    let initial = output.to_vec();
    fma_f64_abt_increasing_k_dispatch(m, n, k, a_values, b_transposed, output);
    repin_nan_outputs(n, k, a_values, b_transposed, &initial, output);
    Ok(())
}

fn fma_f64_abt_increasing_k_dispatch(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f64],
    b_transposed: &[f64],
    output: &mut [f64],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: runtime detection establishes AVX2/FMA support and the
            // validated slices satisfy the implementation's bounds contract.
            unsafe {
                fma_f64_abt_increasing_k_avx2(m, n, k, a_values, b_transposed, output);
            }
            return;
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime detection establishes AVX-512F support and the
            // validated slices satisfy the implementation's bounds contract.
            unsafe {
                fma_f64_abt_increasing_k_avx512(m, n, k, a_values, b_transposed, output);
            }
            return;
        }
    }

    fma_f64_abt_increasing_k_scalar(m, n, k, a_values, b_transposed, output);
}

fn multiply_accumulate_i32_abt_scalar(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[i32],
    b_transposed: &[i64],
    output: &mut [i64],
) {
    for row in 0..m {
        let a_row = &a_values[row * k..(row + 1) * k];
        let output_row = &mut output[row * n..(row + 1) * n];
        for (inner, &a) in a_row.iter().enumerate() {
            let b_row = &b_transposed[inner * n..(inner + 1) * n];
            for (accumulator, &b) in output_row.iter_mut().zip(b_row) {
                *accumulator += i64::from(a) * b;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn multiply_accumulate_i32_abt_avx2(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[i32],
    b_transposed: &[i64],
    output: &mut [i64],
) {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_mul_epi32, _mm256_set1_epi64x,
        _mm256_storeu_si256,
    };

    for row in 0..m {
        let a_row = &a_values[row * k..(row + 1) * k];
        let output_row = &mut output[row * n..(row + 1) * n];
        for (inner, &a) in a_row.iter().enumerate() {
            let b_row = &b_transposed[inner * n..(inner + 1) * n];
            let mut column = 0;
            // Each i64 B lane contains one sign-extended i32 operand. AVX2
            // `mul_epi32` multiplies the low i32 of each i64 lane and returns
            // four exact i64 products.
            unsafe {
                let a_lanes = _mm256_set1_epi64x(i64::from(a));
                while column + 4 <= n {
                    let b_lanes = _mm256_loadu_si256(b_row.as_ptr().add(column).cast::<__m256i>());
                    let accumulators =
                        _mm256_loadu_si256(output_row.as_ptr().add(column).cast::<__m256i>());
                    let result = _mm256_add_epi64(accumulators, _mm256_mul_epi32(a_lanes, b_lanes));
                    _mm256_storeu_si256(
                        output_row.as_mut_ptr().add(column).cast::<__m256i>(),
                        result,
                    );
                    column += 4;
                }
            }
            for (accumulator, &b) in output_row[column..].iter_mut().zip(&b_row[column..]) {
                *accumulator += i64::from(a) * b;
            }
        }
    }
}

/// Apply exact signed-i32 products into i64 accumulators.
///
/// `b_values` is row-major `[n, k]`; this routine performs the transpose once
/// so independent output columns can advance together under SIMD.
pub fn multiply_accumulate_i32_abt(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[i32],
    b_values: &[i32],
    output: &mut [i64],
) -> Result<(), FmaShapeError> {
    validate_abt_shapes(m, n, k, a_values, b_values, output)?;
    let mut b_transposed = vec![0_i64; b_values.len()];
    for column in 0..n {
        for inner in 0..k {
            b_transposed[inner * n + column] = i64::from(b_values[column * k + inner]);
        }
    }

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime detection establishes AVX2 support and validation
        // proves every vector load/store stays within its row slice.
        unsafe {
            multiply_accumulate_i32_abt_avx2(m, n, k, a_values, &b_transposed, output);
        }
        return Ok(());
    }

    multiply_accumulate_i32_abt_scalar(m, n, k, a_values, &b_transposed, output);
    Ok(())
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn supported_host_accepts_round_to_nearest() {
        assert_eq!(set_round_to_nearest(), Ok(()));
    }

    #[test]
    fn simd_fma_is_bitwise_equal_to_scalar_increasing_k() {
        let (m, n, k) = (3_usize, 19_usize, 7_usize);
        let a_values = (0..m * k)
            .map(|index| ((index as f32 - 9.0) * 0.3125).sin())
            .collect::<Vec<_>>();
        let b_values = (0..n * k)
            .map(|index| ((index as f32 + 3.0) * -0.21875).cos())
            .collect::<Vec<_>>();
        let mut b_transposed = vec![0.0_f32; n * k];
        for column in 0..n {
            for inner in 0..k {
                b_transposed[inner * n + column] = b_values[column * k + inner];
            }
        }
        let initial = (0..m * n)
            .map(|index| (index as f32 - 6.0) * -0.046875)
            .collect::<Vec<_>>();
        let mut simd = initial.clone();
        let mut scalar = initial;

        fma_f32_abt_increasing_k(m, n, k, &a_values, &b_transposed, &mut simd).unwrap();
        fma_f32_abt_increasing_k_scalar(m, n, k, &a_values, &b_transposed, &mut scalar);

        assert_eq!(
            simd.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            scalar
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn simd_fma_matches_scalar_bitwise_across_tile_shapes() {
        // Sweep every microkernel path: 4-row and leftover-row tiles, 64- and
        // 16-column vectors, and scalar column tails.
        for &m in &[1_usize, 2, 3, 4, 5, 8, 9] {
            for &n in &[1_usize, 7, 15, 16, 17, 63, 64, 65, 80, 130] {
                for &k in &[1_usize, 2, 7, 32] {
                    let a_values = (0..m * k)
                        .map(|index| ((index as f32 - 11.0) * 0.317).sin() * 3.0)
                        .collect::<Vec<_>>();
                    let b_transposed = (0..n * k)
                        .map(|index| ((index as f32 + 5.0) * -0.213).cos() * 2.0)
                        .collect::<Vec<_>>();
                    let initial = (0..m * n)
                        .map(|index| (index as f32 - 6.0) * -0.047)
                        .collect::<Vec<_>>();
                    let mut simd = initial.clone();
                    let mut scalar = initial;

                    fma_f32_abt_increasing_k(m, n, k, &a_values, &b_transposed, &mut simd).unwrap();
                    fma_f32_abt_increasing_k_scalar(m, n, k, &a_values, &b_transposed, &mut scalar);

                    assert_eq!(
                        simd.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                        scalar
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        "bitwise mismatch at m={m} n={n} k={k}",
                    );
                }
            }
        }
    }

    /// The register-tiled AVX2 kernel, called directly (so it is covered on
    /// AVX-512 hosts too), equals the scalar chains bit for bit over every
    /// tile path: 4-row and single-row tiles, 16- and 8-column vectors, and
    /// scalar tails; NaN/inf/subnormal inputs included.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_register_tile_matches_scalar_bitwise() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            return;
        }
        let special = [
            f32::NAN,
            f32::INFINITY,
            -0.0,
            f32::from_bits(1),
            f32::MAX,
            -f32::MIN_POSITIVE,
        ];
        for &m in &[1_usize, 3, 4, 5, 8, 9, 128] {
            for &n in &[1_usize, 7, 8, 15, 16, 17, 24, 33, 256] {
                for &k in &[1_usize, 2, 7, 16, 32] {
                    let pick = |i: usize, s: f32| {
                        if i % 97 == 5 {
                            special[i % special.len()]
                        } else {
                            ((i as f32 - 11.0) * s).sin() * 3.0
                        }
                    };
                    let a_values = (0..m * k).map(|i| pick(i, 0.317)).collect::<Vec<_>>();
                    let b_transposed = (0..n * k).map(|i| pick(i + 3, -0.213)).collect::<Vec<_>>();
                    let initial = (0..m * n)
                        .map(|i| (i as f32 - 6.0) * -0.047)
                        .collect::<Vec<_>>();
                    let mut simd = initial.clone();
                    let mut scalar = initial;
                    // SAFETY: AVX2/FMA detected above; shapes are consistent.
                    unsafe {
                        fma_f32_abt_increasing_k_avx2(m, n, k, &a_values, &b_transposed, &mut simd)
                    };
                    fma_f32_abt_increasing_k_scalar(m, n, k, &a_values, &b_transposed, &mut scalar);
                    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&simd), bits(&scalar), "m={m} n={n} k={k}");
                }
            }
        }
    }

    #[test]
    fn simd_fma_rejects_malformed_shapes() {
        let error =
            fma_f32_abt_increasing_k(2, 3, 4, &[0.0; 7], &[0.0; 12], &mut [0.0; 6]).unwrap_err();
        assert_eq!(error.to_string(), "A has 7 elements, expected 8");
    }

    #[test]
    fn f64_simd_matches_independent_increasing_k_fma_chains() {
        let (m, n, k) = (3_usize, 11_usize, 7_usize);
        let a_values = (0..m * k)
            .map(|index| ((index as f64 - 9.0) * 0.3125).sin())
            .collect::<Vec<_>>();
        let mut b_transposed = vec![0.0_f64; n * k];
        for inner in 0..k {
            for column in 0..n {
                b_transposed[inner * n + column] =
                    (((column * k + inner) as f64 + 3.0) * -0.21875).cos();
            }
        }
        let initial = (0..m * n)
            .map(|index| (index as f64 - 6.0) * -0.046875)
            .collect::<Vec<_>>();
        let mut actual = initial.clone();

        fma_f64_abt_increasing_k(m, n, k, &a_values, &b_transposed, &mut actual).unwrap();

        let mut expected = Vec::with_capacity(m * n);
        for row in 0..m {
            for column in 0..n {
                let mut accumulator = initial[row * n + column];
                for inner in 0..k {
                    accumulator = a_values[row * k + inner]
                        .mul_add(b_transposed[inner * n + column], accumulator);
                }
                expected.push(accumulator.to_bits());
            }
        }
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected,
        );
    }

    #[test]
    fn integer_simd_matches_independent_dot_products() {
        let (m, n, k) = (3_usize, 11_usize, 17_usize);
        let a_values = (0..m * k)
            .map(|index| (index as i32 % 31) - 15)
            .collect::<Vec<_>>();
        let b_values = (0..n * k)
            .map(|index| 12 - (index as i32 % 29))
            .collect::<Vec<_>>();
        let initial = (0..m * n)
            .map(|index| i64::from(index as i32 - 20) * 1_000_003)
            .collect::<Vec<_>>();
        let mut actual = initial.clone();

        multiply_accumulate_i32_abt(m, n, k, &a_values, &b_values, &mut actual).unwrap();

        let expected = (0..m)
            .flat_map(|row| {
                let initial = initial.clone();
                let a_values = &a_values;
                let b_values = &b_values;
                (0..n).map(move |column| {
                    (0..k).fold(initial[row * n + column], |sum, inner| {
                        sum + i64::from(a_values[row * k + inner])
                            * i64::from(b_values[column * k + inner])
                    })
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}

#[cfg(test)]
mod pinned_nan_tests {
    use super::*;

    /// The SIMD MMA chains give the pinned scalar chain's bits on NaN
    /// outputs (several NaN payloads per chain, invalid products), whichever
    /// vector path the CPU takes and in any build profile.
    #[test]
    fn abt_chains_pin_nan_payloads() {
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let nans32 = [0x7fc0_1234_u32, 0xffa0_0001, 0x7f80_0007, 0xffc0_0000];
        for (m, n, k) in [(5, 37, 9), (16, 64, 16), (3, 8, 1)] {
            let pick = |x: u64| -> f32 {
                match x % 13 {
                    0 => f32::from_bits(nans32[(x >> 8) as usize % 4]),
                    1 => f32::INFINITY,
                    2 => 0.0,
                    _ => f32::from_bits(0x3f00_0000 | (x as u32 & 0x00ff_ffff)),
                }
            };
            let a: Vec<f32> = (0..m * k).map(|_| pick(next())).collect();
            let b: Vec<f32> = (0..k * n).map(|_| pick(next())).collect();
            let init: Vec<f32> = (0..m * n).map(|_| pick(next())).collect();
            let mut fast = init.clone();
            fma_f32_abt_increasing_k(m, n, k, &a, &b, &mut fast).unwrap();
            for row in 0..m {
                for col in 0..n {
                    let mut acc = init[row * n + col];
                    for inner in 0..k {
                        acc = crate::scalar::host_fma_f32(
                            a[row * k + inner],
                            b[inner * n + col],
                            acc,
                        );
                    }
                    assert_eq!(
                        fast[row * n + col].to_bits(),
                        acc.to_bits(),
                        "f32 {m}x{n}x{k} ({row}, {col})"
                    );
                }
            }
            let a: Vec<f64> = a.iter().map(|&v| f64::from(v)).collect();
            let b: Vec<f64> = b.iter().map(|&v| f64::from(v)).collect();
            let init: Vec<f64> = init.iter().map(|&v| f64::from(v)).collect();
            let mut fast = init.clone();
            fma_f64_abt_increasing_k(m, n, k, &a, &b, &mut fast).unwrap();
            for row in 0..m {
                for col in 0..n {
                    let mut acc = init[row * n + col];
                    for inner in 0..k {
                        acc = crate::scalar::host_fma_f64(
                            a[row * k + inner],
                            b[inner * n + col],
                            acc,
                        );
                    }
                    assert_eq!(
                        fast[row * n + col].to_bits(),
                        acc.to_bits(),
                        "f64 {m}x{n}x{k} ({row}, {col})"
                    );
                }
            }
        }
    }
}
