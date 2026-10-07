//! Dense `mma.sync` numerics on register fragments.
//!
//! Legacy source: `engine-rs/src/runtime/matrix_ops.rs` (`raw_mma_sync_*`,
//! `mma_f32`). Each function takes the A/B/C register values of the whole
//! warp and returns the D register values; the engine adapter keeps the
//! full-warp check (`require_full_warp_sync`), D-pointer width checks and the
//! physical stores. Destination register counts are fixed by the form and
//! documented per function (legacy rejected a mismatching `d.len()` with the
//! same message as the operand-count check; the adapter must keep that).

use super::backend::{
    mma_f32_abt_increasing_k, mma_f64_abt_increasing_k, mma_i32_abt_i64, narrow_i64_accumulator,
};
use super::fragments::*;
use crate::cvt::f32_to_tf32;
use crate::scalar::F32RoundingMode;
use crate::types::{OpError, OpResult, WarpValue};

/// f32-accumulator core shared by every float `mma.sync`: `n = 8`, C (if
/// any) is the unscaled initial accumulator (`scale = 1.0`).
pub fn mma_f32(
    m: usize,
    k: usize,
    a_values: &[f32],
    b_values: &[f32],
    input_d: Option<&[f32]>,
) -> OpResult<Vec<f32>> {
    mma_f32_abt_increasing_k(
        m,
        8,
        k,
        a_values,
        b_values,
        input_d.map(|values| (values, 1.0_f32)),
    )
}

fn f32_accumulator(c: Option<&[WarpValue<u32>]>, m: usize) -> Option<Vec<f32>> {
    c.map(|registers| gather_mma_accumulator(m, &decode_fragment(registers, f32::from_bits)))
}

/// `mma.sync.aligned.m16n8k{8,16}.row.col.f32.{f16,bf16}.{f16,bf16}.f32`.
/// A: `k/4` regs, B: `k/8` regs, C: 4 regs. Returns 4 f32 D registers.
pub fn mma_sync_f32_b16(
    a: &[WarpValue<u32>],
    b: &[WarpValue<u32>],
    c: Option<&[WarpValue<u32>]>,
    k: usize,
    dtype: MatrixB16Type,
) -> OpResult<Vec<WarpValue<f32>>> {
    if !matches!(k, 8 | 16) || a.len() != k / 4 || b.len() != k / 8 {
        return Err(OpError::message(format!(
            "mma.sync f32/b16 fragment counts do not match m16n8k{k}"
        )));
    }
    if let Some(c) = c {
        if c.len() != 4 {
            return Err(OpError::message(
                "mma.sync C fragment must contain four f32 registers",
            ));
        }
    }
    let a_values = gather_matrix(16, k, |row, inner| {
        let (lane, slot) = mma_a_b16_owner(row, inner, k);
        packed_b16_value(a, lane, slot, dtype)
    })?;
    let b_values = gather_matrix(8, k, |column, inner| {
        let (lane, slot) = mma_b_b16_owner(inner, column);
        packed_b16_value(b, lane, slot, dtype)
    })?;
    let input_d = f32_accumulator(c, 16);
    let output = mma_f32(16, k, &a_values, &b_values, input_d.as_deref())?;
    matrix_output_registers(16, 4, 0.0_f32, &output)
}

/// `mma.sync.aligned.m16n8k{4,8}.row.col.f32.tf32.tf32.f32`. Operands are
/// rounded with `f32_to_tf32` (RNA to 10 mantissa bits) before the product.
/// A: `k/2` regs, B: `k/4`, C: 4. Returns 4 f32 D registers.
pub fn mma_sync_f32_tf32(
    a: &[WarpValue<u32>],
    b: &[WarpValue<u32>],
    c: Option<&[WarpValue<u32>]>,
    k: usize,
) -> OpResult<Vec<WarpValue<f32>>> {
    if !matches!(k, 4 | 8)
        || a.len() != k / 2
        || b.len() != k / 4
        || c.is_some_and(|value| value.len() != 4)
    {
        return Err(OpError::message(
            "mma.sync TF32 fragment counts are invalid",
        ));
    }
    let a_values = gather_matrix(16, k, |row, inner| {
        let (lane, slot) = mma_a_tf32_owner(row, inner);
        Ok(f32_to_tf32(f32::from_bits(a[slot][lane])))
    })?;
    let b_values = gather_matrix(8, k, |column, inner| {
        let (lane, slot) = mma_b_tf32_owner(column, inner);
        Ok(f32_to_tf32(f32::from_bits(b[slot][lane])))
    })?;
    let input_d = f32_accumulator(c, 16);
    let output = mma_f32(16, k, &a_values, &b_values, input_d.as_deref())?;
    matrix_output_registers(16, 4, 0.0_f32, &output)
}

/// `mma.sync.aligned.m16n8k{8,16}.row.col.f16.f16.f16.f16`: the f16 C is
/// widened, the chain runs in binary32, and each result is rounded to f16
/// (RNE) once at the end. Returns 2 packed f16x2 D registers.
pub fn mma_sync_f16_f16(
    a: &[WarpValue<u32>],
    b: &[WarpValue<u32>],
    c: Option<&[WarpValue<u32>]>,
    k: usize,
) -> OpResult<Vec<WarpValue<u32>>> {
    if !matches!(k, 8 | 16)
        || a.len() != k / 4
        || b.len() != k / 8
        || c.is_some_and(|value| value.len() != 2)
    {
        return Err(OpError::message(
            "mma.sync f16-accumulator fragment counts are invalid",
        ));
    }
    let a_values = gather_matrix(16, k, |row, inner| {
        let (lane, slot) = mma_a_b16_owner(row, inner, k);
        packed_b16_value(a, lane, slot, MatrixB16Type::Fp16)
    })?;
    let b_values = gather_matrix(8, k, |column, inner| {
        let (lane, slot) = mma_b_b16_owner(inner, column);
        packed_b16_value(b, lane, slot, MatrixB16Type::Fp16)
    })?;
    let input_d = c
        .map(|registers| gather_packed_f16_accumulator(16, registers))
        .transpose()?;
    let output = mma_f32(16, k, &a_values, &b_values, input_d.as_deref())?;
    pack_f16_registers(&matrix_output_registers(16, 4, 0.0_f32, &output)?)
}

/// Dense integer and binary `mma.sync` (`m8n8k16/k32`, `m16n8k16/k32/k64`,
/// `b1` `m8n8k128`, `m16n8k128/k256`). Products are exact i32 summed into an
/// i64 starting at C; `.satfinite` clamps to s32, else wraps. `b1.xor`
/// popc(a ^ b) is evaluated as `popc(a) + sum(a' * b)` with `a' = 1 - 2a`;
/// `b1.and` is the plain 0/1 product. Returns `m/4` s32 D registers.
#[allow(clippy::too_many_arguments)]
pub fn mma_sync_packed_integer(
    a: &[WarpValue<u32>],
    b: &[WarpValue<u32>],
    c: Option<&[WarpValue<u32>]>,
    m: usize,
    k: usize,
    a_dtype: MatrixPackedIntType,
    b_dtype: MatrixPackedIntType,
    saturate: bool,
    bit_op: Option<MatrixBitOp>,
) -> OpResult<Vec<WarpValue<i32>>> {
    if !matches!(m, 8 | 16) {
        return Err(OpError::message("packed integer MMA requires M=8 or M=16"));
    }
    let a_bits = matrix_packed_int_bits(a_dtype);
    let b_bits = matrix_packed_int_bits(b_dtype);
    if a_bits != b_bits {
        return Err(OpError::message(
            "packed integer MMA multiplicands must have the same element width",
        ));
    }
    let expected_d = m / 4;
    let expected_a = m * k * a_bits / (32 * 32);
    let expected_b = k * 8 * b_bits / (32 * 32);
    if a.len() != expected_a
        || b.len() != expected_b
        || c.is_some_and(|value| value.len() != expected_d)
    {
        return Err(OpError::message(
            "packed integer MMA fragment counts are invalid",
        ));
    }
    let is_b1 = a_dtype == MatrixPackedIntType::B1 && b_dtype == MatrixPackedIntType::B1;
    if is_b1 != bit_op.is_some() {
        return Err(OpError::message(
            "b1 MMA requires a bit operation and integer MMA forbids one",
        ));
    }
    if saturate && (is_b1 || a_bits == 1) {
        return Err(OpError::message("b1 MMA does not support saturation"));
    }

    let mut a_values = gather_matrix(m, k, |row, inner| {
        let (lane, register, element) = mma_packed_a_owner(m, row, inner, a_bits);
        packed_integer_value(a, lane, register, element, a_dtype)
    })?;
    let b_values = gather_matrix(8, k, |column, inner| {
        let (lane, register, element) = mma_packed_b_owner(column, inner, b_bits);
        packed_integer_value(b, lane, register, element, b_dtype)
    })?;
    let mut output = c
        .map(|registers| {
            gather_mma_accumulator(m, &decode_fragment(registers, |bits| bits as i32))
                .into_iter()
                .map(i64::from)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![0_i64; m * 8]);

    if bit_op == Some(MatrixBitOp::Xor) {
        for row in 0..m {
            let a_row = &mut a_values[row * k..(row + 1) * k];
            let true_count = a_row
                .iter()
                .map(|&value| i64::from(value != 0))
                .sum::<i64>();
            for accumulator in &mut output[row * 8..(row + 1) * 8] {
                *accumulator += true_count;
            }
            for value in a_row {
                *value = 1 - 2 * i32::from(*value != 0);
            }
        }
    }
    mma_i32_abt_i64(m, 8, k, &a_values, &b_values, &mut output)?;
    matrix_output_registers(
        m,
        expected_d,
        0_i32,
        &narrow_i64_accumulator(output, saturate),
    )
}

fn gather_f8_operands(
    a: &[WarpValue<u32>],
    b: &[WarpValue<u32>],
    k: usize,
    a_dtype: MatrixF8Type,
    b_dtype: MatrixF8Type,
) -> OpResult<(Vec<f32>, Vec<f32>)> {
    let a_values = gather_matrix(16, k, |row, inner| {
        let (lane, register, element) = mma_packed_a_owner(16, row, inner, 8);
        packed_f8_value(&a[register..=register], lane, element, a_dtype)
    })?;
    let b_values = gather_matrix(8, k, |column, inner| {
        let (lane, register, element) = mma_packed_b_owner(column, inner, 8);
        packed_f8_value(&b[register..=register], lane, element, b_dtype)
    })?;
    Ok((a_values, b_values))
}

/// `mma.sync.aligned.m16n8k{16,32}.row.col.f32.{e4m3,e5m2}.{e4m3,e5m2}.f32`.
/// A: `k/8` regs, B: `k/16`, C: 4. Returns 4 f32 D registers.
pub fn mma_sync_f32_f8(
    a: &[WarpValue<u32>],
    b: &[WarpValue<u32>],
    c: Option<&[WarpValue<u32>]>,
    k: usize,
    a_dtype: MatrixF8Type,
    b_dtype: MatrixF8Type,
) -> OpResult<Vec<WarpValue<f32>>> {
    if !matches!(k, 16 | 32)
        || a.len() != k / 8
        || b.len() != k / 16
        || c.is_some_and(|value| value.len() != 4)
    {
        return Err(OpError::message("mma.sync f8 fragment counts are invalid"));
    }
    let (a_values, b_values) = gather_f8_operands(a, b, k, a_dtype, b_dtype)?;
    let input_d = f32_accumulator(c, 16);
    let output = mma_f32(16, k, &a_values, &b_values, input_d.as_deref())?;
    matrix_output_registers(16, 4, 0.0_f32, &output)
}

/// f8 multiplicands with an f16 C/D (chain in binary32, one final f16 RNE).
/// Returns 2 packed f16x2 D registers.
pub fn mma_sync_f16_f8(
    a: &[WarpValue<u32>],
    b: &[WarpValue<u32>],
    c: Option<&[WarpValue<u32>]>,
    k: usize,
    a_dtype: MatrixF8Type,
    b_dtype: MatrixF8Type,
) -> OpResult<Vec<WarpValue<u32>>> {
    if !matches!(k, 16 | 32)
        || a.len() != k / 8
        || b.len() != k / 16
        || c.is_some_and(|value| value.len() != 2)
    {
        return Err(OpError::message(
            "mma.sync f8/f16-accumulator fragment counts are invalid",
        ));
    }
    let (a_values, b_values) = gather_f8_operands(a, b, k, a_dtype, b_dtype)?;
    let input_d = c
        .map(|registers| gather_packed_f16_accumulator(16, registers))
        .transpose()?;
    let output = mma_f32(16, k, &a_values, &b_values, input_d.as_deref())?;
    pack_f16_registers(&matrix_output_registers(16, 4, 0.0_f32, &output)?)
}

/// `mma.sync` f64: `m8n8k4` and `m16n8k{4,8,16}`; binary64 increasing-K FMA
/// chain from C with the static rounding mode. Returns `m/4` f64 registers.
pub fn mma_sync_f64(
    a: &[WarpValue<f64>],
    b: &[WarpValue<f64>],
    c: Option<&[WarpValue<f64>]>,
    m: usize,
    k: usize,
    rounding: F32RoundingMode,
) -> OpResult<Vec<WarpValue<f64>>> {
    let valid_shape = (m, k) == (8, 4) || (m == 16 && matches!(k, 4 | 8 | 16));
    let expected_d = m / 4;
    let expected_a = m * k / 32;
    let expected_b = k / 4;
    if !valid_shape
        || a.len() != expected_a
        || b.len() != expected_b
        || c.is_some_and(|value| value.len() != expected_d)
    {
        return Err(OpError::message("mma.sync f64 fragment counts are invalid"));
    }
    let a_values = gather_matrix(m, k, |row, inner| {
        let (lane, slot) = mma_a_f64_owner(m, row, inner);
        Ok(a[slot][lane])
    })?;
    let b_values = gather_matrix(8, k, |column, inner| {
        let (lane, slot) = mma_b_tf32_owner(column, inner);
        Ok(b[slot][lane])
    })?;
    let input_d = c.map(|registers| gather_mma_accumulator(m, registers));
    let output =
        mma_f64_abt_increasing_k(m, 8, k, &a_values, &b_values, input_d.as_deref(), rounding)?;
    matrix_output_registers(m, expected_d, 0.0_f64, &output)
}

/// D registers of a form whose accumulator is either f32 or packed f16.
#[derive(Clone, Debug, PartialEq)]
pub enum MmaFloatOutput {
    /// One f32 per register slot.
    F32(Vec<WarpValue<f32>>),
    /// f16x2 words (even logical slot in the low half).
    PackedF16(Vec<WarpValue<u32>>),
}

/// `mma.sync.aligned.m8n8k4.{row,col}.{row,col}.{f16,f32}.f16.f16.{f16,f32}`:
/// four independent 8x8x4 products, one per quad pair. f16 D returns 4
/// packed registers, f32 D returns 8 f32 registers.
#[allow(clippy::too_many_arguments)]
pub fn mma_sync_m8n8k4_f16(
    a: &[WarpValue<u32>],
    b: &[WarpValue<u32>],
    c: Option<&[WarpValue<u32>]>,
    a_layout: MatrixLayout,
    b_layout: MatrixLayout,
    d_dtype: MatrixAccumulatorType,
    c_dtype: MatrixAccumulatorType,
) -> OpResult<MmaFloatOutput> {
    let c_count = match c_dtype {
        MatrixAccumulatorType::Fp16 => 4,
        MatrixAccumulatorType::Fp32 => 8,
    };
    if a.len() != 2 || b.len() != 2 || c.is_some_and(|value| value.len() != c_count) {
        return Err(OpError::message(
            "mma.sync m8n8k4 f16 fragment counts are invalid",
        ));
    }
    let c_f16 = if c_dtype == MatrixAccumulatorType::Fp16 {
        c
    } else {
        None
    };
    let c_f32 = if c_dtype == MatrixAccumulatorType::Fp32 {
        c.map(|registers| decode_fragment(registers, f32::from_bits))
    } else {
        None
    };
    let mut output = vec![[0.0_f32; 32]; 8];
    for computation in 0..4 {
        let a_values = gather_matrix(8, 4, |row, inner| {
            let (lane, slot) = m8n8k4_f16_a_owner(computation, row, inner, a_layout);
            packed_b16_value(a, lane, slot, MatrixB16Type::Fp16)
        })?;
        let b_values = gather_matrix(8, 4, |column, inner| {
            let (lane, slot) = m8n8k4_f16_b_owner(computation, inner, column, b_layout);
            packed_b16_value(b, lane, slot, MatrixB16Type::Fp16)
        })?;
        let input_d = if let Some(registers) = &c_f16 {
            Some(gather_matrix(8, 8, |row, column| {
                let (lane, slot) = m8n8k4_f16_acc_owner(computation, row, column);
                packed_b16_value(registers, lane, slot, MatrixB16Type::Fp16)
            })?)
        } else if let Some(registers) = &c_f32 {
            Some(gather_matrix(8, 8, |row, column| {
                let (lane, slot) = m8n8k4_f32_acc_owner(computation, row, column)?;
                Ok(registers[slot][lane])
            })?)
        } else {
            None
        };
        let values = mma_f32(8, 4, &a_values, &b_values, input_d.as_deref())?;
        for row in 0..8 {
            for column in 0..8 {
                let (lane, slot) = match d_dtype {
                    MatrixAccumulatorType::Fp16 => m8n8k4_f16_acc_owner(computation, row, column),
                    MatrixAccumulatorType::Fp32 => m8n8k4_f32_acc_owner(computation, row, column)?,
                };
                output[slot][lane] = values[row * 8 + column];
            }
        }
    }
    Ok(match d_dtype {
        MatrixAccumulatorType::Fp16 => MmaFloatOutput::PackedF16(pack_f16_registers(&output)?),
        MatrixAccumulatorType::Fp32 => MmaFloatOutput::F32(output),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cvt::{f32_to_bf16_bits, f32_to_fp16_bits};

    fn set_b16(regs: &mut [WarpValue<u32>], lane: usize, slot: usize, bits: u16) {
        let word = &mut regs[slot / 2][lane];
        let shift = 16 * (slot % 2);
        *word = (*word & !(0xffff << shift)) | (u32::from(bits) << shift);
    }

    fn a_val(row: usize, inner: usize) -> f32 {
        ((row * 7 + inner * 3) % 11) as f32 * 0.25 - 1.0
    }
    fn b_val(inner: usize, col: usize) -> f32 {
        ((inner * 5 + col * 13) % 9) as f32 * 0.5 - 2.0
    }

    fn reference(m: usize, k: usize, c: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        let mut out = vec![0.0; m * 8];
        for row in 0..m {
            for col in 0..8 {
                let mut acc = c(row, col);
                for inner in 0..k {
                    acc = a_val(row, inner).mul_add(b_val(inner, col), acc);
                }
                out[row * 8 + col] = acc;
            }
        }
        out
    }

    #[test]
    fn f32_b16_scatters_ptx_fragments_and_accumulates_c() {
        for (k, dtype) in [(8, MatrixB16Type::Fp16), (16, MatrixB16Type::Bf16)] {
            let encode = |v: f32| match dtype {
                MatrixB16Type::Fp16 => f32_to_fp16_bits(v),
                MatrixB16Type::Bf16 => f32_to_bf16_bits(v),
            };
            let mut a = vec![[0_u32; 32]; k / 4];
            let mut b = vec![[0_u32; 32]; k / 8];
            for row in 0..16 {
                for inner in 0..k {
                    let (lane, slot) = mma_a_b16_owner(row, inner, k);
                    set_b16(&mut a, lane, slot, encode(a_val(row, inner)));
                }
            }
            for inner in 0..k {
                for col in 0..8 {
                    let (lane, slot) = mma_b_b16_owner(inner, col);
                    set_b16(&mut b, lane, slot, encode(b_val(inner, col)));
                }
            }
            let c_val = |row: usize, col: usize| (row * 8 + col) as f32 * 0.125;
            let mut c = vec![[0_u32; 32]; 4];
            for row in 0..16 {
                for col in 0..8 {
                    let (lane, slot) = mma_output_owner(row, col);
                    c[slot][lane] = c_val(row, col).to_bits();
                }
            }
            let d = mma_sync_f32_b16(&a, &b, Some(&c), k, dtype).unwrap();
            let expected = reference(16, k, c_val);
            assert_eq!(gather_mma_accumulator(16, &d), expected);
            let d_no_c = mma_sync_f32_b16(&a, &b, None, k, dtype).unwrap();
            assert_eq!(
                gather_mma_accumulator(16, &d_no_c),
                reference(16, k, |_, _| 0.0)
            );
        }
        assert!(
            mma_sync_f32_b16(&[[0; 32]; 2], &[[0; 32]; 2], None, 8, MatrixB16Type::Fp16).is_err()
        );
    }

    #[test]
    fn f16_accumulator_rounds_once_after_the_f32_chain() {
        // a = [1, 2^-11, 2^-11, 0...]·b = 1: f16 C=0. f32 chain gives
        // 1 + 2^-10, exactly representable in f16. A per-step f16 chain would
        // lose both halves-of-ULP and return 1.0.
        let k = 8;
        let mut a = vec![[0_u32; 32]; 2];
        let mut b = vec![[0_u32; 32]; 1];
        for (inner, value) in [(0, 1.0_f32), (1, 2.0_f32.powi(-11)), (2, 2.0_f32.powi(-11))] {
            let (lane, slot) = mma_a_b16_owner(0, inner, k);
            set_b16(&mut a, lane, slot, f32_to_fp16_bits(value));
            let (lane, slot) = mma_b_b16_owner(inner, 0);
            set_b16(&mut b, lane, slot, f32_to_fp16_bits(1.0));
        }
        let d = mma_sync_f16_f16(&a, &b, None, k).unwrap();
        let out = gather_packed_f16_accumulator(16, &d).unwrap();
        assert_eq!(out[0], 1.0 + 2.0_f32.powi(-10));
    }

    #[test]
    fn integer_saturation_and_b1_xor() {
        // m8n8k16 s8: a row 0 = all 127, b col 0 = all 127, C = i32::MAX.
        let mut a = vec![[0_u32; 32]; 1];
        let mut b = vec![[0_u32; 32]; 1];
        for inner in 0..16 {
            let (lane, register, element) = mma_packed_a_owner(8, 0, inner, 8);
            a[register][lane] |= 0x7f << (8 * element);
            let (lane, register, element) = mma_packed_b_owner(0, inner, 8);
            b[register][lane] |= 0x7f << (8 * element);
        }
        let mut c = vec![[0_u32; 32]; 2];
        let (lane, slot) = mma_output_owner_for_m(8, 0, 0);
        c[slot][lane] = i32::MAX as u32;
        let i8t = MatrixPackedIntType::I8;
        let sat = mma_sync_packed_integer(&a, &b, Some(&c), 8, 16, i8t, i8t, true, None).unwrap();
        assert_eq!(sat[slot][lane], i32::MAX);
        let wrap = mma_sync_packed_integer(&a, &b, Some(&c), 8, 16, i8t, i8t, false, None).unwrap();
        assert_eq!(wrap[slot][lane], i32::MAX.wrapping_add(16 * 127 * 127));

        // b1 m8n8k128: a row 0 = 0xffff_ffff in all words, b col 0 = 0x0000_ffff.
        let mut a = vec![[0_u32; 32]; 1];
        let mut b = vec![[0_u32; 32]; 1];
        for inner in 0..128 {
            let (lane, register, element) = mma_packed_a_owner(8, 0, inner, 1);
            a[register][lane] |= 1 << element;
            if inner % 32 < 16 {
                let (lane, register, element) = mma_packed_b_owner(0, inner, 1);
                b[register][lane] |= 1 << element;
            }
        }
        let b1 = MatrixPackedIntType::B1;
        let xor =
            mma_sync_packed_integer(&a, &b, None, 8, 128, b1, b1, false, Some(MatrixBitOp::Xor))
                .unwrap();
        let and =
            mma_sync_packed_integer(&a, &b, None, 8, 128, b1, b1, false, Some(MatrixBitOp::And))
                .unwrap();
        let (lane, slot) = mma_output_owner_for_m(8, 0, 0);
        assert_eq!(xor[slot][lane], 64);
        assert_eq!(and[slot][lane], 64);
        let (lane, slot) = mma_output_owner_for_m(8, 0, 1);
        assert_eq!(xor[slot][lane], 128);
        assert!(mma_sync_packed_integer(
            &a,
            &b,
            None,
            8,
            128,
            b1,
            b1,
            true,
            Some(MatrixBitOp::And)
        )
        .is_err());
    }

    #[test]
    fn f64_m8n8k4_uses_one_chain() {
        let big = 2.0_f64.powi(53);
        let mut a = vec![[0.0_f64; 32]; 1];
        let mut b = vec![[0.0_f64; 32]; 1];
        for (inner, (av, bv)) in [(1.0, big), (1.0, 1.0), (1.0, -big), (0.0, 0.0)]
            .into_iter()
            .enumerate()
        {
            let (lane, slot) = mma_a_f64_owner(8, 0, inner);
            a[slot][lane] = av;
            let (lane, slot) = mma_b_tf32_owner(0, inner);
            b[slot][lane] = bv;
        }
        let d = mma_sync_f64(&a, &b, None, 8, 4, F32RoundingMode::Nearest).unwrap();
        assert_eq!(d.len(), 2);
        let (lane, slot) = mma_output_owner_for_m(8, 0, 0);
        assert_eq!(d[slot][lane], 0.0);
    }

    #[test]
    fn m8n8k4_f32_output_matches_reference_per_quad_pair() {
        let mut a = vec![[0_u32; 32]; 2];
        let mut b = vec![[0_u32; 32]; 2];
        for computation in 0..4 {
            for row in 0..8 {
                for inner in 0..4 {
                    let (lane, slot) =
                        m8n8k4_f16_a_owner(computation, row, inner, MatrixLayout::Row);
                    set_b16(
                        &mut a,
                        lane,
                        slot,
                        f32_to_fp16_bits(a_val(row + computation, inner)),
                    );
                }
            }
            for col in 0..8 {
                for inner in 0..4 {
                    let (lane, slot) =
                        m8n8k4_f16_b_owner(computation, inner, col, MatrixLayout::Col);
                    set_b16(&mut b, lane, slot, f32_to_fp16_bits(b_val(inner, col)));
                }
            }
        }
        let MmaFloatOutput::F32(d) = mma_sync_m8n8k4_f16(
            &a,
            &b,
            None,
            MatrixLayout::Row,
            MatrixLayout::Col,
            MatrixAccumulatorType::Fp32,
            MatrixAccumulatorType::Fp32,
        )
        .unwrap() else {
            panic!("expected f32 output");
        };
        for computation in 0..4 {
            for row in 0..8 {
                for col in 0..8 {
                    let mut acc = 0.0_f32;
                    for inner in 0..4 {
                        acc = a_val(row + computation, inner).mul_add(b_val(inner, col), acc);
                    }
                    let (lane, slot) = m8n8k4_f32_acc_owner(computation, row, col).unwrap();
                    assert_eq!(d[slot][lane], acc);
                }
            }
        }
    }
}
