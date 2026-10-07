//! `mma.sync` register-fragment layouts: element coordinate <-> (lane, slot)
//! owner maps, operand decoders and accumulator gather/scatter.
//!
//! Legacy source: `engine-rs/src/runtime/matrix_ops.rs` (the dtype enums,
//! `decode_*`, `packed_*_value`, `gather_*`, `matrix_output_registers`,
//! `mma_*_owner` and `m8n8k4_*_owner` helpers, and the word packing of
//! `store_packed_f16_registers`). Engine stores were dropped; callers receive
//! register values and write them themselves.

use crate::cvt::{
    bf16_bits_to_f32, f32_to_fp16_bits, float8_e4m3fn_bits_to_f32, fp16_bits_to_f32,
    narrow_float_bits_to_f32_checked, FLOAT8_E5M2,
};
use crate::types::{OpError, OpResult, WarpValue, WARP_SIZE};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixB16Type {
    Fp16,
    Bf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixLayout {
    Row,
    Col,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixAccumulatorType {
    Fp16,
    Fp32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixPackedIntType {
    I8,
    U8,
    I4,
    U4,
    B1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixBitOp {
    Xor,
    And,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixF8Type {
    E4M3,
    E5M2,
}

pub fn decode_b16(dtype: MatrixB16Type, bits: u16) -> f32 {
    match dtype {
        MatrixB16Type::Fp16 => fp16_bits_to_f32(bits),
        MatrixB16Type::Bf16 => bf16_bits_to_f32(bits),
    }
}

pub fn decode_f8(dtype: MatrixF8Type, bits: u8) -> f32 {
    match dtype {
        MatrixF8Type::E4M3 => float8_e4m3fn_bits_to_f32(bits),
        // The shared codec reports the NaN encodings as `None`; matrix operands
        // carry them through as a quiet NaN, which is what the hand-written
        // E5M2 decoder this replaced did for all 256 payloads.
        MatrixF8Type::E5M2 => {
            narrow_float_bits_to_f32_checked(bits, FLOAT8_E5M2).unwrap_or(f32::NAN)
        }
    }
}

pub fn matrix_packed_int_bits(dtype: MatrixPackedIntType) -> usize {
    match dtype {
        MatrixPackedIntType::I8 | MatrixPackedIntType::U8 => 8,
        MatrixPackedIntType::I4 | MatrixPackedIntType::U4 => 4,
        MatrixPackedIntType::B1 => 1,
    }
}

/// Decode every lane of every register word.
pub fn decode_fragment<T: Copy>(
    registers: &[WarpValue<u32>],
    decode: impl Fn(u32) -> T,
) -> Vec<WarpValue<T>> {
    registers
        .iter()
        .map(|register| std::array::from_fn(|lane| decode(register[lane])))
        .collect()
}

/// The `element_slot`-th b16 element of `lane` (two per 32-bit register,
/// low half first).
pub fn packed_b16_value(
    registers: &[WarpValue<u32>],
    lane: usize,
    element_slot: usize,
    dtype: MatrixB16Type,
) -> OpResult<f32> {
    let word = registers
        .get(element_slot / 2)
        .ok_or_else(|| OpError::message("matrix fragment register index is outside operand"))?
        [lane];
    Ok(decode_b16(
        dtype,
        ((word >> (16 * (element_slot % 2))) & 0xffff) as u16,
    ))
}

/// The `element_slot`-th 8-bit float of `lane` (four per register, low byte
/// first).
pub fn packed_f8_value(
    registers: &[WarpValue<u32>],
    lane: usize,
    element_slot: usize,
    dtype: MatrixF8Type,
) -> OpResult<f32> {
    let word = registers
        .get(element_slot / 4)
        .ok_or_else(|| OpError::message("matrix fragment register index is outside operand"))?
        [lane];
    Ok(decode_f8(
        dtype,
        ((word >> (8 * (element_slot % 4))) & 0xff) as u8,
    ))
}

/// One packed integer element, sign-extended for `I8`/`I4`.
pub fn packed_integer_value(
    registers: &[WarpValue<u32>],
    lane: usize,
    register_slot: usize,
    element_in_register: usize,
    dtype: MatrixPackedIntType,
) -> OpResult<i32> {
    let bits = matrix_packed_int_bits(dtype);
    let word = registers
        .get(register_slot)
        .ok_or_else(|| OpError::message("matrix fragment register index is outside operand"))?
        [lane];
    let mask = (1_u32 << bits) - 1;
    let raw = (word >> (bits * element_in_register)) & mask;
    Ok(match dtype {
        MatrixPackedIntType::I8 => i32::from(raw as u8 as i8),
        MatrixPackedIntType::U8 => raw as i32,
        MatrixPackedIntType::I4 => {
            if raw & 0x8 != 0 {
                raw as i32 - 16
            } else {
                raw as i32
            }
        }
        MatrixPackedIntType::U4 | MatrixPackedIntType::B1 => raw as i32,
    })
}

/// Row-major `rows x columns` gather.
pub fn gather_matrix<T>(
    rows: usize,
    columns: usize,
    mut value_at: impl FnMut(usize, usize) -> OpResult<T>,
) -> OpResult<Vec<T>> {
    let mut values = Vec::with_capacity(rows.saturating_mul(columns));
    for row in 0..rows {
        for column in 0..columns {
            values.push(value_at(row, column)?);
        }
    }
    Ok(values)
}

/// Gather an `m x 8` accumulator (one element per logical register slot).
pub fn gather_mma_accumulator<T: Copy>(m: usize, registers: &[WarpValue<T>]) -> Vec<T> {
    let mut values = Vec::with_capacity(m * 8);
    for row in 0..m {
        for column in 0..8 {
            let (lane, slot) = mma_output_owner_for_m(m, row, column);
            values.push(registers[slot][lane]);
        }
    }
    values
}

/// Gather an `m x 8` accumulator whose f16 elements are packed two per word.
pub fn gather_packed_f16_accumulator(m: usize, registers: &[WarpValue<u32>]) -> OpResult<Vec<f32>> {
    gather_matrix(m, 8, |row, column| {
        let (lane, slot) = mma_output_owner_for_m(m, row, column);
        packed_b16_value(registers, lane, slot, MatrixB16Type::Fp16)
    })
}

/// Scatter a row-major `m x 8` result into `register_count` logical registers.
pub fn matrix_output_registers<T: Copy>(
    m: usize,
    register_count: usize,
    zero: T,
    values: &[T],
) -> OpResult<Vec<WarpValue<T>>> {
    if values.len() != m * 8 {
        return Err(OpError::message(
            "matrix numeric core returned the wrong output shape",
        ));
    }
    let mut registers = vec![[zero; WARP_SIZE]; register_count];
    for row in 0..m {
        for column in 0..8 {
            let (lane, slot) = mma_output_owner_for_m(m, row, column);
            registers[slot][lane] = values[row * 8 + column];
        }
    }
    Ok(registers)
}

/// Pack logical f32 registers pairwise into f16x2 words (`2r` low, `2r+1`
/// high), rounding each value with `f32_to_fp16_bits` (RNE). This is the
/// numeric part of legacy `store_packed_f16_registers`.
pub fn pack_f16_registers(values: &[WarpValue<f32>]) -> OpResult<Vec<WarpValue<u32>>> {
    if values.len() % 2 != 0 {
        return Err(OpError::message(
            "packed f16 matrix output has the wrong logical register count",
        ));
    }
    Ok(values
        .chunks_exact(2)
        .map(|pair| {
            std::array::from_fn(|lane| {
                u32::from(f32_to_fp16_bits(pair[0][lane]))
                    | (u32::from(f32_to_fp16_bits(pair[1][lane])) << 16)
            })
        })
        .collect())
}

/// Accumulator (C/D) owner for `m16n8`: `(lane, slot)`.
pub fn mma_output_owner(row: usize, col: usize) -> (usize, usize) {
    (
        4 * (row % 8) + (col % 8) / 2,
        4 * (col / 8) + 2 * (row / 8) + col % 2,
    )
}

/// Accumulator owner for `m8n8` (`m == 8`) or `m16n8`.
pub fn mma_output_owner_for_m(m: usize, row: usize, col: usize) -> (usize, usize) {
    if m == 8 {
        (4 * row + col / 2, col % 2)
    } else {
        mma_output_owner(row, col)
    }
}

/// Packed sub-word A operand owner: `(lane, register, element_in_register)`.
pub fn mma_packed_a_owner(
    m: usize,
    row: usize,
    inner: usize,
    bits: usize,
) -> (usize, usize, usize) {
    let elements_per_register = 32 / bits;
    if m == 8 {
        (
            4 * row + inner / elements_per_register,
            0,
            inner % elements_per_register,
        )
    } else {
        let k_group = 4 * elements_per_register;
        (
            4 * (row % 8) + (inner % k_group) / elements_per_register,
            2 * (inner / k_group) + row / 8,
            inner % elements_per_register,
        )
    }
}

/// Packed sub-word B operand owner: `(lane, register, element_in_register)`.
pub fn mma_packed_b_owner(col: usize, inner: usize, bits: usize) -> (usize, usize, usize) {
    let elements_per_register = 32 / bits;
    let k_group = 4 * elements_per_register;
    (
        4 * col + (inner % k_group) / elements_per_register,
        inner / k_group,
        inner % elements_per_register,
    )
}

/// b16 A owner for `m16n8k{8,16}`: `(lane, b16 element slot)`.
pub fn mma_a_b16_owner(row: usize, col: usize, k: usize) -> (usize, usize) {
    if k == 8 {
        (4 * (row % 8) + col / 2, 2 * (row / 8) + col % 2)
    } else {
        (
            4 * (row % 8) + (col % 8) / 2,
            4 * (col / 8) + 2 * (row / 8) + col % 2,
        )
    }
}

/// b16 B owner (`row` = K index, `col` = N index): `(lane, b16 element slot)`.
pub fn mma_b_b16_owner(row: usize, col: usize) -> (usize, usize) {
    (4 * col + (row % 8) / 2, 2 * (row / 8) + row % 2)
}

/// TF32 A owner for `m16n8k{4,8}`: `(lane, register)`.
pub fn mma_a_tf32_owner(row: usize, inner: usize) -> (usize, usize) {
    (4 * (row % 8) + inner % 4, 2 * (inner / 4) + row / 8)
}

/// TF32/f64 B owner: `(lane, register)`.
pub fn mma_b_tf32_owner(column: usize, inner: usize) -> (usize, usize) {
    (4 * column + inner % 4, inner / 4)
}

/// f64 A owner for `m8n8k4` and `m16n8k{4,8,16}`: `(lane, register)`.
pub fn mma_a_f64_owner(m: usize, row: usize, inner: usize) -> (usize, usize) {
    if m == 8 {
        (4 * row + inner, 0)
    } else {
        (4 * (row % 8) + inner % 4, 2 * (inner / 4) + row / 8)
    }
}

/// `m8n8k4` f16 A owner for one of the four quad-pair computations.
pub fn m8n8k4_f16_a_owner(
    computation: usize,
    row: usize,
    inner: usize,
    layout: MatrixLayout,
) -> (usize, usize) {
    let high = row / 4;
    match layout {
        MatrixLayout::Row => (16 * high + 4 * computation + row % 4, inner),
        MatrixLayout::Col => (16 * high + 4 * computation + inner, row % 4),
    }
}

/// `m8n8k4` f16 B owner.
pub fn m8n8k4_f16_b_owner(
    computation: usize,
    inner: usize,
    col: usize,
    layout: MatrixLayout,
) -> (usize, usize) {
    let high = col / 4;
    match layout {
        MatrixLayout::Row => (16 * high + 4 * computation + inner, col % 4),
        MatrixLayout::Col => (16 * high + 4 * computation + col % 4, inner),
    }
}

/// `m8n8k4` packed-f16 accumulator owner `(lane, f16 slot)`.
pub fn m8n8k4_f16_acc_owner(computation: usize, row: usize, col: usize) -> (usize, usize) {
    (16 * (row / 4) + 4 * computation + row % 4, col)
}

/// `m8n8k4` f32 accumulator owner `(lane, register)`.
pub fn m8n8k4_f32_acc_owner(
    computation: usize,
    row: usize,
    col: usize,
) -> OpResult<(usize, usize)> {
    let high = row / 4;
    for thread in 0..4 {
        for slot in 0..8 {
            let mapped_row = (thread & 1) + (slot & 2) + 4 * high;
            let mapped_col = (slot & 4) + (thread & 2) + (slot & 1);
            if mapped_row == row && mapped_col == col {
                return Ok((16 * high + 4 * computation + thread, slot));
            }
        }
    }
    Err(OpError::message(
        "m8n8k4 f32 accumulator coordinate has no register owner",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn assert_bijective(pairs: impl Iterator<Item = (usize, usize)>, expected: usize) {
        let set: HashSet<_> = pairs.collect();
        assert_eq!(set.len(), expected);
    }

    #[test]
    fn m16n8_accumulator_owner_is_bijective() {
        assert_bijective(
            (0..16).flat_map(|r| (0..8).map(move |c| mma_output_owner_for_m(16, r, c))),
            128,
        );
        assert_bijective(
            (0..8).flat_map(|r| (0..8).map(move |c| mma_output_owner_for_m(8, r, c))),
            64,
        );
        assert_eq!(mma_output_owner_for_m(16, 9, 3), (5, 3));
    }

    #[test]
    fn b16_owners_are_bijective_and_ptx_shaped() {
        for k in [8, 16] {
            assert_bijective(
                (0..16).flat_map(|r| (0..k).map(move |c| mma_a_b16_owner(r, c, k))),
                16 * k,
            );
            assert_bijective(
                (0..k).flat_map(|r| (0..8).map(move |c| mma_b_b16_owner(r, c))),
                8 * k,
            );
        }
        // PTX m16n8k16 A: a0,a1 row=groupID, col=2*tid+{0,1}; a2,a3 row+8.
        assert_eq!(mma_a_b16_owner(0, 1, 16), (0, 1));
        assert_eq!(mma_a_b16_owner(8, 0, 16), (0, 2));
        assert_eq!(mma_a_b16_owner(0, 8, 16), (0, 4));
    }

    #[test]
    fn m8n8k4_f32_owner_covers_every_cell_per_computation() {
        for computation in 0..4 {
            assert_bijective(
                (0..8).flat_map(|r| {
                    (0..8).map(move |c| m8n8k4_f32_acc_owner(computation, r, c).unwrap())
                }),
                64,
            );
        }
    }

    #[test]
    fn packed_integer_decoding_sign_extends() {
        let mut word = [0_u32; 32];
        word[3] = 0x0000_f0ff;
        let regs = [word];
        assert_eq!(
            packed_integer_value(&regs, 3, 0, 0, MatrixPackedIntType::I8).unwrap(),
            -1
        );
        assert_eq!(
            packed_integer_value(&regs, 3, 0, 0, MatrixPackedIntType::U8).unwrap(),
            255
        );
        assert_eq!(
            packed_integer_value(&regs, 3, 0, 3, MatrixPackedIntType::I4).unwrap(),
            -1
        );
        assert_eq!(
            packed_integer_value(&regs, 3, 0, 2, MatrixPackedIntType::U4).unwrap(),
            0
        );
        assert_eq!(
            packed_integer_value(&regs, 3, 0, 15, MatrixPackedIntType::B1).unwrap(),
            1
        );
        assert!(packed_integer_value(&regs, 3, 1, 0, MatrixPackedIntType::I8).is_err());
    }

    #[test]
    fn e5m2_nan_payloads_decode_to_nan() {
        assert!(decode_f8(MatrixF8Type::E5M2, 0x7f).is_nan());
        assert_eq!(decode_f8(MatrixF8Type::E5M2, 0x3c), 1.0);
        assert_eq!(decode_f8(MatrixF8Type::E4M3, 0x38), 1.0);
    }

    #[test]
    fn f16_packing_places_even_slot_low() {
        let lo = [1.0_f32; 32];
        let hi = [-2.0_f32; 32];
        let words = pack_f16_registers(&[lo, hi]).unwrap();
        assert_eq!(words[0][7], 0xc000_3c00);
        assert!(pack_f16_registers(&[lo]).is_err());
    }
}
