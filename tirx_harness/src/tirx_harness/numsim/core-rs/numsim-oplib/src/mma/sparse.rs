//! Sparse `mma.sp.sync` (2:4 structured sparsity) numerics.
//!
//! Legacy source: `engine-rs/src/runtime/matrix_ops.rs` (`sparse_*`,
//! `validate_sparse_type_pair`, `read_sparse_*`, `raw_mma_sp_sync`). The
//! metadata register is passed in as a plain warp value; the engine keeps the
//! full-warp check and the stores.
//!
//! Numerics: per output row the compressed A elements are paired with the B
//! rows named by the metadata, and the product runs as one increasing-term FMA
//! chain (binary32) or exact i64 sum (integers) in compressed order starting
//! from C. f16 C/D are widened / rounded once like dense `mma.sync`.

use super::backend::narrow_i64_accumulator;
use super::fragments::*;
use super::sync::{mma_f32, MmaFloatOutput};
use crate::cvt::f32_to_tf32;
use crate::types::{OpError, OpResult, WarpMask, WarpValue};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixSparseOperandType {
    Fp16,
    Bf16,
    Tf32,
    I8,
    U8,
    I4,
    U4,
    E4M3,
    E5M2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixSparseAccumulatorType {
    Fp16,
    Fp32,
    I32,
}

pub fn sparse_chunk_width(dtype: MatrixSparseOperandType) -> usize {
    match dtype {
        MatrixSparseOperandType::Tf32 => 2,
        MatrixSparseOperandType::I4 | MatrixSparseOperandType::U4 => 8,
        _ => 4,
    }
}

pub fn sparse_stored_per_chunk(dtype: MatrixSparseOperandType) -> usize {
    match dtype {
        MatrixSparseOperandType::Tf32 => 1,
        MatrixSparseOperandType::I4 | MatrixSparseOperandType::U4 => 4,
        _ => 2,
    }
}

/// `(A registers, B registers)` for one sparse form.
pub fn sparse_fragment_counts(
    k: usize,
    dtype: MatrixSparseOperandType,
) -> OpResult<(usize, usize)> {
    let counts = match dtype {
        MatrixSparseOperandType::Fp16 | MatrixSparseOperandType::Bf16 => {
            if matches!(k, 16 | 32) {
                (k / 8, k / 8)
            } else {
                return Err(OpError::message("sparse b16 MMA requires K=16 or K=32"));
            }
        }
        MatrixSparseOperandType::Tf32 => {
            if matches!(k, 8 | 16) {
                (k / 4, k / 4)
            } else {
                return Err(OpError::message("sparse TF32 MMA requires K=8 or K=16"));
            }
        }
        MatrixSparseOperandType::I8 | MatrixSparseOperandType::U8 => {
            if matches!(k, 32 | 64) {
                (k / 16, k / 16)
            } else {
                return Err(OpError::message("sparse int8 MMA requires K=32 or K=64"));
            }
        }
        MatrixSparseOperandType::I4 | MatrixSparseOperandType::U4 => {
            if matches!(k, 64 | 128) {
                (k / 32, k / 32)
            } else {
                return Err(OpError::message("sparse int4 MMA requires K=64 or K=128"));
            }
        }
        MatrixSparseOperandType::E4M3 | MatrixSparseOperandType::E5M2 => {
            if k == 64 {
                (4, 4)
            } else {
                return Err(OpError::message("sparse FP8 MMA requires K=64"));
            }
        }
    };
    Ok(counts)
}

pub fn validate_sparse_type_pair(
    a_dtype: MatrixSparseOperandType,
    b_dtype: MatrixSparseOperandType,
    accumulator_dtype: MatrixSparseAccumulatorType,
    saturate: bool,
) -> OpResult<()> {
    use MatrixSparseAccumulatorType as Acc;
    use MatrixSparseOperandType as Op;
    let valid = match (a_dtype, b_dtype, accumulator_dtype) {
        (Op::Fp16, Op::Fp16, Acc::Fp16 | Acc::Fp32) => true,
        (Op::Bf16, Op::Bf16, Acc::Fp32) | (Op::Tf32, Op::Tf32, Acc::Fp32) => true,
        (Op::I8 | Op::U8, Op::I8 | Op::U8, Acc::I32)
        | (Op::I4 | Op::U4, Op::I4 | Op::U4, Acc::I32) => true,
        (Op::E4M3 | Op::E5M2, Op::E4M3 | Op::E5M2, Acc::Fp32) => true,
        _ => false,
    };
    if !valid {
        return Err(OpError::message(
            "sparse MMA operand and accumulator types are incompatible",
        ));
    }
    if saturate && !matches!(a_dtype, Op::I8 | Op::U8 | Op::I4 | Op::U4) {
        return Err(OpError::message(
            "sparse MMA saturation requires integer multiplicands",
        ));
    }
    Ok(())
}

/// Compressed-A owner `(lane, element slot)` of stored element `packed` of
/// 2:4 chunk `chunk` in `row`.
pub fn sparse_a_owner(
    row: usize,
    chunk: usize,
    packed: usize,
    dtype: MatrixSparseOperandType,
) -> (usize, usize) {
    let group = row % 8;
    let row_high = row / 8;
    match dtype {
        MatrixSparseOperandType::Fp16 | MatrixSparseOperandType::Bf16 => {
            let block = chunk / 4;
            (4 * group + chunk % 4, 4 * block + 2 * row_high + packed)
        }
        MatrixSparseOperandType::Tf32 => {
            let block = chunk / 4;
            (4 * group + chunk % 4, 2 * block + row_high)
        }
        MatrixSparseOperandType::I8
        | MatrixSparseOperandType::U8
        | MatrixSparseOperandType::E4M3
        | MatrixSparseOperandType::E5M2 => {
            let block = chunk / 8;
            let within = chunk % 8;
            (
                4 * group + within / 2,
                8 * block + 4 * row_high + 2 * (within % 2) + packed,
            )
        }
        MatrixSparseOperandType::I4 | MatrixSparseOperandType::U4 => {
            let block = chunk / 8;
            let within = chunk % 8;
            (
                4 * group + within / 2,
                16 * block + 8 * row_high + 4 * (within % 2) + packed,
            )
        }
    }
}

fn f8_type(dtype: MatrixSparseOperandType) -> MatrixF8Type {
    if dtype == MatrixSparseOperandType::E4M3 {
        MatrixF8Type::E4M3
    } else {
        MatrixF8Type::E5M2
    }
}

fn read_sparse_operand(
    registers: &[WarpValue<u32>],
    lane: usize,
    element_slot: usize,
    dtype: MatrixSparseOperandType,
) -> OpResult<f32> {
    match dtype {
        MatrixSparseOperandType::Fp16 => {
            packed_b16_value(registers, lane, element_slot, MatrixB16Type::Fp16)
        }
        MatrixSparseOperandType::Bf16 => {
            packed_b16_value(registers, lane, element_slot, MatrixB16Type::Bf16)
        }
        MatrixSparseOperandType::Tf32 => {
            let register = registers.get(element_slot).ok_or_else(|| {
                OpError::message("sparse TF32 fragment register index is outside operand")
            })?;
            Ok(f32_to_tf32(f32::from_bits(register[lane])))
        }
        MatrixSparseOperandType::I8
        | MatrixSparseOperandType::U8
        | MatrixSparseOperandType::I4
        | MatrixSparseOperandType::U4 => Err(OpError::message(
            "internal error: integer sparse operand requested as float",
        )),
        MatrixSparseOperandType::E4M3 | MatrixSparseOperandType::E5M2 => {
            packed_f8_value(registers, lane, element_slot, f8_type(dtype))
        }
    }
}

fn sparse_packed_int_type(dtype: MatrixSparseOperandType) -> OpResult<MatrixPackedIntType> {
    match dtype {
        MatrixSparseOperandType::I8 => Ok(MatrixPackedIntType::I8),
        MatrixSparseOperandType::U8 => Ok(MatrixPackedIntType::U8),
        MatrixSparseOperandType::I4 => Ok(MatrixPackedIntType::I4),
        MatrixSparseOperandType::U4 => Ok(MatrixPackedIntType::U4),
        _ => Err(OpError::message(
            "internal error: non-integer sparse operand requested as integer",
        )),
    }
}

fn read_sparse_integer_operand(
    registers: &[WarpValue<u32>],
    lane: usize,
    element_slot: usize,
    dtype: MatrixSparseOperandType,
) -> OpResult<i32> {
    let packed_dtype = sparse_packed_int_type(dtype)?;
    let bits = matrix_packed_int_bits(packed_dtype);
    let elements_per_register = 32 / bits;
    packed_integer_value(
        registers,
        lane,
        element_slot / elements_per_register,
        element_slot % elements_per_register,
        packed_dtype,
    )
}

fn sparse_b_float(
    registers: &[WarpValue<u32>],
    inner: usize,
    col: usize,
    dtype: MatrixSparseOperandType,
) -> OpResult<f32> {
    match dtype {
        MatrixSparseOperandType::Fp16 | MatrixSparseOperandType::Bf16 => {
            let (lane, slot) = mma_b_b16_owner(inner, col);
            read_sparse_operand(registers, lane, slot, dtype)
        }
        MatrixSparseOperandType::Tf32 => {
            let (lane, slot) = mma_b_tf32_owner(col, inner);
            read_sparse_operand(registers, lane, slot, dtype)
        }
        MatrixSparseOperandType::E4M3 | MatrixSparseOperandType::E5M2 => {
            let (lane, register, element) = mma_packed_b_owner(col, inner, 8);
            packed_f8_value(
                &registers[register..=register],
                lane,
                element,
                f8_type(dtype),
            )
        }
        _ => Err(OpError::message(
            "internal error: integer sparse B requested as float",
        )),
    }
}

fn sparse_b_integer(
    registers: &[WarpValue<u32>],
    inner: usize,
    col: usize,
    dtype: MatrixSparseOperandType,
) -> OpResult<i32> {
    let packed_dtype = sparse_packed_int_type(dtype)?;
    let bits = matrix_packed_int_bits(packed_dtype);
    let (lane, register, element) = mma_packed_b_owner(col, inner, bits);
    packed_integer_value(registers, lane, register, element, packed_dtype)
}

/// `(lane, 4-bit code index)` of the metadata code for `(row, chunk)`.
pub fn sparse_metadata_location(
    selector: usize,
    row: usize,
    chunk: usize,
    chunks_per_row: usize,
) -> OpResult<(usize, usize)> {
    let group = row % 8;
    let row_high = row / 8;
    let (lane, code_index) = match chunks_per_row {
        4 => {
            if selector > 3 {
                return Err(OpError::message(
                    "sparse MMA single-thread selector must be in 0..=3",
                ));
            }
            (4 * group + selector, 4 * row_high + chunk)
        }
        8 => {
            if selector > 1 {
                return Err(OpError::message(
                    "sparse MMA thread-pair selector must be 0 or 1",
                ));
            }
            (
                4 * group + 2 * selector + chunk / 4,
                4 * row_high + chunk % 4,
            )
        }
        16 => {
            if selector != 0 {
                return Err(OpError::message(
                    "sparse MMA all-thread metadata requires selector 0",
                ));
            }
            (4 * group + chunk / 4, 4 * row_high + chunk % 4)
        }
        _ => {
            return Err(OpError::message(
                "sparse MMA metadata chunk geometry is unsupported",
            ));
        }
    };
    Ok((lane, code_index))
}

/// Lanes whose metadata register this sparse form reads.
pub fn sparse_metadata_source_mask(
    k: usize,
    dtype: MatrixSparseOperandType,
    selector: usize,
) -> OpResult<WarpMask> {
    let chunks_per_row = k / sparse_chunk_width(dtype);
    let mut bits = 0;
    for row in 0..16 {
        for chunk in 0..chunks_per_row {
            let (lane, _) = sparse_metadata_location(selector, row, chunk, chunks_per_row)?;
            bits |= 1 << lane;
        }
    }
    Ok(WarpMask(bits))
}

fn sparse_metadata_code(
    metadata: &WarpValue<u32>,
    selector: usize,
    row: usize,
    chunk: usize,
    chunks_per_row: usize,
) -> OpResult<u8> {
    let (lane, code_index) = sparse_metadata_location(selector, row, chunk, chunks_per_row)?;
    let word = metadata[lane];
    Ok(((word >> (4 * code_index)) & 0xf) as u8)
}

/// Dense position (within the chunk) of stored element `packed`.
pub fn sparse_dense_position(
    dtype: MatrixSparseOperandType,
    code: u8,
    packed: usize,
    ordered_metadata: bool,
) -> OpResult<usize> {
    if dtype == MatrixSparseOperandType::Tf32 {
        return match code {
            0x4 => Ok(0),
            0xe => Ok(1),
            _ => Err(OpError::message(format!(
                "sparse TF32 metadata code 0x{code:x} is invalid"
            ))),
        };
    }
    let first = usize::from(code & 0x3);
    let second = usize::from((code >> 2) & 0x3);
    if first == second {
        return Err(OpError::message(format!(
            "sparse MMA metadata code 0x{code:x} repeats one position"
        )));
    }
    if ordered_metadata && first > second {
        return Err(OpError::message(format!(
            "ordered sparse MMA metadata code 0x{code:x} has descending indices"
        )));
    }
    if matches!(
        dtype,
        MatrixSparseOperandType::I4 | MatrixSparseOperandType::U4
    ) {
        let pair = if packed < 2 { first } else { second };
        Ok(2 * pair + packed % 2)
    } else {
        Ok(if packed == 0 { first } else { second })
    }
}

/// D registers of `mma.sp.sync` (`F32`/`PackedF16` reuse `MmaFloatOutput`).
#[derive(Clone, Debug, PartialEq)]
pub enum MmaSparseOutput {
    Float(MmaFloatOutput),
    I32(Vec<WarpValue<i32>>),
}

/// `mma.sp{::ordered_metadata}.sync.aligned.m16n8k*`. `accumulator` holds C
/// (2 packed-f16 or 4 f32/s32 registers); D has the same register count.
#[allow(clippy::too_many_arguments)]
pub fn mma_sp_sync(
    a: &[WarpValue<u32>],
    b: &[WarpValue<u32>],
    accumulator: &[WarpValue<u32>],
    metadata: &WarpValue<u32>,
    selector: usize,
    k: usize,
    a_dtype: MatrixSparseOperandType,
    b_dtype: MatrixSparseOperandType,
    accumulator_dtype: MatrixSparseAccumulatorType,
    saturate: bool,
    ordered_metadata: bool,
) -> OpResult<MmaSparseOutput> {
    validate_sparse_type_pair(a_dtype, b_dtype, accumulator_dtype, saturate)?;
    let (expected_a, expected_b) = sparse_fragment_counts(k, a_dtype)?;
    let (_, expected_b_for_type) = sparse_fragment_counts(k, b_dtype)?;
    let expected_c = match accumulator_dtype {
        MatrixSparseAccumulatorType::Fp16 => 2,
        MatrixSparseAccumulatorType::Fp32 | MatrixSparseAccumulatorType::I32 => 4,
    };
    if a.len() != expected_a
        || b.len() != expected_b
        || expected_b != expected_b_for_type
        || accumulator.len() != expected_c
    {
        return Err(OpError::message(
            "sparse MMA fragment register counts are invalid",
        ));
    }
    let chunk_width = sparse_chunk_width(a_dtype);
    let chunks_per_row = k / chunk_width;
    let stored_per_chunk = sparse_stored_per_chunk(a_dtype);
    let compressed_k = chunks_per_row * stored_per_chunk;
    if accumulator_dtype == MatrixSparseAccumulatorType::I32 {
        let accumulator_registers = decode_fragment(accumulator, |bits| bits as i32);
        let mut output = gather_mma_accumulator(16, &accumulator_registers)
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>();
        for row in 0..16 {
            let mut a_values = Vec::with_capacity(compressed_k);
            let mut dense_inner = Vec::with_capacity(compressed_k);
            for chunk in 0..chunks_per_row {
                let code = sparse_metadata_code(metadata, selector, row, chunk, chunks_per_row)?;
                for packed in 0..stored_per_chunk {
                    let (lane, slot) = sparse_a_owner(row, chunk, packed, a_dtype);
                    let position = sparse_dense_position(a_dtype, code, packed, ordered_metadata)?;
                    a_values.push(read_sparse_integer_operand(a, lane, slot, a_dtype)?);
                    dense_inner.push(chunk * chunk_width + position);
                }
            }
            let b_values = gather_matrix(8, compressed_k, |column, term| {
                sparse_b_integer(b, dense_inner[term], column, b_dtype)
            })?;
            crate::fpenv::multiply_accumulate_i32_abt(
                1,
                8,
                compressed_k,
                &a_values,
                &b_values,
                &mut output[row * 8..(row + 1) * 8],
            )
            .map_err(|error| {
                OpError::message(format!("raw sparse MMA SIMD shape error: {error}"))
            })?;
        }
        let output = narrow_i64_accumulator(output, saturate);
        return Ok(MmaSparseOutput::I32(matrix_output_registers(
            16, 4, 0_i32, &output,
        )?));
    }

    let mut output = match accumulator_dtype {
        MatrixSparseAccumulatorType::Fp16 => gather_packed_f16_accumulator(16, accumulator)?,
        MatrixSparseAccumulatorType::Fp32 => {
            gather_mma_accumulator(16, &decode_fragment(accumulator, f32::from_bits))
        }
        MatrixSparseAccumulatorType::I32 => unreachable!(),
    };
    for row in 0..16 {
        let mut a_values = Vec::with_capacity(compressed_k);
        let mut dense_inner = Vec::with_capacity(compressed_k);
        for chunk in 0..chunks_per_row {
            let code = sparse_metadata_code(metadata, selector, row, chunk, chunks_per_row)?;
            for packed in 0..stored_per_chunk {
                let (lane, slot) = sparse_a_owner(row, chunk, packed, a_dtype);
                let position = sparse_dense_position(a_dtype, code, packed, ordered_metadata)?;
                a_values.push(read_sparse_operand(a, lane, slot, a_dtype)?);
                dense_inner.push(chunk * chunk_width + position);
            }
        }
        let b_values = gather_matrix(8, compressed_k, |column, term| {
            sparse_b_float(b, dense_inner[term], column, b_dtype)
        })?;
        let values = mma_f32(
            1,
            compressed_k,
            &a_values,
            &b_values,
            Some(&output[row * 8..(row + 1) * 8]),
        )?;
        output[row * 8..(row + 1) * 8].copy_from_slice(&values);
    }
    let registers = matrix_output_registers(16, 4, 0.0_f32, &output)?;
    Ok(MmaSparseOutput::Float(match accumulator_dtype {
        MatrixSparseAccumulatorType::Fp16 => {
            MmaFloatOutput::PackedF16(pack_f16_registers(&registers)?)
        }
        MatrixSparseAccumulatorType::Fp32 => MmaFloatOutput::F32(registers),
        MatrixSparseAccumulatorType::I32 => unreachable!(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cvt::f32_to_fp16_bits;

    #[test]
    fn metadata_source_masks_follow_the_selector_geometry() {
        // f16 k16: 4 chunks/row -> one thread per group.
        let mask = sparse_metadata_source_mask(16, MatrixSparseOperandType::Fp16, 1).unwrap();
        assert_eq!(mask.bits(), 0x2222_2222);
        // f16 k32: 8 chunks/row -> thread pair.
        let mask = sparse_metadata_source_mask(32, MatrixSparseOperandType::Fp16, 1).unwrap();
        assert_eq!(mask.bits(), 0xcccc_cccc);
        // int8 k64: 16 chunks/row -> all threads.
        let mask = sparse_metadata_source_mask(64, MatrixSparseOperandType::I8, 0).unwrap();
        assert_eq!(mask.bits(), u32::MAX);
        assert!(sparse_metadata_source_mask(16, MatrixSparseOperandType::Fp16, 4).is_err());
    }

    #[test]
    fn dense_positions_validate_codes() {
        use MatrixSparseOperandType as Op;
        assert_eq!(
            sparse_dense_position(Op::Fp16, 0b1101, 0, false).unwrap(),
            1
        );
        assert_eq!(
            sparse_dense_position(Op::Fp16, 0b1101, 1, false).unwrap(),
            3
        );
        assert_eq!(
            sparse_dense_position(Op::Fp16, 0b0111, 1, false).unwrap(),
            1
        );
        assert!(sparse_dense_position(Op::Fp16, 0b0111, 1, true).is_err());
        assert!(sparse_dense_position(Op::Fp16, 0b0101, 0, false).is_err());
        assert_eq!(sparse_dense_position(Op::I4, 0b1000, 3, false).unwrap(), 5);
        assert_eq!(sparse_dense_position(Op::Tf32, 0xe, 0, false).unwrap(), 1);
        assert!(sparse_dense_position(Op::Tf32, 0x5, 0, false).is_err());
    }

    #[test]
    fn sparse_f16_selects_the_metadata_rows_of_b() {
        // m16n8k16 f16 -> f32. Row 0 stores (1.0, 2.0) in chunk 0 at dense
        // positions 1 and 3; B[inner, 0] = inner + 1 so D[0,0] = 1*2 + 2*4.
        let k = 16;
        let a_dtype = MatrixSparseOperandType::Fp16;
        let mut a = vec![[0_u32; 32]; 2];
        for (packed, value) in [(0, 1.0_f32), (1, 2.0_f32)] {
            let (lane, slot) = sparse_a_owner(0, 0, packed, a_dtype);
            a[slot / 2][lane] |= u32::from(f32_to_fp16_bits(value)) << (16 * (slot % 2));
        }
        let mut b = vec![[0_u32; 32]; 2];
        for inner in 0..k {
            let (lane, slot) = mma_b_b16_owner(inner, 0);
            b[slot / 2][lane] |=
                u32::from(f32_to_fp16_bits(inner as f32 + 1.0)) << (16 * (slot % 2));
        }
        // Every chunk code 0b1101 (positions 1 and 3); invalid zeros elsewhere
        // would repeat a position, so fill every nibble.
        let metadata = [0xdddd_dddd_u32; 32];
        let c = vec![[0_u32; 32]; 4];
        let out = mma_sp_sync(
            &a,
            &b,
            &c,
            &metadata,
            0,
            k,
            a_dtype,
            a_dtype,
            MatrixSparseAccumulatorType::Fp32,
            false,
            true,
        )
        .unwrap();
        let MmaSparseOutput::Float(MmaFloatOutput::F32(d)) = out else {
            panic!("expected f32");
        };
        let (lane, slot) = mma_output_owner(0, 0);
        assert_eq!(d[slot][lane], 10.0);
        let (lane, slot) = mma_output_owner(1, 0);
        assert_eq!(d[slot][lane], 0.0);
    }

    #[test]
    fn sparse_type_pairs_are_checked() {
        use MatrixSparseAccumulatorType as Acc;
        use MatrixSparseOperandType as Op;
        assert!(validate_sparse_type_pair(Op::Bf16, Op::Bf16, Acc::Fp16, false).is_err());
        assert!(validate_sparse_type_pair(Op::I8, Op::U8, Acc::I32, true).is_ok());
        assert!(validate_sparse_type_pair(Op::Fp16, Op::Fp16, Acc::Fp32, true).is_err());
        assert_eq!(sparse_fragment_counts(64, Op::E4M3).unwrap(), (4, 4));
        assert!(sparse_fragment_counts(32, Op::E5M2).is_err());
    }
}
