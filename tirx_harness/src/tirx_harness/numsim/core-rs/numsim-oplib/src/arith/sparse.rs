//! Register sparse compression / decompression (`spcompress`, `spdecompress`)
//! on one lane's register vector.
//!
//! Moved from legacy `engine-rs/src/runtime/instructions/sparse_compress.rs`
//! (`compress_sparse_vector`, `sparse_pair_indices`, `spcompress_spec`) and
//! `reg.rs` (`ValidSpDecompressShape`, `valid_spdecompress_shapes!`,
//! `spdecompress_spec`).

use crate::cvt;
use crate::types::{OpError, OpResult};

/// Shared numerical primitive for register and TMEM sparse compression.
///
/// Metadata is packed low-bit first, in source-index order. PTX permits any
/// choice among tied candidates; the model deterministically picks lower
/// indices. Returns `(metadata words, compressed data words)`.
pub fn compress_sparse_vector(
    data: &[u32],
    elem_bits: usize,
    index_bits: usize,
    descriptor: u32,
) -> OpResult<(Vec<u32>, Vec<u32>)> {
    if !matches!(elem_bits, 8 | 16)
        || !matches!(index_bits, 2 | 4)
        || data.is_empty()
        || !data.len().is_multiple_of(2)
        || data.len() > 128
    {
        return Err(OpError::message("spcompress invalid vector shape"));
    }
    let dtype = (descriptor >> 2) & 7;
    if descriptor >> 5 != 0 || dtype > 5 || (elem_bits == 16 && dtype > 1) {
        return Err(OpError::message("spcompress invalid sparsity descriptor"));
    }
    let operation = descriptor & 3;
    let groups = data.len() * 32 / elem_bits / 4;
    let mut metadata = vec![0; (groups * 2 * index_bits).div_ceil(32)];
    let mut compressed = vec![0; data.len() / 2];
    let decode = |bits: u32| -> OpResult<f32> {
        Ok(match (elem_bits, dtype) {
            (16, 0) => cvt::fp16_bits_to_f32(bits as u16),
            (16, 1) => cvt::bf16_bits_to_f32(bits as u16),
            (8, 0) => bits as f32,
            (8, 1) => bits as u8 as i8 as f32,
            (8, 2..=5) => cvt::narrow_float_bits_to_f32_checked(
                bits as u8,
                match dtype {
                    2 => cvt::FLOAT8_E5M2,
                    3 => cvt::FLOAT8_E4M3,
                    4 => cvt::FLOAT6_E3M2,
                    _ => cvt::FLOAT6_E2M3,
                },
            )
            .ok_or_else(|| OpError::message("spcompress invalid narrow-float encoding"))?,
            _ => unreachable!("shape and descriptor were validated above"),
        })
    };
    let elem_mask = (1_u32 << elem_bits) - 1;
    for group in 0..groups {
        let mut bits = [0_u32; 4];
        let mut values = [0_f32; 4];
        for index in 0..4 {
            let bit = (group * 4 + index) * elem_bits;
            bits[index] = (data[bit / 32] >> (bit % 32)) & elem_mask;
            values[index] = decode(bits[index])?;
            if operation & 1 != 0 {
                values[index] = values[index].abs();
            }
        }
        let indices = sparse_pair_indices(values, operation & 2 == 0);
        for (out, &index) in indices.iter().enumerate() {
            let elem = group * 2 + out;
            let mb = elem * index_bits;
            metadata[mb / 32] |= (index as u32) << (mb % 32);
            let cb = elem * elem_bits;
            compressed[cb / 32] |= bits[index] << (cb % 32);
        }
    }
    Ok((metadata, compressed))
}

/// The two kept indices of a 2:4 group, ascending. NaNs rank first; ties
/// prefer the lower index.
pub fn sparse_pair_indices(values: [f32; 4], maximum: bool) -> [usize; 2] {
    let mut indices = [0_usize, 1, 2, 3];
    indices.sort_by(|&a, &b| match (values[a].is_nan(), values[b].is_nan()) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (true, true) => a.cmp(&b),
        _ => {
            let order = values[a].total_cmp(&values[b]);
            (if maximum { order.reverse() } else { order }).then(a.cmp(&b))
        }
    });
    indices[..2].sort_unstable();
    [indices[0], indices[1]]
}

/// Static shape of `spcompress.E.I.N` (`E` element bits, `I` index bits, `N`
/// repetition count).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpCompressShape {
    pub elem_bits: usize,
    pub index_bits: usize,
    pub num: usize,
}

impl SpCompressShape {
    /// Reject shapes outside PTX `spcompress`: element 8/16 bits, index 2/4 bits, num a
    /// power of two in 1..=64.
    pub fn validate(self) -> OpResult<()> {
        if !matches!(self.elem_bits, 8 | 16)
            || !matches!(self.index_bits, 2 | 4)
            || !matches!(self.num, 1 | 2 | 4 | 8 | 16 | 32 | 64)
        {
            return Err(OpError::message("spcompress invalid vector shape"));
        }
        Ok(())
    }
    /// Input data registers.
    pub fn data_registers(self) -> usize {
        2 * self.num
    }
    /// Output registers: metadata words followed by compressed words.
    pub fn output_registers(self) -> usize {
        (self.num * self.index_bits).div_ceil(self.elem_bits) + self.num
    }
}

/// `spcompress` on one lane: returns metadata words followed by compressed
/// words, padded with zeros to [`SpCompressShape::output_registers`].
pub fn spcompress(shape: SpCompressShape, data: &[u32], descriptor: u32) -> OpResult<Vec<u32>> {
    shape.validate()?;
    if data.len() != shape.data_registers() {
        return Err(OpError::message("spcompress invalid vector shape"));
    }
    let (metadata, compressed) =
        compress_sparse_vector(data, shape.elem_bits, shape.index_bits, descriptor)?;
    let mut result = vec![0_u32; shape.output_registers()];
    for (destination, word) in result
        .iter_mut()
        .zip(metadata.into_iter().chain(compressed))
    {
        *destination = word;
    }
    Ok(result)
}

/// Static shape of `spdecompress`: element bits, index bits, sources per
/// group, destinations per group, repetitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpDecompressShape {
    pub elem_bits: usize,
    pub index_bits: usize,
    pub src: usize,
    pub dst: usize,
    pub num: usize,
}

/// The closed PTX 9.4 domain after applying all five vector-size
/// constraints: `(elem, index, src, dst, allowed nums)`.
const SPDECOMPRESS_DOMAIN: &[(usize, usize, usize, usize, &[usize])] = &[
    (8, 2, 1, 2, &[2, 4, 8, 16, 32, 64]),
    (8, 2, 1, 4, &[1, 2, 4, 8, 16, 32, 64]),
    (8, 2, 2, 4, &[1, 2, 4, 8, 16, 32, 64]),
    (8, 4, 1, 2, &[2, 4, 8, 16, 32, 64]),
    (8, 4, 1, 4, &[1, 2, 4, 8, 16, 32, 64]),
    (8, 4, 1, 8, &[1, 2, 4, 8, 16, 32, 64]),
    (8, 4, 1, 16, &[1, 2, 4, 8, 16, 32]),
    (8, 4, 2, 4, &[1, 2, 4, 8, 16, 32, 64]),
    (8, 4, 2, 8, &[1, 2, 4, 8, 16, 32, 64]),
    (8, 4, 2, 16, &[1, 2, 4, 8, 16, 32]),
    (8, 4, 4, 8, &[1, 2, 4, 8, 16, 32, 64]),
    (8, 4, 4, 16, &[1, 2, 4, 8, 16, 32]),
    (16, 2, 1, 2, &[1, 2, 4, 8, 16, 32, 64]),
    (16, 2, 1, 4, &[1, 2, 4, 8, 16, 32, 64]),
    (16, 2, 2, 4, &[1, 2, 4, 8, 16, 32, 64]),
    (16, 4, 1, 2, &[1, 2, 4, 8, 16, 32, 64]),
    (16, 4, 1, 4, &[1, 2, 4, 8, 16, 32, 64]),
    (16, 4, 1, 8, &[1, 2, 4, 8, 16, 32]),
    (16, 4, 1, 16, &[1, 2, 4, 8, 16]),
    (16, 4, 2, 4, &[1, 2, 4, 8, 16, 32, 64]),
    (16, 4, 2, 8, &[1, 2, 4, 8, 16, 32]),
    (16, 4, 2, 16, &[1, 2, 4, 8, 16]),
];

impl SpDecompressShape {
    /// Reject shapes outside the closed PTX 9.4 domain.
    pub fn validate(self) -> OpResult<()> {
        let valid = SPDECOMPRESS_DOMAIN
            .iter()
            .any(|&(elem, index, src, dst, nums)| {
                (elem, index, src, dst) == (self.elem_bits, self.index_bits, self.src, self.dst)
                    && nums.contains(&self.num)
            });
        if !valid {
            return Err(OpError::message(format!(
                "spdecompress shape {self:?} is outside the PTX domain"
            )));
        }
        Ok(())
    }
    /// 32-bit registers holding the packed `src * index_bits * num` metadata bits.
    pub fn metadata_registers(self) -> usize {
        (self.src * self.index_bits * self.num).div_ceil(32)
    }
    /// 32-bit registers holding the `src * num` compressed elements.
    pub fn compressed_registers(self) -> usize {
        (self.src * self.elem_bits * self.num).div_ceil(32)
    }
    /// 32-bit registers of the `dst * num` dense output elements.
    pub fn data_registers(self) -> usize {
        (self.dst * self.elem_bits * self.num).div_ceil(32)
    }
}

/// `spdecompress` on one lane. PTX initializes the entire dense vector to
/// zero; each compressed element then overwrites the indexed destination,
/// which also gives the specified last-source-wins behaviour for duplicate
/// indices. An index `>= dst` fails closed.
pub fn spdecompress(
    shape: SpDecompressShape,
    metadata: &[u32],
    compressed: &[u32],
) -> OpResult<Vec<u32>> {
    shape.validate()?;
    let (metadata_registers, compressed_registers) =
        (shape.metadata_registers(), shape.compressed_registers());
    if metadata.len() != metadata_registers || compressed.len() != compressed_registers {
        return Err(OpError::message(format!(
            "spdecompress operand-vector length mismatch: expected ({metadata_registers}, {compressed_registers}), got ({}, {})",
            metadata.len(),
            compressed.len()
        )));
    }
    let SpDecompressShape {
        elem_bits,
        index_bits,
        src,
        dst,
        num,
    } = shape;
    let mut data = vec![0_u32; shape.data_registers()];
    let index_mask = (1_u32 << index_bits) - 1;
    let element_mask = (1_u32 << elem_bits) - 1;
    for repetition in 0..num {
        for source in 0..src {
            let packed_source = repetition * src + source;
            let metadata_bit = packed_source * index_bits;
            let destination =
                ((metadata[metadata_bit / 32] >> (metadata_bit % 32)) & index_mask) as usize;
            if destination >= dst {
                return Err(OpError::message(format!(
                    "spdecompress metadata index {destination} is outside 0..{dst} at repetition {repetition}, source {source}"
                )));
            }
            let compressed_bit = packed_source * elem_bits;
            let value = (compressed[compressed_bit / 32] >> (compressed_bit % 32)) & element_mask;
            let data_bit = (repetition * dst + destination) * elem_bits;
            let register = &mut data[data_bit / 32];
            let shift = data_bit % 32;
            *register = (*register & !(element_mask << shift)) | (value << shift);
        }
    }
    Ok(data)
}
