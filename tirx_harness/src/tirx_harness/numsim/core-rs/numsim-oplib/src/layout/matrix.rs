//! `ldmatrix` / `stmatrix` thread-value address maps: which providing lane's
//! row address, byte delta, width and bit shift each fragment element uses.
//!
//! Legacy source: `engine-rs/src/runtime/memory_ops.rs`
//! (`raw_ldmatrix_b16_fragments`, `ldmatrix_b8_element`,
//! `raw_ldmatrix_b8_fragments`, `plan_raw_ldmatrix_access`,
//! `StmatrixDescriptor`, `plan_raw_stmatrix_access`, `raw_stmatrix`).
//!
//! Memory is reached through a caller-supplied `read(provider_lane,
//! byte_delta, byte_len)` closure: `byte_delta` is relative to the provider
//! lane's row pointer (the engine resolves and bounds-checks it). Warp-sync,
//! state-space and buffer-kind checks stay in the engine.

use crate::types::{OpError, OpResult, WarpValue, WARP_SIZE};

/// One contiguous access through a providing lane's row address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatrixAccess {
    pub provider_lane: usize,
    pub byte_delta: usize,
    pub byte_len: usize,
    /// Bit shift of the element within the accessed bytes (b8 formats).
    pub shift: usize,
}

/// `ldmatrix.m8n8.b16` accesses feeding fragment register `matrix` of
/// `lane`: one 4-byte read, or two 2-byte reads when transposed. `base` is
/// added to every delta (legacy element-offset operand times itemsize).
pub fn ldmatrix_b16_accesses(
    matrix: usize,
    lane: usize,
    transpose: bool,
    base: impl Fn(usize) -> OpResult<usize>,
) -> OpResult<Vec<MatrixAccess>> {
    let row = lane / 4;
    let fragment = lane % 4;
    let overflow = || OpError::message("ldmatrix source offset overflow");
    if transpose {
        let source_lane0 = matrix * 8 + fragment * 2;
        let column_offset = row * 2;
        [source_lane0, source_lane0 + 1]
            .into_iter()
            .map(|provider_lane| {
                Ok(MatrixAccess {
                    provider_lane,
                    byte_delta: base(provider_lane)?
                        .checked_add(column_offset)
                        .ok_or_else(overflow)?,
                    byte_len: 2,
                    shift: 0,
                })
            })
            .collect()
    } else {
        let provider_lane = matrix * 8 + row;
        Ok(vec![MatrixAccess {
            provider_lane,
            byte_delta: base(provider_lane)?
                .checked_add(fragment * 4)
                .ok_or_else(overflow)?,
            byte_len: 4,
            shift: 0,
        }])
    }
}

/// Byte base of each lane from the legacy element-offset operand.
pub fn ldmatrix_lane_base(
    source_element_offsets: &WarpValue<i64>,
    source_itemsize: usize,
    lane: usize,
) -> OpResult<usize> {
    usize::try_from(source_element_offsets[lane])
        .map_err(|_| OpError::message("negative ldmatrix source element offset"))?
        .checked_mul(source_itemsize)
        .ok_or_else(|| OpError::message("ldmatrix source byte offset overflow"))
}

/// `ldmatrix...b16` fragments: `matrix_count` registers per lane.
pub fn ldmatrix_b16_fragments(
    source_element_offsets: &WarpValue<i64>,
    source_itemsize: usize,
    matrix_count: usize,
    transpose: bool,
    mut read: impl FnMut(usize, usize, usize) -> OpResult<Vec<u8>>,
) -> OpResult<Vec<WarpValue<u32>>> {
    if !matches!(matrix_count, 1 | 2 | 4) {
        return Err(OpError::message(format!(
            "ldmatrix b16 matrix count must be 1, 2, or 4, got {matrix_count}"
        )));
    }
    let base = |lane| ldmatrix_lane_base(source_element_offsets, source_itemsize, lane);
    let mut fragments = vec![[0_u32; WARP_SIZE]; matrix_count];
    for (matrix, matrix_fragments) in fragments.iter_mut().enumerate() {
        for (lane, value) in matrix_fragments.iter_mut().enumerate() {
            let mut bytes = Vec::with_capacity(4);
            for access in ldmatrix_b16_accesses(matrix, lane, transpose, base)? {
                bytes.extend(read(
                    access.provider_lane,
                    access.byte_delta,
                    access.byte_len,
                )?);
            }
            *value = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
    }
    Ok(fragments)
}

/// One output byte's source span for `ldmatrix` b8 formats
/// (`.b8x16.b6x16_p32` / `.b4x16_p64` / m16n16): padding after each packed
/// row is never read. `row_base(provider)` is the provider's row byte
/// offset, which must be 16-byte aligned.
pub fn ldmatrix_b8_element(
    register: usize,
    lane: usize,
    element: usize,
    transpose: bool,
    source_bits: usize,
    row_base: impl Fn(usize) -> OpResult<usize>,
) -> OpResult<MatrixAccess> {
    let (provider, column) = if transpose {
        (
            register / 2 * 16 + lane % 4 * 4 + element,
            register % 2 * 8 + lane / 4,
        )
    } else {
        (register * 8 + lane / 4, lane % 4 * 4 + element)
    };
    let bit = column * source_bits;
    let shift = bit % 8;
    let width = (shift + source_bits).div_ceil(8);
    if row_base(provider)? % 16 != 0 {
        return Err(OpError::message(
            "ldmatrix row address must be 16-byte aligned",
        ));
    }
    Ok(MatrixAccess {
        provider_lane: provider,
        byte_delta: bit / 8,
        byte_len: width,
        shift,
    })
}

/// `ldmatrix` b8-format fragments (each register holds four 8-bit lanes,
/// zero-extended from `source_bits`).
pub fn ldmatrix_b8_fragments(
    register_count: usize,
    transpose: bool,
    source_bits: usize,
    row_base: impl Fn(usize) -> OpResult<usize>,
    mut read: impl FnMut(usize, usize, usize) -> OpResult<Vec<u8>>,
) -> OpResult<Vec<WarpValue<u32>>> {
    let mut fragments = vec![[0_u32; WARP_SIZE]; register_count];
    for (register, fragment) in fragments.iter_mut().enumerate() {
        for (lane, value) in fragment.iter_mut().enumerate() {
            for element in 0..4 {
                let access = ldmatrix_b8_element(
                    register,
                    lane,
                    element,
                    transpose,
                    source_bits,
                    &row_base,
                )?;
                let bytes = read(access.provider_lane, access.byte_delta, access.byte_len)?;
                let packed = u16::from(bytes[0]) | (u16::from(*bytes.get(1).unwrap_or(&0)) << 8);
                *value |=
                    u32::from((packed >> access.shift) & ((1 << source_bits) - 1)) << (element * 8);
            }
        }
    }
    Ok(fragments)
}

/// Every access of one consumer lane (footprint form of `ldmatrix`).
pub fn ldmatrix_lane_accesses(
    register_count: usize,
    lane: usize,
    transpose: bool,
    source_bits: usize,
    row_base: impl Fn(usize) -> OpResult<usize>,
) -> OpResult<Vec<MatrixAccess>> {
    if !matches!(register_count, 1 | 2 | 4) {
        return Err(OpError::message(format!(
            "ldmatrix fragment count must be 1, 2, or 4, got {register_count}"
        )));
    }
    let mut accesses = Vec::new();
    for register in 0..register_count {
        if source_bits != 16 {
            for element in 0..4 {
                accesses.push(ldmatrix_b8_element(
                    register,
                    lane,
                    element,
                    transpose,
                    source_bits,
                    &row_base,
                )?);
            }
        } else {
            accesses.extend(ldmatrix_b16_accesses(register, lane, transpose, |_| Ok(0))?);
        }
    }
    Ok(accesses)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StmatrixShape {
    M8n8B16 { transpose: bool },
    M16n8B8Transposed,
}

/// `(provider lane, byte delta, value bytes)` written by source register
/// `matrix` of `source_lane`, in legacy write order.
pub fn stmatrix_lane_writes(
    shape: StmatrixShape,
    matrix: usize,
    source_lane: usize,
    register: u32,
) -> Vec<(usize, usize, Vec<u8>)> {
    match shape {
        StmatrixShape::M8n8B16 { transpose } => (0..2_usize)
            .map(|half_index| {
                let (row, column) = if transpose {
                    (2 * (source_lane % 4) + half_index, source_lane / 4)
                } else {
                    (source_lane / 4, 2 * (source_lane % 4) + half_index)
                };
                let value = (register >> (half_index * 16)) as u16;
                (matrix * 8 + row, column * 2, value.to_le_bytes().to_vec())
            })
            .collect(),
        StmatrixShape::M16n8B8Transposed => (0..4_usize)
            .map(|byte_index| {
                let row = 2 * (source_lane % 4) + (byte_index % 2);
                let column = source_lane / 4 + 8 * (byte_index / 2);
                (
                    matrix * 8 + row,
                    column,
                    vec![(register >> (byte_index * 8)) as u8],
                )
            })
            .collect(),
    }
}

/// All `stmatrix` writes of a warp (matrix-major, then source lane).
/// `row_address(provider)` is the provider's physical row address, which
/// must be 16-byte aligned.
pub fn stmatrix_writes(
    shape: StmatrixShape,
    sources: &[&WarpValue<u32>],
    row_address: impl Fn(usize) -> OpResult<usize>,
) -> OpResult<Vec<(usize, usize, Vec<u8>)>> {
    if !matches!(sources.len(), 1 | 2 | 4) {
        return Err(OpError::message(format!(
            "stmatrix source count must be 1, 2, or 4, got {}",
            sources.len()
        )));
    }
    for provider_lane in 0..sources.len() * 8 {
        if row_address(provider_lane)? % 16 != 0 {
            return Err(OpError::message(format!(
                "stmatrix row address requires 16-byte alignment on lane {provider_lane}"
            )));
        }
    }
    let mut writes = Vec::new();
    for (matrix, source) in sources.iter().enumerate() {
        for (source_lane, register) in source.iter().enumerate() {
            writes.extend(stmatrix_lane_writes(shape, matrix, source_lane, *register));
        }
    }
    Ok(writes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 8 rows of 16 bytes per matrix; lane `m*8+r` points at row `m*8+r`.
    fn rows(matrices: usize) -> Vec<u8> {
        (0..matrices * 128).map(|i| (i * 7 + 3) as u8).collect()
    }

    #[test]
    fn ldmatrix_b16_matches_ptx_fragment_layout() {
        let memory = rows(2);
        let read = |provider: usize, delta: usize, len: usize| {
            Ok(memory[provider * 16 + delta..provider * 16 + delta + len].to_vec())
        };
        let zeros = [0_i64; WARP_SIZE];
        let plain = ldmatrix_b16_fragments(&zeros, 2, 2, false, read).unwrap();
        let transposed = ldmatrix_b16_fragments(&zeros, 2, 2, true, read).unwrap();
        let element = |m: usize, r: usize, c: usize| {
            let at = m * 128 + r * 16 + c * 2;
            u16::from_le_bytes([memory[at], memory[at + 1]])
        };
        for m in 0..2 {
            for lane in 0..WARP_SIZE {
                let (r, c) = (lane / 4, lane % 4 * 2);
                let expect = u32::from(element(m, r, c)) | u32::from(element(m, r, c + 1)) << 16;
                assert_eq!(plain[m][lane], expect);
                // Transposed: thread holds column r of rows c, c+1.
                let expect_t = u32::from(element(m, c, r)) | u32::from(element(m, c + 1, r)) << 16;
                assert_eq!(transposed[m][lane], expect_t);
            }
        }
        assert!(ldmatrix_b16_fragments(&zeros, 2, 3, false, read).is_err());
        let mut negative = zeros;
        negative[0] = -1;
        assert!(ldmatrix_b16_fragments(&negative, 2, 1, false, read).is_err());
    }

    #[test]
    fn ldmatrix_b8_unpacks_sub_byte_rows_and_checks_alignment() {
        // b4 (source_bits 4): row r holds 16 nibble values, nibble k of row r = (r + k) & 15.
        let mut memory = [0_u8; 8 * 16];
        for r in 0..8 {
            for k in 0..16 {
                memory[r * 16 + k / 2] |= (((r + k) & 15) as u8) << ((k % 2) * 4);
            }
        }
        let fragments = ldmatrix_b8_fragments(
            1,
            false,
            4,
            |provider| Ok(provider * 16),
            |provider, delta, len| {
                Ok(memory[provider * 16 + delta..provider * 16 + delta + len].to_vec())
            },
        )
        .unwrap();
        for (lane, &word) in fragments[0].iter().enumerate() {
            for element in 0..4 {
                let (r, k) = (lane / 4, lane % 4 * 4 + element);
                assert_eq!((word >> (element * 8)) & 0xff, ((r + k) & 15) as u32);
            }
        }
        let access = ldmatrix_b8_element(0, 1, 2, false, 6, |_| Ok(0)).unwrap();
        // column 6 * 6 bits = bit 36 -> byte 4, shift 4, two bytes.
        assert_eq!(
            access,
            MatrixAccess {
                provider_lane: 0,
                byte_delta: 4,
                byte_len: 2,
                shift: 4
            }
        );
        assert!(ldmatrix_b8_element(0, 0, 0, false, 8, |_| Ok(8)).is_err());
        assert_eq!(
            ldmatrix_lane_accesses(4, 5, true, 16, |_| Ok(0))
                .unwrap()
                .len(),
            8
        );
        assert!(ldmatrix_lane_accesses(3, 0, false, 16, |_| Ok(0)).is_err());
    }

    #[test]
    fn stmatrix_inverts_ldmatrix_and_checks_alignment() {
        let memory = rows(1);
        let zeros = [0_i64; WARP_SIZE];
        for transpose in [false, true] {
            let read = |provider: usize, delta: usize, len: usize| {
                Ok(memory[provider * 16 + delta..provider * 16 + delta + len].to_vec())
            };
            let fragments = ldmatrix_b16_fragments(&zeros, 2, 1, transpose, read).unwrap();
            let writes = stmatrix_writes(
                StmatrixShape::M8n8B16 { transpose },
                &[&fragments[0]],
                |provider| Ok(provider * 16),
            )
            .unwrap();
            let mut written = vec![0_u8; 128];
            for (provider, delta, bytes) in writes {
                written[provider * 16 + delta..provider * 16 + delta + bytes.len()]
                    .copy_from_slice(&bytes);
            }
            assert_eq!(written, memory);
        }
        let register = [0x4433_2211_u32; WARP_SIZE];
        let writes = stmatrix_writes(StmatrixShape::M16n8B8Transposed, &[&register], |p| {
            Ok(p * 16)
        })
        .unwrap();
        assert_eq!(writes.len(), 128);
        assert_eq!(writes[0], (0, 0, vec![0x11]));
        assert_eq!(writes[1], (1, 0, vec![0x22]));
        assert_eq!(writes[2], (0, 8, vec![0x33]));
        assert!(stmatrix_writes(
            StmatrixShape::M16n8B8Transposed,
            &[&register],
            |p| Ok(p * 8)
        )
        .unwrap_err()
        .to_string()
        .contains("16-byte alignment on lane 1"));
    }
}
