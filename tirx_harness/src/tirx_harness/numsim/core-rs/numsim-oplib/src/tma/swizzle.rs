//! TMA shared-memory swizzle address functions.
//!
//! Legacy source: `engine-rs/src/runtime/tensor_map.rs`
//! (`SwizzleAtomicity`, `tensor_map_shared_byte_offset_from_layout`).
//!
//! Every function maps a dense (row, byte-in-row) box position to the byte
//! offset relative to the shared destination pointer. The XOR pattern is a
//! function of the *absolute* shared address, so callers pass the absolute
//! byte address of the destination pointer (`absolute_base`; only the
//! byte-offset field of a shared::cluster address participates).

use crate::types::{OpError, OpResult};

/// Swizzle atom granularity ("swizzle atomicity") of a 128B-swizzled map.
///
/// Independent of swizzle width so `tensormap.replace.swizzle_mode` preserves
/// this field. Only `B16` is legal for 32B/64B/96B swizzle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SwizzleAtomicity {
    /// 16-byte atoms (the PTX default).
    B16,
    /// 32-byte atoms.
    B32,
    /// 32-byte atoms with an additional 8-byte flip (load-only).
    B32Flip8,
    /// 64-byte atoms.
    B64,
}

impl SwizzleAtomicity {
    /// Atom size in bytes.
    pub const fn bytes(self) -> usize {
        match self {
            Self::B16 => 16,
            Self::B32 | Self::B32Flip8 => 32,
            Self::B64 => 64,
        }
    }

    /// Bits OR-ed into the NumSim TensorMap image format tag (byte 63).
    pub const fn tag_bits(self) -> u8 {
        match self {
            Self::B16 => 0,
            Self::B32 => 8,
            Self::B32Flip8 => 64,
            Self::B64 => 72,
        }
    }

    /// Decode the atomicity from the image format tag (byte 63).
    pub const fn from_tag(tag: u8) -> Self {
        match tag & 0x48 {
            0 => Self::B16,
            8 => Self::B32,
            64 => Self::B32Flip8,
            _ => Self::B64,
        }
    }
}

/// 96B swizzle (PTX Figure 32): 96-byte rows are packed consecutively, then
/// adjacent 16-byte atoms are exchanged in every odd 128-byte band.
pub fn swizzle_96b_offset(
    outer_linear: usize,
    inner_byte: usize,
    absolute_base: usize,
) -> OpResult<usize> {
    let absolute = outer_linear
        .checked_mul(96)
        .and_then(|row| row.checked_add(inner_byte))
        .and_then(|offset| absolute_base.checked_add(offset))
        .ok_or_else(|| OpError::message("96B swizzle address overflow"))?;
    (absolute ^ ((absolute >> 3) & 16))
        .checked_sub(absolute_base)
        .ok_or_else(|| OpError::message("96B swizzle precedes shared base"))
}

/// XOR-swizzle one absolute shared byte address for a 32B/64B/128B swizzle
/// span with the given atom size. Returns the swizzled absolute address.
///
/// The atom index within the span is XOR-ed with bits `[7, 7 + log2(groups))`
/// of the absolute address; `B32Flip8` additionally flips bit 3 by bit 7.
pub fn swizzle_absolute_address(
    absolute: usize,
    swizzle_bytes: usize,
    atomicity: SwizzleAtomicity,
) -> OpResult<usize> {
    let atom_bytes = atomicity.bytes();
    let groups = swizzle_bytes / atom_bytes;
    if !matches!(groups, 2 | 4 | 8) {
        return Err(OpError::message(format!(
            "TensorMap swizzle has unsupported atom group count {groups}"
        )));
    }
    let flipped = if atomicity == SwizzleAtomicity::B32Flip8 {
        absolute ^ ((absolute >> 4) & 8)
    } else {
        absolute
    };
    Ok(flipped ^ (((absolute >> 7) & (groups - 1)) * atom_bytes))
}

/// Shared byte offset (relative to the destination pointer) of byte
/// `inner_byte` of box row `outer_linear`.
///
/// `swizzle_bytes = None` is the dense layout (`row * inner_row_bytes`).
/// Swizzled rows occupy at least one swizzle span; interleaved rows can span
/// several groups and use the same absolute-address atom XOR.
pub fn shared_byte_offset(
    swizzle_bytes: Option<usize>,
    atomicity: SwizzleAtomicity,
    outer_linear: usize,
    inner_byte: usize,
    inner_row_bytes: usize,
    absolute_base: usize,
) -> OpResult<usize> {
    let Some(swizzle_bytes) = swizzle_bytes else {
        return outer_linear
            .checked_mul(inner_row_bytes)
            .and_then(|row| row.checked_add(inner_byte))
            .ok_or_else(|| OpError::message("dense TensorMap shared offset overflow"));
    };
    if swizzle_bytes == 96 {
        return swizzle_96b_offset(outer_linear, inner_byte, absolute_base);
    }
    let atom_bytes = atomicity.bytes();
    let groups = swizzle_bytes / atom_bytes;
    if !matches!(groups, 2 | 4 | 8) {
        return Err(OpError::message(format!(
            "TensorMap swizzle has unsupported atom group count {groups}"
        )));
    }
    let absolute = outer_linear
        .checked_mul(inner_row_bytes.max(swizzle_bytes))
        .and_then(|row| row.checked_add(inner_byte))
        .and_then(|offset| absolute_base.checked_add(offset))
        .ok_or_else(|| OpError::message("swizzled TensorMap shared offset overflow"))?;
    swizzle_absolute_address(absolute, swizzle_bytes, atomicity)?
        .checked_sub(absolute_base)
        .ok_or_else(|| OpError::message("swizzled TensorMap precedes shared base"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swizzle_matches_rank_coordinates_oracle() {
        // tensor_map.rs rank_coordinates_and_swizzle_are_physical_layout_operations
        assert_eq!(
            shared_byte_offset(Some(32), SwizzleAtomicity::B16, 4, 0, 32, 0).unwrap(),
            144
        );
        // align16 padded FP4 (128B swizzle, row 128 bytes).
        assert_eq!(
            shared_byte_offset(Some(128), SwizzleAtomicity::B16, 1, 0, 128, 0).unwrap(),
            144
        );
        assert_eq!(
            shared_byte_offset(Some(128), SwizzleAtomicity::B16, 1, 16, 128, 0).unwrap(),
            128
        );
    }

    #[test]
    fn swizzle_atom_rotation_matches_pointer_phase_oracle() {
        // Oracle from tensor_copy_round_trips_rank_dtype_swizzle_and_pointer_phase_matrix:
        // one 16-byte row per box row, pointer at `phase * 128`.
        for swizzle in [32_usize, 64, 128] {
            let groups = swizzle / 16;
            let row_shift = match groups {
                2 => 2,
                4 => 1,
                _ => 0,
            };
            for phase in 0..8_usize {
                for outer in 0..16_usize {
                    for byte in [0_usize, 7, 15] {
                        let expected =
                            outer * swizzle + ((outer >> row_shift) + phase) % groups * 16 + byte;
                        assert_eq!(
                            shared_byte_offset(
                                Some(swizzle),
                                SwizzleAtomicity::B16,
                                outer,
                                byte,
                                16,
                                phase * 128
                            )
                            .unwrap(),
                            expected
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn atomicity_tag_round_trips_and_96b_swaps_odd_bands() {
        for atomicity in [
            SwizzleAtomicity::B16,
            SwizzleAtomicity::B32,
            SwizzleAtomicity::B32Flip8,
            SwizzleAtomicity::B64,
        ] {
            assert_eq!(SwizzleAtomicity::from_tag(0xA7 | atomicity.tag_bits()), atomicity);
        }
        // Row 1 starts at absolute 96: band 0 (bit 7 clear) -> unchanged.
        assert_eq!(swizzle_96b_offset(1, 0, 0).unwrap(), 96);
        // Absolute 128 is in odd band: atom exchanged with its neighbour.
        assert_eq!(swizzle_96b_offset(1, 32, 0).unwrap(), 144);
        assert_eq!(swizzle_96b_offset(1, 48, 0).unwrap(), 128);
        // B32Flip8 flips bit 3 by bit 7.
        assert_eq!(
            swizzle_absolute_address(128, 128, SwizzleAtomicity::B32Flip8).unwrap(),
            128 ^ 8 ^ 32
        );
        assert!(shared_byte_offset(Some(32), SwizzleAtomicity::B32, 0, 0, 16, 0).is_err());
    }
}
