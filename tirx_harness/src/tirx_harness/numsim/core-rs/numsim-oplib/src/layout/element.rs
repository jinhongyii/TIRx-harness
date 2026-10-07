//! Pure element-map vocabulary: what a frontend layout program returns for
//! one (logical coordinate, lane), and the address decoders the engine applies
//! to it.
//!
//! Legacy sources: `engine-rs/src/runtime/abi_transport.rs`
//! (`ElementLocation`, `ElementRef`, `ElementMap`) and
//! `engine-rs/src/runtime/instructions/tile.rs` (`MappedElement`,
//! `byte_index`, `byte_offset`, `tmem_coordinates`, `storage_location`).

use crate::types::{OpError, OpResult, WarpMask};

/// Allocation-relative location of one mapped element. Ordinary state spaces
/// use a byte (or byte+bit) offset; TMEM uses the PTX-visible
/// `(TLane, TCol, allocated_addr)` coordinate triple.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElementLocation {
    ByteOffset(i128),
    BitOffset {
        byte_offset: i128,
        bit_offset: u8,
    },
    Tmem {
        mapped_lane: i64,
        tcol_element: i64,
        allocated_addr: i64,
        bit_offset: u8,
    },
}

/// Result of a pure frontend layout mapping for one lane.
///
/// Ownership and bounds are distinct facts: an out-of-bounds element is still
/// assigned to this lane and follows the instruction's fill/error policy,
/// while an unowned element is ignored by this lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ElementRef {
    pub location: ElementLocation,
    pub target_rank: Option<u32>,
    pub owned: bool,
    pub in_bounds: bool,
}

impl ElementRef {
    pub const fn in_bounds(location: ElementLocation, target_rank: Option<u32>) -> Self {
        Self {
            location,
            target_rank,
            owned: true,
            in_bounds: true,
        }
    }

    pub const fn byte(byte_offset: i128) -> Self {
        Self::in_bounds(ElementLocation::ByteOffset(byte_offset), None)
    }

    pub const fn out_of_bounds(target_rank: Option<u32>) -> Self {
        Self {
            location: ElementLocation::ByteOffset(0),
            target_rank,
            owned: true,
            in_bounds: false,
        }
    }

    pub const fn unowned(target_rank: Option<u32>) -> Self {
        Self {
            location: ElementLocation::ByteOffset(0),
            target_rank,
            owned: false,
            in_bounds: false,
        }
    }
}

/// Pure layout program: no engine or memory handle, so mapping cannot
/// perform an effect.
pub trait ElementMap {
    /// Lanes in the current warp that own this logical element (default:
    /// every lane, i.e. probe with [`ElementMap::map`]).
    fn owners(&self, _logical: &[i64]) -> OpResult<WarpMask> {
        Ok(WarpMask::ALL)
    }

    fn map(&self, logical: &[i64], lane: usize) -> OpResult<ElementRef>;
}

impl<F> ElementMap for F
where
    F: Fn(&[i64], usize) -> OpResult<ElementRef>,
{
    fn map(&self, logical: &[i64], lane: usize) -> OpResult<ElementRef> {
        self(logical, lane)
    }
}

/// One mapped element bound to the lane that executes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MappedElement {
    pub execution_lane: usize,
    pub target_cta: Option<usize>,
    pub location: ElementLocation,
}

/// Element index (`offset / itemsize`) of a byte-addressed copy operand, or
/// `(0, false)` when the reference is out of bounds.
pub fn byte_index(reference: ElementRef, itemsize: usize, label: &str) -> OpResult<(i64, bool)> {
    if !reference.in_bounds {
        return Ok((0, false));
    }
    let ElementLocation::ByteOffset(offset) = reference.location else {
        return Err(OpError::message(format!(
            "{label} produced a TMEM coordinate for a byte-addressed copy"
        )));
    };
    if offset < 0 {
        return Err(OpError::message(format!(
            "{label} produced negative in-bounds byte offset {offset}"
        )));
    }
    let itemsize =
        i128::try_from(itemsize).map_err(|_| OpError::message("typed copy itemsize exceeds i128"))?;
    if offset % itemsize != 0 {
        return Err(OpError::message(format!(
            "{label} byte offset {offset} is not aligned to {itemsize} bytes"
        )));
    }
    let index = i64::try_from(offset / itemsize)
        .map_err(|_| OpError::message(format!("{label} element index exceeds i64")))?;
    Ok((index, true))
}

/// Itemsize-aligned byte offset of a byte-addressed mapped element.
pub fn byte_offset(location: MappedElement, itemsize: usize, label: &str) -> OpResult<usize> {
    let ElementLocation::ByteOffset(offset) = location.location else {
        return Err(OpError::message(format!(
            "{label} expected a byte-addressed element"
        )));
    };
    let offset = usize::try_from(offset)
        .map_err(|_| OpError::message(format!("{label} has a negative/oversized offset")))?;
    if offset % itemsize != 0 {
        return Err(OpError::message(format!(
            "{label} byte offset {offset} is not aligned to itemsize {itemsize}"
        )));
    }
    Ok(offset)
}

/// `(TLane, TCol, allocated_addr)` of a whole-cell TMEM element.
pub fn tmem_coordinates(location: MappedElement, label: &str) -> OpResult<(i64, i64, i64)> {
    let ElementLocation::Tmem {
        mapped_lane,
        tcol_element,
        allocated_addr,
        bit_offset: 0,
    } = location.location
    else {
        return Err(OpError::message(format!("{label} expected a TMEM coordinate")));
    };
    Ok((mapped_lane, tcol_element, allocated_addr))
}

/// `(byte offset, TMEM coordinate, bit offset)` of a GEMM operand element.
pub fn storage_location(
    element: MappedElement,
    label: &str,
) -> OpResult<(Option<usize>, Option<(i64, i64, i64)>, u8)> {
    let byte = |offset: i128| {
        usize::try_from(offset).map_err(|_| {
            OpError::message(format!("{label} has a negative/oversized byte offset"))
        })
    };
    match element.location {
        ElementLocation::ByteOffset(offset) => Ok((Some(byte(offset)?), None, 0)),
        ElementLocation::BitOffset {
            byte_offset,
            bit_offset,
        } => Ok((Some(byte(byte_offset)?), None, bit_offset)),
        ElementLocation::Tmem {
            mapped_lane,
            tcol_element,
            allocated_addr,
            bit_offset,
        } => Ok((None, Some((mapped_lane, tcol_element, allocated_addr)), bit_offset)),
    }
}

/// Regular-memory footprint rows `(lane, target CTA, byte offset, len)`.
pub fn regular_footprints(
    elements: &[MappedElement],
    itemsize: usize,
    label: &str,
) -> OpResult<Vec<(usize, Option<usize>, usize, usize)>> {
    elements
        .iter()
        .copied()
        .map(|element| {
            Ok((
                element.execution_lane,
                element.target_cta,
                byte_offset(element, itemsize, label)?,
                itemsize,
            ))
        })
        .collect()
}

/// TMEM footprint rows `(provenance lane, lane, target CTA, TLane, TCol,
/// allocated_addr, len)`.
#[allow(clippy::type_complexity)]
pub fn tmem_footprints(
    elements: &[MappedElement],
    provenance_lane: usize,
    itemsize: usize,
    label: &str,
) -> OpResult<Vec<(usize, usize, Option<usize>, i64, i64, i64, usize)>> {
    elements
        .iter()
        .copied()
        .map(|element| {
            let (mapped_lane, tcol_element, allocated_addr) = tmem_coordinates(element, label)?;
            Ok((
                provenance_lane,
                element.execution_lane,
                element.target_cta,
                mapped_lane,
                tcol_element,
                allocated_addr,
                itemsize,
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoders_validate_space_alignment_and_sign() {
        assert_eq!(byte_index(ElementRef::byte(12), 4, "m").unwrap(), (3, true));
        assert_eq!(byte_index(ElementRef::out_of_bounds(None), 4, "m").unwrap(), (0, false));
        assert!(byte_index(ElementRef::byte(-4), 4, "m")
            .unwrap_err()
            .to_string()
            .contains("negative in-bounds byte offset -4"));
        assert!(byte_index(ElementRef::byte(6), 4, "m")
            .unwrap_err()
            .to_string()
            .contains("not aligned to 4 bytes"));
        let tmem = MappedElement {
            execution_lane: 1,
            target_cta: None,
            location: ElementLocation::Tmem {
                mapped_lane: 3,
                tcol_element: 5,
                allocated_addr: 7,
                bit_offset: 0,
            },
        };
        assert_eq!(tmem_coordinates(tmem, "t").unwrap(), (3, 5, 7));
        assert!(byte_offset(tmem, 4, "t").is_err());
        assert_eq!(storage_location(tmem, "t").unwrap(), (None, Some((3, 5, 7)), 0));
        let bits = MappedElement {
            location: ElementLocation::BitOffset {
                byte_offset: 9,
                bit_offset: 4,
            },
            ..tmem
        };
        assert_eq!(storage_location(bits, "b").unwrap(), (Some(9), None, 4));
        assert!(tmem_coordinates(bits, "b").is_err());
        assert_eq!(
            tmem_footprints(&[tmem], 0, 4, "t").unwrap(),
            vec![(0, 1, None, 3, 5, 7, 4)]
        );
        let byte = MappedElement {
            location: ElementLocation::ByteOffset(8),
            ..tmem
        };
        assert_eq!(regular_footprints(&[byte], 4, "r").unwrap(), vec![(1, None, 8, 4)]);
    }
}
