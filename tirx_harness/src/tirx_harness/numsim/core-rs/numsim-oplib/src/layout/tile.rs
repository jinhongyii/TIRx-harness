//! Logical tile shapes and execution-scope partitions of typed whole-tile
//! operations (`tirx.tile.*`).
//!
//! Legacy source: `engine-rs/src/runtime/instructions/tile.rs`
//! (`StaticShape::EXTENTS`, `coordinates_for`, `element_count_for`,
//! `TileScopeKind`, `scope_lanes`). The legacy const-generic
//! `variant::Shape{1..5}` / scope marker types become plain `&[usize]`
//! extents and the [`TileScope`] enum.

use crate::types::{OpError, OpResult, WarpMask, WARP_SIZE};

/// Execution scope of a typed tile operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TileScope {
    /// Every active lane executes the whole tile.
    Thread,
    /// Element `linear` belongs to lane `linear % 32`.
    Warp,
    /// Element `linear` belongs to thread `linear % 128` of the warpgroup.
    Warpgroup,
    /// Element `linear` belongs to thread `linear % (32 * warps_per_cta)`.
    Cta,
}

/// Row-major (last axis fastest) logical coordinates of element `linear`.
pub fn coordinates_for(extents: &[usize], linear: usize) -> OpResult<Vec<i64>> {
    if extents.is_empty() || extents.contains(&0) {
        return Err(OpError::message(
            "typed tile shape must have positive rank and extents",
        ));
    }
    let mut remainder = linear;
    let mut result = vec![0_i64; extents.len()];
    for axis in (0..extents.len()).rev() {
        let extent = extents[axis];
        result[axis] = i64::try_from(remainder % extent)
            .map_err(|_| OpError::message("typed tile coordinate exceeds i64"))?;
        remainder /= extent;
    }
    Ok(result)
}

/// Number of logical elements of a tile shape.
pub fn element_count_for(extents: &[usize]) -> OpResult<usize> {
    extents.iter().try_fold(1_usize, |count, &extent| {
        if extent == 0 {
            return Err(OpError::message("typed tile extent must be positive"));
        }
        count
            .checked_mul(extent)
            .ok_or_else(|| OpError::message("typed tile element count overflow"))
    })
}

/// Lanes of this warp that execute logical element `linear` under `scope`.
/// `warp_id_in_cta` and `warps_per_cta` describe the issuing warp.
pub fn scope_lanes(
    scope: TileScope,
    active_mask: WarpMask,
    warp_id_in_cta: usize,
    warps_per_cta: usize,
    linear: usize,
) -> OpResult<Vec<usize>> {
    match scope {
        TileScope::Thread => Ok(active_mask.iter().collect()),
        TileScope::Warp => {
            let lane = linear % WARP_SIZE;
            Ok(active_mask.contains(lane).then_some(lane).into_iter().collect())
        }
        TileScope::Warpgroup => {
            let owner = linear % (4 * WARP_SIZE);
            let relative_warp = warp_id_in_cta % 4;
            let lane = owner % WARP_SIZE;
            Ok(
                (owner / WARP_SIZE == relative_warp && active_mask.contains(lane))
                    .then_some(lane)
                    .into_iter()
                    .collect(),
            )
        }
        TileScope::Cta => {
            let threads = warps_per_cta
                .checked_mul(WARP_SIZE)
                .ok_or_else(|| OpError::message("CTA tile thread count overflow"))?;
            let owner = linear % threads;
            let lane = owner % WARP_SIZE;
            Ok(
                (owner / WARP_SIZE == warp_id_in_cta && active_mask.contains(lane))
                    .then_some(lane)
                    .into_iter()
                    .collect(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_shape_is_the_complete_logical_tile_not_a_repeat_count() {
        // tile.rs: variant::Shape2<4, 8>
        let shape = [4_usize, 8];
        assert_eq!(element_count_for(&shape).unwrap(), 32);
        assert_eq!(coordinates_for(&shape, 19).unwrap(), vec![2, 3]);
        assert!(coordinates_for(&[], 0).is_err());
        assert!(element_count_for(&[3, 0]).is_err());
    }

    #[test]
    fn scope_partitions_assign_each_element_to_one_thread() {
        let full = WarpMask::FULL;
        assert_eq!(scope_lanes(TileScope::Thread, WarpMask(0b101), 0, 4, 9).unwrap(), vec![0, 2]);
        assert_eq!(scope_lanes(TileScope::Warp, full, 0, 4, 33).unwrap(), vec![1]);
        assert!(scope_lanes(TileScope::Warp, WarpMask(1), 0, 4, 33).unwrap().is_empty());
        assert_eq!(scope_lanes(TileScope::Warpgroup, full, 5, 8, 32 + 7).unwrap(), vec![7]);
        assert!(scope_lanes(TileScope::Warpgroup, full, 4, 8, 32 + 7).unwrap().is_empty());
        assert_eq!(scope_lanes(TileScope::Cta, full, 2, 3, 96 + 64 + 3).unwrap(), vec![3]);
        assert!(scope_lanes(TileScope::Cta, full, 1, 3, 64 + 3).unwrap().is_empty());
    }
}
