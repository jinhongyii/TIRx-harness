//! Partition/thread-value planning of typed tile operations over pure
//! element maps: copy element pairing (async and synchronous ownership
//! protocols), TCGEN element ownership, GEMM operand matrix maps, and the
//! canonical TMEM fast-path recognizers.
//!
//! Legacy source: `engine-rs/src/runtime/instructions/tile.rs`
//! (`mapped_copy_plan`, `mapped_sync_copy_plan`, `mapped_tcgen_elements`,
//! `map_gemm_element`, `map_gemm_matrix`, `map_register_gemm_matrix`,
//! `map_gemm_scale_matrix`, `fast_tmem_f32_m64_load`,
//! `issue_canonical_32x32b` footprint rows). Legacy `MappedView`s become
//! [`ElementMap`] values; the warp context becomes explicit parameters.

use super::element::{byte_index, tmem_coordinates, ElementLocation, ElementMap, MappedElement};
use super::tile::{coordinates_for, element_count_for, scope_lanes, TileScope};
use crate::types::{OpError, OpResult, WarpMask, WARP_SIZE};

/// Issuing-warp facts that tile partitions depend on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileWarp {
    pub active_mask: WarpMask,
    pub warp_id_in_cta: usize,
    pub warps_per_cta: usize,
}

impl TileWarp {
    fn scope_lanes(self, scope: TileScope, linear: usize) -> OpResult<Vec<usize>> {
        scope_lanes(scope, self.active_mask, self.warp_id_in_cta, self.warps_per_cta, linear)
    }
}

/// One element moved by a typed tile copy (legacy `TypedTileCopyElement`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileCopyElement {
    pub source_index: i64,
    pub source_in_bounds: bool,
    pub source_lane: usize,
    pub destination_index: i64,
    pub destination_in_bounds: bool,
    pub destination_lane: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyPlan {
    pub elements: Vec<TileCopyElement>,
    /// Remote CTA rank selected by the destination map, if any.
    pub destination_rank: Option<usize>,
}

fn wrap<T>(result: OpResult<T>, label: &str) -> OpResult<T> {
    result.map_err(|error| OpError::message(format!("{label}: {error}")))
}

fn note_destination_rank(destination_rank: &mut Option<usize>, rank: Option<u32>) -> OpResult<()> {
    if let Some(rank) = rank {
        let rank =
            usize::try_from(rank).map_err(|_| OpError::message("mapped CTA rank exceeds usize"))?;
        match *destination_rank {
            Some(previous) if previous != rank => {
                return Err(OpError::message(
                    "one typed copy cannot target multiple non-multicast CTA ranks",
                ));
            }
            None => *destination_rank = Some(rank),
            _ => {}
        }
    }
    Ok(())
}

/// Pair source/destination elements for an asynchronous tile copy: each
/// logical element executes on the scope's owner lanes.
pub fn mapped_copy_plan(
    warp: TileWarp,
    source: &dyn ElementMap,
    destination: &dyn ElementMap,
    extents: &[usize],
    scope: TileScope,
    itemsize: usize,
) -> OpResult<CopyPlan> {
    let count = element_count_for(extents)?;
    let mut elements = Vec::new();
    let mut destination_rank = None;
    for linear in 0..count {
        let coordinates = coordinates_for(extents, linear)?;
        for lane in warp.scope_lanes(scope, linear)? {
            let source_ref = wrap(source.map(&coordinates, lane), "typed copy source map")?;
            let destination_ref =
                wrap(destination.map(&coordinates, lane), "typed copy destination map")?;
            if source_ref.target_rank.is_some() {
                return Err(OpError::message(
                    "typed copy source map cannot select a remote CTA",
                ));
            }
            note_destination_rank(&mut destination_rank, destination_ref.target_rank)?;
            let (source_index, source_in_bounds) =
                byte_index(source_ref, itemsize, "typed copy source map")?;
            let (destination_index, destination_in_bounds) =
                byte_index(destination_ref, itemsize, "typed copy destination map")?;
            elements.push(TileCopyElement {
                source_index,
                source_in_bounds,
                source_lane: lane,
                destination_index,
                destination_in_bounds,
                destination_lane: lane,
            });
        }
    }
    Ok(CopyPlan {
        elements,
        destination_rank,
    })
}

/// Pair source/destination owners for a synchronous tile copy
/// (`tirx.tile.copy`). Replicated (every-lane) and single-owner maps are
/// matched lane-to-lane; `destination_lane_private` selects whether a fully
/// replicated element is written by every active lane (local/register
/// destinations) or only by the scope's owner lanes.
pub fn mapped_sync_copy_plan(
    warp: TileWarp,
    source: &dyn ElementMap,
    destination: &dyn ElementMap,
    extents: &[usize],
    scope: TileScope,
    itemsize: usize,
    destination_lane_private: bool,
) -> OpResult<CopyPlan> {
    let count = element_count_for(extents)?;
    let active_lanes: Vec<_> = warp.active_mask.lanes().collect();
    let mut elements = Vec::new();
    let mut destination_rank = None;

    for linear in 0..count {
        let coordinates = coordinates_for(extents, linear)?;
        let mut source_candidates = Vec::new();
        let mut destination_candidates = Vec::new();

        for &lane in &active_lanes {
            let source_ref = wrap(source.map(&coordinates, lane), "typed copy source map")?;
            if source_ref.target_rank.is_some() {
                return Err(OpError::message(
                    "typed copy source map cannot select a remote CTA",
                ));
            }
            if source_ref.owned {
                let (source_index, source_in_bounds) =
                    byte_index(source_ref, itemsize, "typed copy source map")?;
                source_candidates.push((lane, source_index, source_in_bounds));
            }

            let destination_ref =
                wrap(destination.map(&coordinates, lane), "typed copy destination map")?;
            note_destination_rank(&mut destination_rank, destination_ref.target_rank)?;
            if destination_ref.owned {
                let (destination_index, destination_in_bounds) =
                    byte_index(destination_ref, itemsize, "typed copy destination map")?;
                destination_candidates.push((lane, destination_index, destination_in_bounds));
            }
        }

        let source_replicated =
            !active_lanes.is_empty() && source_candidates.len() == active_lanes.len();
        let destination_replicated =
            !active_lanes.is_empty() && destination_candidates.len() == active_lanes.len();
        let default_lanes = warp.scope_lanes(scope, linear)?;

        if source_candidates.is_empty() || destination_candidates.is_empty() {
            let collective_scope = matches!(scope, TileScope::Warpgroup | TileScope::Cta);
            let owned_in_another_warp = collective_scope
                && (source_candidates.is_empty() && destination_replicated
                    || destination_candidates.is_empty() && source_replicated
                    || source_candidates.is_empty() && destination_candidates.is_empty());
            if owned_in_another_warp {
                continue;
            }
            return Err(OpError::message(format!(
                "typed copy logical element {linear} has no source or destination owner in this execution scope"
            )));
        }

        let mut append = |source: (usize, i64, bool), destination: (usize, i64, bool)| {
            let (source_lane, source_index, source_in_bounds) = source;
            let (destination_lane, destination_index, destination_in_bounds) = destination;
            elements.push(TileCopyElement {
                source_index,
                source_in_bounds,
                source_lane,
                destination_index,
                destination_in_bounds,
                destination_lane,
            });
        };
        let source_at = |lane: usize| {
            source_candidates
                .iter()
                .copied()
                .find(|candidate| candidate.0 == lane)
        };
        let destination_at = |lane: usize| {
            destination_candidates
                .iter()
                .copied()
                .find(|candidate| candidate.0 == lane)
        };

        match (source_replicated, destination_replicated) {
            (true, true) => {
                let owner_lanes = if destination_lane_private {
                    &active_lanes
                } else {
                    &default_lanes
                };
                for &lane in owner_lanes {
                    append(
                        source_at(lane).expect("replicated source includes every active lane"),
                        destination_at(lane)
                            .expect("replicated destination includes every active lane"),
                    );
                }
            }
            (true, false) => {
                for destination in destination_candidates.iter().copied() {
                    append(
                        source_at(destination.0)
                            .expect("replicated source includes destination owner lane"),
                        destination,
                    );
                }
            }
            (false, true) => {
                for source in source_candidates.iter().copied() {
                    append(
                        source,
                        destination_at(source.0)
                            .expect("replicated destination includes source owner lane"),
                    );
                }
            }
            (false, false) if source_candidates.len() == 1 => {
                let source = source_candidates[0];
                for destination in destination_candidates.iter().copied() {
                    append(source, destination);
                }
            }
            (false, false) if destination_candidates.len() == 1 => {
                let destination = destination_candidates[0];
                let source = source_at(destination.0).unwrap_or(source_candidates[0]);
                append(source, destination);
            }
            (false, false) if source_candidates.len() == destination_candidates.len() => {
                for (source, destination) in source_candidates
                    .iter()
                    .copied()
                    .zip(destination_candidates.iter().copied())
                {
                    append(source, destination);
                }
            }
            _ => {
                return Err(OpError::message(format!(
                    "typed copy logical element {linear} has incompatible source/destination owner counts {} and {}",
                    source_candidates.len(),
                    destination_candidates.len()
                )));
            }
        }
    }

    Ok(CopyPlan {
        elements,
        destination_rank,
    })
}

/// TCGEN register ownership is part of the frontend layout (e.g. `.16x*b`
/// atoms are not a row-major round-robin partition). Probe every active
/// owner lane; an out-of-bounds map entry means "this lane does not own
/// this logical element".
pub fn mapped_tcgen_elements(
    active_mask: WarpMask,
    view: &dyn ElementMap,
    extents: &[usize],
    label: &str,
) -> OpResult<Vec<MappedElement>> {
    let mut result = Vec::new();
    for linear in 0..element_count_for(extents)? {
        let coordinates = coordinates_for(extents, linear)?;
        let owners = WarpMask(wrap(view.owners(&coordinates), label)?.bits() & active_mask.bits());
        for lane in owners.lanes() {
            let reference = wrap(view.map(&coordinates, lane), label)?;
            if !reference.in_bounds {
                continue;
            }
            result.push(MappedElement {
                execution_lane: lane,
                target_cta: reference.target_rank.map(|rank| rank as usize),
                location: reference.location,
            });
        }
    }
    Ok(result)
}

/// Map one in-bounds GEMM operand element; a map-selected CTA must agree
/// with the instruction's target CTA.
pub fn map_gemm_element(
    view: &dyn ElementMap,
    coordinates: &[i64],
    lane: usize,
    instruction_target: Option<usize>,
    label: &str,
) -> OpResult<MappedElement> {
    let reference = wrap(view.map(coordinates, lane), label)?;
    if !reference.in_bounds {
        return Err(OpError::message(format!(
            "{label} produced an out-of-bounds element"
        )));
    }
    let mapped_target = reference.target_rank.map(|rank| rank as usize);
    if let (Some(mapped), Some(instruction)) = (mapped_target, instruction_target) {
        if mapped != instruction {
            return Err(OpError::message(format!(
                "{label} selected CTA {mapped}, but the instruction targets CTA {instruction}"
            )));
        }
    }
    Ok(MappedElement {
        execution_lane: lane,
        target_cta: instruction_target.or(mapped_target),
        location: reference.location,
    })
}

/// Row-major `rows x columns` operand matrix (coordinates `(row, col)`, or
/// `(col, row)` when `transpose_storage`), rows offset by `row_offset`.
#[allow(clippy::too_many_arguments)]
pub fn map_gemm_matrix(
    view: &dyn ElementMap,
    rows: usize,
    columns: usize,
    transpose_storage: bool,
    lane: usize,
    target_cta: Option<usize>,
    row_offset: usize,
    label: &str,
) -> OpResult<Vec<MappedElement>> {
    let mut elements = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or_else(|| OpError::message("GEMM matrix element count overflow"))?,
    );
    for row in 0..rows {
        for column in 0..columns {
            let row = row
                .checked_add(row_offset)
                .ok_or_else(|| OpError::message("GEMM row offset overflow"))?;
            let row = i64::try_from(row).map_err(|_| OpError::message("GEMM row exceeds i64"))?;
            let column =
                i64::try_from(column).map_err(|_| OpError::message("GEMM column exceeds i64"))?;
            let coordinates = if transpose_storage {
                [column, row]
            } else {
                [row, column]
            };
            elements.push(map_gemm_element(view, &coordinates, lane, target_cta, label)?);
        }
    }
    Ok(elements)
}

/// Register-resident operand matrix: the frontend marks exactly one owning
/// (in-bounds) lane per logical element.
pub fn map_register_gemm_matrix(
    view: &dyn ElementMap,
    active_mask: WarpMask,
    rows: usize,
    columns: usize,
    transpose_storage: bool,
    label: &str,
) -> OpResult<Vec<MappedElement>> {
    let mut elements = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or_else(|| OpError::message("warp GEMM matrix element count overflow"))?,
    );
    for row in 0..rows {
        for column in 0..columns {
            let row = i64::try_from(row).map_err(|_| OpError::message("GEMM row exceeds i64"))?;
            let column =
                i64::try_from(column).map_err(|_| OpError::message("GEMM column exceeds i64"))?;
            let coordinates = if transpose_storage {
                [column, row]
            } else {
                [row, column]
            };
            let mut owner = None;
            for lane in active_mask.lanes() {
                let reference = wrap(view.map(&coordinates, lane), label)?;
                if !reference.in_bounds {
                    continue;
                }
                if reference.target_rank.is_some() {
                    return Err(OpError::message(format!(
                        "{label} register map selected a CTA rank"
                    )));
                }
                if owner.is_some() {
                    return Err(OpError::message(format!(
                        "{label} has more than one owning lane at ({row}, {column})"
                    )));
                }
                owner = Some(MappedElement {
                    execution_lane: lane,
                    target_cta: None,
                    location: reference.location,
                });
            }
            elements.push(owner.ok_or_else(|| {
                OpError::message(format!("{label} has no owning lane at ({row}, {column})"))
            })?);
        }
    }
    Ok(elements)
}

/// TMEM block-scale matrix. With a runtime scale-selector descriptor, a map
/// that repeats its first `values_per_instruction` cells is redirected to
/// `selector + column % values_per_instruction` TMEM columns.
#[allow(clippy::too_many_arguments)]
pub fn map_gemm_scale_matrix(
    view: &dyn ElementMap,
    rows: usize,
    columns: usize,
    lane: usize,
    target_cta: Option<usize>,
    row_offset: usize,
    descriptor_selector: Option<usize>,
    values_per_instruction: usize,
    label: &str,
) -> OpResult<Vec<MappedElement>> {
    let mut elements = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or_else(|| OpError::message("GEMM scale element count overflow"))?,
    );
    for row in 0..rows {
        for column in 0..columns {
            let mapped_row = row
                .checked_add(row_offset)
                .ok_or_else(|| OpError::message("GEMM scale row offset overflow"))?;
            let coordinates = [
                i64::try_from(mapped_row)
                    .map_err(|_| OpError::message("GEMM scale row exceeds i64"))?,
                i64::try_from(column)
                    .map_err(|_| OpError::message("GEMM scale column exceeds i64"))?,
            ];
            elements.push(map_gemm_element(view, &coordinates, lane, target_cta, label)?);
        }
    }
    if let Some(selector) = descriptor_selector {
        if values_per_instruction == 0 {
            return Err(OpError::message(
                "GEMM runtime scale selector has no values per instruction",
            ));
        }
        let reuses_descriptor_cell = columns > values_per_instruction
            && (0..rows).all(|row| {
                (values_per_instruction..columns).all(|column| {
                    elements[row * columns + column].location
                        == elements[row * columns + column % values_per_instruction].location
                })
            });
        if reuses_descriptor_cell {
            for row in 0..rows {
                for column in 0..columns {
                    let slot = selector
                        .checked_add(column % values_per_instruction)
                        .ok_or_else(|| OpError::message("GEMM scale selector overflow"))?;
                    let slot = i64::try_from(slot)
                        .map_err(|_| OpError::message("GEMM scale selector exceeds i64"))?;
                    let element = &mut elements[row * columns + column];
                    element.location = match element.location {
                        ElementLocation::Tmem {
                            mapped_lane,
                            tcol_element,
                            allocated_addr,
                            bit_offset,
                        } => ElementLocation::Tmem {
                            mapped_lane,
                            tcol_element: tcol_element
                                .checked_add(slot)
                                .ok_or_else(|| OpError::message("GEMM scale TMEM column overflow"))?,
                            allocated_addr,
                            bit_offset,
                        },
                        _ => {
                            return Err(OpError::message(format!(
                                "{label} expected a TMEM scale coordinate"
                            )))
                        }
                    };
                }
            }
        }
    }
    Ok(elements)
}

/// TMEM origin of a recognized m64 `.16x256b.x8` f32 load warp slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastTmemF32M64Load {
    pub base_lane: i64,
    pub base_tcol: i64,
    pub allocated_addr: i64,
}

/// Recognize the physical mapping of one complete m64 `.16x256b.x8`
/// `tcgen05.ld` warp slice (32 f32 per lane). Any other map returns `None`
/// and continues through the scalar reference path.
pub fn fast_tmem_f32_m64_load(
    active_mask: WarpMask,
    warp_id_in_cta: usize,
    source: &[MappedElement],
    destination: &[MappedElement],
) -> Option<FastTmemF32M64Load> {
    const ELEMENTS_PER_LANE: usize = 32;
    const ELEMENT_COUNT: usize = WARP_SIZE * ELEMENTS_PER_LANE;

    if active_mask != WarpMask::ALL
        || source.len() != ELEMENT_COUNT
        || destination.len() != ELEMENT_COUNT
    {
        return None;
    }

    let warp_lane_offset = (warp_id_in_cta % 4).checked_mul(WARP_SIZE)?;
    let mut seen = [false; ELEMENT_COUNT];
    let mut base: Option<FastTmemF32M64Load> = None;
    for (&source, &destination) in source.iter().zip(destination) {
        if source.execution_lane != destination.execution_lane
            || source.target_cta.is_some()
            || destination.target_cta.is_some()
        {
            return None;
        }
        let execution_lane = source.execution_lane;
        let ElementLocation::ByteOffset(destination_offset) = destination.location else {
            return None;
        };
        let destination_offset = usize::try_from(destination_offset).ok()?;
        if destination_offset % std::mem::size_of::<f32>() != 0 {
            return None;
        }
        let slot = destination_offset / std::mem::size_of::<f32>();
        if execution_lane >= WARP_SIZE || slot >= ELEMENTS_PER_LANE {
            return None;
        }
        let seen_slot = execution_lane * ELEMENTS_PER_LANE + slot;
        if std::mem::replace(&mut seen[seen_slot], true) {
            return None;
        }

        let (mapped_lane, tcol, allocated_addr) = tmem_coordinates(source, "fast m64 load").ok()?;
        let lane_group = execution_lane / 4;
        let lane_in_group = execution_lane % 4;
        let block = slot / 4;
        let lane_bank = (slot % 4) / 2;
        let pair = slot % 2;
        let expected_lane_offset = warp_lane_offset + lane_bank * 8 + lane_group;
        let expected_tcol_offset = block * 8 + lane_in_group * 2 + pair;
        let candidate = FastTmemF32M64Load {
            base_lane: mapped_lane.checked_sub(i64::try_from(expected_lane_offset).ok()?)?,
            base_tcol: tcol.checked_sub(i64::try_from(expected_tcol_offset).ok()?)?,
            allocated_addr,
        };
        match base {
            Some(base)
                if base.base_lane != candidate.base_lane
                    || base.base_tcol != candidate.base_tcol
                    || base.allocated_addr != candidate.allocated_addr =>
            {
                return None;
            }
            None => base = Some(candidate),
            _ => {}
        }
    }
    seen.into_iter().all(|value| value).then_some(base?)
}

/// TMEM footprint rows of a canonical `.32x32b` warp transfer: each active
/// lane owns TLane `base_lane + 32 * (warp % 4) + lane`, one row of
/// `row_bytes` starting at `base_tcol`. Rows are `(issuer, lane, None, TLane,
/// TCol, allocated_addr, row_bytes)`. `mnemonic` prefixes diagnostics.
#[allow(clippy::type_complexity)]
pub fn canonical_32x32b_tmem_rows(
    active_mask: WarpMask,
    warp_id_in_cta: usize,
    issuer_lane: usize,
    base_lane: i64,
    base_tcol: i64,
    allocated_addr: i64,
    row_bytes: usize,
    mnemonic: &str,
) -> OpResult<Vec<(usize, usize, Option<usize>, i64, i64, i64, usize)>> {
    let warp_lane_offset = (warp_id_in_cta % 4)
        .checked_mul(WARP_SIZE)
        .ok_or_else(|| OpError::message(format!("{mnemonic} warp offset overflow")))?;
    let mut rows = Vec::with_capacity(WARP_SIZE);
    for execution_lane in active_mask.lanes() {
        let lane_offset = warp_lane_offset
            .checked_add(execution_lane)
            .ok_or_else(|| OpError::message(format!("{mnemonic} lane offset overflow")))?;
        let mapped_lane = base_lane
            .checked_add(i64::try_from(lane_offset).map_err(|_| {
                OpError::message(format!("{mnemonic} lane offset exceeds i64"))
            })?)
            .ok_or_else(|| OpError::message(format!("{mnemonic} TLane overflow")))?;
        rows.push((
            issuer_lane,
            execution_lane,
            None,
            mapped_lane,
            base_tcol,
            allocated_addr,
            row_bytes,
        ));
    }
    Ok(rows)
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
