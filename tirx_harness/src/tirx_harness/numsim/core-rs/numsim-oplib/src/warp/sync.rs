//! Membermask-synchronized warp collectives: `vote`, `match`, `redux`,
//! `elect`, plus the lane-local `fns` wrapper and `movmatrix`.
//!
//! Legacy sources: `engine-rs/src/runtime/instructions/warp.rs`
//! (`vote_predicate`, `match_values`, `execute_redux`, `elect_sync`,
//! `movmatrix_spec`) and `engine-rs/src/runtime/warp_ops.rs`
//! (`warp_ballot_sync`, `warp_fns_b32`).
//!
//! All of these validate the `membermask` operands first (see
//! [`super::mask::validate_participants`]) using the legacy operation label
//! (`"vote.sync"`, `"warp ballot"`, `"match.sync"`, `"redux.sync"`,
//! `"elect.sync"`). Where legacy broadcast a uniform result with `R::splat`,
//! every lane (including inactive ones) receives it; per-lane results leave
//! inactive lanes at zero.

use super::mask::validate_participants;
use crate::scalar::{cuda_canonicalize_nan_f32, ptx_fns_b32, ptx_max_f32, ptx_min_f32};
use crate::types::{OpError, OpResult, WarpMask, WarpValue, WARP_SIZE};

fn member_values<T: Copy>(members: u32, values: &WarpValue<T>) -> Vec<T> {
    (0..WARP_SIZE)
        .filter(|lane| members & (1_u32 << lane) != 0)
        .map(|lane| values[lane])
        .collect()
}

fn vote_predicate(
    active_mask: WarpMask,
    member_masks: &WarpValue<u32>,
    predicates: &WarpValue<bool>,
    decide: impl Fn(&[bool]) -> bool,
) -> OpResult<WarpValue<bool>> {
    let members = validate_participants(active_mask, member_masks, "vote.sync")?;
    Ok([decide(&member_values(members, predicates)); WARP_SIZE])
}

/// `vote.sync.all.pred`: true iff every member predicate is true.
pub fn vote_all(
    active_mask: WarpMask,
    member_masks: &WarpValue<u32>,
    predicates: &WarpValue<bool>,
) -> OpResult<WarpValue<bool>> {
    vote_predicate(active_mask, member_masks, predicates, |values| {
        values.iter().copied().all(|value| value)
    })
}

/// `vote.sync.any.pred`: true iff some member predicate is true.
pub fn vote_any(
    active_mask: WarpMask,
    member_masks: &WarpValue<u32>,
    predicates: &WarpValue<bool>,
) -> OpResult<WarpValue<bool>> {
    vote_predicate(active_mask, member_masks, predicates, |values| {
        values.iter().copied().any(|value| value)
    })
}

/// `vote.sync.uni.pred`: true iff all member predicates are equal.
pub fn vote_uni(
    active_mask: WarpMask,
    member_masks: &WarpValue<u32>,
    predicates: &WarpValue<bool>,
) -> OpResult<WarpValue<bool>> {
    vote_predicate(active_mask, member_masks, predicates, |values| {
        values
            .first()
            .is_none_or(|first| values.iter().all(|value| value == first))
    })
}

/// `vote.sync.ballot.b32`: each active lane receives the bitmask of member
/// lanes whose predicate is true; inactive lanes receive 0.
pub fn vote_ballot(
    active_mask: WarpMask,
    participant_masks: &WarpValue<u32>,
    predicates: &WarpValue<bool>,
) -> OpResult<WarpValue<u32>> {
    let participant_mask = validate_participants(active_mask, participant_masks, "warp ballot")?;
    let mut result = [0_u32; WARP_SIZE];
    for lane in active_mask.iter() {
        let mut bits = 0_u32;
        for source_lane in 0..WARP_SIZE {
            if participant_mask & (1_u32 << source_lane) != 0 && predicates[source_lane] {
                bits |= 1_u32 << source_lane;
            }
        }
        result[lane] = bits;
    }
    Ok(result)
}

/// `match.any.sync.b32/b64`: each active lane receives the mask of member lanes
/// holding an equal value; inactive lanes receive 0.
pub fn match_any<T: Copy + PartialEq>(
    active_mask: WarpMask,
    member_masks: &WarpValue<u32>,
    values: &WarpValue<T>,
) -> OpResult<WarpValue<u32>> {
    let members = validate_participants(active_mask, member_masks, "match.sync")?;
    let mut matching = [0_u32; WARP_SIZE];
    for destination_lane in active_mask.iter() {
        let destination_value = values[destination_lane];
        matching[destination_lane] = (0..WARP_SIZE)
            .filter(|source_lane| members & (1_u32 << source_lane) != 0)
            .filter(|source_lane| destination_value == values[*source_lane])
            .fold(0_u32, |mask, source_lane| mask | (1_u32 << source_lane));
    }
    Ok(matching)
}

/// `match.all.sync.b32/b64`: `(membermask or 0, all-equal predicate)`, both
/// broadcast to every lane. Comparison is against the first member lane.
pub fn match_all<T: Copy + PartialEq>(
    active_mask: WarpMask,
    member_masks: &WarpValue<u32>,
    values: &WarpValue<T>,
) -> OpResult<(WarpValue<u32>, WarpValue<bool>)> {
    let members = validate_participants(active_mask, member_masks, "match.sync")?;
    let first = WarpMask::from_bits(members)
        .first_active()
        .expect("validated member mask is nonempty");
    let all_equal = (first + 1..WARP_SIZE)
        .filter(|lane| members & (1_u32 << lane) != 0)
        .all(|lane| values[first] == values[lane]);
    Ok((
        [if all_equal { members } else { 0 }; WARP_SIZE],
        [all_equal; WARP_SIZE],
    ))
}

/// Integer `redux.sync` operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduxIntOp {
    Add,
    Min,
    Max,
    And,
    Or,
    Xor,
}

/// `redux.sync.{min,max}{,.NaN}.f32` operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduxF32Op {
    Min,
    Max,
    MinNan,
    MaxNan,
}

/// Deterministic member fold: start at the lowest member lane and combine the
/// remaining members in ascending lane order.
fn redux_fold<T: Copy>(
    active_mask: WarpMask,
    member_masks: &WarpValue<u32>,
    values: &WarpValue<T>,
    combine: impl Fn(T, T) -> T,
) -> OpResult<T> {
    let members = validate_participants(active_mask, member_masks, "redux.sync")?;
    let first = WarpMask::from_bits(members)
        .first_active()
        .expect("validated member mask is nonempty");
    Ok((first + 1..WARP_SIZE)
        .filter(|lane| members & (1_u32 << lane) != 0)
        .fold(values[first], |accumulator, lane| combine(accumulator, values[lane])))
}

/// `redux.sync.op.u32` / `.b32` (min/max compare unsigned); result broadcast.
pub fn redux_sync_u32(
    active_mask: WarpMask,
    member_masks: &WarpValue<u32>,
    values: &WarpValue<u32>,
    op: ReduxIntOp,
) -> OpResult<WarpValue<u32>> {
    let combine = match op {
        ReduxIntOp::Add => u32::wrapping_add,
        ReduxIntOp::Min => u32::min,
        ReduxIntOp::Max => u32::max,
        ReduxIntOp::And => |lhs: u32, rhs: u32| lhs & rhs,
        ReduxIntOp::Or => |lhs: u32, rhs: u32| lhs | rhs,
        ReduxIntOp::Xor => |lhs: u32, rhs: u32| lhs ^ rhs,
    };
    Ok([redux_fold(active_mask, member_masks, values, combine)?; WARP_SIZE])
}

/// `redux.sync.{add,min,max}.s32`; result broadcast. Bitwise ops are not a
/// legal `.s32` form and are rejected.
pub fn redux_sync_i32(
    active_mask: WarpMask,
    member_masks: &WarpValue<u32>,
    values: &WarpValue<i32>,
    op: ReduxIntOp,
) -> OpResult<WarpValue<i32>> {
    let combine: fn(i32, i32) -> i32 = match op {
        ReduxIntOp::Add => i32::wrapping_add,
        ReduxIntOp::Min => i32::min,
        ReduxIntOp::Max => i32::max,
        ReduxIntOp::And | ReduxIntOp::Or | ReduxIntOp::Xor => {
            return Err(OpError::message(format!(
                "redux.sync.{op:?}.s32 is not a PTX instruction form"
            )))
        }
    };
    Ok([redux_fold(active_mask, member_masks, values, combine)?; WARP_SIZE])
}

/// `redux.sync.{min,max}{,.NaN}.f32` (no `.abs`, no ftz); result broadcast.
/// The final value is NaN-canonicalized (`0x7fffffff`) even for a single
/// member, which never calls `combine`.
pub fn redux_sync_f32(
    active_mask: WarpMask,
    member_masks: &WarpValue<u32>,
    values: &WarpValue<f32>,
    op: ReduxF32Op,
) -> OpResult<WarpValue<f32>> {
    let combine = match op {
        ReduxF32Op::Min => |lhs, rhs| ptx_min_f32(lhs, rhs, false, false),
        ReduxF32Op::Max => |lhs, rhs| ptx_max_f32(lhs, rhs, false, false),
        ReduxF32Op::MinNan => |lhs, rhs| ptx_min_f32(lhs, rhs, false, true),
        ReduxF32Op::MaxNan => |lhs, rhs| ptx_max_f32(lhs, rhs, false, true),
    };
    let reduced = redux_fold(active_mask, member_masks, values, combine)?;
    Ok([cuda_canonicalize_nan_f32(reduced); WARP_SIZE])
}

/// `elect.sync d|p, membermask`: the deterministic representative is the
/// lowest member lane. Returns `(leader lane broadcast, lane == leader)`.
pub fn elect_sync(
    active_mask: WarpMask,
    member_masks: &WarpValue<u32>,
) -> OpResult<(WarpValue<u32>, WarpValue<bool>)> {
    let members = validate_participants(active_mask, member_masks, "elect.sync")?;
    let elected = WarpMask::from_bits(members)
        .first_active()
        .ok_or_else(|| OpError::message("elect.sync has no participating lane"))?;
    Ok((
        [elected as u32; WARP_SIZE],
        std::array::from_fn(|lane| elected == lane),
    ))
}

/// `fns.b32` per active lane; bases outside `0..=31` are rejected.
pub fn fns_b32(
    active_mask: WarpMask,
    masks: &WarpValue<u32>,
    bases: &WarpValue<u32>,
    offsets: &WarpValue<i32>,
) -> OpResult<WarpValue<u32>> {
    let mut result = [0_u32; WARP_SIZE];
    for lane in active_mask.iter() {
        if bases[lane] >= 32 {
            return Err(OpError::message("fns.b32 base is outside the defined 0..31 range"));
        }
        result[lane] = ptx_fns_b32(masks[lane], bases[lane], offsets[lane]);
    }
    Ok(result)
}

/// `movmatrix.sync.aligned.m8n8.trans.b16`: requires a full warp (validated
/// as a 0xffffffff membermask), then transposes the 8x8 b16 fragment.
pub fn movmatrix_m8n8_trans_b16(
    active_mask: WarpMask,
    values: &WarpValue<u32>,
) -> OpResult<WarpValue<u32>> {
    validate_participants(
        active_mask,
        &[u32::MAX; WARP_SIZE],
        "movmatrix.sync.aligned.m8n8.trans.b16",
    )?;
    let mut transposed = [0_u32; WARP_SIZE];
    for destination_lane in 0..WARP_SIZE {
        let column = destination_lane / 4;
        let row_pair = destination_lane % 4;
        let source_pair = column / 2;
        let source_shift = (column % 2) * 16;
        let source_lane_low = 2 * row_pair * 4 + source_pair;
        let source_lane_high = (2 * row_pair + 1) * 4 + source_pair;
        let low = (values[source_lane_low] >> source_shift) & 0xffff;
        let high = (values[source_lane_high] >> source_shift) & 0xffff;
        transposed[destination_lane] = low | (high << 16);
    }
    Ok(transposed)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: [u32; 32] = [u32::MAX; 32];

    fn from_fn<T>(f: impl FnMut(usize) -> T) -> WarpValue<T> {
        std::array::from_fn(f)
    }

    // Ported from instructions/warp.rs
    // `vote_match_and_redux_are_one_instruction_over_the_runtime_member_mask`.
    #[test]
    fn vote_match_and_redux_are_one_instruction_over_the_runtime_member_mask() {
        let ballot = vote_ballot(WarpMask::FULL, &FULL, &from_fn(|lane| lane % 2 == 0)).unwrap();
        assert_eq!(ballot[0], 0x5555_5555);
        let sum = redux_sync_u32(WarpMask::FULL, &FULL, &[1; 32], ReduxIntOp::Add).unwrap();
        assert_eq!(sum[0], 32);
        let matches = match_any(WarpMask::FULL, &FULL, &from_fn(|lane| (lane % 4) as u32)).unwrap();
        assert_eq!(matches[0], 0x1111_1111);
        assert_eq!(matches[1], 0x2222_2222);
        let iota = from_fn(|lane| lane as u32);
        let sums = redux_sync_u32(WarpMask::FULL, &FULL, &iota, ReduxIntOp::Add).unwrap();
        let minima = redux_sync_u32(WarpMask::FULL, &FULL, &iota, ReduxIntOp::Min).unwrap();
        let any = vote_any(WarpMask::FULL, &FULL, &from_fn(|lane| lane == 31)).unwrap();
        for lane in 0..32 {
            assert_eq!(sums[lane], 496);
            assert_eq!(minima[lane], 0);
            assert!(any[lane]);
            assert_eq!(ballot[lane], 0x5555_5555);
        }
        let active = WarpMask::from_bits((1 << 3) | (1 << 7) | (1 << 12));
        let (leader, elected) = elect_sync(active, &[active.bits(); 32]).unwrap();
        for lane in 0..32 {
            assert_eq!(leader[lane], 3);
            assert_eq!(elected[lane], lane == 3);
        }
    }

    #[test]
    fn checked_fns_applies_lane_values_and_rejects_invalid_bases() {
        let masks = [0b10110_u32; 32];
        let bases = [1_u32; 32];
        let offsets = from_fn(|lane| if lane % 2 == 0 { 1 } else { 2 });
        let result = fns_b32(WarpMask::FULL, &masks, &bases, &offsets).unwrap();
        for lane in 0..WARP_SIZE {
            assert_eq!(result[lane], if lane % 2 == 0 { 1 } else { 2 });
        }
        let invalid_bases = from_fn(|lane| if lane == 7 { 32 } else { 0 });
        let error = fns_b32(WarpMask::FULL, &masks, &invalid_bases, &offsets).unwrap_err();
        assert_eq!(error.to_string(), "fns.b32 base is outside the defined 0..31 range");
    }

    #[test]
    fn vote_variants_use_only_member_lanes() {
        let active = WarpMask::from_bits(0xff);
        let members = [0xff_u32; 32];
        let predicates = from_fn(|lane| lane < 8 || lane == 20);
        assert_eq!(vote_all(active, &members, &predicates).unwrap(), [true; 32]);
        assert_eq!(vote_uni(active, &members, &predicates).unwrap(), [true; 32]);
        let mixed = from_fn(|lane| lane == 2);
        assert_eq!(vote_uni(active, &members, &mixed).unwrap(), [false; 32]);
        assert_eq!(vote_all(active, &members, &mixed).unwrap(), [false; 32]);
        let ballot = vote_ballot(active, &members, &predicates).unwrap();
        assert_eq!(ballot[0], 0xff);
        assert_eq!(ballot[20], 0);
        let error = vote_any(active, &FULL, &mixed).unwrap_err();
        assert!(error.to_string().starts_with("vote.sync participant mask names an inactive lane"));
    }

    #[test]
    fn match_all_reports_mask_and_predicate() {
        let (mask, pred) = match_all(WarpMask::FULL, &FULL, &[5_u64; 32]).unwrap();
        assert_eq!((mask[9], pred[9]), (u32::MAX, true));
        let (mask, pred) =
            match_all(WarpMask::FULL, &FULL, &from_fn(|lane| u64::from(lane == 4))).unwrap();
        assert_eq!((mask[0], pred[0]), (0, false));
    }

    #[test]
    fn redux_variants_cover_bitwise_signed_and_float_nan_rules() {
        let values = from_fn(|lane| 1_u32 << (lane % 4));
        assert_eq!(redux_sync_u32(WarpMask::FULL, &FULL, &values, ReduxIntOp::Or).unwrap()[0], 0xf);
        assert_eq!(redux_sync_u32(WarpMask::FULL, &FULL, &values, ReduxIntOp::And).unwrap()[0], 0);
        assert_eq!(redux_sync_u32(WarpMask::FULL, &FULL, &values, ReduxIntOp::Xor).unwrap()[0], 0);
        let signed = from_fn(|lane| lane as i32 - 16);
        assert_eq!(redux_sync_i32(WarpMask::FULL, &FULL, &signed, ReduxIntOp::Min).unwrap()[0], -16);
        assert_eq!(redux_sync_i32(WarpMask::FULL, &FULL, &signed, ReduxIntOp::Max).unwrap()[0], 15);
        assert!(redux_sync_i32(WarpMask::FULL, &FULL, &signed, ReduxIntOp::Xor).is_err());
        let floats = from_fn(|lane| if lane == 5 { f32::NAN } else { lane as f32 });
        assert_eq!(redux_sync_f32(WarpMask::FULL, &FULL, &floats, ReduxF32Op::Max).unwrap()[0], 31.0);
        let nan = redux_sync_f32(WarpMask::FULL, &FULL, &floats, ReduxF32Op::MaxNan).unwrap()[0];
        assert_eq!(nan.to_bits(), 0x7fff_ffff);
        // A single NaN member is canonicalized without combine.
        let single = WarpMask::from_bits(1 << 5);
        let only = redux_sync_f32(single, &[1 << 5; 32], &floats, ReduxF32Op::Min).unwrap()[0];
        assert_eq!(only.to_bits(), 0x7fff_ffff);
    }

    #[test]
    fn movmatrix_transposes_and_requires_full_warp() {
        // Element (row r, col c) of the 8x8 b16 matrix lives in lane r*4 + c/2,
        // half c%2. Encode it as r*8 + c.
        let values = from_fn(|lane| {
            let row = lane / 4;
            let col = (lane % 4) * 2;
            ((row * 8 + col) as u32) | (((row * 8 + col + 1) as u32) << 16)
        });
        let transposed = movmatrix_m8n8_trans_b16(WarpMask::FULL, &values).unwrap();
        for lane in 0..32 {
            let row = lane / 4;
            let col = (lane % 4) * 2;
            assert_eq!(transposed[lane] & 0xffff, (col * 8 + row) as u32);
            assert_eq!(transposed[lane] >> 16, ((col + 1) * 8 + row) as u32);
        }
        assert!(movmatrix_m8n8_trans_b16(WarpMask::from_bits(0xffff), &values).is_err());
    }
}
