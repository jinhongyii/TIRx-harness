//! `shfl.sync` / `redux.sync` lane math, delegating to `numsim_oplib::warp`.

use super::{OpError, OpResult};
use crate::dtype::{Dtype, Ty};
use crate::program::{ReduxOp, ShflMode};
use crate::value::{WarpMask, WarpValue, WARP_SIZE};
use numsim_oplib::warp as lib;

/// Lane selection exactly as legacy `shuffle_sources`; a source outside
/// `members` (undefined in PTX, an error in legacy) reads the source lane's
/// value anyway and reports `false` (see CONTRACT_REQUESTS 2026-10-07: `shfl`
/// cannot return an error).
pub(super) fn shfl(
    mode: ShflMode,
    src: &WarpValue<u64>,
    lane: &WarpValue<u64>,
    clamp: &WarpValue<u64>,
    members: WarpMask,
) -> (WarpValue<u64>, WarpMask) {
    let mode = match mode {
        ShflMode::Idx => lib::ShuffleMode::Index,
        ShflMode::Up => lib::ShuffleMode::Up,
        ShflMode::Down => lib::ShuffleMode::Down,
        ShflMode::Bfly => lib::ShuffleMode::Xor,
    };
    let selectors: WarpValue<u32> = std::array::from_fn(|l| lane[l] as u32);
    let controls: WarpValue<u32> = std::array::from_fn(|l| clamp[l] as u32);
    let participants = [members.bits(); WARP_SIZE];
    if let Ok((values, in_range)) = lib::shfl_sync(members, &participants, src, &selectors, &controls, mode) {
        let mut valid = WarpMask::NONE;
        for l in members.lanes() {
            if in_range[l] {
                valid = valid.or(WarpMask::lane(l));
            }
        }
        return (values, valid);
    }
    // Fallback for a non-member source: resolve each lane independently.
    let mut values = [0u64; WARP_SIZE];
    let mut valid = WarpMask::NONE;
    for l in members.lanes() {
        let one = WarpMask::lane(l);
        let all = WarpMask::ALL;
        let parts = [u32::MAX; WARP_SIZE];
        if let Ok((sources, in_range)) = lib::shuffle_sources(one.or(all), &parts, &selectors, &controls, mode) {
            values[l] = src[sources[l]];
            if in_range[l] && members.contains(sources[l]) {
                valid = valid.or(one);
            }
        }
    }
    (values, valid)
}

/// `redux.sync` over `members`: fold from the lowest member in ascending
/// lane order (legacy order); `ty` selects u32 / s32 / f32 (min/max, NaN
/// canonicalized). `.NaN` variants are not expressible through `ReduxOp`.
pub(super) fn redux(op: ReduxOp, ty: Ty, src: &WarpValue<u64>, members: WarpMask) -> OpResult<u64> {
    let Some(first) = members.first() else {
        return Err(OpError::invalid("redux.sync with an empty membermask"));
    };
    if !ty.is_scalar() {
        return Err(OpError::unsupported(format!("redux.sync.{ty}")));
    }
    let parts = [members.bits(); WARP_SIZE];
    let int_op = match op {
        ReduxOp::Add => lib::ReduxIntOp::Add,
        ReduxOp::Min => lib::ReduxIntOp::Min,
        ReduxOp::Max => lib::ReduxIntOp::Max,
        ReduxOp::And => lib::ReduxIntOp::And,
        ReduxOp::Or => lib::ReduxIntOp::Or,
        ReduxOp::Xor => lib::ReduxIntOp::Xor,
    };
    match ty.elem {
        Dtype::U32 => {
            let values: WarpValue<u32> = std::array::from_fn(|l| src[l] as u32);
            Ok(u64::from(lib::redux_sync_u32(members, &parts, &values, int_op)?[first]))
        }
        Dtype::S32 => {
            let values: WarpValue<i32> = std::array::from_fn(|l| src[l] as u32 as i32);
            if matches!(op, ReduxOp::And | ReduxOp::Or | ReduxOp::Xor) {
                // Bitwise ops are type-agnostic (`.b32`).
                let bits: WarpValue<u32> = std::array::from_fn(|l| src[l] as u32);
                return Ok(u64::from(lib::redux_sync_u32(members, &parts, &bits, int_op)?[first]));
            }
            Ok(u64::from(lib::redux_sync_i32(members, &parts, &values, int_op)?[first] as u32))
        }
        Dtype::F32 => {
            let f_op = match op {
                ReduxOp::Min => lib::ReduxF32Op::Min,
                ReduxOp::Max => lib::ReduxF32Op::Max,
                _ => return Err(OpError::unsupported(format!("redux.sync.{op:?}.f32"))),
            };
            let values: WarpValue<f32> = std::array::from_fn(|l| f32::from_bits(src[l] as u32));
            Ok(u64::from(lib::redux_sync_f32(members, &parts, &values, f_op)?[first].to_bits()))
        }
        other => Err(OpError::unsupported(format!("redux.sync.{op:?}.{other}"))),
    }
}
