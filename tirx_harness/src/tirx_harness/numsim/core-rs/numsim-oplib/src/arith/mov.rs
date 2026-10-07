//! Register moves: `mov`, `mov` pack/unpack of `b32/b64/b128`, `createpolicy`
//! value contract, and the pure special-register representatives.
//!
//! Moved from legacy `engine-rs/src/runtime/instructions/reg.rs`
//! (`move_variant!`, `mov_pack_spec`/`mov_unpack_spec` B32/B64/B128,
//! `createpolicy_spec`, `special_register_variant!` LaneId/Clock64) and
//! `mov_unpack.rs`. The `b16x4`/`b32x4`/`b64x2` forms live in `crate::cvt`.

use crate::scalar::U64x2;
use crate::types::{OpError, OpResult};

/// `mov.type d, a` (value copy).
pub fn mov<T>(value: T) -> T {
    value
}

/// `mov.b32 d, {lo, hi}` from two `b16`.
pub fn mov_pack_b32(low: u16, high: u16) -> u32 {
    u32::from(low) | (u32::from(high) << 16)
}

/// `mov.b32 {lo, hi}, a` into two `b16`.
pub fn mov_unpack_b32(value: u32) -> (u16, u16) {
    (value as u16, (value >> 16) as u16)
}

/// `mov.b64 d, {lo, hi}` from two `b32`.
pub fn mov_pack_b64(low: u32, high: u32) -> u64 {
    u64::from(low) | (u64::from(high) << 32)
}

/// `mov.b64 {lo, hi}, a` into two `b32`.
pub fn mov_unpack_b64(value: u64) -> (u32, u32) {
    (value as u32, (value >> 32) as u32)
}

/// `mov.b128 d, {lo, hi}` from two `b64`.
pub fn mov_pack_b128(low: u64, high: u64) -> U64x2 {
    [low, high]
}

/// `createpolicy.fractional`: PTX policy bits are opaque and every valid
/// policy is equivalent in a model without cache residency; `0` is NumSim's
/// private representative. The fraction must lie in `(0, 1]`.
pub fn createpolicy_fraction(fraction: f32) -> OpResult<u64> {
    if !(fraction > 0.0 && fraction <= 1.0) {
        return Err(OpError::message("createpolicy fraction must be in (0, 1]"));
    }
    Ok(0)
}

/// `createpolicy.cvt.L2`: maps any policy to the representative `0`.
pub fn createpolicy_cvt(_property: u64) -> u64 {
    0
}

/// `createpolicy.range` value contract (the global-space address check is the
/// engine's): `primary` must not exceed `total`.
pub fn createpolicy_range(primary: u32, total: u32) -> OpResult<u64> {
    if primary > total {
        return Err(OpError::message(
            "createpolicy primary size exceeds total size",
        ));
    }
    Ok(0)
}

/// `%laneid` of lane `lane`.
pub fn sreg_laneid(lane: usize) -> u32 {
    lane as u32
}

/// `%clock64` deterministic representative (hardware timing is not modeled).
pub fn sreg_clock64() -> u64 {
    0
}
