//! PTX `shfl.sync` lane selection and data movement.
//!
//! Legacy source: `engine-rs/src/runtime/warp_ops.rs`
//! (`resolve_warp_shuffle_sources`, `warp_shuffle_source_mask`,
//! `warp_shuffle_ptx`) and the mode markers in
//! `engine-rs/src/runtime/instructions/warp.rs`.
//!
//! Operand `c` is packed PTX style: clamp in bits 4:0, segment mask in bits
//! 12:8. For every active lane `l` (only the low 5 bits of lane, `b`, and the
//! clamp/segmask fields are used):
//!
//! ```text
//! maxLane = (l & segmask) | (clamp & !segmask)      minLane = l & segmask
//! up:   j = l - b; valid = j >= maxLane (j < 0 => invalid, j := l)
//! down: j = l + b; valid = j <= maxLane
//! bfly: j = l ^ b; valid = j <= maxLane
//! idx:  j = minLane | (b & !segmask); valid = j <= maxLane
//! ```
//!
//! An invalid (out-of-range) lane reads its own value and gets predicate
//! `false`. The resolved source lane must be both a participant and active,
//! otherwise the whole instruction fails with
//! `"warp shuffle reads a non-participant lane"`. Inactive destination lanes
//! receive `T::default()` (legacy `RuntimeScalar::zero()`) and predicate
//! `false`.

use super::mask::validate_participants;
use crate::types::{OpError, OpResult, WarpMask, WarpValue, WARP_SIZE};

/// `shfl.sync` mode (`.idx`, `.up`, `.down`, `.bfly`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShuffleMode {
    Index,
    Up,
    Down,
    /// `.bfly` (CUDA `__shfl_xor_sync`).
    Xor,
}

/// Resolve the source lane and in-range predicate of each active lane.
///
/// Inactive lanes report source lane 0 and predicate `false`.
pub fn shuffle_sources(
    active_mask: WarpMask,
    participant_masks: &WarpValue<u32>,
    selectors: &WarpValue<u32>,
    controls: &WarpValue<u32>,
    mode: ShuffleMode,
) -> OpResult<(WarpValue<usize>, WarpValue<bool>)> {
    let participant_mask = validate_participants(active_mask, participant_masks, "shfl.sync")?;
    let mut source_lanes = [0_usize; WARP_SIZE];
    let mut in_range = [false; WARP_SIZE];
    for lane in active_mask.lanes() {
        let lane5 = lane & 0x1f;
        let selector = selectors[lane] as usize & 0x1f;
        let control = controls[lane] as usize;
        let clamp = control & 0x1f;
        let segment_mask = (control >> 8) & 0x1f;
        let maximum = (lane5 & segment_mask) | (clamp & !segment_mask & 0x1f);
        let minimum = lane5 & segment_mask;
        let (candidate, valid) = match mode {
            ShuffleMode::Up => {
                let candidate = lane5.checked_sub(selector);
                (
                    candidate.unwrap_or(lane5),
                    candidate.is_some_and(|candidate| candidate >= maximum),
                )
            }
            ShuffleMode::Down => {
                let candidate = lane5 + selector;
                (candidate, candidate <= maximum)
            }
            ShuffleMode::Xor => {
                let candidate = lane5 ^ selector;
                (candidate, candidate <= maximum)
            }
            ShuffleMode::Index => {
                let candidate = minimum | (selector & !segment_mask & 0x1f);
                (candidate, candidate <= maximum)
            }
        };
        let source_lane = if valid { candidate } else { lane };
        if participant_mask & (1_u32 << source_lane) == 0 || !active_mask.contains(source_lane) {
            return Err(OpError::message(
                "warp shuffle reads a non-participant lane",
            ));
        }
        source_lanes[lane] = source_lane;
        in_range[lane] = valid;
    }
    Ok((source_lanes, in_range))
}

/// Lanes whose `a` operand is actually read by one `shfl.sync`.
pub fn shuffle_source_mask(
    active_mask: WarpMask,
    participant_masks: &WarpValue<u32>,
    selectors: &WarpValue<u32>,
    controls: &WarpValue<u32>,
    mode: ShuffleMode,
) -> OpResult<WarpMask> {
    let (source_lanes, _in_range) =
        shuffle_sources(active_mask, participant_masks, selectors, controls, mode)?;
    let mut bits = 0_u32;
    for lane in active_mask.lanes() {
        bits |= 1_u32 << source_lanes[lane];
    }
    Ok(WarpMask(bits))
}

/// One `shfl.sync.mode.b32` returning `(d, p)`.
pub fn shfl_sync<T: Copy + Default>(
    active_mask: WarpMask,
    participant_masks: &WarpValue<u32>,
    values: &WarpValue<T>,
    selectors: &WarpValue<u32>,
    controls: &WarpValue<u32>,
    mode: ShuffleMode,
) -> OpResult<(WarpValue<T>, WarpValue<bool>)> {
    let (source_lanes, in_range) =
        shuffle_sources(active_mask, participant_masks, selectors, controls, mode)?;
    let mut result = [T::default(); WARP_SIZE];
    for lane in active_mask.lanes() {
        result[lane] = values[source_lanes[lane]];
    }
    Ok((result, in_range))
}

macro_rules! shfl_mode_fn {
    ($(#[$doc:meta])* $name:ident, $mode:expr) => {
        $(#[$doc])*
        pub fn $name<T: Copy + Default>(
            active_mask: WarpMask,
            participant_masks: &WarpValue<u32>,
            values: &WarpValue<T>,
            selectors: &WarpValue<u32>,
            controls: &WarpValue<u32>,
        ) -> OpResult<(WarpValue<T>, WarpValue<bool>)> {
            shfl_sync(active_mask, participant_masks, values, selectors, controls, $mode)
        }
    };
}

shfl_mode_fn!(
    /// `shfl.sync.idx.b32`.
    shfl_idx,
    ShuffleMode::Index
);
shfl_mode_fn!(
    /// `shfl.sync.up.b32`.
    shfl_up,
    ShuffleMode::Up
);
shfl_mode_fn!(
    /// `shfl.sync.down.b32`.
    shfl_down,
    ShuffleMode::Down
);
shfl_mode_fn!(
    /// `shfl.sync.bfly.b32`.
    shfl_bfly,
    ShuffleMode::Xor
);

#[cfg(test)]
mod tests {
    use super::super::mask::lanes_below;
    use super::*;

    fn iota() -> WarpValue<u32> {
        std::array::from_fn(|lane| lane as u32)
    }

    #[test]
    fn shuffle_and_shuffle_xor_select_lanes_inside_width_groups() {
        let participants = [u32::MAX; 32];
        let values = iota();
        let (shuffled, _) = shfl_idx(
            WarpMask::ALL,
            &participants,
            &values,
            &std::array::from_fn(|lane| 31 - lane as u32),
            &[31; 32],
        )
        .unwrap();
        let (xored, _) =
            shfl_bfly(WarpMask::ALL, &participants, &values, &[1; 32], &[31; 32]).unwrap();
        for lane in 0..WARP_SIZE {
            assert_eq!(shuffled[lane], (31 - lane) as u32);
            assert_eq!(xored[lane], (lane ^ 1) as u32);
        }
    }

    #[test]
    fn shuffle_xor_width_allows_later_groups_to_read_earlier_groups() {
        let (result, _) = shfl_bfly(
            WarpMask::ALL,
            &[u32::MAX; 32],
            &iota(),
            &[16; 32],
            &[0x100f; 32], // CUDA width 16 lowered to PTX segment/clamp.
        )
        .unwrap();
        for lane in 0..16 {
            assert_eq!(result[lane], lane as u32);
            assert_eq!(result[lane + 16], lane as u32);
        }
    }

    #[test]
    fn shuffle_rejects_nonparticipant_sources() {
        let active = lanes_below(16);
        let error =
            shfl_idx(active, &[active.bits(); 32], &iota(), &[31; 32], &[31; 32]).unwrap_err();
        assert!(error
            .to_string()
            .contains("warp shuffle reads a non-participant lane"));
    }

    // Ported from instructions/warp.rs
    // `raw_shuffle_consumes_packed_ptx_control_and_returns_optional_predicate`.
    #[test]
    fn raw_shuffle_consumes_packed_ptx_control_and_returns_optional_predicate() {
        let (values, predicates) =
            shfl_down(WarpMask::ALL, &[u32::MAX; 32], &iota(), &[1; 32], &[31; 32]).unwrap();
        assert_eq!(values[0], 1);
        assert!(predicates[0]);
        assert_eq!(values[31], 31);
        assert!(!predicates[31]);
    }

    #[test]
    fn shuffle_up_out_of_range_keeps_own_value_and_source_mask_tracks_reads() {
        let (values, predicates) =
            shfl_up(WarpMask::ALL, &[u32::MAX; 32], &iota(), &[2; 32], &[0; 32]).unwrap();
        assert_eq!((values[0], predicates[0]), (0, false));
        assert_eq!((values[1], predicates[1]), (1, false));
        assert_eq!((values[5], predicates[5]), (3, true));
        let mask = shuffle_source_mask(
            WarpMask::ALL,
            &[u32::MAX; 32],
            &[0; 32],
            &[31; 32],
            ShuffleMode::Index,
        )
        .unwrap();
        assert_eq!(mask.bits(), 1);
    }

    #[test]
    fn inactive_destination_lanes_get_default_and_false() {
        let active = lanes_below(8);
        let (values, predicates) =
            shfl_down(active, &[0xff; 32], &iota(), &[1; 32], &[7; 32]).unwrap();
        assert_eq!(values[6], 7);
        assert_eq!((values[7], predicates[7]), (7, false));
        assert_eq!((values[20], predicates[20]), (0, false));
    }
}
