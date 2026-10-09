//! Membermask / participation validation and active-mask queries.
//!
//! Legacy sources: `engine-rs/src/runtime/warp_ops.rs`
//! (`validate_warp_collective_participants`, `require_uniform_i64`),
//! `engine-rs/src/runtime/sync.rs` (`require_full_warp_sync`),
//! `engine-rs/src/runtime/instructions/warp.rs` (`activemask`), and the
//! diagnostic formatting of `engine-rs/src/completion.rs` (`DiagnosticLabel`)
//! plus `executor.rs` (`WarpCollectiveDivergence` display).
//!
//! Error text is reproduced exactly as the legacy `EngineError` rendered it.
//! The legacy divergence errors were a structured `WarpCollectiveDivergence`
//! kind; here they flatten to the rendered message.

use crate::types::{OpError, OpResult, WarpMask, WarpValue, WARP_SIZE};

/// Rendered legacy `WarpCollectiveDivergence { operation, active_mask }`.
pub fn warp_collective_divergence(operation: &str, active_mask: WarpMask) -> OpError {
    OpError::message(format!(
        "{operation} requires all {WARP_SIZE} lanes, got mask 0x{:08x}",
        active_mask.bits()
    ))
}

/// Rendered legacy `DiagnosticLabel::warp_collective_divergence_with_detail`:
/// `"{operation}{detail}: {operation} requires all 32 lanes, got mask 0x..."`.
fn warp_collective_divergence_with_detail(
    operation: &str,
    active_mask: WarpMask,
    detail: std::fmt::Arguments<'_>,
) -> OpError {
    OpError::message(format!(
        "{operation}{detail}: {}",
        warp_collective_divergence(operation, active_mask)
    ))
}

/// Require every lane of the warp to be active (`require_full_warp_sync`).
pub fn require_full_warp(active_mask: WarpMask, operation: &str) -> OpResult<()> {
    if active_mask != WarpMask::ALL {
        return Err(warp_collective_divergence(operation, active_mask));
    }
    Ok(())
}

/// Validate the per-lane `membermask` operands of one `*.sync` collective and
/// return the agreed participant mask.
///
/// Rules (in legacy check order):
/// 1. at least one active lane (`"{op} has no active lane"`);
/// 2. the first active lane's mask is the expected one and must be nonzero
///    (`"{op} participant mask must not be zero"`);
/// 3. the mask may not name an inactive lane;
/// 4. every active lane's mask must equal the expected one;
/// 5. every active (executing) lane must be present in its own mask.
///
/// Rules 3-5 are divergence errors (see [`warp_collective_divergence`]).
/// Note the expected mask may be a strict subset of the active mask; active
/// lanes outside it then fail rule 4/5.
pub fn validate_participants(
    active_mask: WarpMask,
    participant_masks: &WarpValue<u32>,
    operation: &str,
) -> OpResult<u32> {
    let first_lane = active_mask
        .first()
        .ok_or_else(|| OpError::message(format!("{operation} has no active lane")))?;
    let expected = participant_masks[first_lane];
    if expected == 0 {
        return Err(OpError::message(format!(
            "{operation} participant mask must not be zero"
        )));
    }
    if expected & !active_mask.bits() != 0 {
        return Err(warp_collective_divergence_with_detail(
            operation,
            active_mask,
            format_args!(" participant mask names an inactive lane"),
        ));
    }
    for lane in active_mask.lanes() {
        let actual = participant_masks[lane];
        if actual != expected {
            return Err(warp_collective_divergence_with_detail(
                operation,
                active_mask,
                format_args!(
                    " participant masks disagree: lane {lane} has 0x{actual:08x}, expected 0x{expected:08x}"
                ),
            ));
        }
        if actual & (1_u32 << lane) == 0 {
            return Err(warp_collective_divergence_with_detail(
                operation,
                active_mask,
                format_args!(" executing lane {lane} is absent from its participant mask"),
            ));
        }
    }
    Ok(expected)
}

/// Require an `i64` operand to be uniform across the active lanes and return it.
pub fn require_uniform_i64(values: &WarpValue<i64>, mask: WarpMask, label: &str) -> OpResult<i64> {
    let first_lane = mask
        .first()
        .ok_or_else(|| OpError::message(format!("{label} has no active lane")))?;
    let expected = values[first_lane];
    for lane in mask.lanes() {
        if values[lane] != expected {
            return Err(OpError::message(format!(
                "{label} must agree across active lanes: lane {lane} has {}, expected {expected}",
                values[lane]
            )));
        }
    }
    Ok(expected)
}

/// `activemask.b32`: every lane (active or not) receives the active mask bits.
pub fn activemask(active_mask: WarpMask) -> WarpValue<u32> {
    [active_mask.bits(); WARP_SIZE]
}

#[cfg(test)]
/// Mask containing lanes `0..count` (`WarpMask::from_lanes(0..count)` in legacy).
pub(crate) fn lanes_below(count: usize) -> WarpMask {
    WarpMask::first_n(count as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masked_collectives_reject_invalid_participants() {
        let label = "collective";
        let active = lanes_below(16);
        assert!(validate_participants(active, &[u32::MAX; 32], label).is_err());
        assert!(validate_participants(WarpMask::ALL, &[0; 32], label).is_err());
        let missing_self = [!(1_u32 << 7); 32];
        let error = validate_participants(WarpMask::ALL, &missing_self, label).unwrap_err();
        assert!(error.to_string().contains("executing lane 7 is absent"));
        let inconsistent: WarpValue<u32> =
            std::array::from_fn(|lane| if lane == 7 { !(1_u32 << 0) } else { u32::MAX });
        let error = validate_participants(WarpMask::ALL, &inconsistent, label).unwrap_err();
        assert!(error.to_string().contains("participant masks disagree"));
    }

    #[test]
    fn participant_diagnostics_match_legacy_rendering() {
        let error = validate_participants(WarpMask::NONE, &[1; 32], "vote.sync").unwrap_err();
        assert_eq!(error.to_string(), "vote.sync has no active lane");
        let error = validate_participants(WarpMask::ALL, &[0; 32], "vote.sync").unwrap_err();
        assert_eq!(
            error.to_string(),
            "vote.sync participant mask must not be zero"
        );
        let active = lanes_below(16);
        let error = validate_participants(active, &[u32::MAX; 32], "shfl.sync").unwrap_err();
        assert_eq!(
            error.to_string(),
            "shfl.sync participant mask names an inactive lane: \
             shfl.sync requires all 32 lanes, got mask 0x0000ffff"
        );
        // A subset mask is legal when exactly the active lanes agree on it.
        assert_eq!(
            validate_participants(active, &[0xffff; 32], "x").unwrap(),
            0xffff
        );
    }

    #[test]
    fn full_warp_requirement_renders_divergence() {
        assert!(require_full_warp(WarpMask::ALL, "op").is_ok());
        let error = require_full_warp(lanes_below(16), "cuda_warp_reduce").unwrap_err();
        assert_eq!(
            error.to_string(),
            "cuda_warp_reduce requires all 32 lanes, got mask 0x0000ffff"
        );
    }

    #[test]
    fn uniform_values_preserve_diagnostics() {
        let uniform = [7_i64; 32];
        assert_eq!(
            require_uniform_i64(&uniform, WarpMask::ALL, "field").unwrap(),
            7
        );
        let error = require_uniform_i64(&uniform, WarpMask::NONE, "field").unwrap_err();
        assert_eq!(error.to_string(), "field has no active lane");
        let disagreeing: WarpValue<i64> = std::array::from_fn(|lane| if lane == 3 { 9 } else { 7 });
        let error = require_uniform_i64(&disagreeing, WarpMask::ALL, "field").unwrap_err();
        assert_eq!(
            error.to_string(),
            "field must agree across active lanes: lane 3 has 9, expected 7"
        );
    }

    #[test]
    fn activemask_splats_active_bits() {
        assert_eq!(activemask(WarpMask(0x1088)), [0x1088; 32]);
    }
}
