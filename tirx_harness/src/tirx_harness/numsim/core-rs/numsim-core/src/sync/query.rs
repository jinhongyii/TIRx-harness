//! Pinned semantics of the mbarrier side queries behind `Instr::MbarQuery`:
//! `mbarrier.pending_count` and `mbarrier.check_layout`. Also defines the
//! opaque state token that `mbarrier.arrive` returns.
//!
//! These are production-only helpers, not protocol state, so they live
//! outside the ported `mbarrier.rs` and survive `tools/port_sync.py`.
//!
//! # State token
//!
//! `mbarrier.arrive` returns an opaque 64-bit state. The model encodes the
//! generation the arrival applied to (`Outcome::Arrived::gen`). The
//! `.noComplete` forms also carry the pending count, so that
//! `mbarrier.pending_count` can read it back. The layout is private; it
//! keeps the legacy encoding of `hardware_barriers.rs:530-549`:
//!
//! | bits | field |
//! | --- | --- |
//! | 0..43 | generation |
//! | 43 | set only by `.noComplete` arrivals |
//! | 44..64 | pending count before this lane's arrival (20 bits) |
//!
//! The token encodes the generation, not just the phase parity, so
//! `test_wait` / `try_wait` with a state token can reject a token that names
//! neither the current nor the immediately preceding phase
//! (`InvalidStateToken`, PTX 9.4 §9.7.15.16.19).
//!
//! # `mbarrier.pending_count`
//!
//! Reads the pending arrival count captured in a `.noComplete` state token.
//! For a warp whose lanes arrive on one barrier, the handler assigns each lane
//! the pending count before that lane's own arrival. Lanes are taken in
//! ascending order, so lane k sees `pending_before - Σ count(lanes < k)`
//! (`hardware_barriers.rs:500-527`). A token from any other arrive form is
//! rejected: PTX makes `pending_count` meaningful only for
//! `arrive.noComplete` states.
//!
//! # `mbarrier.check_layout`
//!
//! Predicate: does the initialized barrier use the requested layout
//! (`runtime/instructions/sync.rs:521-547`, `hardware_barriers.rs:793-805`)?
//! Querying an uninitialized barrier is `Error::Uninitialized`. The
//! contract's `MbarQueryOp::CheckLayout` does not yet carry the requested
//! layout; see `CONTRACT_REQUESTS.md`.

use super::mbarrier;

const NO_COMPLETE: u64 = 1 << 43;
const PENDING_SHIFT: u32 = 44;

/// Why a state token cannot be decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TokenError {
    /// The generation does not fit the 43-bit field.
    GenerationOverflow { gen: u64 },
    /// The pending count does not fit the 20-bit field.
    PendingOverflow { pending: u64 },
    /// `pending_count` on a token not produced by `.noComplete`.
    NotNoComplete,
}

/// Encode the state returned to one lane.
pub fn encode(gen: u64, pending_before: u64, no_complete: bool) -> Result<u64, TokenError> {
    if gen >= NO_COMPLETE {
        return Err(TokenError::GenerationOverflow { gen });
    }
    if !no_complete {
        return Ok(gen);
    }
    if pending_before > mbarrier::MAX_COUNT {
        return Err(TokenError::PendingOverflow { pending: pending_before });
    }
    Ok(gen | NO_COMPLETE | (pending_before << PENDING_SHIFT))
}

/// Generation named by a state token (input of `Cmd::TestState`).
pub fn generation(token: u64) -> u64 {
    token & (NO_COMPLETE - 1)
}

/// `mbarrier.pending_count`.
pub fn pending_count(token: u64) -> Result<u32, TokenError> {
    if token & NO_COMPLETE == 0 {
        return Err(TokenError::NotNoComplete);
    }
    Ok((token >> PENDING_SHIFT) as u32)
}

/// `mbarrier.check_layout`: `true` iff the barrier was initialized with the
/// requested layout.
pub fn check_layout(s: &mbarrier::State, layout_v1: bool) -> Result<bool, mbarrier::Error> {
    if !s.live {
        return Err(mbarrier::Error::Uninitialized);
    }
    Ok(s.layout_v1 == layout_v1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::Policy;

    #[test]
    fn token_round_trip() {
        let t = encode(5, 3, true).unwrap();
        assert_eq!(generation(t), 5);
        assert_eq!(pending_count(t), Ok(3));
        let plain = encode(5, 3, false).unwrap();
        assert_eq!(generation(plain), 5);
        assert_eq!(pending_count(plain), Err(TokenError::NotNoComplete));
        assert!(encode(NO_COMPLETE, 0, false).is_err());
        assert!(encode(0, mbarrier::MAX_COUNT + 1, true).is_err());
    }

    #[test]
    fn check_layout_requires_initialized_barrier() {
        let mut s = mbarrier::State::new(Policy::Numeric);
        assert_eq!(check_layout(&s, false), Err(mbarrier::Error::Uninitialized));
        mbarrier::step(&mut s, mbarrier::Cmd::Init { count: 1, layout_v1: true }).unwrap();
        assert_eq!(check_layout(&s, true), Ok(true));
        assert_eq!(check_layout(&s, false), Ok(false));
    }
}
