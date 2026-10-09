//! Reference model of the mbarrier side queries: the opaque state token that
//! `mbarrier.arrive` returns, `mbarrier.pending_count` and
//! `mbarrier.check_layout`.
//!
//! Spec (`numsim-core/src/sync/query.rs` module docs, legacy
//! `hardware_barriers.rs:500-549`): the token packs three fields
//!
//! | field | width | meaning |
//! | --- | --- | --- |
//! | generation | 43 bits (0..43) | phase the arrival applied to |
//! | no_complete | 1 bit (43) | set only by `.noComplete` arrivals |
//! | pending | 20 bits (44..64) | pending count before this lane's arrival |
//!
//! A plain arrive's token carries only the generation. `pending_count` is
//! defined only for `.noComplete` tokens (PTX 9.7.15.16.20: "must be the
//! result of a prior mbarrier.arrive.noComplete or
//! mbarrier.arrive_drop.noComplete instruction"). Written independently of
//! the production code (field arithmetic, not shared masks).

use crate::mbarrier;

const GEN_BITS: u32 = 43;
const PENDING_BITS: u32 = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TokenError {
    GenerationOverflow { gen: u64 },
    PendingOverflow { pending: u64 },
    NotNoComplete,
}

/// Decoded token fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fields {
    gen: u64,
    no_complete: bool,
    pending: u64,
}

fn pack(f: Fields) -> u64 {
    let flag = if f.no_complete { 1u64 } else { 0 };
    f.gen + flag * 2u64.pow(GEN_BITS) + f.pending * 2u64.pow(GEN_BITS + 1)
}

fn unpack(token: u64) -> Fields {
    let low = 2u64.pow(GEN_BITS);
    Fields {
        gen: token % low,
        no_complete: (token / low) % 2 == 1,
        pending: token / (low * 2),
    }
}

/// Encode the state returned to one lane.
pub fn encode(gen: u64, pending_before: u64, no_complete: bool) -> Result<u64, TokenError> {
    if gen >= 2u64.pow(GEN_BITS) {
        return Err(TokenError::GenerationOverflow { gen });
    }
    if !no_complete {
        return Ok(pack(Fields { gen, no_complete: false, pending: 0 }));
    }
    if pending_before > mbarrier::MAX_COUNT || pending_before >= 2u64.pow(PENDING_BITS) {
        return Err(TokenError::PendingOverflow { pending: pending_before });
    }
    Ok(pack(Fields { gen, no_complete: true, pending: pending_before }))
}

/// Generation named by a state token.
pub fn generation(token: u64) -> u64 {
    unpack(token).gen
}

/// `mbarrier.pending_count`.
pub fn pending_count(token: u64) -> Result<u32, TokenError> {
    let f = unpack(token);
    if !f.no_complete {
        return Err(TokenError::NotNoComplete);
    }
    Ok(f.pending as u32)
}

/// `mbarrier.check_layout`.
pub fn check_layout(s: &mbarrier::State, layout_v1: bool) -> Result<bool, mbarrier::Error> {
    if !s.live {
        return Err(mbarrier::Error::Uninitialized);
    }
    Ok(s.layout_v1 == layout_v1)
}
