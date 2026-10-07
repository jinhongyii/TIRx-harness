//! Executable reference state machines for NumSim synchronization protocols.
//!
//! This crate is a test oracle, not production code. Each module is a small
//! model of one protocol, written so it is obviously correct: it has no
//! footprints, no waiter registry and no wakers. Its only interface is
//!
//! ```text
//! fn step(&mut State, Cmd) -> Result<Outcome, Error>
//! ```
//!
//! A blocking command returns `Outcome::Blocked`. The scheduler retries the
//! warp on a later round. No call registers a waiter.
//!
//! Contract shared by every module (property-tested in `tests/`):
//!
//! * **Transactional.** `Err` leaves the state bit-for-bit unchanged.
//! * **Deterministic.** The same state and command always produce the same
//!   result.
//! * **Total.** `step` never panics, whatever the command.
//!
//! The specification text, with file:line evidence from the legacy engine and
//! strict models, lives in `docs/development/sync-semantics.md`. Production
//! `SyncTable` step functions are differentially tested against these models.
//! The `Cmd`, `Outcome` and `Error` shapes here are the ones proposed for the
//! `numsim-core` contract.

pub mod async_group;
pub mod cluster;
pub mod mbarrier;
pub mod named;
pub mod setmaxnreg;
pub mod tcgen;

/// Which reading of the ISA a protocol state enforces.
///
/// `Numeric` is what NumSim executes. It rejects only behaviour that the
/// legacy engine rejects in its numeric path.
///
/// `Strict` adds the protocol rules that the legacy strict synccheck models
/// enforced on top of the numeric ones. One example is mbarrier phase
/// consumption before reuse.
///
/// Invariant (tested): whenever `Strict` accepts a command sequence, `Numeric`
/// accepts it with identical outcomes. That makes strict a refinement of
/// numeric.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Policy {
    #[default]
    Numeric,
    Strict,
}

/// Uniform view used by the generic property harness.
pub trait Protocol {
    type State: Clone + core::fmt::Debug + PartialEq;
    type Cmd: Clone + core::fmt::Debug;
    type Outcome: Clone + core::fmt::Debug + PartialEq;
    type Error: Clone + core::fmt::Debug + PartialEq;

    fn step(state: &mut Self::State, cmd: Self::Cmd) -> Result<Self::Outcome, Self::Error>;
}

/// A warp index local to the resource's scope (CTA or cluster).
pub type Warp = u32;

/// Thread mask of one warp.
pub type LaneMask = u32;

pub const FULL_MASK: LaneMask = u32::MAX;
