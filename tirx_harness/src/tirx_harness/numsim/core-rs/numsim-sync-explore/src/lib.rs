//! Prototype of the offline Synccheck explorer (Phase B of today's synccheck).
//!
//! Input is only a `Vec<SyncEvent>` (see [`event`]). The pipeline is:
//!
//! 1. [`program::Program::from_events`]: group events into per-warp command
//!    sequences (each warp's executed path is fixed).
//! 2. [`program::reference_run`]: one complete schedule of the whole program,
//!    recording per-command vector clocks and generations. This plays the role
//!    of the observed NumSim run (today's `ResolvedTransitionLog` clocks).
//! 3. [`projection`]: split into independently explorable transition systems
//!    (`Whole`, warp/resource `Components`, or `PerResource` with
//!    happens-before gating derived from the reference clocks).
//! 4. [`certificate`]: O(n) counting + vector-clock proofs per resource.
//! 5. [`fingerprint`]: deduplicate isomorphic projections.
//! 6. [`explore`]: explicit-state DFS with state hashing, sleep sets, strong
//!    diamonds and a persistent-transition hook.
//!
//! The protocol state machines in [`protocol`] are deliberately tiny local
//! stand-ins. They will be swapped for the shared `step` functions
//! (`numsim-core` SyncTable / `numsim-sync-ref`) once those land.

pub mod certificate;
pub mod check;
pub mod clock;
pub mod event;
pub mod explore;
pub mod fingerprint;
pub mod program;
pub mod projection;
pub mod protocol;
pub mod synth;
pub mod ts;

pub use check::{check, CheckConfig, CheckReport, Finding, IncompleteReason, Verdict};
pub use event::{OpId, ResourceId, SyncEvent, SyncOp};
