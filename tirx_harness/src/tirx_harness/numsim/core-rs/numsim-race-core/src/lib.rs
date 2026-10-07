//! Standalone prototype of the unified racecheck core (redesign plan §2.5).
//!
//! * [`clock`] — packed stamps, chunked `Arc`-shared epochs, join memo,
//!   sparse lane entries.
//! * [`knowledge`] — a clock per proxy view (hb, proxy bridges, tcgen).
//! * [`shadow`] — `IntervalShadow<C>` with the exact-hit fast path.
//! * [`cell`] — write/read frontiers and the contract-aware eviction rule.
//! * [`checker`] — the event loop: conflict rule and every HB edge.
//! * [`input`] — the ONLY definition of the input shape (to be replaced by
//!   the `numsim-core` contract's `Access` / `SyncEvent`).
//!
//! Semantics are specified in `docs/development/racecheck-semantics.md`.

pub mod cell;
pub mod checker;
pub mod clock;
pub mod input;
pub mod knowledge;
pub mod shadow;

pub use checker::{AdvisoryKind, Checker, Finding, FindingKind, Incomplete, OrderingFailure, RaceClass, Report, Severity};
