//! Racecheck (W5): online FastTrack-style checker driven by
//! [`crate::observe::Observer`] callbacks (plan 2.5).
//!
//! Semantics: `docs/development/racecheck-semantics.md`; behaviour changes
//! vs legacy: `docs/development/racecheck-behaviour-deltas.md`.
//!
//! * [`clock`] — packed stamps, `Arc`-shared epoch chunks, join memo,
//!   sparse lane entries.
//! * [`knowledge`] — one clock per proxy view (hb, bridges, tcgen, tensormap).
//! * [`shadow`] — `IntervalShadow<C>` with the exact-hit fast path.
//! * [`cell`] — packed 16-byte witnesses, write/read frontiers.
//! * [`checker`] — the conflict rule and every HB edge.
//! * [`input`] — alias layer over the contract + per-lane event shapes.
//! * [`observer`] — `RaceObserver: Observer`, the contract adapter.
//! * [`payload`] — `report()` / `serialize()`.

pub mod cell;
pub mod checker;
pub mod clock;
pub mod input;
pub mod knowledge;
pub mod observer;
pub mod payload;
pub mod shadow;
pub mod tuning;

pub use checker::{AdvisoryKind, Checker, Finding as RaceFinding, FindingKind as RaceFindingKind, Incomplete, OrderingFailure, RaceClass, Report as RaceReport, Severity};
pub use observer::{RaceObserver, RacecheckConfig};
pub use payload::{report, reports, serialize};

/// Compatibility name used by `numsim-py`.
pub type Racecheck = RaceObserver;
