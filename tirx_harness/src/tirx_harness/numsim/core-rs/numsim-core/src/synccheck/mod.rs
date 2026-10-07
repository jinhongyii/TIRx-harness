//! Synccheck (W6): offline exploration over the recorded `SyncEvent` log
//! (plan 2.6). Input is only a [`RecordingObserver`]; protocol semantics
//! come exclusively from `crate::sync::*::step` (never re-implemented).
//!
//! Per warp<->resource connected component: state = per-actor cursor +
//! resource states + pending completion multiset; DFS + state hashing +
//! sleep sets. Budget exhaustion -> `Status::Incomplete` with coverage.

use crate::observe::RecordingObserver;
use crate::report::Report;
use crate::sync::ResourceInit;

#[derive(Clone, Debug)]
pub struct SynccheckConfig {
    /// Max explored states per component before Incomplete.
    pub state_budget: u64,
    /// Static parameters for implicitly created resources.
    pub init: ResourceInit,
}

impl Default for SynccheckConfig {
    fn default() -> SynccheckConfig {
        SynccheckConfig { state_budget: 1 << 20, init: ResourceInit::default() }
    }
}

/// Explore all interleavings of the logged protocol and report deadlocks
/// and protocol violations reachable under some schedule.
pub fn check(log: &RecordingObserver, config: &SynccheckConfig) -> Report {
    let _ = (log, config);
    unimplemented!("W6: synccheck::check")
}
