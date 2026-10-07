//! Racecheck (W5): online FastTrack-style checker driven by
//! [`crate::observe::Observer`] callbacks (plan 2.5).
//!
//! Owned data: one `Clock` (actors = warps + async virtual actors +
//! cross-CTA actors; proxy is a clock dimension) and one interval shadow per
//! `(AllocId, range)`. Declared-word waits use `WaitVerdicts`. Unmerged
//! global shadow at `end_launch` is a merge point, never a silent pass.
//! Fail closed: budget exhaustion or unsupported semantics produce
//! `Status::Incomplete` findings.

use crate::observe::Observer;
use crate::report::{Finding, Report};

/// Racecheck options.
#[derive(Clone, Debug, Default)]
pub struct RacecheckConfig {
    /// Stop recording new findings after this many (0 = unlimited); the
    /// run continues regardless (races never abort execution).
    pub max_findings: usize,
}

/// The checker. Feed it to `sched::run` as the observer.
#[derive(Debug, Default)]
pub struct Racecheck {
    pub config: RacecheckConfig,
    pub findings: Vec<Finding>,
}

impl Racecheck {
    pub fn new(config: RacecheckConfig) -> Racecheck {
        Racecheck { config, findings: Vec::new() }
    }
    /// Final findings (after the last `end_launch`).
    pub fn finish(self) -> Report {
        Report::new("racecheck", self.findings)
    }
}

impl Observer for Racecheck {
    fn wants_word_history(&self) -> bool {
        true
    }
    // W5: access / sync (all SyncKind variants) / end_launch / inbox_drain.
}
