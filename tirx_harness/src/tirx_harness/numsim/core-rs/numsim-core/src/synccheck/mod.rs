//! Synccheck (W6): offline exploration over the recorded `SyncEvent` log
//! (plan 2.6, spec `docs/development/synccheck-explorer.md`).
//!
//! Input is only a [`RecordingObserver`]. Protocol semantics come only from
//! the protocol `step` functions ([`backend`]; never re-implemented here).
//!
//! 1. **Phase A** (the concrete run): `Protocol` events whose `status` is
//!    `Failed` or `BlockedAtExit` are reported as-is; Phase B runs only when
//!    the run was clean (today's rule, `sync_check_python.rs:151-166`).
//! 2. **Program**: per-warp committed commands; collectives joined; retries
//!    the explorer re-derives (named `Resume`, setmaxnreg `Poll`, failed polls)
//!    dropped ([`program`]).
//! 3. **Reference run**: one complete schedule giving per-command vector
//!    clocks and generations ([`reference`]).
//! 4. **Projection** per resource group with happens-before gates
//!    ([`projection`]), then per projection: causal certificate
//!    ([`certificate`]), fingerprint reuse ([`fingerprint`]), or
//!    explicit-state DFS with state hashing, sleep sets, strong diamonds and a
//!    persistent-transition rule ([`explore`]).
//! 5. Budget exhaustion and unmodeled situations are `Status::Incomplete`.
//!
//! [`serialize`] renders a [`Report`] as today's native payload so the Python
//! layer keeps its pinned keys.

pub mod backend;
pub mod build;
mod certificate;
mod clock;
pub mod explore;
mod fingerprint;
mod kinds;
mod payload;
pub mod program;
pub mod projection;
mod reference;
mod ts;

use std::collections::HashSet;
use std::time::Instant;

use crate::observe::RecordingObserver;
use crate::report::Report;
use crate::sync::ResourceInit;

pub use payload::serialize;
pub use projection::ProjectionMode;

/// Limits echoed into the payload's `coverage.resource_limits`. Only the
/// state and transition budgets (and wall time, checked between projections)
/// bound the offline search; the rest are carried for the report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EchoLimits {
    pub max_schedules: u64,
    pub max_events_per_run: u64,
    pub max_total_events: u64,
    pub max_wall_time_ms: u64,
    pub max_diagnostic_bytes: u64,
}

impl Default for EchoLimits {
    fn default() -> Self {
        Self {
            max_schedules: u64::MAX,
            max_events_per_run: u64::MAX,
            max_total_events: u64::MAX,
            max_wall_time_ms: u64::MAX,
            max_diagnostic_bytes: u64::MAX,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SynccheckConfig {
    /// Max distinct states per projection (`ResourceLimits.max_backtrack_nodes`).
    pub state_budget: u64,
    /// Max explored transitions per projection (`ResourceLimits.max_loop_steps`).
    pub transition_budget: u64,
    /// Static parameters for implicitly created resources.
    pub init: ResourceInit,
    pub mode: ProjectionMode,
    pub certificates: bool,
    pub fingerprints: bool,
    pub explore: explore::Options,
    pub limits: EchoLimits,
}

impl Default for SynccheckConfig {
    fn default() -> SynccheckConfig {
        SynccheckConfig {
            state_budget: 1_000_000,
            transition_budget: 10_000_000,
            init: ResourceInit::default(),
            mode: ProjectionMode::PerResource,
            certificates: true,
            fingerprints: true,
            explore: explore::Options::ALL,
            limits: EchoLimits::default(),
        }
    }
}

/// Resource parameters of a launch (what the scheduler gives its own
/// `SyncTable`): warps per CTA and cluster-barrier participants.
pub fn resource_init(shape: &crate::program::LaunchShape) -> ResourceInit {
    let ctas_per_cluster = shape.cluster.iter().product::<u32>().max(1);
    ResourceInit {
        warps_per_cta: shape.warps_per_cta(),
        cluster_warps: shape.warps_per_cta() * ctas_per_cluster,
        ..ResourceInit::default()
    }
}

/// Split a log by `SyncEvent::kernel` (one launch per kernel index).
pub fn split_launches(log: &RecordingObserver) -> Vec<(u32, RecordingObserver)> {
    let mut out = std::collections::BTreeMap::<u32, RecordingObserver>::new();
    for &(warp, index) in &log.order {
        let event = if warp == u32::MAX { &log.other[index as usize] } else { &log.per_warp[warp as usize][index as usize] };
        crate::observe::Observer::sync(out.entry(event.kernel).or_default(), event);
    }
    out.into_iter().collect()
}

/// One [`Report`] per launch in the log (never merged).
pub fn check_launches(log: &RecordingObserver, config: &SynccheckConfig) -> Vec<Report> {
    split_launches(log).into_iter().map(|(_, l)| check(&l, config)).collect()
}

/// Explore all interleavings of one launch's logged protocol and report
/// deadlocks, protocol violations and non-confluence reachable under some
/// schedule. A log that mixes launches is incomplete (use [`check_launches`]).
pub fn check(log: &RecordingObserver, config: &SynccheckConfig) -> Report {
    let started = Instant::now();
    let mut out = payload::Builder::new(config);
    let kernels = log
        .per_warp
        .iter()
        .flatten()
        .chain(&log.other)
        .map(|e| e.kernel)
        .collect::<std::collections::BTreeSet<_>>();
    if kernels.len() > 1 {
        out.kernel = *kernels.first().expect("non-empty");
        out.program_build(format!("the log mixes launches {kernels:?}; check each launch separately (check_launches)"));
        return out.finish(started);
    }
    let (program, failures) = match program::build(log) {
        Ok(x) => x,
        Err(detail) => {
            out.program_build(detail);
            return out.finish(started);
        }
    };
    if !failures.is_empty() {
        out.phase_a(&failures);
        return out.finish(started);
    }
    out.kernel = program.kernel;
    if program.commands.is_empty() {
        return out.finish(started);
    }
    if let Some((cmd, error)) = program.cta_group_error() {
        out.static_error(&program, cmd, error);
        return out.finish(started);
    }
    let reference = match reference::run(&program, &config.init) {
        Ok(r) => r,
        Err(detail) => {
            out.program_build(detail);
            return out.finish(started);
        }
    };
    if !reference.is_complete() {
        out.reference_failure(&program, &config.init, &reference);
        // Without gates (whole / components) the search does not need the
        // reference clocks; in all-failures mode keep exploring.
        if config.explore.stop_on_first_failure || config.mode == ProjectionMode::PerResource || config.certificates {
            return out.finish(started);
        }
    }
    let limits = explore::Limits {
        max_states: usize::try_from(config.state_budget).unwrap_or(usize::MAX),
        max_transitions: usize::try_from(config.transition_budget).unwrap_or(usize::MAX),
    };
    let mut clean_fingerprints = HashSet::<String>::new();
    for spec in projection::project(&program, config.mode) {
        if started.elapsed().as_millis() > u128::from(config.limits.max_wall_time_ms) {
            out.wall_time_limit(started);
            break;
        }
        let ts = match ts::Ts::new(&program, &spec, &config.init, Some(&reference)) {
            Ok(ts) => ts,
            Err(detail) => {
                out.program_build(detail);
                break;
            }
        };
        out.stats.programs += 1;
        if config.certificates {
            if let Some(result) = certificate::certify(&ts, &reference, config.init.cluster_warps) {
                out.stats.certified += 1;
                out.stats.visited += 1;
                out.stats.transitions += ts.cmds.len() as u64;
                match result {
                    Ok(()) => continue,
                    Err(e) => {
                        out.certificate_failure(&program, &ts, e);
                        break;
                    }
                }
            }
        }
        let key = config.fingerprints.then(|| fingerprint::fingerprint(&ts));
        if key.as_ref().is_some_and(|k| clean_fingerprints.contains(k)) {
            out.stats.reused += 1;
            continue;
        }
        let search = explore::explore(&ts, limits, config.explore);
        out.stats.visited += search.visited_states as u64;
        out.stats.transitions += search.explored_transitions as u64;
        out.stats.diamond_pruned += search.strong_diamond_pruned as u64;
        out.stats.sleep_pruned += search.sleep_pruned as u64;
        if out.search_result(&program, &ts, &search) {
            break;
        }
        if let Some(k) = key {
            clean_fingerprints.insert(k);
        }
    }
    // Terminal states are confluent when clean, so the reference run's exit
    // lints are the lints of every schedule.
    out.lints(&program, &reference);
    out.finish(started)
}
