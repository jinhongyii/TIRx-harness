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
    /// Largest `.exclusive` tcgen05.alloc in columns, from the launch's
    /// arch (`sched::exclusive_tmem_columns`: 576 on sm_107f, 512
    /// elsewhere). `None`: derived from the recording (see
    /// `Program::tcgen_exclusive_max`).
    pub tcgen_exclusive_max: Option<u32>,
    /// Happens-before gates on per-resource projections (bench switch;
    /// off over-approximates the interleavings and may fail closed).
    pub hb_gates: bool,
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
            tcgen_exclusive_max: None,
            hb_gates: true,
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
    // Launches run in order and the engine stops after a failing one, so
    // the first `launches_ended` launches ended and only the last started
    // launch can carry abnormal warp ends.
    let last = log.launches.last().map(|(k, _)| *k);
    for (i, (kernel, shape)) in log.launches.iter().enumerate() {
        let rec = out.entry(*kernel).or_default();
        if !rec.launches.iter().any(|(k, _)| k == kernel) {
            rec.launches.push((*kernel, *shape));
            rec.launches_ended += u32::from((i as u32) < log.launches_ended);
        }
        if Some(*kernel) == last && log.launches.len() > 1 {
            rec.warp_ends = log.warp_ends.clone();
        }
    }
    if log.launches.len() <= 1 {
        if let Some(rec) = out.values_mut().next() {
            rec.warp_ends = log.warp_ends.clone();
        }
    }
    out.into_iter().collect()
}

/// A recording of a launch that did not run to completion (review V2C-23):
/// `None` when the log is complete or carries no launch facts (hand-built
/// logs). Deadlocked ends are not truncation: they are the finding.
fn truncation(log: &RecordingObserver) -> Option<String> {
    use crate::observe::WarpEnd;
    if log.launches.is_empty() {
        return None;
    }
    if (log.launches_ended as usize) < log.launches.len() {
        return Some(format!("{} of {} launches reached end_launch", log.launches_ended, log.launches.len()));
    }
    if let Some((w, end)) = log.warp_ends.iter().find(|(_, e)| matches!(e, WarpEnd::Budget | WarpEnd::Error | WarpEnd::Trapped)) {
        return Some(format!("warp {} ended with {end:?}", w.0));
    }
    if !log.warp_ends.is_empty() {
        let ended = log.warp_ends.iter().map(|(w, _)| w.0).collect::<std::collections::BTreeSet<_>>();
        if let Some(w) = (0..log.per_warp.len() as u32).find(|w| !log.per_warp[*w as usize].is_empty() && !ended.contains(w)) {
            return Some(format!("warp {w} has events but no end"));
        }
    }
    None
}

/// Phase A protocol errors (`ProtocolStatus::Failed`) recorded in the log.
fn protocol_failures(log: &RecordingObserver) -> Vec<program::PhaseAFailure> {
    log.per_warp
        .iter()
        .flatten()
        .filter_map(|event| match &event.kind {
            crate::observe::SyncKind::Protocol { status: crate::observe::ProtocolStatus::Failed(error), .. } => {
                Some(program::PhaseAFailure { event: event.clone(), error: Some(error.clone()) })
            }
            _ => None,
        })
        .collect()
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
    let (mut program, failures) = match program::build(log) {
        Ok(x) => x,
        Err(detail) => {
            // A protocol error the engine hit stops the launch, so the log
            // can be structurally incomplete (e.g. collective records of
            // warps that never got there): the error is the finding.
            let errors = protocol_failures(log);
            if errors.is_empty() {
                out.program_build(detail);
            } else {
                out.phase_a(&errors);
            }
            return out.finish(started);
        }
    };
    // Protocol errors the engine hit are findings even if they stopped the
    // launch; otherwise a truncated recording is incomplete, never a
    // deadlock from missing events.
    let protocol_errors = failures.iter().filter(|f| f.error.is_some()).cloned().collect::<Vec<_>>();
    if !protocol_errors.is_empty() {
        out.phase_a(&protocol_errors);
        return out.finish(started);
    }
    if let Some(detail) = truncation(log) {
        out.truncated(detail);
        return out.finish(started);
    }
    if !failures.is_empty() {
        out.phase_a(&failures);
        return out.finish(started);
    }
    out.kernel = program.kernel;
    if let Some(max) = config.tcgen_exclusive_max {
        program.tcgen_exclusive_max = max;
    }
    if program.commands.is_empty() {
        return out.finish(started);
    }
    if let Some((cmd, error)) = program.cta_group_error() {
        out.static_error(&program, cmd, error);
        return out.finish(started);
    }
    // Launch facts: an explicit `config.init` wins; otherwise the shape the
    // recording captured in `begin_launch` for this kernel.
    let init = if config.init != ResourceInit::default() {
        config.init
    } else {
        log.launches
            .iter()
            .find(|(k, _)| *k == program.kernel)
            .map_or(config.init, |(_, shape)| ResourceInit { policy: config.init.policy, ..resource_init(shape) })
    };
    // Fail closed: never explore a setmaxnreg pool with 0 warps or a cluster
    // barrier with 0 participants because the launch shape is unknown.
    let needs = |pred: fn(&crate::sync::ResourceId) -> bool| program.resources.iter().any(pred);
    if (init.warps_per_cta == 0 && needs(|r| matches!(r, crate::sync::ResourceId::RegPool { .. })))
        || (init.cluster_warps == 0 && needs(|r| matches!(r, crate::sync::ResourceId::Cluster { .. })))
    {
        out.program_build(
            "launch shape unknown: neither the recording (`RecordingObserver::launches`) nor `SynccheckConfig.init` gives warps per CTA / cluster participants".into(),
        );
        return out.finish(started);
    }
    let deadline = (config.limits.max_wall_time_ms != u64::MAX)
        .then(|| started + std::time::Duration::from_millis(config.limits.max_wall_time_ms));
    let reference = match reference::run(&program, &init, deadline) {
        Ok(r) => r,
        Err(reference::RunError::Build(detail)) => {
            out.program_build(detail);
            return out.finish(started);
        }
        Err(reference::RunError::WallTime) => {
            out.wall_time_limit(started);
            return out.finish(started);
        }
    };
    if !reference.is_complete() {
        out.reference_failure(&program, &init, &reference);
        // Without gates (whole / components) the search does not need the
        // reference clocks; in all-failures mode keep exploring.
        if config.explore.stop_on_first_failure || config.mode == ProjectionMode::PerResource || config.certificates {
            return out.finish(started);
        }
    }
    let limits = explore::Limits {
        max_states: usize::try_from(config.state_budget).unwrap_or(usize::MAX),
        max_transitions: usize::try_from(config.transition_budget).unwrap_or(usize::MAX),
        deadline,
    };
    let mut clean_fingerprints = HashSet::<String>::new();
    for spec in projection::project(&program, config.mode) {
        if started.elapsed().as_millis() > u128::from(config.limits.max_wall_time_ms) {
            out.wall_time_limit(started);
            break;
        }
        let spec = if config.hb_gates { spec } else { projection::ProjectionSpec { gated: false, ..spec } };
        let mut ts = match ts::Ts::new(&program, &spec, &init, Some(&reference)) {
            Ok(ts) => ts,
            Err(detail) => {
                out.program_build(detail);
                break;
            }
        };
        ts.rules = config.explore.rules;
        out.stats.programs += 1;
        if config.certificates {
            if let Some(result) = certificate::certify(&ts, &reference, init.cluster_warps) {
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
