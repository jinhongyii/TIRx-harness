//! Driver: events -> program -> reference run -> projections -> certificates
//! -> fingerprint dedup -> DFS, with fail-closed incomplete reasons.

use std::collections::HashMap;

use crate::certificate::certify;
use crate::event::{OpId, SyncEvent};
use crate::explore::{explore, Failure, Limits, Options, Termination};
use crate::fingerprint::fingerprint;
use crate::program::{reference_run, Program, RunOutcome};
use crate::projection::{project, ProjectionMode};
use crate::ts::{ProtocolTs, State, Transition};

#[derive(Clone, Copy, Debug)]
pub struct CheckConfig {
    pub mode: ProjectionMode,
    pub certificates: bool,
    pub fingerprints: bool,
    pub explore: Options,
    pub limits: Limits,
}

impl Default for CheckConfig {
    fn default() -> Self {
        Self {
            mode: ProjectionMode::PerResource,
            certificates: true,
            fingerprints: true,
            explore: Options::ALL,
            limits: Limits::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Clean,
    Error,
    Incomplete,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Finding {
    /// A protocol step failed (today: `fixed_sync_protocol_error`, or a typed
    /// strict-protocol kind when found by the reference run).
    Protocol {
        kind: &'static str,
        op: Option<OpId>,
        related: Vec<OpId>,
        detail: String,
        projection: String,
        witness: Vec<String>,
    },
    /// No transition enabled, warps unfinished (today: `deadlock`).
    Deadlock {
        projection: String,
        unfinished_warps: Vec<u32>,
        blocked_warps: Vec<u32>,
        unready_heads: Vec<String>,
        witness: Vec<String>,
    },
    /// Distinct terminal protocol states (today: `fixed_sync_nonconfluent`).
    NonConfluent {
        projection: String,
        complete_states: usize,
        witnesses: Vec<Vec<String>>,
    },
}

impl Finding {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Protocol { kind, .. } => kind,
            Self::Deadlock { .. } => "deadlock",
            Self::NonConfluent { .. } => "fixed_sync_nonconfluent",
        }
    }
}

/// Reasons the explorer cannot certify clean. Mirrors
/// `FixedSyncVerificationIncomplete` (`sync_fixed_verifier.rs:144-165`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IncompleteReason {
    /// Payload: `reason = "fixed_sync_program_build"`.
    ProgramBuild { detail: String },
    /// Payload: `reason = "fixed_sync_program_model_incomplete"`.
    ProgramModel {
        kind: &'static str,
        op: Option<OpId>,
        detail: String,
        projection: String,
    },
    /// Payload: `reason = "resource_limit", resource = "fixed_sync_states"`.
    StateLimit { projection: String, limit: usize },
    /// Payload: `reason = "resource_limit", resource = "fixed_sync_transitions"`.
    TransitionLimit { projection: String, limit: usize },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Projections checked (today: `program_count`).
    pub programs: usize,
    /// Projections answered by an isomorphic, already-clean one
    /// (today: `reused_clean_program_count`).
    pub reused_clean_programs: usize,
    /// Projections answered by a causal certificate (today: counted in
    /// `programs` with `visited_states += 1`).
    pub certified_programs: usize,
    pub visited_states: usize,
    pub explored_transitions: usize,
    pub strong_diamond_pruned_transitions: usize,
    pub sleep_pruned_transitions: usize,
}

#[derive(Clone, Debug)]
pub struct CheckReport {
    pub verdict: Verdict,
    pub findings: Vec<Finding>,
    pub incomplete: Vec<IncompleteReason>,
    pub stats: Stats,
    /// `coverage.termination.kind`: worklist_exhausted | finding | resource_limit | unsupported.
    pub termination: &'static str,
}

pub fn check(events: &[SyncEvent], config: &CheckConfig) -> CheckReport {
    let mut report = CheckReport {
        verdict: Verdict::Clean,
        findings: Vec::new(),
        incomplete: Vec::new(),
        stats: Stats::default(),
        termination: "worklist_exhausted",
    };
    let program = match Program::from_events(events) {
        Ok(program) => program,
        Err(detail) => {
            report.incomplete.push(IncompleteReason::ProgramBuild { detail });
            return finish(report);
        }
    };
    if program.commands.is_empty() {
        return finish(report);
    }

    // Phase-A analogue: one complete schedule, which also yields the clocks.
    let reference = reference_run(&program);
    match &reference.outcome {
        RunOutcome::Complete => {}
        outcome => {
            let ts = ProtocolTs::new(&program, &program.whole_spec(), None);
            let witness = render(&ts, &reference.schedule);
            match outcome {
                RunOutcome::Error(error) if error.error.incomplete => {
                    report.incomplete.push(IncompleteReason::ProgramModel {
                        kind: error.error.kind,
                        op: error.op,
                        detail: error.error.detail.clone(),
                        projection: "reference".into(),
                    })
                }
                RunOutcome::Error(error) => report.findings.push(Finding::Protocol {
                    kind: error.error.kind,
                    op: error.op,
                    related: Vec::new(),
                    detail: error.error.detail.clone(),
                    projection: "reference".into(),
                    witness,
                }),
                RunOutcome::Deadlock(deadlock) => report.findings.push(Finding::Deadlock {
                    projection: "reference".into(),
                    unfinished_warps: deadlock.unfinished_warps.clone(),
                    blocked_warps: deadlock.blocked_warps.clone(),
                    unready_heads: deadlock.unready_heads.clone(),
                    witness,
                }),
                RunOutcome::Complete => unreachable!(),
            }
            return finish(report);
        }
    }

    let specs = project(&program, config.mode);
    let systems = specs
        .iter()
        .map(|spec| ProtocolTs::new(&program, spec, Some(&reference)))
        .collect::<Vec<_>>();
    let mut fingerprints = HashMap::<Vec<u64>, ()>::new();
    for ts in &systems {
        report.stats.programs += 1;
        let projection = format!("{:?}", ts.key);
        if config.certificates {
            if let Some(result) = certify(ts, &reference) {
                report.stats.certified_programs += 1;
                report.stats.visited_states += 1;
                report.stats.explored_transitions += ts.cmds.len();
                match result {
                    Ok(()) => continue,
                    Err(error) if error.incomplete => {
                        report.incomplete.push(IncompleteReason::ProgramModel {
                            kind: error.kind,
                            op: error.op,
                            detail: error.detail,
                            projection,
                        });
                    }
                    Err(error) => report.findings.push(Finding::Protocol {
                        kind: error.kind,
                        op: error.op,
                        related: error.related,
                        detail: error.detail,
                        projection,
                        witness: Vec::new(),
                    }),
                }
                return finish(report);
            }
        }
        let key = config.fingerprints.then(|| fingerprint(ts)).flatten();
        if let Some(key) = &key {
            if fingerprints.contains_key(key) {
                report.stats.reused_clean_programs += 1;
                continue;
            }
        }
        let search = explore(ts, config.limits, config.explore);
        report.stats.visited_states += search.visited_states;
        report.stats.explored_transitions += search.explored_transitions;
        report.stats.strong_diamond_pruned_transitions += search.strong_diamond_pruned;
        report.stats.sleep_pruned_transitions += search.sleep_pruned;
        if let Some(failure) = search.failures.first() {
            match failure {
                Failure::Error { error, witness, .. } if error.error.incomplete => {
                    let _ = witness;
                    report.incomplete.push(IncompleteReason::ProgramModel {
                        kind: error.error.kind,
                        op: error.op,
                        detail: error.error.detail.clone(),
                        projection,
                    })
                }
                Failure::Error { error, witness, .. } => report.findings.push(Finding::Protocol {
                    kind: error.error.kind,
                    op: error.op,
                    related: Vec::new(),
                    detail: error.error.detail.clone(),
                    projection,
                    witness: render(ts, witness),
                }),
                Failure::Deadlock { deadlock, witness } => report.findings.push(Finding::Deadlock {
                    projection,
                    unfinished_warps: deadlock.unfinished_warps.clone(),
                    blocked_warps: deadlock.blocked_warps.clone(),
                    unready_heads: deadlock.unready_heads.clone(),
                    witness: render(ts, witness),
                }),
            }
            return finish(report);
        }
        match search.termination {
            Termination::StateLimit(limit) => {
                report.incomplete.push(IncompleteReason::StateLimit { projection, limit });
                return finish(report);
            }
            Termination::TransitionLimit(limit) => {
                report.incomplete.push(IncompleteReason::TransitionLimit { projection, limit });
                return finish(report);
            }
            Termination::Exhausted if !search.confluent() => {
                report.findings.push(Finding::NonConfluent {
                    projection,
                    complete_states: search.complete_states,
                    witnesses: search.complete_witnesses.iter().map(|w| render(ts, w)).collect(),
                });
                return finish(report);
            }
            Termination::Exhausted | Termination::FirstFailure => {}
        }
        if let Some(key) = key {
            fingerprints.insert(key, ());
        }
    }
    finish(report)
}

fn finish(mut report: CheckReport) -> CheckReport {
    report.verdict = if !report.findings.is_empty() {
        Verdict::Error
    } else if !report.incomplete.is_empty() {
        Verdict::Incomplete
    } else {
        Verdict::Clean
    };
    report.termination = if !report.findings.is_empty() {
        "finding"
    } else if report.incomplete.iter().any(|reason| {
        matches!(reason, IncompleteReason::StateLimit { .. } | IncompleteReason::TransitionLimit { .. })
    }) {
        "resource_limit"
    } else if !report.incomplete.is_empty() {
        "unsupported"
    } else {
        "worklist_exhausted"
    };
    report
}

/// Replay a witness to name each transition by its operation.
fn render(ts: &ProtocolTs<'_>, witness: &[Transition]) -> Vec<String> {
    let mut state: State = ts.initial();
    let mut out = Vec::with_capacity(witness.len());
    for transition in witness {
        out.push(ts.describe(&state, transition));
        match ts.step(&state, transition) {
            Ok(next) => state = next,
            Err(_) => break,
        }
    }
    out
}
