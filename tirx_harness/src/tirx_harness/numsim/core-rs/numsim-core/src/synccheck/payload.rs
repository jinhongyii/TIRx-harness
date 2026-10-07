//! Mapping explorer results onto `report::{Finding, Evidence}` and rendering
//! a [`Report`] as today's native Synccheck payload (spec section 4).
//!
//! Each finding carries one `Evidence { role: "payload", detail: <json> }`
//! holding its legacy payload entry (with a private `_slot` key naming the
//! list it belongs to), plus `operation` / `witness` evidence for renderers.
//! Search statistics and limits travel in `Report::coverage`.

use std::time::Instant;

use serde_json::{json, Value};

use super::certificate::CertError;
use super::explore::{Failure, SearchResult, Termination};
use super::kinds;
use super::program::{PhaseAFailure, Program};
use super::reference::{whole_spec, ReferenceRun, RunOutcome};
use super::ts::{Deadlock, ErrKind, Transition, Ts, TsError};
use super::SynccheckConfig;
use crate::observe::Actor;
use crate::report::{Evidence, Finding, FindingKind, Report, Status, Verdict};
use crate::site::SiteId;

#[derive(Default)]
pub struct Stats {
    pub programs: u64,
    pub reused: u64,
    pub certified: u64,
    pub visited: u64,
    pub transitions: u64,
    pub diamond_pruned: u64,
    pub sleep_pruned: u64,
}

pub struct Builder<'c> {
    config: &'c SynccheckConfig,
    pub stats: Stats,
    findings: Vec<Finding>,
    commands: u64,
}

fn op_json(program: &Program, cmd: usize) -> Value {
    let c = &program.commands[cmd];
    json!({
        "kernel_index": Value::Null,
        "global_warp_id": c.warp.0,
        "per_warp_sequence": c.seq,
        "source_op_id": c.site.0,
        "loop_frames": c.frames.iter().map(|f| json!({"loop_site_id": f.site.0, "iteration_ordinal": f.iteration})).collect::<Vec<_>>(),
    })
}

/// The one place that builds `report::Evidence` (no memory fields: sync evidence).
fn ev(kernel: u32, role: &str, site: SiteId, actor: Option<Actor>, detail: Option<String>) -> Evidence {
    Evidence {
        role: role.to_owned(),
        kernel,
        site,
        actor,
        buffer: None,
        space: None,
        alloc: None,
        bytes: None,
        detail,
    }
}

fn op_evidence(program: &Program, cmd: usize, role: &str) -> Evidence {
    let c = &program.commands[cmd];
    ev(0, role, c.site, Some(Actor::Warp { warp: c.warp, epoch: c.epoch }), None)
}

fn payload_evidence(payload: &Value) -> Evidence {
    ev(0, "payload", SiteId::NONE, None, Some(payload.to_string()))
}

fn ts_error_message(e: &TsError) -> (String, Option<&'static str>) {
    match &e.kind {
        ErrKind::Protocol(err) => (format!("{err:?}"), Some(kinds::error_kind(err))),
        ErrKind::Incomplete { reason, detail } => (format!("{reason}: {detail}"), None),
    }
}

/// Replay `witness` and describe each step.
fn render(ts: &Ts<'_>, witness: &[Transition]) -> (Vec<String>, Vec<Value>, Vec<Evidence>) {
    let mut state = ts.initial();
    let mut names = Vec::new();
    let mut steps = Vec::new();
    let mut evidence = Vec::new();
    for t in witness {
        let (description, cmd) = ts.describe(&state, t);
        names.push(format!("{t:?}"));
        steps.push(json!({
            "transition": format!("{t:?}"),
            "description": description,
            "operation": cmd.map(|c| op_json(ts.program, c)),
        }));
        if let Some(c) = cmd {
            let mut e = op_evidence(ts.program, c, "witness");
            e.detail = Some(description);
            evidence.push(e);
        }
        match ts.step_fx(&state, t) {
            Ok((next, _)) => state = next,
            Err(_) => break,
        }
    }
    (names, steps, evidence)
}

impl<'c> Builder<'c> {
    pub fn new(config: &'c SynccheckConfig) -> Self {
        Self { config, stats: Stats::default(), findings: Vec::new(), commands: 0 }
    }

    fn push(&mut self, kind: FindingKind, status: Status, message: String, mut evidence: Vec<Evidence>, payload: Value) {
        let mut sites = evidence.iter().map(|e| e.site).filter(|s| *s != SiteId::NONE).collect::<Vec<_>>();
        sites.sort_unstable_by_key(|s| s.0);
        sites.dedup();
        evidence.push(payload_evidence(&payload));
        self.findings.push(Finding { kind, status, message, sites, evidence });
    }

    fn incomplete(&mut self, kind: FindingKind, message: String, evidence: Vec<Evidence>, mut payload: Value) {
        payload["kind"] = json!("analysis_incomplete");
        payload["message"] = json!(message.clone());
        payload["_slot"] = json!("incomplete");
        self.push(kind, Status::Incomplete, message, evidence, payload);
    }

    pub fn program_build(&mut self, detail: String) {
        let message = format!("cannot build fixed synchronization program: {detail}");
        self.incomplete(FindingKind::Unsupported, message, Vec::new(), json!({"reason": "fixed_sync_program_build", "source": detail}));
    }

    pub fn wall_time_limit(&mut self, started: Instant) {
        let usage = started.elapsed().as_millis() as u64;
        let limit = self.config.limits.max_wall_time_ms;
        self.incomplete(
            FindingKind::BudgetExhausted,
            "native fixed synchronization verification exceeded its wall_time resource limit".into(),
            Vec::new(),
            json!({"reason": "resource_limit", "resource": "wall_time",
                   "limit": {"kind": "milliseconds", "value": limit},
                   "usage": {"kind": "milliseconds", "value": usage}}),
        );
    }

    /// Failures the engine already hit in the concrete run.
    pub fn phase_a(&mut self, failures: &[PhaseAFailure]) {
        let mut blocked = Vec::new();
        let mut blocked_evidence = Vec::new();
        for f in failures {
            let crate::observe::SyncKind::Protocol { cmds, .. } = &f.event.kind else { continue };
            let Actor::Warp { warp, epoch } = f.event.actor else { continue };
            let operation = json!({
                "kernel_index": Value::Null,
                "global_warp_id": warp.0,
                "per_warp_sequence": f.event.seq,
                "source_op_id": f.event.site.0,
                "loop_frames": f.event.frames.iter().map(|fr| json!({"loop_site_id": fr.site.0, "iteration_ordinal": fr.iteration})).collect::<Vec<_>>(),
            });
            let evidence = ev(self.config.launch, "operation", f.event.site, Some(f.event.actor), None);
            match &f.error {
                Some(error) => {
                    let effect = cmds
                        .iter()
                        .find(|(_, c)| kinds::same_protocol(c, error))
                        .or(cmds.first())
                        .map_or("unknown", |(_, c)| kinds::effect_name(c));
                    let message = format!("synccheck rejected {effect} at warp {} #{}: {error:?}", warp.0, f.event.seq);
                    let payload = json!({
                        "_slot": "findings",
                        "kind": kinds::error_kind(error),
                        "effect": effect,
                        "operation": operation,
                        "message": message.clone(),
                    });
                    self.push(kinds::finding_kind(error), Status::Error, message, vec![evidence], payload);
                }
                None => {
                    blocked.push(json!({
                        "warp_id": warp.0,
                        "awaited_operation": format!("{:?}", cmds.iter().map(|(r, c)| (r, c)).collect::<Vec<_>>()),
                        "phase": Value::Null,
                        "description": format!("warp {} #{} (epoch {epoch}) blocked at exit", warp.0, f.event.seq),
                        "operation": operation,
                    }));
                    blocked_evidence.push(evidence);
                }
            }
        }
        if !blocked.is_empty() {
            let message = format!("executor deadlock; {} blocked warps", blocked.len());
            let payload = json!({
                "_slot": "execution_error",
                "kind": "deadlock",
                "message": message.clone(),
                "operation": blocked[0]["operation"].clone(),
                "blocked_operations": blocked.clone(),
                "blocked_warp_count": blocked.len(),
                "blocked_operation_count": blocked.len(),
                "stalled_operations": [],
                "diagnostic_retention": "full",
            });
            self.push(FindingKind::Deadlock, Status::Error, message, blocked_evidence, payload);
        }
    }

    fn protocol_error(&mut self, ts: &Ts<'_>, error: &TsError, transition: Option<Transition>, witness: &[Transition]) {
        let (source, source_kind) = ts_error_message(error);
        let (names, steps, mut evidence) = render(ts, witness);
        if let Some(c) = error.cmd {
            evidence.insert(0, op_evidence(ts.program, c, "operation"));
        }
        let operation = error.cmd.map(|c| op_json(ts.program, c));
        match &error.kind {
            ErrKind::Protocol(err) => {
                let protocol = kinds::protocol_name(err);
                let message = format!("fixed synchronization program rejected {transition:?}: {source}");
                let payload = json!({
                    "_slot": "findings",
                    "kind": "fixed_sync_protocol_error",
                    "protocol": protocol,
                    "transition": transition.map(|t| format!("{t:?}")),
                    "source": source,
                    "source_kind": source_kind,
                    "message": message.clone(),
                    "operation": operation,
                    "related_operations": [],
                    "witness": names,
                    "witness_evidence": steps,
                });
                self.push(kinds::finding_kind(err), Status::Error, message, evidence, payload);
            }
            ErrKind::Incomplete { reason, .. } => {
                let legacy = matches!(*reason, "cluster_barrier_rearrival_without_wait_unmodeled" | "cluster_barrier_warp_exit_unmodeled");
                let message = format!("fixed synchronization verification is incomplete at {transition:?}: {source}");
                self.incomplete(
                    FindingKind::Unsupported,
                    message,
                    evidence,
                    json!({
                        "reason": if legacy { *reason } else { "fixed_sync_program_model_incomplete" },
                        "operation": operation,
                        "transition": transition.map(|t| format!("{t:?}")),
                        "source": source,
                        "witness": names,
                    }),
                );
            }
        }
    }

    fn deadlock(&mut self, ts: &Ts<'_>, d: &Deadlock, witness: &[Transition]) {
        let (names, steps, evidence) = render(ts, witness);
        let message = format!(
            "fixed synchronization domain {} can deadlock with unfinished warps {:?}, blocked warps {:?}, and unready heads {:?}",
            d.domain, d.unfinished, d.blocked, d.heads
        );
        let payload = json!({
            "_slot": "findings",
            "kind": "deadlock",
            "verification": "fixed_sync",
            "deadlock": format!("{d:?}"),
            "message": message.clone(),
            "operation": Value::Null,
            "witness": names,
            "witness_evidence": steps,
        });
        self.push(FindingKind::Deadlock, Status::Error, message, evidence, payload);
    }

    pub fn reference_failure(&mut self, program: &Program, init: &crate::sync::ResourceInit, reference: &ReferenceRun) {
        let ts = Ts::new(program, &whole_spec(program), init, None).expect("reference built the same system");
        match &reference.outcome {
            RunOutcome::Error(e) => {
                let (last, prefix) = reference.schedule.split_last().map_or((None, &[][..]), |(l, p)| (Some(*l), p));
                let _ = prefix;
                self.protocol_error(&ts, e, last, &reference.schedule);
            }
            RunOutcome::Deadlock(d) => self.deadlock(&ts, d, &reference.schedule),
            RunOutcome::Complete => {}
        }
    }

    pub fn certificate_failure(&mut self, program: &Program, ts: &Ts<'_>, e: CertError) {
        let mut evidence = Vec::new();
        if let Some(c) = e.cmd {
            evidence.push(op_evidence(program, c, "operation"));
        }
        for &r in &e.related {
            evidence.push(op_evidence(program, r, "related"));
        }
        let operation = e.cmd.map(|c| op_json(program, c));
        if e.incomplete {
            let legacy = matches!(e.kind, "cluster_barrier_rearrival_without_wait_unmodeled" | "cluster_barrier_warp_exit_unmodeled");
            let message = format!("fixed {} verification is incomplete: {}", e.protocol, e.detail);
            self.incomplete(
                FindingKind::Unsupported,
                message,
                evidence,
                json!({
                    "reason": if legacy { e.kind } else { "fixed_sync_program_model_incomplete" },
                    "operation": operation,
                    "transition": "ValidateExit",
                    "source": format!("{}: {}", e.kind, e.detail),
                    "witness": [],
                }),
            );
            return;
        }
        let _ = ts;
        let message = format!("fixed {} protocol rejected the program: {}", e.protocol, e.detail);
        let payload = json!({
            "_slot": "findings",
            "kind": "fixed_sync_protocol_error",
            "protocol": e.protocol,
            "transition": "ValidateExit",
            "source": e.detail,
            "source_kind": e.kind,
            "message": message.clone(),
            "operation": operation,
            "related_operations": e.related.iter().map(|&r| op_json(program, r)).collect::<Vec<_>>(),
            "witness": [],
            "witness_evidence": [],
        });
        self.push(kinds::protocol_finding_kind(e.protocol), Status::Error, message, evidence, payload);
    }

    /// Review-level exit lints (only reported when nothing else was found).
    pub fn lints(&mut self, program: &Program, reference: &ReferenceRun) {
        if !self.findings.is_empty() {
            return;
        }
        for (r, kind, detail) in &reference.lints {
            let message = format!("{:?} at exit: {detail}", program.resources[*r]);
            let payload = json!({"_slot": "review", "kind": "sync_exit_lint", "lint": detail, "resource": format!("{:?}", program.resources[*r]), "message": message.clone()});
            self.push(kind.clone(), Status::Review, message, Vec::new(), payload);
        }
    }

    /// Returns true when the result is terminal (stop checking projections).
    pub fn search_result(&mut self, program: &Program, ts: &Ts<'_>, search: &SearchResult<Transition, TsError, Deadlock>) -> bool {
        if let Some(failure) = search.failures.first() {
            match failure {
                Failure::Error { transition, error, witness } => self.protocol_error(ts, error, Some(*transition), witness),
                Failure::Deadlock { deadlock, witness } => self.deadlock(ts, deadlock, witness),
            }
            return true;
        }
        let first_op = ts.cmds.first().map(|c| op_json(program, c.global));
        match search.termination {
            Termination::StateLimit(limit) | Termination::TransitionLimit(limit) => {
                let resource = if matches!(search.termination, Termination::StateLimit(_)) {
                    "fixed_sync_states"
                } else {
                    "fixed_sync_transitions"
                };
                let message = format!("fixed synchronization program reached its {limit}-{} limit", if resource == "fixed_sync_states" { "state" } else { "transition" });
                self.incomplete(
                    FindingKind::BudgetExhausted,
                    message,
                    Vec::new(),
                    json!({"reason": "resource_limit", "resource": resource, "operation": first_op, "limit": limit}),
                );
                true
            }
            Termination::Exhausted if !search.confluent() => {
                let rendered = search.complete_witnesses.iter().map(|w| render(ts, w)).collect::<Vec<_>>();
                let message = format!("fixed synchronization program has {} distinct terminal protocol states", search.complete_states);
                let evidence = rendered.iter().flat_map(|r| r.2.clone()).collect();
                let payload = json!({
                    "_slot": "findings",
                    "kind": "fixed_sync_nonconfluent",
                    "complete_states": search.complete_states,
                    "message": message.clone(),
                    "operation": first_op,
                    "witness": rendered.last().map(|r| r.0.clone()).unwrap_or_default(),
                    "witnesses": rendered.iter().map(|r| r.0.clone()).collect::<Vec<_>>(),
                    "witnesses_evidence": rendered.iter().map(|r| r.1.clone()).collect::<Vec<_>>(),
                });
                self.push(FindingKind::Other("fixed_sync_nonconfluent".into()), Status::Error, message, evidence, payload);
                true
            }
            Termination::FirstFailure => {
                self.incomplete(
                    FindingKind::Unsupported,
                    "fixed synchronization program stopped without retaining its first failure".into(),
                    Vec::new(),
                    json!({"reason": "fixed_sync_first_failure_unretained", "operation": first_op}),
                );
                true
            }
            Termination::Exhausted => false,
        }
    }

    pub fn finish(mut self, started: Instant) -> Report {
        self.commands = self.stats.transitions;
        let c = self.config;
        let mut findings = std::mem::take(&mut self.findings);
        for f in &mut findings {
            for e in &mut f.evidence {
                e.kernel = c.launch;
            }
        }
        let mut report = Report::new("synccheck", findings);
        report.launch = c.launch;
        report.coverage = vec![
            ("program_count".into(), self.stats.programs),
            ("reused_clean_program_count".into(), self.stats.reused),
            ("certified_program_count".into(), self.stats.certified),
            ("visited_state_count".into(), self.stats.visited),
            ("explored_transition_count".into(), self.stats.transitions),
            ("strong_diamond_pruned_transition_count".into(), self.stats.diamond_pruned),
            ("sleep_pruned_transition_count".into(), self.stats.sleep_pruned),
            ("wall_time_us".into(), started.elapsed().as_micros() as u64),
            ("limit_max_schedules".into(), c.limits.max_schedules),
            ("limit_max_backtrack_nodes".into(), c.state_budget),
            ("limit_max_events_per_run".into(), c.limits.max_events_per_run),
            ("limit_max_total_events".into(), c.limits.max_total_events),
            ("limit_max_loop_steps".into(), c.transition_budget),
            ("limit_max_wall_time_ms".into(), c.limits.max_wall_time_ms),
            ("limit_max_diagnostic_bytes".into(), c.limits.max_diagnostic_bytes),
        ];
        report
    }
}

fn coverage(report: &Report, key: &str) -> u64 {
    report.coverage.iter().find(|(k, _)| k == key).map_or(0, |(_, v)| *v)
}

fn finding_payload(f: &Finding) -> Value {
    f.evidence
        .iter()
        .find(|e| e.role == "payload")
        .and_then(|e| e.detail.as_deref())
        .and_then(|d| serde_json::from_str(d).ok())
        .unwrap_or_else(|| {
            let kind = serde_json::to_value(&f.kind).ok().and_then(|v| v.as_str().map(str::to_owned)).unwrap_or_else(|| format!("{:?}", f.kind));
            let slot = if f.status == Status::Incomplete { "incomplete" } else { "findings" };
            json!({"_slot": slot, "kind": kind, "message": f.message})
        })
}

/// Render a Synccheck [`Report`] as today's native payload (the keys the
/// Python tests pin; spec section 4). Launch facts the explorer does not see
/// (`phase`, `analysis_scope`, `effects`, `stats`, `timing`,
/// `resource_limits`, `replay_resource_limits`, `kernel_index` in
/// operations) are left for the Python layer to add.
pub fn serialize(report: &Report) -> Value {
    let mut findings = Vec::new();
    let mut incomplete = Vec::new();
    let mut execution_error = Value::Null;
    let mut review = Vec::new();
    for f in &report.findings {
        let mut p = finding_payload(f);
        let slot = p.get("_slot").and_then(Value::as_str).unwrap_or("findings").to_owned();
        if let Some(o) = p.as_object_mut() {
            o.remove("_slot");
        }
        match slot.as_str() {
            "incomplete" => incomplete.push(p),
            "review" => review.push(p),
            "execution_error" if execution_error.is_null() => execution_error = p,
            "execution_error" => findings.push(p),
            _ => findings.push(p),
        }
    }
    let verdict = match report.verdict {
        Verdict::Clean => "clean",
        Verdict::Review => "review",
        Verdict::Incomplete => "incomplete",
        Verdict::Error => "error",
    };
    let limit_hit = incomplete.iter().find(|p| p["reason"] == "resource_limit").cloned();
    let visited = coverage(report, "visited_state_count");
    let transitions = coverage(report, "explored_transition_count");
    let wall_ms = coverage(report, "wall_time_us") / 1000;
    let (termination_kind, resource_limit) = if report.verdict == Verdict::Error {
        ("finding", Value::Null)
    } else if let Some(hit) = &limit_hit {
        let (resource, usage) = match hit["resource"].as_str() {
            Some("fixed_sync_states") => ("backtrack_nodes", visited),
            Some("fixed_sync_transitions") => ("loop_steps", transitions),
            Some(other) => (other, 0),
            None => ("unknown", 0),
        };
        let limit = hit.get("limit").cloned().unwrap_or(Value::Null);
        let limit = if limit.is_object() { limit } else { json!({"kind": "count", "value": limit}) };
        ("resource_limit", json!({"resource": resource, "limit": limit, "usage": hit.get("usage").cloned().unwrap_or(json!({"kind": "count", "value": usage}))}))
    } else if !incomplete.is_empty() {
        ("unsupported", Value::Null)
    } else {
        ("worklist_exhausted", Value::Null)
    };
    let run_status = match termination_kind {
        "finding" => "finding",
        "worklist_exhausted" => "complete",
        _ => "incomplete",
    };
    let incomplete_reason = incomplete.first().and_then(|p| p["message"].as_str().map(str::to_owned));
    json!({
        "schema_version": 3,
        "execution_model": "direct_fixed_sync_state",
        "verdict": verdict,
        "findings": findings,
        "incomplete": incomplete,
        "review": review,
        "execution_error": execution_error,
        "counterexample": Value::Null,
        "search": {
            "algorithm": "fixed_sync_state",
            "run_count": 1,
            "backtrack_count": 0,
            "sleep_pruned_branch_count": 0,
            "program_count": coverage(report, "program_count"),
            "reused_clean_program_count": coverage(report, "reused_clean_program_count"),
            "certified_program_count": coverage(report, "certified_program_count"),
            "visited_state_count": visited,
            "explored_transition_count": transitions,
            "strong_diamond_pruned_transition_count": coverage(report, "strong_diamond_pruned_transition_count"),
            "sleep_pruned_transition_count": coverage(report, "sleep_pruned_transition_count"),
            "incomplete_reason": incomplete_reason,
            "runs": [{
                "prefix": [],
                "warp_preemption_bound": 0,
                "trace_digest": Value::Null,
                "trace_digest_hex": Value::Null,
                "coverage_usage": {"warp_preemptions": 0, "completion_schedule_deviations": 0},
                "status": run_status,
            }],
        },
        "coverage": {
            "status": match report.verdict {
                Verdict::Error => "finding",
                Verdict::Clean | Verdict::Review => "complete_within_bounds",
                Verdict::Incomplete => "incomplete",
            },
            "eligible_for_clean": report.verdict == Verdict::Clean,
            "bounds": {"max_warp_preemptions": 0, "max_completion_schedule_deviations": 0},
            "maximum_observed_usage": {"warp_preemptions": 0, "completion_schedule_deviations": 0},
            "resource_limits": {
                "max_schedules": coverage(report, "limit_max_schedules"),
                "max_backtrack_nodes": coverage(report, "limit_max_backtrack_nodes"),
                "max_events_per_run": coverage(report, "limit_max_events_per_run"),
                "max_total_events": coverage(report, "limit_max_total_events"),
                "max_loop_steps": coverage(report, "limit_max_loop_steps"),
                "max_wall_time_ms": coverage(report, "limit_max_wall_time_ms"),
                "max_diagnostic_bytes": coverage(report, "limit_max_diagnostic_bytes"),
            },
            "resource_usage": {
                "schedules": 1,
                "backtrack_nodes": visited,
                "events_in_current_run": transitions,
                "total_events": transitions,
                "loop_steps": transitions,
                "wall_time_ms": wall_ms,
                "diagnostic_bytes": 0,
            },
            "pending_work_items": 0,
            "pending_backtracks": 0,
            "termination": {"kind": termination_kind, "resource_limit": resource_limit},
        },
    })
}
