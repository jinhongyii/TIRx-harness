//! Racecheck results → `report::Report` (the cross-checker contract) and →
//! the legacy native payload JSON the Python tests pin
//! (`race_check_python.rs:184-560`).
//!
//! Race-specific facts (`access_pair`, `ordering_domain`, `ordering_failure`,
//! `proxy_bridge`, `hint`, `occurrences`, incomplete `reason`, and the
//! per-witness `prior` / `current` objects) live in `Finding::attrs`;
//! `prior` / `current` / `overlap` evidence carries site, actor, space,
//! allocation and bytes. [`serialize`] reads only the `Report`, so the
//! payload can be rebuilt from a stored report.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::arena::{AllocId, ByteSpan, Space};
use crate::observe::{Actor, Side, WarpId as CWarpId, Window};
use crate::program::{Proxy, Scope};
use crate::report::{Evidence, Finding, FindingKind, Report, Status};
use crate::site::SiteId;

use super::checker::{
    AdvisoryKind, Finding as RaceFinding, FindingKind as RK, Incomplete, OrderingFailure, RaceClass, Severity, WitnessInfo,
};
use super::input::AccessKind;
use super::observer::{LaunchResult, RaceObserver};

const SPIN_HINT: &str = "If this access relies on a hand-written spin wait, consider wait_until to express the exit condition. The producer must still provide the required release/synchronization; changing the wait API alone does not establish happens-before.";
const TMEM_REVIEW_URL: &str = "https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions";

fn space_name(s: Space) -> &'static str {
    match s {
        Space::Global => "global",
        Space::Shared => "shared",
        Space::Tmem => "tmem",
        Space::Local => "local",
        Space::Param => "param",
        Space::Reg => "reg",
    }
}

fn kind_name(k: AccessKind) -> &'static str {
    match k {
        AccessKind::Read => "read",
        AccessKind::Write => "write",
        AccessKind::Rmw => "atomic_read_modify_write",
    }
}

fn proxy_name(p: Proxy) -> &'static str {
    match p {
        Proxy::Generic => "generic",
        Proxy::Async => "async",
        Proxy::TensorMap => "tensormap",
        Proxy::ReadOnly => "readonly",
        Proxy::Tcgen => "tcgen",
    }
}

fn domain_name(d: Option<Window>) -> &'static str {
    match d {
        Some(Window::Global) => "global",
        Some(Window::SharedCta) => "shared_cta",
        Some(Window::SharedCluster) => "shared_cluster",
        None => "other",
    }
}

fn scope_name(s: Scope) -> &'static str {
    match s {
        Scope::Cta => "cta",
        Scope::Cluster => "cluster",
        Scope::Gpu => "gpu",
        Scope::Sys => "sys",
    }
}

fn class_name(c: RaceClass) -> &'static str {
    match c {
        RaceClass::WriteRead => "write_read",
        RaceClass::ReadWrite => "read_write",
        RaceClass::WriteWrite => "write_write",
    }
}

fn failure_name(f: OrderingFailure) -> (&'static str, &'static str) {
    match f {
        OrderingFailure::MissingInterActorSync => ("missing_inter_actor_sync", "execution"),
        OrderingFailure::MissingSameWarpLaneOrder => ("missing_same_warp_lane_order", "execution"),
        OrderingFailure::MissingReleaseAcquire => ("missing_release_acquire", "memory"),
        OrderingFailure::MissingProxyBridge { .. } => ("missing_proxy_bridge", "memory"),
        OrderingFailure::AsyncLifetimeNotDrained => ("async_lifetime_not_drained", "completion"),
    }
}

fn incomplete_reason(i: &Incomplete) -> (String, Value) {
    let (reason, extra) = match i {
        Incomplete::EpochRegression { warp } => ("shadow_rejected", json!({"cause": "epoch_regression", "warp_id": warp})),
        Incomplete::EpochOverflow { warp } => ("shadow_rejected", json!({"cause": "epoch_overflow", "warp_id": warp})),
        Incomplete::UnknownAlloc { alloc } => ("shadow_rejected", json!({"cause": "unknown_allocation", "allocation_id": alloc.0})),
        Incomplete::UnknownAsyncOp { op } => ("effect_commit_unobserved", json!({"cause": "unknown_async_op", "async_op": op.0})),
        Incomplete::AsyncNeverCompleted { op } => ("effect_commit_unobserved", json!({"effect": "async_operation", "async_op": op.0})),
        Incomplete::WaitExitUnproven { warp } => ("wait_exit_unproven", json!({"warp_id": warp})),
        Incomplete::SyncQualifierUnknown { warp } => ("sync_qualifier_unknown", json!({"warp_id": warp})),
        Incomplete::MbarrierScopeUnknown { warp } => ("mbarrier_scope_unknown", json!({"warp_id": warp})),
        Incomplete::WaitPredicateReadsUnstable { warp } => ("wait_predicate_reads_unstable", json!({"warp_id": warp})),
        Incomplete::EventOutsideLaunch { events } => ("event_outside_launch", json!({"events": events})),
        Incomplete::FindingsTruncated { dropped } => ("findings_truncated", json!({"dropped": dropped})),
        Incomplete::SignalWriteNotRecorded { warp } => ("signal_write_not_recorded", json!({"warp_id": warp})),
        Incomplete::AsyncLaneUnknown { op } => ("async_lane_unknown", json!({"async_op": op.0})),
        Incomplete::CompletionWarpOutOfRange { warp } => ("shadow_rejected", json!({"cause": "completion_warp_out_of_range", "warp_id": warp})),
        Incomplete::KernelMismatch { expected, got } => ("kernel_mismatch", json!({"expected": expected, "got": got})),
    };
    (reason.to_string(), extra)
}

fn actor_of(w: &WitnessInfo) -> Actor {
    match w.async_op {
        Some(op) => Actor::Async { op, side: if w.kind == AccessKind::Read { Side::Read } else { Side::Write } },
        None => Actor::Warp { warp: CWarpId(w.warp), epoch: u64::from(w.epoch) },
    }
}

fn witness_attrs(w: &WitnessInfo) -> Value {
    json!({
        "lane": w.lane,
        "global_warp_id": w.warp,
        "epoch": w.epoch,
        "async_op": w.async_op.map(|o| o.0),
        "access_kind": kind_name(w.kind),
        "proxy": proxy_name(w.proxy),
        "domain": domain_name(w.domain),
        "scope": w.scope.map(scope_name),
        "atomic": w.atomic,
    })
}

fn witness_evidence(role: &str, kernel: u32, w: &WitnessInfo, alloc: AllocId, lr: &LaunchResult) -> Evidence {
    let (buffer, space) = lr.buffers.get(&alloc).map(|(n, s)| (Some(n.clone()), Some(*s))).unwrap_or((None, None));
    let detail = format!(
        "{} by warp {} lane {}{} via the {} proxy",
        kind_name(w.kind),
        w.warp,
        w.lane,
        w.async_op.map(|o| format!(" (async op {})", o.0)).unwrap_or_default(),
        proxy_name(w.proxy)
    );
    Evidence {
        role: role.to_string(),
        kernel,
        site: w.site,
        actor: Some(actor_of(w)),
        buffer,
        space,
        alloc: Some(alloc),
        bytes: Some(ByteSpan::new(w.span.start, w.span.end - w.span.start)),
        detail: Some(detail),
    }
}

fn convert(f: &RaceFinding, lr: &LaunchResult) -> Finding {
    let kernel = lr.kernel;
    let mut ev = Vec::new();
    let mut race = Map::new();
    let (kind, status, message) = match &f.kind {
        RK::SignalProtocolError { class, failure } => {
            let (fname, dom) = failure_name(*failure);
            race.insert("legacy_kind".into(), json!("signal_protocol_error"));
            race.insert("access_pair".into(), json!(class_name(*class)));
            race.insert("ordering_domain".into(), json!(dom));
            race.insert("ordering_failure".into(), json!(fname));
            race.insert("hint".into(), json!("Order initialization/reset and other plain accesses before or after the wait using synchronization. Raw scoped publications remain allowed; not every signal access must use wait_until."));
            let msg = format!(
                "declared word bytes [{}..{}) of allocation {} are accessed by wait_until and a plain operation without a happens-before relationship",
                f.bytes.start, f.bytes.end, f.alloc.0
            );
            (FindingKind::SignalProtocolError, Status::Error, msg)
        }
        RK::DataRace { class, failure } | RK::TmemLifetimeReview { class, failure } => {
            let review = matches!(f.kind, RK::TmemLifetimeReview { .. });
            let (fname, dom) = failure_name(*failure);
            race.insert("legacy_kind".into(), json!(if review { "tmem_lifetime_review" } else { "data_race" }));
            race.insert("access_pair".into(), json!(class_name(*class)));
            race.insert("ordering_domain".into(), json!(dom));
            race.insert("ordering_failure".into(), json!(fname));
            if let OrderingFailure::MissingProxyBridge { prior, current, domain } = failure {
                let cur_dom = f.current.as_ref().and_then(|c| c.domain);
                race.insert(
                    "proxy_bridge".into(),
                    json!({"prior_proxy": proxy_name(*prior), "current_proxy": proxy_name(*current),
                           "prior_domain": domain_name(*domain), "current_domain": domain_name(cur_dom)}),
                );
            }
            if !review && *failure == OrderingFailure::MissingReleaseAcquire {
                race.insert("hint".into(), json!(SPIN_HINT));
            }
            let kind = if review {
                FindingKind::TmemLifetimeReview
            } else {
                match failure {
                    OrderingFailure::MissingProxyBridge { .. } => FindingKind::ProxyRace,
                    OrderingFailure::AsyncLifetimeNotDrained => FindingKind::AsyncRace,
                    _ => FindingKind::DataRace,
                }
            };
            let msg = format!(
                "{} conflict on bytes [{}..{}) of allocation {}: {}",
                class_name(*class),
                f.bytes.start,
                f.bytes.end,
                f.alloc.0,
                fname
            );
            let msg = if review {
                format!("TMEM lifetime conflict requires review: the earlier tcgen05.ld may not have completed before the conflicting reuse ({msg}); inspect the register dependency, otherwise add tcgen05.wait::ld; PTX: {TMEM_REVIEW_URL}")
            } else {
                msg
            };
            (kind, if review { Status::Review } else { Status::Error }, msg)
        }
        RK::ScopeMismatch { release_scope, acquire_scope, release_warp, acquire_warp, release_site, acquire_site } => {
            race.insert("legacy_kind".into(), json!("scope_mismatch"));
            race.insert("ordering_domain".into(), json!("memory"));
            race.insert("ordering_failure".into(), json!("scope_mismatch"));
            race.insert("release_scope".into(), json!(scope_name(*release_scope)));
            race.insert("acquire_scope".into(), json!(scope_name(*acquire_scope)));
            race.insert("release_warp_id".into(), json!(release_warp));
            race.insert("acquire_warp_id".into(), json!(acquire_warp));
            race.insert("release_site".into(), json!(release_site.0));
            race.insert("acquire_site".into(), json!(acquire_site.0));
            for (role, site, w) in [("release", *release_site, *release_warp), ("acquire", *acquire_site, *acquire_warp)] {
                ev.push(Evidence {
                    role: role.into(),
                    kernel,
                    site,
                    actor: None,
                    buffer: None,
                    space: None,
                    alloc: None,
                    bytes: None,
                    detail: Some(format!("{role} by warp {w}")),
                });
            }
            let msg = format!(
                "release .{} (warp {release_warp}) and acquire .{} (warp {acquire_warp}) do not mutually cover each other's thread",
                scope_name(*release_scope),
                scope_name(*acquire_scope)
            );
            (FindingKind::ScopeMismatch, Status::Error, msg)
        }
        RK::Advisory { kind } => {
            let (name, ck) = match kind {
                AdvisoryKind::CrossCtaAsyncOrder => ("cross_cta_async_order", FindingKind::CrossCtaAsyncOrder),
                AdvisoryKind::UndeclaredProtocolWord => ("undeclared_protocol_word", FindingKind::UndeclaredProtocolWord),
                AdvisoryKind::AliasStaleRead => ("alias_stale_read", FindingKind::AliasStaleRead),
            };
            if *kind == AdvisoryKind::AliasStaleRead {
                // Legacy advisory keys.
                let reader = f.current.as_ref().map(|c| c.site);
                let writer = f.prior.as_ref().map(|c| c.site);
                let name = |s: Option<crate::site::SiteId>| s.and_then(|s| lr.buffer_of_site.get(&s).cloned());
                race.insert("reader_buffer".into(), json!(name(reader)));
                race.insert("writer_buffer".into(), json!(name(writer)));
                race.insert("allocation_id".into(), json!(f.alloc.0));
                race.insert("space".into(), json!(lr.buffers.get(&f.alloc).map(|(_, s)| space_name(*s))));
                race.insert(
                    "overlaps".into(),
                    json!([{"allocation_id": f.alloc.0, "byte_offset": f.bytes.start, "byte_len": f.bytes.end - f.bytes.start, "byte_end": f.bytes.end}]),
                );
            }
            race.insert("legacy_kind".into(), json!(name));
            let msg = match kind {
                AdvisoryKind::CrossCtaAsyncOrder => "async-proxy accesses issued from different CTAs are ordered only by base causality; PTX preserves same-proxy order only within one thread block".to_string(),
                AdvisoryKind::UndeclaredProtocolWord => "a strong load observed an unordered strong write on a word not declared for wait_until; the resulting ordering depends on the schedule".to_string(),
                AdvisoryKind::AliasStaleRead => "stale-name read through pool alias: the read observes bytes last written through another logical buffer".to_string(),
            };
            (ck, Status::Review, msg)
        }
        RK::AsyncLifetime { op } => {
            race.insert("legacy_kind".into(), json!("data_race"));
            race.insert("ordering_domain".into(), json!("completion"));
            race.insert("ordering_failure".into(), json!("async_lifetime_not_drained"));
            race.insert("async_op".into(), json!(op.0));
            let msg = format!("allocation {} ended while async op {} still had it in its footprint", f.alloc.0, op.0);
            (FindingKind::AsyncLifetime, Status::Error, msg)
        }
        RK::OutOfBounds { size } => {
            race.insert("legacy_kind".into(), json!("oob"));
            race.insert("allocation_size".into(), json!(size));
            let msg = format!("access [{}..{}) outside allocation {} of {} bytes", f.bytes.start, f.bytes.end, f.alloc.0, size);
            (FindingKind::OutOfBounds, Status::Error, msg)
        }
    };
    race.insert("occurrences".into(), json!(f.occurrences));
    if f.severity == Severity::Review {
        debug_assert_eq!(status, Status::Review);
    }
    let mut sites = Vec::new();
    if let Some(p) = &f.prior {
        ev.push(witness_evidence("prior", kernel, p, f.alloc, lr));
        race.insert("prior".into(), witness_attrs(p));
        sites.push(p.site);
    }
    if let Some(c) = &f.current {
        ev.push(witness_evidence("current", kernel, c, f.alloc, lr));
        race.insert("current".into(), witness_attrs(c));
        sites.push(c.site);
    }
    if f.alloc.0 != u32::MAX {
        let (buffer, space) = lr.buffers.get(&f.alloc).map(|(n, s)| (Some(n.clone()), Some(*s))).unwrap_or((None, None));
        ev.push(Evidence {
            role: "overlap".into(),
            kernel,
            site: sites.last().copied().unwrap_or(SiteId(u32::MAX)),
            actor: None,
            buffer,
            space,
            alloc: Some(f.alloc),
            bytes: Some(ByteSpan::new(f.bytes.start, f.bytes.end - f.bytes.start)),
            detail: None,
        });
    }
    sites.extend(ev.iter().filter(|e| e.role == "release" || e.role == "acquire").map(|e| e.site));
    sites.sort();
    sites.dedup();
    let attrs: BTreeMap<String, Value> = race.into_iter().collect();
    Finding { kind, status, message, attrs, sites, evidence: ev }
}

fn convert_incomplete(i: &Incomplete, count: u64, kernel: u32) -> Finding {
    let (reason, extra) = incomplete_reason(i);
    let mut d = Map::new();
    d.insert("occurrences".into(), json!(count));
    d.insert("legacy_kind".into(), json!("analysis_incomplete"));
    d.insert("reason".into(), json!(reason));
    if let Value::Object(m) = extra {
        d.extend(m);
    }
    let kind = match i {
        Incomplete::FindingsTruncated { .. } => FindingKind::BudgetExhausted,
        Incomplete::SignalWriteNotRecorded { .. } => FindingKind::SignalWriteNotRecorded,
        _ => FindingKind::Unsupported,
    };
    let _ = kernel;
    Finding {
        kind,
        status: Status::Incomplete,
        message: format!("racecheck coverage incomplete: {reason}"),
        attrs: d.into_iter().collect(),
        sites: vec![],
        evidence: vec![],
    }
}

/// One `report::Report` per launch.
pub fn reports(obs: &RaceObserver) -> Vec<Report> {
    obs.launches
        .iter()
        .map(|lr| {
            let mut findings: Vec<Finding> = lr.report.findings.iter().map(|f| convert(f, lr)).collect();
            findings.extend(
                lr.report.incomplete.iter().zip(lr.report.incomplete_counts.iter()).map(|(i, n)| convert_incomplete(i, *n, lr.kernel)),
            );
            let mut r = Report::new("racecheck", findings);
            r.launch = lr.launch;
            r.coverage = vec![
                ("access_count".into(), lr.stats.accesses),
                ("gc_runs".into(), lr.stats.gc_runs),
                ("witnesses_retired".into(), lr.stats.witnesses_retired),
                ("async_slots".into(), lr.stats.async_slots),
                ("async_slots_reclaimed".into(), lr.stats.async_slots_reclaimed),
            ];
            r
        })
        .collect()
}

/// All launches in one report (`launch` = the last launch; coverage summed).
pub fn report(obs: &RaceObserver) -> Report {
    let all = reports(obs);
    let mut findings = Vec::new();
    let mut coverage: Vec<(String, u64)> = Vec::new();
    let mut launch = 0;
    for r in all {
        launch = r.launch;
        findings.extend(r.findings);
        for (k, v) in r.coverage {
            match coverage.iter_mut().find(|(x, _)| *x == k) {
                Some((_, s)) => *s += v,
                None => coverage.push((k, v)),
            }
        }
    }
    let mut r = Report::new("racecheck", findings);
    r.launch = launch;
    r.coverage = coverage;
    r
}

fn span_json(alloc: Option<AllocId>, b: Option<ByteSpan>) -> Value {
    match b {
        Some(b) => json!({"allocation_id": alloc.map(|a| a.0), "byte_offset": b.start, "byte_len": b.len, "byte_end": b.end()}),
        None => Value::Null,
    }
}

fn witness_json(e: &Evidence, d: &Value) -> Value {
    json!({
        "operation": {"kernel_index": e.kernel, "site": e.site.0, "global_warp_id": d["global_warp_id"], "epoch": d["epoch"], "async_op": d["async_op"]},
        "lane": d["lane"],
        "access_kind": d["access_kind"],
        "space": e.space.map(space_name),
        "span": span_json(e.alloc, e.bytes),
        "proxy": d["proxy"],
    })
}

/// The legacy native payload (`race_check_python.rs` keys) for a report.
pub fn serialize(r: &Report) -> Value {
    let mut findings = Vec::new();
    let mut advisories = Vec::new();
    let mut incomplete = Vec::new();
    for f in &r.findings {
        let d: Map<String, Value> = f.attrs.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        if f.status == Status::Incomplete {
            let mut m = d.clone();
            m.insert("kind".into(), json!("analysis_incomplete"));
            m.remove("legacy_kind");
            incomplete.push(Value::Object(m));
            continue;
        }
        let legacy = d.get("legacy_kind").and_then(|v| v.as_str()).unwrap_or("data_race").to_string();
        let mut m = Map::new();
        m.insert(
            "status".into(),
            json!(match f.status {
                Status::Error => "error",
                Status::Review => "review",
                Status::Incomplete => "incomplete",
            }),
        );
        m.insert("kind".into(), json!(legacy));
        for (k, v) in &d {
            if !matches!(k.as_str(), "legacy_kind" | "prior" | "current") {
                m.insert(k.clone(), v.clone());
            }
        }
        if let Some(p) = f.evidence.iter().find(|e| e.role == "prior") {
            m.insert("prior".into(), witness_json(p, d.get("prior").unwrap_or(&Value::Null)));
        }
        if let Some(c) = f.evidence.iter().find(|e| e.role == "current") {
            m.insert("current".into(), witness_json(c, d.get("current").unwrap_or(&Value::Null)));
        }
        if let Some(o) = f.evidence.iter().find(|e| e.role == "overlap") {
            m.insert("overlap".into(), span_json(o.alloc, o.bytes));
        }
        m.insert("message".into(), json!(f.message));
        m.insert("sites".into(), json!(f.sites.iter().map(|s| s.0).collect::<Vec<_>>()));
        if matches!(legacy.as_str(), "undeclared_protocol_word" | "cross_cta_async_order" | "alias_stale_read") {
            advisories.push(Value::Object(m));
        } else {
            findings.push(Value::Object(m));
        }
    }
    let cov = |k: &str| r.coverage.iter().find(|(x, _)| x == k).map(|(_, v)| *v).unwrap_or(0);
    let verdict = serde_json::to_value(r.verdict).unwrap_or(Value::Null);
    json!({
        "schema_version": 5,
        "execution_model": "direct_online_vc",
        "checked_memory_spaces": ["global", "shared", "tmem"],
        "launch": r.launch,
        "verdict": verdict,
        "findings": findings,
        "advisories": advisories,
        "incomplete": incomplete,
        "access_count": cov("access_count"),
        "accesses_complete": true,
        "stats": r.coverage.iter().map(|(k, v)| (k.clone(), json!(v))).collect::<Map<String, Value>>(),
    })
}
