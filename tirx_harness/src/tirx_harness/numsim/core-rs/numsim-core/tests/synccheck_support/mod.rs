// Shared helpers for the synccheck_* integration tests (included with `mod`).
#![allow(dead_code)]

use numsim_core::observe::RecordingObserver;
use numsim_core::report::{Report, Status, Verdict};
use numsim_core::sync::ResourceInit;
use numsim_core::synccheck::explore::Options;
use numsim_core::synccheck::{check, serialize, ProjectionMode, SynccheckConfig};
use serde_json::Value;

pub fn config(init: ResourceInit) -> SynccheckConfig {
    SynccheckConfig { init, ..SynccheckConfig::default() }
}

/// The configurations every scenario must agree on.
pub fn configs(init: ResourceInit) -> Vec<(&'static str, SynccheckConfig)> {
    let base = config(init);
    vec![
        ("default", base.clone()),
        ("per-resource-dfs", SynccheckConfig { certificates: false, fingerprints: false, ..base.clone() }),
        ("components", SynccheckConfig { mode: ProjectionMode::Components, ..base.clone() }),
        (
            "whole-plain",
            SynccheckConfig {
                mode: ProjectionMode::Whole,
                certificates: false,
                fingerprints: false,
                explore: Options::NONE,
                ..base
            },
        ),
    ]
}

pub fn run_all(log: &RecordingObserver, init: ResourceInit, verdict: Verdict) -> Vec<(&'static str, Report)> {
    let out = configs(init).into_iter().map(|(n, c)| (n, check(log, &c))).collect::<Vec<_>>();
    for (name, report) in &out {
        assert_eq!(report.verdict, verdict, "{name}: {:#}", serialize(report));
    }
    out
}

/// Legacy payload entry of the first finding with `status`.
pub fn payload(report: &Report, status: Status) -> Value {
    let p = serialize(report);
    let key = if status == Status::Incomplete { "incomplete" } else { "findings" };
    let list = p[key].as_array().cloned().unwrap_or_default();
    list.into_iter().next().or_else(|| (!p["execution_error"].is_null()).then(|| p["execution_error"].clone())).unwrap_or(Value::Null)
}

/// Payload `kind`, or `source_kind` for `fixed_sync_protocol_error`.
pub fn kind(report: &Report) -> String {
    let p = payload(report, Status::Error);
    if p["kind"] == "fixed_sync_protocol_error" {
        p["source_kind"].as_str().unwrap_or("").to_owned()
    } else {
        p["kind"].as_str().unwrap_or("").to_owned()
    }
}

pub fn stat(report: &Report, key: &str) -> u64 {
    report.coverage.iter().find(|(k, _)| k == key).map_or(0, |(_, v)| *v)
}

pub fn cta(warps_per_cta: u32) -> ResourceInit {
    ResourceInit { warps_per_cta, cluster_warps: warps_per_cta, ..ResourceInit::default() }
}
