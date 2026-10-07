//! `synccheck::serialize` keeps the payload keys the Python tests pin
//! (spec section 4).

mod synccheck_support;

use numsim_core::report::Verdict;
use numsim_core::synccheck::build::*;
use numsim_core::synccheck::{check, serialize};
use synccheck_support::*;

#[test]
fn clean_payload_has_pinned_keys() {
    let r = check(&pipeline(3, 2, 4, 0), &config(cta(3)));
    assert_eq!(r.verdict, Verdict::Clean);
    assert_eq!(r.tool, "synccheck");
    let p = serialize(&r);
    assert_eq!(p["schema_version"], 3);
    assert_eq!(p["execution_model"], "direct_fixed_sync_state");
    assert_eq!(p["verdict"], "clean");
    assert_eq!(p["findings"], serde_json::json!([]));
    assert_eq!(p["incomplete"], serde_json::json!([]));
    assert!(p["execution_error"].is_null() && p["counterexample"].is_null());
    let s = &p["search"];
    assert_eq!(s["algorithm"], "fixed_sync_state");
    assert_eq!(s["run_count"], 1);
    assert_eq!(s["backtrack_count"], 0);
    assert_eq!(s["sleep_pruned_branch_count"], 0);
    for key in ["program_count", "reused_clean_program_count", "visited_state_count", "explored_transition_count", "strong_diamond_pruned_transition_count"] {
        assert!(s[key].is_u64(), "{key}");
    }
    assert!(s["incomplete_reason"].is_null());
    assert_eq!(s["runs"][0]["status"], "complete");
    let c = &p["coverage"];
    assert_eq!(c["status"], "complete_within_bounds");
    assert_eq!(c["eligible_for_clean"], true);
    assert_eq!(c["resource_usage"]["schedules"], 1);
    for key in ["max_schedules", "max_backtrack_nodes", "max_events_per_run", "max_total_events", "max_loop_steps", "max_wall_time_ms", "max_diagnostic_bytes"] {
        assert!(c["resource_limits"][key].is_u64(), "{key}");
    }
    assert_eq!(c["termination"]["kind"], "worklist_exhausted");
    assert!(c["termination"]["resource_limit"].is_null());
}

#[test]
fn protocol_error_payload_has_witness_and_operation() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1)).cmd(1, 2, mbar(0, 0), arrive(1)).cmd(0, 3, mbar(0, 0), wait(0));
    let cfg = numsim_core::synccheck::SynccheckConfig { certificates: false, ..config(cta(2)) };
    let r = check(&log.build(), &cfg);
    let p = serialize(&r);
    assert_eq!(p["verdict"], "error");
    let f = &p["findings"][0];
    assert_eq!(f["kind"], "fixed_sync_protocol_error");
    assert_eq!(f["protocol"], "Mbarrier");
    assert_eq!(f["source_kind"], "mbarrier_use_before_init");
    assert_eq!(f["operation"]["global_warp_id"], 1);
    assert_eq!(f["operation"]["source_op_id"], 2);
    assert!(f["witness"].as_array().is_some_and(|w| !w.is_empty()));
    assert!(f["witness_evidence"][0]["operation"].is_object());
    assert_eq!(p["coverage"]["termination"]["kind"], "finding");
    assert_eq!(p["search"]["runs"][0]["status"], "finding");
    // report::Finding side.
    assert_eq!(r.findings[0].kind, numsim_core::report::FindingKind::MbarrierMisuse);
    assert!(r.findings[0].sites.contains(&numsim_core::site::SiteId(2)));
}

#[test]
fn report_json_round_trips() {
    let r = check(&pipeline(3, 2, 4, 128), &config(cta(3)));
    let back: numsim_core::report::Report = serde_json::from_str(&r.to_json()).unwrap();
    assert_eq!(serialize(&back), serialize(&r));
}
