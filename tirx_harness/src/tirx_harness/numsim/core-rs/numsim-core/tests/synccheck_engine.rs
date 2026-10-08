//! Synccheck on real interpreter event streams (`testutil::scenarios`,
//! W2 engine integration). Budgets are asserted: per-thread async groups and
//! every other scenario must be cheap, not a product of per-lane states.

use std::time::{Duration, Instant};

use numsim_core::observe::RecordingObserver;
use numsim_core::report::Verdict;
use numsim_core::sched::{self, Backend, RunStatus};
use numsim_core::synccheck::{check, resource_init, serialize, SynccheckConfig};
use numsim_core::testutil::scenarios;

fn coverage(r: &numsim_core::report::Report, key: &str) -> u64 {
    r.coverage.iter().find(|(k, _)| k == key).map_or(0, |(_, v)| *v)
}

fn run(s: &scenarios::Scenario) -> (RunStatus, RecordingObserver) {
    let mut log = RecordingObserver::new();
    let o = sched::run_with_config(&s.module, &s.inputs, &mut log, &Backend::Interp, &s.config).unwrap();
    (o.status, log)
}

/// One warp, 32 per-lane cp.async groups (issue, commit, wait_all): used to
/// take ~16 s at a 20K-state budget and not finish at 1M.
#[test]
fn cp_async_per_lane_groups_are_cheap() {
    let (status, log) = run(&scenarios::cp_async_copy());
    assert_eq!(status, RunStatus::Completed);
    let started = Instant::now();
    let s = scenarios::cp_async_copy();
    let shape = sched::resolve_launch(&s.module.kernels[0], &s.inputs).unwrap();
    let r = check(&log, &SynccheckConfig { init: resource_init(&shape), ..SynccheckConfig::default() });
    assert_eq!(r.verdict, Verdict::Clean, "{:#}", serialize(&r));
    assert!(coverage(&r, "visited_state_count") <= 16, "{:?}", r.coverage);
    assert!(started.elapsed() < Duration::from_secs(2), "{:?}", started.elapsed());
}

/// `setmaxnreg_launch_bounds`: the scheduler's host-side `Configure`
/// (launch-bounds register budget) reaches the explorer.
#[test]
fn launch_bounds_configure_reaches_the_explorer() {
    let s = scenarios::all().into_iter().find(|s| s.name == "setmaxnreg_launch_bounds").expect("scenario");
    let (status, log) = run(&s);
    assert_eq!(status, RunStatus::Completed);
    assert!(log.other.iter().any(|e| format!("{:?}", e.kind).contains("Configure")), "no host Configure event logged");
    // The shape comes from the recording (`RecordingObserver::launches`).
    assert!(!log.launches.is_empty());
    let r = check(&log, &SynccheckConfig::default());
    assert_eq!(r.verdict, Verdict::Clean, "{:#}", serialize(&r));
    // The explicit form gives the same result.
    let shape = sched::resolve_launch(&s.module.kernels[0], &s.inputs).unwrap();
    let explicit = check(&log, &SynccheckConfig { init: resource_init(&shape), ..SynccheckConfig::default() });
    assert_eq!(explicit.verdict, Verdict::Clean);
}

/// Every interpreter scenario's stream is checked within a small budget and
/// never exhausts it; completed runs never yield a synccheck error unless
/// the scenario is a deliberate deadlock.
#[test]
fn every_engine_scenario_is_checked_within_budget() {
    for s in scenarios::all() {
        let (status, log) = run(&s);
        let started = Instant::now();
        let cfg = SynccheckConfig { state_budget: 20_000, transition_budget: 200_000, ..SynccheckConfig::default() };
        let r = check(&log, &cfg);
        let p = serialize(&r);
        assert_ne!(p["coverage"]["termination"]["kind"], "resource_limit", "{}: {p:#}", s.name);
        assert!(started.elapsed() < Duration::from_secs(5), "{}: {:?}", s.name, started.elapsed());
        if status == RunStatus::Completed {
            assert_ne!(r.verdict, Verdict::Error, "{}: {p:#}", s.name);
        }
    }
}

/// No launch shape anywhere: fail closed instead of a zero-warp pool (W2-10).
#[test]
fn unknown_launch_shape_is_incomplete() {
    let s = scenarios::all().into_iter().find(|s| s.name == "setmaxnreg_launch_bounds").expect("scenario");
    let (_, mut log) = run(&s);
    log.launches.clear();
    let r = check(&log, &SynccheckConfig::default());
    assert_eq!(r.verdict, Verdict::Incomplete, "{:#}", serialize(&r));
    assert_eq!(serialize(&r)["incomplete"][0]["reason"], "fixed_sync_program_build");
}
