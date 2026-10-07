//! Synccheck scenarios translated from
//! `tirx_harness/tests/analysis_tools/synccheck/test_native_synccheck_artifact.py`
//! and `test_native_kernel_contracts.py`, built from contract `SyncEvent`s.
//! Every scenario runs under four configurations that must agree on the verdict.

mod synccheck_support;

use numsim_core::observe::AsyncTarget;
use numsim_core::report::{Status, Verdict};
use numsim_core::sync::{async_group, mbarrier, named, tcgen, ResourceInit, SyncCmd, SyncError, FULL_MASK};
use numsim_core::synccheck::build::*;
use numsim_core::synccheck::explore::{Limits, Options};
use numsim_core::synccheck::{check, serialize, ProjectionMode, SynccheckConfig};
use synccheck_support::*;

fn one() -> ResourceInit {
    cta(1)
}

/// `native_synccheck_clean`: init, arrive, wait in one warp.
#[test]
fn single_warp_init_arrive_wait_is_clean() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1)).cmd(0, 2, mbar(0, 0), arrive(1)).cmd(0, 3, mbar(0, 0), wait(0));
    let r = run_all(&log.build(), one(), Verdict::Clean);
    assert_eq!(stat(&r[0].1, "program_count"), 1);
    assert_eq!(stat(&r[0].1, "certified_program_count"), 1);
    assert_eq!(stat(&r[0].1, "visited_state_count"), 1);
    let p = serialize(&r[0].1);
    assert_eq!(p["search"]["algorithm"], "fixed_sync_state");
    assert_eq!(p["coverage"]["termination"]["kind"], "worklist_exhausted");
    assert_eq!(p["coverage"]["eligible_for_clean"], true);
}

/// `test_public_native_synccheck_wait_before_init_is_exact_error`.
#[test]
fn wait_before_init_is_use_before_init() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), wait(0)).cmd(0, 2, mbar(0, 0), init(1));
    for (name, r) in run_all(&log.build(), one(), Verdict::Error) {
        assert_eq!(kind(&r), "mbarrier_use_before_init", "{name}");
    }
}

/// The same error already surfaced by the engine (Phase A): reported with
/// today's strict kind and effect; Phase B does not run.
#[test]
fn phase_a_failure_is_reported_from_status() {
    let mut log = LogBuilder::new();
    log.failed(0, 1, mbar(0, 0), wait(0), SyncError::Mbarrier(mbarrier::Error::Uninitialized));
    let r = check(&log.build(), &config(one()));
    assert_eq!(r.verdict, Verdict::Error);
    let p = payload(&r, Status::Error);
    assert_eq!(p["kind"], "mbarrier_use_before_init");
    assert_eq!(p["effect"], "mbarrier.wait");
    assert_eq!(p["operation"]["global_warp_id"], 0);
    assert_eq!(stat(&r, "program_count"), 0);
}

/// A warp still blocked when the concrete run ended (`BlockedAtExit`) is the
/// executor deadlock (`execution_error.kind == "deadlock"`).
#[test]
fn phase_a_blocked_at_exit_is_execution_deadlock() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(2)).blocked_at_exit(0, 2, mbar(0, 0), wait(0));
    let r = check(&log.build(), &config(one()));
    assert_eq!(r.verdict, Verdict::Error);
    let p = serialize(&r);
    assert_eq!(p["execution_error"]["kind"], "deadlock");
    assert_eq!(p["execution_error"]["blocked_operations"][0]["warp_id"], 0);
    assert_eq!(p["findings"].as_array().unwrap().len(), 0);
}

/// Init-before-use missing across warps: the reference run inits first, so
/// only the certificate (HB) or the exhaustive search exposes it.
#[test]
fn cross_warp_init_without_publication_is_an_error() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1)).cmd(1, 2, mbar(0, 0), arrive(1)).cmd(0, 3, mbar(0, 0), wait(0));
    let r = run_all(&log.build(), cta(2), Verdict::Error);
    let kinds = r.iter().map(|(n, r)| (*n, kind(r))).collect::<Vec<_>>();
    assert_eq!(
        kinds,
        [
            ("default", "mbarrier_init_not_happens_before_use".to_owned()),
            ("per-resource-dfs", "mbarrier_use_before_init".to_owned()),
            // One mbarrier forms the whole component, so the certificate applies.
            ("components", "mbarrier_init_not_happens_before_use".to_owned()),
            ("whole-plain", "mbarrier_use_before_init".to_owned()),
        ]
    );
    let p = payload(&r[1].1, Status::Error);
    assert_eq!(p["protocol"], "Mbarrier");
    assert!(p["witness_evidence"][0]["description"].as_str().unwrap().starts_with("warp 1 issues"), "{p:#}");
}

/// `test_public_native_synccheck_accepts_cta_sync_as_mbarrier_init_publication`.
#[test]
fn cta_sync_publishes_init() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    cta_sync(&mut log, 0, &[0, 1], 2);
    log.cmd(1, 2, mbar(0, 0), arrive(1)).cmd(0, 3, mbar(0, 0), wait(0));
    run_all(&log.build(), cta(2), Verdict::Clean);
}

/// `test_native_synccheck_trailing_warp_does_not_hide_real_under_arrival`.
#[test]
fn under_arrival_deadlocks_and_exact_count_is_clean() {
    for (expected, verdict) in [(9 * 32, Verdict::Error), (8 * 32, Verdict::Clean)] {
        let mut log = LogBuilder::new();
        log.cmd(0, 1, mbar(0, 0), init(expected));
        cta_sync(&mut log, 0, &(0..9).collect::<Vec<_>>(), 9);
        log.cmd(0, 2, mbar(0, 0), wait(0));
        for w in 1..9 {
            log.cmd(w, 3, mbar(0, 0), arrive(32));
        }
        for (name, r) in run_all(&log.build(), cta(9), verdict) {
            if verdict == Verdict::Error {
                assert_eq!(kind(&r), "deadlock", "{name}");
                assert_eq!(payload(&r, Status::Error)["verification"], "fixed_sync");
            }
        }
    }
}

/// `test_public_native_synccheck_plain_arrival_overflow_is_typed_error`.
#[test]
fn arrival_overflow_is_typed() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1)).cmd(0, 2, mbar(0, 0), arrive(2));
    for (name, r) in run_all(&log.build(), one(), Verdict::Error) {
        assert_eq!(kind(&r), "mbarrier_arrival_overflow", "{name}");
    }
}

/// `test_public_native_synccheck_rejects_unconsumed_generation_reuse`: the
/// producer can lap the consumer (no back-pressure).
#[test]
fn producer_lap_without_back_pressure_is_an_error() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    cta_sync(&mut log, 0, &[0, 1], 2);
    log.cmd(0, 2, mbar(0, 0), arrive(1)).cmd(0, 2, mbar(0, 0), arrive(1));
    log.cmd(1, 3, mbar(0, 0), wait(0)).cmd(1, 3, mbar(0, 0), wait(1));
    let r = run_all(&log.build(), cta(2), Verdict::Error);
    assert_eq!(kind(&r[0].1), "mbarrier_wait_overtaken");
}

/// `native_mbarrier_depth_two_pipeline`.
#[test]
fn depth_two_multi_generation_pipeline_is_clean() {
    let mut log = LogBuilder::new();
    for b in 0..4 {
        log.cmd(0, 1, mbar(0, 8 * b), init(1));
    }
    cta_sync(&mut log, 0, &[0, 1], 2);
    for i in 0..4u32 {
        let slot = i % 2;
        if i >= 2 {
            log.cmd(0, 2, mbar(0, 8 * (2 + slot)), wait(0));
        }
        log.cmd(0, 3, mbar(0, 8 * slot), arrive(1));
    }
    let mut phase = 0;
    for i in 0..4u32 {
        let slot = i % 2;
        log.cmd(1, 4, mbar(0, 8 * slot), wait(phase));
        log.cmd(1, 5, mbar(0, 8 * (2 + slot)), arrive(1));
        if slot == 1 {
            phase ^= 1;
        }
    }
    run_all(&log.build(), cta(2), Verdict::Clean);
}

/// K-stage producer/consumer ring, with and without a TMA producer.
#[test]
fn k_stage_pipeline_is_clean() {
    for tma in [0, 128] {
        let r = run_all(&pipeline(3, 2, 6, tma), cta(3), Verdict::Clean);
        assert_eq!(stat(&r[0].1, "program_count"), 5);
        assert_eq!(stat(&r[0].1, "certified_program_count"), 5, "tma={tma}");
    }
}

/// The consumer waits the wrong parity on its last trip.
#[test]
fn k_stage_pipeline_with_wrong_parity_is_an_error() {
    let mut log = LogBuilder::new();
    let (full, empty) = (|s: u32| mbar(0, 8 * s), |s: u32| mbar(0, 8 * (2 + s)));
    for s in 0..2 {
        log.cmd(0, 1, full(s), init(1)).cmd(0, 2, empty(s), init(1));
    }
    cta_sync(&mut log, 0, &[0, 1], 2);
    for i in 0..4u32 {
        let (s, round) = (i % 2, i / 2);
        if round >= 1 {
            log.cmd(0, 10, empty(s), wait(u64::from((round - 1) & 1)));
        }
        log.cmd(0, 11, full(s), arrive(1));
    }
    for i in 0..4u32 {
        let (s, round) = (i % 2, i / 2);
        let parity = u64::from(round & 1) ^ u64::from(i == 3);
        log.cmd(1, 20, full(s), wait(parity)).cmd(1, 21, empty(s), arrive(1));
    }
    run_all(&log.build(), cta(2), Verdict::Error);
}

/// `cross_protocol_cycle_is_a_deadlock_not_a_false_clean`.
#[test]
fn cross_protocol_cycle_deadlocks() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, named_bar(0, 2), bar_sync(0, 64))
        .cmd(0, 2, cluster_bar(0), cl_arrive(0))
        .cmd(0, 3, cluster_bar(0), cl_wait(0))
        .cmd(1, 2, cluster_bar(0), cl_arrive(1))
        .cmd(1, 3, cluster_bar(0), cl_wait(1))
        .cmd(1, 1, named_bar(0, 2), bar_sync(1, 64));
    for (name, r) in run_all(&log.build(), cta(2), Verdict::Error) {
        assert_eq!(kind(&r), "deadlock", "{name}");
    }
}

/// Repeated `cta_sync` generations are certified (no arrival permutations).
#[test]
fn repeated_cta_sync_is_certified() {
    let mut log = LogBuilder::new();
    let warps = (0..8).collect::<Vec<_>>();
    for _ in 0..16 {
        cta_sync(&mut log, 0, &warps, 8);
    }
    let r = run_all(&log.build(), cta(8), Verdict::Clean);
    assert_eq!(stat(&r[0].1, "visited_state_count"), 1);
    // Without certificates: a strong-diamond chain per generation, two steps
    // (register + resume) per warp: linear, not 8! per generation.
    assert!(stat(&r[1].1, "visited_state_count") <= 16 * 3 * 8 + 2, "{:?}", r[1].1.coverage);
}

/// A warp that `bar.arrive`s twice without synchronizing can contribute
/// twice to one generation in another schedule.
#[test]
fn named_arrive_reuse_without_ordering_is_an_error() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, named_bar(0, 1), bar_arrive(0, 64))
        .cmd(0, 1, named_bar(0, 1), bar_arrive(0, 64))
        .cmd(1, 2, named_bar(0, 1), bar_sync(1, 64))
        .cmd(1, 2, named_bar(0, 1), bar_sync(1, 64));
    let r = run_all(&log.build(), cta(2), Verdict::Error);
    assert_eq!(kind(&r[0].1), "named_barrier_generation_not_ordered");
    assert_eq!(kind(&r[1].1), "named_barrier_duplicate_contribution");
}

/// Named-barrier lane-mask rule: a strict subset of the live lanes fails closed.
#[test]
fn named_partial_warp_is_an_error() {
    let mut log = LogBuilder::new();
    let c = named::Contribution { warp: 0, mask: 1, live: FULL_MASK, count: 32 };
    log.cmd(0, 1, named_bar(0, 1), SyncCmd::Named(named::Cmd::Sync(c)));
    for (name, r) in run_all(&log.build(), one(), Verdict::Error) {
        assert_eq!(kind(&r), "named_barrier_invalid_arrival_count", "{name}");
    }
}

/// Cluster barrier: a participant that never arrives leaves the waiter blocked.
#[test]
fn cluster_missing_participant_deadlocks() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, cluster_bar(0), cl_arrive(0)).cmd(0, 2, cluster_bar(0), cl_wait(0));
    let init = ResourceInit { cluster_warps: 2, ..cta(2) };
    for (name, r) in run_all(&log.build(), init, Verdict::Error) {
        assert_eq!(kind(&r), "deadlock", "{name}");
    }
}

/// `test_public_native_synccheck_transaction_over_delivery_is_exact_error`.
#[test]
fn transaction_over_delivery_is_an_error() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1)).cmd(0, 2, mbar(0, 0), arrive_tx(1, 16));
    log.issue(0, 3, mbar(0, 0), 32, 0, Vec::new());
    log.cmd(0, 4, mbar(0, 0), wait(0));
    let r = run_all(&log.build(), one(), Verdict::Error);
    assert_eq!(kind(&r[1].1), "mbarrier_transaction_over_delivery");
}

/// `test_native_synccheck_reduces_tma_completion_waiter_interleavings`.
#[test]
fn tma_completion_many_waiters_stays_small() {
    let warps = 16u32;
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    cta_sync(&mut log, 0, &(0..warps).collect::<Vec<_>>(), warps);
    log.cmd(0, 2, mbar(0, 0), arrive_tx(1, 1024));
    log.issue(0, 3, mbar(0, 0), 1024, 0, Vec::new());
    for w in 0..warps {
        log.cmd(w, 4, mbar(0, 0), wait(0));
    }
    let log = log.build();
    let cfg = SynccheckConfig { certificates: false, state_budget: 256, ..config(cta(warps)) };
    let r = check(&log, &cfg);
    assert_eq!(r.verdict, Verdict::Clean, "{:#}", serialize(&r));
    // Python pins <= 32 states for the TMA barrier with the named barrier
    // certified; here both projections are searched (the cta_sync projection
    // is a register + resume chain of about 3 states per warp).
    assert!(stat(&r, "visited_state_count") <= 32 + 3 * u64::from(warps) + 4, "{:?}", r.coverage);
    let certified = check(&log, &config(cta(warps)));
    assert_eq!(certified.verdict, Verdict::Clean);
    assert_eq!(stat(&certified, "visited_state_count"), 2);
}

/// Budget exhaustion is incomplete with `resource_limit` coverage.
#[test]
fn budget_exhaustion_is_incomplete() {
    let log = pipeline(6, 2, 8, 0);
    let tight = SynccheckConfig {
        mode: ProjectionMode::Whole,
        certificates: false,
        explore: Options::NONE,
        state_budget: 500,
        ..config(cta(6))
    };
    let r = check(&log, &tight);
    assert_eq!(r.verdict, Verdict::Incomplete);
    let p = serialize(&r);
    assert_eq!(p["incomplete"][0]["reason"], "resource_limit");
    assert_eq!(p["incomplete"][0]["resource"], "fixed_sync_states");
    assert_eq!(p["coverage"]["termination"]["kind"], "resource_limit");
    assert_eq!(p["coverage"]["termination"]["resource_limit"]["resource"], "backtrack_nodes");
    let fits = check(&log, &SynccheckConfig { state_budget: 500, ..config(cta(6)) });
    assert_eq!(fits.verdict, Verdict::Clean);
}

/// Isomorphic stage projections are searched once.
#[test]
fn fingerprint_reuses_isomorphic_stage_projections() {
    let r = check(&pipeline(4, 4, 16, 0), &SynccheckConfig { certificates: false, ..config(cta(4)) });
    assert_eq!(r.verdict, Verdict::Clean);
    assert_eq!(stat(&r, "program_count"), 9);
    assert_eq!(stat(&r, "reused_clean_program_count"), 6);
}

/// A collective recorded by only one participant cannot form a program.
#[test]
fn malformed_collective_is_program_build_incomplete() {
    let mut log = LogBuilder::new();
    let c = numsim_core::observe::Collective { id: 7, participants: vec![numsim_core::observe::WarpId(0), numsim_core::observe::WarpId(1)] };
    log.event(0, 1, vec![(reg_pool(0), setmax(0, false, 104))], Vec::new(), None, Some(c), numsim_core::observe::ProtocolStatus::Committed);
    let r = check(&log.build(), &config(cta(8)));
    assert_eq!(r.verdict, Verdict::Incomplete);
    assert_eq!(payload(&r, Status::Incomplete)["reason"], "fixed_sync_program_build");
}

// ---- protocols added in the integration (setmaxnreg, tcgen, async groups,
// ---- inval/re-init, .noinc, conditional waits) ----

const WG: [[u32; 4]; 3] = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]];

/// setmaxnreg: an increase waits for a later decrease of another warpgroup
/// (pool grant transition).
#[test]
fn setmaxnreg_increase_is_granted_by_a_later_decrease() {
    let mut log = LogBuilder::new();
    log.collective(&WG[0], 1, vec![(reg_pool(0), setmax(0, true, 232))]);
    log.collective(&WG[1], 2, vec![(reg_pool(0), setmax(1, false, 104))]);
    run_all(&log.build(), cta(12), Verdict::Clean);
}

#[test]
fn setmaxnreg_increase_without_release_deadlocks() {
    let mut log = LogBuilder::new();
    log.collective(&WG[0], 1, vec![(reg_pool(0), setmax(0, true, 232))]);
    for (name, r) in run_all(&log.build(), cta(12), Verdict::Error) {
        assert_eq!(kind(&r), "deadlock", "{name}");
    }
}

/// `test_native_synccheck_does_not_drop_setmaxnreg_inside_loop`.
#[test]
fn setmaxnreg_without_warpgroup_sync_is_an_error() {
    let mut log = LogBuilder::new();
    log.collective(&WG[0], 1, vec![(reg_pool(0), setmax(0, false, 88))]);
    log.collective(&WG[0], 1, vec![(reg_pool(0), setmax(0, false, 88))]);
    for (name, r) in run_all(&log.build(), cta(12), Verdict::Error) {
        assert_eq!(kind(&r), "setmaxnreg_missing_warpgroup_sync", "{name}");
    }
}

/// TMEM lifecycle: single owner alloc/dealloc/relinquish is clean.
#[test]
fn tmem_lifecycle_is_clean() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, tmem(0), tmem_alloc(64)).cmd(0, 2, tmem(0), tmem_dealloc(0, 64)).cmd(0, 3, tmem(0), tmem_relinquish());
    run_all(&log.build(), one(), Verdict::Clean);
}

/// `test_fixed_sync_state_finds_bad_tcgen_order_after_clean_native_execution`:
/// two warps allocate concurrently; the recorded dealloc addresses hold only
/// in the observed order.
#[test]
fn tmem_order_dependent_address_is_an_error() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, tmem(0), tmem_alloc(256)).cmd(0, 2, tmem(0), tmem_dealloc(0, 256));
    log.cmd(1, 1, tmem(0), tmem_alloc(256)).cmd(1, 2, tmem(0), tmem_dealloc(256, 256));
    for (name, r) in run_all(&log.build(), cta(2), Verdict::Error) {
        assert_eq!(kind(&r), "tcgen_deallocation_mismatch", "{name}");
        let p = payload(&r, Status::Error);
        assert_eq!(p["protocol"], "TcgenLifecycle", "{name}");
    }
}

/// Live TMEM at exit.
#[test]
fn tmem_leak_is_an_error() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, tmem(0), tmem_alloc(32));
    for (name, r) in run_all(&log.build(), one(), Verdict::Error) {
        assert_eq!(kind(&r), "tcgen_live_allocations_at_exit", "{name}");
    }
}

/// tcgen05.commit -> deferred mbarrier arrive-on.
#[test]
fn tcgen_commit_arrives_on_mbarrier() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    cta_sync(&mut log, 0, &[0, 1], 2);
    log.cmd(0, 2, tcgen_work(0, 0), work(tcgen::WorkCmd::Issue));
    log.issue(0, 3, mbar(0, 0), 0, 1, vec![(tcgen_work(0, 0), work(tcgen::WorkCmd::Commit))]);
    log.cmd(1, 4, mbar(0, 0), wait(0));
    run_all(&log.build(), cta(2), Verdict::Clean);
}

/// cp.async group: issue, commit, wait_group 0 (milestone transitions).
#[test]
fn cp_async_group_is_clean() {
    let g = async_group_res(0, 0, async_group::Domain::CpAsync);
    let mut log = LogBuilder::new();
    log.cmd(0, 1, g, group(async_group::Cmd::Issue))
        .cmd(0, 2, g, group(async_group::Cmd::Commit))
        .cmd(0, 3, g, group(async_group::Cmd::Wait { n: 0, read: false }));
    run_all(&log.build(), one(), Verdict::Clean);
}

/// `cp.async.mbarrier.arrive.noinc`: the deferred arrival is released by the
/// group's full completion.
#[test]
fn cp_async_mbarrier_arrive_noinc_is_clean() {
    let g = async_group_res(0, 0, async_group::Domain::CpAsync);
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    cta_sync(&mut log, 0, &[0, 1], 2);
    log.cmd(0, 2, g, group(async_group::Cmd::Issue));
    log.issue(0, 3, mbar(0, 0), 0, 1, vec![(g, group(async_group::Cmd::ArriveOn))]);
    log.cmd(1, 4, mbar(0, 0), wait(0));
    run_all(&log.build(), cta(2), Verdict::Clean);
}

/// Without `.noinc` the instruction also raises the pending count, so the
/// phase needs the extra (deferred) arrival on top of the regular one.
#[test]
fn cp_async_mbarrier_arrive_inc_needs_both_arrivals() {
    let g = async_group_res(0, 0, async_group::Domain::CpAsync);
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    log.cmd(0, 2, g, group(async_group::Cmd::Issue));
    log.issue(0, 3, mbar(0, 0), 0, 1, vec![(mbar(0, 0), inc_pending(1)), (g, group(async_group::Cmd::ArriveOn))]);
    log.cmd(0, 4, mbar(0, 0), arrive(1)).cmd(0, 5, mbar(0, 0), wait(0));
    run_all(&log.build(), one(), Verdict::Clean);
}

/// inval + re-init in one warp.
#[test]
fn inval_and_reinit_is_clean() {
    let mut log = LogBuilder::new();
    for site in [0, 10] {
        log.cmd(0, site + 1, mbar(0, 0), init(1))
            .cmd(0, site + 2, mbar(0, 0), arrive(1))
            .cmd(0, site + 3, mbar(0, 0), wait(0))
            .cmd(0, site + 4, mbar(0, 0), inval());
    }
    run_all(&log.build(), one(), Verdict::Clean);
}

#[test]
fn reinit_without_inval_is_an_error() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1)).cmd(0, 2, mbar(0, 0), init(1));
    for (name, r) in run_all(&log.build(), one(), Verdict::Error) {
        assert_eq!(kind(&r), "mbarrier_reinit_without_inval", "{name}");
    }
}

/// A recorded successful `try_wait` (conditional) only admits schedules
/// where it succeeds.
#[test]
fn conditional_try_wait_success_is_clean() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    cta_sync(&mut log, 0, &[0, 1], 2);
    log.cmd(0, 2, mbar(0, 0), arrive(1));
    log.test_ok(1, 3, mbar(0, 0), 0);
    run_all(&log.build(), cta(2), Verdict::Clean);
}

/// Failed polls (`observed_parity: None`) are not program positions.
#[test]
fn failed_polls_are_dropped() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    cta_sync(&mut log, 0, &[0, 1], 2);
    log.cmd(0, 2, mbar(0, 0), arrive(1));
    log.cmd(1, 3, mbar(0, 0), test_parity(0)).cmd(1, 3, mbar(0, 0), test_parity(0));
    log.test_ok(1, 3, mbar(0, 0), 0);
    run_all(&log.build(), cta(2), Verdict::Clean);
}

/// Lane-varying multi-barrier wait = one atomic event with two targets.
#[test]
fn multi_target_wait_is_one_command() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1)).cmd(0, 1, mbar(0, 8), init(1));
    cta_sync(&mut log, 0, &[0, 1], 2);
    log.cmd(0, 2, mbar(0, 0), arrive(1)).cmd(0, 3, mbar(0, 8), arrive(1));
    log.cmds(1, 4, vec![(mbar(0, 0), wait(0)), (mbar(0, 8), wait(0))]);
    let r = run_all(&log.build(), cta(2), Verdict::Clean);
    // The two barriers form one projection (plus cta_sync).
    assert_eq!(stat(&r[0].1, "program_count"), 2);
}

/// Issued target without an explicit `Issue` command gets one synthesized.
#[test]
fn issued_target_synthesizes_issue() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1)).cmd(0, 2, mbar(0, 0), arrive_tx(1, 64));
    log.event(0, 3, Vec::new(), vec![AsyncTarget { res: mbar(0, 0), bytes: 64, arrivals: 0 }], None, None, numsim_core::observe::ProtocolStatus::Committed);
    log.cmd(0, 4, mbar(0, 0), wait(0));
    run_all(&log.build(), one(), Verdict::Clean);
}

/// Limits: explicit per-projection transition budget.
#[test]
fn transition_budget_is_incomplete() {
    let cfg = SynccheckConfig {
        mode: ProjectionMode::Whole,
        certificates: false,
        explore: Options::NONE,
        transition_budget: 10,
        ..config(cta(4))
    };
    let r = check(&pipeline(4, 2, 4, 0), &cfg);
    assert_eq!(r.verdict, Verdict::Incomplete);
    assert_eq!(payload(&r, Status::Incomplete)["resource"], "fixed_sync_transitions");
    let _ = Limits::default();
}

/// A dangling `bar.arrive` at exit is a review lint (W3 `exit_lint`), not an error.
#[test]
fn dangling_named_arrive_is_review() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, named_bar(0, 1), bar_arrive(0, 64));
    let r = run_all(&log.build(), cta(2), Verdict::Review);
    let p = serialize(&r[0].1);
    assert_eq!(p["review"][0]["kind"], "sync_exit_lint");
    assert_eq!(p["findings"], serde_json::json!([]));
}
