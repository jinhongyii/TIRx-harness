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
    // The certificate no longer reports `mbarrier_wait_overtaken`: an
    // overtaken wait is decided by the search.
    for (name, r) in run_all(&log.build(), cta(2), Verdict::Error) {
        assert_ne!(kind(&r), "mbarrier_wait_overtaken", "{name}");
    }
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
    let c = named::Contribution { warp: 0, mask: 1, live: FULL_MASK, count: 32, aligned: true };
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
        assert_eq!(kind(&r), "setmaxnreg_pool_deadlock", "{name}");
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
        // Either the swapped allocation result itself (legacy "allocation
        // result changed") or its consequence, the dealloc mismatch.
        let k = kind(&r);
        assert!(k == "tcgen_allocation_result_changed" || k == "tcgen_deallocation_mismatch", "{name}: {k}");
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


/// Kernel-wide tcgen05 `.cta_group` (W3-3): one group per kernel, checked
/// once over the program; explicit `TcgenGroup` commands are stripped.
#[test]
fn tcgen_cta_group_mismatch_is_an_error() {
    let mut log = LogBuilder::new();
    log.cmds(0, 1, vec![(numsim_core::sync::ResourceId::TcgenKernel, SyncCmd::TcgenGroup(1)), (tmem(0), tmem_alloc(32))]);
    log.cmd(0, 2, tmem(0), tmem_dealloc(0, 32));
    log.cmd(1, 3, tmem(0), SyncCmd::Tcgen(tcgen::Cmd::Relinquish { who: tcgen::Who::Pair }));
    for (name, r) in run_all(&log.build(), cta(2), Verdict::Error) {
        assert_eq!(kind(&r), "tcgen_cta_group_mismatch", "{name}");
        assert_eq!(payload(&r, Status::Error)["protocol"], "TcgenLifecycle");
    }
}

/// `SyncEvent::kernel` becomes `Report::launch`, `Evidence::kernel` and the
/// payload's `operation.kernel_index`; the legacy entry is `Finding::attrs`.
#[test]
fn kernel_index_and_attrs_flow_into_the_report() {
    let mut log = LogBuilder::new();
    log.kernel = 3;
    log.cmd(0, 1, mbar(0, 0), wait(0)).cmd(0, 2, mbar(0, 0), init(1));
    let r = check(&log.build(), &config(one()));
    assert_eq!(r.launch, 3);
    let f = &r.findings[0];
    assert_eq!(f.attrs["kind"], "fixed_sync_protocol_error");
    assert!(f.evidence.iter().all(|e| e.kernel == 3 && e.role != "payload"));
    assert_eq!(serialize(&r)["findings"][0]["operation"]["kernel_index"], 3);
    assert_eq!(r.meta["algorithm"], "fixed_sync_state");
    assert_eq!(r.meta["termination"]["kind"], "finding");
}


// ---- regressions for docs/development/checker-review.md ----

/// S1(a): a parity-1 wait that may pass before generation 0 completes.
/// With the waiter as warp 0 the reference run lets it pass vacuously; the
/// certificate must decline, and the search finds the deadlock where the
/// arrive comes first.
#[test]
fn s1a_vacuous_parity_one_wait_is_not_certified() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    cta_sync(&mut log, 0, &[0, 1], 2);
    log.cmd(0, 2, mbar(0, 0), wait(1));
    log.cmd(1, 3, mbar(0, 0), arrive(1));
    for (name, r) in run_all(&log.build(), cta(2), Verdict::Error) {
        assert_eq!(kind(&r), "deadlock", "{name}");
    }
}

/// S1(b): the reviewer's sequence. W's `wait M(0)` is meant for generation 2
/// but can pass on generation 0, so the reference clocks gate other
/// projections on an edge that a legal schedule does not have.
#[test]
fn s1b_wait_that_can_pass_on_an_older_generation_is_not_certified() {
    let (m, y, e, z) = (mbar(0, 0), mbar(0, 8), mbar(0, 16), mbar(0, 24));
    let (p, c, w) = (0, 1, 2);
    let mut log = LogBuilder::new();
    for b in [m, y, e, z] {
        log.cmd(p, 1, b, init(1));
    }
    cta_sync(&mut log, 0, &[p, c, w], 3);
    log.cmd(p, 2, y, arrive(1)).cmd(p, 3, m, arrive(1)).cmd(p, 4, e, wait(0));
    log.cmd(p, 5, m, arrive(1)).cmd(p, 6, e, wait(1)).cmd(p, 7, m, arrive(1));
    log.cmd(c, 8, y, wait(0)).cmd(c, 9, m, wait(0)).cmd(c, 10, e, arrive(1));
    log.cmd(c, 11, m, wait(1)).cmd(c, 12, e, arrive(1));
    for i in 0..12u32 {
        let parity = u64::from(i & 1);
        log.cmd(w, 13, z, arrive(1)).cmd(w, 14, z, wait(parity));
    }
    log.cmd(w, 15, m, wait(0)).cmd(w, 16, y, arrive(1));
    let log = log.build();
    // The certificate declines; the gated search then leaves the reference
    // generation assignment and fails closed (it used to certify Clean).
    for (name, cfg) in configs(cta(3)) {
        let r = check(&log, &cfg);
        match name {
            "whole-plain" | "components" => {
                assert_eq!(r.verdict, Verdict::Error, "{name}");
                assert_eq!(kind(&r), "mbarrier_arrive_before_consumption", "{name}");
            }
            _ => {
                assert_ne!(r.verdict, Verdict::Clean, "{name}");
                if r.verdict == Verdict::Incomplete {
                    let p = payload(&r, Status::Incomplete);
                    assert!(p["source"].as_str().unwrap().starts_with("generation_assignment_differs"), "{name}: {p:#}");
                }
            }
        }
    }
}

/// S5: a wait that may observe generation 0 or 2 (same parity) in a
/// program that is otherwise clean and confluent. The gated projection must
/// not trust gates built from the reference assignment: it fails closed.
#[test]
fn s5_generation_assignment_is_checked_against_the_reference() {
    let (m, e) = (mbar(0, 0), mbar(0, 8));
    let (p, c, w) = (0, 1, 2);
    let mut log = LogBuilder::new();
    log.cmd(p, 1, m, init(1)).cmd(p, 1, e, init(1));
    cta_sync(&mut log, 0, &[p, c, w], 3);
    log.cmd(p, 2, m, arrive(1)).cmd(p, 3, e, wait(0)).cmd(p, 2, m, arrive(1)).cmd(p, 3, e, wait(1)).cmd(p, 2, m, arrive(1));
    log.cmd(c, 4, m, wait(0)).cmd(c, 5, e, arrive(1)).cmd(c, 4, m, wait(1)).cmd(c, 5, e, arrive(1)).cmd(c, 4, m, wait(0));
    log.cmd(w, 6, m, wait(0));
    let log = log.build();
    let exhaustive = check(&log, &SynccheckConfig { mode: ProjectionMode::Whole, certificates: false, explore: Options::NONE, ..config(cta(3)) });
    assert_eq!(exhaustive.verdict, Verdict::Clean, "{:#}", serialize(&exhaustive));
    for certificates in [false, true] {
        let r = check(&log, &SynccheckConfig { certificates, ..config(cta(3)) });
        assert_eq!(r.verdict, Verdict::Incomplete, "{:#}", serialize(&r));
        assert!(payload(&r, Status::Incomplete)["source"].as_str().unwrap().starts_with("generation_assignment_differs"));
    }
}

/// S8 end to end (the discriminating regression is the explorer unit test
/// `one_step_diamonds_need_the_independence_proof`). After generation 0
/// completes, the lapped waiter's `wait(0)` (warp 0, lowest command ids)
/// commutes in one step with each of the two generation-1 arrivals, which
/// are HB-ordered after the early waiter's consumption. Together the two
/// arrivals complete generation 1 and disable `wait(0)` forever.
#[test]
fn s8_strong_diamond_does_not_hide_a_two_step_lap() {
    let m = mbar(0, 0);
    let mut log = LogBuilder::new();
    // Warp 0 initializes, publishes with a non-blocking `bar.arrive` and
    // immediately waits: it cannot lag before its wait.
    log.cmd(0, 1, m, init(2)).cmd(0, 7, named_bar(0, 0), bar_arrive(0, 192)).cmd(0, 2, m, wait(0));
    for w in 1..6 {
        log.cmd(w, 8, named_bar(0, 0), bar_sync(w, 192));
    }
    log.cmd(1, 3, m, wait(0)).cmd(1, 4, named_bar(0, 1), bar_sync(1, 96)); // consumes, then releases the next producers
    log.cmd(2, 5, m, arrive(1));
    log.cmd(3, 5, m, arrive(1));
    log.cmd(4, 4, named_bar(0, 1), bar_sync(4, 96)).cmd(4, 6, m, arrive(1));
    log.cmd(5, 4, named_bar(0, 1), bar_sync(5, 96)).cmd(5, 6, m, arrive(1));
    let log = log.build();
    let init = cta(6);
    let exhaustive = check(&log, &SynccheckConfig { mode: ProjectionMode::Whole, certificates: false, explore: Options::NONE, ..config(init) });
    assert_eq!(exhaustive.verdict, Verdict::Error, "{:#}", serialize(&exhaustive));
    for mode in [ProjectionMode::Whole, ProjectionMode::Components, ProjectionMode::PerResource] {
        for certificates in [false, true] {
            let reduced = check(&log, &SynccheckConfig { mode, certificates, ..config(init) });
            assert_ne!(reduced.verdict, Verdict::Clean, "{mode:?} certificates={certificates}: {:#}", serialize(&reduced));
        }
    }
}

/// F4: tcgen05.commit no longer couples its barriers and the MMA work queue
/// into one uncertifiable projection. A UMMA ring (TMA producer -> MMA warp
/// committing to empty[s] and tmem_full -> epilogue) is certified.
#[test]
fn f4_umma_ring_is_certified_per_barrier() {
    let r = check(&umma_ring(6, 16, 16), &config(cta(6)));
    assert_eq!(r.verdict, Verdict::Clean, "{:#}", serialize(&r));
    assert_eq!(stat(&r, "certified_program_count"), stat(&r, "program_count"));
    assert!(stat(&r, "visited_state_count") < 1_000, "{:?}", r.coverage);
}

/// A TMA event as the contract delivers it (only an issued target, no
/// `Mbarrier(Issue)` command) is certified.
#[test]
fn tma_event_without_issue_command_is_certified() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    cta_sync(&mut log, 0, &[0, 1], 2);
    log.cmd(0, 2, mbar(0, 0), arrive_tx(1, 64));
    log.event(0, 3, Vec::new(), vec![AsyncTarget { res: mbar(0, 0), bytes: 64, arrivals: 0 }], None, None, numsim_core::observe::ProtocolStatus::Committed);
    log.cmd(1, 4, mbar(0, 0), wait(0));
    let r = check(&log.build(), &config(cta(2)));
    assert_eq!(r.verdict, Verdict::Clean);
    assert_eq!(stat(&r, "certified_program_count"), stat(&r, "program_count"));
}

/// Contract: one Report per launch; launches are never merged.
#[test]
fn launches_are_checked_separately() {
    let mut log = LogBuilder::new();
    log.kernel = 0;
    log.cmd(0, 1, mbar(0, 0), init(1)).cmd(0, 2, mbar(0, 0), arrive(1)).cmd(0, 3, mbar(0, 0), wait(0));
    log.kernel = 1;
    log.restart_seq();
    log.cmd(0, 1, mbar(0, 0), wait(0)).cmd(0, 2, mbar(0, 0), init(1));
    let all = log.build();
    let reports = numsim_core::synccheck::check_launches(&all, &config(one()));
    assert_eq!(reports.iter().map(|r| (r.launch, r.verdict)).collect::<Vec<_>>(), [(0, Verdict::Clean), (1, Verdict::Error)]);
    // `check` on a mixed log fails closed instead of merging.
    let mixed = check(&all, &config(one()));
    assert_eq!(mixed.verdict, Verdict::Incomplete);
    assert_eq!(payload(&mixed, Status::Incomplete)["reason"], "fixed_sync_program_build");
}

/// V2C-15 (`fast_topk_clusters`): every warp exits after its last cluster
/// wait (`Cluster::Exit`, delta C1). The certificate accepts exits that are
/// a warp's last cluster command; the launch is certified, not searched.
#[test]
fn cluster_exit_after_last_wait_is_certified() {
    let warps: Vec<u32> = (0..8).collect();
    let mut log = LogBuilder::new();
    for _ in 0..3 {
        for &w in &warps {
            log.cmd(w, 1, cluster_bar(0), cl_arrive(w));
        }
        for &w in &warps {
            log.cmd(w, 2, cluster_bar(0), cl_wait(w));
        }
    }
    for &w in &warps {
        log.cmd(w, 3, cluster_bar(0), SyncCmd::Cluster(numsim_core::sync::cluster::Cmd::Exit { warp: w, lanes: FULL_MASK }));
    }
    let init = ResourceInit { cluster_warps: 8, ..cta(4) };
    let r = run_all(&log.build(), init, Verdict::Clean);
    assert_eq!(stat(&r[0].1, "certified_program_count"), 1, "{:?}", r[0].1.coverage);
}


/// V2C-23: a launch that stopped early (budget) is `incomplete`
/// (`truncated_launch`), not a deadlock from the blocked warp's missing
/// partner; a `Deadlocked` end of a completed launch stays a deadlock.
#[test]
fn truncated_launch_is_incomplete_not_deadlock() {
    use numsim_core::observe::{Observer, WarpEnd, WarpId};
    let build = |end1: WarpEnd, ended: bool| {
        let mut log = LogBuilder::new();
        log.cmd(0, 1, mbar(0, 0), init(1));
        cta_sync(&mut log, 0, &[0, 1], 2);
        log.blocked_at_exit(0, 2, mbar(0, 0), wait(0));
        let mut rec = log.build();
        rec.launches.push((0, numsim_core::program::LaunchShape { grid: [1, 1, 1], cluster: [1, 1, 1], block: [64, 1, 1], smem_bytes: 0 }));
        rec.warp_done(WarpId(0), WarpEnd::Deadlocked);
        rec.warp_done(WarpId(1), end1);
        if ended {
            rec.launches_ended = 1;
        }
        rec
    };
    for (end1, ended) in [(WarpEnd::Budget, true), (WarpEnd::Exited, false), (WarpEnd::Error, true)] {
        let r = check(&build(end1, ended), &SynccheckConfig::default());
        assert_eq!(r.verdict, Verdict::Incomplete, "{end1:?} {ended}: {:#}", serialize(&r));
        assert_eq!(payload(&r, Status::Incomplete)["reason"], "truncated_launch");
    }
    let r = check(&build(WarpEnd::Exited, true), &SynccheckConfig::default());
    assert_eq!(r.verdict, Verdict::Error);
    assert_eq!(serialize(&r)["execution_error"]["kind"], "deadlock");
}

// ---- round-2 conformance (V2C-4 / V2C-28 / V2C-31) ----

/// V2C-31: the wall-clock budget cuts the search (checked every 256 states)
/// and reports `incomplete` with resource `wall_time`.
#[test]
fn wall_time_limit_is_incomplete() {
    let log = pipeline(6, 2, 8, 0);
    let cfg = SynccheckConfig {
        mode: ProjectionMode::Whole,
        certificates: false,
        explore: Options::NONE,
        limits: numsim_core::synccheck::EchoLimits { max_wall_time_ms: 0, ..Default::default() },
        ..config(cta(6))
    };
    let r = check(&log, &cfg);
    assert_eq!(r.verdict, Verdict::Incomplete, "{:#}", serialize(&r));
    let p = payload(&r, Status::Incomplete);
    assert_eq!(p["reason"], "resource_limit");
    assert_eq!(p["resource"], "wall_time");
}

/// V2C-31: 32 per-lane `cp.async.mbarrier.arrive.noinc` arrivals of one
/// instruction are interchangeable; landing them in every order is 2^32
/// states, the symmetry reduction keeps it linear.
#[test]
fn per_lane_cp_async_arrivals_stay_small() {
    let waiters = 2u32;
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(32));
    log.cmd(0, 1, mbar(0, 8), init(32));
    cta_sync(&mut log, 0, &(0..=waiters).collect::<Vec<_>>(), waiters + 1);
    let groups = (0..32u8).map(|l| async_group_res(0, l, async_group::Domain::CpAsync)).collect::<Vec<_>>();
    for (site, bar) in [(2, mbar(0, 0)), (4, mbar(0, 8))] {
        log.cmds(0, site, groups.iter().map(|&g| (g, group(async_group::Cmd::Issue))).collect());
        let targets = (0..32).map(|_| AsyncTarget { res: bar, bytes: 0, arrivals: 1 }).collect();
        log.event(0, site + 1, groups.iter().map(|&g| (g, group(async_group::Cmd::ArriveOn))).collect(), targets, None, None, numsim_core::observe::ProtocolStatus::Committed);
    }
    for w in 1..=waiters {
        log.cmd(w, 6, mbar(0, 0), wait(0));
        log.cmd(w, 7, mbar(0, 8), wait(0));
    }
    // Count budgets only (no wall-time limit): deterministic under load.
    // 2,541 states with the symmetry rule; the 20k limit without it.
    let cfg = SynccheckConfig { certificates: false, state_budget: 20_000, ..config(cta(waiters + 1)) };
    assert_eq!(cfg.limits.max_wall_time_ms, u64::MAX);
    let r = check(&log.build(), &cfg);
    assert_eq!(r.verdict, Verdict::Clean, "{:#}", serialize(&r));
    assert!(stat(&r, "visited_state_count") < 5_000, "{:?}", r.coverage);
}

/// V2C-4: an engine may list only the recording warp in each record of a
/// collective (cta_group::2 tcgen05 pairs); the members are the union.
#[test]
fn collective_members_are_the_union_of_records() {
    use numsim_core::observe::{Collective, ProtocolStatus, WarpId};
    let mut log = LogBuilder::new();
    // The collective's commands execute once, for all members.
    log.cmd(0, 1, mbar(0, 0), init(1));
    cta_sync(&mut log, 0, &[0, 1, 2], 3);
    for w in [0u32, 1] {
        let c = Collective { id: 9, participants: vec![WarpId(w)] };
        log.event(w, 2, vec![(mbar(0, 0), arrive(1))], Vec::new(), None, Some(c), ProtocolStatus::Committed);
    }
    log.cmd(2, 3, mbar(0, 0), wait(0));
    run_all(&log.build(), cta(3), Verdict::Clean);
}

/// V2C-28: tcgen05.commit arrivals of one warp land in issue order. After
/// the wait on the later commit's barrier, the earlier commit's barrier has
/// completed in every schedule, so the successful test is not
/// schedule-dependent (no `generation_assignment_differs`).
#[test]
fn tcgen_commits_land_in_issue_order() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    log.cmd(0, 1, mbar(0, 8), init(1));
    cta_sync(&mut log, 0, &[0, 1], 2);
    log.cmd(0, 2, tcgen_work(0, 0), work(tcgen::WorkCmd::Issue));
    log.issue(0, 3, mbar(0, 0), 0, 1, vec![(tcgen_work(0, 0), work(tcgen::WorkCmd::Commit))]);
    log.cmd(0, 4, tcgen_work(0, 0), work(tcgen::WorkCmd::Issue));
    log.issue(0, 5, mbar(0, 8), 0, 1, vec![(tcgen_work(0, 0), work(tcgen::WorkCmd::Commit))]);
    log.cmd(1, 6, mbar(0, 8), wait(0));
    log.test_ok(1, 7, mbar(0, 0), 0);
    let cfg = SynccheckConfig { certificates: false, ..config(cta(2)) };
    let r = check(&log.build(), &cfg);
    assert_eq!(r.verdict, Verdict::Clean, "{:#}", serialize(&r));
}

/// A protocol error that stopped the launch is the finding even when the
/// truncated log cannot be built (a collective whose other members never
/// recorded it): no extra `fixed_sync_program_build` incomplete
/// (`sparse_flashmla_decode_head64`, sync delta B1).
#[test]
fn protocol_error_wins_over_unbuildable_truncated_log() {
    use numsim_core::observe::{Collective, ProtocolStatus, WarpId};
    let mut log = LogBuilder::new();
    log.failed(0, 1, named_bar(0, 1), bar_sync(0, 64), SyncError::Named(named::Error::PartialWarp { mask: 0xffff_fffe, live: FULL_MASK }));
    let c = Collective { id: 3, participants: vec![WarpId(1), WarpId(2)] };
    log.event(1, 2, vec![(mbar(0, 0), init(1))], Vec::new(), None, Some(c), ProtocolStatus::Committed);
    let r = check(&log.build(), &config(cta(3)));
    let p = serialize(&r);
    assert_eq!(r.verdict, Verdict::Error, "{p:#}");
    assert!(p["incomplete"].as_array().is_none_or(|a| a.is_empty()), "{p:#}");
}

/// Bulk (TMA / cp.async.bulk) issues are committed implicitly at exit, even
/// though the engine does not log that `Exit` command: no exit lint
/// (`flash_mla_sparse_fwd`, `bsa_backward_blk128`, `sparse_flashmla_prefill_head64_phase1`).
/// Uncommitted `cp.async` stays the `UncommittedAtExit` review lint (delta A3).
#[test]
fn uncommitted_bulk_issue_at_exit_is_not_a_lint() {
    for (domain, verdict) in [(async_group::Domain::Bulk, Verdict::Clean), (async_group::Domain::CpAsync, Verdict::Review)] {
        let g = async_group_res(0, 0, domain);
        let mut log = LogBuilder::new();
        log.cmd(0, 1, g, group(async_group::Cmd::Issue));
        let r = check(&log.build(), &config(one()));
        assert_eq!(r.verdict, verdict, "{domain:?}: {:#}", serialize(&r));
    }
}

/// W2-18: the `.exclusive` limit is 576 columns on sm_107f (PTX Table 58).
/// The recording has no arch; the width of a committed alloc is static and
/// the engine validated it against the arch, so the explorer accepts what
/// the run committed. An explicit `tcgen_exclusive_max` (sm_100: 512) still
/// rejects it.
#[test]
fn tcgen_exclusive_576_follows_the_arch() {
    let alloc = SyncCmd::Tcgen(tcgen::Cmd::Alloc { who: tcgen::Who::One(0), columns: 576, exclusive: true });
    let dealloc = SyncCmd::Tcgen(tcgen::Cmd::Dealloc { who: tcgen::Who::One(0), taddr: 0, columns: 576, exclusive: true });
    let mut log = LogBuilder::new();
    log.cmd(0, 1, tmem(0), alloc).cmd(0, 2, tmem(0), dealloc).cmd(0, 3, tmem(0), tmem_relinquish());
    let log = log.build();
    run_all(&log, one(), Verdict::Clean);
    let sm107 = check(&log, &SynccheckConfig { tcgen_exclusive_max: Some(576), ..config(one()) });
    assert_eq!(sm107.verdict, Verdict::Clean);
    let sm100 = check(&log, &SynccheckConfig { tcgen_exclusive_max: Some(512), ..config(one()) });
    assert_eq!(sm100.verdict, Verdict::Error, "{:#}", serialize(&sm100));
}

/// V2C-14 (`cudnn_sm100_dense_blockscaled_gemm_persistent_{amax,dsrelu_quant}`):
/// in a 6-warp CTA, warpgroup 0 runs setmaxnreg and the CTA-wide aligned
/// `bar.sync` also credits the trailing warps 4-5 (no warpgroup): that credit
/// is a no-op. A setmaxnreg by warps 4-5 is still `IncompleteWarpgroup`.
#[test]
fn trailing_partial_warpgroup_sync_is_not_an_error() {
    let mut log = LogBuilder::new();
    log.collective(&WG[0], 1, vec![(reg_pool(0), setmax(0, false, 64))]);
    log.cmd(4, 2, reg_pool(0), wg_sync(1));
    run_all(&log.build(), cta(6), Verdict::Clean);
    let mut log = LogBuilder::new();
    log.collective(&[4, 5], 1, vec![(reg_pool(0), setmax(1, false, 64))]);
    for (name, r) in run_all(&log.build(), cta(6), Verdict::Error) {
        assert_eq!(kind(&r), "setmaxnreg_incomplete_warpgroup", "{name}");
    }
}
