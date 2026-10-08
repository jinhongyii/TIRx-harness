//! Legacy Python synccheck tests (`tirx_harness/tests/analysis_tools/synccheck/`)
//! that no other Rust test reproduced, ported to contract `SyncEvent` logs.
//!
//! Each test names the legacy `file.py::test_name[param]` it ports. The log is
//! the protocol trace the legacy kernel's concrete run produces (data-dependent
//! branches, lane guards and `elect.sync` already resolved). Where the legacy
//! expectation contradicts a row of `docs/development/sync-behaviour-deltas.md`,
//! the test asserts the new behaviour and cites the row.
//! The coverage map is `scripts/numsim-v2/coverage/synccheck.tsv`.

mod synccheck_support;

use numsim_core::report::{Report, Status, Verdict};
use numsim_core::sync::{async_group, named, tcgen, ResourceInit, SyncCmd, FULL_MASK};
use numsim_core::synccheck::build::*;
use numsim_core::synccheck::{check, serialize};
use synccheck_support::*;

fn one() -> ResourceInit {
    cta(1)
}

/// A named-barrier contribution with an explicit lane mask and form.
fn bar(flavor: fn(named::Contribution) -> named::Cmd, warp_in_cta: u32, count: u64, mask: u32, aligned: bool) -> SyncCmd {
    SyncCmd::Named(flavor(named::Contribution { warp: warp_in_cta, mask, live: FULL_MASK, count, aligned }))
}

/// Full-warp `bar.sync id, count` (aligned) by every warp in `warps` of CTA 0.
fn bar_sync_all(log: &mut LogBuilder, site: u32, id: u8, count: u64, warps: &[u32]) {
    for &w in warps {
        log.cmd(w, site, named_bar(0, id), bar_sync(w, count));
    }
}

/// `barrier.cluster.arrive` + `barrier.cluster.wait` by every warp of cluster 0
/// (cluster-relative warp index == global warp index here).
fn cluster_sync(log: &mut LogBuilder, warps: &[u32]) {
    for &w in warps {
        log.cmd(w, 200, cluster_bar(0), cl_arrive(w)).cmd(w, 201, cluster_bar(0), cl_wait(w));
    }
}

/// `setmaxnreg` executed by the four warps of warpgroup `wg` (one collective).
fn setmax_wg(log: &mut LogBuilder, site: u32, wg: u32, inc: bool, count: u32) {
    let warps = (4 * wg..4 * wg + 4).collect::<Vec<_>>();
    log.collective(&warps, site, vec![(reg_pool(0), setmax(wg, inc, count))]);
}

fn tmem_alloc_on(rank: u8, columns: u32) -> SyncCmd {
    SyncCmd::Tcgen(tcgen::Cmd::Alloc { who: tcgen::Who::One(rank), columns, exclusive: false })
}

fn tmem_dealloc_on(rank: u8, taddr: u32, columns: u32) -> SyncCmd {
    SyncCmd::Tcgen(tcgen::Cmd::Dealloc { who: tcgen::Who::One(rank), taddr, columns, exclusive: false })
}

fn lanes(range: std::ops::Range<u8>, cmd: async_group::Cmd) -> Vec<(numsim_core::sync::ResourceId, SyncCmd)> {
    range.map(|l| (async_group_res(0, l, async_group::Domain::CpAsync), group(cmd))).collect()
}

/// Kinds per configuration, for assertion messages.
fn kinds(r: &[(&'static str, Report)]) -> Vec<(&'static str, String)> {
    r.iter().map(|(n, r)| (*n, kind(r))).collect()
}

/// A missing init publication: the default configuration (causal
/// certificate) reports `mbarrier_init_not_happens_before_use`; the plain
/// searches find the schedule that uses the barrier before its init.
fn assert_init_not_published(r: &[(&'static str, Report)]) {
    assert_eq!(kind(&r[0].1), "mbarrier_init_not_happens_before_use", "{:?}", kinds(r));
    for (name, rep) in r {
        let k = kind(rep);
        assert!(k == "mbarrier_init_not_happens_before_use" || k == "mbarrier_use_before_init", "{name}: {k}");
    }
}

fn assert_kind(r: &[(&'static str, Report)], expected: &str) {
    for (name, rep) in r {
        assert_eq!(kind(rep), expected, "{name}: {:#}", serialize(rep));
    }
}

/// A pool that cannot grant a pending increase: the legacy kernel-contract
/// tests accept either the explorer's `deadlock` or `setmaxnreg_pool_deadlock`.
fn assert_pool_deadlock(r: &[(&'static str, Report)]) {
    for (name, rep) in r {
        let k = kind(rep);
        assert!(k == "deadlock" || k == "setmaxnreg_pool_deadlock", "{name}: {:#}", serialize(rep));
    }
}

// ---------------------------------------------------------------------------
// test_native_synccheck_artifact.py
// ---------------------------------------------------------------------------

/// `test_native_synccheck_artifact.py::test_public_native_synccheck_plain_arrive_before_init_is_exact_error`.
#[test]
fn plain_arrive_before_init_is_use_before_init() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), arrive(1));
    let r = run_all(&log.build(), one(), Verdict::Error);
    assert_kind(&r, "mbarrier_use_before_init");
}

/// `test_native_synccheck_artifact.py::test_public_native_synccheck_arrive_expect_tx_before_init_is_exact_error`.
#[test]
fn arrive_expect_tx_before_init_is_use_before_init() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), arrive_tx(1, 16));
    let r = run_all(&log.build(), one(), Verdict::Error);
    assert_kind(&r, "mbarrier_use_before_init");
}

/// `test_native_synccheck_artifact.py::test_public_native_synccheck_executes_data_dependent_protocol_exactly`:
/// barriers inited with counts 1 and 2; two lanes arrive on `barriers[slot]`.
#[test]
fn data_dependent_slot_arrivals_are_exact() {
    for (slot, verdict) in [(1u32, Verdict::Clean), (0, Verdict::Error)] {
        let mut log = LogBuilder::new();
        log.cmd(0, 1, mbar(0, 0), init(1)).cmd(0, 1, mbar(0, 8), init(2));
        log.cmd(0, 2, mbar(0, 8 * slot), arrive(2)).cmd(0, 3, mbar(0, 8 * slot), wait(0));
        let r = run_all(&log.build(), one(), verdict);
        if verdict == Verdict::Error {
            assert_kind(&r, "mbarrier_arrival_overflow");
        }
    }
}

/// `test_native_synccheck_artifact.py::test_public_native_synccheck_handles_completion_issued_before_future_expectation`:
/// the TMA of generation 1 is issued (and may complete) before that
/// generation's `arrive.expect_tx`.
#[test]
fn completion_issued_before_future_expectation_is_clean() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1)).cmd(0, 2, mbar(0, 0), arrive(1));
    log.issue(0, 3, mbar(0, 0), 16, 0, Vec::new());
    log.cmd(0, 4, mbar(0, 0), wait(0)).cmd(0, 5, mbar(0, 0), arrive_tx(1, 16)).cmd(0, 6, mbar(0, 0), wait(1));
    run_all(&log.build(), one(), Verdict::Clean);
}

/// `test_native_synccheck_artifact.py::test_native_synccheck_tracks_completion_tokens_across_future_generation`.
#[test]
fn tma_completion_tokens_across_two_generations_are_clean() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    for parity in 0..2 {
        log.issue(0, 2, mbar(0, 0), 16, 0, Vec::new());
        log.cmd(0, 3, mbar(0, 0), arrive_tx(1, 16)).cmd(0, 4, mbar(0, 0), wait(parity));
    }
    run_all(&log.build(), one(), Verdict::Clean);
}

/// `test_native_synccheck_artifact.py::test_native_synccheck_accepts_late_wait_ordered_before_next_completion`:
/// warp 1's late `wait(b0, 0)` is ordered (through b2) before the TMA that
/// completes b0's generation 1.
#[test]
fn late_wait_ordered_before_next_completion_is_clean() {
    let (b0, b1, b2) = (mbar(0, 0), mbar(0, 8), mbar(0, 16));
    let mut log = LogBuilder::new();
    for b in [b0, b1, b2] {
        log.cmd(0, 1, b, init(1));
    }
    cta_sync(&mut log, 0, &[0, 1, 2], 3);
    log.cmd(0, 2, b0, arrive(1)).cmd(0, 3, b0, wait(0)).cmd(0, 4, b0, arrive_tx(1, 16)).cmd(0, 5, b1, arrive(1)).cmd(0, 6, b0, wait(1));
    log.cmd(1, 7, b0, wait(0)).cmd(1, 8, b2, arrive(1));
    log.cmd(2, 9, b1, wait(0)).cmd(2, 10, b2, wait(0));
    log.issue(2, 11, b0, 16, 0, Vec::new());
    run_all(&log.build(), cta(3), Verdict::Clean);
}

/// `test_native_synccheck_artifact.py::test_public_native_synccheck_rejects_unconsumed_generation_reuse`
/// (single warp, `consume` = 1 clean, 0 error).
#[test]
fn single_warp_generation_reuse_requires_consumption() {
    for consume in [true, false] {
        let mut log = LogBuilder::new();
        log.cmd(0, 1, mbar(0, 0), init(1)).cmd(0, 2, mbar(0, 0), arrive(1));
        if consume {
            log.cmd(0, 3, mbar(0, 0), wait(0));
        }
        log.cmd(0, 4, mbar(0, 0), arrive(1));
        if consume {
            log.cmd(0, 5, mbar(0, 0), wait(1));
            run_all(&log.build(), one(), Verdict::Clean);
        } else {
            let r = run_all(&log.build(), one(), Verdict::Error);
            assert_kind(&r, "mbarrier_arrive_before_consumption");
        }
    }
}

/// `test_native_synccheck_artifact.py::test_public_native_synccheck_transaction_under_delivery_is_exact_error`
/// (`tma_copy_transaction_mismatch`: expect 68 bytes, the TMA delivers 64).
/// Both the explored program and the concrete run's blocked waiter are a deadlock.
#[test]
fn transaction_under_delivery_deadlocks() {
    let build = |blocked: bool| {
        let mut log = LogBuilder::new();
        log.cmd(0, 1, mbar(0, 0), init(1));
        cta_sync(&mut log, 0, &[0, 1], 2);
        log.issue(0, 2, mbar(0, 0), 64, 0, Vec::new());
        log.cmd(0, 3, mbar(0, 0), arrive_tx(1, 68));
        if blocked {
            log.blocked_at_exit(1, 4, mbar(0, 0), wait(0));
        } else {
            log.cmd(1, 4, mbar(0, 0), wait(0));
        }
        log.build()
    };
    let r = run_all(&build(false), cta(2), Verdict::Error);
    assert_kind(&r, "deadlock");
    let concrete = check(&build(true), &config(cta(2)));
    assert_eq!(concrete.verdict, Verdict::Error);
    assert_eq!(serialize(&concrete)["execution_error"]["kind"], "deadlock");
}

/// `test_native_synccheck_artifact.py::test_public_native_synccheck_transaction_over_delivery_is_exact_error`:
/// the 16-byte TMA is issued before the 8-byte `arrive.expect_tx`.
#[test]
fn completion_before_smaller_expectation_is_over_delivery() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    log.issue(0, 2, mbar(0, 0), 16, 0, Vec::new());
    log.cmd(0, 3, mbar(0, 0), arrive_tx(1, 8)).cmd(0, 4, mbar(0, 0), wait(0));
    let r = run_all(&log.build(), one(), Verdict::Error);
    assert_kind(&r, "mbarrier_transaction_over_delivery");
}

/// `test_native_synccheck_artifact.py::test_public_native_synccheck_executes_one_racy_control_path_without_replay`
/// (`native_synccheck_schedule_sensitive_tma`, concrete path: warp 1 saw
/// `flag == 1` and arrives on the barrier warp 0 inits without publication).
#[test]
fn schedule_sensitive_arrive_without_init_publication_is_an_error() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(2));
    log.issue(0, 2, mbar(0, 0), 16, 0, Vec::new());
    log.cmd(0, 3, mbar(0, 0), arrive_tx(1, 16)).cmd(0, 4, mbar(0, 0), wait(0));
    log.cmd(1, 5, mbar(0, 0), arrive(1));
    let r = run_all(&log.build(), cta(2), Verdict::Error);
    assert_init_not_published(&r);
}

/// `test_native_synccheck_artifact.py::test_synccheck_reports_nonblocking_arrival_successor_error_without_replay`:
/// a non-blocking `bar.arrive` does not publish warp 0's init to warp 1.
#[test]
fn nonblocking_named_arrive_does_not_publish_init() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    log.cmd(1, 2, named_bar(0, 7), bar_arrive(1, 64)).cmd(1, 3, mbar(0, 0), arrive(1)).cmd(1, 4, mbar(0, 0), wait(0));
    log.cmd(2, 2, named_bar(0, 7), bar_arrive(2, 64));
    let r = run_all(&log.build(), cta(3), Verdict::Error);
    assert_init_not_published(&r);
}

/// `test_native_synccheck_artifact.py::test_synccheck_reports_blocking_wait_handoff_successor_error_without_replay`:
/// barrier 1 is inited after the `bar.sync`, so warp 1's arrive on it is unordered.
#[test]
fn blocking_wait_handoff_does_not_publish_successor_init() {
    let (b0, b1) = (mbar(0, 0), mbar(0, 8));
    let mut log = LogBuilder::new();
    log.cmd(0, 1, b0, init(1));
    bar_sync_all(&mut log, 2, 7, 96, &[0, 1, 2]);
    log.cmd(0, 3, b1, init(1));
    log.cmd(1, 4, b0, wait(0)).cmd(1, 5, b1, arrive(1));
    log.cmd(2, 6, b0, arrive(1));
    let r = run_all(&log.build(), cta(3), Verdict::Error);
    assert_init_not_published(&r);
}

/// `test_native_synccheck_artifact.py::test_synccheck_no_waiter_arrival_checks_its_successor_without_replay`.
#[test]
fn no_waiter_arrival_does_not_publish_successor_init() {
    let (b0, b1) = (mbar(0, 0), mbar(0, 8));
    let mut log = LogBuilder::new();
    log.cmd(0, 1, b0, init(1));
    bar_sync_all(&mut log, 2, 7, 96, &[0, 1, 2]);
    log.cmd(0, 3, b1, init(1));
    log.cmd(1, 4, b0, arrive(1)).cmd(1, 5, b0, wait(0)).cmd(1, 6, b1, arrive(1));
    let r = run_all(&log.build(), cta(3), Verdict::Error);
    assert_init_not_published(&r);
}

/// `test_native_synccheck_artifact.py::test_native_synccheck_named_barrier_gateway_reports_execution_contract_error`:
/// `bar.sync 3, 64` and `bar.sync 3, 32` in one generation. Legacy reported
/// the engine's `synchronization_contract_mismatch` execution error; the
/// contract kind is the protocol's `named_barrier_contract_mismatch`.
#[test]
fn named_barrier_count_mismatch_is_contract_mismatch() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, named_bar(0, 3), bar_sync(0, 64)).cmd(1, 2, named_bar(0, 3), bar_sync(1, 32));
    let r = run_all(&log.build(), cta(2), Verdict::Error);
    assert_kind(&r, "named_barrier_contract_mismatch");
}

/// `test_native_synccheck_artifact.py::test_fixed_sync_state_finds_bad_tcgen_order_after_clean_native_execution`
/// (exact kernel shape: 32-column allocations, `cta_sync`s, relinquish). In
/// the swapped allocation order each recorded dealloc frees the other warp's
/// (equal-width) allocation, so every protocol step still succeeds; legacy
/// fixed the allocation results and failed closed ("allocation result
/// changed"). `synccheck_scenarios::tmem_order_dependent_address_is_an_error`
/// keeps the variant without the `cta_sync`s, which the new core rejects.
#[test]
fn order_dependent_tcgen_allocations_with_cta_syncs_are_an_error() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, tmem(0), tmem_alloc(32)).cmd(1, 1, tmem(0), tmem_alloc(32));
    cta_sync(&mut log, 0, &[0, 1], 2);
    log.cmd(0, 2, tmem(0), tmem_dealloc(0, 32)).cmd(1, 2, tmem(0), tmem_dealloc(32, 32));
    cta_sync(&mut log, 0, &[0, 1], 2);
    log.cmd(0, 3, tmem(0), tmem_relinquish());
    let r = run_all(&log.build(), cta(2), Verdict::Error);
    for (name, rep) in &r {
        let p = payload(rep, Status::Error);
        assert_eq!(p["protocol"], "TcgenLifecycle", "{name}");
        assert!(p["source"].as_str().unwrap().contains("allocation result changed"), "{name}: {p:#}");
    }
}

/// `test_native_synccheck_artifact.py::test_public_native_synccheck_accepts_cluster_arrive_through_remote_view`:
/// CTA 1's warp arrives (`.shared::cluster`) on CTA 0's barrier.
#[test]
fn remote_cluster_arrive_is_clean() {
    let init_ = ResourceInit { warps_per_cta: 1, cluster_warps: 2, ..ResourceInit::default() };
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    cluster_sync(&mut log, &[0, 1]);
    log.cmd(0, 2, mbar(0, 0), wait(0));
    log.cmd(1, 3, mbar(0, 0), arrive(1));
    cluster_sync(&mut log, &[0, 1]);
    run_all(&log.build(), init_, Verdict::Clean);
}

/// `test_native_synccheck_artifact.py::test_native_synccheck_deadlock_is_error_even_with_staged_waits`
/// (`mbarrier_missing_arrivals`: count 64, warp 0 arrives with 32 lanes,
/// both warps wait).
#[test]
fn missing_arrivals_with_two_waiters_deadlock() {
    let build = |blocked: bool| {
        let mut log = LogBuilder::new();
        log.cmd(0, 1, mbar(0, 0), init(64));
        cta_sync(&mut log, 0, &[0, 1], 2);
        log.cmd(0, 2, mbar(0, 0), arrive(32));
        for w in 0..2 {
            if blocked {
                log.blocked_at_exit(w, 3, mbar(0, 0), wait(0));
            } else {
                log.cmd(w, 3, mbar(0, 0), wait(0));
            }
        }
        log.build()
    };
    let r = run_all(&build(false), cta(2), Verdict::Error);
    assert_kind(&r, "deadlock");
    let concrete = check(&build(true), &config(cta(2)));
    assert_eq!(serialize(&concrete)["execution_error"]["kind"], "deadlock");
}

// ---------------------------------------------------------------------------
// test_native_kernel_contracts.py
// ---------------------------------------------------------------------------

/// `test_native_kernel_contracts.py::test_native_synccheck_counts_exact_lane_and_elect_participants`
/// [thread-zero, bitwise-data-and-thread-zero, local-leader-relay: one
/// arrival; one-elected-lane-per-warp: one arrival per warp of WG 0, count 4].
#[test]
fn lane_and_elect_participant_counts_are_clean() {
    for (arrivers, count) in [(vec![0u32], 1u64), (vec![1], 1), (vec![0, 1, 2, 3], 4)] {
        let mut log = LogBuilder::new();
        log.cmd(0, 1, mbar(0, 0), init(count));
        cta_sync(&mut log, 0, &(0..8).collect::<Vec<_>>(), 8);
        for &w in &arrivers {
            log.cmd(w, 2, mbar(0, 0), arrive(1));
        }
        log.cmd(4, 3, mbar(0, 0), wait(0));
        run_all(&log.build(), cta(8), Verdict::Clean);
    }
}

/// `test_native_kernel_contracts.py::test_native_synccheck_resolves_data_guard_instead_of_guessing_participation`
/// (`epoch == 1`: nobody arrives).
#[test]
fn data_guard_without_arrival_deadlocks() {
    let build = |blocked: bool| {
        let mut log = LogBuilder::new();
        log.cmd(0, 1, mbar(0, 0), init(1));
        cta_sync(&mut log, 0, &(0..8).collect::<Vec<_>>(), 8);
        if blocked {
            log.blocked_at_exit(4, 3, mbar(0, 0), wait(0));
        } else {
            log.cmd(4, 3, mbar(0, 0), wait(0));
        }
        log.build()
    };
    let r = run_all(&build(false), cta(8), Verdict::Error);
    assert_kind(&r, "deadlock");
    let concrete = check(&build(true), &config(cta(8)));
    assert_eq!(serialize(&concrete)["execution_error"]["kind"], "deadlock");
}

/// `test_native_kernel_contracts.py::test_native_synccheck_accepts_setmaxnreg_with_trailing_warps`
/// (6 warps: one full warpgroup plus two trailing warps).
#[test]
fn setmaxnreg_with_trailing_warps_is_clean() {
    let mut log = LogBuilder::new();
    setmax_wg(&mut log, 1, 0, false, 88);
    run_all(&log.build(), cta(6), Verdict::Clean);
}

type Setting = Option<(bool, u32)>;

fn four_wg_budget(settings: [Setting; 4]) -> numsim_core::observe::RecordingObserver {
    let mut log = LogBuilder::new();
    for (wg, s) in settings.iter().enumerate() {
        if let Some((inc, count)) = *s {
            setmax_wg(&mut log, 1 + wg as u32, wg as u32, inc, count);
        }
    }
    log.build()
}

/// `test_native_kernel_contracts.py::test_native_synccheck_accepts_native_setmaxnreg_budget_shapes`
/// [full-register-file-budget, conserved-asymmetric-split, no-setmaxnreg,
/// missing-wg-default-within-budget].
#[test]
fn setmaxnreg_budget_shapes_are_clean() {
    let shapes: [[Setting; 4]; 4] = [
        [Some((true, 128)); 4],
        [Some((true, 200)), Some((false, 120)), Some((false, 96)), Some((false, 96))],
        [None; 4],
        [Some((true, 160)), Some((false, 80)), Some((false, 80)), None],
    ];
    for s in shapes {
        run_all(&four_wg_budget(s), cta(16), Verdict::Clean);
    }
}

/// `test_native_kernel_contracts.py::test_native_synccheck_rejects_native_setmaxnreg_oversubscription`
/// [oversubscribed-final-allocation, missing-wg-default-tips-budget].
#[test]
fn setmaxnreg_oversubscription_deadlocks() {
    let shapes: [[Setting; 4]; 2] = [
        [Some((true, 224)), Some((true, 232)), Some((false, 48)), Some((false, 64))],
        [Some((true, 232)), Some((false, 120)), Some((false, 120)), None],
    ];
    for s in shapes {
        let r = run_all(&four_wg_budget(s), cta(16), Verdict::Error);
        assert_pool_deadlock(&r);
    }
}

fn six_wg_budget(cg1_target: u32) -> numsim_core::observe::RecordingObserver {
    let mut log = LogBuilder::new();
    for (wg, inc, count) in [(0, true, 104), (1, true, 104), (2, true, 104), (3, true, cg1_target), (4, false, 40), (5, false, 48)] {
        setmax_wg(&mut log, 1 + wg, wg, inc, count);
    }
    log.build()
}

/// `test_native_kernel_contracts.py::test_native_synccheck_accepts_six_warpgroup_launch_allocation`
/// and `::test_native_synccheck_rejects_six_warpgroup_rounding_residual_as_pool`
/// (24 warps start at 80 registers per thread).
#[test]
fn six_warpgroup_setmaxnreg_rounding() {
    run_all(&six_wg_budget(80), cta(24), Verdict::Clean);
    let r = run_all(&six_wg_budget(112), cta(24), Verdict::Error);
    assert_pool_deadlock(&r);
}

/// `test_native_kernel_contracts.py::test_native_synccheck_executes_conditional_tmem_pool_with_exact_topology`
/// [rank-conditional-two-cta (4 warps per CTA), thread-topology-two-warps
/// (2 warps per CTA)]: cluster rank 0 allocates 64 columns, rank 1 allocates
/// 512; the column-count rule is per CTA.
#[test]
fn per_rank_tmem_pools_are_clean() {
    for wpc in [4u32, 2] {
        let init_ = ResourceInit { warps_per_cta: wpc, cluster_warps: 2 * wpc, ..ResourceInit::default() };
        let mut log = LogBuilder::new();
        log.cmd(0, 1, tmem(0), tmem_alloc_on(0, 64)).cmd(wpc, 1, tmem(0), tmem_alloc_on(1, 512));
        cluster_sync(&mut log, &(0..2 * wpc).collect::<Vec<_>>());
        log.cmd(0, 2, tmem(0), tmem_dealloc_on(0, 0, 64)).cmd(wpc, 2, tmem(0), tmem_dealloc_on(1, 0, 512));
        run_all(&log.build(), init_, Verdict::Clean);
    }
}

/// `test_native_kernel_contracts.py::test_native_synccheck_requires_cross_warp_tmem_quiescence_before_dealloc`
/// (clean `ordered` half only: the `unordered` error is a TMEM access after
/// dealloc, not a sync-protocol event).
#[test]
fn cross_warp_tmem_dealloc_after_gate_is_clean() {
    let gate = mbar(0, 0);
    let mut log = LogBuilder::new();
    log.cmd(0, 1, gate, init(1)).cmd(0, 2, tmem(0), tmem_alloc(32));
    cta_sync(&mut log, 0, &[0, 1], 2);
    log.cmd(1, 3, gate, arrive(1));
    log.cmd(0, 4, gate, wait(0)).cmd(0, 5, tmem(0), tmem_relinquish()).cmd(0, 6, tmem(0), tmem_dealloc(0, 32));
    run_all(&log.build(), cta(2), Verdict::Clean);
}

// ---------------------------------------------------------------------------
// test_native_split_site_named_barrier.py
// ---------------------------------------------------------------------------

/// `test_native_split_site_named_barrier.py::test_split_sites_across_full_cta_warps_stay_clean`
/// (32 warps, `bar.sync 3, 1024` from two static sites) and
/// `::test_split_sites_across_sub_cta_warps_stay_clean` (warps 0 and 1 of 3,
/// `bar.sync 3, 64` from two sites). The 32-warp generation runs under the
/// default configuration only: the unreduced whole-program search enumerates
/// every arrival order (32!) and exhausts its budget; a 4-warp split runs
/// under all four.
#[test]
fn split_named_sync_sites_across_warps_are_clean() {
    let split = |warps: u32| {
        let mut log = LogBuilder::new();
        for w in 0..warps {
            log.cmd(w, if w < warps / 2 { 1 } else { 2 }, named_bar(0, 3), bar_sync(w, u64::from(warps) * 32));
        }
        log.build()
    };
    let r = check(&split(32), &config(cta(32)));
    assert_eq!(r.verdict, Verdict::Clean, "{:#}", serialize(&r));
    run_all(&split(4), cta(4), Verdict::Clean);
    let mut log = LogBuilder::new();
    log.cmd(0, 1, named_bar(0, 3), bar_sync(0, 64)).cmd(1, 2, named_bar(0, 3), bar_sync(1, 64));
    run_all(&log.build(), cta(3), Verdict::Clean);
}

/// `test_native_split_site_named_barrier.py::test_split_sites_within_one_warp_are_flagged`:
/// lanes 0-15 and 16-31 reach two different `bar.sync 3, 32` instructions.
// sync delta B2: legacy warp_collective_divergence; new PartialWarp
// (`named_barrier_invalid_arrival_count`).
#[test]
fn named_sync_split_within_one_warp_is_partial_warp() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, named_bar(0, 3), bar(named::Cmd::Sync, 0, 32, 0x0000_ffff, true));
    log.cmd(0, 2, named_bar(0, 3), bar(named::Cmd::Sync, 0, 32, 0xffff_0000, true));
    let r = run_all(&log.build(), one(), Verdict::Error);
    assert_kind(&r, "named_barrier_invalid_arrival_count");
}

/// `test_native_split_site_named_barrier.py::test_split_site_full_cta_arrive_plus_sync_stays_clean`
/// and `::test_arrive_mixed_split_sync_sites_stay_clean` (`bar.arrive` +
/// `bar.sync` from two sites).
#[test]
fn named_arrive_plus_split_syncs_are_clean() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, named_bar(0, 3), bar_arrive(0, 64)).cmd(1, 2, named_bar(0, 3), bar_sync(1, 64));
    run_all(&log.build(), cta(2), Verdict::Clean);
    let mut log = LogBuilder::new();
    log.cmd(0, 1, named_bar(0, 3), bar_arrive(0, 96)).cmd(1, 2, named_bar(0, 3), bar_sync(1, 96)).cmd(2, 3, named_bar(0, 3), bar_sync(2, 96));
    run_all(&log.build(), cta(3), Verdict::Clean);
}

/// `test_native_split_site_named_barrier.py::test_split_site_full_cta_unaligned_sync_stays_clean`
/// (`barrier.sync 3, 64` from two sites).
#[test]
fn unaligned_named_syncs_are_clean() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, named_bar(0, 3), bar(named::Cmd::Sync, 0, 64, FULL_MASK, false));
    log.cmd(1, 2, named_bar(0, 3), bar(named::Cmd::Sync, 1, 64, FULL_MASK, false));
    run_all(&log.build(), cta(2), Verdict::Clean);
}

/// `test_native_split_site_named_barrier.py::test_unaligned_plus_one_aligned_sync_site_is_flagged`
/// and `::test_aligned_plus_one_unaligned_sync_site_is_flagged`.
// sync delta B3: legacy fixed_sync_protocol_error "mixes aligned and unaligned
// blocking syncs"; new: mixing forms is allowed (clean).
#[test]
fn mixed_aligned_and_unaligned_named_syncs_are_clean() {
    for aligned_warp in [1u32, 0] {
        let mut log = LogBuilder::new();
        for w in 0..2 {
            log.cmd(w, 1 + w, named_bar(0, 3), bar(named::Cmd::Sync, w, 64, FULL_MASK, w == aligned_warp));
        }
        run_all(&log.build(), cta(2), Verdict::Clean);
    }
}

/// `test_native_split_site_named_barrier.py::test_uniform_loop_full_cta_bar_sync_stays_clean`
/// (one site, two iterations) and `::test_sequential_full_cta_bar_sync_sites_stay_clean`
/// (two sites).
#[test]
fn repeated_named_sync_generations_are_clean() {
    for sites in [[1, 1], [1, 2]] {
        let mut log = LogBuilder::new();
        for site in sites {
            bar_sync_all(&mut log, site, 3, 64, &[0, 1]);
        }
        run_all(&log.build(), cta(2), Verdict::Clean);
    }
}

/// `test_native_split_site_named_barrier.py::test_same_site_bar_sync_with_skewed_loop_iterations_stays_clean`:
/// warp 0 completes generation 0 of barrier 3 with `bar.arrive`, so the two
/// warps meet at the same static `bar.sync` in different loop iterations.
#[test]
fn skewed_named_sync_iterations_are_clean() {
    let b3 = named_bar(0, 3);
    let b4 = named_bar(0, 4);
    let unaligned = |w| bar(named::Cmd::Sync, w, 64, FULL_MASK, false);
    let mut log = LogBuilder::new();
    log.cmd(0, 1, b3, bar_arrive(0, 64)).cmd(0, 2, b4, unaligned(0)).cmd(0, 3, b3, bar_sync(0, 64)).cmd(0, 3, b3, bar_sync(0, 64));
    log.cmd(1, 3, b3, bar_sync(1, 64)).cmd(1, 2, b4, unaligned(1)).cmd(1, 3, b3, bar_sync(1, 64)).cmd(1, 3, b3, bar_sync(1, 64));
    run_all(&log.build(), cta(2), Verdict::Clean);
}

/// `test_native_split_site_named_barrier.py::test_cta_sync_wrapper_and_handwritten_bar_zero_split_site_stays_clean`.
#[test]
fn cta_sync_wrapper_and_handwritten_bar_zero_are_clean() {
    let mut log = LogBuilder::new();
    log.cmd(0, 100, named_bar(0, 0), bar_sync(0, 64)).cmd(1, 7, named_bar(0, 0), bar_sync(1, 64));
    run_all(&log.build(), cta(2), Verdict::Clean);
}

/// `test_native_split_site_named_barrier.py::test_repeated_warpgroup_sync_generations_stay_clean`:
/// each warpgroup syncs on its own barrier three times.
#[test]
fn per_warpgroup_named_sync_generations_are_independent() {
    let mut log = LogBuilder::new();
    for _ in 0..3 {
        bar_sync_all(&mut log, 1, 6, 128, &[0, 1, 2, 3]);
        bar_sync_all(&mut log, 2, 7, 128, &[4, 5, 6, 7]);
    }
    run_all(&log.build(), cta(8), Verdict::Clean);
}

/// `test_native_split_site_named_barrier.py::test_setmaxnreg_between_warpgroup_syncs_stays_clean`:
/// `dec 88`, `warpgroup_sync(7)` (an aligned `bar.sync 7, 128` that the
/// interpreter credits as the warpgroup sync, logged by the completing warp),
/// `inc 232`.
#[test]
fn setmaxnreg_separated_by_warpgroup_sync_is_clean() {
    let mut log = LogBuilder::new();
    setmax_wg(&mut log, 1, 0, false, 88);
    bar_sync_all(&mut log, 2, 7, 128, &[0, 1, 2, 3]);
    log.cmd(3, 2, reg_pool(0), wg_sync(0));
    setmax_wg(&mut log, 3, 0, true, 232);
    run_all(&log.build(), cta(4), Verdict::Clean);
}

// ---------------------------------------------------------------------------
// test_register_allocation.py
// ---------------------------------------------------------------------------

fn unconfigured_budget_log() -> numsim_core::observe::RecordingObserver {
    let warps = (0..12).collect::<Vec<_>>();
    let mut log = LogBuilder::new();
    setmax_wg(&mut log, 1, 2, false, 96);
    cta_sync(&mut log, 0, &warps, 12);
    setmax_wg(&mut log, 2, 0, true, 232);
    setmax_wg(&mut log, 3, 1, true, 232);
    cta_sync(&mut log, 0, &warps, 12);
    log.build()
}

/// `test_register_allocation.py::test_unconfigured_register_budget_is_still_checked`
/// (synccheck half, verdict): 3 warpgroups at the default 168 registers; WG 2
/// frees 72, WGs 0 and 1 each need 64 more.
#[test]
fn unconfigured_setmaxnreg_budget_deadlocks() {
    let r = run_all(&unconfigured_budget_log(), cta(12), Verdict::Error);
    assert_pool_deadlock(&r);
}

/// `test_register_allocation.py::test_unconfigured_register_budget_is_still_checked`
/// (synccheck half, exact kind): the legacy test requires a
/// `setmaxnreg_pool_deadlock` finding.
#[test]
fn unconfigured_setmaxnreg_budget_is_setmaxnreg_pool_deadlock() {
    let r = run_all(&unconfigured_budget_log(), cta(12), Verdict::Error);
    assert!(r.iter().any(|(_, rep)| kind(rep) == "setmaxnreg_pool_deadlock"), "{:?}", kinds(&r));
}

/// `test_register_allocation.py::test_launch_bounds_allow_register_redistribution_above_initial_count`
/// (synccheck half): `launch_bounds_min_blocks_per_sm = 2` starts the three
/// warpgroups at 80 registers; WG 2 frees 32, covering +24 and +8. The
/// scheduler steps `Configure { count: 80 }` into the live `SyncTable` and
/// logs it as a host-side `Protocol` event before any warp runs.
#[test]
fn launch_bounds_register_redistribution_is_clean() {
    let mut log = LogBuilder::new();
    // The scheduler logs the launch-bounds budget as a host-side Protocol
    // event (W2); synccheck applies it to the initial pool.
    log.host(vec![(reg_pool(0), configure(80))]);
    setmax_wg(&mut log, 1, 0, true, 104);
    setmax_wg(&mut log, 2, 1, true, 88);
    setmax_wg(&mut log, 3, 2, false, 48);
    run_all(&log.build(), cta(12), Verdict::Clean);
}

// ---------------------------------------------------------------------------
// runtime/test_device_protocol_ops.py
// ---------------------------------------------------------------------------

/// `runtime/test_device_protocol_ops.py::test_divergent_unaligned_named_barrier_runtime`:
/// lane 0 and lanes 1-31 reach two different `barrier.sync 5, 32`.
// Partial-warp ruling (sync-isa-answers Q3/Q5, supersedes delta B2 for
// non-aligned forms): the lanes gather (`named::gather`) and the warp
// arrives once; the engine logs that one full-mask contribution. Legacy was
// clean too. The `.aligned` split stays `PartialWarp`
// (`named_sync_split_within_one_warp_is_partial_warp`).
#[test]
fn divergent_unaligned_named_barrier_gathers_into_one_arrival() {
    let mut log = LogBuilder::new();
    log.cmd(0, 2, named_bar(0, 5), bar(named::Cmd::Sync, 0, 32, FULL_MASK, false));
    run_all(&log.build(), one(), Verdict::Clean);
    // A log that still carries the pieces (pre-ruling engine) fails closed.
    let mut log = LogBuilder::new();
    log.cmd(0, 1, named_bar(0, 5), bar(named::Cmd::Sync, 0, 32, 0x0000_0001, false));
    log.cmd(0, 2, named_bar(0, 5), bar(named::Cmd::Sync, 0, 32, 0xffff_fffe, false));
    let r = run_all(&log.build(), one(), Verdict::Error);
    assert_kind(&r, "named_barrier_invalid_arrival_count");
}

/// `runtime/test_device_protocol_ops.py::test_cp_async_predicate_tracks_only_issuing_lanes`:
/// lanes 0-15 issue; every lane commits (lanes 16-31 an empty group) and waits.
#[test]
fn predicated_cp_async_per_lane_groups_are_clean() {
    let mut log = LogBuilder::new();
    log.cmds(0, 1, lanes(0..16, async_group::Cmd::Issue));
    log.cmds(0, 2, lanes(0..32, async_group::Cmd::Commit));
    log.cmds(0, 3, lanes(0..32, async_group::Cmd::Wait { n: 0, read: false }));
    run_all(&log.build(), one(), Verdict::Clean);
}

/// `runtime/test_device_protocol_ops.py::test_uncommitted_cp_async_is_a_protocol_error`.
// sync delta A3: legacy CompletionSourceNotQuiescent error ("uncommitted
// issue"); new UncommittedAtExit review lint.
#[test]
fn uncommitted_cp_async_is_a_review_lint() {
    let mut log = LogBuilder::new();
    log.cmds(0, 1, lanes(0..32, async_group::Cmd::Issue));
    let r = run_all(&log.build(), one(), Verdict::Review);
    let p = serialize(&r[0].1);
    assert_eq!(p["review"][0]["kind"], "sync_exit_lint", "{p:#}");
    assert!(p["review"][0]["lint"].as_str().unwrap().contains("UncommittedAtExit"), "{p:#}");
}

// ---------------------------------------------------------------------------
// test_native_shared_control_relay.py (the relayed value has already chosen
// the waited barrier in the concrete run; `barriers[1]` is never inited)
// ---------------------------------------------------------------------------

fn relay_verdict(log: &LogBuilder, warps: u32, selected: u32) {
    let r = run_all(&log.build(), cta(warps), if selected == 0 { Verdict::Clean } else { Verdict::Error });
    if selected != 0 {
        assert_kind(&r, "mbarrier_use_before_init");
    }
}

/// `test_native_shared_control_relay.py::test_public_native_synccheck_uses_cross_warp_shared_value_for_exact_branch`.
#[test]
fn relayed_branch_selects_the_waited_barrier() {
    for selected in [0u32, 1] {
        let mut log = LogBuilder::new();
        log.cmd(0, 1, mbar(0, 0), init(1));
        cta_sync(&mut log, 0, &[0, 1], 2);
        log.cmd(0, 2, mbar(0, 0), arrive(1)).cmd(1, 3, mbar(0, 8 * selected), wait(0));
        relay_verdict(&log, 2, selected);
    }
}

/// `test_native_shared_control_relay.py::test_public_native_synccheck_resolves_alternate_producer_exactly`
/// [producer_warp = 0, 1].
#[test]
fn alternate_producer_relay_selects_the_waited_barrier() {
    for producer in [0u32, 1] {
        for selected in [0u32, 1] {
            let mut log = LogBuilder::new();
            log.cmd(producer, 1, mbar(0, 0), init(1)).cmd(producer, 2, mbar(0, 0), arrive(1));
            cta_sync(&mut log, 0, &[0, 1, 2], 3);
            log.cmd(2, 3, mbar(0, 8 * selected), wait(0));
            relay_verdict(&log, 3, selected);
        }
    }
}

/// `test_native_shared_control_relay.py::test_public_native_synccheck_resolves_transitive_relay_exactly`.
#[test]
fn transitive_relay_selects_the_waited_barrier() {
    for selected in [0u32, 1] {
        let mut log = LogBuilder::new();
        log.cmd(0, 1, mbar(0, 0), init(1));
        cta_sync(&mut log, 0, &[0, 1, 2], 3);
        log.cmd(0, 2, mbar(0, 0), arrive(1));
        cta_sync(&mut log, 0, &[0, 1, 2], 3);
        log.cmd(2, 3, mbar(0, 8 * selected), wait(0));
        relay_verdict(&log, 3, selected);
    }
}

/// `test_native_shared_control_relay.py::test_public_native_synccheck_uses_concrete_atomic_selected_sync_trace`
/// (the concrete CAS winner is warp 0, so warp 2 waits on `barriers[0]`).
#[test]
fn atomic_winner_relay_is_clean() {
    let mut log = LogBuilder::new();
    log.cmd(2, 1, mbar(0, 0), init(1)).cmd(2, 2, mbar(0, 0), arrive(1));
    cta_sync(&mut log, 0, &[0, 1, 2], 3);
    cta_sync(&mut log, 0, &[0, 1, 2], 3);
    log.cmd(2, 3, mbar(0, 0), wait(0));
    run_all(&log.build(), cta(3), Verdict::Clean);
}

/// `test_native_shared_control_relay.py::test_public_native_synccheck_uses_completed_tma_payload_for_exact_branch`:
/// the TMA payload selects `barriers[1]` (inited) or `barriers[2]` (not).
#[test]
fn tma_payload_relay_selects_the_waited_barrier() {
    for selected in [0u32, 1] {
        let mut log = LogBuilder::new();
        log.cmd(0, 1, mbar(0, 0), init(1)).cmd(0, 1, mbar(0, 8), init(1));
        cta_sync(&mut log, 0, &[0, 1], 2);
        log.issue(0, 2, mbar(0, 0), 16, 0, Vec::new());
        log.cmd(0, 3, mbar(0, 0), arrive_tx(1, 16)).cmd(0, 4, mbar(0, 8), arrive(1));
        log.cmd(1, 5, mbar(0, 0), wait(0)).cmd(1, 6, mbar(0, 8 * (1 + selected)), wait(0));
        relay_verdict(&log, 2, selected);
    }
}

// ---------------------------------------------------------------------------
// test_native_drain_tail.py
// ---------------------------------------------------------------------------

/// `test_native_drain_tail.py::test_public_native_drain_tail_has_exact_generations_and_no_phantom_arrive`
/// [work_total = 1, 3] (verdict only; per-effect generations and loop frames
/// are payload detail): a one-slot ready/free ping-pong whose last `free`
/// generation is never consumed.
#[test]
fn drain_tail_ping_pong_is_clean() {
    for work_total in [1, 3] {
        run_all(&pipeline(2, 1, work_total, 0), cta(2), Verdict::Clean);
    }
}

// ---------------------------------------------------------------------------
// Unreviewed-B batch: synccheck verdicts asserted outside
// tests/analysis_tools/synccheck (racecheck/test_native_raw_async_copy_footprints.py,
// shared/, numsim/runtime/). Coverage map: scripts/numsim-v2/coverage/other_b.tsv.
// ---------------------------------------------------------------------------

/// One event of `warp` that delivers `bytes` to every barrier in `targets`
/// (a multicast copy).
fn issue_multi(log: &mut LogBuilder, warp: u32, site: u32, targets: &[numsim_core::sync::ResourceId], bytes: u64) {
    let issued = targets.iter().map(|&res| numsim_core::observe::AsyncTarget { res, bytes, arrivals: 0 }).collect();
    log.event(warp, site, Vec::new(), issued, None, None, numsim_core::observe::ProtocolStatus::Committed);
}

/// One cluster of `ctas` CTAs with `warps` warps each.
fn cluster_of(ctas: u32, warps: u32) -> ResourceInit {
    ResourceInit { warps_per_cta: warps, cluster_warps: ctas * warps, ..ResourceInit::default() }
}

/// `test_native_raw_async_copy_footprints.py::test_cp_async_mbarrier_arrive_synccheck_is_unchanged[ordered|unwaited]`
/// (both kernels have the same protocol trace): barrier inited for two
/// arrivals; warp 0 lane 0 issues a `cp.async`, `cp.async.mbarrier.arrive`
/// (no `.noinc`: +1 pending, deferred arrival) and a plain arrive; warp 1
/// lane 0 arrives and waits phase 0. Clean (the `cp.async.mbarrier.arrive`
/// tracks the uncommitted `cp.async`, so no `UncommittedAtExit` lint).
#[test]
fn cp_async_mbarrier_arrive_pending_count_protocol() {
    let g = async_group_res(0, 0, async_group::Domain::CpAsync);
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(2));
    cta_sync(&mut log, 0, &[0, 1], 2);
    log.cmd(0, 2, g, group(async_group::Cmd::Issue));
    log.issue(0, 3, mbar(0, 0), 0, 1, vec![(mbar(0, 0), inc_pending(1)), (g, group(async_group::Cmd::ArriveOn))]);
    log.cmd(0, 4, mbar(0, 0), arrive(1));
    log.cmd(1, 5, mbar(0, 0), arrive(1)).cmd(1, 6, mbar(0, 0), wait(0));
    run_all(&log.build(), cta(2), Verdict::Clean);
}

/// `test_native_raw_async_copy_footprints.py::test_bulk_g2s_cta_synccheck_reports_no_finding[plain|ignore_oob|unwaited]`
/// and `::test_bulk_g2s_cta_ignore_oob_short_source_synccheck_reports_no_finding`:
/// init(1), `cta_sync`, `arrive.expect_tx(16)`, one 16-byte G2S copy
/// (`.ignore_oob` still completes the full 16 bytes), wait. The unwaited
/// kernel only moves a data read, so the trace is the same.
#[test]
fn bulk_g2s_cta_protocol_is_clean() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    cta_sync(&mut log, 0, &[0], 1);
    log.cmd(0, 2, mbar(0, 0), arrive_tx(1, 16));
    log.issue(0, 3, mbar(0, 0), 16, 0, Vec::new());
    log.cmd(0, 4, mbar(0, 0), wait(0));
    run_all(&log.build(), one(), Verdict::Clean);
}

/// `test_native_raw_async_copy_footprints.py::test_bulk_g2s_multicast_synccheck_reports_no_finding[ordered|unwaited|unselected]`:
/// two CTAs init their barrier, cluster sync; `ordered`/`unwaited`: both
/// arm 16 bytes and CTA 0 multicasts to both barriers (mask 3), both wait;
/// `unselected`: only CTA 0 arms, the copy targets only CTA 0 (mask 1),
/// CTA 0 waits, cluster sync.
#[test]
fn bulk_g2s_multicast_protocol_is_clean() {
    for selected_both in [true, false] {
        let mut log = LogBuilder::new();
        log.cmd(0, 1, mbar(0, 0), init(1)).cmd(1, 1, mbar(1, 0), init(1));
        cluster_sync(&mut log, &[0, 1]);
        let ctas: &[u32] = if selected_both { &[0, 1] } else { &[0] };
        for &c in ctas {
            log.cmd(c, 2, mbar(c, 0), arrive_tx(1, 16));
        }
        let targets: Vec<_> = ctas.iter().map(|&c| mbar(c, 0)).collect();
        issue_multi(&mut log, 0, 3, &targets, 16);
        for &c in ctas {
            log.cmd(c, 4, mbar(c, 0), wait(0));
        }
        if !selected_both {
            cluster_sync(&mut log, &[0, 1]);
        }
        run_all(&log.build(), cluster_of(2, 1), Verdict::Clean);
    }
}

/// `test_native_raw_async_copy_footprints.py::test_bulk_s2s_cluster_synccheck_reports_no_finding[ordered|unwaited]`:
/// CTA 0 arms CTA 1's barrier through `mapa` (`arrive.expect_tx.shared::cluster`)
/// and pushes 16 bytes into CTA 1 completing on it; CTA 1 waits.
#[test]
fn bulk_s2s_cluster_protocol_is_clean() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1)).cmd(1, 1, mbar(1, 0), init(1));
    cluster_sync(&mut log, &[0, 1]);
    log.cmd(0, 2, mbar(1, 0), arrive_tx(1, 16));
    log.issue(0, 3, mbar(1, 0), 16, 0, Vec::new());
    log.cmd(1, 4, mbar(1, 0), wait(0));
    cluster_sync(&mut log, &[0, 1]);
    run_all(&log.build(), cluster_of(2, 1), Verdict::Clean);
}

/// `test_native_raw_async_copy_footprints.py::test_bulk_s2g_masked_synccheck_reports_no_finding[ordered|unwaited]`
/// (the bulk-group half of `runtime/test_bulk_copy_scopes.py::test_bulk_copy_scope_and_elements`
/// too): lane 0 issues a bulk copy, commits, `wait_group.read 0` (scopes
/// test only), `wait_group 0`.
#[test]
fn bulk_group_issue_commit_wait_is_clean() {
    for read_first in [false, true] {
        let g = async_group_res(0, 0, async_group::Domain::Bulk);
        let mut log = LogBuilder::new();
        log.cmd(0, 1, g, group(async_group::Cmd::Issue)).cmd(0, 2, g, group(async_group::Cmd::Commit));
        if read_first {
            log.cmd(0, 3, g, group(async_group::Cmd::Wait { n: 0, read: true }));
        }
        log.cmd(0, 4, g, group(async_group::Cmd::Wait { n: 0, read: false }));
        run_all(&log.build(), one(), Verdict::Clean);
    }
}

/// `test_native_raw_async_copy_footprints.py::test_tma_gather4_synccheck_reports_no_finding[ordered|unwaited]`:
/// one 64-byte gather4 TMA against `arrive.expect_tx(64)`.
#[test]
fn tma_gather4_protocol_is_clean() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    cta_sync(&mut log, 0, &[0], 1);
    log.cmd(0, 2, mbar(0, 0), arrive_tx(1, 64));
    log.issue(0, 3, mbar(0, 0), 64, 0, Vec::new());
    log.cmd(0, 4, mbar(0, 0), wait(0));
    run_all(&log.build(), one(), Verdict::Clean);
}

/// `shared/test_native_dense_cta2_mma_ordering.py::test_committed_dense_cta2_mma_pipeline_passes_synccheck`:
/// two CTAs x two warps; each CTA's warp 0 inits its barrier; cluster sync;
/// CTA 0 warp 0 issues the `cta_group::2` MMA and commits it to its barrier
/// (`tcgen05.commit.cta_group::2.mbarrier::arrive::one`), then waits;
/// cluster sync.
#[test]
fn committed_dense_cta2_mma_protocol_is_clean() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1)).cmd(2, 1, mbar(1, 0), init(1));
    cluster_sync(&mut log, &[0, 1, 2, 3]);
    log.cmd(0, 2, tcgen_work(0, 0), work(tcgen::WorkCmd::Issue));
    log.issue(0, 3, mbar(0, 0), 0, 1, vec![(tcgen_work(0, 0), work(tcgen::WorkCmd::Commit))]);
    log.cmd(0, 4, mbar(0, 0), wait(0));
    cluster_sync(&mut log, &[0, 1, 2, 3]);
    run_all(&log.build(), cluster_of(2, 2), Verdict::Clean);
}

/// `runtime/test_bulk_g2s_scopes.py::test_strong_g2s_scope_elements_and_predication`
/// (synccheck half; `scope` does not reach the protocol): CTA 0 warp 0 inits
/// and arms `48 * enabled` bytes; cluster sync; the two writers (warps,
/// lanes or CTAs) each deliver their copy (32 and 16 bytes) to CTA 0's
/// barrier when enabled; CTA 0 waits; cluster sync.
#[test]
fn strong_g2s_two_writer_protocol_is_clean() {
    for (ctas, warps, writers) in [(1u32, 2u32, [0u32, 1]), (1, 1, [0, 0]), (2, 1, [0, 1])] {
        for enabled in [false, true] {
            let all: Vec<u32> = (0..ctas * warps).collect();
            let mut log = LogBuilder::new();
            log.cmd(0, 1, mbar(0, 0), init(1)).cmd(0, 2, mbar(0, 0), arrive_tx(1, if enabled { 48 } else { 0 }));
            cluster_sync(&mut log, &all);
            if enabled {
                for (a, &w) in writers.iter().enumerate() {
                    log.issue(w, 3, mbar(0, 0), 32 - 16 * a as u64, 0, Vec::new());
                }
            }
            log.cmd(0, 4, mbar(0, 0), wait(0));
            cluster_sync(&mut log, &all);
            run_all(&log.build(), cluster_of(ctas, warps), Verdict::Clean);
        }
    }
}

/// `runtime/test_mbarrier_drop.py::test_mbarrier_drop_invalid_count[count|no_complete]`
/// (synccheck half): init(4), `cta_sync`, then `mbarrier.arrive_drop[.noComplete]`
/// with count 5 > the 4 pending arrivals. Legacy: error whose kind names an
/// arrival, or an untyped `engine_error`. sync delta M5: an `arrive_drop`
/// that takes the expected count below zero is `DropUnderflow`; sync delta
/// M6: a `.noComplete` arrive that would complete the phase is the typed
/// `NoCompleteWouldComplete`.
#[test]
fn mbarrier_drop_invalid_count_is_an_error() {
    for no_complete in [false, true] {
        let mut log = LogBuilder::new();
        log.cmd(0, 1, mbar(0, 0), init(4));
        cta_sync(&mut log, 0, &[0], 1);
        let drop = SyncCmd::Mbarrier(numsim_core::sync::mbarrier::Cmd::Arrive { count: 5, tx: None, drop: true, no_complete });
        log.cmd(0, 2, mbar(0, 0), drop);
        let r = run_all(&log.build(), one(), Verdict::Error);
        for (name, rep) in &r {
            let k = kind(rep);
            let expected = if no_complete { "mbarrier_no_complete_violated" } else { "mbarrier_drop_underflow" };
            assert_eq!(k, expected, "{name}");
        }
    }
}

/// `runtime/test_mbarrier_lane_semantics.py::test_checkers_group_pending_blocking_waits_on_distinct_barriers[synccheck]`
/// and `::test_synccheck_keeps_one_projection_for_a_multi_barrier_wait`
/// (`checker_lane_varying_pending_wait`): warp 0 lanes 0..4 init four
/// barriers; `cta_sync`; warp 1 lanes 0..4 arrive on and wait their own
/// barrier (one lane-varying instruction each); `cta_sync`; warp 1 waits
/// parity 1 on all four while warp 0 arrives. Clean, and the four-barrier
/// wait is one projection: the `cta_sync` program plus one mbarrier
/// component (legacy `projections=2`).
#[test]
fn lane_varying_multi_barrier_wait_is_one_projection() {
    let bars: Vec<_> = (0..4).map(|l| mbar(0, 8 * l)).collect();
    let all = |cmd: SyncCmd| bars.iter().map(|&b| (b, cmd)).collect::<Vec<_>>();
    let mut log = LogBuilder::new();
    log.cmds(0, 1, all(init(1)));
    cta_sync(&mut log, 0, &[0, 1], 2);
    log.cmds(1, 2, all(arrive(1))).cmds(1, 3, all(wait(0)));
    cta_sync(&mut log, 0, &[0, 1], 2);
    log.cmds(1, 4, all(wait(1)));
    log.cmds(0, 5, all(arrive(1)));
    let r = run_all(&log.build(), cta(2), Verdict::Clean);
    assert_eq!(stat(&r[0].1, "program_count"), 2);
}

/// `runtime/test_mbarrier_lane_semantics.py::test_checkers_allow_distinct_barrier_waits[synccheck]`
/// (`lane_varying_expect_tx`): lanes 0 and 1 init their barrier; `cta_sync`;
/// lane 0 arms barrier 0 and copies 16 bytes into it, then lane 1 the same
/// on barrier 1; lanes 0 and 1 wait their own barrier in one instruction.
#[test]
fn lane_varying_expect_tx_waits_are_clean() {
    let (b0, b1) = (mbar(0, 0), mbar(0, 8));
    let mut log = LogBuilder::new();
    log.cmds(0, 1, vec![(b0, init(1)), (b1, init(1))]);
    cta_sync(&mut log, 0, &[0], 1);
    for b in [b0, b1] {
        log.cmd(0, 2, b, arrive_tx(1, 16));
        log.issue(0, 3, b, 16, 0, Vec::new());
    }
    log.cmds(0, 4, vec![(b0, wait(0)), (b1, wait(0))]);
    run_all(&log.build(), one(), Verdict::Clean);
}

/// `runtime/test_mbarrier_lane_semantics.py::test_relaxed_query_does_not_acquire_arriving_threads_memory`
/// (synccheck half): warp 0 inits, `cta_sync`, warp 0 arrives, warp 1's
/// relaxed `test_wait` loop succeeds. Clean.
#[test]
fn relaxed_test_wait_poll_is_clean() {
    let mut log = LogBuilder::new();
    log.cmd(0, 1, mbar(0, 0), init(1));
    cta_sync(&mut log, 0, &[0, 1], 2);
    log.cmd(0, 2, mbar(0, 0), arrive(1));
    log.cmd(1, 3, mbar(0, 0), test_parity(0));
    log.test_ok(1, 3, mbar(0, 0), 0);
    run_all(&log.build(), cta(2), Verdict::Clean);
}

/// `runtime/test_red_async.py::test_shared_async_reduction_completion`
/// (synccheck half of `run_checked` on the waited kernel): CTA 1 inits its
/// barrier for `lanes` arrivals; cluster sync; CTA 0's `lanes` lanes each
/// arm it remotely with their element size and issue a reduction completing
/// there (4 x 4-byte `red.async`, or 1 x 16-byte bulk reduction); CTA 1
/// waits; cluster sync.
#[test]
fn shared_async_reduction_protocol_is_clean() {
    for (lanes, bytes) in [(4u64, 4u64), (1, 16)] {
        let mut log = LogBuilder::new();
        log.cmd(0, 1, mbar(0, 0), init(lanes)).cmd(1, 1, mbar(1, 0), init(lanes));
        cluster_sync(&mut log, &[0, 1]);
        log.cmd(0, 2, mbar(1, 0), arrive_tx(lanes, lanes * bytes));
        for _ in 0..lanes {
            log.issue(0, 3, mbar(1, 0), bytes, 0, Vec::new());
        }
        log.cmd(1, 4, mbar(1, 0), wait(0));
        cluster_sync(&mut log, &[0, 1]);
        run_all(&log.build(), cluster_of(2, 1), Verdict::Clean);
    }
}
