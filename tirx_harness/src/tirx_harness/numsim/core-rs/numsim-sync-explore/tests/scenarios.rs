//! Scenario tests translated from
//! `tirx_harness/tests/analysis_tools/synccheck/test_native_synccheck_artifact.py`
//! and `test_native_kernel_contracts.py` as handwritten `SyncEvent` logs.
//!
//! Every scenario is checked under several configurations to show that the
//! verdict does not depend on which reductions are enabled.

use numsim_sync_explore::event::LogBuilder;
use numsim_sync_explore::explore::{Limits, Options};
use numsim_sync_explore::projection::ProjectionMode;
use numsim_sync_explore::synth::{pipeline, PipelineShape};
use numsim_sync_explore::{check, CheckConfig, CheckReport, Finding, IncompleteReason, SyncEvent, SyncOp, Verdict};

const CTA_SYNC: u32 = 0;

fn configs() -> Vec<(&'static str, CheckConfig)> {
    let base = CheckConfig::default();
    vec![
        ("default", base),
        (
            "per-resource-dfs",
            CheckConfig { certificates: false, fingerprints: false, ..base },
        ),
        (
            "components",
            CheckConfig { mode: ProjectionMode::Components, ..base },
        ),
        (
            "whole-plain",
            CheckConfig {
                mode: ProjectionMode::Whole,
                certificates: false,
                fingerprints: false,
                explore: Options::NONE,
                ..base
            },
        ),
    ]
}

fn verdicts(events: &[SyncEvent]) -> Vec<(&'static str, CheckReport)> {
    configs()
        .into_iter()
        .map(|(name, config)| (name, check(events, &config)))
        .collect()
}

fn assert_all(events: &[SyncEvent], verdict: Verdict) -> Vec<(&'static str, CheckReport)> {
    let reports = verdicts(events);
    for (name, report) in &reports {
        assert_eq!(report.verdict, verdict, "{name}: {report:#?}");
    }
    reports
}

fn cta_sync(log: &mut LogBuilder, warps: u32) {
    for warp in 0..warps {
        log.push(warp, 100, SyncOp::NamedSync { bar: CTA_SYNC, expected: warps * 32, count: 32 });
    }
}

/// `native_synccheck_clean`: init, arrive, wait in one warp.
#[test]
fn single_warp_init_arrive_wait_is_clean() {
    let mut log = LogBuilder::new();
    log.push(0, 1, SyncOp::MbarInit { bar: 0, expected: 1 })
        .push(0, 2, SyncOp::MbarArrive { bar: 0, count: 1, expect_tx: 0 })
        .push(0, 3, SyncOp::MbarWait { bar: 0, parity: 0 });
    let reports = assert_all(&log.build(), Verdict::Clean);
    let (_, default) = &reports[0];
    assert_eq!(default.stats.programs, 1);
    assert_eq!(default.stats.certified_programs, 1);
    assert_eq!(default.stats.visited_states, 1);
    assert_eq!(default.termination, "worklist_exhausted");
}

/// `test_public_native_synccheck_wait_before_init_is_exact_error`.
#[test]
fn wait_before_init_is_use_before_init() {
    let mut log = LogBuilder::new();
    log.push(0, 1, SyncOp::MbarWait { bar: 0, parity: 0 })
        .push(0, 2, SyncOp::MbarInit { bar: 0, expected: 1 });
    for (name, report) in assert_all(&log.build(), Verdict::Error) {
        assert_eq!(report.findings[0].kind(), "mbarrier_use_before_init", "{name}");
    }
}

/// Init-before-use missing across warps: warp 1 arrives with no
/// synchronization ordering it after warp 0's init. The reference run happens
/// to init first, so only the certificate (HB check) or the exhaustive
/// search exposes the bad order (`cross_warp_mbarrier_use_requires_init_happens_before`,
/// `unordered_mbarrier_init_and_arrive_exposes_the_bad_order`).
#[test]
fn cross_warp_init_without_publication_is_an_error() {
    let mut log = LogBuilder::new();
    log.push(0, 1, SyncOp::MbarInit { bar: 0, expected: 1 })
        .push(1, 2, SyncOp::MbarArrive { bar: 0, count: 1, expect_tx: 0 })
        .push(0, 3, SyncOp::MbarWait { bar: 0, parity: 0 });
    let reports = assert_all(&log.build(), Verdict::Error);
    let kinds = reports
        .iter()
        .map(|(name, report)| (*name, report.findings[0].kind()))
        .collect::<Vec<_>>();
    assert_eq!(
        kinds,
        [
            ("default", "mbarrier_init_not_happens_before_use"),
            ("per-resource-dfs", "mbarrier_use_before_init"),
            // Certificates apply only to single-resource projections.
            ("components", "mbarrier_use_before_init"),
            ("whole-plain", "mbarrier_use_before_init"),
        ]
    );
    // The DFS witness puts the arrive first.
    let (_, dfs) = &reports[1];
    let Finding::Protocol { witness, .. } = &dfs.findings[0] else { panic!() };
    assert!(witness[0].contains("warp 1 issues mbarrier.arrive"), "{witness:?}");
}

/// Same program with a `cta_sync` publishing the init is clean
/// (`test_public_native_synccheck_accepts_cta_sync_as_mbarrier_init_publication`).
#[test]
fn cta_sync_publishes_init() {
    let mut log = LogBuilder::new();
    log.push(0, 1, SyncOp::MbarInit { bar: 0, expected: 1 });
    cta_sync(&mut log, 2);
    log.push(1, 2, SyncOp::MbarArrive { bar: 0, count: 1, expect_tx: 0 })
        .push(0, 3, SyncOp::MbarWait { bar: 0, parity: 0 });
    assert_all(&log.build(), Verdict::Clean);
}

/// `test_native_synccheck_trailing_warp_does_not_hide_real_under_arrival`:
/// 8 warps arrive (one lane-count of 32 each) against expected 9*32.
#[test]
fn under_arrival_deadlocks() {
    let warps = 9;
    let mut log = LogBuilder::new();
    log.push(0, 1, SyncOp::MbarInit { bar: 0, expected: warps * 32 });
    cta_sync(&mut log, warps);
    log.push(0, 2, SyncOp::MbarWait { bar: 0, parity: 0 });
    for warp in 1..warps {
        log.push(warp, 3, SyncOp::MbarArrive { bar: 0, count: 32, expect_tx: 0 });
    }
    for (name, report) in assert_all(&log.build(), Verdict::Error) {
        assert_eq!(report.findings[0].kind(), "deadlock", "{name}");
        let Finding::Deadlock { unfinished_warps, .. } = &report.findings[0] else { panic!() };
        assert_eq!(unfinished_warps, &[0], "{name}");
    }
    // ... and the 8*32 contract is clean.
    let mut log = LogBuilder::new();
    log.push(0, 1, SyncOp::MbarInit { bar: 0, expected: (warps - 1) * 32 });
    cta_sync(&mut log, warps);
    log.push(0, 2, SyncOp::MbarWait { bar: 0, parity: 0 });
    for warp in 1..warps {
        log.push(warp, 3, SyncOp::MbarArrive { bar: 0, count: 32, expect_tx: 0 });
    }
    assert_all(&log.build(), Verdict::Clean);
}

/// `test_public_native_synccheck_plain_arrival_overflow_is_typed_error`:
/// one arrival more than expected in a single operation.
#[test]
fn arrival_count_overflow_is_typed() {
    let mut log = LogBuilder::new();
    log.push(0, 1, SyncOp::MbarInit { bar: 0, expected: 1 })
        .push(0, 2, SyncOp::MbarArrive { bar: 0, count: 2, expect_tx: 0 });
    for (name, report) in assert_all(&log.build(), Verdict::Error) {
        assert_eq!(report.findings[0].kind(), "mbarrier_arrival_overflow", "{name}");
    }
}

/// `test_public_native_synccheck_rejects_unconsumed_generation_reuse`: the
/// producer can lap the consumer (no back-pressure barrier). The reference
/// run interleaves, so it completes; the alternate order is found offline.
#[test]
fn producer_lap_without_back_pressure_is_an_error() {
    let mut log = LogBuilder::new();
    log.push(0, 1, SyncOp::MbarInit { bar: 0, expected: 1 });
    cta_sync(&mut log, 2);
    log.push(0, 2, SyncOp::MbarArrive { bar: 0, count: 1, expect_tx: 0 })
        .push(0, 2, SyncOp::MbarArrive { bar: 0, count: 1, expect_tx: 0 })
        .push(1, 3, SyncOp::MbarWait { bar: 0, parity: 0 })
        .push(1, 3, SyncOp::MbarWait { bar: 0, parity: 1 });
    let reports = assert_all(&log.build(), Verdict::Error);
    // Generation 0's wait is not ordered before generation 1's arrival.
    assert_eq!(reports[0].1.findings[0].kind(), "mbarrier_wait_overtaken");
    assert_eq!(reports[1].1.findings[0].kind(), "mbarrier_arrive_before_consumption");
}

/// `native_mbarrier_depth_two_pipeline` (`test_native_synccheck_preserves_depth_two_multi_generation_pipeline`).
#[test]
fn depth_two_multi_generation_pipeline_is_clean() {
    let mut log = LogBuilder::new();
    for bar in 0..4 {
        log.push(0, 1, SyncOp::MbarInit { bar, expected: 1 });
    }
    cta_sync(&mut log, 2);
    for iteration in 0..4u32 {
        let slot = iteration % 2;
        if iteration >= 2 {
            log.push(0, 2, SyncOp::MbarWait { bar: 2 + slot, parity: 0 });
        }
        log.push(0, 3, SyncOp::MbarArrive { bar: slot, count: 1, expect_tx: 0 });
    }
    let mut phase = 0u8;
    for iteration in 0..4u32 {
        let slot = iteration % 2;
        log.push(1, 4, SyncOp::MbarWait { bar: slot, parity: phase });
        log.push(1, 5, SyncOp::MbarArrive { bar: 2 + slot, count: 1, expect_tx: 0 });
        if slot == 1 {
            phase ^= 1;
        }
    }
    let events = log.build();
    assert_eq!(events.iter().filter(|e| matches!(e.kind, SyncOp::MbarArrive { .. })).count(), 8);
    assert_eq!(events.iter().filter(|e| matches!(e.kind, SyncOp::MbarWait { .. })).count(), 6);
    assert_all(&events, Verdict::Clean);
}

/// K-stage producer/consumer ring must be clean in every configuration.
#[test]
fn k_stage_pipeline_is_clean() {
    for tma_bytes in [0, 128] {
        let events = pipeline(PipelineShape { warps: 3, stages: 2, iterations: 6, tma_bytes });
        let reports = assert_all(&events, Verdict::Clean);
        let (_, default) = &reports[0];
        // full[0..2], empty[0..2], cta_sync.
        assert_eq!(default.stats.programs, 5);
        if tma_bytes == 0 {
            assert_eq!(default.stats.certified_programs, 5);
        }
    }
}

/// A K-stage ring whose consumer waits the wrong parity on its second lap.
#[test]
fn k_stage_pipeline_with_wrong_parity_deadlocks() {
    let mut events = pipeline(PipelineShape { warps: 2, stages: 2, iterations: 4, tma_bytes: 0 });
    let index = events
        .iter()
        .rposition(|e| e.op.warp == 1 && matches!(e.kind, SyncOp::MbarWait { .. }))
        .unwrap();
    let SyncOp::MbarWait { bar, parity } = events[index].kind else { unreachable!() };
    events[index].kind = SyncOp::MbarWait { bar, parity: parity ^ 1 };
    for (name, report) in assert_all(&events, Verdict::Error) {
        assert!(
            matches!(report.findings[0].kind(), "deadlock" | "mbarrier_invalid_phase"),
            "{name}: {:?}",
            report.findings
        );
    }
}

/// `cross_protocol_cycle_is_a_deadlock_not_a_false_clean`.
#[test]
fn cross_protocol_cycle_deadlocks() {
    let mut log = LogBuilder::new();
    log.push(0, 1, SyncOp::NamedSync { bar: 2, expected: 64, count: 32 })
        .push(0, 2, SyncOp::ClusterArrive { bar: 0, participants: 2 })
        .push(0, 3, SyncOp::ClusterWait { bar: 0 })
        .push(1, 2, SyncOp::ClusterArrive { bar: 0, participants: 2 })
        .push(1, 3, SyncOp::ClusterWait { bar: 0 })
        .push(1, 1, SyncOp::NamedSync { bar: 2, expected: 64, count: 32 });
    for (name, report) in assert_all(&log.build(), Verdict::Error) {
        let Finding::Deadlock { unfinished_warps, .. } = &report.findings[0] else {
            panic!("{name}: {report:#?}")
        };
        assert_eq!(unfinished_warps, &[0, 1], "{name}");
    }
}

/// Repeated `cta_sync` generations are certified without enumerating
/// arrival permutations (`repeated_named_barrier_generations_do_not_enumerate_arrival_permutations`).
#[test]
fn repeated_cta_sync_is_certified() {
    let warps = 8;
    let mut log = LogBuilder::new();
    for _ in 0..16 {
        cta_sync(&mut log, warps);
    }
    let reports = assert_all(&log.build(), Verdict::Clean);
    assert_eq!(reports[0].1.stats.visited_states, 1);
    // Without certificates, strong diamonds keep the search linear per generation.
    assert!(reports[1].1.stats.visited_states <= 16 * (warps as usize + 1) + 1, "{:?}", reports[1].1.stats);
}

/// A warp that `bar.arrive`s twice without synchronizing can over-arrive into
/// one generation in another schedule.
#[test]
fn named_arrive_reuse_without_ordering_is_an_error() {
    let mut log = LogBuilder::new();
    log.push(0, 1, SyncOp::NamedArrive { bar: 1, expected: 64, count: 32 })
        .push(0, 1, SyncOp::NamedArrive { bar: 1, expected: 64, count: 32 })
        .push(1, 2, SyncOp::NamedSync { bar: 1, expected: 64, count: 32 })
        .push(1, 2, SyncOp::NamedSync { bar: 1, expected: 64, count: 32 });
    let reports = assert_all(&log.build(), Verdict::Error);
    assert_eq!(reports[0].1.findings[0].kind(), "named_barrier_generation_not_ordered");
    // In the exhaustive search warp 0's two arrivals fill generation 0, so
    // warp 1's syncs can never complete.
    assert_eq!(reports[1].1.findings[0].kind(), "deadlock");
}

/// Cluster barrier: a missing participant at exit is incomplete, not clean
/// (`cluster_barrier_warp_exit_unmodeled`).
#[test]
fn cluster_missing_participant_is_incomplete() {
    let mut log = LogBuilder::new();
    log.push(0, 1, SyncOp::ClusterArrive { bar: 0, participants: 2 });
    for (name, report) in assert_all(&log.build(), Verdict::Incomplete) {
        let IncompleteReason::ProgramModel { kind, .. } = &report.incomplete[0] else {
            panic!("{name}: {report:#?}")
        };
        assert_eq!(*kind, "cluster_barrier_warp_exit_unmodeled", "{name}");
    }
}

/// `test_public_native_synccheck_transaction_over_delivery_is_exact_error`.
#[test]
fn transaction_over_delivery_is_typed() {
    let mut log = LogBuilder::new();
    log.push(0, 1, SyncOp::MbarInit { bar: 0, expected: 1 })
        .push(0, 2, SyncOp::MbarArrive { bar: 0, count: 1, expect_tx: 16 })
        .push(0, 3, SyncOp::MbarTxIssue { bar: 0, tx: 32 })
        .push(0, 4, SyncOp::MbarWait { bar: 0, parity: 0 });
    for (name, report) in assert_all(&log.build(), Verdict::Error) {
        assert_eq!(report.findings[0].kind(), "mbarrier_transaction_over_delivery", "{name}");
    }
}

/// `test_native_synccheck_reduces_tma_completion_waiter_interleavings`
/// (`tma_completion_with_many_waiters_stays_below_fixed_state_budget`):
/// 16 warps wait on one TMA barrier; with reductions the search stays tiny.
#[test]
fn tma_completion_many_waiters_stays_small() {
    let warps = 16;
    let mut log = LogBuilder::new();
    log.push(0, 1, SyncOp::MbarInit { bar: 0, expected: 1 });
    cta_sync(&mut log, warps);
    log.push(0, 2, SyncOp::MbarArrive { bar: 0, count: 1, expect_tx: 1024 })
        .push(0, 3, SyncOp::MbarTxIssue { bar: 0, tx: 1024 });
    for warp in 0..warps {
        log.push(warp, 4, SyncOp::MbarWait { bar: 0, parity: 0 });
    }
    let events = log.build();
    let config = CheckConfig {
        certificates: false,
        limits: Limits { max_states: 64, max_transitions: 1_000_000 },
        ..CheckConfig::default()
    };
    let report = check(&events, &config);
    assert_eq!(report.verdict, Verdict::Clean, "{report:#?}");
    assert_eq!(report.termination, "worklist_exhausted");
    // The Python test pins visited <= 32 and transitions <= 64 for the whole
    // launch with the named barrier certified. Here both projections are
    // searched: the cta_sync projection is a strong-diamond chain of
    // warps + 2 states, the TMA barrier projection takes the rest.
    assert!(report.stats.visited_states <= 32 + warps as usize + 2, "{:?}", report.stats);
    assert!(report.stats.explored_transitions <= 64, "{:?}", report.stats);
    let certified = check(&events, &CheckConfig::default());
    assert_eq!(certified.verdict, Verdict::Clean);
    assert_eq!(certified.stats.visited_states, 2);
}

/// Budget exhaustion is `incomplete` with `resource_limit` coverage, never clean.
#[test]
fn budget_exhaustion_is_incomplete() {
    let events = pipeline(PipelineShape { warps: 6, stages: 2, iterations: 8, tma_bytes: 0 });
    let config = CheckConfig {
        mode: ProjectionMode::Whole,
        certificates: false,
        explore: Options::NONE,
        limits: Limits { max_states: 500, max_transitions: 1_000_000 },
        ..CheckConfig::default()
    };
    let report = check(&events, &config);
    assert_eq!(report.verdict, Verdict::Incomplete);
    assert_eq!(report.termination, "resource_limit");
    assert!(matches!(report.incomplete[0], IncompleteReason::StateLimit { limit: 500, .. }));

    let report = check(&events, &CheckConfig { limits: Limits { max_states: 500, max_transitions: 1_000_000 }, ..CheckConfig::default() });
    assert_eq!(report.verdict, Verdict::Clean, "projection + certificates fit the same budget");
}

/// Isomorphic stage projections are verified once.
#[test]
fn fingerprint_reuses_isomorphic_stage_projections() {
    let events = pipeline(PipelineShape { warps: 4, stages: 4, iterations: 16, tma_bytes: 0 });
    let config = CheckConfig { certificates: false, ..CheckConfig::default() };
    let report = check(&events, &config);
    assert_eq!(report.verdict, Verdict::Clean);
    // 4 full + 4 empty + cta_sync; full[1..3] and empty[1..3] reuse full[0]/empty[0].
    assert_eq!(report.stats.programs, 9);
    assert_eq!(report.stats.reused_clean_programs, 6);
}

/// Duplicate per-warp sequence numbers cannot form a fixed program.
#[test]
fn malformed_log_is_program_build_incomplete() {
    let events = vec![
        SyncEvent::new(0, 0, 1, SyncOp::MbarInit { bar: 0, expected: 1 }),
        SyncEvent::new(0, 0, 2, SyncOp::MbarArrive { bar: 0, count: 1, expect_tx: 0 }),
    ];
    let report = check(&events, &CheckConfig::default());
    assert_eq!(report.verdict, Verdict::Incomplete);
    assert!(matches!(report.incomplete[0], IncompleteReason::ProgramBuild { .. }));
}
