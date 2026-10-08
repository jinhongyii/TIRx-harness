//! Regression tests for docs/development/checker-review.md (racecheck items),
//! built from contract events through `RaceObserver`.
#[path = "racecheck_common/mod.rs"]
mod common;
use common::*;
use numsim_core::arena::Space;
use numsim_core::observe::CtaId;
use numsim_core::racecheck::{report, serialize};
use numsim_core::report::FindingKind as CK;
use numsim_core::sync::completion::ResourceId;

const FLAG: std::ops::Range<u64> = 0..4;

/// S2: a shared::cta store must not evict a shared::cluster (mapa) store:
/// the bridge for the async access is chosen by the prior's window.
#[test]
fn s2_eviction_respects_window_domain() {
    let run = |second_store: bool| {
        let mut k = K::one_warp();
        k.inst_in(0, &[0], PLAIN_ST, Some(Domain::SharedCluster), |_| (SMEM, 0..16));
        if second_store {
            k.st(0, 0, SMEM, 0..16);
        }
        k.fence(0, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
        let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
        k.ar(op, Proxy::Async, SMEM, 0..16).aw(op, Proxy::Async, GMEM, 0..16).done_warp(op, Milestone::Write, 0, 1);
        k.run()
    };
    assert!(has_failure(&run(false), |f| matches!(f, OrderingFailure::MissingProxyBridge { .. })));
    assert!(has_failure(&run(true), |f| matches!(f, OrderingFailure::MissingProxyBridge { domain: Some(Domain::SharedCluster), .. })));
}

/// S3: `scope: None` on an mbarrier / cluster barrier is a lost qualifier:
/// incomplete and no edge (named barriers are identified by ResourceId).
#[test]
fn s3_mbarrier_scope_none_is_lost_qualifier() {
    let mut k = K::new(1, 2, 2);
    k.st(1, 0, GMEM, 0..4);
    k.arrive_q(1, 1, mbar_in(0, 3), 0, Some(true), None);
    k.wait_q(0, 1, mbar_in(0, 3), 0, Some(true), Some(Scope::Cta)).ld(0, 0, GMEM, 0..4);
    let r = k.run();
    assert!(r.incomplete.iter().any(|i| matches!(i, Incomplete::SyncQualifierUnknown { .. })));
    assert!(has_race(&r), "no edge from a lost qualifier");
    // A named barrier with scope None is a full participant barrier.
    let mut k = K::new(2, 1, 1);
    k.st(1, 0, GMEM, 0..4).bar(3, &[0, 1]).ld(0, 0, GMEM, 0..4);
    assert!(clean(&k.run()));
}

/// S4: history numbering per (Access, lane), lanes ascending, for every
/// overlapping word.
#[test]
fn s4_declared_word_numbering() {
    let run = |accepted: u64| {
        let mut k = K::new(1, 1, 3);
        k.declare(GMEM2, FLAG);
        k.st(0, 0, GMEM, 0..4);
        k.inst(1, &[0, 1], atom(MemOrder::Relaxed, Scope::Gpu), |_| (GMEM2, FLAG)); // entries 1, 2
        k.a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, FLAG); // entry 3
        k.wait_until(2, 0, GMEM2, FLAG, Scope::Gpu, accepted, 3).ld(2, 0, GMEM, 0..4);
        k.run()
    };
    assert!(clean(&run(0b1000)), "entry 3 is the release");
    assert!(has_race(&run(0b0100)), "entry 2 is lane 1's relaxed atom");
    // An 8-byte release covering two declared words appends to both.
    let mut k = K::new(1, 1, 2);
    k.declare(GMEM2, 0..4).declare(GMEM2, 4..8);
    k.st(0, 0, GMEM, 0..4).a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, 0..8);
    k.wait_until(1, 0, GMEM2, 4..8, Scope::Gpu, 0b10, 1).ld(1, 0, GMEM, 0..4);
    assert!(clean(&k.run()));
}

/// S6: the tensormap acquire filters by the RELEASING thread.
#[test]
fn s6_tensormap_filters_by_releaser() {
    // X (CTA0) writes the descriptor and hands it to W (CTA1); W releases
    // at .cta and signals C (CTA0); C's .cta acquire does not include W.
    let missed = || {
        let mut k = K::new(2, 1, 2);
        k.st(0, 0, GMEM2, 0..128);
        k.a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM, 0..4).a(2, 0, ld(MemOrder::Acquire, Scope::Gpu), GMEM, 0..4);
        k.fence(2, 1, FenceKind::TensormapRelease { scope: Scope::Cta });
        k.a(2, 0, st(MemOrder::Release, Scope::Gpu), GMEM, 4..8).a(1, 0, ld(MemOrder::Acquire, Scope::Gpu), GMEM, 4..8);
        k.fence(1, 1, FenceKind::TensormapAcquire { scope: Scope::Cta, alloc: GMEM2, span: numsim_core::arena::ByteSpan::new(0, 128) });
        let op = k.issue(1, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
        k.ar(op, Proxy::TensorMap, GMEM2, 0..128).aw(op, Proxy::Async, SMEM, 0..16).done_warp(op, Milestone::Write, 1, 1);
        k.run()
    };
    assert!(has_race(&missed()));
    // Writer remote, releaser local: the local .cta release reaches C.
    let mut k = K::new(2, 1, 2);
    k.st(2, 0, GMEM2, 0..128);
    k.a(2, 0, st(MemOrder::Release, Scope::Gpu), GMEM, 0..4).a(0, 0, ld(MemOrder::Acquire, Scope::Gpu), GMEM, 0..4);
    k.fence(0, 1, FenceKind::TensormapRelease { scope: Scope::Cta }).bar(0, &[0, 1]);
    k.fence(1, 1, FenceKind::TensormapAcquire { scope: Scope::Cta, alloc: GMEM2, span: numsim_core::arena::ByteSpan::new(0, 128) });
    let op = k.issue(1, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::TensorMap, GMEM2, 0..128).aw(op, Proxy::Async, SMEM, 0..16).done_warp(op, Milestone::Write, 1, 1);
    let r = k.run();
    assert!(r.errors().next().is_none() && r.incomplete.is_empty(), "{r:?}");
}

/// S7 / F2: a failed mbarrier scope test reports one ScopeMismatch per site
/// pair with sites, counting occurrences.
#[test]
fn s7_mbarrier_scope_mismatch_is_reported_once() {
    let mut k = K::new(1, 2, 2);
    for p in 0..50u64 {
        k.st(1, 0, GMEM, 0..4);
        k.arrive_q(1, 1, mbar_in(0, 3), p, Some(true), Some(Scope::Cluster));
        k.wait_q(0, 1, mbar_in(0, 3), p, Some(true), Some(Scope::Cta)).ld(0, 0, GMEM, 0..4);
    }
    let obs = k.observe();
    let r = &obs.launches[0].report;
    let sm: Vec<_> = r.findings.iter().filter(|f| matches!(f.kind, FindingKind::ScopeMismatch { .. })).collect();
    assert_eq!(sm.len(), 50, "distinct arrive/wait sites per iteration in this builder");
    let rep = report(&obs);
    let f = rep.findings.iter().find(|f| f.kind == CK::ScopeMismatch).unwrap();
    assert_eq!(f.sites.len(), 2);
    // Same sites repeated: one finding with a count.
    let mut obs = RaceObserver::new(RacecheckConfig::default());
    obs.start_launch(Topology { warps_per_cta: 1, ctas_per_cluster: 2, num_ctas: 2 }, 0);
    use numsim_core::observe::{Actor, Observer, SyncEvent, SyncKind, WarpId as W};
    use numsim_core::site::SiteId;
    use numsim_core::value::WarpMask;
    for p in 0..20u64 {
        for (w, kind) in [
            (1u32, SyncKind::Arrive { obj: mbar_in(0, 3), phase: p, release: Some(true), scope: Some(Scope::Cluster) }),
            (0u32, SyncKind::Wait { obj: mbar_in(0, 3), phase: p, acquire: Some(true), scope: Some(Scope::Cta) }),
        ] {
            obs.sync(&SyncEvent {
                kernel: 0,
                actor: Actor::Warp { warp: W(w), epoch: 2 * p + 1 + w as u64 },
                seq: 0,
                site: SiteId(10 + w),
                frames: vec![],
                lanes: WarpMask(1),
                kind,
            });
        }
    }
    obs.finish_launch();
    let r = &obs.launches[0].report;
    let sm: Vec<_> = r.findings.iter().filter(|f| matches!(f.kind, FindingKind::ScopeMismatch { .. })).collect();
    assert_eq!(sm.len(), 1);
    assert_eq!(sm[0].occurrences, 20);
}

/// F3: fence.sc keeps the latest fence per (thread, scope).
#[test]
fn f3_fence_sc_per_scope() {
    let mut k = K::new(1, 1, 2);
    k.st(0, 0, GMEM, 0..4).fence(0, 1, FenceKind::Sc(Scope::Gpu)).fence(0, 1, FenceKind::Sc(Scope::Cta));
    k.fence(1, 1, FenceKind::Sc(Scope::Gpu)).ld(1, 0, GMEM, 0..4);
    assert!(clean(&k.run()));
}

/// F1: no CrossCtaAsyncOrder for a consumer CTA that observed the prior
/// op's completion itself (multicast / 2-CTA); still reported when the
/// order is base causality alone.
#[test]
fn f1_cross_cta_advisory_only_for_base_causality() {
    let multicast = || {
        let mut k = K::new(1, 2, 2);
        let a = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM1, 0..16)]);
        k.ar(a, Proxy::Async, GMEM, 0..16).aw(a, Proxy::Async, SMEM1, 0..16).done_phase_r(a, Milestone::Write, mbar_in(1, 1), 0);
        k.wait_r(1, 1, mbar_in(1, 1), 0, true);
        let b = k.issue(1, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM1, 0..16)]);
        k.ar(b, Proxy::Async, SMEM1, 0..16).aw(b, Proxy::Async, GMEM2, 0..16).done_warp(b, Milestone::Write, 1, 1);
        k.run()
    };
    let r = multicast();
    assert!(clean(&r) && !has_advisory(&r, AdvisoryKind::CrossCtaAsyncOrder), "{r:?}");
    // CTA0 observes the completion, then hands over through a cluster
    // barrier: base causality only.
    let mut k = K::new(1, 2, 2);
    let a = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM1, 0..16)]);
    k.ar(a, Proxy::Async, GMEM, 0..16).aw(a, Proxy::Async, SMEM1, 0..16).done_phase_r(a, Milestone::Write, mbar_in(0, 1), 0);
    k.wait_r(0, 1, mbar_in(0, 1), 0, true).cluster_bar(&[0, 1]);
    let b = k.issue(1, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM1, 0..16)]);
    k.aw(b, Proxy::Async, SMEM1, 0..16).done_warp(b, Milestone::Write, 1, 1);
    assert!(has_advisory(&k.run(), AdvisoryKind::CrossCtaAsyncOrder));
}

/// R1: an out-of-range warp in a warp-targeted completion is incomplete.
#[test]
fn r1_completion_warp_out_of_range() {
    let mut k = K::one_warp();
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, GMEM, 0..16).done_warp(op, Milestone::Write, 999, 1);
    let r = k.run();
    assert!(r.incomplete.iter().any(|i| matches!(i, Incomplete::CompletionWarpOutOfRange { warp: 999 })));
}

/// R2: repeated incompletes collapse into one entry with a count.
#[test]
fn r2_incompletes_are_deduplicated() {
    let mut k = K::new(1, 2, 2);
    for p in 0..500u64 {
        k.arrive_q(0, 1, ResourceId::Cluster { cluster: 0 }, p, None, Some(Scope::Cluster));
    }
    let obs = k.observe();
    let r = &obs.launches[0].report;
    assert_eq!(r.incomplete.len(), 1);
    assert_eq!(r.incomplete_counts[0], 500);
    let p = serialize(&report(&obs));
    assert_eq!(p["incomplete"][0]["occurrences"], 500);
}

/// R3: the wide-span table is bounded by distinct spans.
#[test]
fn r3_wide_spans_are_interned() {
    let mut k = K::new(1, 1, 1);
    let big = numsim_core::arena::AllocId(42);
    k.alloc(big, Space::Global, 1 << 40);
    for _ in 0..100 {
        k.st(0, 0, big, (1 << 33)..(1 << 33) + (1 << 20));
    }
    let obs = k.observe();
    assert_eq!(obs.launches[0].stats.wide_spans, 1);
}

/// R5: shared memory of a cluster retires without waiting for an
/// unrelated live cluster.
#[test]
fn r5_gc_is_per_cluster_for_shared_memory() {
    let mut k = K::new(2, 1, 2);
    k.gc_every = 8;
    k.alloc_cta(SMEM, 0);
    k.st(2, 0, GMEM2, 0..4); // CTA1's warp stays live and never syncs
    for i in 0..64u64 {
        let off = (i % 16) * 4;
        k.st(0, 0, SMEM, off..off + 4).bar(0, &[0, 1]).ld(1, 0, SMEM, off..off + 4).bar(1, &[0, 1]);
    }
    let obs = k.observe();
    let lr = &obs.launches[0];
    assert!(clean(&lr.report));
    assert!(lr.stats.witnesses_retired >= 64, "{:?}", lr.stats);
}

/// R6: chunk identities are process-global, so checkers on several threads
/// give the same result as one.
#[test]
fn r6_checkers_on_threads_agree() {
    let scenario = || {
        let mut k = K::new(4, 1, 1);
        for i in 0..32u64 {
            k.st((i % 4) as u32, 0, SMEM, i * 4..i * 4 + 4).bar(0, &[0, 1, 2, 3]);
        }
        k.st(1, 0, SMEM, 0..4).ld(2, 0, SMEM, 0..4);
        format!("{:?}", k.run().findings.iter().map(|f| (&f.kind, f.bytes.clone())).collect::<Vec<_>>())
    };
    let one = scenario();
    let handles: Vec<_> = (0..4).map(|_| std::thread::spawn(scenario)).collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), one);
    }
}

/// R7: a zero-length span is not out of bounds.
#[test]
fn r7_zero_length_span() {
    let mut k = K::one_warp();
    k.st(0, 0, SMEM, 4096..4096);
    assert!(clean(&k.run()));
}

/// Contract: events of another kernel are incomplete, not merged.
#[test]
fn kernel_mismatch_is_incomplete() {
    let mut k = K::new(2, 1, 1);
    k.kernel = 1; // the observer was started for kernel 0
    k.bar(0, &[0, 1]);
    let r = k.run();
    assert!(r.incomplete.iter().any(|i| matches!(i, Incomplete::KernelMismatch { expected: 0, got: 1 })));
}

/// Contract: `ALL_LANES` on a warp access is a warp-collective access,
/// attributed to lane 0 (not lane 31).
#[test]
fn all_lanes_warp_access_is_collective() {
    let mut k = K::one_warp();
    k.inst(0, &[numsim_core::observe::ALL_LANES], PLAIN_ST, |_| (SMEM, 0..64));
    k.ld(0, 0, SMEM, 0..4);
    assert!(clean(&k.run()));
}

#[allow(dead_code)]
fn _cta(c: u32) -> CtaId {
    CtaId(c)
}
