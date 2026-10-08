//! Translated from test_native_tcgen_thread_fence.py,
//! test_native_async_lifetime_contracts.py (tcgen rows) and
//! test_native_same_warp_tmem_review.py.
#[path = "racecheck_common/mod.rs"]
mod common;
use common::*;

/// A tcgen05 op writing TMEM bytes `r` (and optionally reading smem).
fn tc(k: &mut K, w: WarpId, kind: AsyncKind, preds: &[AsyncId], write: Option<std::ops::Range<u64>>, read: Option<std::ops::Range<u64>>) -> AsyncId {
    let op = k.issue(w, 0, kind, Proxy::Tcgen, preds, &[(TMEM, 0..4096)]);
    if let Some(r) = read {
        k.aacc(op, Milestone::Read, AccessKind::Read, Proxy::Tcgen, TMEM, r);
    }
    if let Some(r) = write {
        k.aacc(op, Milestone::Write, AccessKind::Write, Proxy::Tcgen, TMEM, r);
    }
    op
}

#[derive(Clone, Copy)]
enum Handoff {
    CtaSync,
    RelaxedFlagWaitUntil,
    MbarArrive,
    MbarArriveRelaxedTestWait,
    MbarArriveRelaxedCluster,
}

/// cp→mma handoff across threads with the fence pair.
fn cp_mma(h: Handoff, before: bool, after: bool) -> Report {
    let mut k = K::new(2, 1, 1);
    let cp = tc(&mut k, 0, AsyncKind::TcgenPipelined, &[], Some(0..16), None);
    if before {
        k.fence(0, 1, FenceKind::TcgenBefore);
    }
    match h {
        Handoff::CtaSync => {
            k.bar(0, &[0, 1]);
        }
        Handoff::RelaxedFlagWaitUntil => {
            k.declare(GMEM, 0..4);
            k.a(0, 0, st(MemOrder::Relaxed, Scope::Cta), GMEM, 0..4);
            k.wait_until(1, 0, GMEM, 0..4, Scope::Cta, 0b10, 1);
        }
        Handoff::MbarArrive => {
            k.arrive(0, 1, 9, 0, true).wait(1, 1, 9, 0, true);
        }
        Handoff::MbarArriveRelaxedTestWait | Handoff::MbarArriveRelaxedCluster => {
            k.arrive(0, 1, 9, 0, false).wait(1, 1, 9, 0, false);
        }
    }
    if after {
        k.fence(1, 1, FenceKind::TcgenAfter);
    }
    let mma = tc(&mut k, 1, AsyncKind::TcgenPipelined, &[], Some(0..32), None);
    let c = k.issue(1, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &[mma], &[]);
    k.done_phase(c, Milestone::Write, 10, 0).wait(1, 1, 10, 0, true);
    let c0 = k.issue(0, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &[cp], &[]);
    k.done_phase(c0, Milestone::Write, 11, 0);
    k.run()
}

#[test]
fn cp_to_mma_handoff_needs_both_fences() {
    for h in [
        Handoff::CtaSync,
        Handoff::RelaxedFlagWaitUntil,
        Handoff::MbarArrive,
        Handoff::MbarArriveRelaxedTestWait,
        Handoff::MbarArriveRelaxedCluster,
    ] {
        assert!(clean(&cp_mma(h, true, true)));
        assert!(has_race(&cp_mma(h, false, true)));
        assert!(has_race(&cp_mma(h, true, false)));
    }
}

#[test]
fn repeated_fence_handoff_stays_clean() {
    let mut k = K::new(2, 1, 1);
    let _cp = tc(&mut k, 0, AsyncKind::TcgenPipelined, &[], Some(0..16), None);
    for _ in 0..128 {
        k.fence(0, 1, FenceKind::TcgenBefore).bar(0, &[0, 1]).fence(1, 1, FenceKind::TcgenAfter).bar(0, &[0, 1]);
    }
    tc(&mut k, 1, AsyncKind::TcgenPipelined, &[], Some(0..32), None);
    let r = k.run();
    assert!(r.findings.is_empty());
}

/// same-thread non-pipelined cp pair[ordering].
fn cp_pair(mode: u8) -> Report {
    let mut k = K::one_warp();
    let cp1 = tc(&mut k, 0, AsyncKind::TcgenPipelined, &[], Some(0..16), None);
    let mut late_wait = None;
    match mode {
        0 => {}
        1 => {
            k.fence(0, 1, FenceKind::TcgenBefore);
        }
        2 => {
            k.fence(0, 1, FenceKind::TcgenAfter);
        }
        3 => {
            let c = k.issue(0, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &[cp1], &[]);
            late_wait = Some(c);
        }
        4 | 5 => {
            let c = k.issue(0, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &[cp1], &[]);
            k.done_phase(c, Milestone::Write, 0, 0).wait(0, u32::MAX, 0, 0, true);
            if mode == 4 {
                k.fence(0, u32::MAX, FenceKind::TcgenAfter);
            }
        }
        _ => unreachable!(),
    }
    let cp2 = tc(&mut k, 0, AsyncKind::TcgenPipelined, &[], Some(0..16), None);
    if let Some(c) = late_wait {
        k.done_phase(c, Milestone::Write, 0, 0).wait(0, 1, 0, 0, true);
    }
    // The final commit tracks every not-yet-committed op of the thread.
    let tracked: Vec<AsyncId> = if mode <= 2 { vec![cp1, cp2] } else { vec![cp2] };
    let c = k.issue(0, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &tracked, &[]);
    k.done_phase(c, Milestone::Write, 1, 0).wait(0, 1, 1, 0, true);
    k.run()
}

#[test]
fn same_thread_cp_pair_ordering() {
    assert!(has_race(&cp_pair(0)));
    assert!(clean(&cp_pair(1)));
    assert!(has_race(&cp_pair(2)));
    assert!(clean(&cp_pair(3)));
    assert!(clean(&cp_pair(4)));
    assert!(clean(&cp_pair(5)));
}

/// cross-thread ld→st needs the producer's wait::ld.
fn ld_st(wait_ld: bool) -> Report {
    let mut k = K::new(2, 1, 1);
    let l = tc(&mut k, 0, AsyncKind::TcgenLd, &[], None, Some(0..16));
    if wait_ld {
        k.done_warp(l, Milestone::Write, 0, u32::MAX);
    }
    k.fence(0, u32::MAX, FenceKind::TcgenBefore).bar(0, &[0, 1]).fence(1, u32::MAX, FenceKind::TcgenAfter);
    let s = tc(&mut k, 1, AsyncKind::TcgenSt, &[], Some(0..16), None);
    k.done_warp(s, Milestone::Write, 1, u32::MAX);
    if !wait_ld {
        k.done_warp(l, Milestone::Write, 0, u32::MAX);
    }
    k.run()
}

#[test]
fn cross_thread_ld_to_st_needs_producer_wait() {
    let r = ld_st(false);
    assert!(review_only(&r), "{r:?}");
    assert!(clean(&ld_st(true)));
}

/// commit forwards only issued work (and that work's causal predecessors).
fn commit_forwards(local_work: bool) -> Report {
    let mut k = K::new(3, 1, 1);
    let cp1 = tc(&mut k, 0, AsyncKind::TcgenPipelined, &[], Some(0..16), None);
    k.fence(0, 1, FenceKind::TcgenBefore).arrive(0, 1, 20, 0, true);
    let c0 = k.issue(0, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &[cp1], &[]);
    k.wait(1, 1, 20, 0, true).fence(1, 1, FenceKind::TcgenAfter);
    let mut preds = vec![];
    if local_work {
        preds.push(tc(&mut k, 1, AsyncKind::TcgenPipelined, &[], Some(64..96), None));
    }
    let c1 = k.issue(1, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &preds, &[]);
    k.done_phase(c1, Milestone::Write, 21, 0);
    k.wait(2, 1, 21, 0, true).fence(2, 1, FenceKind::TcgenAfter);
    let cp2 = tc(&mut k, 2, AsyncKind::TcgenPipelined, &[], Some(0..16), None);
    let c2 = k.issue(2, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &[cp2], &[]);
    k.done_phase(c0, Milestone::Write, 22, 0).done_phase(c2, Milestone::Write, 23, 0);
    k.run()
}

#[test]
fn commit_forwards_only_issued_work() {
    assert!(has_race(&commit_forwards(false)));
    assert!(clean(&commit_forwards(true)));
}

/// tcgen transfer lifetime (same warp): 0 ld,st  1 ld,wait,st  2 st,ld
/// 3 st,wait,ld  4 st,st.
fn lifetime(mode: u8) -> Report {
    let mut k = K::one_warp();
    let (first, second) = match mode {
        0 | 1 => (AsyncKind::TcgenLd, AsyncKind::TcgenSt),
        2 | 3 => (AsyncKind::TcgenSt, AsyncKind::TcgenLd),
        _ => (AsyncKind::TcgenSt, AsyncKind::TcgenSt),
    };
    let rw = |kind| if kind == AsyncKind::TcgenLd { (None, Some(0..16)) } else { (Some(0..16), None) };
    let (w1, r1) = rw(first);
    let a = tc(&mut k, 0, first, &[], w1, r1);
    if mode == 1 || mode == 3 {
        k.done_warp(a, Milestone::Write, 0, u32::MAX);
    }
    let (w2, r2) = rw(second);
    let b = tc(&mut k, 0, second, &[], w2, r2);
    k.done_warp(b, Milestone::Write, 0, u32::MAX);
    if !(mode == 1 || mode == 3) {
        k.done_warp(a, Milestone::Write, 0, u32::MAX);
    }
    k.run()
}

#[test]
fn tcgen_transfer_lifetime() {
    let r0 = lifetime(0);
    assert!(review_only(&r0));
    assert!(matches!(r0.findings[0].kind, FindingKind::TmemLifetimeReview { class: RaceClass::ReadWrite, failure: OrderingFailure::AsyncLifetimeNotDrained }));
    assert!(clean(&lifetime(1)));
    let r2 = lifetime(2);
    assert!(has_failure(&r2, |f| f == OrderingFailure::AsyncLifetimeNotDrained) && has_class(&r2, RaceClass::WriteRead));
    assert!(clean(&lifetime(3)));
    let r4 = lifetime(4);
    assert!(has_class(&r4, RaceClass::WriteWrite));
}

#[test]
fn tcgen_waits_do_not_mask_cross_warpgroup_order() {
    let mut k = K::new(2, 1, 1);
    let l = tc(&mut k, 0, AsyncKind::TcgenLd, &[], None, Some(0..16));
    k.done_warp(l, Milestone::Write, 0, u32::MAX);
    let s = tc(&mut k, 1, AsyncKind::TcgenSt, &[], Some(0..16), None);
    k.done_warp(s, Milestone::Write, 1, u32::MAX);
    let r = k.run();
    assert!(has_failure(&r, |f| f == OrderingFailure::MissingInterActorSync));
}

/// tcgen copy completion vs generic reuse of the smem source.
fn copy_vs_reuse(fence: bool) -> Report {
    let mut k = K::new(2, 1, 1);
    k.st(0, 0, SMEM, 0..64).fence(0, u32::MAX, FenceKind::ProxyAsync(Some(Domain::SharedCta))).bar(0, &[0, 1]);
    let cp = k.issue(0, 0, AsyncKind::TcgenPipelined, Proxy::Tcgen, &[], &[(SMEM, 0..64)]);
    k.ar(cp, Proxy::Async, SMEM, 0..64).aacc(cp, Milestone::Write, AccessKind::Write, Proxy::Tcgen, TMEM, 0..64);
    let c = k.issue(0, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &[cp], &[]);
    k.done_phase(c, Milestone::Write, 3, 0).wait(1, u32::MAX, 3, 0, true);
    if fence {
        k.fence(1, u32::MAX, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    }
    k.st(1, 0, SMEM, 0..4);
    k.run()
}

#[test]
fn tcgen_copy_completion_vs_generic_reuse() {
    let r = copy_vs_reuse(false);
    assert!(has_failure(&r, |f| matches!(f, OrderingFailure::MissingProxyBridge { prior: Proxy::Async, current: Proxy::Generic, domain: Some(Domain::SharedCta) })));
    assert!(clean(&copy_vs_reuse(true)));
}

/// tcgen commit forwards the issuer's generic publication.
fn commit_publication(acquire: bool) -> Report {
    let mut k = K::new(3, 1, 1);
    k.st(0, 0, GMEM, 0..4).a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, 0..4);
    k.a(1, 0, ld(if acquire { MemOrder::Acquire } else { MemOrder::Relaxed }, Scope::Gpu), GMEM2, 0..4);
    let c = k.issue(1, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &[], &[]);
    k.done_phase(c, Milestone::Write, 5, 0).wait(2, 1, 5, 0, true).ld(2, 0, GMEM, 0..4);
    k.run()
}

#[test]
fn tcgen_commit_forwards_issuer_publication() {
    assert!(clean(&commit_publication(true)));
    assert!(has_class(&commit_publication(false), RaceClass::WriteRead));
}

/// `tcgen05.commit...sync_restrict::shared::read::mma::a` (contract shape of
/// CONTRACT_REQUESTS W5-10): the mma's shared-A read is its own async op `ma`
/// (preds `[]`), the restricted commit tracks only `ma`, and the mma itself
/// stays tracked by the next unrestricted commit. Waiting on the restricted
/// commit orders A's read, never B's (legacy `MmaSharedARead`).
fn restricted_commit(reuse_b: bool, shape_ok: bool) -> Report {
    const A: std::ops::Range<u64> = 512..1024;
    const B: std::ops::Range<u64> = 1024..1536;
    let mut k = K::one_warp();
    let mma = k.issue(0, 0, AsyncKind::TcgenPipelined, Proxy::Tcgen, &[], &[(SMEM, 512..1536), (TMEM, 0..4096)]);
    let ma = if shape_ok {
        let ma = k.issue(0, 0, AsyncKind::TcgenPipelined, Proxy::Tcgen, &[], &[(SMEM, A)]);
        k.aacc(ma, Milestone::Read, AccessKind::Read, Proxy::Async, SMEM, A);
        k.aacc(mma, Milestone::Read, AccessKind::Read, Proxy::Async, SMEM, B);
        ma
    } else {
        // Today's engine shape: one op reads A and B, the restricted commit
        // tracks the whole mma (cannot tell A from B).
        k.aacc(mma, Milestone::Read, AccessKind::Read, Proxy::Async, SMEM, A.start..B.end);
        mma
    };
    k.aacc(mma, Milestone::Write, AccessKind::Write, Proxy::Tcgen, TMEM, 0..64);
    let c = k.issue(0, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &[ma], &[]);
    k.done_phase(c, Milestone::Write, 1, 0).wait(0, 1, 1, 0, true);
    k.fence(0, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    k.st(0, 0, SMEM, if reuse_b { B.start..B.start + 2 } else { A.start..A.start + 2 });
    let tracked: Vec<AsyncId> = if shape_ok { vec![mma] } else { vec![] };
    let c = k.issue(0, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &tracked, &[]);
    k.done_phase(c, Milestone::Write, 0, 0).wait(0, 1, 0, 0, true);
    k.run()
}

#[test]
fn restricted_commit_publishes_only_shared_a_read() {
    assert!(clean(&restricted_commit(false, true)));
    assert!(has_race(&restricted_commit(true, true)));
    // Today's shape over-publishes: the false negative is in the events,
    // not the checker (W5-10).
    assert!(!has_race(&restricted_commit(true, false)));
}

/// V2C-5: a tcgen05.ld never followed by `tcgen05.wait::ld` (kernels that
/// wait only on one subtile) is not `AsyncNeverCompleted`: its TMEM read was
/// delivered at issue. It stays unordered: a later TMEM write races.
#[test]
fn unwaited_tcgen_ld_is_not_incomplete() {
    let mut k = K::one_warp();
    let ld = k.issue(0, 0, AsyncKind::TcgenLd, Proxy::Tcgen, &[], &[]);
    k.aacc(ld, Milestone::Read, AccessKind::Read, Proxy::Tcgen, TMEM, 0..64);
    let r = k.run();
    assert!(r.incomplete.is_empty() && r.findings.is_empty(), "{r:?}");
    let mut k = K::one_warp();
    let ld = k.issue(0, 0, AsyncKind::TcgenLd, Proxy::Tcgen, &[], &[]);
    k.aacc(ld, Milestone::Read, AccessKind::Read, Proxy::Tcgen, TMEM, 0..64);
    let st = k.issue(0, 0, AsyncKind::TcgenSt, Proxy::Tcgen, &[], &[]);
    k.aacc(st, Milestone::Write, AccessKind::Write, Proxy::Tcgen, TMEM, 0..64);
    let r = k.run();
    assert!(!r.findings.is_empty(), "{r:?}");
}
