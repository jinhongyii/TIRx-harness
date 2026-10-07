//! Translated from tests/analysis_tools/racecheck/test_native_proxy_async_fence.py
//! and the proxy rows of test_native_shared_publication.py.
mod common;
use common::*;
use numsim_race_core::input::*;
use numsim_race_core::*;

const BAR: SyncObjId = 7;

/// global_proxy_fence[mode]: generic st.global, optional fence, TMA reads
/// the same global bytes through the async proxy.
fn global_proxy_fence(fence: Option<FenceKind>) -> Report {
    let mut k = K::one_warp();
    k.st(0, 0, GMEM, 0..4);
    if let Some(f) = fence {
        k.fence(0, 1, f);
    }
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, GMEM, 0..16).aw(op, Proxy::Async, SMEM, 0..16);
    k.done_phase(op, Milestone::Write, BAR, 0).wait(0, 1, BAR, 0, true).ld(0, 0, SMEM, 0..4);
    k.run()
}

#[test]
fn global_proxy_fence_modes() {
    let none = global_proxy_fence(None);
    assert_eq!(races(&none).len(), 1);
    assert!(has_class(&none, RaceClass::WriteRead));
    assert_eq!(races(&none)[0].bytes, 0..4);
    assert!(has_failure(&none, |f| matches!(f, OrderingFailure::MissingProxyBridge { prior: Proxy::Generic, current: Proxy::Async, .. })));
    assert!(clean(&global_proxy_fence(Some(FenceKind::ProxyAsync(Some(Domain::Global))))));
    assert!(has_race(&global_proxy_fence(Some(FenceKind::ProxyAsync(Some(Domain::SharedCta))))));
    assert!(clean(&global_proxy_fence(Some(FenceKind::ProxyAsync(None)))));
}

/// shared_cta_proxy_fence[mode]: byte stores to smem, fence, bulk S2G.
fn shared_cta_proxy_fence(fence: Option<FenceKind>, fence_first: bool) -> Report {
    let mut k = K::one_warp();
    if fence_first {
        k.fence(0, 1, fence.unwrap());
    }
    for i in 0..16 {
        k.st(0, 0, SMEM, i..i + 1);
    }
    if let (Some(f), false) = (fence, fence_first) {
        k.fence(0, 1, f);
    }
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, SMEM, 0..16).aw(op, Proxy::Async, GMEM, 0..16);
    k.done_warp(op, Milestone::Write, 0, 1).ld(0, 0, GMEM, 0..4).st(0, 0, GMEM, 0..4);
    k.run()
}

#[test]
fn shared_cta_proxy_fence_modes() {
    let none = shared_cta_proxy_fence(None, false);
    assert!(has_class(&none, RaceClass::WriteRead));
    assert!(clean(&shared_cta_proxy_fence(Some(FenceKind::ProxyAsync(Some(Domain::SharedCta))), false)));
    assert!(has_race(&shared_cta_proxy_fence(Some(FenceKind::ProxyAsync(Some(Domain::SharedCluster))), false)));
    assert!(clean(&shared_cta_proxy_fence(Some(FenceKind::ProxyAsync(None)), false)));
    assert!(has_race(&shared_cta_proxy_fence(Some(FenceKind::ProxyAsync(Some(Domain::SharedCta))), true)));
}

/// same_rank_mapa_shared_cluster: the generic store goes through a
/// shared::cluster window; the fence must name that window.
fn mapa(fence: Option<Domain>, unqualified: bool) -> Report {
    let mut k = K::one_warp();
    k.inst_in(0, &[0], PLAIN_ST, Some(Domain::SharedCluster), |_| (SMEM, 0..4));
    if unqualified {
        k.fence(0, 1, FenceKind::ProxyAsync(None));
    } else if let Some(d) = fence {
        k.fence(0, 1, FenceKind::ProxyAsync(Some(d)));
    }
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, GMEM, 0..16).aw(op, Proxy::Async, SMEM, 0..16);
    k.done_phase(op, Milestone::Write, BAR, 0).wait(0, 1, BAR, 0, true);
    k.run()
}

#[test]
fn same_rank_mapa_shared_cluster() {
    let none = mapa(None, false);
    assert!(has_class(&none, RaceClass::WriteWrite));
    assert!(clean(&mapa(Some(Domain::SharedCluster), false)));
    assert!(has_race(&mapa(Some(Domain::SharedCta), false)));
    assert!(clean(&mapa(None, true)));
}

/// proxy_fence_does_not_escape_active_lane: lane 0 fences, lane 1 copies.
#[test]
fn proxy_fence_does_not_escape_active_lane() {
    let mut k = K::new(2, 1, 1);
    k.st(0, 0, SMEM, 0..16).arrive(0, 1, BAR, 0, true);
    k.wait(1, 0b10, BAR, 0, true).fence(1, 0b01, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    let op = k.issue(1, 1, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, SMEM, 0..16).aw(op, Proxy::Async, GMEM, 0..16).done_warp(op, Milestone::Write, 1, 0b10);
    let r = k.run();
    let f = races(&r);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].prior.as_ref().unwrap().warp, 0);
    assert_eq!(f[0].current.as_ref().unwrap().warp, 1);
    assert_eq!(f[0].current.as_ref().unwrap().lane, 1);
    // Control: the copying lane fences.
    let mut k = K::new(2, 1, 1);
    k.st(0, 0, SMEM, 0..16).arrive(0, 1, BAR, 0, true);
    k.wait(1, 0b10, BAR, 0, true).fence(1, 0b10, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    let op = k.issue(1, 1, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, SMEM, 0..16).aw(op, Proxy::Async, GMEM, 0..16).done_warp(op, Milestone::Write, 1, 0b10);
    assert!(clean(&k.run()));
}

/// mbarrier_completion_acquire_lane: only the waiting lane acquires the
/// TMA completion.
fn completion_lane(wait_lane: u8) -> Report {
    let mut k = K::new(2, 1, 1);
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, GMEM, 0..16).aw(op, Proxy::Async, SMEM, 0..16).done_phase(op, Milestone::Write, BAR, 0);
    k.wait(1, 1 << wait_lane, BAR, 0, true).ld(1, 1, SMEM, 0..4);
    k.run()
}

#[test]
fn mbarrier_completion_acquire_lane() {
    let r = completion_lane(0);
    let f = races(&r);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].prior.as_ref().unwrap().warp, 0, "async write attributed to its issuer");
    assert!(f[0].prior.as_ref().unwrap().async_op.is_some());
    assert!(clean(&completion_lane(1)));
}

/// raw try_wait variants: async completion is visible to a relaxed query,
/// a generic arrive's release is not.
fn try_wait(acquire: bool, read_ordinary: bool) -> Report {
    let mut k = K::new(2, 1, 1);
    k.st(0, 0, GMEM, 0..4).arrive(0, 1, BAR, 0, true);
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, GMEM2, 0..16).aw(op, Proxy::Async, SMEM, 0..16).done_phase(op, Milestone::Write, BAR, 0);
    k.wait(1, 1, BAR, 0, acquire);
    if read_ordinary {
        k.ld(1, 0, GMEM, 0..4);
    } else {
        k.ld(1, 0, SMEM, 0..4);
    }
    k.run()
}

#[test]
fn raw_try_wait_acquire_variants() {
    assert!(clean(&try_wait(false, false)));
    assert!(clean(&try_wait(true, false)));
    assert!(has_class(&try_wait(false, true), RaceClass::WriteRead));
    assert!(clean(&try_wait(true, true)));
}

/// clc_response_reuse: generic read of an async-written response, then a
/// second async write into it needs a generic→async fence (WAR).
fn clc(fence: bool) -> Report {
    let mut k = K::one_warp();
    let a = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.aw(a, Proxy::Async, SMEM, 0..16).done_phase(a, Milestone::Write, BAR, 0).wait(0, 1, BAR, 0, true);
    k.a(0, 0, ld(MemOrder::Acquire, Scope::Cta), SMEM, 0..16);
    if fence {
        k.fence(0, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    }
    let b = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.aw(b, Proxy::Async, SMEM, 0..16).done_phase(b, Milestone::Write, BAR, 1).wait(0, 1, BAR, 1, true);
    k.ld(0, 0, SMEM, 0..16);
    k.run()
}

#[test]
fn clc_response_reuse() {
    let r = clc(false);
    assert!(has_class(&r, RaceClass::ReadWrite));
    assert!(clean(&clc(true)));
}

/// v30 Q/V alias anti-dependency: the producer's earlier fence does not
/// cover a generic read it acquired afterwards.
#[test]
fn v30_alias_anti_dependency() {
    let mut k = K::new(2, 1, 1);
    k.st(0, 0, SMEM, 0..4).fence(0, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta))).arrive(0, 1, 1, 0, true);
    k.wait(1, 1, 1, 0, true).ld(1, 0, SMEM, 0..4).arrive(1, 1, 2, 0, true);
    k.wait(0, 1, 2, 0, true);
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, GMEM, 0..16).aw(op, Proxy::Async, SMEM, 0..16).done_phase(op, Milestone::Write, BAR, 0).wait(0, 1, BAR, 0, true);
    let r = k.run();
    let f = races(&r);
    assert_eq!(f.len(), 1);
    assert!(has_class(&r, RaceClass::ReadWrite));
    assert_eq!(f[0].prior.as_ref().unwrap().warp, 1);
}

/// shared→async handoff (test_native_shared_publication.py): writer lane w
/// of W0, W0L0 publishes with st.release, W1L0 acquires, W1 lane r copies.
fn handoff(writer: u8, reader: u8, fence_side: &str) -> Report {
    let mut k = K::new(2, 1, 1);
    if fence_side == "early_producer" {
        k.fence(0, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    }
    k.st(0, writer, SMEM, 0..16);
    if fence_side == "producer" {
        k.fence(0, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    }
    k.a(0, 0, st(MemOrder::Release, Scope::Cta), GMEM, 0..4);
    if fence_side == "early_consumer" {
        k.fence(1, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    }
    k.a(1, 0, ld(MemOrder::Acquire, Scope::Cta), GMEM, 0..4);
    if fence_side == "consumer" {
        k.fence(1, 1 << reader, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    }
    let op = k.issue(1, reader, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, SMEM, 0..16).aw(op, Proxy::Async, GMEM2, 0..16).done_warp(op, Milestone::Write, 1, 1 << reader);
    k.run()
}

#[test]
fn shared_to_async_handoff_matrix() {
    for side in ["producer", "consumer"] {
        assert!(clean(&handoff(0, 0, side)), "{side} (0,0)");
        assert!(has_race(&handoff(1, 0, side)), "{side} (1,0)");
        assert!(has_race(&handoff(0, 1, side)), "{side} (0,1)");
    }
    assert!(has_race(&handoff(0, 0, "early_producer")));
    assert!(has_race(&handoff(0, 0, "early_consumer")));
}
