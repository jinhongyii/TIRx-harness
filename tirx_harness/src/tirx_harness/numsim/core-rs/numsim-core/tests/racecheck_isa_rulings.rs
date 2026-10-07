//! Cases pinning the rulings in docs/development/racecheck-isa-answers.md
//! (R1–R9) that the legacy test suite does not cover.
#[path = "racecheck_common/mod.rs"]
mod common;
use common::*;
use numsim_core::observe::CtaId;
use numsim_core::sync::completion::ResourceId;

const FLAG: std::ops::Range<u64> = 0..4;

/// R4: mbarrier edges need mutual scope inclusion. A remote-CTA
/// `arrive.release.cluster` observed by `try_wait.acquire.cta` gives no edge.
fn remote_arrive(wait_scope: Scope) -> Report {
    let mut k = K::new(1, 2, 2);
    k.st(1, 0, GMEM, 0..4);
    k.arrive_q(1, 1, mbar(3), 0, Some(true), Some(Scope::Cluster));
    k.wait_q(0, 1, mbar(3), 0, Some(true), Some(wait_scope)).ld(0, 0, GMEM, 0..4);
    k.run()
}

#[test]
fn r4_mbarrier_scope_mutual_inclusion() {
    assert!(has_race(&remote_arrive(Scope::Cta)));
    assert!(clean(&remote_arrive(Scope::Cluster)));
}

/// R4: complete-tx is release at cluster scope; a `.cta` wait still sees the
/// copy's own bytes, but not the issuer's earlier writes.
#[test]
fn r4_complete_tx_publishes_only_the_copy() {
    let mut k = K::new(1, 2, 2);
    k.st(1, 0, GMEM2, 0..4);
    let op = k.issue(1, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, GMEM, 0..16).aw(op, Proxy::Async, SMEM, 0..16).done_phase(op, Milestone::Write, 3, 0);
    k.wait(0, 1, 3, 0, true).ld(0, 0, SMEM, 0..16).ld(0, 0, GMEM2, 0..4);
    let r = k.run();
    let f = races(&r);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].alloc, GMEM2);
}

/// R5: `bar.arrive` is a source only; the arriving thread acquires nothing.
#[test]
fn r5_named_barrier_arrive_only_gets_no_acquire() {
    let mut k = K::new(2, 1, 1);
    k.st(1, 0, SMEM, 0..4);
    let bar6 = ResourceId::Named { cta: CtaId(0), id: 6 };
    k.arrive_r(0, u32::MAX, bar6, 0, true); // W0: bar.arrive
    k.arrive_r(1, u32::MAX, bar6, 0, true).wait_r(1, u32::MAX, bar6, 0, true); // W1: bar.sync
    k.ld(0, 0, SMEM, 0..4);
    assert!(has_race(&k.run()));
    // The arriver's prior writes are released to the syncing thread.
    let mut k = K::new(2, 1, 1);
    k.st(0, 0, SMEM, 0..4).arrive_r(0, u32::MAX, bar6, 0, true);
    k.arrive_r(1, u32::MAX, bar6, 0, true).wait_r(1, u32::MAX, bar6, 0, true);
    k.ld(1, 0, SMEM, 0..4);
    assert!(clean(&k.run()));
}

/// R5: cluster barrier defaults to release/acquire at cluster scope; a lost
/// qualifier is incomplete, never assumed relaxed; a `.relaxed` arrive gives
/// no generic edge.
#[test]
fn r5_cluster_barrier_defaults_and_lost_qualifier() {
    let mut k = K::new(1, 2, 2);
    k.st(0, 0, SMEM1, 0..4).cluster_bar(&[0, 1]).ld(1, 0, SMEM1, 0..4);
    assert!(clean(&k.run()));
    let mut k = K::new(1, 2, 2);
    k.st(0, 0, SMEM1, 0..4);
    k.arrive_q(0, u32::MAX, ResourceId::Cluster { cluster: 0 }, 0, None, Some(Scope::Cluster));
    let r = k.run();
    assert!(r.incomplete.iter().any(|i| matches!(i, Incomplete::SyncQualifierUnknown { .. })));
    let mut k = K::new(1, 2, 2);
    let c = ResourceId::Cluster { cluster: 0 };
    k.st(0, 0, SMEM1, 0..4);
    k.arrive_r(0, u32::MAX, c, 0, false).arrive_r(1, u32::MAX, c, 0, true);
    k.wait_r(0, u32::MAX, c, 0, true).wait_r(1, u32::MAX, c, 0, true).ld(1, 0, SMEM1, 0..4);
    assert!(has_race(&k.run()));
}

/// R8: Fence-SC order relates morally strong pairs: mixed scopes in one CTA
/// are related, `.cta` fences in different CTAs are not.
fn sc_pair(s0: Scope, s1: Scope) -> Report {
    let mut k = K::new(1, 1, 2);
    k.st(0, 0, GMEM, 0..4).fence(0, 1, FenceKind::Sc(s0));
    k.fence(1, 1, FenceKind::Sc(s1)).ld(1, 0, GMEM, 0..4);
    k.run()
}

#[test]
fn r8_fence_sc_pairwise_moral_strength() {
    assert!(has_race(&sc_pair(Scope::Cta, Scope::Cta)));
    assert!(has_race(&sc_pair(Scope::Gpu, Scope::Cta)));
    assert!(clean(&sc_pair(Scope::Gpu, Scope::Gpu)));
    // Same CTA, mixed scopes: related.
    let mut k = K::new(2, 1, 1);
    k.st(0, 0, GMEM, 0..4).fence(0, 1, FenceKind::Sc(Scope::Cta));
    k.fence(1, 1, FenceKind::Sc(Scope::Gpu)).ld(1, 0, GMEM, 0..4);
    assert!(clean(&k.run()));
}

/// R2/R3: atomics through different proxies are never morally strong.
#[test]
fn r3_cross_proxy_atomics_race() {
    let mut k = K::new(1, 1, 2);
    k.a(0, 0, atom(MemOrder::Relaxed, Scope::Gpu), GMEM, FLAG);
    let op = k.issue(1, 0, AsyncKind::Copy, Proxy::Async, &[], &[]);
    k.aacc(op, Milestone::Write, AccessKind::Rmw, Proxy::Async, GMEM, FLAG).done_warp(op, Milestone::Write, 1, 1);
    assert!(has_failure(&k.run(), |f| matches!(f, OrderingFailure::MissingProxyBridge { .. })));
}

/// R3: async-proxy writes from two CTAs into one CTA's smem, ordered only by
/// base causality, are a `review` advisory (ISA-silent on the thread block
/// of an async op).
#[test]
fn r3_cross_cta_async_writers_are_advisory() {
    let mut k = K::new(1, 2, 2);
    let a = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(a, Proxy::Async, GMEM, 0..16).aw(a, Proxy::Async, SMEM, 0..16).done_phase(a, Milestone::Write, 1, 0);
    k.wait(0, 1, 1, 0, true).cluster_bar(&[0, 1]);
    let b = k.issue(1, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(b, Proxy::Async, GMEM, 0..16).aw(b, Proxy::Async, SMEM, 0..16).done_phase(b, Milestone::Write, 1, 1);
    k.wait(0, 1, 1, 1, true);
    let r = k.run();
    assert!(r.errors().next().is_none(), "{r:?}");
    assert!(has_advisory(&r, AdvisoryKind::CrossCtaAsyncOrder));
}

/// R9: strong generic load vs strong store, morally strong: not a race
/// (legacy G carve-out dropped); polling an undeclared word is a review
/// advisory instead.
#[test]
fn r9_strong_load_exempt_with_undeclared_word_advisory() {
    let mut k = K::new(1, 1, 2);
    k.a(0, 0, st(MemOrder::Relaxed, Scope::Gpu), GMEM, FLAG);
    k.a(1, 0, ld(MemOrder::Relaxed, Scope::Gpu), GMEM, FLAG);
    let r = k.run();
    assert!(races(&r).is_empty());
    assert!(has_advisory(&r, AdvisoryKind::UndeclaredProtocolWord));
    // Declared: no advisory.
    let mut k = K::new(1, 1, 2);
    k.declare(GMEM, FLAG);
    k.a(0, 0, st(MemOrder::Relaxed, Scope::Gpu), GMEM, FLAG);
    k.a(1, 0, ld(MemOrder::Relaxed, Scope::Gpu), GMEM, FLAG);
    assert!(k.run().findings.is_empty());
    // A weak load is never exempt.
    let mut k = K::new(1, 1, 2);
    k.a(0, 0, st(MemOrder::Relaxed, Scope::Gpu), GMEM, FLAG).ld(1, 0, GMEM, FLAG);
    assert!(has_race(&k.run()));
}

/// R2: mixed-size strong accesses are never morally strong.
#[test]
fn r2_partial_overlap_atomics_race() {
    let mut k = K::new(1, 1, 2);
    k.a(0, 0, atom(MemOrder::Relaxed, Scope::Gpu), GMEM, 0..8);
    k.a(1, 0, atom(MemOrder::Relaxed, Scope::Gpu), GMEM, 4..8);
    assert!(has_race(&k.run()));
}
