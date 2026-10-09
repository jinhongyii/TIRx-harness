//! W6 guards for W5's re-attribution design (completed async ops' witnesses
//! moved from the op's slot to the warps that observed its completion, so
//! the slot can be reclaimed). Contract-event scenarios via
//! `racecheck_common`.
//!
//! Every guard runs with collection at every event (`gc_every = 1`, which
//! is where re-attribution and reclaim would happen) and with the default
//! period. The verdict must be the expected one under both, and the payloads
//! must be equal apart from the collector's own counters (exactness: no
//! precision change).
#[path = "racecheck_common/mod.rs"]
mod common;
use common::*;

/// Unrelated shared-memory traffic, in a byte range private to warp `w`:
/// gives the collector many chances to run (and re-attribute / reclaim)
/// between an observation and a later check.
fn filler(k: &mut K, w: WarpId, n: u64) {
    let base = 1024 + 768 * w as u64;
    for i in 0..n {
        k.st(w, 0, SMEM, base + 4 * i..base + 4 * i + 4);
    }
}

fn strip(r: Report) -> String {
    // The checker report carries findings and incompletes only (no
    // collector counters).
    format!("{r:?}")
}

/// Run `k` with GC at every event and with the default period; the payloads
/// must agree. Returns the eager-GC report.
fn both(mut k: K) -> Report {
    k.gc_every = 1;
    let eager = k.run();
    k.gc_every = DEFAULT_GC;
    let lazy = k.run();
    assert_eq!(strip(eager.clone()), strip(lazy), "collecting at every event changes the payload");
    eager
}

const DEFAULT_GC: u64 = numsim_core::racecheck::observer::DEFAULT_GC_EVERY;

/// Op A: an async (TMA-like) write of GMEM[0..4], completing on mbarrier 0
/// phase 0.
fn async_write(k: &mut K) -> AsyncId {
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(GMEM, 0..4)]);
    k.aw(op, Proxy::Async, GMEM, 0..4);
    k.done_phase(op, Milestone::Write, 0, 0);
    op
}

/// C1: two observers Q1 (warp 1) and Q2 (warp 2) wait on A's phase; the
/// reader (warp 3) learns only from Q2 (Q2 releases on mbarrier 1).
fn two_observers(chain: bool) -> K {
    let mut k = K::new(4, 1, 1);
    async_write(&mut k);
    k.wait(1, u32::MAX, 0, 0, true);
    k.wait(2, u32::MAX, 0, 0, true);
    filler(&mut k, 1, 64);
    if chain {
        k.arrive(2, u32::MAX, 1, 0, true);
        k.wait(3, u32::MAX, 1, 0, true);
    }
    filler(&mut k, 2, 64);
    k.ld(3, 0, GMEM, 0..4);
    k
}

#[test]
fn c1_reader_ordered_through_the_second_observer_is_clean() {
    assert!(clean(&both(two_observers(true))));
}

#[test]
fn c1_reader_without_any_observer_chain_races() {
    assert!(!races(&both(two_observers(false))).is_empty());
}

/// F1 (unsound gap in R2/R4): op A is a bulk-group store (issuer warp 0
/// lane 0). It lands long before the issuer's `wait_group` delivers its
/// `AsyncComplete{Warp}`. If A were re-attributed or marked "unobserved" and
/// its slot reclaimed in between, the late completion could no longer order
/// the issuer's own later write.
fn late_wait_group(own_write: bool) -> K {
    let mut k = K::new(2, 1, 1);
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..4), (GMEM, 0..4)]);
    k.ar(op, Proxy::Async, SMEM, 0..4).aw(op, Proxy::Async, GMEM, 0..4);
    filler(&mut k, 1, 128);
    k.done_warp(op, Milestone::Write, 0, 0b1);
    if own_write {
        k.st(0, 0, GMEM, 0..4);
    } else {
        k.st(1, 0, GMEM, 0..4);
    }
    k
}

#[test]
fn f1_issuer_write_after_late_wait_group_is_clean() {
    assert!(clean(&both(late_wait_group(true))));
}

#[test]
fn f1_other_warp_write_without_wait_races() {
    assert!(!races(&both(late_wait_group(false))).is_empty());
}

/// F2 (lane precision): only lane 0 of warp 1 waits on A's phase. Lane 5
/// (no warp sync, so it never observed A) releases to warp 2, which then
/// reads A's bytes: a race. With a full `__syncwarp` before lane 5's
/// release, lane 5 has observed A: clean.
fn lane_observer(warp_sync: bool) -> K {
    let mut k = K::new(3, 1, 1);
    async_write(&mut k);
    k.wait(1, 0b1, 0, 0, true);
    filler(&mut k, 1, 64);
    if warp_sync {
        k.syncwarp(1, u32::MAX);
    }
    k.arrive(1, 1 << 5, 1, 0, true);
    k.wait(2, u32::MAX, 1, 0, true);
    k.ld(2, 0, GMEM, 0..4);
    k
}

#[test]
fn f2_release_by_a_non_observing_lane_races() {
    assert!(!races(&both(lane_observer(false))).is_empty());
}

#[test]
fn f2_release_after_warp_sync_is_clean() {
    assert!(clean(&both(lane_observer(true))));
}

/// C3 (cross-proxy views): A asynchronously writes SMEM[0..4]. Q (warp 1)
/// observes it on the phase; R (warp 2) learns only Q's `hb` through a
/// release on mbarrier 1, then reads generically, with and without its own
/// `fence.proxy.async`. Whatever the verdict, it must not depend on when the
/// collector runs (pinned by `both`).
fn bridge(fence: bool) -> K {
    let mut k = K::new(3, 1, 1);
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..4)]);
    k.aw(op, Proxy::Async, SMEM, 0..4);
    k.done_phase(op, Milestone::Write, 0, 0);
    k.wait(1, u32::MAX, 0, 0, true);
    filler(&mut k, 1, 64);
    k.arrive(1, u32::MAX, 1, 0, true);
    k.wait(2, u32::MAX, 1, 0, true);
    if fence {
        k.fence(2, u32::MAX, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    }
    filler(&mut k, 2, 64);
    k.ld(2, 0, SMEM, 0..4);
    k
}

#[test]
fn c3_bridge_verdicts_do_not_depend_on_collection() {
    for fence in [false, true] {
        let r = both(bridge(fence));
        // Today's verdict: Q's release carries its bridge rows, so R's
        // generic read is ordered either way.
        assert!(clean(&r), "fence={fence}: {r:#?}");
    }
}

/// v1 rule (a)3 (an a2g observer's stamp counts when found in cur's `hb`):
/// exact only if every path that delivers Q's `hb` also delivers Q's a2g
/// bridge rows. Barrier payloads do (C3). This pins the other path: Q
/// observes A (async write of GMEM[0..4]) on its phase, then publishes with
/// `st.release.gpu` on GMEM2; R `ld.acquire`s it (read-from: a release
/// head, not a barrier payload) and then reads A's bytes generically.
/// Today's verdict must hold under any collection schedule, and under any
/// future re-attribution.
fn read_from_bridge() -> K {
    let mut k = K::new(3, 1, 1);
    async_write(&mut k);
    k.wait(1, u32::MAX, 0, 0, true);
    filler(&mut k, 1, 64);
    k.a(1, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, 0..4);
    k.a(2, 0, ld(MemOrder::Acquire, Scope::Gpu), GMEM2, 0..4);
    filler(&mut k, 2, 64);
    k.ld(2, 0, GMEM, 0..4);
    k
}

#[test]
fn a3_read_from_release_carries_the_observation() {
    let r = both(read_from_bridge());
    // Today: the release head carries Q's bridge rows with its hb, so R's
    // generic read of A's bytes is ordered.
    assert!(clean(&r), "{r:#?}");
}

// ---------------------------------------------------------------------------
// §16 acceptance list for phase-record tokens (B), items 1-9. Baselines:
// today's verdicts, which any token design must reproduce under every
// collection schedule. Item 7 (fork/join ownership and merge order) is
// engine-level: racecheck_parallel_review.
// ---------------------------------------------------------------------------

/// Item 1: A completes on mbarrier 0 phase 0; warp 1 then advances the
/// barrier through phases 1..=10 (each with its own arrive and wait), so
/// phase 0's record is pruned. Warp 2 waits on phase 0 afterwards and reads
/// A's bytes.
fn wait_after_pruning() -> K {
    let mut k = K::new(3, 1, 1);
    async_write(&mut k);
    for p in 1..=10 {
        k.arrive(1, u32::MAX, 0, p, true).wait(1, u32::MAX, 0, p, true);
    }
    k.wait(2, u32::MAX, 0, 0, true);
    k.ld(2, 0, GMEM, 0..4);
    k
}

#[test]
fn item1_wait_on_a_pruned_phase_is_never_ordered() {
    let r = both(wait_after_pruning());
    // Today (contract events): the stale wait acquires nothing of A, so the
    // read races (MissingProxyBridge). It must never become "ordered".
    assert!(!clean(&r), "a wait on a pruned phase must not order A: {r:#?}");
    assert!(!races(&r).is_empty(), "{r:#?}");
}

/// Item 2: A completes on phase 0. Warp 1 (which never observed A) arrives
/// on phases 1 and 2. Warp 2 waits on phase 2 (same parity as 0) and reads
/// A's bytes: phase 2's payload does not include A.
fn parity_alias() -> K {
    let mut k = K::new(3, 1, 1);
    async_write(&mut k);
    k.arrive(1, u32::MAX, 0, 1, true).wait(1, u32::MAX, 0, 1, true);
    k.arrive(1, u32::MAX, 0, 2, true);
    k.wait(2, u32::MAX, 0, 2, true);
    k.ld(2, 0, GMEM, 0..4);
    k
}

#[test]
fn item2_same_parity_two_generations_later_is_not_ordered() {
    assert!(!races(&both(parity_alias())).is_empty());
}

/// Item 3: a relaxed wait parks A's completion until a later acquire fence.
fn relaxed_then_fence(fence: bool) -> K {
    let mut k = K::new(2, 1, 1);
    async_write(&mut k);
    k.wait(1, u32::MAX, 0, 0, false);
    filler(&mut k, 1, 32);
    if fence {
        k.fence(1, u32::MAX, FenceKind::AcqRel(Scope::Cta));
    }
    k.ld(1, 0, GMEM, 0..4);
    k
}

#[test]
fn item3_relaxed_wait_orders_only_after_the_acquire_fence() {
    assert!(!races(&both(relaxed_then_fence(false))).is_empty(), "relaxed wait alone");
    assert!(clean(&both(relaxed_then_fence(true))), "relaxed wait + fence.acquire");
}

/// Item 4: one multicast completion lands in both CTAs' records (2-CTA
/// cluster, one warp each). Warp 1 (CTA 1) waits on its own record; or not.
fn multicast(wait: bool) -> K {
    let mut k = K::new(1, 2, 2);
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(GMEM, 0..4)]);
    k.aw(op, Proxy::Async, GMEM, 0..4);
    k.done_phase_r(op, Milestone::Write, mbar_in(0, 0), 0);
    k.done_phase_r(op, Milestone::Write, mbar_in(1, 0), 0);
    if wait {
        k.wait_r(1, u32::MAX, mbar_in(1, 0), 0, true);
    }
    k.ld(1, 0, GMEM, 0..4);
    k
}

#[test]
fn item4_each_multicast_record_orders_only_its_waiters() {
    assert!(clean(&both(multicast(true))));
    assert!(!races(&both(multicast(false))).is_empty());
}

/// Item 5: warp 0 (CTA 0) observes A on its own record, then a
/// `barrier.cluster` with warp 1 (CTA 1); warp 1 reads A's bytes.
fn cluster_barrier(bar: bool) -> K {
    let mut k = K::new(1, 2, 2);
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(GMEM, 0..4)]);
    k.aw(op, Proxy::Async, GMEM, 0..4);
    k.done_phase_r(op, Milestone::Write, mbar_in(0, 0), 0);
    k.wait_r(0, u32::MAX, mbar_in(0, 0), 0, true);
    if bar {
        k.cluster_bar(&[0, 1]);
    }
    k.ld(1, 0, GMEM, 0..4);
    k
}

#[test]
fn item5_cluster_barrier_carries_the_observation() {
    assert!(clean(&both(cluster_barrier(true))));
    assert!(!races(&both(cluster_barrier(false))).is_empty());
}

/// Item 6: warp 0 observes A, then issues a tcgen05.commit completing on
/// mbarrier 1, which warp 1 waits on before reading A's bytes. The commit
/// forwards its issuer's generic knowledge. If the commit is issued before
/// warp 0 observes A, it forwards nothing about A.
fn commit_forwarding(observe_first: bool) -> K {
    let mut k = K::new(2, 1, 1);
    async_write(&mut k);
    if observe_first {
        k.wait(0, u32::MAX, 0, 0, true);
    }
    let c = k.issue(0, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &[], &[]);
    if !observe_first {
        k.wait(0, u32::MAX, 0, 0, true);
    }
    k.done_phase(c, Milestone::Write, 1, 0);
    k.wait(1, u32::MAX, 1, 0, true);
    k.ld(1, 0, GMEM, 0..4);
    k
}

#[test]
fn item6_commit_forwards_only_what_its_issuer_knew() {
    assert!(clean(&both(commit_forwarding(true))));
    assert!(!races(&both(commit_forwarding(false))).is_empty());
}

/// Item 8: warp 0 (CTA 0) observes A, then arrives `.release.cluster` on
/// CTA 1's mbarrier. Warp 1 (CTA 1) waits with `waiter_scope`: `.cta`
/// (does not cover the remote arriver: no edge, ScopeMismatch) or
/// `.cluster`.
fn scope_filtered(waiter_scope: Scope) -> K {
    let mut k = K::new(1, 2, 2);
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(GMEM, 0..4)]);
    k.aw(op, Proxy::Async, GMEM, 0..4);
    k.done_phase_r(op, Milestone::Write, mbar_in(0, 0), 0);
    k.wait_r(0, u32::MAX, mbar_in(0, 0), 0, true);
    k.arrive_q(0, u32::MAX, mbar_in(1, 0), 0, Some(true), Some(Scope::Cluster));
    k.wait_q(1, u32::MAX, mbar_in(1, 0), 0, Some(true), Some(waiter_scope));
    k.ld(1, 0, GMEM, 0..4);
    k
}

#[test]
fn item8_scope_filtered_arrival_carries_nothing() {
    let r = both(scope_filtered(Scope::Cta));
    assert!(!races(&r).is_empty() && has_scope_mismatch(&r), "{r:#?}");
    assert!(clean(&both(scope_filtered(Scope::Cluster))));
}

/// Item 9: a tensor-map descriptor in GMEM2[0..128], written by warp 0
/// either asynchronously (op A, observed by warp 0 on mbarrier 0) or with
/// plain stores. Warp 0 then (with `fences`)
/// `fence.proxy.tensormap::generic.release`, and hands off to warp 1 on
/// mbarrier 1. Warp 1 (with `fences`) does the tensormap acquire and issues
/// a TMA that reads the descriptor.
fn tensormap_path(async_write: bool, fences: bool) -> K {
    let mut k = K::new(2, 1, 1);
    if async_write {
        let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(GMEM2, 0..128)]);
        k.aw(op, Proxy::Async, GMEM2, 0..128);
        k.done_phase(op, Milestone::Write, 0, 0);
        k.wait(0, u32::MAX, 0, 0, true);
    } else {
        k.st(0, 0, GMEM2, 0..128);
    }
    if fences {
        k.fence(0, 1, FenceKind::TensormapRelease { scope: Scope::Cta });
    }
    k.arrive(0, u32::MAX, 1, 0, true).wait(1, u32::MAX, 1, 0, true);
    if fences {
        k.fence(1, 1, FenceKind::TensormapAcquire { scope: Scope::Cta, alloc: GMEM2, span: numsim_core::arena::ByteSpan::new(0, 128) });
    }
    let tma = k.issue(1, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(tma, Proxy::TensorMap, GMEM2, 0..128).aw(tma, Proxy::Async, SMEM, 0..16).done_warp(tma, Milestone::Write, 1, 1);
    k
}

/// Today: a descriptor written by an async op and observed on its phase is
/// ordered for the TMA's descriptor read by `hb` alone (the mbarrier 1
/// handoff), with or without the tensormap fences. A generic-written
/// descriptor needs the release/acquire pair (deltas I1/I8): that is the
/// `tmap_rel` -> `g2t` path, which a token must ride without losing or
/// adding ordering.
#[test]
fn item9_tensormap_path_carries_the_observation() {
    for fences in [false, true] {
        let r = both(tensormap_path(true, fences));
        assert!(clean(&r), "async descriptor, fences={fences}: {r:#?}");
    }
    assert!(clean(&both(tensormap_path(false, true))), "generic descriptor with the fence pair");
    assert!(!clean(&both(tensormap_path(false, false))), "generic descriptor without the fence pair");
}

/// §17 review (b): a per-warp scalar delivery token is not lane-precise
/// across different acquisitions. Copy A (GMEM[0..4]) completes on
/// mbarrier 0 and copy B (GMEM[64..68]) on mbarrier 1, both phase 0.
/// Lane 0 of warp 1 acquires A's record; lane 5 of warp 1 then acquires B's
/// record (a later, larger delivery-token value). Warp 3 advances
/// mbarrier 0 through phases 1..=6, so A's record is pruned (and, under §17,
/// converted to the delivery token). Lane 5 then releases to warp 2, which
/// reads A's bytes. Lane 5 never acquired A: the read races today, and must
/// keep racing.
fn lane_split_delivery() -> K {
    let mut k = K::new(4, 1, 1);
    async_write(&mut k);
    let b = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(GMEM, 64..68)]);
    k.aw(b, Proxy::Async, GMEM, 64..68);
    k.done_phase(b, Milestone::Write, 1, 0);
    k.wait(1, 0b1, 0, 0, true);
    k.wait(1, 1 << 5, 1, 0, true);
    for p in 1..=6 {
        k.arrive(3, u32::MAX, 0, p, true).wait(3, u32::MAX, 0, p, true);
    }
    filler(&mut k, 3, 32);
    k.arrive(1, 1 << 5, 2, 0, true);
    k.wait(2, u32::MAX, 2, 0, true);
    k.ld(2, 0, GMEM, 0..4);
    k
}

#[test]
fn s17b_lane_split_delivery_tokens_do_not_order() {
    let r = both(lane_split_delivery());
    assert!(!races(&r).is_empty(), "lane 5 never acquired A: {r:#?}");
}
