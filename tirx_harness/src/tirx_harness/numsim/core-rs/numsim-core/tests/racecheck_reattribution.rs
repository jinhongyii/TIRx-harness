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
