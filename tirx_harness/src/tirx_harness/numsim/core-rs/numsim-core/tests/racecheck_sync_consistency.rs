//! Cross-checker consistency (checker-review.md §6): for each synchronization
//! ruling owned by the sync model / synccheck, the happens-before edge that
//! racecheck must (not) add, pinned on contract events (the engine emits the
//! same `SyncEvent`s for both checkers).
#[path = "racecheck_common/mod.rs"]
mod common;
use common::*;
use numsim_core::observe::CtaId;
use numsim_core::sync::completion::ResourceId;

fn named(id: u8) -> ResourceId {
    ResourceId::Named { cta: CtaId(0), id }
}

/// Sync delta B7: a warp that exits releases a count-less named barrier for
/// the others, but its exit publishes nothing (the engine logs the exit
/// arrival only as a synccheck `Protocol` event). A write before the exit is
/// unordered with a read after the barrier.
#[test]
fn b7_exit_release_publishes_no_memory() {
    let mut k = K::new(2, 1, 1);
    k.st(0, 0, GMEM, 0..4);
    // Warp 0 exits (no Arrive event); warp 1's bar.sync completes alone.
    k.arrive_r(1, u32::MAX, named(1), 0, true).wait_r(1, u32::MAX, named(1), 0, true);
    k.ld(1, 0, GMEM, 0..4);
    assert!(!races(&k.run()).is_empty(), "exit must not publish warp 0's write");
}

/// Q3/Q5 gather: lanes of one warp reaching a non-aligned `bar.sync` in two
/// pieces make ONE arrival (the engine's `Arrive` event names every gathered
/// lane), which releases every lane's prior writes.
#[test]
fn q5_gathered_partial_warp_releases_every_lane() {
    let mut k = K::new(2, 1, 1);
    k.st(0, 3, GMEM, 0..4).st(0, 20, GMEM, 4..8);
    k.arrive_r(0, u32::MAX, named(1), 0, true).arrive_r(1, u32::MAX, named(1), 0, true);
    k.wait_r(0, u32::MAX, named(1), 0, true).wait_r(1, u32::MAX, named(1), 0, true);
    k.ld(1, 0, GMEM, 0..8);
    assert!(races(&k.run()).is_empty());
}

/// Q3: `bar.arrive` (arrive without wait) releases but does not acquire.
#[test]
fn q3_arrive_only_warp_acquires_nothing() {
    let mut k = K::new(2, 1, 1);
    k.st(1, 0, GMEM, 0..4);
    k.arrive_r(0, u32::MAX, named(1), 0, true).arrive_r(1, u32::MAX, named(1), 0, true);
    k.wait_r(1, u32::MAX, named(1), 0, true);
    k.ld(0, 0, GMEM, 0..4);
    assert!(!races(&k.run()).is_empty(), "an arrive-only warp acquires nothing");
}

/// Q7: `cp.async.wait_group` completes groups per thread. Lane 0 waits for
/// its own copy (the engine's `AsyncComplete{Warp}` names only the issuing
/// lane); that does not order lane 1, which has no wait of its own.
#[test]
fn q7_wait_group_is_per_lane() {
    let mut k = K::new(1, 1, 1);
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..4)]);
    k.aw(op, Proxy::Async, SMEM, 0..4);
    k.done_warp(op, Milestone::Write, 0, 0b1);
    k.ld(0, 0, SMEM, 0..4);
    let r = k.run();
    assert!(races(&r).is_empty(), "lane 0 waited for its own copy: {r:?}");
    let mut k2 = K::new(1, 1, 1);
    let op = k2.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..4)]);
    k2.aw(op, Proxy::Async, SMEM, 0..4);
    k2.done_warp(op, Milestone::Write, 0, 0b1);
    k2.ld(0, 1, SMEM, 0..4);
    assert!(!races(&k2.run()).is_empty(), "lane 1 is not ordered after lane 0's copy");
}

/// Per-element sync words (W5-14): a `sync_words` buffer is one declared word
/// per element, and a `wait_until` on element 0 acquires only from writes to
/// element 0. CTA 1 publishes `GMEM[0..4]` through element 0 and CTA 2
/// publishes `GMEM[4..8]` through element 1; CTA 0 waits on element 0 only.
#[test]
fn per_element_sync_word_acquires_only_its_element() {
    let run = |read: std::ops::Range<u64>| {
        let mut k = K::new(1, 1, 3);
        k.declare(GMEM2, 0..4).declare(GMEM2, 4..8);
        k.st(1, 0, GMEM, 0..4).a(1, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, 0..4);
        k.st(2, 0, GMEM, 4..8).a(2, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, 4..8);
        k.wait_until(0, 0, GMEM2, 0..4, Scope::Gpu, 0b10, 1).ld(0, 0, GMEM, read);
        k.run()
    };
    let own = run(0..4);
    assert!(clean(&own), "element 0's publisher is acquired: {own:?}");
    let other = run(4..8);
    assert!(has_class(&other, RaceClass::WriteRead), "element 1's publisher is not acquired: {other:?}");
}

fn cluster0() -> ResourceId {
    ResourceId::Cluster { cluster: 0 }
}

/// Q4: an exited warp counts as arrived at `barrier.cluster`, but its exit
/// publishes nothing (the engine emits no `Arrive` for it).
#[test]
fn q4_exited_warp_at_cluster_barrier_publishes_nothing() {
    let mut k = K::new(1, 2, 2);
    k.st(0, 0, GMEM, 0..4);
    k.arrive_r(1, u32::MAX, cluster0(), 0, true).wait_r(1, u32::MAX, cluster0(), 0, true);
    k.ld(1, 0, GMEM, 0..4);
    assert!(!races(&k.run()).is_empty(), "exit must not publish warp 0's write");
}

/// C3/Q11: a non-aligned `barrier.cluster.arrive` reached by a warp in two
/// pieces is one arrival (`cluster::gather`), and its `Arrive` names every
/// gathered lane, so every lane's prior writes are released at `.cluster`.
#[test]
fn c3_gathered_cluster_arrive_releases_every_lane() {
    let mut k = K::new(1, 2, 2);
    k.st(0, 3, GMEM, 0..4).st(0, 20, GMEM, 4..8);
    k.arrive_r(0, u32::MAX, cluster0(), 0, true).arrive_r(1, u32::MAX, cluster0(), 0, true);
    k.wait_r(0, u32::MAX, cluster0(), 0, true).wait_r(1, u32::MAX, cluster0(), 0, true);
    k.ld(1, 0, GMEM, 0..8);
    assert!(races(&k.run()).is_empty());
}

/// Q7 `.read`: `cp.async.bulk.wait_group.read` completes only the source
/// read. The source may be overwritten afterwards; the destination is not
/// yet published.
#[test]
fn q7_wait_group_read_publishes_only_the_source_read() {
    let run = |reuse_source: bool| {
        let mut k = K::new(1, 1, 1);
        let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..4), (GMEM, 0..4)]);
        k.ar(op, Proxy::Async, SMEM, 0..4).aw(op, Proxy::Async, GMEM, 0..4);
        k.done_warp(op, Milestone::Read, 0, 0b1);
        if reuse_source {
            k.st(0, 0, SMEM, 0..4);
        } else {
            k.ld(0, 0, GMEM, 0..4);
        }
        k.run()
    };
    let src = run(true);
    assert!(races(&src).is_empty(), "source reuse after .read is ordered: {src:?}");
    assert!(!races(&run(false)).is_empty(), ".read does not publish the destination");
}

/// C3/Q11 engine shape: the engine's cluster gather must emit exactly one
/// `Arrive` for the warp, naming every gathered lane (the shape
/// `c3_gathered_cluster_arrive_releases_every_lane` assumes).
#[test]
fn engine_cluster_gather_emits_one_arrive_naming_every_lane() {
    use numsim_core::observe::{RecordingObserver, SyncKind};
    use numsim_core::sched::{self, RunStatus};
    use numsim_core::testutil::scenarios;
    let s = scenarios::cluster_partial_arrive(false);
    let mut rec = RecordingObserver::default();
    let o = sched::run_with_config(&s.module, &s.inputs, &mut rec, &s.config).expect("run starts");
    assert_eq!(o.status, RunStatus::Completed, "{:?}", o.status);
    let arrives: Vec<_> = rec
        .per_warp
        .iter()
        .flatten()
        .filter(|e| matches!(e.kind, SyncKind::Arrive { obj: ResourceId::Cluster { .. }, .. }))
        .collect();
    assert_eq!(arrives.len(), 1, "{arrives:?}");
    assert_eq!(arrives[0].lanes.0, u32::MAX, "{arrives:?}");
}
