//! Translated from test_declared_word_regressions.py,
//! test_declared_word_shapes.py and the declared-word rows of
//! test_native_global_scoped_hb_matrix.py.
//!
//! History indices: 0 = launch value, i = i-th write to the word.
#[path = "racecheck_common/mod.rs"]
mod common;
use common::*;

const FLAG: std::ops::Range<u64> = 0..4;

/// C0 writes data then publishes `flag` with `publish`; C1 waits and reads.
fn publication(fence: bool, publish: Op) -> Report {
    let mut k = K::new(1, 1, 2);
    k.declare(GMEM2, FLAG);
    k.st(0, 0, GMEM, 0..4);
    if fence {
        k.fence(0, 1, FenceKind::AcqRel(Scope::Gpu));
    }
    k.a(0, 0, publish, GMEM2, FLAG);
    k.wait_until(1, 0, GMEM2, FLAG, Scope::Gpu, 0b10, 1).ld(1, 0, GMEM, 0..4);
    k.run()
}

#[test]
fn publication_kinds() {
    assert!(has_class(&publication(false, st(MemOrder::Relaxed, Scope::Gpu)), RaceClass::WriteRead));
    assert!(clean(&publication(false, st(MemOrder::Release, Scope::Gpu))));
    assert!(clean(&publication(true, st(MemOrder::Relaxed, Scope::Gpu))));
    assert!(clean(&publication(false, atom(MemOrder::Release, Scope::Gpu))));
}

#[test]
fn launch_value_exit_owes_no_edge() {
    let mut k = K::one_warp();
    k.declare(GMEM2, FLAG).wait_until(0, 0, GMEM2, FLAG, Scope::Gpu, 0b1, 0);
    assert!(clean(&k.run()));
}

#[test]
fn unexplained_exit_is_incomplete() {
    let mut k = K::one_warp();
    k.declare(GMEM2, FLAG).wait_until(0, 0, GMEM2, FLAG, Scope::Gpu, 0b100, 0);
    let r = k.run();
    assert!(r.incomplete.iter().any(|i| matches!(i, Incomplete::WaitExitUnproven { .. })));
}

/// The EARLIEST accepted write decides, not the one this run observed: a
/// relaxed first write that already satisfies the predicate gives no edge
/// even though the run observed a later release.
#[test]
fn earliest_accepted_write_is_schedule_independent() {
    let mut k = K::new(1, 1, 3);
    k.declare(GMEM2, FLAG);
    k.st(0, 0, GMEM, 0..4);
    k.a(2, 0, st(MemOrder::Relaxed, Scope::Gpu), GMEM2, FLAG); // entry 1: accepted
    k.a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, FLAG); // entry 2: accepted, observed
    k.wait_until(1, 0, GMEM2, FLAG, Scope::Gpu, 0b110, 2).ld(1, 0, GMEM, 0..4);
    assert!(has_class(&k.run(), RaceClass::WriteRead));
    // If only entry 2 satisfies the predicate the edge exists.
    let mut k = K::new(1, 1, 3);
    k.declare(GMEM2, FLAG);
    k.st(0, 0, GMEM, 0..4);
    k.a(2, 0, st(MemOrder::Relaxed, Scope::Gpu), GMEM2, FLAG);
    k.a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, FLAG);
    k.wait_until(1, 0, GMEM2, FLAG, Scope::Gpu, 0b100, 2).ld(1, 0, GMEM, 0..4);
    assert!(clean(&k.run()));
}

/// Phase-flip predicate: the edge comes from the arrival that flipped it.
#[test]
fn phase_flip_predicate() {
    let mut k = K::new(1, 1, 2);
    k.declare(GMEM2, FLAG);
    k.st(0, 0, GMEM, 0..4).a(0, 0, atom(MemOrder::Release, Scope::Gpu), GMEM2, FLAG);
    k.a(1, 0, atom(MemOrder::Release, Scope::Gpu), GMEM2, FLAG);
    k.wait_until(1, 0, GMEM2, FLAG, Scope::Gpu, 0b10, 2).ld(1, 0, GMEM, 0..4);
    assert!(clean(&k.run()));
}

#[test]
fn two_waiters_reader_reader_is_no_conflict() {
    let mut k = K::new(1, 1, 3);
    k.declare(GMEM2, FLAG);
    k.st(0, 0, GMEM, 0..4).a(0, 0, atom(MemOrder::Release, Scope::Gpu), GMEM2, FLAG);
    for w in [1, 2] {
        k.wait_until(w, 0, GMEM2, FLAG, Scope::Gpu, 0b10, 1).ld(w, 0, GMEM, 0..4);
    }
    assert!(clean(&k.run()));
}

#[test]
fn morally_strong_store_and_rmw_are_exempt() {
    let mut k = K::new(1, 1, 2);
    k.a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, FLAG);
    k.a(1, 0, atom(MemOrder::Release, Scope::Gpu), GMEM2, FLAG);
    assert!(clean(&k.run()));
}

/// Release scope must reach the waiter.
fn scope_reach(rel: Scope) -> Report {
    let mut k = K::new(1, 1, 2); // two clusters of one CTA
    k.declare(GMEM2, FLAG);
    k.st(0, 0, GMEM, 0..4).a(0, 0, st(MemOrder::Release, rel), GMEM2, FLAG);
    k.wait_until(1, 0, GMEM2, FLAG, Scope::Sys, 0b10, 1).ld(1, 0, GMEM, 0..4);
    k.run()
}

#[test]
fn scope_reaches_or_stops_short() {
    assert!(clean(&scope_reach(Scope::Gpu)));
    let r = scope_reach(Scope::Cta);
    assert!(has_scope_mismatch(&r));
    assert!(has_class(&r, RaceClass::WriteRead));
}

/// The wait does not bridge the async proxy: only the read side of the
/// bulk store completed before the release.
fn async_store_publication(full_wait: bool) -> Report {
    let mut k = K::new(1, 1, 2);
    k.declare(GMEM2, FLAG);
    k.st(0, 0, SMEM, 0..16).fence(0, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, SMEM, 0..16).aw(op, Proxy::Async, GMEM, 0..16);
    k.done_warp(op, if full_wait { Milestone::Write } else { Milestone::Read }, 0, 1);
    k.fence(0, 1, FenceKind::ProxyAsync(Some(Domain::Global)));
    k.a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, FLAG);
    k.wait_until(1, 0, GMEM2, FLAG, Scope::Gpu, 0b10, 1).ld(1, 0, GMEM, 0..4);
    k.run()
}

#[test]
fn wait_does_not_bridge_unfinished_async_store() {
    assert!(has_class(&async_store_publication(false), RaceClass::WriteRead));
    assert!(clean(&async_store_publication(true)));
}

/// RMW release sequence: C0 release-stores, C1 relaxed-adds (continues the
/// sequence), C2 waits on flag>=1 accepting C1's write... and C0's.
fn release_sequence(head: Op) -> Report {
    let mut k = K::new(1, 1, 3);
    k.declare(GMEM2, FLAG);
    k.st(0, 0, GMEM, 0..4).a(0, 0, head, GMEM2, FLAG);
    k.a(1, 0, atom(MemOrder::Relaxed, Scope::Gpu), GMEM2, FLAG);
    // Accept only the RMW's value, so the edge must come through the chain.
    k.wait_until(2, 0, GMEM2, FLAG, Scope::Gpu, 0b100, 2).ld(2, 0, GMEM, 0..4);
    k.run()
}

#[test]
fn rmw_release_sequence() {
    assert!(clean(&release_sequence(atom(MemOrder::Release, Scope::Gpu))));
    assert!(clean(&release_sequence(st(MemOrder::Release, Scope::Gpu))));
    assert!(has_class(&release_sequence(atom(MemOrder::Relaxed, Scope::Gpu)), RaceClass::WriteRead));
}

/// Async publication fallback: the accepted history entry is an async
/// write, so the run-observed entry is used.
#[test]
fn async_publication_falls_back_to_observed() {
    let mut k = K::new(1, 1, 2);
    k.declare(GMEM2, FLAG);
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[]);
    k.aw(op, Proxy::Async, GMEM2, FLAG).done_warp(op, Milestone::Write, 0, 1);
    k.st(0, 0, GMEM, 0..4).a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, FLAG);
    k.wait_until(1, 0, GMEM2, FLAG, Scope::Gpu, 0b110, 2).ld(1, 0, GMEM, 0..4);
    let r = k.run();
    assert!(races(&r).iter().all(|f| f.alloc != GMEM), "{r:?}");
}

/// A predicate that reads other memory (PredProgram `reads_memory`): sound
/// only when that memory is stable across the wait.
fn pred_reads(stable: bool) -> Report {
    let mut k = K::new(1, 1, 3);
    k.declare(GMEM2, FLAG);
    k.st(2, 0, GMEM, 64..68); // predicate input written by C2
    if stable {
        k.a(2, 0, st(MemOrder::Release, Scope::Gpu), GMEM, 128..132);
        k.a(1, 0, ld(MemOrder::Acquire, Scope::Gpu), GMEM, 128..132);
    }
    k.st(0, 0, GMEM, 0..4).a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, FLAG);
    k.wait_until_pred(1, 0, GMEM2, FLAG, Scope::Gpu, 0b10, 1, &[(GMEM, 64..68)]);
    k.run()
}

#[test]
fn predicate_reading_memory() {
    assert!(clean(&pred_reads(true)));
    let r = pred_reads(false);
    assert!(r.incomplete.iter().any(|i| matches!(i, Incomplete::WaitPredicateReadsUnstable { .. })));
}

/// W7 (mega_moe workspace grid sync): a reused grid-sync counter accepts
/// stale values of an earlier round (the predicate compares one bit). The
/// waiter cannot read a value coherence-before its own add (CoWR), so the
/// accepted entry is the earliest at or after its own write.
#[test]
fn accepted_entry_not_before_own_write() {
    let run = |accepted: u64| {
        let mut k = K::new(1, 1, 3);
        k.declare(GMEM2, FLAG);
        k.a(2, 0, atom(MemOrder::Relaxed, Scope::Gpu), GMEM2, FLAG); // entry 1 (earlier round, accepted)
        k.st(0, 0, GMEM, 0..4);
        k.a(0, 0, atom(MemOrder::Release, Scope::Gpu), GMEM2, FLAG); // entry 2
        k.a(1, 0, atom(MemOrder::Release, Scope::Gpu), GMEM2, FLAG); // entry 3: own add
        k.a(2, 0, atom(MemOrder::Release, Scope::Gpu), GMEM2, FLAG); // entry 4: completes the round
        k.wait_until(1, 0, GMEM2, FLAG, Scope::Gpu, accepted, 4).ld(1, 0, GMEM, 0..4);
        k.run()
    };
    assert!(clean(&run(0b1_0010)), "{:?}", run(0b1_0010));
}

