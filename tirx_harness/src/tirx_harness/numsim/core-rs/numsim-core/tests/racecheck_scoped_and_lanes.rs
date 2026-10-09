//! Translated from test_native_global_scoped_hb.py, ..._matrix.py,
//! test_native_atomic_semantics.py, test_native_racecheck_lane_order.py,
//! test_native_racecheck_release_rmw_handoff.py,
//! test_native_shared_publication.py, test_native_racecheck_arrive_snapshot.py,
//! test_native_sub_word_accesses.py and test_native_matrix_collective_sync.py.
#[path = "racecheck_common/mod.rs"]
mod common;
use common::*;

const FLAG: std::ops::Range<u64> = 0..4;

// ------------------------------------------------------- scoped global --

/// Writer C0, reader C1 (different clusters).
fn scoped(mode: u8) -> Report {
    let mut k = K::new(1, 1, 2);
    k.declare(GMEM2, FLAG);
    match mode {
        0 => {
            k.st(0, 0, GMEM, 0..4).a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, FLAG);
            k.wait_until(1, 0, GMEM2, FLAG, Scope::Gpu, 0b10, 1);
        }
        1 => {
            k.st(0, 0, GMEM, 0..4).a(0, 0, st(MemOrder::Release, Scope::Cluster), GMEM2, FLAG);
            k.wait_until(1, 0, GMEM2, FLAG, Scope::Cluster, 0b10, 1);
        }
        2 => {
            k.st(0, 0, GMEM, 0..4).a(0, 0, st(MemOrder::Relaxed, Scope::Gpu), GMEM2, FLAG);
            k.wait_until(1, 0, GMEM2, FLAG, Scope::Gpu, 0b10, 1);
        }
        3 => {
            k.st(0, 0, GMEM, 0..4).a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, FLAG);
            k.a(1, 0, ld(MemOrder::Relaxed, Scope::Gpu), GMEM2, FLAG);
        }
        5 => {
            k.st(0, 0, GMEM, 0..4).fence(0, 1, FenceKind::AcqRel(Scope::Gpu));
            k.a(0, 0, st(MemOrder::Relaxed, Scope::Sys), GMEM2, FLAG);
            k.a(1, 0, ld(MemOrder::Relaxed, Scope::Sys), GMEM2, FLAG).fence(1, 1, FenceKind::AcqRel(Scope::Gpu));
        }
        6 => {
            k.st(0, 0, GMEM, 0..4).a(0, 0, st(MemOrder::Relaxed, Scope::Sys), GMEM2, FLAG);
            k.fence(0, 1, FenceKind::AcqRel(Scope::Gpu));
            k.a(1, 0, ld(MemOrder::Relaxed, Scope::Sys), GMEM2, FLAG).fence(1, 1, FenceKind::AcqRel(Scope::Gpu));
        }
        7 => {
            k.st(0, 0, GMEM, 0..4).fence(0, 1, FenceKind::AcqRel(Scope::Gpu));
            k.a(0, 0, st(MemOrder::Relaxed, Scope::Sys), GMEM2, FLAG);
            k.fence(1, 1, FenceKind::AcqRel(Scope::Gpu)).a(1, 0, ld(MemOrder::Relaxed, Scope::Sys), GMEM2, FLAG);
        }
        _ => unreachable!(),
    }
    k.ld(1, 0, GMEM, 0..4);
    k.run()
}

#[test]
fn global_scoped_hb_modes() {
    assert!(clean(&scoped(0)));
    let r1 = scoped(1);
    assert!(has_scope_mismatch(&r1) && has_class(&r1, RaceClass::WriteRead));
    assert_eq!(races(&r1)[0].bytes, 0..4);
    for m in [2, 3, 6, 7] {
        assert!(has_class(&scoped(m), RaceClass::WriteRead), "mode {m}");
    }
    assert!(clean(&scoped(5)));
}

#[test]
fn plain_flag_spin_races_on_flag_and_data() {
    let mut k = K::new(1, 1, 2);
    k.st(0, 0, GMEM, 0..4).st(0, 0, GMEM2, FLAG).ld(1, 0, GMEM2, FLAG).ld(1, 0, GMEM, 0..4);
    assert!(races(&k.run()).len() >= 2);
}

#[test]
fn typed_cluster_fence_same_cta() {
    let mut k = K::new(2, 1, 1);
    k.declare(GMEM2, FLAG);
    k.st(0, 0, GMEM, 0..4).fence(0, 1, FenceKind::AcqRel(Scope::Cluster));
    k.a(0, 0, st(MemOrder::Relaxed, Scope::Sys), GMEM2, FLAG);
    k.wait_until(1, 0, GMEM2, FLAG, Scope::Sys, 0b10, 1).fence(1, 1, FenceKind::AcqRel(Scope::Cluster)).ld(1, 0, GMEM, 0..4);
    assert!(clean(&k.run()));
}

/// Exact read-from: only the write actually read synchronises.
#[test]
fn exact_read_from_aba() {
    let mut k = K::new(1, 1, 3);
    k.st(0, 0, GMEM, 0..4).a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, FLAG);
    // C1 is unordered with C0 (relaxed spin on another word).
    k.st(1, 0, GMEM, 4..8).a(1, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, FLAG);
    k.a(2, 0, ld(MemOrder::Acquire, Scope::Gpu), GMEM2, FLAG);
    k.ld(2, 0, GMEM, 0..4).ld(2, 0, GMEM, 4..8);
    let r = k.run();
    let f = races(&r);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].bytes, 0..4);
}

/// Lane-precise publication across CTAs of one cluster.
fn lane_precise(mode: u8) -> Report {
    let mut k = K::new(1, 2, 2);
    k.declare(GMEM2, FLAG);
    match mode {
        0 | 1 => {
            k.st(0, 1, GMEM, 0..4);
            if mode == 1 {
                k.syncwarp(0, u32::MAX);
            }
            k.a(0, 0, st(MemOrder::Release, Scope::Cluster), GMEM2, FLAG);
            k.wait_until(1, 0, GMEM2, FLAG, Scope::Cluster, 0b10, 1).ld(1, 0, GMEM, 0..4);
        }
        2 | 3 => {
            k.st(0, 0, GMEM, 0..4).a(0, 0, st(MemOrder::Release, Scope::Cluster), GMEM2, FLAG);
            k.wait_until(1, 0, GMEM2, FLAG, Scope::Cluster, 0b10, 1);
            if mode == 3 {
                k.syncwarp(1, u32::MAX);
            }
            k.ld(1, 1, GMEM, 0..4);
        }
        4 => {
            k.inst(0, &[1, 2], PLAIN_ST, |l| (GMEM, (l as u64 - 1) * 4..(l as u64) * 4));
            k.bar(50, &[0]);
            k.a(0, 0, st(MemOrder::Release, Scope::Cluster), GMEM2, FLAG);
            k.wait_until(1, 0, GMEM2, FLAG, Scope::Cluster, 0b10, 1).ld(1, 0, GMEM, 0..8);
        }
        _ => unreachable!(),
    }
    k.run()
}

#[test]
fn lane_precise_publication() {
    assert!(has_race(&lane_precise(0)));
    assert!(clean(&lane_precise(1)));
    assert!(has_race(&lane_precise(2)));
    assert!(clean(&lane_precise(3)));
    assert!(clean(&lane_precise(4)));
}

#[test]
fn torn_scoped_observation() {
    let mut k = K::new(1, 1, 3);
    k.a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, 0..8);
    k.a(1, 0, st(MemOrder::Relaxed, Scope::Gpu), GMEM2, 4..8);
    let r = k.run();
    let f = races(&r);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].bytes, 4..8);
    assert!(has_failure(&r, |f| f == OrderingFailure::MissingReleaseAcquire));
    assert_eq!(f[0].prior.as_ref().unwrap().span, 0..8);
}

#[test]
fn rmw_cross_cluster_cta_scope_is_not_morally_strong() {
    // Legacy shared shadow exempts every RMW/RMW pair; PTX requires mutual
    // scope coverage (global shadow reports a scope_mismatch here).
    let mut k = K::new(1, 1, 2);
    k.a(0, 0, atom(MemOrder::Relaxed, Scope::Cluster), GMEM2, FLAG);
    k.a(1, 0, atom(MemOrder::Relaxed, Scope::Cluster), GMEM2, FLAG);
    assert!(has_failure(&k.run(), |f| f == OrderingFailure::MissingReleaseAcquire));
}

// ------------------------------------------------------------ atomics --

#[test]
fn atomic_semantics() {
    // unordered atomic vs plain
    let mut k = K::new(2, 1, 1);
    k.st(0, 0, SMEM, 0..4).bar(0, &[0, 1]).a(0, 0, atom(MemOrder::Relaxed, Scope::Gpu), SMEM, 0..4).st(1, 0, SMEM, 0..4);
    let r = k.run();
    assert_eq!(races(&r).len(), 1);
    assert!(has_failure(&r, |f| f == OrderingFailure::MissingReleaseAcquire));
    // unordered atomic vs atomic
    let mut k = K::new(2, 1, 1);
    k.st(0, 0, SMEM, 0..4).bar(0, &[0, 1]);
    k.a(0, 0, atom(MemOrder::Relaxed, Scope::Gpu), SMEM, 0..4).a(1, 0, atom(MemOrder::Relaxed, Scope::Gpu), SMEM, 0..4);
    assert!(clean(&k.run()));
    // barrier orders atomic -> plain read
    let mut k = K::new(2, 1, 1);
    k.a(0, 0, atom(MemOrder::Relaxed, Scope::Gpu), SMEM, 0..4).bar(5, &[0, 1]).ld(1, 0, SMEM, 0..4);
    assert!(clean(&k.run()));
}

#[test]
fn reusable_named_barrier_and_long_loop() {
    let mut k = K::new(2, 1, 1);
    for _ in 0..64 {
        k.st(1, 0, SMEM, 0..4).bar(5, &[0, 1]).ld(0, 0, SMEM, 0..4).bar(6, &[0, 1]);
    }
    for step in 0..1152u64 {
        k.st(0, 0, SMEM, 64 + (step % 64) * 4..64 + (step % 64) * 4 + 4);
    }
    assert!(clean(&k.run()));
}

// --------------------------------------------------------- lane order --

fn lane_handoff(sync: u8, atomic: bool) -> Report {
    let mut k = K::one_warp();
    if atomic {
        k.st(0, 0, SMEM, 0..4).syncwarp(0, u32::MAX);
        k.inst(0, &[0, 1], atom(MemOrder::Relaxed, Scope::Cta), |_| (SMEM, 0..4));
    } else {
        k.st(0, 0, SMEM, 0..4);
    }
    match sync {
        1 => {
            k.syncwarp(0, u32::MAX);
        }
        2 => {
            k.bar(0, &[0]);
        }
        _ => {}
    }
    k.ld(0, 1, SMEM, 0..4);
    k.run()
}

#[test]
fn lane_handoff_matrix() {
    let r = lane_handoff(0, false);
    let f = races(&r);
    assert_eq!(f.len(), 1);
    assert!(has_failure(&r, |f| f == OrderingFailure::MissingSameWarpLaneOrder));
    assert_eq!((f[0].prior.as_ref().unwrap().lane, f[0].current.as_ref().unwrap().lane), (0, 1));
    assert!(clean(&lane_handoff(1, false)));
    assert!(clean(&lane_handoff(2, false)));
    assert!(has_failure(&lane_handoff(0, true), |f| f == OrderingFailure::MissingSameWarpLaneOrder));
    assert!(clean(&lane_handoff(1, true)));
    assert!(clean(&lane_handoff(2, true)));
}

#[test]
fn same_lane_program_order_and_warp_atomics() {
    let mut k = K::one_warp();
    k.st(0, 0, SMEM, 0..4).ld(0, 0, SMEM, 0..4);
    let all: Vec<u8> = (0..32).collect();
    k.inst(0, &all, atom(MemOrder::Relaxed, Scope::Gpu), |_| (GMEM, 0..4));
    k.inst(0, &all, atom(MemOrder::Relaxed, Scope::Gpu), |l| (GMEM2, l as u64 * 4..l as u64 * 4 + 4));
    assert!(clean(&k.run()));
}

#[test]
fn atomic_poll_does_not_publish_shared() {
    let mut k = K::new(2, 1, 1);
    k.declare(GMEM, FLAG);
    k.st(1, 0, SMEM, 0..4).a(1, 0, atom(MemOrder::Relaxed, Scope::Gpu), GMEM, FLAG);
    k.wait_until(0, 0, GMEM, FLAG, Scope::Gpu, 0b10, 1).ld(0, 0, SMEM, 0..4);
    let r = k.run();
    assert_eq!(races(&r).len(), 1);
    assert_eq!(races(&r)[0].prior.as_ref().unwrap().warp, 1);
}

/// test_native_racecheck_release_rmw_handoff.py: same-address RMWs of one
/// instruction are 32 independent threads with unconstrained coherence
/// order (PTX §8.9.1, §8.9.2; racecheck-isa-answers.md R1), so L1's acquire
/// may read from an RMW coherence-before L0's: no observation-order chain,
/// race. With warp_sync after the RMWs it is ordered.
#[test]
fn release_rmw_handoff_sibling_lanes_race() {
    let all: Vec<u8> = (0..32).collect();
    let mut k = K::one_warp();
    k.st(0, 0, GMEM, 0..4);
    k.inst(0, &all, atom(MemOrder::Release, Scope::Gpu), |_| (GMEM2, FLAG));
    k.a(0, 1, ld(MemOrder::Acquire, Scope::Gpu), GMEM2, FLAG).ld(0, 1, GMEM, 0..4);
    let r = k.run();
    let f = races(&r);
    assert_eq!(f.len(), 1);
    assert_eq!((f[0].prior.as_ref().unwrap().lane, f[0].current.as_ref().unwrap().lane), (0, 1));
    let mut k = K::one_warp();
    k.st(0, 0, GMEM, 0..4);
    k.inst(0, &all, atom(MemOrder::Release, Scope::Gpu), |_| (GMEM2, FLAG));
    k.syncwarp(0, u32::MAX);
    k.a(0, 1, ld(MemOrder::Acquire, Scope::Gpu), GMEM2, FLAG).ld(0, 1, GMEM, 0..4);
    assert!(clean(&k.run()));
    // A chain across *different*, ordered instructions does extend it.
    let mut k = K::new(1, 1, 3);
    k.st(0, 0, GMEM, 0..4).a(0, 0, atom(MemOrder::Release, Scope::Gpu), GMEM2, FLAG);
    k.bar(0, &[0, 1]);
    k.a(1, 0, atom(MemOrder::Relaxed, Scope::Gpu), GMEM2, FLAG);
    k.a(2, 0, ld(MemOrder::Acquire, Scope::Gpu), GMEM2, FLAG).ld(2, 0, GMEM, 0..4);
    assert!(clean(&k.run()));
    // Without any release the same shape races.
    let mut k = K::one_warp();
    k.st(0, 0, GMEM, 0..4);
    k.inst(0, &all, atom(MemOrder::Relaxed, Scope::Gpu), |_| (GMEM2, FLAG));
    k.a(0, 1, ld(MemOrder::Acquire, Scope::Gpu), GMEM2, FLAG).ld(0, 1, GMEM, 0..4);
    assert!(has_failure(&k.run(), |f| f == OrderingFailure::MissingSameWarpLaneOrder));
}

// ------------------------------------------------- shared publication --

fn relay(mode: &str) -> Report {
    let mut k = K::new(3, 1, 1);
    k.declare(GMEM, 4..8);
    k.st(0, 0, SMEM, 0..4).a(0, 0, st(MemOrder::Release, Scope::Cta), GMEM, 0..4);
    k.a(1, 0, ld(MemOrder::Acquire, Scope::Cta), GMEM, 0..4);
    match mode {
        "warp" => {
            k.syncwarp(1, u32::MAX).ld(1, 1, SMEM, 0..4);
        }
        "partial_warp" => {
            k.syncwarp(1, 0b110).ld(1, 1, SMEM, 0..4);
        }
        "named" => {
            k.bar(6, &[1, 2]).ld(2, 0, SMEM, 0..4);
        }
        "release" | "relaxed" => {
            let order = if mode == "release" { MemOrder::Release } else { MemOrder::Relaxed };
            k.a(1, 0, st(order, Scope::Cta), GMEM, 4..8);
            k.wait_until(2, 0, GMEM, 4..8, Scope::Cta, 0b10, 1).ld(2, 0, SMEM, 0..4);
        }
        _ => unreachable!(),
    }
    k.run()
}

#[test]
fn shared_frontier_relay() {
    assert!(clean(&relay("warp")));
    assert!(has_race(&relay("partial_warp")));
    assert!(clean(&relay("named")));
    assert!(clean(&relay("release")));
    assert!(has_race(&relay("relaxed")));
}

/// SC fences are totally ordered; acq_rel fences without a flag are not.
fn sc(kind: FenceKind, publisher: u8, consumer: u8, alloc: AllocId) -> Report {
    let mut k = K::one_warp();
    k.st(0, 0, alloc, 0..4).syncwarp(0, u32::MAX).st(0, 0, alloc, 0..4);
    k.fence(0, 1 << publisher, kind).fence(0, 1 << consumer, kind);
    k.ld(0, 1, alloc, 0..4);
    k.run()
}

#[test]
fn sc_causality_and_lane_controls() {
    for alloc in [SMEM, GMEM] {
        assert!(clean(&sc(FenceKind::Sc(Scope::Cta), 0, 1, alloc)));
        assert!(has_race(&sc(FenceKind::AcqRel(Scope::Cta), 0, 1, alloc)));
        assert!(has_race(&sc(FenceKind::Sc(Scope::Cta), 2, 1, alloc)));
        assert!(has_race(&sc(FenceKind::Sc(Scope::Cta), 0, 2, alloc)));
    }
}

/// A late wait acquires only the arrive snapshot.
#[test]
fn late_wait_acquires_arrive_snapshot() {
    for overwrite in [false, true] {
        let mut k = K::new(2, 1, 1);
        k.st(1, 0, SMEM, 0..4).arrive(1, 1, 3, 0, true);
        if overwrite {
            k.st(1, 0, SMEM, 0..4);
        }
        k.wait(0, 1, 3, 0, true).ld(0, 0, SMEM, 0..4);
        let r = k.run();
        assert_eq!(races(&r).len(), overwrite as usize);
    }
}

#[test]
fn sub_word_accesses() {
    for ordered in [true, false] {
        let mut k = K::new(2, 1, 1);
        let all: Vec<u8> = (0..32).collect();
        k.inst(0, &all, PLAIN_ST, |l| (SMEM, l as u64..l as u64 + 1));
        if ordered {
            k.bar(0, &[0, 1]);
        }
        k.inst(1, &all, PLAIN_LD, |l| (SMEM, l as u64..l as u64 + 1));
        let r = k.run();
        if ordered {
            assert!(clean(&r));
        } else {
            let f = races(&r);
            assert!(!f.is_empty());
            for f in f {
                assert_eq!(f.prior.as_ref().unwrap().span.end - f.prior.as_ref().unwrap().span.start, 1);
            }
        }
    }
}

#[test]
fn matrix_collectives_order_lanes() {
    // ldmatrix/stmatrix .sync.aligned are intra-warp rendezvous before and
    // after the access (producer emits WarpSync around them).
    let mut k = K::one_warp();
    let all: Vec<u8> = (0..32).collect();
    k.inst(0, &all, PLAIN_ST, |l| (SMEM, l as u64 * 16..l as u64 * 16 + 16));
    k.syncwarp(0, u32::MAX);
    k.inst(0, &all, PLAIN_LD, |l| (SMEM, ((l as u64 + 1) % 32) * 16..((l as u64 + 1) % 32) * 16 + 16));
    k.syncwarp(0, u32::MAX);
    k.inst(0, &all, PLAIN_ST, |l| (SMEM, ((l as u64 + 3) % 32) * 16..((l as u64 + 3) % 32) * 16 + 16));
    assert!(clean(&k.run()));
}
