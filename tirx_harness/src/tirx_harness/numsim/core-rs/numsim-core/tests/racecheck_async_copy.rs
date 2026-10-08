//! Translated from test_native_raw_async_copy_footprints.py and the copy
//! rows of test_native_async_lifetime_contracts.py.
#[path = "racecheck_common/mod.rs"]
mod common;
use common::*;

/// cp.async.mbarrier.arrive: the copy's completion arrives on the barrier.
fn cp_async_arrive(read_before_wait: bool) -> Report {
    let mut k = K::new(2, 1, 1);
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Generic, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Generic, GMEM, 0..16);
    k.arrive(0, 1, 1, 0, true);
    if read_before_wait {
        k.ld(1, 0, SMEM, 0..16);
    }
    k.aw(op, Proxy::Generic, SMEM, 0..16).done_phase(op, Milestone::Write, 1, 0);
    k.arrive(1, 1, 1, 0, true).wait(1, 1, 1, 0, true);
    if !read_before_wait {
        k.ld(1, 0, SMEM, 0..16);
    }
    k.run()
}

#[test]
fn cp_async_mbarrier_arrive_pending_count() {
    assert!(clean(&cp_async_arrive(false)));
    let r = cp_async_arrive(true);
    let f = races(&r);
    assert_eq!(f.len(), 1);
    assert!(has_class(&r, RaceClass::ReadWrite));
    assert!(f[0].current.as_ref().unwrap().async_op.is_some());
}

#[test]
fn bulk_g2s_waited_and_unwaited() {
    for waited in [true, false] {
        let mut k = K::one_warp();
        let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
        k.ar(op, Proxy::Async, GMEM, 0..16);
        if !waited {
            k.ld(0, 0, SMEM, 0..16);
        }
        k.aw(op, Proxy::Async, SMEM, 0..16).done_phase(op, Milestone::Write, 1, 0).wait(0, 1, 1, 0, true);
        if waited {
            k.ld(0, 0, SMEM, 0..16);
        }
        let r = k.run();
        if waited {
            assert!(clean(&r));
        } else {
            assert!(has_class(&r, RaceClass::ReadWrite));
            assert_eq!(races(&r)[0].current.as_ref().unwrap().span, 0..16);
        }
    }
}

/// Multicast: one op writes both CTAs' smem; each CTA waits its own bar.
fn multicast(mask: u8, c1_reads_early: bool) -> Report {
    let mut k = K::new(1, 2, 2);
    let mut fp = vec![(SMEM, 0..16)];
    if mask & 2 != 0 {
        fp.push((SMEM1, 0..16));
    }
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &fp);
    k.ar(op, Proxy::Async, GMEM, 0..16);
    if c1_reads_early {
        k.ld(1, 0, SMEM1, 0..16);
    }
    k.aw(op, Proxy::Async, SMEM, 0..16).done_phase(op, Milestone::Write, 1, 0);
    if mask & 2 != 0 {
        k.aw(op, Proxy::Async, SMEM1, 0..16).done_phase(op, Milestone::Write, 2, 0);
    }
    k.wait(0, 1, 1, 0, true).ld(0, 0, SMEM, 0..16);
    if mask & 2 != 0 {
        k.wait(1, 1, 2, 0, true);
    }
    if !c1_reads_early {
        k.ld(1, 0, SMEM1, 0..16);
    }
    k.run()
}

#[test]
fn multicast_cases() {
    assert!(clean(&multicast(0b11, false)));
    let r = multicast(0b11, true);
    assert!(has_class(&r, RaceClass::ReadWrite));
    assert_eq!(races(&r)[0].prior.as_ref().unwrap().warp, 1);
    // Unselected CTA: no race (the legacy verdict may be review for an
    // uninitialised-read advisory, which is out of this core's scope).
    assert!(races(&multicast(0b01, true)).is_empty());
}

/// s2g cp_mask: the mask narrows the footprint.
fn cp_mask(byte: u64) -> Report {
    let mut k = K::one_warp();
    k.st(0, 0, SMEM, 0..16).fence(0, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, SMEM, 0..16);
    k.st(0, 0, GMEM, byte..byte + 1);
    k.aw(op, Proxy::Async, GMEM, 0..8).done_warp(op, Milestone::Write, 0, 1);
    k.run()
}

#[test]
fn s2g_cp_mask_footprint() {
    let r = cp_mask(3);
    assert!(has_class(&r, RaceClass::WriteWrite));
    assert_eq!(races(&r)[0].bytes, 3..4);
    assert!(clean(&cp_mask(11)));
}

/// TMA store FIFO: wait_group.read N completes the read side of all but the
/// N most recent groups.
fn fifo(overwrite: usize, read_wait_drains: usize) -> Report {
    let mut k = K::new(2, 1, 1);
    let srcs = [0..16u64, 16..32, 32..48];
    k.st(0, 0, SMEM, 0..48).fence(0, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    let mut ops = vec![];
    for (i, s) in srcs.iter().enumerate().take(2) {
        let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, s.clone())]);
        k.ar(op, Proxy::Async, SMEM, s.clone()).aw(op, Proxy::Async, GMEM, (i as u64 * 16)..(i as u64 * 16 + 16));
        ops.push(op);
    }
    for &op in ops.iter().take(read_wait_drains) {
        k.done_warp(op, Milestone::Read, 0, 1);
    }
    k.arrive(0, 1, 3, 0, true).wait(1, u32::MAX, 3, 0, true);
    k.st(1, 0, SMEM, srcs[overwrite].clone());
    for &op in &ops {
        k.done_warp(op, Milestone::Write, 0, 1);
    }
    k.run()
}

#[test]
fn tma_store_fifo_groups() {
    // wait_group.read 1 with two groups drains only the first.
    assert!(clean(&fifo(0, 1)));
    assert!(has_class(&fifo(1, 1), RaceClass::ReadWrite));
    assert!(clean(&fifo(1, 2)));
}

#[test]
fn uncommitted_tma_store_owns_source() {
    for overwrite in [false, true] {
        let mut k = K::one_warp();
        k.st(0, 0, SMEM, 0..16).fence(0, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
        let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
        k.ar(op, Proxy::Async, SMEM, 0..16);
        if overwrite {
            k.st(0, 0, SMEM, 0..4);
        }
        let r = k.run();
        assert_eq!(!races(&r).is_empty(), overwrite);
        assert!(r.incomplete.iter().any(|i| matches!(i, Incomplete::AsyncNeverCompleted { .. })));
    }
}

/// Classic cp.async (generic proxy) lifetime.
fn cp_async(mode: u8) -> Report {
    let mut k = K::one_warp();
    let lanes: Vec<u8> = (0..32).collect();
    let mut ops = vec![];
    for &l in &lanes {
        let op = k.issue(0, l, AsyncKind::Copy, Proxy::Generic, &[], &[(SMEM, (l as u64 * 16)..(l as u64 * 16 + 16))]);
        k.ar(op, Proxy::Generic, GMEM, (l as u64 * 16)..(l as u64 * 16 + 16));
        k.aw(op, Proxy::Generic, SMEM, (l as u64 * 16)..(l as u64 * 16 + 16));
        ops.push(op);
    }
    let wait = |k: &mut K| {
        for (l, &op) in ops.iter().enumerate() {
            k.done_warp(op, Milestone::Write, 0, 1 << l);
        }
    };
    match mode {
        0 => {
            k.inst(0, &lanes, PLAIN_ST, |l| (GMEM, l as u64 * 16..l as u64 * 16 + 4));
            wait(&mut k);
        }
        2 => {
            k.inst(0, &lanes, PLAIN_LD, |l| (SMEM, l as u64 * 16..l as u64 * 16 + 4));
            wait(&mut k);
        }
        _ => {
            wait(&mut k);
            k.inst(0, &lanes, PLAIN_LD, |l| (SMEM, l as u64 * 16..l as u64 * 16 + 4));
        }
    }
    k.run()
}

#[test]
fn classic_cp_async_lifetime() {
    assert!(has_class(&cp_async(0), RaceClass::ReadWrite));
    assert!(has_class(&cp_async(2), RaceClass::WriteRead));
    assert!(clean(&cp_async(1)));
}

#[test]
fn cp_async_after_prior_same_lane_write() {
    let mut k = K::one_warp();
    k.st(0, 0, SMEM, 0..16);
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Generic, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Generic, GMEM, 0..16).aw(op, Proxy::Generic, SMEM, 0..16).done_warp(op, Milestone::Write, 0, 1);
    assert!(clean(&k.run()));
}

#[test]
fn cp_async_cross_warp_source_race() {
    let mut k = K::new(2, 1, 1);
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Generic, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Generic, GMEM, 0..16).aw(op, Proxy::Generic, SMEM, 0..16).done_warp(op, Milestone::Write, 0, 1);
    k.st(1, 0, GMEM, 0..4);
    let r = k.run();
    assert!(has_failure(&r, |f| f == OrderingFailure::MissingInterActorSync));
}

#[test]
fn tma_store_cross_warp_race_after_wait() {
    let mut k = K::new(2, 1, 1);
    for w in 0..2 {
        let op = k.issue(w, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
        k.ar(op, Proxy::Async, SMEM, 0..16).aw(op, Proxy::Async, GMEM, 0..16).done_warp(op, Milestone::Write, w, 1);
    }
    let r = k.run();
    assert!(has_failure(&r, |f| f == OrderingFailure::MissingInterActorSync));
}

/// Memory reuse under an unfinished async op, and out-of-bounds. (A shared
/// allocation ends only at CTA exit, which drains bulk copies — ruling S7 —
/// so the lifetime case uses a non-shared allocation.)
#[test]
fn reuse_and_oob() {
    let mut k = K::one_warp();
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(GMEM2, 0..16)]);
    k.ar(op, Proxy::Async, GMEM, 0..16);
    k.alloc_end(GMEM2);
    k.st(0, 0, GMEM, 4090..4100);
    let r = k.run();
    assert!(r.findings.iter().any(|f| matches!(f.kind, FindingKind::AsyncLifetime { .. })));
    assert!(r.findings.iter().any(|f| matches!(f.kind, FindingKind::OutOfBounds { size: 4096 })));
}

/// sync-isa-answers.md Q7: `cp.async.bulk.wait_group.read` completes only
/// the source reads; destination writes stay unpublished.
#[test]
fn bulk_wait_read_does_not_publish_destination() {
    for full in [false, true] {
        let mut k = K::one_warp();
        k.st(0, 0, SMEM, 0..16).fence(0, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
        let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
        k.ar(op, Proxy::Async, SMEM, 0..16).aw(op, Proxy::Async, GMEM, 0..16);
        k.done_warp(op, if full { Milestone::Write } else { Milestone::Read }, 0, 1);
        k.st(0, 0, SMEM, 0..16); // source reuse: fine either way
        k.ld(0, 0, GMEM, 0..16); // destination: needs the full wait
        let r = k.run();
        assert_eq!(has_race(&r), !full);
        assert!(races(&r).iter().all(|f| f.alloc == GMEM));
    }
}

/// Q7: async-group completion is per thread; another lane of the same warp
/// does not see it without a warp sync.
#[test]
fn async_group_wait_is_per_lane() {
    for sync in [false, true] {
        let mut k = K::one_warp();
        let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Generic, &[], &[(SMEM, 0..16)]);
        k.ar(op, Proxy::Generic, GMEM, 0..16).aw(op, Proxy::Generic, SMEM, 0..16);
        k.done_warp(op, Milestone::Write, 0, 0b1);
        if sync {
            k.syncwarp(0, u32::MAX);
        }
        k.ld(0, 1, SMEM, 0..4);
        assert_eq!(has_race(&k.run()), !sync);
    }
}

/// Ruling S7: a TMA store still in flight when its CTA exits (no final
/// `cp.async.bulk.wait_group`) is drained by the hardware: neither
/// `AsyncLifetime` nor `AsyncNeverCompleted`. Its global write stays
/// unordered with a later access by a still-running CTA.
#[test]
fn bulk_store_in_flight_at_cta_exit_is_drained() {
    let run = |later_cta_write: bool| {
        let mut k = K::new(1, 1, 2);
        k.st(0, 0, SMEM, 0..16).fence(0, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
        let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16), (GMEM, 0..16)]);
        k.ar(op, Proxy::Async, SMEM, 0..16).aw(op, Proxy::Async, GMEM, 0..16);
        k.alloc_end(SMEM);
        if later_cta_write {
            k.st(1, 0, GMEM, 0..4);
        }
        k.run()
    };
    let r = run(false);
    assert!(r.findings.is_empty() && r.incomplete.is_empty(), "{r:?}");
    assert!(has_race(&run(true)));
}

/// S7: a bulk copy with an empty footprint (TMA store whose box is entirely
/// out of bounds) still in flight at CTA exit drains with the CTA.
#[test]
fn empty_footprint_bulk_copy_drains_at_cta_exit() {
    let mut k = K::one_warp();
    let _op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[]);
    k.alloc_end(SMEM);
    let r = k.run();
    assert!(r.incomplete.is_empty(), "{r:?}");
}
