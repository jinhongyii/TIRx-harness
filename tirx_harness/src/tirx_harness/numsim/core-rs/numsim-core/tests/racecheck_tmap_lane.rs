//! Descriptor reads emitted as warp-lane `Proxy::TensorMap` accesses
//! (gdn_prefill_sm100: in-kernel `tensormap.replace` + TMA per work item).
#[path = "racecheck_common/mod.rs"]
mod common;
use common::*;
use numsim_core::arena::ByteSpan;

fn iteration(k: &mut K) {
    k.st(0, 0, GMEM2, 0..128);
    k.fence(0, u32::MAX, FenceKind::TensormapRelease { scope: Scope::Gpu });
    k.fence(0, 1, FenceKind::TensormapAcquire { scope: Scope::Gpu, alloc: GMEM2, span: ByteSpan::new(0, 128) });
    k.tmap_read(0, 0, GMEM2, 0..128);
}

#[test]
fn replace_release_acquire_tma_twice() {
    let mut k = K::one_warp();
    iteration(&mut k);
    let r1 = k.run();
    iteration(&mut k);
    let r = k.run();
    assert!(r1.findings.is_empty());
    assert!(r.findings.is_empty());
}

/// The descriptor read of iteration 1 does not race the issuing thread's
/// own later `tensormap.replace` (program order; deltas I9), but another
/// warp's unsynchronised rewrite still does.
#[test]
fn descriptor_war_is_ordered_by_hb() {
    let mut k = K::new(2, 1, 1);
    iteration(&mut k);
    k.st(1, 0, GMEM2, 0..128);
    assert!(has_race(&k.run()));
}

