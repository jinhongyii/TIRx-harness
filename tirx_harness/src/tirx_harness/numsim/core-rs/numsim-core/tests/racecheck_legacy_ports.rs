//! Ports of the legacy Python racecheck tests
//! (tests/analysis_tools/racecheck/**/test_*.py) that no other
//! `racecheck_*.rs` file already covered. Each test's doc comment names the
//! legacy `file.py::test_name[param]` it reproduces; where the legacy
//! expectation is overridden by a documented behaviour delta, the test asserts
//! the new behaviour and cites the row of
//! docs/development/racecheck-behaviour-deltas.md. The coverage map is
//! scripts/numsim-v2/coverage/racecheck.tsv.
//!
//! Sections are prefixed g1_..g5_ after the port batches:
//!   g1: racecheck_artifact, raw_async_copy_footprints
//!   g2: declared_word_*, tcgen_thread_fence, same_warp_tmem_review, tmem_arrive_snapshot
//!   g3: proxy_async_fence, global_scoped_hb_matrix, async_lifetime_contracts, shared_publication
//!   g4: exact_oob, global_write_seed, alias_advisory, lane_order, exact_control, global_scoped_hb
//!   g5: the remaining small files (atomic_semantics, arrive_snapshot, vector loads, ...)
//!   g6: checker verdicts asserted outside analysis_tools/racecheck (numsim/runtime,
//!       numsim/integration, analysis_tools/shared); map: coverage/other_b.tsv
#![allow(dead_code, unused_imports, clippy::too_many_arguments)]
#[path = "racecheck_common/mod.rs"]
mod common;
use common::*;
use numsim_core::arena::{ByteSpan, Space};
use numsim_core::observe::{Actor, LaneSpan, Window};
use numsim_core::program::Sem;
use numsim_core::site::SiteId;


// ======================================================================= g1

// ------------------------------------------------ test_native_racecheck_artifact.py --

/// test_native_racecheck_artifact.py::test_native_racecheck_reports_exact_cross_warp_shared_write_race
/// Lane 0 of two warps writes the same shared word: one write_write race
/// with exact prior/current evidence and a 4-byte overlap.
#[test]
fn g1_cross_warp_shared_write_write_race() {
    let mut k = K::new(2, 1, 1);
    k.st(0, 0, SMEM, 0..4).st(1, 0, SMEM, 0..4);
    let r = k.run();
    let f = races(&r);
    assert_eq!(f.len(), 1);
    assert!(has_class(&r, RaceClass::WriteWrite));
    assert!(r.incomplete.is_empty());
    let (p, c) = (f[0].prior.as_ref().unwrap(), f[0].current.as_ref().unwrap());
    assert_eq!((p.warp, p.lane, c.warp, c.lane), (0, 0, 1, 0));
    assert_eq!(f[0].bytes, 0..4);
}

/// test_native_racecheck_artifact.py::test_native_racecheck_oob_is_exact_error_or_clean_from_concrete_mask
/// An active lane reading past the end is an error (never review); the
/// masked run touches only in-bounds bytes and is clean.
/// delta P5: legacy reported `execution_error{oob}` with no finding; new: an
/// `OutOfBounds` finding.
#[test]
fn g1_lane_selected_oob_is_error_or_clean() {
    let mut k = K::one_warp();
    k.ld(0, 1, GMEM, 4096..4100);
    let r = k.run();
    assert!(r.findings.iter().any(|f| matches!(f.kind, FindingKind::OutOfBounds { size: 4096 })));
    assert!(r.findings.iter().all(|f| f.severity != Severity::Review));
    assert!(r.incomplete.is_empty());
    let mut k = K::one_warp();
    k.ld(0, 0, GMEM2, 0..4).st(0, 0, GMEM, 0..4);
    let r = k.run();
    assert!(clean(&r) && r.findings.is_empty());
}

/// test_native_racecheck_artifact.py::test_native_racecheck_raw_bulk_uses_the_concrete_active_issuer_lane
/// A bulk g2s copy issued, waited and consumed by lane 1 only is clean.
#[test]
fn g1_raw_bulk_active_issuer_lane_clean() {
    let mut k = K::one_warp();
    let op = k.issue(0, 1, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..32)]);
    k.ar(op, Proxy::Async, GMEM, 0..32).aw(op, Proxy::Async, SMEM, 0..32).done_phase(op, Milestone::Write, 1, 0);
    k.wait(0, 1 << 1, 1, 0, true).ld(0, 1, SMEM, 0..32);
    let r = k.run();
    assert!(clean(&r) && r.findings.is_empty());
}

/// Two warps each bulk-copy into the same smem bytes, each completing on and
/// waiting its own mbarrier: the async writes are unordered.
fn g1_two_warp_bulk_ww(bytes: u64) -> Report {
    let mut k = K::new(2, 1, 1);
    k.bar(0, &[0, 1]);
    for w in 0..2u32 {
        let op = k.issue(w, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..bytes)]);
        k.ar(op, Proxy::Async, GMEM, 0..bytes).aw(op, Proxy::Async, SMEM, 0..bytes).done_phase(op, Milestone::Write, w + 1, 0);
        k.wait(w, 1, w + 1, 0, true);
    }
    k.run()
}

/// test_native_racecheck_artifact.py::test_native_racecheck_raw_bulk_rejects_a_true_async_write_race
/// test_native_racecheck_artifact.py::test_public_native_racecheck_typed_tma_overlapping_destinations_race
/// test_native_racecheck_artifact.py::test_public_native_racecheck_raw_tma_overlapping_destinations_race
/// Overlapping async destinations of two warps: write_write,
/// missing_inter_actor_sync, overlap = the full footprint.
#[test]
fn g1_two_warp_async_destinations_race() {
    for bytes in [32u64, 16, 4] {
        let r = g1_two_warp_bulk_ww(bytes);
        let f = races(&r);
        assert_eq!(f.len(), 1, "{bytes}: {:?}", r.findings);
        assert!(has_class(&r, RaceClass::WriteWrite));
        assert!(has_failure(&r, |f| f == OrderingFailure::MissingInterActorSync));
        assert_eq!(f[0].alloc, SMEM);
        assert_eq!(f[0].bytes, 0..bytes);
        assert!(f[0].prior.as_ref().unwrap().async_op.is_some() && f[0].current.as_ref().unwrap().async_op.is_some());
        assert!(r.findings.iter().all(|f| f.severity != Severity::Review));
    }
}

/// test_native_racecheck_artifact.py::test_public_native_racecheck_typed_tma_uses_exact_global_subregion
/// The TMA reads only source[1,:,:] (bytes 128..256); another warp's
/// unsynchronised write to source[0,0,0] is disjoint and clean.
#[test]
fn g1_tma_exact_global_subregion_clean() {
    let mut k = K::new(2, 1, 1);
    k.bar(0, &[0, 1]);
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..128)]);
    k.ar(op, Proxy::Async, GMEM, 128..256).aw(op, Proxy::Async, SMEM, 0..128).done_phase(op, Milestone::Write, 1, 0);
    k.wait(0, 1, 1, 0, true);
    k.st(1, 0, GMEM, 0..4);
    let r = k.run();
    assert!(clean(&r) && r.findings.is_empty());
}

/// test_native_racecheck_artifact.py::test_native_racecheck_mbarrier_release_acquire_orders_shared_access
#[test]
fn g1_mbarrier_release_acquire_orders_shared() {
    let mut k = K::new(2, 1, 1);
    k.bar(0, &[0, 1]);
    k.st(0, 0, SMEM, 0..4).arrive(0, 1, 1, 0, true);
    k.wait(1, 1, 1, 0, true).ld(1, 0, SMEM, 0..4).st(1, 0, GMEM, 0..4);
    let r = k.run();
    assert!(clean(&r) && r.findings.is_empty());
}

/// test_native_racecheck_artifact.py::test_public_native_racecheck_accepts_cluster_arrive_through_remote_view
/// CTA 1 arrives on CTA 0's barrier through a mapa view; CTA 0 waits; both
/// then cluster-sync and write disjoint output words.
/// delta B1/V5: the raw `mbarrier.arrive.shared::cluster.b64` has the PTX
/// default `.release.cta`, so a remote arrive observed by a `.cta` wait
/// (with the default `.acquire.cta` wait) fails mutual scope inclusion.
/// Legacy (no scope model) expected clean; new: a `ScopeMismatch` error even
/// though no data flows through the edge. With `.cluster` on both the arrive
/// and the wait the kernel is clean.
fn g1_remote_view(scope: Scope) -> Report {
    let mut k = K::new(1, 2, 2);
    k.cluster_bar(&[0, 1]);
    k.arrive_q(1, 1, mbar_in(0, 0), 0, Some(true), Some(scope));
    k.wait_q(0, 1, mbar_in(0, 0), 0, Some(true), Some(scope));
    k.cluster_bar(&[0, 1]);
    k.st(0, 0, GMEM, 0..4).st(1, 0, GMEM, 4..8);
    k.run()
}

/// test_native_racecheck_artifact.py::test_public_native_racecheck_accepts_cluster_arrive_through_remote_view
#[test]
fn g1_remote_view_cluster_arrive() {
    let r = g1_remote_view(Scope::Cta);
    assert!(r.incomplete.is_empty(), "{:?}", r.incomplete);
    assert!(has_scope_mismatch(&r) && !has_race(&r), "{:?}", r.findings);
    let r = g1_remote_view(Scope::Cluster);
    assert!(clean(&r) && r.findings.is_empty(), "{:?}", r.findings);
}

/// test_native_racecheck_artifact.py::test_native_racecheck_reports_cross_cluster_global_waw_independent_of_workers
#[test]
fn g1_cross_cluster_global_waw() {
    let mut k = K::new(1, 1, 2);
    k.st(0, 0, GMEM, 0..4).st(1, 0, GMEM, 0..4);
    let r = k.run();
    assert_eq!(races(&r).len(), 1);
    assert!(has_class(&r, RaceClass::WriteWrite));
    assert_eq!(races(&r)[0].alloc, GMEM);
    assert!(r.incomplete.is_empty());
}

/// test_native_racecheck_artifact.py::test_native_racecheck_reports_same_cluster_global_waw
#[test]
fn g1_same_cluster_global_waw() {
    let mut k = K::new(2, 1, 1);
    k.st(0, 0, GMEM, 0..4).st(1, 0, GMEM, 0..4);
    let r = k.run();
    assert_eq!(races(&r).len(), 1);
    assert!(has_class(&r, RaceClass::WriteWrite));
    assert_eq!(races(&r)[0].alloc, GMEM);
    assert!(r.incomplete.is_empty());
}

/// test_native_racecheck_artifact.py::test_native_racecheck_reports_cross_cluster_global_tile_copy_conflict
/// Lane 0 of each cluster's CTA copies 128 floats into the same output: one
/// (deduplicated) global write_write finding.
#[test]
fn g1_cross_cluster_wide_global_copy_conflict() {
    let mut k = K::new(1, 1, 2);
    for w in 0..2u32 {
        k.ld(w, 0, GMEM2, 0..512);
        for e in 0..128u64 {
            k.st(w, 0, GMEM, e * 4..e * 4 + 4);
        }
    }
    let r = k.run();
    assert!(has_class(&r, RaceClass::WriteWrite));
    assert!(races(&r).iter().all(|f| f.alloc == GMEM && f.prior.as_ref().unwrap().warp == 0));
    assert!(r.incomplete.is_empty());
}

/// test_native_racecheck_artifact.py::test_native_racecheck_cross_cluster_atomic_modification_order_is_clean
/// Default `atom.add` (relaxed, .gpu) from two clusters: morally strong, clean.
#[test]
fn g1_cross_cluster_gpu_atomics_clean() {
    let mut k = K::new(1, 1, 2);
    k.a(0, 0, atom(MemOrder::Relaxed, Scope::Gpu), GMEM, 0..4);
    k.a(1, 0, atom(MemOrder::Relaxed, Scope::Gpu), GMEM, 0..4);
    let r = k.run();
    assert!(clean(&r) && r.findings.is_empty());
}

// ----------------------------------------- test_native_raw_async_copy_footprints.py --

/// Raw DSMEM push: CTA 0 stages bytes, proxy-fences, and bulk-copies them into
/// CTA 1's landing buffer, completing on CTA 1's barrier.
fn g1_s2s_cluster(waited: bool) -> Report {
    let mut k = K::new(1, 2, 2);
    k.alloc_cta(SMEM1, 1);
    k.cluster_bar(&[0, 1]);
    let lanes: Vec<u8> = (0..16).collect();
    k.inst(0, &lanes, PLAIN_ST, |l| (SMEM, l as u64..l as u64 + 1));
    k.syncwarp(0, u32::MAX).fence(0, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM1, 0..16)]);
    k.ar(op, Proxy::Async, SMEM, 0..16);
    if !waited {
        k.ld(1, 0, SMEM1, 0..16);
    }
    k.aw(op, Proxy::Async, SMEM1, 0..16).done_phase_r(op, Milestone::Write, mbar_in(1, 0), 0);
    k.wait_r(1, 1, mbar_in(1, 0), 0, true);
    if waited {
        k.ld(1, 0, SMEM1, 0..16);
    }
    k.cluster_bar(&[0, 1]);
    k.run()
}

/// test_native_raw_async_copy_footprints.py::test_bulk_s2s_cluster_completes_clean
/// test_native_raw_async_copy_footprints.py::test_bulk_s2s_cluster_flags_the_unwaited_consumer
#[test]
fn g1_bulk_s2s_cluster_waited_and_unwaited() {
    let r = g1_s2s_cluster(true);
    assert!(clean(&r), "{:?} {:?}", r.findings, r.incomplete);
    let r = g1_s2s_cluster(false);
    assert!(r.incomplete.is_empty());
    assert!(has_class(&r, RaceClass::ReadWrite));
    let f = races(&r);
    let (p, c) = (f[0].prior.as_ref().unwrap(), f[0].current.as_ref().unwrap());
    assert_eq!(f[0].alloc, SMEM1);
    assert_eq!((p.warp, p.kind), (1, AccessKind::Read));
    assert_eq!((c.warp, c.kind), (0, AccessKind::Write));
    assert!(c.async_op.is_some());
}

/// test_native_raw_async_copy_footprints.py::test_bulk_s2g_masked_completes_clean
/// A masked s2g copy, fully waited (wait_group 0), then a generic write to a
/// selected destination byte: clean.
#[test]
fn g1_bulk_s2g_masked_waited_clean() {
    let mut k = K::one_warp();
    let lanes: Vec<u8> = (0..16).collect();
    k.inst(0, &lanes, PLAIN_ST, |l| (SMEM, l as u64..l as u64 + 1));
    k.syncwarp(0, u32::MAX).fence(0, u32::MAX, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, SMEM, 0..16).aw(op, Proxy::Async, GMEM, 0..8).done_warp(op, Milestone::Write, 0, 1);
    k.st(0, 0, GMEM, 0..1);
    let r = k.run();
    assert!(clean(&r) && r.findings.is_empty());
}


// ======================================================================= g2

// ===================================================== G2: declared words ==

const G2_FLAG: std::ops::Range<u64> = 0..4;

fn g2_no_incomplete(r: &Report) -> bool {
    r.incomplete.is_empty()
}

/// test_declared_word_regressions.py::test_raw_protocol_reports_only_the_proven_payload_race
///
/// A single `ld.relaxed` of an undeclared flag after `st.release`: the
/// payload is a `write_read` race. The flag pair itself is morally strong.
/// delta R3: legacy reported only the data race and no undeclared-word
/// entry; new: the strong flag pair is exempt and produces a review
/// `UndeclaredProtocolWord` advisory alongside the payload race.
#[test]
fn g2_raw_protocol_reports_only_the_proven_payload_race() {
    let mut k = K::new(1, 1, 2);
    k.st(0, 0, GMEM, 0..4).a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, G2_FLAG);
    k.a(1, 0, ld(MemOrder::Relaxed, Scope::Gpu), GMEM2, G2_FLAG).ld(1, 0, GMEM, 0..4);
    let r = k.run();
    let rs = races(&r);
    assert_eq!(rs.len(), 1, "{r:?}");
    assert_eq!(rs[0].alloc, GMEM);
    assert!(has_class(&r, RaceClass::WriteRead));
    assert!(has_advisory(&r, AdvisoryKind::UndeclaredProtocolWord), "{r:?}");
    assert!(g2_no_incomplete(&r));
}

/// Two warps of one CTA: warp 0 publishes the declared word with a
/// `st.release` of `width` bytes, warp 1 waits on it (accepting write 1).
fn g2_wide_publication(width: u64) -> Report {
    let mut k = K::new(2, 1, 1);
    k.declare(GMEM2, G2_FLAG);
    k.a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, 0..width);
    k.wait_until(1, 0, GMEM2, G2_FLAG, Scope::Gpu, 0b10, 1);
    k.run()
}

/// test_declared_word_regressions.py::test_a_publication_too_wide_to_poll_fails_closed
///
/// A `.v4.b32` release store over the declared word: legacy reported
/// `incomplete` (`signal_write_not_recorded`). racecheck-semantics.md §5
/// lists wide writes among the exits that must be `incomplete`. The new
/// core numbers the wide write as an ordinary history entry (V3: every
/// overlapping word) and, given an accepting verdict bit, reports clean.
/// No delta row states that wide publications are no longer incomplete.
#[test]
fn g2_publication_too_wide_to_poll_fails_closed() {
    let r = g2_wide_publication(16);
    assert!(r.incomplete.iter().any(|i| matches!(i, Incomplete::SignalWriteNotRecorded { .. })), "{r:?}");
}

/// test_declared_word_regressions.py::test_a_publication_a_wait_can_poll_is_clean
#[test]
fn g2_publication_a_wait_can_poll_is_clean() {
    let r = g2_wide_publication(4);
    assert!(r.is_clean(), "{r:?}");
}

/// test_declared_word_regressions.py::test_a_wait_does_not_bridge_the_async_proxy
///
/// Bulk S2G store of the payload, `wait_group.read 0` (read side only),
/// `fence.proxy.async.global`, then `st.release` of the declared flag;
/// the consumer waits and reads the payload: `missing_proxy_bridge`.
/// (The existing `wait_does_not_bridge_unfinished_async_store` asserts the
/// race class but not the failure kind the legacy test pins.)
#[test]
fn g2_wait_does_not_bridge_the_async_proxy() {
    let mut k = K::new(1, 1, 2);
    k.declare(GMEM2, G2_FLAG);
    k.st(0, 0, SMEM, 0..16).fence(0, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, SMEM, 0..16).aw(op, Proxy::Async, GMEM, 0..16);
    k.done_warp(op, Milestone::Read, 0, 1);
    k.fence(0, 1, FenceKind::ProxyAsync(Some(Domain::Global)));
    k.a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, G2_FLAG);
    k.wait_until(1, 0, GMEM2, G2_FLAG, Scope::Gpu, 0b10, 1).ld(1, 0, GMEM, 0..4);
    let r = k.run();
    assert!(has_race(&r), "{r:?}");
    assert!(has_failure(&r, |f| matches!(f, OrderingFailure::MissingProxyBridge { .. })), "{r:?}");
}

/// Ring-credit wait: C0 `red.release.gpu.add`, C1 waits `credit >= lap`.
fn g2_ring_credit(lap: u64) -> Report {
    let mut k = K::new(1, 1, 2);
    k.declare(GMEM2, G2_FLAG);
    k.red(0, 0, MemOrder::Release, Scope::Gpu, GMEM2, G2_FLAG);
    // lap 0: launch value and the red both satisfy `>= 0`; lap 1: only the red.
    let (accepted, observed) = if lap == 0 { (0b11, 0) } else { (0b10, 1) };
    k.wait_until(1, 0, GMEM2, G2_FLAG, Scope::Gpu, accepted, observed);
    k.run()
}

/// test_declared_word_regressions.py::test_a_first_lap_ring_credit_wait_is_explained
#[test]
fn g2_first_lap_ring_credit_wait_is_explained() {
    let r = g2_ring_credit(0);
    assert!(r.is_clean(), "{r:?}");
}

/// test_declared_word_regressions.py::test_a_later_lap_ring_credit_wait_still_needs_its_publication
#[test]
fn g2_later_lap_ring_credit_wait_is_published_by_red_release() {
    let r = g2_ring_credit(1);
    assert!(r.is_clean(), "{r:?}");
}

// ============================================ G2: declared-word shapes ==
// One cluster of two CTAs, one warp each; `slot` = GMEM2[0..8] (u64).

const G2_SLOT: std::ops::Range<u64> = 0..8;
const G2_RLX_CL_ST: Op = st(MemOrder::Relaxed, Scope::Cluster);
const G2_RLX_CL_LD: Op = ld(MemOrder::Relaxed, Scope::Cluster);

/// Prologue: C0 initialises the slot (relaxed.cluster), cluster barrier.
fn g2_shape_prologue(declared: bool) -> K {
    let mut k = K::new(1, 2, 2);
    if declared {
        k.declare(GMEM2, G2_SLOT);
    }
    k.a(0, 0, G2_RLX_CL_ST, GMEM2, G2_SLOT);
    k.cluster_bar(&[0, 1]);
    k
}

/// test_declared_word_shapes.py::test_declared_word_shape_gets_the_verdict_it_earns[payload_in_the_polled_word]
#[test]
fn g2_shape_payload_in_the_polled_word() {
    let mut k = g2_shape_prologue(true);
    k.a(0, 0, G2_RLX_CL_ST, GMEM2, G2_SLOT); // history 2
    k.a(1, 0, G2_RLX_CL_LD, GMEM2, G2_SLOT);
    k.wait_until(1, 0, GMEM2, G2_SLOT, Scope::Cluster, 0b100, 2);
    k.st(1, 0, GMEM, 0..8); // sink
    let r = k.run();
    assert!(r.is_clean(), "{r:?}");
}

/// test_declared_word_shapes.py::test_declared_word_shape_gets_the_verdict_it_earns[payload_in_a_separate_buffer]
///
/// A relaxed publication gives the wait nothing to acquire: the separate
/// payload is a `write_read` race.
#[test]
fn g2_shape_payload_in_a_separate_buffer() {
    let mut k = g2_shape_prologue(true);
    k.st(0, 0, GMEM, 0..8).a(0, 0, G2_RLX_CL_ST, GMEM2, G2_SLOT);
    k.a(1, 0, G2_RLX_CL_LD, GMEM2, G2_SLOT);
    k.wait_until(1, 0, GMEM2, G2_SLOT, Scope::Cluster, 0b100, 2);
    k.ld(1, 0, GMEM, 0..8);
    let r = k.run();
    assert!(has_race(&r));
    assert!(races(&r).iter().all(|f| f.alloc == GMEM && matches!(f.kind, FindingKind::DataRace { class: RaceClass::WriteRead, .. })), "{r:?}");
}

/// test_declared_word_shapes.py::test_declared_word_shape_gets_the_verdict_it_earns[read_once_no_retry]
///
/// delta R3 (R2): legacy G reported the strong `ld.relaxed.cluster` vs
/// `st.relaxed.cluster` as `write_read`; new: the pair is morally strong
/// (exempt) and the undeclared polled word is a review
/// `UndeclaredProtocolWord` advisory instead of an error.
#[test]
fn g2_shape_read_once_no_retry() {
    let mut k = g2_shape_prologue(false);
    k.a(0, 0, G2_RLX_CL_ST, GMEM2, G2_SLOT);
    k.a(1, 0, G2_RLX_CL_LD, GMEM2, G2_SLOT);
    k.st(1, 0, GMEM, 0..8);
    let r = k.run();
    assert!(races(&r).is_empty(), "{r:?}");
    assert!(has_advisory(&r, AdvisoryKind::UndeclaredProtocolWord), "{r:?}");
    assert!(review_only(&r));
}

/// test_declared_word_shapes.py::test_declared_word_shape_gets_the_verdict_it_earns[non_unique_exit_value]
#[test]
fn g2_shape_non_unique_exit_value() {
    let mut k = g2_shape_prologue(true);
    k.a(0, 0, G2_RLX_CL_ST, GMEM2, G2_SLOT).a(0, 0, G2_RLX_CL_ST, GMEM2, G2_SLOT); // 2, 3
    k.a(1, 0, G2_RLX_CL_LD, GMEM2, G2_SLOT);
    k.wait_until(1, 0, GMEM2, G2_SLOT, Scope::Cluster, 0b1100, 3);
    k.st(1, 0, GMEM, 0..8);
    let r = k.run();
    assert!(r.is_clean(), "{r:?}");
}

/// test_declared_word_shapes.py::test_declared_word_shape_gets_the_verdict_it_earns[concurrent_writes_then_read]
#[test]
fn g2_shape_concurrent_writes_then_read() {
    let mut k = K::new(1, 2, 2);
    k.a(0, 0, G2_RLX_CL_ST, GMEM2, G2_SLOT).a(1, 0, G2_RLX_CL_ST, GMEM2, G2_SLOT);
    k.cluster_bar(&[0, 1]);
    k.a(0, 0, G2_RLX_CL_LD, GMEM2, G2_SLOT).st(0, 0, GMEM, 0..8);
    let r = k.run();
    assert!(r.is_clean(), "{r:?}");
}

/// test_declared_word_shapes.py::test_declared_word_shape_gets_the_verdict_it_earns[atomic_add_modification_order]
/// and [raw_float_accumulator] (identical at the contract level: two
/// `atom.relaxed.cluster.add` then cluster barrier then a plain read).
#[test]
fn g2_shape_atomic_add_then_barrier_read() {
    let mut k = K::new(1, 2, 2);
    k.a(0, 0, atom(MemOrder::Relaxed, Scope::Cluster), GMEM2, 0..4);
    k.a(1, 0, atom(MemOrder::Relaxed, Scope::Cluster), GMEM2, 0..4);
    k.cluster_bar(&[0, 1]);
    k.ld(0, 0, GMEM2, 0..4).st(0, 0, GMEM, 0..4);
    let r = k.run();
    assert!(r.is_clean(), "{r:?}");
}

/// test_declared_word_shapes.py::test_declared_word_shape_gets_the_verdict_it_earns[atomic_ticket_order]
#[test]
fn g2_shape_atomic_ticket_order() {
    let mut k = K::new(1, 2, 2);
    k.a(0, 0, atom(MemOrder::Relaxed, Scope::Cluster), GMEM2, 0..4).st(0, 0, GMEM, 0..4);
    k.a(1, 0, atom(MemOrder::Relaxed, Scope::Cluster), GMEM2, 0..4).st(1, 0, GMEM, 4..8);
    let r = k.run();
    assert!(r.is_clean(), "{r:?}");
}

/// test_declared_word_shapes.py::test_declared_word_shape_gets_the_verdict_it_earns[raw_shared_cluster_poll]
///
/// C0 writes its own smem slot through a `mapa` shared::cluster address
/// (relaxed.cluster), C1 polls it with `ld.relaxed.cluster.shared::cluster`.
/// delta R3: legacy verdict `clean`; new: the morally strong pair is exempt
/// (no error) but the undeclared polled word is a review advisory
/// (`UndeclaredProtocolWord`); a shared word can never be declared.
#[test]
fn g2_shape_raw_shared_cluster_poll() {
    let mut k = K::new(1, 2, 2);
    k.inst_in(0, &[0], G2_RLX_CL_ST, Some(Domain::SharedCluster), |_| (SMEM, 0..8));
    k.cluster_bar(&[0, 1]);
    k.inst_in(0, &[0], G2_RLX_CL_ST, Some(Domain::SharedCluster), |_| (SMEM, 0..8));
    for _ in 0..2 {
        k.inst_in(1, &[0], G2_RLX_CL_LD, Some(Domain::SharedCluster), |_| (SMEM, 0..8));
    }
    k.st(1, 0, GMEM, 0..8);
    let r = k.run();
    assert!(races(&r).is_empty(), "{r:?}");
    assert!(r.incomplete.is_empty());
    assert!(has_advisory(&r, AdvisoryKind::UndeclaredProtocolWord), "{r:?}");
    assert!(review_only(&r), "{r:?}");
}

// =============================================================== G2: tcgen ==

fn g2_tc(k: &mut K, w: WarpId, kind: AsyncKind, write: Option<std::ops::Range<u64>>, read: Option<std::ops::Range<u64>>) -> AsyncId {
    let op = k.issue(w, 0, kind, Proxy::Tcgen, &[], &[(TMEM, 0..4096)]);
    if let Some(r) = read {
        k.aacc(op, Milestone::Read, AccessKind::Read, Proxy::Tcgen, TMEM, r);
    }
    if let Some(r) = write {
        k.aacc(op, Milestone::Write, AccessKind::Write, Proxy::Tcgen, TMEM, r);
    }
    op
}

/// cp → mma handoff through an mbarrier with exact qualifiers:
/// `relaxed_wait` = default (release.cta) arrive + `test_wait.relaxed.cta`;
/// `relaxed_arrive_wait` = `arrive.relaxed.cluster` + `test_wait.relaxed.cta`.
fn g2_cp_mma_mbar(relaxed_arrive_cluster: bool, before: bool, after: bool) -> Report {
    let mut k = K::new(2, 1, 1);
    let cp = g2_tc(&mut k, 0, AsyncKind::TcgenPipelined, Some(0..16), None);
    if before {
        k.fence(0, 1, FenceKind::TcgenBefore);
    }
    if relaxed_arrive_cluster {
        k.arrive_q(0, 1, mbar(9), 0, Some(false), Some(Scope::Cluster));
    } else {
        k.arrive_q(0, 1, mbar(9), 0, Some(true), Some(Scope::Cta));
    }
    k.wait_q(1, 1, mbar(9), 0, Some(false), Some(Scope::Cta));
    if after {
        k.fence(1, 1, FenceKind::TcgenAfter);
    }
    let mma = g2_tc(&mut k, 1, AsyncKind::TcgenPipelined, Some(0..32), None);
    let c = k.issue(1, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &[mma], &[]);
    k.done_phase(c, Milestone::Write, 10, 0).wait(1, 1, 10, 0, true);
    let c0 = k.issue(0, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &[cp], &[]);
    k.done_phase(c0, Milestone::Write, 11, 0);
    k.run()
}

/// test_native_tcgen_thread_fence.py::test_cross_thread_cp_to_mma_requires_both_thread_fences[relaxed_wait]
/// and [relaxed_arrive_wait] with their exact arrive/wait qualifiers (the
/// existing `cp_to_mma_handoff_needs_both_fences` models both as a relaxed
/// `.cta` arrive + relaxed wait).
#[test]
fn g2_cp_to_mma_mbarrier_relaxed_variants() {
    for cluster in [false, true] {
        assert!(clean(&g2_cp_mma_mbar(cluster, true, true)), "cluster={cluster}");
        assert!(!g2_cp_mma_mbar(cluster, false, true).is_clean(), "cluster={cluster} missing before");
        assert!(!g2_cp_mma_mbar(cluster, true, false).is_clean(), "cluster={cluster} missing after");
    }
}

/// test_native_same_warp_tmem_review.py::test_unwaited_tmem_load_conflicts_remain_distinct_reviews
///
/// WG0: ld A, ld B, st A (no wait::ld), wait::st, before_thread_sync;
/// bar; WG1: after_thread_sync, st B, wait::st; WG0 wait::ld late. Both
/// the same-warp (ld A / st A) and the cross-warp (ld B / st B) conflict
/// are reported, each as a review, and nothing else.
#[test]
fn g2_unwaited_tmem_load_conflicts_remain_distinct_reviews() {
    let mut k = K::new(2, 1, 1);
    let la = g2_tc(&mut k, 0, AsyncKind::TcgenLd, None, Some(0..16));
    let lb = g2_tc(&mut k, 0, AsyncKind::TcgenLd, None, Some(16..32));
    let sa = g2_tc(&mut k, 0, AsyncKind::TcgenSt, Some(0..16), None);
    k.done_warp(sa, Milestone::Write, 0, u32::MAX);
    k.fence(0, u32::MAX, FenceKind::TcgenBefore).bar(0, &[0, 1]).fence(1, u32::MAX, FenceKind::TcgenAfter);
    let sb = g2_tc(&mut k, 1, AsyncKind::TcgenSt, Some(16..32), None);
    k.done_warp(sb, Milestone::Write, 1, u32::MAX);
    k.done_warp(la, Milestone::Write, 0, u32::MAX).done_warp(lb, Milestone::Write, 0, u32::MAX);
    let r = k.run();
    assert!(review_only(&r), "{r:?}");
    assert!(r.incomplete.is_empty());
    let pairs: Vec<(u32, u32)> = r
        .findings
        .iter()
        .filter_map(|f| Some((f.prior.as_ref()?.warp, f.current.as_ref()?.warp)))
        .collect();
    assert!(pairs.iter().any(|(a, b)| a == b), "{r:?}");
    assert!(pairs.iter().any(|(a, b)| a != b), "{r:?}");
}

// ======================================================= G2: TMEM arrive ==

/// Producer warp `p` writes TMEM (all lanes), optionally reads it before the
/// arrive, `bar.warp.sync`, lane 0 arrives (release.cta), optionally reads
/// after the arrive; consumer lane 0 waits, `bar.warp.sync`, all lanes write.
fn g2_tmem_arrive_snapshot(p: WarpId, load_after_arrive: bool) -> Report {
    let c = 1 - p;
    let all: Vec<u8> = (0..32).collect();
    let mut k = K::new(2, 1, 1);
    let row = |l: u8| (TMEM, l as u64 * 256..l as u64 * 256 + 4);
    k.inst(p, &all, PLAIN_ST, row);
    if !load_after_arrive {
        k.inst(p, &all, PLAIN_LD, row);
    }
    k.syncwarp(p, u32::MAX).arrive(p, 1, 0, 0, true);
    if load_after_arrive {
        k.inst(p, &all, PLAIN_LD, row);
    }
    k.wait(c, 1, 0, 0, true).syncwarp(c, u32::MAX);
    k.inst(c, &all, PLAIN_ST, row);
    k.run()
}

/// test_native_racecheck_tmem_arrive_snapshot.py::test_public_native_tmem_post_arrive_load_races_with_late_consumer_store
/// (producer warp 1) and ::test_public_native_tmem_post_arrive_load_races_with_preblocked_consumer_store
/// (producer warp 0): the late/pre-blocked wait acquires only the arrive
/// snapshot, so the post-arrive TMEM load races the consumer's store; one
/// deduplicated finding, 4 bytes, lane 0 evidence, read by the producer.
#[test]
fn g2_tmem_post_arrive_load_races_with_consumer_store() {
    for p in [1, 0] {
        assert!(g2_tmem_arrive_snapshot(p, false).is_clean(), "p={p}");
        let r = g2_tmem_arrive_snapshot(p, true);
        let rs = races(&r);
        assert_eq!(rs.len(), 1, "p={p} {r:?}");
        assert!(r.incomplete.is_empty());
        let f = rs[0];
        assert_eq!(f.alloc, TMEM);
        assert!(matches!(f.kind, FindingKind::DataRace { class: RaceClass::ReadWrite, .. }), "{f:?}");
        let (prior, cur) = (f.prior.as_ref().unwrap(), f.current.as_ref().unwrap());
        let pw: u32 = prior.warp;
        let cw: u32 = cur.warp;
        assert_eq!((pw, cw), (p, g2_c_of(p)));
        assert_eq!(prior.lane, 0);
        assert_eq!(cur.lane, 0);
        assert_eq!(prior.span.end - prior.span.start, 4);
    }
}

fn g2_c_of(p: WarpId) -> WarpId {
    1 - p
}

// ======================================================================= g3

// ===================================== test_native_global_scoped_hb_matrix.py

const G3_FLAG: std::ops::Range<u64> = 0..4;
const G3_TURN: std::ops::Range<u64> = 4..8;
const G3_READY: std::ops::Range<u64> = 8..12;

/// test_native_global_scoped_hb_matrix.py::test_same_cluster_release_acquire_is_clean:
/// two CTAs of one cluster; C0 writes data, `st.release.cluster` flag; C1
/// `wait_until(.cluster)` on the flag then reads data. Clean.
#[test]
fn g3_same_cluster_release_acquire_is_clean() {
    let mut k = K::new(1, 2, 2);
    k.declare(GMEM2, G3_FLAG);
    k.st(0, 0, GMEM, 0..4).a(0, 0, st(MemOrder::Release, Scope::Cluster), GMEM2, G3_FLAG);
    k.wait_until(1, 0, GMEM2, G3_FLAG, Scope::Cluster, 0b10, 1).ld(1, 0, GMEM, 0..4);
    let r = k.run();
    assert!(clean(&r) && r.findings.is_empty(), "{r:?}");
}

/// test_native_global_scoped_hb_matrix.py::test_host_initialized_version_is_before_all_kernel_reads:
/// two CTAs only `ld.acquire.gpu` a host-initialised word; no kernel write,
/// so nothing conflicts and nothing is incomplete.
#[test]
fn g3_host_initialized_reads_are_clean() {
    let mut k = K::new(1, 1, 2);
    for c in 0..2 {
        k.a(c, 0, ld(MemOrder::Acquire, Scope::Gpu), GMEM, 0..4);
        k.st(c, 0, GMEM2, c as u64 * 4..c as u64 * 4 + 4);
    }
    let r = k.run();
    assert!(clean(&r) && r.findings.is_empty(), "{r:?}");
}

/// test_native_global_scoped_hb_matrix.py::test_same_value_and_aba_observations_use_only_the_exact_writer[aba]:
/// C0 publishes data[0] with `st.release` flag and bumps `turn` relaxed; C1
/// relaxed-polls turn (no edge), writes data[1], optionally (ABA) stores
/// flag=0 relaxed, then `st.release` flag=1 and bumps `ready`; C2 relaxed-
/// polls ready and `ld.acquire`s flag, reading C1's release. Only C1's
/// write is published: data[0] races, data[1] does not.
fn g3_aba(aba: bool) -> Report {
    let mut k = K::new(1, 1, 3);
    k.st(0, 0, GMEM, 0..4).a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, G3_FLAG);
    k.a(0, 0, st(MemOrder::Relaxed, Scope::Gpu), GMEM2, G3_TURN);
    k.a(1, 0, ld(MemOrder::Relaxed, Scope::Gpu), GMEM2, G3_TURN);
    k.st(1, 0, GMEM, 4..8);
    if aba {
        k.a(1, 0, st(MemOrder::Relaxed, Scope::Gpu), GMEM2, G3_FLAG);
    }
    k.a(1, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, G3_FLAG);
    k.a(1, 0, st(MemOrder::Relaxed, Scope::Gpu), GMEM2, G3_READY);
    k.a(2, 0, ld(MemOrder::Relaxed, Scope::Gpu), GMEM2, G3_READY);
    k.a(2, 0, ld(MemOrder::Acquire, Scope::Gpu), GMEM2, G3_FLAG);
    k.ld(2, 0, GMEM, 0..4).ld(2, 0, GMEM, 4..8);
    k.run()
}

#[test]
fn g3_same_value_and_aba_use_only_exact_writer() {
    for aba in [false, true] {
        let r = g3_aba(aba);
        assert!(has_race(&r), "aba={aba}: {r:?}");
        assert!(races(&r).iter().any(|f| f.alloc == GMEM && f.bytes.start == 0), "aba={aba}: {r:?}");
        assert!(!races(&r).iter().any(|f| f.alloc == GMEM && f.bytes.start == 4), "aba={aba}: {r:?}");
        assert!(r.incomplete.is_empty(), "aba={aba}: {r:?}");
    }
}

/// test_native_global_scoped_hb_matrix.py::test_rmw_release_sequence_and_morally_strong_rules[mode]
/// (three single-CTA clusters; flag/turn/ready are declared words).
/// 0: C0 `st.release` flag, C1 continues with `atom.relaxed`, C2 waits
///    flag>=1 (accepts both writes; earliest is the release) → clean.
/// 1: C0's head is `atom.relaxed` (no release), C2 reads flag relaxed → race.
/// 2: C0 and C1 `atom.relaxed.cluster` on one flag from different clusters.
/// 3: C1 `red.relaxed` flag then reads data; it only waited on a relaxed
///    `turn` → race.
fn g3_rmw_matrix(mode: u8) -> Report {
    let mut k = K::new(1, 1, 3);
    for w in [G3_FLAG, G3_TURN, G3_READY] {
        k.declare(GMEM2, w);
    }
    if mode == 2 {
        k.a(0, 0, atom(MemOrder::Relaxed, Scope::Cluster), GMEM2, G3_FLAG);
        k.a(1, 0, atom(MemOrder::Relaxed, Scope::Cluster), GMEM2, G3_FLAG);
        return k.run();
    }
    k.st(0, 0, GMEM, 0..4);
    if mode == 1 {
        k.a(0, 0, atom(MemOrder::Relaxed, Scope::Gpu), GMEM2, G3_FLAG);
    } else {
        k.a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, G3_FLAG);
    }
    k.a(0, 0, st(MemOrder::Relaxed, Scope::Gpu), GMEM2, G3_TURN);
    k.wait_until(1, 0, GMEM2, G3_TURN, Scope::Gpu, 0b10, 1);
    if mode == 3 {
        k.red(1, 0, MemOrder::Relaxed, Scope::Gpu, GMEM2, G3_FLAG);
        k.ld(1, 0, GMEM, 0..4);
        return k.run();
    }
    k.a(1, 0, atom(MemOrder::Relaxed, Scope::Gpu), GMEM2, G3_FLAG);
    k.a(1, 0, st(MemOrder::Relaxed, Scope::Gpu), GMEM2, G3_READY);
    k.wait_until(2, 0, GMEM2, G3_READY, Scope::Gpu, 0b10, 1);
    if mode == 0 {
        k.wait_until(2, 0, GMEM2, G3_FLAG, Scope::Gpu, 0b110, 2);
    } else {
        k.a(2, 0, ld(MemOrder::Relaxed, Scope::Gpu), GMEM2, G3_FLAG);
    }
    k.ld(2, 0, GMEM, 0..4);
    k.run()
}

#[test]
fn g3_rmw_release_sequence_and_morally_strong_rules() {
    let r0 = g3_rmw_matrix(0);
    assert!(clean(&r0) && r0.findings.is_empty(), "{r0:?}");
    for m in [1, 3] {
        let r = g3_rmw_matrix(m);
        assert!(r.incomplete.is_empty(), "mode {m}: {r:?}");
        assert!(races(&r).iter().any(|f| f.alloc == GMEM && f.bytes.start == 0), "mode {m}: {r:?}");
    }
    // delta R4: legacy expected only a `scope_mismatch` finding and no data
    // race; new: an RMW pair that fails mutual scope inclusion is not
    // morally strong, so it is a data race with `missing_release_acquire`.
    let r2 = g3_rmw_matrix(2);
    assert!(has_race(&r2), "{r2:?}");
    assert!(has_failure(&r2, |f| f == OrderingFailure::MissingReleaseAcquire));
}

/// test_native_global_scoped_hb_matrix.py::test_async_store_requires_completion_handoff_before_publication[wait]:
/// C0 fills smem, `fence.proxy.async.shared::cta`, bulk S2G, optionally
/// `wait_group 0`, then `st.release.gpu` flag; C1 waits on the flag and
/// reads the destination. With the full wait it is clean with NO extra
/// `fence.proxy.async.global` (delta X3: completion bridges the op's own
/// results to the generic proxy). Without any wait: a finding or incomplete.
fn g3_async_store_pub(wait: bool) -> Report {
    let mut k = K::new(1, 1, 2);
    k.declare(GMEM2, G3_FLAG);
    k.st(0, 0, SMEM, 0..16).fence(0, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, SMEM, 0..16).aw(op, Proxy::Async, GMEM, 0..16);
    if wait {
        k.done_warp(op, Milestone::Write, 0, 1);
    }
    k.a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, G3_FLAG);
    k.wait_until(1, 0, GMEM2, G3_FLAG, Scope::Gpu, 0b10, 1).ld(1, 0, GMEM, 0..1);
    k.run()
}

#[test]
fn g3_async_store_requires_completion_before_publication() {
    let r = g3_async_store_pub(true);
    assert!(clean(&r) && r.findings.is_empty(), "{r:?}");
    let r = g3_async_store_pub(false);
    assert!(!r.findings.is_empty() || !r.incomplete.is_empty());
    assert!(has_race(&r), "{r:?}");
}

// ===================================== test_native_async_lifetime_contracts.py

/// TMA store FIFO (legacy `native_tma_store_fifo_lifetime`): W0 lanes 0..3
/// fill three 128-byte-aligned sources, CTA barrier, proxy fence; W0L0
/// issues `n_ops` bulk S2G copies; `wait_group.read N` completes the read
/// side of the `drained` oldest ones (FIFO group counting, including empty
/// and predicated-off commits, is decided by the sync engine upstream of
/// these contract events); W0L0 arrives, W1L0 waits, warp sync, W1 lanes
/// 0..3 overwrite source `overwrite`. The write milestones complete at the
/// end (the kernel never full-waits; delta P6 makes that an incomplete).
fn g3_fifo(n_ops: usize, drained: usize, overwrite: usize) -> Report {
    let mut k = K::new(2, 1, 1);
    let src = |i: usize| (i as u64 * 128)..(i as u64 * 128 + 16);
    for i in 0..3 {
        k.inst(0, &[0, 1, 2, 3], PLAIN_ST, |l| (SMEM, src(i).start + l as u64 * 4..src(i).start + l as u64 * 4 + 4));
    }
    k.bar(0, &[0, 1]);
    k.fence(0, u32::MAX, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    k.fence(1, u32::MAX, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    let mut ops = vec![];
    for i in 0..n_ops {
        let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, src(i))]);
        k.ar(op, Proxy::Async, SMEM, src(i)).aw(op, Proxy::Async, GMEM, (i as u64 * 16)..(i as u64 * 16 + 16));
        ops.push(op);
    }
    for &op in ops.iter().take(drained) {
        k.done_warp(op, Milestone::Read, 0, 1);
    }
    k.arrive(0, 1, 3, 0, true).wait(1, 1, 3, 0, true).syncwarp(1, u32::MAX);
    let o = src(overwrite).start;
    k.inst(1, &[0, 1, 2, 3], PLAIN_ST, |l| (SMEM, o + l as u64 * 4..o + l as u64 * 4 + 4));
    for &op in &ops {
        k.done_warp(op, Milestone::Write, 0, 1);
    }
    k.run()
}

/// test_native_async_lifetime_contracts.py::test_tma_store_source_lifetime_uses_fifo_groups
/// [shape,wait_n,overwrite]: (1,1,0) clean, (1,2,0) race (empty group
/// counts), (2,2,0) clean, (2,2,1) race, (2,2,2) race, (3,0,0) race
/// (predicated-off commit forms no group). The 2-group rows are in
/// racecheck_async_copy.rs::tma_store_fifo_groups.
#[test]
fn g3_tma_store_fifo_shapes() {
    let check = |r: Report, race: bool, what: &str| {
        if race {
            assert!(has_class(&r, RaceClass::ReadWrite) || has_class(&r, RaceClass::WriteRead), "{what}: {r:?}");
            assert!(races(&r).iter().all(|f| f.alloc == SMEM), "{what}: {r:?}");
        } else {
            assert!(clean(&r) && r.findings.is_empty(), "{what}: {r:?}");
        }
    };
    // shape 1: [op0], [] ; wait.read 1 drains op0, wait.read 2 drains nothing.
    check(g3_fifo(1, 1, 0), false, "(1,1,0)");
    check(g3_fifo(1, 0, 0), true, "(1,2,0)");
    // shape 2: three groups, wait.read 2 drains op0 only.
    check(g3_fifo(3, 1, 0), false, "(2,2,0)");
    check(g3_fifo(3, 1, 1), true, "(2,2,1)");
    check(g3_fifo(3, 1, 2), true, "(2,2,2)");
    // shape 3: op0 never committed, wait 0 drains nothing.
    check(g3_fifo(1, 0, 0), true, "(3,0,0)");
}

/// test_native_async_lifetime_contracts.py::test_classic_cp_async_wait_group_completes_source_and_destination_accesses[1]:
/// every lane issues `cp.async` (generic proxy), `wait_group 0`, then
/// overwrites its global source. Clean. (mode 3 is
/// racecheck_async_copy.rs::classic_cp_async_lifetime cp_async(1).)
#[test]
fn g3_classic_cp_async_wait_then_source_write_is_clean() {
    let mut k = K::one_warp();
    let lanes: Vec<u8> = (0..32).collect();
    let mut ops = vec![];
    for &l in &lanes {
        let r = (l as u64 * 16)..(l as u64 * 16 + 16);
        let op = k.issue(0, l, AsyncKind::Copy, Proxy::Generic, &[], &[(SMEM, r.clone())]);
        k.ar(op, Proxy::Generic, GMEM, r.clone()).aw(op, Proxy::Generic, SMEM, r);
        ops.push(op);
    }
    for (l, &op) in ops.iter().enumerate() {
        k.done_warp(op, Milestone::Write, 0, 1 << l);
    }
    k.inst(0, &lanes, PLAIN_ST, |l| (GMEM, l as u64 * 16..l as u64 * 16 + 4));
    let r = k.run();
    assert!(clean(&r) && r.findings.is_empty(), "{r:?}");
}

// ======================================== test_native_shared_publication.py

/// test_native_shared_publication.py::test_sc_causality_and_lane_controls[space] order="cuda":
/// `T.cuda.thread_fence()` (`__threadfence` = `membar.gl` = `fence.sc.gpu`)
/// by the publisher and consumer lanes orders lane 0's store before lane 1's
/// load; the other orders are in
/// racecheck_scoped_and_lanes.rs::sc_causality_and_lane_controls.
#[test]
fn g3_cuda_thread_fence_sc_causality() {
    for alloc in [SMEM, GMEM] {
        let mut k = K::one_warp();
        k.st(0, 0, alloc, 0..4).syncwarp(0, u32::MAX).st(0, 0, alloc, 0..4);
        k.fence(0, 1 << 0, FenceKind::Sc(Scope::Gpu)).fence(0, 1 << 1, FenceKind::Sc(Scope::Gpu));
        k.ld(0, 1, alloc, 0..4);
        let r = k.run();
        assert!(clean(&r) && r.findings.is_empty(), "{r:?}");
    }
}

// ======================================================================= g4

// ------------------------------------------- G4: scoped global / lanes --

/// test_native_global_scoped_hb.py::test_scoped_global_missing_edge_or_misplaced_fence_reports_data_race[8]:
/// C0 writes data[0], `fence.acq_rel.gpu`, then data[1] AFTER the fence, then
/// `st.relaxed.sys` flag; C1 waits (sys) + `fence.acq_rel.gpu` and reads
/// data[1]. The fence does not cover the later write: race at byte 4 only.
#[test]
fn g4_scoped_global_write_after_release_fence_races() {
    let mut k = K::new(1, 1, 2);
    k.declare(GMEM2, 0..4);
    k.st(0, 0, GMEM, 0..4).fence(0, 1, FenceKind::AcqRel(Scope::Gpu)).st(0, 0, GMEM, 4..8);
    k.a(0, 0, st(MemOrder::Relaxed, Scope::Sys), GMEM2, 0..4);
    k.wait_until(1, 0, GMEM2, 0..4, Scope::Sys, 0b10, 1).fence(1, 1, FenceKind::AcqRel(Scope::Gpu));
    k.ld(1, 0, GMEM, 4..8);
    let r = k.run();
    let f = races(&r);
    assert_eq!(f.len(), 1, "{r:?}");
    assert_eq!(f[0].alloc, GMEM);
    assert_eq!(f[0].bytes, 4..8);
    assert!(has_class(&r, RaceClass::WriteRead));
    // Control: reading data[0] (covered by the fence) is clean.
    let mut k = K::new(1, 1, 2);
    k.declare(GMEM2, 0..4);
    k.st(0, 0, GMEM, 0..4).fence(0, 1, FenceKind::AcqRel(Scope::Gpu)).st(0, 0, GMEM, 4..8);
    k.a(0, 0, st(MemOrder::Relaxed, Scope::Sys), GMEM2, 0..4);
    k.wait_until(1, 0, GMEM2, 0..4, Scope::Sys, 0b10, 1).fence(1, 1, FenceKind::AcqRel(Scope::Gpu));
    k.ld(1, 0, GMEM, 0..4);
    assert!(clean(&k.run()));
}

/// test_native_racecheck_exact_control.py::test_public_native_racecheck_resolves_data_guarded_cross_lane_hazard_exactly:
/// W0 inits shared[lane], `bar.sync 6, 64`, then (if enabled) W0 rewrites
/// shared[lane] while W1 reads shared[(lane+1)%32]. enabled=0 clean,
/// enabled=1 one shared race with a 4-byte overlap.
fn g4_data_guarded_cross_lane(enabled: bool) -> Report {
    let mut k = K::new(2, 1, 1);
    let all: Vec<u8> = (0..32).collect();
    k.inst(0, &all, PLAIN_ST, |l| (SMEM, l as u64 * 4..l as u64 * 4 + 4));
    k.bar(6, &[0, 1]);
    if enabled {
        k.inst(0, &all, PLAIN_ST, |l| (SMEM, l as u64 * 4..l as u64 * 4 + 4));
    }
    k.inst(1, &all, PLAIN_LD, |l| {
        let o = ((l as u64 + 1) % 32) * 4;
        (SMEM, o..o + 4)
    });
    k.run()
}

#[test]
fn g4_data_guarded_cross_lane_hazard() {
    assert!(clean(&g4_data_guarded_cross_lane(false)));
    let r = g4_data_guarded_cross_lane(true);
    assert!(has_race(&r));
    assert!(races(&r).iter().all(|f| f.alloc == SMEM));
    assert!(has_class(&r, RaceClass::WriteRead) || has_class(&r, RaceClass::ReadWrite));
    let f = races(&r)[0];
    let p = f.prior.as_ref().unwrap();
    assert_eq!(p.span.end - p.span.start, 4);
}

/// test_native_racecheck_exact_control.py::test_public_native_racecheck_preserves_tile_address_local_scalars:
/// two clusters (one CTA each); lanes 0..3 write smem, `fence.proxy.async.shared::cta`,
/// `cta_sync`, lane 0 TMA-stores the 16 bytes to output[cluster*4..] and
/// waits the bulk group. Disjoint destinations: clean.
#[test]
fn g4_tile_store_per_cluster_disjoint_is_clean() {
    let mut k = K::new(1, 1, 2);
    k.alloc_cta(SMEM1, 1);
    for (w, smem) in [(0u32, SMEM), (1u32, SMEM1)] {
        k.inst(w, &[0, 1, 2, 3], PLAIN_ST, |l| (smem, l as u64 * 4..l as u64 * 4 + 4));
        k.fence(w, u32::MAX, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
        k.bar(0, &[w]);
        let op = k.issue(w, 0, AsyncKind::Copy, Proxy::Async, &[], &[(smem, 0..16)]);
        let dst = w as u64 * 16;
        k.ar(op, Proxy::Async, smem, 0..16).aw(op, Proxy::Async, GMEM, dst..dst + 16);
        k.done_warp(op, Milestone::Write, w, 1);
    }
    let r = k.run();
    assert!(clean(&r), "{r:?}");
}

/// test_native_racecheck_lane_order.py::test_synccheck_uses_one_concrete_same_instruction_atomic_lane_order
/// and ::test_address_only_same_instruction_atomic_return_is_clean: a 32-lane
/// same-address `atomicAdd`, optionally `__syncwarp`, then each lane stores
/// to a distinct output word chosen by its ticket. Racecheck: clean.
#[test]
fn g4_same_address_atomic_then_ticket_outputs() {
    let all: Vec<u8> = (0..32).collect();
    for sync in [true, false] {
        for permuted in [false, true] {
            let mut k = K::one_warp();
            k.inst(0, &all, atom(MemOrder::Relaxed, Scope::Gpu), |_| (GMEM, 0..4));
            if sync {
                k.syncwarp(0, u32::MAX);
            }
            k.inst(0, &all, PLAIN_ST, |l| {
                let t = if permuted { (31 - l) as u64 } else { l as u64 };
                (GMEM2, t * 4..t * 4 + 4)
            });
            let r = k.run();
            assert!(clean(&r), "sync={sync} permuted={permuted}: {r:?}");
        }
    }
}

// ------------------------------------------------- G4: alias controls --

/// test_native_alias_advisory.py::test_public_native_non_stale_pool_alias_controls_are_clean[read-writer-name]
/// and [disjoint-lifetimes]: two names over one pooled shared word, all
/// accesses by lane 0 in program order. Clean (and no advisory).
/// (The stale-name case, mode 0, is `alias_stale_read` -- not portable.)
#[test]
fn g4_pool_alias_same_lane_controls_clean() {
    // mode 1: A=1; B=3; read B.
    let mut k = K::one_warp();
    k.st(0, 0, SMEM, 0..4).st(0, 0, SMEM, 0..4).ld(0, 0, SMEM, 0..4).st(0, 0, GMEM, 0..4);
    let r = k.run();
    assert!(clean(&r) && r.findings.is_empty(), "{r:?}");
    // mode 2: A=1; read A; B=4; read B.
    let mut k = K::one_warp();
    k.st(0, 0, SMEM, 0..4).ld(0, 0, SMEM, 0..4).st(0, 0, GMEM, 0..4);
    k.st(0, 0, SMEM, 0..4).ld(0, 0, SMEM, 0..4).st(0, 0, GMEM, 0..4);
    let r = k.run();
    assert!(clean(&r) && r.findings.is_empty(), "{r:?}");
}

/// test_native_alias_advisory.py::test_public_native_tcgen_alloc_result_is_visible_to_every_lane:
/// `tcgen05.alloc.sync.aligned` writes the TMEM address to smem (one store by
/// the first active lane), then every lane reads it. The alloc is a
/// `WarpSync{mask}` (racecheck-semantics.md §3 table, row 3), so: clean.
/// Without the warp sync the same events race (lane order).
#[test]
fn g4_tcgen_alloc_result_visible_to_every_lane() {
    let all: Vec<u8> = (0..32).collect();
    for synced in [true, false] {
        let mut k = K::one_warp();
        k.syncwarp(0, u32::MAX);
        k.st(0, 0, SMEM, 0..4);
        if synced {
            k.syncwarp(0, u32::MAX);
        }
        k.inst(0, &all, PLAIN_LD, |_| (SMEM, 0..4));
        k.inst(0, &all, PLAIN_ST, |l| (GMEM, l as u64 * 4..l as u64 * 4 + 4));
        let r = k.run();
        if synced {
            assert!(clean(&r), "{r:?}");
        } else {
            assert!(has_failure(&r, |f| f == OrderingFailure::MissingSameWarpLaneOrder), "{r:?}");
        }
    }
}

// ======================================================================= g5

const G5_ALL: [u8; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];

/// test_native_atomic_semantics.py::test_public_native_racecheck_accepts_ptx_atom_cas:
/// lane 0 initialises the slot, `cta_sync`, then the same lane issues
/// `atom.relaxed.cta.shared.cas`. Clean.
#[test]
fn g5_atom_cas_after_barrier_is_clean() {
    let mut k = K::one_warp();
    k.st(0, 0, SMEM, 0..4).bar(0, &[0]);
    k.a(0, 0, atom(MemOrder::Relaxed, Scope::Cta), SMEM, 0..4);
    k.st(0, 0, GMEM, 0..4);
    assert!(clean(&k.run()));
}

/// test_native_racecheck_arrive_snapshot.py::test_public_native_racecheck_blocked_wait_acquires_only_arrive_snapshot:
/// the waiter (warp 1) is blocked before the producer (warp 0) arrives; the
/// producer overwrites the data after the wait resumed. The wait acquires
/// only the arrive snapshot, so the overwrite races the read; without the
/// overwrite it is clean.
#[test]
fn g5_blocked_wait_acquires_only_arrive_snapshot() {
    for overwrite in [false, true] {
        let mut k = K::new(2, 1, 1);
        k.st(0, 0, SMEM, 0..4).arrive(0, 1, 3, 0, true);
        k.wait(1, 1, 3, 0, true);
        if overwrite {
            k.st(0, 0, SMEM, 0..4);
        }
        k.ld(1, 0, SMEM, 0..4);
        let r = k.run();
        if overwrite {
            let f = races(&r);
            assert_eq!(f.len(), 1);
            assert_eq!(f[0].bytes, 0..4);
            assert_eq!(f[0].prior.as_ref().unwrap().warp, 0);
            assert_eq!(f[0].current.as_ref().unwrap().warp, 1);
            assert!(r.incomplete.is_empty());
        } else {
            assert!(clean(&r) && r.findings.is_empty());
        }
    }
}

/// Warp 0 writes `shared[2l], shared[2l+1]` (two element stores); warp 1
/// reads them with one `ld.shared.v2` (8 bytes per lane).
fn g5_vector_load(barrier: bool, load: Op) -> Report {
    let mut k = K::new(2, 1, 1);
    for i in 0..2u64 {
        k.inst(0, &G5_ALL, PLAIN_ST, |l| (SMEM, (l as u64 * 2 + i) * 4..(l as u64 * 2 + i) * 4 + 4));
    }
    if barrier {
        k.bar(0, &[0, 1]);
    }
    k.inst(1, &G5_ALL, load, |l| (SMEM, l as u64 * 8..l as u64 * 8 + 8));
    k.run()
}

/// test_native_vector_destination_loads.py::test_racecheck_traces_the_element_footprint_of_a_vector_destination_load
#[test]
fn g5_vector_destination_load_barrier_ordered_is_clean() {
    let r = g5_vector_load(true, PLAIN_LD);
    assert!(clean(&r) && r.findings.is_empty());
}

/// test_native_vector_destination_loads.py::test_racecheck_reports_a_vector_destination_load_racing_an_unordered_write
#[test]
fn g5_vector_destination_load_unordered_races() {
    let r = g5_vector_load(false, PLAIN_LD);
    assert!(has_class(&r, RaceClass::WriteRead));
    assert!(races(&r).iter().all(|f| f.alloc == SMEM));
}

/// test_native_vector_destination_loads.py::test_acquire_vector_destination_load_keeps_its_ordering_specialization
#[test]
fn g5_acquire_vector_destination_load_barrier_ordered_is_clean() {
    let r = g5_vector_load(true, ld(MemOrder::Acquire, Scope::Cta));
    assert!(clean(&r) && r.findings.is_empty());
}

/// test_native_kernel_contracts.py::test_local_scalar_indices_preserve_strided_race_and_nested_clean_case:
/// two warps both write `shared[lane + it*32]` for it in 0..2 (unordered
/// write_write, `missing_inter_actor_sync`); the nested disjoint variant
/// (`warp*64 + lane + it*32`, cta_sync, read own slot) is clean.
#[test]
fn g5_strided_overlap_races_and_nested_disjoint_is_clean() {
    let mut k = K::new(2, 1, 1);
    for w in 0..2 {
        for it in 0..2u64 {
            k.inst(w, &G5_ALL, PLAIN_ST, |l| (SMEM, (l as u64 + it * 32) * 4..(l as u64 + it * 32) * 4 + 4));
        }
    }
    let r = k.run();
    assert!(!races(&r).is_empty());
    assert!(races(&r).iter().all(|f| matches!(f.kind, FindingKind::DataRace { class: RaceClass::WriteWrite, .. })));
    assert!(has_failure(&r, |f| f == OrderingFailure::MissingInterActorSync));
    assert!(r.incomplete.is_empty());

    let mut k = K::new(2, 1, 1);
    for w in 0..2u32 {
        for it in 0..2u64 {
            k.inst(w, &G5_ALL, PLAIN_ST, |l| {
                let i = w as u64 * 64 + l as u64 + it * 32;
                (SMEM, i * 4..i * 4 + 4)
            });
        }
    }
    k.bar(0, &[0, 1]);
    for w in 0..2u32 {
        k.inst(w, &G5_ALL, PLAIN_LD, |l| {
            let i = w as u64 * 64 + l as u64;
            (SMEM, i * 4..i * 4 + 4)
        });
    }
    let r = k.run();
    assert!(clean(&r) && r.findings.is_empty());
}

/// test_native_racecheck_release_rmw_handoff.py::test_same_address_rmw_serialization_is_not_happens_before
/// (advisory facet; the race itself is pinned by
/// racecheck_scoped_and_lanes.rs::release_rmw_handoff_sibling_lanes_race).
/// Legacy additionally asserted no `undeclared_protocol_words` review.
/// delta R3: lane 1's `ld.acquire.gpu` observing its unordered sibling
/// lanes' morally strong `atom.release.gpu` on an undeclared word is now a
/// `UndeclaredProtocolWord` review advisory (not a race).
#[test]
fn g5_release_rmw_sibling_lanes_race_on_data_only() {
    let mut k = K::one_warp();
    k.st(0, 0, GMEM, 0..4);
    k.inst(0, &G5_ALL, atom(MemOrder::Release, Scope::Gpu), |_| (GMEM2, 0..4));
    k.a(0, 1, ld(MemOrder::Acquire, Scope::Gpu), GMEM2, 0..4).ld(0, 1, GMEM, 0..4);
    let r = k.run();
    let f = races(&r);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].alloc, GMEM);
    assert!(has_class(&r, RaceClass::WriteRead));
    assert!(r.incomplete.is_empty());
    // delta R3: the flag pair is an advisory, never an error.
    assert!(r.findings.iter().all(|f| f.alloc == GMEM || f.severity == Severity::Review));
    assert!(has_advisory(&r, AdvisoryKind::UndeclaredProtocolWord));
}

/// Raw spin from test_signal_diagnostics.py: W0L0 `st.{relaxed|release}.gpu`
/// on `flag`, optional `bar.sync`, W1L0 spins with `ld.acquire.gpu` on
/// `flag` (several polls), then writes `out`.
fn g5_raw_spin(ordered: bool) -> Report {
    let mut k = K::new(2, 1, 1);
    let order = if ordered { MemOrder::Release } else { MemOrder::Relaxed };
    k.a(0, 0, st(order, Scope::Gpu), GMEM, 0..4);
    if ordered {
        k.bar(0, &[0, 1]);
    }
    for _ in 0..3 {
        k.a(1, 0, ld(MemOrder::Acquire, Scope::Gpu), GMEM, 0..4);
    }
    k.st(1, 0, GMEM2, 0..4);
    k.run()
}

/// test_signal_diagnostics.py::test_raw_spin_missing_hb_reports_race_with_conditional_wait_hint
/// delta R3 (and R2): legacy G never exempted a strong generic load, so the
/// relaxed-store / acquire-spin pair on `flag` was a data race with a
/// `wait_until` hint. New: the pair is morally strong, not a race; the
/// undeclared protocol word is a `review` advisory.
#[test]
fn g5_raw_spin_relaxed_publication_is_advisory_not_race() {
    let r = g5_raw_spin(false);
    assert!(!has_race(&r)); // delta R3: legacy expected error/data_race
    assert!(has_advisory(&r, AdvisoryKind::UndeclaredProtocolWord));
    assert!(r.incomplete.is_empty());
}

/// test_signal_diagnostics.py::test_hb_ordered_raw_spin_does_not_need_a_wait_declaration:
/// HB-ordered spin is clean with no declaration advisory.
#[test]
fn g5_hb_ordered_raw_spin_needs_no_declaration() {
    let r = g5_raw_spin(true);
    assert!(clean(&r));
    assert!(r.findings.is_empty(), "{:?}", r.findings);
}

/// test_native_racecheck_per_warp.py::test_public_native_racecheck_distinguishes_disjoint_and_overlapping_warp_slices:
/// four warps write `shared[warp*32+lane]` (clean) or `shared[lane]`
/// (write_write between different warps, same lane, 4-byte overlap at
/// `lane*4`). Legacy reported exactly one finding (RS aborted at the first
/// race); delta P1: every race is reported, deduplicated per site pair.
#[test]
fn g5_per_warp_disjoint_vs_overlapping_slices() {
    let run = |overlap: bool| {
        let mut k = K::new(4, 1, 1);
        for w in 0..4u32 {
            k.inst(w, &G5_ALL, PLAIN_ST, |l| {
                let i = if overlap { l as u64 } else { w as u64 * 32 + l as u64 };
                (SMEM, i * 4..i * 4 + 4)
            });
        }
        k.run()
    };
    let r = run(false);
    assert!(clean(&r) && r.findings.is_empty());
    let r = run(true);
    let f = races(&r);
    assert!(!f.is_empty()); // delta P1: legacy len == 1
    assert!(r.incomplete.is_empty());
    for f in f {
        assert!(matches!(f.kind, FindingKind::DataRace { class: RaceClass::WriteWrite, .. }));
        let (p, c) = (f.prior.as_ref().unwrap(), f.current.as_ref().unwrap());
        assert_ne!(p.warp, c.warp);
        assert_eq!(p.span.end - p.span.start, 4);
        assert_eq!(c.span.end - c.span.start, 4);
    }
}

/// test_native_ordered_b128_load.py::test_ordered_b128_acquire_preserves_release_handoff:
/// C0 writes `data` and `response[1]` plainly, `st.release.gpu`
/// `response[0]`, then `st.relaxed.gpu` `ready`. C1 waits on `ready`
/// (accepts the relaxed write: no edge), then waits on `response[0]`
/// (accepts the release write: the handoff), then `ld.relaxed.gpu.b128`
/// over `response[0..2]` and a plain read of `data`. Clean.
#[test]
fn g5_ordered_b128_acquire_preserves_release_handoff() {
    // GMEM: data 0..8. GMEM2: response 0..16, ready 64..68.
    let mut k = K::new(1, 1, 2);
    k.declare(GMEM2, 0..8).declare(GMEM2, 64..68);
    k.st(0, 0, GMEM, 0..8).st(0, 0, GMEM2, 8..16);
    k.a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, 0..8);
    k.a(0, 0, st(MemOrder::Relaxed, Scope::Gpu), GMEM2, 64..68);
    k.wait_until(1, 0, GMEM2, 64..68, Scope::Gpu, 0b10, 1);
    k.wait_until(1, 0, GMEM2, 0..8, Scope::Gpu, 0b10, 1);
    k.a(1, 0, ld(MemOrder::Relaxed, Scope::Gpu), GMEM2, 0..16);
    k.ld(1, 0, GMEM, 0..8);
    let r = k.run();
    assert!(clean(&r), "{:?} {:?}", r.findings, r.incomplete);
    assert!(r.findings.is_empty(), "{:?}", r.findings);
}

/// Async write leg of a bulk reduction (`atomic`, relaxed.gpu, one span per
/// PTX element of `elem` bytes) or of a plain bulk copy (weak, one span).
fn g5_bulk_dst(k: &mut K, op: AsyncId, lane: u8, r: std::ops::Range<u64>, elem: Option<u64>) {
    let spans: Vec<LaneSpan> = match elem {
        Some(e) => (r.start..r.end).step_by(e as usize).map(|s| LaneSpan { lane, span: ByteSpan::new(s, e) }).collect(),
        None => vec![LaneSpan { lane, span: ByteSpan::new(r.start, r.end - r.start) }],
    };
    let atomic = elem.is_some();
    k.ev.push(Ev::Access {
        actor: Actor::Async { op, side: Milestone::Write },
        site: SiteId(950_000 + op.0 as u32),
        alloc: GMEM,
        space: Space::Global,
        kind: if atomic { AccessKind::Rmw } else { AccessKind::Write },
        sem: if atomic { Sem::Relaxed } else { Sem::Weak },
        scope: Scope::Gpu,
        atomic,
        returns_value: false,
        proxy: Proxy::Async,
        window: Some(Window::Global),
        spans,
    });
}

/// Two warps (lane 0): each zeroes its own 16 smem bytes, fences
/// `proxy.async.shared::cta`, then W0 issues `cp.reduce.async.bulk` of
/// `elem`-byte elements into `dst[0..16]` and W1 issues `peer`; each waits
/// its own bulk group.
fn g5_bulk_reduction(elem: u64, peer: &str) -> Report {
    let mut k = K::new(2, 1, 1);
    for w in 0..2u32 {
        let s = w as u64 * 16;
        k.st(w, 0, SMEM, s..s + 16).fence(w, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    }
    let mut ops = vec![];
    for w in 0..2u32 {
        let s = w as u64 * 16;
        let op = k.issue(w, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, s..s + 16)]);
        k.ar(op, Proxy::Async, SMEM, s..s + 16);
        let (dst, e) = match (w, peer) {
            (0, _) => (0..16, Some(elem)),
            (_, "matching_atomic") => (0..16, Some(elem)),
            (_, "overlapping_atomic") => (0..16, Some(4)),
            (_, "inside") => (0..16, None),
            (_, "outside") => (16..32, None),
            _ => unreachable!(),
        };
        g5_bulk_dst(&mut k, op, 0, dst, e);
        ops.push((w, op));
    }
    for (w, op) in ops {
        k.done_warp(op, Milestone::Read, w, 1).done_warp(op, Milestone::Write, w, 1);
    }
    k.run()
}

/// test_bulk_reduction_widths.py::test_bulk_reduction_element_width_and_boundary
/// [f16|u64 x matching_atomic|overlapping_atomic|inside|outside]: bulk
/// reductions are atomic per PTX element; a same-type peer reduction is
/// morally strong (clean), a `.u32` peer is mixed-size (delta R1: never
/// morally strong) and races, a plain bulk copy into the same bytes races,
/// and disjoint bytes are clean.
#[test]
fn g5_bulk_reduction_element_width_and_boundary() {
    for elem in [2u64, 8] {
        for peer in ["matching_atomic", "overlapping_atomic", "inside", "outside"] {
            let r = g5_bulk_reduction(elem, peer);
            if matches!(peer, "matching_atomic" | "outside") {
                assert!(clean(&r), "elem {elem} {peer}: {:?} {:?}", r.findings, r.incomplete);
            } else {
                assert!(has_race(&r), "elem {elem} {peer}: {:?}", r.findings);
                assert!(r.incomplete.is_empty(), "elem {elem} {peer}: {:?}", r.incomplete);
            }
        }
    }
}

// ======================================================================= g6
//
// Checker-verdict tests outside tests/analysis_tools/racecheck (numsim/runtime,
// numsim/integration, analysis_tools/shared), reviewed in the "unreviewed B"
// batch. Coverage map: scripts/numsim-v2/coverage/other_b.tsv.

const G6_TMEM1: AllocId = AllocId(6); // second CTA's tensor memory
const G6_ALL: u32 = u32::MAX;

/// One async-actor access with full control over order, scope, proxy and
/// per-lane spans (the `K::aacc` family is always weak and single-span).
#[allow(clippy::too_many_arguments)]
fn g6_acc(
    k: &mut K,
    op: AsyncId,
    side: Milestone,
    kind: AccessKind,
    sem: Sem,
    scope: Scope,
    proxy: Proxy,
    alloc: AllocId,
    space: Space,
    window: Option<Window>,
    spans: &[(u8, std::ops::Range<u64>)],
) {
    let atomic = kind == AccessKind::Rmw;
    k.ev.push(Ev::Access {
        actor: Actor::Async { op, side },
        site: SiteId(960_000 + op.0 as u32 * 2 + u32::from(side == Milestone::Write)),
        alloc,
        space,
        kind,
        sem,
        scope,
        atomic,
        returns_value: false,
        proxy,
        window,
        spans: spans.iter().map(|(l, r)| LaneSpan { lane: *l, span: ByteSpan::new(r.start, r.end - r.start) }).collect(),
    });
}

/// `n`-byte element spans of `r`, all for `lane`.
fn g6_elems(lane: u8, r: std::ops::Range<u64>, n: u64) -> Vec<(u8, std::ops::Range<u64>)> {
    (r.start..r.end).step_by(n as usize).map(|s| (lane, s..(s + n).min(r.end))).collect()
}

fn g6_lanes(n: u8) -> Vec<u8> {
    (0..n).collect()
}

// ------------------------------------ shared/test_native_dense_cta2_mma_ordering.py --

/// One cluster of two CTAs, two warps each (CTA0 = warps 0,1; CTA1 = 2,3).
/// Warp 0/2 lane 0 stage A/B in their CTA's smem, `fence.proxy.async`,
/// cluster sync; CTA0 warp 0 lane 0 issues one `cta_group::2` MMA that reads
/// both CTAs' smem and writes both CTAs' TMEM accumulators; with
/// `with_commit` it commits to its barrier and warp 0 waits; cluster sync;
/// warp 1 of each CTA reads its CTA's accumulator (a plain TMEM read as in
/// the legacy kernel, or `fence::after_thread_sync` + `tcgen05.ld`).
fn g6_dense_cta2(with_commit: bool, reader_tcgen: bool) -> Report {
    let mut k = K::new(2, 2, 2);
    k.alloc_cta(SMEM1, 1);
    k.alloc(G6_TMEM1, Space::Tmem, 1 << 16);
    k.alloc_cta(G6_TMEM1, 1);
    for (w, s) in [(0u32, SMEM), (2, SMEM1)] {
        k.st(w, 0, s, 0..2560);
    }
    for w in 0..4 {
        k.fence(w, G6_ALL, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    }
    k.cluster_bar(&[0, 1, 2, 3]);
    let mma = k.issue(0, 0, AsyncKind::TcgenPipelined, Proxy::Tcgen, &[], &[(TMEM, 0..8192), (G6_TMEM1, 0..8192)]);
    for s in [SMEM, SMEM1] {
        k.aacc(mma, Milestone::Read, AccessKind::Read, Proxy::Async, s, 0..2560);
    }
    for t in [TMEM, G6_TMEM1] {
        k.aacc(mma, Milestone::Write, AccessKind::Write, Proxy::Tcgen, t, 0..8192);
    }
    if with_commit {
        let c = k.issue(0, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &[mma], &[]);
        k.done_phase(c, Milestone::Write, 0, 0).wait(0, G6_ALL, 0, 0, true);
    }
    k.cluster_bar(&[0, 1, 2, 3]);
    for (w, t) in [(1u32, TMEM), (3, G6_TMEM1)] {
        if reader_tcgen {
            k.fence(w, G6_ALL, FenceKind::TcgenAfter);
            let ld = k.issue(w, 0, AsyncKind::TcgenLd, Proxy::Tcgen, &[], &[(t, 0..8192)]);
            k.aacc(ld, Milestone::Read, AccessKind::Read, Proxy::Tcgen, t, 0..8192).done_warp(ld, Milestone::Write, w, 1);
        } else {
            k.ld(w, 0, t, 0..8192);
        }
        k.st(w, 0, GMEM, u64::from(w) * 1024..u64::from(w) * 1024 + 1024);
    }
    k.run()
}

/// shared/test_native_dense_cta2_mma_ordering.py::test_committed_dense_cta2_mma_pipeline_is_race_free
#[test]
fn g6_committed_dense_cta2_mma_pipeline_is_race_free() {
    for reader_tcgen in [false, true] {
        let r = g6_dense_cta2(true, reader_tcgen);
        assert!(clean(&r) && r.findings.is_empty(), "tcgen reader {reader_tcgen}: {:?} {:?}", r.findings, r.incomplete);
    }
}

/// shared/test_native_dense_cta2_mma_ordering.py::test_uncommitted_dense_cta2_mma_pipeline_is_flagged
/// Every error is the CTA-pair MMA's TMEM write racing a reader
/// (`write_read`). Legacy saw exactly one error; delta P1: findings are
/// deduplicated per `(alloc, class, prior site, current site)`, so each
/// CTA's accumulator reports its own.
#[test]
fn g6_uncommitted_dense_cta2_mma_pipeline_is_flagged() {
    for reader_tcgen in [false, true] {
        let r = g6_dense_cta2(false, reader_tcgen);
        let errors: Vec<_> = r.errors().collect();
        assert!(!errors.is_empty(), "{r:?}");
        for f in errors {
            assert!(matches!(f.kind, FindingKind::DataRace { class: RaceClass::WriteRead, .. }), "{f:?}");
            assert!(f.alloc == TMEM || f.alloc == G6_TMEM1, "{f:?}");
            assert!(f.prior.as_ref().unwrap().async_op.is_some(), "{f:?}");
        }
    }
}

// ------------------------------------------ integration/test_fp8_tmem_a_effects.py --

/// How the TMEM-A stores are published before the `cta_sync`.
#[derive(Clone, Copy, PartialEq, Debug)]
enum G6Publish {
    /// No `wait::st` for the A stores (legacy `publish=False`); the tcgen05
    /// fence pair is kept so the missing wait is the only defect.
    None,
    /// The legacy kernel: `wait::st` + `fence::after_thread_sync`, then `cta_sync`.
    LegacyAfterFence,
    /// `wait::st` + `fence::before_thread_sync`, `cta_sync`, `fence::after_thread_sync`.
    FencePair,
}

/// Warpgroup (4 warps); TMEM cell (warp w's lanes, column c) at
/// `w*4096 + 4c`. Every warp `tcgen05.st` x16 at col 0 + wait::st, then x8
/// (the TMEM A operand) at col 16, published per `publish`; `cta_sync`; two
/// phases of warp 0 lane 0 MMA (reads A cols 16..24 and, in phase 1, D;
/// writes D cols 0..16 of rows 0..m) + commit + warp-0 wait + `cta_sync`;
/// every warp loads D.
fn g6_fp8_tmem_a(m: u32, publish: G6Publish) -> Report {
    let mut k = K::new(4, 1, 1);
    let cell = |w: u32, c: std::ops::Range<u64>| u64::from(w) * 4096 + c.start * 4..u64::from(w) * 4096 + c.end * 4;
    k.st(0, 0, SMEM, 0..512).fence(0, G6_ALL, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    k.bar(0, &[0, 1, 2, 3]);
    let mut late = vec![];
    for w in 0..4u32 {
        let st1 = k.issue(w, 0, AsyncKind::TcgenSt, Proxy::Tcgen, &[], &[(TMEM, cell(w, 0..16))]);
        k.aacc(st1, Milestone::Write, AccessKind::Write, Proxy::Tcgen, TMEM, cell(w, 0..16)).done_warp(st1, Milestone::Write, w, G6_ALL);
        let st2 = k.issue(w, 0, AsyncKind::TcgenSt, Proxy::Tcgen, &[], &[(TMEM, cell(w, 16..24))]);
        k.aacc(st2, Milestone::Write, AccessKind::Write, Proxy::Tcgen, TMEM, cell(w, 16..24));
        match publish {
            G6Publish::None => {
                k.fence(w, G6_ALL, FenceKind::TcgenBefore);
                late.push((w, st2));
            }
            G6Publish::LegacyAfterFence => {
                k.done_warp(st2, Milestone::Write, w, G6_ALL).fence(w, G6_ALL, FenceKind::TcgenAfter);
            }
            G6Publish::FencePair => {
                k.done_warp(st2, Milestone::Write, w, G6_ALL).fence(w, G6_ALL, FenceKind::TcgenBefore);
            }
        }
    }
    k.bar(0, &[0, 1, 2, 3]);
    let fenced = publish != G6Publish::LegacyAfterFence;
    if fenced {
        for w in 0..4u32 {
            k.fence(w, G6_ALL, FenceKind::TcgenAfter);
        }
    }
    let rows = m / 32;
    for phase in 0..2u64 {
        let mma = k.issue(0, 0, AsyncKind::TcgenPipelined, Proxy::Tcgen, &[], &[(TMEM, 0..16384)]);
        k.aacc(mma, Milestone::Read, AccessKind::Read, Proxy::Async, SMEM, 0..512);
        for w in 0..rows {
            k.aacc(mma, Milestone::Read, AccessKind::Read, Proxy::Tcgen, TMEM, cell(w, 16..24));
            if phase == 1 {
                k.aacc(mma, Milestone::Read, AccessKind::Read, Proxy::Tcgen, TMEM, cell(w, 0..16));
            }
            k.aacc(mma, Milestone::Write, AccessKind::Write, Proxy::Tcgen, TMEM, cell(w, 0..16));
        }
        let c = k.issue(0, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &[mma], &[]);
        k.done_phase(c, Milestone::Write, 0, phase).wait(0, G6_ALL, 0, phase, true);
        k.bar(0, &[0, 1, 2, 3]);
    }
    for w in 0..4u32 {
        if fenced {
            k.fence(w, G6_ALL, FenceKind::TcgenAfter);
        }
        let ld = k.issue(w, 0, AsyncKind::TcgenLd, Proxy::Tcgen, &[], &[(TMEM, cell(w, 0..16))]);
        k.aacc(ld, Milestone::Read, AccessKind::Read, Proxy::Tcgen, TMEM, cell(w, 0..16)).done_warp(ld, Milestone::Write, w, G6_ALL);
    }
    // The unwaited stores drain at exit.
    for (w, st2) in late {
        k.done_warp(st2, Milestone::Write, w, G6_ALL);
    }
    k.run()
}

/// integration/test_fp8_tmem_a_effects.py::test_fp8_tmem_a_requires_published_stores[64-False|128-True]
/// (racecheck half, `publish=False`): without `wait::st` the MMA's TMEM-A
/// read races the store (`write_read`, `async_lifetime_not_drained`, TMEM,
/// overlap = the 8 A columns at byte 64 of the warp's row block). With the
/// full `before_thread_sync` / `after_thread_sync` pair it is clean.
#[test]
fn g6_fp8_tmem_a_requires_published_stores() {
    for m in [64u32, 128] {
        let r = g6_fp8_tmem_a(m, G6Publish::FencePair);
        assert!(clean(&r) && r.findings.is_empty(), "m={m}: {:?} {:?}", r.findings, r.incomplete);
        let r = g6_fp8_tmem_a(m, G6Publish::None);
        let errors: Vec<_> = r.errors().collect();
        assert!(!errors.is_empty(), "m={m}: {r:?}");
        for f in &errors {
            assert!(
                matches!(f.kind, FindingKind::DataRace { class: RaceClass::WriteRead, failure: OrderingFailure::AsyncLifetimeNotDrained }),
                "m={m}: {f:?}"
            );
            assert_eq!(f.alloc, TMEM);
            assert_eq!((f.bytes.start % 4096, f.bytes.end - f.bytes.start), (64, 32), "m={m}: {f:?}");
        }
    }
}

/// integration/test_fp8_tmem_a_effects.py::test_fp8_tmem_a_requires_published_stores[64-False|128-True]
/// (the `publish=True` kernel, which both checkers must call clean): the
/// stores are waited and followed by `fence::after_thread_sync` *before*
/// the `cta_sync`; there is no `before_thread_sync`, and no
/// `after_thread_sync` after any `cta_sync` (neither before the MMA nor
/// before the final `tcgen05.ld` of D). Legacy: clean. New: waited tcgen05
/// stores and committed MMA results reach another warp's tcgen05 op only
/// through the fence pair (racecheck-semantics §3 rows 19-23), so both
/// hand-offs race.
#[test]
fn g6_fp8_tmem_a_legacy_publication_is_clean() {
    for m in [64u32, 128] {
        let r = g6_fp8_tmem_a(m, G6Publish::LegacyAfterFence);
        assert!(clean(&r) && r.findings.is_empty(), "m={m}: {:?} {:?}", r.findings, r.incomplete);
    }
}

// ------------------------------------------------- runtime/test_async_release.py --

/// One `st.async.release.gpu.global` (or `red.async.release.gpu.global.add`)
/// write by warp `w` lane `lane` to `alloc[r]`.
fn g6_async_release(k: &mut K, w: WarpId, lane: u8, reduction: bool, alloc: AllocId, r: std::ops::Range<u64>) -> AsyncId {
    let op = k.issue(w, lane, AsyncKind::Copy, Proxy::Generic, &[], &[(alloc, r.clone())]);
    let kind = if reduction { AccessKind::Rmw } else { AccessKind::Write };
    g6_acc(k, op, Milestone::Write, kind, Sem::Release, Scope::Gpu, Proxy::Generic, alloc, Space::Global, Some(Window::Global), &[(lane, r)]);
    // Completes with no observer (no mbarrier, not in a bulk group).
    k.done_warp(op, Milestone::Write, w, 0);
    op
}

/// runtime/test_async_release.py::test_bulk_wait_does_not_acquire_async_release[False|True]
/// Lanes 0..16 each issue an async release store/reduction to `data[lane]`;
/// `cp.async.bulk.commit_group` + `wait_group 0` do not cover it, so every
/// lane's read of `data[lane]` races it.
#[test]
fn g6_bulk_wait_does_not_acquire_async_release() {
    for reduction in [false, true] {
        let mut k = K::one_warp();
        for l in 0..16u8 {
            g6_async_release(&mut k, 0, l, reduction, GMEM, u64::from(l) * 4..u64::from(l) * 4 + 4);
        }
        k.inst(0, &G5_ALL, PLAIN_LD, |l| (GMEM, u64::from(l) * 4..u64::from(l) * 4 + 4));
        let r = k.run();
        assert!(has_class(&r, RaceClass::WriteRead) || has_class(&r, RaceClass::ReadWrite), "red={reduction}: {r:?}");
    }
}

/// Warp 0 lane 0 writes `data` (global, or `scratch` in smem) before or
/// after an async release store/reduction to the declared `flag`; warp 1
/// lane 0 `wait_until`s the flag (accepting that write) and reads the data.
fn g6_async_release_publication(after: bool, shared: bool, reduction: bool) -> Report {
    let mut k = K::new(2, 1, 1);
    k.declare(GMEM2, 0..4);
    let data = if shared { SMEM } else { GMEM };
    if !after {
        k.st(0, 0, data, 0..4);
    }
    g6_async_release(&mut k, 0, 0, reduction, GMEM2, 0..4);
    if after {
        k.st(0, 0, data, 0..4);
    }
    k.a(1, 0, ld(MemOrder::Acquire, Scope::Gpu), GMEM2, 0..4).wait_until(1, 0, GMEM2, 0..4, Scope::Gpu, 0b10, 1);
    k.ld(1, 0, data, 0..4).st(1, 0, GMEM, 64..68);
    k.run()
}

/// runtime/test_async_release.py::test_async_release_publishes_only_pre_issue_work
/// [shared,reduction = F,F | T,F | F,T] (the `after=True` half): data
/// written after the release is not published.
#[test]
fn g6_async_release_does_not_publish_post_issue_work() {
    for (shared, reduction) in [(false, false), (true, false), (false, true)] {
        let r = g6_async_release_publication(true, shared, reduction);
        assert!(has_class(&r, RaceClass::WriteRead) || has_class(&r, RaceClass::ReadWrite), "{shared} {reduction}: {r:?}");
    }
}

/// runtime/test_async_release.py::test_async_release_publishes_only_pre_issue_work
/// (the `after=False` half, which `run_checked` requires clean in both
/// checkers): an async `.release` publication orders the issuer's earlier
/// writes before a waiter that accepts it. The new core gives async-actor
/// writes no release head (`own_rel` only for warp lanes), so the wait owes
/// no edge and the read races, although racecheck-semantics §5 treats async
/// publications as edge sources (the observed-version fallback).
#[test]
fn g6_async_release_publishes_pre_issue_work() {
    for (shared, reduction) in [(false, false), (true, false), (false, true)] {
        let r = g6_async_release_publication(false, shared, reduction);
        assert!(clean(&r), "{shared} {reduction}: {:?} {:?}", r.findings, r.incomplete);
    }
}

// -------------------------------- runtime/test_bulk_copy_scopes.py, test_bulk_g2s_scopes.py --

#[derive(Clone, Copy, PartialEq, Debug)]
enum G6Rel {
    /// Two warps of one CTA.
    Cta,
    /// Two CTAs of one cluster.
    Cluster,
    /// Two clusters.
    Gpu,
}

fn g6_rel_k(rel: G6Rel) -> K {
    let mut k = match rel {
        G6Rel::Cta => K::new(2, 1, 1),
        G6Rel::Cluster => K::new(1, 2, 2),
        G6Rel::Gpu => K::new(1, 1, 2),
    };
    if rel != G6Rel::Cta {
        k.alloc_cta(SMEM1, 1);
    }
    k
}

/// `None` = weak copy; `Some(s)` = `.relaxed.s ... .b128` (strong, atomic
/// per 16-byte element).
fn g6_copy_dst(k: &mut K, op: AsyncId, lane: u8, scope: Option<Scope>, alloc: AllocId, space: Space, window: Window, r: std::ops::Range<u64>) {
    let spans = match scope {
        Some(_) => g6_elems(lane, r, 16),
        None => vec![(lane, r)],
    };
    let sem = if scope.is_some() { Sem::Relaxed } else { Sem::Weak };
    g6_acc(k, op, Milestone::Write, AccessKind::Write, sem, scope.unwrap_or(Scope::Gpu), Proxy::Async, alloc, space, Some(window), &spans);
}

/// test_bulk_copy_scopes.py `copy_case`: actor a (warp a) stages 8 words in
/// its smem, `fence.proxy.async`, lane 0 bulk-copies `32 - 16a` bytes to
/// `destination[4a..]` (actor 0 bytes 0..32, actor 1 bytes 16..32), waits
/// `.read`, rewrites the staging words, then waits the full group.
fn g6_bulk_copy(scope: Option<Scope>, rel: G6Rel, enabled: bool) -> Report {
    let mut k = g6_rel_k(rel);
    for a in 0..2u32 {
        let (smem, off) = match rel {
            G6Rel::Cta => (SMEM, u64::from(a) * 32),
            _ => (if a == 0 { SMEM } else { SMEM1 }, 0),
        };
        let lanes = g6_lanes(8);
        k.inst(a, &lanes, PLAIN_ST, |l| (smem, off + u64::from(l) * 4..off + u64::from(l) * 4 + 4));
        k.syncwarp(a, G6_ALL).fence(a, G6_ALL, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
        let op = enabled.then(|| {
            let dst = u64::from(a) * 16..32;
            let op = k.issue(a, 0, AsyncKind::Copy, Proxy::Async, &[], &[(GMEM, dst.clone())]);
            k.ar(op, Proxy::Async, smem, off..off + dst.end - dst.start);
            g6_copy_dst(&mut k, op, 0, scope, GMEM, Space::Global, Window::Global, dst);
            k.done_warp(op, Milestone::Read, a, 1);
            op
        });
        k.syncwarp(a, G6_ALL);
        k.inst(a, &lanes, PLAIN_ST, |l| (smem, off + u64::from(l) * 4..off + u64::from(l) * 4 + 4));
        k.syncwarp(a, G6_ALL);
        if let Some(op) = op {
            k.done_warp(op, Milestone::Write, a, 1);
        }
    }
    k.run()
}

/// runtime/test_bulk_copy_scopes.py::test_bulk_copy_scope_and_elements[cta|cluster|gpu|sys]
/// (x relation cta|cluster|gpu x enabled 0|1): overlapping strong `.b128`
/// bulk copies are morally strong (clean) when the scopes cover both
/// issuers; otherwise legacy reported `scope_mismatch`. delta R4: an
/// unordered strong pair that fails only on scope is now a data race with
/// `missing_release_acquire` (no `ScopeMismatch`, which is reserved for a
/// release/acquire edge).
#[test]
fn g6_bulk_copy_scope_and_elements() {
    for scope in [Scope::Cta, Scope::Cluster, Scope::Gpu, Scope::Sys] {
        for rel in [G6Rel::Cta, G6Rel::Cluster, G6Rel::Gpu] {
            for enabled in [false, true] {
                let r = g6_bulk_copy(Some(scope), rel, enabled);
                let mismatch = enabled && ((scope == Scope::Cta && rel != G6Rel::Cta) || (scope == Scope::Cluster && rel == G6Rel::Gpu));
                if mismatch {
                    // delta R4: legacy `scope_mismatch`.
                    assert!(has_failure(&r, |f| f == OrderingFailure::MissingReleaseAcquire), "{scope:?} {rel:?}: {r:?}");
                    assert!(has_class(&r, RaceClass::WriteWrite), "{scope:?} {rel:?}: {r:?}");
                    assert!(r.incomplete.is_empty());
                } else {
                    assert!(clean(&r) && r.findings.is_empty(), "{scope:?} {rel:?} {enabled}: {:?} {:?}", r.findings, r.incomplete);
                }
            }
        }
    }
}

/// runtime/test_bulk_copy_scopes.py::test_bulk_copy_weak_overlap_still_races
#[test]
fn g6_bulk_copy_weak_overlap_still_races() {
    let r = g6_bulk_copy(None, G6Rel::Cta, true);
    assert!(has_class(&r, RaceClass::WriteWrite), "{r:?}");
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum G6Writers {
    Warps,
    Lanes,
    /// Two CTAs of one cluster, both writing CTA 0's smem through `mapa`.
    Ctas,
}

/// test_bulk_g2s_scopes.py `g2s_case`: CTA0 warp 0 zeroes the 8-word
/// destination, inits the barrier and arms `expect_tx(48)`; cluster sync;
/// actor a copies `32 - 16a` bytes of `source[8a..]` into `shared[4a..]`
/// completing on CTA 0's barrier; CTA0 warp 0 lane 0 waits; cluster sync;
/// lanes 0..8 read the destination.
fn g6_g2s(scope: Option<Scope>, writers: G6Writers, enabled: bool) -> Report {
    let mut k = match writers {
        G6Writers::Warps => K::new(2, 1, 1),
        G6Writers::Lanes => K::one_warp(),
        G6Writers::Ctas => K::new(1, 2, 2),
    };
    let all: Vec<WarpId> = if writers == G6Writers::Lanes { vec![0] } else { vec![0, 1] };
    let lanes = g6_lanes(8);
    k.inst(0, &lanes, PLAIN_ST, |l| (SMEM, u64::from(l) * 4..u64::from(l) * 4 + 4));
    k.arrive(0, 1, 0, 0, true);
    for &w in &all {
        k.fence(w, G6_ALL, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
    }
    k.cluster_bar(&all);
    let window = if writers == G6Writers::Ctas { Window::SharedCluster } else { Window::SharedCta };
    if enabled {
        for a in 0..2u32 {
            let (w, lane) = if writers == G6Writers::Lanes { (0, a as u8) } else { (a, 0) };
            let dst = u64::from(a) * 16..32;
            let op = k.issue(w, lane, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, dst.clone())]);
            let src = u64::from(a) * 32..u64::from(a) * 32 + (dst.end - dst.start);
            k.ar(op, Proxy::Async, GMEM, src);
            g6_copy_dst(&mut k, op, lane, scope, SMEM, Space::Shared, window, dst);
            k.done_phase(op, Milestone::Write, 0, 0);
        }
    }
    k.wait(0, 1, 0, 0, true);
    k.cluster_bar(&all);
    k.inst(0, &lanes, PLAIN_LD, |l| (SMEM, u64::from(l) * 4..u64::from(l) * 4 + 4));
    k.run()
}

/// runtime/test_bulk_g2s_scopes.py::test_strong_g2s_scope_elements_and_predication[cta|cluster|gpu|sys]
/// (x writers warps|lanes|ctas x enabled 0|1): overlapping strong G2S copies
/// are clean unless the issuers are in different CTAs and the scope is
/// `.cta` (`write_write`; delta R4 adds `missing_release_acquire`).
#[test]
fn g6_strong_g2s_scope_elements_and_predication() {
    for scope in [Scope::Cta, Scope::Cluster, Scope::Gpu, Scope::Sys] {
        for writers in [G6Writers::Warps, G6Writers::Lanes, G6Writers::Ctas] {
            for enabled in [false, true] {
                let r = g6_g2s(Some(scope), writers, enabled);
                if enabled && writers == G6Writers::Ctas && scope == Scope::Cta {
                    assert!(has_class(&r, RaceClass::WriteWrite), "{scope:?} {writers:?}: {r:?}");
                    assert!(has_failure(&r, |f| f == OrderingFailure::MissingReleaseAcquire), "{r:?}");
                } else {
                    assert!(clean(&r) && r.findings.is_empty(), "{scope:?} {writers:?} {enabled}: {:?} {:?}", r.findings, r.incomplete);
                }
            }
        }
    }
}

/// runtime/test_bulk_g2s_scopes.py::test_weak_g2s_overlapping_writes_still_race
#[test]
fn g6_weak_g2s_overlapping_writes_still_race() {
    let r = g6_g2s(None, G6Writers::Warps, true);
    assert!(has_class(&r, RaceClass::WriteWrite), "{r:?}");
}

// ---------------------------------------------- runtime/test_bulk_reduce_s2g_f32.py --

/// runtime/test_bulk_reduce_s2g_f32.py::test_bulk_reduce_partial_overlap_is_elementwise_atomic:
/// two CTAs (two clusters) `cp.reduce.async.bulk.global.shared::cta.add.f32`
/// into `destination[4c..]` (CTA 0: 32 bytes, CTA 1: 16 bytes): the partial
/// overlap is elementwise atomic, clean.
#[test]
fn g6_bulk_reduce_partial_overlap_is_elementwise_atomic() {
    let mut k = K::new(1, 1, 2);
    k.alloc_cta(SMEM1, 1);
    for c in 0..2u32 {
        let smem = if c == 0 { SMEM } else { SMEM1 };
        let len = if c == 0 { 32 } else { 16 };
        k.st(c, 0, smem, 0..32).fence(c, 1, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
        let dst = u64::from(c) * 16..u64::from(c) * 16 + len;
        let op = k.issue(c, 0, AsyncKind::Copy, Proxy::Async, &[], &[(GMEM, dst.clone())]);
        k.ar(op, Proxy::Async, smem, 0..len);
        g6_acc(&mut k, op, Milestone::Write, AccessKind::Rmw, Sem::Relaxed, Scope::Gpu, Proxy::Async, GMEM, Space::Global, Some(Window::Global), &g6_elems(0, dst, 4));
        k.done_warp(op, Milestone::Read, c, 1).done_warp(op, Milestone::Write, c, 1);
    }
    let r = k.run();
    assert!(clean(&r) && r.findings.is_empty(), "{:?} {:?}", r.findings, r.incomplete);
}

// -------------------------------------------- runtime/test_mbarrier_lane_semantics.py --

/// runtime/test_mbarrier_lane_semantics.py::test_checkers_group_pending_blocking_waits_on_distinct_barriers[racecheck]
/// (`checker_lane_varying_pending_wait`): warp 1 lanes 0..4 arrive on and
/// wait `barriers[lane]` (phase 0); `cta_sync`; warp 0 lanes 0..4 arrive
/// (phase 1) while warp 1's one lane-varying wait keeps all four requests
/// pending; warp 1 then writes `output[lane]`. Clean.
#[test]
fn g6_lane_varying_pending_waits_on_distinct_barriers_are_clean() {
    let mut k = K::new(2, 1, 1);
    k.bar(0, &[0, 1]);
    for l in 0..4u32 {
        k.arrive(1, 1 << l, l, 0, true);
    }
    for l in 0..4u32 {
        k.wait(1, 1 << l, l, 0, true);
    }
    k.bar(0, &[0, 1]);
    for l in 0..4u32 {
        k.arrive(0, 1 << l, l, 1, true);
    }
    for l in 0..4u32 {
        k.wait(1, 1 << l, l, 1, true);
    }
    let lanes = g6_lanes(4);
    k.inst(1, &lanes, PLAIN_ST, |l| (GMEM, u64::from(l) * 4..u64::from(l) * 4 + 4));
    let r = k.run();
    assert!(clean(&r) && r.findings.is_empty(), "{:?} {:?}", r.findings, r.incomplete);
}

/// runtime/test_mbarrier_lane_semantics.py::test_checkers_allow_distinct_barrier_waits[racecheck]
/// (`lane_varying_expect_tx`): every lane zeroes `shared[lane]` and
/// `shared[lane+32]`; `fence.proxy.async`; `cta_sync`; lane 0 and lane 1
/// each arm their own barrier and bulk-copy 16 bytes into `shared[0..16]` /
/// `shared[32..48]`; lanes 0 and 1 wait their own barrier; `__syncwarp`;
/// every lane reads `shared[lane]` and `shared[lane+32]`. Clean.
#[test]
fn g6_lane_varying_expect_tx_waits_are_clean() {
    let mut k = K::one_warp();
    for base in [0u64, 32] {
        k.inst(0, &G5_ALL, PLAIN_ST, |l| (SMEM, base + u64::from(l)..base + u64::from(l) + 1));
    }
    k.fence(0, G6_ALL, FenceKind::ProxyAsync(Some(Domain::SharedCta))).bar(0, &[0]);
    for l in 0..2u8 {
        let dst = u64::from(l) * 32..u64::from(l) * 32 + 16;
        k.arrive(0, 1 << l, u32::from(l), 0, true);
        let op = k.issue(0, l, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, dst.clone())]);
        k.ar(op, Proxy::Async, GMEM, dst.clone()).aw(op, Proxy::Async, SMEM, dst).done_phase(op, Milestone::Write, u32::from(l), 0);
    }
    for l in 0..2u32 {
        k.wait(0, 1 << l, l, 0, true);
    }
    k.syncwarp(0, G6_ALL);
    for base in [0u64, 32] {
        k.inst(0, &G5_ALL, PLAIN_LD, |l| (SMEM, base + u64::from(l)..base + u64::from(l) + 1));
    }
    let r = k.run();
    assert!(clean(&r) && r.findings.is_empty(), "{:?} {:?}", r.findings, r.incomplete);
}

/// runtime/test_mbarrier_lane_semantics.py::test_relaxed_query_does_not_acquire_arriving_threads_memory
/// Warp 0 lane 0 writes `data` (global) and `shared[0]`, then arrives
/// (default `.release.cta`); warp 1 lane 0 polls with
/// `mbarrier.test_wait.parity.relaxed.cta` and reads both: two `write_read`
/// races (sync delta B2 agrees: a relaxed wait synchronises nothing).
#[test]
fn g6_relaxed_query_does_not_acquire_arriving_threads_memory() {
    let mut k = K::new(2, 1, 1);
    k.bar(0, &[0, 1]);
    k.st(0, 0, GMEM, 0..4).st(0, 0, SMEM, 0..4).arrive(0, 1, 0, 0, true);
    k.wait(1, 1, 0, 0, false);
    k.ld(1, 0, GMEM, 0..4).ld(1, 0, SMEM, 0..4);
    let r = k.run();
    let errs: Vec<_> = r.errors().filter(|f| matches!(f.kind, FindingKind::DataRace { class: RaceClass::WriteRead, .. })).collect();
    assert_eq!(errs.len(), 2, "{r:?}");
    assert!(errs.iter().any(|f| f.alloc == GMEM) && errs.iter().any(|f| f.alloc == SMEM));
}

// ----------------------------------------------------- runtime/test_memory_ops.py --

/// runtime/test_memory_ops.py::test_bulk_g2s_cta_has_exact_racecheck_payload_accesses
/// (`bulk_g2s_cta`): lane 0 issues a 16-byte G2S copy *before* its
/// `arrive.expect_tx`, waits; `__syncwarp`; lanes 0..16 read one byte each.
/// The waiting lane hands the completed copy to its siblings: clean.
#[test]
fn g6_bulk_g2s_cta_waiter_hands_payload_to_the_warp() {
    let mut k = K::one_warp();
    k.bar(0, &[0]);
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, GMEM, 0..16).aw(op, Proxy::Async, SMEM, 0..16).done_phase(op, Milestone::Write, 0, 0);
    k.arrive(0, 1, 0, 0, true).wait(0, 1, 0, 0, true).syncwarp(0, G6_ALL);
    let lanes = g6_lanes(16);
    k.inst(0, &lanes, PLAIN_LD, |l| (SMEM, u64::from(l)..u64::from(l) + 1));
    k.inst(0, &lanes, PLAIN_ST, |l| (GMEM2, u64::from(l)..u64::from(l) + 1));
    let r = k.run();
    assert!(clean(&r) && r.findings.is_empty(), "{:?} {:?}", r.findings, r.incomplete);
}

// ----------------------------------------------- runtime/test_mixed_version_reads.py --

/// runtime/test_mixed_version_reads.py::test_mixed_version_does_not_publish_later_data
/// [volatile|acquire|relaxed|cas] (`late_data=True`): warps 0 and 1 each
/// `fence.release.gpu` and `.cp_mask` bulk-copy half of the 16-byte flag
/// (`relaxed.gpu .b128`), wait, `cta_sync`; *then* write `data[warp]`.
/// Warp 2 spins on the flag (`fence.proxy.async.global` + a b128 read in
/// `mode`) and reads both data words: neither the release heads nor the
/// barrier publish the later stores, so the reads race; no incomplete.
#[test]
fn g6_mixed_version_does_not_publish_later_data() {
    for mode in ["volatile", "acquire", "relaxed", "cas"] {
        let mut k = K::new(3, 1, 1);
        for w in 0..2u32 {
            let lanes = g6_lanes(16);
            k.inst(w, &lanes, PLAIN_ST, |l| (SMEM, u64::from(w) * 16 + u64::from(l)..u64::from(w) * 16 + u64::from(l) + 1));
            k.syncwarp(w, G6_ALL).fence(w, G6_ALL, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
            k.fence(w, 1, FenceKind::AcqRel(Scope::Gpu));
            let half = u64::from(w) * 8..u64::from(w) * 8 + 8;
            let op = k.issue(w, 0, AsyncKind::Copy, Proxy::Async, &[], &[(GMEM2, half.clone())]);
            k.ar(op, Proxy::Async, SMEM, u64::from(w) * 16..u64::from(w) * 16 + 16);
            g6_acc(&mut k, op, Milestone::Write, AccessKind::Write, Sem::Relaxed, Scope::Gpu, Proxy::Async, GMEM2, Space::Global, Some(Window::Global), &[(0, half)]);
            k.done_warp(op, Milestone::Read, w, 1).done_warp(op, Milestone::Write, w, 1);
        }
        k.bar(0, &[0, 1, 2]);
        for w in 0..2u32 {
            k.st(w, 0, GMEM, u64::from(w) * 4..u64::from(w) * 4 + 4);
        }
        for _ in 0..2 {
            k.fence(2, 1, FenceKind::ProxyAsync(Some(Domain::Global)));
            match mode {
                "volatile" => k.a(2, 0, ld(MemOrder::Relaxed, Scope::Sys), GMEM2, 0..16),
                "acquire" => k.a(2, 0, ld(MemOrder::Acquire, Scope::Gpu), GMEM2, 0..16),
                "relaxed" => k.a(2, 0, ld(MemOrder::Relaxed, Scope::Gpu), GMEM2, 0..16).fence(2, 1, FenceKind::AcqRel(Scope::Gpu)),
                _ => k.a(2, 0, atom(MemOrder::Acquire, Scope::Gpu), GMEM2, 0..16),
            };
        }
        k.ld(2, 0, GMEM, 0..4).ld(2, 0, GMEM, 4..8).st(2, 0, GMEM, 64..72);
        let r = k.run();
        assert!(r.incomplete.is_empty(), "{mode}: {:?}", r.incomplete);
        assert!(
            races(&r).iter().any(|f| f.alloc == GMEM && matches!(f.kind, FindingKind::DataRace { class: RaceClass::WriteRead | RaceClass::ReadWrite, .. })),
            "{mode}: {r:?}"
        );
    }
}

// ------------------------------------------------------- runtime/test_red_async.py --

/// `reduction_kernel`: CTA 1 lane 0 initialises `destination` (and each
/// CTA its `source`), inits its barrier for `lanes` arrivals; cluster sync;
/// CTA 0 lanes 0..lanes arm CTA 1's barrier (`arrive.expect_tx.shared::cluster`,
/// default `.release.cta`) and issue `red.async.relaxed.cluster` (4-byte,
/// generic proxy) or one 16-byte `cp.reduce.async.bulk.shared::cluster`
/// (async proxy, after `fence.proxy.async`) into CTA 1's destination,
/// completing on CTA 1's barrier; CTA 1 lane 0 (optionally) waits and reads
/// the destination; cluster sync.
fn g6_red_async(bulk: bool, wait: bool) -> Report {
    let lanes: u8 = if bulk { 1 } else { 4 };
    let mut k = K::new(1, 2, 2);
    k.alloc_cta(SMEM1, 1);
    k.st(0, 0, SMEM, 16..32).st(1, 0, SMEM1, 0..32);
    if bulk {
        for w in 0..2 {
            k.fence(w, G6_ALL, FenceKind::ProxyAsync(Some(Domain::SharedCta)));
        }
    }
    k.cluster_bar(&[0, 1]);
    let barrier = mbar_in(1, 0);
    for l in 0..lanes {
        k.arrive_q(0, 1 << l, barrier, 0, Some(true), Some(Scope::Cta));
    }
    for l in 0..lanes {
        let op = k.issue(0, l, AsyncKind::Copy, if bulk { Proxy::Async } else { Proxy::Generic }, &[], &[(SMEM1, 0..16)]);
        if bulk {
            k.ar(op, Proxy::Async, SMEM, 16..32);
            g6_acc(&mut k, op, Milestone::Write, AccessKind::Rmw, Sem::Relaxed, Scope::Cluster, Proxy::Async, SMEM1, Space::Shared, Some(Window::SharedCluster), &g6_elems(l, 0..16, 4));
        } else {
            g6_acc(&mut k, op, Milestone::Write, AccessKind::Rmw, Sem::Relaxed, Scope::Cluster, Proxy::Generic, SMEM1, Space::Shared, Some(Window::SharedCluster), &[(l, 0..4)]);
        }
        k.done_phase_r(op, Milestone::Write, barrier, 0);
    }
    if wait {
        k.wait_q(1, 1, barrier, 0, Some(true), Some(Scope::Cta));
    }
    k.ld(1, 0, SMEM1, 0..if bulk { 16 } else { 4 });
    k.cluster_bar(&[0, 1]);
    k.run()
}

/// runtime/test_red_async.py::test_shared_async_reduction_completion
/// [F-u32-add | T-u32-add | T-s32-min | T-b32-xor | T-u64-add] (the op and
/// type only change values): the unwaited read races the remote reduction.
/// The waited kernel (which legacy `run_checked` requires clean) has no data
/// race; delta B1/V5: the qualifier-less remote `arrive.expect_tx` is
/// `.release.cta`, so CTA 1's `.cta` wait reports `ScopeMismatch` on that
/// arrival edge (legacy: clean).
#[test]
fn g6_shared_async_reduction_completion() {
    for bulk in [false, true] {
        let r = g6_red_async(bulk, true);
        assert!(!has_race(&r) && r.incomplete.is_empty(), "bulk={bulk}: {:?} {:?}", r.findings, r.incomplete);
        assert!(r.errors().all(|f| matches!(f.kind, FindingKind::ScopeMismatch { .. })), "bulk={bulk}: {r:?}");
        let r = g6_red_async(bulk, false);
        assert!(has_class(&r, RaceClass::WriteRead) || has_class(&r, RaceClass::ReadWrite), "bulk={bulk}: {r:?}");
        assert!(races(&r).iter().all(|f| f.alloc == SMEM1));
    }
}

// ------------------------------------------------------ runtime/test_store_sinks.py --

/// One warp instruction whose lanes each touch several spans (a vector
/// access: one span per element).
fn g6_vec_store(k: &mut K, w: WarpId, op: Op, alloc: AllocId, spans: &[(u8, std::ops::Range<u64>)]) {
    let epoch = k.tick(w);
    let space = if alloc == GMEM || alloc == GMEM2 { Space::Global } else { Space::Shared };
    let window = if space == Space::Global { Window::Global } else { Window::SharedCta };
    k.ev.push(Ev::Access {
        actor: Actor::Warp { warp: numsim_core::observe::WarpId(w), epoch: u64::from(epoch) },
        site: SiteId(w * 1000 + epoch),
        alloc,
        space,
        kind: op.kind,
        sem: match op.order {
            MemOrder::Weak => Sem::Weak,
            MemOrder::Relaxed => Sem::Relaxed,
            MemOrder::Acquire => Sem::Acquire,
            MemOrder::Release => Sem::Release,
            MemOrder::AcqRel => Sem::AcqRel,
        },
        scope: op.scope.unwrap_or(Scope::Gpu),
        atomic: op.atomic,
        returns_value: op.atomic,
        proxy: Proxy::Generic,
        window: Some(window),
        spans: spans.iter().map(|(l, r)| LaneSpan { lane: *l, span: ByteSpan::new(r.start, r.end - r.start) }).collect(),
    });
}

/// `sink_release_kernel`: warp 0 lane 0 writes `data`, then one
/// `st.{release|relaxed}.{gpu|cta}.v<width>.u<bits>` to the flag (with
/// `sinks`, only elements 0 and 1 are written); warp 1 lane 0 spins with
/// `ld.acquire` on element 1 and reads `data`.
fn g6_store_sinks(bits: u64, release: bool, sinks: bool, shared: bool) -> Report {
    let e = bits / 8;
    let width = if shared { 2 } else { 32 / e };
    let written = if sinks { 2 } else { width };
    let (flag, scope) = if shared { (SMEM, Scope::Cta) } else { (GMEM2, Scope::Gpu) };
    let mut k = K::new(2, 1, 1);
    if shared {
        let lanes = g6_lanes(2);
        k.inst(0, &lanes, PLAIN_ST, |l| (SMEM, u64::from(l) * e..u64::from(l) * e + e));
        k.bar(0, &[0, 1]);
    }
    k.st(0, 0, GMEM, 0..4);
    let order = if release { MemOrder::Release } else { MemOrder::Relaxed };
    g6_vec_store(&mut k, 0, st(order, scope), flag, &g6_elems(0, 0..written * e, e));
    for _ in 0..3 {
        k.a(1, 0, ld(MemOrder::Acquire, scope), flag, e..2 * e);
    }
    k.ld(1, 0, GMEM, 0..4).st(1, 0, GMEM, 64..68);
    k.run()
}

/// runtime/test_store_sinks.py::test_store_sinks_retain_per_element_release
/// [bits 32|64 x (release, sinks, space) = (T,F,global) (T,T,global)
/// (F,T,global) (T,F,shared)]: the release covers element 1, so the spin
/// acquires `data` (no race); a relaxed vector store gives no edge (race).
/// delta R3: the raw (undeclared) acquire spin over a morally strong store
/// is a `review` `UndeclaredProtocolWord` advisory (legacy verdict clean).
#[test]
fn g6_store_sinks_retain_per_element_release() {
    for bits in [32u64, 64] {
        for (release, sinks, shared) in [(true, false, false), (true, true, false), (false, true, false), (true, false, true)] {
            let r = g6_store_sinks(bits, release, sinks, shared);
            let tag = format!("bits={bits} release={release} sinks={sinks} shared={shared}");
            if release {
                assert!(clean(&r), "{tag}: {:?} {:?}", r.findings, r.incomplete);
                assert!(r.findings.iter().all(|f| f.kind == FindingKind::Advisory { kind: AdvisoryKind::UndeclaredProtocolWord }), "{tag}: {r:?}");
            } else {
                assert!(has_class(&r, RaceClass::WriteRead), "{tag}: {r:?}");
                assert!(races(&r).iter().all(|f| f.alloc == GMEM), "{tag}: {r:?}");
            }
        }
    }
}

// ------------------------------------------------ runtime/test_tensormap_publication.py --

/// `tensor_map_publication_case`: lane 0 of warp 0 rewrites the 128-byte
/// descriptor; lane 1 (of warp 0, or warp 1 when `cross_warp`)
/// `fence.proxy.tensormap::generic.release.gpu`; lane 0 acquires and issues
/// a TMA through it, then polls and reads the destination. `ordering`:
/// "before" = update, sync, release, sync, acquire (the control);
/// "after" = the release is not ordered after the update;
/// "stale" = the acquire precedes the update.
fn g6_tensormap(ordering: &str, cross_warp: bool) -> Report {
    let mut k = if cross_warp { K::new(2, 1, 1) } else { K::one_warp() };
    let sync = |k: &mut K| {
        if cross_warp {
            k.bar(0, &[0, 1]);
        } else {
            k.syncwarp(0, G6_ALL);
        }
    };
    let acquire = FenceKind::TensormapAcquire { scope: Scope::Gpu, alloc: GMEM2, span: ByteSpan::new(0, 128) };
    if ordering == "stale" {
        k.fence(0, 1, acquire);
    }
    sync(&mut k);
    k.st(0, 0, GMEM2, 0..128);
    if ordering != "after" {
        sync(&mut k);
    }
    let releaser = u32::from(cross_warp);
    k.fence(releaser, 1 << 1, FenceKind::TensormapRelease { scope: Scope::Gpu });
    sync(&mut k);
    if ordering != "stale" {
        k.fence(0, 1, acquire);
    }
    k.arrive(0, 1, 0, 0, true);
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::TensorMap, GMEM2, 0..128).ar(op, Proxy::Async, GMEM, 0..16).aw(op, Proxy::Async, SMEM, 0..16);
    k.done_phase(op, Milestone::Write, 0, 0).wait(0, 1, 0, 0, true);
    k.ld(0, 0, SMEM, 0..16).st(0, 0, GMEM, 64..80);
    k.run()
}

/// runtime/test_tensormap_publication.py::test_descriptor_publication_does_not_capture_future_writes
/// [replace|bytes|helper x (after, same warp) (after, cross warp) (stale)]:
/// the three update spellings are the same generic descriptor writes at
/// the contract. A release not ordered after the update, or an acquire that
/// precedes it, leaves the TMA's tensormap-proxy read unbridged
/// (`missing_proxy_bridge`); legacy "not acquired"/"dirty". The ordered
/// control is clean.
#[test]
fn g6_descriptor_publication_does_not_capture_future_writes() {
    for cross_warp in [false, true] {
        let r = g6_tensormap("before", cross_warp);
        assert!(clean(&r) && r.findings.is_empty(), "before cross={cross_warp}: {:?} {:?}", r.findings, r.incomplete);
    }
    for (ordering, cross_warp) in [("after", false), ("after", true), ("stale", false)] {
        let r = g6_tensormap(ordering, cross_warp);
        assert!(
            has_failure(&r, |f| matches!(f, OrderingFailure::MissingProxyBridge { current: Proxy::TensorMap, .. })),
            "{ordering} cross={cross_warp}: {r:?}"
        );
    }
}

// --------------------------- runtime/test_tma_multiissuer.py, test_tma_im2col_multiissuer.py --

/// Every lane fills its 128-byte smem row, `fence.proxy.async`, sync, arms
/// `barriers[lane]`, sync; every lane issues its own 32-byte TMA into its
/// row (or, with `overlap`, into row 0), completing on `barriers[lane]`;
/// with `wait` every lane polls its own barrier; sync; every lane reads its
/// row.
fn g6_tma_multiissuer(wait: bool, overlap: bool) -> Report {
    let mut k = K::one_warp();
    let row = |l: u8| (SMEM, u64::from(l) * 128..u64::from(l) * 128 + 128);
    k.inst(0, &G5_ALL, PLAIN_ST, row);
    k.fence(0, G6_ALL, FenceKind::ProxyAsync(Some(Domain::SharedCta))).bar(0, &[0]);
    for l in 0..32u32 {
        k.arrive(0, 1 << l, l, 0, true);
    }
    k.bar(0, &[0]);
    for l in 0..32u8 {
        let dst = if overlap { 0..32 } else { u64::from(l) * 128..u64::from(l) * 128 + 32 };
        let op = k.issue(0, l, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, dst.clone())]);
        k.ar(op, Proxy::Async, GMEM, u64::from(l) * 32..u64::from(l) * 32 + 32).aw(op, Proxy::Async, SMEM, dst);
        k.done_phase(op, Milestone::Write, u32::from(l), 0);
    }
    if wait {
        for l in 0..32u32 {
            k.wait(0, 1 << l, l, 0, true);
        }
    }
    k.bar(0, &[0]);
    k.inst(0, &G5_ALL, PLAIN_LD, row);
    k.run()
}

/// runtime/test_tma_multiissuer.py::test_tma_multiissuer_requires_wait_and_disjoint_destinations
/// [(wait=False, overlap=False) -> read_write/write_read,
///  (wait=True, overlap=True) -> write_write]; the waited, disjoint control
/// is clean.
#[test]
fn g6_tma_multiissuer_requires_wait_and_disjoint_destinations() {
    let r = g6_tma_multiissuer(true, false);
    assert!(clean(&r) && r.findings.is_empty(), "{:?} {:?}", r.findings, r.incomplete);
    let r = g6_tma_multiissuer(false, false);
    assert!(has_class(&r, RaceClass::WriteRead) || has_class(&r, RaceClass::ReadWrite), "{r:?}");
    let r = g6_tma_multiissuer(true, true);
    assert!(has_class(&r, RaceClass::WriteWrite), "{r:?}");
}

/// runtime/test_tma_im2col_multiissuer.py::test_im2col_multiissuer_requires_completion_wait
/// (rank 3, `wait=False`): per-lane im2col TMAs into disjoint rows, read
/// without waiting: `write_read`/`read_write`. (The im2col coordinates and
/// offsets only select the source bytes.)
#[test]
fn g6_im2col_multiissuer_requires_completion_wait() {
    let r = g6_tma_multiissuer(false, false);
    assert!(has_class(&r, RaceClass::WriteRead) || has_class(&r, RaceClass::ReadWrite), "{r:?}");
    assert!(races(&r).iter().all(|f| f.alloc == SMEM));
}

// ------------------------------------------------------- runtime/test_wait_until.py --

/// The poll of a `wait_until`: a strong read of the word (modelled relaxed,
/// so the only edge is the `WaitVerdicts` one) plus the verdicts.
fn g6_wait_until(k: &mut K, w: WarpId, alloc: AllocId, r: std::ops::Range<u64>, accepted: u64, observed: u32) {
    k.a(w, 0, ld(MemOrder::Relaxed, Scope::Gpu), alloc, r.clone()).wait_until(w, 0, alloc, r, Scope::Gpu, accepted, observed);
}

/// `wait_event`: `state` is a declared word whose launch value (7) already
/// satisfies the waiter's predicate. The waiter (warp `waiter`, lane 0,
/// optionally after an unrelated write) `wait_until`s it; the other warp
/// reads `state` and plainly stores 7 into it. `ordering`: a `bar.sync`
/// between the wait and the plain access (either order), only before both
/// ("prefix_only"), or none.
fn g6_wait_event(waiter: WarpId, prior_access: bool, ordering: &str) -> Report {
    let mut k = K::new(2, 1, 1);
    k.declare(GMEM2, 0..4);
    let other = 1 - waiter;
    if prior_access {
        k.st(waiter, 0, GMEM, 0..4);
    }
    let wait = |k: &mut K| {
        g6_wait_until(k, waiter, GMEM2, 0..4, 0b1, 0);
    };
    let plain = |k: &mut K| {
        k.ld(other, 0, GMEM2, 0..4).st(other, 0, GMEM, 4..8).st(other, 0, GMEM2, 0..4);
    };
    match ordering {
        "wait_then_plain" => {
            wait(&mut k);
            k.bar(0, &[0, 1]);
            plain(&mut k);
        }
        "plain_then_wait" => {
            plain(&mut k);
            k.bar(0, &[0, 1]);
            wait(&mut k);
        }
        "prefix_only" => {
            k.bar(0, &[0, 1]);
            wait(&mut k);
            plain(&mut k);
        }
        _ => {
            wait(&mut k);
            plain(&mut k);
        }
    }
    k.run()
}

/// runtime/test_wait_until.py::test_wait_has_its_own_hb_event
/// [waiter 0|1 x prior_access F|T x ordering] (verdict half): only a
/// rendezvous between the wait and the plain access orders the pair; the
/// unordered plain store to the declared word is an error.
#[test]
fn g6_wait_has_its_own_hb_event() {
    for waiter in [0, 1] {
        for prior_access in [false, true] {
            for ordering in ["wait_then_plain", "plain_then_wait", "prefix_only", "none"] {
                let r = g6_wait_event(waiter, prior_access, ordering);
                let tag = format!("waiter={waiter} prior={prior_access} {ordering}");
                if matches!(ordering, "wait_then_plain" | "plain_then_wait") {
                    assert!(clean(&r) && r.findings.is_empty(), "{tag}: {:?} {:?}", r.findings, r.incomplete);
                } else {
                    assert!(r.errors().next().is_some(), "{tag}: {r:?}");
                    assert!(r.errors().all(|f| f.alloc == GMEM2), "{tag}: {r:?}");
                }
            }
        }
    }
}

/// runtime/test_wait_until.py::test_wait_has_its_own_hb_event (finding-kind
/// half): legacy reported the unordered plain access to a declared word as
/// exactly `{signal_protocol_error}` (a declared-word bypass), never a data
/// race. The new core has no bypass kind: the plain store races the wait's
/// strong read as an ordinary `data_race`.
#[test]
fn g6_wait_bypass_is_a_signal_protocol_error_not_a_race() {
    for ordering in ["prefix_only", "none"] {
        let r = g6_wait_event(0, false, ordering);
        assert!(r.errors().next().is_some() && !has_race(&r), "{ordering}: {r:?}");
    }
}

/// `two_arrivals`: warp 0 writes `first` and `red.release.gpu.add`s the
/// declared counter; warp 1 writes `second` and adds; warp 2 waits for
/// `state >= target` and reads `second`. History: entry 1 = warp 0's add,
/// entry 2 = warp 1's add.
fn g6_two_arrivals(target: u32, acquire_poll: bool) -> Report {
    let mut k = K::new(3, 1, 1);
    k.declare(GMEM2, 0..4);
    k.st(0, 0, GMEM, 0..4).red(0, 0, MemOrder::Release, Scope::Gpu, GMEM2, 0..4);
    k.st(1, 0, GMEM, 4..8).red(1, 0, MemOrder::Release, Scope::Gpu, GMEM2, 0..4);
    let accepted = if target == 2 { 0b100 } else { 0b110 };
    if acquire_poll {
        k.a(2, 0, ld(MemOrder::Acquire, Scope::Gpu), GMEM2, 0..4).wait_until(2, 0, GMEM2, 0..4, Scope::Gpu, accepted, 2);
    } else {
        g6_wait_until(&mut k, 2, GMEM2, 0..4, accepted, 2);
    }
    k.ld(2, 0, GMEM, 4..8).st(2, 0, GMEM, 64..68);
    k.run()
}

/// runtime/test_wait_until.py::test_a_wait_that_waits_for_both_arrivals_may_read_both
#[test]
fn g6_wait_for_both_arrivals_may_read_both() {
    let r = g6_two_arrivals(2, false);
    assert!(clean(&r) && r.findings.is_empty(), "{:?} {:?}", r.findings, r.incomplete);
}

/// runtime/test_wait_until.py::test_a_wait_that_waits_for_one_arrival_may_not_read_the_other
/// The earliest accepted write (warp 0's) orders nothing warp 1 published:
/// the read of `second` races warp 1's store.
#[test]
fn g6_wait_for_one_arrival_may_not_read_the_other() {
    let r = g6_two_arrivals(1, false);
    let f = races(&r);
    assert!(!f.is_empty(), "{r:?}");
    assert!(f.iter().all(|f| f.alloc == GMEM && f.bytes == (4..8)), "{r:?}");
    assert!(f.iter().any(|f| f.prior.as_ref().unwrap().warp == 1 && f.current.as_ref().unwrap().warp == 2));
}

/// runtime/test_wait_until.py::test_a_wait_that_waits_for_one_arrival_may_not_read_the_other,
/// with the poll as the interpreter emits it (`sync.rs::wait_until`: an
/// `Access` read with the wait's `.acquire` sem before `WaitVerdicts`). The
/// ordinary read-from rule (racecheck-semantics §3 row 24) then hands the
/// waiter the latest morally strong write (warp 1's), so the race on
/// `second` disappears on this schedule: exactly the run-dependent edge
/// W1 / §5 ("earliest accepted, schedule independent") rule out.
#[test]
fn g6_wait_for_one_arrival_with_acquiring_poll_still_races() {
    let r = g6_two_arrivals(1, true);
    assert!(races(&r).iter().any(|f| f.alloc == GMEM && f.bytes == (4..8)), "{r:?}");
}

/// `woken_by_a_bypass`: warp 0 plainly stores 7 into the declared word;
/// warp 1's `wait_until(!= 0)` exits on that write and stores what it saw.
fn g6_woken_by_plain_write() -> Report {
    let mut k = K::new(2, 1, 1);
    k.declare(GMEM2, 0..4);
    k.st(0, 0, GMEM2, 0..4);
    g6_wait_until(&mut k, 1, GMEM2, 0..4, 0b10, 1);
    k.st(1, 0, GMEM, 0..4);
    k.run()
}

/// runtime/test_wait_until.py::test_a_wait_woken_by_a_plain_write_reports_the_missing_edge
/// The plain store races the wait's strong read; the wait itself builds no
/// edge from a plain write.
#[test]
fn g6_wait_woken_by_a_plain_write_is_an_error() {
    let r = g6_woken_by_plain_write();
    // The plain write races the poll on a declared word: a protocol bypass
    // (signal_protocol_error, an error with the write_read pair).
    assert!(
        r.findings.iter().any(|f| matches!(f.kind, FindingKind::SignalProtocolError { class: RaceClass::WriteRead, .. })),
        "{r:?}"
    );
}

/// runtime/test_wait_until.py::test_a_wait_woken_by_a_plain_write_reports_the_missing_edge
/// (incomplete half): legacy reported `analysis_incomplete` for the wait
/// exit explained only by a plain write; delta W2 also says such an exit is
/// `WaitExitUnproven` incomplete. The core instead accepts the plain
/// history entry with no edge and reports no incomplete.
#[test]
fn g6_wait_woken_by_a_plain_write_is_incomplete() {
    let r = g6_woken_by_plain_write();
    assert!(r.incomplete.iter().any(|i| matches!(i, Incomplete::WaitExitUnproven { .. })), "{r:?}");
}

/// runtime/test_wait_until.py::test_a_wait_on_a_word_that_carries_its_own_payload_is_clean
/// (`_packed_payload(retry=True)`): CTA 0 `st.release.gpu.u64` the declared
/// slot; CTA 1 (another cluster) `wait_until`s it and stores the payload
/// half. Clean.
#[test]
fn g6_wait_on_a_word_that_carries_its_own_payload_is_clean() {
    let mut k = K::new(1, 1, 2);
    k.declare(GMEM2, 0..8);
    k.a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, 0..8);
    g6_wait_until(&mut k, 1, GMEM2, 0..8, 0b10, 1);
    k.st(1, 0, GMEM, 0..8);
    let r = k.run();
    assert!(clean(&r) && r.findings.is_empty(), "{:?} {:?}", r.findings, r.incomplete);
}
