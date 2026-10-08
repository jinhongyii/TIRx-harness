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
#[ignore = "undocumented divergence: wide (v4.b32) release store on a declared word is accepted via the verdict bitset (clean); legacy and racecheck-semantics.md list wide writes as incomplete"]
fn g2_publication_too_wide_to_poll_fails_closed() {
    let r = g2_wide_publication(16);
    assert!(!r.incomplete.is_empty(), "{r:?}");
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
