//! Phase-3 ports: view-aware GC, async-slot reclaim, TensorMap, per-lane
//! tcgen state, `red` vs `atom`, wide witnesses, and the report / payload
//! mapping.
#[path = "racecheck_common/mod.rs"]
mod common;
use common::*;
use numsim_core::arena::Space;
use numsim_core::racecheck::{report, serialize};
use numsim_core::report::{FindingKind as CK, Status};
use numsim_core::sync::completion::ResourceId;

/// The cross-proxy pair survives GC: the generic write is observed by every
/// actor through `hb` but not through the generic→async bridge, so the
/// collector must keep it (legacy G's proxy-blind floor GC dropped it).
#[test]
fn gc_keeps_cross_proxy_witnesses() {
    let mut k = K::new(2, 1, 1);
    k.gc_every = 8;
    k.st(0, 0, SMEM, 0..16);
    for _ in 0..64 {
        k.bar(0, &[0, 1]).st(1, 0, SMEM, 1024..1028);
    }
    let op = k.issue(1, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, SMEM, 0..16).aw(op, Proxy::Async, GMEM, 0..16).done_warp(op, Milestone::Write, 1, 1);
    let obs = k.observe();
    let lr = &obs.launches[0];
    assert!(lr.stats.gc_runs > 4);
    assert!(lr.stats.witnesses_retired > 0, "{:?}", lr.stats);
    assert!(has_failure(&lr.report, |f| matches!(f, OrderingFailure::MissingProxyBridge { .. })));
}

/// Fully ordered history is retired and does not grow the shadow.
#[test]
fn gc_retires_ordered_history() {
    let mut k = K::new(2, 1, 1);
    k.gc_every = 16;
    for i in 0..256u64 {
        let off = (i % 64) * 4;
        k.st(0, 0, SMEM, off..off + 4).bar(0, &[0, 1]).ld(1, 0, SMEM, off..off + 4).bar(1, &[0, 1]);
    }
    let obs = k.observe();
    let lr = &obs.launches[0];
    assert!(clean(&lr.report));
    assert!(lr.stats.witnesses_retired >= 256, "{:?}", lr.stats);
}

/// Completed async ops' actor slots are recycled; a reused slot's new
/// generation is not observed by clocks that saw the old one.
#[test]
fn async_slots_are_reclaimed_and_generations_stay_distinct() {
    let mut k = K::one_warp();
    k.gc_every = 4;
    for i in 0..200u64 {
        let off = (i % 8) * 16;
        let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Generic, &[], &[(SMEM, off..off + 16)]);
        k.ar(op, Proxy::Generic, GMEM, off..off + 16).aw(op, Proxy::Generic, SMEM, off..off + 16);
        k.done_warp(op, Milestone::Write, 0, 1).ld(0, 0, SMEM, off..off + 16).syncwarp(0, u32::MAX);
    }
    // A new op in a recycled slot, not waited: the read must race.
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Generic, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Generic, GMEM, 0..16).aw(op, Proxy::Generic, SMEM, 0..16);
    k.ld(0, 1, SMEM, 0..16).done_warp(op, Milestone::Write, 0, 1);
    let obs = k.observe();
    let lr = &obs.launches[0];
    assert!(lr.stats.async_slots_reclaimed >= 50, "{:?}", lr.stats);
    assert!(lr.stats.async_slots < 32, "{:?}", lr.stats);
    assert!(has_class(&lr.report, RaceClass::WriteRead), "{:?}", lr.report);
    assert_eq!(races(&lr.report).len(), 1);
}

/// TensorMap: descriptor bytes written generically are read by a TMA through
/// the tensormap proxy; release fence (writer) + acquire fence (issuer).
fn tensormap(release: bool, acquire: bool) -> Report {
    let mut k = K::new(2, 1, 1);
    k.st(0, 0, GMEM2, 0..128);
    if release {
        k.fence(0, 1, FenceKind::TensormapRelease { scope: Scope::Cta });
    }
    k.bar(0, &[0, 1]);
    if acquire {
        k.fence(1, 1, FenceKind::TensormapAcquire { scope: Scope::Cta, alloc: GMEM2, span: numsim_core::arena::ByteSpan::new(0, 128) });
    }
    let op = k.issue(1, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::TensorMap, GMEM2, 0..128).ar(op, Proxy::Async, GMEM, 0..16).aw(op, Proxy::Async, SMEM, 0..16);
    k.done_warp(op, Milestone::Write, 1, 1);
    k.run()
}

#[test]
fn tensormap_release_acquire() {
    assert!(clean(&tensormap(true, true)));
    assert!(has_failure(&tensormap(true, false), |f| matches!(f, OrderingFailure::MissingProxyBridge { current: Proxy::TensorMap, .. })));
    assert!(has_race(&tensormap(false, true)));
}

/// tcgen state is per lane: a before_thread_sync by another lane does not
/// publish lane 0's issued cp.
fn tcgen_lane(fence_lane: u8) -> Report {
    let mut k = K::new(2, 1, 1);
    let cp = k.issue(0, 0, AsyncKind::TcgenPipelined, Proxy::Tcgen, &[], &[(TMEM, 0..4096)]);
    k.aacc(cp, Milestone::Write, AccessKind::Write, Proxy::Tcgen, TMEM, 0..16);
    k.fence(0, 1 << fence_lane, FenceKind::TcgenBefore).bar(0, &[0, 1]).fence(1, 1, FenceKind::TcgenAfter);
    let mma = k.issue(1, 0, AsyncKind::TcgenPipelined, Proxy::Tcgen, &[], &[(TMEM, 0..4096)]);
    k.aacc(mma, Milestone::Write, AccessKind::Write, Proxy::Tcgen, TMEM, 0..32);
    let c0 = k.issue(0, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &[cp], &[]);
    let c1 = k.issue(1, 0, AsyncKind::TcgenCommit, Proxy::Tcgen, &[mma], &[]);
    k.done_phase(c0, Milestone::Write, 1, 0).done_phase(c1, Milestone::Write, 2, 0);
    k.run()
}

#[test]
fn tcgen_fences_are_per_thread() {
    assert!(clean(&tcgen_lane(0)));
    assert!(has_race(&tcgen_lane(1)));
}

/// `red` never forms an acquire pattern, even followed by an acquire fence
/// (PTX §8.8); `atom.relaxed` + `fence.acq_rel` does.
fn red_vs_atom(red: bool) -> Report {
    let mut k = K::new(1, 1, 2);
    k.st(0, 0, GMEM, 0..4).a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, 0..4);
    if red {
        k.red(1, 0, MemOrder::Relaxed, Scope::Gpu, GMEM2, 0..4);
    } else {
        k.a(1, 0, atom(MemOrder::Relaxed, Scope::Gpu), GMEM2, 0..4);
    }
    k.fence(1, 1, FenceKind::AcqRel(Scope::Gpu)).ld(1, 0, GMEM, 0..4);
    k.run()
}

#[test]
fn red_never_acquires() {
    assert!(clean(&red_vs_atom(false)));
    assert!(has_class(&red_vs_atom(true), RaceClass::WriteRead));
}

/// Spans beyond the compact witness encoding (offset ≥ 4 GiB, length ≥
/// 32 KiB) keep exact evidence.
#[test]
fn wide_witness_spans() {
    let mut k = K::new(1, 1, 2);
    let big = numsim_core::arena::AllocId(42);
    k.alloc(big, Space::Global, 1 << 40);
    let s = (1u64 << 33) + 7;
    k.st(0, 0, big, s..s + (1 << 20));
    k.ld(1, 0, big, s + 100..s + 104);
    let r = k.run();
    let f = races(&r);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].prior.as_ref().unwrap().span, s..s + (1 << 20));
    assert_eq!(f[0].bytes, s + 100..s + 104);
}

#[test]
fn unfinished_async_work_is_incomplete_at_end_launch() {
    let mut k = K::one_warp();
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, GMEM, 0..16);
    let r = k.run();
    assert!(r.incomplete.iter().any(|i| matches!(i, Incomplete::AsyncNeverCompleted { .. })));
}

/// `report()` maps onto the contract taxonomy with evidence, and
/// `serialize()` emits the legacy payload keys the Python tests pin.
#[test]
fn report_and_payload_mapping() {
    let mut k = K::one_warp();
    k.st(0, 0, GMEM, 0..4);
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, GMEM, 0..16).aw(op, Proxy::Async, SMEM, 0..16);
    k.done_phase(op, Milestone::Write, 7, 0).wait(0, 1, 7, 0, true).ld(0, 0, SMEM, 0..4);
    let op2 = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 32..48)]);
    k.ar(op2, Proxy::Async, GMEM, 32..48);
    let obs = k.observe();
    let rep = report(&obs);
    assert_eq!(rep.tool, "racecheck");
    assert_eq!(rep.verdict, numsim_core::report::Verdict::Error);
    let race = rep.findings.iter().find(|f| f.kind == CK::ProxyRace).expect("proxy race");
    assert_eq!(race.status, Status::Error);
    assert_eq!(race.sites.len(), 2);
    let prior = race.evidence.iter().find(|e| e.role == "prior").unwrap();
    assert_eq!(prior.space, Some(Space::Global));
    assert_eq!(prior.alloc, Some(GMEM));
    assert_eq!(prior.kernel, 0);
    assert!(rep.findings.iter().any(|f| f.status == Status::Incomplete));

    let p = serialize(&rep);
    assert_eq!(p["accesses_complete"], true);
    assert_eq!(p["access_count"], 5);
    let f = &p["findings"][0];
    assert_eq!(f["kind"], "data_race");
    assert_eq!(f["status"], "error");
    assert_eq!(f["access_pair"], "write_read");
    assert_eq!(f["ordering_domain"], "memory");
    assert_eq!(f["ordering_failure"], "missing_proxy_bridge");
    assert_eq!(f["proxy_bridge"]["prior_proxy"], "generic");
    assert_eq!(f["proxy_bridge"]["current_proxy"], "async");
    assert_eq!(f["proxy_bridge"]["prior_domain"], "global");
    assert_eq!(f["prior"]["access_kind"], "write");
    assert_eq!(f["current"]["access_kind"], "read");
    assert_eq!(f["prior"]["space"], "global");
    assert_eq!(f["overlap"]["byte_offset"], 0);
    assert_eq!(f["overlap"]["byte_len"], 4);
    assert_eq!(f["overlap"]["allocation_id"], GMEM.0);
    assert_eq!(p["incomplete"][0]["kind"], "analysis_incomplete");
    assert_eq!(p["incomplete"][0]["reason"], "effect_commit_unobserved");
}

#[test]
fn hint_and_review_payloads() {
    // missing_release_acquire carries the wait_until hint.
    let mut k = K::new(2, 1, 1);
    k.a(0, 0, atom(MemOrder::Relaxed, Scope::Gpu), SMEM, 0..4).st(1, 0, SMEM, 0..4);
    let p = serialize(&report(&k.observe()));
    assert_eq!(p["findings"][0]["ordering_failure"], "missing_release_acquire");
    assert!(p["findings"][0]["hint"].as_str().unwrap().contains("wait_until"));
    // An unwaited tcgen05.ld conflict is a review finding.
    let mut k = K::one_warp();
    let l = k.issue(0, 0, AsyncKind::TcgenLd, Proxy::Tcgen, &[], &[(TMEM, 0..4096)]);
    k.aacc(l, Milestone::Read, AccessKind::Read, Proxy::Tcgen, TMEM, 0..16);
    let s = k.issue(0, 0, AsyncKind::TcgenSt, Proxy::Tcgen, &[], &[(TMEM, 0..4096)]);
    k.aacc(s, Milestone::Write, AccessKind::Write, Proxy::Tcgen, TMEM, 0..16);
    k.done_warp(s, Milestone::Write, 0, 1).done_warp(l, Milestone::Write, 0, 1);
    let rep = report(&k.observe());
    assert_eq!(rep.verdict, numsim_core::report::Verdict::Review);
    let p = serialize(&rep);
    assert_eq!(p["findings"][0]["kind"], "tmem_lifetime_review");
    assert_eq!(p["findings"][0]["status"], "review");
    assert_eq!(p["findings"][0]["access_pair"], "read_write");
}

/// Contract review item 5: a per-thread async op issued by several lanes is
/// one virtual actor per lane; a lane's wait publishes only its own copy.
#[test]
fn per_lane_async_ops_are_not_merged() {
    let mut k = K::one_warp();
    let op = k.issue_lanes(0, 0b11, AsyncKind::Copy, Proxy::Generic, &[(SMEM, 0..32)]);
    k.aacc_lanes(op, Milestone::Read, AccessKind::Read, Proxy::Generic, GMEM, &[(0, 0..16), (1, 16..32)]);
    k.aacc_lanes(op, Milestone::Write, AccessKind::Write, Proxy::Generic, SMEM, &[(0, 0..16), (1, 16..32)]);
    k.done_warp(op, Milestone::Write, 0, 0b01);
    k.ld(0, 0, SMEM, 0..16); // own copy: ordered
    k.ld(0, 0, SMEM, 16..32); // lane 1's copy: lane 1 has not waited
    let r = k.run();
    let f = races(&r);
    assert_eq!(f.len(), 1, "{r:?}");
    assert_eq!(f[0].bytes, 16..32);
    // Spans that do not name the lane of a multi-lane op fail closed.
    let mut k = K::one_warp();
    let op = k.issue_lanes(0, 0b11, AsyncKind::Copy, Proxy::Generic, &[(SMEM, 0..32)]);
    k.aw(op, Proxy::Generic, SMEM, 0..32).done_warp(op, Milestone::Write, 0, 0b11);
    let r = k.run();
    assert!(r.incomplete.iter().any(|i| matches!(i, Incomplete::AsyncLaneUnknown { .. })));
}

/// WaitVerdicts are per lane group: lanes that accepted different writes get
/// different edges (never a lane-wise conjunction).
#[test]
fn wait_verdicts_per_lane_group() {
    let mut k = K::new(1, 1, 3);
    k.declare(GMEM2, 0..4);
    k.st(0, 0, GMEM, 0..4);
    k.a(2, 0, st(MemOrder::Relaxed, Scope::Gpu), GMEM2, 0..4); // entry 1: no release
    k.a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, 0..4); // entry 2: release
    k.wait_until_groups(1, 0b11, GMEM2, 0..4, Scope::Gpu, &[(0b01, 0b110, 1), (0b10, 0b100, 2)]);
    k.ld(1, 1, GMEM, 0..4);
    assert!(clean(&k.run()));
    let mut k = K::new(1, 1, 3);
    k.declare(GMEM2, 0..4);
    k.st(0, 0, GMEM, 0..4);
    k.a(2, 0, st(MemOrder::Relaxed, Scope::Gpu), GMEM2, 0..4);
    k.a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, 0..4);
    k.wait_until_groups(1, 0b11, GMEM2, 0..4, Scope::Gpu, &[(0b01, 0b110, 1), (0b10, 0b100, 2)]);
    k.ld(1, 0, GMEM, 0..4);
    assert!(has_race(&k.run()));
}

/// Tensormap acquire is limited to its address range and to releasers whose
/// scope mutually includes the acquirer.
#[test]
fn tensormap_acquire_range_and_scope() {
    let run = |rel: Scope, acq: Scope, span: (u64, u64), ctas: u32| {
        let mut k = K::new(1, 1, ctas);
        let consumer = ctas - 1;
        k.st(0, 0, GMEM2, 0..128).fence(0, 1, FenceKind::TensormapRelease { scope: rel });
        let c = ResourceId::Cluster { cluster: 0 };
        // A cross-CTA handoff through a gpu-scope release/acquire flag.
        k.a(0, 0, st(MemOrder::Release, Scope::Gpu), GMEM, 0..4);
        k.a(consumer, 0, ld(MemOrder::Acquire, Scope::Gpu), GMEM, 0..4);
        let _ = c;
        k.fence(consumer, 1, FenceKind::TensormapAcquire { scope: acq, alloc: GMEM2, span: numsim_core::arena::ByteSpan::new(span.0, span.1) });
        let op = k.issue(consumer, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
        k.ar(op, Proxy::TensorMap, GMEM2, 0..128).aw(op, Proxy::Async, SMEM, 0..16).done_warp(op, Milestone::Write, consumer, 1);
        k.run()
    };
    assert!(errors_free(&run(Scope::Gpu, Scope::Gpu, (0, 128), 2)));
    assert!(has_race(&run(Scope::Cta, Scope::Gpu, (0, 128), 2)), "release.cta does not reach another CTA");
    assert!(has_race(&run(Scope::Gpu, Scope::Gpu, (0, 64), 2)), "bytes outside the acquired range");
    assert!(errors_free(&run(Scope::Cta, Scope::Cta, (0, 128), 1)));
}

fn errors_free(r: &Report) -> bool {
    r.errors().next().is_none() && r.incomplete.is_empty()
}

#[test]
fn new_kinds_and_attrs() {
    // scope mismatch → FindingKind::ScopeMismatch with attrs
    let mut k = K::new(1, 1, 2);
    k.declare(GMEM2, 0..4);
    k.st(0, 0, GMEM, 0..4).a(0, 0, st(MemOrder::Release, Scope::Cta), GMEM2, 0..4);
    k.wait_until(1, 0, GMEM2, 0..4, Scope::Gpu, 0b10, 1).ld(1, 0, GMEM, 0..4);
    let rep = report(&k.observe());
    let sm = rep.findings.iter().find(|f| f.kind == CK::ScopeMismatch).expect("scope mismatch");
    assert_eq!(sm.attrs["release_scope"], "cta");
    let race = rep.findings.iter().find(|f| f.kind == CK::DataRace).unwrap();
    assert_eq!(race.attrs["access_pair"], "write_read");
    assert_eq!(race.attrs["prior"]["access_kind"], "write");
    // advisory kinds
    let mut k = K::new(1, 1, 2);
    k.a(0, 0, st(MemOrder::Relaxed, Scope::Gpu), GMEM, 0..4).a(1, 0, ld(MemOrder::Relaxed, Scope::Gpu), GMEM, 0..4);
    let rep = report(&k.observe());
    assert!(rep.findings.iter().any(|f| f.kind == CK::UndeclaredProtocolWord && f.status == Status::Review));
    let p = serialize(&rep);
    assert_eq!(p["advisories"][0]["kind"], "undeclared_protocol_word");
    // async lifetime
    let mut k = K::one_warp();
    let op = k.issue(0, 0, AsyncKind::Copy, Proxy::Async, &[], &[(SMEM, 0..16)]);
    k.ar(op, Proxy::Async, GMEM, 0..16).alloc_end(SMEM);
    let rep = report(&k.observe());
    assert!(rep.findings.iter().any(|f| f.kind == CK::AsyncLifetime));
}
