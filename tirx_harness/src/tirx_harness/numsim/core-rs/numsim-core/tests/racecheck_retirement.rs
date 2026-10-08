//! W6 guards for racecheck state retirement (W5's design for mega_moe
//! medium: a per-allocation read summary for allocations never written since
//! launch start, async-slot reclaim once a slot's witnesses are only
//! summarised reads, and possibly a reachability-scoped GC meet).
//!
//! Every scenario is an engine run under the serial `RaceObserver`. The
//! reads mirror mega_moe's weight loads: per-CTA bulk (TMA-like)
//! global->shared copies of a never-written `w` completing on an mbarrier
//! (or plain lane reads). Each test pins the verdict, the race anchors
//! (sites and overlapping bytes) and worker independence at 1/8/32.
use numsim_core::dtype::{Dtype, Ty};
use numsim_core::observe::NoopObserver;
use numsim_core::program::*;
use numsim_core::racecheck::observer::{RaceObserver, RacecheckConfig};
use numsim_core::report::{Finding, FindingKind, Report, Verdict};
use numsim_core::sched::{self, RunConfig, RunStatus};
use numsim_core::site::SiteId;
use numsim_core::sync::async_group::Domain;
use numsim_core::testutil::scenarios::{inputs, u32_buf, Scenario};
use numsim_core::testutil::ProgramBuilder;
use numsim_core::Module;

/// Words of `w` every reader reads (128 bytes).
const W_WORDS: u32 = 32;
/// The writer overwrites words 8..12 (bytes 32..48) of `w`.
const WRITE_LO: u32 = 8;
const WRITE_HI: u32 = 12;

fn run(s: &Scenario, workers: usize) -> (RunStatus, Report) {
    run_gc(s, workers, None)
}

/// `gc_every`: `None` = the default period; `Some(1)` = collect (and reclaim
/// async slots) at every phase end, so retirement runs between the reads and
/// the later write.
fn run_gc(s: &Scenario, workers: usize, gc_every: Option<u64>) -> (RunStatus, Report) {
    let cfg = RunConfig { workers, ..s.config.clone() };
    let mut obs = RaceObserver::new(RacecheckConfig::default());
    obs.fork_join = false;
    if let Some(g) = gc_every {
        obs.gc_every = g;
    }
    let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &cfg).expect("run starts");
    (o.status, obs.finish())
}

fn coverage(r: &Report, key: &str) -> u64 {
    r.coverage.iter().find(|(k, _)| k == key).map_or(0, |(_, v)| *v)
}

/// The payload without the collector's own counters (which depend on when it
/// runs, by construction; see racecheck_parallel_review D7).
fn without_gc_counters(mut r: Report) -> String {
    r.coverage.retain(|(k, _)| !matches!(k.as_str(), "gc_runs" | "witnesses_retired" | "async_slots" | "async_slots_reclaimed"));
    format!("{r:?}")
}

/// Collecting at every phase end changes nothing but the collector counters,
/// at 1, 8 and 32 workers. Returns the aggressive run's report.
fn aggressive_gc_agrees(s: &Scenario) -> Report {
    let mut last = None;
    for w in [1usize, 8, 32] {
        let (st0, r0) = run_gc(s, w, None);
        let (st1, r1) = run_gc(s, w, Some(1));
        assert_eq!(st0, st1);
        assert_eq!(without_gc_counters(r0), without_gc_counters(r1.clone()), "{}: collecting every phase changes the payload at {w} workers", s.name);
        last = Some(r1);
    }
    last.unwrap()
}

/// Status and payload identical at 1, 8 and 32 workers; returns the report.
fn worker_independent(s: &Scenario) -> Report {
    let (st1, r1) = run(s, 1);
    assert_eq!(st1, RunStatus::Completed, "{}: {st1:?}", s.name);
    for w in [8usize, 32] {
        let (st, r) = run(s, w);
        assert_eq!((format!("{st:?}"), format!("{r:?}")), (format!("{st1:?}"), format!("{r1:?}")), "{}: differs at {w} workers", s.name);
    }
    // Racecheck never changes the run.
    let plain = sched::run_with_config(&s.module, &s.inputs, &mut NoopObserver, &s.config).expect("run starts");
    assert_eq!(plain.status, RunStatus::Completed);
    r1
}

/// The race between `read_site` and `write_site` on `w`: the reported
/// conflict is exactly the written bytes 32..48, and every evidence span (an
/// access's own range: the whole read, or one word of the write) overlaps
/// them.
fn assert_race(r: &Report, read_site: SiteId, write_site: SiteId) {
    let hits: Vec<&Finding> = r
        .findings
        .iter()
        .filter(|f| matches!(f.kind, FindingKind::DataRace | FindingKind::ProxyRace | FindingKind::AsyncRace))
        .filter(|f| f.sites.contains(&read_site) && f.sites.contains(&write_site))
        .collect();
    assert!(!hits.is_empty(), "no race between {read_site:?} and {write_site:?}: {r:#?}");
    let (lo, hi) = (4 * WRITE_LO as u64, 4 * WRITE_HI as u64);
    for f in hits {
        assert!(f.message.contains(&format!("bytes [{lo}..{hi})")), "conflict bytes are not the written {lo}..{hi}: {f:#?}");
        let spans: Vec<_> = f.evidence.iter().filter_map(|e| e.bytes).collect();
        assert!(spans.len() >= 2, "race evidence lacks the two accesses' bytes: {f:#?}");
        for b in spans {
            assert!(b.start < hi && b.end() > lo && b.end() <= 4 * W_WORDS as u64, "evidence span {}..{} does not overlap {lo}..{hi} within w: {f:#?}", b.start, b.end());
        }
    }
}

fn assert_clean(r: &Report) {
    assert_eq!(r.verdict, Verdict::Clean, "{r:#?}");
}

/// A register-only loop that keeps a warp busy for many rounds.
fn delay(b: &mut ProgramBuilder, iters: u32) {
    let k = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let kn = b.k_u32(iters);
    b.mov(k, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, k, kn);
    b.loop_if(p);
    b.add_u32(k, k, k1);
    b.loop_end();
}

fn elect_if(b: &mut ProgramBuilder, e: Reg) {
    let full = b.k_u32(u32::MAX);
    b.push(Instr::Elect { dst_pred: e, dst_lane: None, membermask: full });
    b.if_(e);
}

fn mark_elect(p: &mut Program, cond: Reg) {
    for ins in &mut p.code {
        if let Instr::If { cond: Operand::Reg(c), elect, .. } = ins {
            if *c == cond {
                *elect = true;
            }
        }
    }
}


/// Lane-uniform `wait_until(buf[idx] == 1)` (acquire, gpu), with its
/// predicate program appended at `build` time by [`finish_waits`].
struct PendingWait {
    at: Pc,
    dst: Reg,
    addr: Reg,
}

fn wait_eq1(b: &mut ProgramBuilder, buf: Buf, idx: Operand, waits: &mut Vec<PendingWait>) {
    let addr = b.reg(Ty::U64);
    let dst = b.reg(Ty::U32);
    b.addr_of(addr, buf, idx);
    // Racecheck recognises the poll by its site op (`tirx.cuda.wait_until`).
    b.site("tirx.cuda.wait_until", 100 + waits.len() as u32);
    let at = b.push(Instr::Nop);
    b.no_site();
    waits.push(PendingWait { at, dst, addr });
}

fn finish_waits(prog: &mut Program, arg: Reg, res: Reg, one: Operand, waits: &[PendingWait]) {
    if waits.is_empty() {
        return;
    }
    let start = Pc(prog.code.len() as u32);
    prog.code.push(Instr::Compare { op: CmpOp::Eq, ty: Ty::U32, dst: res, a: arg.into(), b: one });
    prog.code_sites.push(SiteId::NONE);
    let pid = PredId(prog.preds.len() as u32);
    prog.preds.push(PredProgram { arg, start, end: Pc(start.0 + 1), result: res, reads_memory: false });
    for w in waits {
        prog.code[w.at.0 as usize] =
            Instr::WaitUntil { dst: w.dst, addr: w.addr.into(), ty: Ty::U32, space: AddrSpace::Generic, sem: Sem::Acquire, scope: Scope::Gpu, pred: pid, captures: vec![] };
    }
    prog.validate().expect("valid");
}

struct Sites {
    read: SiteId,
    write: SiteId,
}

/// `readers` single-CTA clusters read all of `w` (never written before), and
/// one more CTA (the writer, the last cluster) then plainly overwrites words
/// 8..12.
/// - Each reader CTA has two warps. With `bulk`, warp 0's elected lane does
///   `mbarrier.arrive.expect_tx` plus a 128-byte global->shared bulk copy of
///   `w` and warp 1 waits on the mbarrier (a TMA-like async read completing
///   on a phase). Without `bulk`, warp 1 reads `w` with plain lane loads.
/// - With `hb`, warp 1 lane 0 then does `st.release done[cta] = 1`, and the
///   writer spins `ld.acquire` on every `done[c]` before writing: clean.
/// - Without `hb`, the writer only waits a few hundred loop iterations: the
///   read (summarised by then) races the write.
fn ro_then_written(readers: u32, bulk: bool, hb: bool) -> (Scenario, Sites) {
    let mut b = ProgramBuilder::new("ro_then_written", 64);
    b.grid(readers + 1, 1, 1);
    let w = b.global("w", Dtype::U32);
    let done = b.global("done", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    b.declare_sync_words(done);
    let bar = b.shared("bar", Dtype::U64, 1);
    let data = b.shared("data", Dtype::U32, W_WORDS as u64);
    let cta = b.reg(Ty::U32);
    let tid = b.reg(Ty::U32);
    let warp = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let e = b.reg(Ty::PRED);
    let barr = b.reg(Ty::U32);
    let daddr = b.reg(Ty::U32);
    let gaddr = b.reg(Ty::U64);
    let v = b.reg(Ty::U32);
    let c = b.reg(Ty::U32);
    let idx = b.reg(Ty::U32);
    let (arg, res) = (b.reg(Ty::U32), b.reg(Ty::PRED));
    let mut waits = Vec::new();
    b.read_special(cta, SpecialReg::CtaLinear);
    b.thread_rank(tid);
    b.warp_id(warp);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let kr = b.k_u32(readers);
    let kbytes = b.k_u32(4 * W_WORDS);
    b.smem_addr(barr, bar, k0);
    b.smem_addr(daddr, data, k0);
    b.compare(CmpOp::Lt, Ty::U32, p, cta, kr);
    b.if_(p);
    // ---- reader CTA ----
    b.compare(CmpOp::Eq, Ty::U32, p, tid, k0);
    b.if_(p);
    b.mbar_init(barr, 1);
    b.end_if();
    b.fence(FenceKind::MbarrierInit, Sem::Release, Scope::Cluster);
    b.bar_sync(0);
    let read_site;
    if bulk {
        b.compare(CmpOp::Eq, Ty::U32, p, warp, k0);
        b.if_(p);
        elect_if(&mut b, e);
        b.push(Instr::MbarArrive(MbarArriveArgs {
            mbar: barr.into(),
            space: AddrSpace::Shared,
            count: None,
            expect_tx: Some(kbytes),
            drop: false,
            no_complete: false,
            sem: Sem::Release,
            scope: Scope::Cta,
            multicast: None,
            state: None,
        }));
        b.addr_of(gaddr, w, k0);
        read_site = b.site("w_bulk_read", 1);
        b.push(Instr::BulkCopy(BulkCopyArgs {
            dst: daddr.into(),
            dst_space: AddrSpace::SharedCluster,
            src: gaddr.into(),
            src_space: AddrSpace::Global,
            size: kbytes,
            completion: BulkCompletion::Mbarrier { mbar: barr.into(), space: AddrSpace::Shared },
            multicast: None,
            reduce: None,
            byte_mask: None,
            ignore_oob: None,
            report: None,
            mods: MemMods::default(),
        }));
        b.no_site();
        b.end_if();
        b.else_();
        b.mbar_wait_parity(barr, k0);
        b.ld_u32(v, data, lane);
    } else {
        b.compare(CmpOp::Eq, Ty::U32, p, warp, k1);
        b.if_(p);
        read_site = b.site("w_plain_read", 1);
        b.ld_u32(v, w, lane);
        b.no_site();
    }
    b.st_u32(out, cta, v);
    if hb {
        // Order every lane's read before lane 0's release.
        let full = b.k_u32(u32::MAX);
        b.push(Instr::WarpSync { membermask: full });
        b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
        b.if_(p);
        b.push(Instr::Store { ty: Ty::U32, buf: done, offset: cta.into(), value: k1, sem: Sem::Release, scope: Scope::Gpu, mods: MemMods::default() });
        b.end_if();
    }
    b.end_if();
    b.else_();
    // ---- writer CTA: warp 0 lane 0 ----
    b.compare(CmpOp::Eq, Ty::U32, p, tid, k0);
    b.if_(p);
    if hb {
        b.mov(c, k0);
        b.loop_begin();
        b.compare(CmpOp::Lt, Ty::U32, p, c, kr);
        b.loop_if(p);
        wait_eq1(&mut b, done, c.into(), &mut waits);
        b.add_u32(c, c, k1);
        b.loop_end();
    } else {
        delay(&mut b, 400);
    }
    let klo = b.k_u32(WRITE_LO);
    let khi = b.k_u32(WRITE_HI);
    b.mov(idx, klo);
    let write_site = b.site("w_write", 2);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, idx, khi);
    b.loop_if(p);
    b.st_u32(w, idx, idx);
    b.add_u32(idx, idx, k1);
    b.loop_end();
    b.no_site();
    b.end_if();
    b.end_if();
    b.exit();
    let mut prog = b.build();
    mark_elect(&mut prog, e);
    finish_waits(&mut prog, arg, res, k1, &waits);
    let n = (readers + 1) as usize;
    let s = Scenario {
        name: "ro_then_written",
        module: Module::new(vec![prog]),
        inputs: inputs(vec![("w", u32_buf(0..W_WORDS)), ("done", u32_buf(vec![0; n])), ("out", u32_buf(vec![0; n]))]),
        config: RunConfig { loop_budget: 1 << 40, ..RunConfig::default() },
    };
    (s, Sites { read: read_site, write: write_site })
}

/// One CTA, three warps. Warp 0's elected lane issues op A: a 128-byte bulk
/// read of `w` completing on an mbarrier; warp 1 waits on it. After many
/// rounds, warp 2's elected lane issues op B: a bulk store shared->global
/// of words 8..12 of `w` (bulk group, then wait). By then A has completed and
/// its slot may be reclaimed and reused by B.
/// - Without `hb`, warp 2 never synchronises with A's completion, so A's
///   (summarised) read races B's write.
/// - With `hb`, warp 1 publishes `st.release flag` after its wait and warp 2
///   spins `ld.acquire flag` before issuing B: clean.
fn reclaimed_slot_successor(hb: bool) -> (Scenario, Sites) {
    let mut b = ProgramBuilder::new("reclaimed_slot_successor", 96);
    let w = b.global("w", Dtype::U32);
    let flag = b.global("flag", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    b.declare_sync_words(flag);
    let bar = b.shared("bar", Dtype::U64, 1);
    let data = b.shared("data", Dtype::U32, W_WORDS as u64);
    let blk = b.shared("blk", Dtype::U32, (WRITE_HI - WRITE_LO) as u64);
    let tid = b.reg(Ty::U32);
    let warp = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let e = b.reg(Ty::PRED);
    let barr = b.reg(Ty::U32);
    let daddr = b.reg(Ty::U32);
    let saddr = b.reg(Ty::U32);
    let gaddr = b.reg(Ty::U64);
    let v = b.reg(Ty::U32);
    let (arg, res) = (b.reg(Ty::U32), b.reg(Ty::PRED));
    let mut waits = Vec::new();
    b.thread_rank(tid);
    b.warp_id(warp);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k2 = b.k_u32(2);
    let kbytes = b.k_u32(4 * W_WORDS);
    let kwbytes = b.k_u32(4 * (WRITE_HI - WRITE_LO));
    let klo = b.k_u32(WRITE_LO);
    b.smem_addr(barr, bar, k0);
    b.smem_addr(daddr, data, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, tid, k0);
    b.if_(p);
    b.mbar_init(barr, 1);
    b.end_if();
    b.fence(FenceKind::MbarrierInit, Sem::Release, Scope::Cluster);
    b.bar_sync(0);
    b.compare(CmpOp::Eq, Ty::U32, p, warp, k0);
    b.if_(p);
    // ---- warp 0: op A ----
    elect_if(&mut b, e);
    b.push(Instr::MbarArrive(MbarArriveArgs {
        mbar: barr.into(),
        space: AddrSpace::Shared,
        count: None,
        expect_tx: Some(kbytes),
        drop: false,
        no_complete: false,
        sem: Sem::Release,
        scope: Scope::Cta,
        multicast: None,
        state: None,
    }));
    b.addr_of(gaddr, w, k0);
    let read_site = b.site("w_bulk_read", 1);
    b.push(Instr::BulkCopy(BulkCopyArgs {
        dst: daddr.into(),
        dst_space: AddrSpace::SharedCluster,
        src: gaddr.into(),
        src_space: AddrSpace::Global,
        size: kbytes,
        completion: BulkCompletion::Mbarrier { mbar: barr.into(), space: AddrSpace::Shared },
        multicast: None,
        reduce: None,
        byte_mask: None,
        ignore_oob: None,
        report: None,
        mods: MemMods::default(),
    }));
    b.no_site();
    b.end_if();
    b.else_();
    b.compare(CmpOp::Eq, Ty::U32, p, warp, k1);
    b.if_(p);
    // ---- warp 1: wait for A ----
    b.mbar_wait_parity(barr, k0);
    b.ld_u32(v, data, lane);
    b.st_u32(out, lane, v);
    if hb {
        let full = b.k_u32(u32::MAX);
        b.push(Instr::WarpSync { membermask: full });
        b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
        b.if_(p);
        b.push(Instr::Store { ty: Ty::U32, buf: flag, offset: k0, value: k1, sem: Sem::Release, scope: Scope::Gpu, mods: MemMods::default() });
        b.end_if();
    }
    b.else_();
    // ---- warp 2: op B, much later ----
    b.compare(CmpOp::Eq, Ty::U32, p, warp, k2);
    b.if_(p);
    if hb {
        wait_eq1(&mut b, flag, k0, &mut waits);
    } else {
        delay(&mut b, 400);
    }
    let kw = b.k_u32(WRITE_HI - WRITE_LO);
    b.compare(CmpOp::Lt, Ty::U32, p, lane, kw);
    b.if_(p);
    b.st_u32(blk, lane, lane);
    b.end_if();
    b.fence(FenceKind::ProxyAsync(Some(AddrSpace::Shared)), Sem::Weak, Scope::Cta);
    let full = b.k_u32(u32::MAX);
    b.push(Instr::WarpSync { membermask: full });
    let e2 = b.reg(Ty::PRED);
    elect_if(&mut b, e2);
    b.smem_addr(saddr, blk, k0);
    b.addr_of(gaddr, w, klo);
    let write_site = b.site("w_bulk_store", 2);
    b.push(Instr::BulkCopy(BulkCopyArgs {
        dst: gaddr.into(),
        dst_space: AddrSpace::Global,
        src: saddr.into(),
        src_space: AddrSpace::Shared,
        size: kwbytes,
        completion: BulkCompletion::Group,
        multicast: None,
        reduce: None,
        byte_mask: None,
        ignore_oob: None,
        report: None,
        mods: MemMods::default(),
    }));
    b.push(Instr::AsyncCommit { domain: Domain::Bulk });
    b.push(Instr::AsyncWait { domain: Domain::Bulk, n: 0, read: false });
    b.no_site();
    b.end_if();
    b.end_if();
    b.end_if();
    b.end_if();
    b.exit();
    let mut prog = b.build();
    mark_elect(&mut prog, e);
    mark_elect(&mut prog, e2);
    finish_waits(&mut prog, arg, res, k1, &waits);
    let s = Scenario {
        name: "reclaimed_slot_successor",
        module: Module::new(vec![prog]),
        inputs: inputs(vec![("w", u32_buf(0..W_WORDS)), ("flag", u32_buf([0])), ("out", u32_buf(vec![0; 32]))]),
        config: RunConfig { loop_budget: 1 << 40, ..RunConfig::default() },
    };
    (s, Sites { read: read_site, write: write_site })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn bulk_reads_then_write_with_hb_is_clean() {
    let (s, _) = ro_then_written(6, true, true);
    assert_clean(&worker_independent(&s));
}

#[test]
fn bulk_reads_then_write_without_hb_races_with_anchors() {
    let (s, sites) = ro_then_written(6, true, false);
    assert_race(&worker_independent(&s), sites.read, sites.write);
}

#[test]
fn plain_reads_then_write_with_hb_is_clean() {
    let (s, _) = ro_then_written(6, false, true);
    assert_clean(&worker_independent(&s));
}

#[test]
fn plain_reads_then_write_without_hb_races_with_anchors() {
    let (s, sites) = ro_then_written(6, false, false);
    assert_race(&worker_independent(&s), sites.read, sites.write);
}

#[test]
fn summarised_read_vs_reclaimed_slot_successor_races() {
    let (s, sites) = reclaimed_slot_successor(false);
    assert_race(&worker_independent(&s), sites.read, sites.write);
}

#[test]
fn summarised_read_vs_reclaimed_slot_successor_with_hb_is_clean() {
    let (s, _) = reclaimed_slot_successor(true);
    assert_clean(&worker_independent(&s));
}

#[test]
fn aggressive_gc_keeps_every_verdict() {
    for (s, _) in [
        ro_then_written(6, true, true),
        ro_then_written(6, true, false),
        ro_then_written(6, false, true),
        ro_then_written(6, false, false),
        reclaimed_slot_successor(false),
        reclaimed_slot_successor(true),
    ] {
        let r = aggressive_gc_agrees(&s);
        assert!(coverage(&r, "gc_runs") > 0, "{}: the collector never ran", s.name);
    }
}

/// The successor scenario really retires op A before op B is issued:
/// collecting every phase reclaims at least one async slot.
#[test]
fn successor_scenario_reclaims_slots() {
    for hb in [false, true] {
        let (s, _) = reclaimed_slot_successor(hb);
        let (_, r) = run_gc(&s, 1, Some(1));
        assert!(coverage(&r, "async_slots_reclaimed") > 0, "hb={hb}: no async slot was reclaimed: {:?}", r.coverage);
    }
}
