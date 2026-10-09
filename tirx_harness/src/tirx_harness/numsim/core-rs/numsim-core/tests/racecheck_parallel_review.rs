//! W6 review of the parallel racecheck design (racecheck-parallel-design.md
//! §11, scenario list §11.4). Each scenario runs the engine under the serial
//! `RaceObserver` at 1, 8 and 32 workers and requires identical racecheck
//! payloads (the serial baseline the fork/join checker must reproduce).
//!
//! The fork/join arm of each scenario is a separate `#[ignore]`d test that
//! calls [`fork_join_payload`]. W5: implement it with the fork/join observer
//! (decision 17) and remove the `#[ignore]`s; each names the §11 item it
//! guards.
use numsim_core::dtype::{Dtype, Ty};
use numsim_core::program::*;
use numsim_core::racecheck::observer::{RaceObserver, RacecheckConfig};
use numsim_core::report::Report;
use numsim_core::sched::{self, RunConfig, RunStatus};
use numsim_core::testutil::scenarios::{self, inputs, u32_buf, Scenario};
use numsim_core::testutil::{fixtures, ProgramBuilder};
use numsim_core::Module;

const WORKERS: [usize; 3] = [1, 8, 32];

/// Serial racecheck payload of one run, rendered for comparison.
fn serial_payload(s: &Scenario, workers: usize, rc: &RacecheckConfig) -> (String, String) {
    let cfg = RunConfig { workers, ..s.config.clone() };
    let mut obs = RaceObserver::new(rc.clone());
    obs.fork_join = false;
    obs.phase_gc = true;
    let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &cfg).expect("run starts");
    let report: Report = obs.finish();
    (format!("{:?}", o.status), format!("{report:?}"))
}

/// Fork/join racecheck payload (decision 17): the same run with the
/// observer forking one child checker per scheduling partition.
fn fork_join_payload(s: &Scenario, workers: usize, rc: &RacecheckConfig) -> (String, String) {
    let cfg = RunConfig { workers, ..s.config.clone() };
    let mut obs = RaceObserver::new(rc.clone());
    obs.fork_join = true;
    obs.phase_gc = true;
    let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &cfg).expect("run starts");
    let report: Report = obs.finish();
    (format!("{:?}", o.status), format!("{report:?}"))
}

/// The serial payload is identical at every worker count; returns it.
fn serial_is_worker_independent(s: &Scenario, rc: &RacecheckConfig) -> (String, String) {
    let base = serial_payload(s, WORKERS[0], rc);
    for &w in &WORKERS[1..] {
        assert_eq!(serial_payload(s, w, rc), base, "{}: serial racecheck payload differs at {w} workers", s.name);
    }
    base
}

/// The fork/join payload equals the serial one at every worker count.
#[allow(dead_code)]
fn fork_join_matches_serial(s: &Scenario, rc: &RacecheckConfig) {
    let serial = serial_payload(s, 1, rc);
    for &w in &WORKERS {
        assert_eq!(fork_join_payload(s, w, rc), serial, "{}: fork/join differs from serial at {w} workers", s.name);
    }
}

fn st(b: &mut ProgramBuilder, buf: Buf, idx: Operand, value: Operand, sem: Sem) {
    b.push(Instr::Store { ty: Ty::U32, buf, offset: idx, value, sem, scope: Scope::Gpu, mods: MemMods::default() });
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

/// H1: `cross_cluster_flag(false)`: CTA 1's `ld.acquire` of the flag is in
/// the same round as CTA 0's `st.release`, replayed before it (it read the
/// round-start 0). The data read must stay a race.
fn h1() -> Scenario {
    scenarios::cross_cluster_flag(false)
}

/// H2: `wait_until(flag >= limit[0])` in CTA 0 (warp 1 publishes the flag),
/// while CTA 1 plainly stores `limit[0] = 2` (the value it already has) in
/// the same round, after the wait in replay order.
fn h2() -> Scenario {
    let mut b = ProgramBuilder::new("pred_reads_later_writer", 64);
    b.grid(2, 1, 1);
    let flag = b.global("flag", Dtype::U32);
    let limit = b.global("limit", Dtype::U32);
    b.declare_sync_words(flag);
    let cta = b.reg(Ty::U32);
    let w = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let fa = b.reg(Ty::U64);
    let got = b.reg(Ty::U32);
    let arg = b.reg(Ty::U32);
    let lim = b.reg(Ty::U32);
    let res = b.reg(Ty::PRED);
    b.read_special(cta, SpecialReg::CtaLinear);
    b.warp_id(w);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k2 = b.k_u32(2);
    let k3 = b.k_u32(3);
    b.addr_of(fa, flag, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, cta, k1);
    b.if_(p);
    // CTA 1: one plain store to `limit` by warp 0 lane 0.
    b.compare(CmpOp::Eq, Ty::U32, p, w, k0);
    b.if_(p);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.site("limit_store", 1);
    st(&mut b, limit, k0, k2, Sem::Weak);
    b.no_site();
    b.end_if();
    b.end_if();
    b.else_();
    // CTA 0: warp 1 publishes, warp 0 waits.
    b.compare(CmpOp::Eq, Ty::U32, p, w, k1);
    b.if_(p);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    st(&mut b, flag, k0, k3, Sem::Release);
    b.end_if();
    b.else_();
    b.site("wait_until", 2);
    let placeholder = b.push(Instr::Nop);
    b.no_site();
    b.end_if();
    b.end_if();
    b.exit();
    let mut prog = b.build();
    let start = Pc(prog.code.len() as u32);
    prog.code.push(Instr::Load { ty: Ty::U32, dst: lim, buf: limit, offset: k0, sem: Sem::Weak, scope: Scope::Cta, mods: MemMods::default() });
    prog.code.push(Instr::Compare { op: CmpOp::Ge, ty: Ty::U32, dst: res, a: arg.into(), b: lim.into() });
    prog.code_sites.push(numsim_core::site::SiteId::NONE);
    prog.code_sites.push(numsim_core::site::SiteId::NONE);
    prog.preds.push(PredProgram { arg, start, end: Pc(start.0 + 2), result: res, reads_memory: true });
    prog.code[placeholder.0 as usize] =
        Instr::WaitUntil { dst: got, addr: fa.into(), ty: Ty::U32, space: AddrSpace::Generic, sem: Sem::Acquire, scope: Scope::Gpu, pred: PredId(0), captures: vec![] };
    prog.validate().expect("valid");
    let mut s = Scenario {
        name: "pred_reads_later_writer",
        module: Module::new(vec![prog]),
        inputs: inputs(vec![("flag", u32_buf([0])), ("limit", u32_buf([2]))]),
        config: RunConfig::default(),
    };
    s.config.loop_budget = 1 << 40;
    s
}

/// H3: CTA 0 plainly stores `x[0] = 5`; CTA 1 does `atom.add x[0], 1` (a
/// serial-phase global RMW) in the same round, reading CTA 0's write.
fn h3() -> Scenario {
    let mut b = ProgramBuilder::new("serial_atom_reads_same_round_write", 32);
    b.grid(2, 1, 1);
    let x = b.global("x", Dtype::U32);
    let cta = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let xa = b.reg(Ty::U64);
    let old = b.reg(Ty::U32);
    b.read_special(cta, SpecialReg::CtaLinear);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k5 = b.k_u32(5);
    b.addr_of(xa, x, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.compare(CmpOp::Eq, Ty::U32, p, cta, k0);
    b.if_(p);
    b.site("plain_store", 1);
    st(&mut b, x, k0, k5, Sem::Weak);
    b.else_();
    b.site("atom_add", 2);
    b.push(Instr::Atom { op: AtomOp::Add, ty: Ty::U32, dst: Some(old), addr: xa.into(), space: AddrSpace::Global, value: k1, cmp: None, sem: Sem::Relaxed, scope: Scope::Gpu, ftz: false });
    b.no_site();
    b.end_if();
    b.end_if();
    b.exit();
    Scenario { name: "serial_atom_reads_same_round_write", module: b.build_module(), inputs: inputs(vec![("x", u32_buf([0]))]), config: RunConfig::default() }
}

/// H1 in a replay cycle (milestone 2 review). In round 0:
/// - CTA 0 stores `data = 5`, `st.release flag = 1`, then plainly reads `x`
///   (round-start 0).
/// - CTA 1 does `ld.acquire flag` first (round-start 0, before any strong
///   write of its own, so a fork/join child may resolve it), then plainly
///   stores `x = 1`, then reads `data`.
///
/// Each read what the other wrote, so the engine reports
/// `cross_cluster_same_round_cycle` and replays in partition order (CTA 0
/// first). The serial checker then resolves CTA 1's acquire against CTA 0's
/// release (the latest write in `seq`), so the data read is ordered. A child
/// resolving against round-start state would not order it.
fn sb_with_data() -> Scenario {
    let mut b = ProgramBuilder::new("sb_with_data", 32);
    b.grid(2, 1, 1);
    let flag = b.global("flag", Dtype::U32);
    let x = b.global("x", Dtype::U32);
    let data = b.global("data", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let cta = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let v = b.reg(Ty::U32);
    let f = b.reg(Ty::U32);
    b.read_special(cta, SpecialReg::CtaLinear);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k5 = b.k_u32(5);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.compare(CmpOp::Eq, Ty::U32, p, cta, k0);
    b.if_(p);
    b.site("data_store", 1);
    st(&mut b, data, k0, k5, Sem::Weak);
    b.site("flag_release", 2);
    st(&mut b, flag, k0, k1, Sem::Release);
    b.site("x_load", 3);
    b.ld_u32(v, x, k0);
    b.no_site();
    b.st_u32(out, k0, v);
    b.else_();
    b.site("flag_acquire", 4);
    b.push(Instr::Load { ty: Ty::U32, dst: f, buf: flag, offset: k0, sem: Sem::Acquire, scope: Scope::Gpu, mods: MemMods::default() });
    b.site("x_store", 5);
    st(&mut b, x, k0, k1, Sem::Weak);
    b.site("data_load", 6);
    b.ld_u32(v, data, k0);
    b.no_site();
    b.add_u32(v, v, f);
    b.st_u32(out, k1, v);
    b.end_if();
    b.end_if();
    b.exit();
    Scenario {
        name: "sb_with_data",
        module: b.build_module(),
        inputs: inputs(vec![("flag", u32_buf([0])), ("x", u32_buf([0])), ("data", u32_buf([0])), ("out", u32_buf([0, 0]))]),
        config: RunConfig::default(),
    }
}

/// A ring of `n` single-CTA clusters (milestone 2 review; W5's suggestion).
/// Lane 0 of CTA c first does `ld.acquire flag[(c+1) % n]` and reads
/// `data[(c+1) % n]` (round-start values, before any strong write of its
/// own), then stores `data[c]` and `st.release flag[c] = 1`. Each CTA reads
/// what the next one writes: a cycle, so the round replays in partition order
/// and serially. With `bystander`, an extra CTA n (not in the ring) resolves
/// `ld.acquire flag[0]` and reads `data[0]` in the same round.
fn ring(n: u32, bystander: bool) -> Scenario {
    let total = n + u32::from(bystander);
    let mut b = ProgramBuilder::new("ring", 32);
    b.grid(total, 1, 1);
    let flag = b.global("flag", Dtype::U32);
    let data = b.global("data", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let cta = b.reg(Ty::U32);
    let next = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let f = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    b.read_special(cta, SpecialReg::CtaLinear);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let kn = b.k_u32(n);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.compare(CmpOp::Lt, Ty::U32, p, cta, kn);
    b.if_(p);
    // next = (cta + 1) % n
    b.add_u32(next, cta, k1);
    b.compare(CmpOp::Eq, Ty::U32, p, next, kn);
    b.if_(p);
    b.mov(next, k0);
    b.end_if();
    b.site("flag_acquire", 1);
    b.push(Instr::Load { ty: Ty::U32, dst: f, buf: flag, offset: next.into(), sem: Sem::Acquire, scope: Scope::Gpu, mods: MemMods::default() });
    b.site("data_load", 2);
    b.ld_u32(v, data, next);
    b.site("data_store", 3);
    b.st_u32(data, cta, cta);
    b.site("flag_release", 4);
    st(&mut b, flag, cta.into(), k1, Sem::Release);
    b.no_site();
    b.add_u32(v, v, f);
    b.st_u32(out, cta, v);
    b.else_();
    b.site("bystander_acquire", 5);
    b.push(Instr::Load { ty: Ty::U32, dst: f, buf: flag, offset: k0, sem: Sem::Acquire, scope: Scope::Gpu, mods: MemMods::default() });
    b.site("bystander_load", 6);
    b.ld_u32(v, data, k0);
    b.no_site();
    b.add_u32(v, v, f);
    b.st_u32(out, cta, v);
    b.end_if();
    b.end_if();
    b.exit();
    Scenario {
        name: if bystander { "ring_with_bystander" } else { "ring" },
        module: b.build_module(),
        inputs: inputs(vec![("flag", u32_buf(vec![0; total as usize])), ("data", u32_buf(vec![0; total as usize])), ("out", u32_buf(vec![0; total as usize]))]),
        config: RunConfig::default(),
    }
}

/// W6-P1 under racecheck: two clusters each `st.release flag = cta+1` and
/// then `wait_until(flag == cta+1)` (same round). Each verdict must acquire
/// the waiter's own write.
fn same_round_writers() -> Scenario {
    let mut b = ProgramBuilder::new("same_round_writers", 32);
    b.grid(2, 1, 1);
    let flag = b.global("flag", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    b.declare_sync_words(flag);
    let lane = b.reg(Ty::U32);
    let cta = b.reg(Ty::U32);
    let mine = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let fa = b.reg(Ty::U64);
    let got = b.reg(Ty::U32);
    let arg = b.reg(Ty::U32);
    let res = b.reg(Ty::PRED);
    b.lane_id(lane);
    b.read_special(cta, SpecialReg::CtaLinear);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    b.addr_of(fa, flag, k0);
    b.add_u32(mine, cta, k1);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.push(Instr::StoreAddr { ty: Ty::U32, addr: fa.into(), space: AddrSpace::Generic, value: mine.into(), sem: Sem::Release, scope: Scope::Gpu, mods: MemMods::default() });
    b.site("wait_until", 1);
    let placeholder = b.push(Instr::Nop);
    b.no_site();
    b.st_u32(out, cta, got);
    b.end_if();
    b.exit();
    let mut prog = b.build();
    let start = Pc(prog.code.len() as u32);
    prog.code.push(Instr::Compare { op: CmpOp::Eq, ty: Ty::U32, dst: res, a: arg.into(), b: mine.into() });
    prog.code_sites.push(numsim_core::site::SiteId::NONE);
    prog.preds.push(PredProgram { arg, start, end: Pc(start.0 + 1), result: res, reads_memory: false });
    prog.code[placeholder.0 as usize] =
        Instr::WaitUntil { dst: got, addr: fa.into(), ty: Ty::U32, space: AddrSpace::Generic, sem: Sem::Acquire, scope: Scope::Gpu, pred: PredId(0), captures: vec![mine] };
    prog.validate().expect("valid");
    let mut s = Scenario {
        name: "same_round_writers",
        module: Module::new(vec![prog]),
        inputs: inputs(vec![("flag", u32_buf([0])), ("out", u32_buf([0, 0]))]),
        config: RunConfig::default(),
    };
    s.config.loop_budget = 1 << 40;
    s
}

/// D3: `wait_until_chain(8)` with at most 2 resident CTAs, so clusters
/// retire and partitions leave `Scheduler::partitions` (Vec indices are
/// reused by later clusters).
fn turnover() -> Scenario {
    let mut s = scenarios::wait_until_chain(8);
    s.config.max_resident_ctas = 2;
    s
}

/// D5: 8 single-CTA clusters and 9 buffers. Lane 0 of CTA c plainly stores
/// `buf[c]` and `buf[c+1]` from per-CTA sites, so CTAs c and c+1 race on
/// `buf[c+1]`: 7 races with distinct keys (sites, allocation), more than the
/// cap of 3.
fn many_races() -> Scenario {
    const N: u32 = 8;
    let mut b = ProgramBuilder::new("many_races", 32);
    b.grid(N, 1, 1);
    let names: Vec<String> = (0..=N).map(|k| format!("buf{k}")).collect();
    let bufs: Vec<Buf> = names.iter().map(|n| b.global(n, Dtype::U32)).collect();
    let cta = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    b.read_special(cta, SpecialReg::CtaLinear);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    for c in 0..N {
        let kc = b.k_u32(c);
        b.compare(CmpOp::Eq, Ty::U32, p, cta, kc);
        b.if_(p);
        b.site(&format!("own_{c}"), 10 + 2 * c);
        b.st_u32(bufs[c as usize], k0, cta);
        b.site(&format!("next_{c}"), 11 + 2 * c);
        b.st_u32(bufs[c as usize + 1], k0, cta);
        b.no_site();
        b.end_if();
    }
    b.end_if();
    b.exit();
    let args: Vec<(&str, numsim_core::sched::ArgValue)> = names.iter().map(|n| (n.as_str(), u32_buf([0]))).collect();
    Scenario { name: "many_races", module: b.build_module(), inputs: inputs(args), config: RunConfig::default() }
}

fn capped() -> RacecheckConfig {
    RacecheckConfig { max_findings: 3 }
}

// ---------------------------------------------------------------------------
// Serial baselines (must pass today)
// ---------------------------------------------------------------------------

#[test]
fn h1_same_round_flag_serial_is_worker_independent() {
    let (status, report) = serial_is_worker_independent(&h1(), &RacecheckConfig::default());
    assert_eq!(status, format!("{:?}", RunStatus::Completed));
    assert!(report.contains("DataRace") || report.contains("data_race"), "the unsynchronised data read is a race: {report}");
}

#[test]
fn h2_pred_reads_later_writer_serial_is_worker_independent() {
    serial_is_worker_independent(&h2(), &RacecheckConfig::default());
}

#[test]
fn h3_serial_phase_atom_serial_is_worker_independent() {
    serial_is_worker_independent(&h3(), &RacecheckConfig::default());
}

#[test]
fn w6_p1_same_round_writers_serial_is_worker_independent() {
    let (status, _) = serial_is_worker_independent(&same_round_writers(), &RacecheckConfig::default());
    assert_eq!(status, format!("{:?}", RunStatus::Completed));
}

#[test]
fn d3_turnover_serial_is_worker_independent() {
    let (status, _) = serial_is_worker_independent(&turnover(), &RacecheckConfig::default());
    assert_eq!(status, format!("{:?}", RunStatus::Completed));
}

#[test]
fn d5_findings_cap_serial_is_worker_independent() {
    let (_, report) = serial_is_worker_independent(&many_races(), &capped());
    assert!(report.to_lowercase().contains("truncat"), "the scenario must exceed the cap: {report}");
}

#[test]
fn h6_store_buffering_serial_is_worker_independent() {
    serial_is_worker_independent(&scenarios::cross_cluster_sb(), &RacecheckConfig::default());
}

#[test]
fn h1_cycle_sb_with_data_serial_is_worker_independent() {
    let (status, report) = serial_is_worker_independent(&sb_with_data(), &RacecheckConfig::default());
    assert_eq!(status, format!("{:?}", RunStatus::Completed));
    // The engine's stream-cycle diagnostic is a run diagnostic, not a
    // racecheck finding; the payload here is the serial checker's.
    let _ = report;
}

/// Milestone 2 (§14): in a replay cycle, a child would resolve
/// `flag_acquire` against round-start state although CTA 0, replayed
/// earlier (partition-order fallback), wrote it. Since W5-17b (784e2df) a
/// cycle round replays serially with no fork offers; this guards that.
#[test]
fn h1_cycle_sb_with_data_fork_join_matches_serial() {
    fork_join_matches_serial(&sb_with_data(), &RacecheckConfig::default());
}

#[test]
fn h1_ring3_serial_is_worker_independent() {
    serial_is_worker_independent(&ring(3, false), &RacecheckConfig::default());
}

#[test]
fn h1_ring3_fork_join_matches_serial() {
    fork_join_matches_serial(&ring(3, false), &RacecheckConfig::default());
}

#[test]
fn h1_cycle_with_bystander_serial_is_worker_independent() {
    serial_is_worker_independent(&ring(2, true), &RacecheckConfig::default());
}

#[test]
fn h1_cycle_with_bystander_fork_join_matches_serial() {
    fork_join_matches_serial(&ring(2, true), &RacecheckConfig::default());
}

#[test]
fn all_scenarios_serial_is_worker_independent() {
    for s in scenarios::all() {
        serial_is_worker_independent(&s, &RacecheckConfig::default());
    }
}

/// Corpus fixtures (`examples/record_race_fixtures.py`; skipped when absent,
/// as in CI). Release build recommended.
const CORPUS: [&str; 8] = [
    "rmsnorm",
    "deepgemm_sm100_fp8_gemm_1d1d",
    "fp16_bf16_gemm",
    "gdn_decode_bf16_wide_vec_mtp",
    "radix_topk_multi_cta",
    "recurrent_kda_decode_one_warp",
    "selective_state_update_stp_simple",
    "mega_moe_t8_h1024_i512_e24_k2_g1",
];

fn corpus() -> Vec<Scenario> {
    // The recorded corpus runs take minutes unoptimised: release builds
    // only (`cargo test --release --test racecheck_parallel_review`).
    if cfg!(debug_assertions) {
        return Vec::new();
    }
    let dir = fixtures::dir();
    CORPUS
        .iter()
        .filter(|c| fixtures::exists(&dir, c))
        .map(|c| {
            let (module, inputs, config) = fixtures::load(&dir, c);
            Scenario { name: c, module, inputs, config }
        })
        .collect()
}

#[test]
fn corpus_serial_is_worker_independent() {
    for s in corpus() {
        serial_is_worker_independent(&s, &RacecheckConfig::default());
    }
}

// ---------------------------------------------------------------------------
// Fork/join arm
// ---------------------------------------------------------------------------

#[test]
fn h1_same_round_flag_fork_join_matches_serial() {
    fork_join_matches_serial(&h1(), &RacecheckConfig::default());
}

#[test]
fn h2_pred_reads_later_writer_fork_join_matches_serial() {
    fork_join_matches_serial(&h2(), &RacecheckConfig::default());
}

#[test]
fn h3_serial_phase_atom_fork_join_matches_serial() {
    fork_join_matches_serial(&h3(), &RacecheckConfig::default());
}

#[test]
fn w6_p1_same_round_writers_fork_join_matches_serial() {
    fork_join_matches_serial(&same_round_writers(), &RacecheckConfig::default());
}

#[test]
fn d3_turnover_fork_join_matches_serial() {
    fork_join_matches_serial(&turnover(), &RacecheckConfig::default());
}

#[test]
fn d5_findings_cap_fork_join_matches_serial() {
    fork_join_matches_serial(&many_races(), &capped());
}

#[test]
fn h6_store_buffering_fork_join_matches_serial() {
    fork_join_matches_serial(&scenarios::cross_cluster_sb(), &RacecheckConfig::default());
}

#[test]
fn all_scenarios_fork_join_matches_serial() {
    for s in scenarios::all() {
        fork_join_matches_serial(&s, &RacecheckConfig::default());
    }
}

#[test]
fn corpus_fork_join_matches_serial() {
    for s in corpus() {
        fork_join_matches_serial(&s, &RacecheckConfig::default());
    }
}

// ---------------------------------------------------------------------------
// §11.1 D7 and I8 under racecheck (W6 re-review of milestone 1)
// ---------------------------------------------------------------------------

/// Serial payload with an explicit GC mode.
fn serial_payload_gc(s: &Scenario, workers: usize, rc: &RacecheckConfig, phase_gc: bool) -> (String, String) {
    let cfg = RunConfig { workers, ..s.config.clone() };
    let mut obs = RaceObserver::new(rc.clone());
    obs.fork_join = false;
    obs.phase_gc = phase_gc;
    let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &cfg).expect("run starts");
    let mut report: Report = obs.finish();
    // The collector's counters (and the slot peak, which reclaim after GC
    // lowers) depend on when it runs, by construction.
    report.coverage.retain(|(k, _)| !matches!(k.as_str(), "gc_runs" | "witnesses_retired" | "async_slots" | "async_slots_reclaimed"));
    (format!("{:?}", o.status), format!("{report:?}"))
}

fn review_scenarios() -> Vec<(Scenario, RacecheckConfig)> {
    let d = RacecheckConfig::default;
    let mut v = vec![
        (h1(), d()),
        (h2(), d()),
        (h3(), d()),
        (same_round_writers(), d()),
        (turnover(), d()),
        (many_races(), capped()),
        (scenarios::cross_cluster_sb(), d()),
        (sb_with_data(), d()),
        (ring(3, false), d()),
        (ring(2, true), d()),
    ];
    v.extend(scenarios::all().into_iter().map(|s| (s, d())));
    v.extend(corpus().into_iter().map(|s| (s, d())));
    v
}

/// D7: when the collector runs (phase ends vs the periodic default) changes
/// no finding, no reported witness and no other payload field. Only the
/// collector's counters (`gc_runs`, `witnesses_retired`) and the async-slot
/// peak/reclaim counts may differ. Both arms of the review tests pin
/// `phase_gc = true`, so this is the only check that phase-end GC equals the
/// former serial behaviour.
#[test]
fn d7_phase_end_gc_does_not_change_the_payload() {
    for (s, rc) in review_scenarios() {
        let (periodic, phase) = (serial_payload_gc(&s, 1, &rc, false), serial_payload_gc(&s, 1, &rc, true));
        if periodic != phase {
            let (a, b) = (periodic.1.as_bytes(), phase.1.as_bytes());
            let i = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
            let at = |t: &str| t[i.saturating_sub(300)..(i + 300).min(t.len())].to_string();
            panic!("{}: phase-end GC changes the payload at byte {i}:\nperiodic: ...{}...\nphase:    ...{}...", s.name, at(&periodic.1), at(&phase.1));
        }
    }
}

/// I8 under racecheck: status and outputs are identical with no observer
/// and with a `RaceObserver` (serial or fork/join, phase-end GC on), at 1
/// and 8 workers.
#[test]
fn race_observer_does_not_change_the_run() {
    use numsim_core::observe::NoopObserver;
    for (s, rc) in review_scenarios() {
        for workers in [1usize, 8] {
            let cfg = RunConfig { workers, ..s.config.clone() };
            let plain = sched::run_with_config(&s.module, &s.inputs, &mut NoopObserver, &cfg).expect("run starts");
            for fork_join in [false, true] {
                let mut obs = RaceObserver::new(rc.clone());
                obs.fork_join = fork_join;
                obs.phase_gc = true;
                let watched = sched::run_with_config(&s.module, &s.inputs, &mut obs, &cfg).expect("run starts");
                assert_eq!(
                    (format!("{:?}", plain.status), &plain.outputs),
                    (format!("{:?}", watched.status), &watched.outputs),
                    "{}: racecheck (fork_join {fork_join}) changes the run at {workers} workers",
                    s.name
                );
            }
        }
    }
}

/// The cycle scenarios really are replay cycles (otherwise their fork/join
/// arms would not exercise W5-17b): the engine reports
/// `cross_cluster_same_round_cycle` at 8 workers.
#[test]
fn cycle_scenarios_are_replay_cycles() {
    for s in [sb_with_data(), ring(3, false), ring(2, true)] {
        let cfg = RunConfig { workers: 8, ..s.config.clone() };
        let mut obs = RaceObserver::new(RacecheckConfig::default());
        let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &cfg).expect("run starts");
        assert!(
            o.diagnostics.iter().any(|d| d.attrs.get("reason").and_then(|r| r.as_str()) == Some("cross_cluster_same_round_cycle")),
            "{}: no replay cycle reported: {:?}",
            s.name,
            o.diagnostics
        );
    }
}

// ---------------------------------------------------------------------------
// W16 child-side DeclareWord (patch 02): a word declared at its first
// `wait_until` in one partition, written by another partition in the same
// round.
// ---------------------------------------------------------------------------

/// Two single-CTA clusters. `w` is NOT a `sync_words` buffer, so it is
/// declared at its first `wait_until`.
/// - CTA 1 lane 0: `wait_until(w == 7)` (the first use declares `w`), then
///   reads `data`.
/// - CTA 0 lane 0, in the same round: stores `data = 5`, then `w = 7` with
///   `st.release.gpu` (`hb`) or a plain store (no publication edge).
fn first_wait_declare(hb: bool) -> Scenario {
    let mut b = ProgramBuilder::new("first_wait_declare", 32);
    b.grid(2, 1, 1);
    let w = b.global("w", Dtype::U32);
    let data = b.global("data", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let cta = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let wa = b.reg(Ty::U64);
    let got = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    let arg = b.reg(Ty::U32);
    let res = b.reg(Ty::PRED);
    b.read_special(cta, SpecialReg::CtaLinear);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k5 = b.k_u32(5);
    let k7 = b.k_u32(7);
    b.addr_of(wa, w, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.compare(CmpOp::Eq, Ty::U32, p, cta, k0);
    b.if_(p);
    b.site("data_store", 1);
    st(&mut b, data, k0, k5, Sem::Weak);
    b.site("w_publish", 2);
    st(&mut b, w, k0, k7, if hb { Sem::Release } else { Sem::Weak });
    b.no_site();
    b.else_();
    b.site("tirx.cuda.wait_until", 3);
    let placeholder = b.push(Instr::Nop);
    b.site("data_load", 4);
    b.ld_u32(v, data, k0);
    b.no_site();
    b.add_u32(v, v, got);
    b.st_u32(out, k0, v);
    b.end_if();
    b.end_if();
    b.exit();
    let mut prog = b.build();
    let start = Pc(prog.code.len() as u32);
    prog.code.push(Instr::Compare { op: CmpOp::Eq, ty: Ty::U32, dst: res, a: arg.into(), b: k7 });
    prog.code_sites.push(numsim_core::site::SiteId::NONE);
    prog.preds.push(PredProgram { arg, start, end: Pc(start.0 + 1), result: res, reads_memory: false });
    prog.code[placeholder.0 as usize] =
        Instr::WaitUntil { dst: got, addr: wa.into(), ty: Ty::U32, space: AddrSpace::Generic, sem: Sem::Acquire, scope: Scope::Gpu, pred: PredId(0), captures: vec![] };
    prog.validate().expect("valid");
    Scenario {
        name: if hb { "first_wait_declare_hb" } else { "first_wait_declare_plain" },
        module: Module::new(vec![prog]),
        inputs: inputs(vec![("w", u32_buf([0])), ("data", u32_buf([0])), ("out", u32_buf([0]))]),
        config: RunConfig { loop_budget: 1 << 40, ..RunConfig::default() },
    }
}

/// I10 oracle for words declared mid-run: per declared span, the writes
/// delivered since its `DeclareWord` (overlap, as racecheck numbers them);
/// each `WaitVerdicts.observed` must equal that count at delivery.
#[derive(Default)]
struct DeclaredWordOracle {
    words: Vec<(numsim_core::arena::AllocId, numsim_core::arena::ByteSpan, u32)>,
    mismatches: Vec<String>,
    verdicts: usize,
}

impl numsim_core::observe::Observer for DeclaredWordOracle {
    fn wants_word_history(&self) -> bool {
        true
    }
    fn access(&mut self, a: &numsim_core::observe::Access<'_>) {
        if !a.writes() {
            return;
        }
        for (alloc, span, n) in &mut self.words {
            if *alloc == a.alloc {
                *n += a.spans.iter().filter(|s| s.span.overlaps(*span)).count() as u32;
            }
        }
    }
    fn sync(&mut self, e: &numsim_core::observe::SyncEvent) {
        use numsim_core::observe::SyncKind;
        match &e.kind {
            SyncKind::DeclareWord { alloc, span } => self.words.push((*alloc, *span, 0)),
            SyncKind::WaitVerdicts { alloc, span, verdicts, .. } => {
                let n = self.words.iter().find(|(a, s, _)| a == alloc && s.overlaps(*span)).map(|w| w.2);
                for v in verdicts {
                    self.verdicts += 1;
                    if Some(v.observed) != n {
                        self.mismatches.push(format!("observed {} vs delivered writes {:?}", v.observed, n));
                    }
                }
            }
            _ => {}
        }
    }
}

fn declared_word_numbering(s: &Scenario, workers: usize) -> DeclaredWordOracle {
    let mut o = DeclaredWordOracle::default();
    let cfg = RunConfig { workers, ..s.config.clone() };
    sched::run_with_config(&s.module, &s.inputs, &mut o, &cfg).expect("run starts");
    o
}

#[test]
fn w16_first_wait_declare_serial_is_worker_independent() {
    for hb in [true, false] {
        let (status, _) = serial_is_worker_independent(&first_wait_declare(hb), &RacecheckConfig::default());
        assert_eq!(status, format!("{:?}", RunStatus::Completed));
    }
}

#[test]
fn w16_first_wait_declare_fork_join_matches_serial() {
    for hb in [true, false] {
        fork_join_matches_serial(&first_wait_declare(hb), &RacecheckConfig::default());
    }
}

/// The verdict of a wait whose word was declared in the same round in which
/// another partition wrote it must be numbered like the delivered history.
/// Today the engine's history misses CTA 0's same-round write: the
/// declaration lives only in CTA 1's partition table that round, so the
/// write is not logged. The verdict says `observed 0` while one write has
/// been delivered since the `DeclareWord`, and with `hb` racecheck reports
/// a false DataRace on `data`. This is an engine bug, the same on base and
/// on W16's series (CONTRACT_REQUESTS W6-P2).
#[test]
#[ignore = "xfail: CONTRACT_REQUESTS W6-P2 (W2) -- first-wait declaration misses other partitions' same-round writes"]
fn w16_first_wait_declare_numbering_follows_the_delivery() {
    for hb in [true, false] {
        for w in [1usize, 8] {
            let o = declared_word_numbering(&first_wait_declare(hb), w);
            assert!(o.verdicts > 0, "hb={hb} workers={w}: no verdict");
            assert!(o.mismatches.is_empty(), "hb={hb} workers={w}: {:?}", o.mismatches);
        }
    }
}

