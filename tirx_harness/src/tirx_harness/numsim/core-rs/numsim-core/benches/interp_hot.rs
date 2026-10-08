//! Interpreter hot-path guards (W13): one row per engine-side hot spot found
//! by profiling corpus kernels. `cargo bench -p numsim-core --bench
//! interp_hot` (with a private `CARGO_TARGET_DIR`).
//!
//! Throughput is per executed warp instruction (`RunStats::instrs` of one
//! run), so `time/elem` is the cost of one warp instruction including its
//! share of admission and scheduling.
//!
//! * `spin_wait_regs` — a `try_wait` spin loop parked every round while a
//!   second warp counts down, with a corpus-sized register file (the
//!   `WarpState::spin_hash` fixed-point check at every poll-only `LoopEnd`).
//! * `admit_regs` — many short CTAs with a corpus-sized register file
//!   (register-file zero-fill at admission).
//! * `read_special` — a loop of special-register reads (`tid`, `ntid`,
//!   `ctaid`, lane masks): `alu::read_special`.
//! * `tcgen_ld` — `tcgen05.ld.32x32b.x64` into 64 registers (a GEMM
//!   epilogue's accumulator drain), 4 warps: `tcgen::tcgen_ld`.
//! * `reg_indexed` — a loop reading and writing a 64-element register array
//!   at a warp-uniform index (unrolled-accumulator access):
//!   `alu::load_reg_indexed` / `store_reg_indexed`.
//! * `store_v4` — 16-byte vector stores into shared memory (a GEMM
//!   epilogue's staging): `mem::store`.
//! * `observed_overhead` — the cost of being observed: whole corpus runs
//!   with no observer and with a counting observer (events built and
//!   delivered, no word history, no checker), from the recorded fixtures
//!   (`fp16_bf16_gemm` at 1 worker, `mega_moe_t8_h1024_i512_e24_k2_g1` at its
//!   recorded 16 workers); skipped when missing.
//! * `corpus_numsim` — whole corpus kernels (NumSim mode, 1 worker) from the
//!   recorded fixtures: `examples/record_race_fixtures.py OUT rmsnorm
//!   deepgemm_sm100_fp8_gemm_1d1d fp16_bf16_gemm kda_backward_packed` into `$RACE_FIXTURES` or
//!   `core-rs/target/race-fixtures`; skipped when missing.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use numsim_core::dtype::{Dtype, Ty};
use numsim_core::observe::NoopObserver;
use numsim_core::program::*;
use numsim_core::sched::{self, CompletionPolicy};
use numsim_core::testutil::scenarios::{inputs, u32_buf, Scenario};
use numsim_core::testutil::{fixtures, ProgramBuilder};

fn group(c: &mut Criterion, name: &str, row: &str, s: &Scenario) {
    let probe = sched::run_with_config(&s.module, &s.inputs, &mut NoopObserver, &s.config).unwrap();
    assert_eq!(probe.status, sched::RunStatus::Completed, "{}", s.name);
    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Elements(probe.stats.instrs));
    g.sample_size(20);
    g.bench_function(row, |b| b.iter(|| sched::run_with_config(&s.module, &s.inputs, &mut NoopObserver, &s.config).unwrap()));
    g.finish();
}

/// `pad` extra 32-bit registers, each written once (a corpus kernel's
/// register file is mostly SSA temporaries).
fn pad_regs(b: &mut ProgramBuilder, pad: u32, src: Reg) {
    for _ in 0..pad {
        let r = b.reg(Ty::U32);
        b.mov(r, src);
    }
}

/// Warp 1 runs `iters` iterations of register work, then lane 0 arrives on
/// an mbarrier; warp 0 spins on `try_wait` meanwhile (spin-parked each
/// round). `pad` extra registers size the register file.
pub fn spin_wait(pad: u32, iters: u32) -> Scenario {
    let mut b = ProgramBuilder::new("spin_wait", 64);
    let out = b.global("out", Dtype::U32);
    let bar = b.shared("bar", Dtype::U64, 1);
    let w = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let barr = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let done = b.reg(Ty::PRED);
    let notdone = b.reg(Ty::PRED);
    let k = b.reg(Ty::U32);
    let acc = b.reg(Ty::U32);
    b.warp_id(w);
    b.lane_id(lane);
    pad_regs(&mut b, pad, lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k3 = b.k_u32(3);
    let kn = b.k_u32(iters);
    b.smem_addr(barr, bar, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.compare(CmpOp::Eq, Ty::U32, done, w, k0);
    b.if_(done);
    b.if_(p);
    b.mbar_init(barr, 1);
    b.end_if();
    b.end_if();
    b.bar_sync(0);
    b.if_(done);
    let f = b.konst(Ty::PRED, 0);
    b.mov(done, f);
    b.loop_begin();
    b.push(Instr::Unary { op: UnOp::Not, ty: Ty::PRED, dst: notdone, a: done.into() });
    b.loop_if(notdone);
    b.push(Instr::MbarTestWait {
        kind: WaitKind::Try,
        mbar: barr.into(),
        space: AddrSpace::Shared,
        phase: PhaseArg::Parity(k0),
        sem: Sem::Acquire,
        scope: Scope::Cta,
        dst: Some(done),
        report: None,
        report_value: None,
    });
    b.loop_end();
    b.st_u32(out, lane, lane);
    b.else_();
    b.mov(k, k0);
    b.mov(acc, lane);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, notdone, k, kn);
    b.loop_if(notdone);
    b.mul(Ty::U32, acc, acc, k3);
    b.add_u32(k, k, k1);
    b.loop_end();
    b.if_(p);
    b.mbar_arrive(barr, None);
    b.end_if();
    b.end_if();
    b.exit();
    let mut s = Scenario {
        name: "spin_wait",
        module: b.build_module(),
        inputs: inputs(vec![("out", u32_buf([0; 64]))]),
        config: Default::default(),
    };
    s.config.completions = CompletionPolicy::Eager;
    s
}

/// `ctas` CTAs of 128 threads, each a handful of instructions, with `pad`
/// extra registers.
pub fn admit(ctas: u32, pad: u32) -> Scenario {
    let mut b = ProgramBuilder::new("admit", 128);
    b.grid(ctas, 1, 1);
    let out = b.global("out", Dtype::U32);
    let t = b.reg(Ty::U32);
    let c = b.reg(Ty::U32);
    let i = b.reg(Ty::U32);
    b.thread_rank(t);
    b.read_special(c, SpecialReg::CtaId(Axis::X));
    pad_regs(&mut b, pad.min(8), t);
    for _ in 8..pad {
        b.reg(Ty::U32);
    }
    let k128 = b.k_u32(128);
    b.mul(Ty::U32, i, c, k128);
    b.add_u32(i, i, t);
    b.st_u32(out, i, t);
    b.exit();
    Scenario {
        name: "admit",
        module: b.build_module(),
        inputs: inputs(vec![("out", u32_buf(vec![0; ctas as usize * 128]))]),
        config: Default::default(),
    }
}

/// 256 threads (8 warps, 2-D block 64x4) in 8 CTAs of a 2-CTA cluster
/// grid, `iters` loop iterations each reading a mix of special registers.
pub fn read_special(iters: u32) -> Scenario {
    let mut b = ProgramBuilder::new("read_special", 256);
    b.block(64, 4, 1);
    b.grid(8, 1, 1);
    b.cluster(2, 1, 1);
    let out = b.global("out", Dtype::U32);
    let k = b.reg(Ty::U32);
    let acc = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let t = b.reg(Ty::U32);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let kn = b.k_u32(iters);
    b.mov(k, k0);
    b.mov(acc, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, k, kn);
    b.loop_if(p);
    for s in [
        SpecialReg::Tid(Axis::X),
        SpecialReg::Tid(Axis::Y),
        SpecialReg::NTid(Axis::X),
        SpecialReg::CtaId(Axis::X),
        SpecialReg::LaneId,
        SpecialReg::WarpInCta,
        SpecialReg::ClusterCtaRank,
        SpecialReg::LaneMaskLt,
    ] {
        b.read_special(v, s);
        b.add_u32(acc, acc, v);
    }
    b.add_u32(k, k, k1);
    b.loop_end();
    b.thread_rank(t);
    let c = b.reg(Ty::U32);
    b.read_special(c, SpecialReg::CtaLinear);
    let k256 = b.k_u32(256);
    b.mul(Ty::U32, c, c, k256);
    b.add_u32(t, t, c);
    b.st_u32(out, t, acc);
    b.exit();
    let mut s = Scenario {
        name: "read_special",
        module: b.build_module(),
        inputs: inputs(vec![("out", u32_buf(vec![0; 8 * 256]))]),
        config: Default::default(),
    };
    s.config.completions = CompletionPolicy::Eager;
    s
}

/// 4 warps; each stores `tid` into its 32 lanes x 64 columns of TMEM once,
/// then loads them back `iters` times with `tcgen05.ld.32x32b.x64` and
/// stores one register.
pub fn tcgen_ld(iters: u32) -> Scenario {
    let mut b = ProgramBuilder::new("tcgen_ld_x64", 128);
    let out = b.global("out", Dtype::U32);
    let slot = b.shared("taddr", Dtype::U32, 1);
    let tid = b.reg(Ty::U32);
    let w = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let sa = b.reg(Ty::U32);
    let t = b.reg(Ty::U32);
    let t2 = b.reg(Ty::U32);
    let lanebits = b.reg(Ty::U32);
    let k = b.reg(Ty::U32);
    let v: Vec<Reg> = (0..64).map(|_| b.reg(Ty::U32)).collect();
    b.thread_rank(tid);
    b.warp_id(w);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k32 = b.k_u32(32);
    let k64 = b.k_u32(64);
    let k16 = b.k_u32(16);
    let kn = b.k_u32(iters);
    b.smem_addr(sa, slot, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, w, k0);
    b.if_(p);
    b.push(Instr::TcgenAlloc { dst: sa.into(), ncols: k64, cta_group: 1, exclusive: false });
    b.end_if();
    b.bar_sync(0);
    b.ld_u32(t, slot, k0);
    b.mul(Ty::U32, lanebits, w, k32);
    b.binary(BinOp::Shl, Ty::U32, lanebits, lanebits, k16);
    b.add_u32(t2, t, lanebits);
    b.push(Instr::TcgenSt(Box::new(TcgenStArgs {
        srcs: vec![tid.into(); 64],
        taddr: t2.into(),
        row: k0,
        col: k0,
        shape: TcShape::S32x32b,
        num: 64,
        unpack: false,
    })));
    b.push(Instr::TcgenWait { st: true });
    b.mov(k, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, k, kn);
    b.loop_if(p);
    b.push(Instr::TcgenLd(Box::new(TcgenLdArgs {
        dsts: v.clone(),
        taddr: t2.into(),
        row: k0,
        col: k0,
        shape: TcShape::S32x32b,
        num: 64,
        pack: false,
        red: None,
        red_abs: false,
        red_nan: false,
        spcompress: false,
    })));
    b.push(Instr::TcgenWait { st: false });
    b.add_u32(k, k, k1);
    b.loop_end();
    b.st_u32(out, tid, v[63]);
    b.bar_sync(0);
    b.compare(CmpOp::Eq, Ty::U32, p, w, k0);
    b.if_(p);
    b.push(Instr::TcgenDealloc { taddr: t.into(), ncols: k64, cta_group: 1, exclusive: false });
    b.push(Instr::TcgenRelinquish { cta_group: 1 });
    b.end_if();
    b.exit();
    Scenario { name: "tcgen_ld_x64", module: b.build_module(), inputs: inputs(vec![("out", u32_buf([0; 128]))]), config: Default::default() }
}

/// 4 warps, `iters` iterations of `a[k % 64] += a[(k + 1) % 64]` over a
/// 64-register array.
pub fn reg_indexed(iters: u32) -> Scenario {
    let mut b = ProgramBuilder::new("reg_indexed", 128);
    let out = b.global("out", Dtype::U32);
    let tid = b.reg(Ty::U32);
    let k = b.reg(Ty::U32);
    let i = b.reg(Ty::U32);
    let j = b.reg(Ty::U32);
    let x = b.reg(Ty::U32);
    let y = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let arr: Vec<Reg> = (0..64).map(|_| b.reg(Ty::U32)).collect();
    b.thread_rank(tid);
    for &r in &arr {
        b.mov(r, tid);
    }
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k63 = b.k_u32(63);
    let kn = b.k_u32(iters);
    b.mov(k, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, k, kn);
    b.loop_if(p);
    b.binary(BinOp::And, Ty::U32, i, k, k63);
    b.add_u32(j, k, k1);
    b.binary(BinOp::And, Ty::U32, j, j, k63);
    b.push(Instr::LoadRegIndexed { dst: x, base: arr[0], len: 64, idx: i.into() });
    b.push(Instr::LoadRegIndexed { dst: y, base: arr[0], len: 64, idx: j.into() });
    b.add_u32(x, x, y);
    b.push(Instr::StoreRegIndexed { base: arr[0], len: 64, idx: i.into(), value: x.into() });
    b.add_u32(k, k, k1);
    b.loop_end();
    b.st_u32(out, tid, arr[5]);
    b.exit();
    let mut s = Scenario { name: "reg_indexed", module: b.build_module(), inputs: inputs(vec![("out", u32_buf([0; 128]))]), config: Default::default() };
    s.config.completions = CompletionPolicy::Eager;
    s
}

/// 4 warps, `iters` iterations of a `u32x4` store into shared memory.
pub fn store_v4(iters: u32) -> Scenario {
    let v4 = Ty::vector(Dtype::U32, 4);
    let mut b = ProgramBuilder::new("store_v4", 128);
    let inp = b.global("inp", Dtype::U32);
    let sh = b.shared("sh", Dtype::U32, 128 * 4);
    let tid = b.reg(Ty::U32);
    let i4 = b.reg(Ty::U32);
    let k = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let v = b.reg(v4);
    b.thread_rank(tid);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k4 = b.k_u32(4);
    let kn = b.k_u32(iters);
    b.mul(Ty::U32, i4, tid, k4);
    b.ld(v4, v, inp, i4);
    b.mov(k, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, k, kn);
    b.loop_if(p);
    b.st(v4, sh, i4, v);
    b.add_u32(k, k, k1);
    b.loop_end();
    b.exit();
    let mut s = Scenario { name: "store_v4", module: b.build_module(), inputs: inputs(vec![("inp", u32_buf(0..512))]), config: Default::default() };
    s.config.completions = CompletionPolicy::Eager;
    s
}

fn bench(c: &mut Criterion) {
    group(c, "spin_wait_regs", "pad768_iters4096", &spin_wait(768, 4096));
    // A kda_backward_packed-sized register file (~8 MiB per warp).
    group(c, "spin_wait_regs", "pad32768_iters1024", &spin_wait(32768, 1024));
    group(c, "admit_regs", "ctas64_pad256", &admit(64, 256));
    group(c, "read_special", "iters256", &read_special(256));
    group(c, "tcgen_ld", "x64_iters64", &tcgen_ld(64));
    group(c, "reg_indexed", "iters2048", &reg_indexed(2048));
    group(c, "store_v4", "iters2048", &store_v4(2048));
}

/// Events on, history off, no checker: only the cost of building and
/// delivering the event stream.
#[derive(Default)]
struct Counting {
    events: u64,
}
impl numsim_core::observe::Observer for Counting {
    fn access(&mut self, a: &numsim_core::observe::Access<'_>) {
        self.events += 1 + a.spans.len() as u64;
    }
    fn sync(&mut self, _e: &numsim_core::observe::SyncEvent) {
        self.events += 1;
    }
}

fn observed_overhead(c: &mut Criterion) {
    let dir = fixtures::dir();
    for (case, workers) in [("fp16_bf16_gemm", Some(1)), ("mega_moe_t8_h1024_i512_e24_k2_g1", None)] {
        if !fixtures::exists(&dir, case) {
            eprintln!("observed_overhead: fixture {dir}/{case}.* missing, skipped");
            continue;
        }
        let (module, inputs, mut config) = fixtures::load(&dir, case);
        if let Some(w) = workers {
            config.workers = w;
        }
        let mut g = c.benchmark_group("observed_overhead");
        g.sample_size(10);
        g.bench_function(format!("{case}/noop"), |b| b.iter(|| sched::run_with_config(&module, &inputs, &mut NoopObserver, &config).unwrap()));
        g.bench_function(format!("{case}/counting"), |b| {
            b.iter(|| {
                let mut o = Counting::default();
                sched::run_with_config(&module, &inputs, &mut o, &config).unwrap();
                o.events
            })
        });
        g.finish();
    }
}

fn corpus(c: &mut Criterion) {
    let dir = fixtures::dir();
    for case in ["rmsnorm", "deepgemm_sm100_fp8_gemm_1d1d", "fp16_bf16_gemm", "kda_backward_packed"] {
        if !fixtures::exists(&dir, case) {
            eprintln!("corpus_numsim: fixture {dir}/{case}.* missing, skipped");
            continue;
        }
        let (module, inputs, mut config) = fixtures::load(&dir, case);
        config.workers = 1;
        let s = Scenario { name: "corpus", module, inputs, config };
        group(c, "corpus_numsim", case, &s);
    }
}

criterion_group!(benches, bench, corpus, observed_overhead);
criterion_main!(benches);
