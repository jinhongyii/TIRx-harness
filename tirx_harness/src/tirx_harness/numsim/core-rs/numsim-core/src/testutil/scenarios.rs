//! Hand-built interpreter/scheduler scenarios (W2), shared with the codegen
//! differential suite (`tests/codegen_equivalence.rs`) and the benches.
//!
//! Each [`Scenario`] is a module, its inputs and the run configuration its
//! expectations assume. The expectations themselves live in
//! `tests/interp_*.rs`; [`all`] lists every scenario for differential runs.

use crate::arena::BitSet;
use crate::dtype::{Dtype, Ty};
use crate::program::*;
use crate::sched::{ArgValue, CompletionPolicy, Inputs, RunConfig};
use crate::sync::async_group::Domain;
use crate::testutil::ProgramBuilder;
use crate::Module;

/// A runnable test case.
pub struct Scenario {
    pub name: &'static str,
    pub module: Module,
    pub inputs: Inputs,
    pub config: RunConfig,
}

pub fn f32_buf(v: impl IntoIterator<Item = f32>) -> ArgValue {
    ArgValue::Buffer { bytes: v.into_iter().flat_map(|x| x.to_le_bytes()).collect(), valid: None }
}

pub fn u32_buf(v: impl IntoIterator<Item = u32>) -> ArgValue {
    ArgValue::Buffer { bytes: v.into_iter().flat_map(|x| x.to_le_bytes()).collect(), valid: None }
}

/// A buffer of `n` u32 whose bytes are all uninitialized.
pub fn uninit_buf(n: usize) -> ArgValue {
    ArgValue::Buffer { bytes: vec![0; n * 4], valid: Some(BitSet::new(n as u64 * 4, false)) }
}

pub fn inputs(args: Vec<(&str, ArgValue)>) -> Inputs {
    Inputs { args: args.into_iter().map(|(k, v)| (k.to_string(), v)).collect() }
}

fn scenario(name: &'static str, module: Module, inputs: Inputs) -> Scenario {
    Scenario { name, module, inputs, config: RunConfig::default() }
}

/// Set `elect` on every `If` whose condition is `cond`.
fn mark_elect(p: &mut Program, cond: Reg) {
    for ins in &mut p.code {
        if let Instr::If { cond: Operand::Reg(c), elect, .. } = ins {
            if *c == cond {
                *elect = true;
            }
        }
    }
}

/// `tid & mask == 0` predicate.
fn bit_clear(b: &mut ProgramBuilder, dst: Reg, v: Reg, mask: u32) {
    let t = b.reg(Ty::U32);
    let m = b.k_u32(mask);
    b.binary(BinOp::And, Ty::U32, t, v, m);
    let z = b.k_u32(0);
    b.compare(CmpOp::Eq, Ty::U32, dst, t, z);
}

/// W1's vector add (`tests/numsim/v2/test_lowering_vector_add.py`):
/// 8 CTAs x 128 threads, `i = bx*128 + tx`, `if i < n: c[i] = a[i] + b[i]`
/// with `n = 1000` (the guard is exercised) over 1024-element buffers.
pub const VADD_N: u32 = 1000;

pub fn vector_add_program() -> Program {
    let mut b = ProgramBuilder::new("vadd", 128);
    b.grid(8, 1, 1);
    let a = b.global("a", Dtype::F32);
    let bb = b.global("b", Dtype::F32);
    let c = b.global("c", Dtype::F32);
    let bx = b.reg(Ty::S32);
    let tx = b.reg(Ty::S32);
    let m = b.reg(Ty::S32);
    let s = b.reg(Ty::S32);
    let i = b.reg(Ty::S32);
    let p = b.reg(Ty::PRED);
    let va = b.reg(Ty::F32);
    let vb = b.reg(Ty::F32);
    let vc = b.reg(Ty::F32);
    b.read_special(bx, SpecialReg::CtaLinear);
    b.read_special(tx, SpecialReg::ThreadInCta);
    let k128 = b.k_i32(128);
    b.binary(BinOp::Mul, Ty::S32, m, bx, k128);
    b.binary(BinOp::Add, Ty::S32, s, m, tx);
    b.mov(i, s);
    let kn = b.k_i32(VADD_N as i32);
    b.compare(CmpOp::Lt, Ty::S32, p, i, kn);
    b.if_(p);
    b.site("ir.TensorLoad", 10);
    b.ld_f32(va, a, i);
    b.ld_f32(vb, bb, i);
    b.no_site();
    b.add_f32(vc, va, vb);
    b.site("tirx.BufferStore", 10);
    b.st_f32(c, i, vc);
    b.end_if();
    b.no_site();
    b.exit();
    b.build()
}

pub fn vector_add_inputs() -> Inputs {
    inputs(vec![
        ("a", f32_buf((0..1024).map(|v| v as f32 * 0.5))),
        ("b", f32_buf((0..1024).map(|_| 2.0))),
        ("c", f32_buf((0..1024).map(|_| -1.0))),
    ])
}

pub fn vector_add() -> Scenario {
    scenario("vector_add", Module::new(vec![vector_add_program()]), vector_add_inputs())
}

/// Divergent if/else with a nested if and reconvergence:
/// `r = even(tid) ? tid*2 (+1000 if tid<16) : tid+100; out[tid] = r + 1`.
pub fn divergent_if_else() -> Scenario {
    let mut b = ProgramBuilder::new("divergent", 64);
    let out = b.global("out", Dtype::U32);
    let tid = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let q = b.reg(Ty::PRED);
    let r = b.reg(Ty::U32);
    b.thread_rank(tid);
    bit_clear(&mut b, p, tid, 1);
    b.if_(p);
    let k2 = b.k_u32(2);
    b.mul(Ty::U32, r, tid, k2);
    let k16 = b.k_u32(16);
    b.compare(CmpOp::Lt, Ty::U32, q, tid, k16);
    b.if_(q);
    let k1000 = b.k_u32(1000);
    b.add_u32(r, r, k1000);
    b.end_if();
    b.else_();
    let k100 = b.k_u32(100);
    b.add_u32(r, tid, k100);
    b.end_if();
    let k1 = b.k_u32(1);
    b.add_u32(r, r, k1);
    b.st_u32(out, tid, r);
    b.exit();
    scenario("divergent_if_else", b.build_module(), inputs(vec![("out", u32_buf([0; 64]))]))
}

pub fn divergent_expected(tid: u32) -> u32 {
    let r = if tid % 2 == 0 { tid * 2 + if tid < 16 { 1000 } else { 0 } } else { tid + 100 };
    r + 1
}

/// Nested loops with continue and break (per lane):
/// ```text
/// acc = 0; i = 0
/// while i < 8 { j0 = i; i += 1
///   if j0 == lane % 5 { continue }
///   j = 0; while j < 8 { jj = j; j += 1; if jj > j0 { break }; acc += jj }
///   if acc > 40 { break } }
/// out[lane] = acc
/// ```
pub fn nested_loops() -> Scenario {
    let mut b = ProgramBuilder::new("nested_loops", 32);
    let out = b.global("out", Dtype::U32);
    let lane = b.reg(Ty::U32);
    let acc = b.reg(Ty::U32);
    let i = b.reg(Ty::U32);
    let j0 = b.reg(Ty::U32);
    let j = b.reg(Ty::U32);
    let jj = b.reg(Ty::U32);
    let m5 = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k5 = b.k_u32(5);
    let k8 = b.k_u32(8);
    let k40 = b.k_u32(40);
    b.mov(acc, k0);
    b.mov(i, k0);
    b.binary(BinOp::Mod, Ty::U32, m5, lane, k5);
    b.site("outer_loop", 1);
    b.loop_begin();
    b.no_site();
    b.compare(CmpOp::Lt, Ty::U32, p, i, k8);
    b.loop_if(p);
    b.mov(j0, i);
    b.add_u32(i, i, k1);
    b.compare(CmpOp::Eq, Ty::U32, p, j0, m5);
    b.if_(p);
    b.continue_();
    b.end_if();
    b.mov(j, k0);
    b.site("inner_loop", 2);
    b.loop_begin();
    b.no_site();
    b.compare(CmpOp::Lt, Ty::U32, p, j, k8);
    b.loop_if(p);
    b.mov(jj, j);
    b.add_u32(j, j, k1);
    b.compare(CmpOp::Gt, Ty::U32, p, jj, j0);
    b.if_(p);
    b.break_();
    b.end_if();
    b.add_u32(acc, acc, jj);
    b.loop_end();
    b.compare(CmpOp::Gt, Ty::U32, p, acc, k40);
    b.if_(p);
    b.break_();
    b.end_if();
    b.loop_end();
    b.st_u32(out, lane, acc);
    b.exit();
    scenario("nested_loops", b.build_module(), inputs(vec![("out", u32_buf([0; 32]))]))
}

pub fn nested_loops_expected(lane: u32) -> u32 {
    let mut acc = 0u32;
    let mut i = 0u32;
    while i < 8 {
        let j0 = i;
        i += 1;
        if j0 == lane % 5 {
            continue;
        }
        let mut j = 0;
        while j < 8 {
            let jj = j;
            j += 1;
            if jj > j0 {
                break;
            }
            acc += jj;
        }
        if acc > 40 {
            break;
        }
    }
    acc
}

/// Two warps: warp 0's elected lane arms an mbarrier with
/// `arrive.expect_tx(128)` and issues a 128-byte bulk copy global->shared
/// completing on it (complete_tx); warp 1 waits on parity 0, then copies
/// shared -> `out`.
pub fn mbarrier_producer_consumer() -> Scenario {
    let mut b = ProgramBuilder::new("mbar_pc", 64);
    let input = b.global("in", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let bar = b.shared("bar", Dtype::U64, 1);
    let data = b.shared("data", Dtype::U32, 32);
    let tid = b.reg(Ty::U32);
    let w = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let e = b.reg(Ty::PRED);
    let barr = b.reg(Ty::U32);
    let daddr = b.reg(Ty::U32);
    let gaddr = b.reg(Ty::U64);
    let v = b.reg(Ty::U32);
    b.thread_rank(tid);
    b.warp_id(w);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    b.smem_addr(barr, bar, k0);
    b.smem_addr(daddr, data, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, tid, k0);
    b.site("mbar_init", 1);
    b.if_(p);
    b.mbar_init(barr, 1);
    b.end_if();
    b.fence(FenceKind::MbarrierInit, Sem::Release, Scope::Cluster);
    b.bar_sync(0);
    b.compare(CmpOp::Eq, Ty::U32, p, w, k0);
    b.if_(p);
    let full = b.k_u32(u32::MAX);
    b.push(Instr::Elect { dst_pred: e, dst_lane: None, membermask: full });
    b.if_(e);
    b.site("arrive_expect_tx", 2);
    let k128 = b.k_u32(128);
    b.push(Instr::MbarArrive(MbarArriveArgs {
        mbar: barr.into(),
        space: AddrSpace::Shared,
        count: None,
        expect_tx: Some(k128),
        drop: false,
        no_complete: false,
        sem: Sem::Release,
        scope: Scope::Cta,
        multicast: None,
        state: None,
    }));
    b.site("bulk_copy", 3);
    b.addr_of(gaddr, input, k0);
    b.push(Instr::BulkCopy(BulkCopyArgs {
        dst: daddr.into(),
        dst_space: AddrSpace::SharedCluster,
        src: gaddr.into(),
        src_space: AddrSpace::Global,
        size: k128,
        completion: BulkCompletion::Mbarrier { mbar: barr.into(), space: AddrSpace::Shared },
        multicast: None,
        reduce: None,
        byte_mask: None,
        ignore_oob: None,
        report: None,
        mods: MemMods::default(),
    }));
    b.end_if();
    b.else_();
    b.site("consumer_wait", 4);
    b.mbar_wait_parity(barr, k0);
    b.site("consumer_copy", 5);
    b.ld_u32(v, data, lane);
    b.st_u32(out, lane, v);
    b.end_if();
    b.exit();
    let mut prog = b.build();
    mark_elect(&mut prog, e);
    scenario(
        "mbarrier_producer_consumer",
        Module::new(vec![prog]),
        inputs(vec![("in", u32_buf((0..32).map(|x| x * 7 + 3))), ("out", u32_buf([0; 32]))]),
    )
}

/// Named barriers: warps 0-1 fill shared memory and `bar.arrive(1, 128)`;
/// warps 2-3 `bar.sync(1, 128)` then copy it out. Then every thread joins
/// `bar.red.popc(2)` over `tid < 50` and stores the count.
pub fn named_barrier() -> Scenario {
    let mut b = ProgramBuilder::new("named_barrier", 128);
    let out = b.global("out", Dtype::U32);
    let cnt = b.global("cnt", Dtype::U32);
    let s = b.shared("s", Dtype::U32, 64);
    let tid = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let v = b.reg(Ty::U32);
    let t2 = b.reg(Ty::U32);
    let r = b.reg(Ty::U32);
    b.thread_rank(tid);
    let k64 = b.k_u32(64);
    let k3 = b.k_u32(3);
    let k1 = b.k_u32(1);
    let k128 = b.k_u32(128);
    b.compare(CmpOp::Lt, Ty::U32, p, tid, k64);
    b.if_(p);
    b.mul(Ty::U32, v, tid, k3);
    b.st_u32(s, tid, v);
    b.site("bar_arrive", 1);
    b.push(Instr::Barrier { kind: BarKind::Arrive, id: k1, count: Some(k128), aligned: true });
    b.else_();
    b.site("bar_sync", 2);
    b.push(Instr::Barrier { kind: BarKind::Sync, id: k1, count: Some(k128), aligned: true });
    b.no_site();
    b.binary(BinOp::Sub, Ty::U32, t2, tid, k64);
    b.ld_u32(v, s, t2);
    b.st_u32(out, t2, v);
    b.end_if();
    let k50 = b.k_u32(50);
    b.compare(CmpOp::Lt, Ty::U32, p, tid, k50);
    let k2 = b.k_u32(2);
    b.site("bar_red", 3);
    b.push(Instr::Barrier { kind: BarKind::Red { op: BarRedOp::Popc, pred: p.into(), dst: r }, id: k2, count: None, aligned: true });
    b.st_u32(cnt, tid, r);
    b.exit();
    scenario("named_barrier", b.build_module(), inputs(vec![("out", u32_buf([0; 64])), ("cnt", u32_buf([0; 128]))]))
}

/// One elected lane per warp (`If{elect}`) does `red.add(counter, 1)`;
/// every lane stores the elected lane index.
pub fn elect_region() -> Scenario {
    let mut b = ProgramBuilder::new("elect_region", 128);
    let counter = b.global("counter", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let tid = b.reg(Ty::U32);
    let e = b.reg(Ty::PRED);
    let leader = b.reg(Ty::U32);
    let a = b.reg(Ty::U64);
    b.thread_rank(tid);
    let full = b.k_u32(u32::MAX);
    b.push(Instr::Elect { dst_pred: e, dst_lane: Some(leader), membermask: full });
    b.if_(e);
    let k0 = b.k_u32(0);
    b.addr_of(a, counter, k0);
    let k1 = b.k_u32(1);
    b.site("red_add", 1);
    b.red_add(Ty::U32, a, k1);
    b.end_if();
    b.st_u32(out, tid, leader);
    b.exit();
    let mut prog = b.build();
    mark_elect(&mut prog, e);
    scenario("elect_region", Module::new(vec![prog]), inputs(vec![("counter", u32_buf([0])), ("out", u32_buf([7; 128]))]))
}

/// An infinite loop; `loop_budget = 1000` -> Incomplete at the loop site.
pub fn loop_budget() -> Scenario {
    let mut b = ProgramBuilder::new("loop_budget", 32);
    let t = b.konst(Ty::PRED, 1);
    b.site("spin", 7);
    b.loop_begin();
    b.no_site();
    b.loop_if(t);
    b.loop_end();
    b.exit();
    let mut s = scenario("loop_budget", b.build_module(), Inputs::default());
    s.config.loop_budget = 1000;
    s
}

/// Waits on parity 0 of an mbarrier nobody arrives on: Deadlock.
pub fn deadlock_wait() -> Scenario {
    let mut b = ProgramBuilder::new("deadlock_wait", 32);
    let bar = b.shared("bar", Dtype::U64, 1);
    let barr = b.reg(Ty::U32);
    let k0 = b.k_u32(0);
    b.smem_addr(barr, bar, k0);
    b.mbar_init(barr, 1);
    b.site("wait", 1);
    b.mbar_wait_parity(barr, k0);
    b.exit();
    scenario("deadlock_wait", b.build_module(), Inputs::default())
}

/// A `try_wait` spin loop on a barrier nobody arrives on: the loop is
/// spin-parked, so the result is Deadlock (not loop-budget exhaustion).
pub fn deadlock_spin() -> Scenario {
    let mut b = ProgramBuilder::new("deadlock_spin", 32);
    let bar = b.shared("bar", Dtype::U64, 1);
    let barr = b.reg(Ty::U32);
    let done = b.reg(Ty::PRED);
    let notdone = b.reg(Ty::PRED);
    let k0 = b.k_u32(0);
    b.smem_addr(barr, bar, k0);
    b.mbar_init(barr, 1);
    let f = b.konst(Ty::PRED, 0);
    b.mov(done, f);
    b.site("spin", 1);
    b.loop_begin();
    b.no_site();
    b.push(Instr::Unary { op: UnOp::Not, ty: Ty::PRED, dst: notdone, a: done.into() });
    b.loop_if(notdone);
    b.site("try_wait", 2);
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
    b.exit();
    let mut s = scenario("deadlock_spin", b.build_module(), Inputs::default());
    s.config.loop_budget = 1 << 40;
    s
}

/// Reads shared memory nobody wrote and stores it.
pub fn uninit_read() -> Scenario {
    let mut b = ProgramBuilder::new("uninit_read", 32);
    let out = b.global("out", Dtype::U32);
    let s = b.shared("s", Dtype::U32, 32);
    let lane = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    b.lane_id(lane);
    b.site("uninit_ld", 1);
    b.ld_u32(v, s, lane);
    b.st_u32(out, lane, v);
    b.exit();
    scenario("uninit_read", b.build_module(), inputs(vec![("out", u32_buf([9; 32]))]))
}

/// `x[lane]` on a 16-element buffer: lanes 16.. are out of bounds.
pub fn oob() -> Scenario {
    let mut b = ProgramBuilder::new("oob", 32);
    let x = b.global("x", Dtype::F32);
    let y = b.global("y", Dtype::F32);
    let lane = b.reg(Ty::U32);
    let v = b.reg(Ty::F32);
    b.lane_id(lane);
    b.site("oob_ld", 1);
    b.ld_f32(v, x, lane);
    b.st_f32(y, lane, v);
    b.exit();
    scenario("oob", b.build_module(), inputs(vec![("x", f32_buf([1.0; 16])), ("y", f32_buf([0.0; 32]))]))
}

/// `if lane == 0 { mbarrier.wait(b) } else { mbarrier.arrive(b) }` with an
/// expected count of 31: completes only under the divergent scheduling
/// rule (the else arm runs while lane 0 waits).
pub fn divergent_wait_arrive() -> Scenario {
    let mut b = ProgramBuilder::new("divergent_wait_arrive", 32);
    let out = b.global("out", Dtype::U32);
    let bar = b.shared("bar", Dtype::U64, 1);
    let barr = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let k0 = b.k_u32(0);
    b.lane_id(lane);
    b.smem_addr(barr, bar, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.mbar_init(barr, 31);
    b.end_if();
    b.if_(p);
    b.site("wait", 1);
    b.mbar_wait_parity(barr, k0);
    let k1 = b.k_u32(1);
    b.st_u32(out, lane, k1);
    b.else_();
    b.site("arrive", 2);
    b.mbar_arrive(barr, None);
    let k2 = b.k_u32(2);
    b.st_u32(out, lane, k2);
    b.end_if();
    b.exit();
    scenario("divergent_wait_arrive", b.build_module(), inputs(vec![("out", u32_buf([0; 32]))]))
}

/// cp.async 16 bytes per lane global -> shared, commit, wait_all, then
/// copy shared -> out (per-lane async groups).
pub fn cp_async_copy() -> Scenario {
    let mut b = ProgramBuilder::new("cp_async_copy", 32);
    let input = b.global("in", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let s = b.shared("s", Dtype::U32, 128);
    let lane = b.reg(Ty::U32);
    let idx = b.reg(Ty::U32);
    let saddr = b.reg(Ty::U32);
    let gaddr = b.reg(Ty::U64);
    let v = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    b.lane_id(lane);
    let k4 = b.k_u32(4);
    b.mul(Ty::U32, idx, lane, k4);
    b.smem_addr(saddr, s, idx);
    b.addr_of(gaddr, input, idx);
    b.site("cp_async", 1);
    b.push(Instr::CpAsync { dst: saddr.into(), src: gaddr.into(), cp_size: 16, src_size: None, ignore_src: None, mods: MemMods::default() });
    b.cp_async_commit_wait_all();
    b.no_site();
    // Each lane reads back its own 4 words.
    let k1 = b.k_u32(1);
    let j = b.reg(Ty::U32);
    let k0 = b.k_u32(0);
    b.mov(j, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, j, k4);
    b.loop_if(p);
    let at = b.reg(Ty::U32);
    b.add_u32(at, idx, j);
    b.ld_u32(v, s, at);
    b.st_u32(out, at, v);
    b.add_u32(j, j, k1);
    b.loop_end();
    b.exit();
    let _ = Domain::CpAsync;
    scenario("cp_async_copy", b.build_module(), inputs(vec![("in", u32_buf((0..128).map(|x| x ^ 0x55))), ("out", u32_buf([0; 128]))]))
}

/// Warp 1 publishes `flag = 1` (after storing `data`); warp 0 blocks in
/// `wait_until(flag == 1)` then reads `data`. Declares the flag word.
pub fn wait_until_flag() -> Scenario {
    let mut b = ProgramBuilder::new("wait_until_flag", 64);
    let flag = b.global("flag", Dtype::U32);
    let data = b.global("data", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    b.declare_sync_words(flag);
    let w = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let fa = b.reg(Ty::U64);
    let got = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    let arg = b.reg(Ty::U32);
    let res = b.reg(Ty::PRED);
    b.warp_id(w);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    b.addr_of(fa, flag, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, w, k1);
    b.if_(p);
    let k42 = b.k_u32(42);
    b.st_u32(data, lane, k42);
    // Order every lane's data store before lane 0's release (lanes of a
    // warp are otherwise unordered).
    let full = b.k_u32(u32::MAX);
    b.push(Instr::WarpSync { membermask: full });
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.site("publish", 1);
    b.push(Instr::StoreAddr {
        ty: Ty::U32,
        addr: fa.into(),
        space: AddrSpace::Generic,
        value: k1,
        sem: Sem::Release,
        scope: Scope::Gpu,
        mods: MemMods::default(),
    });
    b.end_if();
    b.else_();
    b.site("wait_until", 2);
    let placeholder = b.push(Instr::Nop);
    b.no_site();
    b.ld_u32(v, data, lane);
    b.add_u32(v, v, got);
    b.st_u32(out, lane, v);
    b.end_if();
    b.exit();
    let mut prog = b.build();
    // Predicate sub-program `arg == 1`, placed after the body.
    let start = Pc(prog.code.len() as u32);
    let k1c = match k1 {
        Operand::Const(c) => c,
        _ => unreachable!(),
    };
    prog.code.push(Instr::Compare { op: CmpOp::Eq, ty: Ty::U32, dst: res, a: arg.into(), b: Operand::Const(k1c) });
    prog.code_sites.push(crate::site::SiteId::NONE);
    prog.preds.push(PredProgram { arg, start, end: Pc(start.0 + 1), result: res, reads_memory: false });
    prog.code[placeholder.0 as usize] = Instr::WaitUntil {
        dst: got,
        addr: fa.into(),
        ty: Ty::U32,
        space: AddrSpace::Generic,
        sem: Sem::Acquire,
        scope: Scope::Gpu,
        pred: PredId(0),
        captures: vec![],
    };
    prog.validate().expect("valid");
    scenario(
        "wait_until_flag",
        Module::new(vec![prog]),
        inputs(vec![("flag", u32_buf([0])), ("data", u32_buf([0; 32])), ("out", u32_buf([0; 32]))]),
    )
}

/// A register-only scalar loop (`iters` iterations of fma) per thread:
/// dispatch cost per instruction.
pub fn scalar_loop(iters: u32) -> Scenario {
    let mut b = ProgramBuilder::new("scalar_loop", 32);
    let out = b.global("out", Dtype::U32);
    let lane = b.reg(Ty::U32);
    let k = b.reg(Ty::U32);
    let acc = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k3 = b.k_u32(3);
    let kn = b.k_u32(iters);
    b.mov(k, k0);
    b.mov(acc, lane);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, k, kn);
    b.loop_if(p);
    b.mul(Ty::U32, acc, acc, k3);
    b.add_u32(acc, acc, k);
    b.add_u32(k, k, k1);
    b.loop_end();
    b.st_u32(out, lane, acc);
    b.exit();
    let mut s = scenario("scalar_loop", b.build_module(), inputs(vec![("out", u32_buf([0; 32]))]));
    s.config.completions = CompletionPolicy::Eager;
    s
}

/// Per-warp loads and stores: `iters` rounds of `y[i] = x[i] + y[i]`.
pub fn warp_ldst(iters: u32) -> Scenario {
    let mut b = ProgramBuilder::new("warp_ldst", 32);
    let x = b.global("x", Dtype::U32);
    let y = b.global("y", Dtype::U32);
    let lane = b.reg(Ty::U32);
    let k = b.reg(Ty::U32);
    let a = b.reg(Ty::U32);
    let c = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let kn = b.k_u32(iters);
    b.mov(k, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, k, kn);
    b.loop_if(p);
    b.ld_u32(a, x, lane);
    b.ld_u32(c, y, lane);
    b.add_u32(c, c, a);
    b.st_u32(y, lane, c);
    b.add_u32(k, k, k1);
    b.loop_end();
    b.exit();
    let mut s = scenario("warp_ldst", b.build_module(), inputs(vec![("x", u32_buf(0..32)), ("y", u32_buf([0; 32]))]));
    s.config.completions = CompletionPolicy::Eager;
    s
}

/// tcgen05 lifecycle and data movement: warp 0 allocates 32 TMEM columns
/// (address written to shared memory), every warp stores `tid` to its
/// sub-partition with `tcgen05.st.32x32b.x1`, waits, loads it back with
/// `tcgen05.ld`, waits, and stores it to `out`; warp 0 deallocates and
/// relinquishes.
pub fn tcgen_ld_st() -> Scenario {
    let mut b = ProgramBuilder::new("tcgen_ld_st", 128);
    let out = b.global("out", Dtype::U32);
    let slot = b.shared("taddr", Dtype::U32, 1);
    let tid = b.reg(Ty::U32);
    let w = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let sa = b.reg(Ty::U32);
    let t = b.reg(Ty::U32);
    let t2 = b.reg(Ty::U32);
    let lanebits = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    b.thread_rank(tid);
    b.warp_id(w);
    let k0 = b.k_u32(0);
    let k32 = b.k_u32(32);
    let k16 = b.k_u32(16);
    b.smem_addr(sa, slot, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, w, k0);
    b.site("tcgen_alloc", 1);
    b.if_(p);
    b.push(Instr::TcgenAlloc { dst: sa.into(), ncols: k32, cta_group: 1, exclusive: false });
    b.end_if();
    b.no_site();
    b.bar_sync(0);
    b.ld_u32(t, slot, k0);
    // taddr of this warp's sub-partition: base + ((warp * 32) << 16).
    b.mul(Ty::U32, lanebits, w, k32);
    b.binary(BinOp::Shl, Ty::U32, lanebits, lanebits, k16);
    b.add_u32(t2, t, lanebits);
    b.site("tcgen_st", 2);
    b.push(Instr::TcgenSt(Box::new(TcgenStArgs {
        srcs: vec![tid.into()],
        taddr: t2.into(),
        row: k0,
        col: k0,
        shape: TcShape::S32x32b,
        num: 1,
        unpack: false,
    })));
    b.push(Instr::TcgenWait { st: true });
    b.site("tcgen_ld", 3);
    b.push(Instr::TcgenLd(Box::new(TcgenLdArgs {
        dsts: vec![v],
        taddr: t2.into(),
        row: k0,
        col: k0,
        shape: TcShape::S32x32b,
        num: 1,
        pack: false,
        red: None,
        red_abs: false,
        red_nan: false,
        spcompress: false,
    })));
    b.push(Instr::TcgenWait { st: false });
    b.no_site();
    b.st_u32(out, tid, v);
    b.bar_sync(0);
    b.site("tcgen_dealloc", 4);
    b.if_(p);
    b.push(Instr::TcgenDealloc { taddr: t.into(), ncols: k32, cta_group: 1, exclusive: false });
    b.push(Instr::TcgenRelinquish { cta_group: 1 });
    b.end_if();
    b.exit();
    scenario("tcgen_ld_st", b.build_module(), inputs(vec![("out", u32_buf([0; 128]))]))
}

/// Rows x cols of the TMA scenario's f32 tensor, and its box.
pub const TMA_ROWS: u32 = 8;
pub const TMA_COLS: u32 = 16;
pub const TMA_BOX_ROWS: u32 = 4;

/// TMA tile loads through a `__grid_constant__` tensor map (bound with
/// `ArgValue::TensorMapOf`): two 16x4 f32 boxes at rows 2 and 6 of an 8x16
/// tensor (rows 8..9 of the second box are out of bounds -> zero fill), both
/// completing on one mbarrier armed with `arrive.expect_tx(512)`; the warp
/// waits and copies both boxes to `out`.
pub fn tma_load() -> Scenario {
    let mut b = ProgramBuilder::new("tma_load", 32);
    let _src = b.global("src", Dtype::F32);
    let out = b.global("out", Dtype::F32);
    let bar = b.shared("bar", Dtype::U64, 1);
    let boxes = b.shared("boxes", Dtype::F32, 2 * (TMA_BOX_ROWS * TMA_COLS) as u64);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let barr = b.reg(Ty::U32);
    let s0 = b.reg(Ty::U32);
    let s1 = b.reg(Ty::U32);
    let tmap = b.reg(Ty::U64);
    let idx = b.reg(Ty::U32);
    let v = b.reg(Ty::F32);
    let j = b.reg(Ty::U32);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let box_elems = TMA_BOX_ROWS * TMA_COLS;
    let kbox = b.k_u32(box_elems);
    b.smem_addr(barr, bar, k0);
    b.smem_addr(s0, boxes, k0);
    b.smem_addr(s1, boxes, kbox);
    let tmap_pc = b.push(Instr::Nop); // patched: AddrOf tmap <- tensor-map param
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.mbar_init(barr, 1);
    b.end_if();
    b.fence(FenceKind::MbarrierInit, Sem::Release, Scope::Cluster);
    let full = b.k_u32(u32::MAX);
    let e = b.reg(Ty::PRED);
    b.push(Instr::Elect { dst_pred: e, dst_lane: None, membermask: full });
    b.if_(e);
    let ktx = b.k_u32(2 * box_elems * 4);
    b.push(Instr::MbarArrive(MbarArriveArgs {
        mbar: barr.into(),
        space: AddrSpace::Shared,
        count: None,
        expect_tx: Some(ktx),
        drop: false,
        no_complete: false,
        sem: Sem::Release,
        scope: Scope::Cta,
        multicast: None,
        state: None,
    }));
    for (row, smem) in [(2, s0), (6, s1)] {
        let kr = b.k_i32(row);
        let kc = b.k_i32(0);
        b.site("tma", 1 + row as u32);
        b.push(Instr::Tma(Box::new(TmaArgs {
            dir: TmaDir::Load,
            mode: TmaMode::Tile,
            tmap: tmap.into(),
            tmap_space: AddrSpace::Generic,
            coords: vec![kc, kr],
            im2col_offsets: vec![],
            smem: smem.into(),
            smem_space: AddrSpace::SharedCluster,
            completion: BulkCompletion::Mbarrier { mbar: barr.into(), space: AddrSpace::Shared },
            multicast: None,
            cta_group: 0,
            overrides: vec![],
            report: None,
            mods: MemMods::default(),
        })));
    }
    b.no_site();
    b.end_if();
    b.mbar_wait_parity(barr, k0);
    // Copy both boxes (2 * 64 floats) to `out`, 4 per lane.
    let k1 = b.k_u32(1);
    let k4 = b.k_u32(4);
    let k32 = b.k_u32(32);
    b.mov(j, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, j, k4);
    b.loop_if(p);
    b.mul(Ty::U32, idx, j, k32);
    b.add_u32(idx, idx, lane);
    b.ld_f32(v, boxes, idx);
    b.st_f32(out, idx, v);
    b.add_u32(j, j, k1);
    b.loop_end();
    b.exit();
    let mut prog = b.build();
    mark_elect(&mut prog, e);
    // Tensor-map parameter + its 128-byte Param-space buffer.
    let pid = ParamId(prog.host_abi.len() as u32);
    let buf = Buf(prog.buffers.len() as u32);
    prog.buffers.push(BufferDecl {
        name: "tmap".into(),
        space: crate::arena::Space::Param,
        dtype: Ty::U8,
        shape: vec![DimExpr::Const(128)],
        strides: vec![],
        param_slot: Some(pid),
        base: 0,
        byte_len: Some(DimExpr::Const(128)),
        align: 64,
        view_of: None,
        sync_words: false,
        base_reg: None,
    });
    prog.host_abi.push(ParamSlot {
        name: "tmap".into(),
        local_name: "tmap".into(),
        aliases: vec![],
        kind: ParamKind::TensorMap,
        dtype: None,
        shape: vec![],
        tensor_map: None,
        implicit_base: None,
        buf: Some(buf),
    });
    let kz = match k0 {
        Operand::Const(c) => c,
        _ => unreachable!(),
    };
    prog.code[tmap_pc.0 as usize] = Instr::AddrOf { dst: tmap, buf, offset: Operand::Const(kz) };
    prog.validate().expect("valid");
    let desc = crate::oplib::TensorMapDesc {
        global_address: 0,
        rank: 2,
        elem: Some(Dtype::F32),
        global_dim: [TMA_COLS as u64, TMA_ROWS as u64, 1, 1, 1],
        global_stride: [TMA_COLS as u64 * 4, 0, 0, 0, 0],
        box_dim: [TMA_COLS, TMA_BOX_ROWS, 1, 1, 1],
        element_stride: [1; 5],
        ..Default::default()
    };
    scenario(
        "tma_load",
        Module::new(vec![prog]),
        inputs(vec![
            ("src", f32_buf((0..TMA_ROWS * TMA_COLS).map(|i| ((i / TMA_COLS) * 100 + i % TMA_COLS) as f32))),
            ("out", f32_buf(vec![-1.0; (2 * box_elems) as usize])),
            ("tmap", ArgValue::TensorMapOf { base: "src".into(), offset: 0, desc }),
        ]),
    )
}

/// layout::v1 copy reports: two phases of a 128-byte bulk copy with
/// `mbarrier::report::validity::per_element::ff`; the second source contains
/// a 0xff byte. After each wait, `try_wait.parity` with a report register
/// reads the phase's report bit into `out[phase]`.
pub fn copy_report() -> Scenario {
    let mut b = ProgramBuilder::new("copy_report", 32);
    let ga = b.global("ga", Dtype::U32);
    let gb = b.global("gb", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let bar = b.shared("bar", Dtype::U64, 1);
    let data = b.shared("data", Dtype::U32, 32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let e = b.reg(Ty::PRED);
    let ready = b.reg(Ty::PRED);
    let rep = b.reg(Ty::PRED);
    let repv = b.reg(Ty::U8);
    let barr = b.reg(Ty::U32);
    let daddr = b.reg(Ty::U32);
    let g = b.reg(Ty::U64);
    let v = b.reg(Ty::U32);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    b.smem_addr(barr, bar, k0);
    b.smem_addr(daddr, data, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.push(Instr::MbarInit { mbar: barr.into(), space: AddrSpace::Shared, count: k1, layout_v1: true });
    b.end_if();
    b.fence(FenceKind::MbarrierInit, Sem::Release, Scope::Cluster);
    let full = b.k_u32(u32::MAX);
    let k128 = b.k_u32(128);
    for (phase, src) in [(0u32, ga), (1, gb)] {
        b.push(Instr::Elect { dst_pred: e, dst_lane: None, membermask: full });
        b.if_(e);
        b.push(Instr::MbarArrive(MbarArriveArgs {
            mbar: barr.into(),
            space: AddrSpace::Shared,
            count: None,
            expect_tx: Some(k128),
            drop: false,
            no_complete: false,
            sem: Sem::Release,
            scope: Scope::Cta,
            multicast: None,
            state: None,
        }));
        b.addr_of(g, src, k0);
        b.site("report_copy", 1 + phase);
        b.push(Instr::BulkCopy(BulkCopyArgs {
            dst: daddr.into(),
            dst_space: AddrSpace::SharedCluster,
            src: g.into(),
            src_space: AddrSpace::Global,
            size: k128,
            completion: BulkCompletion::Mbarrier { mbar: barr.into(), space: AddrSpace::Shared },
            multicast: None,
            reduce: None,
            byte_mask: None,
            ignore_oob: None,
            report: Some(ReportMode::PerElementFf),
            mods: MemMods::default(),
        }));
        b.no_site();
        b.end_if();
        let kp = b.k_u32(phase);
        b.mbar_wait_parity(barr, kp);
        b.push(Instr::MbarTestWait {
            kind: WaitKind::Try,
            mbar: barr.into(),
            space: AddrSpace::Shared,
            phase: PhaseArg::Parity(kp),
            sem: Sem::Acquire,
            scope: Scope::Cta,
            dst: Some(ready),
            report: Some(rep),
            report_value: Some(repv),
        });
        b.if_(p);
        b.cast(Ty::PRED, Ty::U32, v, rep);
        b.st_u32(out, kp, v);
        b.end_if();
        // Every lane reads the data before the next phase reuses it; the
        // async-proxy overwrite needs the generic reads fenced.
        b.ld_u32(v, data, lane);
        b.fence(FenceKind::ProxyAsync(Some(AddrSpace::Shared)), Sem::Weak, Scope::Cta);
        b.push(Instr::WarpSync { membermask: full });
    }
    b.exit();
    let mut prog = b.build();
    mark_elect(&mut prog, e);
    let mut gbv: Vec<u32> = (0..32).collect();
    gbv[5] = 0x00ff_0000;
    scenario(
        "copy_report",
        Module::new(vec![prog]),
        inputs(vec![("ga", u32_buf(0..32)), ("gb", u32_buf(gbv)), ("out", u32_buf([7, 7]))]),
    )
}

/// `if lane < 16 { s[lane] = 10*lane; __syncwarp(); x = s[lane ^ 16] } else
/// { s[lane] = 10*lane; __syncwarp(); x = s[lane ^ 16] }; out[lane] = x`:
/// a `__syncwarp` rendezvous across divergent arms.
pub fn divergent_syncwarp() -> Scenario {
    let mut b = ProgramBuilder::new("divergent_syncwarp", 32);
    let out = b.global("out", Dtype::U32);
    let sm = b.shared("s", Dtype::U32, 32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let v = b.reg(Ty::U32);
    let o = b.reg(Ty::U32);
    let x = b.reg(Ty::U32);
    b.lane_id(lane);
    let k16 = b.k_u32(16);
    let k10 = b.k_u32(10);
    let full = b.k_u32(u32::MAX);
    b.compare(CmpOp::Lt, Ty::U32, p, lane, k16);
    b.if_(p);
    for arm in 0..2 {
        if arm == 1 {
            b.else_();
        }
        b.mul(Ty::U32, v, lane, k10);
        b.st_u32(sm, lane, v);
        b.site("syncwarp", 1 + arm);
        b.push(Instr::WarpSync { membermask: full });
        b.no_site();
        b.binary(BinOp::Xor, Ty::U32, o, lane, k16);
        b.ld_u32(x, sm, o);
    }
    b.end_if();
    b.st_u32(out, lane, x);
    b.exit();
    scenario("divergent_syncwarp", b.build_module(), inputs(vec![("out", u32_buf([0; 32]))]))
}

/// Experts of the MoE synthetic.
pub const MOE_EXPERTS: u32 = 8;

/// A Mega-MoE-shaped synthetic: `ctas` CTAs x 256 threads (8 warps). Every
/// iteration each thread loads its token from global memory, mixes it with
/// a neighbour's value exchanged through shared memory (`bar.sync`) and an
/// ALU chain; at the end it stores its result and lane 0 of every warp
/// counts the CTA's tokens into its expert with a global `red.add` (a
/// cross-partition read-modify-write). Expected values: [`moe_expected`].
pub fn moe_synthetic(ctas: u32, iters: u32) -> Scenario {
    let threads = 256u32;
    let mut b = ProgramBuilder::new("moe_synthetic", threads);
    b.grid(ctas, 1, 1);
    let x_in = b.global("tokens", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let counts = b.global("counts", Dtype::U32);
    let sm = b.shared("s", Dtype::U32, threads as u64);
    let tid = b.reg(Ty::U32);
    let cta = b.reg(Ty::U32);
    let gi = b.reg(Ty::U32);
    let acc = b.reg(Ty::U32);
    let x = b.reg(Ty::U32);
    let nb = b.reg(Ty::U32);
    let ni = b.reg(Ty::U32);
    let k = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let lane = b.reg(Ty::U32);
    let expert = b.reg(Ty::U32);
    let ca = b.reg(Ty::U64);
    b.thread_rank(tid);
    b.lane_id(lane);
    b.read_special(cta, SpecialReg::CtaLinear);
    let kt = b.k_u32(threads);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k3 = b.k_u32(3);
    let kmask = b.k_u32(threads - 1);
    let kn = b.k_u32(iters);
    let ke = b.k_u32(MOE_EXPERTS);
    b.mul(Ty::U32, gi, cta, kt);
    b.add_u32(gi, gi, tid);
    b.mov(acc, tid);
    b.add_u32(ni, tid, k1);
    b.binary(BinOp::And, Ty::U32, ni, ni, kmask);
    b.mov(k, k0);
    b.site("moe_loop", 1);
    b.loop_begin();
    b.no_site();
    b.compare(CmpOp::Lt, Ty::U32, p, k, kn);
    b.loop_if(p);
    b.ld_u32(x, x_in, gi);
    b.mul(Ty::U32, acc, acc, k3);
    b.add_u32(acc, acc, x);
    b.add_u32(acc, acc, k);
    b.st_u32(sm, tid, acc);
    b.bar_sync(0);
    b.ld_u32(nb, sm, ni);
    b.add_u32(acc, acc, nb);
    b.bar_sync(0);
    b.add_u32(k, k, k1);
    b.loop_end();
    b.st_u32(out, gi, acc);
    b.binary(BinOp::Mod, Ty::U32, expert, cta, ke);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.addr_of(ca, counts, expert);
    b.site("expert_count", 2);
    b.red_add(Ty::U32, ca, k1);
    b.end_if();
    b.exit();
    let n = (ctas * threads) as usize;
    let mut s = scenario(
        "moe_synthetic",
        b.build_module(),
        inputs(vec![
            ("tokens", u32_buf((0..n as u32).map(|i| i.wrapping_mul(2654435761) >> 7))),
            ("out", u32_buf(vec![0; n])),
            ("counts", u32_buf(vec![0; MOE_EXPERTS as usize])),
        ]),
    );
    s.config.completions = CompletionPolicy::Eager;
    s
}

/// Expected `(out, counts)` of [`moe_synthetic`].
pub fn moe_expected(ctas: u32, iters: u32) -> (Vec<u32>, Vec<u32>) {
    let t = 256usize;
    let mut out = vec![0u32; ctas as usize * t];
    let mut counts = vec![0u32; MOE_EXPERTS as usize];
    for c in 0..ctas as usize {
        let mut acc: Vec<u32> = (0..t as u32).collect();
        for k in 0..iters {
            let mut s = vec![0u32; t];
            for i in 0..t {
                let x = ((c * t + i) as u32).wrapping_mul(2654435761) >> 7;
                acc[i] = acc[i].wrapping_mul(3).wrapping_add(x).wrapping_add(k);
                s[i] = acc[i];
            }
            for i in 0..t {
                acc[i] = acc[i].wrapping_add(s[(i + 1) & (t - 1)]);
            }
        }
        out[c * t..(c + 1) * t].copy_from_slice(&acc);
        counts[c % MOE_EXPERTS as usize] += 8; // lane 0 of each of 8 warps
    }
    (out, counts)
}

/// `ldmatrix.m8n8.x4.b16` (plain into `out_ld`, `.trans` into `out_ldt`)
/// from a 4x8x8 u16 matrix block, and `stmatrix.m8n8.x4` of the plain
/// fragments into a second block copied to `out_st` (a round trip).
pub fn matrix_roundtrip() -> Scenario {
    let mut b = ProgramBuilder::new("matrix_roundtrip", 32);
    let input = b.global("input", Dtype::U16);
    let out_ld = b.global("out_ld", Dtype::U32);
    let out_ldt = b.global("out_ldt", Dtype::U32);
    let out_st = b.global("out_st", Dtype::U16);
    let a = b.shared("a", Dtype::U16, 256);
    let c = b.shared("c", Dtype::U16, 256);
    let lane = b.reg(Ty::U32);
    let i = b.reg(Ty::U32);
    let v = b.reg(Ty::U16);
    let row = b.reg(Ty::U32);
    let ra = b.reg(Ty::U32);
    let rc = b.reg(Ty::U32);
    let frag: Vec<Reg> = (0..4).map(|_| b.reg(Ty::U32)).collect();
    let fragt: Vec<Reg> = (0..4).map(|_| b.reg(Ty::U32)).collect();
    let p = b.reg(Ty::PRED);
    let j = b.reg(Ty::U32);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k4 = b.k_u32(4);
    let k8 = b.k_u32(8);
    let k32 = b.k_u32(32);
    // Fill `a` from `input` (8 u16 per lane).
    b.mov(j, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, j, k8);
    b.loop_if(p);
    b.mul(Ty::U32, i, j, k32);
    b.add_u32(i, i, lane);
    b.ld(Ty::U16, v, input, i);
    b.st(Ty::U16, a, i, v);
    b.add_u32(j, j, k1);
    b.loop_end();
    let full = b.k_u32(u32::MAX);
    b.push(Instr::WarpSync { membermask: full });
    // Lane l provides row (l % 8) of matrix (l / 8): element 8 * l.
    b.mul(Ty::U32, row, lane, k8);
    b.smem_addr(ra, a, row);
    b.smem_addr(rc, c, row);
    b.site("ldmatrix", 1);
    b.push(Instr::LdMatrix { dsts: frag.clone(), addr: ra.into(), space: AddrSpace::Shared, shape: MatrixShape::M8N8, num: 4, trans: false, fmt: MatrixFmt::B16 });
    b.push(Instr::LdMatrix { dsts: fragt.clone(), addr: ra.into(), space: AddrSpace::Shared, shape: MatrixShape::M8N8, num: 4, trans: true, fmt: MatrixFmt::B16 });
    b.site("stmatrix", 2);
    b.push(Instr::StMatrix { srcs: frag.iter().map(|&r| r.into()).collect(), addr: rc.into(), space: AddrSpace::Shared, shape: MatrixShape::M8N8, num: 4, trans: false });
    b.no_site();
    for (k, (&f, &ft)) in frag.iter().zip(&fragt).enumerate() {
        let kk = b.k_u32(k as u32);
        b.mul(Ty::U32, i, lane, k4);
        b.add_u32(i, i, kk);
        b.st_u32(out_ld, i, f);
        b.st_u32(out_ldt, i, ft);
    }
    b.push(Instr::WarpSync { membermask: full });
    b.mov(j, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, j, k8);
    b.loop_if(p);
    b.mul(Ty::U32, i, j, k32);
    b.add_u32(i, i, lane);
    b.ld(Ty::U16, v, c, i);
    b.st(Ty::U16, out_st, i, v);
    b.add_u32(j, j, k1);
    b.loop_end();
    b.exit();
    let u16_buf = |v: Vec<u16>| ArgValue::Buffer { bytes: v.into_iter().flat_map(|x| x.to_le_bytes()).collect(), valid: None };
    scenario(
        "matrix_roundtrip",
        b.build_module(),
        inputs(vec![
            ("input", u16_buf((0..256).map(|x| (x * 7 + 1) as u16).collect())),
            ("out_ld", u32_buf([0; 128])),
            ("out_ldt", u32_buf([0; 128])),
            ("out_st", u16_buf(vec![0; 256])),
        ]),
    )
}

/// Matrix descriptor of the `tcgen_cp_ld` scenario's source (window offset
/// 0, LBO 128 B, SBO 256 B, no swizzle; legacy `encode_matrix_descriptor`).
pub const TCGEN_CP_SDESC: u64 = (8u64 << 16) | (16u64 << 32) | (1u64 << 46);

/// `tcgen05.cp.128x256b` of a 4 KiB shared block into TMEM columns 0..8 of
/// a fresh 32-column allocation, committed to an mbarrier; every warp then
/// reads its sub-partition back with `tcgen05.ld.32x32b.x8` into `out`
/// (`out[tid * 8 + j]` = TMEM lane `tid`, column `j`).
pub fn tcgen_cp_ld() -> Scenario {
    let mut b = ProgramBuilder::new("tcgen_cp_ld", 128);
    let input = b.global("input", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let src = b.shared("src", Dtype::U32, 1024);
    let bar = b.shared("bar", Dtype::U64, 1);
    let slot = b.shared("taddr", Dtype::U32, 1);
    let tid = b.reg(Ty::U32);
    let w = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let i = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    let j = b.reg(Ty::U32);
    let sa = b.reg(Ty::U32);
    let barr = b.reg(Ty::U32);
    let t = b.reg(Ty::U32);
    let t2 = b.reg(Ty::U32);
    let lb = b.reg(Ty::U32);
    let dst: Vec<Reg> = (0..8).map(|_| b.reg(Ty::U32)).collect();
    b.thread_rank(tid);
    b.warp_id(w);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k8 = b.k_u32(8);
    let k16 = b.k_u32(16);
    let k32 = b.k_u32(32);
    let k128 = b.k_u32(128);
    // Fill the source block (8 words per thread).
    b.mov(j, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, j, k8);
    b.loop_if(p);
    b.mul(Ty::U32, i, j, k128);
    b.add_u32(i, i, tid);
    b.ld_u32(v, input, i);
    b.st_u32(src, i, v);
    b.add_u32(j, j, k1);
    b.loop_end();
    b.smem_addr(sa, slot, k0);
    b.smem_addr(barr, bar, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, w, k0);
    b.if_(p);
    b.push(Instr::TcgenAlloc { dst: sa.into(), ncols: k32, cta_group: 1, exclusive: false });
    b.end_if();
    b.compare(CmpOp::Eq, Ty::U32, p, tid, k0);
    b.if_(p);
    b.mbar_init(barr, 1);
    b.end_if();
    b.fence(FenceKind::MbarrierInit, Sem::Release, Scope::Cluster);
    b.fence(FenceKind::ProxyAsync(Some(AddrSpace::Shared)), Sem::Weak, Scope::Cta);
    b.fence(FenceKind::Tcgen05Before, Sem::Weak, Scope::Cta);
    b.bar_sync(0);
    b.fence(FenceKind::Tcgen05After, Sem::Weak, Scope::Cta);
    b.ld_u32(t, slot, k0);
    b.if_(p);
    let kdesc = b.konst(Ty::U64, TCGEN_CP_SDESC as u128);
    b.site("tcgen_cp", 1);
    b.push(Instr::TcgenCp(TcgenCpArgs {
        taddr: t.into(),
        row: k0,
        col: k0,
        sdesc: kdesc,
        rows: 128,
        bits: 256,
        multicast: 0,
        decompress_bits: 0,
        cta_group: 1,
    }));
    b.push(Instr::TcgenCommit { mbar: barr.into(), space: AddrSpace::Shared, cta_group: 1, multicast: None, sync_restrict: false, multicast_width: None });
    b.no_site();
    b.end_if();
    b.mbar_wait_parity(barr, k0);
    b.fence(FenceKind::Tcgen05After, Sem::Weak, Scope::Cta);
    b.mul(Ty::U32, lb, w, k32);
    b.binary(BinOp::Shl, Ty::U32, lb, lb, k16);
    b.add_u32(t2, t, lb);
    b.site("tcgen_ld", 2);
    b.push(Instr::TcgenLd(Box::new(TcgenLdArgs {
        dsts: dst.clone(),
        taddr: t2.into(),
        row: k0,
        col: k0,
        shape: TcShape::S32x32b,
        num: 8,
        pack: false,
        red: None,
        red_abs: false,
        red_nan: false,
        spcompress: false,
    })));
    b.push(Instr::TcgenWait { st: false });
    b.no_site();
    for (k, &d) in dst.iter().enumerate() {
        let kk = b.k_u32(k as u32);
        b.mul(Ty::U32, i, tid, k8);
        b.add_u32(i, i, kk);
        b.st_u32(out, i, d);
    }
    b.fence(FenceKind::Tcgen05Before, Sem::Weak, Scope::Cta);
    b.bar_sync(0);
    b.compare(CmpOp::Eq, Ty::U32, p, w, k0);
    b.if_(p);
    b.push(Instr::TcgenDealloc { taddr: t.into(), ncols: k32, cta_group: 1, exclusive: false });
    b.end_if();
    b.exit();
    scenario(
        "tcgen_cp_ld",
        b.build_module(),
        inputs(vec![("input", u32_buf((0..1024).map(|x| x * 3 + 7))), ("out", u32_buf(vec![0; 1024]))]),
    )
}

/// Elected-lane helper: `elect.sync` into `e`, `If{elect}` on it.
fn elect_if(b: &mut ProgramBuilder, e: Reg) {
    let full = b.k_u32(u32::MAX);
    b.push(Instr::Elect { dst_pred: e, dst_lane: None, membermask: full });
    b.if_(e);
}

/// A 64-byte global->shared bulk copy with `.cp_mask` 0x00ff (bytes 0..8 of
/// every 16) and `.ignore_oob` left 4 / right 8 over a shared block
/// pre-filled with 0xEE; `out` = the block afterwards. Expected bytes:
/// [`bulk_masked_expected`].
pub fn bulk_masked_copy() -> Scenario {
    let mut b = ProgramBuilder::new("bulk_masked_copy", 32);
    let input = b.global("input", Dtype::U8);
    let out = b.global("out", Dtype::U8);
    let bar = b.shared("bar", Dtype::U64, 1);
    let blk = b.shared("blk", Dtype::U8, 64);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let e = b.reg(Ty::PRED);
    let barr = b.reg(Ty::U32);
    let da = b.reg(Ty::U32);
    let g = b.reg(Ty::U64);
    let i = b.reg(Ty::U32);
    let v = b.reg(Ty::U8);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k2 = b.k_u32(2);
    let k32 = b.k_u32(32);
    let kee = b.konst(Ty::U8, 0xee);
    for half in [k0, k32] {
        b.add_u32(i, lane, half);
        b.st(Ty::U8, blk, i, kee);
    }
    b.smem_addr(barr, bar, k0);
    b.smem_addr(da, blk, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.mbar_init(barr, 1);
    b.end_if();
    b.fence(FenceKind::MbarrierInit, Sem::Release, Scope::Cluster);
    b.fence(FenceKind::ProxyAsync(Some(AddrSpace::Shared)), Sem::Weak, Scope::Cta);
    let full = b.k_u32(u32::MAX);
    b.push(Instr::WarpSync { membermask: full });
    elect_if(&mut b, e);
    let k64 = b.k_u32(64);
    b.push(Instr::MbarArrive(MbarArriveArgs {
        mbar: barr.into(),
        space: AddrSpace::Shared,
        count: None,
        expect_tx: Some(k64),
        drop: false,
        no_complete: false,
        sem: Sem::Release,
        scope: Scope::Cta,
        multicast: None,
        state: None,
    }));
    b.addr_of(g, input, k0);
    let mask = b.k_u32(0x00ff);
    let k4 = b.k_u32(4);
    let k8 = b.k_u32(8);
    b.site("masked_bulk", 1);
    b.push(Instr::BulkCopy(BulkCopyArgs {
        dst: da.into(),
        dst_space: AddrSpace::SharedCluster,
        src: g.into(),
        src_space: AddrSpace::Global,
        size: k64,
        completion: BulkCompletion::Mbarrier { mbar: barr.into(), space: AddrSpace::Shared },
        multicast: None,
        reduce: None,
        byte_mask: Some(mask),
        ignore_oob: Some(IgnoreOob { ignore_bytes_left: Some(k4), ignore_bytes_right: Some(k8) }),
        report: None,
        mods: MemMods::default(),
    }));
    b.no_site();
    b.end_if();
    b.mbar_wait_parity(barr, k0);
    for half in [k0, k32] {
        b.add_u32(i, lane, half);
        b.ld(Ty::U8, v, blk, i);
        b.st(Ty::U8, out, i, v);
    }
    let _ = (k1, k2);
    let mut prog = b.build();
    mark_elect(&mut prog, e);
    scenario(
        "bulk_masked_copy",
        Module::new(vec![prog]),
        inputs(vec![
            ("input", ArgValue::Buffer { bytes: (0..64).map(|x| x as u8 + 1).collect(), valid: None }),
            ("out", ArgValue::Buffer { bytes: vec![0; 64], valid: None }),
        ]),
    )
}

/// Expected `out` of [`bulk_masked_copy`].
pub fn bulk_masked_expected() -> Vec<u8> {
    (0..64usize).map(|i| if i % 16 < 8 && (4..56).contains(&i) { i as u8 + 1 } else { 0xee }).collect()
}

/// `ctas` CTAs each bulk-reduce (`cp.reduce.async.bulk .add.u32`, bulk
/// group) their 32-word shared block `cta * 1000 + i` into one global
/// `acc` buffer (cross-partition reductions are serial points).
pub fn bulk_reduce(ctas: u32) -> Scenario {
    let mut b = ProgramBuilder::new("bulk_reduce", 32);
    b.grid(ctas, 1, 1);
    let acc = b.global("acc", Dtype::U32);
    let blk = b.shared("blk", Dtype::U32, 32);
    let lane = b.reg(Ty::U32);
    let cta = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    let e = b.reg(Ty::PRED);
    let sa = b.reg(Ty::U32);
    let g = b.reg(Ty::U64);
    b.lane_id(lane);
    b.read_special(cta, SpecialReg::CtaLinear);
    let k0 = b.k_u32(0);
    let k1000 = b.k_u32(1000);
    b.mul(Ty::U32, v, cta, k1000);
    b.add_u32(v, v, lane);
    b.st_u32(blk, lane, v);
    b.fence(FenceKind::ProxyAsync(Some(AddrSpace::Shared)), Sem::Weak, Scope::Cta);
    let full = b.k_u32(u32::MAX);
    b.push(Instr::WarpSync { membermask: full });
    b.smem_addr(sa, blk, k0);
    b.addr_of(g, acc, k0);
    elect_if(&mut b, e);
    let k128 = b.k_u32(128);
    b.site("bulk_reduce", 1);
    b.push(Instr::BulkCopy(BulkCopyArgs {
        dst: g.into(),
        dst_space: AddrSpace::Global,
        src: sa.into(),
        src_space: AddrSpace::Shared,
        size: k128,
        completion: BulkCompletion::Group,
        multicast: None,
        reduce: Some((AtomOp::Add, Dtype::U32)),
        byte_mask: None,
        ignore_oob: None,
        report: None,
        mods: MemMods::default(),
    }));
    b.push(Instr::AsyncCommit { domain: Domain::Bulk });
    b.push(Instr::AsyncWait { domain: Domain::Bulk, n: 0, read: false });
    b.no_site();
    b.end_if();
    b.exit();
    let mut prog = b.build();
    mark_elect(&mut prog, e);
    scenario("bulk_reduce", Module::new(vec![prog]), inputs(vec![("acc", u32_buf((0..32).map(|i| i * 7)))]))
}

/// Warp 0's lanes break out of a loop at different iterations, then every
/// thread of both warps joins `bar.sync 0`: broken lanes are live again
/// after the loop (not exited), so the barrier completes.
pub fn break_then_barrier() -> Scenario {
    let mut b = ProgramBuilder::new("break_then_barrier", 64);
    let out = b.global("out", Dtype::U32);
    let tid = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let w = b.reg(Ty::U32);
    let k = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    b.thread_rank(tid);
    b.lane_id(lane);
    b.warp_id(w);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k40 = b.k_u32(40);
    b.compare(CmpOp::Eq, Ty::U32, p, w, k0);
    b.if_(p);
    b.mov(k, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, k, k40);
    b.loop_if(p);
    b.compare(CmpOp::Ge, Ty::U32, p, k, lane);
    b.if_(p);
    b.break_();
    b.end_if();
    b.add_u32(k, k, k1);
    b.loop_end();
    b.end_if();
    b.site("bar_after_break", 1);
    b.bar_sync(0);
    b.no_site();
    let k5 = b.k_u32(5);
    b.st_u32(out, tid, k5);
    b.exit();
    scenario("break_then_barrier", b.build_module(), inputs(vec![("out", u32_buf([0; 64]))]))
}

/// Launch-bounds register budget (`regs_per_thread = 128`, a host
/// `Configure`), then warpgroup 0 `setmaxnreg.dec 96` and warpgroup 1
/// `setmaxnreg.inc 160` (granted from the released registers).
pub fn setmaxnreg_launch_bounds() -> Scenario {
    let mut b = ProgramBuilder::new("setmaxnreg_launch_bounds", 256);
    let out = b.global("out", Dtype::U32);
    let tid = b.reg(Ty::U32);
    let wg = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    b.thread_rank(tid);
    b.read_special(wg, SpecialReg::WarpgroupInCta);
    let k0 = b.k_u32(0);
    b.compare(CmpOp::Eq, Ty::U32, p, wg, k0);
    b.site("setmaxnreg", 1);
    b.if_(p);
    b.push(Instr::SetMaxNReg { inc: false, count: 96 });
    b.else_();
    b.push(Instr::SetMaxNReg { inc: true, count: 160 });
    b.end_if();
    b.no_site();
    b.st_u32(out, tid, wg);
    b.exit();
    let mut prog = b.build();
    prog.topology.regs_per_thread = 128;
    scenario("setmaxnreg_launch_bounds", Module::new(vec![prog]), inputs(vec![("out", u32_buf(vec![9; 256]))]))
}

/// Every scenario (for differential runs across backends).
pub fn all() -> Vec<Scenario> {
    vec![
        vector_add(),
        divergent_if_else(),
        nested_loops(),
        mbarrier_producer_consumer(),
        named_barrier(),
        elect_region(),
        loop_budget(),
        deadlock_wait(),
        deadlock_spin(),
        uninit_read(),
        oob(),
        divergent_wait_arrive(),
        cp_async_copy(),
        wait_until_flag(),
        scalar_loop(50),
        warp_ldst(20),
        tcgen_ld_st(),
        tma_load(),
        copy_report(),
        divergent_syncwarp(),
        moe_synthetic(12, 3),
        matrix_roundtrip(),
        tcgen_cp_ld(),
        bulk_masked_copy(),
        bulk_reduce(6),
        break_then_barrier(),
        setmaxnreg_launch_bounds(),
    ]
}
