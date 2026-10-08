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
    Inputs { args: args.into_iter().map(|(k, v)| (k.to_string(), v)).collect(), ..Default::default() }
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

/// One warp, `tcgen05.st/ld.32x32b.x4` (4 consecutive cells per lane):
/// with `store`, two stores (the first onto fresh, invalid TMEM: per-piece
/// path; the second onto valid cells: run path) then a load (run path);
/// without, only the load, of never-written cells (per-piece path; run
/// with `ZeroAndReport`). Sites: st 2 and 5, ld 3. Observer spans must not
/// depend on the path (W5, c06149c).
pub fn tcgen_ld_wide(store: bool) -> Scenario {
    let mut b = ProgramBuilder::new("tcgen_ld_wide", 32);
    let out = b.global("out", Dtype::U32);
    let slot = b.shared("taddr", Dtype::U32, 1);
    let tid = b.reg(Ty::U32);
    let sa = b.reg(Ty::U32);
    let t = b.reg(Ty::U32);
    let v: Vec<Reg> = (0..4).map(|_| b.reg(Ty::U32)).collect();
    b.thread_rank(tid);
    let k0 = b.k_u32(0);
    let k32 = b.k_u32(32);
    b.smem_addr(sa, slot, k0);
    b.push(Instr::TcgenAlloc { dst: sa.into(), ncols: k32, cta_group: 1, exclusive: false });
    b.bar_sync(0);
    b.ld_u32(t, slot, k0);
    if store {
        for site in [2, 5] {
            b.site("tcgen_st", site);
            b.push(Instr::TcgenSt(Box::new(TcgenStArgs {
                srcs: vec![tid.into(), tid.into(), tid.into(), tid.into()],
                taddr: t.into(),
                row: k0,
                col: k0,
                shape: TcShape::S32x32b,
                num: 4,
                unpack: false,
            })));
            b.push(Instr::TcgenWait { st: true });
        }
    }
    b.site("tcgen_ld", 3);
    b.push(Instr::TcgenLd(Box::new(TcgenLdArgs {
        dsts: v.clone(),
        taddr: t.into(),
        row: k0,
        col: k0,
        shape: TcShape::S32x32b,
        num: 4,
        pack: false,
        red: None,
        red_abs: false,
        red_nan: false,
        spcompress: false,
    })));
    b.push(Instr::TcgenWait { st: false });
    b.no_site();
    b.st_u32(out, tid, v[3]);
    b.push(Instr::TcgenDealloc { taddr: t.into(), ncols: k32, cta_group: 1, exclusive: false });
    b.push(Instr::TcgenRelinquish { cta_group: 1 });
    b.exit();
    scenario("tcgen_ld_wide", b.build_module(), inputs(vec![("out", u32_buf([0; 32]))]))
}

/// Readonly-proxy contract (legacy `test_readonly_proxy`): lanes read
/// `data[lane]` with `ld.global.nc` into `out`. `variant`:
/// `"clean"` (no write), `"after"` / `"before"` (write `data[lane]` after /
/// before the read: an error either way), `"disjoint"` (write
/// `data[lane + 32]` before and after: clean), `"cross_cta"` (two CTAs in
/// different partitions: CTA 0 reads, CTA 1 writes `data[lane]`: an error).
pub fn readonly_proxy(variant: &str) -> Scenario {
    let mut b = ProgramBuilder::new("readonly_proxy", 32);
    let cross = variant == "cross_cta";
    if cross {
        b.grid(2, 1, 1);
    }
    let data = b.global("data", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let lane = b.reg(Ty::U32);
    let hi = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    let cta = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    b.lane_id(lane);
    let k32 = b.k_u32(32);
    b.add(Ty::U32, hi, lane, k32);
    let write = |b: &mut ProgramBuilder, at: Reg| b.st_u32(data, at, lane);
    let read = |b: &mut ProgramBuilder| {
        b.push(Instr::Load { ty: Ty::U32, dst: v, buf: data, offset: lane.into(), sem: Sem::Weak, scope: Scope::Cta, mods: MemMods { nc: true, ..Default::default() } });
        b.st_u32(out, lane, v);
    };
    match variant {
        "after" => {
            read(&mut b);
            write(&mut b, lane);
        }
        "before" => {
            write(&mut b, lane);
            read(&mut b);
        }
        "disjoint" => {
            write(&mut b, hi);
            read(&mut b);
            write(&mut b, hi);
        }
        "cross_cta" => {
            b.read_special(cta, SpecialReg::CtaLinear);
            let k0 = b.k_u32(0);
            b.compare(CmpOp::Eq, Ty::U32, p, cta, k0);
            b.if_(p);
            read(&mut b);
            b.else_();
            write(&mut b, lane);
            b.end_if();
        }
        _ => read(&mut b),
    }
    b.exit();
    scenario("readonly_proxy", b.build_module(), inputs(vec![("data", u32_buf(0..64)), ("out", u32_buf([0; 32]))]))
}

/// Non-`.aligned` `barrier.sync` reached by divergent lanes of one warp
/// (Q3 ruling). Two warps; thread 0 stores 7 to shared `s[0]`, then:
/// * `"same"`: thread 0 and the other lanes execute `barrier.sync 1, 64` at
///   two different sites; every thread then copies `s[0]` to `out` (all 7);
/// * `"other_id"`: the other lanes of warp 0 go to barrier 2 instead
///   (error: `PartialWarp`);
/// * `"exit"`: the other lanes exit instead (error: `PartialWarp`).
pub fn divergent_named_barrier(variant: &str) -> Scenario {
    let mut b = ProgramBuilder::new("divergent_named_barrier", 64);
    let out = b.global("out", Dtype::U32);
    let sm = b.shared("s", Dtype::U32, 1);
    let tid = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let v = b.reg(Ty::U32);
    b.thread_rank(tid);
    let k0 = b.k_u32(0);
    let k7 = b.k_u32(7);
    let k64 = b.k_u32(64);
    let id1 = b.k_u32(1);
    let id2 = b.k_u32(2);
    let bar = |b: &mut ProgramBuilder, id: Operand| b.push(Instr::Barrier { kind: BarKind::Sync, id, count: Some(k64), aligned: false });
    b.compare(CmpOp::Eq, Ty::U32, p, tid, k0);
    b.if_(p);
    b.st_u32(sm, k0, k7);
    bar(&mut b, id1);
    b.else_();
    match variant {
        "other_id" => {
            // Warp 1 uses barrier 1 (full warp), warp 0's lanes barrier 2.
            let q = b.reg(Ty::PRED);
            let k32 = b.k_u32(32);
            b.compare(CmpOp::Lt, Ty::U32, q, tid, k32);
            b.if_(q);
            bar(&mut b, id2);
            b.else_();
            bar(&mut b, id1);
            b.end_if();
        }
        "exit" => {
            b.exit();
        }
        _ => {
            bar(&mut b, id1);
        }
    }
    b.end_if();
    b.ld_u32(v, sm, k0);
    b.st_u32(out, tid, v);
    b.exit();
    scenario("divergent_named_barrier", b.build_module(), inputs(vec![("out", u32_buf([0; 64]))]))
}

/// W5-12: one warp's `tcgen05.alloc` writes the TMEM address to shared
/// memory (one lane's store) and every lane reads it right away, without a
/// barrier: the collective orders the warp after its result write (no
/// same-warp race). `out[lane]` = the address (allocation base 0).
pub fn tcgen_alloc_lanes_read() -> Scenario {
    // `cta_group::2` (as in sparse_flashmla_prefill_head128_phase1): the
    // address store happens on the retry after the peer rendezvous.
    let mut b = ProgramBuilder::new("tcgen_alloc_lanes_read", 32);
    b.grid(2, 1, 1);
    b.cluster(2, 1, 1);
    let out = b.global("out", Dtype::U32);
    let slot = b.shared("taddr", Dtype::U32, 1);
    let tid = b.reg(Ty::U32);
    let cta = b.reg(Ty::U32);
    let gi = b.reg(Ty::U32);
    let sa = b.reg(Ty::U32);
    let t = b.reg(Ty::U32);
    b.thread_rank(tid);
    b.read_special(cta, SpecialReg::CtaLinear);
    let k0 = b.k_u32(0);
    let k32 = b.k_u32(32);
    b.mul(Ty::U32, gi, cta, k32);
    b.add_u32(gi, gi, tid);
    b.smem_addr(sa, slot, k0);
    b.push(Instr::TcgenAlloc { dst: sa.into(), ncols: k32, cta_group: 2, exclusive: false });
    b.ld_u32(t, slot, k0);
    b.st_u32(out, gi, t);
    b.push(Instr::TcgenRelinquish { cta_group: 2 });
    b.push(Instr::TcgenDealloc { taddr: t.into(), ncols: k32, cta_group: 2, exclusive: false });
    b.exit();
    scenario("tcgen_alloc_lanes_read", b.build_module(), inputs(vec![("out", u32_buf([7; 64]))]))
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
    let mut gbv: Vec<u32> = (0..32).collect();
    gbv[5] = 0x00ff_0000;
    copy_report_with(ReportMode::PerElementFf, gbv)
}

/// W2-8: `.per_16bytes::80000000` samples the first 32-bit element of each
/// 16-byte source chunk: `gb[5]` (not sampled) never matches; with
/// `sampled`, `gb[8]` (first of chunk 2) does. `out = [0, sampled]`.
pub fn copy_report_16(sampled: bool) -> Scenario {
    let mut gbv: Vec<u32> = (0..32).collect();
    gbv[5] = 0x8000_0000;
    if sampled {
        gbv[8] = 0x8000_0000;
    }
    copy_report_with(ReportMode::Per16BytesPattern { pattern: 0x8000_0000, bits: 32 }, gbv)
}

fn copy_report_with(mode: ReportMode, gbv: Vec<u32>) -> Scenario {
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
            report: Some(mode),
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
    tcgen_cp_ld_with(false)
}

/// [`tcgen_cp_ld`]; with `double_commit`, the copy is committed to `bar`
/// and then AGAIN to `bar2` with nothing new issued in between, and the
/// warps wait only on `bar2`: a commit tracks every prior in-flight
/// tcgen05 op of the thread, not only those since its last commit, so
/// `bar2` must not complete before the copy lands (seeded latency).
pub fn tcgen_cp_ld_with(double_commit: bool) -> Scenario {
    let mut b = ProgramBuilder::new("tcgen_cp_ld", 128);
    let input = b.global("input", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let src = b.shared("src", Dtype::U32, 1024);
    let bar = b.shared("bar", Dtype::U64, 1);
    let bar2 = b.shared("bar2", Dtype::U64, 1);
    let barr2 = b.reg(Ty::U32);
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
    b.smem_addr(barr2, bar2, k0);
    b.if_(p);
    b.mbar_init(barr, 1);
    b.mbar_init(barr2, 1);
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
    if double_commit {
        b.site("second_commit", 3);
        b.push(Instr::TcgenCommit { mbar: barr2.into(), space: AddrSpace::Shared, cta_group: 1, multicast: None, sync_restrict: false, multicast_width: None });
    }
    b.no_site();
    b.end_if();
    b.mbar_wait_parity(if double_commit { barr2 } else { barr }, k0);
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
        if double_commit { "tcgen_cp_double_commit" } else { "tcgen_cp_ld" },
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
    // Masked-in bytes: copied inside the `.ignore_oob` window, zero (and
    // uninitialized, as legacy) on its ignored edges; masked-off bytes keep
    // the prefill.
    (0..64usize)
        .map(|i| match (i % 16 < 8, (4..56).contains(&i)) {
            (true, true) => i as u8 + 1,
            (true, false) => 0,
            (false, _) => 0xee,
        })
        .collect()
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

// ---------------------------------------------------------------------------
// Engine-review regressions (docs/development/engine-review.md)
// ---------------------------------------------------------------------------

fn test_wait(b: &mut ProgramBuilder, dst: Reg, barr: Reg, parity: Operand) {
    b.push(Instr::MbarTestWait {
        kind: WaitKind::Try,
        mbar: barr.into(),
        space: AddrSpace::Shared,
        phase: PhaseArg::Parity(parity),
        sem: Sem::Acquire,
        scope: Scope::Cta,
        dst: Some(dst),
        report: None,
        report_value: None,
    });
}

fn ld_sem(b: &mut ProgramBuilder, dst: Reg, buf: Buf, idx: Operand, sem: Sem, scope: Scope) {
    b.push(Instr::Load { ty: Ty::U32, dst, buf, offset: idx, sem, scope, mods: MemMods::default() });
}

fn st_sem(b: &mut ProgramBuilder, buf: Buf, idx: Operand, value: Operand, sem: Sem, scope: Scope) {
    b.push(Instr::Store { ty: Ty::U32, buf, offset: idx, value, sem, scope, mods: MemMods::default() });
}

/// A register-only loop of `n` iterations (keeps a warp busy for a few
/// rounds without touching memory or sync state).
fn busy_loop(b: &mut ProgramBuilder, n: u32) {
    let k = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let kn = b.k_u32(n);
    b.mov(k, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, k, kn);
    b.loop_if(p);
    b.add_u32(k, k, k1);
    b.loop_end();
}

/// H1 (G8): warp 1 exits, then warp 0 runs the count-less `bar.sync 0`.
/// Exited warps leave the barrier's membership, so it completes. With
/// `counted`, the barrier names an explicit count of 64 threads instead:
/// release by exit is not modeled there, so the hang is `incomplete`.
pub fn exit_then_barrier(counted: bool) -> Scenario {
    let mut b = ProgramBuilder::new("exit_then_barrier", 64);
    let out = b.global("out", Dtype::U32);
    let tid = b.reg(Ty::U32);
    let w = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    b.thread_rank(tid);
    b.warp_id(w);
    let k1 = b.k_u32(1);
    b.compare(CmpOp::Ge, Ty::U32, p, w, k1);
    b.if_(p);
    b.exit();
    b.end_if();
    b.site("bar_after_exit", 1);
    if counted {
        let k0 = b.k_u32(0);
        let k64 = b.k_u32(64);
        b.push(Instr::Barrier { kind: BarKind::Sync, id: k0, count: Some(k64), aligned: true });
    } else {
        b.bar_sync(0);
    }
    b.no_site();
    let k7 = b.k_u32(7);
    b.st_u32(out, tid, k7);
    b.exit();
    let name = if counted { "exit_then_counted_barrier" } else { "exit_then_barrier" };
    scenario(name, b.build_module(), inputs(vec![("out", u32_buf([0; 64]))]))
}

/// H2: a bounded `for k < 3: test_wait(bar)` probe on a barrier nobody
/// arrives on, then `out[lane] = k`. The loop's state advances, so it is
/// never spin-parked: Completed with `out = 3`.
pub fn bounded_probe_loop() -> Scenario {
    let mut b = ProgramBuilder::new("bounded_probe_loop", 32);
    let out = b.global("out", Dtype::U32);
    let bar = b.shared("bar", Dtype::U64, 1);
    let barr = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let k = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let ok = b.reg(Ty::PRED);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k3 = b.k_u32(3);
    b.smem_addr(barr, bar, k0);
    b.mbar_init(barr, 1);
    b.mov(k, k0);
    b.site("probe", 1);
    b.loop_begin();
    b.no_site();
    b.compare(CmpOp::Lt, Ty::U32, p, k, k3);
    b.loop_if(p);
    test_wait(&mut b, ok, barr, k0);
    b.add_u32(k, k, k1);
    b.loop_end();
    b.st_u32(out, lane, k);
    b.exit();
    scenario("bounded_probe_loop", b.build_module(), inputs(vec![("out", u32_buf([9; 32]))]))
}

/// M1: `while (true) { for s < 2 { if (!test_wait(bar[s])) break; } }` on
/// barriers that never complete. The inner loop's failed poll reaches the
/// outer iteration, which is a fixed point: Deadlock within a few rounds.
pub fn nested_spin_break() -> Scenario {
    let mut b = ProgramBuilder::new("nested_spin_break", 32);
    let bars = b.shared("bars", Dtype::U64, 2);
    let barr = b.reg(Ty::U32);
    let s = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let ok = b.reg(Ty::PRED);
    let bad = b.reg(Ty::PRED);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k2 = b.k_u32(2);
    for i in [k0, k1] {
        b.smem_addr(barr, bars, i);
        b.mbar_init(barr, 1);
    }
    let t = b.konst(Ty::PRED, 1);
    b.site("outer_spin", 1);
    b.loop_begin();
    b.no_site();
    b.loop_if(t);
    b.mov(s, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, s, k2);
    b.loop_if(p);
    b.smem_addr(barr, bars, s);
    test_wait(&mut b, ok, barr, k0);
    b.push(Instr::Unary { op: UnOp::Not, ty: Ty::PRED, dst: bad, a: ok.into() });
    b.if_(bad);
    b.break_();
    b.end_if();
    b.add_u32(s, s, k1);
    b.loop_end();
    b.loop_end();
    b.exit();
    let mut sc = scenario("nested_spin_break", b.build_module(), Inputs::default());
    sc.config.loop_budget = 1 << 40;
    sc
}

/// M2: warp 0 spins on `ld.acquire.gpu flag` until warp 1 (after a busy
/// loop) stores `st.release.gpu flag = 1` and the data it guards. With
/// `publish = false` warp 1 never stores: the spin parks and the launch is
/// a Deadlock, not a loop-budget overrun.
pub fn load_flag_spin(publish: bool) -> Scenario {
    let mut b = ProgramBuilder::new("load_flag_spin", 64);
    let flag = b.global("flag", Dtype::U32);
    let data = b.global("data", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let w = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let v = b.reg(Ty::U32);
    b.warp_id(w);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    b.compare(CmpOp::Eq, Ty::U32, p, w, k1);
    b.if_(p);
    if publish {
        busy_loop(&mut b, 600);
        let k5 = b.k_u32(5);
        b.st_u32(data, lane, k5);
        let full = b.k_u32(u32::MAX);
        b.push(Instr::WarpSync { membermask: full });
        b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
        b.if_(p);
        st_sem(&mut b, flag, k0, k1, Sem::Release, Scope::Gpu);
        b.end_if();
    }
    b.else_();
    b.mov(v, k0);
    b.site("flag_spin", 1);
    b.loop_begin();
    b.no_site();
    b.compare(CmpOp::Eq, Ty::U32, p, v, k0);
    b.loop_if(p);
    ld_sem(&mut b, v, flag, k0, Sem::Acquire, Scope::Gpu);
    b.loop_end();
    b.ld_u32(v, data, lane);
    b.st_u32(out, lane, v);
    b.end_if();
    b.exit();
    let mut sc = scenario(
        if publish { "load_flag_spin" } else { "load_flag_spin_never" },
        b.build_module(),
        inputs(vec![("flag", u32_buf([0])), ("data", u32_buf([0; 32])), ("out", u32_buf([0; 32]))]),
    );
    sc.config.loop_budget = 1 << 40;
    sc
}

/// W5-14: a 128-byte `sync_words` shared view of 32 `u32` flags, polled
/// per 4-byte element: warp 1 lane `l` stores `st.release.cta flags[l] =
/// l + 1`; warp 0 lane `l` spins on `ld.acquire.cta flags[l]` and copies the
/// value to `out[l]`. The engine declares 32 four-byte words.
pub fn polled_flag_words() -> Scenario {
    let mut b = ProgramBuilder::new("polled_flag_words", 64);
    let out = b.global("out", Dtype::U32);
    let flags = b.shared("flags", Dtype::U32, 32);
    let w = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let v = b.reg(Ty::U32);
    b.warp_id(w);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let fill = b.k_u32(0);
    b.st_u32(flags, lane, fill);
    b.bar_sync(0);
    b.compare(CmpOp::Eq, Ty::U32, p, w, k1);
    b.if_(p);
    b.add_u32(v, lane, k1);
    st_sem(&mut b, flags, lane.into(), v.into(), Sem::Release, Scope::Cta);
    b.else_();
    b.mov(v, k0);
    b.loop_begin();
    b.compare(CmpOp::Eq, Ty::U32, p, v, k0);
    b.loop_if(p);
    ld_sem(&mut b, v, flags, lane.into(), Sem::Acquire, Scope::Cta);
    b.loop_end();
    b.st_u32(out, lane, v);
    b.end_if();
    b.exit();
    let mut prog = b.build();
    prog.buffers[flags.0 as usize].sync_words = true;
    let mut sc = scenario("polled_flag_words", Module::new(vec![prog]), inputs(vec![("out", u32_buf([0; 32]))]));
    sc.config.loop_budget = 1 << 40;
    sc
}

/// M10: two single-CTA clusters (two partitions). In round 0, CTA 0 lane 0
/// stores `data = 5` then `st.release.gpu flag = 1`; CTA 1 lane 0 reads
/// `ld.acquire.gpu flag` ONCE (seeing the round-start 0) and then reads
/// `data` regardless: a real race, which racecheck must see. With
/// `guarded`, CTA 1 instead spins until the flag is 1 (race-free; also
/// the cross-partition visibility check: one round late).
pub fn cross_cluster_flag(guarded: bool) -> Scenario {
    let mut b = ProgramBuilder::new("cross_cluster_flag", 32);
    b.grid(2, 1, 1);
    let flag = b.global("flag", Dtype::U32);
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
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.compare(CmpOp::Eq, Ty::U32, p, cta, k0);
    b.if_(p);
    let k5 = b.k_u32(5);
    b.site("data_store", 1);
    b.st_u32(data, k0, k5);
    b.site("flag_release", 2);
    st_sem(&mut b, flag, k0, k1, Sem::Release, Scope::Gpu);
    b.no_site();
    b.else_();
    b.mov(f, k0);
    if guarded {
        b.loop_begin();
        b.compare(CmpOp::Eq, Ty::U32, p, f, k0);
        b.loop_if(p);
        b.site("flag_acquire", 3);
        ld_sem(&mut b, f, flag, k0, Sem::Acquire, Scope::Gpu);
        b.no_site();
        b.loop_end();
    } else {
        b.site("flag_acquire", 3);
        ld_sem(&mut b, f, flag, k0, Sem::Acquire, Scope::Gpu);
    }
    b.site("data_load", 4);
    b.ld_u32(v, data, k0);
    b.no_site();
    b.add_u32(v, v, f);
    b.st_u32(out, k0, v);
    b.end_if();
    b.end_if();
    b.exit();
    scenario(
        if guarded { "cross_cluster_flag" } else { "cross_cluster_flag_racy" },
        b.build_module(),
        inputs(vec![("flag", u32_buf([0])), ("data", u32_buf([0])), ("out", u32_buf([0]))]),
    )
}

/// H6: lane 0 `cp.async`s 4 bytes global -> shared WITHOUT commit/wait, one
/// register op, then a read of the destination: under `Seeded` completion
/// some seeds read before the copy lands.
pub fn cp_async_no_wait() -> Scenario {
    let mut b = ProgramBuilder::new("cp_async_no_wait", 32);
    let input = b.global("in", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let sm = b.shared("s", Dtype::U32, 32);
    let lane = b.reg(Ty::U32);
    let saddr = b.reg(Ty::U32);
    let gaddr = b.reg(Ty::U64);
    let v = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.smem_addr(saddr, sm, lane);
    b.addr_of(gaddr, input, lane);
    b.site("cp_async", 1);
    b.push(Instr::CpAsync { dst: saddr.into(), src: gaddr.into(), cp_size: 4, src_size: None, ignore_src: None, mods: MemMods::default() });
    b.no_site();
    b.add_u32(v, lane, lane);
    b.site("early_read", 2);
    b.ld_u32(v, sm, lane);
    b.no_site();
    b.st_u32(out, lane, v);
    b.cp_async_commit_wait_all();
    b.end_if();
    b.exit();
    let mut sc = scenario("cp_async_no_wait", b.build_module(), inputs(vec![("in", u32_buf(0..32)), ("out", u32_buf([0; 32]))]));
    sc.config.quantum = 1;
    sc
}

/// H3: one thread issues two shared->global bulk copies, each its own bulk
/// group, then `cp.async.bulk.wait_group.read 1` (only the OLDER group's
/// reads are done) and finally `wait_group 0`.
pub fn bulk_wait_read() -> Scenario {
    let mut b = ProgramBuilder::new("bulk_wait_read", 32);
    let out = b.global("out", Dtype::U32);
    let blk = b.shared("blk", Dtype::U32, 32);
    let lane = b.reg(Ty::U32);
    let e = b.reg(Ty::PRED);
    let sa = b.reg(Ty::U32);
    let g = b.reg(Ty::U64);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k16 = b.k_u32(16);
    let k64 = b.k_u32(64);
    b.st_u32(blk, lane, lane);
    b.fence(FenceKind::ProxyAsync(Some(AddrSpace::Shared)), Sem::Weak, Scope::Cta);
    let full = b.k_u32(u32::MAX);
    b.push(Instr::WarpSync { membermask: full });
    elect_if(&mut b, e);
    for half in [k0, k16] {
        b.smem_addr(sa, blk, half);
        b.addr_of(g, out, half);
        b.site("bulk_store", 1);
        b.push(Instr::BulkCopy(BulkCopyArgs {
            dst: g.into(),
            dst_space: AddrSpace::Global,
            src: sa.into(),
            src_space: AddrSpace::Shared,
            size: k64,
            completion: BulkCompletion::Group,
            multicast: None,
            reduce: None,
            byte_mask: None,
            ignore_oob: None,
            report: None,
            mods: MemMods::default(),
        }));
        b.push(Instr::AsyncCommit { domain: Domain::Bulk });
    }
    b.site("wait_read_1", 2);
    b.push(Instr::AsyncWait { domain: Domain::Bulk, n: 1, read: true });
    b.site("wait_all", 3);
    b.push(Instr::AsyncWait { domain: Domain::Bulk, n: 0, read: false });
    b.no_site();
    b.end_if();
    b.exit();
    let mut prog = b.build();
    mark_elect(&mut prog, e);
    scenario("bulk_wait_read", Module::new(vec![prog]), inputs(vec![("out", u32_buf([0; 32]))]))
}

/// H4 / M3: lanes 0..16 use `bars[0]`, lanes 16..32 `bars[1]` (count 16
/// each) in one `arrive` and one `wait` instruction.
pub fn lane_split_mbarrier() -> Scenario {
    let mut b = ProgramBuilder::new("lane_split_mbarrier", 32);
    let out = b.global("out", Dtype::U32);
    let bars = b.shared("bars", Dtype::U64, 2);
    let lane = b.reg(Ty::U32);
    let i = b.reg(Ty::U32);
    let m = b.reg(Ty::U32);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k4 = b.k_u32(4);
    b.binary(BinOp::Shr, Ty::U32, i, lane, k4);
    b.smem_addr(m, bars, i);
    b.site("init", 1);
    b.mbar_init(m, 16);
    b.fence(FenceKind::MbarrierInit, Sem::Release, Scope::Cluster);
    b.site("split_arrive", 2);
    b.mbar_arrive(m, None);
    b.site("split_wait", 3);
    b.mbar_wait_parity(m, k0);
    b.no_site();
    b.st_u32(out, lane, i);
    b.exit();
    scenario("lane_split_mbarrier", b.build_module(), inputs(vec![("out", u32_buf([9; 32]))]))
}

/// M4: warp 0 waits in ONE instruction on `A` (lanes 0..16) and `B`
/// (lanes 16..32), both count 1. Warp 1 lane 0 arrives on A, idles, arrives
/// on A again (A has now advanced two phases: parity 0 is pending again),
/// then arrives on B. Lanes that saw A complete left the wait (latched), so
/// the launch completes. Under all schedules the second arrive on A may
/// precede the first wait (nothing orders them), so synccheck rightly
/// reports `ReuseBeforeConsumption`: this is an engine-semantics test, not
/// part of [`all`].
pub fn mbar_latch() -> Scenario {
    let mut b = ProgramBuilder::new("mbar_latch", 64);
    let out = b.global("out", Dtype::U32);
    let bars = b.shared("bars", Dtype::U64, 2);
    let tid = b.reg(Ty::U32);
    let w = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let i = b.reg(Ty::U32);
    let m = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    b.thread_rank(tid);
    b.warp_id(w);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k4 = b.k_u32(4);
    b.compare(CmpOp::Eq, Ty::U32, p, tid, k0);
    b.if_(p);
    for x in [k0, k1] {
        b.smem_addr(m, bars, x);
        b.mbar_init(m, 1);
    }
    b.end_if();
    b.fence(FenceKind::MbarrierInit, Sem::Release, Scope::Cluster);
    b.bar_sync(0);
    b.compare(CmpOp::Eq, Ty::U32, p, w, k0);
    b.if_(p);
    b.binary(BinOp::Shr, Ty::U32, i, lane, k4);
    b.smem_addr(m, bars, i);
    b.site("latched_wait", 1);
    b.mbar_wait_parity(m, k0);
    b.no_site();
    b.st_u32(out, lane, k1);
    b.else_();
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.smem_addr(m, bars, k0);
    b.mbar_arrive(m, None);
    busy_loop(&mut b, 2000);
    b.mbar_arrive(m, None);
    b.smem_addr(m, bars, k1);
    b.mbar_arrive(m, None);
    b.end_if();
    b.end_if();
    b.exit();
    scenario("mbar_latch", b.build_module(), inputs(vec![("out", u32_buf([0; 32]))]))
}

/// H5: `atom.global.exch.b128` and `atom.global.cas.b128` by lane 0.
/// `g` starts as words [1,2,3,4, 5,6,7,8]: exch writes [9,9,9,9] to the
/// first element (old -> out[0..4]); cas on the second compares against
/// [5,6,7,8] and swaps in [7,7,7,7] (old -> out[4..8]).
pub fn atom_b128() -> Scenario {
    let mut b = ProgramBuilder::new("atom_b128", 32);
    let g = b.global("g", Dtype::U32);
    let out = b.global("out", Dtype::B128);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let a = b.reg(Ty::U64);
    let old = b.reg(Ty::B128);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k4 = b.k_u32(4);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    let nine = b.konst(Ty::B128, 0x0000_0009_0000_0009_0000_0009_0000_0009);
    b.addr_of(a, g, k0);
    b.site("atom_exch_b128", 1);
    b.push(Instr::Atom { op: AtomOp::Exch, ty: Ty::B128, dst: Some(old), addr: a.into(), space: AddrSpace::Global, value: nine, cmp: None, sem: Sem::Relaxed, scope: Scope::Gpu, ftz: false });
    b.no_site();
    b.st(Ty::B128, out, k0, old);
    let seven = b.konst(Ty::B128, 0x0000_0007_0000_0007_0000_0007_0000_0007);
    let want = b.konst(Ty::B128, 0x0000_0008_0000_0007_0000_0006_0000_0005);
    b.addr_of(a, g, k4);
    b.site("atom_cas_b128", 2);
    b.push(Instr::Atom { op: AtomOp::Cas, ty: Ty::B128, dst: Some(old), addr: a.into(), space: AddrSpace::Global, value: seven, cmp: Some(want), sem: Sem::Relaxed, scope: Scope::Gpu, ftz: false });
    b.no_site();
    b.st(Ty::B128, out, k1, old);
    b.end_if();
    b.exit();
    scenario(
        "atom_b128",
        b.build_module(),
        inputs(vec![
            ("g", u32_buf(1..=8)),
            ("out", ArgValue::Buffer { bytes: vec![0; 32], valid: None }),
        ]),
    )
}

/// M5 / W5-8: in a one-CTA cluster every lane `st.async`s its word into
/// shared memory, completing on an mbarrier armed with
/// `arrive.expect_tx(128)`; the warp waits and copies the words out.
pub fn st_async_copy() -> Scenario {
    let mut b = ProgramBuilder::new("st_async_copy", 32);
    b.cluster(1, 1, 1);
    let out = b.global("out", Dtype::U32);
    let bar = b.shared("bar", Dtype::U64, 1);
    let sm = b.shared("s", Dtype::U32, 32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let barr = b.reg(Ty::U32);
    let da = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k100 = b.k_u32(100);
    let k128 = b.k_u32(128);
    b.smem_addr(barr, bar, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.mbar_init(barr, 1);
    b.end_if();
    b.fence(FenceKind::MbarrierInit, Sem::Release, Scope::Cluster);
    let full = b.k_u32(u32::MAX);
    b.push(Instr::WarpSync { membermask: full });
    b.if_(p);
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
    b.end_if();
    b.push(Instr::WarpSync { membermask: full });
    b.add_u32(v, lane, k100);
    b.smem_addr(da, sm, lane);
    b.site("st_async", 1);
    b.push(Instr::StAsync(StAsyncArgs { ty: Ty::U32, value: v.into(), addr: da.into(), mbar: Some(barr.into()), red: None, sem: Sem::Release, scope: Scope::Cluster }));
    b.no_site();
    b.mbar_wait_parity(barr, k0);
    b.ld_u32(v, sm, lane);
    b.st_u32(out, lane, v);
    b.exit();
    scenario("st_async_copy", b.build_module(), inputs(vec![("out", u32_buf([0; 32]))]))
}

/// M7: `tcgen05.st` to columns after they were deallocated: BadAddress.
pub fn tcgen_after_dealloc() -> Scenario {
    let mut b = ProgramBuilder::new("tcgen_after_dealloc", 32);
    let slot = b.shared("taddr", Dtype::U32, 1);
    let sa = b.reg(Ty::U32);
    let t = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k32 = b.k_u32(32);
    b.smem_addr(sa, slot, k0);
    b.push(Instr::TcgenAlloc { dst: sa.into(), ncols: k32, cta_group: 1, exclusive: false });
    b.bar_sync(0);
    b.ld_u32(t, slot, k0);
    b.push(Instr::TcgenDealloc { taddr: t.into(), ncols: k32, cta_group: 1, exclusive: false });
    b.push(Instr::TcgenRelinquish { cta_group: 1 });
    b.site("st_after_dealloc", 1);
    b.push(Instr::TcgenSt(Box::new(TcgenStArgs {
        srcs: vec![lane.into()],
        taddr: t.into(),
        row: k0,
        col: k0,
        shape: TcShape::S32x32b,
        num: 1,
        unpack: false,
    })));
    b.push(Instr::TcgenWait { st: true });
    b.no_site();
    b.exit();
    scenario("tcgen_after_dealloc", b.build_module(), Inputs::default())
}

/// M11: warp 1 lane 0 writes a declared word `writes` times (values >= 2),
/// then `flag = 1` (release); warp 0 `wait_until(flag == 1)`. Past
/// `MAX_WORD_HISTORY` writes the verdict cannot be computed: incomplete
/// (with a history-consuming observer).
pub fn word_history_overflow(writes: u32) -> Scenario {
    let mut b = ProgramBuilder::new("word_history_overflow", 64);
    let flag = b.global("flag", Dtype::U32);
    b.declare_sync_words(flag);
    let w = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let k = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    let fa = b.reg(Ty::U64);
    let got = b.reg(Ty::U32);
    let arg = b.reg(Ty::U32);
    let res = b.reg(Ty::PRED);
    b.warp_id(w);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k2 = b.k_u32(2);
    let kn = b.k_u32(writes);
    b.addr_of(fa, flag, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, w, k1);
    b.if_(p);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.mov(k, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, k, kn);
    b.loop_if(p);
    b.add_u32(v, k, k2);
    st_sem(&mut b, flag, k0, v.into(), Sem::Relaxed, Scope::Gpu);
    b.add_u32(k, k, k1);
    b.loop_end();
    st_sem(&mut b, flag, k0, k1, Sem::Release, Scope::Gpu);
    b.end_if();
    b.else_();
    b.site("wait_until", 1);
    let placeholder = b.push(Instr::Nop);
    b.no_site();
    b.end_if();
    b.exit();
    let mut prog = b.build();
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
    let mut sc = scenario("word_history_overflow", Module::new(vec![prog]), inputs(vec![("flag", u32_buf([0]))]));
    sc.config.loop_budget = 1 << 40;
    sc
}

/// M6: `wait_until(flag >= limit[0])` whose predicate reads the bound
/// buffer `limit`: the read must be recorded in the verdict's
/// `pred_reads` (the load fast path must not skip the capture).
pub fn wait_until_pred_reads() -> Scenario {
    let mut b = ProgramBuilder::new("wait_until_pred_reads", 64);
    let flag = b.global("flag", Dtype::U32);
    let limit = b.global("limit", Dtype::U32);
    b.declare_sync_words(flag);
    let w = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let fa = b.reg(Ty::U64);
    let got = b.reg(Ty::U32);
    let arg = b.reg(Ty::U32);
    let lim = b.reg(Ty::U32);
    let res = b.reg(Ty::PRED);
    b.warp_id(w);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k3 = b.k_u32(3);
    b.addr_of(fa, flag, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, w, k1);
    b.if_(p);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    st_sem(&mut b, flag, k0, k3, Sem::Release, Scope::Gpu);
    b.end_if();
    b.else_();
    b.site("wait_until", 1);
    let placeholder = b.push(Instr::Nop);
    b.no_site();
    b.end_if();
    b.exit();
    let mut prog = b.build();
    let start = Pc(prog.code.len() as u32);
    let k0c = match k0 {
        Operand::Const(c) => c,
        _ => unreachable!(),
    };
    prog.code.push(Instr::Load { ty: Ty::U32, dst: lim, buf: limit, offset: Operand::Const(k0c), sem: Sem::Weak, scope: Scope::Cta, mods: MemMods::default() });
    prog.code.push(Instr::Compare { op: CmpOp::Ge, ty: Ty::U32, dst: res, a: arg.into(), b: lim.into() });
    prog.code_sites.push(crate::site::SiteId::NONE);
    prog.code_sites.push(crate::site::SiteId::NONE);
    prog.preds.push(PredProgram { arg, start, end: Pc(start.0 + 2), result: res, reads_memory: true });
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
    scenario("wait_until_pred_reads", Module::new(vec![prog]), inputs(vec![("flag", u32_buf([0])), ("limit", u32_buf([2]))]))
}

/// M9 / review scenario 8: a three-way divergent hand-off inside a loop
/// with `continue` in one arm:
/// `if lane < 8 { wait A; arrive B } else if lane < 16 { arrive A; wait C;
/// continue } else { arrive A; arrive C; wait B }`, A/B/C with counts 24/8/16 (re-armed each
/// iteration), two iterations; each arm appends its iteration's tag to
/// `out[lane]` exactly once per iteration.
pub fn divergent_nesting() -> Scenario {
    let mut b = ProgramBuilder::new("divergent_nesting", 32);
    let out = b.global("out", Dtype::U32);
    let bars = b.shared("bars", Dtype::U64, 3);
    let lane = b.reg(Ty::U32);
    let it = b.reg(Ty::U32);
    let acc = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let q = b.reg(Ty::PRED);
    let ma = b.reg(Ty::U32);
    let mb = b.reg(Ty::U32);
    let mc = b.reg(Ty::U32);
    let par = b.reg(Ty::U32);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k2 = b.k_u32(2);
    let k8 = b.k_u32(8);
    let k16 = b.k_u32(16);
    let k10 = b.k_u32(10);
    b.smem_addr(ma, bars, k0);
    b.smem_addr(mb, bars, k1);
    b.smem_addr(mc, bars, k2);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.mbar_init(ma, 24);
    b.mbar_init(mb, 8);
    b.mbar_init(mc, 16);
    b.end_if();
    b.fence(FenceKind::MbarrierInit, Sem::Release, Scope::Cluster);
    let full = b.k_u32(u32::MAX);
    b.push(Instr::WarpSync { membermask: full });
    b.mov(acc, k0);
    b.mov(it, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, it, k2);
    b.loop_if(p);
    b.binary(BinOp::And, Ty::U32, par, it, k1);
    b.mul(Ty::U32, acc, acc, k10);
    b.add_u32(acc, acc, k1);
    b.add_u32(it, it, k1);
    b.compare(CmpOp::Lt, Ty::U32, p, lane, k8);
    b.if_(p);
    b.site("arm0_wait_a", 1);
    b.mbar_wait_parity(ma, par);
    b.mbar_arrive(mb, None);
    b.no_site();
    b.else_();
    b.compare(CmpOp::Lt, Ty::U32, q, lane, k16);
    b.if_(q);
    b.mbar_arrive(ma, None);
    b.site("arm1_wait_c", 2);
    b.mbar_wait_parity(mc, par);
    b.no_site();
    b.continue_();
    b.else_();
    b.mbar_arrive(ma, None);
    b.mbar_arrive(mc, None);
    b.site("arm2_wait_b", 3);
    b.mbar_wait_parity(mb, par);
    b.no_site();
    b.end_if();
    b.end_if();
    b.add_u32(acc, acc, k1);
    b.loop_end();
    b.st_u32(out, lane, acc);
    b.exit();
    scenario("divergent_nesting", b.build_module(), inputs(vec![("out", u32_buf([0; 32]))]))
}

/// M10 cycle: two single-CTA clusters each `st.release` their own flag and
/// then `ld.acquire` the other's in the same round. Both read the
/// round-start 0 (store buffering); no sequential event stream represents
/// that, so an observed run carries an `incomplete` diagnostic.
pub fn cross_cluster_sb() -> Scenario {
    let mut b = ProgramBuilder::new("cross_cluster_sb", 32);
    b.grid(2, 1, 1);
    let flags = b.global("flags", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let cta = b.reg(Ty::U32);
    let other = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let v = b.reg(Ty::U32);
    b.read_special(cta, SpecialReg::CtaLinear);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    b.binary(BinOp::Xor, Ty::U32, other, cta, k1);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    st_sem(&mut b, flags, cta.into(), k1, Sem::Release, Scope::Gpu);
    ld_sem(&mut b, v, flags, other.into(), Sem::Acquire, Scope::Gpu);
    b.st_u32(out, cta, v);
    b.end_if();
    b.exit();
    scenario("cross_cluster_sb", b.build_module(), inputs(vec![("flags", u32_buf([0, 0])), ("out", u32_buf([9, 9]))]))
}

// ---------------------------------------------------------------------------
// Conformance-sweep regressions (CONTRACT_REQUESTS "v2 conformance")
// ---------------------------------------------------------------------------

/// V2C-9: `ld.global` of a kernel-parameter generic address (the address
/// of a parameter, as kernels take for `__grid_constant__` tensor maps)
/// reads the parameter. With `store`, lane 0 then stores there: an error
/// finding (parameters are read-only), never a crash.
pub fn param_aperture(store: bool) -> Scenario {
    let mut b = ProgramBuilder::new("param_aperture", 32);
    let _x = b.scalar_param("x", Dtype::U64);
    let out = b.global("out", Dtype::U64);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let v = b.reg(Ty::U64);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let pa = b.konst(Ty::U64, crate::arena::addr::GENERIC_PARAM_BASE as u128);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.site("param_ld_global", 1);
    b.push(Instr::LoadAddr { ty: Ty::U64, dst: v, addr: pa, space: AddrSpace::Global, sem: Sem::Weak, scope: Scope::Cta, mods: MemMods::default() });
    b.no_site();
    b.st(Ty::U64, out, k0, v);
    if store {
        b.site("param_st_global", 2);
        b.push(Instr::StoreAddr { ty: Ty::U64, addr: pa, space: AddrSpace::Global, value: v.into(), sem: Sem::Weak, scope: Scope::Cta, mods: MemMods::default() });
        b.no_site();
    }
    b.end_if();
    b.exit();
    scenario(
        if store { "param_aperture_store" } else { "param_aperture" },
        b.build_module(),
        inputs(vec![("x", ArgValue::Scalar(0x1234_5678_9abc)), ("out", ArgValue::Buffer { bytes: vec![0; 8], valid: None })]),
    )
}

/// V2C-14 family: one warpgroup, no launch-bounds `regs_per_thread`,
/// `setmaxnreg.inc 256`. The initial budget is the legacy caller base
/// (min of the even 512 split, the largest inc target and the launch
/// bounds) = 256, so the increase is legal.
pub fn setmaxnreg_default_budget() -> Scenario {
    let mut b = ProgramBuilder::new("setmaxnreg_default_budget", 128);
    let out = b.global("out", Dtype::U32);
    let tid = b.reg(Ty::U32);
    b.thread_rank(tid);
    b.site("setmaxnreg_inc", 1);
    b.push(Instr::SetMaxNReg { inc: true, count: 256 });
    b.no_site();
    b.st_u32(out, tid, tid);
    b.exit();
    scenario("setmaxnreg_default_budget", b.build_module(), inputs(vec![("out", u32_buf(vec![0; 128]))]))
}

/// W8-6: parameters `x` (words 0..4) and `z` (words 2..6) are views of ONE
/// host array `mem`: lane i < 4 writes `x[i] = 10 + i`, then (after a warp
/// sync) lane i < 4 copies `z[i]` to `out[i]`: z[0], z[1] alias x[2], x[3].
pub fn aliased_views() -> Scenario {
    let mut b = ProgramBuilder::new("aliased_views", 32);
    let x = b.global("x", Dtype::U32);
    let z = b.global("z", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let v = b.reg(Ty::U32);
    b.lane_id(lane);
    let k4 = b.k_u32(4);
    let k10 = b.k_u32(10);
    b.compare(CmpOp::Lt, Ty::U32, p, lane, k4);
    b.if_(p);
    b.add_u32(v, lane, k10);
    b.st_u32(x, lane, v);
    b.end_if();
    let full = b.k_u32(u32::MAX);
    b.push(Instr::WarpSync { membermask: full });
    b.if_(p);
    b.ld_u32(v, z, lane);
    b.st_u32(out, lane, v);
    b.end_if();
    b.exit();
    scenario(
        "aliased_views",
        b.build_module(),
        inputs(vec![
            ("mem", u32_buf([100, 101, 102, 103, 104, 105])),
            ("x", ArgValue::View { target: "mem".into(), offset: 0, len: 16 }),
            ("z", ArgValue::View { target: "mem".into(), offset: 8, len: 16 }),
            ("out", u32_buf([0; 4])),
        ]),
    )
}

/// W4-9: `prefetch.L1::32B.valid_addr` needs an addressable global byte
/// (`valid = false`: an unmapped address -> BadAddress), and
/// `applypriority.async.bulk` joins the issuing thread's bulk group, which
/// a later commit / wait_group counts.
pub fn hint_ops(valid: bool) -> Scenario {
    let mut b = ProgramBuilder::new("hint_ops", 32);
    let out = b.global("out", Dtype::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let a = b.reg(Ty::U64);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k64 = b.k_u32(64);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    if valid {
        b.addr_of(a, out, k0);
    } else {
        let bad = b.konst(Ty::U64, 0x40);
        b.mov(a, bad);
    }
    b.site("prefetch_valid_addr", 1);
    b.ptx("tirx.ptx.prefetch_valid_addr", &["global", "L1::32B", "valid_addr"], &[], &[a.into()]);
    b.site("applypriority_bulk", 2);
    b.ptx("tirx.ptx.applypriority_async_bulk", &["async", "bulk", "global", "bulk_group", "L2::evict_normal"], &[], &[a.into(), k64]);
    b.push(Instr::AsyncCommit { domain: Domain::Bulk });
    b.push(Instr::AsyncWait { domain: Domain::Bulk, n: 0, read: false });
    b.no_site();
    b.end_if();
    b.st_u32(out, lane, lane);
    b.exit();
    scenario(if valid { "hint_ops" } else { "hint_ops_bad_addr" }, b.build_module(), inputs(vec![("out", u32_buf([0; 32]))]))
}

/// `tcgen05.alloc.exclusive` of 576 columns, then dealloc: legal on
/// sm_107f (PTX Table 58), an `InvalidColumns` error on sm_100a.
pub fn tcgen_exclusive_576(arch: &str) -> Scenario {
    let mut b = ProgramBuilder::new("tcgen_exclusive_576", 32);
    let out = b.global("out", Dtype::U32);
    let slot = b.shared("taddr", Dtype::U32, 1);
    let sa = b.reg(Ty::U32);
    let t = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k576 = b.k_u32(576);
    b.smem_addr(sa, slot, k0);
    b.site("alloc_exclusive_576", 1);
    b.push(Instr::TcgenAlloc { dst: sa.into(), ncols: k576, cta_group: 1, exclusive: true });
    b.no_site();
    b.bar_sync(0);
    b.ld_u32(t, slot, k0);
    b.push(Instr::TcgenDealloc { taddr: t.into(), ncols: k576, cta_group: 1, exclusive: true });
    b.push(Instr::TcgenRelinquish { cta_group: 1 });
    b.st_u32(out, lane, t);
    b.exit();
    let mut prog = b.build();
    prog.arch = Some(arch.to_string());
    let name = if arch.starts_with("sm_107") { "tcgen_exclusive_576_sm107" } else { "tcgen_exclusive_576_sm100" };
    scenario(name, Module::new(vec![prog]), inputs(vec![("out", u32_buf([9; 32]))]))
}

/// V2C-19/20: a per-lane register-space array (`Space::Reg` buffer) of 4
/// words; each lane writes elements 0 and 1 and reads element 2, which was
/// never written: one `UninitRead` in `space: register`, read as zero.
pub fn reg_buffer_uninit() -> Scenario {
    let mut b = ProgramBuilder::new("reg_buffer_uninit", 32);
    let out = b.global("out", Dtype::U32);
    let rb = b.per_lane("acc", crate::arena::Space::Reg, Dtype::U32, 4);
    let lane = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k2 = b.k_u32(2);
    b.st_u32(rb, k0, lane);
    b.st_u32(rb, k1, lane);
    b.site("reg_uninit_ld", 1);
    b.ld_u32(v, rb, k2);
    b.no_site();
    b.ld_u32(lane, rb, k1);
    b.add_u32(v, v, lane);
    b.st_u32(out, lane, v);
    b.exit();
    scenario("reg_buffer_uninit", b.build_module(), inputs(vec![("out", u32_buf([9; 32]))]))
}

/// W1 ruling (`mapa.shared::cluster.u64`): in a 2-CTA cluster, CTA 1 lane 0
/// maps CTA 0's mbarrier with `mapa.shared::cluster` and arrives on it
/// through the u64 value, once as a `SharedCluster` operand and once as a
/// `Generic` one (a rank-tagged 32-bit window address zero-extended); CTA 0
/// (count 2) waits for the phase both arrivals complete.
pub fn mapa_cluster_arrive() -> Scenario {
    let mut b = ProgramBuilder::new("mapa_cluster_arrive", 32);
    b.grid(2, 1, 1);
    b.cluster(2, 1, 1);
    let out = b.global("out", Dtype::U32);
    let bar = b.shared("bar", Dtype::U64, 1);
    let lane = b.reg(Ty::U32);
    let rank = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let q = b.reg(Ty::PRED);
    let m = b.reg(Ty::U32);
    let rem = b.reg(Ty::U64);
    let cta = b.reg(Ty::U32);
    b.lane_id(lane);
    b.read_special(rank, SpecialReg::ClusterCtaRank);
    b.read_special(cta, SpecialReg::CtaLinear);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    b.smem_addr(m, bar, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.mbar_init(m, 2);
    b.end_if();
    b.fence(FenceKind::MbarrierInit, Sem::Release, Scope::Cluster);
    b.push(Instr::ClusterArrive { sem: Sem::Release, aligned: true });
    b.push(Instr::ClusterWait { acquire: true, aligned: true });
    b.compare(CmpOp::Eq, Ty::U32, q, rank, k1);
    b.if_(p);
    b.if_(q);
    b.site("mapa", 1);
    b.push(Instr::Mapa { dst: rem, src: m.into(), rank: k0, space: AddrSpace::SharedCluster });
    for (i, space) in [AddrSpace::SharedCluster, AddrSpace::Generic].into_iter().enumerate() {
        b.site(if i == 0 { "arrive_cluster" } else { "arrive_generic" }, 2 + i as u32);
        b.push(Instr::MbarArrive(MbarArriveArgs {
            mbar: rem.into(),
            space,
            count: None,
            expect_tx: None,
            drop: false,
            no_complete: false,
            sem: Sem::Release,
            scope: Scope::Cluster,
            multicast: None,
            state: None,
        }));
    }
    b.no_site();
    b.else_();
    b.push(Instr::MbarWait { mbar: m.into(), space: AddrSpace::Shared, phase: PhaseArg::Parity(k0), sem: Sem::Acquire, scope: Scope::Cluster });
    b.end_if();
    b.st_u32(out, cta, k1);
    b.end_if();
    // Keep CTA 0's barrier alive until CTA 1 is done with it.
    b.push(Instr::ClusterArrive { sem: Sem::Release, aligned: true });
    b.push(Instr::ClusterWait { acquire: true, aligned: true });
    b.exit();
    scenario("mapa_cluster_arrive", b.build_module(), inputs(vec![("out", u32_buf([0, 0]))]))
}

/// V2C-24: a TMEM view (`Space::Tmem` buffer, 4 columns) used without any
/// `tcgen05.alloc` under `Requirements::implicit_tmem`: lane l stores
/// `l * 10 + j` to its row's columns j and reads them back.
pub fn implicit_tmem() -> Scenario {
    let mut b = ProgramBuilder::new("implicit_tmem", 32);
    let out = b.global("out", Dtype::U32);
    let tm = b.per_lane("acc_tmem", crate::arena::Space::Tmem, Dtype::U32, 128 * 4);
    let lane = b.reg(Ty::U32);
    let idx = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    b.lane_id(lane);
    let k4 = b.k_u32(4);
    let k10 = b.k_u32(10);
    for j in 0..4u32 {
        let kj = b.k_u32(j);
        b.mul(Ty::U32, idx, lane, k4);
        b.add_u32(idx, idx, kj);
        b.mul(Ty::U32, v, lane, k10);
        b.add_u32(v, v, kj);
        b.st_u32(tm, idx, v);
    }
    // TMEM stores are tcgen-proxy writes: order them before the loads.
    b.push(Instr::TcgenWait { st: true });
    for j in 0..4u32 {
        let kj = b.k_u32(j);
        b.mul(Ty::U32, idx, lane, k4);
        b.add_u32(idx, idx, kj);
        b.ld_u32(v, tm, idx);
        b.st_u32(out, idx, v);
    }
    b.exit();
    let mut prog = b.build();
    {
        let d = prog.buffers.iter_mut().find(|d| d.name == "acc_tmem").expect("tmem buffer");
        d.shape = vec![DimExpr::Const(128), DimExpr::Const(4)];
    }
    prog.requirements.implicit_tmem = true;
    scenario("implicit_tmem", Module::new(vec![prog]), inputs(vec![("out", u32_buf(vec![0; 128]))]))
}

/// Packed-FP4 TMA store tensor map of [`fp4_tma_store`] (dst `dst`: 40 x 2
/// E2M1 elements, 32-byte rows; box 32 x 2 at column 32, so columns 40..64
/// are out of bounds).
pub fn fp4_store_desc() -> crate::oplib::TensorMapDesc {
    crate::oplib::TensorMapDesc {
        global_address: 0,
        rank: 2,
        elem: Some(Dtype::E2M1),
        global_dim: [40, 2, 1, 1, 1],
        global_stride: [32, 0, 0, 0, 0],
        box_dim: [32, 2, 1, 1, 1],
        element_stride: [1; 5],
        ..Default::default()
    }
}

/// W4-11: a packed FP4 TMA store, partly out of bounds, whose plan carries
/// masked sub-byte fragments: shared byte i = 0x10 * (i % 16) + i / 2 + 1,
/// `dst` pre-filled with 0xEE (bytes outside the written nibbles keep it).
pub fn fp4_tma_store() -> Scenario {
    let mut b = ProgramBuilder::new("fp4_tma_store", 32);
    let _dst = b.global("dst", Dtype::U8);
    let tile = b.shared("tile", Dtype::U8, 64);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let v = b.reg(Ty::U32);
    let w = b.reg(Ty::U8);
    let sa = b.reg(Ty::U32);
    let tmap = b.reg(Ty::U64);
    let idx = b.reg(Ty::U32);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k2 = b.k_u32(2);
    let k15 = b.k_u32(15);
    let k16 = b.k_u32(16);
    let k32 = b.k_u32(32);
    let k1 = b.k_u32(1);
    for half in [k0, k32] {
        b.add_u32(idx, lane, half);
        b.binary(BinOp::And, Ty::U32, v, idx, k15);
        b.mul(Ty::U32, v, v, k16);
        let t = b.reg(Ty::U32);
        b.binary(BinOp::Div, Ty::U32, t, idx, k2);
        b.add_u32(v, v, t);
        b.add_u32(v, v, k1);
        b.cast(Ty::U32, Ty::U8, w, v);
        b.st(Ty::U8, tile, idx, w);
    }
    b.smem_addr(sa, tile, k0);
    let tmap_pc = b.push(Instr::Nop);
    b.fence(FenceKind::ProxyAsync(Some(AddrSpace::Shared)), Sem::Weak, Scope::Cta);
    let full = b.k_u32(u32::MAX);
    b.push(Instr::WarpSync { membermask: full });
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    let kc = b.k_i32(32);
    let kr = b.k_i32(0);
    b.site("fp4_tma_store", 1);
    b.push(Instr::Tma(Box::new(TmaArgs {
        dir: TmaDir::Store,
        mode: TmaMode::Tile,
        tmap: tmap.into(),
        tmap_space: AddrSpace::Generic,
        coords: vec![kc, kr],
        im2col_offsets: vec![],
        smem: sa.into(),
        smem_space: AddrSpace::Shared,
        completion: BulkCompletion::Group,
        multicast: None,
        cta_group: 0,
        overrides: vec![],
        report: None,
        mods: MemMods::default(),
    })));
    b.push(Instr::AsyncCommit { domain: Domain::Bulk });
    b.push(Instr::AsyncWait { domain: Domain::Bulk, n: 0, read: false });
    b.no_site();
    b.end_if();
    b.exit();
    let mut prog = b.build();
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
    scenario(
        "fp4_tma_store",
        Module::new(vec![prog]),
        inputs(vec![
            ("dst", ArgValue::Buffer { bytes: vec![0xee; 64], valid: None }),
            ("tmap", ArgValue::TensorMapOf { base: "dst".into(), offset: 0, desc: fp4_store_desc() }),
        ]),
    )
}

/// Ruling: `%smid`, `%clock`, `%clock64`, `%globaltimer`, `%gridid` read
/// the representative 0 (legacy `mov_sreg`); `%nsmid` keeps its value.
pub fn physical_sregs() -> Scenario {
    let mut b = ProgramBuilder::new("physical_sregs", 32);
    b.grid(3, 1, 1);
    let out = b.global("out", Dtype::U64);
    let cta = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let idx = b.reg(Ty::U32);
    let v = b.reg(Ty::U64);
    b.read_special(cta, SpecialReg::CtaLinear);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k5 = b.k_u32(5);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    for (i, r) in [SpecialReg::SmId, SpecialReg::Clock, SpecialReg::Clock64, SpecialReg::GlobalTimer, SpecialReg::GridId].into_iter().enumerate() {
        let ki = b.k_u32(i as u32);
        b.read_special(v, r);
        b.mul(Ty::U32, idx, cta, k5);
        b.add_u32(idx, idx, ki);
        b.st(Ty::U64, out, idx, v);
    }
    b.end_if();
    b.exit();
    scenario("physical_sregs", b.build_module(), inputs(vec![("out", ArgValue::Buffer { bytes: vec![0xab; 15 * 8], valid: None })]))
}

/// Ruling: a multicast CTA mask naming a rank outside the cluster is an
/// error (BadAddress), not incomplete: a 1-CTA cluster arrives with
/// multicast mask 0b10.
pub fn multicast_outside_cluster() -> Scenario {
    let mut b = ProgramBuilder::new("multicast_outside_cluster", 32);
    b.cluster(1, 1, 1);
    let bar = b.shared("bar", Dtype::U64, 1);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let m = b.reg(Ty::U32);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k2 = b.k_u32(2);
    b.smem_addr(m, bar, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.mbar_init(m, 1);
    b.site("multicast_arrive", 1);
    b.push(Instr::MbarArrive(MbarArriveArgs {
        mbar: m.into(),
        space: AddrSpace::SharedCluster,
        count: None,
        expect_tx: None,
        drop: false,
        no_complete: false,
        sem: Sem::Release,
        scope: Scope::Cluster,
        multicast: Some(k2),
        state: None,
    }));
    b.no_site();
    b.end_if();
    b.exit();
    scenario("multicast_outside_cluster", b.build_module(), Inputs::default())
}

/// Contract batch 4: a 16-bit TMEM view (`implicit_tmem`) packs two
/// elements per 32-bit cell: lane l stores u16 `l * 100 + j` to elements
/// j = 0..4 of its row (cells 0..2) and reads them back; then lane l
/// overwrites element 1 only (a cell read-modify-write) and reads the cell.
pub fn tmem_subword() -> Scenario {
    let mut b = ProgramBuilder::new("tmem_subword", 32);
    let out = b.global("out", Dtype::U16);
    let cells = b.global("cells", Dtype::U32);
    let tm = b.per_lane("h_tmem", crate::arena::Space::Tmem, Dtype::U16, 128 * 4);
    let tw = b.per_lane("w_tmem", crate::arena::Space::Tmem, Dtype::U32, 128 * 2);
    let lane = b.reg(Ty::U32);
    let idx = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    let h = b.reg(Ty::U16);
    let c = b.reg(Ty::U32);
    b.lane_id(lane);
    let k2 = b.k_u32(2);
    let k4 = b.k_u32(4);
    let k100 = b.k_u32(100);
    for j in 0..4u32 {
        let kj = b.k_u32(j);
        b.mul(Ty::U32, idx, lane, k4);
        b.add_u32(idx, idx, kj);
        b.mul(Ty::U32, v, lane, k100);
        b.add_u32(v, v, kj);
        b.cast(Ty::U32, Ty::U16, h, v);
        b.st(Ty::U16, tm, idx, h);
    }
    b.push(Instr::TcgenWait { st: true });
    for j in 0..4u32 {
        let kj = b.k_u32(j);
        b.mul(Ty::U32, idx, lane, k4);
        b.add_u32(idx, idx, kj);
        b.ld(Ty::U16, h, tm, idx);
        b.st(Ty::U16, out, idx, h);
    }
    // Overwrite element 1 (high half of cell 0) with 0xBEEF.
    let k1 = b.k_u32(1);
    let kbeef = b.konst(Ty::U16, 0xbeef);
    b.mul(Ty::U32, idx, lane, k4);
    b.add_u32(idx, idx, k1);
    b.st(Ty::U16, tm, idx, kbeef);
    b.push(Instr::TcgenWait { st: true });
    b.mul(Ty::U32, idx, lane, k2);
    b.ld_u32(c, tw, idx);
    b.st_u32(cells, lane, c);
    b.exit();
    let mut prog = b.build();
    for d in prog.buffers.iter_mut() {
        // TMEM view shapes count 32-bit cells per row: 4 u16 = 2 cells.
        if d.name == "h_tmem" {
            d.shape = vec![DimExpr::Const(128), DimExpr::Const(2)];
        }
        if d.name == "w_tmem" {
            d.shape = vec![DimExpr::Const(128), DimExpr::Const(2)];
        }
    }
    prog.requirements.implicit_tmem = true;
    scenario(
        "tmem_subword",
        Module::new(vec![prog]),
        inputs(vec![("out", ArgValue::Buffer { bytes: vec![0; 256], valid: None }), ("cells", u32_buf(vec![0; 32]))]),
    )
}

/// W1 batch 4 repro: a 16-bit TMEM view with rows of 4 cells (8 elements);
/// thread t of 2 warps writes elements `t*8 .. t*8+8` (TMEM lane t, warp 1
/// in lanes 32..64) and reads them back.
pub fn tmem_f16_rows() -> Scenario {
    let mut b = ProgramBuilder::new("tmem_f16_rows", 64);
    let out = b.global("out", Dtype::U16);
    let tm = b.per_lane("physical", crate::arena::Space::Tmem, Dtype::F16, 128 * 8);
    let tid = b.reg(Ty::U32);
    let idx = b.reg(Ty::U32);
    let h = b.reg(Ty::U16);
    let v = b.reg(Ty::U32);
    b.thread_rank(tid);
    let k8 = b.k_u32(8);
    for j in 0..8u32 {
        let kj = b.k_u32(j);
        b.mul(Ty::U32, idx, tid, k8);
        b.add_u32(idx, idx, kj);
        b.add_u32(v, idx, kj);
        b.cast(Ty::U32, Ty::U16, h, v);
        b.st(Ty::U16, tm, idx, h);
    }
    b.push(Instr::TcgenWait { st: true });
    for j in 0..8u32 {
        let kj = b.k_u32(j);
        b.mul(Ty::U32, idx, tid, k8);
        b.add_u32(idx, idx, kj);
        b.ld(Ty::U16, h, tm, idx);
        b.st(Ty::U16, out, idx, h);
    }
    b.exit();
    let mut prog = b.build();
    for d in prog.buffers.iter_mut() {
        if d.name == "physical" {
            d.shape = vec![DimExpr::Const(128), DimExpr::Const(4)];
        }
    }
    prog.requirements.implicit_tmem = true;
    scenario("tmem_f16_rows", Module::new(vec![prog]), inputs(vec![("out", ArgValue::Buffer { bytes: vec![0; 64 * 16], valid: None })]))
}

/// W5-11: warp 0's lanes `cp.async` a word each into shared memory and
/// `cp.async.mbarrier.arrive.noinc` on `bar` (count 32); warp 1 waits on
/// the phase and reads the words: the arrive publishes the copies to that
/// phase (race-free).
pub fn cp_async_mbar_publish() -> Scenario {
    let mut b = ProgramBuilder::new("cp_async_mbar_publish", 64);
    let input = b.global("in", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let bar = b.shared("bar", Dtype::U64, 1);
    let sm = b.shared("s", Dtype::U32, 32);
    let tid = b.reg(Ty::U32);
    let w = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let barr = b.reg(Ty::U32);
    let sa = b.reg(Ty::U32);
    let ga = b.reg(Ty::U64);
    let v = b.reg(Ty::U32);
    b.thread_rank(tid);
    b.warp_id(w);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    b.smem_addr(barr, bar, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, tid, k0);
    b.if_(p);
    b.mbar_init(barr, 32);
    b.end_if();
    b.fence(FenceKind::MbarrierInit, Sem::Release, Scope::Cluster);
    b.bar_sync(0);
    b.compare(CmpOp::Eq, Ty::U32, p, w, k0);
    b.if_(p);
    b.smem_addr(sa, sm, lane);
    b.addr_of(ga, input, lane);
    b.site("cp_async", 1);
    b.push(Instr::CpAsync { dst: sa.into(), src: ga.into(), cp_size: 4, src_size: None, ignore_src: None, mods: MemMods::default() });
    b.site("cp_async_mbar_arrive", 2);
    b.push(Instr::CpAsyncMbarArrive { mbar: barr.into(), space: AddrSpace::Shared, noinc: true });
    b.no_site();
    b.else_();
    b.site("consumer_wait", 3);
    b.mbar_wait_parity(barr, k0);
    b.no_site();
    b.ld_u32(v, sm, lane);
    b.st_u32(out, lane, v);
    b.end_if();
    b.exit();
    scenario("cp_async_mbar_publish", b.build_module(), inputs(vec![("in", u32_buf((0..32).map(|i| i * 11 + 1))), ("out", u32_buf([0; 32]))]))
}

/// Contract batch 4 / W1: [`tma_load`] with its tensor map encoded by the
/// host prelude from a spec whose box rows are a runtime `DimExpr`
/// (`0 - neg_rows`, `neg_rows` an int32 -4: also checks scalar sign
/// extension). Same output as [`tma_load`].
pub fn tma_load_param_box() -> Scenario {
    let mut s = tma_load();
    let prog = &mut s.module.kernels[0];
    let pid = ParamId(prog.host_abi.len() as u32);
    prog.host_abi.push(ParamSlot {
        name: "neg_rows".into(),
        local_name: "neg_rows".into(),
        aliases: vec![],
        kind: ParamKind::Scalar,
        dtype: Some(Ty::S32),
        shape: vec![],
        tensor_map: None,
        implicit_base: None,
        buf: None,
    });
    let src = ParamId(prog.host_abi.iter().position(|p| p.name == "src").expect("src") as u32);
    let slot = prog.host_abi.iter_mut().find(|p| p.name == "tmap").expect("tmap");
    slot.tensor_map = Some(TensorMapSpec {
        dtype: Dtype::F32,
        rank: 2,
        global_dim: vec![DimExpr::Const(TMA_COLS as i64), DimExpr::Const(TMA_ROWS as i64)],
        global_stride: vec![DimExpr::Const(TMA_COLS as i64 * 4)],
        box_dim: vec![DimExpr::Const(TMA_COLS as i64), DimExpr::Sub(Box::new(DimExpr::Const(0)), Box::new(DimExpr::Param(pid)))],
        element_stride: vec![DimExpr::Const(1), DimExpr::Const(1)],
        interleave: 0,
        swizzle: 0,
        l2_promotion: 0,
        oob_fill: 0,
        base_offset: DimExpr::Const(0),
        force_cu_dtype: None,
    });
    slot.implicit_base = Some(src);
    prog.validate().expect("valid");
    s.inputs.args.remove("tmap");
    s.inputs.args.insert("neg_rows".into(), ArgValue::Scalar((-(TMA_BOX_ROWS as i32)) as u32 as u64));
    s.name = "tma_load_param_box";
    s
}

/// Scenarios that are deliberately racy or only meaningful with a specific
/// configuration (each test states its expectation): not in [`all`].
pub fn special() -> Vec<Scenario> {
    vec![mbar_latch(), tcgen_exclusive_576("sm_107f"), implicit_tmem(), tmem_subword(), tmem_f16_rows(), cross_cluster_flag(false), cross_cluster_sb(), cp_async_no_wait(), word_history_overflow(MAX_HISTORY_PROBE), readonly_proxy("after"), readonly_proxy("cross_cta"), tcgen_ld_wide(false), divergent_named_barrier("other_id"), divergent_named_barrier("exit")]
}

/// Writes in [`word_history_overflow`] past `MAX_WORD_HISTORY`.
pub const MAX_HISTORY_PROBE: u32 = (crate::interp::aux::MAX_WORD_HISTORY as u32) + 8;

/// Every scenario (for differential runs across backends).
pub fn all() -> Vec<Scenario> {
    vec![
        vector_add(),
        polled_flag_words(),
        copy_report_16(true),
        tcgen_alloc_lanes_read(),
        divergent_named_barrier("same"),
        tcgen_ld_wide(true),
        readonly_proxy("clean"),
        readonly_proxy("disjoint"),
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
        exit_then_barrier(false),
        exit_then_barrier(true),
        bounded_probe_loop(),
        nested_spin_break(),
        load_flag_spin(true),
        load_flag_spin(false),
        cross_cluster_flag(true),
        bulk_wait_read(),
        lane_split_mbarrier(),
        atom_b128(),
        st_async_copy(),
        tcgen_after_dealloc(),
        wait_until_pred_reads(),
        divergent_nesting(),
        param_aperture(false),
        param_aperture(true),
        setmaxnreg_default_budget(),
        aliased_views(),
        hint_ops(true),
        hint_ops(false),
        tcgen_cp_ld_with(true),
        tcgen_exclusive_576("sm_100a"),
        reg_buffer_uninit(),
        mapa_cluster_arrive(),
        tma_load_param_box(),
        cp_async_mbar_publish(),
        fp4_tma_store(),
        physical_sregs(),
        multicast_outside_cluster(),
    ]
}
