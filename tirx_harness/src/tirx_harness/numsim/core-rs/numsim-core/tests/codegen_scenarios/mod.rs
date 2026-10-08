//! Hand-built programs shared by `tests/codegen_equivalence.rs` and
//! `benches/codegen.rs` (W7). Each covers a dispatch shape the codegen
//! printer must get right: straight-line code, divergence, loops with
//! break/continue, barriers, mbarriers, atomics, generic PTX ops, errors,
//! multi-kernel modules.

#![allow(dead_code)]

use numsim_core::program::*;
use numsim_core::sched::{ArgValue, Inputs};
use numsim_core::testutil::ProgramBuilder;
use numsim_core::{Dtype, Module, Ty};

pub struct Scenario {
    pub name: &'static str,
    pub module: Module,
    pub inputs: Inputs,
}

fn f32_buf(v: impl IntoIterator<Item = f32>) -> ArgValue {
    ArgValue::Buffer { bytes: v.into_iter().flat_map(|x| x.to_le_bytes()).collect(), valid: None }
}

fn u32_buf(v: impl IntoIterator<Item = u32>) -> ArgValue {
    ArgValue::Buffer { bytes: v.into_iter().flat_map(|x| x.to_le_bytes()).collect(), valid: None }
}

fn inputs(args: Vec<(&str, ArgValue)>) -> Inputs {
    Inputs { args: args.into_iter().map(|(k, v)| (k.to_string(), v)).collect(), ..Default::default() }
}

/// `i = ctaid * ntid + tid`.
fn global_index(b: &mut ProgramBuilder, threads: u32) -> Reg {
    let cta = b.reg(Ty::U32);
    let tid = b.reg(Ty::U32);
    let i = b.reg(Ty::U32);
    b.read_special(cta, SpecialReg::CtaLinear);
    b.thread_rank(tid);
    let n = b.k_u32(threads);
    b.mul(Ty::U32, i, cta, n);
    b.add_u32(i, i, tid);
    i
}

/// `z[i] = x[i] + y[i]`, 2 CTAs x 64 threads.
pub fn vadd() -> Scenario {
    let mut b = ProgramBuilder::new("vadd", 64);
    b.grid(2, 1, 1);
    b.site("vadd.ld", 1);
    let x = b.global("x", Dtype::F32);
    let y = b.global("y", Dtype::F32);
    let z = b.global("z", Dtype::F32);
    let i = global_index(&mut b, 64);
    let a = b.reg(Ty::F32);
    let c = b.reg(Ty::F32);
    b.ld_f32(a, x, i);
    b.ld_f32(c, y, i);
    b.add_f32(a, a, c);
    b.site("vadd.st", 2);
    b.st_f32(z, i, a);
    Scenario {
        name: "vadd",
        module: b.build_module(),
        inputs: inputs(vec![
            ("x", f32_buf((0..128).map(|v| v as f32 * 0.5))),
            ("y", f32_buf((0..128).map(|v| 1.0 - v as f32))),
            ("z", f32_buf([0.0; 128])),
        ]),
    }
}

/// Register-only loop: `acc = fma(acc, a, b)` `iters` times per thread.
pub fn scalar_loop(iters: u32) -> Scenario {
    let mut b = ProgramBuilder::new("scalar_loop", 32);
    let out = b.global("out", Dtype::F32);
    let tid = b.reg(Ty::U32);
    let k = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let acc = b.reg(Ty::F32);
    let tf = b.reg(Ty::F32);
    b.thread_rank(tid);
    b.cast(Ty::U32, Ty::F32, tf, tid);
    let zero = b.k_u32(0);
    b.mov(k, zero);
    b.mov(acc, tf);
    let n = b.k_u32(iters);
    let one = b.k_u32(1);
    let a = b.k_f32(0.999);
    let c = b.k_f32(0.25);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, k, n);
    b.loop_if(p);
    b.fma(Ty::F32, acc, acc, a, c);
    b.add_f32(acc, acc, tf);
    b.add_u32(k, k, one);
    b.loop_end();
    b.st_f32(out, tid, acc);
    Scenario { name: "scalar_loop", module: b.build_module(), inputs: inputs(vec![("out", f32_buf([0.0; 32]))]) }
}

/// Memory-bound: each thread sums `x[(tid + j*stride) % n]` for `iters`
/// iterations and stores partial sums back into `x` every 8 iterations.
pub fn memory_heavy(iters: u32) -> Scenario {
    const N: u32 = 1024;
    let mut b = ProgramBuilder::new("memory_heavy", 128);
    b.site("mem.ld", 1);
    let x = b.global("x", Dtype::F32);
    let y = b.global("y", Dtype::F32);
    let tid = b.reg(Ty::U32);
    let j = b.reg(Ty::U32);
    let idx = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let q = b.reg(Ty::PRED);
    let t = b.reg(Ty::U32);
    let v = b.reg(Ty::F32);
    let acc = b.reg(Ty::F32);
    b.thread_rank(tid);
    let zero = b.k_u32(0);
    let fzero = b.k_f32(0.0);
    b.mov(j, zero);
    b.mov(acc, fzero);
    let n = b.k_u32(iters);
    let one = b.k_u32(1);
    let stride = b.k_u32(37);
    let mask = b.k_u32(N - 1);
    let seven = b.k_u32(7);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, j, n);
    b.loop_if(p);
    b.mul(Ty::U32, idx, j, stride);
    b.add_u32(idx, idx, tid);
    b.binary(BinOp::And, Ty::U32, idx, idx, mask);
    b.ld_f32(v, x, idx);
    b.add_f32(acc, acc, v);
    b.binary(BinOp::And, Ty::U32, t, j, seven);
    b.compare(CmpOp::Eq, Ty::U32, q, t, zero);
    b.if_(q);
    b.site("mem.st", 2);
    b.st_f32(y, tid, acc);
    b.site("mem.ld", 1);
    b.end_if();
    b.add_u32(j, j, one);
    b.loop_end();
    Scenario {
        name: "memory_heavy",
        module: b.build_module(),
        inputs: inputs(vec![("x", f32_buf((0..N).map(|v| (v % 13) as f32))), ("y", f32_buf([0.0; 128]))]),
    }
}

/// Nested divergent if/else, partial last warp, early exit of some lanes.
pub fn divergence() -> Scenario {
    let mut b = ProgramBuilder::new("divergence", 48);
    let out = b.global("out", Dtype::U32);
    let tid = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let q = b.reg(Ty::PRED);
    let v = b.reg(Ty::U32);
    b.thread_rank(tid);
    b.lane_id(lane);
    let k3 = b.k_u32(3);
    let k1 = b.k_u32(1);
    let k100 = b.k_u32(100);
    let k40 = b.k_u32(40);
    b.mov(v, tid);
    b.binary(BinOp::And, Ty::U32, v, lane, k3);
    b.compare(CmpOp::Eq, Ty::U32, p, v, k1);
    b.if_(p);
    b.add_u32(v, tid, k100);
    b.compare(CmpOp::Gt, Ty::U32, q, tid, k40);
    b.if_(q);
    b.st_u32(out, tid, v);
    b.exit();
    b.end_if();
    b.else_();
    b.mul(Ty::U32, v, tid, k3);
    b.end_if();
    b.st_u32(out, tid, v);
    Scenario { name: "divergence", module: b.build_module(), inputs: inputs(vec![("out", u32_buf([0; 48]))]) }
}

/// Loop with per-lane break and continue.
pub fn break_continue() -> Scenario {
    let mut b = ProgramBuilder::new("break_continue", 32);
    let out = b.global("out", Dtype::U32);
    let lane = b.reg(Ty::U32);
    let i = b.reg(Ty::U32);
    let s = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let t = b.reg(Ty::U32);
    b.lane_id(lane);
    let zero = b.k_u32(0);
    let one = b.k_u32(1);
    let k1 = b.k_u32(1);
    let k20 = b.k_u32(20);
    b.mov(i, zero);
    b.mov(s, zero);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, i, k20);
    b.loop_if(p);
    b.add_u32(i, i, one);
    // continue on odd i
    b.binary(BinOp::And, Ty::U32, t, i, k1);
    b.compare(CmpOp::Eq, Ty::U32, p, t, k1);
    b.if_(p);
    b.continue_();
    b.end_if();
    // break when i > lane
    b.compare(CmpOp::Gt, Ty::U32, p, i, lane);
    b.if_(p);
    b.break_();
    b.end_if();
    b.add_u32(s, s, i);
    b.loop_end();
    b.st_u32(out, lane, s);
    Scenario { name: "break_continue", module: b.build_module(), inputs: inputs(vec![("out", u32_buf([0; 32]))]) }
}

/// Shared-memory exchange across warps through `bar.sync`.
pub fn shared_barrier() -> Scenario {
    let mut b = ProgramBuilder::new("shared_barrier", 96);
    let out = b.global("out", Dtype::U32);
    let s = b.shared("s", Dtype::U32, 96);
    let tid = b.reg(Ty::U32);
    let j = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    b.thread_rank(tid);
    let k7 = b.k_u32(7);
    let k95 = b.k_u32(95);
    b.mul(Ty::U32, v, tid, k7);
    b.site("smem.st", 1);
    b.st_u32(s, tid, v);
    b.site("bar", 2);
    b.bar_sync(0);
    b.site("smem.ld", 3);
    b.binary(BinOp::Sub, Ty::U32, j, k95, tid);
    b.ld_u32(v, s, j);
    b.st_u32(out, tid, v);
    Scenario { name: "shared_barrier", module: b.build_module(), inputs: inputs(vec![("out", u32_buf([0; 96]))]) }
}

/// mbarrier init (one thread) / arrive (all) / wait parity 0.
pub fn mbarrier() -> Scenario {
    let mut b = ProgramBuilder::new("mbarrier", 64);
    let out = b.global("out", Dtype::U32);
    let bar = b.shared("bar", Dtype::U64, 1);
    let data = b.shared("data", Dtype::U32, 64);
    let tid = b.reg(Ty::U32);
    let m = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let v = b.reg(Ty::U32);
    b.thread_rank(tid);
    let zero = b.k_u32(0);
    b.smem_addr(m, bar, zero);
    b.compare(CmpOp::Eq, Ty::U32, p, tid, zero);
    b.site("mbar.init", 1);
    b.if_(p);
    b.mbar_init(m, 64);
    b.end_if();
    b.bar_sync(0);
    b.site("data.st", 2);
    b.st_u32(data, tid, tid);
    b.site("mbar.arrive", 3);
    b.mbar_arrive(m, None);
    b.site("mbar.wait", 4);
    b.mbar_wait_parity(m, zero);
    let k63 = b.k_u32(63);
    let j = b.reg(Ty::U32);
    b.binary(BinOp::Sub, Ty::U32, j, k63, tid);
    b.site("data.ld", 5);
    b.ld_u32(v, data, j);
    b.st_u32(out, tid, v);
    Scenario { name: "mbarrier", module: b.build_module(), inputs: inputs(vec![("out", u32_buf([0; 64]))]) }
}

/// `red.add` of every thread into one counter, 2 CTAs.
pub fn atomics() -> Scenario {
    let mut b = ProgramBuilder::new("atomics", 64);
    b.grid(2, 1, 1);
    let ctr = b.global("ctr", Dtype::U32);
    let tid = b.reg(Ty::U32);
    let a = b.reg(Ty::U64);
    b.thread_rank(tid);
    let zero = b.k_u32(0);
    b.addr_of(a, ctr, zero);
    b.site("red", 1);
    b.red_add(Ty::U32, a, tid);
    Scenario { name: "atomics", module: b.build_module(), inputs: inputs(vec![("ctr", u32_buf([0]))]) }
}

/// A generic `Ptx` op (bit cast) and a `Cast`.
pub fn ptx_ops() -> Scenario {
    let mut b = ProgramBuilder::new("ptx_ops", 32);
    let out = b.global("out", Dtype::U32);
    let tid = b.reg(Ty::U32);
    let f = b.reg(Ty::F32);
    let u = b.reg(Ty::U32);
    b.thread_rank(tid);
    b.cast(Ty::U32, Ty::F32, f, tid);
    let half = b.k_f32(0.5);
    b.mul_f32(f, f, half);
    b.ptx("tirx.cuda.float_as_uint", &[], &[u], &[f.into()]);
    b.st_u32(out, tid, u);
    Scenario { name: "ptx_ops", module: b.build_module(), inputs: inputs(vec![("out", u32_buf([0; 32]))]) }
}

/// A failing assert in some lanes: identical error status expected.
pub fn assert_fail() -> Scenario {
    let mut b = ProgramBuilder::new("assert_fail", 32);
    let out = b.global("out", Dtype::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    b.lane_id(lane);
    b.st_u32(out, lane, lane);
    let k20 = b.k_u32(20);
    b.compare(CmpOp::Lt, Ty::U32, p, lane, k20);
    b.site("assert", 1);
    let msg = b.string("lane < 20");
    b.push(Instr::Assert { cond: p.into(), msg: Some(msg) });
    Scenario { name: "assert_fail", module: b.build_module(), inputs: inputs(vec![("out", u32_buf([0; 32]))]) }
}

/// Out-of-bounds load: identical error status expected.
pub fn oob() -> Scenario {
    let mut b = ProgramBuilder::new("oob", 32);
    let x = b.global("x", Dtype::F32);
    let lane = b.reg(Ty::U32);
    let v = b.reg(Ty::F32);
    b.lane_id(lane);
    b.site("oob.ld", 1);
    b.ld_f32(v, x, lane);
    b.st_f32(x, lane, v);
    Scenario { name: "oob", module: b.build_module(), inputs: inputs(vec![("x", f32_buf([1.0; 16]))]) }
}

/// Two kernels sharing a buffer by name.
pub fn two_kernels() -> Scenario {
    let mut k0 = ProgramBuilder::new("k0", 32);
    let buf = k0.global("buf", Dtype::U32);
    let t = k0.reg(Ty::U32);
    k0.thread_rank(t);
    let k5 = k0.k_u32(5);
    let v = k0.reg(Ty::U32);
    k0.add_u32(v, t, k5);
    k0.st_u32(buf, t, v);
    let mut k1 = ProgramBuilder::new("k1", 32);
    let buf1 = k1.global("buf", Dtype::U32);
    let t1 = k1.reg(Ty::U32);
    let w = k1.reg(Ty::U32);
    k1.thread_rank(t1);
    k1.ld_u32(w, buf1, t1);
    k1.mul(Ty::U32, w, w, w);
    k1.st_u32(buf1, t1, w);
    Scenario {
        name: "two_kernels",
        module: Module::new(vec![k0.build(), k1.build()]),
        inputs: inputs(vec![("buf", u32_buf([0; 32]))]),
    }
}

/// Every scenario of the differential suite.
pub fn all() -> Vec<Scenario> {
    vec![
        vadd(),
        scalar_loop(50),
        memory_heavy(40),
        divergence(),
        break_continue(),
        shared_barrier(),
        mbarrier(),
        atomics(),
        ptx_ops(),
        assert_fail(),
        oob(),
        two_kernels(),
    ]
}
