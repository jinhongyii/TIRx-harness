//! The printer: `Module` -> Rust source of the generated cdylib.
//!
//! Every instruction becomes one arm of a resumable
//! `loop { match pc { k => step!(k, handlers::f(ctx, <consts>)) } }`.
//! `pc` is a local whose value after an arm is a constant (`k + 1`) on the
//! `Flow::Next` path, so LLVM jump-threads the loop back-edge straight into
//! the next arm: straight-line code between yield points, a jump table only
//! where the scheduler resumes or a handler jumps. Control-flow handlers
//! (`if_`, `loop_end`, ...) still decide every transfer at run time, so
//! mask-stack and loop-budget semantics are the interpreter's.
//!
//! Per-instruction bookkeeping and `Flow` application are the
//! interpreter's own `interp::{begin_instr, end_instr, fall_off_end}`
//! (`progress` printed as a constant); the printer's argument lists mirror
//! `interp::handlers::dispatch` one line per variant.

use super::lit::lit;
use super::{CHECK_SYMBOL_PREFIX, STEP_SYMBOL_PREFIX};
use crate::program::{Instr, Module, Program};
use std::fmt::Write;

/// Version of the printed shape; part of the build cache key.
pub const EMIT_VERSION: u32 = 1;

/// Handler call for `ins` at `pc` (`ctx` and `p: &Program` in scope).
pub(crate) fn handler_call(pc: usize, ins: &Instr) -> String {
    use Instr::*;
    // Slice fields print as promoted `&[..]` constants.
    fn sl<T: serde::Serialize>(v: &[T]) -> String {
        format!("&{}", lit(v))
    }
    // Boxed payloads (Vec fields) are borrowed from the program itself.
    let borrowed = |variant: &str, f: &str| {
        format!("match &p.code[{pc}] {{ Instr::{variant}(a) => h::{f}(ctx, a), _ => unreachable!() }}")
    };
    let c = |f: &str, args: &[String]| {
        let mut s = format!("h::{f}(ctx");
        for a in args {
            s.push_str(", ");
            s.push_str(a);
        }
        s.push(')');
        s
    };
    let l = |v: &dyn erased::Lit| v.lit();
    match ins {
        Nop => c("nop", &[]),
        If { cond, else_pc, end_pc, elect } => c("if_", &[l(cond), l(else_pc), l(end_pc), l(elect)]),
        Else { end_pc } => c("else_", &[l(end_pc)]),
        EndIf => c("end_if", &[]),
        LoopBegin { end_pc } => c("loop_begin", &[l(end_pc)]),
        LoopIf { cond, end_pc } => c("loop_if", &[l(cond), l(end_pc)]),
        LoopEnd { head_pc } => c("loop_end", &[l(head_pc)]),
        Break => c("break_", &[]),
        Continue => c("continue_", &[]),
        Exit => c("exit", &[]),
        Assert { cond, msg } => c("assert", &[l(cond), l(msg)]),
        Unsupported { reason } => c("unsupported", &[l(reason)]),
        Mov { dst, src } => c("mov", &[l(dst), l(src)]),
        ReadSpecial { dst, sreg } => c("read_special", &[l(dst), l(sreg)]),
        ReadParam { dst, slot } => c("read_param", &[l(dst), l(slot)]),
        Unary { op, ty, dst, a } => c("unary", &[l(op), l(ty), l(dst), l(a)]),
        Binary { op, ty, dst, a, b } => c("binary", &[l(op), l(ty), l(dst), l(a), l(b)]),
        Ternary { op, ty, dst, a, b, c: cc } => c("ternary", &[l(op), l(ty), l(dst), l(a), l(b), l(cc)]),
        Compare { op, ty, dst, a, b } => c("compare", &[l(op), l(ty), l(dst), l(a), l(b)]),
        Select { ty, dst, cond, a, b } => c("select", &[l(ty), l(dst), l(cond), l(a), l(b)]),
        Cast { from, to, dst, src, rnd, sat } => c("cast", &[l(from), l(to), l(dst), l(src), l(rnd), l(sat)]),
        Ptx { op, dsts, srcs, pred, keep_dst } => c("ptx", &[l(op), sl(dsts), sl(srcs), l(pred), l(keep_dst)]),
        LoadRegIndexed { dst, base, len, idx } => c("load_reg_indexed", &[l(dst), l(base), l(len), l(idx)]),
        StoreRegIndexed { base, len, idx, value } => c("store_reg_indexed", &[l(base), l(len), l(idx), l(value)]),
        Shfl { mode, ty, dst, dst_pred, src, lane, clamp, membermask } => {
            c("shfl", &[l(mode), l(ty), l(dst), l(dst_pred), l(src), l(lane), l(clamp), l(membermask)])
        }
        Vote { mode, dst, pred, membermask } => c("vote", &[l(mode), l(dst), l(pred), l(membermask)]),
        Redux { op, ty, dst, src, membermask } => c("redux", &[l(op), l(ty), l(dst), l(src), l(membermask)]),
        Elect { dst_pred, dst_lane, membermask } => c("elect", &[l(dst_pred), l(dst_lane), l(membermask)]),
        WarpSync { membermask } => c("warp_sync", &[l(membermask)]),
        LdMatrix { dsts, addr, space, shape, num, trans, fmt } => {
            c("ldmatrix", &[sl(dsts), l(addr), l(space), l(shape), l(num), l(trans), l(fmt)])
        }
        StMatrix { srcs, addr, space, shape, num, trans } => {
            c("stmatrix", &[sl(srcs), l(addr), l(space), l(shape), l(num), l(trans)])
        }
        Load { ty, dst, buf, offset, sem, scope, mods } => {
            c("load", &[l(ty), l(dst), l(buf), l(offset), l(sem), l(scope), l(mods)])
        }
        Store { ty, buf, offset, value, sem, scope, mods } => {
            c("store", &[l(ty), l(buf), l(offset), l(value), l(sem), l(scope), l(mods)])
        }
        LoadAddr { ty, dst, addr, space, sem, scope, mods } => {
            c("load_addr", &[l(ty), l(dst), l(addr), l(space), l(sem), l(scope), l(mods)])
        }
        StoreAddr { ty, addr, space, value, sem, scope, mods } => {
            c("store_addr", &[l(ty), l(addr), l(space), l(value), l(sem), l(scope), l(mods)])
        }
        AddrOf { dst, buf, offset } => c("addr_of", &[l(dst), l(buf), l(offset)]),
        Atom { op, ty, dst, addr, space, value, cmp, sem, scope, ftz } => c(
            "atom",
            &[l(op), l(ty), l(dst), l(addr), l(space), l(value), l(cmp), l(sem), l(scope), l(ftz)],
        ),
        StBulk { addr, space, size } => c("st_bulk", &[l(addr), l(space), l(size)]),
        Discard { addr, space, size } => c("discard", &[l(addr), l(space), l(size)]),
        Cvta { dst, src, space, to_generic } => c("cvta", &[l(dst), l(src), l(space), l(to_generic)]),
        Isspacep { dst, src, space } => c("isspacep", &[l(dst), l(src), l(space)]),
        Mapa { dst, src, rank, space } => c("mapa", &[l(dst), l(src), l(rank), l(space)]),
        GetCtaRank { dst, src, space } => c("getctarank", &[l(dst), l(src), l(space)]),
        CpAsync { dst, src, cp_size, src_size, ignore_src, mods } => {
            c("cp_async", &[l(dst), l(src), l(cp_size), l(src_size), l(ignore_src), l(mods)])
        }
        AsyncCommit { domain } => c("async_commit", &[l(domain)]),
        AsyncWait { domain, n, read } => c("async_wait", &[l(domain), l(n), l(read)]),
        CpAsyncMbarArrive { mbar, space, noinc } => c("cp_async_mbar_arrive", &[l(mbar), l(space), l(noinc)]),
        BulkCopy(args) => c("bulk_copy", &[l(args)]),
        Tma(_) => borrowed("Tma", "tma"),
        StAsync(args) => c("st_async", &[l(args)]),
        TensorMapReplace { tmap, space, field, ord, value } => {
            c("tensormap_replace", &[l(tmap), l(space), l(field), l(ord), l(value)])
        }
        TensorMapCopyFence { dst, src, size, scope } => c("tensormap_cp_fence", &[l(dst), l(src), l(size), l(scope)]),
        Barrier { kind, id, count, aligned } => c("barrier", &[l(kind), l(id), l(count), l(aligned)]),
        ClusterArrive { sem, aligned } => c("cluster_arrive", &[l(sem), l(aligned)]),
        ClusterWait { acquire, aligned } => c("cluster_wait", &[l(acquire), l(aligned)]),
        GridSync => c("grid_sync", &[]),
        MbarInit { mbar, space, count, layout_v1 } => c("mbar_init", &[l(mbar), l(space), l(count), l(layout_v1)]),
        MbarInval { mbar, space } => c("mbar_inval", &[l(mbar), l(space)]),
        MbarArrive(args) => c("mbar_arrive", &[l(args)]),
        MbarTx { op, mbar, space, bytes, multicast, scope } => {
            c("mbar_tx", &[l(op), l(mbar), l(space), l(bytes), l(multicast), l(scope)])
        }
        MbarTestWait { kind, mbar, space, phase, sem, scope, dst } => {
            c("mbar_test_wait", &[l(kind), l(mbar), l(space), l(phase), l(sem), l(scope), l(dst)])
        }
        MbarWait { mbar, space, phase, sem, scope } => c("mbar_wait", &[l(mbar), l(space), l(phase), l(sem), l(scope)]),
        MbarQuery { dst, op } => c("mbar_query", &[l(dst), l(op)]),
        Fence { kind, sem, scope } => c("fence", &[l(kind), l(sem), l(scope)]),
        SetMaxNReg { inc, count } => c("setmaxnreg", &[l(inc), l(count)]),
        WaitUntil { dst, addr, ty, space, sem, scope, pred, captures } => c(
            "wait_until",
            &[l(dst), l(addr), l(ty), l(space), l(sem), l(scope), l(pred), sl(captures)],
        ),
        GridDepControl { launch_dependents } => c("griddepcontrol", &[l(launch_dependents)]),
        ClcTryCancel { resp, mbar, multicast } => c("clc_try_cancel", &[l(resp), l(mbar), l(multicast)]),
        TcgenAlloc { dst, ncols, cta_group, exclusive } => {
            c("tcgen_alloc", &[l(dst), l(ncols), l(cta_group), l(exclusive)])
        }
        TcgenDealloc { taddr, ncols, cta_group, exclusive } => {
            c("tcgen_dealloc", &[l(taddr), l(ncols), l(cta_group), l(exclusive)])
        }
        TcgenRelinquish { cta_group } => c("tcgen_relinquish", &[l(cta_group)]),
        TcgenCommit { mbar, space, cta_group, multicast } => {
            c("tcgen_commit", &[l(mbar), l(space), l(cta_group), l(multicast)])
        }
        TcgenLd(_) => borrowed("TcgenLd", "tcgen_ld"),
        TcgenSt(_) => borrowed("TcgenSt", "tcgen_st"),
        TcgenWait { st } => c("tcgen_wait", &[l(st)]),
        TcgenCp(args) => c("tcgen_cp", &[l(args)]),
        TcgenMma(_) => borrowed("TcgenMma", "tcgen_mma"),
        Tile(_) => borrowed("Tile", "tile"),
    }
}

/// Object-safe shim so the match above can pass heterogeneous fields.
mod erased {
    pub trait Lit {
        fn lit(&self) -> String;
    }
    impl<T: serde::Serialize> Lit for T {
        fn lit(&self) -> String {
            super::lit(self)
        }
    }
}

/// One-line comment text for an instruction (no `*/` or newlines).
fn comment(ins: &Instr) -> String {
    let mut s = ins.to_string().replace(['\n', '\r'], " ");
    s = s.replace("*/", "* /");
    if s.len() > 100 {
        let mut cut = 100;
        while !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push_str("...");
    }
    s
}

const PRELUDE: &str = r#"// @generated by numsim_core::codegen. Do not edit.
#![allow(unused_imports, unused_variables, unused_mut, unreachable_code, unreachable_patterns, non_upper_case_globals, clippy::all)]

use numsim_core::codegen::rt;
use numsim_core::dtype::*;
use numsim_core::interp::handlers as h;
use numsim_core::interp::{begin_instr, end_instr, fall_off_end, ExecCtx, StepResult};
use numsim_core::program::*;
use numsim_core::sync::async_group::Domain;

/// One instruction: the interpreter's prologue, the handler call, the
/// interpreter's epilogue (`progress` = `Instr::is_progress`, a constant).
macro_rules! step {
    ($ctx:ident, $pc:ident, $k:literal, $progress:literal, $call:expr) => {{
        begin_instr($ctx);
        let r = $call;
        if let Some(done) = end_instr($ctx, Pc($k), $progress, r) {
            return done;
        }
        $pc = $ctx.warp.pc.0;
    }};
}

#[no_mangle]
pub extern "Rust" fn numsim_abi_fingerprint() -> [u64; numsim_core::codegen::ABI_WORDS] {
    numsim_core::codegen::abi_fingerprint()
}
"#;

/// Print the generated crate's `lib.rs` for `module`.
///
/// `module_sha256` is embedded so a loaded library can be matched against
/// the module it is run with.
pub fn emit_module(module: &Module, module_sha256: &[u8; 32]) -> String {
    let mut out = String::with_capacity(4096);
    out.push_str(PRELUDE);
    writeln!(
        out,
        "\n#[no_mangle]\npub extern \"Rust\" fn numsim_module_sha256() -> [u8; 32] {{\n    {:?}\n}}",
        module_sha256
    )
    .unwrap();
    for (k, p) in module.kernels.iter().enumerate() {
        emit_kernel(&mut out, k, p);
    }
    out
}

fn emit_kernel(out: &mut String, k: usize, p: &Program) {
    let name_comment = p.name.replace(['\n', '\r'], " ");
    writeln!(out, "\n// ===== kernel {k}: {name_comment} ({} instrs) =====", p.code.len()).unwrap();

    // Constant pool and op table (checked against the runtime program at load).
    writeln!(out, "pub static K{k}_CONSTS: [Const; {}] = [", p.consts.len()).unwrap();
    for c in &p.consts {
        writeln!(out, "    {},", lit(c)).unwrap();
    }
    out.push_str("];\n");
    writeln!(out, "pub static K{k}_OPS: [(&str, &[&str]); {}] = [", p.ops.len()).unwrap();
    for o in &p.ops {
        writeln!(out, "    ({:?}, &{:?}),", o.name, o.mods).unwrap();
    }
    out.push_str("];\n");
    writeln!(
        out,
        "\n#[no_mangle]\npub extern \"Rust\" fn {CHECK_SYMBOL_PREFIX}{k}(p: &Program) -> bool {{\n    \
         p.code.len() == {n} && p.consts[..] == K{k}_CONSTS[..] && p.ops.len() == K{k}_OPS.len()\n        \
         && p.ops.iter().zip(K{k}_OPS.iter()).all(|(o, (n, m))| o.name == *n && o.mods.len() == m.len() && o.mods.iter().zip(m.iter()).all(|(a, b)| a == b))\n}}",
        n = p.code.len()
    )
    .unwrap();

    // The step function.
    writeln!(
        out,
        "\n#[no_mangle]\npub extern \"Rust\" fn {STEP_SYMBOL_PREFIX}{k}(ctx: &mut ExecCtx<'_>, quantum: u32) -> StepResult {{\n    \
         rt::guard(ctx, quantum, body_{k})\n}}\n\n\
         #[inline(always)]\nfn body_{k}(ctx: &mut ExecCtx<'_>, quantum: u32) -> StepResult {{\n    \
         let p: &Program = ctx.program;\n    \
         let mut pc: u32 = ctx.warp.pc.0;\n    \
         let mut budget: u32 = quantum;\n    \
         loop {{\n        \
         if budget == 0 {{\n            return StepResult::Continue;\n        }}\n        \
         budget -= 1;\n        \
         match pc {{"
    )
    .unwrap();
    for (pc, ins) in p.code.iter().enumerate() {
        writeln!(out, "            // {}", comment(ins)).unwrap();
        writeln!(out, "            {pc} => step!(ctx, pc, {pc}, {}, {}),", ins.is_progress(), handler_call(pc, ins)).unwrap();
    }
    out.push_str(
        "            _ => return fall_off_end(ctx),\n        }\n    }\n}\n",
    );
}
