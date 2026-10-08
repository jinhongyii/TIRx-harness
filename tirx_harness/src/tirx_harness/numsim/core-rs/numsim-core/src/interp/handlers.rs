//! The handler interface: one `#[inline] pub fn` per `Instr` family.
//!
//! **Contract (W2 implements bodies, W7 prints calls):**
//! * Each handler takes `&mut ExecCtx` and the instruction's fields as
//!   plain arguments (Copy fields by value, `&[T]` for lists, `&Args` for
//!   boxed payloads), in field-declaration order. The codegen printer emits
//!   exactly `handlers::<name>(ctx, <field constants>...)` and applies the
//!   returned [`Flow`] exactly like [`super::step_warp`].
//! * The dispatcher sets `ctx.warp.pc` and bumps `ctx.warp.epoch` before
//!   the call; the site is `ctx.site()`. Handlers express control transfer
//!   only through the returned `Flow`.
//! * Handlers operate on `ctx.warp.active` lanes only and never assume which
//!   backend called them.
//!
//! [`dispatch`] is the interpreter's match and the reference for the
//! codegen printer's argument order.


/// Non-control handlers do nothing for an empty active mask (code skipped
/// by divergence, or after `Break`/`Continue`/`Exit` until the frame that
/// restores lanes).
macro_rules! active_or_next {
    ($ctx:expr) => {
        if $ctx.warp.active.is_empty() {
            return Ok($crate::interp::Flow::Next);
        }
    };
}

mod alu;
mod async_copy;
mod control;
pub(crate) mod mem;
mod sync;
mod tcgen;
mod warp;

use super::{ExecCtx, ExecError, Flow};
use crate::dtype::Ty;
use crate::program::*;
use crate::sync::async_group::Domain;

/// Result of a handler.
pub type HResult = Result<Flow, ExecError>;

#[inline(always)]
pub fn nop(ctx: &mut ExecCtx<'_>) -> HResult {
    control::nop(ctx)
}

/// Push a `Then` frame; active &= cond; jump to `else_pc` if none active.
#[inline(always)]
pub fn if_(ctx: &mut ExecCtx<'_>, cond: Operand, else_pc: Pc, end_pc: Pc, elect: bool) -> HResult {
    control::if_(ctx, cond, else_pc, end_pc, elect)
}

/// Top frame becomes `Else`: active = entry & !taken (& live & !broken); jump to `end_pc` if empty.
#[inline(always)]
pub fn else_(ctx: &mut ExecCtx<'_>, end_pc: Pc) -> HResult {
    control::else_(ctx, end_pc)
}

/// Pop: active = entry & live & !broken-in-enclosing-loop.
#[inline(always)]
pub fn end_if(ctx: &mut ExecCtx<'_>) -> HResult {
    control::end_if(ctx)
}

/// Push a `Loop` frame (iteration 0, empty break/continue masks).
#[inline(always)]
pub fn loop_begin(ctx: &mut ExecCtx<'_>, end_pc: Pc) -> HResult {
    control::loop_begin(ctx, end_pc)
}

/// active &= cond; if empty: pop the loop frame, active = entry & live, jump to `end_pc + 1`.
#[inline(always)]
pub fn loop_if(ctx: &mut ExecCtx<'_>, cond: Operand, end_pc: Pc) -> HResult {
    control::loop_if(ctx, cond, end_pc)
}

/// active |= continued; iteration += 1 (budget -> `ExecErrorKind::Budget`); spin parking -> `Flow::Blocked`; quantum -> `Flow::Yield(head_pc)`; else `Flow::Jump(head_pc)`.
#[inline(always)]
pub fn loop_end(ctx: &mut ExecCtx<'_>, head_pc: Pc) -> HResult {
    control::loop_end(ctx, head_pc)
}

/// Innermost loop: broken |= active; active = 0.
#[inline(always)]
pub fn break_(ctx: &mut ExecCtx<'_>) -> HResult {
    control::break_(ctx)
}

/// Innermost loop: continued |= active; active = 0.
#[inline(always)]
pub fn continue_(ctx: &mut ExecCtx<'_>) -> HResult {
    control::continue_(ctx)
}

/// Active lanes retire (removed from `live`); `Flow::Exit` when no live lanes remain.
#[inline(always)]
pub fn exit(ctx: &mut ExecCtx<'_>) -> HResult {
    control::exit(ctx)
}

/// `ExecErrorKind::Trap` for active lanes where cond is false.
#[inline(always)]
pub fn assert(ctx: &mut ExecCtx<'_>, cond: Operand, msg: Option<StrId>) -> HResult {
    control::assert(ctx, cond, msg)
}

/// `ExecErrorKind::Unsupported` if any lane is active.
#[inline(always)]
pub fn unsupported(ctx: &mut ExecCtx<'_>, reason: StrId) -> HResult {
    control::unsupported(ctx, reason)
}

#[inline(always)]
pub fn mov(ctx: &mut ExecCtx<'_>, dst: Reg, src: Operand) -> HResult {
    alu::mov(ctx, dst, src)
}

#[inline(always)]
pub fn read_special(ctx: &mut ExecCtx<'_>, dst: Reg, sreg: SpecialReg) -> HResult {
    alu::read_special(ctx, dst, sreg)
}

/// Scalar parameter bytes from the launch's param block.
#[inline(always)]
pub fn read_param(ctx: &mut ExecCtx<'_>, dst: Reg, slot: ParamId) -> HResult {
    alu::read_param(ctx, dst, slot)
}

#[inline(always)]
pub fn unary(ctx: &mut ExecCtx<'_>, op: UnOp, ty: Ty, dst: Reg, a: Operand) -> HResult {
    alu::unary(ctx, op, ty, dst, a)
}

#[inline(always)]
pub fn binary(ctx: &mut ExecCtx<'_>, op: BinOp, ty: Ty, dst: Reg, a: Operand, b: Operand) -> HResult {
    alu::binary(ctx, op, ty, dst, a, b)
}

#[inline(always)]
pub fn ternary(ctx: &mut ExecCtx<'_>, op: TerOp, ty: Ty, dst: Reg, a: Operand, b: Operand, c: Operand) -> HResult {
    alu::ternary(ctx, op, ty, dst, a, b, c)
}

#[inline(always)]
pub fn compare(ctx: &mut ExecCtx<'_>, op: CmpOp, ty: Ty, dst: Reg, a: Operand, b: Operand) -> HResult {
    alu::compare(ctx, op, ty, dst, a, b)
}

#[inline(always)]
pub fn select(ctx: &mut ExecCtx<'_>, ty: Ty, dst: Reg, cond: Operand, a: Operand, b: Operand) -> HResult {
    alu::select(ctx, ty, dst, cond, a, b)
}

#[inline(always)]
pub fn cast(ctx: &mut ExecCtx<'_>, from: Ty, to: Ty, dst: Reg, src: Operand, rnd: Rounding, sat: bool) -> HResult {
    alu::cast(ctx, from, to, dst, src, rnd, sat)
}

/// Run `ctx.loaded.ops[op]` (resolved oplib function) over the operand slots; `pred`/`keep_dst` per `Instr::Ptx`.
#[inline(always)]
pub fn ptx(ctx: &mut ExecCtx<'_>, op: OpId, dsts: &[Reg], srcs: &[Operand], pred: Option<Operand>, keep_dst: bool) -> HResult {
    alu::ptx(ctx, op, dsts, srcs, pred, keep_dst)
}

#[inline(always)]
pub fn load_reg_indexed(ctx: &mut ExecCtx<'_>, dst: Reg, base: Reg, len: u32, idx: Operand) -> HResult {
    alu::load_reg_indexed(ctx, dst, base, len, idx)
}

#[inline(always)]
pub fn store_reg_indexed(ctx: &mut ExecCtx<'_>, base: Reg, len: u32, idx: Operand, value: Operand) -> HResult {
    alu::store_reg_indexed(ctx, base, len, idx, value)
}

#[inline(always)]
pub fn shfl(ctx: &mut ExecCtx<'_>, mode: ShflMode, ty: Ty, dst: Reg, dst_pred: Option<Reg>, src: Operand, lane: Operand, clamp: Operand, membermask: Operand) -> HResult {
    warp::shfl(ctx, mode, ty, dst, dst_pred, src, lane, clamp, membermask)
}

#[inline(always)]
pub fn vote(ctx: &mut ExecCtx<'_>, mode: VoteMode, dst: Reg, pred: Operand, membermask: Operand) -> HResult {
    warp::vote(ctx, mode, dst, pred, membermask)
}

#[inline(always)]
pub fn redux(ctx: &mut ExecCtx<'_>, op: ReduxOp, ty: Ty, dst: Reg, src: Operand, membermask: Operand) -> HResult {
    warp::redux(ctx, op, ty, dst, src, membermask)
}

#[inline(always)]
pub fn elect(ctx: &mut ExecCtx<'_>, dst_pred: Reg, dst_lane: Option<Reg>, membermask: Operand) -> HResult {
    warp::elect(ctx, dst_pred, dst_lane, membermask)
}

#[inline(always)]
pub fn warp_sync(ctx: &mut ExecCtx<'_>, membermask: Operand) -> HResult {
    warp::warp_sync(ctx, membermask)
}

#[inline(always)]
pub fn ldmatrix(ctx: &mut ExecCtx<'_>, dsts: &[Reg], addr: Operand, space: AddrSpace, shape: MatrixShape, num: u8, trans: bool, fmt: MatrixFmt) -> HResult {
    warp::ldmatrix(ctx, dsts, addr, space, shape, num, trans, fmt)
}

#[inline(always)]
pub fn stmatrix(ctx: &mut ExecCtx<'_>, srcs: &[Operand], addr: Operand, space: AddrSpace, shape: MatrixShape, num: u8, trans: bool) -> HResult {
    warp::stmatrix(ctx, srcs, addr, space, shape, num, trans)
}

/// Resolve `buf` (+ offset * elem bytes), check OOB/alignment/validity, read, write `dst`; emit one `Access`.
#[inline(always)]
pub fn load(ctx: &mut ExecCtx<'_>, ty: Ty, dst: Reg, buf: Buf, offset: Operand, sem: Sem, scope: Scope, mods: MemMods) -> HResult {
    mem::load(ctx, ty, dst, buf, offset, sem, scope, mods)
}

/// Resolve, check, write (sets validity); emit `Access`.
#[inline(always)]
pub fn store(ctx: &mut ExecCtx<'_>, ty: Ty, buf: Buf, offset: Operand, value: Operand, sem: Sem, scope: Scope, mods: MemMods) -> HResult {
    mem::store(ctx, ty, buf, offset, value, sem, scope, mods)
}

/// As `load` but through an address value in `space`.
#[inline(always)]
pub fn load_addr(ctx: &mut ExecCtx<'_>, ty: Ty, dst: Reg, addr: Operand, space: AddrSpace, sem: Sem, scope: Scope, mods: MemMods) -> HResult {
    mem::load_addr(ctx, ty, dst, addr, space, sem, scope, mods)
}

/// As `store` but through an address value (remote shared::cluster stores apply at issue).
#[inline(always)]
pub fn store_addr(ctx: &mut ExecCtx<'_>, ty: Ty, addr: Operand, space: AddrSpace, value: Operand, sem: Sem, scope: Scope, mods: MemMods) -> HResult {
    mem::store_addr(ctx, ty, addr, space, value, sem, scope, mods)
}

#[inline(always)]
pub fn addr_of(ctx: &mut ExecCtx<'_>, dst: Reg, buf: Buf, offset: Operand) -> HResult {
    mem::addr_of(ctx, dst, buf, offset)
}

/// Per-lane RMW in lane order (one per vector element).
#[inline(always)]
pub fn atom(ctx: &mut ExecCtx<'_>, op: AtomOp, ty: Ty, dst: Option<Reg>, addr: Operand, space: AddrSpace, value: Operand, cmp: Option<Operand>, sem: Sem, scope: Scope, ftz: bool) -> HResult {
    mem::atom(ctx, op, ty, dst, addr, space, value, cmp, sem, scope, ftz)
}

#[inline(always)]
pub fn st_bulk(ctx: &mut ExecCtx<'_>, addr: Operand, space: AddrSpace, size: Operand) -> HResult {
    mem::st_bulk(ctx, addr, space, size)
}

#[inline(always)]
pub fn discard(ctx: &mut ExecCtx<'_>, addr: Operand, space: AddrSpace, size: u32) -> HResult {
    mem::discard(ctx, addr, space, size)
}

#[inline(always)]
pub fn cvta(ctx: &mut ExecCtx<'_>, dst: Reg, src: Operand, space: AddrSpace, to_generic: bool) -> HResult {
    mem::cvta(ctx, dst, src, space, to_generic)
}

#[inline(always)]
pub fn isspacep(ctx: &mut ExecCtx<'_>, dst: Reg, src: Operand, space: AddrSpace) -> HResult {
    mem::isspacep(ctx, dst, src, space)
}

#[inline(always)]
pub fn mapa(ctx: &mut ExecCtx<'_>, dst: Reg, src: Operand, rank: Operand, space: AddrSpace) -> HResult {
    mem::mapa(ctx, dst, src, rank, space)
}

#[inline(always)]
pub fn getctarank(ctx: &mut ExecCtx<'_>, dst: Reg, src: Operand, space: AddrSpace) -> HResult {
    mem::getctarank(ctx, dst, src, space)
}

/// Issue an AsyncOp into this lane's cp.async group (`async_group::Cmd::Issue`).
#[inline(always)]
pub fn cp_async(ctx: &mut ExecCtx<'_>, dst: Operand, src: Operand, cp_size: u8, src_size: Option<Operand>, ignore_src: Option<Operand>, mods: MemMods) -> HResult {
    async_copy::cp_async(ctx, dst, src, cp_size, src_size, ignore_src, mods)
}

#[inline(always)]
pub fn async_commit(ctx: &mut ExecCtx<'_>, domain: Domain) -> HResult {
    async_copy::async_commit(ctx, domain)
}

#[inline(always)]
pub fn async_wait(ctx: &mut ExecCtx<'_>, domain: Domain, n: u32, read: bool) -> HResult {
    async_copy::async_wait(ctx, domain, n, read)
}

#[inline(always)]
pub fn cp_async_mbar_arrive(ctx: &mut ExecCtx<'_>, mbar: Operand, space: AddrSpace, noinc: bool) -> HResult {
    async_copy::cp_async_mbar_arrive(ctx, mbar, space, noinc)
}

/// Resolve spans; AsyncOp + mbarrier `Issue` (complete_tx) or bulk-group `Issue`.
#[inline(always)]
pub fn bulk_copy(ctx: &mut ExecCtx<'_>, args: BulkCopyArgs) -> HResult {
    async_copy::bulk_copy(ctx, args)
}

/// Decode the tensor map, `oplib::tma_plan`, AsyncOp + completion binding.
#[inline(always)]
pub fn tma(ctx: &mut ExecCtx<'_>, args: &TmaArgs) -> HResult {
    async_copy::tma(ctx, args)
}

#[inline(always)]
pub fn st_async(ctx: &mut ExecCtx<'_>, args: StAsyncArgs) -> HResult {
    async_copy::st_async(ctx, args)
}

#[inline(always)]
pub fn tensormap_replace(ctx: &mut ExecCtx<'_>, tmap: Operand, space: AddrSpace, field: TmapField, ord: Option<u8>, value: Operand) -> HResult {
    async_copy::tensormap_replace(ctx, tmap, space, field, ord, value)
}

#[inline(always)]
pub fn tensormap_cp_fence(ctx: &mut ExecCtx<'_>, dst: Operand, src: Operand, size: u32, scope: Scope) -> HResult {
    async_copy::tensormap_cp_fence(ctx, dst, src, size, scope)
}

/// named `Sync`/`Arrive` with an explicit `Contribution`, then `Resume{gen}` on retries.
#[inline(always)]
pub fn barrier(ctx: &mut ExecCtx<'_>, kind: BarKind, id: Operand, count: Option<Operand>, aligned: bool) -> HResult {
    sync::barrier(ctx, kind, id, count, aligned)
}

#[inline(always)]
pub fn cluster_arrive(ctx: &mut ExecCtx<'_>, sem: Sem, aligned: bool) -> HResult {
    sync::cluster_arrive(ctx, sem, aligned)
}

#[inline(always)]
pub fn cluster_wait(ctx: &mut ExecCtx<'_>, acquire: bool, aligned: bool) -> HResult {
    sync::cluster_wait(ctx, acquire, aligned)
}

#[inline(always)]
pub fn grid_sync(ctx: &mut ExecCtx<'_>) -> HResult {
    sync::grid_sync(ctx)
}

#[inline(always)]
pub fn mbar_init(ctx: &mut ExecCtx<'_>, mbar: Operand, space: AddrSpace, count: Operand, layout_v1: bool) -> HResult {
    sync::mbar_init(ctx, mbar, space, count, layout_v1)
}

#[inline(always)]
pub fn mbar_inval(ctx: &mut ExecCtx<'_>, mbar: Operand, space: AddrSpace) -> HResult {
    sync::mbar_inval(ctx, mbar, space)
}

/// mbarrier `Arrive` (local, remote or multicast: applied at issue, all-or-nothing).
#[inline(always)]
pub fn mbar_arrive(ctx: &mut ExecCtx<'_>, args: MbarArriveArgs) -> HResult {
    sync::mbar_arrive(ctx, args)
}

#[inline(always)]
pub fn mbar_tx(ctx: &mut ExecCtx<'_>, op: TxOp, mbar: Operand, space: AddrSpace, bytes: Operand, multicast: Option<Operand>, scope: Scope) -> HResult {
    sync::mbar_tx(ctx, op, mbar, space, bytes, multicast, scope)
}

/// Non-blocking `TestParity`/`TestState`; NotReady -> `note_failed_poll`.
#[inline(always)]
pub fn mbar_test_wait(ctx: &mut ExecCtx<'_>, kind: WaitKind, mbar: Operand, space: AddrSpace, phase: PhaseArg, sem: Sem, scope: Scope, dst: Option<Reg>, report: Option<Reg>, report_value: Option<Reg>) -> HResult {
    sync::mbar_test_wait(ctx, kind, mbar, space, phase, sem, scope, dst, report, report_value)
}

/// Blocking `WaitParity`.
#[inline(always)]
pub fn mbar_wait(ctx: &mut ExecCtx<'_>, mbar: Operand, space: AddrSpace, phase: PhaseArg, sem: Sem, scope: Scope) -> HResult {
    sync::mbar_wait(ctx, mbar, space, phase, sem, scope)
}

#[inline(always)]
pub fn mbar_query(ctx: &mut ExecCtx<'_>, dst: Reg, op: MbarQueryOp) -> HResult {
    sync::mbar_query(ctx, dst, op)
}

#[inline(always)]
pub fn fence(ctx: &mut ExecCtx<'_>, kind: FenceKind, sem: Sem, scope: Scope) -> HResult {
    sync::fence(ctx, kind, sem, scope)
}

#[inline(always)]
pub fn setmaxnreg(ctx: &mut ExecCtx<'_>, inc: bool, count: u32) -> HResult {
    sync::setmaxnreg(ctx, inc, count)
}

/// Load the word; run the predicate sub-program; all active lanes accepted -> write `dst`, emit `WaitVerdicts`; else `Flow::Blocked(ResourceId::Word{..})`.
#[inline(always)]
pub fn wait_until(ctx: &mut ExecCtx<'_>, dst: Reg, addr: Operand, ty: Ty, space: AddrSpace, sem: Sem, scope: Scope, pred: PredId, captures: &[Reg]) -> HResult {
    sync::wait_until(ctx, dst, addr, ty, space, sem, scope, pred, captures)
}

#[inline(always)]
pub fn griddepcontrol(ctx: &mut ExecCtx<'_>, launch_dependents: bool) -> HResult {
    sync::griddepcontrol(ctx, launch_dependents)
}

#[inline(always)]
pub fn clc_try_cancel(ctx: &mut ExecCtx<'_>, resp: Operand, mbar: Operand, multicast: bool) -> HResult {
    sync::clc_try_cancel(ctx, resp, mbar, multicast)
}

#[inline(always)]
pub fn tcgen_alloc(ctx: &mut ExecCtx<'_>, dst: Operand, ncols: Operand, cta_group: u8, exclusive: bool) -> HResult {
    tcgen::tcgen_alloc(ctx, dst, ncols, cta_group, exclusive)
}

#[inline(always)]
pub fn tcgen_dealloc(ctx: &mut ExecCtx<'_>, taddr: Operand, ncols: Operand, cta_group: u8, exclusive: bool) -> HResult {
    tcgen::tcgen_dealloc(ctx, taddr, ncols, cta_group, exclusive)
}

#[inline(always)]
pub fn tcgen_relinquish(ctx: &mut ExecCtx<'_>, cta_group: u8) -> HResult {
    tcgen::tcgen_relinquish(ctx, cta_group)
}

#[inline(always)]
pub fn tcgen_commit(ctx: &mut ExecCtx<'_>, mbar: Operand, space: AddrSpace, cta_group: u8, multicast: Option<Operand>, sync_restrict: bool, multicast_width: Option<u8>) -> HResult {
    tcgen::tcgen_commit(ctx, mbar, space, cta_group, multicast, sync_restrict, multicast_width)
}

#[inline(always)]
pub fn tcgen_ld(ctx: &mut ExecCtx<'_>, args: &TcgenLdArgs) -> HResult {
    tcgen::tcgen_ld(ctx, args)
}

#[inline(always)]
pub fn tcgen_st(ctx: &mut ExecCtx<'_>, args: &TcgenStArgs) -> HResult {
    tcgen::tcgen_st(ctx, args)
}

#[inline(always)]
pub fn tcgen_wait(ctx: &mut ExecCtx<'_>, st: bool) -> HResult {
    tcgen::tcgen_wait(ctx, st)
}

#[inline(always)]
pub fn tcgen_cp(ctx: &mut ExecCtx<'_>, args: TcgenCpArgs) -> HResult {
    tcgen::tcgen_cp(ctx, args)
}

/// Evaluate operands, build `TcgenMmaPayload`, issue an AsyncOp ordered after the CTA's previous tcgen05 op, `WorkCmd::Issue`.
#[inline(always)]
pub fn tcgen_mma(ctx: &mut ExecCtx<'_>, args: &TcgenMmaArgs) -> HResult {
    tcgen::tcgen_mma(ctx, args)
}

/// Provisional: execute a tile op over its element maps.
#[inline(always)]
pub fn tile(ctx: &mut ExecCtx<'_>, args: &TileArgs) -> HResult {
    tcgen::tile(ctx, args)
}

/// Interpreter dispatch: call the handler of `ins` with its fields.
#[inline(always)]
pub fn dispatch(ctx: &mut ExecCtx<'_>, ins: &Instr) -> HResult {
    match ins {
        Instr::Nop => nop(ctx),
        Instr::If { cond, else_pc, end_pc, elect } => if_(ctx, *cond, *else_pc, *end_pc, *elect),
        Instr::Else { end_pc } => else_(ctx, *end_pc),
        Instr::EndIf => end_if(ctx),
        Instr::LoopBegin { end_pc } => loop_begin(ctx, *end_pc),
        Instr::LoopIf { cond, end_pc } => loop_if(ctx, *cond, *end_pc),
        Instr::LoopEnd { head_pc } => loop_end(ctx, *head_pc),
        Instr::Break => break_(ctx),
        Instr::Continue => continue_(ctx),
        Instr::Exit => exit(ctx),
        Instr::Assert { cond, msg } => assert(ctx, *cond, *msg),
        Instr::Unsupported { reason } => unsupported(ctx, *reason),
        Instr::Mov { dst, src } => mov(ctx, *dst, *src),
        Instr::ReadSpecial { dst, sreg } => read_special(ctx, *dst, *sreg),
        Instr::ReadParam { dst, slot } => read_param(ctx, *dst, *slot),
        Instr::Unary { op, ty, dst, a } => unary(ctx, *op, *ty, *dst, *a),
        Instr::Binary { op, ty, dst, a, b } => binary(ctx, *op, *ty, *dst, *a, *b),
        Instr::Ternary { op, ty, dst, a, b, c } => ternary(ctx, *op, *ty, *dst, *a, *b, *c),
        Instr::Compare { op, ty, dst, a, b } => compare(ctx, *op, *ty, *dst, *a, *b),
        Instr::Select { ty, dst, cond, a, b } => select(ctx, *ty, *dst, *cond, *a, *b),
        Instr::Cast { from, to, dst, src, rnd, sat } => cast(ctx, *from, *to, *dst, *src, *rnd, *sat),
        Instr::Ptx { op, dsts, srcs, pred, keep_dst } => ptx(ctx, *op, dsts, srcs, *pred, *keep_dst),
        Instr::LoadRegIndexed { dst, base, len, idx } => load_reg_indexed(ctx, *dst, *base, *len, *idx),
        Instr::StoreRegIndexed { base, len, idx, value } => store_reg_indexed(ctx, *base, *len, *idx, *value),
        Instr::Shfl { mode, ty, dst, dst_pred, src, lane, clamp, membermask } => shfl(ctx, *mode, *ty, *dst, *dst_pred, *src, *lane, *clamp, *membermask),
        Instr::Vote { mode, dst, pred, membermask } => vote(ctx, *mode, *dst, *pred, *membermask),
        Instr::Redux { op, ty, dst, src, membermask } => redux(ctx, *op, *ty, *dst, *src, *membermask),
        Instr::Elect { dst_pred, dst_lane, membermask } => elect(ctx, *dst_pred, *dst_lane, *membermask),
        Instr::WarpSync { membermask } => warp_sync(ctx, *membermask),
        Instr::LdMatrix { dsts, addr, space, shape, num, trans, fmt } => ldmatrix(ctx, dsts, *addr, *space, *shape, *num, *trans, *fmt),
        Instr::StMatrix { srcs, addr, space, shape, num, trans } => stmatrix(ctx, srcs, *addr, *space, *shape, *num, *trans),
        Instr::Load { ty, dst, buf, offset, sem, scope, mods } => load(ctx, *ty, *dst, *buf, *offset, *sem, *scope, *mods),
        Instr::Store { ty, buf, offset, value, sem, scope, mods } => store(ctx, *ty, *buf, *offset, *value, *sem, *scope, *mods),
        Instr::LoadAddr { ty, dst, addr, space, sem, scope, mods } => load_addr(ctx, *ty, *dst, *addr, *space, *sem, *scope, *mods),
        Instr::StoreAddr { ty, addr, space, value, sem, scope, mods } => store_addr(ctx, *ty, *addr, *space, *value, *sem, *scope, *mods),
        Instr::AddrOf { dst, buf, offset } => addr_of(ctx, *dst, *buf, *offset),
        Instr::Atom { op, ty, dst, addr, space, value, cmp, sem, scope, ftz } => atom(ctx, *op, *ty, *dst, *addr, *space, *value, *cmp, *sem, *scope, *ftz),
        Instr::StBulk { addr, space, size } => st_bulk(ctx, *addr, *space, *size),
        Instr::Discard { addr, space, size } => discard(ctx, *addr, *space, *size),
        Instr::Cvta { dst, src, space, to_generic } => cvta(ctx, *dst, *src, *space, *to_generic),
        Instr::Isspacep { dst, src, space } => isspacep(ctx, *dst, *src, *space),
        Instr::Mapa { dst, src, rank, space } => mapa(ctx, *dst, *src, *rank, *space),
        Instr::GetCtaRank { dst, src, space } => getctarank(ctx, *dst, *src, *space),
        Instr::CpAsync { dst, src, cp_size, src_size, ignore_src, mods } => cp_async(ctx, *dst, *src, *cp_size, *src_size, *ignore_src, *mods),
        Instr::AsyncCommit { domain } => async_commit(ctx, *domain),
        Instr::AsyncWait { domain, n, read } => async_wait(ctx, *domain, *n, *read),
        Instr::CpAsyncMbarArrive { mbar, space, noinc } => cp_async_mbar_arrive(ctx, *mbar, *space, *noinc),
        Instr::BulkCopy(args) => bulk_copy(ctx, *args),
        Instr::Tma(args) => tma(ctx, args),
        Instr::StAsync(args) => st_async(ctx, *args),
        Instr::TensorMapReplace { tmap, space, field, ord, value } => tensormap_replace(ctx, *tmap, *space, *field, *ord, *value),
        Instr::TensorMapCopyFence { dst, src, size, scope } => tensormap_cp_fence(ctx, *dst, *src, *size, *scope),
        Instr::Barrier { kind, id, count, aligned } => barrier(ctx, *kind, *id, *count, *aligned),
        Instr::ClusterArrive { sem, aligned } => cluster_arrive(ctx, *sem, *aligned),
        Instr::ClusterWait { acquire, aligned } => cluster_wait(ctx, *acquire, *aligned),
        Instr::GridSync => grid_sync(ctx),
        Instr::MbarInit { mbar, space, count, layout_v1 } => mbar_init(ctx, *mbar, *space, *count, *layout_v1),
        Instr::MbarInval { mbar, space } => mbar_inval(ctx, *mbar, *space),
        Instr::MbarArrive(args) => mbar_arrive(ctx, *args),
        Instr::MbarTx { op, mbar, space, bytes, multicast, scope } => mbar_tx(ctx, *op, *mbar, *space, *bytes, *multicast, *scope),
        Instr::MbarTestWait { kind, mbar, space, phase, sem, scope, dst, report, report_value } => mbar_test_wait(ctx, *kind, *mbar, *space, *phase, *sem, *scope, *dst, *report, *report_value),
        Instr::MbarWait { mbar, space, phase, sem, scope } => mbar_wait(ctx, *mbar, *space, *phase, *sem, *scope),
        Instr::MbarQuery { dst, op } => mbar_query(ctx, *dst, *op),
        Instr::Fence { kind, sem, scope } => fence(ctx, *kind, *sem, *scope),
        Instr::SetMaxNReg { inc, count } => setmaxnreg(ctx, *inc, *count),
        Instr::WaitUntil { dst, addr, ty, space, sem, scope, pred, captures } => wait_until(ctx, *dst, *addr, *ty, *space, *sem, *scope, *pred, captures),
        Instr::GridDepControl { launch_dependents } => griddepcontrol(ctx, *launch_dependents),
        Instr::ClcTryCancel { resp, mbar, multicast } => clc_try_cancel(ctx, *resp, *mbar, *multicast),
        Instr::TcgenAlloc { dst, ncols, cta_group, exclusive } => tcgen_alloc(ctx, *dst, *ncols, *cta_group, *exclusive),
        Instr::TcgenDealloc { taddr, ncols, cta_group, exclusive } => tcgen_dealloc(ctx, *taddr, *ncols, *cta_group, *exclusive),
        Instr::TcgenRelinquish { cta_group } => tcgen_relinquish(ctx, *cta_group),
        Instr::TcgenCommit { mbar, space, cta_group, multicast, sync_restrict, multicast_width } => tcgen_commit(ctx, *mbar, *space, *cta_group, *multicast, *sync_restrict, *multicast_width),
        Instr::TcgenLd(args) => tcgen_ld(ctx, args),
        Instr::TcgenSt(args) => tcgen_st(ctx, args),
        Instr::TcgenWait { st } => tcgen_wait(ctx, *st),
        Instr::TcgenCp(args) => tcgen_cp(ctx, *args),
        Instr::TcgenMma(args) => tcgen_mma(ctx, args),
        Instr::Tile(args) => tile(ctx, args),
    }
}

