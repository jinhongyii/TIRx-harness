//! Warp collectives (shfl/vote/redux/elect/syncwarp) and ldmatrix/stmatrix.
//!
//! Structured SIMT: a collective is executed by the active lanes together.
//! `membermask` must name exactly the active lanes among the live ones
//! (lanes it names that are not executing would hang on hardware; executing
//! lanes it omits are undefined): otherwise `Divergence`.

use super::HResult;
use crate::dtype::Ty;
use crate::interp::support::{self, lane_val, reg_ty, uniform_over, write_lane, Accesses};
use crate::interp::{ExecCtx, ExecError, ExecErrorKind, Flow};
use crate::observe::{AccessKind, SyncKind};
use crate::oplib;
use crate::program::*;
use crate::value::{WarpMask, WarpValue};

/// Validate `membermask` against the active/live lanes; returns the members.
pub fn members(ctx: &ExecCtx<'_>, membermask: Operand) -> Result<WarpMask, ExecError> {
    let active = ctx.warp.active;
    let m = WarpMask(uniform_over(ctx, membermask, active)? as u32);
    let missing = m.and(ctx.warp.live).and_not(active);
    let extra = active.and_not(m);
    if !missing.is_empty() || !extra.is_empty() {
        return Err(support::err(
            ctx,
            ExecErrorKind::Divergence,
            active,
            format!("membermask {m} does not match the executing lanes {active} (live {})", ctx.warp.live),
        ));
    }
    Ok(active)
}

#[inline]
pub fn shfl(
    ctx: &mut ExecCtx<'_>,
    mode: ShflMode,
    ty: Ty,
    dst: Reg,
    dst_pred: Option<Reg>,
    src: Operand,
    lane: Operand,
    clamp: Operand,
    membermask: Operand,
) -> HResult {
    active_or_next!(ctx);
    let m = members(ctx, membermask)?;
    let lanes = ctx.read(lane);
    let clamps = ctx.read(clamp);
    let mm = ctx.read(membermask);
    let mut inrange = WarpMask::NONE;
    for i in 0..ty.slots().min(reg_ty(ctx, dst).slots()) {
        let s = ctx.read_slot(src, i);
        let (v, ok) = oplib::shfl_sync(mode, &s, &lanes, &clamps, &mm, m).map_err(|e| support::op_err(ctx, e))?;
        inrange = ok;
        ctx.write_slot(dst, i, &v);
    }
    if let Some(p) = dst_pred {
        super::alu::write_pred(ctx, p, inrange.and(m), m);
    }
    Ok(Flow::Next)
}

#[inline]
pub fn vote(ctx: &mut ExecCtx<'_>, mode: VoteMode, dst: Reg, pred: Operand, membermask: Operand) -> HResult {
    active_or_next!(ctx);
    let m = members(ctx, membermask)?;
    let yes = super::control::cond_mask(ctx, pred, m);
    let v = match mode {
        VoteMode::All => (yes == m) as u64,
        VoteMode::Any => (!yes.is_empty()) as u64,
        VoteMode::Uni => (yes == m || yes.is_empty()) as u64,
        VoteMode::Ballot => yes.bits() as u64,
    };
    for l in m.lanes() {
        write_lane(ctx, dst, l, v);
    }
    Ok(Flow::Next)
}

#[inline]
pub fn redux(ctx: &mut ExecCtx<'_>, op: ReduxOp, ty: Ty, dst: Reg, src: Operand, membermask: Operand) -> HResult {
    active_or_next!(ctx);
    let m = members(ctx, membermask)?;
    let s = ctx.read(src);
    let v = oplib::redux(op, ty, &s, m).map_err(|e| support::op_err(ctx, e))?;
    for l in m.lanes() {
        write_lane(ctx, dst, l, v);
    }
    Ok(Flow::Next)
}

#[inline]
pub fn elect(ctx: &mut ExecCtx<'_>, dst_pred: Reg, dst_lane: Option<Reg>, membermask: Operand) -> HResult {
    active_or_next!(ctx);
    let m = members(ctx, membermask)?;
    let leader = m.first().unwrap_or(0);
    super::alu::write_pred(ctx, dst_pred, WarpMask::lane(leader), m);
    if let Some(d) = dst_lane {
        for l in m.lanes() {
            write_lane(ctx, d, l, leader as u64);
        }
    }
    Ok(Flow::Next)
}

#[inline]
pub fn warp_sync(ctx: &mut ExecCtx<'_>, membermask: Operand) -> HResult {
    active_or_next!(ctx);
    let active = ctx.warp.active;
    let m = WarpMask(uniform_over(ctx, membermask, active)? as u32);
    if !active.and_not(m).is_empty() {
        return Err(support::err(ctx, ExecErrorKind::Divergence, active, format!("__syncwarp({m}) executed by lanes outside the mask")));
    }
    if !m.and(ctx.warp.live).and_not(active).is_empty() {
        // Lanes named by the mask reach a different __syncwarp later: the
        // structured interpreter cannot rendezvous divergent paths.
        return Err(support::unsupported(ctx, "__syncwarp across divergent paths"));
    }
    support::sync_event(ctx, active, SyncKind::WarpSync { mask: active });
    Ok(Flow::Next)
}

/// Bytes of one `m8n8.b16` row and rows per matrix.
const ROW_BYTES: usize = 16;

fn check_ldst_matrix(ctx: &ExecCtx<'_>, shape: MatrixShape, num: u8, fmt: MatrixFmt) -> Result<(), ExecError> {
    if shape != MatrixShape::M8N8 || fmt != MatrixFmt::B16 || !matches!(num, 1 | 2 | 4) {
        return Err(support::unsupported(ctx, &format!("ldmatrix/stmatrix {shape:?} {fmt:?} x{num}")));
    }
    if ctx.warp.active != ctx.warp.live {
        return Err(support::err(ctx, ExecErrorKind::Divergence, ctx.warp.active, ".sync.aligned matrix op with divergent lanes"));
    }
    Ok(())
}

/// Byte offset inside the 8x16-byte matrix tile of the 4 bytes thread `t`
/// holds: `(row, byte_in_row)` for each of its two b16 halves.
#[inline]
fn frag(t: usize, trans: bool) -> [(usize, usize); 2] {
    if trans {
        let c = t / 4;
        let r = 2 * (t % 4);
        [(r, 2 * c), (r + 1, 2 * c)]
    } else {
        let r = t / 4;
        let c = 2 * (t % 4);
        [(r, 2 * c), (r, 2 * c + 2)]
    }
}

#[inline]
pub fn ldmatrix(
    ctx: &mut ExecCtx<'_>,
    dsts: &[Reg],
    addr: Operand,
    space: AddrSpace,
    shape: MatrixShape,
    num: u8,
    trans: bool,
    fmt: MatrixFmt,
) -> HResult {
    active_or_next!(ctx);
    check_ldst_matrix(ctx, shape, num, fmt)?;
    let active = ctx.warp.active;
    support::sync_event(ctx, active, SyncKind::WarpSync { mask: active });
    let mut acc = Accesses::default();
    // rows[i][r] = 16 bytes of row r of matrix i.
    let mut rows = [[[0u8; ROW_BYTES]; 8]; 4];
    for i in 0..num as usize {
        for (r, row) in rows[i].iter_mut().enumerate() {
            let lane = 8 * i + r;
            if !active.contains(lane) {
                continue;
            }
            let a = lane_val(ctx, addr, lane);
            let loc = support::resolve(ctx, space, a, lane, ROW_BYTES as u64)?;
            support::mem_read(ctx, loc, lane, row)?;
            if ctx.observing {
                acc.push(loc, lane as u8, ROW_BYTES as u64);
            }
        }
    }
    for t in active.lanes() {
        let mut bytes = [0u8; 16];
        for i in 0..num as usize {
            for (h, (r, c)) in frag(t, trans).into_iter().enumerate() {
                bytes[4 * i + 2 * h..4 * i + 2 * h + 2].copy_from_slice(&rows[i][r][c..c + 2]);
            }
        }
        let mut pos = 0usize;
        for &d in dsts {
            let n = reg_ty(ctx, d).mem_bytes() as usize;
            let end = (pos + n).min(4 * num as usize);
            support::write_lane_bytes(ctx, d, t, &bytes[pos..end]);
            pos += n;
        }
    }
    let sp = support::spec(ctx, AccessKind::Read, Sem::Weak, Scope::Cta, Proxy::Generic);
    support::emit(ctx, sp, &mut acc);
    support::sync_event(ctx, active, SyncKind::WarpSync { mask: active });
    Ok(Flow::Next)
}

#[inline]
pub fn stmatrix(
    ctx: &mut ExecCtx<'_>,
    srcs: &[Operand],
    addr: Operand,
    space: AddrSpace,
    shape: MatrixShape,
    num: u8,
    trans: bool,
) -> HResult {
    active_or_next!(ctx);
    check_ldst_matrix(ctx, shape, num, MatrixFmt::B16)?;
    let active = ctx.warp.active;
    support::sync_event(ctx, active, SyncKind::WarpSync { mask: active });
    let mut rows = [[[0u8; ROW_BYTES]; 8]; 4];
    for t in active.lanes() {
        let mut bytes = [0u8; 16];
        let mut pos = 0usize;
        for &s in srcs {
            let ty = support::operand_ty(ctx, s);
            let n = ty.mem_bytes() as usize;
            let mut tmp = [0u8; 32];
            support::lane_bytes(ctx, s, ty, t, &mut tmp);
            let end = (pos + n).min(16);
            bytes[pos..end].copy_from_slice(&tmp[..end - pos]);
            pos += n;
        }
        for i in 0..num as usize {
            for (h, (r, c)) in frag(t, trans).into_iter().enumerate() {
                rows[i][r][c..c + 2].copy_from_slice(&bytes[4 * i + 2 * h..4 * i + 2 * h + 2]);
            }
        }
    }
    let mut acc = Accesses::default();
    for i in 0..num as usize {
        for (r, row) in rows[i].iter().enumerate() {
            let lane = 8 * i + r;
            if !active.contains(lane) {
                continue;
            }
            let a = lane_val(ctx, addr, lane);
            let loc = support::resolve(ctx, space, a, lane, ROW_BYTES as u64)?;
            support::mem_write(ctx, loc, lane, row)?;
            if ctx.observing {
                acc.push(loc, lane as u8, ROW_BYTES as u64);
            }
        }
    }
    let sp = support::spec(ctx, AccessKind::Write, Sem::Weak, Scope::Cta, Proxy::Generic);
    support::emit(ctx, sp, &mut acc);
    support::sync_event(ctx, active, SyncKind::WarpSync { mask: active });
    let _: Option<WarpValue<u64>> = None;
    Ok(Flow::Next)
}
