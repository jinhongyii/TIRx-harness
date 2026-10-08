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
            ExecErrorKind::WarpCollectiveDivergence,
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
        return Err(support::err(ctx, ExecErrorKind::WarpCollectiveDivergence, active, format!("__syncwarp({m}) executed by lanes outside the mask")));
    }
    let required = m.and(ctx.warp.live);
    if !required.and_not(active).is_empty() {
        // Lanes named by the mask reach a `__syncwarp` on another path:
        // a rendezvous per warp, completed by the arm that brings the last
        // lanes (the divergent scheduling rule runs the other arm while this
        // one blocks).
        let id = ctx.warp.id;
        let tag = 1u64 << 59;
        let entry = ctx.aux.warp_sync.entry(id).or_default();
        if let Some(tok) = ctx.warp.resume.filter(|t| t & tag != 0) {
            if entry.1 > tok & !tag {
                ctx.warp.resume = None;
                support::sync_event(ctx, active, SyncKind::WarpSync { mask: required });
                return Ok(Flow::Next);
            }
            return Ok(Flow::Blocked(crate::sync::ResourceId::WarpSync { warp: id }));
        }
        entry.0 = entry.0.or(active);
        if entry.0.and(required) == required {
            entry.0 = WarpMask::NONE;
            entry.1 += 1;
            support::sync_event(ctx, active, SyncKind::WarpSync { mask: required });
            return Ok(Flow::Next);
        }
        ctx.warp.resume = Some(tag | entry.1);
        return Ok(Flow::Blocked(crate::sync::ResourceId::WarpSync { warp: id }));
    }
    support::sync_event(ctx, active, SyncKind::WarpSync { mask: active });
    Ok(Flow::Next)
}

fn full_warp(ctx: &ExecCtx<'_>, what: &str) -> Result<(), ExecError> {
    if ctx.warp.active != ctx.warp.live {
        return Err(support::err(ctx, ExecErrorKind::WarpCollectiveDivergence, ctx.warp.active, format!("{what}.sync.aligned with divergent lanes")));
    }
    Ok(())
}

/// Spread 32-bit words (per lane) over destination registers by bytes.
fn write_words(ctx: &mut ExecCtx<'_>, dsts: &[Reg], words: &[WarpValue<u32>], lanes: WarpMask) {
    for t in lanes.lanes() {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w[t].to_le_bytes()).collect();
        let mut pos = 0usize;
        for &d in dsts {
            let n = reg_ty(ctx, d).mem_bytes() as usize;
            let end = (pos + n).min(bytes.len());
            if pos < end {
                support::write_lane_bytes(ctx, d, t, &bytes[pos..end]);
            }
            pos += n;
        }
    }
}

/// Source operands flattened to 32-bit words per lane, register-major.
fn read_words(ctx: &ExecCtx<'_>, srcs: &[Operand], count: usize) -> Vec<WarpValue<u32>> {
    let mut out = vec![[0u32; 32]; count];
    for t in 0..32 {
        let mut bytes = Vec::with_capacity(count * 4);
        for &s in srcs {
            let ty = support::operand_ty(ctx, s);
            let mut tmp = [0u8; 32];
            support::lane_bytes(ctx, s, ty, t, &mut tmp);
            bytes.extend_from_slice(&tmp[..ty.mem_bytes() as usize]);
        }
        bytes.resize(count * 4, 0);
        for (r, w) in out.iter_mut().enumerate() {
            w[t] = u32::from_le_bytes(bytes[4 * r..4 * r + 4].try_into().unwrap());
        }
    }
    out
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
    full_warp(ctx, "ldmatrix")?;
    let plan = oplib::ldmatrix_plan(shape, num, trans, fmt).map_err(|e| support::op_err(ctx, e))?;
    let active = ctx.warp.active;
    support::sync_event(ctx, active, SyncKind::WarpSync { mask: active });
    let rows: Vec<u64> = (0..32).map(|l| lane_val(ctx, addr, l)).collect();
    let mut acc = Accesses::default();
    let mut err = None;
    let frags = {
        let observing = ctx.observing;
        let ctx_cell = std::cell::RefCell::new(&mut *ctx);
        let acc_cell = std::cell::RefCell::new(&mut acc);
        let read = |provider: usize, delta: usize, len: usize| -> oplib::OpResult<Vec<u8>> {
            let mut c = ctx_cell.borrow_mut();
            let loc = match support::resolve(&c, space, rows[provider].wrapping_add(delta as u64), provider, len as u64) {
                Ok(l) => l,
                Err(e) => {
                    err = Some(e);
                    return Err(oplib::OpError::invalid("ldmatrix row"));
                }
            };
            let mut out = vec![0u8; len];
            if let Err(e) = support::mem_read(&mut c, loc, provider, &mut out) {
                err = Some(e);
                return Err(oplib::OpError::invalid("ldmatrix row"));
            }
            if observing {
                acc_cell.borrow_mut().push(loc, provider as u8, len as u64);
            }
            Ok(out)
        };
        oplib::ldmatrix_fragments(&plan, |p| Ok(rows[p]), read)
    };
    if let Some(e) = err {
        return Err(e);
    }
    let frags = frags.map_err(|e| support::op_err(ctx, e))?;
    write_words(ctx, dsts, &frags, active);
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
    full_warp(ctx, "stmatrix")?;
    let plan = oplib::stmatrix_plan(shape, num, trans).map_err(|e| support::op_err(ctx, e))?;
    let active = ctx.warp.active;
    support::sync_event(ctx, active, SyncKind::WarpSync { mask: active });
    let rows: Vec<u64> = (0..32).map(|l| lane_val(ctx, addr, l)).collect();
    let sources = read_words(ctx, srcs, plan.registers);
    let writes = oplib::stmatrix_writes(&plan, &sources, |p| Ok(rows[p])).map_err(|e| support::op_err(ctx, e))?;
    let mut acc = Accesses::default();
    for (provider, delta, bytes) in writes {
        let loc = support::resolve(ctx, space, rows[provider].wrapping_add(delta as u64), provider, bytes.len() as u64)?;
        support::mem_write(ctx, loc, provider, &bytes)?;
        if ctx.observing {
            acc.push(loc, provider as u8, bytes.len() as u64);
        }
    }
    let sp = support::spec(ctx, AccessKind::Write, Sem::Weak, Scope::Cta, Proxy::Generic);
    support::emit(ctx, sp, &mut acc);
    support::sync_event(ctx, active, SyncKind::WarpSync { mask: active });
    Ok(Flow::Next)
}
