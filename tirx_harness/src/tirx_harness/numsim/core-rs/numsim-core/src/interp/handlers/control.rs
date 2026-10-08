//! Structured control flow: mask stack, loops (budget, quantum, spin
//! parking), exit, assert.
//!
//! Masks (see `Instr` docs): `active`, per-frame `entry`/`taken`, `live`.
//! Every restore intersects `live` and the innermost enclosing loop's
//! non-broken, non-continued lanes.

use super::HResult;
use crate::interp::support::{self, lane_val, ProtoExtra};
use crate::interp::{ExecCtx, ExecErrorKind, Flow, FrameKind, MaskFrame};
use crate::observe::SyncKind;
use crate::program::{Instr, Operand, Pc, StrId};
use crate::sync::{async_group, cluster, ResourceId, SyncCmd};
use crate::value::WarpMask;

/// `WarpState::resume` marker of a spin-parked `LoopEnd`.
pub const PARKED: u64 = u64::MAX - 1;

/// Iterations between voluntary yields of a running loop.
pub const LOOP_QUANTUM: u64 = 64;

/// Lanes of `v` that are non-zero, within `mask`.
#[inline]
pub fn cond_mask(ctx: &ExecCtx<'_>, cond: Operand, mask: WarpMask) -> WarpMask {
    match cond {
        Operand::Const(c) => {
            if ctx.program.consts[c.0 as usize].bits != 0 {
                mask
            } else {
                WarpMask::NONE
            }
        }
        Operand::Reg(_) => {
            let mut m = 0u32;
            for l in mask.lanes() {
                if lane_val(ctx, cond, l) != 0 {
                    m |= 1 << l;
                }
            }
            WarpMask(m)
        }
    }
}

/// `live` minus lanes broken/continued in the innermost loop frame at or
/// below `depth` (exclusive upper bound into `frames`).
#[inline]
fn alive_below(ctx: &ExecCtx<'_>, depth: usize) -> WarpMask {
    let live = ctx.warp.live;
    for f in ctx.warp.frames[..depth].iter().rev() {
        if let FrameKind::Loop { broken, continued, .. } = f.kind {
            return live.and_not(broken.or(continued));
        }
    }
    live
}

fn innermost_loop(ctx: &ExecCtx<'_>) -> Option<usize> {
    ctx.warp.frames.iter().rposition(|f| matches!(f.kind, FrameKind::Loop { .. }))
}

fn internal(ctx: &ExecCtx<'_>, msg: &str) -> crate::interp::ExecError {
    ctx.error(ExecErrorKind::Internal, msg.to_string())
}

#[inline]
pub fn nop(_ctx: &mut ExecCtx<'_>) -> HResult {
    Ok(Flow::Next)
}

#[inline]
pub fn if_(ctx: &mut ExecCtx<'_>, cond: Operand, else_pc: Pc, _end_pc: Pc, _elect: bool) -> HResult {
    let entry = ctx.warp.active;
    let taken = if entry.is_empty() { entry } else { cond_mask(ctx, cond, entry) };
    let origin = ctx.warp.pc;
    ctx.warp.frames.push(MaskFrame { kind: FrameKind::Then, origin, entry, taken });
    ctx.warp.active = taken;
    if taken.is_empty() {
        Ok(Flow::Jump(else_pc))
    } else {
        Ok(Flow::Next)
    }
}

#[inline]
pub fn else_(ctx: &mut ExecCtx<'_>, end_pc: Pc) -> HResult {
    let depth = ctx.warp.frames.len();
    let Some(top) = ctx.warp.frames.last_mut() else { return Err(internal(ctx, "Else without frame")) };
    if top.kind == FrameKind::Else {
        // A then-arm resumed after its else arm already ran (divergent
        // scheduling rule): go to the EndIf.
        return Ok(Flow::Jump(end_pc));
    }
    if top.kind != FrameKind::Then {
        return Err(internal(ctx, "Else does not match a Then frame"));
    }
    top.kind = FrameKind::Else;
    let (entry, taken) = (top.entry, top.taken);
    let alive = alive_below(ctx, depth - 1);
    let a = entry.and_not(taken).and(alive);
    ctx.warp.active = a;
    if a.is_empty() {
        Ok(Flow::Jump(end_pc))
    } else {
        Ok(Flow::Next)
    }
}

#[inline]
pub fn end_if(ctx: &mut ExecCtx<'_>) -> HResult {
    let depth = ctx.warp.frames.len().wrapping_sub(1);
    if let Some(k) = ctx.warp.suspended.iter().position(|s| s.depth == depth) {
        // The other arm of this If is suspended at a blocking instruction:
        // resume it (divergent scheduling rule).
        let s = ctx.warp.suspended.swap_remove(k);
        let w = &mut *ctx.warp;
        w.frames[depth].kind = FrameKind::Else;
        w.frames.extend(s.frames);
        w.active = s.active.and(w.live);
        w.resume = s.resume;
        w.poll = s.poll;
        return Ok(Flow::Jump(s.pc));
    }
    let Some(top) = ctx.warp.frames.pop() else { return Err(internal(ctx, "EndIf without frame")) };
    if !matches!(top.kind, FrameKind::Then | FrameKind::Else) {
        return Err(internal(ctx, "EndIf does not match an If frame"));
    }
    let depth = ctx.warp.frames.len();
    ctx.warp.active = top.entry.and(alive_below(ctx, depth));
    Ok(Flow::Next)
}

#[inline]
pub fn loop_begin(ctx: &mut ExecCtx<'_>, _end_pc: Pc) -> HResult {
    let entry = ctx.warp.active;
    let begin = ctx.warp.pc;
    ctx.warp.frames.push(MaskFrame {
        kind: FrameKind::Loop { begin, iteration: 0, broken: WarpMask::NONE, continued: WarpMask::NONE },
        origin: begin,
        entry,
        taken: entry,
    });
    Ok(Flow::Next)
}

/// Pop the loop frame on top and restore its entry lanes.
fn exit_loop(ctx: &mut ExecCtx<'_>) {
    if let Some(f) = ctx.warp.frames.pop() {
        let depth = ctx.warp.frames.len();
        ctx.warp.active = f.entry.and(alive_below(ctx, depth));
    }
}

#[inline]
pub fn loop_if(ctx: &mut ExecCtx<'_>, cond: Operand, end_pc: Pc) -> HResult {
    let a = ctx.warp.active;
    let a = if a.is_empty() { a } else { cond_mask(ctx, cond, a) };
    let Some(top) = ctx.warp.frames.last_mut() else { return Err(internal(ctx, "LoopIf without frame")) };
    if !matches!(top.kind, FrameKind::Loop { .. }) {
        return Err(internal(ctx, "LoopIf outside its loop frame"));
    }
    top.taken = a;
    ctx.warp.active = a;
    if a.is_empty() {
        exit_loop(ctx);
        Ok(Flow::Jump(Pc(end_pc.0 + 1)))
    } else {
        Ok(Flow::Next)
    }
}

#[inline]
pub fn loop_end(ctx: &mut ExecCtx<'_>, head_pc: Pc) -> HResult {
    if ctx.warp.resume == Some(PARKED) {
        // Retry of a spin-parked iteration: run the next one.
        ctx.warp.resume = None;
        return Ok(Flow::Jump(head_pc));
    }
    // Defensive: frames opened inside the body belong to this iteration.
    while ctx.warp.frames.last().is_some_and(|f| !matches!(f.kind, FrameKind::Loop { .. })) {
        ctx.warp.frames.pop();
    }
    let live = ctx.warp.live;
    let budget = ctx.config.loop_budget;
    let Some(top) = ctx.warp.frames.last_mut() else { return Err(internal(ctx, "LoopEnd without frame")) };
    let FrameKind::Loop { begin, iteration, broken, continued } = &mut top.kind else {
        return Err(internal(ctx, "LoopEnd without loop frame"));
    };
    *iteration += 1;
    let it = *iteration;
    let begin = *begin;
    *continued = WarpMask::NONE;
    let next = top.taken.and(live).and_not(*broken);
    top.taken = next;
    if it > budget {
        let mut e = ctx.error(
            ExecErrorKind::Budget,
            format!("loop exceeded its iteration budget of {budget} (raise loop_budget)"),
        );
        e.site = ctx.program.site_of(begin);
        return Err(e);
    }
    ctx.warp.active = next;
    if next.is_empty() {
        exit_loop(ctx);
        ctx.warp.poll = Default::default();
        return Ok(Flow::Next);
    }
    let poll = std::mem::take(&mut ctx.warp.poll);
    if let (Some(r), false) = (poll.failed_on, poll.progressed) {
        // A whole iteration of failed polls: park until the next round.
        ctx.warp.resume = Some(PARKED);
        return Ok(Flow::Blocked(r));
    }
    if it % LOOP_QUANTUM == 0 {
        Ok(Flow::Yield(head_pc))
    } else {
        Ok(Flow::Jump(head_pc))
    }
}

/// End-pc of the innermost loop (its `LoopEnd`).
fn loop_end_pc(ctx: &ExecCtx<'_>, i: usize) -> Option<Pc> {
    let FrameKind::Loop { begin, .. } = ctx.warp.frames[i].kind else { return None };
    match ctx.program.code[begin.0 as usize] {
        Instr::LoopBegin { end_pc } => Some(end_pc),
        _ => None,
    }
}

/// After `Break`/`Continue`: if no lane of the loop iteration remains
/// active in any enclosing frame, skip straight to the `LoopEnd`.
fn skip_if_iteration_done(ctx: &mut ExecCtx<'_>, li: usize) -> HResult {
    let FrameKind::Loop { broken, continued, .. } = ctx.warp.frames[li].kind else { return Ok(Flow::Next) };
    let remaining = ctx.warp.frames[li].taken.and(ctx.warp.live).and_not(broken.or(continued));
    if remaining.is_empty() {
        if let Some(end) = loop_end_pc(ctx, li) {
            ctx.warp.frames.truncate(li + 1);
            return Ok(Flow::Jump(end));
        }
    }
    Ok(Flow::Next)
}

#[inline]
pub fn break_(ctx: &mut ExecCtx<'_>) -> HResult {
    let a = ctx.warp.active;
    if a.is_empty() {
        return Ok(Flow::Next);
    }
    let Some(li) = innermost_loop(ctx) else { return Err(internal(ctx, "Break outside a loop")) };
    if let FrameKind::Loop { broken, .. } = &mut ctx.warp.frames[li].kind {
        *broken = broken.or(a);
    }
    ctx.warp.active = WarpMask::NONE;
    skip_if_iteration_done(ctx, li)
}

#[inline]
pub fn continue_(ctx: &mut ExecCtx<'_>) -> HResult {
    let a = ctx.warp.active;
    if a.is_empty() {
        return Ok(Flow::Next);
    }
    let Some(li) = innermost_loop(ctx) else { return Err(internal(ctx, "Continue outside a loop")) };
    if let FrameKind::Loop { continued, .. } = &mut ctx.warp.frames[li].kind {
        *continued = continued.or(a);
    }
    ctx.warp.active = WarpMask::NONE;
    skip_if_iteration_done(ctx, li)
}

/// Retire the active lanes: cluster-barrier membership, bulk async groups
/// (implicit commit), grid barrier; `Flow::Exit` when the warp is done.
pub fn exit(ctx: &mut ExecCtx<'_>) -> HResult {
    let lanes = ctx.warp.active;
    if lanes.is_empty() {
        return Ok(Flow::Next);
    }
    ctx.warp.live = ctx.warp.live.and_not(lanes);
    ctx.warp.active = WarpMask::NONE;
    if ctx.loaded.uses_cluster_barrier {
        let res = ResourceId::Cluster { cluster: ctx.cta.cluster };
        let warp = ctx.cta.rank_in_cluster * ctx.launch.warps_per_cta() + ctx.warp.warp_in_cta;
        let cmd = SyncCmd::Cluster(cluster::Cmd::Exit { warp, lanes: lanes.bits() });
        support::step(ctx, res, cmd)?;
        support::protocol(ctx, lanes, vec![(res, cmd)], ProtoExtra::default());
    }
    // Bulk issues are committed implicitly at exit.
    for lane in lanes.lanes() {
        let res = ResourceId::AsyncGroup { warp: ctx.warp.id, lane: lane as u8, domain: async_group::Domain::Bulk };
        if ctx.aux.groups.open.get(&res).is_some_and(|v| !v.is_empty()) {
            let cmd = SyncCmd::AsyncGroup(async_group::Cmd::Exit);
            support::step(ctx, res, cmd)?;
            if let Some(crate::sync::Resource::AsyncGroup(s)) = ctx.sync.get(res) {
                if let Some(g) = s.groups.back() {
                    let due = ctx.aux.groups.commit(res, g.ordinal);
                    ctx.sync.completions.extend(due);
                }
            }
        }
    }
    if ctx.warp.live.is_empty() {
        ctx.aux.exited_warps += 1;
        // A grid barrier waiting only on exited warps completes.
        let g = &mut ctx.aux.grid;
        if !g.arrived.is_empty() && g.arrived.len() as u32 >= ctx.launch.num_warps() - ctx.aux.exited_warps {
            g.gen += 1;
            g.arrived.clear();
        }
        if ctx.observing && ctx.loaded.uses_cluster_barrier {
            let _ = SyncKind::WarpSync { mask: WarpMask::NONE };
        }
        return Ok(Flow::Exit);
    }
    Ok(Flow::Next)
}

#[inline]
pub fn assert(ctx: &mut ExecCtx<'_>, cond: Operand, msg: Option<StrId>) -> HResult {
    active_or_next!(ctx);
    let a = ctx.warp.active;
    let ok = cond_mask(ctx, cond, a);
    let bad = a.and_not(ok);
    if bad.is_empty() {
        return Ok(Flow::Next);
    }
    let text = msg.and_then(|s| ctx.program.strings.get(s.0 as usize)).cloned().unwrap_or_else(|| "assertion failed".into());
    Err(support::err(ctx, ExecErrorKind::Trap, bad, text))
}

#[inline]
pub fn unsupported(ctx: &mut ExecCtx<'_>, reason: StrId) -> HResult {
    active_or_next!(ctx);
    let r = ctx.program.strings.get(reason.0 as usize).cloned().unwrap_or_default();
    Err(ctx.error(ExecErrorKind::Unsupported, r))
}
