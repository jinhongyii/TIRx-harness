//! Registers and TIR arithmetic. Numerics are oplib's; this file only
//! gathers operand slots and writes results under the active mask.

use super::HResult;
use crate::dtype::Ty;
use crate::interp::support::{self, extend, lane_int, lane_val, operand_ty, reg_ty, write_lane};
use crate::interp::{ExecCtx, ExecErrorKind, Flow};
use crate::oplib::{self, PtxIo};
use crate::program::*;
use crate::value::{WarpMask, WarpValue};

type Slots = [WarpValue<u64>; 4];

#[inline(always)]
fn zero_slots() -> Slots {
    [[0u64; 32]; 4]
}

/// Mask the top slot of a register value to its type's width.
#[inline]
fn clamp_top(ty: Ty, out: &mut Slots, mask: WarpMask) {
    let bits = ty.bits();
    let n = ty.slots();
    let rem = bits - (n - 1) * 64;
    if rem < 64 {
        let m = (1u64 << rem) - 1;
        let top = &mut out[(n - 1) as usize];
        for l in mask.lanes() {
            top[l] &= m;
        }
    }
}

#[inline]
pub fn mov(ctx: &mut ExecCtx<'_>, dst: Reg, src: Operand) -> HResult {
    active_or_next!(ctx);
    let ty = reg_ty(ctx, dst);
    let n = ty.slots();
    let mask = ctx.warp.active;
    if n == 1 && ty.bits() == 64 || (n == 1 && support::operand_ty(ctx, src).bits() <= ty.bits()) {
        // Single slot, no truncation needed.
        let v = ctx.read_slot(src, 0);
        let s = ctx.slot(dst);
        support::write_masked(ctx.warp.regs.get_mut(s), &v, mask);
        return Ok(Flow::Next);
    }
    let mut out = zero_slots();
    support::gather(ctx, src, n, &mut out);
    clamp_top(ty, &mut out, mask);
    support::scatter(ctx, dst, n, &out, mask);
    Ok(Flow::Next)
}

/// Value of a special register in one lane.
fn special(ctx: &ExecCtx<'_>, sreg: SpecialReg, lane: usize) -> u64 {
    let sh = ctx.launch;
    let [bx, by, _bz] = sh.block;
    let t = ctx.warp.warp_in_cta as u64 * 32 + lane as u64;
    let tid = [t % bx as u64, (t / bx as u64) % by as u64, t / (bx as u64 * by as u64)];
    let ax = |a: Axis| match a {
        Axis::X => 0usize,
        Axis::Y => 1,
        Axis::Z => 2,
    };
    let cl = sh.cluster;
    let cid = ctx.cta.ctaid;
    let lane_mask = |f: fn(u32, u32) -> bool| -> u64 {
        let mut m = 0u32;
        for i in 0..32u32 {
            if f(i, lane as u32) {
                m |= 1 << i;
            }
        }
        m as u64
    };
    match sreg {
        SpecialReg::LaneId => lane as u64,
        SpecialReg::WarpInCta => ctx.warp.warp_in_cta as u64,
        SpecialReg::WarpgroupInCta => (ctx.warp.warp_in_cta / 4) as u64,
        SpecialReg::ThreadInCta => t,
        SpecialReg::Tid(a) => tid[ax(a)],
        SpecialReg::NTid(a) => sh.block[ax(a)] as u64,
        SpecialReg::CtaId(a) => cid[ax(a)] as u64,
        SpecialReg::NCtaId(a) => sh.grid[ax(a)] as u64,
        SpecialReg::CtaLinear => ctx.cta.id.0 as u64,
        SpecialReg::ClusterId(a) => (cid[ax(a)] / cl[ax(a)].max(1)) as u64,
        SpecialReg::NClusterId(a) => (sh.grid[ax(a)] / cl[ax(a)].max(1)) as u64,
        SpecialReg::ClusterLinear => ctx.cta.cluster as u64,
        SpecialReg::ClusterCtaId(a) => (cid[ax(a)] % cl[ax(a)].max(1)) as u64,
        SpecialReg::ClusterNCtaId(a) => cl[ax(a)] as u64,
        SpecialReg::ClusterCtaRank => ctx.cta.rank_in_cluster as u64,
        SpecialReg::ClusterNCtaRank => sh.ctas_per_cluster() as u64,
        SpecialReg::LaneMaskEq => lane_mask(|i, l| i == l),
        SpecialReg::LaneMaskLt => lane_mask(|i, l| i < l),
        SpecialReg::LaneMaskLe => lane_mask(|i, l| i <= l),
        SpecialReg::LaneMaskGt => lane_mask(|i, l| i > l),
        SpecialReg::LaneMaskGe => lane_mask(|i, l| i >= l),
        SpecialReg::ActiveMask => ctx.warp.active.bits() as u64,
        // Deterministic representatives.
        // Physical SM id, hardware clocks and grid-launch tokens read a
        // deterministic representative zero (SUPPORTED_OPS.md `mov_sreg`,
        // legacy behaviour).
        SpecialReg::SmId => 0,
        SpecialReg::NSmId => NUM_SMS,
        SpecialReg::GridId => 0,
        SpecialReg::Clock => 0,
        SpecialReg::Clock64 => 0,
        SpecialReg::GlobalTimer => 0,
        SpecialReg::DynamicSmemSize => {
            (sh.smem_bytes.saturating_sub(ctx.program.topology.static_smem_bytes)) as u64
        }
        SpecialReg::TotalSmemSize => sh.smem_bytes as u64,
        // Deterministic representative: max warps per SM (sm_100).
        SpecialReg::NWarpId => 64,
    }
}

/// `%nsmid` representative (B200).
const NUM_SMS: u64 = 148;

#[inline]
pub fn read_special(ctx: &mut ExecCtx<'_>, dst: Reg, sreg: SpecialReg) -> HResult {
    active_or_next!(ctx);
    for l in ctx.warp.active.lanes() {
        let v = special(ctx, sreg, l);
        write_lane(ctx, dst, l, v);
    }
    Ok(Flow::Next)
}

#[inline]
pub fn read_param(ctx: &mut ExecCtx<'_>, dst: Reg, slot: ParamId) -> HResult {
    active_or_next!(ctx);
    let off = ctx.loaded.param_offsets[slot.0 as usize];
    let ty = reg_ty(ctx, dst);
    let n = (ty.mem_bytes() as usize).min(8);
    let raw = ctx.arena.read_raw(ctx.cta.params, crate::arena::ByteSpan::new(off, n as u64));
    let mut b = [0u8; 8];
    b[..n].copy_from_slice(&raw);
    let v = u64::from_le_bytes(b);
    for l in ctx.warp.active.lanes() {
        write_lane(ctx, dst, l, v);
    }
    Ok(Flow::Next)
}

#[inline]
pub fn unary(ctx: &mut ExecCtx<'_>, op: UnOp, ty: Ty, dst: Reg, a: Operand) -> HResult {
    active_or_next!(ctx);
    let n = ty.slots();
    let mask = ctx.warp.active;
    let mut sa = zero_slots();
    support::gather(ctx, a, n, &mut sa);
    // Output sized by the destination type (`Is*` write predicates).
    let nd = reg_ty(ctx, dst).slots();
    let mut out = zero_slots();
    oplib::unary(op, ty, &sa[..n as usize], &mut out[..nd as usize], mask).map_err(|e| support::op_err(ctx, e))?;
    support::scatter(ctx, dst, nd, &out, mask);
    Ok(Flow::Next)
}

/// An ALU error names the lanes that fault (e.g. the divide-by-zero lane),
/// not the whole active mask: re-run lane by lane (error path only). The
/// structured attrs carry the faulting lanes, the operation and the first
/// faulting lane's operand bits (W11-pin-message item 1).
#[cold]
fn narrow(
    ctx: &ExecCtx<'_>,
    e: oplib::OpError,
    mask: WarpMask,
    operation: String,
    operands: impl Fn(usize) -> Vec<u64>,
    mut f: impl FnMut(WarpMask) -> oplib::OpResult,
) -> crate::interp::ExecError {
    let mut bad = 0u32;
    for l in mask.lanes() {
        if f(WarpMask::lane(l)).is_err() {
            bad |= 1 << l;
        }
    }
    let mut err = support::op_err(ctx, e);
    if bad != 0 {
        err.lanes = WarpMask(bad);
        let lanes: Vec<u32> = (0..32).filter(|l| bad >> l & 1 == 1).collect();
        err.attrs.insert("operands".into(), serde_json::json!(operands(lanes[0] as usize)));
        err.attrs.insert("faulting_lanes".into(), serde_json::json!(lanes));
    }
    err.attrs.insert("operation".into(), serde_json::json!(operation));
    err
}

#[inline]
pub fn binary(ctx: &mut ExecCtx<'_>, op: BinOp, ty: Ty, dst: Reg, a: Operand, b: Operand) -> HResult {
    active_or_next!(ctx);
    let n = ty.slots();
    let mask = ctx.warp.active;
    if n == 1 && reg_ty(ctx, dst).slots() == 1 {
        let sa = [ctx.read_slot(a, 0)];
        let sb = [ctx.read_slot(b, 0)];
        let mut out = [[0u64; 32]];
        if let Err(e) = oplib::binary(op, ty, &sa, &sb, &mut out, mask) {
            let mut tmp = [[0u64; 32]];
            return Err(narrow(ctx, e, mask, format!("{op:?}.{ty}"), |l| vec![sa[0][l], sb[0][l]], |m| oplib::binary(op, ty, &sa, &sb, &mut tmp, m)));
        }
        let s = ctx.slot(dst);
        support::write_masked(ctx.warp.regs.get_mut(s), &out[0], mask);
        return Ok(Flow::Next);
    }
    let mut sa = zero_slots();
    let mut sb = zero_slots();
    support::gather(ctx, a, n, &mut sa);
    support::gather(ctx, b, n, &mut sb);
    let mut out = zero_slots();
    if let Err(e) = oplib::binary(op, ty, &sa[..n as usize], &sb[..n as usize], &mut out[..n as usize], mask) {
        let mut tmp = zero_slots();
        return Err(narrow(ctx, e, mask, format!("{op:?}.{ty}"), |l| sa[..n as usize].iter().chain(&sb[..n as usize]).map(|s| s[l]).collect(), |m| oplib::binary(op, ty, &sa[..n as usize], &sb[..n as usize], &mut tmp[..n as usize], m)));
    }
    support::scatter(ctx, dst, reg_ty(ctx, dst).slots().min(n), &out, mask);
    Ok(Flow::Next)
}

#[inline]
pub fn ternary(ctx: &mut ExecCtx<'_>, op: TerOp, ty: Ty, dst: Reg, a: Operand, b: Operand, c: Operand) -> HResult {
    active_or_next!(ctx);
    let n = ty.slots();
    let mask = ctx.warp.active;
    if n == 1 && reg_ty(ctx, dst).slots() == 1 {
        let sa = [ctx.read_slot(a, 0)];
        let sb = [ctx.read_slot(b, 0)];
        let sc = [ctx.read_slot(c, 0)];
        let mut out = [[0u64; 32]];
        if let Err(e) = oplib::ternary(op, ty, &sa, &sb, &sc, &mut out, mask) {
            let mut tmp = [[0u64; 32]];
            return Err(narrow(ctx, e, mask, format!("{op:?}.{ty}"), |l| vec![sa[0][l], sb[0][l], sc[0][l]], |m| oplib::ternary(op, ty, &sa, &sb, &sc, &mut tmp, m)));
        }
        let s = ctx.slot(dst);
        support::write_masked(ctx.warp.regs.get_mut(s), &out[0], mask);
        return Ok(Flow::Next);
    }
    let mut sa = zero_slots();
    let mut sb = zero_slots();
    let mut sc = zero_slots();
    support::gather(ctx, a, n, &mut sa);
    support::gather(ctx, b, n, &mut sb);
    support::gather(ctx, c, n, &mut sc);
    let mut out = zero_slots();
    let k = n as usize;
    if let Err(e) = oplib::ternary(op, ty, &sa[..k], &sb[..k], &sc[..k], &mut out[..k], mask) {
        let mut tmp = zero_slots();
        return Err(narrow(ctx, e, mask, format!("{op:?}.{ty}"), |l| sa[..k].iter().chain(&sb[..k]).chain(&sc[..k]).map(|s| s[l]).collect(), |m| oplib::ternary(op, ty, &sa[..k], &sb[..k], &sc[..k], &mut tmp[..k], m)));
    }
    support::scatter(ctx, dst, reg_ty(ctx, dst).slots().min(n), &out, mask);
    Ok(Flow::Next)
}

/// Write a predicate result: 1 in `yes`, 0 in `mask & !yes`.
#[inline]
pub fn write_pred(ctx: &mut ExecCtx<'_>, dst: Reg, yes: WarpMask, mask: WarpMask) {
    let s = ctx.slot(dst);
    let v = ctx.warp.regs.get_mut(s);
    let (y, m) = (yes.bits(), mask.bits());
    for (l, d) in v.iter_mut().enumerate() {
        if (m >> l) & 1 == 1 {
            *d = ((y >> l) & 1) as u64;
        }
    }
}

#[inline]
pub fn compare(ctx: &mut ExecCtx<'_>, op: CmpOp, ty: Ty, dst: Reg, a: Operand, b: Operand) -> HResult {
    active_or_next!(ctx);
    let n = ty.slots();
    let mask = ctx.warp.active;
    if n == 1 {
        let sa = [ctx.read_slot(a, 0)];
        let sb = [ctx.read_slot(b, 0)];
        let yes = oplib::compare(op, ty, &sa, &sb, mask).map_err(|e| support::op_err(ctx, e))?;
        write_pred(ctx, dst, yes.and(mask), mask);
        return Ok(Flow::Next);
    }
    let mut sa = zero_slots();
    let mut sb = zero_slots();
    support::gather(ctx, a, n, &mut sa);
    support::gather(ctx, b, n, &mut sb);
    let yes = oplib::compare(op, ty, &sa[..n as usize], &sb[..n as usize], mask).map_err(|e| support::op_err(ctx, e))?;
    write_pred(ctx, dst, yes.and(mask), mask);
    Ok(Flow::Next)
}

#[inline]
pub fn select(ctx: &mut ExecCtx<'_>, ty: Ty, dst: Reg, cond: Operand, a: Operand, b: Operand) -> HResult {
    active_or_next!(ctx);
    let mask = ctx.warp.active;
    let yes = super::control::cond_mask(ctx, cond, mask);
    let n = ty.slots().min(reg_ty(ctx, dst).slots());
    let base = ctx.slot(dst);
    for i in 0..n {
        let va = ctx.read_slot(a, i);
        let vb = ctx.read_slot(b, i);
        let d = ctx.warp.regs.get_mut(base + i);
        for l in mask.lanes() {
            d[l] = if yes.contains(l) { va[l] } else { vb[l] };
        }
    }
    Ok(Flow::Next)
}

#[inline]
pub fn cast(ctx: &mut ExecCtx<'_>, from: Ty, to: Ty, dst: Reg, src: Operand, rnd: Rounding, sat: bool) -> HResult {
    active_or_next!(ctx);
    let mask = ctx.warp.active;
    let n1 = from.slots();
    let n2 = reg_ty(ctx, dst).slots();
    let mut s = zero_slots();
    support::gather(ctx, src, n1, &mut s);
    let mut out = zero_slots();
    oplib::cast(from, to, rnd, sat, &s[..n1 as usize], &mut out[..n2 as usize], mask).map_err(|e| support::op_err(ctx, e))?;
    support::scatter(ctx, dst, n2, &out, mask);
    Ok(Flow::Next)
}

/// Scratch buffers reused across `Ptx` executions.
#[derive(Default)]
pub struct PtxScratch {
    pub dsts: Vec<WarpValue<u64>>,
    pub srcs: Vec<WarpValue<u64>>,
    pub dst_tys: Vec<Ty>,
    pub src_tys: Vec<Ty>,
}

thread_local! {
    static PTX_SCRATCH: std::cell::RefCell<PtxScratch> = std::cell::RefCell::new(PtxScratch::default());
}

#[inline]
pub fn ptx(ctx: &mut ExecCtx<'_>, op: OpId, dsts: &[Reg], srcs: &[Operand], pred: Option<Operand>, keep_dst: bool) -> HResult {
    active_or_next!(ctx);
    // Multi-signature ops pick the resolution of this site's operand types.
    let variants = &ctx.loaded.op_variants[op.0 as usize];
    let (f, error) = if variants.is_empty() {
        (ctx.loaded.ops[op.0 as usize], ctx.loaded.op_errors[op.0 as usize].as_ref())
    } else {
        let v = variants
            .iter()
            .find(|v| v.dst_tys.iter().copied().eq(dsts.iter().map(|&d| reg_ty(ctx, d))) && v.src_tys.iter().copied().eq(srcs.iter().map(|&s| operand_ty(ctx, s))))
            .expect("every Ptx site's signature is resolved at load");
        (v.f, v.error.as_ref())
    };
    if let Some(msg) = error {
        let key = &ctx.program.ops[op.0 as usize];
        return Err(ctx.error(ExecErrorKind::Unsupported, format!("{} {:?}: {msg}", key.name, key.mods)));
    }
    let active = ctx.warp.active;
    let exec = match pred {
        Some(p) => super::control::cond_mask(ctx, p, active),
        None => active,
    };
    match ctx.loaded.op_effects[op.0 as usize] {
        crate::interp::OpEffect::None => {}
        effect => op_effect(ctx, effect, srcs, exec)?,
    }
    PTX_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        let sc = &mut *sc;
        sc.dsts.clear();
        sc.srcs.clear();
        sc.dst_tys.clear();
        sc.src_tys.clear();
        for &d in dsts {
            let ty = reg_ty(ctx, d);
            sc.dst_tys.push(ty);
            for i in 0..ty.slots() {
                sc.dsts.push(*ctx.reg_slot(d, i));
            }
        }
        for &s in srcs {
            let ty = operand_ty(ctx, s);
            sc.src_tys.push(ty);
            for i in 0..ty.slots() {
                sc.srcs.push(ctx.read_slot(s, i));
            }
        }
        if !exec.is_empty() {
            let mut io = PtxIo { dsts: &mut sc.dsts, dst_tys: &sc.dst_tys, srcs: &sc.srcs, src_tys: &sc.src_tys, mask: exec };
            if let Err(e) = f.call(&mut io) {
                let key = &ctx.program.ops[op.0 as usize];
                let operation = std::iter::once(key.name.as_str()).chain(key.mods.iter().map(String::as_str)).collect::<Vec<_>>().join(".");
                let srcs = &sc.srcs;
                let mut tmp = sc.dsts.clone();
                return Err(narrow(ctx, e, exec, operation, |l| srcs.iter().map(|s| s[l]).collect(), |m| {
                    f.call(&mut PtxIo { dsts: &mut tmp, dst_tys: &sc.dst_tys, srcs: &sc.srcs, src_tys: &sc.src_tys, mask: m })
                }));
            }
        }
        let zero_lanes = if keep_dst { WarpMask::NONE } else { active.and_not(exec) };
        let mut k = 0usize;
        for &d in dsts {
            let ty = reg_ty(ctx, d);
            let base = ctx.slot(d);
            for i in 0..ty.slots() {
                let v = sc.dsts[k];
                k += 1;
                let r = ctx.warp.regs.get_mut(base + i);
                for l in exec.lanes() {
                    r[l] = v[l];
                }
                for l in zero_lanes.lanes() {
                    r[l] = 0;
                }
            }
        }
        Ok(Flow::Next)
    })
}

fn reg_index(ctx: &ExecCtx<'_>, base: Reg, len: u32, idx: Operand, lane: usize) -> Result<Reg, crate::interp::ExecError> {
    let i = lane_int(ctx, idx, lane);
    if i < 0 || i >= len as i64 {
        return Err(support::err(
            ctx,
            ExecErrorKind::OutOfBounds,
            WarpMask::lane(lane),
            format!("register-array index {i} out of bounds [0, {len})"),
        ));
    }
    Ok(Reg(base.0 + i as u32))
}

#[inline]
pub fn load_reg_indexed(ctx: &mut ExecCtx<'_>, dst: Reg, base: Reg, len: u32, idx: Operand) -> HResult {
    active_or_next!(ctx);
    let n = reg_ty(ctx, dst).slots();
    for l in ctx.warp.active.lanes() {
        let r = reg_index(ctx, base, len, idx, l)?;
        for i in 0..n.min(reg_ty(ctx, r).slots()) {
            let v = ctx.reg_slot(r, i)[l];
            let s = ctx.slot(dst) + i;
            ctx.warp.regs.get_mut(s)[l] = v;
        }
    }
    Ok(Flow::Next)
}

#[inline]
pub fn store_reg_indexed(ctx: &mut ExecCtx<'_>, base: Reg, len: u32, idx: Operand, value: Operand) -> HResult {
    active_or_next!(ctx);
    for l in ctx.warp.active.lanes() {
        let r = reg_index(ctx, base, len, idx, l)?;
        let ty = reg_ty(ctx, r);
        let s = ctx.slot(r);
        for i in 0..ty.slots() {
            let mut v = support::lane_slot(ctx, value, i, l);
            let rem = ty.bits().saturating_sub(i * 64);
            if rem < 64 {
                v &= (1u64 << rem) - 1;
            }
            ctx.warp.regs.get_mut(s + i)[l] = v;
        }
    }
    let _ = (extend, lane_val);
    Ok(Flow::Next)
}

/// Engine effects of hint ops (CONTRACT_REQUESTS W4-9).
#[cold]
fn op_effect(ctx: &mut ExecCtx<'_>, effect: crate::interp::OpEffect, srcs: &[Operand], exec: WarpMask) -> Result<(), crate::interp::ExecError> {
    use crate::interp::OpEffect;
    use crate::sync::async_group::{self, Domain};
    use crate::sync::SyncCmd;
    match effect {
        OpEffect::None => Ok(()),
        OpEffect::ValidGlobalAddr => {
            let Some(&a) = srcs.first() else { return Ok(()) };
            for l in exec.lanes() {
                let v = support::lane_val(ctx, a, l);
                support::resolve(ctx, crate::program::AddrSpace::Global, v, l, 1)?;
            }
            Ok(())
        }
        OpEffect::BulkGroupOp => {
            let mut cmds = Vec::new();
            for l in exec.lanes() {
                let gres = super::async_copy::group_res(ctx, l, Domain::Bulk);
                let c = SyncCmd::AsyncGroup(async_group::Cmd::Issue);
                support::step(ctx, gres, c)?;
                cmds.push((gres, c));
                let op = super::async_copy::issue_async(
                    ctx,
                    WarpMask::lane(l),
                    super::async_copy::Issue {
                        kind: crate::sync::AsyncKind::Bulk,
                        class: crate::observe::AsyncClass::Copy,
                        proxy: crate::program::Proxy::Async,
                        payload: crate::sync::Payload::None,
                        signals: Vec::new(),
                        after: Vec::new(),
                        targets: Vec::new(),
                        queue: true,
                        fill_pattern: Vec::new(),
                        tf32_round: false,
                        report: None,
                        lut_b: None,
                        strong: None,
                        restricted: false,
                        preds: None,
                    },
                );
                ctx.aux.groups.issue(gres, op);
            }
            support::protocol(ctx, exec, cmds, support::ProtoExtra::default());
            Ok(())
        }
    }
}
