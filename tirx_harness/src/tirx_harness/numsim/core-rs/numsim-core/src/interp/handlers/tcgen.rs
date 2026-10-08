//! tcgen05: TMEM lifecycle (alloc/dealloc/relinquish), ld/st/wait, mma,
//! commit; plus the provisional tile op.
//!
//! Kernel-wide `.cta_group` uniformity: lifecycle commands are checked
//! inside `SyncTable::step`; work commands (mma/commit) carry
//! `(TcgenKernel, TcgenGroup(g))` in the same `step_all` batch.
//! `cta_group::2` lifecycle ops are a two-CTA rendezvous (one warp of each
//! peer CTA, any warp index) resolved here before `step`.

use super::async_copy::{issue_async, mbar_in_rank, mbar_issue, mbar_res, ranks_of, Issue};
use super::HResult;
use crate::arena::{addr, ByteSpan};
use crate::interp::support::{self, lane_val, uniform_over, write_lane_bytes, Accesses, AccessSpec, ProtoExtra};
use crate::interp::{ExecCtx, ExecError, ExecErrorKind, Flow};
use crate::observe::{AccessKind, Actor, AsyncClass, AsyncTarget, Collective, LaneSpan, PublishTarget, Side, SyncKind};
use crate::program::*;
use crate::sync::completion::{AsyncKind, Payload, TcgenMmaPayload};
use crate::sync::{tcgen, AsyncId, Completion, Outcome, ResourceId, Step, SyncCmd};
use crate::value::WarpMask;

fn group(cta_group: u8) -> u8 {
    if cta_group == 0 {
        1
    } else {
        cta_group
    }
}

/// Lifecycle resource of the executing CTA's pair and its `Who`.
fn lifecycle(ctx: &ExecCtx<'_>, cta_group: u8) -> (ResourceId, tcgen::Who) {
    let rank = ctx.cta.rank_in_cluster;
    let who = if group(cta_group) == 2 { tcgen::Who::Pair } else { tcgen::Who::One((rank & 1) as u8) };
    (ResourceId::TcgenLifecycle { cluster: ctx.cta.cluster, pair_rank: (rank >> 1) as u8 }, who)
}

/// Global id of the even CTA of the executing CTA's pair (rendezvous key).
fn pair_cta(ctx: &ExecCtx<'_>) -> crate::observe::CtaId {
    let even = (ctx.cta.rank_in_cluster & !1) as usize;
    ctx.cta.cluster_ctas.get(even).copied().unwrap_or(ctx.cta.id)
}

fn full_warp(ctx: &ExecCtx<'_>, what: &str) -> Result<(), ExecError> {
    if ctx.warp.active != ctx.warp.live {
        return Err(support::err(ctx, ExecErrorKind::Divergence, ctx.warp.active, format!("{what}.sync.aligned with divergent lanes")));
    }
    Ok(())
}

/// Write the allocated taddr to shared memory at `dst` (every active lane
/// writes the same word).
fn write_taddr(ctx: &mut ExecCtx<'_>, dst: Operand, base: u32) -> Result<(), ExecError> {
    let l = ctx.warp.active.first().unwrap_or(0);
    let a = lane_val(ctx, dst, l);
    let space = if a >> 32 == 0 { AddrSpace::Shared } else { AddrSpace::Generic };
    let loc = support::resolve(ctx, space, a, l, 4)?;
    let taddr = addr::tmem_addr(0, base);
    support::mem_write(ctx, loc, l, &taddr.to_le_bytes())?;
    let mut acc = Accesses::default();
    if ctx.observing {
        acc.push(loc, l as u8, 4);
    }
    let sp = support::spec(ctx, AccessKind::Write, Sem::Weak, Scope::Cta, Proxy::Generic);
    support::emit(ctx, sp, &mut acc);
    Ok(())
}

/// Resume token tag of a `cta_group::2` rendezvous.
const PAIR_TAG: u64 = 1 << 61;

/// Outcome of a `cta_group::2` rendezvous attempt.
enum Pair {
    /// Wait for the peer (token stored in `resume`).
    Wait,
    /// This warp commits the pair command now.
    Commit(u64),
    /// The peer committed instance `epoch`; result available.
    Done(u64),
}

fn pair_rendezvous(ctx: &mut ExecCtx<'_>, kind: u8) -> Pair {
    let pair = pair_cta(ctx);
    let me = (ctx.cta.id, ctx.warp.id);
    let token = ctx.warp.resume.filter(|t| t & PAIR_TAG != 0).map(|t| t & !PAIR_TAG);
    let rv = ctx.aux.tcgen_pairs.entry((pair, kind)).or_default();
    if let Some(e) = token {
        if rv.results.contains_key(&e) {
            return Pair::Done(e);
        }
    }
    match rv.first {
        None => {
            rv.first = Some(me);
            let e = rv.epoch;
            ctx.warp.resume = Some(PAIR_TAG | e);
            Pair::Wait
        }
        Some(f) if f == me => Pair::Wait,
        Some(f) if f.0 == me.0 => Pair::Wait,
        Some(_) => Pair::Commit(rv.epoch),
    }
}

fn pair_committed(ctx: &mut ExecCtx<'_>, kind: u8, epoch: u64, value: u32) {
    let pair = pair_cta(ctx);
    let rv = ctx.aux.tcgen_pairs.entry((pair, kind)).or_default();
    rv.results.insert(epoch, value);
    rv.epoch = epoch + 1;
    rv.first = None;
    ctx.warp.resume = None;
}

fn pair_collective(ctx: &ExecCtx<'_>, epoch: u64, kind: u8) -> Collective {
    let pair = pair_cta(ctx);
    Collective { id: (1u64 << 63) | ((pair.0 as u64) << 24) | ((kind as u64) << 20) | epoch, participants: vec![ctx.warp.id] }
}

#[inline]
pub fn tcgen_alloc(ctx: &mut ExecCtx<'_>, dst: Operand, ncols: Operand, cta_group: u8, exclusive: bool) -> HResult {
    active_or_next!(ctx);
    full_warp(ctx, "tcgen05.alloc")?;
    let active = ctx.warp.active;
    let columns = uniform_over(ctx, ncols, active)? as u32;
    let (res, who) = lifecycle(ctx, cta_group);
    let cmd = SyncCmd::Tcgen(tcgen::Cmd::Alloc { who, columns, exclusive });
    let mut collective = None;
    if group(cta_group) == 2 {
        match pair_rendezvous(ctx, 0) {
            Pair::Wait => return Ok(Flow::Blocked(res)),
            Pair::Done(e) => {
                let base = ctx.aux.tcgen_pairs[&(pair_cta(ctx), 0)].results[&e];
                ctx.warp.resume = None;
                let c = pair_collective(ctx, e, 0);
                support::protocol(ctx, active, vec![(res, cmd)], ProtoExtra { collective: Some(c), ..Default::default() });
                write_taddr(ctx, dst, base)?;
                return Ok(Flow::Next);
            }
            Pair::Commit(e) => collective = Some((e, pair_collective(ctx, e, 0))),
        }
    }
    match support::step(ctx, res, cmd)? {
        Step::Blocked(r) => Ok(Flow::Blocked(r)),
        Step::Done(Outcome::Tcgen(tcgen::Outcome::Allocated { base })) => {
            if let Some((e, _)) = &collective {
                pair_committed(ctx, 0, *e, base);
            }
            let extra = ProtoExtra { collective: collective.map(|c| c.1), ..Default::default() };
            support::protocol(ctx, active, vec![(res, cmd)], extra);
            write_taddr(ctx, dst, base)?;
            Ok(Flow::Next)
        }
        Step::Done(o) => Err(ctx.error(ExecErrorKind::Internal, format!("tcgen05.alloc returned {o:?}"))),
    }
}


/// Dealloc / relinquish. `cta_group::2` is a two-CTA rendezvous: the first
/// peer warp blocks until the second commits for both (CONTRACT_REQUESTS.md
/// W2-3: `may_block`). Returns `Ok(None)` to block, `Ok(Some(committed_here))`.
fn lifecycle_collective(ctx: &mut ExecCtx<'_>, cta_group: u8, kind: u8, cmd: SyncCmd) -> Result<Option<bool>, ExecError> {
    let active = ctx.warp.active;
    let (res, _) = lifecycle(ctx, cta_group);
    if group(cta_group) == 2 {
        match pair_rendezvous(ctx, kind) {
            Pair::Wait => return Ok(None),
            Pair::Done(e) => {
                ctx.warp.resume = None;
                let c = pair_collective(ctx, e, kind);
                support::protocol(ctx, active, vec![(res, cmd)], ProtoExtra { collective: Some(c), ..Default::default() });
                return Ok(Some(false));
            }
            Pair::Commit(e) => {
                support::step(ctx, res, cmd)?;
                pair_committed(ctx, kind, e, 0);
                let c = pair_collective(ctx, e, kind);
                support::protocol(ctx, active, vec![(res, cmd)], ProtoExtra { collective: Some(c), ..Default::default() });
                return Ok(Some(true));
            }
        }
    }
    support::step(ctx, res, cmd)?;
    support::protocol(ctx, active, vec![(res, cmd)], ProtoExtra::default());
    Ok(Some(true))
}

#[inline]
pub fn tcgen_dealloc(ctx: &mut ExecCtx<'_>, taddr: Operand, ncols: Operand, cta_group: u8, exclusive: bool) -> HResult {
    active_or_next!(ctx);
    full_warp(ctx, "tcgen05.dealloc")?;
    let active = ctx.warp.active;
    let columns = uniform_over(ctx, ncols, active)? as u32;
    let t = uniform_over(ctx, taddr, active)? as u32;
    let (_, who) = lifecycle(ctx, cta_group);
    let cmd = SyncCmd::Tcgen(tcgen::Cmd::Dealloc { who, taddr: t, columns, exclusive });
    let Some(committed) = lifecycle_collective(ctx, cta_group, 1, cmd)? else {
        return Ok(Flow::Blocked(lifecycle(ctx, cta_group).0));
    };
    if committed {
        // Deallocated columns hold undefined data.
        let (_, col) = addr::tmem_decode(t);
        let mut tmems = vec![ctx.cta.tmem];
        if group(cta_group) == 2 {
            let peer = (ctx.cta.rank_in_cluster ^ 1) as usize;
            if let Some(&a) = ctx.cta.cluster_tmem.get(peer) {
                tmems.push(a);
            }
        }
        tmems.dedup();
        for alloc in tmems {
            if ctx.arena.get(alloc).size == 0 {
                continue;
            }
            let v = support::whole(ctx.arena, alloc);
            let spans: Vec<ByteSpan> = (0..addr::TMEM_LANES)
                .map(|ln| ByteSpan::new(addr::tmem_byte_offset(ln, col), columns.min(addr::TMEM_COLS - col) as u64 * 4))
                .collect();
            let _ = ctx.arena.invalidate(v, &spans);
        }
    }
    Ok(Flow::Next)
}

#[inline]
pub fn tcgen_relinquish(ctx: &mut ExecCtx<'_>, cta_group: u8) -> HResult {
    active_or_next!(ctx);
    full_warp(ctx, "tcgen05.relinquish_alloc_permit")?;
    let (_, who) = lifecycle(ctx, cta_group);
    match lifecycle_collective(ctx, cta_group, 2, SyncCmd::Tcgen(tcgen::Cmd::Relinquish { who }))? {
        Some(_) => Ok(Flow::Next),
        None => Ok(Flow::Blocked(lifecycle(ctx, cta_group).0)),
    }
}

fn work_res(ctx: &ExecCtx<'_>, lane: usize) -> ResourceId {
    ResourceId::TcgenWork { warp: ctx.warp.id, lane: lane as u8 }
}

#[inline]
pub fn tcgen_commit(
    ctx: &mut ExecCtx<'_>,
    mbar: Operand,
    space: AddrSpace,
    cta_group: u8,
    multicast: Option<Operand>,
    sync_restrict: bool,
    multicast_width: Option<u8>,
) -> HResult {
    active_or_next!(ctx);
    if multicast_width.is_some() {
        return Err(support::unsupported(ctx, "tcgen05.commit .multicast::cluster.width"));
    }
    let active = ctx.warp.active;
    let g = group(cta_group);
    let mut all = Vec::new();
    for l in active.lanes() {
        let batch = [
            (ResourceId::TcgenKernel, SyncCmd::TcgenGroup(g)),
            (work_res(ctx, l), SyncCmd::TcgenWork(tcgen::WorkCmd::Commit)),
        ];
        support::step_all(ctx, &batch)?;
        all.extend(batch);
        let tracked = ctx.aux.tcgen_uncommitted.remove(&(ctx.warp.id, l as u8)).unwrap_or_default();
        let res = mbar_res(ctx, space, lane_val(ctx, mbar, l), l)?;
        if sync_restrict && multicast.is_none() {
            if let ResourceId::Mbarrier { cta, .. } = res {
                if cta != ctx.cta.id {
                    return Err(support::err(
                        ctx,
                        ExecErrorKind::BadAddress,
                        WarpMask::lane(l),
                        "tcgen05.commit.sync_restrict names another CTA's mbarrier",
                    ));
                }
            }
        }
        let targets_res: Vec<ResourceId> = match multicast {
            Some(m) => ranks_of(ctx, lane_val(ctx, m, l)).into_iter().filter_map(|r| mbar_in_rank(ctx, res, r)).collect(),
            None => vec![res],
        };
        let mut signals = Vec::new();
        let mut targets = Vec::new();
        for r in targets_res {
            let gen = mbar_issue(ctx, r)?;
            all.push((r, SyncCmd::Mbarrier(crate::sync::mbarrier::Cmd::Issue)));
            signals.push(Completion::MbarArrive { res: r, gen, count: 1 });
            targets.push(AsyncTarget { res: r, bytes: 0, arrivals: 1 });
        }
        issue_async(
            ctx,
            WarpMask::lane(l),
            Issue {
                // No dedicated AsyncKind for commit (CONTRACT_REQUESTS.md).
                kind: AsyncKind::TcgenMma,
                class: AsyncClass::TcgenCommit,
                proxy: Proxy::Tcgen,
                payload: Payload::None,
                signals,
                after: tracked,
                targets,
                queue: true,
                fill_pattern: Vec::new(),
                tf32_round: false,
            },
        );
    }
    support::protocol(ctx, active, all, ProtoExtra::default());
    Ok(Flow::Next)
}

/// Check a 32x32b access by this warp: returns the TMEM base lane/column.
fn ldst_base(ctx: &ExecCtx<'_>, taddr: Operand, shape: TcShape, flags: bool) -> Result<(u32, u32), ExecError> {
    if shape != TcShape::S32x32b || flags {
        return Err(support::unsupported(ctx, &format!("tcgen05.ld/st shape {shape:?} (or pack/red/spcompress)")));
    }
    full_warp(ctx, "tcgen05.ld/st")?;
    let t = uniform_over(ctx, taddr, ctx.warp.active)? as u32;
    let (lane, col) = addr::tmem_decode(t);
    let sp = ctx.warp.warp_in_cta % 4;
    if lane != 32 * sp {
        return Err(ctx.error(
            ExecErrorKind::BadAddress,
            format!("warp {} (subpartition {sp}) cannot access TMEM lanes from {lane}", ctx.warp.warp_in_cta),
        ));
    }
    Ok((lane, col))
}

/// Register spans (in the warp's metadata `Reg` allocation) of `regs` in `lanes`.
fn reg_spans(_ctx: &ExecCtx<'_>, regs: &[Reg], lanes: WarpMask) -> Vec<LaneSpan> {
    let mut v = Vec::new();
    for l in lanes.lanes() {
        for r in regs {
            v.push(LaneSpan { lane: l as u8, span: ByteSpan::new((r.0 as u64 * 32 + l as u64) * 8, 8) });
        }
    }
    v
}

fn emit_async_spans(ctx: &mut ExecCtx<'_>, op: AsyncId, side: Side, kind: AccessKind, alloc: crate::arena::AllocId, spans: Vec<LaneSpan>) {
    if !ctx.observing || spans.is_empty() {
        return;
    }
    let mut acc = Accesses::default();
    acc.items = spans.into_iter().map(|s| (alloc, None, s)).collect();
    let spec = AccessSpec {
        actor: Actor::Async { op, side },
        site: ctx.site(),
        kind,
        sem: Sem::Weak,
        scope: Scope::Cta,
        atomic: false,
        returns_value: false,
        proxy: Proxy::Tcgen,
    };
    support::emit(ctx, spec, &mut acc);
}

#[inline]
pub fn tcgen_ld(ctx: &mut ExecCtx<'_>, args: &TcgenLdArgs) -> HResult {
    active_or_next!(ctx);
    let flags = args.pack || args.red.is_some() || args.spcompress;
    let (lane0, col) = ldst_base(ctx, args.taddr, args.shape, flags)?;
    let active = ctx.warp.active;
    let num = args.num as u32;
    if col + num > addr::TMEM_COLS {
        return Err(ctx.error(ExecErrorKind::OutOfBounds, "tcgen05.ld beyond TMEM columns"));
    }
    let op = issue_async(
        ctx,
        active,
        Issue {
            kind: AsyncKind::TcgenMma,
            class: AsyncClass::TcgenLd,
            proxy: Proxy::Tcgen,
            payload: Payload::None,
            signals: Vec::new(),
            after: Vec::new(),
            targets: Vec::new(),
            queue: false,
            fill_pattern: Vec::new(),
            tf32_round: false,
        },
    );
    let tmem = ctx.cta.tmem;
    let mut tspans = Vec::new();
    let mut buf = vec![0u8; num as usize * 4];
    for t in active.lanes() {
        let off = addr::tmem_byte_offset(lane0 + t as u32, col);
        let loc = support::Loc { alloc: tmem, offset: off, window: None, remote: None };
        support::mem_read(ctx, loc, t, &mut buf)?;
        let mut pos = 0usize;
        for &d in &args.dsts {
            let n = support::reg_ty(ctx, d).mem_bytes() as usize;
            let end = (pos + n).min(buf.len());
            write_lane_bytes(ctx, d, t, &buf[pos..end]);
            pos += n;
        }
        tspans.push(LaneSpan { lane: t as u8, span: ByteSpan::new(off, num as u64 * 4) });
    }
    emit_async_spans(ctx, op, Side::Read, AccessKind::Read, tmem, tspans);
    let rs = reg_spans(ctx, &args.dsts, active);
    if let Some(&ra) = ctx.aux.reg_allocs.get(ctx.warp.id.0 as usize) {
        emit_async_spans(ctx, op, Side::Write, AccessKind::Write, ra, rs);
    }
    let cmds: Vec<_> = active.lanes().map(|l| (work_res(ctx, l), SyncCmd::TcgenWork(tcgen::WorkCmd::Load))).collect();
    support::step_all(ctx, &cmds)?;
    ctx.aux.tcgen_ldst.entry(ctx.warp.id).or_default().push((op, active, false));
    support::protocol(ctx, active, cmds, ProtoExtra::default());
    Ok(Flow::Next)
}

#[inline]
pub fn tcgen_st(ctx: &mut ExecCtx<'_>, args: &TcgenStArgs) -> HResult {
    active_or_next!(ctx);
    let (lane0, col) = ldst_base(ctx, args.taddr, args.shape, args.unpack)?;
    let active = ctx.warp.active;
    let num = args.num as u32;
    if col + num > addr::TMEM_COLS {
        return Err(ctx.error(ExecErrorKind::OutOfBounds, "tcgen05.st beyond TMEM columns"));
    }
    let op = issue_async(
        ctx,
        active,
        Issue {
            kind: AsyncKind::TcgenMma,
            class: AsyncClass::TcgenSt,
            proxy: Proxy::Tcgen,
            payload: Payload::None,
            signals: Vec::new(),
            after: Vec::new(),
            targets: Vec::new(),
            queue: false,
            fill_pattern: Vec::new(),
            tf32_round: false,
        },
    );
    let tmem = ctx.cta.tmem;
    let mut tspans = Vec::new();
    for t in active.lanes() {
        let mut buf = vec![0u8; num as usize * 4];
        let mut pos = 0usize;
        for &s in &args.srcs {
            let ty = support::operand_ty(ctx, s);
            let n = ty.mem_bytes() as usize;
            let mut tmp = [0u8; 32];
            support::lane_bytes(ctx, s, ty, t, &mut tmp);
            let end = (pos + n).min(buf.len());
            if pos < end {
                buf[pos..end].copy_from_slice(&tmp[..end - pos]);
            }
            pos += n;
        }
        let off = addr::tmem_byte_offset(lane0 + t as u32, col);
        let loc = support::Loc { alloc: tmem, offset: off, window: None, remote: None };
        support::mem_write(ctx, loc, t, &buf)?;
        tspans.push(LaneSpan { lane: t as u8, span: ByteSpan::new(off, num as u64 * 4) });
    }
    let regs: Vec<Reg> = args.srcs.iter().filter_map(|s| if let Operand::Reg(r) = s { Some(*r) } else { None }).collect();
    let rs = reg_spans(ctx, &regs, active);
    if let Some(&ra) = ctx.aux.reg_allocs.get(ctx.warp.id.0 as usize) {
        emit_async_spans(ctx, op, Side::Read, AccessKind::Read, ra, rs);
    }
    emit_async_spans(ctx, op, Side::Write, AccessKind::Write, tmem, tspans);
    let cmds: Vec<_> = active.lanes().map(|l| (work_res(ctx, l), SyncCmd::TcgenWork(tcgen::WorkCmd::Store))).collect();
    support::step_all(ctx, &cmds)?;
    ctx.aux.tcgen_ldst.entry(ctx.warp.id).or_default().push((op, active, true));
    support::protocol(ctx, active, cmds, ProtoExtra::default());
    Ok(Flow::Next)
}

#[inline]
pub fn tcgen_wait(ctx: &mut ExecCtx<'_>, st: bool) -> HResult {
    active_or_next!(ctx);
    let active = ctx.warp.active;
    let w = if st { tcgen::WorkCmd::WaitSt } else { tcgen::WorkCmd::WaitLd };
    let cmds: Vec<_> = active.lanes().map(|l| (work_res(ctx, l), SyncCmd::TcgenWork(w))).collect();
    support::step_all(ctx, &cmds)?;
    support::protocol(ctx, active, cmds, ProtoExtra::default());
    let mut done = Vec::new();
    if let Some(v) = ctx.aux.tcgen_ldst.get_mut(&ctx.warp.id) {
        v.retain_mut(|(op, lanes, is_st)| {
            if *is_st != st || lanes.and(active).is_empty() {
                return true;
            }
            done.push((*op, lanes.and(active)));
            *lanes = lanes.and_not(active);
            !lanes.is_empty()
        });
    }
    for (op, lanes) in done {
        let target = PublishTarget::Warp { warp: ctx.warp.id, lanes };
        support::sync_event(ctx, lanes, SyncKind::AsyncComplete { op, milestone: Side::Write, target });
    }
    Ok(Flow::Next)
}

#[inline]
pub fn tcgen_cp(ctx: &mut ExecCtx<'_>, args: TcgenCpArgs) -> HResult {
    active_or_next!(ctx);
    let _ = args;
    // Needs the shared-memory matrix layout of the source descriptor
    // (oplib has no tcgen05.cp plan yet): fail closed.
    Err(support::unsupported(ctx, "tcgen05.cp"))
}

#[inline]
pub fn tcgen_mma(ctx: &mut ExecCtx<'_>, args: &TcgenMmaArgs) -> HResult {
    active_or_next!(ctx);
    if args.lut_b {
        // The LUT's TMEM address is not an operand of the contract form.
        return Err(support::unsupported(ctx, "tcgen05.mma .lut_b"));
    }
    let active = ctx.warp.active;
    let g = group(args.cta_group);
    let mut all = Vec::new();
    for l in active.lanes() {
        let v = |o: Operand| lane_val(ctx, o, l);
        let a = match args.a {
            TcA::Smem(o) | TcA::Tmem(o) => v(o),
        };
        // CTA index within the issuing group: 0 = this CTA (cta_group::1),
        // or [even, odd] CTA of the pair (cta_group::2), as oplib expects.
        let (smem, tmem) = if g == 2 {
            let even = (ctx.cta.rank_in_cluster & !1) as usize;
            let pick = |v: &Vec<crate::arena::AllocId>, own| -> Vec<crate::arena::AllocId> {
                (even..even + 2).map(|r| v.get(r).copied().unwrap_or(own)).collect()
            };
            (pick(&ctx.cta.cluster_smem, ctx.cta.smem), pick(&ctx.cta.cluster_tmem, ctx.cta.tmem))
        } else {
            (vec![ctx.cta.smem], vec![ctx.cta.tmem])
        };
        let payload = TcgenMmaPayload {
            args: args.clone(),
            d_taddr: v(args.d) as u32,
            a,
            b_desc: v(args.b_desc),
            idesc: v(args.idesc) as u32,
            enable_input_d: v(args.enable_input_d) != 0,
            scale_taddrs: args.block_scale.map(|(sa, sb, _)| (v(sa) as u32, v(sb) as u32)),
            scale_input_d: args.scale_input_d.map(|o| v(o) as u32),
            sparse_meta: args.sparse_meta.map(|o| v(o) as u32),
            disable_output_lane: args.disable_output_lane.iter().map(|&o| v(o) as u32).collect(),
            smem,
            tmem,
        };
        let batch = [
            (ResourceId::TcgenKernel, SyncCmd::TcgenGroup(g)),
            (work_res(ctx, l), SyncCmd::TcgenWork(tcgen::WorkCmd::Issue)),
        ];
        support::step_all(ctx, &batch)?;
        all.extend(batch);
        let after: Vec<AsyncId> = ctx.aux.tcgen_last.get(&ctx.cta.id).copied().into_iter().collect();
        let op = issue_async(
            ctx,
            WarpMask::lane(l),
            Issue {
                kind: AsyncKind::TcgenMma,
                class: AsyncClass::TcgenPipelined,
                proxy: Proxy::Tcgen,
                payload: Payload::TcgenMma(Box::new(payload)),
                signals: Vec::new(),
                after,
                targets: Vec::new(),
                queue: true,
                fill_pattern: Vec::new(),
                tf32_round: false,
            },
        );
        ctx.aux.tcgen_last.insert(ctx.cta.id, op);
        ctx.aux.tcgen_uncommitted.entry((ctx.warp.id, l as u8)).or_default().push(op);
    }
    support::protocol(ctx, active, all, ProtoExtra::default());
    Ok(Flow::Next)
}

#[inline]
pub fn tile(ctx: &mut ExecCtx<'_>, args: &TileArgs) -> HResult {
    active_or_next!(ctx);
    // Provisional instruction (contract decision 11): lowering is expected
    // to dispatch tile ops to PTX-level IR. Fail closed.
    Err(support::unsupported(ctx, &format!("tile op {:?}", args.op)))
}
