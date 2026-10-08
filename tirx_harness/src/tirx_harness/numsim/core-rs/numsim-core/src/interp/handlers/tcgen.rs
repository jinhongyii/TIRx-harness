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
use crate::oplib;
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
        return Err(support::err(ctx, ExecErrorKind::WarpCollectiveDivergence, ctx.warp.active, format!("{what}.sync.aligned with divergent lanes")));
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
    // The result store is ONE 32-bit write by the issuing warp (the alloc
    // is warp-collective): a warp-collective access (`ALL_LANES`), not lane
    // `l`'s (W5-12).
    let mut acc = Accesses::default();
    if ctx.observing {
        acc.push(loc, crate::observe::ALL_LANES, 4);
    }
    let sp = support::spec(ctx, AccessKind::Write, Sem::Weak, Scope::Cta, Proxy::Generic);
    support::emit(ctx, sp, &mut acc);
    // `tcgen05.alloc` is `.aligned` and warp-collective: the result write is
    // the warp's, so every lane is ordered after it (W5-12;
    // racecheck-semantics §3 row 3).
    let active = ctx.warp.active;
    support::sync_event(ctx, active, crate::observe::SyncKind::WarpSync { mask: active });
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
    let me = ctx.warp.id;
    let rv = ctx.aux.tcgen_pairs.entry((pair, kind)).or_default();
    rv.results.insert(epoch, value);
    // Both member warps of the instance (W6-4: every record lists the full
    // participant set).
    let mut members: Vec<crate::observe::WarpId> = rv.first.map(|f| f.1).into_iter().chain(std::iter::once(me)).collect();
    members.sort();
    members.dedup();
    rv.participants.insert(epoch, members);
    rv.epoch = epoch + 1;
    rv.first = None;
    ctx.warp.resume = None;
}

fn pair_collective(ctx: &ExecCtx<'_>, epoch: u64, kind: u8) -> Collective {
    let pair = pair_cta(ctx);
    let participants = ctx
        .aux
        .tcgen_pairs
        .get(&(pair, kind))
        .and_then(|rv| rv.participants.get(&epoch).cloned())
        .unwrap_or_else(|| vec![ctx.warp.id]);
    Collective { id: (1u64 << 63) | ((pair.0 as u64) << 24) | ((kind as u64) << 20) | epoch, participants }
}

#[inline]
pub fn tcgen_alloc(ctx: &mut ExecCtx<'_>, dst: Operand, ncols: Operand, cta_group: u8, exclusive: bool) -> HResult {
    active_or_next!(ctx);
    full_warp(ctx, "tcgen05.alloc")?;
    let active = ctx.warp.active;
    // `tcgen05.alloc.sync.aligned` orders the warp's lanes
    // (racecheck-semantics §3 row 3), on every attempt that runs it.
    if ctx.warp.resume.is_none() {
        support::sync_event(ctx, active, crate::observe::SyncKind::WarpSync { mask: active });
    }
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
            let collective = match collective {
                Some((e, _)) => {
                    pair_committed(ctx, 0, e, base);
                    Some(pair_collective(ctx, e, 0))
                }
                None => None,
            };
            let extra = ProtoExtra { collective, ..Default::default() };
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

fn active_lanes(ctx: &ExecCtx<'_>) -> WarpMask {
    ctx.warp.active
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
    // `.multicast::cluster::16b/32b`: the width of the CTA-mask operand.
    let width_mask: u64 = match multicast_width {
        Some(w) if w < 64 => (1u64 << w) - 1,
        _ => u64::MAX,
    };
    let active = ctx.warp.active;
    let g = group(cta_group);
    let mut all = Vec::new();
    let mut issued_all = Vec::new();
    for l in active.lanes() {
        let batch = [
            (ResourceId::TcgenKernel, SyncCmd::TcgenGroup(g)),
            (work_res(ctx, l), SyncCmd::TcgenWork(tcgen::WorkCmd::Commit)),
        ];
        support::step_all(ctx, &batch)?;
        all.extend(batch);
        // PTX: the mbarrier tracks completion of ALL prior async tcgen05
        // operations of this thread, not only those since its last commit:
        // every one still in flight (a second commit right after a first
        // must not complete before the mma ops the first one tracks).
        // `.sync_restrict` (W5-10): only the shared-A reads are tracked;
        // the MMA itself stays for a later unrestricted commit.
        // Tracked = every op issued since the previous commit of this kind
        // (landed or not: the commit's preds name them) plus earlier
        // committed ops still in flight.
        // A multicast `.sync_restrict` commit is restricted too (W5-13).
        let restricted = sync_restrict;
        let key = (ctx.warp.id, l as u8);
        let fresh = if restricted {
            ctx.aux.tcgen_shared_reads.remove(&key).unwrap_or_default()
        } else {
            ctx.aux.tcgen_uncommitted.remove(&key).unwrap_or_default()
        };
        let inflight = if restricted { &mut ctx.aux.tcgen_inflight_shared } else { &mut ctx.aux.tcgen_inflight };
        let earlier = inflight.entry(key).or_default();
        {
            let live = &ctx.sync.async_ops;
            earlier.retain(|id| live.iter().any(|o| o.id == *id));
        }
        let mut tracked = earlier.clone();
        for id in fresh {
            if !tracked.contains(&id) {
                tracked.push(id);
            }
            earlier.push(id);
        }
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
            Some(m) => ranks_of(ctx, lane_val(ctx, m, l) & width_mask)?.into_iter().filter_map(|r| mbar_in_rank(ctx, res, r)).collect(),
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
        issued_all.extend(targets.iter().copied());
        issue_async(
            ctx,
            WarpMask::lane(l),
            Issue {
                kind: AsyncKind::TcgenCommit,
                class: AsyncClass::TcgenCommit,
                proxy: Proxy::Tcgen,
                payload: Payload::None,
                signals,
                after: tracked,
                targets,
                queue: true,
                fill_pattern: Vec::new(),
                tf32_round: false,
                report: None,
                lut_b: None,
                strong: None,
                restricted,
            },
        );
    }
    support::protocol(ctx, active, all, ProtoExtra { issued: issued_all, ..Default::default() });
    Ok(Flow::Next)
}

/// Effective taddr of an ld/st/cp: `taddr` plus the (warp-uniform) row and
/// column offsets, each field mod 2^16.
fn effective_taddr(ctx: &ExecCtx<'_>, taddr: Operand, row: Operand, col: Operand, lanes: WarpMask) -> Result<u32, ExecError> {
    let t = uniform_over(ctx, taddr, lanes)? as u32;
    let r = uniform_over(ctx, row, lanes)? as u32;
    let c = uniform_over(ctx, col, lanes)? as u32;
    let lane = ((t >> 16).wrapping_add(r)) & 0xffff;
    let column = ((t & 0xffff).wrapping_add(c)) & 0xffff;
    Ok((lane << 16) | column)
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
    let mut acc = Accesses {
        items: spans.into_iter().map(|s| (alloc, None, s)).collect(),
    };
    let spec = AccessSpec {
        actor: Actor::Async { op, side },
        site: ctx.site(),
        kind,
        sem: Sem::Weak,
        scope: Scope::Cta,
        atomic: false,
        returns_value: false,
        proxy: Proxy::Tcgen,
        // The TMEM address (or the register file): the only pointer operand.
        operand: 0,
    };
    support::emit(ctx, spec, &mut acc);
}

/// Issue the observer-only async op of a tcgen05.ld/st.
fn ldst_op(ctx: &mut ExecCtx<'_>, class: AsyncClass) -> AsyncId {
    let active = ctx.warp.active;
    issue_async(
        ctx,
        active,
        Issue {
            kind: AsyncKind::TcgenMma,
            class,
            proxy: Proxy::Tcgen,
            payload: Payload::None,
            signals: Vec::new(),
            after: Vec::new(),
            targets: Vec::new(),
            queue: false,
            fill_pattern: Vec::new(),
            tf32_round: false,
            report: None,
            lut_b: None,
            strong: None,
            restricted: false,
        },
    )
}

/// A tcgen05.ld/st piece must address a column of a live allocation
/// (PTX: TMEM outside `tcgen05.alloc` results is not addressable).
fn check_piece_live(ctx: &ExecCtx<'_>, p: &oplib::TcgenLdstPiece, t: usize, what: &str) -> Result<(), crate::interp::ExecError> {
    if support::tmem_live(ctx, p.column, 1) {
        return Ok(());
    }
    Err(support::err(
        ctx,
        ExecErrorKind::BadAddress,
        WarpMask::lane(t),
        format!("{what}: tmem column {} (lane {}) is not in a live tcgen05 allocation", p.column, p.tmem_lane),
    ))
}

/// Byte offset in the CTA's TMEM allocation of a plan piece.
#[inline]
fn piece_offset(p: &oplib::TcgenLdstPiece) -> u64 {
    addr::tmem_byte_offset(p.tmem_lane, p.column) + p.cell_byte as u64
}

/// Byte images of the live, fully valid cell runs of a `tcgen05.ld/st`
/// map (W4-16), read once per op. `locate` maps a piece to its image byte.
struct RunImages<'m> {
    runs: &'m [oplib::TcgenCellRun],
    /// Image start per run (`None`: that run takes the per-piece path).
    at: Vec<Option<usize>>,
    image: Vec<u8>,
}

impl<'m> RunImages<'m> {
    fn load(ctx: &mut ExecCtx<'_>, tmem: crate::arena::AllocId, runs: &'m [oplib::TcgenCellRun], enabled: bool) -> Result<Self, crate::interp::ExecError> {
        let mut at = Vec::with_capacity(runs.len());
        let mut image = Vec::new();
        let size = ctx.arena.get(tmem).size;
        for r in runs {
            let span = ByteSpan::new(addr::tmem_byte_offset(r.tmem_lane, r.column), r.cells as u64 * 4);
            let ok = enabled
                && r.column as u64 + r.cells as u64 <= addr::TMEM_COLS as u64
                && span.end() <= size
                && support::tmem_live(ctx, r.column, r.cells)
                && ctx.arena.first_invalid(support::whole(ctx.arena, tmem), span).is_none();
            if ok {
                let k = image.len();
                image.resize(k + span.len as usize, 0);
                ctx.arena
                    .read(support::whole(ctx.arena, tmem), &[span], &mut image[k..])
                    .map_err(|e| support::arena_err(ctx, e, WarpMask::ALL))?;
                at.push(Some(k));
            } else {
                at.push(None);
            }
        }
        Ok(RunImages { runs, at, image })
    }

    /// Image byte of piece `p` when its run is fast. `hint`: last run index.
    #[inline]
    fn locate(&self, p: &oplib::TcgenLdstPiece, hint: &mut usize) -> Option<usize> {
        let inside = |r: &oplib::TcgenCellRun| r.tmem_lane == p.tmem_lane && p.column >= r.column && p.column < r.column + r.cells;
        let i = if self.runs.get(*hint).is_some_and(inside) {
            *hint
        } else {
            let i = self.runs.partition_point(|r| (r.tmem_lane, r.column) <= (p.tmem_lane, p.column)).checked_sub(1)?;
            if !inside(&self.runs[i]) {
                return None;
            }
            *hint = i;
            i
        };
        self.at[i].map(|k| k + (p.column - self.runs[i].column) as usize * 4 + p.cell_byte as usize)
    }

    /// Write every fast run's image back (one write per run).
    fn store(&self, ctx: &mut ExecCtx<'_>, tmem: crate::arena::AllocId) -> Result<(), crate::interp::ExecError> {
        for (r, k) in self.runs.iter().zip(&self.at) {
            if let Some(k) = *k {
                let span = ByteSpan::new(addr::tmem_byte_offset(r.tmem_lane, r.column), r.cells as u64 * 4);
                let v = support::whole(ctx.arena, tmem);
                ctx.arena
                    .write(v, &[span], &self.image[k..k + span.len as usize])
                    .map_err(|e| support::arena_err(ctx, e, WarpMask::ALL))?;
            }
        }
        Ok(())
    }
}

#[inline]
pub fn tcgen_ld(ctx: &mut ExecCtx<'_>, args: &TcgenLdArgs) -> HResult {
    active_or_next!(ctx);
    full_warp(ctx, "tcgen05.ld")?;
    // `.spcompress`: the selection op and `.abs` ride in `red` / `red_abs`
    // (W4-17: a form without `.red` carries `red = (op, [])`).
    let compress = if args.spcompress {
        match &args.red {
            Some((op, _)) => Some((*op == crate::program::ReduxOp::Max, args.red_abs)),
            None => return Err(support::unsupported(ctx, "tcgen05.ld .spcompress without its max/min op")),
        }
    } else {
        None
    };
    let active = ctx.warp.active;
    let taddr = effective_taddr(ctx, args.taddr, args.row, args.col, active)?;
    let map = oplib::tcgen_ldst_map(args.shape, args.num, args.pack, ctx.warp.warp_in_cta, taddr)
        .map_err(|e| support::op_err(ctx, e))?;
    let nregs = map.registers;
    let red = match &args.red {
        Some((_, regs)) if regs.is_empty() => None,
        Some((op, regs)) => {
            let ty = regs.first().map(|&r| support::reg_ty(ctx, r).elem).unwrap_or(crate::dtype::Dtype::U32);
            Some((oplib::TcgenLdRed::new(*op, ty, args.red_abs, args.red_nan).map_err(|e| support::op_err(ctx, e))?, regs.clone()))
        }
        None => None,
    };
    let op = ldst_op(ctx, AsyncClass::TcgenLd);
    let tmem = ctx.cta.tmem;
    let mut tspans = Vec::with_capacity(if ctx.observing { active.count() as usize * nregs } else { 0 });
    // One register image buffer for every lane (no per-lane allocation).
    let mut bytes = vec![0u8; nregs * 4];
    // W4-16: each live, fully valid cell run is read once into `image`;
    // pieces of other runs take the per-piece path (same errors / findings).
    let runs = RunImages::load(ctx, tmem, map.cell_runs(), ctx.aux.capture_reads.is_none())?;
    for t in active.lanes() {
        bytes.fill(0);
        let mut hint = 0usize;
        for r in 0..nregs {
            for p in map.pieces(r, t) {
                let at = 4 * r + p.reg_byte as usize;
                let n = p.len as usize;
                if let Some(k) = runs.locate(p, &mut hint) {
                    bytes[at..at + n].copy_from_slice(&runs.image[k..k + n]);
                } else {
                    check_piece_live(ctx, p, t, "tcgen05.ld")?;
                    let loc = support::Loc { alloc: tmem, offset: piece_offset(p), window: None, remote: None };
                    support::mem_read(ctx, loc, t, &mut bytes[at..at + n])?;
                }
                if ctx.observing {
                    tspans.push(LaneSpan { lane: t as u8, span: ByteSpan::new(piece_offset(p), p.len as u64) });
                }
            }
        }
        let compressed: Vec<u8>;
        let out: &[u8] = match compress {
            Some((max, abs)) => {
                let words: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
                // Unreadable (invalid) words already failed or reported at the read.
                let valid = vec![true; words.len()];
                let (kept, _) = oplib::tcgen_ld_spcompress(&words, &valid, max, abs).map_err(|e| support::op_err(ctx, e))?;
                compressed = kept.iter().flat_map(|w| w.to_le_bytes()).collect();
                &compressed
            }
            None => &bytes,
        };
        let mut pos = 0usize;
        for &d in &args.dsts {
            let n = support::reg_ty(ctx, d).mem_bytes() as usize;
            let end = (pos + n).min(out.len());
            if pos < end {
                write_lane_bytes(ctx, d, t, &out[pos..end]);
            }
            pos += n;
        }
        if let Some((red, regs)) = &red {
            let words: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
            let v = oplib::tcgen_ld_reduce(*red, &words).map_err(|e| support::op_err(ctx, e))?;
            if let Some(&rr) = regs.first() {
                write_lane_bytes(ctx, rr, t, &v.to_le_bytes());
            }
        }
    }
    emit_async_spans(ctx, op, Side::Read, AccessKind::Read, tmem, tspans);
    let mut written: Vec<Reg> = args.dsts.clone();
    if let Some((_, regs)) = &red {
        written.extend(regs.iter().copied());
    }
    let rs = if ctx.observing { reg_spans(ctx, &written, active) } else { Vec::new() };
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
    full_warp(ctx, "tcgen05.st")?;
    let active = ctx.warp.active;
    let taddr = effective_taddr(ctx, args.taddr, args.row, args.col, active)?;
    let map = oplib::tcgen_ldst_map(args.shape, args.num, args.unpack, ctx.warp.warp_in_cta, taddr)
        .map_err(|e| support::op_err(ctx, e))?;
    let nregs = map.registers;
    let op = ldst_op(ctx, AsyncClass::TcgenSt);
    let tmem = ctx.cta.tmem;
    let mut tspans = Vec::with_capacity(if ctx.observing { active.count() as usize * nregs } else { 0 });
    // W4-16: live, fully valid cell runs are patched in `image` and written
    // back once (validity is unchanged: already valid); other pieces take
    // the per-piece path. The images are flushed before any error returns.
    let fast = !(ctx.aux.wants_history && !ctx.aux.words.is_empty());
    let mut runs = RunImages::load(ctx, tmem, map.cell_runs(), fast)?;
    let mut bytes = Vec::with_capacity(nregs * 4);
    let res = (|| -> Result<(), crate::interp::ExecError> {
        for t in active.lanes() {
            bytes.clear();
            for &s in &args.srcs {
                let ty = support::operand_ty(ctx, s);
                let mut tmp = [0u8; 32];
                support::lane_bytes(ctx, s, ty, t, &mut tmp);
                bytes.extend_from_slice(&tmp[..ty.mem_bytes() as usize]);
            }
            bytes.resize(nregs * 4, 0);
            let mut hint = 0usize;
            for r in 0..nregs {
                for p in map.pieces(r, t) {
                    let at = 4 * r + p.reg_byte as usize;
                    let n = p.len as usize;
                    if let Some(k) = runs.locate(p, &mut hint) {
                        runs.image[k..k + n].copy_from_slice(&bytes[at..at + n]);
                    } else {
                        check_piece_live(ctx, p, t, "tcgen05.st")?;
                        let loc = support::Loc { alloc: tmem, offset: piece_offset(p), window: None, remote: None };
                        support::mem_write(ctx, loc, t, &bytes[at..at + n])?;
                    }
                    if ctx.observing {
                        tspans.push(LaneSpan { lane: t as u8, span: ByteSpan::new(piece_offset(p), p.len as u64) });
                    }
                }
            }
        }
        Ok(())
    })();
    runs.store(ctx, tmem)?;
    res?;
    let regs: Vec<Reg> = args.srcs.iter().filter_map(|s| if let Operand::Reg(r) = s { Some(*r) } else { None }).collect();
    let rs = if ctx.observing { reg_spans(ctx, &regs, active) } else { Vec::new() };
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
    full_warp(ctx, if st { "tcgen05.wait::st" } else { "tcgen05.wait::ld" })?;
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

/// `tcgen05.cp`: shared memory (matrix descriptor) -> TMEM, asynchronous,
/// pipelined with mma (oplib `tcgen_cp_plan`). `cta_group::2` applies the
/// plan in both CTAs of the pair, each reading its own shared window.
#[inline]
pub fn tcgen_cp(ctx: &mut ExecCtx<'_>, args: TcgenCpArgs) -> HResult {
    active_or_next!(ctx);
    let active = ctx.warp.active;
    let g = group(args.cta_group);
    let arch = match ctx.program.arch.as_deref() {
        Some(a) if a.starts_with("sm_103") => oplib::TcArch::Sm103,
        Some(a) if a.starts_with("sm_107") => oplib::TcArch::Sm107,
        _ => oplib::TcArch::Sm100,
    };
    let mut all = Vec::new();
    for l in active.lanes() {
        let one = WarpMask::lane(l);
        let taddr = effective_taddr(ctx, args.taddr, args.row, args.col, one)?;
        let sdesc = lane_val(ctx, args.sdesc, l);
        let plan = oplib::tcgen_cp_plan(args.rows, args.bits, args.multicast, args.decompress_bits, sdesc, taddr, g, arch)
            .map_err(|e| support::op_err(ctx, e))?;
        let (srcs, cells) = plan.pairs();
        let ranks: Vec<usize> = if g == 2 {
            let even = (ctx.cta.rank_in_cluster & !1) as usize;
            vec![even, even + 1]
        } else {
            vec![ctx.cta.rank_in_cluster as usize]
        };
        let mut src = Vec::new();
        let mut dst = Vec::new();
        for r in ranks {
            let (Some(&sm), Some(&tm)) = (ctx.cta.cluster_smem.get(r), ctx.cta.cluster_tmem.get(r)) else {
                return Err(ctx.error(ExecErrorKind::BadAddress, "tcgen05.cp peer CTA outside the cluster"));
            };
            for (s, &(lane, col)) in srcs.iter().zip(&cells) {
                let off = addr::decode_shared(s.start as u32).1 as u64;
                src.push((sm, ByteSpan::new(off, s.len)));
                dst.push((tm, ByteSpan::new(addr::tmem_byte_offset(lane, col), 4)));
            }
        }
        let batch = [
            (ResourceId::TcgenKernel, SyncCmd::TcgenGroup(g)),
            (work_res(ctx, l), SyncCmd::TcgenWork(tcgen::WorkCmd::Issue)),
        ];
        support::step_all(ctx, &batch)?;
        all.extend(batch);
        let after: Vec<AsyncId> = ctx.aux.tcgen_last.get(&ctx.cta.id).copied().into_iter().collect();
        let op = issue_async(
            ctx,
            one,
            Issue {
                kind: AsyncKind::TcgenCp,
                class: AsyncClass::TcgenPipelined,
                proxy: Proxy::Tcgen,
                payload: Payload::TcgenCp { src, dst, decompress_bits: args.decompress_bits },
                signals: Vec::new(),
                after,
                targets: Vec::new(),
                queue: true,
                fill_pattern: Vec::new(),
                tf32_round: false,
                report: None,
                lut_b: None,
                strong: None,
                restricted: false,
            },
        );
        ctx.aux.tcgen_last.insert(ctx.cta.id, op);
        ctx.aux.tcgen_uncommitted.entry((ctx.warp.id, l as u8)).or_default().push(op);
    }
    support::protocol(ctx, active, all, ProtoExtra::default());
    Ok(Flow::Next)
}


#[inline]
pub fn tcgen_mma(ctx: &mut ExecCtx<'_>, args: &TcgenMmaArgs) -> HResult {
    active_or_next!(ctx);
    // `tcgen05.mma` is issued by a single thread (sync-semantics §6.3): two
    // lanes executing one MMA site issue two MMAs into the same TMEM
    // accumulator, a kernel bug (W12-gaps 3).
    if ctx.warp.active.count() > 1 {
        let m = ctx.warp.active;
        return Err(support::err(
            ctx,
            ExecErrorKind::Op(crate::oplib::OpErrorKind::Invalid),
            m,
            format!("tcgen05.mma must be issued by a single thread; lanes {:#010x} issued it", m.bits()),
        ));
    }
    // `.lut_b`: the lookup table's TMEM address must name a column inside
    // a live allocation (fail closed otherwise).
    let lut_b = match args.lut_b_addr {
        Some(o) => {
            let t = uniform_over(ctx, o, active_lanes(ctx))? as u32;
            let (_, col) = addr::tmem_decode(t);
            if !support::tmem_live(ctx, col, 1) {
                return Err(support::unsupported(ctx, "tcgen05.mma .lut_b address outside a live TMEM allocation"));
            }
            Some(t)
        }
        None => None,
    };
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
        // Decision 16 (W11-7): a runtime instruction descriptor must encode the
        // shape the typed op declared.
        if let Some(declared) = args.declared {
            crate::oplib::tcgen_mma_check_declared(args.kind, v(args.idesc) as u32, declared)
                .map_err(|e| support::op_err(ctx, e))?;
        }
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
            // A `.ws` zero-column mask is one 64-bit operand: pass it as
            // [low, high] words (oplib reads both; W4-15).
            disable_output_lane: args
                .disable_output_lane
                .iter()
                .flat_map(|&o| {
                    let x = v(o);
                    if args.ws && support::operand_ty(ctx, o).bits() == 64 {
                        vec![x as u32, (x >> 32) as u32]
                    } else {
                        vec![x as u32]
                    }
                })
                .collect(),
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
        let mma_payload = matches!(args.a, TcA::Smem(_)).then(|| payload.clone());
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
                report: None,
                lut_b,
                strong: None,
                restricted: false,
            },
        );
        ctx.aux.tcgen_last.insert(ctx.cta.id, op);
        ctx.aux.tcgen_uncommitted.entry((ctx.warp.id, l as u8)).or_default().push(op);
        if let (TcA::Smem(_), Some(p)) = (args.a, mma_payload) {
            shared_a_read(ctx, l, op, &p)?;
        }
    }
    support::protocol(ctx, active, all, ProtoExtra::default());
    Ok(Flow::Next)
}

/// Shared-memory A footprint of an MMA: run oplib's MMA twice on recording
/// closures (no memory), the second time with A's descriptor start moved by
/// 1024 bytes (a swizzle-atom multiple, so swizzling is unchanged); reads
/// that moved are A's, the others B's. Cached per descriptor set.
fn mma_a_footprint(ctx: &mut ExecCtx<'_>, p: &TcgenMmaPayload) -> Result<Vec<(u32, u32, u32)>, ExecError> {
    // The A footprint depends on A's descriptor (layout, LBO/SBO, swizzle),
    // the instruction descriptor and the group, and moves with A's start
    // address; swizzling sees the start only modulo 1024 bytes. Cache it
    // relative to the start, keyed by everything else, so a pipeline
    // walking its stages reuses one probe.
    let start = ((p.a & 0x3fff) << 4) as u32;
    let key = (p.a & !0x3fff, (start % 1024) as u64, p.idesc, p.args.cta_group);
    if let Some(v) = ctx.aux.mma_a_footprints.get(&key) {
        return Ok(v.iter().map(|&(c, rel, n)| (c, rel.wrapping_add(start), n)).collect());
    }
    let arch = match ctx.program.arch.as_deref() {
        Some(a) if a.starts_with("sm_103") => oplib::TcArch::Sm103,
        Some(a) if a.starts_with("sm_107") => oplib::TcArch::Sm107,
        _ => oplib::TcArch::Sm100,
    };
    let options = oplib::TcMmaOptions { arch, ti16: p.args.kind == TcMmaKind::Ti16, ..Default::default() };
    let out = mma_a_footprint_probe(p, &options);
    ctx.aux.mma_a_footprints.insert(key, out.iter().map(|&(c, a, n)| (c, a.wrapping_sub(start), n)).collect());
    Ok(out)
}

/// The MMA's shared-A read as its own async op (legacy `MmaSharedARead`,
/// W5-10): `preds: []` in its AsyncIssue, lands right after the MMA (so the
/// MMA has read A when it completes), and is what a `.sync_restrict`
/// commit tracks.
fn shared_a_read(ctx: &mut ExecCtx<'_>, l: usize, mma: AsyncId, p: &TcgenMmaPayload) -> Result<(), ExecError> {
    // The op exists whether or not anyone observes (landing order and
    // commit tracking must not depend on the observer); only its access
    // footprint is computed for observers.
    let reads = if ctx.observing { mma_a_footprint(ctx, p)? } else { Vec::new() };
    let mut spans: Vec<(crate::arena::AllocId, ByteSpan)> = reads
        .iter()
        .filter_map(|&(cta, a, n)| p.smem.get(cta as usize).map(|&al| (al, ByteSpan::new(addr::decode_shared(a).1 as u64, n as u64))))
        .collect();
    spans.sort();
    let mut merged: Vec<(crate::arena::AllocId, ByteSpan)> = Vec::new();
    for (a, s) in spans {
        match merged.last_mut() {
            Some((la, ls)) if *la == a && s.start <= ls.end() => {
                let end = ls.end().max(s.end());
                ls.len = end - ls.start;
            }
            _ => merged.push((a, s)),
        }
    }
    let op = issue_async(
        ctx,
        WarpMask::lane(l),
        Issue {
            kind: AsyncKind::TcgenMma,
            class: AsyncClass::TcgenPipelined,
            proxy: Proxy::Async,
            payload: Payload::None,
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
        },
    );
    // Lands after the MMA (whose landing performs the read).
    if let Some(o) = ctx.sync.async_ops.iter_mut().rev().find(|o| o.id == op) {
        o.after.push(mma);
    }
    if let Some(m) = ctx.aux.async_meta.get_mut(&op) {
        m.a_reads = merged.clone();
        m.is_a_read = true;
    }
    if let Some(m) = ctx.aux.async_meta.get_mut(&mma) {
        m.a_reads = merged;
    }
    ctx.aux.tcgen_uncommitted.entry((ctx.warp.id, l as u8)).or_default().push(op);
    ctx.aux.tcgen_shared_reads.entry((ctx.warp.id, l as u8)).or_default().push(op);
    Ok(())
}

#[inline]
pub fn tile(ctx: &mut ExecCtx<'_>, args: &TileArgs) -> HResult {
    active_or_next!(ctx);
    // Contract decision 11: tile ops are lowered through TVM's dispatch to
    // PTX-level IR, so `Instr::Tile` is never executed by the engine. A
    // program that still contains one fails closed (incomplete).
    Err(support::unsupported(
        ctx,
        &format!("Instr::Tile ({:?}): tile ops must be lowered through TVM dispatch to PTX-level IR", args.op),
    ))
}

/// Window reads `(cta, address, len)` of the MMA's shared-memory A operand
/// (see [`mma_a_footprint`]); pure, no memory.
pub(crate) fn mma_a_footprint_probe(p: &TcgenMmaPayload, options: &oplib::TcMmaOptions) -> Vec<(u32, u32, u32)> {
    use std::cell::RefCell;
    let probe = |payload: &TcgenMmaPayload| -> Vec<(u32, u32, u32)> {
        let reads: RefCell<Vec<(u32, u32, u32)>> = RefCell::new(Vec::new());
        let smem = |cta: u32, a: u32, out: &mut [u8]| -> oplib::OpResult {
            out.fill(0);
            reads.borrow_mut().push((cta, a, out.len() as u32));
            Ok(())
        };
        let tr = |_: u32, _: u32, _: u32, out: &mut [u8]| -> oplib::OpResult {
            out.fill(0);
            Ok(())
        };
        let mut tw = |_: u32, _: u32, _: u32, _: &[u8]| -> oplib::OpResult { Ok(()) };
        let _ = oplib::tc_mma_ctas(payload, options, &smem, &tr, &mut tw);
        reads.into_inner()
    };
    let first = probe(p);
    let start = p.a & 0x3fff;
    let delta: i64 = if start + 64 < 0x4000 { 64 } else { -64 };
    let mut moved = p.clone();
    moved.a = (p.a & !0x3fff) | ((start as i64 + delta) as u64 & 0x3fff);
    let second = probe(&moved);
    let mut out = Vec::new();
    if first.len() == second.len() {
        for (x, y) in first.iter().zip(&second) {
            if x.0 == y.0 && y.1 as i64 == x.1 as i64 + delta * 16 {
                out.push(*x);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::AllocId;
    use numsim_oplib::tcgen05::encode::{encode_dense_instr_descriptor_fields, encode_matrix_descriptor};

    #[test]
    fn shared_a_footprint_excludes_b() {
        let k = 16usize;
        let (m, n) = (128usize, 16usize);
        // K-major no-swizzle: 8x16-byte core matrices, lbo 128, sbo 256.
        let desc = |start: u32| encode_matrix_descriptor(start, 128 >> 4, 256 >> 4, 0);
        let idesc = encode_dense_instr_descriptor_fields("float32", "bfloat16", "bfloat16", m as i64, n as i64, k as i64, false, false, 1, false, false, false, false).unwrap() as u32;
        let c = Operand::Const(ConstId(0));
        let p = TcgenMmaPayload {
            args: TcgenMmaArgs {
                kind: TcMmaKind::F16,
                cta_group: 1,
                d: c,
                a: TcA::Smem(c),
                b_desc: c,
                idesc: c,
                enable_input_d: c,
                ws: false,
                ws_b_buffer: 0,
                block_scale: None,
                scale_input_d: None,
                sparse_meta: None,
                disable_output_lane: Vec::new(),
                collector_a: CollectorOp::None,
                collector_b: CollectorOp::None,
                ashift: false,
                lut_b: false,
                lut_b_addr: None,
                declared: None,
            },
            d_taddr: 0,
            a: desc(0x1000),
            b_desc: desc(0x8000),
            idesc,
            enable_input_d: false,
            scale_taddrs: None,
            scale_input_d: None,
            sparse_meta: None,
            disable_output_lane: Vec::new(),
            smem: vec![AllocId(0)],
            tmem: vec![AllocId(1)],
        };
        let a = mma_a_footprint_probe(&p, &oplib::TcMmaOptions::default());
        assert!(!a.is_empty());
        let bytes: u32 = a.iter().map(|x| x.2).sum();
        assert_eq!(bytes as usize, m * k * 2, "every A byte, once");
        assert!(a.iter().all(|&(_, addr, len)| addr >= 0x1000 && addr + len <= 0x8000), "{a:?}");
    }
}
