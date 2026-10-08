//! Async copies: cp.async, bulk copies, TMA, st.async, async groups,
//! tensor-map updates.
//!
//! Every data effect becomes an [`AsyncOp`] whose payload is resolved at
//! issue (spans, captured bytes) and lands later in the scheduler; issue
//! emits `SyncKind::AsyncIssue` (racecheck-semantics §3 rows 12-17). Sync
//! completions are bound at issue: mbarrier `Issue` captures the
//! generation (`Completion::MbarTx/MbarArrive`), async-group membership is
//! tracked in `LaunchAux::groups` and milestones are queued when the
//! group's ops have landed.

use super::HResult;
use crate::arena::{AllocId, ByteSpan};
use crate::interp::aux::AsyncMeta;
use crate::interp::support::{self, lane_bytes, lane_int, lane_val, Accesses, Loc, ProtoExtra};
use crate::interp::{ExecCtx, ExecError, ExecErrorKind, Flow};
use crate::observe::{AccessKind, AsyncClass, AsyncTarget, FenceEvent, PublishTarget, Side, SyncKind};
use crate::oplib::TensorMapDesc;
use crate::program::*;
use crate::sync::async_group::{self, Domain};
use crate::sync::completion::{AsyncKind, AsyncOp, AsyncSource, Payload};
use crate::sync::{mbarrier, AsyncId, Completion, Outcome, ResourceId, Step, SyncCmd};
use crate::value::WarpMask;

/// Spans an async payload touches (lifetime footprint).
pub fn footprint(p: &Payload) -> Vec<(AllocId, ByteSpan)> {
    match p {
        Payload::None | Payload::TcgenMma(_) => Vec::new(),
        Payload::Copy { src, dst, zero_fill } => src.iter().chain(dst).chain(zero_fill).copied().collect(),
        Payload::Reduce { src, dst, .. } => src.iter().chain(dst).copied().collect(),
        Payload::Data { dst, .. } | Payload::ReduceData { dst, .. } => dst.clone(),
        Payload::TcgenCp { src, dst, .. } => src.iter().chain(dst).copied().collect(),
    }
}

/// Everything about an async op the issuing handler decides.
pub struct Issue {
    pub kind: AsyncKind,
    pub class: AsyncClass,
    pub proxy: Proxy,
    pub payload: Payload,
    pub signals: Vec<Completion>,
    pub after: Vec<AsyncId>,
    pub targets: Vec<AsyncTarget>,
    /// Queue the op for landing (false for ops whose data moved at issue,
    /// e.g. tcgen05.ld/st; their id is only an observer actor).
    pub queue: bool,
    /// TMA loads: OOB fill pattern and TF32 rounding (`oplib::TmaPlan`).
    pub fill_pattern: Vec<u8>,
    pub tf32_round: bool,
}

/// Issue an async op: allocate its id, emit `AsyncIssue`, queue it.
pub fn issue_async(ctx: &mut ExecCtx<'_>, lanes: WarpMask, is: Issue) -> AsyncId {
    let id = ctx.sync.next_async_id();
    let phases = is
        .signals
        .iter()
        .filter_map(|c| match *c {
            Completion::MbarTx { res, gen, .. } | Completion::MbarArrive { res, gen, .. } => Some((res, gen)),
            _ => None,
        })
        .collect();
    if ctx.observing {
        let fp = footprint(&is.payload);
        support::sync_event(
            ctx,
            lanes,
            SyncKind::AsyncIssue {
                op: id,
                class: is.class,
                proxy: is.proxy,
                preds: is.after.clone(),
                footprint: fp,
                targets: is.targets.clone(),
            },
        );
    }
    let lane = lanes.first().unwrap_or(0) as u8;
    ctx.aux.async_meta.insert(
        id,
        AsyncMeta { lane, class: is.class, proxy: is.proxy, phases, fill_pattern: is.fill_pattern, tf32_round: is.tf32_round },
    );
    if is.queue {
        let source = AsyncSource { warp: ctx.warp.id, cta: ctx.cta.id, site: ctx.site(), seq: ctx.warp.sync_seq };
        ctx.sync.async_ops.push_back(AsyncOp {
            id,
            source,
            kind: is.kind,
            payload: is.payload,
            signals: is.signals,
            after: is.after,
        });
    }
    id
}

/// mbarrier resource named by an address (8-byte aligned shared word).
pub fn mbar_res(ctx: &ExecCtx<'_>, space: AddrSpace, a: u64, lane: usize) -> Result<ResourceId, ExecError> {
    let loc = support::resolve(ctx, space, a, lane, 8)?;
    if ctx.arena.get(loc.alloc).space != crate::arena::Space::Shared {
        return Err(support::err(ctx, ExecErrorKind::BadAddress, WarpMask::lane(lane), "mbarrier not in shared memory"));
    }
    if loc.offset % 8 != 0 {
        return Err(support::err(ctx, ExecErrorKind::Misaligned, WarpMask::lane(lane), "mbarrier address is not 8-byte aligned"));
    }
    let cta = loc.remote.unwrap_or(ctx.cta.id);
    Ok(ResourceId::Mbarrier { cta, alloc: loc.alloc, offset: loc.offset as u32 })
}

/// Same mbarrier offset in CTA `rank` of the cluster.
pub fn mbar_in_rank(ctx: &ExecCtx<'_>, res: ResourceId, rank: u32) -> Option<ResourceId> {
    let ResourceId::Mbarrier { offset, .. } = res else { return None };
    let alloc = *ctx.cta.cluster_smem.get(rank as usize)?;
    let cta = *ctx.cta.cluster_ctas.get(rank as usize)?;
    Some(ResourceId::Mbarrier { cta, alloc, offset })
}

/// `mbarrier::Cmd::Issue` on `res`: the bound generation.
pub fn mbar_issue(ctx: &mut ExecCtx<'_>, res: ResourceId) -> Result<u64, ExecError> {
    match support::step(ctx, res, SyncCmd::Mbarrier(mbarrier::Cmd::Issue))? {
        Step::Done(Outcome::Mbarrier(mbarrier::Outcome::Issued { gen })) => Ok(gen),
        other => Err(ctx.error(ExecErrorKind::Internal, format!("mbarrier Issue returned {other:?}"))),
    }
}

/// Ranks named by a `.multicast::cluster` CTA mask.
pub fn ranks_of(ctx: &ExecCtx<'_>, mask: u64) -> Vec<u32> {
    let n = ctx.launch.ctas_per_cluster();
    (0..n.min(64)).filter(|r| mask >> r & 1 == 1).collect()
}

/// Is `loc` a shared-window location (own or cluster peer)?
fn smem_loc_in_rank(ctx: &ExecCtx<'_>, loc: Loc, rank: u32) -> Option<(AllocId, u64)> {
    Some((*ctx.cta.cluster_smem.get(rank as usize)?, loc.offset))
}

fn group_res(ctx: &ExecCtx<'_>, lane: usize, domain: Domain) -> ResourceId {
    ResourceId::AsyncGroup { warp: ctx.warp.id, lane: lane as u8, domain }
}

#[inline]
pub fn cp_async(
    ctx: &mut ExecCtx<'_>,
    dst: Operand,
    src: Operand,
    cp_size: u8,
    src_size: Option<Operand>,
    ignore_src: Option<Operand>,
    mods: MemMods,
) -> HResult {
    active_or_next!(ctx);
    let _ = mods;
    let active = ctx.warp.active;
    let cp = cp_size as u64;
    // One async op per thread (async groups are per thread).
    let mut lanes_ops = Vec::with_capacity(active.count() as usize);
    for l in active.lanes() {
        let da = lane_val(ctx, dst, l);
        let dspace = if da >> 32 == 0 { AddrSpace::Shared } else { AddrSpace::Generic };
        let dl = support::resolve(ctx, dspace, da, l, cp)?;
        let mut n = match src_size {
            Some(o) => lane_val(ctx, o, l).min(cp),
            None => cp,
        };
        if let Some(p) = ignore_src {
            if lane_val(ctx, p, l) != 0 {
                n = 0;
            }
        }
        let (mut s, mut d, mut z) = (Vec::new(), Vec::new(), Vec::new());
        if n > 0 {
            let sl = support::resolve(ctx, AddrSpace::Generic, lane_val(ctx, src, l), l, n)?;
            s.push((sl.alloc, sl.span(n)));
            d.push((dl.alloc, dl.span(n)));
        }
        if n < cp {
            z.push((dl.alloc, ByteSpan::new(dl.offset + n, cp - n)));
        }
        lanes_ops.push((l, Payload::Copy { src: s, dst: d, zero_fill: z }));
    }
    let cmds: Vec<(ResourceId, SyncCmd)> = active
        .lanes()
        .map(|l| (group_res(ctx, l, Domain::CpAsync), SyncCmd::AsyncGroup(async_group::Cmd::Issue)))
        .collect();
    support::step_all(ctx, &cmds)?;
    for (l, payload) in lanes_ops {
        let op = issue_async(
            ctx,
            WarpMask::lane(l),
            Issue {
                kind: AsyncKind::CpAsync,
                class: AsyncClass::Copy,
                proxy: Proxy::Generic,
                payload,
                signals: Vec::new(),
                after: Vec::new(),
                targets: Vec::new(),
                queue: true,
                fill_pattern: Vec::new(),
                tf32_round: false,
            },
        );
        ctx.aux.groups.issue(group_res(ctx, l, Domain::CpAsync), op);
    }
    support::protocol(ctx, active, cmds, ProtoExtra::default());
    Ok(Flow::Next)
}

#[inline]
pub fn async_commit(ctx: &mut ExecCtx<'_>, domain: Domain) -> HResult {
    active_or_next!(ctx);
    let active = ctx.warp.active;
    let cmds: Vec<(ResourceId, SyncCmd)> = active
        .lanes()
        .map(|l| (group_res(ctx, l, domain), SyncCmd::AsyncGroup(async_group::Cmd::Commit)))
        .collect();
    let Step::Done(outs) = support::step_all(ctx, &cmds)? else {
        return Err(ctx.error(ExecErrorKind::Internal, "commit_group blocked"));
    };
    for ((res, _), out) in cmds.iter().zip(outs) {
        if let Outcome::AsyncGroup(async_group::Outcome::Committed { ordinal, .. }) = out {
            let due = ctx.aux.groups.commit(*res, ordinal);
            ctx.sync.completions.extend(due);
        }
    }
    support::protocol(ctx, active, cmds, ProtoExtra::default());
    Ok(Flow::Next)
}

#[inline]
pub fn async_wait(ctx: &mut ExecCtx<'_>, domain: Domain, n: u32, read: bool) -> HResult {
    active_or_next!(ctx);
    let active = ctx.warp.active;
    let cmd = SyncCmd::AsyncGroup(async_group::Cmd::Wait { n: n as u64, read });
    // Ops of the awaited prefix, per lane (for the AsyncComplete events).
    let mut covered: Vec<(AsyncId, WarpMask)> = Vec::new();
    let mut retired_keys = Vec::new();
    for l in active.lanes() {
        let res = group_res(ctx, l, domain);
        match support::step(ctx, res, cmd)? {
            Step::Blocked(r) => return Ok(Flow::Blocked(r)),
            Step::Done(_) => {}
        }
        if !ctx.observing {
            continue;
        }
        // Prefix ordinals: groups no longer in the state were retired by
        // this wait; still-present ones are covered when `.read`.
        let present: Vec<u64> = match ctx.sync.get(res) {
            Some(crate::sync::Resource::AsyncGroup(s)) => s.groups.iter().map(|g| g.ordinal).collect(),
            _ => Vec::new(),
        };
        let first_present = present.first().copied().unwrap_or(u64::MAX);
        for (&(r, ord), ops) in ctx.aux.groups.members.iter() {
            if r != res {
                continue;
            }
            let retired = ord < first_present;
            if retired || read {
                for &op in ops {
                    match covered.iter_mut().find(|(o, _)| *o == op) {
                        Some((_, m)) => *m = m.or(WarpMask::lane(l)),
                        None => covered.push((op, WarpMask::lane(l))),
                    }
                }
            }
            if retired && !read {
                retired_keys.push((r, ord));
            }
        }
    }
    for k in retired_keys {
        ctx.aux.groups.members.remove(&k);
    }
    covered.sort_by_key(|(op, _)| *op);
    let cmds: Vec<(ResourceId, SyncCmd)> = active.lanes().map(|l| (group_res(ctx, l, domain), cmd)).collect();
    support::protocol(ctx, active, cmds, ProtoExtra::default());
    let milestone = if read { Side::Read } else { Side::Write };
    for (op, lanes) in covered {
        let target = PublishTarget::Warp { warp: ctx.warp.id, lanes };
        support::sync_event(ctx, lanes, SyncKind::AsyncComplete { op, milestone, target });
    }
    if !read {
        // Free bookkeeping of fully retired groups even when not observing.
        for l in active.lanes() {
            let res = group_res(ctx, l, domain);
            let first = match ctx.sync.get(res) {
                Some(crate::sync::Resource::AsyncGroup(s)) => s.groups.front().map(|g| g.ordinal).unwrap_or(u64::MAX),
                _ => u64::MAX,
            };
            ctx.aux.groups.members.retain(|(r, o), _| *r != res || *o >= first);
        }
    }
    Ok(Flow::Next)
}

#[inline]
pub fn cp_async_mbar_arrive(ctx: &mut ExecCtx<'_>, mbar: Operand, space: AddrSpace, noinc: bool) -> HResult {
    active_or_next!(ctx);
    let active = ctx.warp.active;
    let mut all_cmds = Vec::new();
    let mut targets = Vec::new();
    for l in active.lanes() {
        let res = mbar_res(ctx, space, lane_val(ctx, mbar, l), l)?;
        if !noinc {
            let c = SyncCmd::Mbarrier(mbarrier::Cmd::IncPending { count: 1 });
            support::step(ctx, res, c)?;
            all_cmds.push((res, c));
        }
        let gen = mbar_issue(ctx, res)?;
        all_cmds.push((res, SyncCmd::Mbarrier(mbarrier::Cmd::Issue)));
        let arrival = Completion::MbarArrive { res, gen, count: 1 };
        let gres = group_res(ctx, l, Domain::CpAsync);
        let c = SyncCmd::AsyncGroup(async_group::Cmd::ArriveOn);
        let out = support::step(ctx, gres, c)?;
        all_cmds.push((gres, c));
        match out {
            Step::Done(Outcome::AsyncGroup(async_group::Outcome::ArriveOn { group: Some(ord) })) => {
                if ctx.aux.groups.open.get(&gres).is_some_and(|v| !v.is_empty()) {
                    // The open issues became a non-closing batch.
                    let due = ctx.aux.groups.commit(gres, ord);
                    ctx.sync.completions.extend(due);
                }
                ctx.aux.groups.arrivals.entry((gres, ord)).or_default().push(arrival);
            }
            Step::Done(_) => ctx.sync.completions.push_back(arrival),
            Step::Blocked(r) => return Ok(Flow::Blocked(r)),
        }
        targets.push(AsyncTarget { res, bytes: 0, arrivals: 1 });
    }
    let extra = ProtoExtra { issued: targets, ..Default::default() };
    support::protocol(ctx, active, all_cmds, extra);
    Ok(Flow::Next)
}

/// Bind a mbarrier completion: `Issue` on each target, `MbarTx` signals.
fn bind_mbar_tx(
    ctx: &mut ExecCtx<'_>,
    targets: &[ResourceId],
    bytes: u64,
    signals: &mut Vec<Completion>,
    issued: &mut Vec<AsyncTarget>,
    cmds: &mut Vec<(ResourceId, SyncCmd)>,
) -> Result<(), ExecError> {
    for &res in targets {
        let gen = mbar_issue(ctx, res)?;
        cmds.push((res, SyncCmd::Mbarrier(mbarrier::Cmd::Issue)));
        signals.push(Completion::MbarTx { res, gen, bytes });
        issued.push(AsyncTarget { res, bytes, arrivals: 0 });
    }
    Ok(())
}

#[inline]
pub fn bulk_copy(ctx: &mut ExecCtx<'_>, args: BulkCopyArgs) -> HResult {
    active_or_next!(ctx);
    let active = ctx.warp.active;
    if args.report.is_some() {
        return Err(support::unsupported(ctx, "cp.async.bulk report forms"));
    }
    let mut cmds = Vec::new();
    let mut issued = Vec::new();
    for l in active.lanes() {
        let size = lane_val(ctx, args.size, l);
        let sa = lane_val(ctx, args.src, l);
        // `.ignore_oob`: only the in-bounds prefix of the source is read;
        // the destination keeps its bytes beyond it (tx counts `size`).
        let copy = if args.ignore_oob { in_bounds_prefix(ctx, args.src_space, sa, l, size)? } else { size };
        let sl = support::resolve(ctx, args.src_space, sa, l, copy)?;
        let dl = support::resolve(ctx, args.dst_space, lane_val(ctx, args.dst, l), l, size)?;
        let mask = args.byte_mask.map(|m| lane_val(ctx, m, l) as u16);
        // Destination CTAs (multicast: same offset in every masked CTA).
        let mut dsts: Vec<(AllocId, ByteSpan)> = Vec::new();
        let mut ranks: Vec<Option<u32>> = Vec::new();
        match args.multicast {
            Some(m) => {
                for r in ranks_of(ctx, lane_val(ctx, m, l)) {
                    let (alloc, off) = smem_loc_in_rank(ctx, dl, r)
                        .ok_or_else(|| ctx.error(ExecErrorKind::BadAddress, "multicast rank out of range"))?;
                    dsts.push((alloc, ByteSpan::new(off, size)));
                    ranks.push(Some(r));
                }
            }
            None => {
                dsts.push((dl.alloc, dl.span(size)));
                ranks.push(None);
            }
        }
        // Byte runs actually copied (byte mask per 16-byte chunk, OOB prefix).
        let runs = byte_runs(copy, mask);
        let mut srcs: Vec<(AllocId, ByteSpan)> = Vec::new();
        let mut dsts2: Vec<(AllocId, ByteSpan)> = Vec::new();
        for (da, ds) in &dsts {
            for &(o, n) in &runs {
                srcs.push((sl.alloc, ByteSpan::new(sl.offset + o, n)));
                dsts2.push((*da, ByteSpan::new(ds.start + o, n)));
            }
        }
        let dsts = dsts2;
        let (kind, payload) = match args.reduce {
            Some((op, dtype)) => (AsyncKind::BulkReduce, Payload::Reduce { op, dtype, src: srcs, dst: dsts }),
            None => (AsyncKind::Bulk, Payload::Copy { src: srcs, dst: dsts, zero_fill: Vec::new() }),
        };
        let mut signals = Vec::new();
        let mut targets = Vec::new();
        let mut group = None;
        match args.completion {
            BulkCompletion::Mbarrier { mbar, space } => {
                let res = mbar_res(ctx, space, lane_val(ctx, mbar, l), l)?;
                let res_list: Vec<ResourceId> = ranks
                    .iter()
                    .map(|r| match r {
                        Some(r) => mbar_in_rank(ctx, res, *r).unwrap_or(res),
                        None => res,
                    })
                    .collect();
                bind_mbar_tx(ctx, &res_list, size, &mut signals, &mut targets, &mut cmds)?;
            }
            BulkCompletion::Group => {
                let gres = group_res(ctx, l, Domain::Bulk);
                let c = SyncCmd::AsyncGroup(async_group::Cmd::Issue);
                support::step(ctx, gres, c)?;
                cmds.push((gres, c));
                group = Some(gres);
            }
        }
        issued.extend(targets.iter().copied());
        let op = issue_async(
            ctx,
            WarpMask::lane(l),
            Issue { kind, class: AsyncClass::Copy, proxy: Proxy::Async, payload, signals, after: Vec::new(), targets, queue: true, fill_pattern: Vec::new(), tf32_round: false },
        );
        if let Some(g) = group {
            ctx.aux.groups.issue(g, op);
        }
    }
    let extra = ProtoExtra { issued, ..Default::default() };
    support::protocol(ctx, active, cmds, extra);
    Ok(Flow::Next)
}

/// Bytes of `[a, a+size)` that lie inside the allocation `a` resolves to.
fn in_bounds_prefix(ctx: &ExecCtx<'_>, space: AddrSpace, a: u64, lane: usize, size: u64) -> Result<u64, ExecError> {
    let loc = support::resolve(ctx, space, a, lane, 0)?;
    let avail = ctx.arena.get(loc.alloc).size.saturating_sub(loc.offset);
    Ok(size.min(avail))
}

/// `(offset, len)` runs of the first `n` bytes selected by a `.cp_mask`
/// 16-bit byte mask (bit i = byte i of every 16-byte chunk).
fn byte_runs(n: u64, mask: Option<u16>) -> Vec<(u64, u64)> {
    let Some(m) = mask else { return if n > 0 { vec![(0, n)] } else { Vec::new() } };
    let mut out: Vec<(u64, u64)> = Vec::new();
    for i in 0..n {
        if (m >> (i % 16)) & 1 == 0 {
            continue;
        }
        match out.last_mut() {
            Some((o, l)) if *o + *l == i => *l += 1,
            _ => out.push((i, 1)),
        }
    }
    out
}

/// Read and decode the 128-byte tensor map at `addr` (+ overrides).
fn read_tmap(
    ctx: &mut ExecCtx<'_>,
    tmap: Operand,
    space: AddrSpace,
    lane: usize,
    acc: &mut Accesses,
) -> Result<(TensorMapDesc, Loc), ExecError> {
    let a = lane_val(ctx, tmap, lane);
    let loc = support::resolve(ctx, space, a, lane, TensorMapDesc::BYTES as u64)?;
    let mut b = [0u8; 128];
    support::mem_read(ctx, loc, lane, &mut b)?;
    if ctx.observing {
        acc.push(loc, lane as u8, 128);
    }
    let d = TensorMapDesc::decode(&b).map_err(|e| support::op_err(ctx, e))?;
    Ok((d, loc))
}

#[inline]
pub fn tma(ctx: &mut ExecCtx<'_>, args: &TmaArgs) -> HResult {
    active_or_next!(ctx);
    if args.report.is_some() {
        return Err(support::unsupported(ctx, "cp.async.bulk.tensor report forms"));
    }
    let active = ctx.warp.active;
    let mut cmds = Vec::new();
    let mut issued = Vec::new();
    let mut acc = Accesses::default();
    for l in active.lanes() {
        let (mut desc, _) = read_tmap(ctx, args.tmap, args.tmap_space, l, &mut acc)?;
        for o in &args.overrides {
            let v = lane_val(ctx, o.value, l);
            desc.replace(o.field, o.ord, v).map_err(|e| support::op_err(ctx, e))?;
        }
        let coords: Vec<i64> = args.coords.iter().map(|&c| lane_int(ctx, c, l)).collect();
        let offs: Vec<i64> = args.im2col_offsets.iter().map(|&c| lane_int(ctx, c, l)).collect();
        let sa = lane_val(ctx, args.smem, l);
        let sloc = support::resolve(ctx, args.smem_space, sa, l, 1)?;
        let pdir = match args.dir {
            TmaDir::Load | TmaDir::Prefetch => crate::oplib::TmaPlanDir::Load,
            TmaDir::Store | TmaDir::Reduce(_) => crate::oplib::TmaPlanDir::Store,
        };
        let plan = crate::oplib::tma_plan_dir(&desc, pdir, args.mode, &coords, &offs, sloc.offset).map_err(|e| support::op_err(ctx, e))?;
        if args.dir == TmaDir::Prefetch {
            continue;
        }
        let smem_alloc = sloc.alloc;
        let mut global = Vec::with_capacity(plan.global.len());
        for s in &plan.global {
            let (alloc, off) = ctx
                .arena
                .resolve_global(s.start, s.len)
                .map_err(|e| support::arena_err(ctx, e, WarpMask::lane(l)))?;
            global.push((alloc, ByteSpan::new(off, s.len)));
        }
        // Destination CTAs for loads (multicast).
        let ranks: Vec<Option<u32>> = match (args.dir, args.multicast) {
            (TmaDir::Load, Some(m)) => ranks_of(ctx, lane_val(ctx, m, l)).into_iter().map(Some).collect(),
            _ => vec![None],
        };
        let smem_in = |ctx: &ExecCtx<'_>, r: Option<u32>| -> AllocId {
            match r {
                Some(r) => ctx.cta.cluster_smem.get(r as usize).copied().unwrap_or(smem_alloc),
                None => smem_alloc,
            }
        };
        let (kind, payload) = match args.dir {
            TmaDir::Load => {
                let mut src = Vec::new();
                let mut dst = Vec::new();
                let mut zf = Vec::new();
                for &r in &ranks {
                    let a = smem_in(ctx, r);
                    src.extend(global.iter().copied());
                    dst.extend(plan.smem.iter().map(|s| (a, *s)));
                    zf.extend(plan.smem_oob_fill.iter().map(|s| (a, *s)));
                }
                (AsyncKind::Tma, Payload::Copy { src, dst, zero_fill: zf })
            }
            TmaDir::Store => {
                let src = plan.smem.iter().map(|s| (smem_alloc, *s)).collect();
                (AsyncKind::Tma, Payload::Copy { src, dst: global, zero_fill: Vec::new() })
            }
            TmaDir::Reduce(op) => {
                let dtype = desc.elem.ok_or_else(|| ctx.error(ExecErrorKind::Op(crate::oplib::OpErrorKind::Invalid), "tensor map without element type"))?;
                let src = plan.smem.iter().map(|s| (smem_alloc, *s)).collect();
                (AsyncKind::TmaReduce, Payload::Reduce { op, dtype, src, dst: global })
            }
            TmaDir::Prefetch => unreachable!(),
        };
        let mut signals = Vec::new();
        let mut targets = Vec::new();
        let mut group = None;
        match args.completion {
            BulkCompletion::Mbarrier { mbar, space } => {
                let res = mbar_res(ctx, space, lane_val(ctx, mbar, l), l)?;
                let list: Vec<ResourceId> = ranks
                    .iter()
                    .map(|r| match r {
                        Some(r) => mbar_in_rank(ctx, res, *r).unwrap_or(res),
                        None => res,
                    })
                    .collect();
                bind_mbar_tx(ctx, &list, plan.bytes, &mut signals, &mut targets, &mut cmds)?;
            }
            BulkCompletion::Group => {
                let gres = group_res(ctx, l, Domain::Bulk);
                let c = SyncCmd::AsyncGroup(async_group::Cmd::Issue);
                support::step(ctx, gres, c)?;
                cmds.push((gres, c));
                group = Some(gres);
            }
        }
        issued.extend(targets.iter().copied());
        let (fill_pattern, tf32_round) =
            if args.dir == TmaDir::Load { (plan.fill_pattern.clone(), plan.tf32_round) } else { (Vec::new(), false) };
        let op = issue_async(
            ctx,
            WarpMask::lane(l),
            Issue { kind, class: AsyncClass::Copy, proxy: Proxy::Async, payload, signals, after: Vec::new(), targets, queue: true, fill_pattern, tf32_round },
        );
        if let Some(g) = group {
            ctx.aux.groups.issue(g, op);
        }
    }
    let sp = support::spec(ctx, AccessKind::Read, Sem::Weak, Scope::Cta, Proxy::TensorMap);
    support::emit(ctx, sp, &mut acc);
    if !cmds.is_empty() {
        let extra = ProtoExtra { issued, ..Default::default() };
        support::protocol(ctx, active, cmds, extra);
    }
    Ok(Flow::Next)
}

#[inline]
pub fn st_async(ctx: &mut ExecCtx<'_>, args: StAsyncArgs) -> HResult {
    active_or_next!(ctx);
    let active = ctx.warp.active;
    let n = args.ty.mem_bytes() as u64;
    let mut cmds = Vec::new();
    let mut all_targets = Vec::new();
    for l in active.lanes() {
        let loc = support::resolve(ctx, AddrSpace::SharedCluster, lane_val(ctx, args.addr, l), l, n)?;
        super::mem::check_align(ctx, loc, n, l)?;
        let mut b = [0u8; 32];
        lane_bytes(ctx, args.value, args.ty, l, &mut b);
        let dst = vec![(loc.alloc, loc.span(n))];
        let bytes = b[..n as usize].to_vec();
        let mut signals = Vec::new();
        let mut targets = Vec::new();
        // `st.async.release` without an mbarrier: ordered by `sem` only.
        if let Some(mbar) = args.mbar {
            let res = mbar_res(ctx, AddrSpace::SharedCluster, lane_val(ctx, mbar, l), l)?;
            bind_mbar_tx(ctx, &[res], n, &mut signals, &mut targets, &mut cmds)?;
        }
        let payload = match args.red {
            Some(op) => Payload::ReduceData { op, dtype: args.ty.elem, dst, bytes },
            None => Payload::Data { dst, bytes },
        };
        all_targets.extend(targets.iter().copied());
        issue_async(
            ctx,
            WarpMask::lane(l),
            Issue {
                kind: AsyncKind::StAsync,
                class: AsyncClass::Copy,
                proxy: Proxy::Async,
                payload,
                signals,
                after: Vec::new(),
                targets,
                queue: true,
                fill_pattern: Vec::new(),
                tf32_round: false,
            },
        );
    }
    let extra = ProtoExtra { issued: all_targets, ..Default::default() };
    support::protocol(ctx, active, cmds, extra);
    Ok(Flow::Next)
}

#[inline]
pub fn tensormap_replace(ctx: &mut ExecCtx<'_>, tmap: Operand, space: AddrSpace, field: TmapField, ord: Option<u8>, value: Operand) -> HResult {
    active_or_next!(ctx);
    let mut racc = Accesses::default();
    let mut wacc = Accesses::default();
    for l in ctx.warp.active.lanes() {
        let (mut d, loc) = read_tmap(ctx, tmap, space, l, &mut racc)?;
        d.replace(field, ord, lane_val(ctx, value, l)).map_err(|e| support::op_err(ctx, e))?;
        let b = d.try_encode().map_err(|e| support::op_err(ctx, e))?;
        support::mem_write(ctx, loc, l, &b)?;
        if ctx.observing {
            wacc.push(loc, l as u8, 128);
        }
    }
    let sp = support::spec(ctx, AccessKind::Read, Sem::Weak, Scope::Cta, Proxy::Generic);
    support::emit(ctx, sp, &mut racc);
    let sp = support::spec(ctx, AccessKind::Write, Sem::Weak, Scope::Cta, Proxy::Generic);
    support::emit(ctx, sp, &mut wacc);
    Ok(Flow::Next)
}

#[inline]
pub fn tensormap_cp_fence(ctx: &mut ExecCtx<'_>, dst: Operand, src: Operand, size: u32, scope: Scope) -> HResult {
    active_or_next!(ctx);
    let n = size as u64;
    let mut racc = Accesses::default();
    let mut wacc = Accesses::default();
    for l in ctx.warp.active.lanes() {
        let sa = lane_val(ctx, src, l);
        let sspace = if sa >> 32 == 0 { AddrSpace::Shared } else { AddrSpace::Generic };
        let sl = support::resolve(ctx, sspace, sa, l, n)?;
        let dl = support::resolve(ctx, AddrSpace::Generic, lane_val(ctx, dst, l), l, n)?;
        let mut b = vec![0u8; n as usize];
        support::mem_read(ctx, sl, l, &mut b)?;
        support::mem_write(ctx, dl, l, &b)?;
        if ctx.observing {
            racc.push(sl, l as u8, n);
            wacc.push(dl, l as u8, n);
        }
    }
    let sp = support::spec(ctx, AccessKind::Read, Sem::Weak, scope, Proxy::Generic);
    support::emit(ctx, sp, &mut racc);
    let sp = support::spec(ctx, AccessKind::Write, Sem::Weak, scope, Proxy::Generic);
    support::emit(ctx, sp, &mut wacc);
    let lanes = ctx.warp.active;
    support::sync_event(ctx, lanes, SyncKind::Fence(FenceEvent::TensormapRelease { scope }));
    Ok(Flow::Next)
}
