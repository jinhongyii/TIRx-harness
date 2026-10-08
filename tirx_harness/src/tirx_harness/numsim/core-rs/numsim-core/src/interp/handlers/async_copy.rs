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
    /// `_report` copy forms (layout::v1 completion mbarriers).
    pub report: Option<ReportMode>,
    /// tcgen05.mma `.lut_b`: TMEM address of the lookup table.
    pub lut_b: Option<u32>,
    /// `st.async` / `red.async`: the landing is a strong release write in
    /// the generic proxy at this scope (W5-8).
    pub strong: Option<Scope>,
}

/// Issue an async op: allocate its id, emit `AsyncIssue`, queue it.
pub fn issue_async(ctx: &mut ExecCtx<'_>, lanes: WarpMask, is: Issue) -> AsyncId {
    let id = ctx.aux.next_async_id();
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
        AsyncMeta {
            lane,
            class: is.class,
            proxy: is.proxy,
            phases,
            fill_pattern: is.fill_pattern,
            tf32_round: is.tf32_round,
            report: is.report,
            lut_b: is.lut_b,
            strong: is.strong,
            bit_frags: Vec::new(),
            dead: Vec::new(),
            a_reads: Vec::new(),
            is_a_read: false,
        },
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
    // sync-semantics §2.1: an EXPLICIT `.shared::cta` mbarrier operand must
    // name the executing CTA; a remote-rank (`mapa`) address there is an
    // error. Unqualified (generic) and `.shared::cluster` forms accept it.
    if space == AddrSpace::Shared {
        let (rank, _) = crate::arena::addr::decode_shared(a as u32);
        if rank != ctx.cta.rank_in_cluster {
            return Err(support::err(
                ctx,
                ExecErrorKind::BadAddress,
                WarpMask::lane(lane),
                format!("shared::cta mbarrier address {a:#x} names CTA rank {rank}, not the executing CTA (rank {})", ctx.cta.rank_in_cluster),
            ));
        }
    }
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
/// CTA ranks of a multicast mask. A bit naming a rank outside the cluster
/// is a kernel bug (BadAddress), not a silently dropped target.
pub fn ranks_of(ctx: &ExecCtx<'_>, mask: u64) -> Result<Vec<u32>, ExecError> {
    let n = ctx.launch.ctas_per_cluster().min(64);
    let outside = if n >= 64 { 0 } else { mask >> n };
    if outside != 0 {
        return Err(ctx.error(
            ExecErrorKind::BadAddress,
            format!("multicast CTA mask {mask:#x} names ranks outside the {n}-CTA cluster"),
        ));
    }
    Ok((0..n).filter(|r| mask >> r & 1 == 1).collect())
}

/// Is `loc` a shared-window location (own or cluster peer)?
fn smem_loc_in_rank(ctx: &ExecCtx<'_>, loc: Loc, rank: u32) -> Option<(AllocId, u64)> {
    Some((*ctx.cta.cluster_smem.get(rank as usize)?, loc.offset))
}

pub fn group_res(ctx: &ExecCtx<'_>, lane: usize, domain: Domain) -> ResourceId {
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
                report: None,
                lut_b: None,
                strong: None,
            },
        );
        ctx.aux.groups.issue(group_res(ctx, l, Domain::CpAsync), op);
        ctx.aux.cp_async_unpublished.entry((ctx.warp.id, l as u8)).or_default().push(op);
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
    // Ops of the awaited prefix, per lane (for the AsyncComplete events):
    // exactly the groups `wait_prefix_len` covers, never the `n` youngest.
    // All lanes or none (a partial retirement would lose its events).
    let mut covered: Vec<(AsyncId, WarpMask)> = Vec::new();
    let mut prefixes: Vec<(usize, ResourceId, Vec<u64>)> = Vec::new();
    for l in active.lanes() {
        let res = group_res(ctx, l, domain);
        let prefix: Vec<u64> = match ctx.sync.get(res) {
            Some(crate::sync::Resource::AsyncGroup(s)) => {
                let k = async_group::wait_prefix_len(s, n as u64);
                s.groups.iter().take(k).map(|g| g.ordinal).collect()
            }
            _ => Vec::new(),
        };
        prefixes.push((l, res, prefix));
    }
    let all: Vec<(ResourceId, SyncCmd)> = prefixes.iter().map(|(_, r, _)| (*r, cmd)).collect();
    if let Step::Blocked(r) = support::step_all(ctx, &all)? {
        return Ok(Flow::Blocked(r));
    }
    for (l, res, prefix) in prefixes {
        for ord in prefix {
            let key = (res, ord);
            // A full wait retires the group: its bookkeeping goes too.
            let ops = if read { ctx.aux.groups.members.get(&key).cloned() } else { ctx.aux.groups.members.remove(&key) };
            if !ctx.observing {
                continue;
            }
            for op in ops.unwrap_or_default() {
                match covered.iter_mut().find(|(o, _)| *o == op) {
                    Some((_, m)) => *m = m.or(WarpMask::lane(l)),
                    None => covered.push((op, WarpMask::lane(l))),
                }
            }
        }
    }
    covered.sort_by_key(|(op, _)| *op);
    let cmds: Vec<(ResourceId, SyncCmd)> = active.lanes().map(|l| (group_res(ctx, l, domain), cmd)).collect();
    support::protocol(ctx, active, cmds, ProtoExtra::default());
    let milestone = if read { Side::Read } else { Side::Write };
    for (op, lanes) in covered {
        let target = PublishTarget::Warp { warp: ctx.warp.id, lanes };
        support::sync_event(ctx, lanes, SyncKind::AsyncComplete { op, milestone, target });
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
        // W5-11: when the arrive fires, every prior cp.async of this lane
        // is published to that phase (AsyncComplete{Write, Phase}).
        let prior = ctx.aux.cp_async_unpublished.remove(&(ctx.warp.id, l as u8)).unwrap_or_default();
        match out {
            Step::Done(Outcome::AsyncGroup(async_group::Outcome::ArriveOn { group: Some(ord) })) => {
                if ctx.aux.groups.open.get(&gres).is_some_and(|v| !v.is_empty()) {
                    // The open issues became a non-closing batch.
                    let due = ctx.aux.groups.commit(gres, ord);
                    ctx.sync.completions.extend(due);
                }
                ctx.aux.groups.arrivals.entry((gres, ord)).or_default().push(arrival);
                ctx.aux.cp_arrive_publish.entry((gres, ord)).or_default().push((res, gen, prior));
            }
            Step::Done(_) => {
                ctx.sync.completions.push_back(arrival);
                // Nothing pending: the prior copies already landed.
                for op in prior {
                    let target = PublishTarget::Phase { obj: res, phase: gen };
                    support::async_event(ctx, op, SyncKind::AsyncComplete { op, milestone: Side::Write, target });
                }
            }
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
    check_report(ctx, args.report, &args.completion)?;
    let mut cmds = Vec::new();
    let mut issued = Vec::new();
    for l in active.lanes() {
        let size = lane_val(ctx, args.size, l);
        let sa = lane_val(ctx, args.src, l);
        // `.ignore_oob`: the first `left` and last `right` source bytes are
        // not read and their destination bytes are left unchanged (tx still
        // counts `size`).
        let (left, right) = match args.ignore_oob {
            Some(io) => {
                let left = io.ignore_bytes_left.map(|o| lane_val(ctx, o, l)).unwrap_or(0);
                let right = io.ignore_bytes_right.map(|o| lane_val(ctx, o, l)).unwrap_or(0);
                // PTX: each ignored-byte count must be in 0..=15.
                for (what, v) in [("ignoreBytesLeft", left), ("ignoreBytesRight", right)] {
                    if v > 15 {
                        return Err(support::err(
                            ctx,
                            ExecErrorKind::Op(crate::oplib::OpErrorKind::Invalid),
                            WarpMask::lane(l),
                            format!("cp.async.bulk .ignore_oob {what} = {v} must be in 0..=15"),
                        ));
                    }
                }
                (left.min(size), right)
            }
            None => (0, 0),
        };
        let copy = size.saturating_sub(left).saturating_sub(right);
        let sl = support::resolve(ctx, args.src_space, sa.wrapping_add(left), l, copy)?;
        let dl = support::resolve(ctx, args.dst_space, lane_val(ctx, args.dst, l), l, size)?;
        let mask = args.byte_mask.map(|m| lane_val(ctx, m, l) as u16);
        // Destination CTAs (multicast: same offset in every masked CTA).
        let mut dsts: Vec<(AllocId, ByteSpan)> = Vec::new();
        let mut ranks: Vec<Option<u32>> = Vec::new();
        match args.multicast {
            Some(m) => {
                for r in ranks_of(ctx, lane_val(ctx, m, l))? {
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
        // Byte runs actually copied (byte mask per 16-byte chunk of the
        // full range, minus the ignored edges).
        let runs: Vec<(u64, u64)> = byte_runs(size, mask)
            .into_iter()
            .filter_map(|(o, n)| {
                let lo = o.max(left);
                let hi = (o + n).min(left + copy);
                (lo < hi).then_some((lo, hi - lo))
            })
            .collect();
        let mut srcs: Vec<(AllocId, ByteSpan)> = Vec::new();
        let mut dsts2: Vec<(AllocId, ByteSpan)> = Vec::new();
        for (da, ds) in &dsts {
            for &(o, n) in &runs {
                srcs.push((sl.alloc, ByteSpan::new(sl.offset + (o - left), n)));
                dsts2.push((*da, ByteSpan::new(ds.start + o, n)));
            }
        }
        // `.ignore_oob` dead bytes (outside the copied middle, inside the
        // byte mask): legacy writes them as zero with validity cleared.
        let mut dead: Vec<(AllocId, ByteSpan)> = Vec::new();
        if args.ignore_oob.is_some() && args.reduce.is_none() {
            for (o, n) in byte_runs(size, mask) {
                for (lo, hi) in [(o, (o + n).min(left)), (o.max(left + copy), o + n)] {
                    if lo < hi {
                        for (da, ds) in &dsts {
                            dead.push((*da, ByteSpan::new(ds.start + lo, hi - lo)));
                        }
                    }
                }
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
                if args.report.is_some() {
                    require_layout_v1(ctx, res)?;
                }
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
            Issue {
                kind,
                class: AsyncClass::Copy,
                proxy: Proxy::Async,
                payload,
                signals,
                after: Vec::new(),
                targets,
                queue: true,
                fill_pattern: Vec::new(),
                tf32_round: false,
                report: args.report,
                lut_b: None,
                strong: None,
            },
        );
        if !dead.is_empty() {
            if let Some(m) = ctx.aux.async_meta.get_mut(&op) {
                m.dead = dead;
            }
        }
        if let Some(g) = group {
            ctx.aux.groups.issue(g, op);
        }
    }
    let extra = ProtoExtra { issued, ..Default::default() };
    support::protocol(ctx, active, cmds, extra);
    Ok(Flow::Next)
}

/// `_report` copy forms: completion must be an mbarrier; the pattern of
/// `.per_16bytes` is not carried by `ReportMode` yet (W2-8), fail closed.
fn check_report(ctx: &ExecCtx<'_>, report: Option<ReportMode>, completion: &BulkCompletion) -> Result<(), ExecError> {
    match (report, completion) {
        (None, _) => Ok(()),
        (Some(ReportMode::Per16Bytes), _) => {
            Err(support::unsupported(ctx, "copy report .per_16bytes (pattern not in the contract, W2-8)"))
        }
        (Some(_), BulkCompletion::Group) => Err(ctx.error(
            ExecErrorKind::Op(crate::oplib::OpErrorKind::Invalid),
            "copy report forms require an mbarrier completion",
        )),
        (Some(ReportMode::PerElementFf), BulkCompletion::Mbarrier { .. }) => Ok(()),
    }
}

/// The completion mbarrier of a report copy must use layout::v1.
pub fn require_layout_v1(ctx: &ExecCtx<'_>, res: ResourceId) -> Result<(), ExecError> {
    match ctx.sync.get(res) {
        Some(crate::sync::Resource::Mbarrier(s)) if s.live && s.layout_v1 => Ok(()),
        _ => Err(ctx.error(
            ExecErrorKind::Op(crate::oplib::OpErrorKind::Invalid),
            "copy reporting requires an initialized mbarrier with layout::v1",
        )),
    }
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
    check_report(ctx, args.report, &args.completion)?;
    let active = ctx.warp.active;
    let mut cmds = Vec::new();
    let mut issued = Vec::new();
    let mut acc = Accesses::default();
    for l in active.lanes() {
        let (mut desc, tloc) = read_tmap(ctx, args.tmap, args.tmap_space, l, &mut acc)?;
        // A descriptor modified by tensormap.replace must be published by a
        // fence.proxy.tensormap::generic.release before the tensormap proxy
        // reads it (legacy rejected the unpublished use).
        if ctx.aux.tmap_dirty.contains_key(&(tloc.alloc, tloc.offset)) {
            return Err(support::err(
                ctx,
                ExecErrorKind::Op(crate::oplib::OpErrorKind::Invalid),
                WarpMask::lane(l),
                "tensor map modified by tensormap.replace is used before a fence.proxy.tensormap::generic.release published it (dirty descriptor)",
            ));
        }
        if !args.overrides.is_empty() {
            let ov: Vec<_> = args.overrides.iter().map(|o| (o.field, o.ord, lane_val(ctx, o.value, l))).collect();
            desc.apply_overrides(&ov).map_err(|e| support::op_err(ctx, e))?;
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
        let mut bit_frags = Vec::with_capacity(plan.global_bits.len());
        for f in &plan.global_bits {
            let (alloc, off) = ctx.arena.resolve_global(f.global, 1).map_err(|e| support::arena_err(ctx, e, WarpMask::lane(l)))?;
            bit_frags.push(crate::interp::aux::BitFrag {
                global: (alloc, off),
                smem: (smem_alloc, f.smem),
                src_shift: f.source_shift,
                tgt_shift: f.target_shift,
                mask: f.mask,
            });
        }
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
            (TmaDir::Load, Some(m)) => ranks_of(ctx, lane_val(ctx, m, l))?.into_iter().map(Some).collect(),
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
                crate::oplib::tma_reduce_valid(op, dtype).map_err(|e| support::op_err(ctx, e))?;
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
                // `.cta_group::2`: the mbarrier operand is a shared::cluster
                // address whose CTA-id bit 0 picks the CTA of each
                // destination's pair that is signalled (kernels mask bit 24
                // to signal the leader CTA): signal rank = (dst & !1) |
                // (mbar rank & 1). Otherwise the same offset in each
                // destination CTA.
                let a = lane_val(ctx, mbar, l);
                let pair = args.cta_group == 2;
                let res = mbar_res(ctx, if pair { AddrSpace::SharedCluster } else { space }, a, l)?;
                if args.report.is_some() {
                    require_layout_v1(ctx, res)?;
                }
                let mbar_rank = crate::arena::addr::decode_shared(a as u32).0;
                let own = ctx.cta.rank_in_cluster;
                let list: Vec<ResourceId> = ranks
                    .iter()
                    .map(|r| match (r, pair) {
                        (Some(r), true) => mbar_in_rank(ctx, res, (*r & !1) | (mbar_rank & 1)).unwrap_or(res),
                        (None, true) => mbar_in_rank(ctx, res, (own & !1) | (mbar_rank & 1)).unwrap_or(res),
                        (Some(r), false) => mbar_in_rank(ctx, res, *r).unwrap_or(res),
                        (None, false) => res,
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
            Issue {
                kind,
                class: AsyncClass::Copy,
                proxy: Proxy::Async,
                payload,
                signals,
                after: Vec::new(),
                targets,
                queue: true,
                fill_pattern,
                tf32_round,
                report: args.report,
                lut_b: None,
                strong: None,
            },
        );
        if !bit_frags.is_empty() {
            if let Some(m) = ctx.aux.async_meta.get_mut(&op) {
                m.bit_frags = bit_frags;
            }
        }
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
        // `st.async` / `red.async` target `.shared::cluster` (32-bit
        // window address) or, in the `.release.<scope>.global` form, global
        // memory (a 64-bit address): never resolve a global address as shared.
        let a = lane_val(ctx, args.addr, l);
        let space = if a >> 32 != 0 { AddrSpace::Generic } else { AddrSpace::SharedCluster };
        let loc = support::resolve(ctx, space, a, l, n)?;
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
                // Performed in the generic proxy (PTX §9.7.10.12, W5-8).
                proxy: Proxy::Generic,
                payload,
                signals,
                after: Vec::new(),
                targets,
                queue: true,
                fill_pattern: Vec::new(),
                tf32_round: false,
                report: None,
                lut_b: None,
                strong: Some(args.scope),
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
        ctx.aux.tmap_dirty.insert((loc.alloc, loc.offset), ctx.warp.id);
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
        // `tensormap.cp_fenceproxy` publishes the copy (release).
        ctx.aux.tmap_dirty.remove(&(dl.alloc, dl.offset));
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
