//! Barriers, mbarriers, fences, setmaxnreg, wait_until.
//!
//! Lane aggregation and target resolution happen here; protocol state
//! changes only through `SyncTable::step`/`step_all`. A blocking command
//! returns `Flow::Blocked(resource)` and is retried at the same pc; a
//! registration half (named-barrier generation, setmaxnreg epoch, grid
//! generation) is kept in `WarpState::resume` so retries never
//! re-register. Events follow racecheck-semantics §3 (rows 7-11, 26-33)
//! and the W6 `Protocol` shape.

use super::async_copy::{issue_async, mbar_in_rank, mbar_issue, mbar_res, ranks_of, Issue};
use super::HResult;
use crate::arena::ByteSpan;
use crate::dtype::Ty;
use crate::interp::support::{self, lane_val, uniform_over, write_lane, Accesses, ProtoExtra};
use crate::interp::{ExecCtx, ExecError, ExecErrorKind, Flow};
use crate::observe::{AccessKind, AsyncClass, AsyncTarget, Collective, Counts, FenceEvent, LaneVerdict, SyncKind, WarpId, Window};
use crate::program::*;
use crate::sync::completion::{AsyncKind, Payload};
use crate::sync::{cluster, mbarrier, named, query, setmaxnreg, Completion, Outcome, ResourceId, Step, SyncCmd};
use crate::value::WarpMask;

fn internal(ctx: &ExecCtx<'_>, what: &str, o: impl std::fmt::Debug) -> ExecError {
    ctx.error(ExecErrorKind::Internal, format!("{what}: unexpected outcome {o:?}"))
}

// ---------------------------------------------------------------------------
// Named barriers
// ---------------------------------------------------------------------------

/// Finish a `bar.sync`/`bar.red` whose generation `gen` completed.
fn bar_ready(ctx: &mut ExecCtx<'_>, kind: BarKind, res: ResourceId, id: u8, gen: u64, aligned: bool) -> Result<(), ExecError> {
    let active = ctx.warp.active;
    support::sync_event(ctx, active, SyncKind::Wait { obj: res, phase: gen, acquire: Some(true), scope: None });
    if let BarKind::Red { op, dst, .. } = kind {
        let acc = ctx.aux.bar_red.get(&(ctx.cta.id, id, gen)).copied().unwrap_or_default();
        let v = match op {
            BarRedOp::Popc => acc.popc,
            BarRedOp::And => acc.all as u64,
            BarRedOp::Or => acc.any as u64,
        };
        for l in active.lanes() {
            write_lane(ctx, dst, l, v);
        }
    }
    // An aligned full-warp bar.sync of all warps of a warpgroup credits the
    // setmaxnreg warpgroup sync.
    if ctx.loaded.uses_setmaxnreg && aligned && matches!(kind, BarKind::Sync) && active == ctx.warp.live {
        let key = (ctx.cta.id, id, gen);
        let w = ctx.warp.warp_in_cta;
        let bits = ctx.aux.bar_aligned.entry(key).or_default();
        *bits |= 1u64 << (w.min(63));
        let bits = *bits;
        let wg = w / 4;
        let wpc = ctx.launch.warps_per_cta();
        let n = (wpc - 4 * wg).min(4);
        let need = ((1u64 << n) - 1) << (4 * wg);
        if bits & need == need && ctx.aux.wg_credited.insert((ctx.cta.id, id, gen, wg)) {
            let r = ResourceId::RegPool { cta: ctx.cta.id };
            let c = SyncCmd::RegPool(setmaxnreg::Cmd::WarpgroupSync { wg });
            support::step(ctx, r, c)?;
            support::protocol(ctx, active, vec![(r, c)], ProtoExtra::default());
        }
    }
    Ok(())
}

#[inline]
pub fn barrier(ctx: &mut ExecCtx<'_>, kind: BarKind, id: Operand, count: Option<Operand>, aligned: bool) -> HResult {
    active_or_next!(ctx);
    let active = ctx.warp.active;
    let idv = uniform_over(ctx, id, active)?;
    if idv >= named::NUM_IDS as u64 {
        return Err(ctx.error(ExecErrorKind::Protocol(crate::sync::SyncError::Named(named::Error::InvalidCount { count: idv })), format!("barrier id {idv} >= 16")));
    }
    let idb = idv as u8;
    let res = ResourceId::Named { cta: ctx.cta.id, id: idb };
    // The count-less form waits for every non-exited thread of the CTA.
    let b = match count {
        Some(c) => uniform_over(ctx, c, active)?,
        None => {
            let gone = ctx.aux.cta_exited.get(&ctx.cta.id).map_or(0, |m| m.count_ones());
            (ctx.launch.warps_per_cta() - gone) as u64 * 32
        }
    };
    let b = match ctx.warp.resume {
        Some(_) => ctx.aux.named_registered.get(&ctx.warp.id).copied().unwrap_or(b),
        None => b,
    };
    // Non-`.aligned` forms under a partial mask (or while the warp's partial
    // arrival is in progress): the executing lanes wait for the rest of the
    // warp (Q3 ruling).
    if !aligned && ctx.warp.resume.is_none() && (active != ctx.warp.live || ctx.aux.named_partial.contains_key(&ctx.warp.id)) {
        return barrier_partial(ctx, kind, res, idb, b, count.is_none());
    }
    let extra = || ProtoExtra {
        counts: Counts { expected_threads: Some(b), contributed_threads: Some(named::WARP_SIZE), ..Default::default() },
        ..Default::default()
    };
    let contribution = named::Contribution { warp: ctx.warp.warp_in_cta, mask: active.bits(), live: ctx.warp.live.bits(), count: b, aligned };
    let cmd = SyncCmd::Named(match kind {
        BarKind::Sync => named::Cmd::Sync(contribution),
        BarKind::Arrive => named::Cmd::Arrive(contribution),
        BarKind::Red { .. } => named::Cmd::Red(contribution),
    });
    if let Some(gen) = ctx.warp.resume {
        // Retry of a registered Sync/Red. The instruction's one Protocol
        // event (its contribution) is logged when it completes.
        let resume = SyncCmd::Named(named::Cmd::Resume { gen });
        return match support::step(ctx, res, resume)? {
            Step::Blocked(r) => Ok(Flow::Blocked(r)),
            Step::Done(_) => {
                ctx.warp.resume = None;
                ctx.aux.named_registered.remove(&ctx.warp.id);
                support::protocol(ctx, active, vec![(res, cmd)], extra());
                bar_ready(ctx, kind, res, idb, gen, aligned)?;
                Ok(Flow::Next)
            }
        };
    }
    let out = support::step(ctx, res, cmd)?;
    let Step::Done(Outcome::Named(o)) = out else { return Err(internal(ctx, "named barrier", out)) };
    let gen = match o {
        named::Outcome::Arrived { gen, .. } | named::Outcome::Registered { gen } | named::Outcome::Ready { gen } => gen,
        named::Outcome::Blocked => return Err(internal(ctx, "named barrier", o)),
    };
    match count {
        None => {
            ctx.aux.named_implicit.insert((ctx.cta.id, idb), gen);
        }
        Some(_) => {
            ctx.aux.named_implicit.remove(&(ctx.cta.id, idb));
        }
    }
    if let BarKind::Red { pred, .. } = kind {
        let yes = super::control::cond_mask(ctx, pred, active);
        let a = ctx.aux.bar_red.entry((ctx.cta.id, idb, gen)).or_default();
        if !a.started {
            a.all = true;
            a.started = true;
        }
        a.popc += yes.count() as u64;
        a.all &= yes == active;
        a.any |= !yes.is_empty();
    }
    support::sync_event(ctx, active, SyncKind::Arrive { obj: res, phase: gen, release: Some(true), scope: None });
    match o {
        named::Outcome::Arrived { .. } => {
            support::protocol(ctx, active, vec![(res, cmd)], extra());
            Ok(Flow::Next)
        }
        named::Outcome::Ready { gen } => {
            support::protocol(ctx, active, vec![(res, cmd)], extra());
            bar_ready(ctx, kind, res, idb, gen, aligned)?;
            Ok(Flow::Next)
        }
        named::Outcome::Registered { gen } => {
            ctx.warp.resume = Some(gen);
            ctx.aux.named_registered.insert(ctx.warp.id, b);
            Ok(Flow::Blocked(res))
        }
        named::Outcome::Blocked => unreachable!(),
    }
}

fn bar_flavor(kind: BarKind) -> u8 {
    match kind {
        BarKind::Arrive => 0,
        BarKind::Sync => 1,
        BarKind::Red { .. } => 2,
    }
}

/// The fail-closed Q3/Q5 case: lanes of a warp at a non-`.aligned` barrier
/// whose missing lanes reached a different barrier (id or flavor) or exited.
pub(crate) fn named_partial_error(ctx: &ExecCtx<'_>, mask: WarpMask, live: WarpMask, why: &str) -> ExecError {
    ctx.error(
        ExecErrorKind::Protocol(crate::sync::SyncError::Named(named::Error::PartialWarp { mask: mask.bits(), live: live.bits() })),
        format!("named barrier reached by lanes {:#010x} of the warp (non-exited {:#010x}); {why}", mask.bits(), live.bits()),
    )
}

/// Lanes `lanes` passed the warp's partial-arrival barrier.
fn partial_done(ctx: &mut ExecCtx<'_>, lanes: WarpMask) {
    let w = ctx.warp.id;
    if let Some(p) = ctx.aux.named_partial.get_mut(&w) {
        p.waiting = p.waiting.and_not(lanes);
        if p.waiting.is_empty() {
            ctx.aux.named_partial.remove(&w);
        }
    }
}

/// A non-`.aligned` `barrier.{sync,arrive,red}` executed by a strict subset
/// of the warp's non-exited lanes, or by lanes joining / retrying such an
/// arrival. The lanes block until every non-exited lane has executed a
/// barrier with the same id and flavor (any site); then the last group
/// makes ONE warp arrival (full mask) and each group continues once the
/// generation completes (`arrive`: at once). Missing lanes reaching a
/// different barrier, or exiting, fail closed (`PartialWarp`).
fn barrier_partial(ctx: &mut ExecCtx<'_>, kind: BarKind, res: ResourceId, idb: u8, b: u64, implicit: bool) -> HResult {
    let active = ctx.warp.active;
    let live = ctx.warp.live;
    let flavor = bar_flavor(kind);
    let prev = ctx.aux.named_partial.get(&ctx.warp.id).cloned();
    if let Some(p) = &prev {
        if p.waiting.and(active) == active {
            // Retry of a group that already executed the barrier.
            let Some(gen) = p.gen else { return Ok(Flow::Blocked(res)) };
            if flavor != 0 {
                if let Step::Blocked(r) = support::step(ctx, res, SyncCmd::Named(named::Cmd::Resume { gen }))? {
                    return Ok(Flow::Blocked(r));
                }
                bar_ready(ctx, kind, res, idb, gen, false)?;
            }
            partial_done(ctx, active);
            return Ok(Flow::Next);
        }
        if p.gen.is_some() {
            // Lanes past the barrier reached it again before the rest of the
            // warp left the previous generation: wait for them.
            return Ok(Flow::Blocked(res));
        }
        if p.id != idb || p.flavor != flavor {
            return Err(named_partial_error(ctx, p.lanes, live, &format!("the other lanes reached barrier {} (flavor {})", idb, flavor)));
        }
        if p.count != b {
            return Err(ctx.error(
                ExecErrorKind::Protocol(crate::sync::SyncError::Named(named::Error::ContractMismatch { expected: p.count, observed: b })),
                format!("lanes of one warp reached barrier {idb} with thread counts {} and {b}", p.count),
            ));
        }
    }
    let mut p = prev.unwrap_or(crate::interp::aux::NamedPartial {
        id: idb,
        flavor,
        count: b,
        lanes: WarpMask::NONE,
        waiting: WarpMask::NONE,
        gen: None,
        red: Default::default(),
    });
    p.lanes = p.lanes.or(active);
    p.waiting = p.waiting.or(active);
    if let BarKind::Red { pred, .. } = kind {
        let yes = super::control::cond_mask(ctx, pred, active);
        if !p.red.started {
            p.red.all = true;
            p.red.started = true;
        }
        p.red.popc += yes.count() as u64;
        p.red.all &= yes == active;
        p.red.any |= !yes.is_empty();
    }
    if p.lanes.and(live) != live {
        ctx.aux.named_partial.insert(ctx.warp.id, p);
        return Ok(Flow::Blocked(res));
    }
    // Every non-exited lane is here: one warp arrival.
    let contribution = named::Contribution { warp: ctx.warp.warp_in_cta, mask: live.bits(), live: live.bits(), count: b, aligned: false };
    let cmd = SyncCmd::Named(match kind {
        BarKind::Sync => named::Cmd::Sync(contribution),
        BarKind::Arrive => named::Cmd::Arrive(contribution),
        BarKind::Red { .. } => named::Cmd::Red(contribution),
    });
    let out = support::step(ctx, res, cmd)?;
    let Step::Done(Outcome::Named(o)) = out else { return Err(internal(ctx, "named barrier", out)) };
    let gen = match o {
        named::Outcome::Arrived { gen, .. } | named::Outcome::Registered { gen } | named::Outcome::Ready { gen } => gen,
        named::Outcome::Blocked => return Err(internal(ctx, "named barrier", o)),
    };
    if implicit {
        ctx.aux.named_implicit.insert((ctx.cta.id, idb), gen);
    } else {
        ctx.aux.named_implicit.remove(&(ctx.cta.id, idb));
    }
    if flavor == 2 {
        let a = ctx.aux.bar_red.entry((ctx.cta.id, idb, gen)).or_default();
        if !a.started {
            a.all = true;
            a.started = true;
        }
        a.popc += p.red.popc;
        a.all &= p.red.all;
        a.any |= p.red.any;
    }
    let lanes = p.lanes.and(live);
    p.gen = Some(gen);
    ctx.aux.named_partial.insert(ctx.warp.id, p);
    support::sync_event(ctx, lanes, SyncKind::Arrive { obj: res, phase: gen, release: Some(true), scope: None });
    let extra = ProtoExtra {
        counts: Counts { expected_threads: Some(b), contributed_threads: Some(named::WARP_SIZE), ..Default::default() },
        ..Default::default()
    };
    support::protocol(ctx, live, vec![(res, cmd)], extra);
    match o {
        named::Outcome::Arrived { .. } => {
            partial_done(ctx, active);
            Ok(Flow::Next)
        }
        named::Outcome::Ready { gen } => {
            bar_ready(ctx, kind, res, idb, gen, false)?;
            partial_done(ctx, active);
            Ok(Flow::Next)
        }
        _ => Ok(Flow::Blocked(res)),
    }
}

// ---------------------------------------------------------------------------
// Cluster / grid barriers
// ---------------------------------------------------------------------------

fn cluster_warp(ctx: &ExecCtx<'_>) -> u32 {
    ctx.cta.rank_in_cluster * ctx.launch.warps_per_cta() + ctx.warp.warp_in_cta
}

#[inline]
pub fn cluster_arrive(ctx: &mut ExecCtx<'_>, sem: Sem, aligned: bool) -> HResult {
    active_or_next!(ctx);
    if ctx.warp.active != ctx.warp.live || ctx.aux.cluster_partial.contains_key(&ctx.warp.id) {
        return cluster_partial(ctx, false, sem != Sem::Relaxed, aligned);
    }
    let active = ctx.warp.active;
    let res = ResourceId::Cluster { cluster: ctx.cta.cluster };
    let cmd = SyncCmd::Cluster(cluster::Cmd::Arrive { warp: cluster_warp(ctx), mask: active.bits(), aligned });
    let out = support::step(ctx, res, cmd)?;
    let Step::Done(Outcome::Cluster(cluster::Outcome::Arrived { gen, .. })) = out else {
        return Err(internal(ctx, "cluster arrive", out));
    };
    let participants = ctx.launch.ctas_per_cluster() * ctx.launch.warps_per_cta();
    let extra = ProtoExtra { counts: Counts { participants: Some(participants), ..Default::default() }, ..Default::default() };
    support::protocol(ctx, active, vec![(res, cmd)], extra);
    support::sync_event(ctx, active, SyncKind::Arrive { obj: res, phase: gen, release: Some(sem != Sem::Relaxed), scope: Some(Scope::Cluster) });
    Ok(Flow::Next)
}

#[inline]
pub fn cluster_wait(ctx: &mut ExecCtx<'_>, acquire: bool, aligned: bool) -> HResult {
    active_or_next!(ctx);
    let _ = acquire; // the wait always acquires (sync-semantics §4.1)
    if ctx.warp.active != ctx.warp.live || ctx.aux.cluster_partial.contains_key(&ctx.warp.id) {
        return cluster_partial(ctx, true, true, aligned);
    }
    let active = ctx.warp.active;
    let res = ResourceId::Cluster { cluster: ctx.cta.cluster };
    let cmd = SyncCmd::Cluster(cluster::Cmd::Wait { warp: cluster_warp(ctx), mask: active.bits(), aligned });
    match support::step(ctx, res, cmd)? {
        Step::Blocked(r) => Ok(Flow::Blocked(r)),
        Step::Done(Outcome::Cluster(cluster::Outcome::Ready { gen })) => {
            support::protocol(ctx, active, vec![(res, cmd)], ProtoExtra::default());
            support::sync_event(ctx, active, SyncKind::Wait { obj: res, phase: gen, acquire: Some(true), scope: Some(Scope::Cluster) });
            Ok(Flow::Next)
        }
        out => Err(internal(ctx, "cluster wait", out)),
    }
}

/// `barrier.cluster.{arrive,wait}` executed by part of a warp, or by lanes
/// joining / retrying such a gather (sync §4.6, Q11; W6). The lanes wait
/// for the rest of the warp to execute the same kind (any site); then the
/// warp makes ONE command with the full live mask, and ONE `Arrive` / `Wait`
/// event names every gathered lane (checker-review §3). `.aligned` partial
/// forms, missing lanes that exit or run the other kind, and a second
/// execution before completion fail closed (`cluster::gather`).
fn cluster_partial(ctx: &mut ExecCtx<'_>, wait: bool, release: bool, aligned: bool) -> HResult {
    use crate::interp::aux::{ClusterPartial, ClusterPass};
    let active = ctx.warp.active;
    let live = ctx.warp.live;
    let res = ResourceId::Cluster { cluster: ctx.cta.cluster };
    let k = wait as usize;
    let cw = cluster_warp(ctx);
    let wid = ctx.warp.id;
    let entry = ctx.aux.cluster_partial.entry(wid).or_insert_with(|| ClusterPartial { gather: cluster::Gather { warp: cw, pending: None }, pass: [None, None] });
    // Retry by lanes already gathered into a completed command of this kind.
    if let Some(p) = entry.pass[k].filter(|p| p.waiting.and(active) == active) {
        if wait && !p.passed {
            let cmd = SyncCmd::Cluster(cluster::Cmd::Wait { warp: cw, mask: live.bits(), aligned: false });
            match support::step(ctx, res, cmd)? {
                Step::Blocked(r) => return Ok(Flow::Blocked(r)),
                Step::Done(Outcome::Cluster(cluster::Outcome::Ready { gen })) => {
                    support::protocol(ctx, live, vec![(res, cmd)], ProtoExtra::default());
                    support::sync_event(ctx, p.lanes, SyncKind::Wait { obj: res, phase: gen, acquire: Some(true), scope: Some(Scope::Cluster) });
                }
                out => return Err(internal(ctx, "cluster wait", out)),
            }
        }
        cluster_pass(ctx, k, active, true);
        return Ok(Flow::Next);
    }
    let entry = ctx.aux.cluster_partial.get_mut(&wid).expect("entry");
    // Gathered lanes whose command is not complete yet: keep waiting.
    if entry.gather.pending.is_some_and(|p| p.wait == wait && p.lanes & active.bits() == active.bits()) {
        return Ok(Flow::Blocked(res));
    }
    let out = cluster::gather(&mut entry.gather, cluster::GatherCmd::Execute { wait, mask: active.bits(), live: live.bits(), aligned })
        .map_err(|e| support::sync_err(ctx, crate::sync::SyncError::Cluster(e)))?;
    match out {
        cluster::GatherOutcome::Wait => Ok(Flow::Blocked(res)),
        cluster::GatherOutcome::Idle => Err(internal(ctx, "cluster gather", out)),
        cluster::GatherOutcome::Complete { mask } => {
            let lanes = WarpMask(mask);
            ctx.aux.cluster_partial.get_mut(&wid).expect("entry").pass[k] = Some(ClusterPass { lanes, waiting: lanes, passed: false });
            if !wait {
                let cmd = SyncCmd::Cluster(cluster::Cmd::Arrive { warp: cw, mask: live.bits(), aligned: false });
                let out = support::step(ctx, res, cmd)?;
                let Step::Done(Outcome::Cluster(cluster::Outcome::Arrived { gen, .. })) = out else {
                    return Err(internal(ctx, "cluster arrive", out));
                };
                let participants = ctx.launch.ctas_per_cluster() * ctx.launch.warps_per_cta();
                let extra = ProtoExtra { counts: Counts { participants: Some(participants), ..Default::default() }, ..Default::default() };
                support::protocol(ctx, live, vec![(res, cmd)], extra);
                support::sync_event(ctx, lanes, SyncKind::Arrive { obj: res, phase: gen, release: Some(release), scope: Some(Scope::Cluster) });
                cluster_pass(ctx, k, active, true);
                return Ok(Flow::Next);
            }
            let cmd = SyncCmd::Cluster(cluster::Cmd::Wait { warp: cw, mask: live.bits(), aligned: false });
            match support::step(ctx, res, cmd)? {
                Step::Blocked(r) => Ok(Flow::Blocked(r)),
                Step::Done(Outcome::Cluster(cluster::Outcome::Ready { gen })) => {
                    support::protocol(ctx, live, vec![(res, cmd)], ProtoExtra::default());
                    support::sync_event(ctx, lanes, SyncKind::Wait { obj: res, phase: gen, acquire: Some(true), scope: Some(Scope::Cluster) });
                    cluster_pass(ctx, k, active, true);
                    Ok(Flow::Next)
                }
                out => Err(internal(ctx, "cluster wait", out)),
            }
        }
    }
}

/// Lanes `lanes` passed the gathered cluster command of kind `k`.
fn cluster_pass(ctx: &mut ExecCtx<'_>, k: usize, lanes: WarpMask, passed: bool) {
    let wid = ctx.warp.id;
    if let Some(e) = ctx.aux.cluster_partial.get_mut(&wid) {
        if let Some(p) = e.pass[k].as_mut() {
            p.passed |= passed;
            p.waiting = p.waiting.and_not(lanes);
            if p.waiting.is_empty() {
                e.pass[k] = None;
            }
        }
        if e.pass.iter().all(Option::is_none) && e.gather.pending.is_none() {
            ctx.aux.cluster_partial.remove(&wid);
        }
    }
}

#[inline]
pub fn grid_sync(ctx: &mut ExecCtx<'_>) -> HResult {
    active_or_next!(ctx);
    let active = ctx.warp.active;
    if active != ctx.warp.live {
        return Err(support::err(ctx, ExecErrorKind::WarpCollectiveDivergence, active, "grid.sync with divergent lanes"));
    }
    if let Some(g) = ctx.warp.resume {
        if ctx.aux.grid.gen > g {
            ctx.warp.resume = None;
            support::sync_event(ctx, active, SyncKind::Wait { obj: ResourceId::Grid, phase: g, acquire: Some(true), scope: Some(Scope::Gpu) });
            return Ok(Flow::Next);
        }
        return Ok(Flow::Blocked(ResourceId::Grid));
    }
    let g = ctx.aux.grid.gen;
    ctx.aux.grid.arrived.insert(ctx.warp.id);
    support::sync_event(ctx, active, SyncKind::Arrive { obj: ResourceId::Grid, phase: g, release: Some(true), scope: Some(Scope::Gpu) });
    let expected = ctx.launch.num_warps() - ctx.aux.exited_warps;
    if ctx.aux.grid.arrived.len() as u32 >= expected {
        ctx.aux.grid.gen += 1;
        ctx.aux.grid.arrived.clear();
        support::sync_event(ctx, active, SyncKind::Wait { obj: ResourceId::Grid, phase: g, acquire: Some(true), scope: Some(Scope::Gpu) });
        return Ok(Flow::Next);
    }
    ctx.warp.resume = Some(g);
    Ok(Flow::Blocked(ResourceId::Grid))
}

// ---------------------------------------------------------------------------
// mbarrier
// ---------------------------------------------------------------------------

/// Distinct targets in lane order with a per-target value.
fn aggregate<T: Copy>(v: &mut Vec<(ResourceId, T)>, res: ResourceId, x: T, f: impl Fn(T, T) -> T) {
    match v.iter_mut().find(|(r, _)| *r == res) {
        Some((_, acc)) => *acc = f(*acc, x),
        None => v.push((res, x)),
    }
}

fn mbar_collapse(ctx: &mut ExecCtx<'_>, mbar: Operand, space: AddrSpace, val: Option<Operand>) -> Result<Vec<(ResourceId, u64)>, ExecError> {
    let mut v: Vec<(ResourceId, u64)> = Vec::new();
    for l in ctx.warp.active.lanes() {
        let res = mbar_res(ctx, space, lane_val(ctx, mbar, l), l)?;
        let x = val.map(|o| lane_val(ctx, o, l)).unwrap_or(0);
        match v.iter().find(|(r, _)| *r == res) {
            Some((_, y)) if *y != x => {
                return Err(support::err(ctx, ExecErrorKind::Divergence, ctx.warp.active, "lanes initialize one mbarrier with different counts"))
            }
            Some(_) => {}
            None => v.push((res, x)),
        }
    }
    Ok(v)
}

#[inline]
pub fn mbar_init(ctx: &mut ExecCtx<'_>, mbar: Operand, space: AddrSpace, count: Operand, layout_v1: bool) -> HResult {
    active_or_next!(ctx);
    let active = ctx.warp.active;
    let t = mbar_collapse(ctx, mbar, space, Some(count))?;
    // Two lanes initializing one barrier in one instruction while other
    // lanes name other barriers is a concurrent non-atomic init (legacy
    // rule, W12-gaps 7): the pointer must be warp-uniform or one-to-one.
    let n = active.count() as usize;
    if t.len() != 1 && t.len() != n {
        return Err(support::err(
            ctx,
            ExecErrorKind::Divergence,
            active,
            format!("mbarrier.init pointer must be warp-uniform or one-to-one across active lanes ({} lanes name {} barriers)", n, t.len()),
        ));
    }
    let cmds: Vec<(ResourceId, SyncCmd)> =
        t.iter().map(|&(r, c)| (r, SyncCmd::Mbarrier(mbarrier::Cmd::Init { count: c, layout_v1 }))).collect();
    support::step_all(ctx, &cmds)?;
    let pcmds = cmds
        .iter()
        .zip(&t)
        .map(|(&(res, cmd), &(_, c))| crate::observe::ProtocolCmd {
            res,
            cmd,
            counts: Counts { arrivals: Some(c), ..Default::default() },
            observed_parity: None,
        })
        .collect();
    support::protocol_cmds(ctx, active, pcmds, None, Vec::new());
    Ok(Flow::Next)
}

#[inline]
pub fn mbar_inval(ctx: &mut ExecCtx<'_>, mbar: Operand, space: AddrSpace) -> HResult {
    active_or_next!(ctx);
    let active = ctx.warp.active;
    let t = mbar_collapse(ctx, mbar, space, None)?;
    let cmds: Vec<(ResourceId, SyncCmd)> = t.iter().map(|&(r, _)| (r, SyncCmd::Mbarrier(mbarrier::Cmd::Inval))).collect();
    support::step_all(ctx, &cmds)?;
    support::protocol(ctx, active, cmds, ProtoExtra::default());
    Ok(Flow::Next)
}

/// Targets of one lane: the named barrier, or the same offset in every
/// CTA of a multicast mask.
fn lane_targets(ctx: &ExecCtx<'_>, mbar: Operand, space: AddrSpace, multicast: Option<Operand>, l: usize) -> Result<Vec<ResourceId>, ExecError> {
    let res = mbar_res(ctx, space, lane_val(ctx, mbar, l), l)?;
    Ok(match multicast {
        None => vec![res],
        Some(m) => ranks_of(ctx, lane_val(ctx, m, l))?.into_iter().filter_map(|r| mbar_in_rank(ctx, res, r)).collect(),
    })
}

#[inline]
pub fn mbar_arrive(ctx: &mut ExecCtx<'_>, args: MbarArriveArgs) -> HResult {
    active_or_next!(ctx);
    let active = ctx.warp.active;
    // (target, count, tx) aggregated over lanes; per-lane (target, count).
    let mut agg: Vec<(ResourceId, (u64, u64))> = Vec::new();
    let mut lanes: Vec<(usize, ResourceId, u64)> = Vec::new();
    for l in active.lanes() {
        let count = args.count.map(|c| lane_val(ctx, c, l)).unwrap_or(1);
        let tx = args.expect_tx.map(|t| lane_val(ctx, t, l)).unwrap_or(0);
        for res in lane_targets(ctx, args.mbar, args.space, args.multicast, l)? {
            aggregate(&mut agg, res, (count, tx), |a, b| (a.0 + b.0, a.1 + b.1));
            lanes.push((l, res, count));
        }
    }
    let cmd_of = |c: u64, tx: u64| {
        SyncCmd::Mbarrier(mbarrier::Cmd::Arrive {
            count: c,
            tx: args.expect_tx.map(|_| tx),
            drop: args.drop,
            no_complete: args.no_complete,
        })
    };
    // Cross-CTA targets apply synchronously (single arena / sync table).
    let all_cmds: Vec<(ResourceId, SyncCmd)> = agg.iter().map(|&(r, (c, tx))| (r, cmd_of(c, tx))).collect();
    let outs = match support::step_all(ctx, &all_cmds)? {
        Step::Done(o) => o,
        Step::Blocked(r) => return Err(internal(ctx, "mbarrier arrive", r)),
    };
    let local = all_cmds.clone();
    let pcmds = all_cmds
        .iter()
        .map(|&(res, cmd)| {
            let (c, tx) = agg.iter().find(|(r, _)| *r == res).map(|x| x.1).unwrap_or_default();
            crate::observe::ProtocolCmd {
                res,
                cmd,
                counts: Counts { arrivals: Some(c), tx_bytes: args.expect_tx.map(|_| tx), ..Default::default() },
                observed_parity: None,
            }
        })
        .collect();
    support::protocol_cmds(ctx, active, pcmds, None, Vec::new());
    let release = args.sem != Sem::Relaxed;
    for ((res, _), out) in local.iter().zip(&outs) {
        let Outcome::Mbarrier(mbarrier::Outcome::Arrived { gen, pending_before, .. }) = *out else {
            return Err(internal(ctx, "mbarrier arrive", out));
        };
        // Each target's Arrive carries only the lanes that arrived on it.
        let mine = lanes.iter().filter(|(_, r, _)| r == res).fold(WarpMask::NONE, |m, &(l, _, _)| m.or(WarpMask::lane(l)));
        support::sync_event(ctx, mine, SyncKind::Arrive { obj: *res, phase: gen, release: Some(release), scope: Some(args.scope) });
        if let Some(dst) = args.state {
            // Lane k sees pending_before minus the counts of lower lanes.
            let mut pending = pending_before;
            for &(l, r, c) in &lanes {
                if r != *res {
                    continue;
                }
                let tok = query::encode(gen, pending, args.no_complete)
                    .map_err(|e| ctx.error(ExecErrorKind::Internal, format!("state token: {e:?}")))?;
                write_lane(ctx, dst, l, tok);
                pending = pending.saturating_sub(c);
            }
        }
    }
    Ok(Flow::Next)
}

#[inline]
pub fn mbar_tx(ctx: &mut ExecCtx<'_>, op: TxOp, mbar: Operand, space: AddrSpace, bytes: Operand, multicast: Option<Operand>, scope: Scope) -> HResult {
    active_or_next!(ctx);
    let _ = scope;
    let active = ctx.warp.active;
    let mut agg: Vec<(ResourceId, u64)> = Vec::new();
    for l in active.lanes() {
        let b = lane_val(ctx, bytes, l);
        for res in lane_targets(ctx, mbar, space, multicast, l)? {
            aggregate(&mut agg, res, b, |a, b| a + b);
        }
    }
    let mut cmds = Vec::new();
    match op {
        TxOp::Expect => {
            let all: Vec<(ResourceId, SyncCmd)> =
                agg.iter().map(|&(r, b)| (r, SyncCmd::Mbarrier(mbarrier::Cmd::ExpectTx { bytes: b }))).collect();
            support::step_all(ctx, &all)?;
            cmds.extend(all);
        }
        TxOp::Complete => {
            for &(r, b) in &agg {
                let gen = mbar_issue(ctx, r)?;
                cmds.push((r, SyncCmd::Mbarrier(mbarrier::Cmd::Issue)));
                let cmd = SyncCmd::Mbarrier(mbarrier::Cmd::CompleteTx { gen, bytes: b });
                cmds.push((r, cmd));
                support::step(ctx, r, cmd)?;
            }
        }
    }
    let pcmds = cmds
        .iter()
        .map(|&(res, cmd)| crate::observe::ProtocolCmd {
            res,
            cmd,
            counts: Counts { tx_bytes: agg.iter().find(|(r, _)| *r == res).map(|x| x.1), ..Default::default() },
            observed_parity: None,
        })
        .collect();
    support::protocol_cmds(ctx, active, pcmds, None, Vec::new());
    Ok(Flow::Next)
}

fn phase_cmd(ctx: &ExecCtx<'_>, phase: PhaseArg, l: usize, blocking: bool) -> SyncCmd {
    SyncCmd::Mbarrier(match phase {
        PhaseArg::Parity(p) => {
            let parity = lane_val(ctx, p, l);
            if blocking {
                mbarrier::Cmd::WaitParity { parity }
            } else {
                mbarrier::Cmd::TestParity { parity }
            }
        }
        PhaseArg::State(s) => mbarrier::Cmd::TestState { gen: query::generation(lane_val(ctx, s, l)) },
    })
}

/// Successful waits/tests of one instruction: ONE protocol event (per-target
/// observed parity), then one HB `Wait` per target with its own lanes.
fn mbar_observed(ctx: &mut ExecCtx<'_>, lanes: WarpMask, done: &[(ResourceId, SyncCmd, WarpMask, Option<u64>)], sem: Sem, scope: Scope) {
    if done.is_empty() {
        return;
    }
    let pcmds = done
        .iter()
        .map(|&(res, cmd, _, gen)| crate::observe::ProtocolCmd {
            res,
            cmd,
            counts: Counts::default(),
            observed_parity: Some(gen.map_or(1, |g| (g & 1) as u8)),
        })
        .collect();
    support::protocol_cmds(ctx, lanes, pcmds, None, Vec::new());
    for &(res, _, m, gen) in done {
        if let Some(g) = gen {
            support::sync_event(ctx, m, SyncKind::Wait { obj: res, phase: g, acquire: Some(sem != Sem::Relaxed), scope: Some(scope) });
        }
    }
}

#[inline]
pub fn mbar_test_wait(
    ctx: &mut ExecCtx<'_>,
    kind: WaitKind,
    mbar: Operand,
    space: AddrSpace,
    phase: PhaseArg,
    sem: Sem,
    scope: Scope,
    dst: Option<Reg>,
    report: Option<Reg>,
    report_value: Option<Reg>,
) -> HResult {
    active_or_next!(ctx);
    let _ = kind;
    let active = ctx.warp.active;
    // One query per distinct (target, command), in lane order.
    let mut seen: Vec<(ResourceId, SyncCmd, Option<Option<u64>>, WarpMask)> = Vec::new();
    for l in active.lanes() {
        let res = mbar_res(ctx, space, lane_val(ctx, mbar, l), l)?;
        let cmd = phase_cmd(ctx, phase, l, false);
        let i = match seen.iter().position(|(r, c, _, _)| *r == res && *c == cmd) {
            Some(i) => i,
            None => {
                let ready = match support::step(ctx, res, cmd)? {
                    Step::Done(Outcome::Mbarrier(mbarrier::Outcome::Ready { gen })) => Some(gen),
                    Step::Done(Outcome::Mbarrier(mbarrier::Outcome::NotReady)) | Step::Blocked(_) => None,
                    other => return Err(internal(ctx, "mbarrier test_wait", other)),
                };
                seen.push((res, cmd, ready, WarpMask::NONE));
                seen.len() - 1
            }
        };
        seen[i].3 = seen[i].3.or(WarpMask::lane(l));
        if let Some(d) = dst {
            write_lane(ctx, d, l, seen[i].2.is_some() as u64);
        }
        // Report forms: the report bit of the observed phase, from the same
        // snapshot (legacy `query_many`); the value register is always 0
        // for copy-validity reports.
        if let Some(r) = report {
            let bit = match seen[i].2 {
                Some(Some(g)) => ctx.aux.mbar_reports.get(&(res, g)).copied().unwrap_or(false),
                _ => false,
            };
            write_lane(ctx, r, l, bit as u64);
        }
        if let Some(r) = report_value {
            write_lane(ctx, r, l, 0);
        }
    }
    let mut done = Vec::new();
    let mut ok_lanes = WarpMask::NONE;
    for (res, cmd, ready, lanes) in seen {
        match ready {
            Some(gen) => {
                ok_lanes = ok_lanes.or(lanes);
                done.push((res, cmd, lanes, gen));
            }
            None => ctx.note_failed_poll(res),
        }
    }
    mbar_observed(ctx, ok_lanes, &done, sem, scope);
    Ok(Flow::Next)
}

#[inline]
pub fn mbar_wait(ctx: &mut ExecCtx<'_>, mbar: Operand, space: AddrSpace, phase: PhaseArg, sem: Sem, scope: Scope) -> HResult {
    active_or_next!(ctx);
    let active = ctx.warp.active;
    // Lanes whose target completed on an earlier attempt already left the
    // wait (latched); only the others are retried. Lane-varying waits go
    // through `step` one target at a time (a blocked command inside
    // `step_all` would drop the `armed` registration).
    let key = (ctx.warp.id, ctx.warp.pc);
    let earlier = ctx.aux.mbar_latch.remove(&key).unwrap_or_default();
    let latched = earlier.iter().fold(WarpMask::NONE, |m, d| m.or(d.2));
    let mut done = Vec::new();
    let mut seen: Vec<(ResourceId, SyncCmd, WarpMask)> = Vec::new();
    for l in active.and_not(latched).lanes() {
        let res = mbar_res(ctx, space, lane_val(ctx, mbar, l), l)?;
        let cmd = phase_cmd(ctx, phase, l, true);
        match seen.iter_mut().find(|(r, c, _)| *r == res && *c == cmd) {
            Some((_, _, m)) => *m = m.or(WarpMask::lane(l)),
            None => seen.push((res, cmd, WarpMask::lane(l))),
        }
    }
    let mut blocked = None;
    for &(res, cmd, lanes) in &seen {
        match support::step(ctx, res, cmd) {
            Ok(Step::Done(Outcome::Mbarrier(mbarrier::Outcome::Ready { gen }))) => done.push((res, cmd, lanes, gen)),
            Ok(Step::Done(Outcome::Mbarrier(mbarrier::Outcome::NotReady)) | Step::Blocked(_)) => {
                blocked.get_or_insert(res);
            }
            Ok(other) => return Err(internal(ctx, "mbarrier wait", other)),
            Err(e) => return Err(e),
        }
    }
    // Targets that completed now are observed now (their lanes leave the
    // wait at this point, and the observed phase is consumed): one
    // Protocol event per attempt that completes something.
    let now = done.iter().fold(WarpMask::NONE, |m, d| m.or(d.2));
    mbar_observed(ctx, now, &done, sem, scope);
    if let Some(r) = blocked {
        let mut all = earlier;
        all.extend(done);
        if !all.is_empty() {
            ctx.aux.mbar_latch.insert(key, all);
        }
        return Ok(Flow::Blocked(r));
    }
    Ok(Flow::Next)
}

#[inline]
pub fn mbar_query(ctx: &mut ExecCtx<'_>, dst: Reg, op: MbarQueryOp) -> HResult {
    active_or_next!(ctx);
    for l in ctx.warp.active.lanes() {
        let v = match op {
            MbarQueryOp::PendingCount { state } => query::pending_count(lane_val(ctx, state, l))
                .map_err(|e| support::err(ctx, ExecErrorKind::Protocol(crate::sync::SyncError::Mbarrier(mbarrier::Error::UnknownToken { gen: 0 })), WarpMask::lane(l), format!("pending_count: {e:?}")))?
                as u64,
            MbarQueryOp::CheckLayout { mbar, space, layout_v1 } => {
                let res = mbar_res(ctx, space, lane_val(ctx, mbar, l), l)?;
                let st = match ctx.sync.get(res) {
                    Some(crate::sync::Resource::Mbarrier(s)) => s.clone(),
                    _ => mbarrier::State::default(),
                };
                query::check_layout(&st, layout_v1)
                    .map_err(|e| support::sync_err(ctx, crate::sync::SyncError::Mbarrier(e)))? as u64
            }
        };
        write_lane(ctx, dst, l, v);
    }
    Ok(Flow::Next)
}

// ---------------------------------------------------------------------------
// Fences, setmaxnreg, misc
// ---------------------------------------------------------------------------

#[inline]
pub fn fence(ctx: &mut ExecCtx<'_>, kind: FenceKind, sem: Sem, scope: Scope) -> HResult {
    active_or_next!(ctx);
    let ev = match kind {
        FenceKind::Thread => {
            if sem == Sem::Sc {
                FenceEvent::Sc(scope)
            } else {
                FenceEvent::AcqRel(scope)
            }
        }
        FenceKind::MbarrierInit => FenceEvent::MbarrierInit,
        FenceKind::ProxyAsync(space) => FenceEvent::ProxyAsync(match space {
            Some(AddrSpace::Global) => Some(Window::Global),
            Some(AddrSpace::Shared) => Some(Window::SharedCta),
            Some(AddrSpace::SharedCluster) => Some(Window::SharedCluster),
            _ => None,
        }),
        FenceKind::ProxyAlias => FenceEvent::ProxyAlias,
        FenceKind::TensormapRelease => {
            let w = ctx.warp.id;
            let mine: Vec<_> = ctx.aux.tmap_dirty.iter().filter(|(_, o)| **o == w).map(|(k, _)| *k).collect();
            for k in mine {
                ctx.aux.tmap_dirty.remove(&k);
                *ctx.aux.tmap_published.entry(k).or_default() += 1;
            }
            FenceEvent::TensormapRelease { scope }
        }
        FenceKind::TensormapAcquire { addr, space } => {
            // One event per distinct tensor map named by the active lanes.
            let mut seen: Vec<(crate::arena::AllocId, ByteSpan, WarpMask)> = Vec::new();
            for l in ctx.warp.active.lanes() {
                let loc = support::resolve(ctx, space, lane_val(ctx, addr, l), l, 128)?;
                match seen.iter_mut().find(|(a, s, _)| *a == loc.alloc && *s == loc.span(128)) {
                    Some((_, _, m)) => *m = m.or(WarpMask::lane(l)),
                    None => seen.push((loc.alloc, loc.span(128), WarpMask::lane(l))),
                }
            }
            for (alloc, span, lanes) in seen {
                if let Some(&g) = ctx.aux.tmap_published.get(&(alloc, span.start)) {
                    ctx.aux.tmap_acquired.insert((ctx.cta.id, alloc, span.start), g);
                }
                support::sync_event(ctx, lanes, SyncKind::Fence(FenceEvent::TensormapAcquire { scope, alloc, span }));
            }
            return Ok(Flow::Next);
        }
        FenceKind::Tcgen05Before => FenceEvent::TcgenBefore,
        FenceKind::Tcgen05After => FenceEvent::TcgenAfter,
    };
    let lanes = ctx.warp.active;
    support::sync_event(ctx, lanes, SyncKind::Fence(ev));
    Ok(Flow::Next)
}

/// `resume` tokens of setmaxnreg are tagged so they cannot be confused
/// with other blocking instructions' tokens.
const SETMAX_TAG: u64 = 1 << 62;

/// `setmaxnreg.{inc,dec}.sync.aligned`: a warpgroup collective. Every warp
/// contributes and waits until the whole warpgroup has (both directions;
/// see CONTRACT_REQUESTS.md W2-3 for `may_block`); the last arriver commits
/// `Set`. An `inc` then polls its grant. One Protocol event per warp, at
/// completion, tagged with the collective instance.
#[inline]
pub fn setmaxnreg(ctx: &mut ExecCtx<'_>, inc: bool, count: u32) -> HResult {
    active_or_next!(ctx);
    let active = ctx.warp.active;
    if active != ctx.warp.live {
        return Err(support::err(ctx, ExecErrorKind::WarpCollectiveDivergence, active, "setmaxnreg.sync.aligned with divergent lanes"));
    }
    let wg = ctx.warp.warp_in_cta / 4;
    let key = (ctx.cta.id, wg);
    let res = ResourceId::RegPool { cta: ctx.cta.id };
    let set = SyncCmd::RegPool(setmaxnreg::Cmd::Set { wg, inc, count });
    let wpc = ctx.launch.warps_per_cta();
    let n = (wpc - 4 * wg).min(4) as usize;
    let want = match ctx.warp.resume.filter(|t| t & SETMAX_TAG != 0) {
        Some(tok) => tok & !SETMAX_TAG,
        None => {
            let w = ctx.warp.warp_in_cta;
            let r = ctx.aux.setmax.entry(key).or_default();
            if !r.arrived.is_empty() && (r.inc != inc || r.count != count) {
                return Err(support::err(ctx, ExecErrorKind::WarpCollectiveDivergence, active, "warps of a warpgroup execute different setmaxnreg"));
            }
            if r.arrived.contains(&w) {
                return Err(support::err(ctx, ExecErrorKind::WarpCollectiveDivergence, active, "warp executed setmaxnreg twice before its warpgroup"));
            }
            r.arrived.push(w);
            r.inc = inc;
            r.count = count;
            let my_epoch = r.epoch;
            if r.arrived.len() >= n {
                match support::step(ctx, res, set)? {
                    Step::Done(Outcome::RegPool(setmaxnreg::Outcome::Pending { .. })) => {
                        ctx.sync.completions.push_back(Completion::SetmaxGrant { res, wg });
                    }
                    Step::Done(_) => {}
                    Step::Blocked(b) => return Err(internal(ctx, "setmaxnreg set", b)),
                }
                let r = ctx.aux.setmax.get_mut(&key).expect("present");
                r.epoch += 1;
                r.arrived.clear();
            }
            ctx.warp.resume = Some(SETMAX_TAG | (my_epoch + 1));
            my_epoch + 1
        }
    };
    if ctx.aux.setmax.get(&key).map_or(0, |r| r.epoch) < want {
        return Ok(Flow::Blocked(res));
    }
    if inc {
        let c = SyncCmd::RegPool(setmaxnreg::Cmd::Poll { wg });
        if let Step::Blocked(r) = support::step(ctx, res, c)? {
            return Ok(Flow::Blocked(r));
        }
    }
    ctx.warp.resume = None;
    let epoch = want - 1;
    let base = ctx.cta.id.0 * wpc + 4 * wg;
    let participants = (0..n as u32).map(|i| WarpId(base + i)).collect();
    let id = ((ctx.cta.id.0 as u64) << 40) | ((wg as u64) << 32) | epoch;
    let extra = ProtoExtra { collective: Some(Collective { id, participants }), ..Default::default() };
    support::protocol(ctx, active, vec![(res, set)], extra);
    Ok(Flow::Next)
}

/// Evaluate predicate `pid` with `values` bound to its argument, over
/// `lanes`. Runs the sub-program through the ordinary handlers.
fn eval_pred(ctx: &mut ExecCtx<'_>, pid: PredId, values: &[u64; 32], lanes: WarpMask) -> Result<WarpMask, ExecError> {
    let pp = ctx.program.preds[pid.0 as usize].clone();
    let saved = ctx.warp.active;
    ctx.warp.active = lanes;
    for l in lanes.lanes() {
        write_lane(ctx, pp.arg, l, values[l]);
    }
    let program = ctx.program;
    let mut res = Ok(());
    for pc in pp.start.0..pp.end.0 {
        match super::dispatch(ctx, &program.code[pc as usize]) {
            Ok(Flow::Next) => {}
            Ok(f) => {
                res = Err(ctx.error(ExecErrorKind::Internal, format!("predicate instruction returned {f:?}")));
                break;
            }
            Err(e) => {
                res = Err(e);
                break;
            }
        }
    }
    let yes = super::control::cond_mask(ctx, Operand::Reg(pp.result), lanes);
    ctx.warp.active = saved;
    res.map(|_| yes)
}

/// `resume` tag of a `WaitUntil` (low 32 bits: lanes already accepted).
const WAIT_TAG: u64 = 1 << 60;

/// Block until the predicate accepts the word at `addr`. Lanes latch
/// individually: a lane whose predicate accepts gets its `dst` (the value
/// it accepted) and its verdicts at that poll; the instruction completes
/// when every active lane has latched. `captures` are registers of this
/// warp, which cannot change while it waits here, so the register file is
/// their issue-time snapshot.
#[inline]
pub fn wait_until(
    ctx: &mut ExecCtx<'_>,
    dst: Reg,
    a: Operand,
    ty: Ty,
    space: AddrSpace,
    sem: Sem,
    scope: Scope,
    pred: PredId,
    captures: &[Reg],
) -> HResult {
    active_or_next!(ctx);
    let active = ctx.warp.active;
    let n = ty.mem_bytes() as u64;
    if n > 8 {
        return Err(support::unsupported(ctx, "wait_until on a word wider than 64 bits"));
    }
    let latched = match ctx.warp.resume {
        Some(t) if t & WAIT_TAG != 0 => WarpMask(t as u32),
        _ => WarpMask::NONE,
    };
    let todo = active.and_not(latched);
    let mut vals = [0u64; 32];
    let mut locs: Vec<(usize, support::Loc)> = Vec::with_capacity(todo.count() as usize);
    for l in todo.lanes() {
        let loc = support::resolve(ctx, space, lane_val(ctx, a, l), l, n)?;
        super::mem::check_align(ctx, loc, n, l)?;
        if ctx.aux.wants_history && ctx.aux.words.region(loc.alloc, loc.span(n)).is_none() {
            // Undeclared word: declared at first use (history starts now).
            ctx.aux.words.declare(ctx.arena, loc.alloc, loc.span(n));
            support::sync_event(ctx, WarpMask::lane(l), SyncKind::DeclareWord { alloc: loc.alloc, span: loc.span(n) });
        }
        let mut b = [0u8; 8];
        support::mem_read(ctx, loc, l, &mut b[..n as usize])?;
        vals[l] = u64::from_le_bytes(b);
        locs.push((l, loc));
    }
    let reads_memory = ctx.program.preds[pred.0 as usize].reads_memory;
    if reads_memory && ctx.observing {
        ctx.aux.capture_reads = Some(Vec::new());
    }
    let accepted = eval_pred(ctx, pred, &vals, todo);
    let pred_reads = ctx.aux.capture_reads.take().unwrap_or_default();
    let accepted = accepted?;
    if !accepted.is_empty() {
        let mut acc = Accesses::default();
        for &(l, loc) in &locs {
            if accepted.contains(l) {
                write_lane(ctx, dst, l, vals[l]);
                if ctx.observing {
                    acc.push(loc, l as u8, n);
                }
            }
        }
        let sp = support::spec(ctx, AccessKind::Read, sem, scope, Proxy::Generic);
        support::emit(ctx, sp, &mut acc);
        if ctx.aux.wants_history && ctx.observing {
            let locs: Vec<(usize, support::Loc)> = locs.iter().copied().filter(|(l, _)| accepted.contains(*l)).collect();
            emit_verdicts(ctx, pred, scope, captures, &locs, n, pred_reads)?;
        }
    }
    let latched = latched.or(accepted);
    if latched == active {
        ctx.warp.resume = None;
        return Ok(Flow::Next);
    }
    ctx.warp.resume = Some(WAIT_TAG | latched.bits() as u64);
    let l = active.and_not(latched).first().unwrap_or(0);
    let loc = locs.iter().find(|(x, _)| *x == l).map(|x| x.1).unwrap_or(locs[0].1);
    Ok(Flow::Blocked(ResourceId::Word { alloc: loc.alloc, offset: loc.offset }))
}

/// `WaitVerdicts` for lanes that just accepted: per word, lanes grouped by
/// identical acceptance bitsets. Only history entries newer than the last
/// verdict of this (warp, pc, word, capture values) are evaluated.
fn emit_verdicts(
    ctx: &mut ExecCtx<'_>,
    pred: PredId,
    scope: Scope,
    captures: &[Reg],
    locs: &[(usize, support::Loc)],
    n: u64,
    mut reads: Vec<(crate::arena::AllocId, ByteSpan)>,
) -> Result<(), ExecError> {
    reads.sort();
    reads.dedup();
    let mut words: Vec<(crate::arena::AllocId, ByteSpan, WarpMask)> = Vec::new();
    for &(l, loc) in locs {
        let span = loc.span(n);
        match words.iter_mut().find(|(a, s, _)| *a == loc.alloc && *s == span) {
            Some((_, _, m)) => *m = m.or(WarpMask::lane(l)),
            None => words.push((loc.alloc, span, WarpMask::lane(l))),
        }
    }
    for (alloc, span, lanes) in words {
        if ctx.aux.words.overflowed(alloc, span) {
            return Err(ctx.error(
                ExecErrorKind::Unsupported,
                format!(
                    "declared-word history exceeded {} writes; wait_until verdicts over it would be computed on a truncated history",
                    crate::interp::aux::MAX_WORD_HISTORY
                ),
            ));
        }
        let hist = ctx.aux.words.history(alloc, span).unwrap_or_default();
        let key = (ctx.warp.id, ctx.warp.pc, alloc, span.start);
        let mut cache = ctx.aux.verdicts.remove(&key).unwrap_or_default();
        if cache.bits.len() < 32 {
            cache.bits = vec![Vec::new(); 32];
            cache.captures = vec![Vec::new(); 32];
        }
        for l in lanes.lanes() {
            let cap: Vec<u64> = captures
                .iter()
                .flat_map(|&r| (0..support::reg_ty(ctx, r).slots()).map(move |i| (r, i)))
                .map(|(r, i)| ctx.reg_slot(r, i)[l])
                .collect();
            if cache.captures[l] != cap {
                cache.captures[l] = cap;
                cache.evaluated[l] = 0;
                cache.bits[l].clear();
            }
            cache.bits[l].resize(hist.len().div_ceil(64).max(1), 0);
        }
        let start = lanes.lanes().map(|l| cache.evaluated[l]).min().unwrap_or(0);
        let saved_obs = ctx.observing;
        ctx.observing = false;
        let mut res = Ok(());
        for (i, &v) in hist.iter().enumerate().skip(start) {
            let mut need = 0u32;
            for l in lanes.lanes() {
                if cache.evaluated[l] <= i {
                    need |= 1 << l;
                }
            }
            match eval_pred(ctx, pred, &[v; 32], WarpMask(need)) {
                Ok(ok) => {
                    for l in ok.lanes() {
                        cache.bits[l][i / 64] |= 1 << (i % 64);
                    }
                }
                Err(e) => {
                    res = Err(e);
                    break;
                }
            }
        }
        ctx.observing = saved_obs;
        res?;
        for l in lanes.lanes() {
            cache.evaluated[l] = hist.len();
        }
        let observed = hist.len().saturating_sub(1) as u32;
        let mut verdicts: Vec<LaneVerdict> = Vec::new();
        for l in lanes.lanes() {
            match verdicts.iter_mut().find(|v| v.accepted == cache.bits[l]) {
                Some(v) => v.lanes = v.lanes.or(WarpMask::lane(l)),
                None => verdicts.push(LaneVerdict { lanes: WarpMask::lane(l), accepted: cache.bits[l].clone(), observed }),
            }
        }
        ctx.aux.verdicts.insert(key, cache);
        support::sync_event(ctx, lanes, SyncKind::WaitVerdicts { alloc, span, scope, verdicts, pred_reads: reads.clone() });
    }
    Ok(())
}

#[inline]
pub fn griddepcontrol(ctx: &mut ExecCtx<'_>, launch_dependents: bool) -> HResult {
    // Programmatic dependent launch: kernels of a Module run in order, so
    // both forms are ordering-only no-ops.
    let _ = (ctx, launch_dependents);
    Ok(Flow::Next)
}

/// Bytes of a `clusterlaunchcontrol.try_cancel` response.
pub const CLC_RESPONSE_BYTES: u64 = 16;

#[inline]
pub fn clc_try_cancel(ctx: &mut ExecCtx<'_>, resp: Operand, mbar: Operand, multicast: bool) -> HResult {
    active_or_next!(ctx);
    // Every cluster of the launch is resident unless an execution subset
    // names the resident ones; then each non-resident cluster's task goes to
    // exactly one `try_cancel` (`LaunchAux::clc`, legacy `ClcTaskCounter`,
    // W12-gaps 6). The response's first word is the claimed cluster's linear
    // base CTA id, or legacy's "no cluster" 0xFFFF_FFFF; the rest is zero.
    // Claims are launch-wide state: inside an arena shard (parallel phase)
    // the instruction is a serial point, re-run on the main arena in
    // partition order, so who claims what does not depend on the workers.
    if ctx.arena.is_shard() && ctx.aux.clc.claimable() {
        ctx.aux.serial_request = true;
        return Ok(Flow::Yield(ctx.pc()));
    }
    let active = ctx.warp.active;
    let mut cmds = Vec::new();
    let mut issued_all = Vec::new();
    for l in active.lanes() {
        let rl = support::resolve(ctx, AddrSpace::SharedCluster, lane_val(ctx, resp, l), l, CLC_RESPONSE_BYTES)?;
        let res = mbar_res(ctx, AddrSpace::SharedCluster, lane_val(ctx, mbar, l), l)?;
        let ranks: Vec<u32> = if multicast { (0..ctx.launch.ctas_per_cluster()).collect() } else { vec![ctx.cta.rank_in_cluster] };
        let mut dst = Vec::new();
        let mut signals = Vec::new();
        let mut targets = Vec::new();
        for r in ranks {
            let alloc = ctx.cta.cluster_smem[r as usize];
            dst.push((alloc, rl.span(CLC_RESPONSE_BYTES)));
            let mres = mbar_in_rank(ctx, res, r).unwrap_or(res);
            let gen = mbar_issue(ctx, mres)?;
            cmds.push((mres, SyncCmd::Mbarrier(mbarrier::Cmd::Issue)));
            signals.push(Completion::MbarTx { res: mres, gen, bytes: CLC_RESPONSE_BYTES });
            targets.push(AsyncTarget { res: mres, bytes: CLC_RESPONSE_BYTES, arrivals: 0 });
        }
        issued_all.extend(targets.iter().copied());
        let mut one = vec![0u8; CLC_RESPONSE_BYTES as usize];
        one[..4].copy_from_slice(&ctx.aux.clc.try_cancel().to_le_bytes());
        let bytes: Vec<u8> = (0..dst.len()).flat_map(|_| one.iter().copied()).collect();
        issue_async(
            ctx,
            WarpMask::lane(l),
            Issue {
                kind: AsyncKind::ClcResponse,
                class: AsyncClass::Copy,
                proxy: Proxy::Async,
                payload: Payload::Data { dst, bytes },
                signals,
                after: Vec::new(),
                targets,
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
    }
    support::protocol(ctx, active, cmds, ProtoExtra { issued: issued_all, ..Default::default() });
    Ok(Flow::Next)
}
