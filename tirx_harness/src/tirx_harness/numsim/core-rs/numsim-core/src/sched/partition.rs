//! A scheduling partition: the unit of ownership (and of parallelism).
//!
//! A partition owns some resident CTAs (one cluster, or every cluster for
//! launches with grid-wide state), their warps, its own `SyncTable` and
//! `LaunchAux` (every resource and bookkeeping entry is partition-local), an
//! event buffer, and the list of async ops / serial requests it produced.
//! A round runs every partition against its own `Arena` shard; the
//! scheduler then merges shards, replays buffered events and runs serial
//! requests in partition order, so the result does not depend on how many
//! threads executed the partitions.

use super::{CompletionPolicy, CtaState, InboxMsg, Rng};
use crate::arena::{addr, AllocId, Arena, ByteSpan, Space};
use crate::interp::support::{self, AccessSpec, Accesses};
use crate::interp::{ExecCtx, ExecError, ExecErrorKind, LaunchAux, LaunchCounters, Loaded, StepResult, WarpStatus, WarpStepFn};
use crate::observe::{
    Access, AccessKind, AccessSeq, Actor, CtaId, LaneSpan, Observer, PublishTarget, Side, SyncEvent, SyncKind, WarpEnd,
    WarpId, Window, ALL_LANES,
};
use crate::program::{LaunchShape, Pc, Program, Proxy, Scope, Sem};
use crate::site::SiteId;
use crate::sync::completion::Payload;
use crate::sync::{async_group, mbarrier, Completion, Outcome, Step, SyncTable};
use crate::value::WarpMask;

/// Read-only launch facts shared by every partition of a round.
pub(crate) struct Env<'a> {
    pub program: &'a Program,
    pub loaded: &'a Loaded,
    pub shape: &'a LaunchShape,
    pub config: &'a super::RunConfig,
    pub kernel: u32,
    pub step_fn: WarpStepFn,
    pub round: u64,
    pub observing: bool,
}

pub(crate) fn sched_error(kind: ExecErrorKind, kernel: u32, warp: WarpId, site: SiteId, message: String) -> ExecError {
    ExecError { kind, kernel, warp, pc: Pc(0), site, lanes: WarpMask::NONE, message }
}

/// An owned copy of an [`Access`] (its `seq` is assigned at replay).
#[derive(Clone, Debug)]
struct OwnedAccess {
    actor: Actor,
    site: SiteId,
    alloc: AllocId,
    space: Space,
    kind: AccessKind,
    sem: Sem,
    scope: Scope,
    atomic: bool,
    returns_value: bool,
    proxy: Proxy,
    window: Option<Window>,
    spans: Vec<LaneSpan>,
    declared_word: bool,
}

#[derive(Clone, Debug)]
enum Event {
    Access(OwnedAccess),
    Sync(SyncEvent),
    WarpDone(WarpId, WarpEnd),
    InboxDrain(CtaId, u64),
}

/// Observer that records a partition's callbacks for ordered replay.
#[derive(Debug, Default)]
pub(crate) struct EventBuffer {
    pub enabled: bool,
    pub history: bool,
    events: Vec<Event>,
}

impl EventBuffer {
    /// Deliver and clear the buffered events; `Access::seq` is assigned
    /// here, in delivery order.
    pub fn replay(&mut self, observer: &mut dyn Observer, next_seq: &mut u64) {
        for e in self.events.drain(..) {
            match e {
                Event::Access(a) => {
                    let acc = Access {
                        seq: AccessSeq(*next_seq),
                        actor: a.actor,
                        site: a.site,
                        alloc: a.alloc,
                        space: a.space,
                        kind: a.kind,
                        sem: a.sem,
                        scope: a.scope,
                        atomic: a.atomic,
                        returns_value: a.returns_value,
                        proxy: a.proxy,
                        window: a.window,
                        spans: &a.spans,
                        declared_word: a.declared_word,
                    };
                    *next_seq += 1;
                    observer.access(&acc);
                }
                Event::Sync(s) => observer.sync(&s),
                Event::WarpDone(w, end) => observer.warp_done(w, end),
                Event::InboxDrain(c, r) => observer.inbox_drain(c, r),
            }
        }
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }
}

impl Observer for EventBuffer {
    fn enabled(&self) -> bool {
        self.enabled
    }
    fn wants_word_history(&self) -> bool {
        self.history
    }
    fn access(&mut self, a: &Access<'_>) {
        if !self.enabled {
            return;
        }
        self.events.push(Event::Access(OwnedAccess {
            actor: a.actor,
            site: a.site,
            alloc: a.alloc,
            space: a.space,
            kind: a.kind,
            sem: a.sem,
            scope: a.scope,
            atomic: a.atomic,
            returns_value: a.returns_value,
            proxy: a.proxy,
            window: a.window,
            spans: a.spans.to_vec(),
            declared_word: a.declared_word,
        }));
    }
    fn sync(&mut self, e: &SyncEvent) {
        if self.enabled {
            self.events.push(Event::Sync(e.clone()));
        }
    }
    fn warp_done(&mut self, w: WarpId, end: WarpEnd) {
        self.events.push(Event::WarpDone(w, end));
    }
    fn inbox_drain(&mut self, c: CtaId, r: u64) {
        self.events.push(Event::InboxDrain(c, r));
    }
}

/// One partition.
pub struct Partition {
    /// Clusters (linear ids) whose CTAs this partition owns, in order.
    pub clusters: Vec<u32>,
    /// Resident CTAs (cluster-contiguous, in admission order).
    pub ctas: Vec<CtaState>,
    pub sync: SyncTable,
    pub aux: LaunchAux,
    pub counters: LaunchCounters,
    pub completions: u64,
    pub(crate) outbox: Vec<InboxMsg>,
    pub(crate) rng: Rng,
    pub(crate) events: EventBuffer,
    /// `(cta index, warp index)` of warps parked at a serial point
    /// (global RMW inside a shard), in the order they were reached.
    pub(crate) serial: Vec<(usize, usize)>,
    /// Warp ends of this round, in order (merged into the scheduler).
    pub(crate) ends: Vec<(WarpId, WarpEnd)>,
    /// Async ops selected to land this round whose landing is a global
    /// read-modify-write inside a shard: landed by the serial phase.
    pub(crate) deferred: Vec<crate::sync::AsyncId>,
}

impl Partition {
    pub(crate) fn new(first_cluster: u32, sync: SyncTable, aux: LaunchAux, seed: u64, observing: bool, history: bool) -> Partition {
        Partition {
            clusters: vec![first_cluster],
            ctas: Vec::new(),
            sync,
            aux,
            counters: LaunchCounters::default(),
            completions: 0,
            outbox: Vec::new(),
            rng: Rng::new(seed ^ ((first_cluster as u64 + 1) << 32)),
            events: EventBuffer { enabled: observing, history, events: Vec::new() },
            serial: Vec::new(),
            ends: Vec::new(),
            deferred: Vec::new(),
        }
    }

    /// Allocations this partition owns (moved into its shard).
    pub(crate) fn private_allocs(&self) -> Vec<AllocId> {
        let mut v = Vec::new();
        for c in &self.ctas {
            v.push(c.ctx.smem);
            v.push(c.ctx.tmem);
            for w in &c.warps {
                v.extend(w.local);
                v.extend(w.regbuf);
                if let Some(&ra) = self.aux.reg_allocs.get(w.id.0 as usize) {
                    if ra.0 != u32::MAX {
                        v.push(ra);
                    }
                }
            }
        }
        v.sort();
        v.dedup();
        v
    }

    pub(crate) fn finished(&self) -> bool {
        self.ctas.iter().all(|c| c.finished() && c.inbox.msgs.is_empty())
    }

    /// One round of this partition: per CTA, drain its inbox, run one slice
    /// per runnable warp, land async ops and apply enabled completions.
    pub(crate) fn run_round(&mut self, env: &Env<'_>, arena: &mut Arena) -> Result<bool, ExecError> {
        let mut progress = false;
        for ci in 0..self.ctas.len() {
            progress |= self.drain_inbox(ci, env, arena)?;
            progress |= self.run_cta(ci, env, arena)?;
            let cta = self.ctas[ci].ctx.id;
            progress |= self.land(Some(cta), env, arena, env.config.completions == CompletionPolicy::Eager)?;
            progress |= self.apply_completions(env)?;
        }
        progress |= self.route_outbox();
        Ok(progress)
    }

    /// Run one warp for at most `quantum` instructions; returns the result
    /// and whether a progress instruction completed.
    fn slice(&mut self, ci: usize, w: usize, env: &Env<'_>, arena: &mut Arena, quantum: u32) -> (StepResult, bool) {
        let Partition { ctas, sync, outbox, counters, aux, events, .. } = self;
        let CtaState { ctx: cctx, warps, buffers, .. } = &mut ctas[ci];
        let prog0 = counters.progress;
        let mut ctx = ExecCtx {
            program: env.program,
            loaded: env.loaded,
            launch: env.shape,
            config: env.config,
            warp: &mut warps[w],
            cta: cctx,
            buffers,
            arena,
            sync,
            outbox,
            observer: events,
            observing: env.observing,
            counters,
            aux,
        };
        let r = crate::codegen::rt::guard(&mut ctx, quantum, env.step_fn);
        let progressed = ctx.counters.progress != prog0;
        (r, progressed)
    }

    /// Apply a slice result to the warp. `Err` ends the launch.
    fn settle(&mut self, ci: usize, w: usize, r: StepResult, progressed: bool) -> Result<bool, ExecError> {
        let serial = std::mem::take(&mut self.aux.serial_request);
        let warp = &mut self.ctas[ci].warps[w];
        Ok(match r {
            StepResult::Continue | StepResult::Yield => {
                warp.status = WarpStatus::Running;
                if serial {
                    self.serial.push((ci, w));
                }
                true
            }
            StepResult::Blocked(res) => {
                // Only committed effects count: a warp that merely moved
                // to (or swapped between) blocking points changed nothing
                // another warp could wait on.
                warp.status = WarpStatus::Blocked(res);
                progressed
            }
            StepResult::Exit => {
                warp.status = WarpStatus::Exited;
                let id = warp.id;
                self.ends.push((id, WarpEnd::Exited));
                self.events.warp_done(id, WarpEnd::Exited);
                true
            }
            StepResult::Error(e) => {
                let end = match e.kind {
                    ExecErrorKind::Trap => WarpEnd::Trapped,
                    ExecErrorKind::Budget => WarpEnd::Budget,
                    _ => WarpEnd::Error,
                };
                warp.status = if end == WarpEnd::Trapped { WarpStatus::Trapped } else { WarpStatus::Errored };
                let id = warp.id;
                self.ends.push((id, end));
                self.events.warp_done(id, end);
                return Err(e);
            }
        })
    }

    /// One slice per runnable warp of CTA `ci`, from a seeded rotation.
    fn run_cta(&mut self, ci: usize, env: &Env<'_>, arena: &mut Arena) -> Result<bool, ExecError> {
        let nw = self.ctas[ci].warps.len();
        if nw == 0 {
            return Ok(false);
        }
        let cid = self.ctas[ci].ctx.id.0 as u64;
        let start = (Rng::new(env.config.seed ^ env.round.wrapping_mul(0x2545_f491_4f6c_dd1d) ^ (cid << 32)).next_u64()
            % nw as u64) as usize;
        let mut progress = false;
        for k in 0..nw {
            let w = (start + k) % nw;
            if !matches!(self.ctas[ci].warps[w].status, WarpStatus::Running | WarpStatus::Blocked(_)) {
                continue;
            }
            if self.serial.iter().any(|&(c, x)| c == ci && x == w) {
                continue;
            }
            let (r, progressed) = self.slice(ci, w, env, arena, env.config.quantum);
            progress |= self.settle(ci, w, r, progressed)?;
        }
        Ok(progress)
    }

    /// Serial phase (main arena): execute every parked serial point (one
    /// instruction each, in order) and land deferred global reductions.
    ///
    /// Only the deferred reductions land here: every other async op keeps
    /// the completion policy's (seeded) latency.
    pub(crate) fn run_serial(&mut self, env: &Env<'_>, arena: &mut Arena) -> Result<bool, ExecError> {
        if self.serial.is_empty() && self.deferred.is_empty() {
            return Ok(false);
        }
        let mut progress = false;
        for (ci, w) in std::mem::take(&mut self.serial) {
            let (r, progressed) = self.slice(ci, w, env, arena, 1);
            self.settle(ci, w, r, progressed)?;
            progress = true;
        }
        for id in std::mem::take(&mut self.deferred) {
            if let Some(i) = self.sync.async_ops.iter().position(|o| o.id == id) {
                self.fire_op(i, env, arena)?;
                progress = true;
            }
        }
        progress |= self.apply_completions(env)?;
        Ok(progress)
    }

    /// Deliver CTA `ci`'s inbox (cross-CTA effects apply synchronously, so
    /// this is a wake-up point; the drain is still observable).
    fn drain_inbox(&mut self, ci: usize, env: &Env<'_>, arena: &mut Arena) -> Result<bool, ExecError> {
        let msgs = std::mem::take(&mut self.ctas[ci].inbox.msgs);
        let cta = self.ctas[ci].ctx.id;
        let any = !msgs.is_empty();
        for m in msgs {
            match m {
                InboxMsg::Write { alloc, offset, bytes, actor, site } => {
                    let span = ByteSpan::new(offset, bytes.len() as u64);
                    arena
                        .write(support::whole(arena, alloc), &[span], &bytes)
                        .map_err(|e| sched_error(ExecErrorKind::OutOfBounds, env.kernel, actor_warp(actor), site, e.to_string()))?;
                    if env.observing {
                        let mut acc = Accesses::default();
                        acc.items.push((alloc, Some(Window::SharedCluster), LaneSpan { lane: ALL_LANES, span }));
                        let spec = AccessSpec {
                            actor,
                            site,
                            kind: AccessKind::Write,
                            sem: Sem::Weak,
                            scope: Scope::Cluster,
                            atomic: false,
                            returns_value: false,
                            proxy: Proxy::Generic,
                        };
                        support::emit_accesses(&mut self.events, &mut self.counters, &mut self.aux, arena, spec, &mut acc);
                    }
                }
                InboxMsg::Sync { resource, cmd, actor, site } => {
                    let out = self.sync.step(resource, cmd).map_err(|e| {
                        sched_error(ExecErrorKind::Protocol(e.clone()), env.kernel, actor_warp(actor), site, format!("{e:?}"))
                    })?;
                    if let Step::Done(Outcome::Mbarrier(mbarrier::Outcome::Arrived { gen, .. })) = out {
                        if env.observing {
                            self.events.sync(&SyncEvent {
                                kernel: env.kernel,
                                actor,
                                seq: 0,
                                site,
                                frames: Vec::new(),
                                lanes: WarpMask::NONE,
                                kind: SyncKind::Arrive { obj: resource, phase: gen, release: None, scope: None },
                            });
                        }
                    }
                }
            }
        }
        self.events.inbox_drain(cta, env.round);
        Ok(any)
    }

    /// Move outbox messages to their target CTAs' inboxes (within the
    /// partition; cross-CTA effects of a cluster never leave it).
    fn route_outbox(&mut self) -> bool {
        if self.outbox.is_empty() {
            return false;
        }
        for m in std::mem::take(&mut self.outbox) {
            let target = match &m {
                InboxMsg::Sync { resource: crate::sync::ResourceId::Mbarrier { cta, .. }, .. } => Some(*cta),
                _ => m.target_hint().and_then(|a| self.aux.owner_cta.get(&a).copied()),
            };
            if let Some(c) = target.and_then(|t| self.ctas.iter_mut().find(|c| c.ctx.id == t)) {
                c.inbox.msgs.push(m);
            }
        }
        true
    }

    /// Does landing `op` read-modify-write a shared (global) allocation
    /// through this shard? Such landings are serial points.
    fn is_global_reduce(op: &crate::sync::AsyncOp, arena: &Arena) -> bool {
        match &op.payload {
            Payload::Reduce { dst, .. } | Payload::ReduceData { dst, .. } => dst.iter().any(|&(a, _)| arena.is_overlaid(a)),
            _ => false,
        }
    }

    /// Land ready async ops (of `cta`, or all with `None`). `all` = every
    /// ready op; otherwise a seeded subset. Global reductions inside a
    /// shard are deferred to the serial phase.
    pub(crate) fn land(&mut self, cta: Option<CtaId>, env: &Env<'_>, arena: &mut Arena, all: bool) -> Result<bool, ExecError> {
        let mut any = false;
        loop {
            let mut landed_one = false;
            let mut i = 0;
            while i < self.sync.async_ops.len() {
                let op = &self.sync.async_ops[i];
                let mine = cta.is_none_or(|c| op.source.cta == c);
                let ready = op.after.iter().all(|d| !self.sync.async_ops.iter().any(|o| o.id == *d));
                if mine && ready && !self.deferred.contains(&op.id) && (all || self.rng.below(2) == 0) {
                    if Self::is_global_reduce(op, arena) {
                        self.deferred.push(op.id);
                    } else {
                        self.fire_op(i, env, arena)?;
                        landed_one = true;
                        any = true;
                        continue;
                    }
                }
                i += 1;
            }
            if !landed_one || !all {
                break;
            }
        }
        Ok(any)
    }

    /// Apply enabled sync completions in FIFO order until none is enabled.
    pub(crate) fn apply_completions(&mut self, env: &Env<'_>) -> Result<bool, ExecError> {
        let mut any = false;
        // Passes over the queue (not a rescan from the head after every
        // application, which was quadratic in the number of pending
        // milestones): each pass applies enabled completions in queue
        // order; another pass runs while the previous one applied any (an
        // application can enable an earlier entry).
        let mut start = 0usize;
        let mut applied_in_pass = false;
        loop {
            let found = self.sync.completions.iter().skip(start).position(|c| self.sync.enabled(c)).map(|k| k + start);
            let Some(i) = found else {
                if applied_in_pass {
                    start = 0;
                    applied_in_pass = false;
                    continue;
                }
                break;
            };
            start = i;
            applied_in_pass = true;
            let c = self.sync.completions.remove(i).expect("index valid");
            match self.sync.apply_completion(c) {
                Ok(Step::Done(out)) => {
                    any = true;
                    self.completions += 1;
                    if let (
                        Completion::GroupMilestone { res, ordinal, milestone: async_group::Milestone::FullyDone },
                        Outcome::AsyncGroup(async_group::Outcome::Completed { .. }),
                    ) = (c, out)
                    {
                        if let Some(arr) = self.aux.groups.arrivals.remove(&(res, ordinal)) {
                            self.sync.completions.extend(arr);
                        }
                        // W5-11: the fired cp.async.mbarrier.arrive publishes
                        // every prior cp.async of its lane to that phase.
                        if let Some(pubs) = self.aux.cp_arrive_publish.remove(&(res, ordinal)) {
                            if env.observing {
                                for (mbar, gen, ops) in pubs {
                                    for op in ops {
                                        self.events.sync(&SyncEvent {
                                            kernel: env.kernel,
                                            actor: Actor::Async { op, side: Side::Write },
                                            seq: 0,
                                            site: SiteId::NONE,
                                            frames: Vec::new(),
                                            lanes: WarpMask::NONE,
                                            kind: SyncKind::AsyncComplete { op, milestone: Side::Write, target: PublishTarget::Phase { obj: mbar, phase: gen } },
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(Step::Blocked(_)) => {
                    self.sync.completions.push_back(c);
                    break;
                }
                Err(e) => {
                    return Err(sched_error(
                        ExecErrorKind::Protocol(e.clone()),
                        env.kernel,
                        WarpId(u32::MAX),
                        SiteId::NONE,
                        format!("completion {c:?}: {e:?}"),
                    ))
                }
            }
        }
        Ok(any)
    }

    /// Land async op `index` (payload, then its completions), emitting
    /// observer events.
    pub(crate) fn fire_op(&mut self, index: usize, env: &Env<'_>, arena: &mut Arena) -> Result<(), ExecError> {
        let kernel = env.kernel;
        let Some(op) = self.sync.async_ops.remove(index) else {
            return Err(sched_error(ExecErrorKind::Internal, kernel, WarpId(u32::MAX), SiteId::NONE, "no such async op".into()));
        };
        let meta = self.aux.async_meta.remove(&op.id);
        // st.async / red.async: performed in the generic proxy (W5-8).
        let strong = meta.as_ref().and_then(|m| m.strong);
        let proxy = if strong.is_some() { Proxy::Generic } else { meta.as_ref().map(|m| m.proxy).unwrap_or_default() };
        let lane = meta.as_ref().map(|m| m.lane).unwrap_or(ALL_LANES);
        let src_err = |e: String| sched_error(ExecErrorKind::OutOfBounds, kernel, op.source.warp, op.source.site, e);
        let mut reads: Vec<(AllocId, ByteSpan)> = Vec::new();
        let mut writes: Vec<(AllocId, ByteSpan)> = Vec::new();
        let mut rmw = false;
        match &op.payload {
            Payload::None => {}
            Payload::Copy { src, dst, zero_fill } => {
                copy_spans(arena, src, dst).map_err(src_err)?;
                reads.extend(src.iter().copied());
                writes.extend(dst.iter().copied());
                let pattern = meta.as_ref().map(|m| m.fill_pattern.as_slice()).unwrap_or(&[]);
                for &(a, s) in zero_fill {
                    if pattern.is_empty() {
                        arena.fill(support::whole(arena, a), &[s], 0).map_err(|e| src_err(e.to_string()))?;
                    } else {
                        let bytes: Vec<u8> = (0..s.len as usize).map(|i| pattern[i % pattern.len()]).collect();
                        arena.write(support::whole(arena, a), &[s], &bytes).map_err(|e| src_err(e.to_string()))?;
                    }
                    writes.push((a, s));
                }
                if meta.as_ref().is_some_and(|m| m.tf32_round) {
                    // TF32 tensor-map loads round each copied f32 element.
                    for &(a, s) in dst {
                        let v = support::whole(arena, a);
                        let mut off = s.start;
                        while off + 4 <= s.end() {
                            let e = ByteSpan::new(off, 4);
                            if arena.is_valid(v, &[e]).unwrap_or(false) {
                                let w = arena.read_raw(a, e);
                                let x = crate::oplib::tma_tf32_round(u32::from_le_bytes(w.try_into().unwrap()));
                                arena.write(v, &[e], &x.to_le_bytes()).map_err(|e| src_err(e.to_string()))?;
                            }
                            off += 4;
                        }
                    }
                }
            }
            Payload::TcgenCp { src, dst, decompress_bits } => {
                // Pairwise: `src[k]` (2/3/4 bytes) lands decoded as the
                // 4-byte TMEM cell `dst[k]` (oplib `tcgen_cp_decode`).
                if src.len() != dst.len() {
                    return Err(sched_error(ExecErrorKind::Internal, kernel, op.source.warp, op.source.site, "tcgen05.cp payload pairs".into()));
                }
                for (&(sa, ss), &(da, ds)) in src.iter().zip(dst) {
                    let mut b = vec![0u8; ss.len as usize];
                    arena.read(support::whole(arena, sa), &[ss], &mut b).map_err(|e| src_err(e.to_string()))?;
                    let cell = crate::oplib::tcgen_cp_decode(&b, *decompress_bits)
                        .map_err(|e| sched_error(ExecErrorKind::Op(e.kind), kernel, op.source.warp, op.source.site, e.message))?;
                    arena.write(support::whole(arena, da), &[ds], &cell).map_err(|e| src_err(e.to_string()))?;
                }
                reads.extend(src.iter().copied());
                writes.extend(dst.iter().copied());
            }
            Payload::Reduce { op: aop, dtype, src, dst } => {
                let s = gather_bytes(arena, src).map_err(src_err)?;
                let mut d = gather_bytes(arena, dst).map_err(src_err)?;
                crate::interp::handlers::mem::rmw_bytes(*aop, *dtype, &mut d, &s, &[], false)
                    .map_err(|e| sched_error(ExecErrorKind::Op(e.kind), kernel, op.source.warp, op.source.site, e.message))?;
                scatter_bytes(arena, dst, &d).map_err(src_err)?;
                reads.extend(src.iter().copied());
                writes.extend(dst.iter().copied());
                rmw = true;
            }
            Payload::Data { dst, bytes } => {
                scatter_bytes(arena, dst, bytes).map_err(src_err)?;
                writes.extend(dst.iter().copied());
            }
            Payload::ReduceData { op: aop, dtype, dst, bytes } => {
                let mut d = gather_bytes(arena, dst).map_err(src_err)?;
                crate::interp::handlers::mem::rmw_bytes(*aop, *dtype, &mut d, bytes, &[], false)
                    .map_err(|e| sched_error(ExecErrorKind::Op(e.kind), kernel, op.source.warp, op.source.site, e.message))?;
                scatter_bytes(arena, dst, &d).map_err(src_err)?;
                writes.extend(dst.iter().copied());
                rmw = true;
            }
            Payload::TcgenMma(p) => {
                let (r, w) = run_mma(arena, p, tc_arch(env.program.arch.as_deref()), meta.as_ref().and_then(|m| m.lut_b))
                    .map_err(|e| sched_error(ExecErrorKind::Op(e.kind), kernel, op.source.warp, op.source.site, e.message))?;
                reads.extend(r);
                writes.extend(w);
                rmw = true;
            }
        }
        // Readonly-proxy contract: no global byte the kernel read through
        // `ld.global.nc` may be written (any write path).
        if arena.readonly_tracking() {
            let dead = meta.as_ref().map(|m| m.dead.as_slice()).unwrap_or(&[]);
            let frags = meta.as_ref().map(|m| m.bit_frags.as_slice()).unwrap_or(&[]);
            let all = writes.iter().copied().chain(dead.iter().copied()).chain(frags.iter().map(|f| (f.global.0, ByteSpan::new(f.global.1, 1))));
            for (a, sp) in all {
                if let Err(b) = arena.note_global_write(a, sp) {
                    let m = support::readonly_conflict_message(&arena.get(a).name, a, b);
                    return Err(sched_error(ExecErrorKind::BadAddress, kernel, op.source.warp, op.source.site, m));
                }
            }
        }
        // `.ignore_oob` dead bytes: zero, then invalid (legacy validity
        // `false`); they are part of the copy's write footprint.
        if let Some(m) = meta.as_ref().filter(|m| !m.dead.is_empty()) {
            for &(a, sp) in &m.dead {
                let v = support::whole(arena, a);
                arena.fill(v, &[sp], 0).map_err(|e| src_err(e.to_string()))?;
                arena.invalidate(v, &[sp]).map_err(|e| src_err(e.to_string()))?;
                writes.push((a, sp));
            }
        }
        // Sub-byte TMA store fragments, after the byte spans (W4-11).
        let mut frag_writes: Vec<(AllocId, ByteSpan)> = Vec::new();
        if let Some(m) = meta.as_ref().filter(|m| !m.bit_frags.is_empty()) {
            for f in &m.bit_frags {
                let sspan = ByteSpan::new(f.smem.1, 1);
                let gspan = ByteSpan::new(f.global.1, 1);
                let mut sb = [0u8; 1];
                let mut gb = [0u8; 1];
                arena.read(support::whole(arena, f.smem.0), &[sspan], &mut sb).map_err(|e| src_err(e.to_string()))?;
                arena.read(support::whole(arena, f.global.0), &[gspan], &mut gb).map_err(|e| src_err(e.to_string()))?;
                let mask = f.mask << f.tgt_shift;
                let g = (gb[0] & !mask) | (((sb[0] >> f.src_shift) & f.mask) << f.tgt_shift);
                arena.write(support::whole(arena, f.global.0), &[gspan], &[g]).map_err(|e| src_err(e.to_string()))?;
                reads.push((f.smem.0, sspan));
                frag_writes.push((f.global.0, gspan));
            }
        }
        // ZeroAndReport: an async op that reads uninitialized bytes is
        // reported like a synchronous read (legacy reports it at the async
        // read; the bytes' invalidity still propagates to the destination).
        if arena.policy() == crate::arena::ValidityPolicy::ZeroAndReport {
            for &(a, s) in &reads {
                self.report_async_uninit(env, arena, op.id, op.source.site, lane, a, s);
            }
        }
        if env.observing {
            let site = op.source.site;
            let mk = |side, kind| AccessSpec {
                actor: Actor::Async { op: op.id, side },
                site,
                kind,
                sem: Sem::Weak,
                scope: Scope::Gpu,
                atomic: false,
                returns_value: false,
                proxy,
            };
            let window = |arena: &Arena, a: AllocId| match arena.get(a).space {
                Space::Global => Some(Window::Global),
                Space::Shared => Some(Window::SharedCta),
                _ => None,
            };
            let mut acc = Accesses::default();
            // W5-10: the MMA's shared-A reads belong to its separate
            // shared-A read op (reported when that op lands).
            let a_reads: &[(AllocId, ByteSpan)] = meta.as_ref().map(|m| m.a_reads.as_slice()).unwrap_or(&[]);
            let is_a_read = meta.as_ref().is_some_and(|m| m.is_a_read);
            let reads: Vec<(AllocId, ByteSpan)> = if is_a_read {
                a_reads.to_vec()
            } else if a_reads.is_empty() {
                reads.clone()
            } else {
                subtract_spans(&reads, a_reads)
            };
            // MMA / tcgen05.cp operand reads of shared memory are async-proxy
            // reads (legacy reports a missing proxy fence there); TMEM stays
            // in the tcgen proxy.
            let tc_op = matches!(op.kind, crate::sync::AsyncKind::TcgenMma | crate::sync::AsyncKind::TcgenCp);
            let (shared_reads, other_reads): (Vec<_>, Vec<_>) =
                reads.iter().copied().partition(|&(a, _)| tc_op && arena.get(a).space == Space::Shared);
            for (group, prox) in [(shared_reads, Proxy::Async), (other_reads, proxy)] {
                if group.is_empty() {
                    continue;
                }
                acc.items = group.iter().map(|&(a, s)| (a, window(arena, a), LaneSpan { lane, span: s })).collect();
                let mut sp = mk(Side::Read, AccessKind::Read);
                sp.proxy = prox;
                support::emit_accesses(&mut self.events, &mut self.counters, &mut self.aux, arena, sp, &mut acc);
            }
            let reduce_elem = match &op.payload {
                Payload::Reduce { dtype, .. } | Payload::ReduceData { dtype, .. } => Some(dtype.mem_bytes() as u64),
                _ => None,
            };
            match reduce_elem {
                // Bulk / TMA / async reductions: one atomic relaxed RMW per
                // element (CONTRACT_REQUESTS.md W2-9), element-granular spans.
                Some(eb) => {
                    acc.items = writes
                        .iter()
                        .flat_map(|&(a, s)| {
                            let w = window(arena, a);
                            (0..s.len / eb.max(1)).map(move |k| (a, w, LaneSpan { lane, span: ByteSpan::new(s.start + k * eb, eb) }))
                        })
                        .collect();
                    let mut sp = mk(Side::Write, AccessKind::Rmw);
                    sp.atomic = true;
                    sp.sem = Sem::Relaxed;
                    sp.scope = Scope::Gpu;
                    if let Some(scope) = strong {
                        sp.sem = Sem::Release;
                        sp.scope = scope;
                    }
                    support::emit_accesses(&mut self.events, &mut self.counters, &mut self.aux, arena, sp, &mut acc);
                }
                None => {
                    acc.items = writes.iter().map(|&(a, s)| (a, window(arena, a), LaneSpan { lane, span: s })).collect();
                    let wk = if rmw { AccessKind::Rmw } else { AccessKind::Write };
                    let mut sp = mk(Side::Write, wk);
                    if let Some(scope) = strong {
                        // A strong release write at the instruction's scope,
                        // with or without a completion mbarrier.
                        sp.atomic = true;
                        sp.sem = Sem::Release;
                        sp.scope = scope;
                    }
                    support::emit_accesses(&mut self.events, &mut self.counters, &mut self.aux, arena, sp, &mut acc);
                }
            }
            if !frag_writes.is_empty() {
                // One 1-byte global read-modify-write per fragment.
                acc.items = frag_writes.iter().map(|&(a, s)| (a, window(arena, a), LaneSpan { lane, span: s })).collect();
                support::emit_accesses(&mut self.events, &mut self.counters, &mut self.aux, arena, mk(Side::Write, AccessKind::Rmw), &mut acc);
            }
            for c in &op.signals {
                if let Completion::MbarTx { res, gen, .. } | Completion::MbarArrive { res, gen, .. } = *c {
                    self.events.sync(&SyncEvent {
                        kernel,
                        actor: Actor::Async { op: op.id, side: Side::Write },
                        seq: 0,
                        site,
                        frames: Vec::new(),
                        lanes: WarpMask::NONE,
                        kind: SyncKind::AsyncComplete { op: op.id, milestone: Side::Write, target: PublishTarget::Phase { obj: res, phase: gen } },
                    });
                }
            }
        }
        // layout::v1 copy reports: inspect the copied source bytes, OR the
        // result into every completion phase's report bit.
        if let (Some(crate::program::ReportMode::PerElementFf), Payload::Copy { src, .. }) =
            (meta.as_ref().and_then(|m| m.report), &op.payload)
        {
            let matched = src.iter().any(|&(a, s)| arena.read_raw(a, s).contains(&0xff));
            for c in &op.signals {
                if let Completion::MbarTx { res, gen, .. } = *c {
                    *self.aux.mbar_reports.entry((res, gen)).or_default() |= matched;
                }
            }
        }
        self.sync.completions.extend(op.signals.iter().copied());
        let open = self.aux.groups.is_open(op.id);
        let due = self.aux.groups.landed(op.id, open);
        self.sync.completions.extend(due);
        Ok(())
    }
}

impl Partition {
    /// One `UninitRead` finding per maximal invalid range of `span` of
    /// `alloc` read by async op `op` (deduplicated like synchronous reads).
    fn report_async_uninit(&mut self, env: &Env<'_>, arena: &Arena, op: crate::sync::AsyncId, site: SiteId, lane: u8, alloc: AllocId, span: ByteSpan) {
        let v = support::whole(arena, alloc);
        let mut pos = span.start;
        while pos < span.end() {
            let Some(first) = arena.first_invalid(v, ByteSpan::new(pos, span.end() - pos)) else { break };
            let end = arena.first_valid(v, ByteSpan::new(first, span.end() - first)).unwrap_or(span.end());
            let bad = ByteSpan::new(first, end - first);
            if self.aux.uninit_seen.insert((site, alloc, bad)) {
                let a = arena.get(alloc);
                let actor = Actor::Async { op, side: Side::Read };
                let l = if lane == ALL_LANES { 0 } else { lane as usize };
                self.aux.diagnostics.push(support::uninit_finding(env.kernel, site, Some(actor), a.space, alloc, &a.name, bad, l));
            }
            pos = end;
        }
    }
}

/// `spans` minus every byte of `minus` (per allocation).
fn subtract_spans(spans: &[(AllocId, ByteSpan)], minus: &[(AllocId, ByteSpan)]) -> Vec<(AllocId, ByteSpan)> {
    let mut out = Vec::new();
    for &(a, s) in spans {
        let mut pieces = vec![s];
        for &(ma, m) in minus.iter().filter(|(ma, _)| *ma == a) {
            let _ = ma;
            let mut next = Vec::new();
            for p in pieces {
                if !p.overlaps(m) {
                    next.push(p);
                    continue;
                }
                if p.start < m.start {
                    next.push(ByteSpan::new(p.start, m.start - p.start));
                }
                if m.end() < p.end() {
                    next.push(ByteSpan::new(m.end(), p.end() - m.end()));
                }
            }
            pieces = next;
        }
        out.extend(pieces.into_iter().map(|p| (a, p)));
    }
    out
}

fn actor_warp(a: Actor) -> WarpId {
    match a {
        Actor::Warp { warp, .. } => warp,
        _ => WarpId(u32::MAX),
    }
}

/// Copy concatenated `src` spans onto concatenated `dst` spans (equal
/// totals), carrying validity.
fn copy_spans(arena: &mut Arena, src: &[(AllocId, ByteSpan)], dst: &[(AllocId, ByteSpan)]) -> Result<(), String> {
    let total = |v: &[(AllocId, ByteSpan)]| v.iter().map(|s| s.1.len).sum::<u64>();
    if total(src) != total(dst) {
        return Err(format!("copy length mismatch: {} vs {}", total(src), total(dst)));
    }
    // Common case (every source byte valid): gather the source as merged
    // contiguous runs, then store each merged destination run with one
    // memcpy + one validity range (direct when the allocation is not a
    // shard overlay).
    let src_runs = merge_runs(src);
    if src_runs.iter().all(|&(a, s)| !arena.get(a).metadata_only && arena.first_invalid(support::whole(arena, a), s).is_none()) {
        let mut bytes = Vec::with_capacity(total(src) as usize);
        for &(a, s) in &src_runs {
            if arena.is_overlaid(a) {
                bytes.extend(arena.read_raw(a, s));
            } else {
                bytes.extend_from_slice(&arena.get(a).bytes[s.start as usize..s.end() as usize]);
            }
        }
        let mut pos = 0usize;
        for (a, s) in merge_runs(dst) {
            let n = s.len as usize;
            let direct = !arena.is_overlaid(a) && !arena.get(a).metadata_only && s.end() <= arena.get(a).size;
            if direct {
                let al = arena.get_mut(a);
                al.bytes[s.start as usize..s.end() as usize].copy_from_slice(&bytes[pos..pos + n]);
                al.valid.set_range(s.start, s.len, true);
            } else {
                arena.write(support::whole(arena, a), &[s], &bytes[pos..pos + n]).map_err(|e| e.to_string())?;
            }
            pos += n;
        }
        return Ok(());
    }
    let (mut si, mut so, mut di, mut doff) = (0usize, 0u64, 0usize, 0u64);
    while si < src.len() && di < dst.len() {
        let (sa, ss) = src[si];
        let (da, ds) = dst[di];
        let n = (ss.len - so).min(ds.len - doff);
        if n > 0 {
            arena
                .copy_with_validity((sa, ByteSpan::new(ss.start + so, n)), (da, ds.start + doff))
                .map_err(|e| e.to_string())?;
        }
        so += n;
        doff += n;
        if so == ss.len {
            si += 1;
            so = 0;
        }
        if doff == ds.len {
            di += 1;
            doff = 0;
        }
    }
    Ok(())
}

/// Consecutive same-allocation spans that touch, merged (order kept).
fn merge_runs(v: &[(AllocId, ByteSpan)]) -> Vec<(AllocId, ByteSpan)> {
    let mut out: Vec<(AllocId, ByteSpan)> = Vec::with_capacity(v.len());
    for &(a, s) in v {
        match out.last_mut() {
            Some((la, ls)) if *la == a && ls.end() == s.start => ls.len += s.len,
            _ => out.push((a, s)),
        }
    }
    out
}

fn gather_bytes(arena: &Arena, spans: &[(AllocId, ByteSpan)]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for &(a, s) in spans {
        let mut b = vec![0u8; s.len as usize];
        arena.read(support::whole(arena, a), &[s], &mut b).map_err(|e| e.to_string())?;
        out.extend(b);
    }
    Ok(out)
}

fn scatter_bytes(arena: &mut Arena, spans: &[(AllocId, ByteSpan)], bytes: &[u8]) -> Result<(), String> {
    let mut pos = 0usize;
    for &(a, s) in spans {
        let n = s.len as usize;
        if pos + n > bytes.len() {
            return Err("payload shorter than its spans".into());
        }
        arena.write(support::whole(arena, a), &[s], &bytes[pos..pos + n]).map_err(|e| e.to_string())?;
        pos += n;
    }
    Ok(())
}

type Spans = Vec<(AllocId, ByteSpan)>;

/// Target architecture of a program (`Program::arch`).
fn tc_arch(arch: Option<&str>) -> crate::oplib::TcArch {
    match arch {
        Some(a) if a.starts_with("sm_103") => crate::oplib::TcArch::Sm103,
        Some(a) if a.starts_with("sm_107") => crate::oplib::TcArch::Sm107,
        _ => crate::oplib::TcArch::Sm100,
    }
}

/// tcgen05.mma numerics through oplib (`tc_mma_ctas`), recording the spans
/// it touched. `p.smem` / `p.tmem` are indexed by CTA within the issuing
/// group (0 = even CTA of the pair for `cta_group::2`).
fn run_mma(arena: &mut Arena, p: &crate::sync::completion::TcgenMmaPayload, arch: crate::oplib::TcArch, lut_b: Option<u32>) -> crate::oplib::OpResult<(Spans, Spans)> {
    use crate::oplib::OpError;
    use std::cell::RefCell;
    let cell = RefCell::new(arena);
    let reads: RefCell<Spans> = RefCell::new(Vec::new());
    let mut writes: Spans = Vec::new();
    let options = crate::oplib::TcMmaOptions { arch, ti16: p.args.kind == crate::program::TcMmaKind::Ti16, lut_b, ..Default::default() };
    // oplib reads/writes one small piece at a time: record them merged with
    // the previous piece when contiguous (most are), and read/write private
    // (non-overlaid) allocations directly when every byte is valid.
    fn note(v: &mut Spans, al: AllocId, span: ByteSpan) {
        if let Some((la, ls)) = v.last_mut() {
            if *la == al && ls.end() == span.start {
                ls.len += span.len;
                return;
            }
        }
        v.push((al, span));
    }
    fn fast_read(ar: &Arena, al: AllocId, span: ByteSpan, out: &mut [u8]) -> bool {
        if ar.is_overlaid(al) {
            return false;
        }
        let a = ar.get(al);
        if a.metadata_only || span.end() > a.size || a.valid.first_clear(span.start, span.len).is_some() {
            return false;
        }
        out.copy_from_slice(&a.bytes[span.start as usize..span.end() as usize]);
        true
    }
    let smem = |cta: u32, a: u32, out: &mut [u8]| -> crate::oplib::OpResult {
        let al = *p.smem.get(cta as usize).ok_or_else(|| OpError::invalid("mma smem operand of a CTA outside the group"))?;
        let off = addr::decode_shared(a).1 as u64;
        let ar = cell.borrow();
        let span = ByteSpan::new(off, out.len() as u64);
        if !fast_read(&ar, al, span, out) {
            ar.read(support::whole(&ar, al), &[span], out).map_err(|e| OpError::invalid(e.to_string()))?;
        }
        note(&mut reads.borrow_mut(), al, span);
        Ok(())
    };
    // A buffer may span several consecutive cells of one lane (W4-16);
    // it must not run past the lane's last column.
    let tmem_of = |cta: u32, lane: u32, col: u32, len: usize| -> crate::oplib::OpResult<(AllocId, u64)> {
        let al = *p.tmem.get(cta as usize).ok_or_else(|| OpError::invalid("mma tmem operand of a CTA outside the group"))?;
        if lane >= addr::TMEM_LANES || col >= addr::TMEM_COLS || col as u64 * 4 + len as u64 > addr::TMEM_COLS as u64 * 4 {
            return Err(OpError::invalid(format!("tmem cells ({lane}, {col}) + {len} bytes out of range")));
        }
        Ok((al, addr::tmem_byte_offset(lane, col)))
    };
    let tmem_read = |cta: u32, lane: u32, col: u32, out: &mut [u8]| -> crate::oplib::OpResult {
        let (al, off) = tmem_of(cta, lane, col, out.len())?;
        let ar = cell.borrow();
        let span = ByteSpan::new(off, out.len() as u64);
        if !fast_read(&ar, al, span, out) {
            ar.read(support::whole(&ar, al), &[span], out).map_err(|e| OpError::invalid(e.to_string()))?;
        }
        note(&mut reads.borrow_mut(), al, span);
        Ok(())
    };
    let mut tmem_write = |cta: u32, lane: u32, col: u32, data: &[u8]| -> crate::oplib::OpResult {
        let (al, off) = tmem_of(cta, lane, col, data.len())?;
        let mut ar = cell.borrow_mut();
        let span = ByteSpan::new(off, data.len() as u64);
        if !ar.is_overlaid(al) && !ar.get(al).metadata_only && span.end() <= ar.get(al).size {
            let a = ar.get_mut(al);
            a.bytes[span.start as usize..span.end() as usize].copy_from_slice(data);
            a.valid.set_range(span.start, span.len, true);
        } else {
            let v = support::whole(&ar, al);
            ar.write(v, &[span], data).map_err(|e| OpError::invalid(e.to_string()))?;
        }
        note(&mut writes, al, span);
        Ok(())
    };
    crate::oplib::tc_mma_ctas(p, &options, &smem, &tmem_read, &mut tmem_write)?;
    let mut r = reads.into_inner();
    coalesce(&mut r);
    coalesce(&mut writes);
    Ok((r, writes))
}

fn coalesce(v: &mut Spans) {
    v.sort();
    let mut out: Spans = Vec::with_capacity(v.len());
    for (a, s) in v.drain(..) {
        match out.last_mut() {
            Some((la, ls)) if *la == a && s.start <= ls.end() => {
                let end = ls.end().max(s.end());
                ls.len = end - ls.start;
            }
            _ => out.push((a, s)),
        }
    }
    *v = out;
}
