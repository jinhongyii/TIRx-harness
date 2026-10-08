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
    pub(crate) fn run_serial(&mut self, env: &Env<'_>, arena: &mut Arena) -> Result<bool, ExecError> {
        if self.serial.is_empty() && self.sync.async_ops.is_empty() && self.sync.completions.is_empty() {
            return Ok(false);
        }
        let mut progress = false;
        for (ci, w) in std::mem::take(&mut self.serial) {
            let (r, progressed) = self.slice(ci, w, env, arena, 1);
            self.settle(ci, w, r, progressed)?;
            progress = true;
        }
        progress |= self.land(None, env, arena, true)?;
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
                if mine && ready && !Self::is_global_reduce(op, arena) && (all || self.rng.below(2) == 0) {
                    self.fire_op(i, env, arena)?;
                    landed_one = true;
                    any = true;
                    continue;
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
        loop {
            let Some(i) = self.sync.completions.iter().position(|c| self.sync.enabled(c)) else { break };
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
        let proxy = meta.as_ref().map(|m| m.proxy).unwrap_or_default();
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
            acc.items = reads.iter().map(|&(a, s)| (a, window(arena, a), LaneSpan { lane, span: s })).collect();
            support::emit_accesses(&mut self.events, &mut self.counters, &mut self.aux, arena, mk(Side::Read, AccessKind::Read), &mut acc);
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
                    support::emit_accesses(&mut self.events, &mut self.counters, &mut self.aux, arena, sp, &mut acc);
                }
                None => {
                    acc.items = writes.iter().map(|&(a, s)| (a, window(arena, a), LaneSpan { lane, span: s })).collect();
                    let wk = if rmw { AccessKind::Rmw } else { AccessKind::Write };
                    support::emit_accesses(&mut self.events, &mut self.counters, &mut self.aux, arena, mk(Side::Write, wk), &mut acc);
                }
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
    let smem = |cta: u32, a: u32, out: &mut [u8]| -> crate::oplib::OpResult {
        let al = *p.smem.get(cta as usize).ok_or_else(|| OpError::invalid("mma smem operand of a CTA outside the group"))?;
        let off = addr::decode_shared(a).1 as u64;
        let ar = cell.borrow();
        let span = ByteSpan::new(off, out.len() as u64);
        ar.read(support::whole(&ar, al), &[span], out).map_err(|e| OpError::invalid(e.to_string()))?;
        reads.borrow_mut().push((al, span));
        Ok(())
    };
    let tmem_of = |cta: u32, lane: u32, col: u32| -> crate::oplib::OpResult<(AllocId, u64)> {
        let al = *p.tmem.get(cta as usize).ok_or_else(|| OpError::invalid("mma tmem operand of a CTA outside the group"))?;
        if lane >= addr::TMEM_LANES || col >= addr::TMEM_COLS {
            return Err(OpError::invalid(format!("tmem cell ({lane}, {col}) out of range")));
        }
        Ok((al, addr::tmem_byte_offset(lane, col)))
    };
    let tmem_read = |cta: u32, lane: u32, col: u32, out: &mut [u8]| -> crate::oplib::OpResult {
        let (al, off) = tmem_of(cta, lane, col)?;
        let ar = cell.borrow();
        let span = ByteSpan::new(off, out.len() as u64);
        ar.read(support::whole(&ar, al), &[span], out).map_err(|e| OpError::invalid(e.to_string()))?;
        reads.borrow_mut().push((al, span));
        Ok(())
    };
    let mut tmem_write = |cta: u32, lane: u32, col: u32, data: &[u8]| -> crate::oplib::OpResult {
        let (al, off) = tmem_of(cta, lane, col)?;
        let mut ar = cell.borrow_mut();
        let span = ByteSpan::new(off, data.len() as u64);
        let v = support::whole(&ar, al);
        ar.write(v, &[span], data).map_err(|e| OpError::invalid(e.to_string()))?;
        writes.push((al, span));
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
