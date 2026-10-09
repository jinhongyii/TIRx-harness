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

use super::{CompletionPolicy, CtaState, Rng};
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
use crate::sync::{async_group, Completion, Outcome, Step, SyncTable};
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
    ExecError { kind, kernel, warp, pc: Pc(0), site, lanes: WarpMask::NONE, message, attrs: Default::default() }
}

/// An owned copy of an [`Access`] (its `seq` is assigned at replay). Its
/// lane spans live in the buffer's span pool (W13: one allocation reused
/// across rounds instead of one per access, freed on another thread).
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
    spans: std::ops::Range<usize>,
    declared_word: bool,
    operand: u8,
}

#[derive(Clone, Debug)]
enum Event {
    Access(OwnedAccess),
    Sync(SyncEvent),
    WarpDone(WarpId, WarpEnd),
    RoundBoundary(CtaId, u64),
}

/// Observer that records a partition's callbacks for ordered replay.
#[derive(Debug, Default)]
pub(crate) struct EventBuffer {
    pub enabled: bool,
    pub history: bool,
    events: Vec<Event>,
    /// Lane spans of the buffered accesses (`OwnedAccess::spans`).
    spans: Vec<LaneSpan>,
    /// Number of buffered `Event::Access` entries.
    accesses: u64,
}

impl EventBuffer {
    /// Rewrite the buffered `WaitVerdicts` history indices (`observed` and
    /// the `accepted` bits) through `map(alloc, span, local index)`: the
    /// partition numbered them against its local history; the launch
    /// history (delivery order) may have other partitions' same-round
    /// entries before its own (W6-P1). Unmapped bits are dropped.
    pub(crate) fn remap_verdicts(&mut self, mut map: impl FnMut(AllocId, ByteSpan, u32) -> Option<u32>) {
        for e in &mut self.events {
            let Event::Sync(se) = e else { continue };
            let SyncKind::WaitVerdicts { alloc, span, verdicts, .. } = &mut se.kind else { continue };
            for v in verdicts.iter_mut() {
                if let Some(o) = map(*alloc, *span, v.observed) {
                    v.observed = o;
                }
                let mut nb: Vec<u64> = Vec::new();
                for (wi, &w) in v.accepted.iter().enumerate() {
                    for b in 0..64 {
                        if w >> b & 1 == 1 {
                            if let Some(g) = map(*alloc, *span, (wi * 64 + b) as u32) {
                                let g = g as usize;
                                if nb.len() <= g / 64 {
                                    nb.resize(g / 64 + 1, 0);
                                }
                                nb[g / 64] |= 1 << (g % 64);
                            }
                        }
                    }
                }
                if nb.is_empty() {
                    nb.push(0);
                }
                v.accepted = nb;
            }
        }
    }
}

impl EventBuffer {
    /// Deliver and clear the buffered events; `Access::seq` is assigned
    /// here, in delivery order.
    pub fn replay(&mut self, observer: &mut dyn Observer, next_seq: &mut u64) {
        debug_assert_eq!(self.accesses, self.events.iter().filter(|e| matches!(e, Event::Access(_))).count() as u64);
        let pool = &self.spans;
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
                        spans: &pool[a.spans],
                        declared_word: a.declared_word,
                        operand: a.operand,
                    };
                    *next_seq += 1;
                    observer.access(&acc);
                }
                Event::Sync(s) => observer.sync(&s),
                Event::WarpDone(w, end) => observer.warp_done(w, end),
                Event::RoundBoundary(c, r) => observer.round_boundary(c, r),
            }
        }
        self.spans.clear();
        self.accesses = 0;
    }

    pub fn clear(&mut self) {
        self.events.clear();
        self.spans.clear();
        self.accesses = 0;
    }

    /// Buffered accesses (each takes one `Access::seq` at replay).
    pub fn access_count(&self) -> u64 {
        self.accesses
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
        let at = self.spans.len();
        self.spans.extend_from_slice(a.spans);
        self.accesses += 1;
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
            spans: at..self.spans.len(),
            declared_word: a.declared_word,
            operand: a.operand,
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
    fn round_boundary(&mut self, c: CtaId, r: u64) {
        self.events.push(Event::RoundBoundary(c, r));
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
            rng: Rng::new(seed ^ ((first_cluster as u64 + 1) << 32)),
            events: EventBuffer { enabled: observing, history, events: Vec::new(), spans: Vec::new(), accesses: 0 },
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
        self.ctas.iter().all(|c| c.finished())
    }

    /// One round of this partition: per CTA, mark the round boundary, run one
    /// slice per runnable warp, land async ops and apply enabled completions.
    pub(crate) fn run_round(&mut self, env: &Env<'_>, arena: &mut Arena) -> Result<bool, ExecError> {
        let mut progress = false;
        for ci in 0..self.ctas.len() {
            self.events.round_boundary(self.ctas[ci].ctx.id, env.round);
            progress |= self.run_cta(ci, env, arena)?;
            let cta = self.ctas[ci].ctx.id;
            progress |= self.land(Some(cta), env, arena, env.config.completions == CompletionPolicy::Eager)?;
            progress |= self.apply_completions(env)?;
        }
        Ok(progress)
    }

    /// Run one warp for at most `quantum` instructions; returns the result
    /// and whether a progress instruction completed.
    fn slice(&mut self, ci: usize, w: usize, env: &Env<'_>, arena: &mut Arena, quantum: u32) -> (StepResult, bool) {
        let Partition { ctas, sync, counters, aux, events, .. } = self;
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
            observer: events,
            observing: env.observing,
            counters,
            aux,
        };
        let r = crate::interp::guard_step(&mut ctx, quantum, env.step_fn);
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
            // W13: a retry that would only re-block at the same
            // `mbarrier.try_wait` (`interp::BlockedWait`) applies its
            // counters without re-executing; same state either way.
            let repeats = self.ctas[ci].warps[w].blocked_wait_repeats(&self.sync, arena);
            if repeats && !cfg!(debug_assertions) {
                self.ctas[ci].warps[w].repeat_blocked_wait();
                self.counters.instrs += 1;
                let WarpStatus::Blocked(res) = self.ctas[ci].warps[w].status else { unreachable!("checked by blocked_wait_repeats") };
                progress |= self.settle(ci, w, StepResult::Blocked(res), false)?;
                continue;
            }
            // Debug builds execute the retry and check the prediction.
            let before = repeats.then(|| {
                let wp = &self.ctas[ci].warps[w];
                (wp.status, wp.epoch, wp.steps, self.counters.instrs, wp.blocked_wait.as_ref().and_then(|b| b.state.as_ref().and(self.sync.get(b.res)).cloned()))
            });
            let (r, progressed) = self.slice(ci, w, env, arena, env.config.quantum);
            if let Some((status, epoch, steps, instrs, res_state)) = before {
                let wp = &self.ctas[ci].warps[w];
                let res = wp.blocked_wait.as_ref().map(|b| b.res);
                debug_assert!(
                    matches!(r, StepResult::Blocked(x) if Some(x) == res)
                        && !progressed
                        && wp.status == status
                        && wp.epoch == epoch + 1
                        && wp.steps == steps + 1
                        && self.counters.instrs == instrs + 1
                        && wp.blocked_wait.as_ref().and_then(|b| b.state.as_ref().and(self.sync.get(b.res)).cloned()) == res_state
                        && res.is_some(),
                    "blocked_wait_repeats predicted a re-block the retry did not reproduce: {r:?}"
                );
            }
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
        // Live op ids, built only when an op with `after` dependencies is
        // tested (most ops have none): readiness is then a set lookup, not
        // a rescan of the queue per dependency. Same order and RNG draws.
        let mut live: Option<std::collections::HashSet<crate::sync::AsyncId>> = None;
        loop {
            let mut landed_one = false;
            let mut i = 0;
            while i < self.sync.async_ops.len() {
                let op = &self.sync.async_ops[i];
                let mine = cta.is_none_or(|c| op.source.cta == c);
                let ready = mine
                    && (op.after.is_empty() || {
                        let ops = &self.sync.async_ops;
                        let l = live.get_or_insert_with(|| ops.iter().map(|o| o.id).collect());
                        op.after.iter().all(|d| !l.contains(d))
                    });
                if mine && ready && !self.deferred.contains(&op.id) && (all || self.rng.below(2) == 0) {
                    if Self::is_global_reduce(op, arena) {
                        self.deferred.push(op.id);
                    } else {
                        let id = op.id;
                        self.fire_op(i, env, arena)?;
                        if let Some(l) = live.as_mut() {
                            l.remove(&id);
                        }
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
        let mut reads: Vec<(AllocId, ByteSpan)> = take_spans();
        let mut writes: Vec<(AllocId, ByteSpan)> = take_spans();
        let mut rmw = false;
        // Reads whose bytes were invalid when read but are written by the same
        // op (MMA accumulator D): reported from the read-time check.
        let mut read_time_uninit: Vec<(AllocId, ByteSpan)> = Vec::new();
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
                // Fast path (as `run_mma`'s): private (non-overlaid) source
                // and destination whose source bytes are all valid move
                // through the allocations' byte arrays directly; anything
                // else takes the generic arena read/write (same result).
                let direct = |ar: &Arena, al: AllocId| !ar.is_overlaid(al) && !ar.get(al).metadata_only;
                for (&(sa, ss), &(da, ds)) in src.iter().zip(dst) {
                    let mut buf = [0u8; 8];
                    let n = ss.len as usize;
                    let fast = n <= buf.len()
                        && direct(arena, sa)
                        && direct(arena, da)
                        && ss.end() <= arena.get(sa).size
                        && ds.end() <= arena.get(da).size
                        && arena.get(sa).valid.first_clear(ss.start, ss.len).is_none();
                    let b: &mut [u8] = &mut buf[..n.min(8)];
                    let mut heap;
                    let b: &mut [u8] = if fast {
                        b.copy_from_slice(&arena.get(sa).bytes[ss.start as usize..ss.end() as usize]);
                        b
                    } else {
                        heap = vec![0u8; n];
                        arena.read(support::whole(arena, sa), &[ss], &mut heap).map_err(|e| src_err(e.to_string()))?;
                        &mut heap
                    };
                    let cell = crate::oplib::tcgen_cp_decode(b, *decompress_bits)
                        .map_err(|e| sched_error(ExecErrorKind::Op(e.kind), kernel, op.source.warp, op.source.site, e.message))?;
                    if fast && cell.len() as u64 == ds.len {
                        let a = arena.get_mut(da);
                        a.bytes[ds.start as usize..ds.end() as usize].copy_from_slice(&cell);
                        a.valid.set_range(ds.start, ds.len, true);
                    } else {
                        arena.write(support::whole(arena, da), &[ds], &cell).map_err(|e| src_err(e.to_string()))?;
                    }
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
                // Read spans feed only access events and the ZeroAndReport
                // uninit reports. Without an observer only the reports need
                // them, and a piece every byte of which was valid when read
                // contributes nothing to those (W13): a landing only makes
                // bytes valid (D writes), so every byte invalid at report
                // time lies in a piece read through the general path, and
                // the maximal invalid runs `report_async_uninit` finds in
                // the merged spans are the same with or without the valid
                // pieces. So only those pieces are noted then.
                let track_reads = env.observing || arena.policy() == crate::arena::ValidityPolicy::ZeroAndReport;
                let (r, w, u) = run_mma(arena, p, tc_arch(env.program.arch.as_deref()), meta.as_ref().and_then(|m| m.lut_b), track_reads, env.observing)
                    .map_err(|e| sched_error(ExecErrorKind::Op(e.kind), kernel, op.source.warp, op.source.site, e.message))?;
                // Take the lists over (no copy); they go back to the pool
                // at the end of the landing.
                if reads.is_empty() {
                    give_spans(std::mem::replace(&mut reads, r));
                } else {
                    reads.extend_from_slice(&r);
                    give_spans(r);
                }
                read_time_uninit = u;
                if writes.is_empty() {
                    give_spans(std::mem::replace(&mut writes, w));
                } else {
                    writes.extend_from_slice(&w);
                    give_spans(w);
                }
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
            for &(a, s) in &read_time_uninit {
                self.report_read_time_uninit(env, arena, op.id, op.source.site, lane, a, s);
            }
        }
        // Declared-word writes are counted (and, under a history observer,
        // logged with images) in every mode, in the order the write accesses
        // are reported below (W13-1).
        if !self.aux.words.is_empty() {
            let window = |arena: &Arena, a: AllocId| match arena.get(a).space {
                Space::Global => Some(Window::Global),
                Space::Shared => Some(Window::SharedCta),
                _ => None,
            };
            let elem = match &op.payload {
                Payload::Reduce { dtype, .. } | Payload::ReduceData { dtype, .. } => Some(dtype.mem_bytes() as u64),
                _ => None,
            };
            // Only allocations holding declared words are logged (the rest
            // log nothing); filtering keeps the logged order (W13).
            let words = &self.aux.words;
            let mut items: Vec<(AllocId, Option<Window>, LaneSpan)> = match elem {
                Some(eb) => writes
                    .iter()
                    .filter(|(a, _)| words.has(*a))
                    .flat_map(|&(a, s)| {
                        let w = window(arena, a);
                        (0..s.len / eb.max(1)).map(move |k| (a, w, LaneSpan { lane, span: ByteSpan::new(s.start + k * eb, eb) }))
                    })
                    .collect(),
                None => writes.iter().filter(|(a, _)| words.has(*a)).map(|&(a, s)| (a, window(arena, a), LaneSpan { lane, span: s })).collect(),
            };
            support::log_async_writes(&mut self.aux, arena, &mut items);
            let words = &self.aux.words;
            let mut items: Vec<(AllocId, Option<Window>, LaneSpan)> =
                frag_writes.iter().filter(|(a, _)| words.has(*a)).map(|&(a, s)| (a, window(arena, a), LaneSpan { lane, span: s })).collect();
            support::log_async_writes(&mut self.aux, arena, &mut items);
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
                // Copies: destination 0, source 1 (PTX operand order); the
                // read side is refined below for MMA operands.
                operand: if side == Side::Write { 0 } else { 1 },
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
            // Operand of each read group (`Access::operand` indexes the
            // site's pointer operands, W5-15). MMA: TMEM reads take their
            // pointer operand's index (d, [a_tmem], [lut | sp_meta],
            // [sfa, sfb]: `mma_tmem_operands`); shared reads go through the
            // A/B descriptors, which are not pointer operands, and take
            // MMA_SHARED_A / MMA_SHARED_B (past every pointer operand, so
            // they never borrow a TMEM operand's buffer name).
            // tcgen05.cp taddr/s-desc = 0/1; copies src = 1.
            let is_mma = op.kind == crate::sync::AsyncKind::TcgenMma;
            let shared_operand = if is_mma { if is_a_read { MMA_SHARED_A } else { MMA_SHARED_B } } else { 1 };
            let other_operand = if tc_op { 0 } else { 1 };
            let mut groups: Vec<ReadGroup> = vec![(shared_reads, Proxy::Async, shared_operand)];
            match &op.payload {
                Payload::TcgenMma(p) if is_mma && !is_a_read => {
                    let bases = mma_tmem_operands(p, meta.as_ref().and_then(|m| m.lut_b));
                    for (operand, spans) in split_tmem_reads(&other_reads, &bases) {
                        groups.push((spans, proxy, operand));
                    }
                }
                _ => groups.push((other_reads, proxy, other_operand)),
            }
            for (group, prox, operand) in groups {
                if group.is_empty() {
                    continue;
                }
                acc.items = group.iter().map(|&(a, s)| (a, window(arena, a), LaneSpan { lane, span: s })).collect();
                let mut sp = mk(Side::Read, AccessKind::Read);
                sp.proxy = prox;
                sp.operand = operand;
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
        if let (Some(mode), Payload::Copy { src, .. }) = (meta.as_ref().and_then(|m| m.report), &op.payload) {
            let matched = match mode {
                crate::program::ReportMode::PerElementFf => src.iter().any(|&(a, s)| arena.read_raw(a, s).contains(&0xff)),
                crate::program::ReportMode::Per16BytesPattern { pattern, bits } => report_16(arena, src, pattern, bits),
                crate::program::ReportMode::Per16Bytes => false,
            };
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
        give_spans(reads);
        give_spans(writes);
        give_spans(read_time_uninit);
        Ok(())
    }
}

impl Partition {
    /// One `UninitRead` finding for `span` of `alloc`, whose bytes were
    /// invalid when the op read them (since overwritten by the op itself).
    fn report_read_time_uninit(&mut self, env: &Env<'_>, arena: &Arena, op: crate::sync::AsyncId, site: SiteId, lane: u8, alloc: AllocId, span: ByteSpan) {
        if self.aux.uninit_seen.insert((site, alloc, span)) {
            let a = arena.get(alloc);
            let actor = Actor::Async { op, side: Side::Read };
            let l = if lane == ALL_LANES { 0 } else { lane as usize };
            self.aux.diagnostics.push(support::uninit_finding(env.kernel, site, Some(actor), a.space, alloc, &a.name, span, l));
        }
    }

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
    // W13: each span minus the `minus` spans of its allocation, as
    // ascending pieces, in input order, without re-splitting every span
    // against every `minus` span (quadratic; 62% of an observed Mega MoE
    // run). Per allocation: the union of the non-empty `minus` spans
    // (sorted, overlapping/touching merged), the raw non-empty spans, and
    // the empty ones, which only cut a piece they fall strictly inside
    // (`ByteSpan::overlaps` of the previous piecewise loop).
    struct Minus {
        alloc: AllocId,
        union: Vec<ByteSpan>,
        raw: Vec<ByteSpan>,
        cuts: Vec<u64>,
    }
    let mut per: Vec<Minus> = Vec::new();
    {
        let mut m: Vec<(AllocId, ByteSpan)> = minus.to_vec();
        m.sort_unstable();
        for (a, s) in m {
            if per.last().is_none_or(|x| x.alloc != a) {
                per.push(Minus { alloc: a, union: Vec::new(), raw: Vec::new(), cuts: Vec::new() });
            }
            let x = per.last_mut().expect("entry");
            if s.len == 0 {
                x.cuts.push(s.start);
                continue;
            }
            x.raw.push(s);
            match x.union.last_mut() {
                Some(l) if s.start <= l.end() => {
                    let end = l.end().max(s.end());
                    l.len = end - l.start;
                }
                _ => x.union.push(s),
            }
        }
    }
    let mut out = Vec::with_capacity(spans.len());
    for &(a, s) in spans {
        let Ok(k) = per.binary_search_by_key(&a, |x| x.alloc) else {
            out.push((a, s));
            continue;
        };
        let m = &per[k];
        if s.len == 0 {
            // An empty span goes when a non-empty one strictly contains it.
            if !m.raw.iter().any(|r| s.overlaps(*r)) {
                out.push((a, s));
            }
            continue;
        }
        let emit = |lo: u64, hi: u64, out: &mut Vec<(AllocId, ByteSpan)>| {
            // Cuts strictly inside [lo, hi) split the piece.
            let mut at = lo;
            let mut c = m.cuts.partition_point(|&x| x <= lo);
            while c < m.cuts.len() && m.cuts[c] < hi {
                if m.cuts[c] > at {
                    out.push((a, ByteSpan::new(at, m.cuts[c] - at)));
                    at = m.cuts[c];
                }
                c += 1;
            }
            out.push((a, ByteSpan::new(at, hi - at)));
        };
        let mut k = m.union.partition_point(|u| u.end() <= s.start);
        let mut at = s.start;
        while k < m.union.len() && m.union[k].start < s.end() {
            if m.union[k].start > at {
                emit(at, m.union[k].start, &mut out);
            }
            at = at.max(m.union[k].end());
            k += 1;
        }
        if at < s.end() {
            emit(at, s.end(), &mut out);
        }
    }
    out
}

#[cfg(test)]
mod subtract_tests {
    use super::*;

    /// The previous (quadratic) implementation: the reference.
    fn subtract_spans_reference(spans: &[(AllocId, ByteSpan)], minus: &[(AllocId, ByteSpan)]) -> Vec<(AllocId, ByteSpan)> {
        let mut out = Vec::new();
        for &(a, s) in spans {
            let mut pieces = vec![s];
            for &(_, m) in minus.iter().filter(|(ma, _)| *ma == a) {
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

    #[test]
    fn matches_reference() {
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = |n: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % n
        };
        for case in 0..3000 {
            let width = 8 + rnd(300);
            let maxlen = 1 + rnd(if case % 2 == 0 { 8 } else { 80 });
            let gen = |rnd: &mut dyn FnMut(u64) -> u64| -> Vec<(AllocId, ByteSpan)> {
                let n = rnd(40);
                (0..n).map(|_| (AllocId(rnd(3) as u32), ByteSpan::new(rnd(width), rnd(maxlen + 1)))).collect()
            };
            let spans = gen(&mut rnd);
            let minus = gen(&mut rnd);
            assert_eq!(subtract_spans(&spans, &minus), subtract_spans_reference(&spans, &minus), "case {case}: {spans:?} - {minus:?}");
        }
    }
}

/// `.per_16bytes` copy report (W2-8): in each 16-byte chunk of the source
/// address space the copy reads, the lowest-addressed copied element of
/// `bits` bits is compared with `pattern`.
fn report_16(arena: &Arena, src: &[(AllocId, ByteSpan)], pattern: u32, bits: u8) -> bool {
    let eb = (bits as u64).div_ceil(8).max(1);
    let mut first: std::collections::BTreeMap<(AllocId, u64), u64> = std::collections::BTreeMap::new();
    for &(a, s) in src {
        let base = arena.get(a).base;
        let mut off = s.start;
        while off < s.end() {
            let chunk = (base + off) / 16;
            let e = first.entry((a, chunk)).or_insert(off);
            *e = (*e).min(off);
            off = (chunk + 1) * 16 - base;
        }
    }
    first.into_iter().any(|((a, _), off)| {
        let b = arena.read_raw(a, ByteSpan::new(off, eb));
        let v = b.iter().rev().fold(0u32, |acc, &x| (acc << 8) | x as u32);
        let v = if bits == 4 { v & 0xf } else { v };
        v == pattern
    })
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
    let mut src_runs = take_spans();
    merge_runs_into(src, &mut src_runs);
    if src_runs.iter().all(|&(a, s)| !arena.get(a).metadata_only && arena.first_invalid(support::whole(arena, a), s).is_none()) {
        thread_local! {
            /// Reusable gather buffer (W13: a fresh one per landing was
            /// GBs of allocation per Mega MoE run).
            static COPY_BUF: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let mut bytes = COPY_BUF.with(|b| std::mem::take(&mut *b.borrow_mut()));
        bytes.clear();
        bytes.resize(total(src) as usize, 0);
        let mut at = 0usize;
        for &(a, s) in &src_runs {
            let out = &mut bytes[at..at + s.len as usize];
            if arena.is_overlaid(a) {
                arena.read_raw_into(a, s, out);
            } else {
                out.copy_from_slice(&arena.get(a).bytes[s.start as usize..s.end() as usize]);
            }
            at += s.len as usize;
        }
        let mut dst_runs = take_spans();
        merge_runs_into(dst, &mut dst_runs);
        let mut pos = 0usize;
        for &(a, s) in &dst_runs {
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
        give_spans(src_runs);
        give_spans(dst_runs);
        if bytes.capacity() <= 1 << 24 {
            COPY_BUF.with(|b| *b.borrow_mut() = bytes);
        }
        return Ok(());
    }
    give_spans(src_runs);
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
fn merge_runs_into(v: &[(AllocId, ByteSpan)], out: &mut Vec<(AllocId, ByteSpan)>) {
    out.clear();
    for &(a, s) in v {
        match out.last_mut() {
            Some((la, ls)) if *la == a && ls.end() == s.start => ls.len += s.len,
            _ => out.push((a, s)),
        }
    }
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
/// Returns (reads, writes, TMEM reads that saw invalid bytes when they were
/// made: the accumulator D read before the MMA writes D, W9 phase 6).
/// Read spans emitted as one access group: (spans, proxy, operand).
type ReadGroup = (Vec<(AllocId, ByteSpan)>, Proxy, u8);

/// `Access::operand` of an MMA's shared-memory A / B reads (through
/// descriptors, not pointer operands).
pub const MMA_SHARED_A: u8 = 240;
pub const MMA_SHARED_B: u8 = 241;

/// TMEM pointer operands of an MMA in site operand order (the TVM table's
/// `addr` slots: d, [a_tmem], [b_decompress_metadata | sp_meta_tmem],
/// [sfa_tmem, sfb_tmem]): (operand index, lane, column) of each base.
fn mma_tmem_operands(p: &crate::sync::completion::TcgenMmaPayload, lut_b: Option<u32>) -> Vec<(u8, u32, u32)> {
    let mut v: Vec<u32> = vec![p.d_taddr];
    if matches!(p.args.a, crate::program::TcA::Tmem(_)) {
        v.push(p.a as u32);
    }
    if let Some(t) = lut_b.filter(|_| p.args.lut_b) {
        v.push(t);
    }
    if let Some(t) = p.sparse_meta {
        v.push(t);
    }
    if let Some((a, b)) = p.scale_taddrs {
        v.push(a);
        v.push(b);
    }
    v.iter()
        .enumerate()
        .map(|(i, &t)| {
            let (lane, col) = addr::tmem_decode(t);
            (i as u8, lane, col)
        })
        .collect()
}

/// Split TMEM read spans by operand: each byte belongs to the operand with
/// the greatest base column at or below its column (ties: greatest base lane
/// at or below its lane); bytes below every base are D's. Pieces are cut at
/// lane rows and base columns. Groups come out in operand order.
fn split_tmem_reads(reads: &[(AllocId, ByteSpan)], bases: &[(u8, u32, u32)]) -> Vec<(u8, Vec<(AllocId, ByteSpan)>)> {
    let row = addr::TMEM_COLS as u64 * 4;
    let owner = |lane: u32, col: u32| -> u8 {
        bases
            .iter()
            .filter(|&&(_, bl, bc)| bc <= col && bl <= lane)
            .max_by_key(|&&(i, bl, bc)| (bc, bl, std::cmp::Reverse(i)))
            .or_else(|| bases.iter().filter(|&&(_, _, bc)| bc <= col).max_by_key(|&&(i, _, bc)| (bc, std::cmp::Reverse(i))))
            .map_or(0, |b| b.0)
    };
    let mut out: Vec<(u8, Vec<(AllocId, ByteSpan)>)> = Vec::new();
    for &(a, s) in reads {
        let mut x = s.start;
        while x < s.end() {
            let lane = (x / row) as u32;
            let col = ((x % row) / 4) as u32;
            let mut end = s.end().min((x / row + 1) * row);
            for &(_, _, bc) in bases {
                let b = (x / row) * row + bc as u64 * 4;
                if b > x && b < end {
                    end = b;
                }
            }
            let o = owner(lane, col);
            let piece = ByteSpan::new(x, end - x);
            match out.iter_mut().find(|g| g.0 == o) {
                Some(g) => match g.1.last_mut() {
                    Some((la, ls)) if *la == a && ls.end() == piece.start => ls.len += piece.len,
                    _ => g.1.push((a, piece)),
                },
                None => out.push((o, vec![(a, piece)])),
            }
            x = end;
        }
    }
    out.sort_by_key(|g| g.0);
    out
}

thread_local! {
    /// Reusable span lists for MMA landings (W13: their growth was ~10 GB of
    /// allocation per Mega MoE medium run). Per thread; contents never
    /// outlive a landing.
    static SPAN_POOL: std::cell::RefCell<Vec<Spans>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// An empty span list, reusing a returned one's capacity when available.
fn take_spans() -> Spans {
    SPAN_POOL.with(|p| p.borrow_mut().pop()).unwrap_or_default()
}

/// Return a span list for reuse (bounded count and capacity).
fn give_spans(mut v: Spans) {
    v.clear();
    if v.capacity() == 0 || v.capacity() > 1 << 20 {
        return;
    }
    SPAN_POOL.with(|p| {
        let mut p = p.borrow_mut();
        if p.len() < 16 {
            p.push(v);
        }
    });
}

fn run_mma(
    arena: &mut Arena,
    p: &crate::sync::completion::TcgenMmaPayload,
    arch: crate::oplib::TcArch,
    lut_b: Option<u32>,
    track_reads: bool,
    note_all: bool,
) -> crate::oplib::OpResult<(Spans, Spans, Spans)> {
    use crate::oplib::OpError;
    use std::cell::RefCell;
    let cell = RefCell::new(arena);
    let reads: RefCell<Spans> = RefCell::new(take_spans());
    let uninit: RefCell<Spans> = RefCell::new(take_spans());
    let mut writes: Spans = take_spans();
    let options = crate::oplib::TcMmaOptions { arch, ti16: p.args.kind == crate::program::TcMmaKind::Ti16, lut_b, ..Default::default() };
    // oplib reads/writes one small piece at a time: record them merged with
    // the previous piece when contiguous (most are; only when a consumer
    // needs the read spans), and read/write private (non-overlaid)
    // allocations directly when every byte is valid.
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
        let fast = fast_read(&ar, al, span, out);
        if !fast {
            ar.read(support::whole(&ar, al), &[span], out).map_err(|e| OpError::invalid(e.to_string()))?;
        }
        if track_reads && (note_all || !fast) {
            note(&mut reads.borrow_mut(), al, span);
        }
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
        let fast = fast_read(&ar, al, span, out);
        if !fast {
            if ar.first_invalid(support::whole(&ar, al), span).is_some() {
                note(&mut uninit.borrow_mut(), al, span);
            }
            ar.read(support::whole(&ar, al), &[span], out).map_err(|e| OpError::invalid(e.to_string()))?;
        }
        if track_reads && (note_all || !fast) {
            note(&mut reads.borrow_mut(), al, span);
        }
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
    // Resolved operand windows (contract W4-mma-window): every piece that
    // is in bounds and fully valid is served from the allocation directly;
    // the rest go to the callbacks above. Window-served pieces are fully
    // valid, so they are recorded only when observing (`note_all`), like the
    // callbacks' fast path.
    let window_reads: RefCell<Vec<(crate::oplib::TcSpace, u32, u64, u64)>> = RefCell::new(Vec::new());
    let window = |al: AllocId| -> Option<crate::oplib::TcWindow<'_>> {
        let ar = cell.borrow();
        if ar.is_overlaid(al) || ar.get(al).metadata_only {
            return None;
        }
        let bytes = std::cell::Ref::map(cell.borrow(), |ar| &ar.get(al).bytes[..]);
        let valid = std::cell::Ref::map(cell.borrow(), |ar| &ar.get(al).valid);
        Some(crate::oplib::TcWindow { bytes, valid })
    };
    let per_cta = |v: &[AllocId]| -> [Option<crate::oplib::TcWindow<'_>>; 2] { [v.first().and_then(|&a| window(a)), v.get(1).and_then(|&a| window(a))] };
    let io = crate::oplib::TcMmaIo {
        windows: RefCell::new(Some([per_cta(&p.smem), per_cta(&p.tmem)])),
        reads: (track_reads && note_all).then_some(&window_reads),
    };
    crate::oplib::tc_mma_ctas(p, &options, &smem, &tmem_read, &mut tmem_write, Some(&io))?;
    drop(io);
    let mut r = reads.into_inner();
    // Merged with the callbacks' notes; `coalesce` makes the order moot.
    for (space, cta, off, len) in window_reads.into_inner() {
        let v = match space {
            crate::oplib::TcSpace::Shared => &p.smem,
            crate::oplib::TcSpace::Tmem => &p.tmem,
        };
        if let Some(&al) = v.get(cta as usize) {
            r.push((al, ByteSpan::new(off, len)));
        }
    }
    coalesce(&mut r);
    coalesce(&mut writes);
    let mut u = uninit.into_inner();
    coalesce(&mut u);
    Ok((r, writes, u))
}

/// Sort and merge overlapping/adjacent spans per allocation.
fn coalesce(v: &mut Spans) {
    if v.len() >= 64 && coalesce_dense(v) {
        return;
    }
    coalesce_sorted(v);
}

fn coalesce_sorted(v: &mut Spans) {
    // Unstable is exact here: equal `(AllocId, ByteSpan)` keys are
    // indistinguishable (W13: the stable sort's buffer moves were ~10% of
    // an fp8 GEMM run).
    v.sort_unstable();
    // Merge in place (W13: no second list).
    let mut n = 0usize;
    for i in 0..v.len() {
        let (a, s) = v[i];
        if n > 0 {
            let (la, ls) = &mut v[n - 1];
            if *la == a && s.start <= ls.end() {
                let end = ls.end().max(s.end());
                ls.len = end - ls.start;
                continue;
            }
        }
        v[n] = (a, s);
        n += 1;
    }
    v.truncate(n);
}

/// [`coalesce_sorted`] through one bitmap per allocation (W13): an MMA
/// notes thousands of small spans over a few compact shared/TMEM ranges,
/// where marking bytes and reading back the maximal runs is linear. The
/// result is the same: the union of the spans as maximal runs (touching
/// spans merged), in (allocation, start) order. Returns false, leaving `v`
/// untouched, when the spans are not dense (or a span is empty, which the
/// sorted merge keeps as its own entry).
fn coalesce_dense(v: &mut Spans) -> bool {
    let mut ranges: Vec<(AllocId, u64, u64)> = Vec::new();
    for &(a, s) in v.iter() {
        if s.len == 0 {
            return false;
        }
        match ranges.iter_mut().find(|r| r.0 == a) {
            Some(r) => {
                r.1 = r.1.min(s.start);
                r.2 = r.2.max(s.end());
            }
            None => {
                if ranges.len() == 8 {
                    return false;
                }
                ranges.push((a, s.start, s.end()));
            }
        }
    }
    let words: u64 = ranges.iter().map(|r| (r.2 - r.1).div_ceil(64)).sum();
    if words > 8 * v.len() as u64 + 1024 {
        return false;
    }
    ranges.sort_unstable_by_key(|r| r.0);
    thread_local! {
        /// One reusable word buffer for every range's bitmap (W13: a fresh
        /// bitmap per call was ~10 GB of allocation per Mega MoE medium run).
        static BITS: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let mut buf = BITS.with(|b| std::mem::take(&mut *b.borrow_mut()));
    buf.clear();
    buf.resize(words as usize, 0);
    let mut starts: [usize; 9] = [0; 9];
    for (k, r) in ranges.iter().enumerate() {
        starts[k + 1] = starts[k] + (r.2 - r.1).div_ceil(64) as usize;
    }
    let mut bits: Vec<&mut [u64]> = Vec::with_capacity(ranges.len());
    {
        let mut rest: &mut [u64] = &mut buf;
        for k in 0..ranges.len() {
            let (head, tail) = rest.split_at_mut(starts[k + 1] - starts[k]);
            bits.push(head);
            rest = tail;
        }
    }
    for &(a, s) in v.iter() {
        let k = ranges.iter().position(|r| r.0 == a).expect("range");
        let (b, lo) = (&mut bits[k], ranges[k].1);
        let (mut i, end) = (s.start - lo, s.end() - lo);
        while i < end {
            let w = (i / 64) as usize;
            let bit = i % 64;
            let n = (64 - bit).min(end - i);
            b[w] |= if n == 64 { u64::MAX } else { ((1u64 << n) - 1) << bit };
            i += n;
        }
    }
    v.clear();
    for (r, b) in ranges.iter().zip(&bits) {
        // Maximal runs of set bits, word by word (bits past the range's
        // end are never set).
        let mut start: Option<u64> = None;
        for (wi, &w) in b.iter().enumerate() {
            let base = wi as u64 * 64;
            match start {
                None if w == 0 => continue,
                Some(_) if w == u64::MAX => continue,
                _ => {}
            }
            let mut pos = 0u32;
            while pos < 64 {
                let y = if start.is_some() { !w } else { w } >> pos;
                if y == 0 {
                    break;
                }
                let t = pos + y.trailing_zeros();
                match start.take() {
                    Some(st) => v.push((r.0, ByteSpan::new(r.1 + st, base + t as u64 - st))),
                    None => start = Some(base + t as u64),
                }
                pos = t;
            }
        }
        if let Some(st) = start {
            let end = b.len() as u64 * 64;
            v.push((r.0, ByteSpan::new(r.1 + st, end - st)));
        }
    }
    drop(bits);
    BITS.with(|b| *b.borrow_mut() = buf);
    true
}

#[cfg(test)]
mod coalesce_tests {
    use super::*;

    #[test]
    fn dense_matches_sorted_merge() {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = |n: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % n
        };
        let mut dense = 0;
        for case in 0..400 {
            let n = 64 + rnd(600) as usize;
            let allocs = 1 + rnd(4) as u32;
            let width = 64 + rnd(8192);
            let v: Spans = (0..n)
                .map(|_| (AllocId(rnd(allocs as u64) as u32 * 3), ByteSpan::new(1000 + rnd(width), 1 + rnd(if case % 3 == 0 { 3 } else { 40 }))))
                .collect();
            let mut want = v.clone();
            coalesce_sorted(&mut want);
            let mut got = v.clone();
            if coalesce_dense(&mut got) {
                dense += 1;
                assert_eq!(got, want, "case {case}");
            } else {
                assert_eq!(got, v, "case {case}: a declined dense pass leaves the spans");
            }
            let mut via = v;
            coalesce(&mut via);
            assert_eq!(via, want, "case {case}");
        }
        assert!(dense > 300, "{dense} dense cases");
    }
}
