//! `RaceObserver`: the contract adapter. Converts `observe::Access` batches
//! and `observe::SyncEvent`s into the core's per-lane events and runs the
//! [`Checker`] online, inside the engine callbacks (on the CTA thread).
//!
//! # Merge design (`inbox_drain`)
//!
//! The scheduler is single-threaded (`sched` module doc): one observer sees
//! one total event order for the whole launch. The checker therefore keeps
//! a single global shadow, which is always merged, and `inbox_drain` is only
//! a GC safe point. When CTA parallelism lands (plan 2.4), each CTA thread
//! gets its own `Checker` view of shared memory and TMEM (CTA-private)
//! plus a per-stripe global shadow owned by the stripe's thread. The global
//! accesses a CTA performs in a round are then buffered and applied to the
//! owning stripe at the receiving CTA's next drain, in `(round, cta, seq)`
//! order. That is exactly the point where cross-CTA acquires become visible,
//! so validation is unchanged. `end_launch` drains every pending buffer
//! before finalising: an unmerged shadow is a merge point, never a silent
//! pass.

use std::collections::HashMap;

use crate::arena::{AllocId, ByteSpan, Space};
use crate::observe::{
    Access as CAccess, Actor, CtaId, FenceEvent, LaunchInfo, Observer, PublishTarget, SyncEvent as CSync, SyncKind,
    WarpEnd, WarpId as CWarpId,
};
use crate::program::{Proxy, Scope, Sem};
use crate::sync::completion::AsyncId;
use crate::value::LaneMask;

use super::checker::{Checker, Incomplete, Report as RaceReport, Stats};
use super::input as ri;

/// Racecheck options (the shape `numsim-py` constructs).
#[derive(Clone, Debug, Default)]
pub struct RacecheckConfig {
    /// Stop recording new findings after this many per launch (0 =
    /// unlimited); the run continues regardless (races never abort).
    pub max_findings: usize,
}

/// Internal ids of per-lane sub-ops (above any engine-assigned `AsyncId`).
const SUBOP_BASE: u64 = 1 << 62;

/// Default collector period (events).
pub const DEFAULT_GC_EVERY: u64 = 1 << 14;

/// One finished launch.
#[derive(Clone, Debug)]
pub struct LaunchResult {
    /// Launch ordinal within this observer's run.
    pub launch: u32,
    /// Kernel index within the `Module` (sites index its table).
    pub kernel: u32,
    pub report: RaceReport,
    pub stats: Stats,
    /// Buffer name and space per allocation, for evidence.
    pub buffers: HashMap<AllocId, (String, Space)>,
}

/// The checker as an [`Observer`]. Feed it to `sched::run`.
pub struct RaceObserver {
    pub config: RacecheckConfig,
    /// Run the dominated-frontier GC / async-slot reclaim every this many
    /// events (0 = never).
    pub gc_every: u64,
    checker: Option<Checker>,
    kernel: u32,
    buffers: HashMap<AllocId, (String, Space)>,
    /// Events delivered outside `begin_launch`/`end_launch`.
    outside_launch: u64,
    /// Per-thread async ops issued by several lanes in one instruction are
    /// split into one virtual actor per issuing lane (PTX async-groups and
    /// completions are per thread; contract review item 5). Never merged.
    subops: HashMap<AsyncId, Vec<(u8, AsyncId)>>,
    next_sub: u64,
    pub launches: Vec<LaunchResult>,
}

impl RaceObserver {
    pub fn new(config: RacecheckConfig) -> RaceObserver {
        RaceObserver { config, gc_every: DEFAULT_GC_EVERY, checker: None, kernel: 0, buffers: HashMap::new(), outside_launch: 0, subops: HashMap::new(), next_sub: SUBOP_BASE, launches: Vec::new() }
    }

    /// Start a launch from an explicit topology (tests and replay; the
    /// engine path uses `begin_launch`).
    pub fn start_launch(&mut self, topo: ri::Topology, kernel: u32) {
        if self.checker.is_some() {
            self.finish_launch();
        }
        let mut c = Checker::new(topo);
        c.gc_every = self.gc_every;
        c.max_findings = self.config.max_findings;
        self.kernel = kernel;
        self.checker = Some(c);
    }

    /// Register an allocation (normally via `begin_launch` or `AllocBegin`).
    pub fn register_alloc(&mut self, alloc: AllocId, space: Space, size: u64, name: &str) {
        self.buffers.insert(alloc, (name.to_string(), space));
        if let Some(c) = &mut self.checker {
            c.sync(ri::SyncEvent::AllocBegin { alloc, space, size, cta: 0 });
        }
    }

    /// Finalise the current launch (also called by `end_launch`).
    pub fn finish_launch(&mut self) {
        let Some(mut c) = self.checker.take() else { return };
        c.gc();
        let stats = c.stats;
        let mut report = c.finish();
        if self.outside_launch > 0 {
            report.incomplete.push(Incomplete::EventOutsideLaunch { events: self.outside_launch });
            self.outside_launch = 0;
        }
        self.launches.push(LaunchResult {
            launch: self.launches.len() as u32,
            kernel: self.kernel,
            report,
            stats,
            buffers: self.buffers.clone(),
        });
    }

    /// The live checker (tests).
    pub fn checker(&self) -> Option<&Checker> {
        self.checker.as_ref()
    }

    /// Final report over every launch (`numsim-py` entry point).
    pub fn finish(mut self) -> crate::report::Report {
        self.finish_launch();
        super::payload::report(&self)
    }
}

impl Default for RaceObserver {
    fn default() -> Self {
        RaceObserver::new(RacecheckConfig::default())
    }
}

fn span_range(s: ByteSpan) -> std::ops::Range<u64> {
    s.start..s.end()
}

/// Contract `Sem` → (order, strong scope, preceded by an SC fence).
fn normalize(sem: Sem, scope: Scope, atomic: bool) -> (ri::MemOrder, Option<Scope>, bool) {
    match sem {
        Sem::Weak if atomic => (ri::MemOrder::Relaxed, Some(scope), false),
        Sem::Weak => (ri::MemOrder::Weak, None, false),
        Sem::Relaxed => (ri::MemOrder::Relaxed, Some(scope), false),
        Sem::Acquire => (ri::MemOrder::Acquire, Some(scope), false),
        Sem::Release => (ri::MemOrder::Release, Some(scope), false),
        Sem::AcqRel => (ri::MemOrder::AcqRel, Some(scope), false),
        Sem::Sc => (ri::MemOrder::AcqRel, Some(scope), true),
        Sem::Volatile | Sem::Mmio => (ri::MemOrder::Relaxed, Some(Scope::Sys), false),
    }
}

impl Observer for RaceObserver {
    fn wants_word_history(&self) -> bool {
        true
    }

    fn begin_launch(&mut self, info: &LaunchInfo<'_>) {
        let s = info.shape;
        let topo = ri::Topology { warps_per_cta: s.warps_per_cta(), ctas_per_cluster: s.ctas_per_cluster().max(1), num_ctas: s.num_ctas() };
        self.start_launch(topo, info.kernel_index);
        for (id, a) in info.arena.iter() {
            if matches!(a.space, Space::Global | Space::Shared | Space::Tmem) {
                self.register_alloc(id, a.space, a.size, &a.name);
            }
        }
    }

    fn end_launch(&mut self, info: &LaunchInfo<'_>) {
        for (id, a) in info.arena.iter() {
            self.buffers.entry(id).or_insert_with(|| (a.name.clone(), a.space));
        }
        self.finish_launch();
    }

    fn access(&mut self, a: &CAccess<'_>) {
        let Some(c) = self.checker.as_mut() else {
            self.outside_launch += 1;
            return;
        };
        if !matches!(a.space, Space::Global | Space::Shared | Space::Tmem) {
            return; // local / param / reg are thread-private
        }
        let (order, scope, sc) = normalize(a.sem, a.scope, a.atomic);
        // The readonly proxy (ld.global.nc) is modelled as generic.
        let proxy = if a.proxy == Proxy::ReadOnly { Proxy::Generic } else { a.proxy };
        let base = |who| ri::Access {
            who,
            alloc: a.alloc,
            range: 0..0,
            kind: a.kind,
            order,
            scope,
            atomic: a.atomic,
            returns_value: a.returns_value,
            proxy,
            domain: a.window,
            site: a.site,
        };
        match a.actor {
            Actor::Warp { warp, epoch } => {
                if sc {
                    let mut lanes = LaneMask::NONE;
                    for s in a.spans {
                        lanes = lanes.or(LaneMask::lane(s.lane as usize & 31));
                    }
                    c.sync(ri::SyncEvent::Fence { warp: warp.0, lanes, kind: ri::FenceKind::Sc(scope.unwrap()), epoch });
                }
                for s in a.spans {
                    let mut x = base(ri::Who::Lane { warp: warp.0, lane: s.lane & 31, epoch });
                    x.range = span_range(s.span);
                    c.access(&x);
                }
            }
            Actor::Async { op, side } => {
                let subs = self.subops.get(&op);
                for s in a.spans {
                    let id = match subs {
                        None => op,
                        Some(subs) => match subs.iter().find(|(l, _)| *l == s.lane) {
                            Some((_, id)) => *id,
                            None => {
                                // A multi-lane per-thread op whose span does
                                // not name its lane: attributing it to the
                                // whole warp would merge lanes (false
                                // negatives) — fail closed.
                                c.note_incomplete(Incomplete::AsyncLaneUnknown { op });
                                continue;
                            }
                        },
                    };
                    let mut x = base(ri::Who::Async { op: id, side });
                    x.range = span_range(s.span);
                    c.access(&x);
                }
            }
            Actor::Host => {}
        }
    }

    fn sync(&mut self, e: &CSync) {
        if let SyncKind::AllocBegin { alloc, space, .. } = &e.kind {
            self.buffers.entry(*alloc).or_insert_with(|| (format!("{alloc}"), *space));
        }
        let Some(c) = self.checker.as_mut() else {
            self.outside_launch += 1;
            return;
        };
        let wa = match e.actor {
            Actor::Warp { warp, epoch } => Some((warp.0, epoch)),
            _ => None,
        };
        let lanes = e.lanes;
        let ev = match &e.kind {
            SyncKind::AllocBegin { alloc, space, size, cta } => {
                ri::SyncEvent::AllocBegin { alloc: *alloc, space: *space, size: *size, cta: cta.0 }
            }
            SyncKind::AllocEnd { alloc } => ri::SyncEvent::AllocEnd { alloc: *alloc },
            SyncKind::DeclareWord { alloc, span } => ri::SyncEvent::DeclareWord { alloc: *alloc, range: span_range(*span) },
            SyncKind::Protocol { .. } => return, // synccheck's input
            SyncKind::AsyncComplete { op, milestone, target } => {
                let (t, only) = match *target {
                    PublishTarget::Phase { obj, phase } => (ri::CompletionTarget::Phase { obj, phase }, LaneMask::ALL),
                    PublishTarget::Warp { warp, lanes } => (ri::CompletionTarget::Warp { warp: warp.0, lanes }, lanes),
                };
                match self.subops.get(op) {
                    // Per-thread completion: only the waiting lanes' sub-ops.
                    Some(subs) => {
                        for (l, id) in subs.clone() {
                            if only.contains(l as usize) {
                                c.sync(ri::SyncEvent::AsyncComplete { op: id, milestone: *milestone, target: t });
                            }
                        }
                        return;
                    }
                    None => ri::SyncEvent::AsyncComplete { op: *op, milestone: *milestone, target: t },
                }
            }
            kind => {
                let Some((warp, epoch)) = wa else {
                    self.outside_launch += 1;
                    return;
                };
                match kind {
                    SyncKind::WarpSync { mask } => ri::SyncEvent::WarpSync { warp, mask: *mask, epoch },
                    // `None` qualifier = lost in lowering → incomplete;
                    // `scope: None` = named barrier (participants).
                    SyncKind::Arrive { obj, phase, release, scope } => {
                        ri::SyncEvent::Arrive { warp, lanes, obj: *obj, phase: *phase, release: *release, scope: *scope, epoch }
                    }
                    SyncKind::Wait { obj, phase, acquire, scope } => {
                        ri::SyncEvent::Wait { warp, lanes, obj: *obj, phase: *phase, acquire: *acquire, scope: *scope, epoch }
                    }
                    SyncKind::Fence(f) => {
                        let kind = match *f {
                            FenceEvent::AcqRel(s) => ri::FenceKind::AcqRel(s),
                            FenceEvent::Sc(s) => ri::FenceKind::Sc(s),
                            FenceEvent::ProxyAsync(w) => ri::FenceKind::ProxyAsync(w),
                            FenceEvent::TcgenBefore => ri::FenceKind::TcgenBefore,
                            FenceEvent::TcgenAfter => ri::FenceKind::TcgenAfter,
                            FenceEvent::TensormapRelease { scope } => ri::FenceKind::TensormapRelease(scope),
                            FenceEvent::TensormapAcquire { scope, alloc, span } => {
                                ri::FenceKind::TensormapAcquire { scope, alloc, range: span_range(span) }
                            }
                            // mbarrier init visibility is synccheck's; the
                            // alias proxy orders virtual aliases, and the
                            // shadow is keyed by physical bytes already.
                            FenceEvent::MbarrierInit | FenceEvent::ProxyAlias => return,
                        };
                        ri::SyncEvent::Fence { warp, lanes, kind, epoch }
                    }
                    SyncKind::AsyncIssue { op, class, proxy, preds, footprint, .. } => {
                        let preds: Vec<AsyncId> = preds
                            .iter()
                            .flat_map(|p| match self.subops.get(p) {
                                Some(s) => s.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
                                None => vec![*p],
                            })
                            .collect();
                        let footprint: Vec<_> = footprint.iter().map(|(a, s)| (*a, span_range(*s))).collect();
                        if *class == crate::observe::AsyncClass::Copy && lanes.count() > 1 {
                            let mut subs = Vec::new();
                            for l in lanes.lanes() {
                                let id = AsyncId(self.next_sub);
                                self.next_sub += 1;
                                subs.push((l as u8, id));
                                c.sync(ri::SyncEvent::AsyncIssue {
                                    op: id,
                                    warp,
                                    lanes: LaneMask::lane(l),
                                    kind: *class,
                                    proxy: *proxy,
                                    preds: preds.clone(),
                                    footprint: footprint.clone(),
                                    site: e.site,
                                    epoch,
                                });
                            }
                            self.subops.insert(*op, subs);
                            return;
                        }
                        ri::SyncEvent::AsyncIssue {
                            op: *op,
                            warp,
                            lanes,
                            kind: *class,
                            proxy: *proxy,
                            preds,
                            footprint,
                            site: e.site,
                            epoch,
                        }
                    }
                    SyncKind::WaitVerdicts { alloc, span, scope, verdicts, pred_reads } => ri::SyncEvent::WaitVerdicts {
                        warp,
                        lanes,
                        alloc: *alloc,
                        range: span_range(*span),
                        scope: *scope,
                        verdicts: verdicts.iter().map(|v| (v.lanes, v.accepted.clone(), v.observed)).collect(),
                        pred_reads: pred_reads.iter().map(|(a, s)| (*a, span_range(*s))).collect(),
                        epoch,
                    },
                    _ => unreachable!("handled above"),
                }
            }
        };
        c.sync(ev);
    }

    fn warp_done(&mut self, warp: CWarpId, _end: WarpEnd) {
        if let Some(c) = &mut self.checker {
            c.warp_done(warp.0);
        }
    }

    fn inbox_drain(&mut self, _cta: CtaId, _round: u64) {
        // Single-threaded scheduler: the global shadow is always merged
        // (module doc). The drain is a safe point for the collectors.
        if let Some(c) = &mut self.checker {
            c.safe_point();
        }
    }
}
