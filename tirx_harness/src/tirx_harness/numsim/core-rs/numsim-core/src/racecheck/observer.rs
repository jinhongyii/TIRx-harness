//! `RaceObserver`: the contract adapter. Converts `observe::Access` batches
//! and `observe::SyncEvent`s into the core's per-lane events and runs the
//! [`Checker`] online, inside the engine callbacks (on the CTA thread).
//!
//! # Merge design (`round_boundary`)
//!
//! The scheduler is single-threaded (`sched` module doc): one observer sees
//! one total event order for the whole launch. The checker therefore keeps
//! a single global shadow, which is always merged, and `round_boundary` is only
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
use std::ops::Range;

use crate::arena::{AllocId, ByteSpan, Space};
use crate::observe::{
    Access as CAccess, Actor, CtaId, FenceEvent, ForkedObserver, LaunchInfo, Observer, PartitionInfo, PublishTarget,
    SyncEvent as CSync, SyncKind, WarpEnd, WarpId as CWarpId, ALL_LANES,
};
use crate::program::{Proxy, Scope, Sem};
use crate::sync::completion::AsyncId;
use crate::value::LaneMask;

use super::checker::{pool_key, Checker, Incomplete, Report as RaceReport, Stashed, Stats};
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
    /// Logical buffer name per site (`SiteInfo::buffer`).
    pub buffer_of_site: HashMap<crate::site::SiteId, String>,
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
    subops: Subops,
    pub launches: Vec<LaunchResult>,
    buffer_of_site: HashMap<crate::site::SiteId, String>,
    /// Decision 17: process each scheduling partition's replayed events in
    /// a forked child checker (default: `tuning::FORK_JOIN`).
    pub fork_join: bool,
    /// Collect only at the scheduler's `phase_end` (review D7; on whenever
    /// `fork_join` is).
    pub phase_gc: bool,
}

impl RaceObserver {
    pub fn new(config: RacecheckConfig) -> RaceObserver {
        let fork_join = super::tuning::on(&super::tuning::FORK_JOIN);
        RaceObserver {
            config,
            gc_every: DEFAULT_GC_EVERY,
            checker: None,
            kernel: 0,
            buffers: HashMap::new(),
            outside_launch: 0,
            subops: Subops::default(),
            launches: Vec::new(),
            buffer_of_site: HashMap::new(),
            fork_join,
            // D7: collect at the scheduler's phase ends in both modes.
            phase_gc: true,
        }
    }

    /// Start a launch from an explicit topology (tests and replay; the
    /// engine path uses `begin_launch`).
    pub fn start_launch(&mut self, topo: ri::Topology, kernel: u32) {
        if self.checker.is_some() {
            self.finish_launch();
        }
        let mut c = Checker::new(topo);
        c.gc_every = self.gc_every;
        c.set_collect_at_phase_end(self.phase_gc || self.fork_join);
        c.set_fork_join(self.fork_join);
        c.max_findings = self.config.max_findings;
        self.kernel = kernel;
        self.checker = Some(c);
    }

    /// Logical buffer names per site (normally from `Program::sites`); enables
    /// the `alias_stale_read` advisory.
    pub fn set_site_buffers(&mut self, names: Vec<(crate::site::SiteId, String)>) {
        if let Some(c) = &mut self.checker {
            c.site_buffer = names.iter().map(|(s, b)| (*s, std::sync::Arc::<str>::from(b.as_str()))).collect();
        }
        self.buffer_of_site = names.into_iter().collect();
    }

    /// Sites of `wait_until` polls (normally from the program's site table).
    pub fn set_poll_sites(&mut self, sites: impl IntoIterator<Item = crate::site::SiteId>) {
        if let Some(c) = &mut self.checker {
            c.poll_sites = sites.into_iter().collect();
        }
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
        let mut stats = c.stats;
        stats.wide_spans = c.wide_span_count();
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
            buffer_of_site: std::mem::take(&mut self.buffer_of_site),
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

/// The core packs a 32-bit epoch into its stamps (2^32 instructions per warp
/// per launch). Beyond that the run is `incomplete`, never truncated.
fn core_epoch(c: &mut dyn Sink, warp: u32, epoch: u64) -> Option<u32> {
    match u32::try_from(epoch) {
        Ok(e) => Some(e),
        Err(_) => {
            c.note(Incomplete::EpochOverflow { warp });
            None
        }
    }
}

/// A warp actor's span lane. `ALL_LANES` marks a warp-collective access
/// (ldmatrix/stmatrix, 32x32b tcgen05.ld/st, tile ops), performed by the
/// warp as one rendezvous: it is attributed to lane 0 (the engine emits the
/// surrounding `WarpSync`s that order the other lanes).
fn warp_lane(lane: u8) -> u8 {
    if lane == ALL_LANES {
        0
    } else {
        lane & 31
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

/// Where converted events go: the checker itself (serial), or a checker
/// partition that may hand them back for the main checker (decision 17).
pub(crate) trait Sink {
    fn ev(&mut self, e: ri::Event);
    fn note(&mut self, i: Incomplete);
}

impl Sink for Checker {
    fn ev(&mut self, e: ri::Event) {
        self.event(e);
    }
    fn note(&mut self, i: Incomplete) {
        self.note_incomplete(i);
    }
}

/// Per-lane sub-ops, by the parent op's slot pool (so a partition's
/// entries move with its pool).
#[derive(Default)]
pub(crate) struct Subops(HashMap<u32, HashMap<AsyncId, Vec<(u8, AsyncId)>>>);

impl Subops {
    fn get(&self, op: AsyncId) -> Option<&Vec<(u8, AsyncId)>> {
        self.0.get(&pool_key(op)).and_then(|m| m.get(&op))
    }
    fn insert(&mut self, op: AsyncId, subs: Vec<(u8, AsyncId)>) {
        self.0.entry(pool_key(op)).or_default().insert(op, subs);
    }
}

/// Contract access → core per-lane accesses.
fn conv_access(a: &CAccess<'_>, subops: &Subops, sink: &mut dyn Sink) {
        if !matches!(a.space, Space::Global | Space::Shared | Space::Tmem) {
            return; // local / param / reg are thread-private
        }
        let (order, scope, sc) = normalize(a.sem, a.scope, a.atomic);
        // The readonly proxy (ld.global.nc) is modelled as generic.
        let proxy = if a.proxy == Proxy::ReadOnly { Proxy::Generic } else { a.proxy };
        let base = |who| ri::Access {
            seq: a.seq.0,
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
            operand: a.operand,
        };
        match a.actor {
            Actor::Warp { warp, epoch } => {
                let Some(epoch) = core_epoch(sink, warp.0, epoch) else { return };
                if sc {
                    let mut lanes = LaneMask::NONE;
                    for s in a.spans {
                        lanes = lanes.or(LaneMask::lane(warp_lane(s.lane) as usize));
                    }
                    sink.ev(ri::Event::Sync(ri::SyncEvent::Fence { warp: warp.0, lanes, kind: ri::FenceKind::Sc(scope.unwrap()), site: a.site, epoch }));
                }
                for s in a.spans {
                    let mut x = base(ri::Who::Lane { warp: warp.0, lane: warp_lane(s.lane), epoch });
                    x.range = span_range(s.span);
                    sink.ev(ri::Event::Access(x));
                }
            }
            Actor::Async { op, side } => {
                let subs = subops.get(op);
                // One async op touches its footprint as many small spans
                // (TMA swizzle fragments, MMA operand rows). Coalesce
                // touching/overlapping byte ranges of one lane: the shadow
                // is byte-exact, so the union is the same set of accesses.
                // Only weak, non-atomic accesses: a strong or atomic span is
                // one element whose exact extent decides moral strength.
                let mut spans: Vec<(u8, Range<u64>)> = a.spans.iter().map(|s| (s.lane, span_range(s.span))).collect();
                if spans.len() > 1 && scope.is_none() && !a.atomic && super::tuning::on(&super::tuning::ASYNC_SPAN_MERGE) {
                    spans.sort_unstable_by_key(|(l, r)| (*l, r.start));
                    let mut out: Vec<(u8, Range<u64>)> = Vec::with_capacity(spans.len());
                    for (l, r) in spans {
                        match out.last_mut() {
                            Some((pl, pr)) if *pl == l && r.start <= pr.end => pr.end = pr.end.max(r.end),
                            _ => out.push((l, r)),
                        }
                    }
                    spans = out;
                }
                for (lane, range) in spans {
                    let id = match subs {
                        None => op,
                        Some(subs) => match subs.iter().find(|(l, _)| *l == lane) {
                            Some((_, id)) => *id,
                            None => {
                                // A multi-lane per-thread op whose span does
                                // not name its lane: attributing it to the
                                // whole warp would merge lanes (false
                                // negatives) — fail closed.
                                sink.note(Incomplete::AsyncLaneUnknown { op });
                                continue;
                            }
                        },
                    };
                    let mut x = base(ri::Who::Async { op: id, side });
                    x.range = range;
                    sink.ev(ri::Event::Access(x));
                }
            }
            Actor::Host => {}
        }
    }

/// Contract sync event → core events. `false`: the event had no warp actor
/// where one is required (counted as outside the launch).
fn conv_sync(e: &CSync, subops: &mut Subops, sink: &mut dyn Sink) -> bool {
        let wa = match e.actor {
            Actor::Warp { warp, epoch } => match core_epoch(sink, warp.0, epoch) {
                Some(epoch) => Some((warp.0, epoch)),
                None => return true,
            },
            _ => None,
        };
        let lanes = e.lanes;
        let ev = match &e.kind {
            SyncKind::AllocBegin { alloc, space, size, cta } => {
                ri::SyncEvent::AllocBegin { alloc: *alloc, space: *space, size: *size, cta: cta.0 }
            }
            SyncKind::AllocEnd { alloc } => ri::SyncEvent::AllocEnd { alloc: *alloc },
            SyncKind::DeclareWord { alloc, span } => ri::SyncEvent::DeclareWord { alloc: *alloc, range: span_range(*span) },
            SyncKind::Protocol { .. } => return true, // synccheck's input
            SyncKind::AsyncComplete { op, milestone, target } => {
                let (t, only) = match *target {
                    PublishTarget::Phase { obj, phase } => (ri::CompletionTarget::Phase { obj, phase }, LaneMask::ALL),
                    PublishTarget::Warp { warp, lanes } => (ri::CompletionTarget::Warp { warp: warp.0, lanes }, lanes),
                };
                match subops.get(*op) {
                    // Per-thread completion: only the waiting lanes' sub-ops.
                    Some(subs) => {
                        for (l, id) in subs.clone() {
                            if only.contains(l as usize) {
                                sink.ev(ri::Event::Sync(ri::SyncEvent::AsyncComplete { op: id, milestone: *milestone, target: t }));
                            }
                        }
                        return true;
                    }
                    None => ri::SyncEvent::AsyncComplete { op: *op, milestone: *milestone, target: t },
                }
            }
            kind => {
                let Some((warp, epoch)) = wa else {
                    return false;
                };
                match kind {
                    SyncKind::WarpSync { mask } => ri::SyncEvent::WarpSync { warp, mask: *mask, epoch },
                    // `None` qualifier = lost in lowering → incomplete;
                    // `scope: None` = named barrier (participants).
                    SyncKind::Arrive { obj, phase, release, scope } => {
                        ri::SyncEvent::Arrive { warp, lanes, obj: *obj, phase: *phase, release: *release, scope: *scope, site: e.site, epoch }
                    }
                    SyncKind::Wait { obj, phase, acquire, scope } => {
                        ri::SyncEvent::Wait { warp, lanes, obj: *obj, phase: *phase, acquire: *acquire, scope: *scope, site: e.site, epoch }
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
                            FenceEvent::MbarrierInit | FenceEvent::ProxyAlias => return true,
                        };
                        ri::SyncEvent::Fence { warp, lanes, kind, site: e.site, epoch }
                    }
                    SyncKind::AsyncIssue { op, class, proxy, preds, footprint, restricted, .. } => {
                        let preds: Vec<AsyncId> = preds
                            .iter()
                            .flat_map(|p| match subops.get(*p) {
                                Some(s) => s.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
                                None => vec![*p],
                            })
                            .collect();
                        let footprint: Vec<_> = footprint.iter().map(|(a, s)| (*a, span_range(*s))).collect();
                        if *class == crate::observe::AsyncClass::Copy && lanes.count() > 1 {
                            let mut subs = Vec::new();
                            for l in lanes.lanes() {
                                // Derived from the parent id: deterministic
                                // in any partition, and in the parent's
                                // slot pool (`checker::pool_key`).
                                let id = AsyncId(SUBOP_BASE | ((l as u64) << 56) | op.0);
                                subs.push((l as u8, id));
                                sink.ev(ri::Event::Sync(ri::SyncEvent::AsyncIssue {
                                    op: id,
                                    warp,
                                    lanes: LaneMask::lane(l),
                                    kind: *class,
                                    proxy: *proxy,
                                    preds: preds.clone(),
                                    footprint: footprint.clone(),
                                    restricted: *restricted,
                                    site: e.site,
                                    epoch,
                                }));
                            }
                            subops.insert(*op, subs);
                            return true;
                        }
                        ri::SyncEvent::AsyncIssue {
                            op: *op,
                            warp,
                            lanes,
                            kind: *class,
                            proxy: *proxy,
                            preds,
                            footprint,
                            restricted: *restricted,
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
                        site: e.site,
                        pred_reads: pred_reads.iter().map(|(a, s)| (*a, span_range(*s))).collect(),
                        epoch,
                    },
                    _ => unreachable!("handled above"),
                }
            }
        };
        sink.ev(ri::Event::Sync(ev));
        true
    }

impl Observer for RaceObserver {
    fn wants_word_history(&self) -> bool {
        true
    }

    fn begin_launch(&mut self, info: &LaunchInfo<'_>) {
        let s = info.shape;
        let topo = ri::Topology { warps_per_cta: s.warps_per_cta(), ctas_per_cluster: s.ctas_per_cluster().max(1), num_ctas: s.num_ctas() };
        self.start_launch(topo, info.kernel_index);
        let names: Vec<(crate::site::SiteId, String)> = info
            .program
            .sites
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.buffer().map(str::to_string).filter(|b| !b.is_empty()).map(|b| (crate::site::SiteId(i as u32), b)))
            .collect();
        self.set_site_buffers(names);
        if let Some(c) = &mut self.checker {
            // W5-15: one logical name per pointer operand.
            c.operand_buffer = info
                .program
                .sites
                .iter()
                .enumerate()
                .flat_map(|(i, s)| {
                    s.buffers.iter().enumerate().filter_map(move |(o, b)| {
                        b.as_deref().filter(|b| !b.is_empty()).map(|b| ((crate::site::SiteId(i as u32), o as u8), std::sync::Arc::<str>::from(b)))
                    })
                })
                .collect();
            // The site's buffer names ONE operand (the first pointer of a
            // multi-operand op such as tensormap.cp_fenceproxy): record its
            // space so an access in another space is not given that name.
            let space_of: HashMap<&str, Space> = info.program.buffers.iter().map(|b| (b.name.as_str(), b.space)).collect();
            c.site_buffer_space = info
                .program
                .sites
                .iter()
                .enumerate()
                .filter_map(|(i, s)| s.buffer().and_then(|b| space_of.get(b)).map(|sp| (crate::site::SiteId(i as u32), *sp)))
                .collect();
            c.poll_sites = info
                .program
                .sites
                .iter()
                .enumerate()
                .filter(|(_, s)| s.op_name == "tirx.cuda.wait_until")
                .map(|(i, _)| crate::site::SiteId(i as u32))
                .collect();
        }
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
        conv_access(a, &self.subops, c);
    }

    fn sync(&mut self, e: &CSync) {
        if let SyncKind::AllocBegin { alloc, space, .. } = &e.kind {
            self.buffers.entry(*alloc).or_insert_with(|| (format!("{alloc}"), *space));
        }
        let Some(c) = self.checker.as_mut() else {
            self.outside_launch += 1;
            return;
        };
        if e.kernel != self.kernel && !matches!(e.actor, Actor::Host) {
            // Per-launch reports: an event of another kernel cannot be
            // judged with this launch's topology and clocks.
            c.note_incomplete(Incomplete::KernelMismatch { expected: self.kernel, got: e.kernel });
            return;
        }
        if !conv_sync(e, &mut self.subops, c) {
            self.outside_launch += 1;
        }
    }

    fn warp_done(&mut self, warp: CWarpId, _end: WarpEnd) {
        if let Some(c) = &mut self.checker {
            c.warp_done(warp.0);
        }
    }

    fn round_boundary(&mut self, _cta: CtaId, _round: u64) {
        // Single-threaded scheduler: the global shadow is always merged
        // (module doc). The drain is a safe point for the collectors.
        if let Some(c) = &mut self.checker {
            c.safe_point();
        }
    }

    fn fork(&mut self, part: &PartitionInfo<'_>) -> Option<Box<dyn ForkedObserver>> {
        if !self.fork_join {
            return None;
        }
        let c = self.checker.as_mut()?;
        let ctas: Vec<u32> = part.ctas.iter().map(|c| c.0).collect();
        let meta = std::sync::Arc::new(c.global_meta());
        let child = c.split(part.key, &ctas, meta, FORK_RESERVE);
        let mut subops = Subops::default();
        if let Some(m) = self.subops.0.remove(&(part.key + 1)) {
            subops.0.insert(part.key + 1, m);
        }
        Some(Box::new(RaceChild { checker: child, subops, stash: Vec::new(), kernel: self.kernel, buffers: Vec::new(), outside_launch: 0 }))
    }

    fn join(&mut self, _part: &PartitionInfo<'_>, child: Box<dyn ForkedObserver>) {
        let child = child.into_any().downcast::<RaceChild>().expect("a RaceObserver child");
        let RaceChild { checker, subops, stash, buffers, outside_launch, .. } = *child;
        for (k, m) in subops.0 {
            self.subops.0.insert(k, m);
        }
        for (a, b) in buffers {
            self.buffers.entry(a).or_insert(b);
        }
        self.outside_launch += outside_launch;
        if let Some(c) = self.checker.as_mut() {
            c.absorb(checker, stash);
        }
    }

    fn phase_end(&mut self, _round: u64) {
        if let Some(c) = &mut self.checker {
            c.phase_end();
        }
    }
}

/// Async slots a child gets per phase before it must hand over (a fresh
/// slot beyond them is the main checker's).
const FORK_RESERVE: usize = 4;

/// A checker partition as an observer (decision 17): converts its
/// partition's events like `RaceObserver` and runs them in its child
/// checker until the first one that needs the main checker; that one and
/// everything after are stashed for `join`.
pub struct RaceChild {
    checker: Checker,
    subops: Subops,
    stash: Vec<Stashed>,
    kernel: u32,
    buffers: Vec<(AllocId, (String, Space))>,
    outside_launch: u64,
}

struct ChildSink<'a> {
    c: &'a mut Checker,
    stash: &'a mut Vec<Stashed>,
}

impl Sink for ChildSink<'_> {
    fn ev(&mut self, e: ri::Event) {
        if !self.stash.is_empty() {
            self.stash.push(Stashed::Event(e));
        } else if let Err(e) = self.c.part_event(e) {
            self.stash.push(Stashed::Event(e));
        }
    }
    fn note(&mut self, i: Incomplete) {
        if self.stash.is_empty() {
            self.c.note_incomplete(i);
        } else {
            self.stash.push(Stashed::Note(i));
        }
    }
}

impl Observer for RaceChild {
    fn wants_word_history(&self) -> bool {
        true
    }

    fn access(&mut self, a: &CAccess<'_>) {
        if !matches!(a.space, Space::Global | Space::Shared | Space::Tmem) {
            return;
        }
        let mut sink = ChildSink { c: &mut self.checker, stash: &mut self.stash };
        conv_access(a, &self.subops, &mut sink);
    }

    fn sync(&mut self, e: &CSync) {
        if let SyncKind::AllocBegin { alloc, space, .. } = &e.kind {
            self.buffers.push((*alloc, (format!("{alloc}"), *space)));
        }
        let mut sink = ChildSink { c: &mut self.checker, stash: &mut self.stash };
        if e.kernel != self.kernel && !matches!(e.actor, Actor::Host) {
            sink.note(Incomplete::KernelMismatch { expected: self.kernel, got: e.kernel });
            return;
        }
        if !conv_sync(e, &mut self.subops, &mut sink) {
            self.outside_launch += 1;
        }
    }

    fn warp_done(&mut self, warp: CWarpId, _end: WarpEnd) {
        if self.stash.is_empty() && self.checker.holds_warp(warp.0) {
            self.checker.warp_done(warp.0);
        } else {
            self.stash.push(Stashed::WarpDone(warp.0));
        }
    }
}

impl ForkedObserver for RaceChild {
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

