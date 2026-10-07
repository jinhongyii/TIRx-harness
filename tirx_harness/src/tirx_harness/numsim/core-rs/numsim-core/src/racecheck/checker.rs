//! The unified online checker: one `Clock` model, one `IntervalShadow<Cell>`
//! per allocation, one conflict rule — for shared memory, TMEM and global
//! memory alike. Memory-space differences are data (actor sets, the proxy
//! view consulted, scope topology), not separate engines.
//!
//! Races never abort: a finding is recorded and checking continues.
//!
//! Memory is bounded by two collectors run every `gc_every` events
//! ([`Checker::gc`]):
//! * **view-aware dominated-frontier GC** drops a witness once the meet of
//!   every live actor's knowledge observes it *in every view a future access
//!   could judge it by* (hb, proxy bridges, tcgen, tensormap), so cross-proxy
//!   pairs survive GC (legacy G's proxy-blind floor GC did not);
//! * **async-slot reclaim** recycles the actor index of a completed async op
//!   once no witness of it remains; the slot's next generation starts above
//!   every epoch of the previous one, so old clocks never observe it.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use super::cell::{effective_heads, overlap, Cell, Entry, WideSpans, Witness};
use super::clock::{ActorId, Clock, Epoch, JoinMemo, LaneVec, Stamp};
use super::input::*;
use super::knowledge::{fence_domains, select_view, Heads, Knowledge, Rel, View, NDOM};
use super::shadow::IntervalShadow;

// ---------------------------------------------------------------- report --

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RaceClass {
    WriteRead,
    ReadWrite,
    WriteWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderingFailure {
    MissingInterActorSync,
    MissingSameWarpLaneOrder,
    MissingReleaseAcquire,
    AsyncLifetimeNotDrained,
    MissingProxyBridge { prior: Proxy, current: Proxy, domain: Option<Domain> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Review,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WitnessInfo {
    /// Issuing warp/lane for async ops (the op id is extra evidence).
    pub warp: WarpId,
    pub lane: u8,
    pub epoch: Epoch,
    pub async_op: Option<AsyncId>,
    pub kind: AccessKind,
    pub proxy: Proxy,
    pub domain: Option<Domain>,
    pub scope: Option<Scope>,
    pub atomic: bool,
    pub site: SiteId,
    pub span: Range<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FindingKind {
    DataRace { class: RaceClass, failure: OrderingFailure },
    /// Same evidence as a data race whose prior is an unwaited tcgen05.ld.
    TmemLifetimeReview { class: RaceClass, failure: OrderingFailure },
    /// A release/acquire pair whose scopes do not mutually cover.
    ScopeMismatch { release_scope: Scope, acquire_scope: Scope, release_warp: WarpId, acquire_warp: WarpId },
    /// Advisory (`review`): not a proven race, an unresolved risk.
    Advisory { kind: AdvisoryKind },
    /// An allocation ended while an async op with a footprint in it had not
    /// reached any completion milestone.
    AsyncLifetime { op: AsyncId },
    OutOfBounds { size: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AdvisoryKind {
    /// Two async-proxy accesses issued from different CTAs, ordered only by
    /// base causality (PTX §8.9.5 "same thread block"; ISA-silent).
    CrossCtaAsyncOrder,
    /// A strong load observed an unordered, morally strong write on a word
    /// that is not a declared `wait_until` word: not a race (§8.7.1), but
    /// the edge it yields is schedule dependent.
    UndeclaredProtocolWord,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finding {
    pub kind: FindingKind,
    pub severity: Severity,
    pub alloc: AllocId,
    /// Hull of the overlapping bytes over all deduplicated occurrences.
    pub bytes: Range<u64>,
    pub prior: Option<WitnessInfo>,
    pub current: Option<WitnessInfo>,
    /// Occurrences folded into this finding.
    pub occurrences: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Incomplete {
    EpochRegression { warp: WarpId },
    EpochOverflow { warp: WarpId },
    UnknownAsyncOp { op: AsyncId },
    UnknownAlloc { alloc: AllocId },
    /// `wait_until` exited but no history entry explains it.
    WaitExitUnproven { warp: WarpId },
    /// A barrier arrive/wait whose `.sem` qualifier was lost in lowering.
    SyncQualifierUnknown { warp: WarpId },
    /// An mbarrier arrival crossed CTAs but the event carries no scope
    /// (contract gap): the edge can be neither proven nor refuted.
    MbarrierScopeUnknown { warp: WarpId },
    /// A `wait_until` predicate read memory written concurrently with the
    /// wait, so the verdict bitset does not determine the exit.
    WaitPredicateReadsUnstable { warp: WarpId },
    /// Async op never reached a milestone by launch end.
    AsyncNeverCompleted { op: AsyncId },
    /// Events delivered outside a launch (no topology to judge them by).
    EventOutsideLaunch { events: u64 },
    /// `max_findings` reached; later findings were not recorded.
    FindingsTruncated { dropped: u64 },
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub findings: Vec<Finding>,
    pub incomplete: Vec<Incomplete>,
}

impl Report {
    pub fn errors(&self) -> impl Iterator<Item = &Finding> {
        self.findings.iter().filter(|f| f.severity == Severity::Error)
    }
    pub fn races(&self) -> impl Iterator<Item = &Finding> {
        self.findings.iter().filter(|f| matches!(f.kind, FindingKind::DataRace { .. }))
    }
    pub fn is_clean(&self) -> bool {
        self.findings.is_empty() && self.incomplete.is_empty()
    }
}

/// Counters exposed for tests, payload `stats` and benchmarks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub accesses: u64,
    pub gc_runs: u64,
    pub witnesses_retired: u64,
    pub async_slots_reclaimed: u64,
    pub async_slots: u64,
}

// ----------------------------------------------------------------- state --

/// Bridge-row slots: g2a[0..NDOM], a2g[NDOM..2NDOM], tensormap release.
const NSLOT: usize = 2 * NDOM + 1;
const TSLOT: usize = 2 * NDOM;

struct Warp {
    actor: ActorId,
    epoch: Epoch,
    done: bool,
    /// Lane-order matrix: `row[c][p]` = latest epoch of lane `p` that lane
    /// `c` has observed (warp sync / collectives). Diagonal unused.
    row: Box<[LaneVec; 32]>,
    /// Knowledge common to every lane.
    base: Knowledge,
    /// Lane-specific acquisitions. Sparse: `None` for lanes that never
    /// diverged.
    extra: Vec<Option<Box<Knowledge>>>,
    /// Own-warp part of the bridges: `bridge_rows[s][c][p]`.
    bridge_rows: Option<Box<[[LaneVec; 32]; NSLOT]>>,
    /// Latest release-fence head per lane.
    fence_rel: Vec<Option<Arc<Rel>>>,
    /// Release payloads observed by relaxed strong loads / relaxed waits,
    /// awaiting an acquire fence.
    pending_acq: Vec<(u8, Arc<Rel>)>,
    /// tcgen05 ordering state, per lane (per PTX thread).
    tcgen: Vec<Clock>,
    tcgen_in: Vec<Clock>,
    tcgen_issued: Vec<Clock>,
    tcgen_waited: Vec<Clock>,
    tcgen_pub: Vec<Clock>,
    /// `(epoch, site)` of every instruction that accessed memory, ascending;
    /// pruned below the oldest epoch any witness still references.
    sites: Vec<(Epoch, SiteId)>,
}

impl Warp {
    fn new(actor: ActorId) -> Self {
        Warp {
            actor,
            epoch: 0,
            done: false,
            row: Box::new([[0; 32]; 32]),
            base: Knowledge::default(),
            extra: vec![None; 32],
            bridge_rows: None,
            fence_rel: vec![None; 32],
            pending_acq: Vec::new(),
            tcgen: vec![Clock::default(); 32],
            tcgen_in: vec![Clock::default(); 32],
            tcgen_issued: vec![Clock::default(); 32],
            tcgen_waited: vec![Clock::default(); 32],
            tcgen_pub: vec![Clock::default(); 32],
            sites: Vec::new(),
        }
    }

    #[inline(always)]
    fn knows(&self, lane: u8, v: View, s: Stamp, prior_lane: u8) -> bool {
        self.base.observes(v, s, prior_lane)
            || self.extra[lane as usize].as_ref().is_some_and(|x| x.observes(v, s, prior_lane))
    }

    /// What lanes `lanes` release at epoch `e` (barrier arrive, st.release,
    /// fence-release head, async issue). The own-warp component is exact per
    /// lane: lane `p` is published up to what some releasing lane observed.
    fn publication(&self, lanes: LaneMask, e: Epoch, memo: &JoinMemo) -> Knowledge {
        let mut k = self.base.propagating();
        let mut v = [0; 32];
        for c in lanes.lanes8() {
            if let Some(x) = &self.extra[c as usize] {
                k.join_propagating(x, memo);
            }
            if !self.tcgen_pub[c as usize].is_empty() {
                k.tcgen_rel.join(&self.tcgen_pub[c as usize], memo);
            }
            for (p, slot) in v.iter_mut().enumerate() {
                let seen = if p == c as usize { e } else { self.row[c as usize][p] };
                *slot = (*slot).max(seen);
            }
        }
        k.hb.raise_lanes(self.actor, &v);
        if let Some(br) = &self.bridge_rows {
            for (s, rows) in br.iter().enumerate() {
                let mut bv = [0; 32];
                for c in lanes.lanes8() {
                    for (p, slot) in bv.iter_mut().enumerate() {
                        *slot = (*slot).max(rows[c as usize][p]);
                    }
                }
                if bv.iter().any(|x| *x != 0) {
                    slot_mut(&mut k, s).raise_lanes(self.actor, &bv);
                }
            }
        }
        k
    }

    /// The tcgen05 fence frontier lanes `lanes` carry through any sync.
    fn tcgen_publication(&self, lanes: LaneMask, memo: &JoinMemo) -> Clock {
        let mut c = Clock::default();
        for l in lanes.lanes8() {
            if !self.tcgen_pub[l as usize].is_empty() {
                c.join(&self.tcgen_pub[l as usize], memo);
            }
        }
        c
    }

    fn acquire(&mut self, lanes: LaneMask, k: &Knowledge, memo: &JoinMemo) {
        let join = |x: &mut Knowledge| {
            x.hb.join(&k.hb, memo);
            for d in 0..NDOM {
                x.g2a[d].join(&k.g2a[d], memo);
                x.a2g[d].join(&k.a2g[d], memo);
            }
            x.tmap_rel.join(&k.tmap_rel, memo);
        };
        if lanes.is_all() {
            join(&mut self.base);
        } else {
            for c in lanes.lanes8() {
                join(self.extra[c as usize].get_or_insert_with(Default::default));
            }
        }
        if !k.tcgen_rel.is_empty() {
            for c in lanes.lanes8() {
                self.tcgen_in[c as usize].join(&k.tcgen_rel, memo);
            }
        }
    }

    fn lane_view(&self, lane: u8, v: View, memo: &JoinMemo) -> Clock {
        let mut c = self.base.view(v).clone();
        if let Some(x) = &self.extra[lane as usize] {
            c.join(x.view(v), memo);
        }
        c
    }
}

fn slot_mut(k: &mut Knowledge, s: usize) -> &mut Clock {
    if s < NDOM {
        &mut k.g2a[s]
    } else if s < 2 * NDOM {
        &mut k.a2g[s - NDOM]
    } else {
        &mut k.tmap_rel
    }
}

struct AsyncActor {
    op: AsyncId,
    actor: ActorId,
    /// Epoch base of the slot's current generation (sides are base+1/+2).
    gen_base: Epoch,
    in_use: bool,
    warp: WarpId,
    lane: u8,
    issue_epoch: Epoch,
    site: SiteId,
    kind: AsyncKind,
    k: Knowledge,
    preds: Vec<usize>,
    footprint: Vec<(AllocId, Range<u64>)>,
    /// Highest milestone reached (0 none, 1 read, 2 write/full).
    done: u8,
}

struct Alloc {
    size: u64,
    space: Space,
    shadow: IntervalShadow<Cell>,
    /// Once any non-generic proxy touched the allocation, generic witnesses
    /// are retired only when every view observes them.
    sensitive: bool,
    /// Generic witnesses retired while the allocation was not proxy
    /// sensitive, summarised per `(actor, lane, write, window)`: the latest
    /// epoch and the byte hull. A later async/tensormap access is checked
    /// against them, so GC never hides a cross-proxy race (legacy RS
    /// `RetiredGenericHistory`; bounded by actors × 32 × 2 × 3).
    retired: HashMap<(ActorId, u8, bool, u8), RetiredGeneric>,
}

#[derive(Clone, Debug)]
struct RetiredGeneric {
    stamp: Stamp,
    lane: u8,
    write: bool,
    domain: Option<Domain>,
    lo: u64,
    hi: u64,
    info: WitnessInfo,
}

fn domain_code(d: Option<Domain>) -> u8 {
    match d {
        None => 0,
        Some(Domain::Global) => 1,
        Some(Domain::SharedCta) => 2,
        Some(Domain::SharedCluster) => 3,
    }
}

#[derive(Default)]
struct Phase {
    /// Release arrivals grouped by `(arriver CTA, scope)` (`None` = named
    /// barrier), each with a representative warp. Mutual scope inclusion
    /// depends on the arriver only through its CTA / cluster, so a group is
    /// judged once and joined once per waiter (O(groups), not O(arrivers)).
    arrivals: Vec<(WarpId, Option<Scope>, Arc<Knowledge>)>,
    /// tcgen05 fence frontier carried by every arrive, relaxed included.
    tcgen_rel: Clock,
    /// Async completions (complete-tx: release at cluster scope for the
    /// op's own bytes; accepted by an acquire wait of any scope).
    completion: Knowledge,
}

struct HistEntry {
    rel: Option<Heads>,
    is_async: bool,
}

struct Word {
    alloc: AllocId,
    range: Range<u64>,
    history: Vec<HistEntry>,
}

/// Who performs the access being checked.
#[derive(Clone, Copy)]
enum Cur {
    Lane { w: usize, lane: u8, epoch: Epoch },
    Async { a: usize },
}

pub struct Checker {
    topo: Topology,
    pub memo: JoinMemo,
    warps: Vec<Warp>,
    asyncs: Vec<AsyncActor>,
    free_slots: Vec<usize>,
    async_index: HashMap<AsyncId, usize>,
    reclaimed: HashSet<AsyncId>,
    allocs: HashMap<AllocId, Alloc>,
    phases: HashMap<(SyncObjId, u64), Phase>,
    /// Latest `fence.sc` per thread `(warp, lane)` with its scope.
    sc: HashMap<(WarpId, u8), (Scope, Arc<Knowledge>)>,
    words: Vec<Word>,
    wide: WideSpans,
    report: Report,
    dedup: HashMap<(AllocId, RaceClass, SiteId, SiteId, bool), usize>,
    /// Run the collectors every this many events (0 = never).
    pub gc_every: u64,
    /// Stop recording new findings after this many (0 = unlimited).
    pub max_findings: usize,
    /// The mbarrier arrive/wait scope is assumed (`.cta`, the PTX default)
    /// because the event does not carry it. A cross-CTA arrival that fails
    /// the assumed mutual-inclusion test is then `incomplete`, not silently
    /// edge-less.
    pub mbarrier_scope_assumed: bool,
    dropped_findings: u64,
    since_gc: u64,
    pub stats: Stats,
}

pub(crate) fn required_scope(t: &Topology, a: WarpId, b: WarpId) -> Scope {
    if t.cta_of(a) == t.cta_of(b) {
        Scope::Cta
    } else if t.cluster_of(a) == t.cluster_of(b) {
        Scope::Cluster
    } else {
        Scope::Gpu
    }
}

/// Proxies a future access to an allocation of this space can use.
fn future_proxies(space: Space) -> &'static [Proxy] {
    match space {
        Space::Tmem => &[Proxy::Tcgen],
        _ => &[Proxy::Generic, Proxy::Async, Proxy::TensorMap],
    }
}

impl Checker {
    pub fn new(topo: Topology) -> Self {
        let n = topo.num_warps();
        Checker {
            topo,
            memo: JoinMemo::default(),
            warps: (0..n).map(Warp::new).collect(),
            asyncs: Vec::new(),
            free_slots: Vec::new(),
            async_index: HashMap::new(),
            reclaimed: HashSet::new(),
            allocs: HashMap::new(),
            phases: HashMap::new(),
            sc: HashMap::new(),
            words: Vec::new(),
            wide: WideSpans::default(),
            report: Report::default(),
            dedup: HashMap::new(),
            gc_every: 1 << 14,
            max_findings: 0,
            mbarrier_scope_assumed: false,
            dropped_findings: 0,
            since_gc: 0,
            stats: Stats::default(),
        }
    }

    pub fn topology(&self) -> Topology {
        self.topo
    }

    pub fn run(topo: Topology, events: impl IntoIterator<Item = Event>) -> Report {
        let mut c = Checker::new(topo);
        for e in events {
            c.event(e);
        }
        c.finish()
    }

    pub fn event(&mut self, e: Event) {
        match e {
            Event::Access(a) => self.access(&a),
            Event::Sync(s) => self.sync(s),
        }
    }

    /// Findings so far (the run is not finalised).
    pub fn report(&self) -> &Report {
        &self.report
    }

    /// Finalise: outstanding async work is incomplete.
    pub fn finalize(&mut self) {
        for a in &self.asyncs {
            if a.in_use && a.done == 0 && a.kind != AsyncKind::TcgenCommit {
                self.report.incomplete.push(Incomplete::AsyncNeverCompleted { op: a.op });
            }
        }
        if self.dropped_findings > 0 {
            self.report.incomplete.push(Incomplete::FindingsTruncated { dropped: self.dropped_findings });
            self.dropped_findings = 0;
        }
    }

    pub fn finish(mut self) -> Report {
        self.finalize();
        self.report
    }

    pub fn warp_done(&mut self, w: WarpId) {
        if let Some(w) = self.warps.get_mut(w as usize) {
            w.done = true;
        }
    }

    fn covers(&self, s: Scope, a: WarpId, b: WarpId) -> bool {
        s >= required_scope(&self.topo, a, b)
    }

    fn tick(&mut self, w: WarpId, epoch: Epoch) -> bool {
        let Some(warp) = self.warps.get_mut(w as usize) else {
            self.report.incomplete.push(Incomplete::EpochRegression { warp: w });
            return false;
        };
        if epoch < warp.epoch {
            self.report.incomplete.push(Incomplete::EpochRegression { warp: w });
            return false;
        }
        if epoch >= u32::MAX - 1 {
            self.report.incomplete.push(Incomplete::EpochOverflow { warp: w });
            return false;
        }
        warp.epoch = epoch;
        true
    }

    fn async_idx(&mut self, op: AsyncId) -> Option<usize> {
        let r = self.async_index.get(&op).copied();
        if r.is_none() {
            self.report.incomplete.push(Incomplete::UnknownAsyncOp { op });
        }
        r
    }

    fn push_finding(&mut self, f: Finding) -> Option<usize> {
        if self.max_findings != 0 && self.report.findings.len() >= self.max_findings {
            self.dropped_findings += 1;
            return None;
        }
        self.report.findings.push(f);
        Some(self.report.findings.len() - 1)
    }

    /// A natural pause (inbox drain): collect if a quarter period elapsed.
    pub fn safe_point(&mut self) {
        if self.gc_every != 0 && self.since_gc >= self.gc_every / 4 {
            self.gc();
        }
    }

    fn maybe_gc(&mut self) {
        self.since_gc += 1;
        if self.gc_every != 0 && self.since_gc >= self.gc_every {
            self.gc();
        }
    }

    // ------------------------------------------------- witness decoding --

    #[inline(always)]
    fn slot_of(&self, actor: ActorId) -> Option<&AsyncActor> {
        let nw = self.topo.num_warps();
        (actor >= nw).then(|| &self.asyncs[(actor - nw) as usize])
    }

    /// Performing warp (issuing warp for async actors), for scope tests.
    #[inline(always)]
    fn warp_of(&self, w: &Witness) -> WarpId {
        match self.slot_of(w.stamp.actor()) {
            Some(a) => a.warp,
            None => w.stamp.actor(),
        }
    }

    fn site_of(&self, w: &Witness) -> SiteId {
        match self.slot_of(w.stamp.actor()) {
            Some(a) => a.site,
            None => {
                let sites = &self.warps[w.stamp.actor() as usize].sites;
                let e = w.stamp.epoch();
                let i = sites.partition_point(|(x, _)| *x <= e);
                if i == 0 {
                    SiteId(u32::MAX)
                } else {
                    sites[i - 1].1
                }
            }
        }
    }

    // ------------------------------------------------------- ordering --

    /// Is `prior` ordered before an access by `cur` in proxy `cur_proxy`?
    #[inline]
    fn ordered(&self, cur: Cur, prior: &Witness, cur_proxy: Proxy) -> bool {
        let view = select_view(prior.proxy(), cur_proxy, prior.domain());
        match cur {
            Cur::Lane { w, lane, epoch } => {
                let warp = &self.warps[w];
                if prior.stamp.actor() == warp.actor && view == View::Hb {
                    // Program order within a lane; sibling lanes of one
                    // instruction are simultaneous; otherwise the lane-order
                    // matrix, then knowledge that came back around.
                    let pl = prior.lane();
                    if pl == lane {
                        return true;
                    }
                    if prior.stamp.epoch() == epoch {
                        return false;
                    }
                    if warp.row[lane as usize][pl as usize] >= prior.stamp.epoch() {
                        return true;
                    }
                }
                if view == View::Tcgen {
                    return warp.tcgen[lane as usize].observes(prior.stamp, prior.lane());
                }
                warp.knows(lane, view, prior.stamp, prior.lane())
            }
            Cur::Async { a } => {
                let act = &self.asyncs[a];
                prior.stamp.actor() == act.actor || act.k.observes(view, prior.stamp, prior.lane())
            }
        }
    }

    fn morally_strong(&self, p: &Witness, c: &Witness) -> bool {
        match (p.scope(), c.scope()) {
            (Some(ps), Some(cs)) => {
                let (pw, cw) = (self.warp_of(p), self.warp_of(c));
                p.proxy() == c.proxy() && p.same_span(c, &self.wide) && self.covers(ps, pw, cw) && self.covers(cs, cw, pw)
            }
            _ => false,
        }
    }

    fn classify(&self, cur: Cur, prior: &Witness, cw: &Witness) -> OrderingFailure {
        let (pp, cp) = (prior.proxy(), cw.proxy());
        if pp != cp && pp != Proxy::Tcgen && cp != Proxy::Tcgen {
            return OrderingFailure::MissingProxyBridge { prior: pp, current: cp, domain: prior.domain() };
        }
        if let Cur::Lane { w, .. } = cur {
            if prior.stamp.actor() == self.warps[w].actor {
                return OrderingFailure::MissingSameWarpLaneOrder;
            }
        }
        if let Some(a) = self.slot_of(prior.stamp.actor()) {
            let issue = Stamp::new(a.warp, a.issue_epoch);
            let issued_before = match cur {
                Cur::Lane { w, lane, .. } => {
                    (a.warp == w as u32 && (a.lane == lane || self.warps[w].row[lane as usize][a.lane as usize] >= a.issue_epoch))
                        || self.warps[w].knows(lane, View::Hb, issue, a.lane)
                }
                Cur::Async { a: x } => self.asyncs[x].k.hb.observes(issue, a.lane),
            };
            if issued_before {
                return OrderingFailure::AsyncLifetimeNotDrained;
            }
        }
        if prior.scope().is_some() || cw.scope().is_some() || prior.kind() == AccessKind::Rmw || cw.kind() == AccessKind::Rmw {
            OrderingFailure::MissingReleaseAcquire
        } else {
            OrderingFailure::MissingInterActorSync
        }
    }

    /// Async-proxy pair whose two ops were issued from different CTAs.
    fn cross_cta_async(&self, prior: &Witness, cur: Cur, cur_proxy: Proxy) -> bool {
        let Cur::Async { a } = cur else { return false };
        prior.proxy() == Proxy::Async
            && cur_proxy == Proxy::Async
            && prior.stamp.actor() != self.asyncs[a].actor
            && self.slot_of(prior.stamp.actor()).is_some()
            && self.topo.cta_of(self.warp_of(prior)) != self.topo.cta_of(self.asyncs[a].warp)
    }

    fn info(&self, w: &Witness) -> WitnessInfo {
        let (lane, op) = match self.slot_of(w.stamp.actor()) {
            Some(a) => (a.lane, Some(a.op)),
            None => (w.lane(), None),
        };
        let (s, e) = w.span(&self.wide);
        WitnessInfo {
            warp: self.warp_of(w),
            lane,
            epoch: w.stamp.epoch(),
            async_op: op,
            kind: w.kind(),
            proxy: w.proxy(),
            domain: w.domain(),
            scope: w.scope(),
            atomic: w.atomic(),
            site: self.site_of(w),
            span: s..e,
        }
    }

    fn report_advisory(&mut self, alloc: AllocId, bytes: Range<u64>, prior: &Witness, cw: &Witness, kind: AdvisoryKind) {
        if !(prior.writes() || cw.writes()) {
            return;
        }
        let site = self.site_of(cw);
        if self
            .report
            .findings
            .iter()
            .any(|f| f.kind == FindingKind::Advisory { kind } && f.alloc == alloc && f.current.as_ref().is_some_and(|c| c.site == site))
        {
            return;
        }
        let f = Finding {
            kind: FindingKind::Advisory { kind },
            severity: Severity::Review,
            alloc,
            bytes,
            prior: Some(self.info(prior)),
            current: Some(self.info(cw)),
            occurrences: 1,
        };
        self.push_finding(f);
    }

    fn report_race(&mut self, alloc: AllocId, bytes: Range<u64>, cur: Cur, prior: &Witness, cw: &Witness) {
        self.report_race_with(alloc, bytes, cur, prior, cw, None)
    }

    fn report_race_with(&mut self, alloc: AllocId, bytes: Range<u64>, cur: Cur, prior: &Witness, cw: &Witness, prior_info: Option<WitnessInfo>) {
        let class = match (prior.writes(), cw.writes()) {
            (true, true) => RaceClass::WriteWrite,
            (true, false) => RaceClass::WriteRead,
            _ => RaceClass::ReadWrite,
        };
        let failure = self.classify(cur, prior, cw);
        let prior_is_ld = self.slot_of(prior.stamp.actor()).is_some_and(|a| a.kind == AsyncKind::TcgenLd);
        let review = prior_is_ld && failure == OrderingFailure::AsyncLifetimeNotDrained;
        let prior_site = prior_info.as_ref().map_or_else(|| self.site_of(prior), |i| i.site);
        let key = (alloc, class, prior_site, self.site_of(cw), review);
        if let Some(&i) = self.dedup.get(&key) {
            let f = &mut self.report.findings[i];
            f.bytes = f.bytes.start.min(bytes.start)..f.bytes.end.max(bytes.end);
            f.occurrences += 1;
            return;
        }
        let kind = if review {
            FindingKind::TmemLifetimeReview { class, failure }
        } else {
            FindingKind::DataRace { class, failure }
        };
        let f = Finding {
            kind,
            severity: if review { Severity::Review } else { Severity::Error },
            alloc,
            bytes,
            prior: Some(prior_info.unwrap_or_else(|| self.info(prior))),
            current: Some(self.info(cw)),
            occurrences: 1,
        };
        if let Some(i) = self.push_finding(f) {
            self.dedup.insert(key, i);
        }
    }

    // --------------------------------------------------------- access --

    pub fn access(&mut self, a: &Access) {
        self.stats.accesses += 1;
        self.maybe_gc();
        let Some(size) = self.allocs.get(&a.alloc).map(|x| x.size) else {
            self.report.incomplete.push(Incomplete::UnknownAlloc { alloc: a.alloc });
            return;
        };
        let (cur, stamp, lane) = match a.who {
            Who::Lane { warp, lane, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                let sites = &mut self.warps[warp as usize].sites;
                if sites.last().is_none_or(|(e, _)| *e != epoch) {
                    sites.push((epoch, a.site));
                }
                (Cur::Lane { w: warp as usize, lane, epoch }, Stamp::new(warp, epoch), lane)
            }
            Who::Async { op, side } => {
                let Some(i) = self.async_idx(op) else { return };
                let act = &self.asyncs[i];
                (Cur::Async { a: i }, Stamp::new(act.actor, act.gen_base + side_index(side)), 0)
            }
        };
        if a.range.end > size || a.range.start >= a.range.end {
            let current = matches!(cur, Cur::Lane { .. }).then(|| {
                let w = Witness::pack(stamp, lane, a.proxy, a.domain, a.kind, a.scope, a.atomic, (a.range.start, a.range.end), &mut self.wide);
                self.info(&w)
            });
            let f = Finding {
                kind: FindingKind::OutOfBounds { size },
                severity: Severity::Error,
                alloc: a.alloc,
                bytes: a.range.clone(),
                prior: None,
                current,
                occurrences: 1,
            };
            self.push_finding(f);
            return;
        }
        let w = Witness::pack(stamp, lane, a.proxy, a.domain, a.kind, a.scope, a.atomic, (a.range.start, a.range.end), &mut self.wide);
        let writes = w.writes();
        let strong = a.scope.is_some();

        // Own release head for strong writes.
        let own_rel: Option<Arc<Rel>> = match (cur, writes) {
            (Cur::Lane { w: wi, lane, epoch }, true) if strong => {
                let warp = &self.warps[wi];
                if matches!(a.order, MemOrder::Release | MemOrder::AcqRel) {
                    let k = warp.publication(one_lane(lane), epoch, &self.memo);
                    Some(Arc::new(Rel { k, scope: a.scope, warp: wi as u32 }))
                } else if let Some(f) = &warp.fence_rel[lane as usize] {
                    Some(f.clone())
                } else {
                    let k = Knowledge { tcgen_rel: warp.tcgen_publication(one_lane(lane), &self.memo), ..Default::default() };
                    Some(Arc::new(Rel { k, scope: None, warp: wi as u32 }))
                }
            }
            _ => None,
        };

        let mut races: Vec<(Range<u64>, Witness)> = Vec::new();
        if a.proxy != Proxy::Generic {
            self.check_retired_generic(a, cur, &w);
        }
        let mut advisories: Vec<(Range<u64>, Witness, AdvisoryKind)> = Vec::new();
        let mut acquired: Vec<Heads> = Vec::new();
        let mut word_rel: Option<Option<Heads>> = None;
        let word_start = self
            .words
            .iter()
            .find(|x| x.alloc == a.alloc && x.range.start < a.range.end && a.range.start < x.range.end)
            .map(|x| x.range.start);
        let in_word = word_start.is_some();

        let mut shadow = std::mem::take(&mut self.allocs.get_mut(&a.alloc).unwrap().shadow);
        {
            let this = &*self;
            let wide = &self.wide;
            shadow.update(a.range.clone(), |seg, cell| {
                // 1. check
                let mut check = |p: &Witness| {
                    let ordered = this.ordered(cur, p, a.proxy);
                    let ms = this.morally_strong(p, &w);
                    if !ordered && !ms {
                        races.push((overlap(p.span(wide), &seg), *p));
                    } else if ordered && this.cross_cta_async(p, cur, a.proxy) {
                        advisories.push((overlap(p.span(wide), &seg), *p, AdvisoryKind::CrossCtaAsyncOrder));
                    } else if !ordered && ms && !writes && !in_word && p.writes() {
                        advisories.push((overlap(p.span(wide), &seg), *p, AdvisoryKind::UndeclaredProtocolWord));
                    }
                };
                for p in cell.writes.as_slice() {
                    check(&p.w);
                }
                if writes {
                    for p in cell.reads.as_slice() {
                        check(&p.w);
                    }
                }
                // The write this access reads from: the latest one that is
                // not a sibling lane of the same instruction.
                let prev = cell
                    .writes
                    .as_slice()
                    .iter()
                    .rev()
                    .find(|e| !(e.w.stamp == w.stamp && e.w.lane() != w.lane()))
                    .filter(|e| this.morally_strong(&e.w, &w));
                let inherited = prev.and_then(|e| effective_heads(&cell.writes, e));
                // 2. read-from (strong reads and atomics)
                if strong && (!writes || a.kind == AccessKind::Rmw) {
                    if let Some(h) = &inherited {
                        if !acquired.iter().any(|r| Arc::ptr_eq(r, h)) {
                            acquired.push(h.clone());
                        }
                    }
                }
                // 3. record
                if writes {
                    // Observation order (PTX §8.9.2): an atomic continues the
                    // chain of the write it reads from; heads stay separate.
                    let base = if a.kind == AccessKind::Rmw { inherited.clone() } else { None };
                    let rel = match (&base, &own_rel) {
                        (None, None) => None,
                        (Some(b), None) => Some(b.clone()),
                        (b, Some(own)) => {
                            let mut v: Vec<Arc<Rel>> = b.as_ref().map(|b| b.to_vec()).unwrap_or_default();
                            v.push(own.clone());
                            Some(Arc::new(v))
                        }
                    };
                    if word_start.is_some_and(|s| seg.start <= s && s < seg.end) {
                        word_rel = Some(rel.clone());
                    }
                    cell.writes.record(Entry { w, rel, base }, wide, |p| this.ordered(cur, p, a.proxy));
                    if !strong {
                        // A plain write supersedes the readers it is ordered
                        // after; a strong write keeps them (a later access
                        // morally strong with it may still race them).
                        cell.reads.retain(|r| !this.ordered(cur, &r.w, a.proxy));
                    }
                } else {
                    cell.reads.record(Entry { w, rel: None, base: None }, wide, |p| this.ordered(cur, p, a.proxy));
                }
            });
        }
        self.allocs.get_mut(&a.alloc).unwrap().shadow = shadow;
        for (bytes, prior) in races {
            self.report_race(a.alloc, bytes, cur, &prior, &w);
        }
        for (bytes, prior, kind) in advisories {
            self.report_advisory(a.alloc, bytes, &prior, &w, kind);
        }
        if let Some(rel) = word_rel {
            let ws = word_start.unwrap();
            if let Some(word) = self.words.iter_mut().find(|x| x.alloc == a.alloc && x.range.start == ws) {
                word.history.push(HistEntry { rel, is_async: matches!(cur, Cur::Async { .. }) });
            }
        }
        if let Cur::Lane { w: wi, lane, .. } = cur {
            // `red` never forms an acquire pattern (PTX §8.8).
            let can_acquire = a.kind != AccessKind::Rmw || a.returns_value;
            for heads in acquired {
                for rel in heads.iter() {
                    self.warps[wi].tcgen_in[lane as usize].join(&rel.k.tcgen_rel, &self.memo);
                    if !can_acquire {
                        continue;
                    }
                    match a.order {
                        MemOrder::Acquire | MemOrder::AcqRel => self.acquire_rel(wi as u32, one_lane(lane), a.scope.unwrap(), rel),
                        MemOrder::Relaxed | MemOrder::Release => self.warps[wi].pending_acq.push((lane, rel.clone())),
                        MemOrder::Weak => {}
                    }
                }
            }
        }
    }

    /// Check a non-generic access against the generic witnesses GC folded
    /// into the allocation's summary, then mark the allocation sensitive.
    fn check_retired_generic(&mut self, a: &Access, cur: Cur, cw: &Witness) {
        let Some(alloc) = self.allocs.get_mut(&a.alloc) else { return };
        alloc.sensitive = true;
        if alloc.retired.is_empty() {
            return;
        }
        let entries: Vec<RetiredGeneric> = alloc
            .retired
            .values()
            .filter(|e| e.lo < a.range.end && a.range.start < e.hi && (e.write || cw.writes()))
            .cloned()
            .collect();
        for e in entries {
            let kind = if e.write { AccessKind::Write } else { AccessKind::Read };
            let pw = Witness::pack(e.stamp, e.lane, Proxy::Generic, e.domain, kind, None, false, (e.lo, e.hi), &mut self.wide);
            if !self.ordered(cur, &pw, a.proxy) {
                let bytes = e.lo.max(a.range.start)..e.hi.min(a.range.end);
                self.report_race_with(a.alloc, bytes, cur, &pw, cw, Some(e.info.clone()));
            }
        }
    }

    /// Acquire a release payload, checking that the scopes mutually cover.
    fn acquire_rel(&mut self, me: WarpId, lanes: LaneMask, my_scope: Scope, rel: &Rel) {
        let Some(rs) = rel.scope else {
            for c in lanes.lanes8() {
                self.warps[me as usize].tcgen_in[c as usize].join(&rel.k.tcgen_rel, &self.memo);
            }
            return;
        };
        if self.covers(rs, rel.warp, me) && self.covers(my_scope, me, rel.warp) {
            let memo = &self.memo;
            self.warps[me as usize].acquire(lanes, &rel.k, memo);
        } else {
            let f = Finding {
                kind: FindingKind::ScopeMismatch { release_scope: rs, acquire_scope: my_scope, release_warp: rel.warp, acquire_warp: me },
                severity: Severity::Error,
                alloc: AllocId(u32::MAX),
                bytes: 0..0,
                prior: None,
                current: None,
                occurrences: 1,
            };
            self.push_finding(f);
        }
    }

    // ----------------------------------------------------------- sync --

    pub fn sync(&mut self, s: SyncEvent) {
        self.maybe_gc();
        match s {
            SyncEvent::AllocBegin { alloc, size, space, .. } => {
                self.allocs.insert(alloc, Alloc { size, space, shadow: IntervalShadow::new(), sensitive: false, retired: HashMap::new() });
            }
            SyncEvent::AllocEnd { alloc } => {
                let mut lifetime = Vec::new();
                for a in &self.asyncs {
                    if a.in_use && a.done == 0 {
                        if let Some((_, r)) = a.footprint.iter().find(|(al, _)| *al == alloc) {
                            lifetime.push((a.op, r.clone()));
                        }
                    }
                }
                for (op, range) in lifetime {
                    let f = Finding {
                        kind: FindingKind::AsyncLifetime { op },
                        severity: Severity::Error,
                        alloc,
                        bytes: range,
                        prior: None,
                        current: None,
                        occurrences: 1,
                    };
                    self.push_finding(f);
                }
                self.allocs.remove(&alloc);
                self.words.retain(|w| w.alloc != alloc);
            }
            SyncEvent::DeclareWord { alloc, range } => {
                if !self.words.iter().any(|w| w.alloc == alloc && w.range == range) {
                    self.words.push(Word { alloc, range, history: Vec::new() });
                }
            }
            SyncEvent::WarpSync { warp, mask, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                self.warp_sync(warp, mask, epoch);
            }
            SyncEvent::Arrive { warp, lanes, obj, phase, release, scope, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                let Some(release) = release else {
                    self.report.incomplete.push(Incomplete::SyncQualifierUnknown { warp });
                    return;
                };
                let w = &self.warps[warp as usize];
                let tc = w.tcgen_publication(lanes, &self.memo);
                let pubk = release.then(|| Arc::new(w.publication(lanes, epoch, &self.memo)));
                let ph = self.phases.entry((obj, phase)).or_default();
                ph.tcgen_rel.join(&tc, &self.memo);
                if let Some(k) = pubk {
                    let cta = self.topo.cta_of(warp);
                    let topo = self.topo;
                    match ph.arrivals.iter_mut().find(|(aw, s, _)| *s == scope && topo.cta_of(*aw) == cta) {
                        Some((_, _, g)) => Arc::make_mut(g).join_propagating(&k, &self.memo),
                        None => ph.arrivals.push((warp, scope, k)),
                    }
                }
            }
            SyncEvent::Wait { warp, lanes, obj, phase, acquire, scope, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                let Some(acquire) = acquire else {
                    self.report.incomplete.push(Incomplete::SyncQualifierUnknown { warp });
                    return;
                };
                let ph = self.phases.entry((obj, phase)).or_default();
                let topo = self.topo;
                let assumed = self.mbarrier_scope_assumed && matches!(obj, SyncObjId::Mbarrier { .. });
                let mut unknown_scope = false;
                let memo = &self.memo;
                let w = &mut self.warps[warp as usize];
                if !ph.tcgen_rel.is_empty() {
                    for c in lanes.lanes8() {
                        w.tcgen_in[c as usize].join(&ph.tcgen_rel, memo);
                    }
                }
                for (aw, ascope, k) in &ph.arrivals {
                    if acquire {
                        let ok = match (ascope, scope) {
                            (Some(a), Some(s)) => *a >= required_scope(&topo, *aw, warp) && s >= required_scope(&topo, warp, *aw),
                            _ => true, // named barrier: participants, no scope
                        };
                        if ok {
                            w.acquire(lanes, k, memo);
                        } else if assumed {
                            unknown_scope = true;
                        }
                    } else {
                        let rel = Arc::new(Rel { k: (**k).clone(), scope: Some(ascope.unwrap_or(Scope::Cta)), warp: *aw });
                        for c in lanes.lanes8() {
                            w.pending_acq.push((c, rel.clone()));
                        }
                    }
                }
                if unknown_scope {
                    self.report.incomplete.push(Incomplete::MbarrierScopeUnknown { warp });
                }
                let ph = &self.phases[&(obj, phase)];
                let memo = &self.memo;
                let w = &mut self.warps[warp as usize];
                if acquire {
                    w.acquire(lanes, &ph.completion, memo);
                } else {
                    // Any later acquire fence of the waiter suffices for the
                    // copy's own bytes: scope Sys, attributed to the waiter.
                    let rel = Arc::new(Rel { k: ph.completion.clone(), scope: Some(Scope::Sys), warp });
                    for c in lanes.lanes8() {
                        w.pending_acq.push((c, rel.clone()));
                    }
                }
            }
            SyncEvent::Fence { warp, lanes, kind, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                self.fence(warp, lanes, kind, epoch);
            }
            SyncEvent::AsyncIssue { op, warp, lanes, kind, proxy: _, preds, footprint, site, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                self.async_issue(op, warp, lanes, kind, preds, footprint, site, epoch);
            }
            SyncEvent::AsyncComplete { op, milestone, target } => {
                let Some(i) = self.async_idx(op) else { return };
                let m = side_index(milestone);
                let a = &mut self.asyncs[i];
                a.done = a.done.max(m as u8);
                let (actor, kind, gen_base) = (a.actor, a.kind, a.gen_base);
                if kind == AsyncKind::TcgenCommit {
                    for p in a.preds.clone() {
                        self.asyncs[p].done = 2;
                    }
                }
                let c = self.completion(i, m);
                match target {
                    CompletionTarget::Phase { obj, phase } => {
                        let ph = self.phases.entry((obj, phase)).or_default();
                        ph.completion.join_propagating(&c, &self.memo);
                    }
                    CompletionTarget::Warp { warp, lanes } => {
                        let memo = &self.memo;
                        let w = &mut self.warps[warp as usize];
                        match kind {
                            AsyncKind::TcgenLd | AsyncKind::TcgenSt | AsyncKind::TcgenPipelined => {
                                // tcgen05.wait::{ld,st}: orders the waiting
                                // thread's own later tcgen05 work; others
                                // need the fence pair.
                                for l in lanes.lanes8() {
                                    w.tcgen[l as usize].raise(actor, gen_base + 2);
                                    w.tcgen_waited[l as usize].raise(actor, gen_base + 2);
                                }
                            }
                            _ => w.acquire(lanes, &c, memo),
                        }
                    }
                }
            }
            SyncEvent::WaitVerdicts { warp, lanes, alloc, range, scope, accepted, observed, pred_reads, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                if !self.pred_reads_stable(warp, lanes, epoch, &pred_reads) {
                    self.report.incomplete.push(Incomplete::WaitPredicateReadsUnstable { warp });
                    return;
                }
                self.wait_verdicts(warp, lanes, alloc, range, scope, &accepted, observed);
            }
        }
    }

    fn warp_sync(&mut self, warp: WarpId, mask: LaneMask, epoch: Epoch) {
        let memo = &self.memo;
        let w = &mut self.warps[warp as usize];
        let mut j = [0; 32];
        for c in mask.lanes8() {
            for (p, slot) in j.iter_mut().enumerate() {
                let seen = if p == c as usize { epoch } else { w.row[c as usize][p] };
                *slot = (*slot).max(seen);
            }
        }
        for c in mask.lanes8() {
            w.row[c as usize] = j;
        }
        if let Some(br) = &mut w.bridge_rows {
            for rows in br.iter_mut() {
                let mut j = [0; 32];
                for c in mask.lanes8() {
                    for (p, slot) in j.iter_mut().enumerate() {
                        *slot = (*slot).max(rows[c as usize][p]);
                    }
                }
                for c in mask.lanes8() {
                    rows[c as usize] = j;
                }
            }
        }
        // A warp sync is a thread sync: tcgen fence frontiers flow too.
        let tp = w.tcgen_publication(mask, memo);
        for c in mask.lanes8() {
            w.tcgen_in[c as usize].join(&tp, memo);
        }
        let mut jx: Option<Knowledge> = None;
        for c in mask.lanes8() {
            if let Some(x) = &w.extra[c as usize] {
                let j = jx.get_or_insert_with(Default::default);
                j.join_propagating(x, memo);
                j.g2t.join(&x.g2t, memo);
            }
        }
        if let Some(jx) = jx {
            if mask.is_all() {
                w.base.join_propagating(&jx, memo);
                for x in w.extra.iter_mut() {
                    *x = None;
                }
            } else {
                for c in mask.lanes8() {
                    w.extra[c as usize] = Some(Box::new(jx.clone()));
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn async_issue(
        &mut self,
        op: AsyncId,
        warp: WarpId,
        lanes: LaneMask,
        kind: AsyncKind,
        preds: Vec<AsyncId>,
        footprint: Vec<(AllocId, Range<u64>)>,
        site: SiteId,
        epoch: Epoch,
    ) {
        if kind == AsyncKind::TcgenCommit {
            // tcgen05.commit carries an implicit fence::before_thread_sync
            // for its issuing thread.
            self.fence(warp, lanes, FenceKind::TcgenBefore, epoch);
        }
        let w = &self.warps[warp as usize];
        let mut k = w.publication(lanes, epoch, &self.memo);
        for c in lanes.lanes8() {
            k.tcgen.join(&w.tcgen[c as usize], &self.memo);
            k.g2t.join(&w.lane_view(c, View::G2t, &self.memo), &self.memo);
        }
        let mut pred_idx = Vec::new();
        for p in preds {
            let Some(pi) = self.async_index.get(&p).copied() else {
                if !self.reclaimed.contains(&p) {
                    self.report.incomplete.push(Incomplete::UnknownAsyncOp { op: p });
                }
                // A reclaimed op completed and left no witness behind.
                continue;
            };
            let pa = &self.asyncs[pi];
            if kind != AsyncKind::TcgenCommit {
                k.tcgen.join(&pa.k.tcgen, &self.memo);
                k.tcgen.raise(pa.actor, pa.gen_base + 2);
            }
            pred_idx.push(pi);
        }
        let lane = lanes.lanes8().next().unwrap_or(0);
        let nw = self.topo.num_warps();
        let idx = match self.free_slots.pop() {
            Some(i) => i,
            None => {
                self.asyncs.push(AsyncActor {
                    op,
                    actor: nw + self.asyncs.len() as u32,
                    gen_base: 0,
                    in_use: false,
                    warp,
                    lane,
                    issue_epoch: epoch,
                    site,
                    kind,
                    k: Knowledge::default(),
                    preds: Vec::new(),
                    footprint: Vec::new(),
                    done: 0,
                });
                self.stats.async_slots += 1;
                self.asyncs.len() - 1
            }
        };
        let slot = &mut self.asyncs[idx];
        let actor = slot.actor;
        slot.op = op;
        slot.in_use = true;
        slot.warp = warp;
        slot.lane = lane;
        slot.issue_epoch = epoch;
        slot.site = site;
        slot.kind = kind;
        slot.k = k;
        slot.preds = pred_idx;
        slot.footprint = footprint;
        slot.done = 0;
        let gen_base = slot.gen_base;
        if kind == AsyncKind::TcgenPipelined {
            for c in lanes.lanes8() {
                self.warps[warp as usize].tcgen_issued[c as usize].raise(actor, gen_base + 2);
            }
        }
        self.async_index.insert(op, idx);
    }

    /// Knowledge an async milestone publishes.
    fn completion(&self, i: usize, m: u32) -> Knowledge {
        let a = &self.asyncs[i];
        if a.kind == AsyncKind::TcgenCommit {
            // Commit forwards the issuer's generic knowledge at commit and,
            // for every tracked op, that op plus its causal predecessors —
            // but not tcgen knowledge the issuer merely holds.
            let mut c = a.k.propagating();
            c.tcgen_rel = Clock::default();
            c.hb.raise(a.actor, a.gen_base + m);
            for &p in &a.preds {
                let pa = &self.asyncs[p];
                c.hb.raise(pa.actor, pa.gen_base + 2);
                c.tcgen_rel.raise(pa.actor, pa.gen_base + 2);
                c.tcgen_rel.join(&pa.k.tcgen, &self.memo);
            }
            return c;
        }
        // Copy completion projection: only the op's own milestone, plus the
        // implicit async→generic bridge for it in every domain.
        let mut c = Knowledge::default();
        c.hb.raise(a.actor, a.gen_base + m);
        for d in 0..NDOM {
            c.a2g[d].raise(a.actor, a.gen_base + m);
        }
        c
    }

    /// Record the own-warp rows of bridge slot `s` for lanes `lanes`.
    fn bridge_rows(&mut self, warp: WarpId, lanes: LaneMask, slots: &[usize], epoch: Epoch) {
        let w = &mut self.warps[warp as usize];
        let br = w.bridge_rows.get_or_insert_with(|| Box::new([[[0; 32]; 32]; NSLOT]));
        for &s in slots {
            for c in lanes.lanes8() {
                let mut r = w.row[c as usize];
                r[c as usize] = epoch;
                for (dst, src) in br[s][c as usize].iter_mut().zip(r.iter()) {
                    *dst = (*dst).max(*src);
                }
            }
        }
    }

    /// `slot ⊔= hb` for lanes `lanes` (base when the whole warp fences).
    fn snapshot_hb_into(&mut self, warp: WarpId, lanes: LaneMask, slots: &[usize]) {
        let memo = &self.memo;
        let w = &mut self.warps[warp as usize];
        if lanes.is_all() {
            let hb = w.base.hb.clone();
            for &s in slots {
                slot_mut(&mut w.base, s).join(&hb, memo);
            }
            for x in w.extra.iter_mut().flatten() {
                let hb = x.hb.clone();
                for &s in slots {
                    slot_mut(x, s).join(&hb, memo);
                }
            }
        } else {
            let base_hb = w.base.hb.clone();
            for c in lanes.lanes8() {
                let x = w.extra[c as usize].get_or_insert_with(Default::default);
                let mut hb = x.hb.clone();
                hb.join(&base_hb, memo);
                for &s in slots {
                    slot_mut(x, s).join(&hb, memo);
                }
            }
        }
    }

    fn fence(&mut self, warp: WarpId, lanes: LaneMask, kind: FenceKind, epoch: Epoch) {
        match kind {
            FenceKind::AcqRel(scope) | FenceKind::Sc(scope) => {
                // Acquire half: pending relaxed observations.
                let pending = std::mem::take(&mut self.warps[warp as usize].pending_acq);
                let mut keep = Vec::new();
                for (lane, rel) in pending {
                    if lanes.has(lane) {
                        self.acquire_rel(warp, one_lane(lane), scope, &rel);
                    } else {
                        keep.push((lane, rel));
                    }
                }
                self.warps[warp as usize].pending_acq = keep;
                if let FenceKind::Sc(_) = kind {
                    // Fence-SC order relates every pair of *morally strong*
                    // fence.sc (PTX §8.9.3). Delivery order is a legal
                    // runtime order.
                    for c in lanes.lanes8() {
                        let incoming: Vec<Arc<Knowledge>> = self
                            .sc
                            .iter()
                            .filter(|((ow, ol), (os, _))| (*ow, *ol) != (warp, c) && self.covers(*os, *ow, warp) && self.covers(scope, warp, *ow))
                            .map(|(_, (_, k))| k.clone())
                            .collect();
                        for k in incoming {
                            self.warps[warp as usize].acquire(one_lane(c), &k, &self.memo);
                        }
                    }
                    for c in lanes.lanes8() {
                        let k = Arc::new(self.warps[warp as usize].publication(one_lane(c), epoch, &self.memo));
                        self.sc.insert((warp, c), (scope, k));
                    }
                }
                // Release half: the head a later relaxed strong write carries.
                let w = &mut self.warps[warp as usize];
                for c in lanes.lanes8() {
                    let k = w.publication(one_lane(c), epoch, &self.memo);
                    w.fence_rel[c as usize] = Some(Arc::new(Rel { k, scope: Some(scope), warp }));
                }
            }
            FenceKind::ProxyAsync(dom) => {
                let slots: Vec<usize> = fence_domains(dom).iter().flat_map(|&d| [d, NDOM + d]).collect();
                self.bridge_rows(warp, lanes, &slots, epoch);
                self.snapshot_hb_into(warp, lanes, &slots);
            }
            FenceKind::TensormapRelease => {
                self.bridge_rows(warp, lanes, &[TSLOT], epoch);
                self.snapshot_hb_into(warp, lanes, &[TSLOT]);
            }
            FenceKind::TensormapAcquire => {
                // The acquiring thread's tensormap view: every release that
                // reached it, including its own warp's released lanes.
                let memo = &self.memo;
                let w = &mut self.warps[warp as usize];
                for c in lanes.lanes8() {
                    let mut t = w.base.tmap_rel.clone();
                    if let Some(x) = &w.extra[c as usize] {
                        t.join(&x.tmap_rel, memo);
                    }
                    if let Some(br) = &w.bridge_rows {
                        let own = br[TSLOT][c as usize];
                        if own.iter().any(|x| *x != 0) {
                            t.raise_lanes(w.actor, &own);
                        }
                    }
                    let x = w.extra[c as usize].get_or_insert_with(Default::default);
                    x.g2t.join(&t, memo);
                }
            }
            FenceKind::TcgenBefore => {
                // Publish (and order before the thread's own later tcgen
                // work) every issued pipelined op, every waited ld/st and
                // the fenced view.
                let memo = &self.memo;
                let w = &mut self.warps[warp as usize];
                for c in lanes.lanes8() {
                    let c = c as usize;
                    let mut p = w.tcgen_issued[c].clone();
                    p.join(&w.tcgen_waited[c], memo);
                    p.join(&w.tcgen[c], memo);
                    let issued = w.tcgen_issued[c].clone();
                    w.tcgen[c].join(&issued, memo);
                    w.tcgen_pub[c].join(&p, memo);
                }
            }
            FenceKind::TcgenAfter => {
                let memo = &self.memo;
                let w = &mut self.warps[warp as usize];
                for c in lanes.lanes8() {
                    let i = w.tcgen_in[c as usize].clone();
                    w.tcgen[c as usize].join(&i, memo);
                }
            }
        }
    }

    /// Every write to the predicate's extra inputs happens before the wait
    /// (for every waiting lane).
    fn pred_reads_stable(&self, warp: WarpId, lanes: LaneMask, epoch: Epoch, reads: &[(AllocId, Range<u64>)]) -> bool {
        reads.iter().all(|(alloc, r)| {
            let Some(a) = self.allocs.get(alloc) else { return false };
            let mut ok = true;
            a.shadow.visit(r.clone(), |_, cell| {
                for p in cell.writes.as_slice() {
                    for lane in lanes.lanes8() {
                        let cur = Cur::Lane { w: warp as usize, lane, epoch };
                        ok &= self.ordered(cur, &p.w, Proxy::Generic);
                    }
                }
            });
            ok
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn wait_verdicts(&mut self, warp: WarpId, lanes: LaneMask, alloc: AllocId, range: Range<u64>, scope: Scope, accepted: &[u64], observed: u32) {
        let Some(wi) = self.words.iter().position(|w| w.alloc == alloc && w.range == range) else {
            self.report.incomplete.push(Incomplete::WaitExitUnproven { warp });
            return;
        };
        // Earliest predicate-accepted history entry: schedule independent.
        let first = accepted.iter().enumerate().find(|(_, b)| **b != 0).map(|(i, b)| i as u32 * 64 + b.trailing_zeros());
        let Some(mut idx) = first else {
            self.report.incomplete.push(Incomplete::WaitExitUnproven { warp });
            return;
        };
        if idx == 0 {
            return; // the launch value satisfied the predicate: no edge owed
        }
        let Some(e) = self.words[wi].history.get(idx as usize - 1) else {
            self.report.incomplete.push(Incomplete::WaitExitUnproven { warp });
            return;
        };
        if e.is_async {
            // Async publications land their bytes at completion; the run's
            // observed version is the only edge available (degraded,
            // schedule-dependent fallback, async publications only).
            idx = observed;
            if idx == 0 {
                return;
            }
        }
        let Some(heads) = self.words[wi].history.get(idx as usize - 1).and_then(|e| e.rel.clone()) else {
            return; // plain publication: no edge, later reads race
        };
        for rel in heads.iter() {
            self.acquire_rel(warp, lanes, scope, rel);
        }
    }

    // ------------------------------------------------------------- GC --

    /// View-aware dominated-frontier GC plus async-slot reclaim.
    pub fn gc(&mut self) {
        self.since_gc = 0;
        self.stats.gc_runs += 1;
        // Meet of what every live actor knows, per view. Lane extras only
        // add knowledge, so `base` is a sound lower bound for a warp; the
        // per-lane tcgen views are met lane by lane.
        let mut meet: Option<Knowledge> = None;
        let mut fold = |k: &Knowledge, tcgen: Clock| {
            let m = meet.get_or_insert_with(|| {
                let mut x = k.clone();
                x.tcgen = tcgen.clone();
                x
            });
            m.hb = m.hb.meet(&k.hb);
            for d in 0..NDOM {
                m.g2a[d] = m.g2a[d].meet(&k.g2a[d]);
                m.a2g[d] = m.a2g[d].meet(&k.a2g[d]);
            }
            m.g2t = m.g2t.meet(&k.g2t);
            m.tcgen = m.tcgen.meet(&tcgen);
        };
        for w in self.warps.iter().filter(|w| !w.done) {
            let mut t = w.tcgen[0].clone();
            for l in 1..32 {
                t = t.meet(&w.tcgen[l]);
            }
            fold(&w.base, t);
        }
        for a in self.asyncs.iter().filter(|a| a.in_use && a.done < 2 && a.kind != AsyncKind::TcgenCommit) {
            fold(&a.k, a.k.tcgen.clone());
        }
        let Some(meet) = meet else { return };
        let fully_dead = |w: &Witness, space: Space| {
            future_proxies(space).iter().all(|pc| meet.view(select_view(w.proxy(), *pc, w.domain())).observes(w.stamp, w.lane()))
        };
        let same_proxy_dead = |w: &Witness| meet.view(select_view(w.proxy(), w.proxy(), w.domain())).observes(w.stamp, w.lane());
        let mut retired = 0u64;
        let mut live_actors: HashSet<ActorId> = HashSet::new();
        let mut min_epoch: HashMap<ActorId, Epoch> = HashMap::new();
        let mut allocs = std::mem::take(&mut self.allocs);
        for alloc in allocs.values_mut() {
            let (space, sensitive) = (alloc.space, alloc.sensitive);
            let mut folded: Vec<Witness> = Vec::new();
            alloc.shadow.retain_mut(|cell| {
                let last = cell.writes.last().map(|e| e.w);
                // Decide per witness: drop (dead in every view), fold into the
                // retired-generic summary (dead in its own proxy, allocation
                // not proxy sensitive), or keep.
                let mut decide = |w: &Witness, pinned: bool| -> bool {
                    if pinned {
                        return true;
                    }
                    if fully_dead(w, space) {
                        retired += 1;
                        return false;
                    }
                    // Async-actor stamps may be summarised too: once every
                    // live actor's hb observes them, any view carrying a
                    // later generation of the slot was snapshotted after
                    // that, so it observes the old stamp as well.
                    if !sensitive && w.proxy() == Proxy::Generic && space != Space::Tmem && same_proxy_dead(w) {
                        folded.push(*w);
                        retired += 1;
                        return false;
                    }
                    true
                };
                // Keep the latest write while it carries release heads: a
                // future reader that reads from it needs them.
                cell.writes.retain(|e| decide(&e.w, Some(e.w) == last && e.rel.is_some()));
                cell.reads.retain(|e| decide(&e.w, false));
                for e in cell.writes.as_slice().iter().chain(cell.reads.as_slice()) {
                    live_actors.insert(e.w.stamp.actor());
                    let m = min_epoch.entry(e.w.stamp.actor()).or_insert(u32::MAX);
                    *m = (*m).min(e.w.stamp.epoch());
                }
                !cell.is_empty()
            });
            for w in folded {
                let info = self.info(&w);
                let (lo, hi) = w.span(&self.wide);
                let key = (w.stamp.actor(), w.lane(), w.writes(), domain_code(w.domain()));
                let e = alloc.retired.entry(key).or_insert_with(|| RetiredGeneric {
                    stamp: w.stamp,
                    lane: w.lane(),
                    write: w.writes(),
                    domain: w.domain(),
                    lo,
                    hi,
                    info: info.clone(),
                });
                if w.stamp.epoch() >= e.stamp.epoch() {
                    e.stamp = w.stamp;
                    e.info = info;
                }
                e.lo = e.lo.min(lo);
                e.hi = e.hi.max(hi);
                e.info.span = e.lo..e.hi;
            }
        }
        self.allocs = allocs;
        self.stats.witnesses_retired += retired;
        // Async-slot reclaim: completed ops with no witness left.
        for i in 0..self.asyncs.len() {
            let a = &self.asyncs[i];
            if a.in_use && a.done >= 2 && !live_actors.contains(&a.actor) {
                let op = a.op;
                self.async_index.remove(&op);
                self.reclaimed.insert(op);
                let a = &mut self.asyncs[i];
                a.in_use = false;
                a.gen_base += 2; // next generation starts above every old epoch
                a.k = Knowledge::default();
                a.preds.clear();
                a.footprint.clear();
                self.free_slots.push(i);
                self.stats.async_slots_reclaimed += 1;
            }
        }
        // Site tables: keep only epochs a witness may still name.
        for w in &mut self.warps {
            let keep_from = min_epoch.get(&w.actor).copied().unwrap_or(w.epoch);
            let i = w.sites.partition_point(|(e, _)| *e <= keep_from);
            if i > 1 {
                w.sites.drain(..i - 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::cell::Frontier;
    use super::*;

    #[test]
    fn required_scope_matrix() {
        let t = Topology { warps_per_cta: 4, ctas_per_cluster: 2, num_ctas: 4 };
        assert_eq!(required_scope(&t, 0, 3), Scope::Cta);
        assert_eq!(required_scope(&t, 0, 4), Scope::Cluster);
        assert_eq!(required_scope(&t, 0, 8), Scope::Gpu);
    }

    #[test]
    fn frontier_one_steady_state() {
        let mut wide = WideSpans::default();
        let mut f = Frontier::Empty;
        let mut w = |e| Witness::pack(Stamp::new(0, e), 0, Proxy::Generic, None, AccessKind::Read, None, false, (0, 4), &mut wide);
        let (w1, w2, w3) = (w(1), w(2), w(3));
        f.record(Entry { w: w1, rel: None, base: None }, &wide, |_| true);
        f.record(Entry { w: w2, rel: None, base: None }, &wide, |_| true);
        assert!(matches!(f, Frontier::One(_)));
        f.record(Entry { w: w3, rel: None, base: None }, &wide, |_| false);
        assert_eq!(f.as_slice().len(), 2);
    }
}
