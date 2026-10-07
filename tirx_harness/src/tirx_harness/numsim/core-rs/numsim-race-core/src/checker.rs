//! The unified online checker: one `Clock` model, one `IntervalShadow<Cell>`
//! per allocation, one conflict rule — for shared memory, TMEM and global
//! memory alike. Memory-space differences are data (actor sets, the proxy
//! view consulted, scope topology), not separate engines.
//!
//! Races never abort: a finding is recorded and checking continues (the
//! legacy shared/TMEM shadow aborts on its first race; see the spec).

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use crate::cell::{effective_heads, overlap, Cell, Entry, Witness};
use crate::clock::{ActorId, Clock, Epoch, JoinMemo, LaneVec, Stamp};
use crate::input::*;
use crate::knowledge::{fence_domains, select_view, Heads, Knowledge, Rel, View, NDOM};
use crate::shadow::IntervalShadow;

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
    /// Issuing warp/lane for async ops (legacy attributes async accesses to
    /// their issuer; the op id is kept as extra evidence).
    pub warp: WarpId,
    pub lane: u8,
    pub async_op: Option<AsyncId>,
    pub kind: AccessKind,
    pub proxy: Proxy,
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
    /// base causality. PTX §8.9.5 preserves same-proxy order only "by the
    /// same thread block"; which block an async op belongs to is ISA-silent.
    CrossCtaAsyncOrder,
    /// A strong load observed an unordered, morally strong write on a word
    /// that is not a declared `wait_until` word: not a race (§8.7.1), but the
    /// edge it yields is schedule dependent.
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Incomplete {
    EpochRegression { warp: WarpId },
    EpochOverflow { warp: WarpId },
    UnknownAsyncOp { op: AsyncId },
    UnknownAlloc { alloc: AllocId },
    /// `wait_until` exited but no history entry explains it.
    WaitExitUnproven { warp: WarpId },
    /// A `wait_until` predicate read memory that was written concurrently
    /// with the wait, so the verdict bitset does not determine the exit.
    WaitPredicateReadsUnstable { warp: WarpId },
    /// Async op never reached a milestone by launch end.
    AsyncNeverCompleted { op: AsyncId },
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

// ----------------------------------------------------------------- state --

const NSLOT: usize = 2 * NDOM; // bridge rows: g2a[0..3], a2g[0..3]

struct Warp {
    actor: ActorId,
    epoch: Epoch,
    /// Lane-order matrix: `row[c][p]` = latest epoch of lane `p` that lane
    /// `c` has observed (via warp sync / collectives). Diagonal unused.
    row: Box<[LaneVec; 32]>,
    /// Knowledge common to every lane.
    base: Knowledge,
    /// Lane-specific acquisitions (single-lane ld.acquire, lane-masked
    /// mbarrier waits). Sparse: `None` for lanes that never diverged.
    extra: Vec<Option<Box<Knowledge>>>,
    /// Own-warp part of each proxy bridge: `bridge_rows[s][c][p]` = latest
    /// epoch of lane `p` bridged for lane `c`.
    bridge_rows: Option<Box<[[LaneVec; 32]; NSLOT]>>,
    /// Latest release-fence head per lane.
    fence_rel: Vec<Option<Arc<Rel>>>,
    /// Release payloads observed by relaxed strong loads, awaiting an
    /// acquire fence.
    pending_acq: Vec<(u8, Arc<Rel>)>,
    // tcgen05 ordering (per warp: the issuing lane is elected).
    tcgen: Clock,
    tcgen_in: Clock,
    tcgen_issued: Clock,
    tcgen_waited: Clock,
}

impl Warp {
    fn new(actor: ActorId) -> Self {
        Warp {
            actor,
            epoch: 0,
            row: Box::new([[0; 32]; 32]),
            base: Knowledge::default(),
            extra: vec![None; 32],
            bridge_rows: None,
            fence_rel: vec![None; 32],
            pending_acq: Vec::new(),
            tcgen: Clock::default(),
            tcgen_in: Clock::default(),
            tcgen_issued: Clock::default(),
            tcgen_waited: Clock::default(),
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
        for c in lanes.lanes() {
            if let Some(x) = &self.extra[c as usize] {
                k.join_propagating(x, memo);
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
                for c in lanes.lanes() {
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

    fn acquire(&mut self, lanes: LaneMask, k: &Knowledge, memo: &JoinMemo) {
        if lanes.is_full() {
            self.base.hb.join(&k.hb, memo);
            for d in 0..NDOM {
                self.base.g2a[d].join(&k.g2a[d], memo);
                self.base.a2g[d].join(&k.a2g[d], memo);
            }
        } else {
            for c in lanes.lanes() {
                let x = self.extra[c as usize].get_or_insert_with(Default::default);
                x.hb.join(&k.hb, memo);
                for d in 0..NDOM {
                    x.g2a[d].join(&k.g2a[d], memo);
                    x.a2g[d].join(&k.a2g[d], memo);
                }
            }
        }
        self.tcgen_in.join(&k.tcgen_rel, memo);
    }
}

fn slot_mut(k: &mut Knowledge, s: usize) -> &mut Clock {
    if s < NDOM {
        &mut k.g2a[s]
    } else {
        &mut k.a2g[s - NDOM]
    }
}

struct AsyncActor {
    op: AsyncId,
    actor: ActorId,
    warp: WarpId,
    lane: u8,
    issue_epoch: Epoch,
    kind: AsyncKind,
    k: Knowledge,
    preds: Vec<usize>,
    footprint: Vec<(AllocId, Range<u64>)>,
    done: u8,
}

struct Alloc {
    size: u64,
    shadow: IntervalShadow<Cell>,
}

#[derive(Default)]
struct Phase {
    arrive: Knowledge,
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
    async_index: HashMap<AsyncId, usize>,
    allocs: HashMap<AllocId, Alloc>,
    phases: HashMap<(SyncObjId, u32), Phase>,
    sc: HashMap<(Scope, u32), Knowledge>,
    words: Vec<Word>,
    report: Report,
    dedup: HashMap<(AllocId, RaceClass, SiteId, SiteId, Option<AsyncId>), usize>,
}

fn required_scope(t: &Topology, a: WarpId, b: WarpId) -> Scope {
    if t.cta_of(a) == t.cta_of(b) {
        Scope::Cta
    } else if t.cluster_of(a) == t.cluster_of(b) {
        Scope::Cluster
    } else {
        Scope::Gpu
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
            async_index: HashMap::new(),
            allocs: HashMap::new(),
            phases: HashMap::new(),
            sc: HashMap::new(),
            words: Vec::new(),
            report: Report::default(),
            dedup: HashMap::new(),
        }
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

    pub fn finish(mut self) -> Report {
        for a in &self.asyncs {
            if a.done == 0 && a.kind != AsyncKind::TcgenCommit {
                self.report.incomplete.push(Incomplete::AsyncNeverCompleted { op: a.op });
            }
        }
        self.report
    }

    fn covers(&self, s: Scope, a: WarpId, b: WarpId) -> bool {
        s >= required_scope(&self.topo, a, b)
    }

    fn tick(&mut self, w: WarpId, epoch: Epoch) -> bool {
        let warp = &mut self.warps[w as usize];
        if epoch < warp.epoch {
            self.report.incomplete.push(Incomplete::EpochRegression { warp: w });
            return false;
        }
        if epoch == u32::MAX {
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

    // ------------------------------------------------------- ordering --

    /// Is `prior` ordered before an access by `cur` in proxy `cur_proxy`?
    #[inline]
    fn ordered(&self, cur: Cur, prior: &Witness, cur_proxy: Proxy) -> bool {
        let view = select_view(prior.proxy, cur_proxy, prior.domain);
        match cur {
            Cur::Lane { w, lane, epoch } => {
                let warp = &self.warps[w];
                if prior.stamp.actor() == warp.actor && view == View::Hb {
                    // Program order within a lane; sibling lanes of one
                    // instruction are simultaneous; otherwise the
                    // lane-order matrix, then knowledge that came back
                    // around through other actors.
                    if prior.lane == lane {
                        return true;
                    }
                    if prior.stamp.epoch() == epoch {
                        return false;
                    }
                    if warp.row[lane as usize][prior.lane as usize] >= prior.stamp.epoch() {
                        return true;
                    }
                }
                if view == View::Tcgen {
                    return warp.tcgen.observes(prior.stamp, prior.lane);
                }
                warp.knows(lane, view, prior.stamp, prior.lane)
            }
            Cur::Async { a } => {
                let act = &self.asyncs[a];
                prior.stamp.actor() == act.actor || act.k.observes(view, prior.stamp, prior.lane)
            }
        }
    }

    fn morally_strong(&self, p: &Witness, c: &Witness) -> bool {
        match (p.scope, c.scope) {
            (Some(ps), Some(cs)) => {
                p.proxy == c.proxy
                    && p.span == c.span
                    && self.covers(ps, p.warp, c.warp)
                    && self.covers(cs, c.warp, p.warp)
            }
            _ => false,
        }
    }

    fn classify(&self, cur: Cur, prior: &Witness, cw: &Witness) -> OrderingFailure {
        if prior.proxy != cw.proxy && prior.proxy != Proxy::Tcgen && cw.proxy != Proxy::Tcgen {
            return OrderingFailure::MissingProxyBridge { prior: prior.proxy, current: cw.proxy, domain: prior.domain };
        }
        if let Cur::Lane { w, .. } = cur {
            if prior.stamp.actor() == self.warps[w].actor {
                return OrderingFailure::MissingSameWarpLaneOrder;
            }
        }
        if prior.stamp.actor() >= self.topo.num_warps() {
            let a = &self.asyncs[(prior.stamp.actor() - self.topo.num_warps()) as usize];
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
        if prior.scope.is_some() || cw.scope.is_some() || prior.kind == AccessKind::Rmw || cw.kind == AccessKind::Rmw {
            OrderingFailure::MissingReleaseAcquire
        } else {
            OrderingFailure::MissingInterActorSync
        }
    }

    /// Async-proxy pair whose two ops were issued from different CTAs.
    fn cross_cta_async(&self, prior: &Witness, cur: Cur, cur_proxy: Proxy) -> bool {
        let nw = self.topo.num_warps();
        let Cur::Async { a } = cur else { return false };
        prior.proxy == Proxy::Async
            && cur_proxy == Proxy::Async
            && prior.stamp.actor() >= nw
            && prior.stamp.actor() != self.asyncs[a].actor
            && self.topo.cta_of(prior.warp) != self.topo.cta_of(self.asyncs[a].warp)
            && (prior.writes() || cur_proxy == Proxy::Async)
    }

    fn report_advisory(&mut self, alloc: AllocId, bytes: Range<u64>, prior: &Witness, cw: &Witness, kind: AdvisoryKind) {
        if !(prior.writes() || cw.writes()) {
            return;
        }
        if self
            .report
            .findings
            .iter()
            .any(|f| f.kind == FindingKind::Advisory { kind } && f.alloc == alloc && f.current.as_ref().is_some_and(|c| c.site == cw.site))
        {
            return;
        }
        self.report.findings.push(Finding {
            kind: FindingKind::Advisory { kind },
            severity: Severity::Review,
            alloc,
            bytes,
            prior: Some(self.info(prior)),
            current: Some(self.info(cw)),
        });
    }

    fn info(&self, w: &Witness) -> WitnessInfo {
        let nw = self.topo.num_warps();
        let (lane, op) = if w.stamp.actor() >= nw {
            let a = &self.asyncs[(w.stamp.actor() - nw) as usize];
            (a.lane, Some(a.op))
        } else {
            (w.lane, None)
        };
        WitnessInfo { warp: w.warp, lane, async_op: op, kind: w.kind, proxy: w.proxy, site: w.site, span: w.span.0..w.span.1 }
    }

    fn report_race(&mut self, alloc: AllocId, bytes: Range<u64>, cur: Cur, prior: &Witness, cw: &Witness) {
        let class = match (prior.writes(), cw.writes()) {
            (true, true) => RaceClass::WriteWrite,
            (true, false) => RaceClass::WriteRead,
            _ => RaceClass::ReadWrite,
        };
        let failure = self.classify(cur, prior, cw);
        let nw = self.topo.num_warps();
        let prior_is_ld = prior.stamp.actor() >= nw
            && self.asyncs[(prior.stamp.actor() - nw) as usize].kind == AsyncKind::TcgenLd;
        let review = prior_is_ld && failure == OrderingFailure::AsyncLifetimeNotDrained;
        let key = (alloc, class, prior.site, cw.site, if review { Some(0) } else { None });
        if let Some(&i) = self.dedup.get(&key) {
            let f = &mut self.report.findings[i];
            f.bytes = f.bytes.start.min(bytes.start)..f.bytes.end.max(bytes.end);
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
            prior: Some(self.info(prior)),
            current: Some(self.info(cw)),
        };
        self.dedup.insert(key, self.report.findings.len());
        self.report.findings.push(f);
    }

    // --------------------------------------------------------- access --

    fn access(&mut self, a: &Access) {
        let Some(size) = self.allocs.get(&a.alloc).map(|x| x.size) else {
            self.report.incomplete.push(Incomplete::UnknownAlloc { alloc: a.alloc });
            return;
        };
        let (cur, stamp, lane, warp) = match a.who {
            Who::Lane { warp, lane, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                (Cur::Lane { w: warp as usize, lane, epoch }, Stamp::new(warp, epoch), lane, warp)
            }
            Who::Async { op, side } => {
                let Some(i) = self.async_idx(op) else { return };
                let act = &self.asyncs[i];
                (Cur::Async { a: i }, Stamp::new(act.actor, side as u32), 0, act.warp)
            }
        };
        if a.range.end > size || a.range.start >= a.range.end {
            self.report.findings.push(Finding {
                kind: FindingKind::OutOfBounds { size },
                severity: Severity::Error,
                alloc: a.alloc,
                bytes: a.range.clone(),
                prior: None,
                current: None,
            });
            return;
        }
        let w = Witness {
            stamp,
            lane,
            proxy: a.proxy,
            domain: a.domain,
            kind: a.kind,
            scope: a.scope,
            atomic: a.atomic,
            span: (a.range.start, a.range.end),
            site: a.site,
            warp,
        };
        let writes = w.writes();
        let strong = a.scope.is_some();

        // Release payload for writes, minus the RMW release-sequence part.
        let own_rel: Option<Arc<Rel>> = match (cur, writes) {
            (Cur::Lane { w: wi, lane, epoch }, true) if strong => {
                let warp = &self.warps[wi];
                if matches!(a.order, MemOrder::Release | MemOrder::AcqRel) {
                    let mut k = warp.publication(LaneMask::lane(lane), epoch, &self.memo);
                    k.tcgen_rel = warp.base.tcgen_rel.clone();
                    Some(Arc::new(Rel { k, scope: a.scope, warp: wi as u32 }))
                } else if let Some(f) = &warp.fence_rel[lane as usize] {
                    Some(f.clone())
                } else {
                    let k = Knowledge { tcgen_rel: warp.base.tcgen_rel.clone(), ..Default::default() };
                    Some(Arc::new(Rel { k, scope: None, warp: wi as u32 }))
                }
            }
            _ => None,
        };

        let mut races: Vec<(Range<u64>, Witness)> = Vec::new();
        let mut advisories: Vec<(Range<u64>, Witness, AdvisoryKind)> = Vec::new();
        let mut acquired: Vec<Heads> = Vec::new();
        let mut word_rel: Option<Option<Heads>> = None;
        let word_start = self.words.iter().find(|x| x.alloc == a.alloc && x.range.start < a.range.end && a.range.start < x.range.end).map(|x| x.range.start);
        let in_word = word_start.is_some();

        let mut shadow = std::mem::take(&mut self.allocs.get_mut(&a.alloc).unwrap().shadow);
        {
            let this = &*self;
            shadow.update(a.range.clone(), |seg, cell| {
                // 1. check
                let mut check = |p: &Witness| {
                    let ordered = this.ordered(cur, p, a.proxy);
                    let ms = this.morally_strong(p, &w);
                    if !ordered && !ms {
                        races.push((overlap(p.span, &seg), p.clone()));
                    } else if ordered && this.cross_cta_async(p, cur, a.proxy) {
                        advisories.push((overlap(p.span, &seg), p.clone(), AdvisoryKind::CrossCtaAsyncOrder));
                    } else if !ordered && ms && !writes && !in_word && p.writes() {
                        advisories.push((overlap(p.span, &seg), p.clone(), AdvisoryKind::UndeclaredProtocolWord));
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
                    .find(|e| !(e.w.stamp == w.stamp && e.w.lane != w.lane))
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
                    // chain of the write it reads from; heads stay separate so
                    // each is scope-checked against the eventual acquirer.
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
                    let entry = Entry { w: w.clone(), rel, base };
                    cell.writes.record(entry, |p| this.ordered(cur, p, a.proxy));
                    if !strong {
                        // A plain write supersedes the readers it is ordered
                        // after; a strong write must keep them, since a later
                        // access morally strong with it may still race them.
                        cell.reads.retain(|r| !this.ordered(cur, &r.w, a.proxy));
                    }
                } else {
                    cell.reads.record(Entry { w: w.clone(), rel: None, base: None }, |p| this.ordered(cur, p, a.proxy));
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
            for heads in acquired {
                for rel in heads.iter() {
                    self.warps[wi].tcgen_in.join(&rel.k.tcgen_rel, &self.memo);
                    match a.order {
                        MemOrder::Acquire | MemOrder::AcqRel => {
                            self.acquire_rel(wi as u32, LaneMask::lane(lane), a.scope.unwrap(), rel)
                        }
                        MemOrder::Relaxed | MemOrder::Release => self.warps[wi].pending_acq.push((lane, rel.clone())),
                        MemOrder::Weak => {}
                    }
                }
            }
        }
    }

    /// Acquire a release payload, checking that the scopes mutually cover.
    fn acquire_rel(&mut self, me: WarpId, lanes: LaneMask, my_scope: Scope, rel: &Rel) {
        let Some(rs) = rel.scope else {
            self.warps[me as usize].tcgen_in.join(&rel.k.tcgen_rel, &self.memo);
            return;
        };
        if self.covers(rs, rel.warp, me) && self.covers(my_scope, me, rel.warp) {
            let memo = &self.memo;
            self.warps[me as usize].acquire(lanes, &rel.k, memo);
        } else {
            self.report.findings.push(Finding {
                kind: FindingKind::ScopeMismatch { release_scope: rs, acquire_scope: my_scope, release_warp: rel.warp, acquire_warp: me },
                severity: Severity::Error,
                alloc: u32::MAX,
                bytes: 0..0,
                prior: None,
                current: None,
            });
        }
    }

    // ----------------------------------------------------------- sync --

    fn sync(&mut self, s: SyncEvent) {
        match s {
            SyncEvent::AllocBegin { alloc, size, .. } => {
                self.allocs.insert(alloc, Alloc { size, shadow: IntervalShadow::new() });
            }
            SyncEvent::AllocEnd { alloc, site: _ } => {
                for a in &self.asyncs {
                    if a.done == 0 && a.footprint.iter().any(|(al, _)| *al == alloc) {
                        let range = a.footprint.iter().find(|(al, _)| *al == alloc).unwrap().1.clone();
                        self.report.findings.push(Finding {
                            kind: FindingKind::AsyncLifetime { op: a.op },
                            severity: Severity::Error,
                            alloc,
                            bytes: range,
                            prior: None,
                            current: None,
                        });
                    }
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
                let memo = &self.memo;
                let w = &mut self.warps[warp as usize];
                let mut j = [0; 32];
                for c in mask.lanes() {
                    for (p, slot) in j.iter_mut().enumerate() {
                        let seen = if p == c as usize { epoch } else { w.row[c as usize][p] };
                        *slot = (*slot).max(seen);
                    }
                }
                for c in mask.lanes() {
                    w.row[c as usize] = j;
                }
                if let Some(br) = &mut w.bridge_rows {
                    for rows in br.iter_mut() {
                        let mut j = [0; 32];
                        for c in mask.lanes() {
                            for (p, slot) in j.iter_mut().enumerate() {
                                *slot = (*slot).max(rows[c as usize][p]);
                            }
                        }
                        for c in mask.lanes() {
                            rows[c as usize] = j;
                        }
                    }
                }
                let mut jx: Option<Knowledge> = None;
                for c in mask.lanes() {
                    if let Some(x) = &w.extra[c as usize] {
                        jx.get_or_insert_with(Default::default).join_propagating(x, memo);
                    }
                }
                if let Some(jx) = jx {
                    if mask.is_full() {
                        w.base.join_propagating(&jx, memo);
                        w.base.tcgen_rel = w.base.tcgen_rel.clone(); // own publication unchanged
                        for x in w.extra.iter_mut() {
                            *x = None;
                        }
                    } else {
                        for c in mask.lanes() {
                            w.extra[c as usize] = Some(Box::new(jx.clone()));
                        }
                    }
                }
            }
            SyncEvent::Arrive { warp, lanes, obj, phase, release, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                let w = &self.warps[warp as usize];
                let pubk = if release {
                    w.publication(lanes, epoch, &self.memo)
                } else {
                    Knowledge { tcgen_rel: w.base.tcgen_rel.clone(), ..Default::default() }
                };
                let ph = self.phases.entry((obj, phase)).or_default();
                ph.arrive.join_propagating(&pubk, &self.memo);
            }
            SyncEvent::Wait { warp, lanes, obj, phase, acquire, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                let ph = self.phases.entry((obj, phase)).or_default();
                let memo = &self.memo;
                let w = &mut self.warps[warp as usize];
                if acquire {
                    w.acquire(lanes, &ph.arrive, memo);
                } else {
                    w.tcgen_in.join(&ph.arrive.tcgen_rel, memo);
                }
                w.acquire(lanes, &ph.completion, memo);
            }
            SyncEvent::Fence { warp, lanes, kind, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                self.fence(warp, lanes, kind, epoch);
            }
            SyncEvent::AsyncIssue { op, warp, lanes, kind, proxy: _, preds, footprint, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                let idx = self.asyncs.len();
                let actor = self.topo.num_warps() + idx as u32;
                let w = &self.warps[warp as usize];
                let mut k = w.publication(lanes, epoch, &self.memo);
                k.tcgen = w.tcgen.clone();
                let mut pred_idx = Vec::new();
                for p in preds {
                    let Some(pi) = self.async_index.get(&p).copied() else {
                        self.report.incomplete.push(Incomplete::UnknownAsyncOp { op: p });
                        continue;
                    };
                    let pa = &self.asyncs[pi];
                    if kind != AsyncKind::TcgenCommit {
                        k.tcgen.join(&pa.k.tcgen, &self.memo);
                        k.tcgen.raise(pa.actor, 2);
                    }
                    pred_idx.push(pi);
                }
                let lane = lanes.lanes().next().unwrap_or(0);
                if kind == AsyncKind::TcgenCommit {
                    // tcgen05.commit carries an implicit
                    // fence::before_thread_sync for its issuing thread.
                    self.fence(warp, lanes, FenceKind::TcgenBefore, epoch);
                }
                if kind == AsyncKind::TcgenPipelined {
                    self.warps[warp as usize].tcgen_issued.raise(actor, 2);
                }
                self.async_index.insert(op, idx);
                self.asyncs.push(AsyncActor {
                    op,
                    actor,
                    warp,
                    lane,
                    issue_epoch: epoch,
                    kind,
                    k,
                    preds: pred_idx,
                    footprint,
                    done: 0,
                });
            }
            SyncEvent::AsyncComplete { op, milestone, target } => {
                let Some(i) = self.async_idx(op) else { return };
                let m = milestone as u32;
                let a = &mut self.asyncs[i];
                a.done = a.done.max(m as u8);
                let (actor, kind) = (a.actor, a.kind);
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
                                // tcgen05.wait::{ld,st}: orders the warp's own
                                // later tcgen05 work; others need the fence pair.
                                w.tcgen.raise(actor, 2);
                                w.tcgen_waited.raise(actor, 2);
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

    /// Knowledge an async milestone publishes.
    fn completion(&self, i: usize, m: u32) -> Knowledge {
        let a = &self.asyncs[i];
        if a.kind == AsyncKind::TcgenCommit {
            // Commit forwards the issuer's generic knowledge at commit and,
            // for every tracked op, that op plus its causal predecessors —
            // but not tcgen knowledge the issuer merely holds.
            let mut c = a.k.propagating();
            c.tcgen_rel = Clock::default();
            c.hb.raise(a.actor, m);
            for &p in &a.preds {
                let pa = &self.asyncs[p];
                c.hb.raise(pa.actor, 2);
                c.tcgen_rel.raise(pa.actor, 2);
                c.tcgen_rel.join(&pa.k.tcgen, &self.memo);
            }
            return c;
        }
        // Copy completion projection: only the op's own milestone, plus the
        // implicit async→generic bridge for it in every domain.
        let mut c = Knowledge::default();
        c.hb.raise(a.actor, m);
        for d in 0..NDOM {
            c.a2g[d].raise(a.actor, m);
        }
        c
    }

    fn fence(&mut self, warp: WarpId, lanes: LaneMask, kind: FenceKind, epoch: Epoch) {
        match kind {
            FenceKind::AcqRel(scope) | FenceKind::Sc(scope) => {
                // Acquire half: pending relaxed observations.
                let pending = std::mem::take(&mut self.warps[warp as usize].pending_acq);
                let mut keep = Vec::new();
                for (lane, rel) in pending {
                    if lanes.contains(lane) {
                        self.acquire_rel(warp, LaneMask::lane(lane), scope, &rel);
                    } else {
                        keep.push((lane, rel));
                    }
                }
                self.warps[warp as usize].pending_acq = keep;
                if let FenceKind::Sc(_) = kind {
                    // SC fences of one scope instance are totally ordered by
                    // their runtime linearisation (delivery order).
                    let inst = match scope {
                        Scope::Cta => self.topo.cta_of(warp),
                        Scope::Cluster => self.topo.cluster_of(warp),
                        _ => 0,
                    };
                    let k = self.sc.remove(&(scope, inst)).unwrap_or_default();
                    self.warps[warp as usize].acquire(lanes, &k, &self.memo);
                    let mut k2 = k;
                    let pubk = self.warps[warp as usize].publication(lanes, epoch, &self.memo);
                    k2.join_propagating(&pubk, &self.memo);
                    self.sc.insert((scope, inst), k2);
                }
                // Release half: the head a later relaxed strong write carries.
                let w = &mut self.warps[warp as usize];
                for c in lanes.lanes() {
                    let mut k = w.publication(LaneMask::lane(c), epoch, &self.memo);
                    k.tcgen_rel = w.base.tcgen_rel.clone();
                    w.fence_rel[c as usize] = Some(Arc::new(Rel { k, scope: Some(scope), warp }));
                }
            }
            FenceKind::ProxyAsync(dom) => {
                let memo = &self.memo;
                let w = &mut self.warps[warp as usize];
                let br = w.bridge_rows.get_or_insert_with(|| Box::new([[[0; 32]; 32]; NSLOT]));
                for &d in fence_domains(dom) {
                    for s in [d, NDOM + d] {
                        for c in lanes.lanes() {
                            let mut r = w.row[c as usize];
                            r[c as usize] = epoch;
                            for (dst, src) in br[s][c as usize].iter_mut().zip(r.iter()) {
                                *dst = (*dst).max(*src);
                            }
                        }
                    }
                }
                if lanes.is_full() {
                    let hb = w.base.hb.clone();
                    for &d in fence_domains(dom) {
                        w.base.g2a[d].join(&hb, memo);
                        w.base.a2g[d].join(&hb, memo);
                    }
                    for x in w.extra.iter_mut().flatten() {
                        let hb = x.hb.clone();
                        for &d in fence_domains(dom) {
                            x.g2a[d].join(&hb, memo);
                            x.a2g[d].join(&hb, memo);
                        }
                    }
                } else {
                    let base_hb = w.base.hb.clone();
                    for c in lanes.lanes() {
                        let x = w.extra[c as usize].get_or_insert_with(Default::default);
                        let mut hb = x.hb.clone();
                        hb.join(&base_hb, memo);
                        for &d in fence_domains(dom) {
                            x.g2a[d].join(&hb, memo);
                            x.a2g[d].join(&hb, memo);
                        }
                    }
                }
            }
            FenceKind::TcgenBefore => {
                // Publish (and order before the warp's own later tcgen work)
                // every issued pipelined op, every waited ld/st and the
                // fenced view.
                let memo = &self.memo;
                let w = &mut self.warps[warp as usize];
                let mut p = w.tcgen_issued.clone();
                p.join(&w.tcgen_waited, memo);
                p.join(&w.tcgen, memo);
                w.tcgen.join(&w.tcgen_issued, memo);
                w.base.tcgen_rel.join(&p, memo);
            }
            FenceKind::TcgenAfter => {
                let memo = &self.memo;
                let w = &mut self.warps[warp as usize];
                let i = w.tcgen_in.clone();
                w.tcgen.join(&i, memo);
            }
        }
    }

    /// Every write to the predicate's extra inputs happens before the wait
    /// (for every waiting lane), so those inputs were constant while the
    /// declared word's history was being judged.
    fn pred_reads_stable(&self, warp: WarpId, lanes: LaneMask, epoch: Epoch, reads: &[(AllocId, Range<u64>)]) -> bool {
        reads.iter().all(|(alloc, r)| {
            let Some(a) = self.allocs.get(alloc) else { return false };
            let mut ok = true;
            a.shadow.visit(r.clone(), |_, cell| {
                for p in cell.writes.as_slice() {
                    for lane in lanes.lanes() {
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
        let first = accepted
            .iter()
            .enumerate()
            .find(|(_, b)| **b != 0)
            .map(|(i, b)| i as u32 * 64 + b.trailing_zeros());
        let Some(mut idx) = first else {
            self.report.incomplete.push(Incomplete::WaitExitUnproven { warp });
            return;
        };
        if idx == 0 {
            return; // the launch value satisfied the predicate: no edge owed
        }
        let hist = &self.words[wi].history;
        let Some(e) = hist.get(idx as usize - 1) else {
            self.report.incomplete.push(Incomplete::WaitExitUnproven { warp });
            return;
        };
        if e.is_async {
            // Async publications land their bytes at completion; the run's
            // observed version is the only edge available (degraded,
            // schedule-dependent fallback).
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Frontier;

    #[test]
    fn required_scope_matrix() {
        let t = Topology { warps_per_cta: 4, ctas_per_cluster: 2, num_ctas: 4 };
        assert_eq!(required_scope(&t, 0, 3), Scope::Cta);
        assert_eq!(required_scope(&t, 0, 4), Scope::Cluster);
        assert_eq!(required_scope(&t, 0, 8), Scope::Gpu);
    }

    #[test]
    fn frontier_one_steady_state() {
        let mut f = Frontier::Empty;
        let w = |e| Witness {
            stamp: Stamp::new(0, e),
            lane: 0,
            proxy: Proxy::Generic,
            domain: None,
            kind: AccessKind::Read,
            scope: None,
            atomic: false,
            span: (0, 4),
            site: 0,
            warp: 0,
        };
        f.record(Entry { w: w(1), rel: None, base: None }, |_| true);
        f.record(Entry { w: w(2), rel: None, base: None }, |_| true);
        assert!(matches!(f, Frontier::One(_)));
        f.record(Entry { w: w(3), rel: None, base: None }, |_| false);
        assert_eq!(f.as_slice().len(), 2);
    }
}
