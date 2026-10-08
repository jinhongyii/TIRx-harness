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
use super::knowledge::{fence_domains, join_tmap, select_view, Heads, Knowledge, Rel, View, NDOM};
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
    /// A plain (weak) access racing on a declared `wait_until` word: it
    /// bypasses the signalling protocol (legacy `signal_protocol_error`).
    SignalProtocolError { class: RaceClass, failure: OrderingFailure },
    /// Same evidence as a data race whose prior is an unwaited tcgen05.ld.
    TmemLifetimeReview { class: RaceClass, failure: OrderingFailure },
    /// A release/acquire pair whose scopes do not mutually cover.
    ScopeMismatch {
        release_scope: Scope,
        acquire_scope: Scope,
        release_warp: WarpId,
        acquire_warp: WarpId,
        release_site: SiteId,
        acquire_site: SiteId,
    },
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
    /// A read through one logical buffer name observes bytes last written
    /// through another name of the same pooled allocation (legacy
    /// `alias_stale_read`; ordered, so not a race, but likely a stale name).
    AliasStaleRead,
    /// A raw (non-`wait_until`) strong read of a DECLARED protocol word
    /// raced its publication (morally strong, so not a race, §8.7.1): the
    /// protocol says to read the word through `wait_until` (deltas T18).
    DeclaredWordRawRead,
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
    /// For TMEM: the union of overlapped `(lanes, columns)` over all
    /// occurrences (the byte hull spans rows, so it cannot be projected to
    /// columns; TMEM byte = (lane * 512 + column) * 4).
    /// `(lanes, columns)`: sorted, disjoint, merged only when overlapping or
    /// adjacent, over every occurrence's exact overlap.
    pub tmem: Option<(Vec<Range<u64>>, Vec<Range<u64>>)>,
    /// `AliasStaleRead` only: the merged byte spans of every occurrence
    /// (legacy `overlaps`); empty for other kinds.
    pub spans: Vec<Range<u64>>,
    /// `AliasStaleRead` only: (reader, writer) logical names (W5-15).
    pub names: Option<(Arc<str>, Arc<str>)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
    /// A multi-lane per-thread async op's access did not name its lane.
    AsyncLaneUnknown { op: AsyncId },
    /// A `wait_until` exit is explained only by a write that does not exactly
    /// cover the declared word (wider, narrower or misaligned).
    SignalWriteNotRecorded { warp: WarpId },
    /// An `AsyncComplete` named a warp outside the launch.
    CompletionWarpOutOfRange { warp: WarpId },
    /// A sync event's `kernel` differs from the launch being checked.
    KernelMismatch { expected: u32, got: u32 },
    /// `max_findings` reached; later findings were not recorded.
    FindingsTruncated { dropped: u64 },
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub findings: Vec<Finding>,
    /// Distinct incomplete reasons, first occurrence order.
    pub incomplete: Vec<Incomplete>,
    /// Occurrences of each `incomplete[i]`.
    pub incomplete_counts: Vec<u64>,
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
    /// Distinct spans in the wide-span side table.
    pub wide_spans: u64,
}

// ----------------------------------------------------------------- state --

const MAX_PENDING: usize = 1024;
/// Bridge-row slots: g2a[0..NDOM], a2g[NDOM..2NDOM].
const NSLOT: usize = 2 * NDOM;

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
    /// Tensormap ranges each lane acquired (`fence.proxy.tensormap::generic
    /// .acquire`), with the release knowledge that reached it.
    g2t_ranges: Vec<Arc<Vec<(AllocId, Range<u64>, Clock)>>>,
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
            g2t_ranges: vec![Arc::new(Vec::new()); 32],
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

    /// Park a relaxed observation for a later acquire fence. Deduplicated by
    /// payload and bounded: dropping the oldest only loses edges (more races
    /// reported, never fewer).
    fn push_pending(&mut self, lane: u8, rel: Arc<Rel>) {
        if self.pending_acq.iter().any(|(l, r)| *l == lane && Arc::ptr_eq(r, &rel)) {
            return;
        }
        if self.pending_acq.len() >= MAX_PENDING {
            self.pending_acq.drain(..MAX_PENDING / 2);
        }
        self.pending_acq.push((lane, rel));
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
            join_tmap(&mut x.tmap_rel, &k.tmap_rel, memo);
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
    /// Epoch base of the slot's current generation (sides are base+1/+2).
    gen_base: Epoch,
    in_use: bool,
    warp: WarpId,
    lane: u8,
    issue_epoch: Epoch,
    site: SiteId,
    kind: AsyncKind,
    k: Knowledge,
    /// Tensormap ranges the issuing lanes acquired (shared with the lane).
    g2t_ranges: Arc<Vec<(AllocId, Range<u64>, Clock)>>,
    /// CTAs that observed this op's completion directly (its mbarrier's
    /// CTA, or the waiting warp's CTA).
    completed_ctas: Vec<u32>,
    /// `(slot, generation)` of each predecessor (a reclaimed slot's new
    /// generation is a different op).
    preds: Vec<(usize, Epoch)>,
    footprint: Vec<(AllocId, Range<u64>)>,
    /// Highest milestone reached (0 none, 1 read, 2 write/full).
    done: u8,
    /// No further completion is expected: a bulk copy still in flight when
    /// its CTA's shared memory ended (implicit CTA exit, deltas S7), or an
    /// mbarrier-less st.async / red.async `.release` that landed at issue
    /// (T13). Its accesses stay unordered with everything after them.
    drained: bool,
}

struct Alloc {
    size: u64,
    space: Space,
    /// Owning CTA (shared / TMEM reach: its cluster).
    cta: u32,
    shadow: IntervalShadow<Cell>,
    /// Proxies that have accessed the allocation (bit = proxy code).
    seen: u8,
    /// Generic witnesses that GC retired while some proxy had not yet
    /// accessed the allocation, summarised per `(actor, lane, write, window,
    /// 4 KiB page)`: the latest epoch and the byte hull. Every non-generic
    /// access is checked against them, so GC never hides a cross-proxy race
    /// (legacy RS `RetiredGenericHistory`, made view-aware; legacy G's
    /// proxy-blind floor dropped them).
    /// Keyed `(start page, actor, lane, write, window)`, so a range query by
    /// page finds the candidates (review R4).
    retired: std::collections::BTreeMap<(u64, ActorId, u8, bool, u8), RetiredGeneric>,
    /// Widest retired hull, in pages (bounds the backward page scan).
    retired_span_pages: u64,
}

#[derive(Clone, Debug)]
struct RetiredGeneric {
    /// Latest witness, packed once with the hull (re-packed only when the
    /// hull grows).
    w: Witness,
    lo: u64,
    hi: u64,
    info: WitnessInfo,
}

fn proxy_bit(p: Proxy) -> u8 {
    1 << match p {
        Proxy::Generic => 0,
        Proxy::Async => 1,
        Proxy::TensorMap => 2,
        Proxy::ReadOnly => 3,
        Proxy::Tcgen => 4,
    }
}

fn domain_code(d: Option<Domain>) -> u8 {
    match d {
        None => 0,
        Some(Domain::Global) => 1,
        Some(Domain::SharedCta) => 2,
        Some(Domain::SharedCluster) => 3,
    }
}

struct Arrival {
    /// Representative arriving warp (the group's first).
    warp: WarpId,
    scope: Option<Scope>,
    site: SiteId,
    k: Arc<Knowledge>,
}

#[derive(Default)]
struct Phase {
    /// Release arrivals grouped by `(arriver CTA, scope)`. Mutual scope
    /// inclusion depends on the arriver only through its CTA / cluster, so a
    /// group is judged once and joined once per waiter (O(groups)).
    arrivals: Vec<Arrival>,
    /// tcgen05 fence frontier carried by every arrive, relaxed included.
    tcgen_rel: Clock,
    /// Async completions (complete-tx: release at cluster scope for the
    /// op's own bytes; accepted by an acquire wait of any scope).
    completion: Knowledge,
}

struct PollStash {
    lane: u8,
    alloc: AllocId,
    range: Range<u64>,
    heads: Vec<Heads>,
    order: MemOrder,
    scope: Scope,
    site: SiteId,
}

struct HistEntry {
    rel: Option<Heads>,
    is_async: bool,
    /// The write's span is not exactly the word: a mixed-size access is
    /// not single-copy atomic with the word's polls (PTX §8.7.2), so a wait
    /// accepting it is `incomplete` (legacy `signal_write_not_recorded`).
    mixed_size: bool,
    /// GC found the payload dominated by every live actor and dropped it.
    consumed: bool,
}

/// A declared word. History numbering follows the contract (README
/// decision 14): bit 0 = launch value, bit i = the i-th `(Access, lane)`
/// write overlapping the word, in delivery order, lanes ascending — kept for
/// every overlapping word, not only the first.
struct Word {
    range: Range<u64>,
    history: Vec<HistEntry>,
    /// Last `(access seq, lane)` appended (one entry per lane per Access).
    last: Option<(u64, u8)>,
    /// Per (warp, lane): history index (1-based) of its own latest write.
    /// Coherence (CoWR): a later read of that thread cannot read from an
    /// earlier entry, so a wait's accepted entry is at least this one.
    own: HashMap<(WarpId, u8), u32>,
}

/// One segment of the alias tracker: `[start, end)` last written through
/// logical name `buf` by `w` (warp `warp`, site `site`).
#[derive(Clone)]
struct AliasSeg {
    end: u64,
    buf: Arc<str>,
    warp: WarpId,
    site: SiteId,
    w: Witness,
}

#[derive(Clone, Copy)]
/// Who performs the access being checked.
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
    allocs: HashMap<AllocId, Alloc>,
    /// Per sync object, its most recent phases (older ones can no longer be
    /// waited on: mbarrier parity / barrier generations).
    phases: HashMap<SyncObjId, std::collections::BTreeMap<u64, Phase>>,
    incomplete_index: HashMap<Incomplete, usize>,
    scope_dedup: HashMap<(SiteId, SiteId, Scope, Scope), usize>,
    /// Latest `fence.sc` per `(warp, lane, scope)`.
    sc: HashMap<(WarpId, u8, Scope), Arc<Knowledge>>,
    words: HashMap<AllocId, Vec<Word>>,
    /// Read-froms of possible `wait_until` polls, per warp, held back until
    /// the warp's next event.
    poll_stash: HashMap<WarpId, Vec<PollStash>>,
    advisory_dedup: HashMap<(AdvisoryKind, AllocId, SiteId), usize>,
    /// `alias_stale_read` (legacy alias tracker): per allocation, the last
    /// named warp-lane writer of every byte, keyed by segment start.
    alias_writers: HashMap<AllocId, std::collections::BTreeMap<u64, AliasSeg>>,
    /// The tensormap view of the current warp-lane `Proxy::TensorMap`
    /// access (set per access).
    lane_g2t: Option<Clock>,
    /// One advisory per (alloc, reader name, writer name, reader warp/site,
    /// writer warp/site), as legacy keyed them.
    alias_dedup: HashMap<(AllocId, Arc<str>, Arc<str>, WarpId, SiteId, WarpId, SiteId), usize>,
    /// Logical buffer name per site (`SiteInfo::buffer`), for
    /// `AliasStaleRead`. Empty = advisory off.
    pub site_buffer: HashMap<SiteId, Arc<str>>,
    /// W5-15: logical buffer of each (site, pointer operand).
    pub operand_buffer: HashMap<(SiteId, u8), Arc<str>>,
    /// Declared space of each site's named buffer (see `alias_access`).
    pub site_buffer_space: HashMap<SiteId, Space>,
    /// Sites of `wait_until` polls (lowering's `tirx.cuda.wait_until`).
    pub poll_sites: HashSet<SiteId>,
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
    /// Accesses between collections: `gc_every`, raised to twice the live
    /// shadow cells after each collection so a pass (linear in the cells)
    /// stays amortised O(1) per access while memory stays within 2x live.
    gc_period: u64,
    pub stats: Stats,
}

/// Insert `r` into sorted, disjoint, non-adjacent `spans` (bounded: past
/// 1024 spans the last one absorbs the rest).
fn add_span(spans: &mut Vec<Range<u64>>, r: Range<u64>) {
    if r.is_empty() {
        return;
    }
    let i = spans.partition_point(|s| s.end < r.start);
    let mut j = i;
    let (mut lo, mut hi) = (r.start, r.end);
    while j < spans.len() && spans[j].start <= hi {
        lo = lo.min(spans[j].start);
        hi = hi.max(spans[j].end);
        j += 1;
    }
    spans.splice(i..j, std::iter::once(lo..hi));
    if spans.len() > 1024 {
        let last = spans.pop().unwrap();
        let l = spans.last_mut().unwrap();
        l.end = l.end.max(last.end);
    }
}

/// Fold the TMEM byte range `r` into `(lanes, columns)` (exact per-lane
/// column segments of the taddr-encoded range).
fn add_tmem(t: &mut (Vec<Range<u64>>, Vec<Range<u64>>), r: &Range<u64>) {
    let row = 512 * 4;
    let (l0, l1) = (r.start / row, (r.end - 1) / row);
    add_span(&mut t.0, l0..l1 + 1);
    let (c0, c1) = ((r.start % row) / 4, ((r.end - 1) % row) / 4 + 1);
    if l0 == l1 {
        add_span(&mut t.1, c0..c1);
    } else if l1 > l0 + 1 || c0 <= c1 {
        add_span(&mut t.1, 0..512);
    } else {
        add_span(&mut t.1, c0..512);
        add_span(&mut t.1, 0..c1);
    }
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
        Space::Global => &[Proxy::Generic, Proxy::Async, Proxy::TensorMap],
        _ => &[Proxy::Generic, Proxy::Async],
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
            allocs: HashMap::new(),
            phases: HashMap::new(),
            incomplete_index: HashMap::new(),
            scope_dedup: HashMap::new(),
            sc: HashMap::new(),
            words: HashMap::new(),
            poll_stash: HashMap::new(),
            advisory_dedup: HashMap::new(),
            alias_writers: HashMap::new(),
            lane_g2t: None,
            alias_dedup: HashMap::new(),
            site_buffer: HashMap::new(),
            poll_sites: HashSet::new(),
            site_buffer_space: HashMap::new(),
            operand_buffer: HashMap::new(),
            wide: WideSpans::default(),
            report: Report::default(),
            dedup: HashMap::new(),
            gc_every: 1 << 14,
            max_findings: 0,
            mbarrier_scope_assumed: false,
            dropped_findings: 0,
            since_gc: 0,
            gc_period: 0,
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

    /// Record an incomplete reason once (adapter use).
    pub fn note_incomplete(&mut self, i: Incomplete) {
        match self.incomplete_index.get(&i) {
            Some(&k) => self.report.incomplete_counts[k] += 1,
            None => {
                self.incomplete_index.insert(i.clone(), self.report.incomplete.len());
                self.report.incomplete.push(i);
                self.report.incomplete_counts.push(1);
            }
        }
    }

    pub fn wide_span_count(&self) -> u64 {
        self.wide.spans.len() as u64
    }

    /// Findings so far (the run is not finalised).
    pub fn report(&self) -> &Report {
        &self.report
    }

    /// Finalise: outstanding async work is incomplete.
    pub fn finalize(&mut self) {
        let warps: Vec<WarpId> = self.poll_stash.keys().copied().collect();
        for w in warps {
            self.flush_polls(w);
        }
        let never: Vec<AsyncId> = self
            .asyncs
            .iter()
            // tcgen05.ld / st are observer-only ops whose every access is
            // delivered at issue: an op never waited (`tcgen05.wait::ld/st`)
            // leaves no unobserved effect. Its accesses stay unordered with
            // everything after them (races / TmemLifetimeReview still fire);
            // the destination-register side belongs to Space::Reg.
            .filter(|a| a.in_use && a.done == 0 && !a.drained && !matches!(a.kind, AsyncKind::TcgenCommit | AsyncKind::TcgenLd | AsyncKind::TcgenSt))
            .map(|a| a.op)
            .collect();
        for op in never {
            self.note_incomplete(Incomplete::AsyncNeverCompleted { op });
        }
        if self.dropped_findings > 0 {
            self.note_incomplete(Incomplete::FindingsTruncated { dropped: self.dropped_findings });
            self.dropped_findings = 0;
        }
    }

    pub fn finish(mut self) -> Report {
        self.finalize();
        self.stats.wide_spans = self.wide.spans.len() as u64;
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
        let Some(cur) = self.warps.get(w as usize).map(|x| x.epoch) else {
            self.note_incomplete(Incomplete::CompletionWarpOutOfRange { warp: w });
            return false;
        };
        if epoch < cur {
            self.note_incomplete(Incomplete::EpochRegression { warp: w });
            return false;
        }
        if epoch >= u32::MAX - 1 {
            self.note_incomplete(Incomplete::EpochOverflow { warp: w });
            return false;
        }
        self.warps[w as usize].epoch = epoch;
        true
    }

    fn async_idx(&mut self, op: AsyncId) -> Option<usize> {
        let r = self.async_index.get(&op).copied();
        if r.is_none() {
            self.note_incomplete(Incomplete::UnknownAsyncOp { op });
        }
        r
    }

    fn is_tmem(&self, alloc: AllocId) -> bool {
        self.allocs.get(&alloc).is_some_and(|al| al.space == Space::Tmem)
    }

    /// Fold one more occurrence over `bytes` into finding `i`.
    fn bump(&mut self, i: usize, bytes: &Range<u64>) {
        let tmem = self.is_tmem(self.report.findings[i].alloc);
        let f = &mut self.report.findings[i];
        f.occurrences += 1;
        if tmem && !bytes.is_empty() {
            add_tmem(f.tmem.get_or_insert_with(Default::default), bytes);
        }
        if matches!(f.kind, FindingKind::Advisory { kind: AdvisoryKind::AliasStaleRead }) {
            add_span(&mut f.spans, bytes.clone());
        }
    }

    fn push_finding(&mut self, mut f: Finding) -> Option<usize> {
        if self.max_findings != 0 && self.report.findings.len() >= self.max_findings {
            self.dropped_findings += 1;
            return None;
        }
        if matches!(f.kind, FindingKind::Advisory { kind: AdvisoryKind::AliasStaleRead }) && f.spans.is_empty() {
            f.spans.push(f.bytes.clone());
        }
        if f.tmem.is_none() && !f.bytes.is_empty() && self.is_tmem(f.alloc) {
            let mut t = Default::default();
            add_tmem(&mut t, &f.bytes);
            f.tmem = Some(t);
        }
        self.report.findings.push(f);
        Some(self.report.findings.len() - 1)
    }

    /// A natural pause (inbox drain): collect if a quarter period elapsed.
    pub fn safe_point(&mut self) {
        if self.gc_every != 0 && self.since_gc >= self.gc_every.max(self.gc_period) / 4 {
            self.gc();
        }
    }

    fn maybe_gc(&mut self) {
        self.since_gc += 1;
        if self.gc_every != 0 && self.since_gc >= self.gc_every.max(self.gc_period) {
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
        let mut view = select_view(prior.proxy(), cur_proxy, prior.domain());
        // W2-19 (deltas T14): a warp-lane `Proxy::Tcgen` witness is a
        // buffer-form TMEM access, synchronous in its thread (completed at
        // the instruction, like a waited tcgen05.ld/st): hb orders it, not
        // the tcgen pipeline view.
        if view == View::Tcgen && prior.stamp.actor() < self.topo.num_warps() {
            view = View::Hb;
        }
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
                if view == View::G2t {
                    // A lane's descriptor read: only its acquired ranges.
                    return self.lane_g2t.as_ref().is_some_and(|v| v.observes(prior.stamp, prior.lane()));
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
            && self.slot_of(prior.stamp.actor()).is_some_and(|p| {
                // Ordered through the prior op's own completion, observed in
                // the current issuer's CTA (the multicast / 2-CTA consumer
                // pattern), is not "base causality alone" (review F1).
                let cur_cta = self.topo.cta_of(self.asyncs[a].warp);
                self.topo.cta_of(p.warp) != cur_cta && !p.completed_ctas.contains(&cur_cta)
            })
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
        if let Some(&i) = self.advisory_dedup.get(&(kind, alloc, site)) {
            self.bump(i, &bytes);
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
            tmem: None,
            spans: Vec::new(),
            names: None,        };
        if let Some(i) = self.push_finding(f) {
            self.advisory_dedup.insert((kind, alloc, site), i);
        }
    }

    /// Legacy `alias_stale_read` (engine-rs `AliasTracker`): a warp-lane
    /// read through logical name R of bytes whose last named warp-lane
    /// writer used another name W is a review advisory over exactly those
    /// bytes (adjacent bytes of one writer coalesce). Unnamed accesses
    /// neither report nor overwrite; async copies carry no name. No
    /// ordering requirement: the advisory is about logical identity.
    fn alias_access(&mut self, alloc: AllocId, r: Range<u64>, site: SiteId, operand: u8, warp: WarpId, cw: &Witness) {
        // W5-15: the access's own operand names it when lowering provides
        // per-operand buffers; otherwise the site's single name, guarded by
        // the space rule below (deltas P7 interim).
        let exact = self.operand_buffer.get(&(site, operand)).cloned();
        if exact.is_none() && operand != 0 {
            return;
        }
        let Some(buf) = exact.clone().or_else(|| self.site_buffer.get(&site).cloned()).filter(|b| !b.is_empty()) else {
            return;
        };
        // A site names one operand's buffer; an access in another space
        // (e.g. tensormap.cp_fenceproxy's shared-memory source, named after
        // its global destination) has no known logical name (deltas P7).
        let space = self.allocs.get(&alloc).map(|a| a.space);
        if exact.is_none() && self.site_buffer_space.get(&site).is_some_and(|s| Some(*s) != space) {
            return;
        }
        if cw.kind() != AccessKind::Write {
            let mut hits: Vec<(Range<u64>, AliasSeg)> = Vec::new();
            if let Some(segs) = self.alias_writers.get(&alloc) {
                let first = segs.range(..r.start).next_back().filter(|(_, s)| s.end > r.start).map(|(k, _)| *k).unwrap_or(r.start);
                for (&st, seg) in segs.range(first..r.end) {
                    if seg.buf == buf {
                        continue;
                    }
                    let o = st.max(r.start)..seg.end.min(r.end);
                    match hits.last_mut() {
                        Some((h, prev)) if h.end == o.start && prev.w == seg.w && prev.buf == seg.buf => h.end = o.end,
                        _ => hits.push((o, seg.clone())),
                    }
                }
            }
            for (o, seg) in hits {
                let key = (alloc, buf.clone(), seg.buf.clone(), warp, site, seg.warp, seg.site);
                if let Some(&i) = self.alias_dedup.get(&key) {
                    self.bump(i, &o);
                    continue;
                }
                let mut prior = self.info(&seg.w);
                prior.site = seg.site;
                let f = Finding {
                    kind: FindingKind::Advisory { kind: AdvisoryKind::AliasStaleRead },
                    severity: Severity::Review,
                    alloc,
                    bytes: o,
                    prior: Some(prior),
                    current: Some(self.info(cw)),
                    occurrences: 1,
                    tmem: None,
                    spans: Vec::new(),
                    names: Some((buf.clone(), seg.buf.clone())),
                };
                if let Some(i) = self.push_finding(f) {
                    self.alias_dedup.insert(key, i);
                }
            }
        }
        if cw.writes() {
            let segs = self.alias_writers.entry(alloc).or_default();
            // Split the segment straddling r.start, drop covered ones, keep
            // the tail of one straddling r.end.
            if let Some((&st, seg)) = segs.range(..r.start).next_back() {
                if seg.end > r.start {
                    let tail = seg.clone();
                    segs.get_mut(&st).unwrap().end = r.start;
                    if tail.end > r.end {
                        segs.insert(r.end, tail);
                    }
                }
            }
            let inside: Vec<u64> = segs.range(r.start..r.end).map(|(k, _)| *k).collect();
            for k in inside {
                let seg = segs.remove(&k).unwrap();
                if seg.end > r.end {
                    segs.insert(r.end, seg);
                }
            }
            segs.insert(r.start, AliasSeg { end: r.end, buf, warp, site, w: *cw });
        }
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
            self.bump(i, &bytes);
            return;
        }
        // A plain access racing on a declared word bypasses the wait_until
        // protocol: legacy `signal_protocol_error`.
        let on_word = self
            .words
            .get(&alloc)
            .is_some_and(|ws| ws.iter().any(|w| w.range.start < bytes.end && bytes.start < w.range.end));
        let kind = if review {
            FindingKind::TmemLifetimeReview { class, failure }
        } else if on_word && (prior.scope().is_none() || cw.scope().is_none()) {
            FindingKind::SignalProtocolError { class, failure }
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
            tmem: None,
            spans: Vec::new(),
            names: None,        };
        if let Some(i) = self.push_finding(f) {
            self.dedup.insert(key, i);
        }
    }

    // --------------------------------------------------------- access --

    /// Apply the held-back poll read-froms of `warp` (no WaitVerdicts came).
    fn flush_polls(&mut self, warp: WarpId) {
        if self.poll_stash.is_empty() {
            return;
        }
        if let Some(v) = self.poll_stash.remove(&warp) {
            for p in v {
                self.apply_read_from(warp as usize, p.lane, AccessKind::Read, false, p.order, Some(p.scope), p.site, p.heads);
            }
        }
    }

    pub fn access(&mut self, a: &Access) {
        if let Who::Lane { warp, .. } = a.who {
            self.flush_polls(warp);
        }
        self.stats.accesses += 1;
        self.maybe_gc();
        let Some(size) = self.allocs.get(&a.alloc).map(|x| x.size) else {
            self.note_incomplete(Incomplete::UnknownAlloc { alloc: a.alloc });
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
        if a.range.start == a.range.end {
            return; // e.g. cp.async zfill with src-size 0: no bytes (review R7)
        }
        // A `mapa` to the accessor's own rank is the CTA's own window: the
        // hardware encoding gives `mapa(p, own rank) == p` (PTX §9.7.9.20),
        // so it is the shared::cta window, not shared::cluster.
        let accessor_cta = match cur {
            Cur::Lane { w, .. } => self.topo.cta_of(w as u32),
            Cur::Async { a: i } => self.topo.cta_of(self.asyncs[i].warp),
        };
        let domain = match (a.domain, self.allocs.get(&a.alloc)) {
            (Some(Domain::SharedCluster), Some(al)) if al.space == Space::Shared && al.cta == accessor_cta => Some(Domain::SharedCta),
            (d, _) => d,
        };
        if a.range.end > size || a.range.start > a.range.end {
            let current = matches!(cur, Cur::Lane { .. }).then(|| {
                let w = Witness::pack(stamp, lane, a.proxy, domain, a.kind, a.scope, a.atomic, (a.range.start, a.range.end), &mut self.wide);
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
                tmem: None,
                spans: Vec::new(),
            names: None,            };
            self.push_finding(f);
            return;
        }
        self.lane_g2t = None;
        if a.proxy == Proxy::TensorMap {
            // The acquired tensormap ranges of the reader: an async op's
            // (inherited at issue) or, for a descriptor read the engine
            // emits at TMA issue as a warp-lane access, the lane's own.
            let ranges = match cur {
                Cur::Async { a: i } => self.asyncs[i].g2t_ranges.clone(),
                Cur::Lane { w, lane, .. } => self.warps[w].g2t_ranges[lane as usize].clone(),
            };
            // Split at acquired-range boundaries so each piece has one view.
            let mut cuts: Vec<u64> = ranges
                .iter()
                .filter(|(al, _, _)| *al == a.alloc)
                .flat_map(|(_, r, _)| [r.start, r.end])
                .filter(|x| a.range.start < *x && *x < a.range.end)
                .collect();
            if !cuts.is_empty() {
                cuts.sort_unstable();
                cuts.dedup();
                let mut lo = a.range.start;
                for hi in cuts.into_iter().chain([a.range.end]) {
                    let mut piece = a.clone();
                    piece.range = lo..hi;
                    self.stats.accesses -= 1; // counted once per access
                    self.access(&piece);
                    lo = hi;
                }
                return;
            }
            // The tensormap view for exactly these descriptor bytes.
            let mut v = Clock::default();
            for (al, r, k) in ranges.iter() {
                if *al == a.alloc && r.start < a.range.end && a.range.start < r.end {
                    v.join(k, &self.memo);
                }
            }
            match cur {
                Cur::Async { a: i } => self.asyncs[i].k.g2t = v,
                Cur::Lane { .. } => self.lane_g2t = Some(v),
            }
        }
        let w = Witness::pack(stamp, lane, a.proxy, domain, a.kind, a.scope, a.atomic, (a.range.start, a.range.end), &mut self.wide);
        let writes = w.writes();
        let strong = a.scope.is_some();

        // Own release head for strong writes.
        let own_rel: Option<Arc<Rel>> = match (cur, writes) {
            (Cur::Lane { w: wi, lane, epoch }, true) if strong => {
                let warp = &self.warps[wi];
                if matches!(a.order, MemOrder::Release | MemOrder::AcqRel) {
                    let k = warp.publication(one_lane(lane), epoch, &self.memo);
                    Some(Arc::new(Rel { k, scope: a.scope, warp: wi as u32, site: a.site }))
                } else if let Some(f) = &warp.fence_rel[lane as usize] {
                    Some(f.clone())
                } else {
                    let k = Knowledge { tcgen_rel: warp.tcgen_publication(one_lane(lane), &self.memo), ..Default::default() };
                    Some(Arc::new(Rel { k, scope: None, warp: wi as u32, site: a.site }))
                }
            }
            // st.async / red.async `.release` (PTX §9.7.10.12, §9.7.15.7: a
            // strong release at `.scope`, performed in the generic proxy):
            // releases what the issuing thread knew at issue.
            (Cur::Async { a: i }, true) if strong && matches!(a.order, MemOrder::Release | MemOrder::AcqRel) => {
                // W2-20 (1): the mbarrier-less form has no completion event;
                // the write lands at issue (generic proxy), so the op is
                // complete there: neither never-completed nor a lifetime
                // finding (deltas T13).
                if a.proxy == Proxy::Generic {
                    self.asyncs[i].drained = true;
                }
                let act = &self.asyncs[i];
                Some(Arc::new(Rel { k: act.k.propagating(), scope: a.scope, warp: act.warp, site: act.site }))
            }
            _ => None,
        };

        let mut races: Vec<(Range<u64>, Witness)> = Vec::new();
        if let Some(al) = self.allocs.get_mut(&a.alloc) {
            al.seen |= proxy_bit(a.proxy);
        }
        if a.proxy != Proxy::Generic {
            self.check_retired_generic(a, cur, &w);
        }
        let mut advisories: Vec<(Range<u64>, Witness, AdvisoryKind)> = Vec::new();
        let mut acquired: Vec<Heads> = Vec::new();
        // Every declared word this access overlaps, with the byte whose
        // segment decides the word's entry.
        let word_points: Vec<(usize, u64, bool)> = self
            .words
            .get(&a.alloc)
            .map(|ws| {
                ws.iter()
                    .enumerate()
                    .filter(|(_, x)| x.range.start < a.range.end && a.range.start < x.range.end)
                    .map(|(i, x)| (i, x.range.start.max(a.range.start), x.range == a.range))
                    .collect()
            })
            .unwrap_or_default();
        let in_word = !word_points.is_empty();
        let alias_space = !(self.site_buffer.is_empty() && self.operand_buffer.is_empty())
            && self.allocs.get(&a.alloc).is_some_and(|al| matches!(al.space, Space::Shared | Space::Tmem));
        let mut word_rels: Vec<(usize, Option<Heads>, bool)> = Vec::new();

        let mut shadow = std::mem::take(&mut self.allocs.get_mut(&a.alloc).unwrap().shadow);
        {
            let this = &*self;
            let wide = &self.wide;
            shadow.update(a.range.clone(), |seg, cell| {
                // 1. check
                let mut check = |p: &Witness| {
                    // Sibling lanes of ONE warp instruction storing to the
                    // same bytes: CUDA defines the outcome (the writes are
                    // serialised, one of them is the final value), so this is
                    // not reported as a race (deltas V11). Reads are not
                    // involved: one instruction is one access kind.
                    if writes && p.writes() && p.stamp == w.stamp && p.lane() != w.lane() && matches!(cur, Cur::Lane { .. }) {
                        return;
                    }
                    let ordered = this.ordered(cur, p, a.proxy);
                    let ms = this.morally_strong(p, &w);
                    if !ordered && !ms {
                        races.push((overlap(p.span(wide), &seg), *p));
                    } else if ordered && this.cross_cta_async(p, cur, a.proxy) {
                        advisories.push((overlap(p.span(wide), &seg), *p, AdvisoryKind::CrossCtaAsyncOrder));
                    } else if !ordered && ms && in_word && !writes && p.writes() {
                        // T18: a strong read of a declared word that is not
                        // a `wait_until` poll observed an unordered write of
                        // it. Only this order: a spin's early reads (before
                        // the publication) are the loop working.
                        if !this.poll_sites.contains(&a.site) {
                            advisories.push((overlap(p.span(wide), &seg), *p, AdvisoryKind::DeclaredWordRawRead));
                        }
                    } else if !ordered && ms && !in_word && !writes && p.writes() {
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
                // 2. read-from (strong reads and atomics). A pure read of a
                // declared word is a `wait_until` poll: its edge comes only
                // from the WaitVerdicts earliest accepted write (W1), never
                // from the run's latest write.
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
                    for (i, pt, exact) in &word_points {
                        if seg.start <= *pt && *pt < seg.end {
                            word_rels.push((*i, rel.clone(), !*exact));
                        }
                    }
                    cell.writes.record(Entry { w, rel, base }, wide, |p| this.ordered(cur, p, a.proxy));
                    if !strong {
                        // A plain write supersedes the readers it is ordered
                        // after; a strong write keeps them (a later access
                        // morally strong with it may still race them).
                        // Only readers in the same proxy and window: a reader
                        // judged by a different bridge is not covered.
                        cell.reads.retain(|r| !(w.same_view_class(&r.w) && this.ordered(cur, &r.w, a.proxy)));
                    }
                    // An un-waited tcgen05.ld read is reported once, against
                    // the first write that overwrites it (the review names
                    // the hazard; legacy shadow semantics, deltas T17).
                    cell.reads.retain(|r| !this.slot_of(r.w.stamp.actor()).is_some_and(|x| x.kind == AsyncKind::TcgenLd));
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
        if alias_space {
            if let Cur::Lane { w: wi, .. } = cur {
                self.alias_access(a.alloc, a.range.clone(), a.site, a.operand, wi as WarpId, &w);
            }
        }
        if !word_rels.is_empty() {
            let is_async = matches!(cur, Cur::Async { .. });
            let ws = self.words.get_mut(&a.alloc).unwrap();
            for (i, rel, mixed_size) in word_rels {
                let word = &mut ws[i];
                if word.last != Some((a.seq, lane)) {
                    word.last = Some((a.seq, lane));
                    word.history.push(HistEntry { rel, is_async, consumed: false, mixed_size });
                    if let Cur::Lane { w: wi, lane, .. } = cur {
                        word.own.insert((wi as WarpId, lane), word.history.len() as u32);
                    }
                }
            }
        }
        if let Cur::Lane { w: wi, lane, .. } = cur {
            // A pure strong read of a declared word may be a `wait_until`
            // poll. Its read-from edge (the run's latest write) is held back:
            // a following WaitVerdicts for this lane and word replaces it
            // with the earliest-accepted edge (W1); any other event of the
            // warp applies it as an ordinary read-from.
            if in_word && !writes && !acquired.is_empty() {
                let order = a.order;
                let scope = a.scope.unwrap();
                self.poll_stash.entry(wi as u32).or_default().push(PollStash {
                    lane,
                    alloc: a.alloc,
                    range: a.range.clone(),
                    heads: acquired,
                    order,
                    scope,
                    site: a.site,
                });
                return;
            }
            self.apply_read_from(wi, lane, a.kind, a.returns_value, a.order, a.scope, a.site, acquired);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_read_from(&mut self, wi: usize, lane: u8, kind: AccessKind, returns_value: bool, order: MemOrder, scope: Option<Scope>, site: SiteId, acquired: Vec<Heads>) {
        {
            // `red` never forms an acquire pattern (PTX §8.8).
            let can_acquire = kind != AccessKind::Rmw || returns_value;
            for heads in acquired {
                for rel in heads.iter() {
                    if !can_acquire {
                        continue; // `red` observes nothing, tcgen included
                    }
                    self.warps[wi].tcgen_in[lane as usize].join(&rel.k.tcgen_rel, &self.memo);
                    match order {
                        MemOrder::Acquire | MemOrder::AcqRel => self.acquire_rel(wi as u32, one_lane(lane), scope.unwrap(), rel, site),
                        MemOrder::Relaxed | MemOrder::Release => self.warps[wi].push_pending(lane, rel.clone()),
                        MemOrder::Weak => {}
                    }
                }
            }
        }
    }

    /// Check a non-generic access against the generic witnesses GC folded
    /// into the allocation's summary.
    fn check_retired_generic(&mut self, a: &Access, cur: Cur, cw: &Witness) {
        let Some(alloc) = self.allocs.get(&a.alloc) else { return };
        if alloc.retired.is_empty() {
            return;
        }
        let lo_page = (a.range.start >> 12).saturating_sub(alloc.retired_span_pages);
        let hi_page = (a.range.end - 1) >> 12;
        let hits: Vec<(Witness, Range<u64>, WitnessInfo)> = alloc
            .retired
            .range((lo_page, 0, 0, false, 0)..=(hi_page, u32::MAX, u8::MAX, true, u8::MAX))
            .map(|(_, e)| e)
            .filter(|e| e.lo < a.range.end && a.range.start < e.hi && (e.w.writes() || cw.writes()))
            .filter(|e| !self.ordered(cur, &e.w, a.proxy))
            .map(|e| (e.w, e.lo.max(a.range.start)..e.hi.min(a.range.end), e.info.clone()))
            .collect();
        for (pw, bytes, info) in hits {
            self.report_race_with(a.alloc, bytes, cur, &pw, cw, Some(info));
        }
    }

    /// Acquire a release payload, checking that the scopes mutually cover.
    fn acquire_rel(&mut self, me: WarpId, lanes: LaneMask, my_scope: Scope, rel: &Rel, acq_site: SiteId) {
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
            self.report_scope_mismatch(rs, my_scope, rel.warp, me, rel.site, acq_site);
        }
    }

    /// One `ScopeMismatch` per (release site, acquire site, scopes), with an
    /// occurrence count (review F2 / R2).
    fn report_scope_mismatch(&mut self, rs: Scope, acq: Scope, rw: WarpId, aw: WarpId, rsite: SiteId, asite: SiteId) {
        let key = (rsite, asite, rs, acq);
        if let Some(&i) = self.scope_dedup.get(&key) {
            self.report.findings[i].occurrences += 1;
            return;
        }
        let f = Finding {
            kind: FindingKind::ScopeMismatch {
                release_scope: rs,
                acquire_scope: acq,
                release_warp: rw,
                acquire_warp: aw,
                release_site: rsite,
                acquire_site: asite,
            },
            severity: Severity::Error,
            alloc: AllocId(u32::MAX),
            bytes: 0..0,
            prior: None,
            current: None,
            occurrences: 1,
            tmem: None,
            spans: Vec::new(),
            names: None,        };
        if let Some(i) = self.push_finding(f) {
            self.scope_dedup.insert(key, i);
        }
    }

    // ----------------------------------------------------------- sync --

    pub fn sync(&mut self, s: SyncEvent) {
        self.maybe_gc();
        match &s {
            // The verdicts replace the polls of these lanes on this word.
            SyncEvent::WaitVerdicts { warp, lanes, alloc, range, .. } => {
                if let Some(v) = self.poll_stash.get_mut(warp) {
                    v.retain(|p| !(lanes.has(p.lane) && p.alloc == *alloc && p.range.start < range.end && range.start < p.range.end));
                }
                self.flush_polls(*warp);
            }
            SyncEvent::WarpSync { warp, .. }
            | SyncEvent::Arrive { warp, .. }
            | SyncEvent::Wait { warp, .. }
            | SyncEvent::Fence { warp, .. }
            | SyncEvent::AsyncIssue { warp, .. } => self.flush_polls(*warp),
            _ => {}
        }
        match s {
            SyncEvent::AllocBegin { alloc, size, space, cta } => {
                self.allocs.insert(alloc, Alloc {
                        size,
                        space,
                        cta,
                        shadow: IntervalShadow::new(),
                        seen: proxy_bit(Proxy::Generic),
                        retired: Default::default(),
                        retired_span_pages: 0,
                    });
            }
            SyncEvent::AllocEnd { alloc } => {
                self.alias_writers.remove(&alloc);
                // Shared memory ends only at CTA exit, and the hardware keeps
                // it until the CTA's outstanding bulk copies are done (ruling
                // S7: no final `cp.async.bulk.wait_group` is not an error).
                let shared = self.allocs.get(&alloc).is_some_and(|al| al.space == Space::Shared);
                let mut lifetime = Vec::new();
                for a in self.asyncs.iter_mut() {
                    if a.in_use && a.done == 0 {
                        if let Some((_, r)) = a.footprint.iter().find(|(al, _)| *al == alloc) {
                            if shared && a.kind == AsyncKind::Copy {
                                a.drained = true;
                            } else if !a.drained {
                                lifetime.push((a.op, r.clone()));
                            }
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
                        tmem: None,
                        spans: Vec::new(),
            names: None,                    };
                    self.push_finding(f);
                }
                self.allocs.remove(&alloc);
                self.words.remove(&alloc);
            }
            SyncEvent::DeclareWord { alloc, range } => {
                let ws = self.words.entry(alloc).or_default();
                if !ws.iter().any(|w| w.range == range) {
                    ws.push(Word { range, history: Vec::new(), last: None, own: HashMap::new() });
                }
            }
            SyncEvent::WarpSync { warp, mask, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                self.warp_sync(warp, mask, epoch);
            }
            SyncEvent::Arrive { warp, lanes, obj, phase, release, scope, site, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                self.arrive(warp, lanes, obj, phase, release, scope, site, epoch);
            }
            SyncEvent::Wait { warp, lanes, obj, phase, acquire, scope, site, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                self.wait(warp, lanes, obj, phase, acquire, scope, site);
            }
            SyncEvent::Fence { warp, lanes, kind, site, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                self.fence(warp, lanes, kind, site, epoch);
            }
            SyncEvent::AsyncIssue { op, warp, lanes, kind, proxy: _, preds, footprint, restricted, site, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                self.async_issue(op, warp, lanes, kind, preds, footprint, restricted, site, epoch);
            }
            SyncEvent::AsyncComplete { op, milestone, target } => {
                let Some(i) = self.async_idx(op) else { return };
                let m = side_index(milestone);
                let a = &mut self.asyncs[i];
                a.done = a.done.max(m as u8);
                let (actor, kind, gen_base) = (a.actor, a.kind, a.gen_base);
                if kind == AsyncKind::TcgenCommit {
                    for p in self.commit_closure(i) {
                        self.asyncs[p].done = 2;
                    }
                }
                let c = self.completion(i, m);
                match target {
                    CompletionTarget::Phase { obj, phase } => {
                        if let SyncObjId::Mbarrier { cta, .. } = obj {
                            self.asyncs[i].completed_ctas.push(cta.0);
                        }
                        self.phase_mut(obj, phase);
                        let ph = self.phases.get_mut(&obj).and_then(|m| m.get_mut(&phase)).unwrap();
                        ph.completion.join_propagating(&c, &self.memo);
                    }
                    CompletionTarget::Warp { warp, lanes } => {
                        if warp as usize >= self.warps.len() {
                            self.note_incomplete(Incomplete::CompletionWarpOutOfRange { warp });
                            return;
                        }
                        let cta = self.topo.cta_of(warp);
                        self.asyncs[i].completed_ctas.push(cta);
                        let memo = &self.memo;
                        let w = &mut self.warps[warp as usize];
                        match kind {
                            AsyncKind::TcgenLd | AsyncKind::TcgenSt | AsyncKind::TcgenPipelined => {
                                // tcgen05.wait::{ld,st}: the work has
                                // completed for the waiting thread. Completed
                                // work is ordered like any memory effect the
                                // thread observed: it travels through ordinary
                                // thread sync (hb). Only uncompleted
                                // (pipelined) work needs the fence pair.
                                for l in lanes.lanes8() {
                                    w.tcgen[l as usize].raise(actor, gen_base + 2);
                                    w.tcgen_waited[l as usize].raise(actor, gen_base + 2);
                                }
                                let mut done = Knowledge::default();
                                done.hb.raise(actor, gen_base + 2);
                                w.acquire(lanes, &done, memo);
                            }
                            _ => w.acquire(lanes, &c, memo),
                        }
                    }
                }
            }
            SyncEvent::WaitVerdicts { warp, lanes, alloc, range, scope, verdicts, pred_reads, site, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                if !self.pred_reads_stable(warp, lanes, epoch, &pred_reads) {
                    self.note_incomplete(Incomplete::WaitPredicateReadsUnstable { warp });
                    return;
                }
                // Each lane group judged by its own verdicts (never a
                // lane-wise conjunction).
                for (glanes, accepted, observed) in verdicts {
                    self.wait_verdicts(warp, glanes, alloc, range.clone(), scope, &accepted, observed, site);
                }
            }
        }
    }

    /// The phase record, keeping only the most recent phases per object.
    fn phase_mut(&mut self, obj: SyncObjId, phase: u64) -> &mut Phase {
        let m = self.phases.entry(obj).or_default();
        if !m.contains_key(&phase) {
            m.insert(phase, Phase::default());
            while m.len() > 4 {
                m.pop_first();
            }
        }
        m.entry(phase).or_default()
    }

    #[allow(clippy::too_many_arguments)]
    fn arrive(&mut self, warp: WarpId, lanes: LaneMask, obj: SyncObjId, phase: u64, release: Option<bool>, scope: Option<Scope>, site: SiteId, epoch: Epoch) {
        let Some(release) = release else {
            self.note_incomplete(Incomplete::SyncQualifierUnknown { warp });
            return;
        };
        let w = &self.warps[warp as usize];
        let tc = w.tcgen_publication(lanes, &self.memo);
        let pubk = release.then(|| Arc::new(w.publication(lanes, epoch, &self.memo)));
        let topo = self.topo;
        let cta = topo.cta_of(warp);
        self.phase_mut(obj, phase);
        let ph = self.phases.get_mut(&obj).and_then(|m| m.get_mut(&phase)).unwrap();
        ph.tcgen_rel.join(&tc, &self.memo);
        if let Some(k) = pubk {
            match ph.arrivals.iter_mut().find(|x| x.scope == scope && topo.cta_of(x.warp) == cta) {
                Some(x) => Arc::make_mut(&mut x.k).join_propagating(&k, &self.memo),
                None => ph.arrivals.push(Arrival { warp, scope, site, k }),
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn wait(&mut self, warp: WarpId, lanes: LaneMask, obj: SyncObjId, phase: u64, acquire: Option<bool>, scope: Option<Scope>, site: SiteId) {
        let Some(acquire) = acquire else {
            self.note_incomplete(Incomplete::SyncQualifierUnknown { warp });
            return;
        };
        let named = matches!(obj, SyncObjId::Named { .. });
        self.phase_mut(obj, phase);
        let topo = self.topo;
        let ph = &self.phases[&obj][&phase];
        let mut lost = false;
        let mut mismatches = Vec::new();
        let mut acquires: Vec<Arc<Knowledge>> = Vec::new();
        let mut pend: Vec<Arc<Rel>> = Vec::new();
        for x in &ph.arrivals {
            // Named barriers synchronise their participants without a scope;
            // every other object needs both scopes (None = qualifier lost).
            let (a, s) = match (named, x.scope, scope) {
                (true, _, _) => (Scope::Sys, Scope::Sys),
                (false, Some(a), Some(s)) => (a, s),
                _ => {
                    lost = true;
                    continue;
                }
            };
            if acquire {
                if a >= required_scope(&topo, x.warp, warp) && s >= required_scope(&topo, warp, x.warp) {
                    acquires.push(x.k.clone());
                } else {
                    mismatches.push((a, s, x.warp, x.site));
                }
            } else {
                pend.push(Arc::new(Rel { k: (*x.k).clone(), scope: Some(a), warp: x.warp, site: x.site }));
            }
        }
        let tcgen_rel = ph.tcgen_rel.clone();
        let completion = ph.completion.clone();
        if lost {
            self.note_incomplete(Incomplete::SyncQualifierUnknown { warp });
        }
        for (a, s, aw, asite) in mismatches {
            self.report_scope_mismatch(a, s, aw, warp, asite, site);
        }
        let memo = &self.memo;
        let w = &mut self.warps[warp as usize];
        if !tcgen_rel.is_empty() {
            for c in lanes.lanes8() {
                w.tcgen_in[c as usize].join(&tcgen_rel, memo);
            }
        }
        for k in acquires {
            w.acquire(lanes, &k, memo);
        }
        if acquire {
            w.acquire(lanes, &completion, memo);
        } else {
            // A relaxed wait synchronises nothing until a later acquire
            // fence; the copy's own bytes then need no scope match.
            pend.push(Arc::new(Rel { k: completion, scope: Some(Scope::Sys), warp, site }));
            for rel in pend {
                for c in lanes.lanes8() {
                    w.push_pending(c, rel.clone());
                }
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
        restricted: bool,
        site: SiteId,
        epoch: Epoch,
    ) {
        if kind == AsyncKind::TcgenCommit {
            // tcgen05.commit carries an implicit fence::before_thread_sync
            // for its issuing thread.
            self.fence(warp, lanes, FenceKind::TcgenBefore, site, epoch);
        }
        let w = &self.warps[warp as usize];
        let mut k = w.publication(lanes, epoch, &self.memo);
        let mut g2t_ranges: Arc<Vec<(AllocId, Range<u64>, Clock)>> = Arc::new(Vec::new());
        for (n, c) in lanes.lanes8().enumerate() {
            k.tcgen.join(&w.tcgen[c as usize], &self.memo);
            if matches!(kind, AsyncKind::TcgenPipelined | AsyncKind::TcgenLd | AsyncKind::TcgenSt) {
                // Completed tcgen05 work the issuer knows through hb (waited
                // ld/st, committed and observed MMA) is ordered before this
                // op without the fence pair.
                let hb = k.hb.clone();
                k.tcgen.join(&hb, &self.memo);
            }
            if n == 0 {
                g2t_ranges = w.g2t_ranges[c as usize].clone(); // shared, not copied
            } else if !w.g2t_ranges[c as usize].is_empty() {
                Arc::make_mut(&mut g2t_ranges).extend(w.g2t_ranges[c as usize].iter().cloned());
            }
        }
        let mut pred_idx = Vec::new();
        for p in preds {
            let Some(pi) = self.async_index.get(&p).copied() else {
                // Not live: completed and reclaimed (no witness left), or
                // unknown. Either way adding no ordering is conservative.
                continue;
            };
            let pa = &self.asyncs[pi];
            if kind != AsyncKind::TcgenCommit {
                k.tcgen.join(&pa.k.tcgen, &self.memo);
                k.tcgen.raise(pa.actor, pa.gen_base + 2);
            }
            pred_idx.push((pi, pa.gen_base));
        }
        if kind == AsyncKind::TcgenCommit && !restricted {
            // PTX: the commit tracks ALL prior async tcgen05 ops of the
            // thread. The engine names only the ops it still has in flight
            // (one that already landed, possibly behind an earlier commit
            // whose arrival is not yet delivered, is omitted), so every
            // earlier pipelined op of the thread the checker still holds
            // is tracked here too (deltas T15).
            for (si, sa) in self.asyncs.iter().enumerate() {
                if sa.in_use
                    && sa.kind == AsyncKind::TcgenPipelined
                    && sa.warp == warp
                    && lanes.has(sa.lane)
                    && !pred_idx.iter().any(|(p, _)| *p == si)
                {
                    pred_idx.push((si, sa.gen_base));
                }
            }
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
                    g2t_ranges: Arc::new(Vec::new()),
                    completed_ctas: Vec::new(),
                    preds: Vec::new(),
                    footprint: Vec::new(),
                    done: 0,
                    drained: false,
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
        slot.g2t_ranges = g2t_ranges;
        slot.completed_ctas.clear();
        slot.preds = pred_idx;
        slot.footprint = footprint;
        slot.done = 0;
        slot.drained = false;
        let gen_base = slot.gen_base;
        if kind == AsyncKind::TcgenPipelined {
            for c in lanes.lanes8() {
                self.warps[warp as usize].tcgen_issued[c as usize].raise(actor, gen_base + 2);
            }
        }
        self.async_index.insert(op, idx);
    }

    /// The tcgen05 ops a commit's completion covers: its tracked ops and,
    /// transitively, their architected-pipeline predecessors (a later op of
    /// the thread's pipe completing implies the earlier ones did; PTX
    /// §9.7.18: commit tracks *all* prior async tcgen05 operations).
    fn commit_closure(&self, i: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let mut stack: Vec<(usize, Epoch)> = self.asyncs[i].preds.clone();
        while let Some((p, g)) = stack.pop() {
            let pa = &self.asyncs[p];
            if pa.gen_base != g || !pa.in_use || !seen.insert(p) {
                continue;
            }
            out.push(p);
            // An op already completed had its chain completed with it.
            if pa.done < 2 {
                stack.extend(pa.preds.iter().copied());
            }
        }
        out
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
            // Direct preds; their tcgen views already carry the pipeline
            // chain behind them.
            for &(p, g) in &a.preds {
                let pa = &self.asyncs[p];
                if pa.gen_base != g {
                    continue;
                }
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

    fn fence(&mut self, warp: WarpId, lanes: LaneMask, kind: FenceKind, site: SiteId, epoch: Epoch) {
        match kind {
            FenceKind::AcqRel(scope) | FenceKind::Sc(scope) => {
                // Acquire half: pending relaxed observations.
                let pending = std::mem::take(&mut self.warps[warp as usize].pending_acq);
                let mut keep = Vec::new();
                for (lane, rel) in pending {
                    if lanes.has(lane) {
                        self.acquire_rel(warp, one_lane(lane), scope, &rel, site);
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
                            .filter(|((ow, ol, os), _)| (*ow, *ol) != (warp, c) && self.covers(*os, *ow, warp) && self.covers(scope, warp, *ow))
                            .map(|(_, k)| k.clone())
                            .collect();
                        for k in incoming {
                            self.warps[warp as usize].acquire(one_lane(c), &k, &self.memo);
                        }
                    }
                    for c in lanes.lanes8() {
                        let k = Arc::new(self.warps[warp as usize].publication(one_lane(c), epoch, &self.memo));
                        // Latest fence per (thread, scope): a later fence of the
                        // same thread and scope covers the earlier one, but a
                        // narrower later fence must not hide a wider one.
                        self.sc.insert((warp, c, scope), k);
                    }
                }
                // Release half: the head a later relaxed strong write carries.
                let w = &mut self.warps[warp as usize];
                for c in lanes.lanes8() {
                    let k = w.publication(one_lane(c), epoch, &self.memo);
                    w.fence_rel[c as usize] = Some(Arc::new(Rel { k, scope: Some(scope), warp, site }));
                }
            }
            FenceKind::ProxyAsync(dom) => {
                let slots: Vec<usize> = fence_domains(dom).iter().flat_map(|&d| [d, NDOM + d]).collect();
                self.bridge_rows(warp, lanes, &slots, epoch);
                self.snapshot_hb_into(warp, lanes, &slots);
            }
            FenceKind::TensormapRelease(scope) => {
                // One head per releasing lane: (this warp, scope, what the
                // lane knows now, own lanes exact).
                let heads: Vec<(u8, Clock)> = {
                    let w = &self.warps[warp as usize];
                    lanes.lanes8().map(|c| (c, w.publication(one_lane(c), epoch, &self.memo).hb)).collect()
                };
                let memo = &self.memo;
                let w = &mut self.warps[warp as usize];
                for (c, hb) in heads {
                    let x = w.extra[c as usize].get_or_insert_with(Default::default);
                    join_tmap(&mut x.tmap_rel, &Some(Arc::new(vec![(warp, scope, hb)])), memo);
                }
            }
            FenceKind::TensormapAcquire { scope, alloc, range } => {
                // Keep the heads whose releasing fence and this acquire
                // mutually include each other's thread (review S6: filter by
                // the releaser, not by each component's own actor).
                let topo = self.topo;
                let mut acquired = Vec::new();
                {
                    let w = &self.warps[warp as usize];
                    for c in lanes.lanes8() {
                        let mut heads = w.base.tmap_rel.clone();
                        if let Some(x) = &w.extra[c as usize] {
                            join_tmap(&mut heads, &x.tmap_rel, &self.memo);
                        }
                        let mut t = Clock::default();
                        for (rw, rs, clk) in heads.iter().flat_map(|h| h.iter()) {
                            if *rs >= required_scope(&topo, *rw, warp) && scope >= required_scope(&topo, warp, *rw) {
                                t.join(clk, &self.memo);
                            }
                        }
                        acquired.push((c, t));
                    }
                }
                let memo = &self.memo;
                let w = &mut self.warps[warp as usize];
                for (c, t) in acquired {
                    let v = Arc::make_mut(&mut w.g2t_ranges[c as usize]);
                    match v.iter_mut().find(|(al, r, _)| *al == alloc && *r == range) {
                        Some((_, _, k)) => {
                            k.join(&t, memo);
                        }
                        None => v.push((alloc, range.clone(), t)),
                    }
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
    fn wait_verdicts(&mut self, warp: WarpId, lanes: LaneMask, alloc: AllocId, range: Range<u64>, scope: Scope, accepted: &[u64], observed: u32, site: SiteId) {
        let exact = self.words.get(&alloc).and_then(|ws| ws.iter().position(|w| w.range == range));
        // A declared region polled element by element (a `sync_words`
        // buffer declared whole): the launch value needs no history.
        let within = self.words.get(&alloc).and_then(|ws| ws.iter().position(|w| w.range.start <= range.start && range.end <= w.range.end));
        let Some(wi) = exact.or(within) else {
            self.note_incomplete(Incomplete::WaitExitUnproven { warp });
            return;
        };
        if exact.is_none() {
            if accepted.first().is_some_and(|b| b & 1 != 0) {
                return; // the launch value satisfied the predicate: no edge owed
            }
            self.note_incomplete(Incomplete::WaitExitUnproven { warp });
            return;
        }
        // Earliest predicate-accepted history entry: schedule independent,
        // but never coherence-before the waiting lanes' own latest write of
        // the word (CoWR; deltas W7): a grid-sync counter accepts stale
        // values of earlier rounds that the waiter can no longer read.
        let floor = self.words[&alloc][wi].own.iter().filter(|((w, l), _)| *w == warp && lanes.has(*l)).map(|(_, i)| *i).max().unwrap_or(0);
        // If no accepted entry is at or after it (the predicate's history
        // view is coarser than the waiter's own writes), keep the earliest.
        let all = || accepted.iter().enumerate().flat_map(|(i, b)| (0..64u32).filter(move |k| b >> k & 1 != 0).map(move |k| i as u32 * 64 + k));
        let first = all().find(|idx| *idx >= floor).or_else(|| all().next());
        let Some(mut idx) = first else {
            self.note_incomplete(Incomplete::WaitExitUnproven { warp });
            return;
        };
        if idx == 0 {
            return; // the launch value satisfied the predicate: no edge owed
        }
        let word = &self.words[&alloc][wi];
        let Some(e) = word.history.get(idx as usize - 1) else {
            self.note_incomplete(Incomplete::WaitExitUnproven { warp });
            return;
        };
        if e.mixed_size {
            self.note_incomplete(Incomplete::SignalWriteNotRecorded { warp });
            return;
        }
        if e.is_async {
            // Async publications land their bytes at completion; the run's
            // observed version is the only edge available (degraded,
            // schedule-dependent fallback, async publications only).
            idx = observed;
            if idx == 0 {
                return;
            }
        }
        let word = &self.words[&alloc][wi];
        let Some(e) = word.history.get(idx as usize - 1) else {
            self.note_incomplete(Incomplete::WaitExitUnproven { warp });
            return;
        };
        let Some(heads) = e.rel.clone() else {
            if !e.consumed {
                // Explained only by a plain write: no edge can be proven
                // (delta W2). The plain write itself still races the poll.
                self.note_incomplete(Incomplete::WaitExitUnproven { warp });
            }
            return;
        };
        for rel in heads.iter() {
            self.acquire_rel(warp, lanes, scope, rel, site);
        }
    }

    // ------------------------------------------------------------- GC --

    /// View-aware dominated-frontier GC plus async-slot reclaim.
    ///
    /// The meet is taken per reach: shared memory and TMEM of a cluster can
    /// only be accessed by that cluster's warps (plus in-flight async ops),
    /// so a CTA's witnesses retire without waiting for unrelated CTAs or
    /// later waves (review R5); global memory uses every live actor.
    pub fn gc(&mut self) {
        self.since_gc = 0;
        self.stats.gc_runs += 1;
        fn fold(meet: &mut Option<Knowledge>, k: &Knowledge, tcgen: &Clock) {
            match meet {
                None => {
                    let mut x = k.propagating();
                    x.tcgen = tcgen.clone();
                    x.g2t = k.g2t.clone();
                    *meet = Some(x);
                }
                Some(m) => {
                    m.hb = m.hb.meet(&k.hb);
                    for d in 0..NDOM {
                        m.g2a[d] = m.g2a[d].meet(&k.g2a[d]);
                        m.a2g[d] = m.a2g[d].meet(&k.a2g[d]);
                    }
                    m.g2t = m.g2t.meet(&k.g2t);
                    m.tcgen = m.tcgen.meet(tcgen);
                }
            }
        }
        fn meet2(a: &Option<Knowledge>, b: &Option<Knowledge>) -> Option<Knowledge> {
            match (a, b) {
                (None, x) | (x, None) => x.clone(),
                (Some(a), Some(b)) => {
                    let mut m = Some(a.clone());
                    fold(&mut m, b, &b.tcgen);
                    m
                }
            }
        }
        // Lane extras only add knowledge, so `base` is a sound lower bound
        // for a warp; per-lane tcgen views are met lane by lane.
        let mut async_meet: Option<Knowledge> = None;
        for a in self.asyncs.iter().filter(|a| a.in_use && a.done < 2 && a.kind != AsyncKind::TcgenCommit) {
            fold(&mut async_meet, &a.k, &a.k.tcgen);
        }
        let ncl = (self.topo.num_ctas / self.topo.ctas_per_cluster.max(1)).max(1) as usize;
        let mut cluster_meet: Vec<Option<Knowledge>> = vec![None; ncl];
        let mut any_live = vec![false; ncl];
        for w in self.warps.iter().filter(|w| !w.done) {
            let cl = (self.topo.cluster_of(w.actor) as usize).min(ncl - 1);
            let mut t = w.tcgen[0].clone();
            for l in 1..32 {
                t = t.meet(&w.tcgen[l]);
            }
            fold(&mut cluster_meet[cl], &w.base, &t);
            any_live[cl] = true;
        }
        let mut global_meet: Option<Knowledge> = async_meet.clone();
        for m in &cluster_meet {
            global_meet = meet2(&global_meet, m);
        }
        let any_global = global_meet.is_some();
        let cluster_meet: Vec<Option<Knowledge>> = cluster_meet.iter().map(|m| meet2(m, &async_meet)).collect();
        let ctas_per_cluster = self.topo.ctas_per_cluster.max(1);
        // `None` meet with nobody live: nothing can access it any more.
        let dead_in = |meet: &Option<Knowledge>, live: bool, w: &Witness, pcs: &[Proxy]| match meet {
            None => !live,
            Some(m) => pcs.iter().all(|pc| m.view(select_view(w.proxy(), *pc, w.domain())).observes(w.stamp, w.lane())),
        };
        let mut retired = 0u64;
        let mut live_actors: HashSet<ActorId> = HashSet::new();
        let mut min_epoch: HashMap<ActorId, Epoch> = HashMap::new();
        let mut allocs = std::mem::take(&mut self.allocs);
        for alloc in allocs.values_mut() {
            let (space, seen) = (alloc.space, alloc.seen);
            let (meet, live) = match space {
                Space::Shared | Space::Tmem => {
                    let cl = ((alloc.cta / ctas_per_cluster) as usize).min(ncl - 1);
                    (&cluster_meet[cl], any_live[cl] || async_meet.is_some())
                }
                _ => (&global_meet, any_global),
            };
            let seen_proxies: Vec<Proxy> =
                [Proxy::Async, Proxy::TensorMap, Proxy::ReadOnly, Proxy::Tcgen].into_iter().filter(|p| seen & proxy_bit(*p) != 0).collect();
            let fully_dead = |w: &Witness| dead_in(meet, live, w, future_proxies(space));
            let same_proxy_dead = |w: &Witness| dead_in(meet, live, w, &[w.proxy()]);
            let dead_in_seen = |w: &Witness| dead_in(meet, live, w, &seen_proxies);
            let mut folded: Vec<Witness> = Vec::new();
            alloc.shadow.retain_mut(|cell| {
                let last = cell.writes.last().map(|e| e.w);
                // Decide per witness: drop (dead in every view), fold into the
                // retired-generic summary (dead in every view of a proxy that
                // already accessed the allocation; the summary answers for the
                // others), or keep.
                let mut decide = |w: &Witness, pinned: bool| -> bool {
                    if pinned {
                        return true;
                    }
                    if fully_dead(w) {
                        retired += 1;
                        return false;
                    }
                    // Async-actor stamps may be summarised too: once every
                    // live actor's hb observes them, any view carrying a
                    // later generation of the slot was snapshotted after
                    // that, so it observes the old stamp as well.
                    if w.proxy() == Proxy::Generic && space != Space::Tmem && same_proxy_dead(w) && dead_in_seen(w) {
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
                let page = lo >> 12;
                alloc.retired_span_pages = alloc.retired_span_pages.max((hi.saturating_sub(1) >> 12) - page);
                let key = (page, w.stamp.actor(), w.lane(), w.writes(), domain_code(w.domain()));
                let e = alloc.retired.entry(key).or_insert_with(|| RetiredGeneric { w, lo, hi, info: info.clone() });
                if w.stamp.epoch() >= e.w.stamp.epoch() {
                    e.info = info;
                    e.w = w;
                }
                if lo < e.lo || hi > e.hi {
                    e.lo = e.lo.min(lo);
                    e.hi = e.hi.max(hi);
                    alloc.retired_span_pages = alloc.retired_span_pages.max((e.hi.saturating_sub(1) >> 12) - page);
                }
                let kind = if e.w.writes() { AccessKind::Write } else { AccessKind::Read };
                e.w = Witness::pack(e.w.stamp, e.w.lane(), Proxy::Generic, e.w.domain(), kind, None, false, (e.lo, e.hi), &mut self.wide);
                e.info.span = e.lo..e.hi;
            }
        }
        let cells: u64 = allocs.values().map(|a| a.shadow.len() as u64).sum();
        self.gc_period = 2 * cells;
        self.allocs = allocs;
        self.stats.witnesses_retired += retired;
        // Declared-word history: a release whose payload every live actor
        // already holds adds nothing when acquired; drop the payload, keep
        // the index (verdict bitsets index absolute positions).
        if let Some(m) = &global_meet {
            for words in self.words.values_mut() {
                for word in words.iter_mut() {
                    for e in word.history.iter_mut() {
                        let useless = e.rel.as_ref().is_some_and(|h| {
                            h.iter().all(|r| {
                                r.k.hb.leq(&m.hb, &self.memo)
                                    && (0..NDOM).all(|d| r.k.g2a[d].leq(&m.g2a[d], &self.memo) && r.k.a2g[d].leq(&m.a2g[d], &self.memo))
                                    && r.k.tcgen_rel.is_empty()
                                    && r.k.tmap_rel.is_none()
                            })
                        });
                        if useless {
                            e.rel = None;
                            e.consumed = true;
                        }
                    }
                }
            }
        }
        // Async-slot reclaim: completed ops with no witness left.
        for i in 0..self.asyncs.len() {
            let a = &self.asyncs[i];
            if a.in_use && a.done >= 2 && !live_actors.contains(&a.actor) {
                let op = a.op;
                self.async_index.remove(&op);
                let a = &mut self.asyncs[i];
                a.in_use = false;
                a.gen_base += 2; // next generation starts above every old epoch
                a.k = Knowledge::default();
                a.g2t_ranges = Arc::new(Vec::new());
                a.completed_ctas.clear();
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
