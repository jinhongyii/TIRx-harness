//! Launch-wide engine bookkeeping that is *not* protocol state.
//!
//! Protocol state lives in [`crate::sync::SyncTable`]; this holds what the
//! engine needs around it: which async ops belong to which per-lane async
//! group, deferred arrive-ons released by a group's completion,
//! declared-word write histories, collective rendezvous (setmaxnreg
//! warpgroups, `cta_group::2` pairs, `bar.red` accumulators, grid barrier),
//! and the tcgen05 pipeline order.
//!
//! Everything here is deterministic (ordered maps where iteration order is
//! observable).

use crate::arena::{AllocId, Arena, ByteSpan};
use crate::observe::{AsyncClass, CtaId, WarpId};
use crate::program::Proxy;
use crate::sync::{AsyncId, Completion, FxBuild, ResourceId};
use crate::value::WarpMask;
use std::collections::{BTreeMap, HashMap, HashSet};

/// What landing an async op must report (kept beside the `AsyncOp`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsyncMeta {
    /// Issuing lane (async ops are per thread; their accesses carry it).
    pub lane: u8,
    pub class: AsyncClass,
    pub proxy: Proxy,
    /// Phase targets `(mbarrier, generation)` its signals land on.
    pub phases: Vec<(ResourceId, u64)>,
    /// TMA loads: byte pattern of the OOB fill (empty = zeros).
    pub fill_pattern: Vec<u8>,
    /// TMA loads of TF32 maps: round copied f32 elements to tf32.
    pub tf32_round: bool,
    /// `_report` copy forms: inspect the copied source bytes and OR the
    /// result into the report bit of every completion phase.
    pub report: Option<crate::program::ReportMode>,
    /// tcgen05.mma `.lut_b`: TMEM address of the lookup table.
    pub lut_b: Option<u32>,
    /// `st.async` / `red.async` (PTX §9.7.10.12, §9.7.15.7): the landing
    /// is performed in the generic proxy as a strong release write at this
    /// scope (CONTRACT_REQUESTS W5-8).
    pub strong: Option<crate::program::Scope>,
    /// Sub-byte TMA stores (FP4/U6 maps, oplib `TmaPlan::global_bits`):
    /// masked partial-byte writes applied after the byte spans.
    pub bit_frags: Vec<BitFrag>,
    /// tcgen05.mma with a shared-memory A: the MMA op excludes these A
    /// spans from its read accesses; the separate shared-A read op
    /// (`MmaSharedARead`, `is_a_read`) reports them (W5-10).
    pub a_reads: Vec<(AllocId, ByteSpan)>,
    pub is_a_read: bool,
    /// `cp.async.bulk .ignore_oob` dead destination bytes: written as zero
    /// and left *uninitialized* (legacy `raw_bulk_copy_g2s_cta_ignore_oob`
    /// writes `0` with validity `false`), so a later read reports.
    pub dead: Vec<(AllocId, ByteSpan)>,
}

/// One masked partial-byte global write of a sub-byte TMA store:
/// `g = (g & !(mask << tgt)) | (((s >> src) & mask) << tgt)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitFrag {
    pub global: (AllocId, u64),
    pub smem: (AllocId, u64),
    pub src_shift: u8,
    pub tgt_shift: u8,
    pub mask: u8,
}

/// Per-lane async-group membership of in-flight async ops.
#[derive(Clone, Debug, Default)]
pub struct GroupTracker {
    /// Issued but not yet committed ops, per `AsyncGroup` resource.
    pub open: HashMap<ResourceId, Vec<AsyncId>, FxBuild>,
    /// Ops of each committed group (all, landed or not).
    pub members: HashMap<(ResourceId, u64), Vec<AsyncId>, FxBuild>,
    /// Not-yet-landed ops per committed group.
    pub pending: HashMap<(ResourceId, u64), u32, FxBuild>,
    /// Groups each in-flight op still has to report its landing to.
    pub op_groups: HashMap<AsyncId, Vec<(ResourceId, u64)>, FxBuild>,
    /// Ops that landed while still uncommitted.
    pub landed_open: HashSet<AsyncId, FxBuild>,
    /// Deferred mbarrier arrive-ons released by a group's full completion.
    pub arrivals: HashMap<(ResourceId, u64), Vec<Completion>, FxBuild>,
}

impl GroupTracker {
    /// Record an issue into `res`'s open set.
    pub fn issue(&mut self, res: ResourceId, op: AsyncId) {
        self.open.entry(res).or_default().push(op);
    }

    /// Move `res`'s open ops into group `ordinal`. Returns the milestone
    /// completions that are due immediately (every member already landed).
    pub fn commit(&mut self, res: ResourceId, ordinal: u64) -> Vec<Completion> {
        let ops = self.open.remove(&res).unwrap_or_default();
        if ops.is_empty() {
            // An empty group is born complete in the protocol state
            // (`async_group::close`): no milestone completion to queue (one
            // would never be enabled and would sit in the queue forever).
            self.members.insert((res, ordinal), ops);
            return Vec::new();
        }
        let mut pending = 0u32;
        for &op in &ops {
            if self.landed_open.remove(&op) {
                continue;
            }
            pending += 1;
            self.op_groups.entry(op).or_default().push((res, ordinal));
        }
        self.members.insert((res, ordinal), ops);
        if pending == 0 {
            milestones(res, ordinal)
        } else {
            self.pending.insert((res, ordinal), pending);
            Vec::new()
        }
    }

    /// `op` landed: milestone completions of groups it completed.
    pub fn landed(&mut self, op: AsyncId, still_open: bool) -> Vec<Completion> {
        let mut out = Vec::new();
        if let Some(groups) = self.op_groups.remove(&op) {
            for key in groups {
                if let Some(n) = self.pending.get_mut(&key) {
                    *n -= 1;
                    if *n == 0 {
                        self.pending.remove(&key);
                        out.extend(milestones(key.0, key.1));
                    }
                }
            }
        }
        if still_open {
            self.landed_open.insert(op);
        }
        out
    }

    /// Is `op` in some lane's open (uncommitted) set?
    pub fn is_open(&self, op: AsyncId) -> bool {
        self.open.values().any(|v| v.contains(&op))
    }
}

fn milestones(res: ResourceId, ordinal: u64) -> Vec<Completion> {
    use crate::sync::async_group::Milestone;
    vec![
        Completion::GroupMilestone { res, ordinal, milestone: Milestone::ReadsDone },
        Completion::GroupMilestone { res, ordinal, milestone: Milestone::FullyDone },
    ]
}

/// One declared synchronization region and its write log.
#[derive(Clone, Debug)]
pub struct WordRegion {
    /// Allocation-relative span of the region.
    pub span: ByteSpan,
    /// Region bytes at declaration (history index 0).
    pub init: Vec<u8>,
    /// Post-image of the region after each write `Access` overlapping it,
    /// with the allocation-relative spans that access wrote.
    pub log: Vec<(Vec<ByteSpan>, Vec<u8>)>,
    /// Writes counted so far (always kept, with or without an observer;
    /// equals `log.len()` when the table keeps images).
    pub count: usize,
    /// More than [`MAX_WORD_HISTORY`] writes arrived: later ones were not
    /// counted, so `wait_until` over this region is `incomplete` (with or
    /// without an observer, W13-1).
    pub overflow: bool,
}

/// Declared-word writes logged per region; a region that receives more is
/// marked `overflow` and its `wait_until` verdicts fail closed
/// (`Unsupported` -> `incomplete`), never computed on a truncated history.
pub const MAX_WORD_HISTORY: usize = 1 << 16;

/// Declared words (`BufferDecl::sync_words`, or declared at first wait).
#[derive(Clone, Debug, Default)]
pub struct WordTable {
    /// Declared regions per allocation, sorted by `(span.start, span.len)`
    /// (lookups are binary searches: kernels declare 10^5+ words).
    pub regions: HashMap<AllocId, Vec<WordRegion>>,
    /// Longest region per allocation (bounds overlap searches).
    max_len: HashMap<AllocId, u64>,
    /// Regions declared or logged since the last [`Self::clear_dirty`]
    /// (the scheduler merges only these), with a dedupe set.
    dirty: Vec<(AllocId, ByteSpan)>,
    dirty_set: HashSet<(AllocId, u64, u64)>,
    /// Keep post-images (a history-consuming observer is attached);
    /// otherwise only counts and overflow are kept.
    pub images: bool,
    /// Declared words not yet materialized (count-only mode): per
    /// allocation, `(base, width, n)` arrays of `n` words of `width` bytes.
    /// A word's region is created (count 0) on its first write or wait, so
    /// a kernel declaring 10^5+ words pays only for the words it touches.
    arrays: HashMap<AllocId, Vec<(u64, u64, u64)>>,
}

impl WordTable {
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty() && self.arrays.is_empty()
    }

    /// Declare `spans` of `alloc` (a buffer's sync words). With images the
    /// regions are created now (index 0 = current bytes); without, a run of
    /// equal-width adjacent words is recorded as one lazy array.
    pub fn declare_words(&mut self, arena: &Arena, alloc: AllocId, spans: &[ByteSpan]) {
        if self.images {
            for &s in spans {
                self.declare(arena, alloc, s);
            }
            return;
        }
        let mut i = 0;
        while i < spans.len() {
            let (base, w) = (spans[i].start, spans[i].len);
            let mut j = i + 1;
            while j < spans.len() && spans[j].len == w && spans[j].start == base + (j - i) as u64 * w {
                j += 1;
            }
            if w > 0 {
                self.arrays.entry(alloc).or_default().push((base, w, (j - i) as u64));
            }
            i = j;
        }
    }

    /// Declare `n` words of `w` bytes at `base` of `alloc` lazily
    /// (count-only mode; no image is ever needed).
    pub fn declare_array(&mut self, alloc: AllocId, base: u64, w: u64, n: u64) {
        debug_assert!(!self.images, "lazy words have no declaration image");
        if w > 0 && n > 0 {
            self.arrays.entry(alloc).or_default().push((base, w, n));
        }
    }

    /// Create the regions of lazily declared words of `alloc` overlapping
    /// `span` (count 0, no image: count-only mode).
    pub fn materialize(&mut self, alloc: AllocId, span: ByteSpan) {
        let Some(arrays) = self.arrays.get(&alloc) else { return };
        let mut new: Vec<ByteSpan> = Vec::new();
        for &(base, w, n) in arrays {
            let end = base + w * n;
            let (lo, hi) = (span.start.max(base), span.end().max(span.start + 1).min(end));
            if lo >= hi {
                continue;
            }
            for k in (lo - base) / w..(hi - base).div_ceil(w) {
                new.push(ByteSpan::new(base + k * w, w));
            }
        }
        for s in new {
            if self.position(alloc, s).is_none() {
                self.insert(alloc, WordRegion { span: s, init: Vec::new(), log: Vec::new(), count: 0, overflow: false });
            }
        }
    }

    fn arrays_overlap(&self, alloc: AllocId, span: ByteSpan) -> bool {
        self.arrays.get(&alloc).is_some_and(|v| v.iter().any(|&(base, w, n)| span.start < base + w * n && base < span.end()))
    }

    fn mark(&mut self, alloc: AllocId, span: ByteSpan) {
        if self.dirty_set.insert((alloc, span.start, span.len)) {
            self.dirty.push((alloc, span));
        }
    }

    /// Regions declared or logged since the last [`Self::clear_dirty`], in
    /// span order per allocation (deterministic).
    pub fn dirty(&self) -> Vec<(AllocId, ByteSpan)> {
        let mut v = self.dirty.clone();
        v.sort_by_key(|&(a, s)| (a, s.start, s.len));
        v
    }

    pub fn clear_dirty(&mut self) {
        self.dirty.clear();
        self.dirty_set.clear();
    }

    /// Does `alloc` hold any declared region? (Writes to other allocations
    /// log nothing, so they may take paths that skip the history.)
    pub fn has(&self, alloc: AllocId) -> bool {
        self.regions.get(&alloc).is_some_and(|rs| !rs.is_empty()) || self.arrays.contains_key(&alloc)
    }

    /// Index range of `alloc`'s regions that may overlap `span` (each one
    /// still needs an overlap test).
    fn window(&self, alloc: AllocId, span: ByteSpan) -> std::ops::Range<usize> {
        let Some(rs) = self.regions.get(&alloc) else { return 0..0 };
        let ml = self.max_len.get(&alloc).copied().unwrap_or(0);
        let lo_start = span.start.saturating_sub(ml);
        let lo = rs.partition_point(|r| r.span.start < lo_start);
        let hi = rs.partition_point(|r| r.span.start < span.end().max(span.start + 1));
        lo..hi.max(lo)
    }

    /// Index of the region of `alloc` with exactly `span`.
    pub fn position(&self, alloc: AllocId, span: ByteSpan) -> Option<usize> {
        self.regions.get(&alloc)?.binary_search_by_key(&(span.start, span.len), |r| (r.span.start, r.span.len)).ok()
    }

    /// The region of `alloc` with exactly `span`.
    pub fn exact(&self, alloc: AllocId, span: ByteSpan) -> Option<&WordRegion> {
        let i = self.position(alloc, span)?;
        self.regions.get(&alloc).map(|rs| &rs[i])
    }

    /// Mutable [`Self::exact`].
    pub fn exact_mut(&mut self, alloc: AllocId, span: ByteSpan) -> Option<&mut WordRegion> {
        let i = self.position(alloc, span)?;
        self.regions.get_mut(&alloc).map(|rs| &mut rs[i])
    }

    /// Insert `r` in sorted position unless a region with its span exists;
    /// returns the region's index.
    pub fn insert(&mut self, alloc: AllocId, r: WordRegion) -> usize {
        let ml = self.max_len.entry(alloc).or_insert(0);
        *ml = (*ml).max(r.span.len);
        let rs = self.regions.entry(alloc).or_default();
        match rs.binary_search_by_key(&(r.span.start, r.span.len), |t| (t.span.start, t.span.len)) {
            Ok(i) => i,
            Err(i) => {
                rs.insert(i, r);
                i
            }
        }
    }

    /// Does any span overlap a declared region of `alloc`?
    pub fn overlaps(&self, alloc: AllocId, spans: &[ByteSpan]) -> bool {
        if spans.iter().any(|s| self.arrays_overlap(alloc, *s)) {
            return true;
        }
        let Some(rs) = self.regions.get(&alloc) else { return false };
        spans.iter().any(|s| rs[self.window(alloc, *s)].iter().any(|r| s.overlaps(r.span)))
    }

    /// Region of `alloc` covering `span`, if declared (the first in span
    /// order when declared regions overlap).
    pub fn region(&self, alloc: AllocId, span: ByteSpan) -> Option<&WordRegion> {
        let rs = self.regions.get(&alloc)?;
        rs[self.window(alloc, span)].iter().find(|r| r.span.start <= span.start && span.end() <= r.span.end())
    }

    /// Region of `alloc` containing byte `at`, if any (first in span order).
    pub fn region_at(&self, alloc: AllocId, at: u64) -> Option<&WordRegion> {
        self.region(alloc, ByteSpan::new(at, 1))
    }

    /// Declare `span` of `alloc` with its current bytes as history index 0
    /// (no-op if exactly `span` is already declared).
    pub fn declare(&mut self, arena: &Arena, alloc: AllocId, span: ByteSpan) {
        if self.position(alloc, span).is_some() {
            return;
        }
        let init = snapshot(arena, alloc, span);
        self.insert(alloc, WordRegion { span, init, log: Vec::new(), count: 0, overflow: false });
        self.mark(alloc, span);
    }

    /// Append one history entry per declared region overlapping a lane's
    /// write of `bytes` at `span` (allocation-relative): the region's last
    /// image with this lane's bytes merged in (README decision 14: entries
    /// are (write Access, lane) pairs in delivery order, lanes ascending).
    pub fn log_lane(&mut self, alloc: AllocId, span: ByteSpan, bytes: &[u8]) {
        if self.arrays.contains_key(&alloc) {
            self.materialize(alloc, span);
        }
        let images = self.images;
        let w = self.window(alloc, span);
        let Some(rs) = self.regions.get_mut(&alloc) else { return };
        let mut touched: Vec<ByteSpan> = Vec::new();
        for r in rs[w].iter_mut() {
            if !span.overlaps(r.span) {
                continue;
            }
            touched.push(r.span);
            if r.count >= MAX_WORD_HISTORY {
                r.overflow = true;
                continue;
            }
            r.count += 1;
            if !images {
                continue;
            }
            let mut img = r.log.last().map(|e| e.1.clone()).unwrap_or_else(|| r.init.clone());
            let lo = span.start.max(r.span.start);
            let hi = span.end().min(r.span.end());
            for x in lo..hi {
                img[(x - r.span.start) as usize] = bytes[(x - span.start) as usize];
            }
            r.log.push((vec![span], img));
        }
        for t in touched {
            self.mark(alloc, t);
        }
    }

    /// [`Self::log_lane`] with the bytes currently in the arena (single
    /// writer of `span`, e.g. an async landing).
    pub fn log_from_arena(&mut self, arena: &Arena, alloc: AllocId, span: ByteSpan) {
        if !self.overlaps(alloc, &[span]) {
            return;
        }
        let bytes = snapshot(arena, alloc, span);
        self.log_lane(alloc, span, &bytes);
    }

    /// Did the region covering `span` overflow its history?
    pub fn overflowed(&self, alloc: AllocId, span: ByteSpan) -> bool {
        self.region(alloc, span).is_some_and(|r| r.overflow)
    }

    /// Values of the word `span` (<= 8 bytes, little-endian) over its history:
    /// index 0 = declaration value, then one entry per write overlapping it.
    pub fn history(&self, alloc: AllocId, span: ByteSpan) -> Option<Vec<u64>> {
        let r = self.region(alloc, span)?;
        let lo = (span.start - r.span.start) as usize;
        let hi = lo + span.len as usize;
        let val = |img: &[u8]| {
            let mut b = [0u8; 8];
            b[..hi - lo].copy_from_slice(&img[lo..hi]);
            u64::from_le_bytes(b)
        };
        let mut out = vec![val(&r.init)];
        for (hit, img) in &r.log {
            if hit.iter().any(|s| s.overlaps(span)) {
                out.push(val(img));
            }
        }
        Some(out)
    }
}

fn snapshot(arena: &Arena, alloc: AllocId, span: ByteSpan) -> Vec<u8> {
    arena.read_raw(alloc, span)
}

/// Incremental `WaitUntil` verdicts of one (warp, pc, word): per lane, the
/// history prefix already evaluated, its acceptance bits, and the capture
/// values they were evaluated under (a change resets the lane).
#[derive(Clone, Debug, Default)]
pub struct VerdictCache {
    pub evaluated: [usize; 32],
    pub bits: Vec<Vec<u64>>,
    pub captures: Vec<Vec<u64>>,
}

/// `bar.red` accumulator of one generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RedAcc {
    pub popc: u64,
    pub all: bool,
    pub any: bool,
    pub started: bool,
}

/// A warp's partial-mask arrival at a non-`.aligned` named barrier: the
/// executing lanes wait for the rest of the warp (PTX §9.7.15.1, Q3); one
/// warp arrival happens when the arrived lanes cover the non-exited ones.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamedPartial {
    pub id: u8,
    /// 0 = arrive, 1 = sync, 2 = red.
    pub flavor: u8,
    pub count: u64,
    /// Lanes that executed the barrier so far.
    pub lanes: crate::value::WarpMask,
    /// Arrived lanes not yet past the barrier.
    pub waiting: crate::value::WarpMask,
    /// Generation of the warp arrival, once made.
    pub gen: Option<u64>,
    /// `bar.red` predicate accumulation of the arrived lanes.
    pub red: RedAcc,
}

/// See [`LaunchAux::cluster_partial`].
#[derive(Clone, Debug)]
pub struct ClusterPartial {
    pub gather: crate::sync::cluster::Gather,
    /// `[arrive, wait]`: the completed warp command's lanes still to pass.
    pub pass: [Option<ClusterPass>; 2],
}

#[derive(Clone, Copy, Debug)]
pub struct ClusterPass {
    /// Every gathered lane (the warp's one command).
    pub lanes: crate::value::WarpMask,
    /// Gathered lanes that have not passed yet.
    pub waiting: crate::value::WarpMask,
    /// `wait`: the warp's Wait completed (later groups pass without
    /// stepping again).
    pub passed: bool,
}

/// One setmaxnreg warpgroup rendezvous.
#[derive(Clone, Debug, Default)]
pub struct SetmaxRendezvous {
    /// Warps (index in CTA) that contributed to the current instance.
    pub arrived: Vec<u32>,
    pub inc: bool,
    pub count: u32,
    /// Instances committed so far.
    pub epoch: u64,
}

/// A two-CTA (`cta_group::2`) tcgen05 lifecycle rendezvous.
#[derive(Clone, Debug, Default)]
pub struct PairRendezvous {
    /// `(cta, warp)` of the first arriver of the open instance.
    pub first: Option<(CtaId, WarpId)>,
    pub epoch: u64,
    /// Result of each committed instance (alloc base), by epoch.
    pub results: BTreeMap<u64, u32>,
    /// Member warps of each committed instance, by epoch.
    pub participants: BTreeMap<u64, Vec<WarpId>>,
}

/// Cooperative grid barrier.
#[derive(Clone, Debug, Default)]
pub struct GridBarrier {
    pub gen: u64,
    pub arrived: HashSet<WarpId>,
}

/// A latched `mbarrier` wait: (target, command, lanes, observed generation).
pub type LatchedWait = (ResourceId, crate::sync::SyncCmd, WarpMask, Option<u64>);
/// A deferred `cp.async.mbarrier.arrive` publication: (mbarrier, phase,
/// prior cp.async ops).
pub type DeferredPublish = (ResourceId, u64, Vec<AsyncId>);

/// Launch-wide CLC task queue (legacy `ClcTaskCounter`): the clusters of an
/// execution subset are the resident ones; `try_cancel` walks the logical
/// cluster ids once and hands each non-resident cluster's task to exactly
/// one caller (its linear base CTA id), then `u32::MAX` ("no cluster").
/// Without a subset every cluster is resident and nothing is claimable.
/// Shared by all partitions; claims are made only on the main arena (the
/// serial phase, or a single partition), so their order is deterministic.
#[derive(Debug, Default)]
pub struct ClcTasks {
    next: std::sync::atomic::AtomicU32,
    clusters: u32,
    ctas_per_cluster: u32,
    /// Resident clusters (sorted); `None` = all.
    resident: Option<Vec<u32>>,
}

impl ClcTasks {
    pub fn new(clusters: u32, ctas_per_cluster: u32, resident: Option<Vec<u32>>) -> Self {
        let resident = resident.map(|mut r| {
            r.sort_unstable();
            r.dedup();
            r
        });
        ClcTasks { next: Default::default(), clusters, ctas_per_cluster, resident }
    }

    /// Can a claim still return a task? (Otherwise every response is the
    /// "no cluster" sentinel and no serialization is needed.)
    pub fn claimable(&self) -> bool {
        let Some(r) = &self.resident else { return false };
        let mut t = self.next.load(std::sync::atomic::Ordering::Relaxed);
        while t < self.clusters {
            if r.binary_search(&t).is_err() {
                return true;
            }
            t += 1;
        }
        false
    }

    /// Claim the next non-resident cluster: its base CTA id, or `u32::MAX`.
    pub fn try_cancel(&self) -> u32 {
        let Some(r) = &self.resident else { return u32::MAX };
        loop {
            let t = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if t >= self.clusters {
                self.next.store(self.clusters, std::sync::atomic::Ordering::Relaxed);
                return u32::MAX;
            }
            if r.binary_search(&t).is_err() {
                return t.saturating_mul(self.ctas_per_cluster);
            }
        }
    }
}

/// All launch-wide engine bookkeeping.
#[derive(Clone, Debug, Default)]
pub struct LaunchAux {
    /// Launch-wide CLC task queue (shared by every partition).
    pub clc: std::sync::Arc<ClcTasks>,
    /// Kernel index within the `Module`.
    pub kernel: u32,
    /// `ValidityPolicy::ZeroAndReport` findings (one per read range).
    pub diagnostics: Vec<crate::report::Finding>,
    /// Dedupe key of reported uninitialized reads: (site, alloc, span).
    pub uninit_seen: HashSet<(crate::site::SiteId, AllocId, ByteSpan)>,
    /// Metadata-only register allocation per launch warp (`Space::Reg`).
    pub reg_allocs: Vec<AllocId>,
    pub groups: GroupTracker,
    pub async_meta: HashMap<AsyncId, AsyncMeta>,
    pub words: WordTable,
    /// Observer `wants_word_history` (cached).
    pub wants_history: bool,
    /// While evaluating a `WaitUntil` predicate: memory it read.
    pub capture_reads: Option<Vec<(AllocId, ByteSpan)>>,
    pub bar_red: HashMap<(CtaId, u8, u64), RedAcc>,
    /// `WaitUntil` verdict caches keyed by (warp, pc, alloc, word offset).
    pub verdicts: HashMap<(WarpId, crate::program::Pc, AllocId, u64), VerdictCache>,
    /// Warps (bitmask by warp-in-CTA) that completed an aligned full-warp
    /// `bar.sync` per `(cta, id, gen)` (setmaxnreg warpgroup sync credit).
    pub bar_aligned: HashMap<(CtaId, u8, u64), u64>,
    /// Warpgroup syncs already credited per `(cta, id, gen, wg)`.
    pub wg_credited: HashSet<(CtaId, u8, u64, u32)>,
    pub setmax: HashMap<(CtaId, u32), SetmaxRendezvous>,
    pub tcgen_pairs: HashMap<(CtaId, u8), PairRendezvous>,
    /// Last tcgen05 pipelined op issued by each CTA (pipeline order).
    pub tcgen_last: HashMap<CtaId, AsyncId>,
    /// Last pipelined tcgen05 op per issuing thread (`AsyncIssue.preds`).
    pub tcgen_last_thread: HashMap<(WarpId, u8), AsyncId>,
    /// tcgen05.mma collector buffer state per issuing thread (bit 0 = A,
    /// bits 1..5 = B buffers b0..b3; `oplib::tc_collector_transition`).
    pub tcgen_collectors: HashMap<(WarpId, u8), u8>,
    /// tcgen05 mma/cp ops per issuing thread issued since its last
    /// unrestricted commit.
    pub tcgen_uncommitted: HashMap<(WarpId, u8), Vec<AsyncId>>,
    /// tcgen05.ld/st ops not yet waited, per warp: (op, lanes, is_store).
    pub tcgen_ldst: HashMap<WarpId, Vec<(AsyncId, WarpMask, bool)>>,
    pub grid: GridBarrier,
    /// layout::v1 copy-report bits per (mbarrier, generation): OR of the
    /// inspections of copies completing on that phase (legacy
    /// `PhysicalBarrierEntry.phase.report`); read by report test/try_wait.
    pub mbar_reports: HashMap<(ResourceId, u64), bool>,
    /// Divergent `__syncwarp` rendezvous per warp: (lanes arrived, generation).
    pub warp_sync: HashMap<WarpId, (WarpMask, u64)>,
    /// Warps whose every lane exited.
    pub exited_warps: u32,
    /// Tensor maps modified by `tensormap.replace` and not yet published by
    /// a `fence.proxy.tensormap::generic.release` of the modifying warp,
    /// keyed by (allocation, offset) -> modifying warp.
    pub tmap_dirty: HashMap<(AllocId, u64), WarpId>,
    /// In-kernel publications of a tensor map in memory (generation per
    /// descriptor) and the generation each CTA acquired with
    /// `fence.proxy.tensormap::generic.acquire` (legacy "latest published
    /// generation is not acquired within this CTA").
    pub tmap_published: HashMap<(AllocId, u64), u64>,
    pub tmap_acquired: HashMap<(CtaId, AllocId, u64), u64>,
    /// Lane-varying `mbarrier.wait` targets that already completed while
    /// another target of the same instruction blocks, per (warp, pc):
    /// (target, command, lanes, observed generation). Those lanes left the
    /// wait (per-lane latching, as for `wait_until`).
    pub mbar_latch: HashMap<(WarpId, crate::program::Pc), Vec<LatchedWait>>,
    /// Exited warps per CTA (bit = warp in CTA): they leave the membership
    /// of count-less named barriers (sync-isa-answers Q3/Q4, PTX §9.7.14.7).
    pub cta_exited: HashMap<CtaId, u64>,
    /// Named-barrier generations opened by the count-less (whole-CTA) form,
    /// per `(cta, id)`: only those shrink when a member warp exits.
    pub named_implicit: HashMap<(CtaId, u8), u64>,
    /// Thread count a blocked `bar.sync`/`bar.red` registered with (its
    /// Protocol event, logged at completion, carries that contribution even
    /// if exits shrank the count-less barrier meanwhile).
    pub named_registered: HashMap<WarpId, u64>,
    /// Non-`.aligned` named-barrier arrival of a divergent warp in progress
    /// (lanes accumulate until every non-exited lane arrived; Q3 ruling).
    pub named_partial: HashMap<WarpId, NamedPartial>,
    /// Non-`.aligned` `barrier.cluster` arrive/wait executed by part of a
    /// warp (sync §4.6, Q11): the per-warp gather plus, per kind, the
    /// gathered lanes still to pass the instruction.
    pub cluster_partial: HashMap<WarpId, ClusterPartial>,
    /// Shared-A read ops (`MmaSharedARead`) per issuing thread that a
    /// `tcgen05.commit.sync_restrict` tracks (W5-10).
    pub tcgen_shared_reads: HashMap<(WarpId, u8), Vec<AsyncId>>,
    /// Ops already tracked by an earlier (unrestricted / restricted) commit
    /// of the thread that may still be in flight (pruned when they land).
    pub tcgen_inflight: HashMap<(WarpId, u8), Vec<AsyncId>>,
    pub tcgen_inflight_shared: HashMap<(WarpId, u8), Vec<AsyncId>>,
    /// cp.async ops per issuing thread not yet covered by a
    /// `cp.async.mbarrier.arrive` (W5-11).
    pub cp_async_unpublished: HashMap<(WarpId, u8), Vec<AsyncId>>,
    /// Deferred `cp.async.mbarrier.arrive`s per (group, ordinal): the
    /// (mbarrier, phase, prior cp.async ops) published when it fires.
    pub cp_arrive_publish: HashMap<(ResourceId, u64), Vec<DeferredPublish>, FxBuild>,
    /// Next async op id (partition-scoped: high bits name the partition).
    pub next_async: u64,
    /// Set by a handler that must run as a serial point (a global
    /// read-modify-write inside an arena shard); the scheduler re-runs the
    /// instruction after the round's merge.
    pub serial_request: bool,
}

impl LaunchAux {
    pub fn next_async_id(&mut self) -> AsyncId {
        let id = AsyncId(self.next_async);
        self.next_async += 1;
        id
    }

}
