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
use crate::sync::{AsyncId, Completion, ResourceId};
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
}

/// Per-lane async-group membership of in-flight async ops.
#[derive(Clone, Debug, Default)]
pub struct GroupTracker {
    /// Issued but not yet committed ops, per `AsyncGroup` resource.
    pub open: HashMap<ResourceId, Vec<AsyncId>>,
    /// Ops of each committed group (all, landed or not).
    pub members: HashMap<(ResourceId, u64), Vec<AsyncId>>,
    /// Not-yet-landed ops per committed group.
    pub pending: HashMap<(ResourceId, u64), u32>,
    /// Groups each in-flight op still has to report its landing to.
    pub op_groups: HashMap<AsyncId, Vec<(ResourceId, u64)>>,
    /// Ops that landed while still uncommitted.
    pub landed_open: HashSet<AsyncId>,
    /// Deferred mbarrier arrive-ons released by a group's full completion.
    pub arrivals: HashMap<(ResourceId, u64), Vec<Completion>>,
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
}

/// History entries beyond this make verdicts truncated (the oldest entries
/// are kept; `observed` saturates).
pub const MAX_WORD_HISTORY: usize = 1 << 16;

/// Declared words (`BufferDecl::sync_words`, or declared at first wait).
#[derive(Clone, Debug, Default)]
pub struct WordTable {
    pub regions: HashMap<AllocId, Vec<WordRegion>>,
}

impl WordTable {
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// Does any span overlap a declared region of `alloc`?
    pub fn overlaps(&self, alloc: AllocId, spans: &[ByteSpan]) -> bool {
        self.regions
            .get(&alloc)
            .is_some_and(|rs| rs.iter().any(|r| spans.iter().any(|s| s.overlaps(r.span))))
    }

    /// Region of `alloc` covering `span`, if declared.
    pub fn region(&self, alloc: AllocId, span: ByteSpan) -> Option<&WordRegion> {
        self.regions
            .get(&alloc)?
            .iter()
            .find(|r| r.span.start <= span.start && span.end() <= r.span.end())
    }

    /// Declare `span` of `alloc` with its current bytes as history index 0.
    pub fn declare(&mut self, arena: &Arena, alloc: AllocId, span: ByteSpan) {
        let init = snapshot(arena, alloc, span);
        self.regions.entry(alloc).or_default().push(WordRegion { span, init, log: Vec::new() });
    }

    /// Append one history entry per declared region overlapping a lane's
    /// write of `bytes` at `span` (allocation-relative): the region's last
    /// image with this lane's bytes merged in (README decision 14: entries
    /// are (write Access, lane) pairs in delivery order, lanes ascending).
    pub fn log_lane(&mut self, alloc: AllocId, span: ByteSpan, bytes: &[u8]) {
        let Some(rs) = self.regions.get_mut(&alloc) else { return };
        for r in rs.iter_mut() {
            if !span.overlaps(r.span) || r.log.len() >= MAX_WORD_HISTORY {
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
    }

    /// [`Self::log_lane`] with the bytes currently in the arena (single
    /// writer of `span`, e.g. an async landing).
    pub fn log_from_arena(&mut self, arena: &Arena, alloc: AllocId, span: ByteSpan) {
        if !self.regions.get(&alloc).is_some_and(|rs| rs.iter().any(|r| r.span.overlaps(span))) {
            return;
        }
        let bytes = snapshot(arena, alloc, span);
        self.log_lane(alloc, span, &bytes);
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
    let a = arena.get(alloc);
    let end = (span.end() as usize).min(a.bytes.len());
    let start = (span.start as usize).min(end);
    let mut v = a.bytes[start..end].to_vec();
    v.resize(span.len as usize, 0);
    v
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
}

/// Cooperative grid barrier.
#[derive(Clone, Debug, Default)]
pub struct GridBarrier {
    pub gen: u64,
    pub arrived: HashSet<WarpId>,
}

/// All launch-wide engine bookkeeping.
#[derive(Clone, Debug, Default)]
pub struct LaunchAux {
    /// Kernel index within the `Module`.
    pub kernel: u32,
    /// `ValidityPolicy::ZeroAndReport` findings (one per read range).
    pub diagnostics: Vec<crate::report::Finding>,
    /// Dedupe key of reported uninitialized reads: (site, alloc, span).
    pub uninit_seen: HashSet<(crate::site::SiteId, AllocId, ByteSpan)>,
    /// Metadata-only register allocation per launch warp (`Space::Reg`).
    pub reg_allocs: Vec<AllocId>,
    /// Owning CTA of every CTA-private allocation (inbox routing).
    pub owner_cta: HashMap<AllocId, CtaId>,
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
    /// Uncommitted tcgen05 mma/cp ops per issuing thread.
    pub tcgen_uncommitted: HashMap<(WarpId, u8), Vec<AsyncId>>,
    /// tcgen05.ld/st ops not yet waited, per warp: (op, lanes, is_store).
    pub tcgen_ldst: HashMap<WarpId, Vec<(AsyncId, WarpMask, bool)>>,
    pub grid: GridBarrier,
    /// Warps whose every lane exited.
    pub exited_warps: u32,
    /// Next collective instance id.
    pub next_collective: u64,
}

impl LaunchAux {
    pub fn next_collective_id(&mut self) -> u64 {
        let id = self.next_collective;
        self.next_collective += 1;
        id
    }
}
