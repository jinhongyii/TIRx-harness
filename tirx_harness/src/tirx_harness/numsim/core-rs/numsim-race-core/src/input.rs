//! The checker's input shape — the ONLY file that defines it.
//!
//! This is a local stand-in for the `numsim-core` contract (`Access` on the
//! hot path, `SyncEvent` on the cold path, plan section 1). When the contract
//! lands, this file becomes a thin adapter and nothing else in the crate
//! changes: the core only reads the fields named here.
//!
//! Conventions every producer must honour:
//! * Events are delivered in one total order per checker instance (the
//!   scheduler's order). Within a warp, accesses of one dynamic instruction are
//!   delivered with the same `epoch` and differing `lane`; the next instruction
//!   of that warp uses a strictly larger epoch. Sibling lanes of one
//!   instruction are simultaneous (never ordered with each other).
//! * Async operations (TMA, cp.async, cp.async.bulk, tcgen05.*) are virtual
//!   actors: `AsyncIssue` creates one, their memory accesses carry
//!   `Who::Async`, and `AsyncComplete` publishes one of their two milestones.
//! * Declared-word history indices: index 0 is the word's value before the
//!   launch; index `i >= 1` is the `i`-th checker-observed write overlapping
//!   the declared word, in delivery order. Engine and checker number the same
//!   stream, so the engine's verdict bitset is directly interpretable.

use std::ops::Range;

/// Launch-global warp id (`cta * warps_per_cta + warp_in_cta`).
pub type WarpId = u32;
/// Producer-chosen id of one async operation instance (unique per launch).
pub type AsyncId = u32;
/// Synchronisation object (mbarrier phase owner, named barrier, cluster barrier,
/// cp.async group, ...). Its phase numbering is the SyncTable's business; the
/// checker only needs "arrive into phase p" and "wait for phase p".
pub type SyncObjId = u32;
pub type AllocId = u32;
pub type SiteId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LaneMask(pub u32);

impl LaneMask {
    pub const FULL: LaneMask = LaneMask(u32::MAX);
    pub const fn lane(l: u8) -> LaneMask {
        LaneMask(1 << l)
    }
    pub fn lanes(self) -> impl Iterator<Item = u8> {
        (0..32u8).filter(move |l| self.0 & (1 << l) != 0)
    }
    pub fn is_full(self) -> bool {
        self.0 == u32::MAX
    }
    pub fn contains(self, lane: u8) -> bool {
        self.0 & (1 << lane) != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Space {
    Shared,
    Global,
    Tmem,
}

/// Proxy memory domain, the granularity of `fence.proxy.async.<domain>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Domain {
    Global = 0,
    SharedCta = 1,
    SharedCluster = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Scope {
    Cta = 0,
    Cluster = 1,
    Gpu = 2,
    Sys = 3,
}

/// The proxy an access is performed through. TMEM is only touched through
/// `Tcgen`; TMA / bulk copies / tcgen05.mma operand reads use `Async`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Proxy {
    Generic = 0,
    Async = 1,
    Tcgen = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccessKind {
    Read,
    Write,
    /// Read-modify-write (atom / red). Conflicts as a write.
    Rmw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemOrder {
    /// `.weak` / plain ld/st (no scope).
    Weak,
    Relaxed,
    Acquire,
    Release,
    AcqRel,
}

/// Which side of an async op performed an access. Read-side accesses are
/// stamped at milestone 1, write-side at milestone 2, so a read-only
/// completion (`cp.async.bulk.wait_group.read`) orders exactly the reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Milestone {
    Read = 1,
    Write = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Who {
    Lane { warp: WarpId, lane: u8, epoch: u32 },
    Async { op: AsyncId, side: Milestone },
}

/// Hot path: one lane's (or one async op's) contiguous byte footprint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Access {
    pub who: Who,
    pub alloc: AllocId,
    pub range: Range<u64>,
    pub kind: AccessKind,
    pub order: MemOrder,
    /// `Some(scope)` for strong accesses (scoped relaxed/acquire/release and
    /// all atomics; atomics default to `.gpu`).
    pub scope: Option<Scope>,
    pub atomic: bool,
    pub proxy: Proxy,
    /// The window the address was formed in (`shared::cta`, a mapa'd
    /// `shared::cluster` window, or global). `None` for TMEM.
    pub domain: Option<Domain>,
    pub site: SiteId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenceKind {
    /// `fence.acq_rel.<scope>`.
    AcqRel(Scope),
    /// `fence.sc.<scope>` and `membar.{cta,gl,sys}`.
    Sc(Scope),
    /// `fence.proxy.async[.<domain>]`; `None` covers every domain.
    ProxyAsync(Option<Domain>),
    /// `tcgen05.fence::before_thread_sync`.
    TcgenBefore,
    /// `tcgen05.fence::after_thread_sync`.
    TcgenAfter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsyncKind {
    /// TMA, cp.async.bulk[.tensor], cp.async, CLC try_cancel, multicast legs.
    Copy,
    /// tcgen05.mma / cp / shift: ordered by `fence::before_thread_sync` and
    /// by commit, without waiting.
    TcgenPipelined,
    /// tcgen05.ld: ordered only by `tcgen05.wait::ld` (or commit).
    TcgenLd,
    /// tcgen05.st: ordered only by `tcgen05.wait::st`.
    TcgenSt,
    /// tcgen05.commit: its completion (mbarrier arrive) publishes the
    /// issuer's generic knowledge and the tracked ops with their causal
    /// predecessors.
    TcgenCommit,
}

/// Where an async milestone is published.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionTarget {
    /// mbarrier complete_tx / tcgen05.commit → mbarrier arrive: the
    /// milestone joins the phase clock that waiters acquire.
    Phase { obj: SyncObjId, phase: u32 },
    /// The issuing warp observes completion directly (cp.async.wait_group,
    /// cp.async.bulk.wait_group[.read], tcgen05.wait::ld/st).
    Warp { warp: WarpId, lanes: LaneMask },
}

/// Cold path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SyncEvent {
    AllocBegin { alloc: AllocId, space: Space, size: u64, cta: u32 },
    /// End of an allocation's lifetime (smem reuse boundary, TMEM dealloc).
    AllocEnd { alloc: AllocId, site: SiteId },
    /// `__syncwarp(mask)` and warp collectives (shfl, vote, ldmatrix, ...).
    WarpSync { warp: WarpId, mask: LaneMask, epoch: u32 },
    /// Arrive into a phase.
    /// * `release`: `Some(true)` for release arrives (mbarrier default,
    ///   `bar.*`, `barrier.cluster.arrive` default), `Some(false)` for
    ///   `.relaxed` (carries only the tcgen05 fence frontier), `None` when
    ///   lowering lost the qualifier — reported `incomplete`, never assumed
    ///   relaxed.
    /// * `scope`: `Some(s)` for mbarrier (default `.cta`) and cluster
    ///   barrier (`.cluster`); `None` for named barriers, which synchronise
    ///   their participants without a scope (PTX §9.7.15.1).
    Arrive { warp: WarpId, lanes: LaneMask, obj: SyncObjId, phase: u32, release: Option<bool>, scope: Option<Scope>, epoch: u32 },
    /// Observe a completed phase (bar.sync/red, successful mbarrier
    /// try/test wait, barrier.cluster.wait). `acquire == Some(false)`
    /// (`.relaxed`) synchronises nothing by itself: arrivals and async
    /// completions are parked until a later `fence.acquire`/`acq_rel`
    /// (PTX §8.8). A `bar.arrive`-only thread emits no `Wait`.
    Wait { warp: WarpId, lanes: LaneMask, obj: SyncObjId, phase: u32, acquire: Option<bool>, scope: Option<Scope>, epoch: u32 },
    Fence { warp: WarpId, lanes: LaneMask, kind: FenceKind, epoch: u32 },
    AsyncIssue {
        op: AsyncId,
        warp: WarpId,
        lanes: LaneMask,
        kind: AsyncKind,
        /// Proxy of the op's memory accesses (cp.async is `Generic`; TMA and
        /// bulk copies `Async`; TMEM accesses of tcgen05 ops `Tcgen`).
        proxy: Proxy,
        /// Architected-pipeline predecessors (PTX 9.7.17.6.2 pairs, e.g.
        /// mma→mma of one class), resolved by the producer. For
        /// `TcgenCommit`, the ops the commit tracks.
        preds: Vec<AsyncId>,
        /// Lifetime footprint, checked against `AllocEnd`.
        footprint: Vec<(AllocId, Range<u64>)>,
        epoch: u32,
    },
    AsyncComplete { op: AsyncId, milestone: Milestone, target: CompletionTarget },
    /// A declared-word wait (`wait_until`) succeeded. `accepted` is the
    /// predicate verdict bitset over the word's write history (see module
    /// doc); bit `i` set means history entry `i` satisfies the predicate.
    WaitVerdicts {
        warp: WarpId,
        lanes: LaneMask,
        alloc: AllocId,
        range: Range<u64>,
        scope: Scope,
        accepted: Vec<u64>,
        /// The history index the run actually observed (only used for the
        /// async-publication fallback).
        observed: u32,
        /// Other memory the predicate reads (a lowered PredProgram flagged
        /// `reads_memory`). The verdict bitset is only meaningful if those
        /// bytes were stable for the whole wait: every write to them must
        /// happen before the wait, else the wait is `incomplete`.
        pred_reads: Vec<(AllocId, Range<u64>)>,
        epoch: u32,
    },
    /// Marks a byte range as a declared word (history is recorded for it).
    DeclareWord { alloc: AllocId, range: Range<u64> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Topology {
    pub warps_per_cta: u32,
    pub ctas_per_cluster: u32,
    pub num_ctas: u32,
}

impl Topology {
    pub fn num_warps(&self) -> u32 {
        self.warps_per_cta * self.num_ctas
    }
    pub fn cta_of(&self, warp: WarpId) -> u32 {
        warp / self.warps_per_cta
    }
    pub fn cluster_of(&self, warp: WarpId) -> u32 {
        self.cta_of(warp) / self.ctas_per_cluster
    }
}

/// Everything the checker consumes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    Access(Access),
    Sync(SyncEvent),
}
