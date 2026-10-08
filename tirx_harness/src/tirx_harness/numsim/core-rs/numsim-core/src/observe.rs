//! The observation interface between the engine and the checkers.
//!
//! Two representations, never unified (plan section 1):
//! * [`Access`]: hot, *borrowed*, built on the stack by memory handlers;
//!   one per (warp instruction or async-op side, allocation).
//! * [`SyncEvent`]: cold, *owned*; everything else (protocol commands,
//!   arrive/wait phases, fences, async issue/complete, allocation lifetime,
//!   declared words, wait verdicts).
//!
//! Consumers: Racecheck (W5, `numsim-race-core/src/input.rs`) and the
//! Synccheck explorer (W6, `numsim-sync-explore/src/event.rs`). Both map
//! onto these types with a thin adapter; name mappings are noted inline.
//!
//! # Delivery order (one total order per run)
//!
//! * Per actor, program order. Every *warp instruction* gets a fresh
//!   `epoch` (strictly increasing per warp); its sibling lanes share it and
//!   are simultaneous.
//! * `SyncKind::AsyncIssue{op}` precedes every access by `Actor::Async{op,..}`,
//!   which precede `AsyncComplete{op, milestone: Write}`.
//! * A `Protocol` event is delivered after the transition committed.
//!   Blocked attempts are *not* delivered (`seq` counts committed commands
//!   only); a warp still blocked at launch end gets one final event with
//!   `ProtocolStatus::BlockedAtExit`.
//! * Declared-word history (README decision 14): index 0 = value before the
//!   launch (or at declaration); index i >= 1 = the i-th delivered
//!   (write `Access`, lane) pair with `declared_word` overlapping the word,
//!   in delivery order with lanes ascending inside one `Access`; the value
//!   is the byte-merged post-image of that lane's write. Engine and checker
//!   number the same stream, per word.
//!
//! Observers must never change program-visible behaviour. Handlers check
//! [`Observer::enabled`] before building anything ([`NoopObserver`] for NumSim).

use crate::arena::{AllocId, Arena, ByteSpan, Space};
use crate::program::{LaunchShape, Program, Proxy, Scope, Sem};
use crate::site::SiteId;
use crate::sync::completion::{AsyncId, ResourceId};
use crate::sync::{SyncCmd, SyncError};
use crate::value::LaneMask;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Launch-global warp index: `cta * warps_per_cta + warp_in_cta`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WarpId(pub u32);

/// Launch-global linear CTA index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CtaId(pub u32);

/// Which side of an async op (W5 `Milestone`): read-side accesses are
/// published at `Read`, write-side at `Write`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Side {
    Read,
    Write,
}

/// Who performed an action (W5 `Who`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Actor {
    /// A warp instruction; `epoch` is per instruction (lanes in the event's
    /// lane set/spans).
    Warp { warp: WarpId, epoch: u64 },
    /// The virtual actor of an async op (TMA, cp.async(.bulk), st.async,
    /// tcgen05.*), and which side of it.
    Async { op: AsyncId, side: Side },
    /// Host initialization / readback.
    Host,
}

impl fmt::Display for Actor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Actor::Warp { warp, epoch } => write!(f, "warp{}#{epoch}", warp.0),
            Actor::Async { op, side } => write!(f, "async{}.{side:?}", op.0),
            Actor::Host => f.write_str("host"),
        }
    }
}

/// Launch-global delivery index of an `Access`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AccessSeq(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessKind {
    Read,
    Write,
    /// atom / red; conflicts as a write.
    Rmw,
}

/// The address window an access was formed in (W5 `Domain`; renamed to
/// avoid clashing with `async_group::Domain`). `None` for TMEM / local / param.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Window {
    Global,
    SharedCta,
    SharedCluster,
}

/// One lane's byte range. For async actors `lane` is the ISSUING lane of
/// that per-thread async op (async groups and copies are per thread, PTX ISA
/// §9.7.10.28); [`ALL_LANES`] is permitted only for genuinely warp-collective
/// accesses (ldmatrix/stmatrix, tcgen05.ld/st 32x32b, tile ops).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LaneSpan {
    pub lane: u8,
    pub span: ByteSpan,
}

pub const ALL_LANES: u8 = u8::MAX;

/// A batch of byte accesses by one warp instruction (or one async-op side)
/// to one allocation. W5's per-lane `Access` = one item of [`Access::per_lane`].
#[derive(Clone, Copy, Debug)]
pub struct Access<'a> {
    pub seq: AccessSeq,
    pub actor: Actor,
    pub site: SiteId,
    pub alloc: AllocId,
    pub space: Space,
    pub kind: AccessKind,
    /// Memory order (W5 `order`): Weak/Relaxed/Acquire/Release/AcqRel;
    /// `Sc` = fence.sc + op, `Volatile`/`Mmio` = relaxed.sys.
    pub sem: Sem,
    /// Meaningful when `atomic || sem != Weak` (W5 `scope: Option`).
    pub scope: Scope,
    pub atomic: bool,
    /// For RMW accesses: `true` for `atom` (returns the old value and can
    /// form an acquire pattern), `false` for `red` (never an acquire
    /// pattern, PTX ISA §8.8). Ignored for plain loads and stores.
    pub returns_value: bool,
    pub proxy: Proxy,
    pub window: Option<Window>,
    /// Allocation-relative ranges, sorted by (lane, start); lanes are the
    /// active lanes that touched memory.
    pub spans: &'a [LaneSpan],
    /// At least one span overlaps a declared word (only when the observer
    /// `wants_word_history`).
    pub declared_word: bool,
}

impl<'a> Access<'a> {
    pub fn writes(&self) -> bool {
        matches!(self.kind, AccessKind::Write | AccessKind::Rmw)
    }
    /// Per-lane view: `(lane, range)`.
    pub fn per_lane(&self) -> impl Iterator<Item = (u8, ByteSpan)> + 'a {
        self.spans.iter().map(|s| (s.lane, s.span))
    }
}

/// One enclosing loop: its `LoopBegin` site and current iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LoopFrame {
    pub site: SiteId,
    pub iteration: u64,
}

/// Explicit counts (W6 point 2): consumers never derive them from masks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Counts {
    /// mbarrier arrival count.
    pub arrivals: Option<u64>,
    /// expect_tx / complete_tx bytes.
    pub tx_bytes: Option<u64>,
    /// Named barrier: expected thread count `b` of the generation.
    pub expected_threads: Option<u64>,
    /// Named barrier: threads this warp contributed.
    pub contributed_threads: Option<u64>,
    /// Cluster barrier: participant warps per generation.
    pub participants: Option<u32>,
}

/// One target of a committed protocol instruction.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProtocolCmd {
    pub res: ResourceId,
    pub cmd: SyncCmd,
    pub counts: Counts,
    /// Successful `test_wait`/`try_wait` on `res`: the parity observed (W6
    /// point 8). Failed polls are never logged.
    pub observed_parity: Option<u8>,
}

/// Predicate verdicts for one group of lanes that accepted the same write.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LaneVerdict {
    pub lanes: LaneMask,
    pub accepted: Vec<u64>,
    /// Index of the history entry whose value these lanes observed.
    pub observed: u32,
}

/// A warp-collective rendezvous (setmaxnreg warpgroup, cta_group::2 tcgen).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Collective {
    /// Launch-unique id of this rendezvous instance.
    pub id: u64,
    pub participants: Vec<WarpId>,
}

/// An async completion promised at issue (W6 point 4); no generations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AsyncTarget {
    pub res: ResourceId,
    /// complete_tx bytes delivered to `res` (0 = none).
    pub bytes: u64,
    /// Arrivals performed at completion (cp.async.mbarrier.arrive,
    /// tcgen05.commit); 0 = none.
    pub arrivals: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProtocolStatus {
    Committed,
    Failed(SyncError),
    /// Final event of a warp still blocked when the launch ended.
    BlockedAtExit,
}

/// W5 async-op class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AsyncClass {
    /// TMA, cp.async(.bulk), st.async, CLC response, multicast legs.
    Copy,
    /// tcgen05.mma / cp / shift.
    TcgenPipelined,
    TcgenLd,
    TcgenSt,
    TcgenCommit,
}

/// Where an async milestone is published (W5 `CompletionTarget`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PublishTarget {
    /// Joins the phase clock of `obj` (mbarrier complete_tx / commit arrive).
    Phase { obj: ResourceId, phase: u64 },
    /// Observed directly by the issuing warp (wait_group, tcgen05.wait).
    Warp { warp: WarpId, lanes: LaneMask },
}

/// Fence as checkers see it (W5 `FenceKind` plus the remaining PTX kinds).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FenceEvent {
    AcqRel(Scope),
    /// `fence.sc` and `membar`.
    Sc(Scope),
    /// `fence.proxy.async[.window]`; None = all windows.
    ProxyAsync(Option<Window>),
    TcgenBefore,
    TcgenAfter,
    MbarrierInit,
    ProxyAlias,
    /// `fence.proxy.tensormap::generic.release.<scope>` (W5-4).
    TensormapRelease { scope: Scope },
    /// `fence.proxy.tensormap::generic.acquire.<scope> [addr], size` (W5-4).
    TensormapAcquire { scope: Scope, alloc: AllocId, span: ByteSpan },
}

/// Payload of a [`SyncEvent`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SyncKind {
    // ---- lifetime / declarations (W5) ----
    AllocBegin { alloc: AllocId, space: Space, size: u64, cta: CtaId },
    /// End of an allocation's lifetime (smem reuse, TMEM dealloc).
    AllocEnd { alloc: AllocId },
    DeclareWord { alloc: AllocId, span: ByteSpan },

    // ---- protocol commands (W6) ----
    /// One committed instruction's protocol commands: all targets of an
    /// atomic multi-resource instruction (W6 point 5) in one event.
    Protocol {
        /// One entry per target resource; counts and observed parity are
        /// per target so lane-varying wait batches carry per-barrier values
        /// (contract review item 4).
        cmds: Vec<ProtocolCmd>,
        collective: Option<Collective>,
        /// Async completions this instruction issued (TMA, commit, ...).
        issued: Vec<AsyncTarget>,
        status: ProtocolStatus,
    },

    // ---- happens-before view (W5) ----
    /// `__syncwarp(mask)` and warp collectives.
    WarpSync { mask: LaneMask },
    /// Arrive into `phase` of `obj` (bar.arrive/sync, mbarrier arrive,
    /// cluster arrive). `release = false` for `.relaxed`.
    /// `release`/`scope` are `None` when lowering lost the qualifier; the
    /// checker then reports `incomplete`, never assumes relaxed (ISA R5).
    Arrive { obj: ResourceId, phase: u64, release: Option<bool>, scope: Option<Scope> },
    /// Observed completed `phase` of `obj` (bar wait, successful mbarrier
    /// wait/test, cluster wait). Named barriers use `scope: None`.
    Wait { obj: ResourceId, phase: u64, acquire: Option<bool>, scope: Option<Scope> },
    Fence(FenceEvent),
    AsyncIssue {
        op: AsyncId,
        class: AsyncClass,
        proxy: Proxy,
        /// Architected-pipeline predecessors; for TcgenCommit the tracked ops.
        preds: Vec<AsyncId>,
        /// Lifetime footprint.
        footprint: Vec<(AllocId, ByteSpan)>,
        /// W6 view of the same issue.
        targets: Vec<AsyncTarget>,
    },
    AsyncComplete { op: AsyncId, milestone: Side, target: PublishTarget },
    /// A `WaitUntil` succeeded for the lanes in `verdicts` (plan 2.5).
    /// Verdicts are PER LANE GROUP: lanes that accepted different writes get
    /// separate entries, never a lane-wise conjunction (contract review item 6).
    /// `accepted` bit 0 = launch value, bit i = i-th write in delivery order
    /// of the word's history as of this wait; the engine evaluates the
    /// predicate incrementally (only writes newer than the last verdict).
    WaitVerdicts {
        alloc: AllocId,
        span: ByteSpan,
        scope: Scope,
        verdicts: Vec<LaneVerdict>,
        /// Memory the predicate sub-program read (`PredProgram::reads_memory`).
        pred_reads: Vec<(AllocId, ByteSpan)>,
    },
}

/// One cold-path event.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SyncEvent {
    /// Kernel index within the `Module` (W6-1 item 5).
    pub kernel: u32,
    pub actor: Actor,
    /// Per-warp sequence of committed `Protocol` events (W6 cursor);
    /// for other kinds, the warp's next protocol seq (not incremented).
    pub seq: u32,
    pub site: SiteId,
    /// Enclosing loops, outermost first (W6 point 7).
    pub frames: Vec<LoopFrame>,
    pub lanes: LaneMask,
    pub kind: SyncKind,
}

/// Why a warp stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarpEnd {
    Exited,
    Trapped,
    Error,
    Deadlocked,
    Budget,
}

/// Launch context for `begin_launch`/`end_launch`.
#[derive(Clone, Copy)]
pub struct LaunchInfo<'a> {
    pub program: &'a Program,
    pub kernel_index: u32,
    pub shape: LaunchShape,
    pub arena: &'a Arena,
}

/// The observer interface (object safe; the engine holds `&mut dyn Observer`).
pub trait Observer {
    /// False = handlers skip building events entirely.
    fn enabled(&self) -> bool {
        true
    }
    /// True = engine keeps declared-word histories, sets
    /// `Access::declared_word` and emits `WaitVerdicts`.
    fn wants_word_history(&self) -> bool {
        false
    }
    fn begin_launch(&mut self, _info: &LaunchInfo<'_>) {}
    fn end_launch(&mut self, _info: &LaunchInfo<'_>) {}
    fn access(&mut self, _a: &Access<'_>) {}
    fn sync(&mut self, _e: &SyncEvent) {}
    fn warp_done(&mut self, _warp: WarpId, _end: WarpEnd) {}
    /// Cross-CTA effects became visible to `cta` (inbox drained).
    fn inbox_drain(&mut self, _cta: CtaId, _round: u64) {}
}

/// NumSim's observer.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopObserver;

impl Observer for NoopObserver {
    fn enabled(&self) -> bool {
        false
    }
}

/// Records every [`SyncEvent`] per warp (indexed by `WarpId.0`) plus
/// non-warp events in delivery order. Synccheck's input.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingObserver {
    pub per_warp: Vec<Vec<SyncEvent>>,
    pub other: Vec<SyncEvent>,
    /// Total delivery order as (warp index or u32::MAX, index in its list).
    pub order: Vec<(u32, u32)>,
}

impl RecordingObserver {
    pub fn new() -> RecordingObserver {
        RecordingObserver::default()
    }
    pub fn total(&self) -> usize {
        self.order.len()
    }
}

impl Observer for RecordingObserver {
    fn sync(&mut self, e: &SyncEvent) {
        match e.actor {
            Actor::Warp { warp, .. } => {
                let i = warp.0 as usize;
                if self.per_warp.len() <= i {
                    self.per_warp.resize_with(i + 1, Vec::new);
                }
                self.order.push((warp.0, self.per_warp[i].len() as u32));
                self.per_warp[i].push(e.clone());
            }
            _ => {
                self.order.push((u32::MAX, self.other.len() as u32));
                self.other.push(e.clone());
            }
        }
    }
}

/// Fan-out to two observers.
impl<A: Observer, B: Observer> Observer for (A, B) {
    fn enabled(&self) -> bool {
        self.0.enabled() || self.1.enabled()
    }
    fn wants_word_history(&self) -> bool {
        self.0.wants_word_history() || self.1.wants_word_history()
    }
    fn begin_launch(&mut self, i: &LaunchInfo<'_>) {
        self.0.begin_launch(i);
        self.1.begin_launch(i);
    }
    fn end_launch(&mut self, i: &LaunchInfo<'_>) {
        self.0.end_launch(i);
        self.1.end_launch(i);
    }
    fn access(&mut self, a: &Access<'_>) {
        self.0.access(a);
        self.1.access(a);
    }
    fn sync(&mut self, e: &SyncEvent) {
        self.0.sync(e);
        self.1.sync(e);
    }
    fn warp_done(&mut self, w: WarpId, end: WarpEnd) {
        self.0.warp_done(w, end);
        self.1.warp_done(w, end);
    }
    fn inbox_drain(&mut self, c: CtaId, r: u64) {
        self.0.inbox_drain(c, r);
        self.1.inbox_drain(c, r);
    }
}
