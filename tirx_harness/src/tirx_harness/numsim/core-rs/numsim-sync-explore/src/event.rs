//! The local `SyncEvent` shape this explorer consumes.
//!
//! This is the *only* input of the offline explorer. Everything the old
//! `FixedSyncProgram` reconstructed from `ResolvedTransitionLog` summaries,
//! completion hints, setmax/tcgen side tables and recorded vector clocks is
//! reduced here to: who (warp, per-warp sequence, static site), which resource,
//! which protocol operation with which counts. Clocks and generations are
//! recomputed by [`crate::program::reference_run`], so the observer does not
//! have to record them.
//!
//! When `numsim-core` publishes its `SyncEvent`, this module becomes a
//! `From<numsim_core::SyncEvent>` adapter.

/// Global warp id inside one launch.
pub type WarpId = u32;

/// Stable identity of one executed synchronization operation.
///
/// Stand-in for today's `DynamicOpId` (kernel index, global warp id,
/// per-warp sequence, static op id, loop frames). Loop frames are omitted:
/// they only matter for reporting and for the named-barrier "same static
/// instruction" check, which uses `site`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpId {
    pub warp: WarpId,
    /// Position in the warp's executed synchronization sequence.
    pub seq: u32,
    /// Static source site (TIR op id) for reporting.
    pub site: u32,
}

/// A synchronization resource. Raw ids are opaque; equal ids name the same
/// physical object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceId {
    /// Physical mbarrier (allocation, byte offset, target CTA) collapsed to one id.
    Mbarrier(u32),
    /// `bar.sync` / `bar.arrive` id within one CTA.
    Named(u32),
    /// `barrier.cluster` of one cluster.
    Cluster(u32),
}

/// One synchronization operation as executed by one warp.
///
/// Counts are in *threads* (lanes) for arrivals, bytes for transactions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SyncOp {
    /// `mbarrier.init` with the expected arrival count.
    MbarInit { bar: u32, expected: u32 },
    /// `mbarrier.arrive[.expect_tx]`: `count` arrivals, optional expected bytes.
    MbarArrive { bar: u32, count: u32, expect_tx: u32 },
    /// `mbarrier.expect_tx` without arrival.
    MbarExpectTx { bar: u32, tx: u32 },
    /// Issue of an async producer (TMA, cp.async.bulk, tcgen05.commit) whose
    /// later completion delivers `tx` bytes to the generation that is current
    /// at issue. The completion is a separate, independently schedulable
    /// transition.
    MbarTxIssue { bar: u32, tx: u32 },
    /// Blocking parity wait (`mbarrier.try_wait.parity` loop).
    MbarWait { bar: u32, parity: u8 },
    /// `bar.arrive id, expected` contributing `count` threads.
    NamedArrive { bar: u32, expected: u32, count: u32 },
    /// `bar.sync id, expected` contributing `count` threads and blocking.
    NamedSync { bar: u32, expected: u32, count: u32 },
    /// `barrier.cluster.arrive` by a full warp; `participants` warps per generation.
    ClusterArrive { bar: u32, participants: u32 },
    /// `barrier.cluster.wait` by a full warp.
    ClusterWait { bar: u32 },
}

impl SyncOp {
    pub const fn resource(&self) -> ResourceId {
        match *self {
            Self::MbarInit { bar, .. }
            | Self::MbarArrive { bar, .. }
            | Self::MbarExpectTx { bar, .. }
            | Self::MbarTxIssue { bar, .. }
            | Self::MbarWait { bar, .. } => ResourceId::Mbarrier(bar),
            Self::NamedArrive { bar, .. } | Self::NamedSync { bar, .. } => ResourceId::Named(bar),
            Self::ClusterArrive { bar, .. } | Self::ClusterWait { bar } => ResourceId::Cluster(bar),
        }
    }

    pub const fn name(&self) -> &'static str {
        match self {
            Self::MbarInit { .. } => "mbarrier.init",
            Self::MbarArrive { .. } => "mbarrier.arrive",
            Self::MbarExpectTx { .. } => "mbarrier.expect_tx",
            Self::MbarTxIssue { .. } => "mbarrier.completion_issue",
            Self::MbarWait { .. } => "mbarrier.wait",
            Self::NamedArrive { .. } => "bar.arrive.register",
            Self::NamedSync { .. } => "bar.sync.register",
            Self::ClusterArrive { .. } => "barrier.cluster.arrive",
            Self::ClusterWait { .. } => "barrier.cluster.wait.register",
        }
    }
}

/// One record of the observer's synchronization log.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SyncEvent {
    pub op: OpId,
    pub kind: SyncOp,
}

impl SyncEvent {
    pub const fn new(warp: WarpId, seq: u32, site: u32, kind: SyncOp) -> Self {
        Self {
            op: OpId { warp, seq, site },
            kind,
        }
    }
}

/// Small builder that assigns per-warp sequence numbers, for tests and benches.
#[derive(Default)]
pub struct LogBuilder {
    events: Vec<SyncEvent>,
    next_seq: std::collections::BTreeMap<WarpId, u32>,
}

impl LogBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, warp: WarpId, site: u32, kind: SyncOp) -> &mut Self {
        let seq = self.next_seq.entry(warp).or_insert(0);
        self.events.push(SyncEvent::new(warp, *seq, site, kind));
        *seq += 1;
        self
    }

    pub fn build(&self) -> Vec<SyncEvent> {
        self.events.clone()
    }
}
