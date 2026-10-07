//! Resource identities, sync-level completions and async operations.
//!
//! Two queues (owned by [`super::SyncTable`]):
//!
//! * [`Completion`] (W3 spec section 1.3): a *sync-level* effect bound at issue
//!   to the generation/ordinal it must land on: `MbarTx`, `MbarArrive`,
//!   `GroupMilestone`, `SetmaxGrant`. Applying one is a `step` call; whether
//!   it is enabled now is `SyncTable::enabled`.
//! * [`AsyncOp`]: a *data* effect (TMA, cp.async(.bulk), st.async,
//!   tcgen05.mma/cp, CLC response) resolved at issue into a [`Payload`].
//!   When the scheduler lands it, the payload bytes move (read then write,
//!   atomically w.r.t. warps), then its `signals` (Completions) are queued.
//!   Its id is also its virtual actor (`Actor::Async`).
//!
//! NumSim may land everything immediately (the legacy eager path); checkers
//! may delay. Both use the same path.

use crate::arena::{AllocId, ByteSpan};
use crate::dtype::Dtype;
use crate::observe::{CtaId, WarpId};
use crate::program::{AtomOp, TcgenMmaArgs};
use crate::site::SiteId;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Launch-unique id of an async operation (also its virtual actor).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AsyncId(pub u64);

/// Physical identity of a synchronization object. Equal ids name the same
/// object; remote (cluster-mapped) operations name the *target* CTA's object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceId {
    /// mbarrier: owning CTA, its shared-window allocation, byte offset.
    Mbarrier { cta: CtaId, alloc: AllocId, offset: u32 },
    /// Named barrier 0..15 of a CTA.
    Named { cta: CtaId, id: u8 },
    /// Cluster barrier.
    Cluster { cluster: u32 },
    /// Async-group queue of one thread (`domain` = cp.async or bulk).
    AsyncGroup { warp: WarpId, lane: u8, domain: super::async_group::Domain },
    /// tcgen05 TMEM lifecycle of a CTA pair (`pair` = global id of the even CTA).
    TcgenLifecycle { pair: CtaId },
    /// tcgen05 per-thread work queue (commit / wait::ld / wait::st).
    TcgenWork { warp: WarpId, lane: u8 },
    /// setmaxnreg register pool of a CTA.
    RegPool { cta: CtaId },
    /// A declared synchronization word (`WaitUntil`).
    Word { alloc: AllocId, offset: u64 },
    /// Cooperative grid barrier.
    Grid,
    /// `__syncwarp` rendezvous of a divergent warp.
    WarpSync { warp: WarpId },
}

impl fmt::Display for ResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// A sync-level completion (W3 spec 1.3). Carries the generation/ordinal
/// captured at issue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Completion {
    /// Transaction bytes landing on mbarrier generation `gen`.
    MbarTx { res: ResourceId, gen: u64, bytes: u64 },
    /// Deferred arrive-on (cp.async.mbarrier.arrive, tcgen05.commit).
    MbarArrive { res: ResourceId, gen: u64, count: u64 },
    /// Async-group milestone of group `ordinal`.
    GroupMilestone { res: ResourceId, ordinal: u64, milestone: super::async_group::Milestone },
    /// setmaxnreg grant for warpgroup `wg`.
    SetmaxGrant { res: ResourceId, wg: u32 },
}

impl Completion {
    pub fn resource(&self) -> ResourceId {
        match *self {
            Completion::MbarTx { res, .. }
            | Completion::MbarArrive { res, .. }
            | Completion::GroupMilestone { res, .. }
            | Completion::SetmaxGrant { res, .. } => res,
        }
    }
}

/// Category of an async op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AsyncKind {
    CpAsync,
    Bulk,
    BulkReduce,
    Tma,
    TmaReduce,
    StAsync,
    TcgenMma,
    TcgenCp,
    ClcResponse,
}

/// Who issued an async op (ties completions to the issuing SyncEvent).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AsyncSource {
    pub warp: WarpId,
    pub cta: CtaId,
    pub site: SiteId,
    /// `SyncEvent::seq` of the issuing warp's event (the issue command).
    pub seq: u32,
}

/// Resolved tcgen05.mma operands (uniform values evaluated at issue).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TcgenMmaPayload {
    pub args: TcgenMmaArgs,
    pub d_taddr: u32,
    /// Smem descriptor, or tmem address zero-extended, per `args.a`.
    pub a: u64,
    pub b_desc: u64,
    pub idesc: u32,
    pub enable_input_d: bool,
    pub scale_taddrs: Option<(u32, u32)>,
    pub scale_input_d: Option<u32>,
    pub sparse_meta: Option<u32>,
    pub disable_output_lane: Vec<u32>,
    /// Shared windows / tmem allocations of participating CTAs (1 or 2).
    pub smem: Vec<AllocId>,
    pub tmem: Vec<AllocId>,
}

/// Data effect performed when an async op lands.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Payload {
    None,
    /// Copy concatenated `src` spans to concatenated `dst` spans (equal
    /// totals), then zero-fill `zero_fill` (TMA OOB).
    Copy { src: Vec<(AllocId, ByteSpan)>, dst: Vec<(AllocId, ByteSpan)>, zero_fill: Vec<(AllocId, ByteSpan)> },
    /// Element-wise `dst = op(dst, src)`.
    Reduce { op: AtomOp, dtype: Dtype, src: Vec<(AllocId, ByteSpan)>, dst: Vec<(AllocId, ByteSpan)> },
    /// Bytes captured at issue (st.async values, CLC response).
    Data { dst: Vec<(AllocId, ByteSpan)>, bytes: Vec<u8> },
    /// Element-wise reduction of captured bytes (red.async).
    ReduceData { op: AtomOp, dtype: Dtype, dst: Vec<(AllocId, ByteSpan)>, bytes: Vec<u8> },
    TcgenMma(Box<TcgenMmaPayload>),
    /// tcgen05.cp (resolved spans; decompression by oplib).
    TcgenCp { src: Vec<(AllocId, ByteSpan)>, dst: Vec<(AllocId, ByteSpan)>, decompress_bits: u8 },
}

/// One in-flight async operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AsyncOp {
    pub id: AsyncId,
    pub source: AsyncSource,
    pub kind: AsyncKind,
    pub payload: Payload,
    /// Sync completions queued when the payload has landed (read-side
    /// `GroupMilestone{ReadsDone}` may be queued before the write).
    pub signals: Vec<Completion>,
    /// Must not land before these (tcgen05 pipeline order).
    pub after: Vec<AsyncId>,
}
