//! The core's input vocabulary: an alias layer over the contract
//! (`crate::observe`, `crate::arena`, `crate::program`) plus the two
//! per-lane event shapes the checker consumes.
//!
//! Name mapping (W5 ↔ contract): `Domain` = `observe::Window`,
//! `Milestone` = `observe::Side`, `AsyncKind` = `observe::AsyncClass`,
//! `SyncObjId` = `sync::ResourceId`, `Who` ⊂ `observe::Actor`,
//! `order` = `Sem` (normalised to [`MemOrder`] by the adapter). The
//! contract `Access` is a per-instruction batch; [`super::observer`] splits
//! it into per-lane [`Access`] items (`Access::per_lane`), and converts
//! `observe::SyncEvent` into [`SyncEvent`].

use std::ops::Range;

pub use crate::arena::{AllocId, Space};
pub use crate::observe::{AccessKind, AsyncClass as AsyncKind, Side as Milestone, Window as Domain};
pub use crate::program::{Proxy, Scope};
pub use crate::site::SiteId;
pub use crate::sync::completion::{AsyncId, ResourceId as SyncObjId};
pub use crate::value::LaneMask;

/// Dense launch-global warp index (`observe::WarpId.0`).
pub type WarpId = u32;

/// Lane-mask helpers over `WarpMask` with `u8` lanes.
pub trait Lanes: Copy {
    fn lanes8(self) -> impl Iterator<Item = u8>;
    fn has(self, lane: u8) -> bool;
}

impl Lanes for LaneMask {
    fn lanes8(self) -> impl Iterator<Item = u8> {
        self.lanes().map(|l| l as u8)
    }
    fn has(self, lane: u8) -> bool {
        self.contains(lane as usize)
    }
}

pub fn one_lane(lane: u8) -> LaneMask {
    LaneMask::lane(lane as usize)
}

/// Epoch of an async side: read-side 1, write-side 2 (within a slot
/// generation, see `Checker::async_epoch`).
pub fn side_index(m: Milestone) -> u32 {
    match m {
        Milestone::Read => 1,
        Milestone::Write => 2,
    }
}

/// Normalised memory order (contract `Sem`: `Volatile`/`Mmio` → relaxed at
/// `.sys`; `Sc` → an SC fence followed by an `AcqRel` access).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemOrder {
    Weak,
    Relaxed,
    Acquire,
    Release,
    AcqRel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Who {
    Lane { warp: WarpId, lane: u8, epoch: u32 },
    Async { op: AsyncId, side: Milestone },
}

/// One lane's (or one async side's) contiguous footprint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Access {
    pub who: Who,
    pub alloc: AllocId,
    pub range: Range<u64>,
    pub kind: AccessKind,
    pub order: MemOrder,
    /// `Some` for strong accesses.
    pub scope: Option<Scope>,
    pub atomic: bool,
    /// `atom` (true) vs `red` (false); only meaningful for RMW.
    pub returns_value: bool,
    pub proxy: Proxy,
    pub domain: Option<Domain>,
    pub site: SiteId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenceKind {
    AcqRel(Scope),
    Sc(Scope),
    ProxyAsync(Option<Domain>),
    TcgenBefore,
    TcgenAfter,
    /// `fence.proxy.tensormap::generic.release`.
    TensormapRelease,
    /// `fence.proxy.tensormap::generic.acquire`.
    TensormapAcquire,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionTarget {
    Phase { obj: SyncObjId, phase: u64 },
    Warp { warp: WarpId, lanes: LaneMask },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SyncEvent {
    AllocBegin { alloc: AllocId, space: Space, size: u64, cta: u32 },
    AllocEnd { alloc: AllocId },
    DeclareWord { alloc: AllocId, range: Range<u64> },
    WarpSync { warp: WarpId, mask: LaneMask, epoch: u32 },
    /// `release: None` = qualifier lost in lowering (incomplete).
    /// `scope: None` = named barrier (participants, no scope).
    Arrive { warp: WarpId, lanes: LaneMask, obj: SyncObjId, phase: u64, release: Option<bool>, scope: Option<Scope>, epoch: u32 },
    Wait { warp: WarpId, lanes: LaneMask, obj: SyncObjId, phase: u64, acquire: Option<bool>, scope: Option<Scope>, epoch: u32 },
    Fence { warp: WarpId, lanes: LaneMask, kind: FenceKind, epoch: u32 },
    AsyncIssue {
        op: AsyncId,
        warp: WarpId,
        lanes: LaneMask,
        kind: AsyncKind,
        proxy: Proxy,
        preds: Vec<AsyncId>,
        footprint: Vec<(AllocId, Range<u64>)>,
        site: SiteId,
        epoch: u32,
    },
    AsyncComplete { op: AsyncId, milestone: Milestone, target: CompletionTarget },
    WaitVerdicts {
        warp: WarpId,
        lanes: LaneMask,
        alloc: AllocId,
        range: Range<u64>,
        scope: Scope,
        accepted: Vec<u64>,
        observed: u32,
        pred_reads: Vec<(AllocId, Range<u64>)>,
        epoch: u32,
    },
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
        warp / self.warps_per_cta.max(1)
    }
    pub fn cluster_of(&self, warp: WarpId) -> u32 {
        self.cta_of(warp) / self.ctas_per_cluster.max(1)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    Access(Access),
    Sync(SyncEvent),
}
