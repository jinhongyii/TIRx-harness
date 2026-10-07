//! Synchronization protocols: one transition function per GPU concept
//! (plan section 1). The engine, the online checkers and the offline
//! explorer call the same `step`.
//!
//! **Shapes are the reference crate's** (`numsim-sync-ref`, spec
//! `docs/development/sync-semantics.md` section 1): each protocol module has
//! `State`, `Cmd`, `Outcome`, `Error` copied verbatim (plus serde) and
//!
//! ```ignore
//! pub fn step(state: &mut State, cmd: Cmd) -> Result<Outcome, Error>;
//! pub fn quiescent(state: &State) -> Result<(), Error>;   // exit check
//! ```
//!
//! so production `step` (W3) is differentially tested against the reference
//! mechanically. Rules: transactional (`Err` leaves state unchanged),
//! deterministic, total (never panics). A blocking command returns the
//! protocol's `Outcome::Blocked` with the state unchanged; the scheduler
//! retries later (no wakers, no waiter registry). [`SyncTable::step`] lifts
//! that to [`Step::Blocked`] naming the resource.
//!
//! Lane aggregation, address resolution, multicast expansion and collective
//! rendezvous (setmaxnreg warpgroups, cta_group::2 pairs) happen *before*
//! `step`, in the handlers. A multi-target instruction applies `step` to
//! cloned states and commits all or none ([`SyncTable::step_all`]).

pub mod async_group;
pub mod cluster;
pub mod completion;
pub mod mbarrier;
pub mod named;
pub mod setmaxnreg;
pub mod tcgen;

pub use completion::{AsyncId, AsyncKind, AsyncOp, AsyncSource, Completion, Payload, ResourceId};

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;

/// Which reading of the ISA a protocol state enforces (reference crate).
/// NumSim and the online checkers run `Numeric`; the offline explorer runs
/// `Strict`. Strict refines Numeric.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Policy {
    #[default]
    Numeric,
    Strict,
}

/// Uniform view of a protocol (reference crate's trait).
pub trait Protocol {
    type State: Clone + fmt::Debug + PartialEq;
    type Cmd: Clone + fmt::Debug;
    type Outcome: Clone + fmt::Debug + PartialEq;
    type Error: Clone + fmt::Debug + PartialEq;
    fn step(state: &mut Self::State, cmd: Self::Cmd) -> Result<Self::Outcome, Self::Error>;
}

/// A warp index local to the resource's scope (CTA or cluster).
pub type Warp = u32;
/// Thread mask of one warp (reference-crate spelling; same bits as `WarpMask`).
pub type LaneMask = u32;
pub const FULL_MASK: LaneMask = u32::MAX;

/// Result of [`SyncTable::step`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Step<O> {
    Done(O),
    /// Cannot proceed; state unchanged; retry after `ResourceId` changes.
    Blocked(ResourceId),
}

/// A command to any protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SyncCmd {
    Mbarrier(mbarrier::Cmd),
    Named(named::Cmd),
    Cluster(cluster::Cmd),
    AsyncGroup(async_group::Cmd),
    Tcgen(tcgen::Cmd),
    TcgenWork(tcgen::WorkCmd),
    RegPool(setmaxnreg::Cmd),
}

/// Outcome of any protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Outcome {
    Mbarrier(mbarrier::Outcome),
    Named(named::Outcome),
    Cluster(cluster::Outcome),
    AsyncGroup(async_group::Outcome),
    Tcgen(tcgen::Outcome),
    TcgenWork(tcgen::WorkOutcome),
    RegPool(setmaxnreg::Outcome),
}

impl Outcome {
    /// The protocol said `Blocked`.
    pub fn is_blocked(&self) -> bool {
        matches!(
            self,
            Outcome::Mbarrier(mbarrier::Outcome::Blocked)
                | Outcome::Named(named::Outcome::Blocked)
                | Outcome::Cluster(cluster::Outcome::Blocked)
                | Outcome::AsyncGroup(async_group::Outcome::Blocked)
                | Outcome::RegPool(setmaxnreg::Outcome::Blocked)
        )
    }
}

/// Error of any protocol.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SyncError {
    Mbarrier(mbarrier::Error),
    Named(named::Error),
    Cluster(cluster::Error),
    AsyncGroup(async_group::Error),
    Tcgen(tcgen::Error),
    RegPool(setmaxnreg::Error),
    /// Command kind does not match the resource kind (engine bug).
    WrongResource { resource: ResourceId },
}

impl fmt::Display for SyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SyncError {}

/// State of one resource.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Resource {
    Mbarrier(mbarrier::State),
    Named(named::State),
    Cluster(cluster::State),
    AsyncGroup(async_group::State),
    Tcgen(tcgen::State),
    TcgenWork(tcgen::WorkState),
    RegPool(setmaxnreg::State),
}

/// Launch facts needed to create resources implicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ResourceInit {
    pub policy: Policy,
    /// Participant warps of each cluster barrier.
    pub cluster_warps: u32,
    pub warps_per_cta: u32,
}

/// All sync state of one launch.
#[derive(Clone, Debug, Default)]
pub struct SyncTable {
    pub init: ResourceInit,
    pub resources: HashMap<ResourceId, Resource>,
    /// Sync-level completions waiting to be applied.
    pub completions: VecDeque<Completion>,
    /// Data-carrying async ops in flight.
    pub async_ops: VecDeque<AsyncOp>,
    next_async: u64,
}

impl SyncTable {
    pub fn new(init: ResourceInit) -> SyncTable {
        SyncTable { init, ..SyncTable::default() }
    }

    pub fn get(&self, id: ResourceId) -> Option<&Resource> {
        self.resources.get(&id)
    }

    pub fn next_async_id(&mut self) -> AsyncId {
        let id = AsyncId(self.next_async);
        self.next_async += 1;
        id
    }

    /// Fresh state for a resource on first use. mbarriers start
    /// uninitialized (`Init` makes them live).
    pub fn fresh(&self, id: ResourceId) -> Option<Resource> {
        let p = self.init.policy;
        // Constructors follow the reference crate's current API.
        Some(match id {
            ResourceId::Mbarrier { .. } => Resource::Mbarrier(mbarrier::State::new(p)),
            ResourceId::Named { .. } => Resource::Named(named::State::default()),
            ResourceId::Cluster { .. } => Resource::Cluster(cluster::State::new(self.init.cluster_warps)),
            ResourceId::AsyncGroup { domain, .. } => Resource::AsyncGroup(async_group::State::new(domain)),
            ResourceId::TcgenLifecycle { .. } => Resource::Tcgen(tcgen::State::default()),
            ResourceId::TcgenWork { .. } => Resource::TcgenWork(tcgen::WorkState::default()),
            ResourceId::RegPool { .. } => Resource::RegPool(setmaxnreg::State::new(self.init.warps_per_cta)),
            ResourceId::Word { .. } | ResourceId::Grid | ResourceId::WarpSync { .. } => return None,
        })
    }

    fn apply(res: &mut Resource, id: ResourceId, cmd: SyncCmd) -> Result<Outcome, SyncError> {
        match (res, cmd) {
            (Resource::Mbarrier(s), SyncCmd::Mbarrier(c)) => {
                mbarrier::step(s, c).map(Outcome::Mbarrier).map_err(SyncError::Mbarrier)
            }
            (Resource::Named(s), SyncCmd::Named(c)) => named::step(s, c).map(Outcome::Named).map_err(SyncError::Named),
            (Resource::Cluster(s), SyncCmd::Cluster(c)) => {
                cluster::step(s, c).map(Outcome::Cluster).map_err(SyncError::Cluster)
            }
            (Resource::AsyncGroup(s), SyncCmd::AsyncGroup(c)) => {
                async_group::step(s, c).map(Outcome::AsyncGroup).map_err(SyncError::AsyncGroup)
            }
            (Resource::Tcgen(s), SyncCmd::Tcgen(c)) => tcgen::step(s, c).map(Outcome::Tcgen).map_err(SyncError::Tcgen),
            (Resource::TcgenWork(s), SyncCmd::TcgenWork(c)) => {
                Ok(Outcome::TcgenWork(tcgen::work_step(s, c)))
            }
            (Resource::RegPool(s), SyncCmd::RegPool(c)) => {
                setmaxnreg::step(s, c).map(Outcome::RegPool).map_err(SyncError::RegPool)
            }
            _ => Err(SyncError::WrongResource { resource: id }),
        }
    }

    /// Apply `cmd` to resource `id` (created on first use).
    pub fn step(&mut self, id: ResourceId, cmd: SyncCmd) -> Result<Step<Outcome>, SyncError> {
        if !self.resources.contains_key(&id) {
            let r = self.fresh(id).ok_or(SyncError::WrongResource { resource: id })?;
            self.resources.insert(id, r);
        }
        let res = self.resources.get_mut(&id).expect("inserted above");
        let out = Self::apply(res, id, cmd)?;
        Ok(if out.is_blocked() { Step::Blocked(id) } else { Step::Done(out) })
    }

    /// All-or-nothing multi-target step (multicast arrive, lane-varying
    /// batches, 2-CTA ops): applies to clones and commits only if every
    /// command succeeded and none blocked.
    pub fn step_all(&mut self, cmds: &[(ResourceId, SyncCmd)]) -> Result<Step<Vec<Outcome>>, SyncError> {
        let mut staged: Vec<(ResourceId, Resource)> = Vec::with_capacity(cmds.len());
        let mut outs = Vec::with_capacity(cmds.len());
        for &(id, cmd) in cmds {
            let pos = staged.iter().position(|(r, _)| *r == id);
            let mut res = match pos {
                Some(i) => staged.remove(i).1,
                None => match self.resources.get(&id) {
                    Some(r) => r.clone(),
                    None => self.fresh(id).ok_or(SyncError::WrongResource { resource: id })?,
                },
            };
            let out = Self::apply(&mut res, id, cmd)?;
            if out.is_blocked() {
                return Ok(Step::Blocked(id));
            }
            outs.push(out);
            staged.push((id, res));
        }
        for (id, r) in staged {
            self.resources.insert(id, r);
        }
        Ok(Step::Done(outs))
    }

    /// Is this completion enabled now (W3 spec 1.3 rules: mbarrier action
    /// generation, async-group FIFO, grant availability)?
    pub fn enabled(&self, c: &Completion) -> bool {
        let _ = c;
        unimplemented!("W3: SyncTable::enabled")
    }

    /// Apply a completion through the protocol `step`.
    pub fn apply_completion(&mut self, c: Completion) -> Result<Step<Outcome>, SyncError> {
        let cmd = match c {
            Completion::MbarTx { gen, bytes, .. } => SyncCmd::Mbarrier(mbarrier::Cmd::CompleteTx { gen, bytes }),
            Completion::MbarArrive { gen, count, .. } => {
                SyncCmd::Mbarrier(mbarrier::Cmd::DeferredArrive { gen, count })
            }
            Completion::GroupMilestone { ordinal, milestone, .. } => {
                SyncCmd::AsyncGroup(async_group::Cmd::Complete { ordinal, milestone })
            }
            Completion::SetmaxGrant { wg, .. } => SyncCmd::RegPool(setmaxnreg::Cmd::Grant { wg }),
        };
        self.step(c.resource(), cmd)
    }

    /// Exit check over every resource (after the queues drained): each
    /// protocol's `quiescent`. Empty = clean.
    pub fn quiescent(&self) -> Vec<(ResourceId, SyncError)> {
        let mut out: Vec<(ResourceId, SyncError)> = Vec::new();
        for (id, r) in &self.resources {
            let e = match r {
                Resource::Mbarrier(s) => mbarrier::quiescent(s).map_err(SyncError::Mbarrier),
                Resource::Named(s) => named::quiescent(s).map_err(SyncError::Named),
                Resource::Cluster(s) => cluster::quiescent(s).map_err(SyncError::Cluster),
                Resource::AsyncGroup(s) => async_group::quiescent(s).map_err(SyncError::AsyncGroup),
                Resource::Tcgen(s) => tcgen::quiescent(s).map_err(SyncError::Tcgen),
                Resource::TcgenWork(_) => Ok(()),
                Resource::RegPool(s) => setmaxnreg::quiescent(s).map_err(SyncError::RegPool),
            };
            if let Err(e) = e {
                out.push((*id, e));
            }
        }
        out.sort_by_cached_key(|(id, _)| format!("{id:?}"));
        out
    }
}
