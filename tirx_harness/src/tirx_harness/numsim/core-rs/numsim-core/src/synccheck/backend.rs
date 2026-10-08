//! The protocol `step` functions the explorer drives: the production
//! `crate::sync::*::step` (the same functions the engine calls). This is the
//! only file in `synccheck/` that names them.

use crate::sync::{Outcome, Policy, ResourceId, ResourceInit, SyncCmd, SyncError};
use crate::sync as p;

/// Abstract state of one resource (the production `State` types after the swap).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Res {
    Mbarrier(p::mbarrier::State),
    Named(p::named::State),
    Cluster(p::cluster::State),
    AsyncGroup(p::async_group::State),
    Tcgen(p::tcgen::State),
    TcgenWork(p::tcgen::WorkState),
    RegPool(p::setmaxnreg::State),
    /// Kernel-wide `.cta_group`; normally checked statically
    /// (`Program::cta_group_error`) and never explored.
    TcgenKernel(p::tcgen::KernelState),
}

/// Fresh state on first use. The offline explorer runs the `Strict` policy.
pub fn fresh(id: ResourceId, init: &ResourceInit) -> Option<Res> {
    Some(match id {
        ResourceId::Mbarrier { .. } => Res::Mbarrier(p::mbarrier::State::new(Policy::Strict)),
        ResourceId::Named { .. } => Res::Named(p::named::State::default()),
        ResourceId::Cluster { .. } => Res::Cluster(p::cluster::State::new(init.cluster_warps)),
        ResourceId::AsyncGroup { domain, .. } => Res::AsyncGroup(p::async_group::State::new(domain)),
        ResourceId::TcgenLifecycle { .. } => Res::Tcgen(p::tcgen::State::default()),
        ResourceId::TcgenWork { .. } => Res::TcgenWork(p::tcgen::WorkState::default()),
        ResourceId::RegPool { .. } => Res::RegPool(p::setmaxnreg::State::new(init.warps_per_cta)),
        ResourceId::TcgenKernel => Res::TcgenKernel(p::tcgen::KernelState::default()),
        ResourceId::Word { .. } | ResourceId::Grid | ResourceId::WarpSync { .. } => return None,
    })
}

/// One transactional protocol step (`Err` leaves `res` unchanged).
pub fn step(res: &mut Res, id: ResourceId, cmd: SyncCmd) -> Result<Outcome, SyncError> {
    match (res, cmd) {
        (Res::Mbarrier(s), SyncCmd::Mbarrier(c)) => p::mbarrier::step(s, c)
            .map(Outcome::Mbarrier)
            .map_err(SyncError::Mbarrier),
        (Res::Named(s), SyncCmd::Named(c)) => p::named::step(s, c)
            .map(Outcome::Named)
            .map_err(SyncError::Named),
        (Res::Cluster(s), SyncCmd::Cluster(c)) => p::cluster::step(s, c)
            .map(Outcome::Cluster)
            .map_err(SyncError::Cluster),
        (Res::AsyncGroup(s), SyncCmd::AsyncGroup(c)) => p::async_group::step(s, c)
            .map(Outcome::AsyncGroup)
            .map_err(SyncError::AsyncGroup),
        (Res::Tcgen(s), SyncCmd::Tcgen(c)) => p::tcgen::step(s, c)
            .map(Outcome::Tcgen)
            .map_err(SyncError::Tcgen),
        (Res::TcgenWork(s), SyncCmd::TcgenWork(c)) => Ok(Outcome::TcgenWork(p::tcgen::work_step(s, c))),
        (Res::RegPool(s), SyncCmd::RegPool(c)) => p::setmaxnreg::step(s, c)
            .map(Outcome::RegPool)
            .map_err(SyncError::RegPool),
        (Res::TcgenKernel(k), SyncCmd::TcgenGroup(g)) => p::tcgen::use_cta_group(k, g)
            .map(|()| Outcome::Tcgen(p::tcgen::Outcome::Done))
            .map_err(SyncError::Tcgen),
        _ => Err(SyncError::WrongResource { resource: id }),
    }
}

/// End-of-launch check.
pub fn quiescent(res: &Res) -> Result<(), SyncError> {
    match res {
        Res::Mbarrier(s) => p::mbarrier::quiescent(s).map_err(SyncError::Mbarrier),
        Res::Named(s) => p::named::quiescent(s).map_err(SyncError::Named),
        Res::Cluster(s) => p::cluster::quiescent(s).map_err(SyncError::Cluster),
        Res::AsyncGroup(s) => p::async_group::quiescent(s).map_err(SyncError::AsyncGroup),
        Res::Tcgen(s) => p::tcgen::quiescent(s).map_err(SyncError::Tcgen),
        Res::TcgenWork(_) | Res::TcgenKernel(_) => Ok(()),
        Res::RegPool(s) => p::setmaxnreg::quiescent(s).map_err(SyncError::RegPool),
    }
}

/// setmaxnreg grants the scheduler may fire now.
pub fn enabled_grants(res: &Res) -> Vec<u32> {
    match res {
        Res::RegPool(s) => p::setmaxnreg::enabled_grants(s),
        _ => Vec::new(),
    }
}

/// Async-group bookkeeping the explorer needs: `next_ordinal` and whether
/// group `ordinal` has reached full completion (or was retired).
pub fn async_next_ordinal(res: &Res) -> Option<u64> {
    match res {
        Res::AsyncGroup(s) => Some(s.next_ordinal),
        _ => None,
    }
}

pub fn async_group_pending(res: &Res, ordinal: u64) -> bool {
    match res {
        Res::AsyncGroup(s) => s
            .groups
            .iter()
            .any(|g| g.ordinal == ordinal && g.milestone != p::async_group::Milestone::FullyDone),
        _ => false,
    }
}

/// Non-empty groups created with ordinals in `from..` (they need milestones).
pub fn async_new_groups(res: &Res, from: u64) -> Vec<u64> {
    match res {
        Res::AsyncGroup(s) => s
            .groups
            .iter()
            .filter(|g| g.ordinal >= from && g.milestone == p::async_group::Milestone::Pending)
            .map(|g| g.ordinal)
            .collect(),
        _ => Vec::new(),
    }
}

/// Review-level exit lint of a terminal resource state (W3 `exit_lint`).
pub fn exit_lint(res: &Res) -> Option<(crate::report::FindingKind, String)> {
    match res {
        Res::Named(s) => p::named::exit_lint(s).map(|l| (p::named::lint_kind(&l), format!("{l:?}"))),
        Res::AsyncGroup(s) => p::async_group::exit_lint(s).map(|l| (p::async_group::lint_kind(&l), format!("{l:?}"))),
        _ => None,
    }
}

/// How a command interacts with a barrier's phase, for the strong-diamond
/// independence proof (`Ts::independent_of_future`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Reads the completed phase (parity waits/tests, named `Resume`,
    /// cluster wait); never completes a phase.
    Observer,
    /// Adds `n` arrivals (or only counters/tx with `n == 0`) to the open phase.
    Contributor(u64),
    /// Anything else (init, inval, pending increments, drops, state tokens,
    /// other protocols): no reduction.
    Other,
}

pub fn classify(cmd: &SyncCmd) -> Class {
    use crate::sync::{cluster, mbarrier, named};
    match cmd {
        SyncCmd::Mbarrier(c) => match *c {
            mbarrier::Cmd::WaitParity { .. } | mbarrier::Cmd::TestParity { .. } => Class::Observer,
            mbarrier::Cmd::Arrive { count, drop: false, no_complete: false, .. } => Class::Contributor(count),
            mbarrier::Cmd::ExpectTx { .. } | mbarrier::Cmd::Issue | mbarrier::Cmd::CompleteTx { .. } => {
                Class::Contributor(0)
            }
            mbarrier::Cmd::DeferredArrive { count, .. } => Class::Contributor(count),
            _ => Class::Other,
        },
        SyncCmd::Named(named::Cmd::Resume { .. }) => Class::Observer,
        SyncCmd::Named(named::Cmd::Arrive(_) | named::Cmd::Sync(_) | named::Cmd::Red(_)) => {
            Class::Contributor(p::named::WARP_SIZE)
        }
        SyncCmd::Cluster(cluster::Cmd::Wait { .. }) => Class::Observer,
        SyncCmd::Cluster(cluster::Cmd::Arrive { .. }) => Class::Contributor(1),
        _ => Class::Other,
    }
}

/// Arrivals still needed before the open phase can complete, in the units of
/// [`Class::Contributor`]. `fresh` is the thread count a named barrier's next
/// generation will expect. `None` = unknown (no reduction).
pub fn remaining(res: &Res, fresh: Option<u64>) -> Option<u64> {
    match res {
        Res::Mbarrier(s) if s.live => Some(if s.complete { s.expected } else { s.required() - s.arrived }),
        Res::Named(s) => match s.expected {
            Some(e) if !s.complete => Some(e - s.arrived),
            _ => fresh,
        },
        Res::Cluster(s) => {
            let members = s.live.iter().filter(|&&m| m != 0).count() as u64;
            Some(members.saturating_sub(s.arrived.len() as u64))
        }
        _ => None,
    }
}

/// Observers of this resource are never disabled once enabled (named
/// `Resume{gen}` and cluster waits stay ready); mbarrier parity observers
/// are disabled by the next phase completion.
pub fn observers_stable(res: &Res) -> bool {
    matches!(res, Res::Named(_) | Res::Cluster(_))
}

/// An mbarrier parity observer that a completion of the open phase would
/// flip from ready to blocked (it observes the last completed parity).
/// Observers waiting for the open phase are only enabled by its completion.
pub fn observer_disabled_by_completion(res: &Res, cmd: &SyncCmd) -> bool {
    use crate::sync::mbarrier;
    match (res, cmd) {
        (
            Res::Mbarrier(s),
            SyncCmd::Mbarrier(mbarrier::Cmd::WaitParity { parity } | mbarrier::Cmd::TestParity { parity }),
        ) => !s.live || *parity == s.completed_parity(),
        (Res::Mbarrier(_), _) => true,
        _ => false,
    }
}
