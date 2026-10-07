#![allow(unused_variables, dead_code)]
//! Reference model of one thread's async-group queue in one domain
//! (`cp.async` or `cp.async.bulk`).
//!
//! Spec: `docs/development/sync-semantics.md` §5, with the ISA answers in
//! `sync-isa-answers.md` Q7. Async-groups are per thread (PTX 9.4
//! §9.7.10.28.1.1), so this state lives per (warp, lane, domain). A wait
//! makes completions visible only to the executing lane. The legacy engine
//! keeps the same per-lane state (`async_groups.rs:987-1022`). A warp-level
//! wait blocks until every lane in its mask is ready; the caller builds that
//! from 32 instances of this model.
//!
//! Each committed group passes through two milestones in FIFO order: source
//! reads done, then full completion. Read milestones may run ahead of full
//! completions (`async_groups.rs:794-817`). The scheduler fires milestones
//! with `Complete`.
//!
//! CONTRACT: the types in this file are copied verbatim from
//! `numsim-sync-ref/src/async_group.rs` (plus serde derives) so production `step`
//! is differentially tested against the reference mechanically. Change
//! them only together with the reference crate, via the coordinator.
//! Function bodies marked `W3` are the production implementation to write.

use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Domain {
    #[default]
    CpAsync,
    Bulk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum Milestone {
    Pending,
    ReadsDone,
    FullyDone,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Group {
    pub ordinal: u64,
    /// `false` for a batch cut by `cp.async.mbarrier.arrive`. Such a batch
    /// schedules its copies but does not close a PTX group
    /// (`async_groups.rs:733`).
    pub closes: bool,
    pub ops: u32,
    pub milestone: Milestone,
    /// Deferred mbarrier arrive-ons released at full completion.
    pub arrivals: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct State {
    pub domain: Domain,
    /// Issued, not yet committed.
    pub open: u32,
    pub groups: VecDeque<Group>,
    pub next_ordinal: u64,
}

impl State {
    pub fn new(domain: Domain) -> Self {
        Self {
            domain,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Cmd {
    Issue,
    Commit,
    /// `cp.async.mbarrier.arrive[.noinc]` (cp.async domain only). The
    /// mbarrier half is `mbarrier::Cmd::{IncPending, Issue}`.
    ArriveOn,
    /// The scheduler fires the next milestone of group `ordinal`.
    Complete {
        ordinal: u64,
        milestone: Milestone,
    },
    /// `cp.async.wait_group n`, `cp.async.bulk.wait_group[.read] n`.
    /// `cp.async.wait_all` lowers to `Commit` followed by `Wait { n: 0 }`.
    Wait {
        n: u64,
        read: bool,
    },
    /// Launch exit: bulk issues are committed implicitly
    /// (`async_groups.rs:1417-1454`).
    Exit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Outcome {
    Done,
    Committed {
        ordinal: u64,
        empty: bool,
    },
    /// Where the arrive-on attached: `Some(ordinal)` means it is released by
    /// that group's full completion, and `None` means it is released now.
    ArriveOn {
        group: Option<u64>,
    },
    /// A milestone fired; `arrivals` deferred arrive-ons are now due.
    Completed {
        arrivals: u32,
    },
    /// `acquired` is the milestone the wait guarantees for this thread only.
    /// `ReadsDone` (`.read`) releases the sources for reuse and never makes
    /// destination writes visible (PTX 9.4 §9.7.10.28.6.2).
    Ready {
        retired: u32,
        acquired: Milestone,
    },
    Blocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Error {
    /// `.read` on `cp.async`, or arrive-on outside the cp.async domain.
    InvalidForm,
    NotEnabled {
        ordinal: u64,
        milestone: Milestone,
    },
    PendingAtExit {
        ordinal: u64,
    },
}

pub struct AsyncGroup;

impl super::Protocol for AsyncGroup {
    type State = State;
    type Cmd = Cmd;
    type Outcome = Outcome;
    type Error = Error;
    fn step(state: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
        step(state, cmd)
    }
}

pub fn step(state: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
    unimplemented!("W3: async_group::step")
}



/// Number of oldest groups a `wait_group n` must cover: everything up to and
/// including the (n+1)-th newest closing group (`async_groups.rs:897-909`).
pub fn wait_prefix_len(s: &State, n: u64) -> usize {
    unimplemented!("W3: async_group::wait_prefix_len")
}

/// Review-level lint. The ISA does not require a commit before exit
/// (sync-isa-answers Q7); the legacy engine treated this as an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Lint {
    UncommittedAtExit { open: u32 },
}

pub fn exit_lint(s: &State) -> Option<Lint> {
    unimplemented!("W3: async_group::exit_lint")
}

/// Run after the completion pump has drained. Committed groups always
/// complete, so an error here is an infrastructure fault.
pub fn quiescent(s: &State) -> Result<(), Error> {
    unimplemented!("W3: async_group::quiescent")
}

pub fn check_invariants(s: &State) -> Result<(), String> {
    unimplemented!("W3: async_group::check_invariants")
}
