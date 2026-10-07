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
    match cmd {
        Cmd::Issue => {
            state.open += 1;
            Ok(Outcome::Done)
        }
        Cmd::Commit => {
            let g = close(state, true, 0);
            Ok(Outcome::Committed { ordinal: g.ordinal, empty: g.ops == 0 })
        }
        Cmd::ArriveOn => {
            if state.domain != Domain::CpAsync {
                return Err(Error::InvalidForm);
            }
            if state.open != 0 {
                let g = close(state, false, 1);
                return Ok(Outcome::ArriveOn { group: Some(g.ordinal) });
            }
            // FIFO completion: the newest unfinished group completes last.
            let newest = state.groups.iter_mut().rev().find(|g| g.milestone != Milestone::FullyDone);
            Ok(Outcome::ArriveOn {
                group: newest.map(|g| {
                    g.arrivals += 1;
                    g.ordinal
                }),
            })
        }
        Cmd::Complete { ordinal, milestone } => {
            let Some(i) = enabled_index(state, milestone).filter(|&i| state.groups[i].ordinal == ordinal) else {
                return Err(Error::NotEnabled { ordinal, milestone });
            };
            let g = &mut state.groups[i];
            g.milestone = milestone;
            let arrivals = if milestone == Milestone::FullyDone { std::mem::take(&mut g.arrivals) } else { 0 };
            Ok(Outcome::Completed { arrivals })
        }
        Cmd::Wait { n, read } => {
            if read && state.domain == Domain::CpAsync {
                return Err(Error::InvalidForm);
            }
            let acquired = if read { Milestone::ReadsDone } else { Milestone::FullyDone };
            let prefix = wait_prefix_len(state, n);
            if state.groups.range(..prefix).any(|g| g.milestone < acquired) {
                return Ok(Outcome::Blocked);
            }
            let retired = if read {
                // `.read` retires only empty (complete) groups of the prefix.
                let mut kept = VecDeque::with_capacity(state.groups.len());
                let mut retired = 0u32;
                for (i, g) in std::mem::take(&mut state.groups).into_iter().enumerate() {
                    if i < prefix && g.ops == 0 {
                        retired += 1;
                    } else {
                        kept.push_back(g);
                    }
                }
                state.groups = kept;
                retired
            } else {
                state.groups.drain(..prefix);
                prefix as u32
            };
            Ok(Outcome::Ready { retired, acquired })
        }
        Cmd::Exit => {
            if state.domain == Domain::Bulk && state.open != 0 {
                close(state, true, 0);
            }
            Ok(Outcome::Done)
        }
    }
}



/// Number of oldest groups a `wait_group n` must cover: everything up to and
/// including the (n+1)-th newest closing group (`async_groups.rs:897-909`).
pub fn wait_prefix_len(s: &State, n: u64) -> usize {
    let mut remaining = n;
    for (i, g) in s.groups.iter().enumerate().rev() {
        if !g.closes {
            continue;
        }
        if remaining == 0 {
            return i + 1;
        }
        remaining -= 1;
    }
    0
}

/// Review-level lint. The ISA does not require a commit before exit
/// (sync-isa-answers Q7); the legacy engine treated this as an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Lint {
    UncommittedAtExit { open: u32 },
}

pub fn exit_lint(s: &State) -> Option<Lint> {
    (s.open != 0).then_some(Lint::UncommittedAtExit { open: s.open })
}

/// Run after the completion pump has drained. Committed groups always
/// complete, so an error here is an infrastructure fault.
pub fn quiescent(s: &State) -> Result<(), Error> {
    match s.groups.iter().find(|g| g.milestone != Milestone::FullyDone) {
        Some(g) => Err(Error::PendingAtExit { ordinal: g.ordinal }),
        None => Ok(()),
    }
}

pub fn check_invariants(s: &State) -> Result<(), String> {
    let mut last_ordinal = None;
    let mut last_nonempty = None;
    for g in &s.groups {
        if last_ordinal.is_some_and(|o| o >= g.ordinal) || g.ordinal >= s.next_ordinal {
            return Err("ordinals not increasing".into());
        }
        last_ordinal = Some(g.ordinal);
        if g.ops == 0 && g.milestone != Milestone::FullyDone {
            return Err("empty group not complete".into());
        }
        if g.milestone == Milestone::FullyDone && g.arrivals != 0 {
            return Err("arrivals held past full completion".into());
        }
        if g.ops != 0 {
            if last_nonempty.is_some_and(|m| m < g.milestone) {
                return Err("milestone overtook an older group (FIFO)".into());
            }
            last_nonempty = Some(g.milestone);
        }
    }
    Ok(())
}

/// Move the open issues into a new group. Empty groups are born complete.
fn close(s: &mut State, closes: bool, arrivals: u32) -> Group {
    let ops = std::mem::take(&mut s.open);
    let milestone = if ops == 0 { Milestone::FullyDone } else { Milestone::Pending };
    let g = Group { ordinal: s.next_ordinal, closes, ops, milestone, arrivals };
    s.next_ordinal += 1;
    s.groups.push_back(g);
    g
}

/// Index of the group whose `milestone` may fire next: reads on the oldest
/// pending group; full completion on the oldest unfinished group once its
/// reads are done (`async_groups.rs:794-817`).
fn enabled_index(s: &State, milestone: Milestone) -> Option<usize> {
    match milestone {
        Milestone::ReadsDone => s.groups.iter().position(|g| g.milestone == Milestone::Pending),
        Milestone::FullyDone => s
            .groups
            .iter()
            .position(|g| g.milestone != Milestone::FullyDone)
            .filter(|&i| s.groups[i].milestone == Milestone::ReadsDone),
        Milestone::Pending => None,
    }
}

/// Is `Complete { ordinal, milestone }` enabled now?
pub fn milestone_enabled(s: &State, ordinal: u64, milestone: Milestone) -> bool {
    enabled_index(s, milestone).is_some_and(|i| s.groups[i].ordinal == ordinal)
}

use crate::report::FindingKind;

/// Map a protocol error to its report kind.
pub fn finding_kind(e: &Error) -> FindingKind {
    match e {
        Error::InvalidForm => FindingKind::AsyncGroupMisuse,
        Error::NotEnabled { .. } | Error::PendingAtExit { .. } => FindingKind::RuntimeError,
    }
}

/// Report kind of an exit lint (reported with `Status::Review`).
pub fn lint_kind(_l: &Lint) -> FindingKind {
    FindingKind::UnwaitedAsync
}
