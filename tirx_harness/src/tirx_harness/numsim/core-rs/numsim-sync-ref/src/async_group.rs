//! Reference model of one thread's async-group queue in one domain
//! (`cp.async` or `cp.async.bulk`).
//!
//! Spec: `docs/development/sync-semantics.md` §5. The legacy engine keeps
//! this state per lane (`async_groups.rs:987-1022`). A warp-level wait blocks
//! until every lane in its mask is ready. The caller builds that from 32
//! instances of this model.
//!
//! Each committed group passes through two milestones in FIFO order: source
//! reads done, then full completion. Read milestones may run ahead of full
//! completions (`async_groups.rs:794-817`). The scheduler fires milestones
//! with `Complete`.

use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Domain {
    #[default]
    CpAsync,
    Bulk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Milestone {
    Pending,
    ReadsDone,
    FullyDone,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    Ready {
        retired: u32,
    },
    Blocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Error {
    /// `.read` on `cp.async`, or arrive-on outside the cp.async domain.
    InvalidForm,
    NotEnabled {
        ordinal: u64,
        milestone: Milestone,
    },
    UncommittedAtExit {
        open: u32,
    },
    PendingAtExit {
        ordinal: u64,
    },
}

pub struct AsyncGroup;

impl crate::Protocol for AsyncGroup {
    type State = State;
    type Cmd = Cmd;
    type Outcome = Outcome;
    type Error = Error;
    fn step(state: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
        step(state, cmd)
    }
}

pub fn step(state: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
    let mut next = state.clone();
    let outcome = apply(&mut next, cmd)?;
    *state = next;
    Ok(outcome)
}

fn push(s: &mut State, closes: bool, arrivals: u32) -> Group {
    let ops = std::mem::take(&mut s.open);
    let group = Group {
        ordinal: s.next_ordinal,
        closes,
        ops,
        // An empty group is born complete (async_groups.rs:718-740).
        milestone: if ops == 0 {
            Milestone::FullyDone
        } else {
            Milestone::Pending
        },
        arrivals,
    };
    s.next_ordinal += 1;
    s.groups.push_back(group);
    group
}

fn apply(s: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
    match cmd {
        Cmd::Issue => {
            s.open += 1;
            Ok(Outcome::Done)
        }
        Cmd::Commit => {
            let g = push(s, true, 0);
            Ok(Outcome::Committed {
                ordinal: g.ordinal,
                empty: g.ops == 0,
            })
        }
        Cmd::ArriveOn => {
            if s.domain != Domain::CpAsync {
                return Err(Error::InvalidForm);
            }
            if s.open > 0 {
                let g = push(s, false, 1);
                return Ok(Outcome::ArriveOn {
                    group: Some(g.ordinal),
                });
            }
            // FIFO completion makes the newest unfinished group the last one
            // to complete (async_groups.rs:1355-1373).
            match s
                .groups
                .iter_mut()
                .rev()
                .find(|g| g.milestone != Milestone::FullyDone)
            {
                Some(g) => {
                    g.arrivals += 1;
                    Ok(Outcome::ArriveOn {
                        group: Some(g.ordinal),
                    })
                }
                None => Ok(Outcome::ArriveOn { group: None }),
            }
        }
        Cmd::Complete { ordinal, milestone } => {
            let enabled = match milestone {
                Milestone::ReadsDone => s
                    .groups
                    .iter()
                    .find(|g| g.milestone == Milestone::Pending)
                    .map(|g| g.ordinal),
                Milestone::FullyDone => s
                    .groups
                    .iter()
                    .find(|g| g.milestone != Milestone::FullyDone)
                    .filter(|g| g.milestone == Milestone::ReadsDone)
                    .map(|g| g.ordinal),
                Milestone::Pending => None,
            };
            if enabled != Some(ordinal) {
                return Err(Error::NotEnabled { ordinal, milestone });
            }
            let g = s
                .groups
                .iter_mut()
                .find(|g| g.ordinal == ordinal)
                .expect("enabled group exists");
            g.milestone = milestone;
            let arrivals = if milestone == Milestone::FullyDone {
                std::mem::take(&mut g.arrivals)
            } else {
                0
            };
            Ok(Outcome::Completed { arrivals })
        }
        Cmd::Wait { n, read } => {
            if read && s.domain == Domain::CpAsync {
                return Err(Error::InvalidForm);
            }
            let prefix = wait_prefix_len(s, n);
            let need = if read {
                Milestone::ReadsDone
            } else {
                Milestone::FullyDone
            };
            if s.groups.iter().take(prefix).any(|g| g.milestone < need) {
                return Ok(Outcome::Blocked);
            }
            let retired = if read {
                // `.read` only drops empty complete groups. Non-empty ones
                // stay for a later full wait (async_groups.rs:953-983).
                let before = s.groups.len();
                let mut index = 0;
                s.groups.retain(|g| {
                    let keep = index >= prefix || g.ops != 0;
                    index += 1;
                    keep
                });
                before - s.groups.len()
            } else {
                s.groups.drain(..prefix).count()
            };
            Ok(Outcome::Ready {
                retired: retired as u32,
            })
        }
        Cmd::Exit => {
            if s.domain == Domain::Bulk && s.open > 0 {
                push(s, true, 0);
            }
            Ok(Outcome::Done)
        }
    }
}

/// Number of oldest groups a `wait_group n` must cover: everything up to and
/// including the (n+1)-th newest closing group (`async_groups.rs:897-909`).
pub fn wait_prefix_len(s: &State, n: u64) -> usize {
    let mut closing_seen = 0u64;
    for (index, group) in s.groups.iter().enumerate().rev() {
        if group.closes {
            closing_seen += 1;
            if closing_seen > n {
                return index + 1;
            }
        }
    }
    0
}

/// Run after the completion pump has drained.
pub fn quiescent(s: &State) -> Result<(), Error> {
    if s.open > 0 {
        return Err(Error::UncommittedAtExit { open: s.open });
    }
    if let Some(g) = s
        .groups
        .iter()
        .find(|g| g.milestone != Milestone::FullyDone)
    {
        return Err(Error::PendingAtExit { ordinal: g.ordinal });
    }
    Ok(())
}

pub fn check_invariants(s: &State) -> Result<(), String> {
    let mut previous: Option<&Group> = None;
    let mut previous_nonempty: Option<&Group> = None;
    for g in &s.groups {
        if let Some(p) = previous {
            if p.ordinal >= g.ordinal {
                return Err("ordinals not increasing".into());
            }
        }
        if g.ops != 0 {
            if let Some(p) = previous_nonempty {
                if p.milestone < g.milestone {
                    return Err("milestone overtook an older group (FIFO)".into());
                }
            }
            previous_nonempty = Some(g);
        }
        if g.ops == 0 && g.milestone != Milestone::FullyDone {
            return Err("empty group not complete".into());
        }
        if g.milestone == Milestone::FullyDone && g.arrivals != 0 {
            return Err("arrivals held past full completion".into());
        }
        if g.ordinal >= s.next_ordinal {
            return Err("ordinal from the future".into());
        }
        previous = Some(g);
    }
    Ok(())
}
