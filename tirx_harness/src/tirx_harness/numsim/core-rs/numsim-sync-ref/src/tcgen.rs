//! Reference model of the tcgen05 TMEM lifecycle of one CTA pair, plus the
//! per-thread work queues that `tcgen05.commit` drains.
//!
//! Spec: `docs/development/sync-semantics.md` §6. Index 0 of `ctas` is the
//! even CTA of the pair and index 1 is its peer. A `cta_group::2` command has
//! already been matched by the collective layer. That layer rejects
//! `CollectiveArgumentMismatch`, `MissingPeerCta`, and a peer that never
//! arrives. The command then mutates both CTAs atomically at one common base
//! (`tcgen.rs:713-733, 798-950`).
//!
//! Numerics are eager. "Pending" tcgen work exists only as tokens that
//! `commit` and `wait::ld/st` drain. The `commit` arrival itself is
//! `mbarrier::Cmd::{Issue, DeferredArrive { count: 1 }}` on each target
//! barrier.

pub const TMEM_COLUMNS: u32 = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Allocation {
    pub base: u32,
    pub columns: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct CtaTmem {
    /// Sorted by base.
    pub allocations: Vec<Allocation>,
    pub relinquished: bool,
    /// Sticky after the first lifecycle action (`tcgen.rs:844-858`).
    pub cta_group: Option<u8>,
    /// Engine rule: later allocations may not be wider (`tcgen.rs:871-887`).
    pub last_alloc_columns: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    pub capacity: u32,
    pub ctas: [CtaTmem; 2],
}

impl Default for State {
    fn default() -> Self {
        Self {
            capacity: TMEM_COLUMNS,
            ctas: Default::default(),
        }
    }
}

/// Which CTAs a lifecycle command acts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Who {
    /// `cta_group::1` on CTA 0 or 1.
    One(u8),
    /// `cta_group::2` on both CTAs of the pair.
    Pair,
}

impl Who {
    fn group(self) -> u8 {
        match self {
            Who::One(_) => 1,
            Who::Pair => 2,
        }
    }

    fn indices(self) -> &'static [usize] {
        match self {
            Who::One(0) => &[0],
            Who::One(_) => &[1],
            Who::Pair => &[0, 1],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cmd {
    /// `tcgen05.alloc[.exclusive?].cta_group::N.sync.aligned` (full warp).
    Alloc {
        who: Who,
        columns: u32,
        exclusive: bool,
    },
    /// `tcgen05.dealloc.cta_group::N.sync.aligned` with the base `taddr`.
    Dealloc { who: Who, taddr: u32, columns: u32 },
    /// `tcgen05.relinquish_alloc_permit.cta_group::N.sync.aligned`.
    Relinquish { who: Who },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    Allocated { base: u32 },
    Done,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Error {
    InvalidCtaGroup,
    InvalidColumns {
        columns: u32,
    },
    CtaGroupMismatch {
        established: u8,
        requested: u8,
    },
    AllocAfterRelinquish,
    AllocationSizeIncrease {
        previous: u32,
        requested: u32,
    },
    DeallocationMismatch {
        taddr: u32,
        columns: u32,
    },
    /// PTX alloc blocks until columns are free. The engine and verifier fail
    /// instead (`tcgen.rs:952-966`), so this model does too; see spec §6.5.
    AllocationUnavailable {
        columns: u32,
    },
    LiveAllocationsAtExit {
        cta: u8,
    },
}

pub struct Tcgen;

impl crate::Protocol for Tcgen {
    type State = State;
    type Cmd = Cmd;
    type Outcome = Outcome;
    type Error = Error;
    fn step(state: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
        step(state, cmd)
    }
}

/// `tcgen.rs:703-710`: a power of two in `[32, 512]`, or any multiple of 32
/// for `.exclusive`, bounded by the TMEM capacity.
pub fn valid_columns(columns: u32, exclusive: bool, capacity: u32) -> bool {
    (32..=capacity.min(TMEM_COLUMNS)).contains(&columns)
        && if exclusive {
            columns.is_multiple_of(32)
        } else {
            columns.is_power_of_two()
        }
}

pub fn step(state: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
    let mut next = state.clone();
    let outcome = apply(&mut next, cmd)?;
    *state = next;
    Ok(outcome)
}

fn apply(s: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
    let who = match cmd {
        Cmd::Alloc { who, .. } | Cmd::Dealloc { who, .. } | Cmd::Relinquish { who } => who,
    };
    if let Who::One(i) = who {
        if i > 1 {
            return Err(Error::InvalidCtaGroup);
        }
    }
    let group = who.group();
    for &i in who.indices() {
        if let Some(established) = s.ctas[i].cta_group {
            if established != group {
                return Err(Error::CtaGroupMismatch {
                    established,
                    requested: group,
                });
            }
        }
    }
    let outcome = match cmd {
        Cmd::Alloc {
            columns, exclusive, ..
        } => {
            if !valid_columns(columns, exclusive, s.capacity) {
                return Err(Error::InvalidColumns { columns });
            }
            for &i in who.indices() {
                let cta = &s.ctas[i];
                if cta.relinquished {
                    return Err(Error::AllocAfterRelinquish);
                }
                if let Some(previous) = cta.last_alloc_columns {
                    if columns > previous {
                        return Err(Error::AllocationSizeIncrease {
                            previous,
                            requested: columns,
                        });
                    }
                }
            }
            let base =
                first_fit(s, who, columns).ok_or(Error::AllocationUnavailable { columns })?;
            for &i in who.indices() {
                let cta = &mut s.ctas[i];
                cta.allocations.push(Allocation { base, columns });
                cta.allocations.sort_by_key(|a| a.base);
                cta.last_alloc_columns = Some(columns);
            }
            Outcome::Allocated { base }
        }
        Cmd::Dealloc { taddr, columns, .. } => {
            let wanted = Allocation {
                base: taddr,
                columns,
            };
            for &i in who.indices() {
                if !s.ctas[i].allocations.contains(&wanted) {
                    return Err(Error::DeallocationMismatch { taddr, columns });
                }
            }
            for &i in who.indices() {
                s.ctas[i].allocations.retain(|a| *a != wanted);
            }
            Outcome::Done
        }
        Cmd::Relinquish { .. } => {
            for &i in who.indices() {
                s.ctas[i].relinquished = true;
            }
            Outcome::Done
        }
    };
    for &i in who.indices() {
        s.ctas[i].cta_group = Some(group);
    }
    Ok(outcome)
}

/// Lowest 32-aligned base free in every participating CTA.
fn first_fit(s: &State, who: Who, columns: u32) -> Option<u32> {
    let limit = s.capacity.min(TMEM_COLUMNS);
    (0..=limit.checked_sub(columns)?).step_by(32).find(|&base| {
        who.indices().iter().all(|&i| {
            s.ctas[i]
                .allocations
                .iter()
                .all(|a| base + columns <= a.base || a.base + a.columns <= base)
        })
    })
}

pub fn quiescent(s: &State) -> Result<(), Error> {
    for (i, cta) in s.ctas.iter().enumerate() {
        if !cta.allocations.is_empty() {
            return Err(Error::LiveAllocationsAtExit { cta: i as u8 });
        }
    }
    Ok(())
}

pub fn check_invariants(s: &State) -> Result<(), String> {
    for cta in &s.ctas {
        let mut end = 0;
        for a in &cta.allocations {
            if a.base < end {
                return Err("overlapping or unsorted allocations".into());
            }
            if !a.base.is_multiple_of(32) || a.base + a.columns > s.capacity.min(TMEM_COLUMNS) {
                return Err("allocation outside TMEM".into());
            }
            end = a.base + a.columns;
        }
        if !cta.allocations.is_empty() && cta.cta_group.is_none() {
            return Err("allocation without established cta_group".into());
        }
    }
    Ok(())
}

/// Per-thread tcgen05 work queues (`ordering.rs:67-99`). Tokens are counted
/// because the lifecycle model has no footprints.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct WorkState {
    /// mma / cp / shift issued per cta_group, not yet committed.
    pub uncommitted: [u32; 2],
    pub loads: u32,
    pub stores: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkCmd {
    Issue {
        cta_group: u8,
    },
    Load,
    Store,
    /// `tcgen05.commit.cta_group::N`: drains only that group's work. Work
    /// issued under the other group stays (`ordering.rs:815-849`).
    Commit {
        cta_group: u8,
    },
    WaitLd,
    WaitSt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkOutcome {
    Queued,
    Drained { tokens: u32 },
}

pub fn work_step(s: &mut WorkState, cmd: WorkCmd) -> Result<WorkOutcome, Error> {
    let slot = |g: u8| match g {
        1 => Ok(0),
        2 => Ok(1),
        _ => Err(Error::InvalidCtaGroup),
    };
    Ok(match cmd {
        WorkCmd::Issue { cta_group } => {
            s.uncommitted[slot(cta_group)?] += 1;
            WorkOutcome::Queued
        }
        WorkCmd::Load => {
            s.loads += 1;
            WorkOutcome::Queued
        }
        WorkCmd::Store => {
            s.stores += 1;
            WorkOutcome::Queued
        }
        WorkCmd::Commit { cta_group } => WorkOutcome::Drained {
            tokens: std::mem::take(&mut s.uncommitted[slot(cta_group)?]),
        },
        WorkCmd::WaitLd => WorkOutcome::Drained {
            tokens: std::mem::take(&mut s.loads),
        },
        WorkCmd::WaitSt => WorkOutcome::Drained {
            tokens: std::mem::take(&mut s.stores),
        },
    })
}
