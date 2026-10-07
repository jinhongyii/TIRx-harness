//! Reference model of the tcgen05 TMEM lifecycle of one CTA pair, the
//! kernel-wide `cta_group` rule, and the per-thread work queues that
//! `tcgen05.commit` drains.
//!
//! Spec: `docs/development/sync-semantics.md` §6, with the ISA answers in
//! `sync-isa-answers.md` Q1/Q6. All PTX citations refer to PTX 9.4
//! §9.7.18.7.1.
//!
//! - `tcgen05.alloc` blocks until the columns are free. An `.exclusive`
//!   allocation blocks until no other allocation is live. While one is live,
//!   every other allocation blocks.
//! - The allocated column count may not increase between any two allocations
//!   of a CTA, in execution order. The last count is sticky across deallocs.
//! - Allocating after `relinquish_alloc_permit` is illegal.
//!
//! In `ctas`, index 0 is the even CTA of the pair and index 1 is its peer.
//! A `cta_group::2` command has already been matched by the collective layer.
//! That layer requires one warp from each peer CTA and no particular warp
//! index. The command mutates both CTAs atomically, at one common base.
//!
//! Every tcgen05 instruction of a kernel first steps the kernel-wide
//! [`KernelState`], because all of them must use the same `.cta_group`.

pub const TMEM_COLUMNS: u32 = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Allocation {
    pub base: u32,
    pub columns: u32,
    pub exclusive: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct CtaTmem {
    /// Sorted by base.
    pub allocations: Vec<Allocation>,
    pub relinquished: bool,
    pub last_alloc_columns: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    /// Columns available to non-exclusive allocations (512).
    pub capacity: u32,
    /// Largest `.exclusive` allocation: 512 on sm_100f/103/110, 576 on
    /// sm_107f (PTX Table 58).
    pub exclusive_max: u32,
    pub ctas: [CtaTmem; 2],
}

impl Default for State {
    fn default() -> Self {
        Self::new(TMEM_COLUMNS)
    }
}

impl State {
    pub fn new(exclusive_max: u32) -> Self {
        Self {
            capacity: TMEM_COLUMNS,
            exclusive_max,
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
    pub fn group(self) -> u8 {
        match self {
            Who::One(_) => 1,
            Who::Pair => 2,
        }
    }

    fn indices(self) -> Result<&'static [usize], Error> {
        match self {
            Who::One(0) => Ok(&[0]),
            Who::One(1) => Ok(&[1]),
            Who::One(_) => Err(Error::InvalidCtaGroup),
            Who::Pair => Ok(&[0, 1]),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cmd {
    /// `tcgen05.alloc[.exclusive].cta_group::N.sync.aligned`, by a full warp.
    /// Retried while `Blocked`.
    Alloc {
        who: Who,
        columns: u32,
        exclusive: bool,
    },
    /// `tcgen05.dealloc[.exclusive].cta_group::N.sync.aligned` at `taddr`.
    Dealloc {
        who: Who,
        taddr: u32,
        columns: u32,
        exclusive: bool,
    },
    /// `tcgen05.relinquish_alloc_permit.cta_group::N.sync.aligned`.
    Relinquish { who: Who },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    Allocated { base: u32 },
    Blocked,
    Done,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Error {
    InvalidCtaGroup,
    /// "All tcgen05 instructions within a kernel must specify the same value
    /// for the .cta_group qualifier."
    CtaGroupMismatch {
        established: u8,
        requested: u8,
    },
    InvalidColumns {
        columns: u32,
    },
    AllocAfterRelinquish,
    AllocationSizeIncrease {
        previous: u32,
        requested: u32,
    },
    /// No live allocation with this base, width and exclusivity.
    DeallocationMismatch {
        taddr: u32,
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

/// Non-exclusive widths are powers of two in [32, 512]. Exclusive widths are
/// multiples of 32 in [32, exclusive_max].
pub fn valid_columns(s: &State, columns: u32, exclusive: bool) -> bool {
    if exclusive {
        (32..=s.exclusive_max).contains(&columns) && columns.is_multiple_of(32)
    } else {
        (32..=s.capacity).contains(&columns) && columns.is_power_of_two()
    }
}

pub fn step(state: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
    let mut next = state.clone();
    let outcome = apply(&mut next, cmd)?;
    *state = next;
    Ok(outcome)
}

fn apply(s: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
    match cmd {
        Cmd::Alloc {
            who,
            columns,
            exclusive,
        } => {
            let idx = who.indices()?;
            if !valid_columns(s, columns, exclusive) {
                return Err(Error::InvalidColumns { columns });
            }
            for &i in idx {
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
            let Some(base) = first_fit(s, idx, columns, exclusive) else {
                return Ok(Outcome::Blocked);
            };
            for &i in idx {
                let cta = &mut s.ctas[i];
                cta.allocations.push(Allocation {
                    base,
                    columns,
                    exclusive,
                });
                cta.allocations.sort_by_key(|a| a.base);
                cta.last_alloc_columns = Some(columns);
            }
            Ok(Outcome::Allocated { base })
        }
        Cmd::Dealloc {
            who,
            taddr,
            columns,
            exclusive,
        } => {
            let idx = who.indices()?;
            let wanted = Allocation {
                base: taddr,
                columns,
                exclusive,
            };
            for &i in idx {
                if !s.ctas[i].allocations.contains(&wanted) {
                    return Err(Error::DeallocationMismatch { taddr, columns });
                }
            }
            for &i in idx {
                s.ctas[i].allocations.retain(|a| *a != wanted);
            }
            Ok(Outcome::Done)
        }
        Cmd::Relinquish { who } => {
            for &i in who.indices()? {
                s.ctas[i].relinquished = true;
            }
            Ok(Outcome::Done)
        }
    }
}

/// Lowest 32-aligned base that is free in every participating CTA, or
/// `None` while the allocation must block.
fn first_fit(s: &State, idx: &[usize], columns: u32, exclusive: bool) -> Option<u32> {
    let ctas = || idx.iter().map(|&i| &s.ctas[i]);
    if exclusive {
        return ctas().all(|c| c.allocations.is_empty()).then_some(0);
    }
    if ctas().any(|c| c.allocations.iter().any(|a| a.exclusive)) {
        return None;
    }
    (0..=s.capacity.checked_sub(columns)?)
        .step_by(32)
        .find(|&base| {
            ctas().all(|c| {
                c.allocations
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
            if a.base < end || !a.base.is_multiple_of(32) {
                return Err("overlapping, unsorted or misaligned allocations".into());
            }
            let limit = if a.exclusive {
                s.exclusive_max
            } else {
                s.capacity
            };
            if a.base + a.columns > limit {
                return Err("allocation outside TMEM".into());
            }
            end = a.base + a.columns;
        }
        if cta.allocations.iter().any(|a| a.exclusive) && cta.allocations.len() != 1 {
            return Err("exclusive allocation is not the sole live allocation".into());
        }
        if !cta.allocations.is_empty() && cta.last_alloc_columns.is_none() {
            return Err("allocation without a recorded width".into());
        }
    }
    Ok(())
}

/// Kernel-wide `.cta_group` uniformity across every tcgen05 instruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct KernelState {
    pub cta_group: Option<u8>,
}

pub fn use_cta_group(k: &mut KernelState, cta_group: u8) -> Result<(), Error> {
    if !matches!(cta_group, 1 | 2) {
        return Err(Error::InvalidCtaGroup);
    }
    match k.cta_group {
        Some(established) if established != cta_group => Err(Error::CtaGroupMismatch {
            established,
            requested: cta_group,
        }),
        _ => {
            k.cta_group = Some(cta_group);
            Ok(())
        }
    }
}

/// Per-thread tcgen05 work queues (`ordering.rs:67-99`). Tokens are counted
/// because the lifecycle model has no footprints. Kernel-wide `cta_group`
/// uniformity is checked through [`use_cta_group`] before every command.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct WorkState {
    /// mma / cp / shift issued, not yet committed.
    pub uncommitted: u32,
    pub loads: u32,
    pub stores: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkCmd {
    Issue,
    Load,
    Store,
    /// `tcgen05.commit`. The arrive-on is
    /// `mbarrier::Cmd::{Issue, DeferredArrive { count: 1 }}`.
    Commit,
    WaitLd,
    WaitSt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkOutcome {
    Queued,
    Drained { tokens: u32 },
}

pub fn work_step(s: &mut WorkState, cmd: WorkCmd) -> WorkOutcome {
    let drain = |n: &mut u32| WorkOutcome::Drained {
        tokens: std::mem::take(n),
    };
    match cmd {
        WorkCmd::Issue => {
            s.uncommitted += 1;
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
        WorkCmd::Commit => drain(&mut s.uncommitted),
        WorkCmd::WaitLd => drain(&mut s.loads),
        WorkCmd::WaitSt => drain(&mut s.stores),
    }
}
