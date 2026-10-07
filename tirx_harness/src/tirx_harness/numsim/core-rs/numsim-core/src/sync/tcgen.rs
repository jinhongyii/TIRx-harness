#![allow(unused_variables, dead_code)]
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
//!
//! CONTRACT: the types in this file are copied verbatim from
//! `numsim-sync-ref/src/tcgen.rs` (plus serde derives) so production `step`
//! is differentially tested against the reference mechanically. Change
//! them only together with the reference crate, via the coordinator.
//! Function bodies marked `W3` are the production implementation to write.

pub const TMEM_COLUMNS: u32 = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Allocation {
    pub base: u32,
    pub columns: u32,
    pub exclusive: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CtaTmem {
    /// Sorted by base.
    pub allocations: Vec<Allocation>,
    pub relinquished: bool,
    pub last_alloc_columns: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Outcome {
    Allocated { base: u32 },
    Blocked,
    Done,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

impl super::Protocol for Tcgen {
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
    let (max, shape_ok) = if exclusive {
        (s.exclusive_max, columns.is_multiple_of(32))
    } else {
        (s.capacity, columns.is_power_of_two())
    };
    columns >= 32 && columns <= max && shape_ok
}

pub fn step(state: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
    match cmd {
        Cmd::Alloc { who, columns, exclusive } => {
            let ctas = participants(who)?;
            if !valid_columns(state, columns, exclusive) {
                return Err(Error::InvalidColumns { columns });
            }
            for &i in ctas {
                let cta = &state.ctas[i];
                if cta.relinquished {
                    return Err(Error::AllocAfterRelinquish);
                }
                if let Some(previous) = cta.last_alloc_columns.filter(|&p| columns > p) {
                    return Err(Error::AllocationSizeIncrease { previous, requested: columns });
                }
            }
            // PTX §9.7.18.7.1: alloc blocks until the columns are free.
            let Some(base) = free_base(state, ctas, columns, exclusive) else {
                return Ok(Outcome::Blocked);
            };
            for &i in ctas {
                let cta = &mut state.ctas[i];
                let at = cta.allocations.partition_point(|a| a.base < base);
                cta.allocations.insert(at, Allocation { base, columns, exclusive });
                cta.last_alloc_columns = Some(columns);
            }
            Ok(Outcome::Allocated { base })
        }
        Cmd::Dealloc { who, taddr, columns, exclusive } => {
            let ctas = participants(who)?;
            let wanted = Allocation { base: taddr, columns, exclusive };
            let mut found = [0usize; 2];
            for (k, &i) in ctas.iter().enumerate() {
                found[k] = state.ctas[i]
                    .allocations
                    .iter()
                    .position(|a| *a == wanted)
                    .ok_or(Error::DeallocationMismatch { taddr, columns })?;
            }
            for (k, &i) in ctas.iter().enumerate() {
                state.ctas[i].allocations.remove(found[k]);
            }
            Ok(Outcome::Done)
        }
        Cmd::Relinquish { who } => {
            for &i in participants(who)? {
                state.ctas[i].relinquished = true;
            }
            Ok(Outcome::Done)
        }
    }
}



pub fn quiescent(s: &State) -> Result<(), Error> {
    match s.ctas.iter().position(|c| !c.allocations.is_empty()) {
        Some(i) => Err(Error::LiveAllocationsAtExit { cta: i as u8 }),
        None => Ok(()),
    }
}

pub fn check_invariants(s: &State) -> Result<(), String> {
    for cta in &s.ctas {
        for w in cta.allocations.windows(2) {
            if w[0].base + w[0].columns > w[1].base {
                return Err("overlapping or unsorted allocations".into());
            }
        }
        for a in &cta.allocations {
            let limit = if a.exclusive { s.exclusive_max } else { s.capacity };
            if a.base % 32 != 0 || a.base + a.columns > limit {
                return Err("allocation outside TMEM or misaligned".into());
            }
            if a.exclusive && cta.allocations.len() != 1 {
                return Err("exclusive allocation is not the sole live allocation".into());
            }
        }
        if !cta.allocations.is_empty() && cta.last_alloc_columns.is_none() {
            return Err("allocation without a recorded width".into());
        }
    }
    Ok(())
}

/// Kernel-wide `.cta_group` uniformity across every tcgen05 instruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct KernelState {
    pub cta_group: Option<u8>,
}

pub fn use_cta_group(k: &mut KernelState, cta_group: u8) -> Result<(), Error> {
    if cta_group != 1 && cta_group != 2 {
        return Err(Error::InvalidCtaGroup);
    }
    if let Some(established) = k.cta_group.filter(|&g| g != cta_group) {
        return Err(Error::CtaGroupMismatch { established, requested: cta_group });
    }
    k.cta_group = Some(cta_group);
    Ok(())
}

/// Per-thread tcgen05 work queues (`ordering.rs:67-99`). Tokens are counted
/// because the lifecycle model has no footprints. Kernel-wide `cta_group`
/// uniformity is checked through [`use_cta_group`] before every command.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct WorkState {
    /// mma / cp / shift issued, not yet committed.
    pub uncommitted: u32,
    pub loads: u32,
    pub stores: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum WorkOutcome {
    Queued,
    Drained { tokens: u32 },
}

pub fn work_step(s: &mut WorkState, cmd: WorkCmd) -> WorkOutcome {
    let counter = match cmd {
        WorkCmd::Issue | WorkCmd::Commit => &mut s.uncommitted,
        WorkCmd::Load | WorkCmd::WaitLd => &mut s.loads,
        WorkCmd::Store | WorkCmd::WaitSt => &mut s.stores,
    };
    match cmd {
        WorkCmd::Issue | WorkCmd::Load | WorkCmd::Store => {
            *counter += 1;
            WorkOutcome::Queued
        }
        WorkCmd::Commit | WorkCmd::WaitLd | WorkCmd::WaitSt => WorkOutcome::Drained { tokens: std::mem::take(counter) },
    }
}

fn participants(who: Who) -> Result<&'static [usize], Error> {
    match who {
        Who::One(0) => Ok(&[0]),
        Who::One(1) => Ok(&[1]),
        Who::Pair => Ok(&[0, 1]),
        Who::One(_) => Err(Error::InvalidCtaGroup),
    }
}

/// Lowest 32-aligned base free in every participating CTA; `None` = block.
/// `.exclusive` needs no live allocation at all; any allocation blocks while
/// an exclusive one is live.
fn free_base(s: &State, ctas: &[usize], columns: u32, exclusive: bool) -> Option<u32> {
    let live = |i: usize| s.ctas[i].allocations.iter();
    if exclusive {
        return ctas.iter().all(|&i| live(i).next().is_none()).then_some(0);
    }
    if ctas.iter().any(|&i| live(i).any(|a| a.exclusive)) {
        return None;
    }
    let mut base = 0u32;
    while base + columns <= s.capacity {
        // Jump past the first conflicting allocation instead of probing
        // every 32-column step.
        let conflict = ctas
            .iter()
            .flat_map(|&i| live(i))
            .filter(|a| base < a.base + a.columns && a.base < base + columns)
            .map(|a| a.base + a.columns)
            .max();
        match conflict {
            None => return Some(base),
            Some(end) => base = end.div_ceil(32) * 32,
        }
    }
    None
}

use crate::report::FindingKind;

/// Map a protocol error to its report kind.
pub fn finding_kind(_e: &Error) -> FindingKind {
    FindingKind::TmemMisuse
}
