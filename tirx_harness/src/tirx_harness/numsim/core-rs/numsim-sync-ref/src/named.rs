//! Reference model of one CTA-local named barrier (`bar.sync`, `bar.arrive`,
//! `barrier.sync[.aligned]`, `bar.red` without the reduction value).
//!
//! Spec: `docs/development/sync-semantics.md` §3. Arrivals are counted per
//! active lane (`hardware_barriers.rs:3019-3036`). A generation completes when
//! its lane count reaches the count `b` carried by its contributions. The next
//! contribution then rolls lazily to a fresh generation, which may use a
//! different `b` (`hardware_barriers.rs:2983-2997`).
//!
//! Blocking: `Sync` contributes and returns `Registered { gen }`, or `Ready`
//! when it completed the generation. The warp then retries `Resume { gen }`
//! until it is `Ready`. Readiness is a pure function of `(gen, state)`, so the
//! model needs no waiter registry.

use std::collections::BTreeMap;

use crate::{LaneMask, Policy, Warp, FULL_MASK};

/// Hardware barrier ids are `0..16`; the caller resolves the id to a `State`.
pub const NUM_IDS: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Flavor {
    Arrive,
    Sync,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Contribution {
    pub warp: Warp,
    pub mask: LaneMask,
    /// Thread count operand `b` (whole CTA when the source omitted it).
    pub count: u64,
    /// `bar.*` and `barrier.*.aligned` forms.
    pub aligned: bool,
    /// Entry mask of an enclosing `elect.sync` region, if any. The engine
    /// waives the full-warp check under elect (`kernel_engine.rs:7902-7915`);
    /// strict requires the participation to equal that entry mask
    /// (`strict_named_barrier.rs:350-369`).
    pub elect_entry: Option<LaneMask>,
    /// Static instruction identity, for the strict aligned-origin contract.
    pub site: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct State {
    pub policy: Policy,
    /// Count `b` of the current generation; `None` before first use.
    pub expected: Option<u64>,
    pub gen: u64,
    pub complete: bool,
    pub arrived: u64,
    pub lanes: BTreeMap<(Warp, Flavor), LaneMask>,
    /// Strict only: aligned flag and static sites of sync contributions.
    pub sync_origins: BTreeMap<Warp, Vec<(bool, u32)>>,
}

impl State {
    pub fn new(policy: Policy) -> Self {
        Self {
            policy,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cmd {
    Arrive(Contribution),
    Sync(Contribution),
    /// Retry of a blocked `Sync` that registered on `gen`.
    Resume {
        gen: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    Arrived { gen: u64, completed: bool },
    Registered { gen: u64 },
    Ready { gen: u64 },
    Blocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Error {
    /// `b == 0` or `b % 32 != 0` (`runtime/sync.rs:1427-1436`).
    InvalidCount {
        count: u64,
    },
    EmptyMask,
    /// `.aligned` with a partial warp outside an elect region.
    PartialWarp {
        mask: LaneMask,
    },
    /// Strict: an elect-gated participation differs from the elect entry mask.
    ElectSyncParticipation {
        entry: LaneMask,
        mask: LaneMask,
    },
    /// Contributions to one generation carry different `b`.
    ContractMismatch {
        expected: u64,
        observed: u64,
    },
    /// The same lanes contributed twice with the same flavor in one generation.
    Duplicate {
        warp: Warp,
        overlap: LaneMask,
    },
    ArrivalOverflow {
        expected: u64,
        arrived: u64,
    },
    /// Strict: aligned and unaligned syncs mixed, or one warp's aligned syncs
    /// from different static instructions.
    AlignedSyncContractMismatch {
        warp: Warp,
    },
    /// `Resume` on a generation that this barrier never reached.
    ResumeFuture {
        gen: u64,
    },
    /// Exit with an incomplete generation (`hardware_barriers.rs:3132-3154`).
    IncompleteAtExit {
        gen: u64,
        arrived: u64,
        expected: u64,
    },
}

pub struct Named;

impl crate::Protocol for Named {
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

fn apply(s: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
    match cmd {
        Cmd::Arrive(c) => {
            let (gen, completed) = contribute(s, c, Flavor::Arrive)?;
            Ok(Outcome::Arrived { gen, completed })
        }
        Cmd::Sync(c) => {
            let (gen, completed) = contribute(s, c, Flavor::Sync)?;
            Ok(if completed {
                Outcome::Ready { gen }
            } else {
                Outcome::Registered { gen }
            })
        }
        Cmd::Resume { gen } => {
            if gen > s.gen || s.expected.is_none() {
                Err(Error::ResumeFuture { gen })
            } else if gen < s.gen || s.complete {
                Ok(Outcome::Ready { gen })
            } else {
                Ok(Outcome::Blocked)
            }
        }
    }
}

fn contribute(s: &mut State, c: Contribution, flavor: Flavor) -> Result<(u64, bool), Error> {
    if c.count == 0 || !c.count.is_multiple_of(32) {
        return Err(Error::InvalidCount { count: c.count });
    }
    if c.mask == 0 {
        return Err(Error::EmptyMask);
    }
    // Engine: every arrive and every aligned sync is warp-collective unless
    // elect-gated (kernel_engine.rs:4191-4228, 7902-7915).
    let collective = flavor == Flavor::Arrive || c.aligned;
    if collective && c.mask != FULL_MASK && c.elect_entry.is_none() {
        return Err(Error::PartialWarp { mask: c.mask });
    }
    if s.policy == Policy::Strict && collective {
        if let Some(entry) = c.elect_entry {
            if entry != c.mask {
                return Err(Error::ElectSyncParticipation {
                    entry,
                    mask: c.mask,
                });
            }
        }
    }
    if s.complete || s.expected.is_none() {
        if s.complete {
            s.gen += 1;
        }
        s.complete = false;
        s.expected = Some(c.count);
        s.arrived = 0;
        s.lanes.clear();
        s.sync_origins.clear();
    }
    let expected = s.expected.expect("set above");
    if expected != c.count {
        return Err(Error::ContractMismatch {
            expected,
            observed: c.count,
        });
    }
    let prior = s.lanes.get(&(c.warp, flavor)).copied().unwrap_or(0);
    if prior & c.mask != 0 {
        return Err(Error::Duplicate {
            warp: c.warp,
            overlap: prior & c.mask,
        });
    }
    let arrived = s.arrived + u64::from(c.mask.count_ones());
    if arrived > expected {
        return Err(Error::ArrivalOverflow { expected, arrived });
    }
    s.arrived = arrived;
    s.lanes.insert((c.warp, flavor), prior | c.mask);
    if flavor == Flavor::Sync {
        s.sync_origins
            .entry(c.warp)
            .or_default()
            .push((c.aligned, c.site));
    }
    if arrived == expected {
        if s.policy == Policy::Strict {
            check_aligned_origins(s)?;
        }
        s.complete = true;
    }
    Ok((s.gen, s.complete))
}

/// `strict_named_barrier.rs:719-773`: if any sync of the generation is
/// aligned, every sync is aligned, and each warp uses one static site.
fn check_aligned_origins(s: &State) -> Result<(), Error> {
    let any_aligned = s
        .sync_origins
        .values()
        .flatten()
        .any(|&(aligned, _)| aligned);
    if !any_aligned {
        return Ok(());
    }
    for (&warp, origins) in &s.sync_origins {
        let first_site = origins[0].1;
        if origins
            .iter()
            .any(|&(aligned, site)| !aligned || site != first_site)
        {
            return Err(Error::AlignedSyncContractMismatch { warp });
        }
    }
    Ok(())
}

/// Launch-exit check: an incomplete generation is an error, including one fed
/// only by `bar.arrive`.
pub fn quiescent(s: &State) -> Result<(), Error> {
    match s.expected {
        Some(expected) if !s.complete => Err(Error::IncompleteAtExit {
            gen: s.gen,
            arrived: s.arrived,
            expected,
        }),
        _ => Ok(()),
    }
}

pub fn check_invariants(s: &State) -> Result<(), String> {
    let expected = match s.expected {
        None => {
            return (s.gen == 0 && !s.complete && s.arrived == 0 && s.lanes.is_empty())
                .then_some(())
                .ok_or_else(|| "state before first use".into())
        }
        Some(e) => e,
    };
    if s.arrived > expected {
        return Err("arrived > expected".into());
    }
    if s.complete != (s.arrived == expected) {
        return Err("complete iff arrived == expected".into());
    }
    let lanes: u64 = s.lanes.values().map(|m| u64::from(m.count_ones())).sum();
    if lanes != s.arrived {
        return Err("lane bookkeeping disagrees with count".into());
    }
    Ok(())
}
