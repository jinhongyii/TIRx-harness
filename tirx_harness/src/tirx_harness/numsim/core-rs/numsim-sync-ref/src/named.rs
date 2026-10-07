//! Reference model of one CTA-local named barrier: `bar.sync`, `bar.arrive`,
//! `bar.red`, and `barrier.{sync,arrive,red}[.aligned]`.
//!
//! Spec: `docs/development/sync-semantics.md` §3, with the ISA answers in
//! `sync-isa-answers.md` Q3/Q5.
//!
//! A barrier instruction makes each executing thread wait for all non-exited
//! threads of its warp, then marks the **warp's** arrival (PTX 9.4 §9.7.15.1).
//! The model therefore requires each contribution's lane mask to equal the
//! warp's non-exited lanes. A strict subset is an error. That covers the
//! elect-gated single-lane case and lanes that reach different barrier
//! instructions; the latter fail closed. A warp arrival counts 32 threads
//! toward `b`, which must be a multiple of the warp size.
//!
//! `.aligned` is a convergence promise, not barrier state. Mixing aligned and
//! unaligned forms on one barrier is legal. `Contribution::aligned` is carried
//! for the checkers (the synccheck aligned-site rule) and never changes a
//! transition. Mixing `.red` with `sync`/`arrive`
//! on one active barrier is "unpredictable" and is rejected.
//!
//! Blocking: `Sync`/`Red` contribute and return `Registered { gen }`, or
//! `Ready` when they complete the generation. The warp then retries
//! `Resume { gen }`. Readiness is a pure function of `gen`, so no waiter
//! registry is needed.

use std::collections::BTreeMap;

use crate::{LaneMask, Warp};

/// Hardware barrier ids are `0..16`; the caller resolves the id to a `State`.
pub const NUM_IDS: u32 = 16;
pub const WARP_SIZE: u64 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Flavor {
    Arrive,
    Sync,
    Red,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Contribution {
    pub warp: Warp,
    /// Lanes executing this barrier instruction.
    pub mask: LaneMask,
    /// Non-exited lanes of the warp at this point.
    pub live: LaneMask,
    /// Thread count `b`. Whole CTA when the source omitted it.
    pub count: u64,
    /// `bar.*` / `barrier.*.aligned` form. No effect on the transition.
    pub aligned: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct State {
    /// `b` of the current generation; `None` before first use.
    pub expected: Option<u64>,
    pub gen: u64,
    pub complete: bool,
    /// Threads counted so far (32 per warp arrival).
    pub arrived: u64,
    /// Warp arrivals of the current generation, by flavor.
    pub warps: BTreeMap<(Warp, Flavor), ()>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cmd {
    Arrive(Contribution),
    Sync(Contribution),
    Red(Contribution),
    /// Retry of a blocked `Sync`/`Red` that registered on `gen`.
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
    /// `b == 0` or `b` not a multiple of 32 (PTX §9.7.15.1).
    InvalidCount {
        count: u64,
    },
    /// Executed by no lane, or by a strict subset of the warp's non-exited
    /// lanes (PTX §9.7.15.1, sync-isa-answers Q3/Q5).
    PartialWarp {
        mask: LaneMask,
        live: LaneMask,
    },
    /// Contributions to one generation carry different `b`
    /// ("using the same barrier name and thread count").
    ContractMismatch {
        expected: u64,
        observed: u64,
    },
    /// One warp contributed twice with the same flavor in one generation
    /// ("keep a warp from executing more barrier instructions than intended").
    Duplicate {
        warp: Warp,
    },
    /// `.red` mixed with `sync`/`arrive` on one active barrier.
    RedMixed {
        warp: Warp,
    },
    ArrivalOverflow {
        expected: u64,
        arrived: u64,
    },
    /// `Resume` on a generation that this barrier never reached.
    ResumeFuture {
        gen: u64,
    },
}

/// Review-level lint: not a kernel error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lint {
    /// Generation left incomplete at exit (a dangling `bar.arrive`). PTX
    /// releases barriers waiting only on exited threads (§9.7.14.7).
    DanglingAtExit {
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
    let blocking = |(gen, completed)| {
        if completed {
            Outcome::Ready { gen }
        } else {
            Outcome::Registered { gen }
        }
    };
    match cmd {
        Cmd::Arrive(c) => {
            let (gen, completed) = contribute(s, c, Flavor::Arrive)?;
            Ok(Outcome::Arrived { gen, completed })
        }
        Cmd::Sync(c) => contribute(s, c, Flavor::Sync).map(blocking),
        Cmd::Red(c) => contribute(s, c, Flavor::Red).map(blocking),
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
    if c.count == 0 || !c.count.is_multiple_of(WARP_SIZE) {
        return Err(Error::InvalidCount { count: c.count });
    }
    if c.mask == 0 || c.mask != c.live {
        return Err(Error::PartialWarp {
            mask: c.mask,
            live: c.live,
        });
    }
    if s.complete || s.expected.is_none() {
        if s.complete {
            s.gen += 1;
        }
        s.complete = false;
        s.expected = Some(c.count);
        s.arrived = 0;
        s.warps.clear();
    }
    let expected = s.expected.expect("set above");
    if expected != c.count {
        return Err(Error::ContractMismatch {
            expected,
            observed: c.count,
        });
    }
    if s.warps.contains_key(&(c.warp, flavor)) {
        return Err(Error::Duplicate { warp: c.warp });
    }
    let red_now = flavor == Flavor::Red;
    if s.warps.keys().any(|&(_, f)| (f == Flavor::Red) != red_now) {
        return Err(Error::RedMixed { warp: c.warp });
    }
    let arrived = s.arrived + WARP_SIZE;
    if arrived > expected {
        return Err(Error::ArrivalOverflow { expected, arrived });
    }
    s.arrived = arrived;
    s.warps.insert((c.warp, flavor), ());
    s.complete = arrived == expected;
    Ok((s.gen, s.complete))
}

/// Exit check. An incomplete generation is never an error, only a lint.
pub fn exit_lint(s: &State) -> Option<Lint> {
    match s.expected {
        Some(expected) if !s.complete => Some(Lint::DanglingAtExit {
            gen: s.gen,
            arrived: s.arrived,
            expected,
        }),
        _ => None,
    }
}

/// Launch-exit check. Never an error: a dangling generation is reported by [`exit_lint`] instead.
pub fn quiescent(_s: &State) -> Result<(), Error> {
    Ok(())
}

pub fn check_invariants(s: &State) -> Result<(), String> {
    let Some(expected) = s.expected else {
        return (s.gen == 0 && !s.complete && s.arrived == 0 && s.warps.is_empty())
            .then_some(())
            .ok_or_else(|| "state before first use".into());
    };
    if s.arrived > expected || s.complete != (s.arrived == expected) {
        return Err("complete iff arrived == expected".into());
    }
    if s.arrived != WARP_SIZE * s.warps.len() as u64 {
        return Err("warp bookkeeping disagrees with count".into());
    }
    let reds = s.warps.keys().filter(|&&(_, f)| f == Flavor::Red).count();
    if reds != 0 && reds != s.warps.len() {
        return Err(".red mixed with sync/arrive".into());
    }
    Ok(())
}
