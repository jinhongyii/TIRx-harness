#![allow(unused_variables, dead_code)]
//! Reference model of one CTA-local named barrier: `bar.sync`, `bar.arrive`,
//! `bar.red`, and `barrier.{sync,arrive,red}[.aligned]`.
//!
//! Spec: `docs/development/sync-semantics.md` §3, with the ISA answers in
//! `sync-isa-answers.md` Q3/Q5.
//!
//! A barrier instruction makes each executing thread wait for all non-exited
//! threads of its warp, then marks the **warp's** arrival (PTX 9.4 §9.7.15.1).
//! The model therefore requires each contribution's lane mask to equal the
//! warp's non-exited lanes. A non-aligned instruction reached by a partial
//! warp first goes through the per-warp [`Gather`] (the lanes wait for the
//! rest of the warp, which then arrives once). An `.aligned` partial warp,
//! or missing lanes that exit or reach a different barrier id, are
//! `PartialWarp`. A warp arrival counts 32 threads toward `b`, which must be
//! a multiple of the warp size.
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
//!
//! CONTRACT: the types in this file are copied verbatim from
//! `numsim-sync-ref/src/named.rs` (plus serde derives) so production `step`
//! is differentially tested against the reference mechanically. Change
//! them only together with the reference crate, via the coordinator.
//! Function bodies marked `W3` are the production implementation to write.

use std::collections::BTreeMap;

use super::{LaneMask, Warp};

/// Hardware barrier ids are `0..16`; the caller resolves the id to a `State`.
pub const NUM_IDS: u32 = 16;
pub const WARP_SIZE: u64 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum Flavor {
    Arrive,
    Sync,
    Red,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Cmd {
    Arrive(Contribution),
    Sync(Contribution),
    Red(Contribution),
    /// Retry of a blocked `Sync`/`Red` that registered on `gen`.
    Resume {
        gen: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Outcome {
    Arrived { gen: u64, completed: bool },
    Registered { gen: u64 },
    Ready { gen: u64 },
    Blocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Error {
    /// `b == 0` or `b` not a multiple of 32 (PTX §9.7.15.1).
    InvalidCount {
        count: u64,
    },
    /// Executed by no lane, or by a strict subset of the warp's non-exited
    /// lanes reaching this barrier as one contribution (PTX §9.7.15.1,
    /// sync-isa-answers Q3/Q5). Non-aligned pieces are gathered first
    /// ([`gather`]); there it means the missing lanes exited or reached a
    /// different barrier id or flavor, or an `.aligned` form ran partial.
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

impl super::Protocol for Named {
    type State = State;
    type Cmd = Cmd;
    type Outcome = Outcome;
    type Error = Error;
    fn step(state: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
        step(state, cmd)
    }
}

pub fn step(state: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
    let (c, flavor) = match cmd {
        Cmd::Resume { gen } => {
            return match state.expected {
                Some(_) if gen < state.gen || (gen == state.gen && state.complete) => Ok(Outcome::Ready { gen }),
                Some(_) if gen == state.gen => Ok(Outcome::Blocked),
                _ => Err(Error::ResumeFuture { gen }),
            };
        }
        Cmd::Arrive(c) => (c, Flavor::Arrive),
        Cmd::Sync(c) => (c, Flavor::Sync),
        Cmd::Red(c) => (c, Flavor::Red),
    };
    if c.count == 0 || c.count % WARP_SIZE != 0 {
        return Err(Error::InvalidCount { count: c.count });
    }
    // PTX §9.7.15.1: every non-exited lane of the warp executes the barrier.
    if c.mask == 0 || c.mask != c.live {
        return Err(Error::PartialWarp { mask: c.mask, live: c.live });
    }
    // Validate against the generation this contribution joins: a fresh one
    // (first use or after completion) or the current one.
    let fresh = state.complete || state.expected.is_none();
    let expected = if fresh { c.count } else { state.expected.unwrap_or(c.count) };
    if expected != c.count {
        return Err(Error::ContractMismatch { expected, observed: c.count });
    }
    let (arrived_before, warps_empty) = if fresh { (0, true) } else { (state.arrived, state.warps.is_empty()) };
    if !warps_empty {
        if state.warps.contains_key(&(c.warp, flavor)) {
            return Err(Error::Duplicate { warp: c.warp });
        }
        let red = flavor == Flavor::Red;
        if state.warps.keys().any(|&(_, f)| (f == Flavor::Red) != red) {
            return Err(Error::RedMixed { warp: c.warp });
        }
    }
    let arrived = arrived_before + WARP_SIZE;
    if arrived > expected {
        return Err(Error::ArrivalOverflow { expected, arrived });
    }
    if fresh {
        if state.complete {
            state.gen += 1;
        }
        state.expected = Some(expected);
        state.warps.clear();
    }
    state.arrived = arrived;
    state.warps.insert((c.warp, flavor), ());
    state.complete = arrived == expected;
    let (gen, completed) = (state.gen, state.complete);
    Ok(match (flavor, completed) {
        (Flavor::Arrive, _) => Outcome::Arrived { gen, completed },
        (_, true) => Outcome::Ready { gen },
        (_, false) => Outcome::Registered { gen },
    })
}



// ---------------------------------------------------------------- partial warps

/// Lanes of one warp reaching a barrier instruction in pieces (coordinator
/// ruling on sync-isa-answers Q3/Q5). A **non-aligned** `barrier.sync` /
/// `bar.sync` / `barrier.arrive` / `barrier.red` executed by a strict subset
/// of the warp's non-exited lanes is not an error. Those lanes wait for the
/// warp's remaining non-exited lanes to reach the same barrier id, at any
/// instruction site, with the same flavor and `b`. When they all have, the
/// warp makes its single arrival ([`GatherOutcome::Arrive`] with the full
/// `live` mask; that is the one [`Contribution`] the barrier sees).
/// `PartialWarp` fires when the missing lanes exit or reach a different
/// barrier id or flavor. `.aligned` forms keep the immediate full-warp
/// requirement. This is per-warp state in front of the per-id [`State`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Gather {
    pub warp: Warp,
    pub pending: Option<Pending>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Pending {
    pub id: u32,
    pub flavor: Flavor,
    pub count: u64,
    /// Lanes that have executed the instruction so far.
    pub lanes: LaneMask,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum GatherCmd {
    /// The active lanes `mask` execute a barrier instruction on `id`.
    Execute {
        id: u32,
        flavor: Flavor,
        count: u64,
        mask: LaneMask,
        live: LaneMask,
        aligned: bool,
    },
    /// Lanes of the warp exited while others may be waiting.
    Exit {
        live: LaneMask,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum GatherOutcome {
    /// The lanes wait for the rest of their warp.
    Wait,
    /// Every non-exited lane is here: the warp arrives once with `mask`.
    Arrive { mask: LaneMask },
    /// Nothing pending (an exit with no gathering in progress).
    Idle,
}

pub fn gather(g: &mut Gather, cmd: GatherCmd) -> Result<GatherOutcome, Error> {
    match cmd {
        GatherCmd::Exit { live } => match g.pending {
            Some(p) => Err(Error::PartialWarp { mask: p.lanes, live }),
            None => Ok(GatherOutcome::Idle),
        },
        GatherCmd::Execute { id, flavor, count, mask, live, aligned } => {
            let so_far = g.pending.map_or(0, |p| p.lanes);
            if mask == 0 || mask & !live != 0 {
                return Err(Error::PartialWarp { mask: mask | so_far, live });
            }
            if aligned {
                if g.pending.is_some() || mask != live {
                    return Err(Error::PartialWarp { mask: mask | so_far, live });
                }
                return Ok(GatherOutcome::Arrive { mask: live });
            }
            if let Some(p) = g.pending {
                if p.id != id || p.flavor != flavor {
                    return Err(Error::PartialWarp { mask: p.lanes, live });
                }
                if p.count != count {
                    return Err(Error::ContractMismatch { expected: p.count, observed: count });
                }
                if p.lanes & mask != 0 {
                    return Err(Error::Duplicate { warp: g.warp });
                }
            }
            let lanes = so_far | mask;
            if lanes & !live != 0 {
                return Err(Error::PartialWarp { mask: lanes, live });
            }
            if lanes == live {
                g.pending = None;
                Ok(GatherOutcome::Arrive { mask: live })
            } else {
                g.pending = Some(Pending { id, flavor, count, lanes });
                Ok(GatherOutcome::Wait)
            }
        }
    }
}

/// Exit check. An incomplete generation is never an error, only a lint.
pub fn exit_lint(s: &State) -> Option<Lint> {
    let expected = s.expected?;
    (!s.complete).then_some(Lint::DanglingAtExit { gen: s.gen, arrived: s.arrived, expected })
}

/// Launch-exit check. Never an error: a dangling generation is reported by [`exit_lint`] instead.
pub fn quiescent(_s: &State) -> Result<(), Error> {
    Ok(())
}

pub fn check_invariants(s: &State) -> Result<(), String> {
    let Some(expected) = s.expected else {
        let pristine = s.gen == 0 && !s.complete && s.arrived == 0 && s.warps.is_empty();
        return if pristine { Ok(()) } else { Err("state before first use".into()) };
    };
    if s.arrived > expected || s.complete != (s.arrived == expected) {
        return Err("complete iff arrived == expected".into());
    }
    if s.arrived != WARP_SIZE * s.warps.len() as u64 {
        return Err("warp bookkeeping disagrees with count".into());
    }
    let reds = s.warps.keys().filter(|(_, f)| *f == Flavor::Red).count();
    if reds != 0 && reds != s.warps.len() {
        return Err(".red mixed with sync/arrive".into());
    }
    Ok(())
}

use crate::report::FindingKind;

/// Map a protocol error to its report kind.
pub fn finding_kind(e: &Error) -> FindingKind {
    match e {
        Error::ResumeFuture { .. } => FindingKind::RuntimeError,
        _ => FindingKind::BarrierMismatch,
    }
}

/// Report kind of an exit lint (reported with `Status::Review`).
pub fn lint_kind(_l: &Lint) -> FindingKind {
    FindingKind::BarrierMismatch
}
