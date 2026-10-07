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
//! warp's non-exited lanes. A strict subset is an error. That covers the
//! elect-gated single-lane case and lanes that reach different barrier
//! instructions; the latter fail closed. A warp arrival counts 32 threads
//! toward `b`, which must be a multiple of the warp size.
//!
//! `.aligned` is a convergence promise, not barrier state. Mixing aligned and
//! unaligned forms on one barrier is legal. Mixing `.red` with `sync`/`arrive`
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
    unimplemented!("W3: named::step")
}



/// Exit check. An incomplete generation is never an error, only a lint.
pub fn exit_lint(s: &State) -> Option<Lint> {
    unimplemented!("W3: named::exit_lint")
}

pub fn check_invariants(s: &State) -> Result<(), String> {
    unimplemented!("W3: named::check_invariants")
}

/// End-of-launch check (contract addition; not yet in the reference).
pub fn quiescent(s: &State) -> Result<(), Error> {
    unimplemented!("W3: named::quiescent")
}
