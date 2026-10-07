#![allow(unused_variables, dead_code)]
//! Reference model of the per-cluster hardware barrier:
//! `barrier.cluster.arrive[.release|.relaxed][.aligned]` and
//! `barrier.cluster.wait[.acquire][.aligned]`.
//!
//! Spec: `docs/development/sync-semantics.md` §4, with the ISA answers in
//! `sync-isa-answers.md` Q4/Q5.
//!
//! Membership is exit-aware. The barrier completes when every **non-exited**
//! thread of the cluster has arrived (PTX 9.4 §9.7.15.3), and barriers waiting
//! only on exited threads are released (§9.7.14.7). The model's unit is the
//! warp. Each instruction must be executed by exactly the warp's non-exited
//! lanes. A strict subset is an error and fails closed. A warp whose lanes
//! have all exited leaves the membership. If that leaves every remaining
//! member arrived, the generation completes.
//!
//! Release, relaxed and acquire change no barrier state. They select HB edges
//! in the checkers only, so the commands do not carry them.
//!
//! CONTRACT: the types in this file are copied verbatim from
//! `numsim-sync-ref/src/cluster.rs` (plus serde derives) so production `step`
//! is differentially tested against the reference mechanically. Change
//! them only together with the reference crate, via the coordinator.
//! Function bodies marked `W3` are the production implementation to write.

use std::collections::BTreeSet;

use super::{LaneMask, Warp, FULL_MASK};

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct State {
    /// Non-exited lanes per participant warp; the warps are `0..live.len()`.
    pub live: Vec<LaneMask>,
    /// Current (incomplete) generation. Completion increments it eagerly.
    pub gen: u64,
    pub arrived: BTreeSet<Warp>,
    pub last_arrival: std::collections::BTreeMap<Warp, u64>,
    pub last_waited: std::collections::BTreeMap<Warp, u64>,
}

impl State {
    pub fn new(participants: u32) -> Self {
        Self {
            live: vec![FULL_MASK; participants as usize],
            ..Self::default()
        }
    }

    fn member(&self, warp: Warp) -> bool {
        self.live.get(warp as usize).is_some_and(|&m| m != 0)
    }

    /// Every remaining member has arrived, and at least one warp arrived
    /// (an all-exit with no arrival completes nothing observable).
    fn all_arrived(&self) -> bool {
        !self.arrived.is_empty()
            && (0..self.live.len() as Warp)
                .filter(|&w| self.member(w))
                .all(|w| self.arrived.contains(&w))
    }

    fn complete_generation(&mut self) {
        self.gen += 1;
        self.arrived.clear();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Cmd {
    Arrive {
        warp: Warp,
        mask: LaneMask,
        aligned: bool,
    },
    /// Retried by the scheduler until `Ready`.
    Wait {
        warp: Warp,
        mask: LaneMask,
        aligned: bool,
    },
    /// Lanes of `warp` exit the kernel.
    Exit { warp: Warp, lanes: LaneMask },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Outcome {
    /// `rearrival_without_wait`: the warp's previous arrival completed but the
    /// warp never waited on it. The ISA is silent on this case
    /// (sync-isa-answers Q4), so checkers report it as unmodeled.
    Arrived {
        gen: u64,
        completed: bool,
        rearrival_without_wait: bool,
    },
    Ready {
        gen: u64,
    },
    Blocked,
    /// The exit completed the current generation.
    Exited {
        completed: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Error {
    UnexpectedParticipant {
        warp: Warp,
    },
    /// Executed by no lane, or by a strict subset of the warp's non-exited
    /// lanes.
    PartialWarp {
        mask: LaneMask,
        live: LaneMask,
    },
    /// "Each thread must arrive at the barrier only once before the barrier
    /// completes."
    EarlyArrival {
        warp: Warp,
    },
    /// Waiting before one's own arrive waits on oneself: a guaranteed hang.
    WaitBeforeArrival {
        warp: Warp,
    },
    DuplicateWait {
        warp: Warp,
        gen: u64,
    },
}

pub struct Cluster;

impl super::Protocol for Cluster {
    type State = State;
    type Cmd = Cmd;
    type Outcome = Outcome;
    type Error = Error;
    fn step(state: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
        step(state, cmd)
    }
}

pub fn step(state: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
    unimplemented!("W3: cluster::step")
}



pub fn check_invariants(s: &State) -> Result<(), String> {
    unimplemented!("W3: cluster::check_invariants")
}

/// End-of-launch check (contract addition; not yet in the reference).
pub fn quiescent(s: &State) -> Result<(), Error> {
    unimplemented!("W3: cluster::quiescent")
}
