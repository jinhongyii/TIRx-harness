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

use std::collections::BTreeSet;

use crate::{LaneMask, Warp, FULL_MASK};

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

impl crate::Protocol for Cluster {
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

fn check_lanes(s: &State, warp: Warp, mask: LaneMask) -> Result<(), Error> {
    if !s.member(warp) {
        return Err(Error::UnexpectedParticipant { warp });
    }
    let live = s.live[warp as usize];
    if mask != live {
        return Err(Error::PartialWarp { mask, live });
    }
    Ok(())
}

fn apply(s: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
    match cmd {
        Cmd::Arrive { warp, mask, .. } => {
            check_lanes(s, warp, mask)?;
            if s.arrived.contains(&warp) {
                return Err(Error::EarlyArrival { warp });
            }
            let gen = s.gen;
            let rearrival_without_wait = s
                .last_arrival
                .get(&warp)
                .is_some_and(|previous| s.last_waited.get(&warp) != Some(previous));
            s.arrived.insert(warp);
            s.last_arrival.insert(warp, gen);
            let completed = s.all_arrived();
            if completed {
                s.complete_generation();
            }
            Ok(Outcome::Arrived {
                gen,
                completed,
                rearrival_without_wait,
            })
        }
        Cmd::Wait { warp, mask, .. } => {
            check_lanes(s, warp, mask)?;
            let Some(&gen) = s.last_arrival.get(&warp) else {
                return Err(Error::WaitBeforeArrival { warp });
            };
            if s.last_waited.get(&warp).is_some_and(|&w| w >= gen) {
                return Err(Error::DuplicateWait { warp, gen });
            }
            if gen < s.gen {
                s.last_waited.insert(warp, gen);
                Ok(Outcome::Ready { gen })
            } else {
                Ok(Outcome::Blocked)
            }
        }
        Cmd::Exit { warp, lanes } => {
            if !s.member(warp) {
                return Err(Error::UnexpectedParticipant { warp });
            }
            s.live[warp as usize] &= !lanes;
            let completed = s.all_arrived();
            if completed {
                s.complete_generation();
            }
            Ok(Outcome::Exited { completed })
        }
    }
}

pub fn check_invariants(s: &State) -> Result<(), String> {
    if s.all_arrived() {
        return Err("a complete generation was not rolled".into());
    }
    for (w, &g) in &s.last_arrival {
        if g > s.gen || (g == s.gen) != s.arrived.contains(w) {
            return Err("arrival bookkeeping out of step".into());
        }
    }
    for (w, &g) in &s.last_waited {
        if g >= s.gen || s.last_arrival.get(w).is_none_or(|&a| a < g) {
            return Err("wait on an incomplete or unarrived generation".into());
        }
    }
    Ok(())
}
