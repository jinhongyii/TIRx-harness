//! Reference model of the per-cluster hardware barrier
//! (`barrier.cluster.arrive[.release|.relaxed][.aligned]`,
//! `barrier.cluster.wait[.acquire][.aligned]`).
//!
//! Spec: `docs/development/sync-semantics.md` §4. The unit is the warp. A
//! warp counts once all 32 lanes have arrived in the current generation
//! (`cluster_barriers.rs:192-214`). The generation completes when every
//! participant warp has arrived. A wait targets the warp's latest full arrival
//! and is satisfied once that generation has completed.
//!
//! Release/relaxed/acquire change no barrier state. They only select HB edges
//! in the checkers, so the commands do not carry them.

use std::collections::BTreeMap;

use crate::{LaneMask, Policy, Warp, FULL_MASK};

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct State {
    pub policy: Policy,
    /// Participant warps are `0..participants` (every warp of every CTA of
    /// the cluster in the launch).
    pub participants: u32,
    /// Current (incomplete) generation. Completion increments it eagerly
    /// (`cluster_barriers.rs:216-221`).
    pub gen: u64,
    pub lanes: BTreeMap<Warp, LaneMask>,
    pub last_arrival: BTreeMap<Warp, u64>,
    pub last_waited: BTreeMap<Warp, u64>,
}

impl State {
    pub fn new(policy: Policy, participants: u32) -> Self {
        Self {
            policy,
            participants,
            ..Self::default()
        }
    }

    fn arrived_warps(&self) -> u32 {
        self.lanes.values().filter(|&&m| m == FULL_MASK).count() as u32
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// `rearrival_without_wait`: the warp's previous full arrival completed
    /// but the warp never waited on it. This is not an error. Strict synccheck
    /// reports it as unmodeled (`strict_cluster_barrier.rs:366-372`).
    Arrived {
        gen: u64,
        completed: bool,
        rearrival_without_wait: bool,
    },
    Ready {
        gen: u64,
    },
    Blocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Error {
    UnexpectedParticipant {
        warp: Warp,
    },
    /// `.aligned` with a partial warp. Strict also rejects partial unaligned
    /// arrivals. Both reject partial waits, which the warp-unit model cannot
    /// represent.
    PartialWarp {
        mask: LaneMask,
    },
    EarlyArrival {
        warp: Warp,
        overlap: LaneMask,
    },
    WaitBeforeArrival {
        warp: Warp,
    },
    DuplicateWait {
        warp: Warp,
        gen: u64,
    },
    IncompleteAtExit {
        gen: u64,
        arrived: u32,
        expected: u32,
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

fn apply(s: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
    match cmd {
        Cmd::Arrive {
            warp,
            mask,
            aligned,
        } => {
            if warp >= s.participants {
                return Err(Error::UnexpectedParticipant { warp });
            }
            if mask != FULL_MASK && (aligned || s.policy == Policy::Strict) {
                return Err(Error::PartialWarp { mask });
            }
            let prior = s.lanes.get(&warp).copied().unwrap_or(0);
            if prior & mask != 0 {
                return Err(Error::EarlyArrival {
                    warp,
                    overlap: prior & mask,
                });
            }
            if mask == 0 {
                return Ok(Outcome::Arrived {
                    gen: s.gen,
                    completed: false,
                    rearrival_without_wait: false,
                });
            }
            let gen = s.gen;
            let lanes = prior | mask;
            s.lanes.insert(warp, lanes);
            let mut rearrival_without_wait = false;
            if lanes == FULL_MASK {
                if let Some(&previous) = s.last_arrival.get(&warp) {
                    rearrival_without_wait = s.last_waited.get(&warp) != Some(&previous);
                }
                s.last_arrival.insert(warp, gen);
            }
            let completed = s.arrived_warps() == s.participants;
            if completed {
                s.gen += 1;
                s.lanes.clear();
            }
            Ok(Outcome::Arrived {
                gen,
                completed,
                rearrival_without_wait,
            })
        }
        Cmd::Wait {
            warp,
            mask,
            aligned: _,
        } => {
            if warp >= s.participants {
                return Err(Error::UnexpectedParticipant { warp });
            }
            if mask != FULL_MASK {
                return Err(Error::PartialWarp { mask });
            }
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
    }
}

/// Launch-exit check. Exited threads are not modeled as arrived: hardware
/// counts them, and the legacy engine reports a non-quiescent source.
pub fn quiescent(s: &State) -> Result<(), Error> {
    if s.lanes.values().any(|&m| m != 0) {
        return Err(Error::IncompleteAtExit {
            gen: s.gen,
            arrived: s.arrived_warps(),
            expected: s.participants,
        });
    }
    Ok(())
}

pub fn check_invariants(s: &State) -> Result<(), String> {
    if s.participants > 0 && s.arrived_warps() >= s.participants {
        return Err("a complete generation was not rolled".into());
    }
    if s.lanes.keys().any(|&w| w >= s.participants) {
        return Err("non-participant lanes".into());
    }
    for (w, &g) in &s.last_arrival {
        if g > s.gen {
            return Err("arrival in a future generation".into());
        }
        if g == s.gen && s.lanes.get(w) != Some(&FULL_MASK) {
            return Err("current arrival without full lanes".into());
        }
    }
    for (w, &g) in &s.last_waited {
        if g >= s.gen || s.last_arrival.get(w).is_none_or(|&a| a < g) {
            return Err("wait on an incomplete or unarrived generation".into());
        }
    }
    Ok(())
}
