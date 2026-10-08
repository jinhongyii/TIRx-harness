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
    match cmd {
        Cmd::Arrive { warp, mask, .. } => {
            participation(state, warp, mask)?;
            if state.arrived.contains(&warp) {
                return Err(Error::EarlyArrival { warp });
            }
            let gen = state.gen;
            let previous = state.last_arrival.insert(warp, gen);
            let rearrival_without_wait = previous.is_some() && state.last_waited.get(&warp) != previous.as_ref();
            state.arrived.insert(warp);
            let completed = roll_if_complete(state);
            Ok(Outcome::Arrived { gen, completed, rearrival_without_wait })
        }
        Cmd::Wait { warp, mask, .. } => {
            participation(state, warp, mask)?;
            let gen = *state.last_arrival.get(&warp).ok_or(Error::WaitBeforeArrival { warp })?;
            if matches!(state.last_waited.get(&warp), Some(&w) if w >= gen) {
                return Err(Error::DuplicateWait { warp, gen });
            }
            if gen >= state.gen {
                return Ok(Outcome::Blocked);
            }
            state.last_waited.insert(warp, gen);
            Ok(Outcome::Ready { gen })
        }
        Cmd::Exit { warp, lanes } => {
            if live_lanes(state, warp) == 0 {
                return Err(Error::UnexpectedParticipant { warp });
            }
            state.live[warp as usize] &= !lanes;
            Ok(Outcome::Exited { completed: roll_if_complete(state) })
        }
    }
}



/// Launch-exit check. Never an error: exit-aware membership completes every generation by kernel exit.
pub fn quiescent(_s: &State) -> Result<(), Error> {
    Ok(())
}

// ---------------------------------------------------------------- partial warps

/// Lanes of one warp reaching `barrier.cluster.{arrive,wait}` in pieces.
/// Same ruling as named barriers (sync-semantics §4.6, sync-isa-answers
/// Q4/Q5): "barrier.cluster instructions cause the executing thread to wait
/// for all non-exited threads from its warp", and only `.aligned` requires
/// every thread of the warp to execute the *same* instruction. A
/// **non-aligned** arrive or wait executed by a strict subset of the warp's
/// non-exited lanes waits for the rest of the warp to execute the same kind
/// (arrive or wait, any site); the warp then makes its single arrive or wait
/// with the full live mask. `PartialWarp` fires when the missing lanes exit
/// or execute the other kind, and for `.aligned` partial forms. A lane that
/// executes the same kind twice before its warp completes is `EarlyArrival`
/// ("each thread must arrive at the barrier only once").
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Gather {
    pub warp: Warp,
    pub pending: Option<Pending>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Pending {
    /// `barrier.cluster.wait` (else `arrive`).
    pub wait: bool,
    pub lanes: LaneMask,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum GatherCmd {
    Execute { wait: bool, mask: LaneMask, live: LaneMask, aligned: bool },
    /// Lanes of the warp exited while others may be waiting.
    Exit { live: LaneMask },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum GatherOutcome {
    /// The lanes wait for the rest of their warp.
    Wait,
    /// Every non-exited lane is here: issue the warp's one command with `mask`.
    Complete { mask: LaneMask },
    /// Nothing pending.
    Idle,
}

pub fn gather(g: &mut Gather, cmd: GatherCmd) -> Result<GatherOutcome, Error> {
    match cmd {
        GatherCmd::Exit { live } => match g.pending {
            Some(p) => Err(Error::PartialWarp { mask: p.lanes, live }),
            None => Ok(GatherOutcome::Idle),
        },
        GatherCmd::Execute { wait, mask, live, aligned } => {
            let so_far = g.pending.map_or(0, |p| p.lanes);
            if mask == 0 || mask & !live != 0 {
                return Err(Error::PartialWarp { mask: mask | so_far, live });
            }
            if aligned {
                if g.pending.is_some() || mask != live {
                    return Err(Error::PartialWarp { mask: mask | so_far, live });
                }
                return Ok(GatherOutcome::Complete { mask: live });
            }
            if let Some(p) = g.pending {
                if p.wait != wait {
                    return Err(Error::PartialWarp { mask: p.lanes, live });
                }
                if p.lanes & mask != 0 {
                    return Err(Error::EarlyArrival { warp: g.warp });
                }
            }
            let lanes = so_far | mask;
            if lanes & !live != 0 {
                return Err(Error::PartialWarp { mask: lanes, live });
            }
            if lanes == live {
                g.pending = None;
                Ok(GatherOutcome::Complete { mask: live })
            } else {
                g.pending = Some(Pending { wait, lanes });
                Ok(GatherOutcome::Wait)
            }
        }
    }
}

pub fn check_invariants(s: &State) -> Result<(), String> {
    if complete(s) {
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

fn live_lanes(s: &State, warp: Warp) -> LaneMask {
    s.live.get(warp as usize).copied().unwrap_or(0)
}

/// PTX §9.7.15.3: the instruction is executed by exactly the warp's
/// non-exited lanes.
fn participation(s: &State, warp: Warp, mask: LaneMask) -> Result<(), Error> {
    let live = live_lanes(s, warp);
    if live == 0 {
        return Err(Error::UnexpectedParticipant { warp });
    }
    if mask != live {
        return Err(Error::PartialWarp { mask, live });
    }
    Ok(())
}

/// All non-exited member warps arrived (and at least one warp arrived).
fn complete(s: &State) -> bool {
    let members = s.live.iter().filter(|&&m| m != 0).count();
    let arrived_members = s.arrived.iter().filter(|&&w| live_lanes(s, w) != 0).count();
    !s.arrived.is_empty() && arrived_members == members
}

fn roll_if_complete(s: &mut State) -> bool {
    let done = complete(s);
    if done {
        s.gen += 1;
        s.arrived.clear();
    }
    done
}

use crate::report::FindingKind;

/// Map a protocol error to its report kind.
pub fn finding_kind(e: &Error) -> FindingKind {
    match e {
        Error::UnexpectedParticipant { .. } => FindingKind::RuntimeError,
        _ => FindingKind::BarrierMismatch,
    }
}
