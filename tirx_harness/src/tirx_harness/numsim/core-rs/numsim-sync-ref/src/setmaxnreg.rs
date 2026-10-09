//! Reference model of the per-CTA `setmaxnreg` register pool.
//!
//! Spec: `docs/development/sync-semantics.md` §7. Each command is already
//! warpgroup-collective: all four warps contributed the same action and
//! count. The collective layer rejects `Divergence` and full-warp violations.
//!
//! Semantics follow the checker-mode `SetmaxnregHub` (`setmaxnreg.rs`). The
//! pool starts empty: only registers released by `dec` can satisfy `inc`.
//! An `inc` that does not fit stays pending until the scheduler fires
//! `Grant`. The warpgroup's warps retry `Poll` until it is `Ready`. NumSim's
//! legacy occurrence-only path (`ordering.rs:923-1033`) has no pool and no
//! direction check. The redesign adopts this model in every mode (spec §7.6).

pub const CTA_REGISTER_POOL: u32 = 512;
pub const MIN_COUNT: u32 = 24;
pub const MAX_COUNT: u32 = 256;
pub const GRANULARITY: u32 = 8;
pub const WARPS_PER_GROUP: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Pending {
    pub target: u32,
    pub required: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct State {
    pub warps_per_cta: u32,
    pub current: Vec<u32>,
    pub pending: Vec<Option<Pending>>,
    /// A setmaxnreg completed and no warpgroup-wide `.aligned bar.sync` has
    /// happened since (`setmaxnreg.rs:1066-1074`).
    pub needs_sync: Vec<bool>,
    pub available: u32,
    /// Some setmaxnreg has executed, so `Configure` is no longer allowed.
    pub started: bool,
    pub configured: Option<u32>,
}

impl State {
    pub fn new(warps_per_cta: u32) -> Self {
        let groups = warps_per_cta.div_ceil(WARPS_PER_GROUP).max(1);
        let default = default_count(warps_per_cta);
        Self {
            warps_per_cta,
            current: vec![default; groups as usize],
            pending: vec![None; groups as usize],
            needs_sync: vec![false; groups as usize],
            ..Self::default()
        }
    }

    fn total(&self) -> u32 {
        self.available + self.current.iter().sum::<u32>()
    }
}

/// `setmaxnreg.rs:973-977`.
pub fn default_count(warps_per_cta: u32) -> u32 {
    let groups = warps_per_cta.div_ceil(WARPS_PER_GROUP).max(1);
    CTA_REGISTER_POOL / groups / GRANULARITY * GRANULARITY
}

fn valid_count(count: u32) -> bool {
    (MIN_COUNT..=MAX_COUNT).contains(&count) && count.is_multiple_of(GRANULARITY)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cmd {
    /// Launch-configured initial per-thread count (`setmaxnreg.rs:580-630`).
    Configure { count: u32 },
    /// `setmaxnreg.inc/dec.sync.aligned.u32 count` by warpgroup `wg`.
    Set { wg: u32, inc: bool, count: u32 },
    /// A completed `.aligned bar.sync` with all four warps of `wg` on the
    /// same generation and full masks (`setmaxnreg.rs:740-791`).
    WarpgroupSync { wg: u32 },
    /// Scheduler grant of a pending increase.
    Grant { wg: u32 },
    /// Retry by a warp of `wg` parked on its increase.
    Poll { wg: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    Done,
    Applied { count: u32 },
    Pending { required: u32 },
    Ready,
    Blocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Error {
    InvalidCount { count: u32 },
    ConfigureConflict,
    IncompleteWarpgroup { wg: u32 },
    MissingWarpgroupSync { wg: u32 },
    InvalidDirection { inc: bool, current: u32, count: u32 },
    WarpgroupPending { wg: u32 },
    GrantNotEnabled { wg: u32 },
    PendingAtExit { wg: u32 },
}

pub struct Setmaxnreg;

impl crate::Protocol for Setmaxnreg {
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

fn check_wg(s: &State, wg: u32) -> Result<usize, Error> {
    if (wg + 1) * WARPS_PER_GROUP > s.warps_per_cta {
        return Err(Error::IncompleteWarpgroup { wg });
    }
    Ok(wg as usize)
}

fn apply(s: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
    match cmd {
        Cmd::Configure { count } => {
            if !valid_count(count) || count > default_count(s.warps_per_cta) {
                return Err(Error::InvalidCount { count });
            }
            if s.started || s.configured.is_some_and(|c| c != count) {
                return Err(Error::ConfigureConflict);
            }
            s.configured = Some(count);
            s.current.iter_mut().for_each(|c| *c = count);
            s.available = 0;
            Ok(Outcome::Done)
        }
        Cmd::Set { wg, inc, count } => {
            if !valid_count(count) {
                return Err(Error::InvalidCount { count });
            }
            let i = check_wg(s, wg)?;
            if s.pending[i].is_some() {
                return Err(Error::WarpgroupPending { wg });
            }
            if s.needs_sync[i] {
                return Err(Error::MissingWarpgroupSync { wg });
            }
            let current = s.current[i];
            if (inc && count < current) || (!inc && count > current) {
                return Err(Error::InvalidDirection {
                    inc,
                    current,
                    count,
                });
            }
            s.started = true;
            s.needs_sync[i] = true;
            if !inc {
                s.available += current - count;
                s.current[i] = count;
                return Ok(Outcome::Applied { count });
            }
            let required = count - current;
            if required <= s.available {
                s.available -= required;
                s.current[i] = count;
                Ok(Outcome::Applied { count })
            } else {
                s.pending[i] = Some(Pending {
                    target: count,
                    required,
                });
                Ok(Outcome::Pending { required })
            }
        }
        Cmd::WarpgroupSync { wg } => {
            // A warpgroup is four contiguous warps whose first warp rank is a
            // multiple of 4 (PTX ISA "warpgroup"). The trailing warps of a CTA
            // whose warp count is not a multiple of 4 (e.g. warps 4-5 of a
            // 6-warp CTA) form no warpgroup, so `setmaxnreg` there is still
            // `IncompleteWarpgroup` (UB: "If a setmaxnreg instruction is not
            // executed by all warps in the warpgroup, then the behavior is
            // undefined", PTX 9.7.21.5). But an aligned `bar.sync` among
            // them is legal and owes no setmaxnreg sync: crediting it is a
            // no-op, not an error (V2C-14).
            match check_wg(s, wg) {
                Ok(i) => s.needs_sync[i] = false,
                Err(Error::IncompleteWarpgroup { .. }) => {}
                Err(e) => return Err(e),
            }
            Ok(Outcome::Done)
        }
        Cmd::Grant { wg } => {
            let i = check_wg(s, wg)?;
            match s.pending[i] {
                Some(p) if p.required <= s.available => {
                    s.available -= p.required;
                    s.current[i] = p.target;
                    s.pending[i] = None;
                    Ok(Outcome::Applied { count: p.target })
                }
                _ => Err(Error::GrantNotEnabled { wg }),
            }
        }
        Cmd::Poll { wg } => {
            let i = check_wg(s, wg)?;
            Ok(if s.pending[i].is_some() {
                Outcome::Blocked
            } else {
                Outcome::Ready
            })
        }
    }
}

/// Grants the scheduler may fire now. The legacy hub fires only the one with
/// the lowest hashed action id per pump; the verifier explores all of them.
pub fn enabled_grants(s: &State) -> Vec<u32> {
    s.pending
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.filter(|p| p.required <= s.available).map(|_| i as u32))
        .collect()
}

pub fn quiescent(s: &State) -> Result<(), Error> {
    match s.pending.iter().position(Option::is_some) {
        Some(i) => Err(Error::PendingAtExit { wg: i as u32 }),
        None => Ok(()),
    }
}

/// The initial pool total, which every transition conserves.
pub fn initial_total(s: &State) -> u32 {
    let per = s.configured.unwrap_or(default_count(s.warps_per_cta));
    per * s.current.len() as u32
}

pub fn check_invariants(s: &State) -> Result<(), String> {
    if s.total() != initial_total(s) {
        return Err("register pool not conserved".into());
    }
    if s.total() > CTA_REGISTER_POOL {
        return Err("register oversubscription".into());
    }
    for (c, p) in s.current.iter().zip(&s.pending) {
        if p.is_some_and(|p| p.target != c + p.required) {
            return Err("pending target inconsistent".into());
        }
    }
    Ok(())
}
