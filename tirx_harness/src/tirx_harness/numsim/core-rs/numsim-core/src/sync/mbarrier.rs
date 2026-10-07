#![allow(unused_variables, dead_code)]
//! Reference model of one physical shared-memory `mbarrier` slot.
//!
//! Spec: `docs/development/sync-semantics.md` §2, with the ISA answers in
//! `sync-isa-answers.md` (Q2, limits, S1, noComplete). Arrival counts are already
//! aggregated per instruction and target. Operand validation is done by the
//! caller: lane-uniformity, address space, and the rule that a lane-varying
//! instruction resolves to distinct targets.
//!
//! Phase identity: `gen` is the 0-based generation of the phase currently
//! being filled. Its hardware parity is `gen & 1`. After `init` the barrier
//! behaves as if a phase of parity 1 has just completed. A parity-1 wait
//! therefore succeeds vacuously, and a parity-0 wait blocks until generation 0
//! completes.
//!
//! Roll-over is lazy, as in the engine (`hardware_barriers.rs:2628`). When a
//! phase completes, `complete` is set. The next command that touches the
//! counters (arrive, expect_tx, pending increment, or a deferred arrival)
//! starts generation `gen + 1`.
//!
//! CONTRACT: the types in this file are copied verbatim from
//! `numsim-sync-ref/src/mbarrier.rs` (plus serde derives) so production `step`
//! is differentially tested against the reference mechanically. Change
//! them only together with the reference crate, via the coordinator.
//! Function bodies marked `W3` are the production implementation to write.

use std::collections::BTreeMap;

use super::Policy;

/// `mbarrier.init` count limit (PTX: 2^20 - 1). `hardware_barriers.rs:16`.
pub const MAX_COUNT: u64 = (1 << 20) - 1;
/// Limit under `mbarrier.init.layout::v1`. `hardware_barriers.rs:19-25`.
pub const MAX_COUNT_V1: u64 = (1 << 9) - 1;
/// The tx-count is signed and its state range is `-(2^20-1) ..= 2^20-1`
/// (PTX 9.4 §9.7.15.16.3, Table 43). The range is checked on the barrier
/// state, never on the 32-bit `txCount` operand.
pub const MAX_TX: u64 = (1 << 20) - 1;

pub const fn arrival_limit(layout_v1: bool) -> u64 {
    if layout_v1 {
        MAX_COUNT_V1
    } else {
        MAX_COUNT
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct State {
    pub policy: Policy,
    pub live: bool,
    pub layout_v1: bool,
    /// Steady-state expected arrivals (`mbarrier.init` count, minus drops).
    pub expected: u64,
    /// Generation of the current (pending, or completed but not rolled) phase.
    pub gen: u64,
    /// The current generation has completed; roll-over pending.
    pub complete: bool,
    /// Strict bookkeeping: a wait has observed the current completion.
    pub consumed: bool,
    /// Generation whose completion a parity query observes.
    pub last_completed: Option<u64>,
    pub arrived: u64,
    /// Phase-only extra arrivals: `cp.async.mbarrier.arrive` without `.noinc`,
    /// and the current-phase share of `arrive_drop`.
    pub extra: u64,
    pub tx_expected: u64,
    pub tx_completed: u64,
    /// Bytes delivered to generation `gen + 1` before it started.
    pub buffered_next: u64,
    /// Async completions issued (token captured) but not landed, per generation.
    pub outstanding: BTreeMap<u64, u32>,
    /// A blocking wait is parked on the next completion. It replaces the
    /// waiter registry: strict consumption happens at completion when set.
    pub armed: bool,
}

impl State {
    pub fn new(policy: Policy) -> Self {
        Self {
            policy,
            ..Self::default()
        }
    }

    pub const fn required(&self) -> u64 {
        self.expected.saturating_add(self.extra)
    }

    /// Parity of the most recently completed phase (1 right after init).
    pub fn completed_parity(&self) -> u64 {
        self.last_completed.map_or(1, |g| g & 1)
    }

    fn strict(&self) -> bool {
        self.policy == Policy::Strict
    }

    fn outstanding_total(&self) -> u64 {
        self.outstanding.values().map(|&n| u64::from(n)).sum()
    }

    /// The phase holds work that re-initialization would silently discard.
    fn active(&self) -> bool {
        self.armed
            || !self.outstanding.is_empty()
            || self.buffered_next != 0
            || (!self.complete
                && (self.arrived != 0
                    || self.extra != 0
                    || self.tx_expected != 0
                    || self.tx_completed != 0))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Op {
    Arrive,
    ExpectTx,
    IncPending,
    DeferredArrive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Cmd {
    /// `mbarrier.init[.layout::v1]` (lane-collapsed).
    Init { count: u64, layout_v1: bool },
    /// `mbarrier.inval`.
    Inval,
    /// `mbarrier.arrive[.expect_tx][_drop][.noComplete]`. `tx` is `Some` for
    /// the `.expect_tx` forms. A count of 0 is a fully-masked no-op.
    Arrive {
        count: u64,
        tx: Option<u64>,
        drop: bool,
        no_complete: bool,
    },
    /// Standalone `mbarrier.expect_tx`.
    ExpectTx { bytes: u64 },
    /// Immediate half of `cp.async.mbarrier.arrive` (without `.noinc`).
    IncPending { count: u64 },
    /// Issue of an async op that later completes on this barrier (TMA
    /// complete_tx, `cp.async.mbarrier.arrive`, `tcgen05.commit`). It captures
    /// the generation the completion is bound to.
    Issue,
    /// Landing of transaction bytes bound to `gen` (from `Issue`).
    CompleteTx { gen: u64, bytes: u64 },
    /// Landing of a deferred arrive-on bound to `gen` (from `Issue`).
    DeferredArrive { gen: u64, count: u64 },
    /// `mbarrier.test_wait/try_wait.parity` (non-blocking; try_wait chooses
    /// the zero-suspension execution, `runtime/instructions/sync.rs:1072-1112`).
    TestParity { parity: u64 },
    /// Blocking parity wait (`WaitUntilParity`).
    WaitParity { parity: u64 },
    /// `mbarrier.test_wait/try_wait` with an opaque state token naming `gen`.
    TestState { gen: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Outcome {
    Done,
    /// Arrival applied to `gen` (also the state token). `completed` is true
    /// iff this arrival flipped the phase.
    Arrived {
        gen: u64,
        pending_before: u64,
        completed: bool,
    },
    /// Counter-only update of `gen` (expect_tx, pending increment).
    Updated {
        gen: u64,
    },
    Issued {
        gen: u64,
    },
    /// A completion landed; `completed` is the generation it completed, if any.
    Landed {
        completed: Option<u64>,
    },
    /// Query or wait succeeded. `gen` is the observed completion; `None` means
    /// the vacuous parity-1 success right after init.
    Ready {
        gen: Option<u64>,
    },
    NotReady,
    Blocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Error {
    Uninitialized,
    InvalidCount {
        count: u64,
        limit: u64,
    },
    InvalidPhase {
        parity: u64,
    },
    InvalidStateToken {
        gen: u64,
        current: u64,
    },
    ReinitActive,
    ReinitWithoutInval,
    ReinitBeforeConsumption {
        gen: u64,
    },
    InvalWithOutstanding,
    ArrivalOverflow {
        required: u64,
        arrived: u64,
    },
    DropUnderflow {
        expected: u64,
        count: u64,
    },
    NoCompleteWouldComplete {
        count: u64,
        pending: u64,
    },
    /// Net tx-count (expected minus completed) left `±(2^20-1)`.
    TxCountOutOfRange {
        tx_count: i64,
    },
    TxOverDelivery {
        expected: u64,
        completed: u64,
    },
    PendingOverflow {
        required: u64,
    },
    ReuseBeforeConsumption {
        op: Op,
        gen: u64,
    },
    UnknownToken {
        gen: u64,
    },
    StaleCompletion {
        gen: u64,
        current: u64,
    },
    CompletionAfterComplete {
        gen: u64,
    },
    FutureNotBufferable {
        gen: u64,
        current: u64,
    },
    /// Launch exit with in-flight or unresolved work on a live slot: an
    /// issued completion that never landed, bytes buffered for a phase that
    /// never started, or a tx-count left unresolved while arrivals are still
    /// missing (`hardware_barriers.rs:2348-2396`). A phase whose arrivals are
    /// all in but whose bytes are missing is a tolerated terminal
    /// reservation (`hardware_barriers.rs:2364-2373`).
    IncompleteAtExit {
        gen: u64,
        outstanding: u32,
        buffered: u64,
        tx_count: i64,
    },
}

pub struct Mbarrier;

impl super::Protocol for Mbarrier {
    type State = State;
    type Cmd = Cmd;
    type Outcome = Outcome;
    type Error = Error;
    fn step(state: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
        step(state, cmd)
    }
}

/// Apply one command. On `Err` the state is unchanged.
pub fn step(state: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
    // Production form: every command validates against `&State` and builds a
    // small plan; only a fully validated plan mutates. No per-step clone.
    match cmd {
        Cmd::Init { count, layout_v1 } => {
            let limit = arrival_limit(layout_v1);
            if count == 0 || count > limit {
                return Err(Error::InvalidCount { count, limit });
            }
            if state.live {
                // PTX §9.7.15.16.12: init on a valid object is UB in every
                // policy; report the most specific diagnosis.
                return Err(if state.complete && !state.consumed {
                    Error::ReinitBeforeConsumption { gen: state.gen }
                } else if state.active() {
                    Error::ReinitActive
                } else {
                    Error::ReinitWithoutInval
                });
            }
            let policy = state.policy;
            *state = State { policy, live: true, layout_v1, expected: count, ..State::default() };
            Ok(Outcome::Done)
        }
        _ if !state.live => Err(Error::Uninitialized),
        Cmd::Inval => {
            if state.armed || !state.outstanding.is_empty() {
                return Err(Error::InvalWithOutstanding);
            }
            *state = State::new(state.policy);
            Ok(Outcome::Done)
        }
        Cmd::Arrive { count, tx, drop, no_complete } => {
            if count == 0 {
                return Ok(Outcome::Arrived { gen: state.gen, pending_before: 0, completed: false });
            }
            if no_complete {
                // Pending before roll-over and drop (hardware_barriers.rs:1157-1184).
                let pending = if state.complete { state.expected } else { state.required() - state.arrived };
                if count >= pending {
                    return Err(Error::NoCompleteWouldComplete { count, pending });
                }
            }
            let mut p = Phase::of(state, Op::Arrive)?;
            let mut expected = state.expected;
            if drop {
                if count >= expected {
                    return Err(Error::DropUnderflow { expected, count });
                }
                expected -= count;
                p.extra += count;
            }
            let pending_before = expected.saturating_add(p.extra) - p.arrived;
            p.arrive(expected, count, tx.unwrap_or(0))?;
            let gen = p.gen;
            p.commit(state);
            state.expected = expected;
            let completed = complete_if_ready(state);
            Ok(Outcome::Arrived { gen, pending_before, completed })
        }
        Cmd::ExpectTx { bytes } => {
            let mut p = Phase::of(state, Op::ExpectTx)?;
            p.tx_expected = p.tx_expected.saturating_add(bytes);
            range_check(p.tx_expected, p.tx_completed)?;
            let gen = p.gen;
            p.commit(state);
            Ok(Outcome::Updated { gen })
        }
        Cmd::IncPending { count } => {
            if count == 0 {
                return Ok(Outcome::Updated { gen: state.gen });
            }
            let mut p = Phase::of(state, Op::IncPending)?;
            let required = state.expected.saturating_add(p.extra).saturating_add(count);
            if required > arrival_limit(state.layout_v1) {
                return Err(Error::PendingOverflow { required });
            }
            p.extra += count;
            let gen = p.gen;
            p.commit(state);
            Ok(Outcome::Updated { gen })
        }
        Cmd::Issue => {
            let gen = state.gen + u64::from(state.complete);
            *state.outstanding.entry(gen).or_default() += 1;
            Ok(Outcome::Issued { gen })
        }
        Cmd::CompleteTx { gen, bytes } => {
            if !state.outstanding.contains_key(&gen) {
                return Err(Error::UnknownToken { gen });
            }
            if bytes == 0 {
                take_token(state, gen);
                return Ok(Outcome::Landed { completed: None });
            }
            if landing_target(state, gen)? {
                let buffered = state.buffered_next.saturating_add(bytes);
                if buffered > MAX_TX {
                    return Err(Error::TxCountOutOfRange { tx_count: -(buffered.min(i64::MAX as u64) as i64) });
                }
                take_token(state, gen);
                state.buffered_next = buffered;
                return Ok(Outcome::Landed { completed: None });
            }
            let completed_tx = state.tx_completed.saturating_add(bytes);
            if state.arrived == state.required() && completed_tx > state.tx_expected {
                return Err(Error::TxOverDelivery { expected: state.tx_expected, completed: completed_tx });
            }
            range_check(state.tx_expected, completed_tx)?;
            take_token(state, gen);
            state.tx_completed = completed_tx;
            let completed = complete_if_ready(state).then_some(state.gen);
            Ok(Outcome::Landed { completed })
        }
        Cmd::DeferredArrive { gen, count } => {
            if count == 0 {
                return Err(Error::InvalidCount { count, limit: arrival_limit(state.layout_v1) });
            }
            if !state.outstanding.contains_key(&gen) {
                return Err(Error::UnknownToken { gen });
            }
            landing_target(state, gen)?;
            let mut p = Phase::of(state, Op::DeferredArrive)?;
            p.arrive(state.expected, count, 0)?;
            take_token(state, gen);
            p.commit(state);
            let completed = complete_if_ready(state).then_some(state.gen);
            Ok(Outcome::Landed { completed })
        }
        Cmd::TestParity { parity } => parity_query(state, parity, false),
        Cmd::WaitParity { parity } => parity_query(state, parity, true),
        Cmd::TestState { gen } => {
            let preceding = gen.checked_add(1) == Some(state.gen);
            if gen != state.gen && !preceding {
                return Err(Error::InvalidStateToken { gen, current: state.gen });
            }
            if gen == state.gen && !state.complete {
                return Ok(Outcome::NotReady);
            }
            if gen == state.gen {
                state.consumed = true;
            }
            Ok(Outcome::Ready { gen: Some(gen) })
        }
    }
}






/// Signed tx-count of the current phase.
pub fn tx_count(s: &State) -> i128 {
    s.tx_expected as i128 - s.tx_completed as i128
}



enum Target {
    Current,
    Next,
}




/// Launch-exit check, run after the completion queue has drained.
pub fn quiescent(s: &State) -> Result<(), Error> {
    if !s.live {
        return Ok(());
    }
    let tx = tx_count(s);
    let outstanding = s.outstanding_total();
    if outstanding == 0 && s.buffered_next == 0 && (s.complete || tx == 0 || s.arrived >= s.required()) {
        return Ok(());
    }
    Err(Error::IncompleteAtExit {
        gen: s.gen,
        outstanding: outstanding.min(u32::MAX as u64) as u32,
        buffered: s.buffered_next,
        tx_count: tx.clamp(i64::MIN as i128, i64::MAX as i128) as i64,
    })
}

/// Invariants every reachable state satisfies (checked by the property tests).
pub fn check_invariants(s: &State) -> Result<(), String> {
    if !s.live {
        return if *s == State::new(s.policy) { Ok(()) } else { Err("dead slot carries state".into()) };
    }
    let checks: [(bool, &str); 9] = [
        (s.expected <= arrival_limit(s.layout_v1), "expected above limit"),
        (s.arrived <= s.required(), "arrived > required"),
        (tx_count(s).unsigned_abs() <= MAX_TX as u128 && s.buffered_next <= MAX_TX, "tx-count out of range"),
        (!s.complete || (s.arrived == s.required() && s.tx_completed == s.tx_expected), "complete without balanced counters"),
        (!s.consumed || s.complete, "consumed but not complete"),
        (s.buffered_next == 0 || s.complete, "buffer for next gen while pending"),
        (s.last_completed == if s.complete { Some(s.gen) } else { s.gen.checked_sub(1) }, "last_completed out of step"),
        (s.outstanding.keys().all(|&g| g <= s.gen + u64::from(s.complete)), "token bound to an unreachable generation"),
        (s.outstanding.values().all(|&n| n > 0), "empty token entry"),
    ];
    match checks.iter().find(|(ok, _)| !ok) {
        Some((_, msg)) => Err(msg.to_string()),
        None => Ok(()),
    }
}

/// Counters of the phase a mutating command acts on: the current phase, or
/// the next one when the current phase has completed (lazy roll-over).
struct Phase {
    gen: u64,
    rolled: bool,
    arrived: u64,
    extra: u64,
    tx_expected: u64,
    tx_completed: u64,
}

impl Phase {
    /// PTX §9.7.15.16.5.1: rolling over a completed phase needs a successful
    /// wait on it for arrive-on operations (every policy) and, in Strict,
    /// also for `expect_tx`.
    fn of(s: &State, op: Op) -> Result<Phase, Error> {
        if !s.complete {
            return Ok(Phase {
                gen: s.gen,
                rolled: false,
                arrived: s.arrived,
                extra: s.extra,
                tx_expected: s.tx_expected,
                tx_completed: s.tx_completed,
            });
        }
        if !s.consumed && (op != Op::ExpectTx || s.policy == Policy::Strict) {
            return Err(Error::ReuseBeforeConsumption { op, gen: s.gen });
        }
        Ok(Phase { gen: s.gen + 1, rolled: true, arrived: 0, extra: 0, tx_expected: 0, tx_completed: s.buffered_next })
    }

    /// Validate and stage `count` arrivals that also expect `tx` bytes.
    fn arrive(&mut self, expected: u64, count: u64, tx: u64) -> Result<(), Error> {
        let required = expected.saturating_add(self.extra);
        let arrived = self.arrived.saturating_add(count);
        if arrived > required {
            return Err(Error::ArrivalOverflow { required, arrived });
        }
        let tx_expected = self.tx_expected.saturating_add(tx);
        if arrived == required && self.tx_completed > tx_expected {
            return Err(Error::TxOverDelivery { expected: tx_expected, completed: self.tx_completed });
        }
        range_check(tx_expected, self.tx_completed)?;
        self.arrived = arrived;
        self.tx_expected = tx_expected;
        Ok(())
    }

    fn commit(self, s: &mut State) {
        if self.rolled {
            s.gen = self.gen;
            s.complete = false;
            s.consumed = false;
            s.buffered_next = 0;
        }
        s.arrived = self.arrived;
        s.extra = self.extra;
        s.tx_expected = self.tx_expected;
        s.tx_completed = self.tx_completed;
    }
}

/// Signed tx-count range of ISA Table 43, checked on the state.
fn range_check(tx_expected: u64, tx_completed: u64) -> Result<(), Error> {
    let tx = tx_expected as i128 - tx_completed as i128;
    if tx.unsigned_abs() > MAX_TX as u128 {
        return Err(Error::TxCountOutOfRange { tx_count: tx.clamp(i64::MIN as i128, i64::MAX as i128) as i64 });
    }
    Ok(())
}

fn complete_if_ready(s: &mut State) -> bool {
    let ready = !s.complete && s.arrived == s.required() && s.tx_completed == s.tx_expected;
    if ready {
        s.complete = true;
        s.last_completed = Some(s.gen);
        s.consumed = s.armed;
        s.armed = false;
    }
    ready
}

/// Where a completion bound to `gen` lands: `Ok(false)` current phase,
/// `Ok(true)` buffered for the next phase.
fn landing_target(s: &State, gen: u64) -> Result<bool, Error> {
    if gen < s.gen {
        Err(Error::StaleCompletion { gen, current: s.gen })
    } else if gen == s.gen && s.complete {
        Err(Error::CompletionAfterComplete { gen })
    } else if gen == s.gen {
        Ok(false)
    } else if s.complete && gen == s.gen + 1 {
        Ok(true)
    } else {
        Err(Error::FutureNotBufferable { gen, current: s.gen })
    }
}

fn take_token(s: &mut State, gen: u64) {
    if let Some(n) = s.outstanding.get_mut(&gen) {
        *n -= 1;
        if *n == 0 {
            s.outstanding.remove(&gen);
        }
    }
}

fn parity_query(s: &mut State, parity: u64, blocking: bool) -> Result<Outcome, Error> {
    if parity > 1 {
        return Err(Error::InvalidPhase { parity });
    }
    if parity != s.completed_parity() {
        if !blocking {
            return Ok(Outcome::NotReady);
        }
        // Parked wait: consumes the next completion (strict_mbarrier.rs:1669-1689).
        s.armed = true;
        return Ok(Outcome::Blocked);
    }
    if s.complete && s.last_completed == Some(s.gen) {
        s.consumed = true;
    }
    Ok(Outcome::Ready { gen: s.last_completed })
}

use crate::report::FindingKind;

/// Map a protocol error to its report kind.
pub fn finding_kind(e: &Error) -> FindingKind {
    match e {
        // A landing the scheduler should never produce: infrastructure.
        Error::UnknownToken { .. } | Error::FutureNotBufferable { .. } => FindingKind::RuntimeError,
        Error::IncompleteAtExit { .. } => FindingKind::UnwaitedAsync,
        _ => FindingKind::MbarrierMisuse,
    }
}
