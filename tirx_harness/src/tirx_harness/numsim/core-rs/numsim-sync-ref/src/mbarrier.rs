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

use std::collections::BTreeMap;

use crate::Policy;

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

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    Arrive,
    ExpectTx,
    IncPending,
    DeferredArrive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
}

pub struct Mbarrier;

impl crate::Protocol for Mbarrier {
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
    let mut next = state.clone();
    let outcome = apply(&mut next, cmd)?;
    *state = next;
    Ok(outcome)
}

fn apply(s: &mut State, cmd: Cmd) -> Result<Outcome, Error> {
    if let Cmd::Init { count, layout_v1 } = cmd {
        return init(s, count, layout_v1);
    }
    if !s.live {
        return Err(Error::Uninitialized);
    }
    match cmd {
        Cmd::Init { .. } => unreachable!(),
        Cmd::Inval => {
            if s.armed || !s.outstanding.is_empty() {
                return Err(Error::InvalWithOutstanding);
            }
            *s = State::new(s.policy);
            Ok(Outcome::Done)
        }
        Cmd::Arrive {
            count,
            tx,
            drop,
            no_complete,
        } => arrive(s, count, tx, drop, no_complete),
        Cmd::ExpectTx { bytes } => {
            begin(s, Op::ExpectTx)?;
            s.tx_expected = s.tx_expected.saturating_add(bytes);
            check_tx(s)?;
            Ok(Outcome::Updated { gen: s.gen })
        }
        Cmd::IncPending { count } => {
            if count == 0 {
                return Ok(Outcome::Updated { gen: s.gen });
            }
            begin(s, Op::IncPending)?;
            let required = s.required().saturating_add(count);
            if required > arrival_limit(s.layout_v1) {
                return Err(Error::PendingOverflow { required });
            }
            s.extra += count;
            Ok(Outcome::Updated { gen: s.gen })
        }
        Cmd::Issue => {
            let gen = if s.complete { s.gen + 1 } else { s.gen };
            *s.outstanding.entry(gen).or_default() += 1;
            Ok(Outcome::Issued { gen })
        }
        Cmd::CompleteTx { gen, bytes } => {
            take_token(s, gen)?;
            if bytes == 0 {
                return Ok(Outcome::Landed { completed: None });
            }
            match target(s, gen)? {
                Target::Current => {
                    let completed = s.tx_completed + bytes;
                    if s.arrived == s.required() && completed > s.tx_expected {
                        return Err(Error::TxOverDelivery {
                            expected: s.tx_expected,
                            completed,
                        });
                    }
                    s.tx_completed = completed;
                    check_tx(s)?;
                    Ok(Outcome::Landed {
                        completed: maybe_complete(s).then_some(s.gen),
                    })
                }
                Target::Next => {
                    // Buffering does not advance the phase, so strict does not
                    // require consumption here (strict_mbarrier.rs:1291-1330).
                    // These bytes become the next phase's negative tx-count.
                    s.buffered_next = s.buffered_next.saturating_add(bytes);
                    if s.buffered_next > MAX_TX {
                        return Err(Error::TxCountOutOfRange {
                            tx_count: -(s.buffered_next.min(i64::MAX as u64) as i64),
                        });
                    }
                    Ok(Outcome::Landed { completed: None })
                }
            }
        }
        Cmd::DeferredArrive { gen, count } => {
            if count == 0 {
                return Err(Error::InvalidCount {
                    count,
                    limit: arrival_limit(s.layout_v1),
                });
            }
            take_token(s, gen)?;
            target(s, gen)?;
            begin(s, Op::DeferredArrive)?;
            let (_, completed) = add_arrivals(s, count, 0)?;
            Ok(Outcome::Landed {
                completed: completed.then_some(s.gen),
            })
        }
        Cmd::TestParity { parity } => query_parity(s, parity, false),
        Cmd::WaitParity { parity } => query_parity(s, parity, true),
        Cmd::TestState { gen } => {
            let ready = if gen == s.gen {
                s.complete
            } else if gen.checked_add(1) == Some(s.gen) {
                true
            } else {
                return Err(Error::InvalidStateToken {
                    gen,
                    current: s.gen,
                });
            };
            if !ready {
                return Ok(Outcome::NotReady);
            }
            if gen == s.gen {
                s.consumed = true;
            }
            Ok(Outcome::Ready { gen: Some(gen) })
        }
    }
}

fn init(s: &mut State, count: u64, layout_v1: bool) -> Result<Outcome, Error> {
    let limit = arrival_limit(layout_v1);
    if !(1..=limit).contains(&count) {
        return Err(Error::InvalidCount { count, limit });
    }
    if s.live {
        // PTX §9.7.15.16.12: init on a valid mbarrier object is UB under
        // every policy. Report the most specific diagnosis.
        if s.complete && !s.consumed {
            return Err(Error::ReinitBeforeConsumption { gen: s.gen });
        }
        if s.active() {
            return Err(Error::ReinitActive);
        }
        return Err(Error::ReinitWithoutInval);
    }
    *s = State {
        policy: s.policy,
        live: true,
        layout_v1,
        expected: count,
        ..State::default()
    };
    Ok(Outcome::Done)
}

/// Roll a completed phase over before mutating counters.
fn begin(s: &mut State, op: Op) -> Result<(), Error> {
    if !s.complete {
        return Ok(());
    }
    // PTX §9.7.15.16.5.1: at least one successful test_wait/try_wait per
    // primary phase before an arrive-on in the next phase. Arrive-on
    // operations (arrive, pending increment, deferred arrive-on) are
    // checked under every policy. Extending the rule to expect_tx is Strict
    // only.
    if !s.consumed && (op != Op::ExpectTx || s.strict()) {
        return Err(Error::ReuseBeforeConsumption { op, gen: s.gen });
    }
    s.gen += 1;
    s.complete = false;
    s.consumed = false;
    s.arrived = 0;
    s.extra = 0;
    s.tx_expected = 0;
    s.tx_completed = std::mem::take(&mut s.buffered_next);
    Ok(())
}

fn arrive(
    s: &mut State,
    count: u64,
    tx: Option<u64>,
    drop: bool,
    no_complete: bool,
) -> Result<Outcome, Error> {
    if count == 0 {
        return Ok(Outcome::Arrived {
            gen: s.gen,
            pending_before: 0,
            completed: false,
        });
    }
    if no_complete {
        // hardware_barriers.rs:1157-1184: computed before roll-over and drop.
        let pending = if s.complete {
            s.expected
        } else {
            s.required() - s.arrived
        };
        if count >= pending {
            return Err(Error::NoCompleteWouldComplete { count, pending });
        }
    }
    begin(s, Op::Arrive)?;
    if drop {
        // Future phases expect `count` fewer arrivals; this phase is
        // unchanged because the same count is added to the phase-only extra.
        // PTX §9.7.15.16.17: dropping the expected count to zero is UB.
        if count >= s.expected {
            return Err(Error::DropUnderflow {
                expected: s.expected,
                count,
            });
        }
        s.expected -= count;
        s.extra += count;
    }
    let pending_before = s.required() - s.arrived;
    let (gen, completed) = add_arrivals(s, count, tx.unwrap_or(0))?;
    Ok(Outcome::Arrived {
        gen,
        pending_before,
        completed,
    })
}

fn add_arrivals(s: &mut State, count: u64, tx: u64) -> Result<(u64, bool), Error> {
    let arrived = s.arrived.saturating_add(count);
    if arrived > s.required() {
        return Err(Error::ArrivalOverflow {
            required: s.required(),
            arrived,
        });
    }
    let tx_expected = s.tx_expected.saturating_add(tx);
    if arrived == s.required() && s.tx_completed > tx_expected {
        return Err(Error::TxOverDelivery {
            expected: tx_expected,
            completed: s.tx_completed,
        });
    }
    s.arrived = arrived;
    s.tx_expected = tx_expected;
    check_tx(s)?;
    Ok((s.gen, maybe_complete(s)))
}

/// Signed tx-count of the current phase.
pub fn tx_count(s: &State) -> i128 {
    i128::from(s.tx_expected) - i128::from(s.tx_completed)
}

fn check_tx(s: &State) -> Result<(), Error> {
    let tx_count = tx_count(s);
    if tx_count.unsigned_abs() > u128::from(MAX_TX) {
        return Err(Error::TxCountOutOfRange {
            tx_count: tx_count.clamp(i64::MIN.into(), i64::MAX.into()) as i64,
        });
    }
    Ok(())
}

fn maybe_complete(s: &mut State) -> bool {
    if s.complete || s.arrived != s.required() || s.tx_completed != s.tx_expected {
        return false;
    }
    s.complete = true;
    s.last_completed = Some(s.gen);
    s.consumed = std::mem::take(&mut s.armed);
    true
}

enum Target {
    Current,
    Next,
}

/// Where a completion captured for `gen` lands.
fn target(s: &State, gen: u64) -> Result<Target, Error> {
    if gen < s.gen {
        Err(Error::StaleCompletion {
            gen,
            current: s.gen,
        })
    } else if gen == s.gen {
        if s.complete {
            Err(Error::CompletionAfterComplete { gen })
        } else {
            Ok(Target::Current)
        }
    } else if s.complete && gen == s.gen + 1 {
        Ok(Target::Next)
    } else {
        Err(Error::FutureNotBufferable {
            gen,
            current: s.gen,
        })
    }
}

fn take_token(s: &mut State, gen: u64) -> Result<(), Error> {
    match s.outstanding.get_mut(&gen) {
        Some(n) => {
            *n -= 1;
            if *n == 0 {
                s.outstanding.remove(&gen);
            }
            Ok(())
        }
        None => Err(Error::UnknownToken { gen }),
    }
}

fn query_parity(s: &mut State, parity: u64, blocking: bool) -> Result<Outcome, Error> {
    if parity > 1 {
        return Err(Error::InvalidPhase { parity });
    }
    if parity == s.completed_parity() {
        if s.complete && s.last_completed == Some(s.gen) {
            s.consumed = true;
        }
        return Ok(Outcome::Ready {
            gen: s.last_completed,
        });
    }
    if blocking {
        s.armed = true;
        Ok(Outcome::Blocked)
    } else {
        Ok(Outcome::NotReady)
    }
}

/// Invariants every reachable state satisfies (checked by the property tests).
pub fn check_invariants(s: &State) -> Result<(), String> {
    if !s.live {
        return (*s == State::new(s.policy))
            .then_some(())
            .ok_or_else(|| "dead slot carries state".into());
    }
    let ensure = |ok: bool, msg: &str| if ok { Ok(()) } else { Err(msg.to_string()) };
    ensure(
        s.expected <= arrival_limit(s.layout_v1),
        "expected above limit",
    )?;
    ensure(s.arrived <= s.required(), "arrived > required")?;
    ensure(
        tx_count(s).unsigned_abs() <= u128::from(MAX_TX) && s.buffered_next <= MAX_TX,
        "tx-count out of range",
    )?;
    ensure(
        !s.complete || (s.arrived == s.required() && s.tx_completed == s.tx_expected),
        "complete without balanced counters",
    )?;
    ensure(!s.consumed || s.complete, "consumed but not complete")?;
    ensure(
        s.buffered_next == 0 || s.complete,
        "buffer for next gen while pending",
    )?;
    let expected_last = if s.complete {
        Some(s.gen)
    } else {
        s.gen.checked_sub(1)
    };
    ensure(
        s.last_completed == expected_last,
        "last_completed out of step",
    )?;
    ensure(
        s.outstanding
            .keys()
            .all(|&g| g == s.gen || (s.complete && g == s.gen + 1) || g < s.gen),
        "token bound to an unreachable generation",
    )?;
    ensure(s.outstanding_total() < u64::MAX, "token count overflow")?;
    Ok(())
}
