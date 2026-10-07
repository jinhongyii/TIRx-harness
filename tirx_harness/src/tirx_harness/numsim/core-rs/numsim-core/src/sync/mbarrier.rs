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
    unimplemented!("W3: mbarrier::step")
}






/// Signed tx-count of the current phase.
pub fn tx_count(s: &State) -> i128 {
    unimplemented!("W3: mbarrier::tx_count")
}



enum Target {
    Current,
    Next,
}




/// Invariants every reachable state satisfies (checked by the property tests).
pub fn check_invariants(s: &State) -> Result<(), String> {
    unimplemented!("W3: mbarrier::check_invariants")
}

/// End-of-launch check (contract addition; not yet in the reference).
pub fn quiescent(s: &State) -> Result<(), Error> {
    unimplemented!("W3: mbarrier::quiescent")
}
