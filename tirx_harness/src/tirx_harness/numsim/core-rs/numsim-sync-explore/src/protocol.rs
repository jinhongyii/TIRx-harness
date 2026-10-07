//! Tiny local protocol state machines (mbarrier, named barrier, cluster barrier).
//!
//! PLACEHOLDER: these exist only so the explorer can run standalone. They will
//! be replaced by the shared `step` functions of the SyncTable (W3) / the
//! `numsim-sync-ref` reference machine. Semantics follow today's
//! `StrictMbarrierProtocol` / `StrictNamedBarrierProtocol` /
//! `StrictClusterBarrierProtocol` closely enough for the explorer tests, with
//! these deliberate simplifications:
//!
//! * no waiter registry: a blocking wait is a transition that is enabled only
//!   when ready (the plan's `Blocked(resource)` retry model);
//! * a generation is "consumed" by the first wait that observes it (today's
//!   strict model also releases the slot on the first consuming wait);
//! * no invalidate / re-init, no `.noinc` pending-count raises, no drop;
//! * error `kind` strings reuse today's stable finding kinds where one exists.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtoErr {
    /// Stable machine-readable kind (matches today's payload kinds when possible).
    pub kind: &'static str,
    pub detail: String,
    /// Incomplete (unsupported/unmodeled) rather than a protocol violation.
    pub incomplete: bool,
}

impl ProtoErr {
    fn error(kind: &'static str, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
            incomplete: false,
        }
    }

    fn incomplete(kind: &'static str, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
            incomplete: true,
        }
    }
}

impl fmt::Display for ProtoErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.detail)
    }
}

pub type ProtoResult<T> = Result<T, ProtoErr>;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct MbarState {
    pub initialized: bool,
    pub expected: u32,
    /// Arrivals still required in the current phase.
    pub remaining: u32,
    /// Expected minus completed transaction bytes of the current phase.
    pub tx: i64,
    /// Number of completed phases; the current phase parity is `gen & 1`.
    pub gen: u32,
    /// The last completed generation has not been observed by any wait yet.
    pub unconsumed: bool,
}

impl MbarState {
    fn require_init(&self, what: &str) -> ProtoResult<()> {
        if self.initialized {
            Ok(())
        } else {
            Err(ProtoErr::error(
                "mbarrier_use_before_init",
                format!("{what} used an uninitialized mbarrier"),
            ))
        }
    }

    pub fn init(&mut self, expected: u32) -> ProtoResult<()> {
        if self.initialized {
            return Err(ProtoErr::error(
                "mbarrier_reinit_without_inval",
                "mbarrier.init on an initialized mbarrier",
            ));
        }
        if expected == 0 {
            return Err(ProtoErr::error(
                "mbarrier_invalid_expected_arrivals",
                "expected arrival count must be positive",
            ));
        }
        *self = Self {
            initialized: true,
            expected,
            remaining: expected,
            tx: 0,
            gen: 0,
            unconsumed: false,
        };
        Ok(())
    }

    fn require_consumed(&self, kind: &'static str, what: &str) -> ProtoResult<()> {
        if self.unconsumed {
            Err(ProtoErr::error(
                kind,
                format!(
                    "{what} mutates generation {} before generation {} was consumed",
                    self.gen,
                    self.gen - 1
                ),
            ))
        } else {
            Ok(())
        }
    }

    /// Returns `(contributed generation, completed generation)`.
    pub fn arrive(&mut self, count: u32, expect_tx: u32) -> ProtoResult<(u32, Option<u32>)> {
        self.require_init("mbarrier.arrive")?;
        self.require_consumed("mbarrier_arrive_before_consumption", "mbarrier.arrive")?;
        if count > self.remaining {
            return Err(ProtoErr::error(
                "mbarrier_arrival_overflow",
                format!(
                    "{count} arrivals exceed the {} still required by generation {}",
                    self.remaining, self.gen
                ),
            ));
        }
        let gen = self.gen;
        self.tx += i64::from(expect_tx);
        self.remaining -= count;
        Ok((gen, self.maybe_complete()?))
    }

    pub fn expect_tx(&mut self, tx: u32) -> ProtoResult<u32> {
        self.require_init("mbarrier.expect_tx")?;
        self.require_consumed("mbarrier_expect_tx_before_consumption", "mbarrier.expect_tx")?;
        self.tx += i64::from(tx);
        Ok(self.gen)
    }

    /// Capture the generation an async completion will deliver to.
    pub fn capture(&self) -> ProtoResult<u32> {
        self.require_init("mbarrier completion issue")?;
        Ok(self.gen)
    }

    pub fn complete_tx(&mut self, gen: u32, tx: u32) -> ProtoResult<Option<u32>> {
        if gen != self.gen {
            return Err(ProtoErr::error(
                "mbarrier_stale_completion",
                format!(
                    "completion captured generation {gen} but the barrier is at {}",
                    self.gen
                ),
            ));
        }
        self.tx -= i64::from(tx);
        self.maybe_complete()
    }

    fn maybe_complete(&mut self) -> ProtoResult<Option<u32>> {
        if self.remaining != 0 {
            return Ok(None);
        }
        if self.tx < 0 {
            return Err(ProtoErr::error(
                "mbarrier_transaction_over_delivery",
                format!(
                    "generation {} received {} more transaction bytes than expected",
                    self.gen, -self.tx
                ),
            ));
        }
        if self.tx > 0 {
            return Ok(None);
        }
        let completed = self.gen;
        self.gen += 1;
        self.remaining = self.expected;
        self.unconsumed = true;
        Ok(Some(completed))
    }

    /// A wait is enabled when ready, or when the barrier is uninitialized so
    /// the step can expose use-before-init.
    pub fn wait_enabled(&self, parity: u8) -> bool {
        !self.initialized || (self.gen & 1) as u8 != parity
    }

    /// Returns the consumed generation, if the wait observed a completed one.
    pub fn wait(&mut self, parity: u8) -> ProtoResult<Option<u32>> {
        self.require_init("mbarrier.try_wait")?;
        if (self.gen & 1) as u8 == parity {
            return Err(ProtoErr::error(
                "mbarrier_acquire_before_completion",
                "wait stepped before its phase completed",
            ));
        }
        if self.gen > 0 && ((self.gen - 1) & 1) as u8 == parity {
            self.unconsumed = false;
            return Ok(Some(self.gen - 1));
        }
        Ok(None)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct NamedState {
    pub expected: u32,
    pub arrived: u32,
    pub gen: u32,
}

impl NamedState {
    /// Returns `(contributed generation, completed?)`.
    pub fn contribute(&mut self, expected: u32, count: u32) -> ProtoResult<(u32, bool)> {
        if expected == 0 || count == 0 {
            return Err(ProtoErr::error(
                "named_barrier_invalid_expected_arrivals",
                "named barrier counts must be positive",
            ));
        }
        if self.arrived > 0 && self.expected != expected {
            return Err(ProtoErr::error(
                "named_barrier_contract_mismatch",
                format!(
                    "generation {} expects {} threads, contribution expects {expected}",
                    self.gen, self.expected
                ),
            ));
        }
        if self.arrived + count > expected {
            return Err(ProtoErr::error(
                "named_barrier_arrival_overflow",
                format!(
                    "generation {} over-arrives: {} > {expected}",
                    self.gen,
                    self.arrived + count
                ),
            ));
        }
        self.expected = expected;
        self.arrived += count;
        let gen = self.gen;
        if self.arrived == expected {
            self.arrived = 0;
            self.gen += 1;
            return Ok((gen, true));
        }
        Ok((gen, false))
    }

    pub fn exit_check(&self) -> ProtoResult<()> {
        if self.arrived == 0 {
            Ok(())
        } else {
            Err(ProtoErr::error(
                "named_barrier_incomplete_generation",
                format!(
                    "generation {} has {} of {} required arrivals at exit",
                    self.gen, self.arrived, self.expected
                ),
            ))
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ClusterState {
    pub participants: u32,
    pub arrived: u32,
    pub gen: u32,
    /// Per projection-local warp: the generation it arrived in and has not waited for.
    pub arrived_gen: Box<[Option<u32>]>,
}

impl ClusterState {
    pub fn with_warps(warps: usize) -> Self {
        Self {
            arrived_gen: vec![None; warps].into_boxed_slice(),
            ..Self::default()
        }
    }

    /// Returns `(contributed generation, completed?)`.
    pub fn arrive(&mut self, warp: usize, participants: u32) -> ProtoResult<(u32, bool)> {
        if self.arrived > 0 && self.participants != participants {
            return Err(ProtoErr::error(
                "cluster_barrier_contract_mismatch",
                "participant count changes within a generation",
            ));
        }
        match self.arrived_gen[warp] {
            Some(g) if g == self.gen => {
                return Err(ProtoErr::error(
                    "cluster_barrier_early_arrival",
                    format!("warp arrives twice in generation {g}"),
                ))
            }
            Some(g) => {
                return Err(ProtoErr::incomplete(
                    "cluster_barrier_rearrival_without_wait_unmodeled",
                    format!("warp re-arrives without consuming generation {g}"),
                ))
            }
            None => {}
        }
        self.participants = participants;
        self.arrived_gen[warp] = Some(self.gen);
        self.arrived += 1;
        let gen = self.gen;
        if self.arrived == participants {
            self.arrived = 0;
            self.gen += 1;
            return Ok((gen, true));
        }
        Ok((gen, false))
    }

    pub fn wait_enabled(&self, warp: usize) -> bool {
        self.arrived_gen[warp].is_none_or(|g| g < self.gen)
    }

    /// Returns the consumed generation.
    pub fn wait(&mut self, warp: usize) -> ProtoResult<u32> {
        match self.arrived_gen[warp].take() {
            None => Err(ProtoErr::error(
                "cluster_barrier_wait_before_arrival",
                "barrier.cluster.wait without a preceding arrive",
            )),
            Some(g) if g < self.gen => Ok(g),
            Some(_) => Err(ProtoErr::error(
                "cluster_barrier_resume_before_completion",
                "wait stepped before its generation completed",
            )),
        }
    }

    pub fn exit_check(&self) -> ProtoResult<()> {
        if self.arrived == 0 {
            Ok(())
        } else {
            Err(ProtoErr::incomplete(
                "cluster_barrier_warp_exit_unmodeled",
                format!(
                    "generation {} is missing {} participant warps; exit-aware membership is not modeled",
                    self.gen,
                    self.participants - self.arrived
                ),
            ))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResState {
    Mbar(MbarState),
    Named(NamedState),
    Cluster(ClusterState),
}
