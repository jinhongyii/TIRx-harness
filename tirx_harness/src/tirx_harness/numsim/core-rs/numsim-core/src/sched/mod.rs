//! Scheduler: CTA-lockstep execution, seeded warp rotation, per-CTA inbox,
//! completion firing, deadlock and budget detection.
//!
//! # Contract (W2 implements)
//!
//! * A *round* visits every CTA; within a CTA every runnable warp gets one
//!   slice of at most `RunConfig::quantum` instructions, starting at a
//!   seeded rotation offset (`seed` + round), in ascending warp order from
//!   there. Blocked warps are retried every round (correct, if not optimal);
//!   W2 may skip warps whose `ResourceId` did not change.
//! * After a CTA's slices, a seeded subset of in-flight `AsyncOp`s lands
//!   (`CompletionPolicy`; `after` deps respected) and enabled sync
//!   `Completion`s are applied (`SyncTable::enabled`/`apply_completion`).
//! * Cross-CTA effects produced during a round (`ExecCtx::outbox`) are
//!   delivered at the target CTA's next inbox drain (once per round, before
//!   its slices); each drain emits `Observer::inbox_drain`.
//! * Termination: all warps exited and no completion pending -> Completed.
//!   No warp runnable, no completion can fire, no inbox message pending ->
//!   Deadlock (with the blocked resources). Round / step budget exceeded ->
//!   Incomplete. Any `ExecError` -> Error (the rest of the launch stops;
//!   Budget/Unsupported errors map to Incomplete).
//! * Fully deterministic for a fixed (module, inputs, config, seed).
//!
//! Single-threaded by design for now: one `Arena`, one `SyncTable`. CTA
//! parallelism (global memory by stripe, plan 2.4) is a later change
//! inside this module that must not alter the observer stream order
//! guarantees.

use crate::arena::{AllocId, Arena, BitSet, ValidityPolicy};
use crate::interp::{BufBinding, CtaCtx, ExecError, WarpState, WarpStepFn};
use crate::observe::{Actor, CtaId, Observer, WarpEnd, WarpId};
use crate::program::{LaunchShape, Module, Program};
use crate::site::SiteId;
use crate::sync::{ResourceId, SyncCmd, SyncError, SyncTable};
use std::collections::BTreeMap;
use std::fmt;

/// Which executor runs warp slices.
#[derive(Clone)]
pub enum Backend {
    Interp,
    /// One compiled step function per kernel of the module (same order as
    /// `Module::kernels`), produced by the codegen backend.
    Codegen(Vec<WarpStepFn>),
}

impl fmt::Debug for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Backend::Interp => f.write_str("Interp"),
            Backend::Codegen(v) => write!(f, "Codegen({} kernels)", v.len()),
        }
    }
}

/// When pending async completions fire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CompletionPolicy {
    /// Fire every ready completion at the end of the issuing CTA's slice.
    Eager,
    /// Fire a seeded random subset each round (default; exercises latency).
    #[default]
    Seeded,
}

#[derive(Clone, Debug)]
pub struct RunConfig {
    pub seed: u64,
    /// Max instructions per warp slice.
    pub quantum: u32,
    /// Max iterations of one loop instance per warp before Incomplete.
    pub loop_budget: u64,
    /// Max scheduler rounds per launch before Incomplete.
    pub max_rounds: u64,
    pub validity: ValidityPolicy,
    pub completions: CompletionPolicy,
    /// Register budget per CTA for setmaxnreg accounting.
    pub reg_pool: u32,
}

impl Default for RunConfig {
    fn default() -> RunConfig {
        RunConfig {
            seed: 0,
            quantum: 256,
            loop_budget: 1 << 24,
            max_rounds: 1 << 32,
            validity: ValidityPolicy::Error,
            completions: CompletionPolicy::Seeded,
            reg_pool: 65536,
        }
    }
}

/// A host argument value, bound by `ParamSlot::name` (or an alias).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArgValue {
    /// Device buffer contents; `valid = None` = all bytes valid.
    Buffer { bytes: Vec<u8>, valid: Option<BitSet> },
    /// By-value scalar bits.
    Scalar(u64),
    /// 128-byte tensor map image (`oplib::TensorMapDesc::encode`).
    TensorMap(Vec<u8>),
    /// Pointer to byte `offset` of another (buffer) argument.
    Pointer { target: String, offset: u64 },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Inputs {
    pub args: BTreeMap<String, ArgValue>,
}

/// Final contents of every buffer argument after the module ran.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outputs {
    pub buffers: BTreeMap<String, (Vec<u8>, BitSet)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RunStatus {
    Completed,
    Deadlock { blocked: Vec<(WarpId, ResourceId)> },
    /// Coverage could not be established (budget, unsupported op).
    Incomplete { reason: String, site: Option<SiteId> },
    Error(ExecError),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RunStats {
    pub instrs: u64,
    pub rounds: u64,
    pub completions: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunOutcome {
    /// Status of the first kernel that did not complete, else Completed.
    pub status: RunStatus,
    pub outputs: Outputs,
    pub stats: RunStats,
    /// `SyncTable::finish` violations of every launch.
    pub sync_leftovers: Vec<(ResourceId, SyncError)>,
}

/// Errors before execution starts (bad module, missing/ill-typed inputs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunError {
    InvalidProgram(String),
    MissingArg(String),
    BadArg { name: String, message: String },
    Backend(String),
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RunError {}

/// A cross-CTA effect awaiting delivery.
#[derive(Clone, Debug, PartialEq)]
pub enum InboxMsg {
    /// Bytes for a remote CTA's shared window (st to shared::cluster,
    /// multicast copies are delivered by completions instead).
    Write { alloc: AllocId, offset: u64, bytes: Vec<u8>, actor: Actor, site: SiteId },
    /// A sync command against a remote resource (remote mbarrier arrive,
    /// multicast arrive/expect_tx, cta_group::2 tcgen peer).
    Sync { resource: ResourceId, cmd: SyncCmd, actor: Actor, site: SiteId },
}

impl InboxMsg {
    /// Which CTA's inbox receives it (W2 maps allocs/resources to CTAs).
    pub fn target_hint(&self) -> Option<AllocId> {
        match self {
            InboxMsg::Write { alloc, .. } => Some(*alloc),
            InboxMsg::Sync { resource: ResourceId::Mbarrier { alloc, .. }, .. } => Some(*alloc),
            _ => None,
        }
    }
}

/// Per-CTA queue of incoming cross-CTA effects.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Inbox {
    pub msgs: Vec<InboxMsg>,
}

/// One CTA: its facts, its warps and its inbox.
#[derive(Clone, Debug)]
pub struct CtaState {
    pub ctx: CtaCtx,
    pub warps: Vec<WarpState>,
    pub inbox: Inbox,
    /// Buffer bindings for this CTA (shared-window buffers differ per CTA).
    pub buffers: Vec<BufBinding>,
}

/// Small deterministic RNG (splitmix64); no external dependency so the
/// schedule is stable across crate upgrades.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in `0..n` (n > 0).
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// Executes one launch of one kernel.
pub struct Scheduler<'p> {
    pub program: &'p Program,
    pub kernel_index: u32,
    pub shape: LaunchShape,
    pub config: RunConfig,
    pub ctas: Vec<CtaState>,
    pub sync: SyncTable,
    pub rng: Rng,
    pub round: u64,
    pub stats: RunStats,
}

impl<'p> Scheduler<'p> {
    /// Allocate per-CTA state (shared windows, tmem, local, params) in
    /// `arena` and bind buffers; host buffers must already be in `arena`
    /// (`globals` maps `ParamSlot::name` to its allocation).
    pub fn new(
        program: &'p Program,
        kernel_index: u32,
        shape: LaunchShape,
        arena: &mut Arena,
        globals: &BTreeMap<String, AllocId>,
        config: RunConfig,
    ) -> Result<Scheduler<'p>, RunError> {
        let _ = (program, kernel_index, shape, arena, globals, config);
        unimplemented!("W2: Scheduler::new")
    }

    /// Run the launch to completion / deadlock / error / budget.
    pub fn run(&mut self, arena: &mut Arena, observer: &mut dyn Observer, backend: &Backend) -> RunStatus {
        let _ = (arena, observer, backend);
        unimplemented!("W2: Scheduler::run")
    }

    /// Fire one pending completion (payload, then targets), emitting
    /// observer events. Exposed for tests.
    pub fn fire_completion(&mut self, index: usize, arena: &mut Arena, observer: &mut dyn Observer) -> Result<(), ExecError> {
        let _ = (index, arena, observer);
        unimplemented!("W2: Scheduler::fire_completion")
    }

    /// Why a warp ended, for `Observer::warp_done`.
    pub fn end_reason(&self, warp: WarpId) -> Option<WarpEnd> {
        let _ = warp;
        unimplemented!("W2: Scheduler::end_reason")
    }
}

/// Resolve a declared launch against inputs (`DimExpr` over scalar params).
pub fn resolve_launch(program: &Program, inputs: &Inputs) -> Result<LaunchShape, RunError> {
    let param = |p: crate::program::ParamId| -> Option<i64> {
        let slot = program.host_abi.get(p.0 as usize)?;
        match inputs.args.get(&slot.name) {
            Some(ArgValue::Scalar(v)) => Some(*v as i64),
            _ => None,
        }
    };
    let eval = |e: &crate::program::DimExpr, what: &str| -> Result<u32, RunError> {
        let v = e.eval(&param).ok_or_else(|| RunError::BadArg {
            name: what.to_string(),
            message: "cannot evaluate launch extent (missing scalar argument?)".into(),
        })?;
        u32::try_from(v).map_err(|_| RunError::BadArg { name: what.to_string(), message: format!("extent {v} out of range") })
    };
    let t = &program.topology;
    Ok(LaunchShape {
        grid: [eval(&t.grid[0], "grid.x")?, eval(&t.grid[1], "grid.y")?, eval(&t.grid[2], "grid.z")?],
        cluster: t.cluster,
        block: t.block,
        smem_bytes: t.static_smem_bytes + eval(&t.dyn_smem_bytes, "dyn_smem_bytes")?,
    })
}

/// Run every kernel of `module` in order with default config and `seed`.
pub fn run(
    module: &Module,
    inputs: &Inputs,
    observer: &mut dyn Observer,
    backend: &Backend,
    seed: u64,
) -> Result<RunOutcome, RunError> {
    let config = RunConfig { seed, ..RunConfig::default() };
    run_with_config(module, inputs, observer, backend, &config)
}

/// Run every kernel of `module` in order. Host buffers are allocated once
/// (shared by name across kernels), each kernel gets a fresh `SyncTable`
/// and CTA-private allocations; outputs are read back at the end.
pub fn run_with_config(
    module: &Module,
    inputs: &Inputs,
    observer: &mut dyn Observer,
    backend: &Backend,
    config: &RunConfig,
) -> Result<RunOutcome, RunError> {
    let _ = (module, inputs, observer, backend, config);
    unimplemented!("W2: sched::run_with_config")
}

/// Convenience: CtaId of global warp `w`.
pub fn cta_of(shape: &LaunchShape, w: WarpId) -> CtaId {
    CtaId(w.0 / shape.warps_per_cta().max(1))
}

