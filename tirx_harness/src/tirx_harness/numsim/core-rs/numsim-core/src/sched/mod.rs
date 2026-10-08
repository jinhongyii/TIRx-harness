//! Scheduler: rounds over scheduling partitions, seeded warp rotation,
//! async landing, serial points, deadlock and budget detection.
//!
//! # Rounds
//!
//! * A *round* visits every resident CTA; within a CTA every runnable warp
//!   gets one slice of at most `RunConfig::quantum` instructions, starting
//!   at a seeded rotation offset (`seed`, round, CTA), in ascending warp
//!   order from there. Blocked warps are retried every round (a blocking
//!   handler re-checks its resource and returns `Blocked` again).
//! * After a CTA's slices, its ready `AsyncOp`s land (`CompletionPolicy::
//!   Eager`: all of them; `Seeded`, the default: a seeded random subset, so
//!   an op may stay in flight for several rounds; `after` dependencies are
//!   respected) and enabled sync `Completion`s are applied.
//! * Cross-CTA effects inside a cluster (remote shared stores, remote
//!   mbarrier commands) apply synchronously at issue; the start of each
//!   CTA's turn in a round is observable as `Observer::round_boundary`.
//! * Residency: clusters are admitted in linear cluster order while at most
//!   `RunConfig::max_resident_ctas` CTAs are resident (all of them for
//!   cooperative launches and `grid.sync`); a cluster's CTAs are always
//!   co-resident. A retired cluster's shared/TMEM/local allocations are
//!   reset and reused (`AllocEnd` + `AllocBegin`).
//!
//! # Partitions and parallelism
//!
//! The unit of ownership is the cluster: each resident cluster is a
//! [`Partition`] with its own CTAs, `SyncTable`, `LaunchAux`, event buffer
//! and RNG. One partition holds every CTA instead when the program has
//! launch-wide state (grid sync / cooperative launch, declared sync words
//! or `wait_until`, mixed tcgen05 `cta_group`s) or `RunConfig::
//! single_partition` is set. The rule depends only on the program and the
//! config, never on the observer or the worker count.
//!
//! A round has three phases:
//!
//! 1. **Parallel phase.** With one partition it runs directly on the arena
//!    (sequential semantics). With several, each runs against its own arena
//!    shard (`Arena::make_shard`): private allocations in place, global and
//!    param allocations through a copy-on-write 4 KiB stripe overlay over
//!    the round-start state, so a cluster sees other clusters' global writes
//!    of this round only from the next round on. Partitions run on up to
//!    `RunConfig::workers` threads (a launch-lifetime pool); the result
//!    does not depend on which thread ran what.
//! 2. **Merge.** Shards merge in partition order (the later partition wins
//!    a byte both wrote). Partitions after the first failing one are
//!    discarded. Buffered events replay in an order consistent with what
//!    each partition observed (`Arena::shard_replay_order`: a partition
//!    that read global bytes another wrote this round goes first; same-byte
//!    writers keep partition order); `Access::seq` is assigned at replay.
//!    When no such order exists (each read what the other wrote: a
//!    store-buffering outcome), partition order is used and an
//!    `incomplete` diagnostic says the stream is not faithful. Only the
//!    replay order (never results) depends on whether anyone observes.
//! 3. **Serial phase** (main arena, partition order): global
//!    read-modify-writes inside shards are serial points (atom/red re-run as
//!    one instruction; bulk/tensor/async reductions into global memory land
//!    here), so no update is lost.
//!
//! Results and observer streams are identical for any worker count.
//!
//! # Progress, deadlock, termination
//!
//! A slice made progress if it ended other than `Blocked` or completed an
//! instruction with `Instr::is_progress` (a write, a committed sync
//! transition, an async issue). A loop iteration whose only effects were
//! failed polls (`test_wait` false, a non-weak load) is spin-parked at its
//! `LoopEnd` only when the warp's registers and masks are exactly those of
//! the previous such iteration (a fixed point: the next iteration repeats
//! unless another actor or an async op changes what it polls). A loop whose
//! state advances (a bounded probe, a retry counter) is never parked.
//!
//! * All warps exited, no cluster left and the async queues drained ->
//!   Completed.
//! * A round with no progress anywhere (no slice progress, landing,
//!   completion or admission, after force-landing every
//!   ready op) -> Deadlock with the blocked resources, except:
//!   a blocked warp with a divergent mask -> Incomplete (`divergent_block`:
//!   structured SIMT cannot interleave its arms further); a warp blocked on
//!   an explicit-count named barrier of a CTA with exited warps ->
//!   Incomplete (G8). Count-less named barriers are exit-aware (exited
//!   warps leave their membership; PTX §9.7.14.7).
//! * Round budget -> Incomplete. Any `ExecError` -> Error, except Budget /
//!   Unsupported / `Op(Unsupported)` -> Incomplete.
//! * Fully deterministic for a fixed (module, inputs, config, seed).
//!
//! # Divergent scheduling limits
//!
//! A warp blocked inside one arm of a divergent `If` with an `Else` runs
//! the other arm (`interp::divergent_switch`), swapping between suspended
//! arms; a swapped warp reports the resumed arm's resource as blocked.
//! Not covered (reported `divergent_block` incomplete, never Deadlock): an
//! `If` without an `Else` whose skipped lanes would produce what the
//! blocked lanes wait for (`if lane != 0 { wait } ; if lane == 0 { arrive }`)
//! and loop-exit divergence (`for i < lane { wait }`).

mod partition;
pub use partition::{MMA_SHARED_A, MMA_SHARED_B};
mod pool;

pub use partition::Partition;
use partition::{sched_error, Env};

use crate::arena::{addr, AllocId, Arena, BitSet, ByteSpan, Init, Owner, Space, ValidityPolicy, View};
use crate::interp::support;
use crate::interp::{step_warp, BufBinding, CtaCtx, ExecError, ExecErrorKind, LaunchAux, Loaded, WarpState, WarpStatus, WarpStepFn};
use crate::observe::{Actor, CtaId, LaunchInfo, Observer, ProtocolStatus, SyncEvent, SyncKind, WarpEnd, WarpId};
use crate::oplib::OpErrorKind;
use crate::program::{Instr, LaunchShape, Module, ParamKind, Program};
use crate::report::Finding;
use crate::site::SiteId;
use crate::sync::{async_group, cluster, mbarrier, setmaxnreg, Policy, ResourceId, ResourceInit, SyncCmd, SyncError, SyncTable};
use crate::value::WarpMask;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;

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
    /// Worker threads (CTA/cluster parallelism). `<= 1` = single-threaded.
    /// Results and observer streams do not depend on it (module docs).
    pub workers: usize,
    /// Run every resident CTA in one partition (one arena, no shards, no
    /// round-snapshot isolation between clusters): the sequential reference
    /// the partitioned scheduler is tested against. Default `false`.
    pub single_partition: bool,
    /// Max co-resident CTAs (0 = all). Clusters are admitted whole.
    pub max_resident_ctas: u32,
    /// Resident cluster ids (W8-4): only these clusters run; `None` = all.
    pub subset: Option<Vec<u32>>,
}

impl Default for RunConfig {
    fn default() -> RunConfig {
        RunConfig {
            seed: 0,
            quantum: 256,
            loop_budget: 1 << 24,
            max_rounds: 1 << 32,
            validity: ValidityPolicy::ZeroAndReport,
            completions: CompletionPolicy::Seeded,
            reg_pool: 65536,
            workers: 1,
            single_partition: false,
            max_resident_ctas: 1024,
            subset: None,
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
    /// A tensor map over buffer argument `base` (+ byte `offset`): its
    /// `global_address` is resolved at bind time to the engine's VA.
    TensorMapOf { base: String, offset: u64, desc: crate::oplib::TensorMapDesc },
    /// Pointer to byte `offset` of another (buffer) argument.
    Pointer { target: String, offset: u64 },
    /// Bytes `[offset, offset + len)` of the `Buffer` argument `target`
    /// (CONTRACT_REQUESTS W8-6): parameters bound to overlapping host memory
    /// share ONE allocation, so writes through one name are visible through
    /// the other and checkers see the aliasing. Accepted for `Buffer`
    /// slots, as a `Pointer`/`TensorMapOf` target, and read back in
    /// `Outputs` as that slice of the target.
    View { target: String, offset: u64, len: u64 },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Inputs {
    pub args: BTreeMap<String, ArgValue>,
    /// Host address of `Buffer` arguments (when the host array has one).
    /// Not used for placement (V2C-35 ruling): every top-level buffer gets
    /// a fresh synthetic base aligned like `cudaMalloc` (at least 256
    /// bytes); only `View` arguments sit at an offset inside their region.
    pub host_addrs: BTreeMap<String, u64>,
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
    /// `attrs`: structured facts for the report (`warp`, `lanes`, `budget`,
    /// `max_rounds`, ... — W11-pin-message item 5).
    Incomplete { reason: String, site: Option<SiteId>, attrs: BTreeMap<String, serde_json::Value> },
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
    /// Kernel index of `status` when it is not Completed (W8-2).
    pub failed_kernel: Option<u32>,
    pub outputs: Outputs,
    pub stats: RunStats,
    /// `SyncTable::finish` violations of every launch.
    pub sync_leftovers: Vec<(ResourceId, SyncError)>,
    /// Review-level runtime findings (uninitialized reads under
    /// `ValidityPolicy::ZeroAndReport`), W8-5.
    pub diagnostics: Vec<Finding>,
    /// `RunConfig::subset` echoed back (checkers report subset runs as
    /// `Incomplete { reason: "subset_execution" }`), W8-4.
    pub subset: Option<Vec<u32>>,
}

/// Errors before execution starts (bad module, missing/ill-typed inputs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunError {
    InvalidProgram(String),
    MissingArg(String),
    BadArg { name: String, message: String },
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RunError {}

/// One CTA: its facts and its warps.
#[derive(Clone, Debug)]
pub struct CtaState {
    pub ctx: CtaCtx,
    pub warps: Vec<WarpState>,
    /// Buffer bindings for this CTA (shared-window buffers differ per CTA).
    pub buffers: Vec<BufBinding>,
}

impl CtaState {
    fn finished(&self) -> bool {
        self.warps.iter().all(|w| !matches!(w.status, WarpStatus::Running | WarpStatus::Blocked(_)))
    }
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
    /// Resident partitions, in admission (cluster) order.
    pub partitions: Vec<Partition>,
    pub round: u64,
    pub stats: RunStats,
    pub loaded: Loaded,
    /// Kernel parameter block.
    pub params: AllocId,
    /// Buffer bindings (identical for every CTA: shared bindings are
    /// window-relative).
    bindings: Vec<BufBinding>,
    /// Clusters not yet admitted, in order.
    pending: VecDeque<u32>,
    /// Reusable (smem, tmem) allocations of retired CTAs, and local ones.
    free_cta: Vec<(AllocId, AllocId)>,
    free_local: Vec<AllocId>,
    free_regbuf: Vec<AllocId>,
    ends: BTreeMap<WarpId, WarpEnd>,
    observing: bool,
    wants_history: bool,
    /// One partition for every cluster (grid-wide state); see `single_partition`.
    single: bool,
    /// Next `Access::seq` (assigned at replay).
    next_seq: u64,
    /// Exit-check violations of retired partitions.
    leftovers: Vec<(ResourceId, SyncError)>,
    /// A round's replay order was not faithful (see `stream_cycle`).
    stream_cycle_reported: bool,
    /// Review diagnostics (uninitialized reads), in partition order.
    pub diagnostics: Vec<Finding>,
    /// Instructions of retired partitions.
    retired_instrs: u64,
    retired_completions: u64,
    /// Declared words seeded into new partitions (launch-scope regions).
    launch_words: crate::interp::aux::WordTable,
    /// Launch-wide CLC task queue (W12-gaps 6), shared by every partition.
    clc: std::sync::Arc<crate::interp::aux::ClcTasks>,
}

/// Linear cluster id and rank of a CTA (ctaid coordinates); the inverse
/// of [`cta_coords`] (tests).
#[cfg(test)]
fn cluster_of(shape: &LaunchShape, c: [u32; 3]) -> (u32, u32) {
    let cl = shape.cluster.map(|x| x.max(1));
    let ncl = [shape.grid[0] / cl[0], shape.grid[1] / cl[1]];
    let cc = [c[0] / cl[0], c[1] / cl[1], c[2] / cl[2]];
    let id = cc[0] + cc[1] * ncl[0] + cc[2] * ncl[0] * ncl[1];
    let rank = c[0] % cl[0] + (c[1] % cl[1]) * cl[0] + (c[2] % cl[2]) * cl[0] * cl[1];
    (id, rank)
}

/// ctaid of rank `rank` of cluster `id`.
fn cta_coords(shape: &LaunchShape, id: u32, rank: u32) -> [u32; 3] {
    let cl = shape.cluster.map(|x| x.max(1));
    let ncl = [shape.grid[0] / cl[0], shape.grid[1] / cl[1]];
    let cc = [id % ncl[0], (id / ncl[0]) % ncl[1], id / (ncl[0] * ncl[1])];
    let r = [rank % cl[0], (rank / cl[0]) % cl[1], rank / (cl[0] * cl[1])];
    [cc[0] * cl[0] + r[0], cc[1] * cl[1] + r[1], cc[2] * cl[2] + r[2]]
}

/// Per declared region `(alloc, region start)`: the merged-history log
/// position of each of a partition's local log entries (`merge_words`).
type PositionMap = HashMap<(AllocId, u64), Vec<Option<usize>>>;

/// One partition's work item in a parallel round: the partition, its arena
/// shard and its result slot.
type RoundItem<'a> = (&'a mut Partition, &'a mut Arena, &'a mut Option<Result<bool, ExecError>>);

fn linear(shape: &LaunchShape, c: [u32; 3]) -> u32 {
    c[0] + c[1] * shape.grid[0] + c[2] * shape.grid[0] * shape.grid[1]
}

/// The error kinds that mean "coverage could not be established".
fn is_incomplete(k: &ExecErrorKind) -> bool {
    matches!(k, ExecErrorKind::Budget | ExecErrorKind::Unsupported | ExecErrorKind::Op(OpErrorKind::Unsupported))
}

/// Status of a failed launch.
pub fn classify(e: ExecError) -> RunStatus {
    if is_incomplete(&e.kind) {
        let mut attrs = e.attrs.clone();
        if e.warp != WarpId(u32::MAX) {
            attrs.insert("warp".into(), serde_json::json!(e.warp.0));
        }
        if !e.lanes.is_empty() {
            attrs.insert("lanes".into(), serde_json::json!(e.lanes.0));
        }
        attrs.insert("kernel".into(), serde_json::json!(e.kernel));
        RunStatus::Incomplete { reason: format!("{:?}: {}", e.kind, e.message), site: Some(e.site), attrs }
    } else {
        RunStatus::Error(e)
    }
}

impl<'p> Scheduler<'p> {
    /// Allocate per-launch state and bind buffers; host buffers must
    /// already be in `arena` (`globals` maps `ParamSlot::name` to its
    /// allocation). Scalar parameters read as zero (use
    /// [`run_with_config`] to bind them).
    pub fn new(
        program: &'p Program,
        kernel_index: u32,
        shape: LaunchShape,
        arena: &mut Arena,
        globals: &BTreeMap<String, AllocId>,
        config: RunConfig,
    ) -> Result<Scheduler<'p>, RunError> {
        let loaded = Loaded::new(program);
        let params = arena.alloc(Space::Param, Owner::Launch, "params", loaded.param_bytes, Init::Zeroed);
        for (i, p) in program.host_abi.iter().enumerate() {
            if let Some(&a) = globals.get(&p.name) {
                let va = arena.get(a).base;
                write_param(arena, params, loaded.param_offsets[i], &va.to_le_bytes());
            }
        }
        Self::with_params(program, kernel_index, shape, arena, globals, &BTreeMap::new(), params, loaded, config)
    }

    fn with_params(
        program: &'p Program,
        kernel_index: u32,
        shape: LaunchShape,
        arena: &mut Arena,
        globals: &BTreeMap<String, AllocId>,
        view_lens: &BTreeMap<String, u64>,
        params: AllocId,
        loaded: Loaded,
        config: RunConfig,
    ) -> Result<Scheduler<'p>, RunError> {
        let invalid = |m: String| RunError::InvalidProgram(m);
        let cl = shape.cluster.map(|x| x.max(1));
        for (g, c) in shape.grid.iter().zip(cl) {
            if !g.is_multiple_of(c) {
                return Err(invalid(format!("grid {:?} is not a multiple of cluster {:?}", shape.grid, shape.cluster)));
            }
        }
        if shape.threads_per_cta() == 0 {
            return Err(invalid("empty CTA".into()));
        }
        let scalar = |arena: &Arena, p: crate::program::ParamId| -> Option<i64> {
            let off = *loaded.param_offsets.get(p.0 as usize)? as usize;
            let b = arena.read_raw(params, ByteSpan::new(off as u64, 8));
            Some(i64::from_le_bytes(b.try_into().ok()?))
        };
        // Buffer bindings.
        let n = program.buffers.len();
        let mut bindings: Vec<Option<BufBinding>> = vec![None; n];
        let mut local_off = 0u64;
        let mut local_offsets = vec![0u64; n];
        for (i, b) in program.buffers.iter().enumerate() {
            if b.space == Space::Local && b.view_of.is_none() {
                let len = b.byte_len.as_ref().and_then(|e| e.eval(&|p| scalar(arena, p))).unwrap_or(0).max(0) as u64;
                let align = (b.align as u64).max(1);
                local_off = local_off.div_ceil(align) * align;
                local_offsets[i] = local_off;
                local_off += len;
            }
        }
        // Register-space buffers: their own per-lane layout (same rule as
        // `Loaded::reg_per_lane`).
        let mut reg_off = 0u64;
        for (i, b) in program.buffers.iter().enumerate() {
            if b.space == Space::Reg && b.view_of.is_none() {
                let len = b.byte_len.as_ref().and_then(|e| e.eval(&|p| scalar(arena, p))).unwrap_or(0).max(0) as u64;
                let align = (b.align as u64).max(1);
                reg_off = reg_off.div_ceil(align) * align;
                local_offsets[i] = reg_off;
                reg_off += len;
            }
        }
        fn bind(
            i: usize,
            program: &Program,
            arena: &mut Arena,
            globals: &BTreeMap<String, AllocId>,
            view_lens: &BTreeMap<String, u64>,
            loaded: &Loaded,
            params: AllocId,
            local_offsets: &[u64],
            out: &mut Vec<Option<BufBinding>>,
            depth: u32,
            scalar: &dyn Fn(&Arena, crate::program::ParamId) -> Option<i64>,
        ) -> Result<BufBinding, RunError> {
            if let Some(b) = out[i] {
                return Ok(b);
            }
            if depth > 64 {
                return Err(RunError::InvalidProgram("cyclic buffer views".into()));
            }
            let d = &program.buffers[i];
            let len = d.byte_len.as_ref().and_then(|e| e.eval(&|p| scalar(arena, p))).map(|v| v.max(0) as u64);
            let b = if let Some(parent) = d.view_of {
                let pb = bind(parent.0 as usize, program, arena, globals, view_lens, loaded, params, local_offsets, out, depth + 1, scalar)?;
                match pb {
                    BufBinding::View(v) => {
                        let len = len.unwrap_or(v.len.saturating_sub(d.base));
                        BufBinding::View(View { alloc: v.alloc, offset: v.offset + d.base, len })
                    }
                    BufBinding::SharedWindow { offset, len: pl } => BufBinding::SharedWindow {
                        offset: offset + d.base as u32,
                        len: len.unwrap_or(pl.saturating_sub(d.base)),
                    },
                    BufBinding::Local { offset, per_lane } => {
                        BufBinding::Local { offset: offset + d.base, per_lane: len.unwrap_or(per_lane.saturating_sub(d.base)) }
                    }
                    BufBinding::Reg { offset, per_lane } => {
                        BufBinding::Reg { offset: offset + d.base, per_lane: len.unwrap_or(per_lane.saturating_sub(d.base)) }
                    }
                    BufBinding::Tmem { base_col, cols, base_reg } => {
                        BufBinding::Tmem { base_col: base_col + d.base as u32, cols, base_reg: d.base_reg.or(base_reg) }
                    }
                    BufBinding::Unbound => BufBinding::Unbound,
                }
            } else {
                match d.space {
                    Space::Shared => BufBinding::SharedWindow { offset: d.base as u32, len: len.unwrap_or(0) },
                    Space::Local => BufBinding::Local { offset: local_offsets[i], per_lane: len.unwrap_or(0) },
                    Space::Global | Space::Param => match d.param_slot {
                        Some(slot) => {
                            let ps = &program.host_abi[slot.0 as usize];
                            let poff = loaded.param_offsets[slot.0 as usize];
                            match ps.kind {
                                ParamKind::TensorMap => BufBinding::View(View { alloc: params, offset: poff, len: 128 }),
                                ParamKind::Buffer | ParamKind::Pointer => {
                                    let va = u64::from_le_bytes(
                                        arena.read_raw(params, ByteSpan::new(poff, 8)).try_into().unwrap(),
                                    );
                                    match arena.resolve_global(va, 0) {
                                        Ok((alloc, off)) => {
                                            let size = arena.get(alloc).size;
                                            // A view-bound slot ends where its view ends.
                                            let avail = size.saturating_sub(off).min(view_lens.get(&ps.name).copied().unwrap_or(u64::MAX));
                                            BufBinding::View(View { alloc, offset: off, len: len.unwrap_or(avail).min(avail) })
                                        }
                                        Err(_) => match globals.get(&ps.name) {
                                            Some(&a) => BufBinding::View(arena.view(a)),
                                            None => BufBinding::Unbound,
                                        },
                                    }
                                }
                                ParamKind::Scalar | ParamKind::ImplicitShape { .. } => BufBinding::Unbound,
                            }
                        }
                        None => {
                            // Kernel-private global scratch.
                            let size = len.unwrap_or(0);
                            let a = arena.alloc(Space::Global, Owner::Launch, &d.name, size, Init::Uninit);
                            BufBinding::View(arena.view(a))
                        }
                    },
                    Space::Tmem => {
                        let cols = d.shape.last().and_then(|e| e.eval(&|p| scalar(arena, p))).unwrap_or(addr::TMEM_COLS as i64);
                        BufBinding::Tmem {
                            base_col: d.base as u32,
                            cols: cols.clamp(1, addr::TMEM_COLS as i64) as u32,
                            base_reg: d.base_reg,
                        }
                    }
                    Space::Reg => BufBinding::Reg { offset: local_offsets[i], per_lane: len.unwrap_or(0) },
                }
            };
            out[i] = Some(b);
            Ok(b)
        }
        for i in 0..n {
            bind(i, program, arena, globals, view_lens, &loaded, params, &local_offsets, &mut bindings, 0, &scalar)?;
        }
        let bindings: Vec<BufBinding> = bindings.into_iter().map(|b| b.unwrap_or(BufBinding::Unbound)).collect();

        let mut pending: VecDeque<u32> = (0..shape.num_clusters()).collect();
        if let Some(s) = &config.subset {
            pending.retain(|c| s.contains(c));
        }
        let clc = std::sync::Arc::new(crate::interp::aux::ClcTasks::new(shape.num_clusters(), shape.ctas_per_cluster(), config.subset.clone()));
        Ok(Scheduler {
            program,
            kernel_index,
            shape,
            config,
            partitions: Vec::new(),
            round: 0,
            stats: RunStats::default(),
            loaded,
            params,
            bindings,
            pending,
            free_cta: Vec::new(),
            free_local: Vec::new(),
            free_regbuf: Vec::new(),
            ends: BTreeMap::new(),
            observing: false,
            wants_history: false,
            single: false,
            next_seq: 0,
            leftovers: Vec::new(),
            stream_cycle_reported: false,
            diagnostics: Vec::new(),
            retired_instrs: 0,
            retired_completions: 0,
            launch_words: Default::default(),
            clc,
        })
    }

    fn host_event(&self, observer: &mut dyn Observer, kind: SyncKind) {
        if self.observing {
            observer.sync(&SyncEvent { kernel: self.kernel_index, actor: Actor::Host, seq: 0, site: SiteId::NONE, frames: Vec::new(), lanes: WarpMask::NONE, kind });
        }
    }

    fn all_resident(&self) -> bool {
        self.program.topology.cooperative
            || self.config.max_resident_ctas == 0
            || self.program.code.iter().any(|i| matches!(i, Instr::GridSync))
    }

    /// Launches with launch-wide state run as one partition: cooperative /
    /// `grid.sync` (grid barrier), declared synchronization words /
    /// `wait_until` (a word's history is one stream), and kernels mixing
    /// `cta_group` values (the kernel-wide tcgen05 rule). Independent of the
    /// worker count and of the observer.
    /// The per-thread register count every warpgroup starts with
    /// (setmaxnreg `Configure`), or `None` when the program has no
    /// setmaxnreg. `Launch::regs_per_thread` when lowering set it; otherwise
    /// the legacy caller base (`frontend-rs emit/sync.rs`
    /// `calling_initial_count`): the even split of the 512-register
    /// per-thread pool over the CTA's warpgroups, capped by the largest
    /// `setmaxnreg.inc` target (the compiler's register cap, default 256)
    /// and by the launch bounds (`min_blocks_per_sm` CTAs resident).
    fn initial_regs_per_thread(&self) -> Result<Option<u32>, ExecError> {
        let t = &self.program.topology;
        if t.regs_per_thread != 0 {
            return Ok(Some(t.regs_per_thread));
        }
        if !self.loaded.uses_setmaxnreg {
            return Ok(None);
        }
        let wg = self.shape.warps_per_cta().div_ceil(setmaxnreg::WARPS_PER_GROUP).max(1);
        let g = setmaxnreg::GRANULARITY;
        let default = setmaxnreg::CTA_REGISTER_POOL / wg / g * g;
        let cap = self
            .program
            .code
            .iter()
            .filter_map(|i| match *i {
                Instr::SetMaxNReg { inc: true, count } if (setmaxnreg::MIN_COUNT..=setmaxnreg::MAX_COUNT).contains(&count) && count % g == 0 => Some(count),
                _ => None,
            })
            .max()
            .unwrap_or(setmaxnreg::MAX_COUNT);
        let min_blocks = t.min_blocks_per_sm.unwrap_or(1).max(1);
        let resident = setmaxnreg::CTA_REGISTER_POOL / (wg * min_blocks) / g * g;
        if resident < setmaxnreg::MIN_COUNT {
            return Err(sched_error(
                ExecErrorKind::Unsupported,
                self.kernel_index,
                WarpId(u32::MAX),
                SiteId::NONE,
                "launch bounds cannot provide the minimum 24 registers per thread required by setmaxnreg".into(),
            ));
        }
        Ok(Some(default.min(cap).min(resident)))
    }

    fn single_partition(&self) -> bool {
        let p = self.program;
        let mut groups = std::collections::BTreeSet::new();
        for i in &p.code {
            let g = match i {
                Instr::TcgenAlloc { cta_group, .. }
                | Instr::TcgenDealloc { cta_group, .. }
                | Instr::TcgenRelinquish { cta_group }
                | Instr::TcgenCommit { cta_group, .. } => Some(*cta_group),
                Instr::TcgenMma(a) => Some(a.cta_group),
                Instr::TcgenCp(a) => Some(a.cta_group),
                _ => None,
            };
            if let Some(g) = g {
                groups.insert(g.max(1));
            }
        }
        // Never depends on the observer: observers must not change
        // program-visible behaviour (and partitioning changes when other
        // partitions' global writes become visible).
        // Declared words / `wait_until` do NOT force one partition: the
        // launch-wide word history is merged in replay order after every
        // phase (`merge_words`), which keeps verdict numbering identical to
        // the delivery order (README / engine-review "partitioned words").
        self.config.single_partition || self.all_resident() || groups.len() > 1
    }

    fn new_partition(&self, cluster: u32) -> Partition {
        let wpc = self.shape.warps_per_cta();
        let init = ResourceInit { policy: Policy::Numeric, cluster_warps: self.shape.ctas_per_cluster() * wpc, warps_per_cta: wpc };
        let mut aux = LaunchAux { kernel: self.kernel_index, wants_history: self.wants_history, ..LaunchAux::default() };
        aux.words = self.launch_words.clone();
        aux.words.clear_dirty();
        aux.clc = self.clc.clone();
        // Async op ids are partition-scoped so they do not depend on the
        // order partitions run in.
        aux.next_async = (cluster as u64 + 1) << 40;
        Partition::new(cluster, SyncTable::new(init), aux, self.config.seed, self.observing, self.wants_history)
    }

    /// Admit cluster `id`: allocate its CTAs and warps into a partition.
    fn admit(&mut self, id: u32, arena: &mut Arena, observer: &mut dyn Observer) -> Result<(), ExecError> {
        let shape = self.shape;
        let n = shape.ctas_per_cluster().max(1);
        let wpc = shape.warps_per_cta();
        let nslots = *self.loaded.slots.last().unwrap_or(&0) as usize;
        let tmem_bytes = if self.loaded.uses_tmem { addr::TMEM_BYTES } else { 0 };
        let pi = if self.single && !self.partitions.is_empty() {
            self.partitions[0].clusters.push(id);
            0
        } else {
            let p = self.new_partition(id);
            self.partitions.push(p);
            self.partitions.len() - 1
        };
        // tcgen05 lifecycle state of each CTA pair, with the target's
        // `.exclusive` allocation limit (PTX Table 58: 576 columns on
        // sm_107f, 512 elsewhere; sync-isa-answers column rule).
        if self.loaded.uses_tmem {
            let limit = exclusive_tmem_columns(self.program.arch.as_deref());
            let part = &mut self.partitions[pi];
            for pair_rank in 0..n.div_ceil(2) {
                let res = ResourceId::TcgenLifecycle { cluster: id, pair_rank: pair_rank as u8 };
                part.sync.resources.insert(res, crate::sync::Resource::Tcgen(crate::sync::tcgen::State::new(limit)));
            }
        }
        let mut ctas: Vec<CtaCtx> = Vec::with_capacity(n as usize);
        for rank in 0..n {
            let c = cta_coords(&shape, id, rank);
            let cid = CtaId(linear(&shape, c));
            let reuse = self.free_cta.iter().position(|&(s, t)| {
                arena.get(s).size == shape.smem_bytes as u64 && arena.get(t).size == tmem_bytes
            });
            let (smem, tmem) = match reuse {
                Some(i) => {
                    let (s, t) = self.free_cta.swap_remove(i);
                    arena.reset(s);
                    arena.reset(t);
                    (s, t)
                }
                None => (
                    arena.alloc(Space::Shared, Owner::Cta(cid.0), &format!("smem[cta{}]", cid.0), shape.smem_bytes as u64, Init::Uninit),
                    arena.alloc(Space::Tmem, Owner::Cta(cid.0), &format!("tmem[cta{}]", cid.0), tmem_bytes, Init::Uninit),
                ),
            };
            ctas.push(CtaCtx {
                id: cid,
                ctaid: c,
                cluster: id,
                rank_in_cluster: rank,
                cluster_ctas: Vec::new(),
                cluster_smem: Vec::new(),
                cluster_tmem: Vec::new(),
                smem,
                tmem,
                params: self.params,
            });
        }
        let ids: Vec<CtaId> = ctas.iter().map(|c| c.id).collect();
        let smems: Vec<AllocId> = ctas.iter().map(|c| c.smem).collect();
        let tmems: Vec<AllocId> = ctas.iter().map(|c| c.tmem).collect();
        for ctx in ctas {
            let mut ctx = ctx;
            ctx.cluster_ctas = ids.clone();
            ctx.cluster_smem = smems.clone();
            ctx.cluster_tmem = tmems.clone();
            let cid = ctx.id;
            if self.observing {
                for (a, sp) in [(ctx.smem, Space::Shared), (ctx.tmem, Space::Tmem)] {
                    let size = arena.get(a).size;
                    self.host_event(observer, SyncKind::AllocBegin { alloc: a, space: sp, size, cta: cid });
                }
            }
            let mut warps = Vec::with_capacity(wpc as usize);
            for w in 0..wpc {
                let wid = WarpId(cid.0 * wpc + w);
                let mut ws = WarpState::new_deferred(wid, cid, w, nslots, shape.warp_lanes(w));
                if self.loaded.local_per_lane > 0 {
                    let size = self.loaded.local_per_lane * 32;
                    let a = match self.free_local.pop() {
                        Some(a) => {
                            arena.reset(a);
                            a
                        }
                        None => arena.alloc(Space::Local, Owner::Warp(wid.0), &format!("local[w{}]", wid.0), size, Init::Uninit),
                    };
                    ws.local = Some(a);
                    self.host_event(observer, SyncKind::AllocBegin { alloc: a, space: Space::Local, size, cta: cid });
                }
                if self.loaded.reg_per_lane > 0 {
                    let size = self.loaded.reg_per_lane * 32;
                    let a = match self.free_regbuf.pop() {
                        Some(a) => {
                            arena.reset(a);
                            a
                        }
                        None => arena.alloc_register_buffer(Owner::Warp(wid.0), &format!("regs[w{}]", wid.0), size, Init::Uninit),
                    };
                    ws.regbuf = Some(a);
                    self.host_event(observer, SyncKind::AllocBegin { alloc: a, space: Space::Reg, size, cta: cid });
                }
                if self.observing {
                    let size = self.program.regs.len() as u64 * 32 * 8;
                    let ra = arena.alloc(Space::Reg, Owner::Warp(wid.0), &format!("regs[w{}]", wid.0), size, Init::Uninit);
                    let part = &mut self.partitions[pi];
                    let i = wid.0 as usize;
                    if part.aux.reg_allocs.len() <= i {
                        part.aux.reg_allocs.resize(i + 1, AllocId(u32::MAX));
                    }
                    part.aux.reg_allocs[i] = ra;
                    self.host_event(observer, SyncKind::AllocBegin { alloc: ra, space: Space::Reg, size, cta: cid });
                }
                warps.push(ws);
            }
            // Declared words in the shared window: validated and declared
            // in every mode (counts and overflow are observer-independent;
            // W13-1/3).
            {
                for (i, b) in self.program.buffers.iter().enumerate() {
                    if let (true, BufBinding::SharedWindow { offset, len }) = (b.sync_words, self.bindings[i]) {
                        let (base, w, n) = sync_word_array(b, offset as u64, len)
                            .map_err(|m| sched_error(ExecErrorKind::Unsupported, self.kernel_index, WarpId(u32::MAX), SiteId::NONE, m))?;
                        if self.observing || self.wants_history {
                            let spans: Vec<ByteSpan> = (0..n).map(|k| ByteSpan::new(base + k * w, w)).collect();
                            self.partitions[pi].aux.words.declare_words(arena, ctx.smem, &spans);
                            for span in spans {
                                self.host_event(observer, SyncKind::DeclareWord { alloc: ctx.smem, span });
                            }
                        } else {
                            self.partitions[pi].aux.words.declare_array(ctx.smem, base, w, n);
                        }
                    }
                }
            }
            let regs = self.initial_regs_per_thread()?;
            let part = &mut self.partitions[pi];
            if let Some(count) = regs {
                let res = ResourceId::RegPool { cta: cid };
                let cmd = SyncCmd::RegPool(setmaxnreg::Cmd::Configure { count });
                part.sync.step(res, cmd).map_err(|e| {
                    sched_error(ExecErrorKind::Protocol(e.clone()), self.kernel_index, WarpId(cid.0 * wpc), SiteId::NONE, format!("{e:?}"))
                })?;
                // Launch-bounds register budget: a host-side protocol command
                // that precedes every warp of the CTA.
                if self.observing {
                    observer.sync(&SyncEvent {
                        kernel: self.kernel_index,
                        actor: Actor::Host,
                        seq: 0,
                        site: SiteId::NONE,
                        frames: Vec::new(),
                        lanes: WarpMask::NONE,
                        kind: SyncKind::Protocol {
                            cmds: vec![crate::observe::ProtocolCmd { res, cmd, counts: Default::default(), observed_parity: None }],
                            collective: None,
                            issued: Vec::new(),
                            status: ProtocolStatus::Committed,
                        },
                    });
                }
            }
            if self.loaded.uses_cluster_barrier {
                // Lanes beyond the CTA's thread count never take part.
                for ws in &warps {
                    let missing = WarpMask::ALL.and_not(ws.live);
                    if !missing.is_empty() {
                        let res = ResourceId::Cluster { cluster: id };
                        let warp = ctx.rank_in_cluster * wpc + ws.warp_in_cta;
                        let cmd = SyncCmd::Cluster(cluster::Cmd::Exit { warp, lanes: missing.bits() });
                        part.sync.step(res, cmd).map_err(|e| {
                            sched_error(ExecErrorKind::Internal, self.kernel_index, ws.id, SiteId::NONE, format!("{e:?}"))
                        })?;
                    }
                }
            }
            part.ctas.push(CtaState { ctx, warps, buffers: self.bindings.clone() });
        }
        Ok(())
    }

    /// Retire finished clusters and empty partitions, admit pending
    /// clusters. Returns whether anything changed.
    fn turnover(&mut self, arena: &mut Arena, observer: &mut dyn Observer) -> Result<bool, ExecError> {
        let mut changed = false;
        if !self.pending.is_empty() {
            for pi in 0..self.partitions.len() {
                let mut i = 0;
                while i < self.partitions[pi].ctas.len() {
                    let part = &self.partitions[pi];
                    let cl = part.ctas[i].ctx.cluster;
                    let mut j = i;
                    while j < part.ctas.len() && part.ctas[j].ctx.cluster == cl {
                        j += 1;
                    }
                    let done = part.ctas[i..j].iter().all(|c| c.finished());
                    // Async ops still writing into the cluster keep it resident.
                    let busy = part.sync.async_ops.iter().any(|op| part.ctas[i..j].iter().any(|c| c.ctx.id == op.source.cta));
                    if done && !busy {
                        let gone: Vec<CtaState> = self.partitions[pi].ctas.drain(i..j).collect();
                        self.partitions[pi].clusters.retain(|&c| c != cl);
                        for c in gone {
                            for a in [c.ctx.smem, c.ctx.tmem] {
                                self.host_event(observer, SyncKind::AllocEnd { alloc: a });
                            }
                            self.free_cta.push((c.ctx.smem, c.ctx.tmem));
                            for w in c.warps {
                                if let Some(a) = w.local {
                                    self.host_event(observer, SyncKind::AllocEnd { alloc: a });
                                    self.free_local.push(a);
                                }
                                if let Some(a) = w.regbuf {
                                    self.host_event(observer, SyncKind::AllocEnd { alloc: a });
                                    self.free_regbuf.push(a);
                                }
                            }
                        }
                        changed = true;
                        continue;
                    }
                    i = j;
                }
            }
            // Drop emptied partitions (their exit checks are final).
            let mut k = 0;
            while k < self.partitions.len() {
                if self.partitions[k].ctas.is_empty() && self.partitions[k].sync.async_ops.is_empty() {
                    let p = self.partitions.remove(k);
                    self.leftovers.extend(p.sync.quiescent());
                    self.retired_instrs += p.counters.instrs;
                    self.retired_completions += p.completions;
                    continue;
                }
                k += 1;
            }
        }
        let cap = if self.all_resident() { u32::MAX } else { self.config.max_resident_ctas };
        let per = self.shape.ctas_per_cluster().max(1);
        // `admit` adds exactly `per` CTAs (W13: counted, not re-summed per
        // admission, which was quadratic in the cluster count).
        let mut resident: u32 = self.partitions.iter().map(|p| p.ctas.len() as u32).sum();
        while let Some(&id) = self.pending.front() {
            if resident > 0 && resident + per > cap {
                break;
            }
            self.pending.pop_front();
            self.admit(id, arena, observer)?;
            resident += per;
            changed = true;
        }
        Ok(changed)
    }

    /// Run the launch to completion / deadlock / error / budget.
    pub fn run(&mut self, arena: &mut Arena, observer: &mut dyn Observer) -> RunStatus {
        let step_fn: WarpStepFn = step_warp;
        self.observing = observer.enabled();
        self.wants_history = self.observing && observer.wants_word_history();
        self.single = self.single_partition();
        observer.begin_launch(&LaunchInfo { program: self.program, kernel_index: self.kernel_index, shape: self.shape, arena });
        // Launch-scope allocations and declared global words.
        if self.observing {
            let mut seen = Vec::new();
            for b in &self.bindings {
                if let BufBinding::View(v) = b {
                    if !seen.contains(&v.alloc) {
                        seen.push(v.alloc);
                    }
                }
            }
            seen.push(self.params);
            seen.sort();
            for a in seen {
                let al = arena.get(a);
                let (space, size) = (al.space, al.size);
                self.host_event(observer, SyncKind::AllocBegin { alloc: a, space, size, cta: CtaId(u32::MAX) });
            }
        }
        self.launch_words.images = self.wants_history;
        {
            for (i, b) in self.program.buffers.iter().enumerate() {
                if let (true, BufBinding::View(v)) = (b.sync_words, self.bindings[i]) {
                    let (base, w, n) = match sync_word_array(b, v.offset, v.len) {
                        Ok(a) => a,
                        Err(m) => return classify(sched_error(ExecErrorKind::Unsupported, self.kernel_index, WarpId(u32::MAX), SiteId::NONE, m)),
                    };
                    if self.observing || self.wants_history {
                        let spans: Vec<ByteSpan> = (0..n).map(|k| ByteSpan::new(base + k * w, w)).collect();
                        self.launch_words.declare_words(arena, v.alloc, &spans);
                        for span in spans {
                            self.host_event(observer, SyncKind::DeclareWord { alloc: v.alloc, span });
                        }
                    } else {
                        self.launch_words.declare_array(v.alloc, base, w, n);
                    }
                }
            }
        }
        let workers = self.config.workers.max(1);
        // Deterministic numerics: the engine's FP environment on this
        // thread for the whole run, the caller's restored afterwards.
        let _fp = pool::FpEnvGuard::enter();
        let result = if workers > 1 && !self.single {
            // A launch-lifetime pool; the calling thread is one of the workers.
            let pool = pool::Pool::new(workers - 1);
            std::thread::scope(|scope| {
                for _ in 0..workers - 1 {
                    scope.spawn(|| pool.worker());
                }
                // Stop the workers however the loop ends (a panic on this
                // thread must not leave the scope joining blocked workers).
                let _stop = pool::ShutdownGuard(&pool);
                self.run_loop(arena, observer, step_fn, Some(&pool))
            })
        } else {
            self.run_loop(arena, observer, step_fn, None)
        };
        let status = match result {
            Ok(s) => s,
            Err(e) => classify(e),
        };
        self.finish(arena, observer, &status);
        status
    }

    /// Deliver the parallel phase's buffered events of partitions `order`
    /// (replay order), after sequence assignment and verdict renumbering
    /// (decision 17; W6 H4). Each partition is offered to
    /// `Observer::fork` (keyed by its first cluster id, D3); a child gets
    /// that partition's events on the worker pool, with the exact `seq`
    /// values serial replay would assign; the children are joined, and the
    /// other partitions replayed into `observer`, serially in replay order
    /// after the pool returns (D4). Then one `phase_end` (H3). With the
    /// default observer (no children) this is today's serial replay.
    fn replay_partitions(&mut self, order: &[usize], observer: &mut dyn Observer, pool: Option<&pool::Pool>) {
        use crate::observe::{ForkedObserver, PartitionInfo};
        if order.is_empty() {
            return;
        }
        let ctas_of = |p: &Partition| -> Vec<CtaId> { p.ctas.iter().map(|c| c.ctx.id).collect() };
        let key_of = |p: &Partition| p.clusters.first().copied().unwrap_or(0);
        let mut children: Vec<Option<Box<dyn ForkedObserver>>> = Vec::with_capacity(order.len());
        let mut starts: Vec<u64> = Vec::with_capacity(order.len());
        let mut seq = self.next_seq;
        // Fork only when the children can actually run in parallel (a pool
        // and more than one partition); otherwise serial replay below. Within
        // a phase every partition is offered a fork or none is (W5-17a).
        let parallel = pool.is_some() && order.len() > 1;
        for &k in order {
            let p = &self.partitions[k];
            let ctas = ctas_of(p);
            let n = p.events.access_count();
            children.push(if parallel { observer.fork(&PartitionInfo { key: key_of(p), ctas: &ctas, accesses: n }) } else { None });
            starts.push(seq);
            seq += n;
        }
        if children.iter().any(Option::is_some) {
            let mut evs: Vec<Option<&mut partition::EventBuffer>> = self.partitions.iter_mut().map(|p| Some(&mut p.events)).collect();
            type Item<'a> = (&'a mut partition::EventBuffer, &'a mut Box<dyn ForkedObserver>, u64, Option<Box<dyn std::any::Any + Send>>);
            let items: Vec<std::cell::UnsafeCell<Item<'_>>> = order
                .iter()
                .zip(children.iter_mut())
                .zip(&starts)
                .filter_map(|((&k, c), &st)| c.as_mut().map(|c| (k, c, st)))
                .map(|(k, c, st)| std::cell::UnsafeCell::new((evs[k].take().expect("each partition once"), c, st, None)))
                .collect();
            struct Items<'a, 'b>(&'b [std::cell::UnsafeCell<Item<'a>>]);
            // SAFETY: `par_for` hands every index to exactly one thread, and
            // each item borrows a distinct event buffer and child.
            unsafe impl Sync for Items<'_, '_> {}
            impl<'a> Items<'a, '_> {
                #[allow(clippy::mut_from_ref)]
                unsafe fn item(&self, i: usize) -> &mut Item<'a> {
                    &mut *self.0[i].get()
                }
            }
            let shared = Items(&items);
            let run = |i: usize| {
                // SAFETY: see `Items`.
                let (ev, child, start, panic) = unsafe { shared.item(i) };
                let mut s = *start;
                let child: &mut dyn Observer = child.as_mut();
                if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ev.replay(child, &mut s))) {
                    *panic = Some(payload);
                }
            };
            match pool {
                Some(pool) if items.len() > 1 => pool.par_for(items.len(), &run),
                _ => (0..items.len()).for_each(run),
            }
            for it in items {
                if let Some(payload) = it.into_inner().3 {
                    std::panic::resume_unwind(payload);
                }
            }
        }
        for (i, &k) in order.iter().enumerate() {
            match children[i].take() {
                Some(child) => {
                    let p = &self.partitions[k];
                    let ctas = ctas_of(p);
                    observer.join(&PartitionInfo { key: key_of(p), ctas: &ctas, accesses: p.events.access_count() }, child);
                }
                None => {
                    let mut s = starts[i];
                    self.partitions[k].events.replay(observer, &mut s);
                }
            }
        }
        self.next_seq = seq;
        observer.phase_end(self.round);
    }

    /// Replay partition `pi`'s buffered events and absorb its round results.
    fn absorb(&mut self, pi: usize, observer: &mut dyn Observer) {
        self.partitions[pi].events.replay(observer, &mut self.next_seq);
        self.absorb_state(pi);
    }

    /// Absorb partition `pi`'s warp ends and diagnostics (no events).
    fn absorb_state(&mut self, pi: usize) {
        let p = &mut self.partitions[pi];
        for (w, e) in p.ends.drain(..) {
            self.ends.insert(w, e);
        }
        self.diagnostics.append(&mut p.aux.diagnostics);
    }

    /// The parallel phase of a round: every partition runs its CTAs against
    /// its own arena shard (or directly on `arena` when one partition is
    /// resident), on up to `workers` threads. Shards are merged in partition
    /// order; buffered events are replayed in partition order. Results do
    /// not depend on the worker count.
    fn parallel_phase(
        &mut self,
        arena: &mut Arena,
        observer: &mut dyn Observer,
        step_fn: WarpStepFn,
        pool: Option<&pool::Pool>,
    ) -> Result<bool, ExecError> {
        let n = self.partitions.len();
        if n == 0 {
            return Ok(false);
        }
        if n == 1 {
            let env = Env {
                program: self.program,
                loaded: &self.loaded,
                shape: &self.shape,
                config: &self.config,
                kernel: self.kernel_index,
                step_fn,
                round: self.round,
                observing: self.observing,
            };
            let r = self.partitions[0].run_round(&env, arena);
            self.replay_partitions(&[0], observer, None);
            self.absorb_state(0);
            self.merge_words(&[0]);
            return r;
        }
        let mut shards: Vec<Arena> = Vec::with_capacity(n);
        for p in &self.partitions {
            // The ownership list only feeds a debug assertion.
            let private = if cfg!(debug_assertions) { p.private_allocs() } else { Vec::new() };
            let mut sh = arena.make_shard(&private);
            if self.observing {
                // Replay order must follow what each partition observed.
                sh.track_shard_reads();
            }
            shards.push(sh);
        }
        let mut results: Vec<Option<Result<bool, ExecError>>> = (0..n).map(|_| None).collect();
        {
            let env = Env {
                program: self.program,
                loaded: &self.loaded,
                shape: &self.shape,
                config: &self.config,
                kernel: self.kernel_index,
                step_fn,
                round: self.round,
                observing: self.observing,
            };
            let kernel = self.kernel_index;
            let items: Vec<std::cell::UnsafeCell<RoundItem<'_>>> = self
                .partitions
                .iter_mut()
                .zip(shards.iter_mut())
                .zip(results.iter_mut())
                .map(|((p, a), r)| std::cell::UnsafeCell::new((p, a, r)))
                .collect();
            struct Items<'a, 'b>(&'b [std::cell::UnsafeCell<RoundItem<'a>>]);
            // SAFETY: `par_for` hands every index to exactly one thread.
            unsafe impl Sync for Items<'_, '_> {}
            impl<'a> Items<'a, '_> {
                #[allow(clippy::mut_from_ref)]
                unsafe fn item(&self, i: usize) -> &mut RoundItem<'a> {
                    &mut *self.0[i].get()
                }
            }
            let items = Items(&items);
            let env = &env;
            let run_one = |i: usize| {
                // SAFETY: see `Items`.
                let (p, a, r) = unsafe { items.item(i) };
                let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.run_round(env, a)));
                **r = Some(out.unwrap_or_else(|payload| {
                    Err(sched_error(
                        ExecErrorKind::Internal,
                        kernel,
                        WarpId(u32::MAX),
                        SiteId::NONE,
                        format!("panic in partition: {}", crate::interp::panic_message(&*payload)),
                    ))
                }));
            };
            match pool {
                Some(pool) => pool.par_for(n, &run_one),
                None => (0..n).for_each(run_one),
            }
        }
        // Partitions after the first failing one are discarded, as if never
        // run. The rest merge in partition order (the last writer of a byte
        // in partition order wins).
        let kept = (0..n).find(|&k| matches!(results[k], Some(Err(_)))).map_or(n, |e| e + 1);
        for k in kept..n {
            self.partitions[k].events.clear();
            self.partitions[k].ends.clear();
        }
        // Event replay follows observation: a partition that read global
        // bytes another one wrote this round saw the round-start value, so
        // its events go first (Arena::shard_replay_order). Without such
        // conflicts this is partition order. A cycle (each read what the
        // other wrote) has no faithful sequential stream: partition order is
        // used and an `incomplete` diagnostic says so.
        let order = if self.observing {
            match Arena::shard_replay_order(&shards[..kept]) {
                Ok(o) => o,
                Err((a, b)) => {
                    self.stream_cycle(a, b);
                    (0..kept).collect()
                }
            }
        } else {
            (0..kept).collect()
        };
        let mut ro_err = None;
        for (k, shard) in shards.into_iter().enumerate() {
            if k < kept {
                // Readonly-proxy conflicts between partitions of one round.
                if ro_err.is_none() {
                    if let Some((a, b)) = arena.readonly_merge_conflict(&shard) {
                        let m = support::readonly_conflict_message(&arena.get(a).name, a, b);
                        ro_err = Some(sched_error(ExecErrorKind::BadAddress, self.kernel_index, WarpId(u32::MAX), SiteId::NONE, m));
                    }
                }
                arena.merge_shard(shard);
            } else {
                arena.discard_shard(shard);
            }
        }
        // Declared-word history: each partition's new entries, in the order
        // its events are about to be delivered; buffered verdicts are
        // renumbered to the merged (delivery-order) history first (W6-P1).
        self.merge_words(&order);
        self.replay_partitions(&order, observer, pool);
        let mut progress = false;
        let mut first_err = None;
        for (k, result) in results.iter_mut().enumerate().take(kept) {
            self.absorb_state(k);
            match result.take().expect("ran") {
                Ok(p) => progress |= p,
                Err(e) => first_err = Some(e),
            }
        }
        match first_err.or(ro_err) {
            Some(e) => Err(e),
            None => Ok(progress),
        }
    }

    /// Record (once per launch) that a round's event stream could not be
    /// ordered consistently with what partitions `a` and `b` observed.
    fn stream_cycle(&mut self, a: usize, b: usize) {
        if self.stream_cycle_reported {
            return;
        }
        self.stream_cycle_reported = true;
        let cl = |k: usize| self.partitions[k].clusters.first().copied().unwrap_or(0);
        let mut attrs = BTreeMap::new();
        attrs.insert("reason".to_string(), serde_json::Value::from("cross_cluster_same_round_cycle"));
        attrs.insert("round".to_string(), serde_json::Value::from(self.round));
        self.diagnostics.push(Finding {
            kind: crate::report::FindingKind::Unsupported,
            status: crate::report::Status::Incomplete,
            message: format!(
                "clusters {} and {} each read global bytes the other wrote in round {} (a non-sequentially-consistent outcome); the observer stream cannot order them faithfully, so checker verdicts over this launch are incomplete",
                cl(a),
                cl(b),
                self.round
            ),
            attrs,
            sites: Vec::new(),
            evidence: Vec::new(),
        });
    }

    /// The serial phase of a round (main arena): parked global RMWs and
    /// deferred global reductions, in partition order.
    fn serial_phase(&mut self, arena: &mut Arena, observer: &mut dyn Observer, step_fn: WarpStepFn) -> Result<bool, ExecError> {
        let mut progress = false;
        for pi in 0..self.partitions.len() {
            let env = Env {
                program: self.program,
                loaded: &self.loaded,
                shape: &self.shape,
                config: &self.config,
                kernel: self.kernel_index,
                step_fn,
                round: self.round,
                observing: self.observing,
            };
            let r = self.partitions[pi].run_serial(&env, arena);
            self.absorb(pi, observer);
            // Serial work runs partition by partition on the main arena:
            // merge each partition's history before the next one runs, so a
            // later partition's verdicts see earlier entries (W6 S-b).
            self.merge_words(&[pi]);
            if let Err(e) = r {
                observer.phase_end(self.round);
                return Err(e);
            }
            progress |= r?;
        }
        if !self.partitions.is_empty() {
            observer.phase_end(self.round);
        }
        Ok(progress)
    }

    /// Merge the partitions' declared-word histories into the launch table
    /// (`order` = the order their events were just delivered) and give every
    /// partition the merged table back.
    ///
    /// Each partition ran the phase with a copy of the launch table, so its
    /// entries past the copy's length are new. They are appended rebased
    /// onto the launch table's last image (each entry keeps only the bytes
    /// it wrote). Indices stay equal to the delivery order: the replay
    /// order puts a partition that read bytes another one wrote this round
    /// before the writer, so no partition delivered earlier wrote a word a
    /// later one waited on (a read/write cycle is reported `incomplete`).
    /// History exists only for history-consuming observers; nothing
    /// program-visible depends on it.
    fn merge_words(&mut self, order: &[usize]) {
        use crate::interp::aux::{WordRegion, MAX_WORD_HISTORY};
        // Write counts (and overflow) are merged in every mode (W13-1);
        // post-images, positions and verdict renumbering only when a
        // history-consuming observer is attached (`images`).
        let images = self.wants_history;
        // Launch-wide log length of each region before this merge, recorded
        // when the merge first touches it (`None`: created by this merge).
        // Every partition's log of a region agrees with the launch-wide log
        // on that prefix; only the suffix past it is partition-local. The
        // table is therefore never cloned: the merge appends the partitions'
        // suffixes and the refresh rewrites only the suffixes of the
        // regions it touched (cost O(touched regions + new entries)).
        let mut touched: Vec<((AllocId, ByteSpan), Option<usize>)> = Vec::new();
        let mut touched_ix: HashMap<(AllocId, u64), usize> = HashMap::new();
        // Per partition (index into `order`): the merged log position of each
        // of its log entries, per region.
        let mut maps: Vec<PositionMap> = Vec::with_capacity(order.len());
        for &k in order {
            let local = &self.partitions[k].aux.words;
            let global = &mut self.launch_words;
            let mut map: PositionMap = HashMap::new();
            // Only regions this partition declared or logged since the last
            // merge can differ from the launch table (the others equal its
            // prefix: identity positions, see `hist_map`).
            for (alloc, span) in local.dirty() {
                let Some(r) = local.exact(alloc, span) else { continue };
                let alloc = &alloc;
                {
                    let start = match touched_ix.get(&(*alloc, r.span.start)) {
                        Some(&t) => touched[t].1,
                        None => global.exact(*alloc, r.span).map(|t| t.count),
                    };
                    let mut m: Vec<Option<usize>> = if images { (0..start.unwrap_or(0).min(r.log.len())).map(Some).collect() } else { Vec::new() };
                    if start == Some(r.count) && !r.overflow {
                        map.insert((*alloc, r.span.start), m);
                        continue;
                    }
                    touched_ix.entry((*alloc, r.span.start)).or_insert_with(|| {
                        touched.push(((*alloc, r.span), start));
                        touched.len() - 1
                    });
                    if global.position(*alloc, r.span).is_none() {
                        global.insert(*alloc, WordRegion { span: r.span, init: r.init.clone(), log: Vec::new(), count: 0, overflow: false });
                    }
                    let t = global.exact_mut(*alloc, r.span).expect("region just inserted");
                    if !images {
                        let add = r.count.saturating_sub(start.unwrap_or(0));
                        if t.count + add > MAX_WORD_HISTORY {
                            t.count = MAX_WORD_HISTORY;
                            t.overflow = true;
                        } else {
                            t.count += add;
                        }
                    }
                    for (spans, img) in r.log.iter().skip(start.unwrap_or(0)).take(if images { usize::MAX } else { 0 }) {
                        if t.count >= MAX_WORD_HISTORY {
                            t.overflow = true;
                            break;
                        }
                        m.push(Some(t.log.len()));
                        t.count += 1;
                        let mut merged = t.log.last().map(|e| e.1.clone()).unwrap_or_else(|| t.init.clone());
                        for sp in spans {
                            let lo = sp.start.max(r.span.start);
                            let hi = sp.end().min(r.span.end());
                            for x in lo..hi {
                                let i = (x - r.span.start) as usize;
                                merged[i] = img[i];
                            }
                        }
                        t.log.push((spans.clone(), merged));
                    }
                    t.overflow |= r.overflow;
                    map.insert((*alloc, r.span.start), m);
                }
            }
            maps.push(map);
        }
        for &k in order {
            self.partitions[k].aux.words.clear_dirty();
        }
        if touched.is_empty() {
            return;
        }
        if !images {
            // Counts only: every partition's copy of a touched region takes
            // the launch-wide count and overflow.
            let global = &self.launch_words;
            for p in &mut self.partitions {
                for &((alloc, span), _) in &touched {
                    let Some(g) = global.exact(alloc, span) else { continue };
                    match p.aux.words.exact_mut(alloc, span) {
                        Some(l) => {
                            l.count = g.count;
                            l.overflow = g.overflow;
                        }
                        None => {
                            p.aux.words.insert(alloc, g.clone());
                        }
                    }
                }
            }
            return;
        }
        // Renumber each partition's buffered verdicts and rebase its verdict
        // cache wherever the merged history interleaves other partitions'
        // entries before its own (reads the partition's pre-refresh table).
        for (oi, &k) in order.iter().enumerate() {
            let map = &maps[oi];
            let shifted = map.values().any(|m| m.iter().enumerate().any(|(i, g)| *g != Some(i)));
            if !shifted {
                continue;
            }
            let global = &self.launch_words;
            let Partition { aux, events, .. } = &mut self.partitions[k];
            let local = &aux.words;
            let hist_map = |alloc: AllocId, span: ByteSpan, h: u32| -> Option<u32> {
                if h == 0 {
                    return Some(0);
                }
                let lr = local.region(alloc, span)?;
                let gr = global.region(alloc, span)?;
                let Some(m) = map.get(&(alloc, lr.span.start)) else {
                    // A region this partition did not touch: its log is the
                    // launch log's prefix, positions unchanged.
                    return Some(h);
                };
                let lp = lr.log.iter().enumerate().filter(|(_, e)| e.0.iter().any(|s| s.overlaps(span))).nth(h as usize - 1)?.0;
                let gp = (*m.get(lp)?)?;
                Some(1 + gr.log[..gp].iter().filter(|e| e.0.iter().any(|s| s.overlaps(span))).count() as u32)
            };
            events.remap_verdicts(hist_map);
            // Cached evaluations at or past the first foreign entry are
            // dropped: the next poll evaluates the merged history there.
            let mut first_foreign: HashMap<(AllocId, u64), usize> = HashMap::new();
            for ((alloc, rs), m) in map {
                let mut images: Vec<usize> = m.iter().flatten().copied().collect();
                images.sort_unstable();
                let f = images.iter().enumerate().find(|(i, g)| *i != **g).map(|(i, _)| i).unwrap_or(images.len());
                first_foreign.insert((*alloc, *rs), f + 1);
            }
            let words = &aux.words;
            for ((_, _, alloc, start), cache) in aux.verdicts.iter_mut() {
                let Some(r) = words.region_at(*alloc, *start) else { continue };
                let Some(&f) = first_foreign.get(&(*alloc, r.span.start)) else { continue };
                for l in 0..32 {
                    if cache.evaluated[l] > f {
                        cache.evaluated[l] = f;
                    }
                    if let Some(b) = cache.bits.get_mut(l) {
                        for (wi, w) in b.iter_mut().enumerate() {
                            for bit in 0..64 {
                                if wi * 64 + bit >= f {
                                    *w &= !(1u64 << bit);
                                }
                            }
                        }
                    }
                }
            }
        }
        // Refresh: every partition's copy of each touched region becomes the
        // launch-wide one (shared prefix kept, suffix rewritten).
        let global = &self.launch_words;
        for p in &mut self.partitions {
            for &((alloc, span), start) in &touched {
                let Some(g) = global.exact(alloc, span) else { continue };
                match p.aux.words.exact_mut(alloc, span) {
                    Some(l) => {
                        let keep = start.unwrap_or(0).min(l.log.len());
                        if start.is_none() {
                            l.init.clone_from(&g.init);
                        }
                        l.log.truncate(keep);
                        l.log.extend_from_slice(&g.log[keep..]);
                        l.count = g.count;
                        l.overflow = g.overflow;
                    }
                    None => {
                        p.aux.words.insert(alloc, g.clone());
                    }
                }
            }
        }
    }

    /// Land / apply everything ready in every partition (main arena).
    fn drain_all(&mut self, arena: &mut Arena, observer: &mut dyn Observer, step_fn: WarpStepFn) -> Result<bool, ExecError> {
        let mut any = false;
        for pi in 0..self.partitions.len() {
            let env = Env {
                program: self.program,
                loaded: &self.loaded,
                shape: &self.shape,
                config: &self.config,
                kernel: self.kernel_index,
                step_fn,
                round: self.round,
                observing: self.observing,
            };
            let p = &mut self.partitions[pi];
            let r = p.land(None, &env, arena, true).and_then(|a| Ok(a | p.apply_completions(&env)?));
            self.absorb(pi, observer);
            self.merge_words(&[pi]);
            if let Err(e) = r {
                observer.phase_end(self.round);
                return Err(e);
            }
            any |= r?;
        }
        if !self.partitions.is_empty() {
            observer.phase_end(self.round);
        }
        Ok(any)
    }

    fn run_loop(
        &mut self,
        arena: &mut Arena,
        observer: &mut dyn Observer,
        step_fn: WarpStepFn,
        pool: Option<&pool::Pool>,
    ) -> Result<RunStatus, ExecError> {
        self.turnover(arena, observer)?;
        loop {
            if self.round >= self.config.max_rounds {
                let mut attrs = BTreeMap::new();
                attrs.insert("max_rounds".into(), serde_json::json!(self.config.max_rounds));
                return Ok(RunStatus::Incomplete { reason: format!("round budget of {} exhausted", self.config.max_rounds), site: None, attrs });
            }
            let mut progress = self.parallel_phase(arena, observer, step_fn, pool)?;
            progress |= self.serial_phase(arena, observer, step_fn)?;
            progress |= self.turnover(arena, observer)?;
            self.round += 1;
            self.stats.rounds = self.round;
            let all_done = self.pending.is_empty() && self.partitions.iter().all(|p| p.finished());
            if all_done {
                // Drain the async queues.
                while self.drain_all(arena, observer, step_fn)? {}
                return Ok(RunStatus::Completed);
            }
            if !progress {
                // Before declaring a deadlock, land everything that is ready.
                if self.drain_all(arena, observer, step_fn)? {
                    continue;
                }
                let mut blocked = Vec::new();
                let mut divergent = None;
                let mut after_exit = None;
                // Per partition: (warp, resource, site, lanes) of each
                // blocked warp, for the stuck-resource diagnosis below.
                let mut per_partition: Vec<Vec<(WarpId, ResourceId, SiteId, WarpMask)>> = vec![Vec::new(); self.partitions.len()];
                for (pi, p) in self.partitions.iter().enumerate() {
                    for c in &p.ctas {
                        for w in &c.warps {
                            if let WarpStatus::Blocked(r) = w.status {
                                blocked.push((w.id, r));
                                per_partition[pi].push((w.id, r, self.program.site_of(w.pc), w.active));
                                if divergent.is_none() && crate::interp::is_divergent(w) {
                                    divergent = Some((w.id, self.program.site_of(w.pc), w.active));
                                }
                                // G8: a named barrier with an explicit count may
                                // be waiting only on warps that exited.
                                if let ResourceId::Named { cta, .. } = r {
                                    if after_exit.is_none() && p.aux.cta_exited.get(&cta).is_some_and(|m| *m != 0) {
                                        after_exit = Some((w.id, self.program.site_of(w.pc), w.active));
                                    }
                                }
                            }
                        }
                    }
                }
                let at = |w: WarpId, lanes: WarpMask| {
                    let mut a = BTreeMap::new();
                    a.insert("warp".to_string(), serde_json::json!(w.0));
                    a.insert("lanes".to_string(), serde_json::json!(lanes.0));
                    a
                };
                if let Some((w, site, lanes)) = after_exit {
                    return Ok(RunStatus::Incomplete {
                        attrs: at(w, lanes),
                        reason: format!(
                            "named_barrier_after_exit (G8): no progress while warp {} waits on a named barrier of a CTA with exited warps; release of explicit-count barriers by exit is not modeled",
                            w.0
                        ),
                        site: Some(site),
                    });
                }
                // A divergent warp's lanes may be waiting on each other in a
                // way structured SIMT cannot interleave: not a proof.
                if let Some((w, site, lanes)) = divergent {
                    return Ok(RunStatus::Incomplete {
                        attrs: at(w, lanes),
                        reason: format!("divergent_block: no progress while warp {} is blocked with a divergent mask", w.0),
                        site: Some(site),
                    });
                }
                // A blocked resource that can provably never complete (sync
                // §1.5 / §2.9, e.g. an mbarrier whose arrivals are all in
                // with nothing in flight but fewer tx bytes than expected) is
                // a kernel error at the warp blocked on it, not a generic
                // deadlock (W6).
                for (pi, warps) in per_partition.iter().enumerate() {
                    let ids: Vec<ResourceId> = warps.iter().map(|w| w.1).collect();
                    if let Some((res, err)) = self.partitions[pi].sync.stuck(&ids).into_iter().next() {
                        let &(warp, _, site, lanes) = warps.iter().find(|w| w.1 == res).expect("stuck resource is blocked on");
                        let message = match &err {
                            crate::sync::SyncError::Mbarrier(crate::sync::mbarrier::Error::TxUnderDelivered { gen, expected, completed }) => {
                                format!("mbarrier transaction under-delivery: {completed} of {expected} bytes for phase {gen}")
                            }
                            other => format!("{other:?}"),
                        };
                        let mut e = sched_error(ExecErrorKind::Protocol(err), self.kernel_index, warp, site, message);
                        e.lanes = lanes;
                        return Ok(RunStatus::Error(e));
                    }
                }
                return Ok(RunStatus::Deadlock { blocked });
            }
        }
    }

    /// `SyncTable::quiescent` of every partition (retired and resident).
    pub fn leftovers(&self) -> Vec<(ResourceId, SyncError)> {
        let mut out = self.leftovers.clone();
        for p in &self.partitions {
            out.extend(p.sync.quiescent());
        }
        out.sort_by_cached_key(|(id, _)| format!("{id:?}"));
        out
    }

    /// Async ops never landed, over every partition.
    pub fn pending_async_ops(&self) -> usize {
        self.partitions.iter().map(|p| p.sync.async_ops.len()).sum()
    }

    /// End-of-launch events: blocked-at-exit, warp_done, AllocEnd, end_launch.
    fn finish(&mut self, arena: &mut Arena, observer: &mut dyn Observer, status: &RunStatus) {
        let end = match status {
            RunStatus::Deadlock { .. } => WarpEnd::Deadlocked,
            RunStatus::Incomplete { .. } => WarpEnd::Budget,
            _ => WarpEnd::Error,
        };
        let mut blocked_events = Vec::new();
        for c in self.partitions.iter().flat_map(|p| p.ctas.iter()) {
            for w in &c.warps {
                if self.ends.contains_key(&w.id) {
                    continue;
                }
                if let WarpStatus::Blocked(r) = w.status {
                    blocked_events.push(SyncEvent {
                        kernel: self.kernel_index,
                        actor: Actor::Warp { warp: w.id, epoch: w.epoch },
                        seq: w.sync_seq,
                        site: self.program.site_of(w.pc),
                        frames: w.loop_frames(self.program),
                        lanes: w.active,
                        kind: SyncKind::Protocol {
                            cmds: vec![crate::observe::ProtocolCmd {
                                res: r,
                                cmd: blocked_cmd(r),
                                counts: Default::default(),
                                observed_parity: None,
                            }],
                            collective: None,
                            issued: Vec::new(),
                            status: ProtocolStatus::BlockedAtExit,
                        },
                    });
                }
            }
        }
        if self.observing {
            for e in &blocked_events {
                observer.sync(e);
            }
        }
        let mut done = Vec::new();
        for c in self.partitions.iter().flat_map(|p| p.ctas.iter()) {
            for w in &c.warps {
                if !self.ends.contains_key(&w.id) {
                    done.push(w.id);
                }
            }
        }
        for w in done {
            self.ends.insert(w, end);
            observer.warp_done(w, end);
        }
        if self.observing {
            let mut ends = Vec::new();
            for c in self.partitions.iter().flat_map(|p| p.ctas.iter()) {
                ends.push(c.ctx.smem);
                ends.push(c.ctx.tmem);
                ends.extend(c.warps.iter().filter_map(|w| w.local));
                ends.extend(c.warps.iter().filter_map(|w| w.regbuf));
            }
            for a in ends {
                self.host_event(observer, SyncKind::AllocEnd { alloc: a });
            }
        }
        self.stats.instrs = self.retired_instrs + self.partitions.iter().map(|p| p.counters.instrs).sum::<u64>();
        self.stats.completions = self.retired_completions + self.partitions.iter().map(|p| p.completions).sum::<u64>();
        observer.end_launch(&LaunchInfo { program: self.program, kernel_index: self.kernel_index, shape: self.shape, arena });
    }
}

/// A representative command naming what a warp was blocked on.
fn blocked_cmd(r: ResourceId) -> SyncCmd {
    match r {
        ResourceId::Mbarrier { .. } => SyncCmd::Mbarrier(mbarrier::Cmd::WaitParity { parity: 0 }),
        ResourceId::Named { .. } => SyncCmd::Named(crate::sync::named::Cmd::Resume { gen: 0 }),
        ResourceId::Cluster { .. } => SyncCmd::Cluster(cluster::Cmd::Wait { warp: 0, mask: 0, aligned: true }),
        ResourceId::AsyncGroup { .. } => SyncCmd::AsyncGroup(async_group::Cmd::Wait { n: 0, read: false }),
        ResourceId::TcgenLifecycle { .. } => SyncCmd::Tcgen(crate::sync::tcgen::Cmd::Relinquish { who: crate::sync::tcgen::Who::One(0) }),
        ResourceId::RegPool { .. } => SyncCmd::RegPool(setmaxnreg::Cmd::Poll { wg: 0 }),
        _ => SyncCmd::TcgenGroup(0),
    }
}

/// Declared sync words of a `sync_words` buffer (W5-14 ruling): one word
/// per element of the polled view's dtype (`bits / 8` bytes, at least 1),
/// each with its own history; never one word over the whole buffer.
/// A dtype that is not a whole number of bytes, or a buffer that is not a
/// whole number of words, has no well-defined word: fail closed
/// (`incomplete`, coordinator ruling on W5-14).
/// A `sync_words` buffer's words as `(base, width, n)`; validated in every
/// mode (W13-3: a malformed span is `incomplete` with or without an
/// observer).
fn sync_word_array(b: &crate::program::BufferDecl, offset: u64, len: u64) -> Result<(u64, u64, u64), String> {
    let bits = b.dtype.bits() as u64;
    if bits == 0 || !bits.is_multiple_of(8) || !len.is_multiple_of(bits / 8) {
        return Err(format!(
            "sync_words buffer {}: {len} bytes are not a whole number of {}-bit words; declared-word verdicts are not modelled for sub-word sizes",
            b.name, bits
        ));
    }
    let w = bits / 8;
    Ok((offset, w, len / w))
}

fn write_param(arena: &mut Arena, params: AllocId, off: u64, bytes: &[u8]) {
    let v = support::whole(arena, params);
    let _ = arena.write(v, &[ByteSpan::new(off, bytes.len() as u64)], bytes);
}

/// Resolve a declared launch against inputs (`DimExpr` over scalar params).
pub fn resolve_launch(program: &Program, inputs: &Inputs) -> Result<LaunchShape, RunError> {
    let param = |p: crate::program::ParamId| -> Option<i64> { scalar_param(program, inputs, p) };
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

/// Value of a scalar-like parameter: a bound `Scalar`, or an
/// `ImplicitShape` derived from its buffer argument (1-D buffers: byte
/// length / element size; otherwise the shape must be bound by name).
fn scalar_param(program: &Program, inputs: &Inputs, p: crate::program::ParamId) -> Option<i64> {
    let slot = program.host_abi.get(p.0 as usize)?;
    match lookup(inputs, slot) {
        // Scalar bits are the parameter's raw bits: sign- or zero-extend by
        // its declared type (an int32 -7 arrives as 0xFFFF_FFF9).
        Some(ArgValue::Scalar(v)) => {
            let v = *v;
            return Some(match slot.dtype {
                Some(t) if t.elem.is_signed_int() && t.elem.bits() < 64 => {
                    let sh = 64 - t.elem.bits();
                    ((v << sh) as i64) >> sh
                }
                Some(t) if t.elem.bits() < 64 => (v & ((1u64 << t.elem.bits()) - 1)) as i64,
                _ => v as i64,
            });
        }
        Some(_) => return None,
        None => {}
    }
    let ParamKind::ImplicitShape { buffer, axis } = slot.kind else { return None };
    let b = program.host_abi.get(buffer.0 as usize)?;
    let nbytes = match lookup(inputs, b) {
        Some(ArgValue::Buffer { bytes, .. }) => bytes.len(),
        Some(ArgValue::View { len, .. }) => *len as usize,
        _ => return None,
    };
    let elem = b.dtype.map(|t| t.mem_bytes() as usize).unwrap_or(1).max(1);
    if b.shape.len() <= 1 && axis == 0 {
        return Some((nbytes / elem) as i64);
    }
    // Multi-dimensional: the other extents must be constants.
    let mut other = 1i64;
    for (i, d) in b.shape.iter().enumerate() {
        if i == axis as usize {
            continue;
        }
        other = other.checked_mul(d.eval(&|_| None)?)?;
    }
    if other <= 0 {
        return None;
    }
    Some((nbytes / elem) as i64 / other)
}

/// Argument bound to a parameter slot (by name, then aliases).
fn lookup<'a>(inputs: &'a Inputs, slot: &crate::program::ParamSlot) -> Option<&'a ArgValue> {
    inputs.args.get(&slot.name).or_else(|| slot.aliases.iter().find_map(|a| inputs.args.get(a)))
}

/// Run every kernel of `module` in order with default config and `seed`.
pub fn run(
    module: &Module,
    inputs: &Inputs,
    observer: &mut dyn Observer,
    seed: u64,
) -> Result<RunOutcome, RunError> {
    let config = RunConfig { seed, ..RunConfig::default() };
    run_with_config(module, inputs, observer, &config)
}

/// Allocate (once per module run) the global allocation of buffer argument `name`.
/// The allocation and byte range argument `name` names: a `Buffer` (its
/// whole allocation) or a `View` into one.
fn host_region(
    arena: &mut Arena,
    globals: &mut BTreeMap<String, AllocId>,
    inputs: &Inputs,
    name: &str,
) -> Result<(AllocId, u64, u64), RunError> {
    if let Some(ArgValue::View { target, offset, len }) = inputs.args.get(name) {
        if matches!(inputs.args.get(target), Some(ArgValue::View { .. })) {
            return Err(RunError::BadArg { name: name.into(), message: format!("view target {target} must be a buffer, not a view") });
        }
        let a = host_buffer(arena, globals, inputs, target)?;
        let size = arena.get(a).size;
        if offset.checked_add(*len).is_none_or(|e| e > size) {
            return Err(RunError::BadArg {
                name: name.into(),
                message: format!("view [{offset}, {offset}+{len}) exceeds {target} ({size} bytes)"),
            });
        }
        return Ok((a, *offset, *len));
    }
    let a = host_buffer(arena, globals, inputs, name)?;
    Ok((a, 0, arena.get(a).size))
}

fn host_buffer(
    arena: &mut Arena,
    globals: &mut BTreeMap<String, AllocId>,
    inputs: &Inputs,
    name: &str,
) -> Result<AllocId, RunError> {
    if let Some(&a) = globals.get(name) {
        return Ok(a);
    }
    match inputs.args.get(name) {
        Some(ArgValue::Buffer { bytes, valid }) => {
            let size = bytes.len() as u64;
            let init = match valid {
                Some(v) => {
                    if v.len() != size {
                        return Err(RunError::BadArg { name: name.into(), message: "validity length differs from bytes".into() });
                    }
                    Init::BytesWithValidity(bytes.clone(), v.clone())
                }
                None => Init::Bytes(bytes.clone()),
            };
            // V2C-35 ruling: a top-level buffer is a fresh device
            // allocation (aligned like cudaMalloc); the host pointer's bits
            // are not copied. Sub-views keep their offset in the region.
            let a = arena.alloc(Space::Global, Owner::Launch, name, size, init);
            globals.insert(name.to_string(), a);
            Ok(a)
        }
        Some(_) => Err(RunError::BadArg { name: name.into(), message: "expected a buffer argument".into() }),
        None => Err(RunError::MissingArg(name.into())),
    }
}

/// Allocate every host buffer `module` binds, in one fixed order: per
/// kernel, its `Buffer` slots (in signature order), then the targets of
/// its `Pointer` / `TensorMapOf` arguments. `run_with_config` and
/// [`plan_global_addresses`] share it, so planned addresses are the ones a
/// run uses.
fn allocate_host(
    module: &Module,
    inputs: &Inputs,
    arena: &mut Arena,
    globals: &mut BTreeMap<String, AllocId>,
    views: &mut BTreeMap<String, (u64, u64)>,
) -> Result<(), RunError> {
    for program in &module.kernels {
        for slot in &program.host_abi {
            if slot.kind == ParamKind::Buffer {
                let name = if inputs.args.contains_key(&slot.name) {
                    slot.name.clone()
                } else {
                    slot.aliases.iter().find(|a| inputs.args.contains_key(*a)).cloned().unwrap_or(slot.name.clone())
                };
                let (a, off, len) = host_region(arena, globals, inputs, &name)?;
                globals.entry(slot.name.clone()).or_insert(a);
                if matches!(inputs.args.get(&name), Some(ArgValue::View { .. })) {
                    views.insert(slot.name.clone(), (off, len));
                }
            }
        }
        for slot in &program.host_abi {
            match (slot.kind, lookup(inputs, slot)) {
                (ParamKind::Pointer, Some(ArgValue::Pointer { target, .. })) => {
                    host_region(arena, globals, inputs, target)?;
                }
                (ParamKind::Pointer, Some(ArgValue::View { target, .. })) => {
                    host_buffer(arena, globals, inputs, target)?;
                }
                (ParamKind::Pointer, Some(ArgValue::Buffer { .. })) => {
                    host_buffer(arena, globals, inputs, &slot.name)?;
                }
                (ParamKind::TensorMap, Some(ArgValue::TensorMapOf { base, .. })) => {
                    host_region(arena, globals, inputs, base)?;
                }
                _ => {}
            }
        }
    }
    // W8-8: every buffer argument is placed, also those no parameter
    // references (e.g. the base array a descriptor image bound to a plain
    // buffer parameter points at), after the referenced ones (name order).
    for (name, arg) in &inputs.args {
        match arg {
            ArgValue::Buffer { .. } => {
                host_buffer(arena, globals, inputs, name)?;
            }
            ArgValue::View { .. } => {
                host_region(arena, globals, inputs, name)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// The engine (synthetic global) address every `Buffer` / `View` argument
/// of `inputs` will have when `module` runs with them (CONTRACT_REQUESTS
/// W8-7), for raw-pointer inputs that must embed them. Arguments the module
/// never binds are absent.
pub fn plan_global_addresses(module: &Module, inputs: &Inputs) -> Result<BTreeMap<String, u64>, RunError> {
    let mut arena = Arena::new(ValidityPolicy::Allow);
    let mut globals = BTreeMap::new();
    let mut views = BTreeMap::new();
    for program in &module.kernels {
        program.validate().map_err(|e| RunError::InvalidProgram(e.to_string()))?;
    }
    allocate_host(module, inputs, &mut arena, &mut globals, &mut views)?;
    let mut out = BTreeMap::new();
    for (name, arg) in &inputs.args {
        match arg {
            ArgValue::Buffer { .. } => {
                if let Some(&a) = globals.get(name) {
                    out.insert(name.clone(), arena.get(a).base);
                }
            }
            ArgValue::View { target, offset, .. } => {
                if let Some(&a) = globals.get(target) {
                    out.insert(name.clone(), arena.get(a).base + offset);
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// Largest `tcgen05.alloc.exclusive` column count on `arch` (PTX Table 58).
pub fn exclusive_tmem_columns(arch: Option<&str>) -> u32 {
    match arch {
        Some(a) if a.starts_with("sm_107") => 576,
        _ => crate::sync::tcgen::TMEM_COLUMNS,
    }
}

/// Encode a tensor map from its host-prelude spec over `va`.
fn encode_spec(spec: &crate::program::TensorMapSpec, va: u64, scalar: &dyn Fn(crate::program::ParamId) -> Option<i64>) -> Result<Vec<u8>, String> {
    let mut d = crate::oplib::TensorMapDesc::default();
    let ev = |e: &crate::program::DimExpr| e.eval(scalar).ok_or_else(|| "cannot evaluate tensor-map extent".to_string());
    d.global_address = va.wrapping_add(ev(&spec.base_offset)? as u64);
    d.rank = spec.rank;
    d.elem = Some(spec.dtype);
    // The host's raw CUtensorMapDataType, when it says more than `dtype`.
    use crate::dtype::Dtype as D;
    let canonical: Option<u8> = match spec.dtype {
        D::U8 => Some(0),
        D::U16 => Some(1),
        D::U32 => Some(2),
        D::S32 => Some(3),
        D::U64 => Some(4),
        D::S64 => Some(5),
        D::F16 => Some(6),
        D::F32 => Some(7),
        D::F64 => Some(8),
        D::BF16 => Some(9),
        _ => None,
    };
    match spec.force_cu_dtype {
        None => {}
        Some(c) if Some(c) == canonical => {}
        Some(11) => d.elem = Some(D::TF32),
        Some(13) => d.elem = Some(D::E2M1),
        Some(14) => {
            d.elem = Some(D::E2M1);
            d.fp4_padded = true;
        }
        Some(15) => d.elem = Some(D::U6),
        Some(c) => return Err(format!("CUtensorMapDataType {c} for a {:?} tensor map is not modeled", spec.dtype)),
    }
    for (i, e) in spec.global_dim.iter().enumerate().take(5) {
        d.global_dim[i] = ev(e)? as u64;
    }
    for (i, e) in spec.global_stride.iter().enumerate().take(5) {
        d.global_stride[i] = ev(e)? as u64;
    }
    let ev32 = |e: &crate::program::DimExpr, what: &str| -> Result<u32, String> {
        let v = ev(e)?;
        u32::try_from(v).map_err(|_| format!("tensor-map {what} {v} out of range"))
    };
    for (i, b) in spec.box_dim.iter().enumerate().take(5) {
        d.box_dim[i] = ev32(b, "box dim")?;
    }
    for (i, b) in spec.element_stride.iter().enumerate().take(5) {
        d.element_stride[i] = ev32(b, "element stride")?;
    }
    d.interleave = spec.interleave;
    d.swizzle = spec.swizzle;
    d.l2_promotion = spec.l2_promotion;
    d.oob_fill = spec.oob_fill;
    d.try_encode().map(|b| b.to_vec()).map_err(|e| format!("host tensor-map encode failed: {}", e.message))
}

/// Run every kernel of `module` in order. Host buffers are allocated once
/// (shared by name across kernels), each kernel gets a fresh `SyncTable`
/// and CTA-private allocations; outputs are read back at the end.
pub fn run_with_config(
    module: &Module,
    inputs: &Inputs,
    observer: &mut dyn Observer,
    config: &RunConfig,
) -> Result<RunOutcome, RunError> {
    let mut arena = Arena::new(config.validity);
    let mut globals: BTreeMap<String, AllocId> = BTreeMap::new();
    // Buffer slots bound to an `ArgValue::View`: (offset, len) in the
    // target's allocation.
    let mut views: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut outcome = RunOutcome {
        status: RunStatus::Completed,
        failed_kernel: None,
        outputs: Outputs::default(),
        stats: RunStats::default(),
        sync_leftovers: Vec::new(),
        diagnostics: Vec::new(),
        subset: config.subset.clone(),
    };
    // Validate and bind everything before running anything.
    let mut prepared = Vec::with_capacity(module.kernels.len());
    for program in &module.kernels {
        program.validate().map_err(|e| RunError::InvalidProgram(e.to_string()))?;
        prepared.push(resolve_launch(program, inputs)?);
    }
    allocate_host(module, inputs, &mut arena, &mut globals, &mut views)?;
    for (k, program) in module.kernels.iter().enumerate() {
        let shape = prepared[k];
        let loaded = Loaded::new(program);
        let params = arena.alloc(Space::Param, Owner::Launch, &format!("params[k{k}]"), loaded.param_bytes, Init::Zeroed);
        let scalar = |p: crate::program::ParamId| -> Option<i64> { scalar_param(program, inputs, p) };
        for (i, slot) in program.host_abi.iter().enumerate() {
            let off = loaded.param_offsets[i];
            let arg = lookup(inputs, slot);
            let bytes: Vec<u8> = match (slot.kind, arg) {
                (ParamKind::Buffer, _) => {
                    let off = views.get(&slot.name).map_or(0, |v| v.0);
                    (arena.get(globals[&slot.name]).base + off).to_le_bytes().to_vec()
                }
                (ParamKind::Scalar, Some(ArgValue::Scalar(v))) => v.to_le_bytes().to_vec(),
                (ParamKind::Scalar, None) => return Err(RunError::MissingArg(slot.name.clone())),
                (ParamKind::ImplicitShape { .. }, _) => match scalar_param(program, inputs, crate::program::ParamId(i as u32)) {
                    Some(v) => v.to_le_bytes().to_vec(),
                    None => {
                        return Err(RunError::BadArg {
                            name: slot.name.clone(),
                            message: "cannot derive the implicit shape from the bound buffer".into(),
                        })
                    }
                },
                (ParamKind::Pointer, Some(ArgValue::Pointer { target, offset })) => {
                    let (a, base_off, _) = host_region(&mut arena, &mut globals, inputs, target)?;
                    (arena.get(a).base + base_off + offset).to_le_bytes().to_vec()
                }
                (ParamKind::Pointer, Some(ArgValue::View { target, offset, .. })) => {
                    let a = host_buffer(&mut arena, &mut globals, inputs, target)?;
                    (arena.get(a).base + offset).to_le_bytes().to_vec()
                }
                (ParamKind::Pointer, Some(ArgValue::Buffer { .. })) => {
                    let a = host_buffer(&mut arena, &mut globals, inputs, &slot.name)?;
                    arena.get(a).base.to_le_bytes().to_vec()
                }
                (ParamKind::Pointer, Some(ArgValue::Scalar(v))) => v.to_le_bytes().to_vec(),
                (ParamKind::Pointer, None) => return Err(RunError::MissingArg(slot.name.clone())),
                (ParamKind::TensorMap, Some(ArgValue::TensorMap(b))) => {
                    if b.len() != 128 {
                        return Err(RunError::BadArg { name: slot.name.clone(), message: "tensor map must be 128 bytes".into() });
                    }
                    b.clone()
                }
                (ParamKind::TensorMap, Some(ArgValue::TensorMapOf { base, offset, desc })) => {
                    let (a, base_off, _) = host_region(&mut arena, &mut globals, inputs, base)?;
                    let mut d = desc.clone();
                    d.global_address = arena.get(a).base + base_off + offset;
                    d.encode().to_vec()
                }
                (ParamKind::TensorMap, None) => {
                    let (Some(spec), Some(base)) = (&slot.tensor_map, slot.implicit_base) else {
                        return Err(RunError::MissingArg(slot.name.clone()));
                    };
                    let bname = &program.host_abi[base.0 as usize].name;
                    let a = *globals.get(bname).ok_or_else(|| RunError::MissingArg(bname.clone()))?;
                    let va = arena.get(a).base + views.get(bname).map_or(0, |v| v.0);
                    encode_spec(spec, va, &scalar).map_err(|m| RunError::BadArg { name: slot.name.clone(), message: m })?
                }
                (_, Some(_)) => {
                    return Err(RunError::BadArg { name: slot.name.clone(), message: format!("argument does not match a {:?} parameter", slot.kind) })
                }
            };
            write_param(&mut arena, params, off, &bytes);
        }
        let view_lens: BTreeMap<String, u64> = views.iter().map(|(k, v)| (k.clone(), v.1)).collect();
        let mut sched = Scheduler::with_params(program, k as u32, shape, &mut arena, &globals, &view_lens, params, loaded, config.clone())?;
        // Readonly-proxy contract, for kernels with `ld.global.nc` loads.
        let nc_loads = program.code.iter().any(|i| matches!(i, Instr::Load { mods, .. } | Instr::LoadAddr { mods, .. } if mods.nc));
        arena.set_readonly_tracking(program.requirements.readonly_proxy || nc_loads);
        let status = sched.run(&mut arena, observer);
        arena.set_readonly_tracking(false);
        outcome.stats.instrs += sched.stats.instrs;
        outcome.stats.rounds += sched.stats.rounds;
        outcome.stats.completions += sched.stats.completions;
        outcome.diagnostics.append(&mut sched.diagnostics);
        if status == RunStatus::Completed {
            let left = sched.leftovers();
            // A kernel that exits with live TMEM allocations is a protocol
            // error (sync-semantics tcgen05 lifecycle; legacy rejected it).
            if let Some((res, e)) = left.iter().find(|(_, e)| matches!(e, SyncError::Tcgen(crate::sync::tcgen::Error::LiveAllocationsAtExit { .. }))) {
                let err = sched_error(
                    ExecErrorKind::Protocol(e.clone()),
                    k as u32,
                    WarpId(u32::MAX),
                    SiteId::NONE,
                    format!("kernel exited with live TMEM allocations ({res:?})"),
                );
                outcome.sync_leftovers.extend(left);
                outcome.status = RunStatus::Error(err);
                outcome.failed_kernel = Some(k as u32);
                break;
            }
            outcome.sync_leftovers.extend(left);
            if sched.pending_async_ops() != 0 {
                let e = sched_error(
                    ExecErrorKind::Internal,
                    k as u32,
                    WarpId(u32::MAX),
                    SiteId::NONE,
                    format!("{} async operations never became ready", sched.pending_async_ops()),
                );
                outcome.status = RunStatus::Error(e);
                outcome.failed_kernel = Some(k as u32);
                break;
            }
        } else {
            outcome.status = status;
            outcome.failed_kernel = Some(k as u32);
            break;
        }
    }
    for (name, arg) in &inputs.args {
        match arg {
            ArgValue::Buffer { .. } => {
                if let Some(&a) = globals.get(name) {
                    let al = arena.get(a);
                    outcome.outputs.buffers.insert(name.clone(), (al.bytes.clone(), al.valid.clone()));
                }
            }
            ArgValue::View { target, offset, len } => {
                if let Some(&a) = globals.get(target) {
                    let al = arena.get(a);
                    let (lo, hi) = (*offset as usize, (*offset + *len) as usize);
                    let valid = al.valid.slice(*offset, *len);
                    outcome.outputs.buffers.insert(name.clone(), (al.bytes[lo..hi].to_vec(), valid));
                }
            }
            _ => {}
        }
    }
    Ok(outcome)
}

/// Convenience: CtaId of global warp `w`.
pub fn cta_of(shape: &LaunchShape, w: WarpId) -> CtaId {
    CtaId(w.0 / shape.warps_per_cta().max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_coords_roundtrip() {
        let shape = LaunchShape { grid: [4, 2, 2], cluster: [2, 1, 2], block: [32, 1, 1], smem_bytes: 0 };
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..4 {
                    let (id, rank) = cluster_of(&shape, [x, y, z]);
                    assert_eq!(cta_coords(&shape, id, rank), [x, y, z]);
                }
            }
        }
    }
}
