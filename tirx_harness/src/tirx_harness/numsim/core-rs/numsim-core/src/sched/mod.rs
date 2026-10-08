//! Scheduler: CTA-lockstep execution, seeded warp rotation, per-CTA inbox,
//! completion firing, deadlock and budget detection.
//!
//! # Contract
//!
//! * A *round* visits every resident CTA; within a CTA every runnable warp
//!   gets one slice of at most `RunConfig::quantum` instructions, starting
//!   at a seeded rotation offset (`seed`, round, CTA), in ascending warp
//!   order from there. Blocked warps are retried every round (a blocking
//!   handler re-checks its resource and returns `Blocked` again).
//! * After a CTA's slices, in-flight `AsyncOp`s issued by that CTA land
//!   (`CompletionPolicy::Eager`: all ready ones; `Seeded`: a seeded subset;
//!   `after` dependencies respected) and enabled sync `Completion`s are
//!   applied (`SyncTable::enabled`/`apply_completion`).
//! * Cross-CTA effects produced during a round (`ExecCtx::outbox`: remote
//!   shared stores, remote mbarrier arrive / expect_tx / complete_tx) are
//!   delivered at the target CTA's next inbox drain (start of its turn in
//!   the next round); each drain emits `Observer::inbox_drain`.
//! * Residency: clusters are admitted in linear cluster order while at most
//!   `RunConfig::max_resident_ctas` CTAs are resident (all of them for
//!   cooperative launches and programs using `grid.sync`); a cluster's
//!   CTAs are always co-resident. A retired cluster's shared/TMEM/local
//!   allocations are reset and reused (`AllocEnd` + `AllocBegin`).
//! * Termination: all warps exited, no cluster left to admit and the async
//!   queues drained -> Completed. A round in which no warp made progress,
//!   no async op landed, no completion applied, no inbox message was
//!   delivered and no cluster was admitted (after force-landing every ready
//!   async op) -> Deadlock (with the blocked resources). Round budget ->
//!   Incomplete. Any `ExecError` -> Error, except Budget / Unsupported /
//!   `Op(Unsupported)` -> Incomplete.
//! * Fully deterministic for a fixed (module, inputs, config, seed).
//!
//! # Progress
//!
//! A slice made progress if it ended other than `Blocked` or completed an
//! instruction with `Instr::is_progress` (a write, a committed sync
//! transition, an async issue). A spin loop
//! whose iteration consists of failed polls is parked by `LoopEnd`
//! (`Blocked` at the same pc, no progress instruction), so a launch whose
//! only activity is such spinning is a deadlock, not a budget overrun.
//!
//! # CTA parallelism (design; `RunConfig::workers > 1`)
//!
//! Not implemented yet: `workers > 1` currently runs the single-threaded
//! path (results are identical by construction, since the parallel design
//! must reproduce this schedule). The design:
//!
//! * **Unit of ownership = cluster.** A worker thread owns a cluster for a
//!   round: its CTAs' `WarpState`s, shared windows, TMEM, local and
//!   register allocations, and the cluster's `SyncTable` partition (every
//!   `ResourceId` except `Grid` and global `Word`s is cluster-local:
//!   mbarriers, named barriers, cluster barrier, async groups, tcgen,
//!   reg pool). DSMEM and remote-mbarrier effects stay inside the worker.
//!   This needs the `Arena` split into a shared global arena plus one
//!   private arena per cluster (`AllocId` high bits = arena shard), and
//!   `ExecCtx` holding `&mut` to the private shard and `&` to the global
//!   one.
//! * **Global memory** within a round: each worker reads a snapshot of the
//!   global arena taken at the round start overlaid with its own writes
//!   (write log per worker, keyed by stripe of 4 KiB); atomics and
//!   `wait_until` on global memory are executed against the snapshot and
//!   *re-validated* at the merge: if two clusters touched the same stripe
//!   with at least one write in the round, the round is re-executed
//!   sequentially for those clusters (deterministic fallback). At the round
//!   barrier the write logs are merged in cluster order, which is exactly
//!   the single-threaded order (cluster i's round slice precedes cluster
//!   i+1's), and `inbox_drain` is the cross-CTA acquire point checkers
//!   already use.
//! * **Observer stream**: each worker buffers its events per cluster; the
//!   coordinator replays them in cluster order after the merge, so the
//!   observer sees the single-threaded order (contract: observers never
//!   change behaviour, so buffering is invisible).
//! * **Grid barrier / cooperative launches** stay single-threaded.

mod partition;
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
use std::collections::{BTreeMap, VecDeque};
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
    /// Worker threads (CTA/cluster parallelism). `<= 1` = single-threaded;
    /// see the module docs for the parallel design (not yet implemented;
    /// larger values currently run single-threaded with identical results).
    pub workers: usize,
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
    ends: BTreeMap<WarpId, WarpEnd>,
    observing: bool,
    wants_history: bool,
    /// One partition for every cluster (grid-wide state); see `single_partition`.
    single: bool,
    /// Next `Access::seq` (assigned at replay).
    next_seq: u64,
    /// Exit-check violations of retired partitions.
    leftovers: Vec<(ResourceId, SyncError)>,
    /// Review diagnostics (uninitialized reads), in partition order.
    pub diagnostics: Vec<Finding>,
    /// Instructions of retired partitions.
    retired_instrs: u64,
    retired_completions: u64,
    /// Declared words seeded into new partitions (launch-scope regions).
    launch_words: crate::interp::aux::WordTable,
}

/// Linear cluster id and rank of a CTA (ctaid coordinates).
#[allow(dead_code)]
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
        RunStatus::Incomplete { reason: format!("{:?}: {}", e.kind, e.message), site: Some(e.site) }
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
        Self::with_params(program, kernel_index, shape, arena, globals, params, loaded, config)
    }

    fn with_params(
        program: &'p Program,
        kernel_index: u32,
        shape: LaunchShape,
        arena: &mut Arena,
        globals: &BTreeMap<String, AllocId>,
        params: AllocId,
        loaded: Loaded,
        config: RunConfig,
    ) -> Result<Scheduler<'p>, RunError> {
        let invalid = |m: String| RunError::InvalidProgram(m);
        let cl = shape.cluster.map(|x| x.max(1));
        for d in 0..3 {
            if shape.grid[d] % cl[d] != 0 {
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
        fn bind(
            i: usize,
            program: &Program,
            arena: &mut Arena,
            globals: &BTreeMap<String, AllocId>,
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
                let pb = bind(parent.0 as usize, program, arena, globals, loaded, params, local_offsets, out, depth + 1, scalar)?;
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
                                            let avail = size.saturating_sub(off);
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
                    Space::Reg => BufBinding::Unbound,
                }
            };
            out[i] = Some(b);
            Ok(b)
        }
        for i in 0..n {
            bind(i, program, arena, globals, &loaded, params, &local_offsets, &mut bindings, 0, &scalar)?;
        }
        let bindings: Vec<BufBinding> = bindings.into_iter().map(|b| b.unwrap_or(BufBinding::Unbound)).collect();

        let mut pending: VecDeque<u32> = (0..shape.num_clusters()).collect();
        if let Some(s) = &config.subset {
            pending.retain(|c| s.contains(c));
        }
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
            ends: BTreeMap::new(),
            observing: false,
            wants_history: false,
            single: false,
            next_seq: 0,
            leftovers: Vec::new(),
            diagnostics: Vec::new(),
            retired_instrs: 0,
            retired_completions: 0,
            launch_words: Default::default(),
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
        self.all_resident()
            || groups.len() > 1
            || p.buffers.iter().any(|b| b.sync_words)
            || p.code.iter().any(|i| matches!(i, Instr::WaitUntil { .. }))
    }

    fn new_partition(&self, cluster: u32) -> Partition {
        let wpc = self.shape.warps_per_cta();
        let init = ResourceInit { policy: Policy::Numeric, cluster_warps: self.shape.ctas_per_cluster() * wpc, warps_per_cta: wpc };
        let mut aux = LaunchAux { kernel: self.kernel_index, wants_history: self.wants_history, ..LaunchAux::default() };
        aux.words = self.launch_words.clone();
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
            let part = &mut self.partitions[pi];
            part.aux.owner_cta.insert(smem, cid);
            part.aux.owner_cta.insert(tmem, cid);
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
                let mut ws = WarpState::new(wid, cid, w, nslots, shape.warp_lanes(w));
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
            // Declared words in the shared window.
            if self.wants_history {
                for (i, b) in self.program.buffers.iter().enumerate() {
                    if let (true, BufBinding::SharedWindow { offset, len }) = (b.sync_words, self.bindings[i]) {
                        let span = ByteSpan::new(offset as u64, len);
                        self.partitions[pi].aux.words.declare(arena, ctx.smem, span);
                        self.host_event(observer, SyncKind::DeclareWord { alloc: ctx.smem, span });
                    }
                }
            }
            let part = &mut self.partitions[pi];
            if self.program.topology.regs_per_thread != 0 {
                let res = ResourceId::RegPool { cta: cid };
                let cmd = SyncCmd::RegPool(setmaxnreg::Cmd::Configure { count: self.program.topology.regs_per_thread });
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
            part.ctas.push(CtaState { ctx, warps, inbox: Inbox::default(), buffers: self.bindings.clone() });
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
                    let done = part.ctas[i..j].iter().all(|c| c.finished() && c.inbox.msgs.is_empty());
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
        while let Some(&id) = self.pending.front() {
            let resident: u32 = self.partitions.iter().map(|p| p.ctas.len() as u32).sum();
            if resident > 0 && resident + per > cap {
                break;
            }
            self.pending.pop_front();
            self.admit(id, arena, observer)?;
            changed = true;
        }
        Ok(changed)
    }

    /// Run the launch to completion / deadlock / error / budget.
    pub fn run(&mut self, arena: &mut Arena, observer: &mut dyn Observer, backend: &Backend) -> RunStatus {
        let step_fn: WarpStepFn = match backend {
            Backend::Interp => step_warp,
            Backend::Codegen(v) => match v.get(self.kernel_index as usize) {
                Some(f) => *f,
                None => {
                    return RunStatus::Error(sched_error(
                        ExecErrorKind::Internal,
                        self.kernel_index,
                        WarpId(0),
                        SiteId::NONE,
                        format!("codegen backend has no step function for kernel {}", self.kernel_index),
                    ))
                }
            },
        };
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
        if self.wants_history {
            for (i, b) in self.program.buffers.iter().enumerate() {
                if let (true, BufBinding::View(v)) = (b.sync_words, self.bindings[i]) {
                    let span = ByteSpan::new(v.offset, v.len);
                    self.launch_words.declare(arena, v.alloc, span);
                    self.host_event(observer, SyncKind::DeclareWord { alloc: v.alloc, span });
                }
            }
        }
        let workers = self.config.workers.max(1);
        let result = if workers > 1 && !self.single {
            // A launch-lifetime pool; the calling thread is one of the workers.
            let pool = pool::Pool::new(workers - 1);
            std::thread::scope(|scope| {
                for _ in 0..workers - 1 {
                    scope.spawn(|| pool.worker());
                }
                let r = self.run_loop(arena, observer, step_fn, Some(&pool));
                pool.shutdown();
                r
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

    /// Replay partition `pi`'s buffered events and absorb its round results.
    fn absorb(&mut self, pi: usize, observer: &mut dyn Observer) {
        let p = &mut self.partitions[pi];
        p.events.replay(observer, &mut self.next_seq);
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
            self.absorb(0, observer);
            return r;
        }
        let mut shards: Vec<Arena> = Vec::with_capacity(n);
        for p in &self.partitions {
            // The ownership list only feeds a debug assertion.
            let private = if cfg!(debug_assertions) { p.private_allocs() } else { Vec::new() };
            shards.push(arena.make_shard(&private));
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
            let items: Vec<std::cell::UnsafeCell<(&mut Partition, &mut Arena, &mut Option<Result<bool, ExecError>>)>> = self
                .partitions
                .iter_mut()
                .zip(shards.iter_mut())
                .zip(results.iter_mut())
                .map(|((p, a), r)| std::cell::UnsafeCell::new((p, a, r)))
                .collect();
            struct Items<'a, 'b>(&'b [std::cell::UnsafeCell<(&'a mut Partition, &'a mut Arena, &'a mut Option<Result<bool, ExecError>>)>]);
            // SAFETY: `par_for` hands every index to exactly one thread.
            unsafe impl Sync for Items<'_, '_> {}
            impl<'a> Items<'a, '_> {
                #[allow(clippy::mut_from_ref)]
                unsafe fn item(&self, i: usize) -> &mut (&'a mut Partition, &'a mut Arena, &'a mut Option<Result<bool, ExecError>>) {
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
                        format!("panic in partition: {}", crate::codegen::rt::panic_message(&*payload)),
                    ))
                }));
            };
            match pool {
                Some(pool) => pool.par_for(n, &run_one),
                None => (0..n).for_each(run_one),
            }
        }
        // Merge in partition order; stop at the first error (later
        // partitions' effects are discarded, as if never run).
        let mut progress = false;
        let mut first_err = None;
        for (k, shard) in shards.into_iter().enumerate() {
            if first_err.is_some() {
                arena.discard_shard(shard);
                self.partitions[k].events.clear();
                self.partitions[k].ends.clear();
                continue;
            }
            arena.merge_shard(shard);
            self.absorb(k, observer);
            match results[k].take().expect("ran") {
                Ok(p) => progress |= p,
                Err(e) => first_err = Some(e),
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(progress),
        }
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
            progress |= r?;
        }
        Ok(progress)
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
            any |= r?;
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
                return Ok(RunStatus::Incomplete { reason: format!("round budget of {} exhausted", self.config.max_rounds), site: None });
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
                for c in self.partitions.iter().flat_map(|p| p.ctas.iter()) {
                    for w in &c.warps {
                        if let WarpStatus::Blocked(r) = w.status {
                            blocked.push((w.id, r));
                            if divergent.is_none() && crate::interp::is_divergent(w) {
                                divergent = Some((w.id, self.program.site_of(w.pc)));
                            }
                        }
                    }
                }
                // A divergent warp's lanes may be waiting on each other in a
                // way structured SIMT cannot interleave: not a proof.
                if let Some((w, site)) = divergent {
                    return Ok(RunStatus::Incomplete {
                        reason: format!("divergent_block: no progress while warp {} is blocked with a divergent mask", w.0),
                        site: Some(site),
                    });
                }
                return Ok(RunStatus::Deadlock { blocked });
            }
        }
    }

    /// Fire async op `index` of the first partition (tests).
    pub fn fire_completion(&mut self, index: usize, arena: &mut Arena, observer: &mut dyn Observer) -> Result<(), ExecError> {
        let env = Env {
            program: self.program,
            loaded: &self.loaded,
            shape: &self.shape,
            config: &self.config,
            kernel: self.kernel_index,
            step_fn: step_warp,
            round: self.round,
            observing: self.observing,
        };
        let r = match self.partitions.first_mut() {
            Some(p) => p.fire_op(index, &env, arena),
            None => Err(sched_error(ExecErrorKind::Internal, self.kernel_index, WarpId(u32::MAX), SiteId::NONE, "no partition".into())),
        };
        if !self.partitions.is_empty() {
            self.absorb(0, observer);
        }
        r
    }

    /// Why a warp ended, for `Observer::warp_done`.
    pub fn end_reason(&self, warp: WarpId) -> Option<WarpEnd> {
        self.ends.get(&warp).copied()
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
        Some(ArgValue::Scalar(v)) => return Some(*v as i64),
        Some(_) => return None,
        None => {}
    }
    let ParamKind::ImplicitShape { buffer, axis } = slot.kind else { return None };
    let b = program.host_abi.get(buffer.0 as usize)?;
    let Some(ArgValue::Buffer { bytes, .. }) = lookup(inputs, b) else { return None };
    let elem = b.dtype.map(|t| t.mem_bytes() as usize).unwrap_or(1).max(1);
    if b.shape.len() <= 1 && axis == 0 {
        return Some((bytes.len() / elem) as i64);
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
    Some((bytes.len() / elem) as i64 / other)
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
    backend: &Backend,
    seed: u64,
) -> Result<RunOutcome, RunError> {
    let config = RunConfig { seed, ..RunConfig::default() };
    run_with_config(module, inputs, observer, backend, &config)
}

/// Allocate (once per module run) the global allocation of buffer argument `name`.
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
            let a = arena.alloc(Space::Global, Owner::Launch, name, size, init);
            globals.insert(name.to_string(), a);
            Ok(a)
        }
        Some(_) => Err(RunError::BadArg { name: name.into(), message: "expected a buffer argument".into() }),
        None => Err(RunError::MissingArg(name.into())),
    }
}

/// Encode a tensor map from its host-prelude spec over `va`.
fn encode_spec(spec: &crate::program::TensorMapSpec, va: u64, scalar: &dyn Fn(crate::program::ParamId) -> Option<i64>) -> Result<Vec<u8>, String> {
    let mut d = crate::oplib::TensorMapDesc::default();
    let ev = |e: &crate::program::DimExpr| e.eval(scalar).ok_or_else(|| "cannot evaluate tensor-map extent".to_string());
    d.global_address = va.wrapping_add(ev(&spec.base_offset)? as u64);
    d.rank = spec.rank;
    d.elem = Some(spec.dtype);
    for (i, e) in spec.global_dim.iter().enumerate().take(5) {
        d.global_dim[i] = ev(e)? as u64;
    }
    for (i, e) in spec.global_stride.iter().enumerate().take(5) {
        d.global_stride[i] = ev(e)? as u64;
    }
    for (i, &b) in spec.box_dim.iter().enumerate().take(5) {
        d.box_dim[i] = b;
    }
    for (i, &b) in spec.element_stride.iter().enumerate().take(5) {
        d.element_stride[i] = b;
    }
    d.interleave = spec.interleave;
    d.swizzle = spec.swizzle;
    d.l2_promotion = spec.l2_promotion;
    d.oob_fill = spec.oob_fill;
    Ok(d.encode().to_vec())
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
    if let Backend::Codegen(v) = backend {
        if v.len() != module.kernels.len() {
            return Err(RunError::Backend(format!("{} step functions for {} kernels", v.len(), module.kernels.len())));
        }
    }
    let mut arena = Arena::new(config.validity);
    let mut globals: BTreeMap<String, AllocId> = BTreeMap::new();
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
        let shape = resolve_launch(program, inputs)?;
        for slot in &program.host_abi {
            if slot.kind == ParamKind::Buffer {
                let name = if inputs.args.contains_key(&slot.name) {
                    slot.name.clone()
                } else {
                    slot.aliases.iter().find(|a| inputs.args.contains_key(*a)).cloned().unwrap_or(slot.name.clone())
                };
                let a = host_buffer(&mut arena, &mut globals, inputs, &name)?;
                globals.entry(slot.name.clone()).or_insert(a);
            }
        }
        prepared.push(shape);
    }
    for (k, program) in module.kernels.iter().enumerate() {
        let shape = prepared[k];
        let loaded = Loaded::new(program);
        let params = arena.alloc(Space::Param, Owner::Launch, &format!("params[k{k}]"), loaded.param_bytes, Init::Zeroed);
        let scalar = |p: crate::program::ParamId| -> Option<i64> { scalar_param(program, inputs, p) };
        for (i, slot) in program.host_abi.iter().enumerate() {
            let off = loaded.param_offsets[i];
            let arg = lookup(inputs, slot);
            let bytes: Vec<u8> = match (slot.kind, arg) {
                (ParamKind::Buffer, _) => arena.get(globals[&slot.name]).base.to_le_bytes().to_vec(),
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
                    let a = host_buffer(&mut arena, &mut globals, inputs, base)?;
                    let mut d = desc.clone();
                    d.global_address = arena.get(a).base + offset;
                    d.encode().to_vec()
                }
                (ParamKind::TensorMap, None) => {
                    let (Some(spec), Some(base)) = (&slot.tensor_map, slot.implicit_base) else {
                        return Err(RunError::MissingArg(slot.name.clone()));
                    };
                    let bname = &program.host_abi[base.0 as usize].name;
                    let a = *globals.get(bname).ok_or_else(|| RunError::MissingArg(bname.clone()))?;
                    let va = arena.get(a).base;
                    encode_spec(spec, va, &scalar).map_err(|m| RunError::BadArg { name: slot.name.clone(), message: m })?
                }
                (_, Some(_)) => {
                    return Err(RunError::BadArg { name: slot.name.clone(), message: format!("argument does not match a {:?} parameter", slot.kind) })
                }
            };
            write_param(&mut arena, params, off, &bytes);
        }
        let mut sched = Scheduler::with_params(program, k as u32, shape, &mut arena, &globals, params, loaded, config.clone())?;
        let status = sched.run(&mut arena, observer, backend);
        outcome.stats.instrs += sched.stats.instrs;
        outcome.stats.rounds += sched.stats.rounds;
        outcome.stats.completions += sched.stats.completions;
        outcome.diagnostics.append(&mut sched.diagnostics);
        if status == RunStatus::Completed {
            outcome.sync_leftovers.extend(sched.leftovers());
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
        if let (ArgValue::Buffer { .. }, Some(&a)) = (arg, globals.get(name)) {
            let al = arena.get(a);
            outcome.outputs.buffers.insert(name.clone(), (al.bytes.clone(), al.valid.clone()));
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
