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

use crate::arena::{addr, AllocId, Arena, BitSet, ByteSpan, Init, Owner, Space, ValidityPolicy, View};
use crate::interp::support::{self, AccessSpec, Accesses};
use crate::interp::{
    step_warp, BufBinding, CtaCtx, ExecCtx, ExecError, ExecErrorKind, LaunchAux, LaunchCounters, Loaded, StepResult,
    WarpState, WarpStatus, WarpStepFn,
};
use crate::observe::{
    AccessKind, Actor, CtaId, LaneSpan, LaunchInfo, Observer, ProtocolStatus, PublishTarget, Side, SyncEvent, SyncKind,
    WarpEnd, WarpId, Window, ALL_LANES,
};
use crate::oplib::OpErrorKind;
use crate::program::{Instr, LaunchShape, Module, ParamKind, Pc, Program, Scope, Sem};
use crate::report::Finding;
use crate::site::SiteId;
use crate::sync::completion::Payload;
use crate::sync::{async_group, cluster, mbarrier, setmaxnreg, Completion, Outcome, Policy, ResourceId, ResourceInit, Step, SyncCmd, SyncError, SyncTable};
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
    /// Resident CTAs (cluster-contiguous, in admission order).
    pub ctas: Vec<CtaState>,
    pub sync: SyncTable,
    pub rng: Rng,
    pub round: u64,
    pub stats: RunStats,
    pub loaded: Loaded,
    pub aux: LaunchAux,
    pub counters: LaunchCounters,
    /// Kernel parameter block.
    pub params: AllocId,
    outbox: Vec<InboxMsg>,
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

fn sched_error(kind: ExecErrorKind, kernel: u32, warp: WarpId, site: SiteId, message: String) -> ExecError {
    ExecError { kind, kernel, warp, pc: Pc(0), site, lanes: WarpMask::NONE, message }
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
            let b = &arena.get(params).bytes;
            Some(i64::from_le_bytes(b[off..off + 8].try_into().ok()?))
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
                    BufBinding::Tmem { base_col, cols } => BufBinding::Tmem { base_col: base_col + d.base as u32, cols },
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
                                        arena.get(params).bytes[poff as usize..poff as usize + 8].try_into().unwrap(),
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
                        BufBinding::Tmem { base_col: d.base as u32, cols: cols.clamp(1, addr::TMEM_COLS as i64) as u32 }
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

        let wpc = shape.warps_per_cta();
        let init = ResourceInit { policy: Policy::Numeric, cluster_warps: shape.ctas_per_cluster() * wpc, warps_per_cta: wpc };
        let mut pending: VecDeque<u32> = (0..shape.num_clusters()).collect();
        if let Some(s) = &config.subset {
            pending.retain(|c| s.contains(c));
        }
        let aux = LaunchAux { kernel: kernel_index, ..LaunchAux::default() };
        Ok(Scheduler {
            program,
            kernel_index,
            shape,
            rng: Rng::new(config.seed),
            config,
            ctas: Vec::new(),
            sync: SyncTable::new(init),
            round: 0,
            stats: RunStats::default(),
            loaded,
            aux,
            counters: LaunchCounters::default(),
            params,
            outbox: Vec::new(),
            bindings,
            pending,
            free_cta: Vec::new(),
            free_local: Vec::new(),
            ends: BTreeMap::new(),
            observing: false,
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

    /// Admit cluster `id`: allocate its CTAs and warps.
    fn admit(&mut self, id: u32, arena: &mut Arena, observer: &mut dyn Observer) -> Result<(), ExecError> {
        let shape = self.shape;
        let n = shape.ctas_per_cluster().max(1);
        let wpc = shape.warps_per_cta();
        let nslots = *self.loaded.slots.last().unwrap_or(&0) as usize;
        let tmem_bytes = if self.loaded.uses_tmem { addr::TMEM_BYTES } else { 0 };
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
            self.aux.owner_cta.insert(smem, cid);
            self.aux.owner_cta.insert(tmem, cid);
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
                    let i = wid.0 as usize;
                    if self.aux.reg_allocs.len() <= i {
                        self.aux.reg_allocs.resize(i + 1, AllocId(u32::MAX));
                    }
                    self.aux.reg_allocs[i] = ra;
                    self.host_event(observer, SyncKind::AllocBegin { alloc: ra, space: Space::Reg, size, cta: cid });
                }
                warps.push(ws);
            }
            // Declared words in the shared window.
            if self.aux.wants_history {
                for (i, b) in self.program.buffers.iter().enumerate() {
                    if let (true, BufBinding::SharedWindow { offset, len }) = (b.sync_words, self.bindings[i]) {
                        let span = ByteSpan::new(offset as u64, len);
                        self.aux.words.declare(arena, ctx.smem, span);
                        self.host_event(observer, SyncKind::DeclareWord { alloc: ctx.smem, span });
                    }
                }
            }
            if self.program.topology.regs_per_thread != 0 {
                let res = ResourceId::RegPool { cta: cid };
                let cmd = SyncCmd::RegPool(setmaxnreg::Cmd::Configure { count: self.program.topology.regs_per_thread });
                self.sync.step(res, cmd).map_err(|e| {
                    sched_error(ExecErrorKind::Protocol(e.clone()), self.kernel_index, WarpId(cid.0 * wpc), SiteId::NONE, format!("{e:?}"))
                })?;
            }
            if self.loaded.uses_cluster_barrier {
                // Lanes beyond the CTA's thread count never take part.
                for ws in &warps {
                    let missing = WarpMask::ALL.and_not(ws.live);
                    if !missing.is_empty() {
                        let res = ResourceId::Cluster { cluster: id };
                        let warp = ctx.rank_in_cluster * wpc + ws.warp_in_cta;
                        let cmd = SyncCmd::Cluster(cluster::Cmd::Exit { warp, lanes: missing.bits() });
                        self.sync.step(res, cmd).map_err(|e| {
                            sched_error(ExecErrorKind::Internal, self.kernel_index, ws.id, SiteId::NONE, format!("{e:?}"))
                        })?;
                    }
                }
            }
            self.ctas.push(CtaState { ctx, warps, inbox: Inbox::default(), buffers: self.bindings.clone() });
        }
        Ok(())
    }

    /// Retire finished clusters (all CTAs of the cluster finished) and
    /// admit pending ones. Returns whether anything changed.
    fn turnover(&mut self, arena: &mut Arena, observer: &mut dyn Observer) -> Result<bool, ExecError> {
        let mut changed = false;
        let mut i = 0;
        while i < self.ctas.len() {
            let cl = self.ctas[i].ctx.cluster;
            let mut j = i;
            while j < self.ctas.len() && self.ctas[j].ctx.cluster == cl {
                j += 1;
            }
            let done = self.ctas[i..j].iter().all(|c| c.finished() && c.inbox.msgs.is_empty());
            // Async ops still writing into the cluster keep it resident.
            let busy = self.sync.async_ops.iter().any(|op| self.ctas[i..j].iter().any(|c| c.ctx.id == op.source.cta));
            if done && !busy && !self.pending.is_empty() {
                let gone: Vec<CtaState> = self.ctas.drain(i..j).collect();
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
        let cap = if self.all_resident() { u32::MAX } else { self.config.max_resident_ctas };
        let per = self.shape.ctas_per_cluster().max(1);
        while let Some(&id) = self.pending.front() {
            let resident = self.ctas.len() as u32;
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
        self.aux.wants_history = self.observing && observer.wants_word_history();
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
        if self.aux.wants_history {
            for (i, b) in self.program.buffers.iter().enumerate() {
                if let (true, BufBinding::View(v)) = (b.sync_words, self.bindings[i]) {
                    let span = ByteSpan::new(v.offset, v.len);
                    self.aux.words.declare(arena, v.alloc, span);
                    self.host_event(observer, SyncKind::DeclareWord { alloc: v.alloc, span });
                }
            }
        }
        let status = match self.run_loop(arena, observer, step_fn) {
            Ok(s) => s,
            Err(e) => classify(e),
        };
        self.finish(arena, observer, &status);
        status
    }

    fn run_loop(&mut self, arena: &mut Arena, observer: &mut dyn Observer, step_fn: WarpStepFn) -> Result<RunStatus, ExecError> {
        self.turnover(arena, observer)?;
        loop {
            if self.round >= self.config.max_rounds {
                return Ok(RunStatus::Incomplete { reason: format!("round budget of {} exhausted", self.config.max_rounds), site: None });
            }
            let mut progress = false;
            for ci in 0..self.ctas.len() {
                progress |= self.drain_inbox(ci, arena, observer)?;
                progress |= self.run_cta(ci, arena, observer, step_fn)?;
                let cta = self.ctas[ci].ctx.id;
                progress |= self.land(Some(cta), arena, observer, self.config.completions == CompletionPolicy::Eager)?;
                progress |= self.apply_completions(arena, observer)?;
            }
            progress |= self.route_outbox();
            progress |= self.turnover(arena, observer)?;
            self.round += 1;
            self.stats.rounds = self.round;
            let all_done = self.pending.is_empty() && self.ctas.iter().all(|c| c.finished() && c.inbox.msgs.is_empty());
            if all_done {
                // Drain the async queues.
                loop {
                    let a = self.land(None, arena, observer, true)?;
                    let b = self.apply_completions(arena, observer)?;
                    let c = self.route_outbox();
                    let mut d = false;
                    for ci in 0..self.ctas.len() {
                        d |= self.drain_inbox(ci, arena, observer)?;
                    }
                    if !(a || b || c || d) {
                        break;
                    }
                }
                return Ok(RunStatus::Completed);
            }
            if !progress {
                // Before declaring a deadlock, land everything that is ready.
                let a = self.land(None, arena, observer, true)?;
                let b = self.apply_completions(arena, observer)?;
                if a || b {
                    continue;
                }
                let mut blocked = Vec::new();
                let mut divergent = None;
                for c in &self.ctas {
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

    /// One slice per runnable warp of resident CTA `ci`.
    fn run_cta(&mut self, ci: usize, arena: &mut Arena, observer: &mut dyn Observer, step_fn: WarpStepFn) -> Result<bool, ExecError> {
        let nw = self.ctas[ci].warps.len();
        if nw == 0 {
            return Ok(false);
        }
        let cid = self.ctas[ci].ctx.id.0 as u64;
        let start = (Rng::new(self.config.seed ^ self.round.wrapping_mul(0x2545_f491_4f6c_dd1d) ^ (cid << 32)).next_u64() % nw as u64) as usize;
        let mut progress = false;
        for k in 0..nw {
            let w = (start + k) % nw;
            let Scheduler { program, loaded, shape, config, ctas, sync, outbox, counters, aux, observing, ends, .. } = self;
            let cta = &mut ctas[ci];
            let CtaState { ctx: cctx, warps, buffers, .. } = cta;
            let warp = &mut warps[w];
            if !matches!(warp.status, WarpStatus::Running | WarpStatus::Blocked(_)) {
                continue;
            }
            let (pc0, prog0) = (warp.pc, counters.progress);
            let mut ctx = ExecCtx {
                program,
                loaded,
                launch: shape,
                config,
                warp,
                cta: cctx,
                buffers,
                arena: &mut *arena,
                sync,
                outbox,
                observer: &mut *observer,
                observing: *observing,
                counters,
                aux,
            };
            let r = crate::codegen::rt::guard(&mut ctx, config.quantum, step_fn);
            let (pc1, prog1) = (ctx.warp.pc, ctx.counters.progress);
            drop(ctx);
            let warp = &mut ctas[ci].warps[w];
            match r {
                StepResult::Continue | StepResult::Yield => {
                    warp.status = WarpStatus::Running;
                    progress = true;
                }
                StepResult::Blocked(res) => {
                    // Only committed effects count: a warp that merely moved
                    // to (or swapped between) blocking points changed
                    // nothing another warp could wait on.
                    let _ = (pc0, pc1);
                    if prog1 != prog0 {
                        progress = true;
                    }
                    warp.status = WarpStatus::Blocked(res);
                }
                StepResult::Exit => {
                    warp.status = WarpStatus::Exited;
                    ends.insert(warp.id, WarpEnd::Exited);
                    observer.warp_done(warp.id, WarpEnd::Exited);
                    progress = true;
                }
                StepResult::Error(e) => {
                    let end = match e.kind {
                        ExecErrorKind::Trap => WarpEnd::Trapped,
                        ExecErrorKind::Budget => WarpEnd::Budget,
                        _ => WarpEnd::Error,
                    };
                    warp.status = if end == WarpEnd::Trapped { WarpStatus::Trapped } else { WarpStatus::Errored };
                    ends.insert(warp.id, end);
                    observer.warp_done(warp.id, end);
                    return Err(e);
                }
            }
        }
        Ok(progress)
    }

    /// Deliver CTA `ci`'s inbox.
    fn drain_inbox(&mut self, ci: usize, arena: &mut Arena, observer: &mut dyn Observer) -> Result<bool, ExecError> {
        let msgs = std::mem::take(&mut self.ctas[ci].inbox.msgs);
        let cta = self.ctas[ci].ctx.id;
        let any = !msgs.is_empty();
        for m in msgs {
            match m {
                InboxMsg::Write { alloc, offset, bytes, actor, site } => {
                    let span = ByteSpan::new(offset, bytes.len() as u64);
                    arena
                        .write(support::whole(arena, alloc), &[span], &bytes)
                        .map_err(|e| sched_error(ExecErrorKind::OutOfBounds, self.kernel_index, actor_warp(actor), site, e.to_string()))?;
                    if self.observing {
                        let mut acc = Accesses::default();
                        acc.items.push((alloc, Some(Window::SharedCluster), LaneSpan { lane: ALL_LANES, span }));
                        let spec = AccessSpec {
                            actor,
                            site,
                            kind: AccessKind::Write,
                            sem: Sem::Weak,
                            scope: Scope::Cluster,
                            atomic: false,
                            returns_value: false,
                            proxy: crate::program::Proxy::Generic,
                        };
                        support::emit_accesses(observer, &mut self.counters, &mut self.aux, arena, spec, &mut acc);
                    }
                }
                InboxMsg::Sync { resource, cmd, actor, site } => {
                    let out = self.sync.step(resource, cmd).map_err(|e| {
                        sched_error(ExecErrorKind::Protocol(e.clone()), self.kernel_index, actor_warp(actor), site, format!("{e:?}"))
                    })?;
                    if let Step::Done(Outcome::Mbarrier(mbarrier::Outcome::Arrived { gen, .. })) = out {
                        if self.observing {
                            observer.sync(&SyncEvent {
                                kernel: self.kernel_index,
                                actor,
                                seq: 0,
                                site,
                                frames: Vec::new(),
                                lanes: WarpMask::NONE,
                                kind: SyncKind::Arrive { obj: resource, phase: gen, release: None, scope: None },
                            });
                        }
                    }
                }
            }
        }
        observer.inbox_drain(cta, self.round);
        Ok(any)
    }

    /// Move outbox messages to their target CTAs' inboxes.
    fn route_outbox(&mut self) -> bool {
        if self.outbox.is_empty() {
            return false;
        }
        for m in std::mem::take(&mut self.outbox) {
            let target = match &m {
                InboxMsg::Sync { resource: ResourceId::Mbarrier { cta, .. }, .. } => Some(*cta),
                _ => m.target_hint().and_then(|a| self.aux.owner_cta.get(&a).copied()),
            };
            match target.and_then(|t| self.ctas.iter_mut().find(|c| c.ctx.id == t)) {
                Some(c) => c.inbox.msgs.push(m),
                // The target CTA is gone: the effect is lost (accessing an
                // exited CTA's shared memory is undefined).
                None => {}
            }
        }
        true
    }

    /// Land ready async ops (of `cta`, or all with `None`). `all` = every
    /// ready op; otherwise a seeded subset. Returns whether any landed.
    fn land(&mut self, cta: Option<CtaId>, arena: &mut Arena, observer: &mut dyn Observer, all: bool) -> Result<bool, ExecError> {
        let mut any = false;
        loop {
            let mut landed_one = false;
            let mut i = 0;
            while i < self.sync.async_ops.len() {
                let op = &self.sync.async_ops[i];
                let mine = cta.is_none_or(|c| op.source.cta == c);
                let ready = op.after.iter().all(|d| !self.sync.async_ops.iter().any(|o| o.id == *d));
                if mine && ready && (all || self.rng.below(2) == 0) {
                    self.fire_completion(i, arena, observer)?;
                    landed_one = true;
                    any = true;
                    continue;
                }
                i += 1;
            }
            if !landed_one || !all {
                break;
            }
        }
        Ok(any)
    }

    /// Apply enabled sync completions in FIFO order until none is enabled.
    fn apply_completions(&mut self, _arena: &mut Arena, _observer: &mut dyn Observer) -> Result<bool, ExecError> {
        let mut any = false;
        loop {
            let Some(i) = self.sync.completions.iter().position(|c| self.sync.enabled(c)) else { break };
            let c = self.sync.completions.remove(i).expect("index valid");
            match self.sync.apply_completion(c) {
                Ok(Step::Done(out)) => {
                    any = true;
                    self.stats.completions += 1;
                    if let (
                        Completion::GroupMilestone { res, ordinal, milestone: async_group::Milestone::FullyDone },
                        Outcome::AsyncGroup(async_group::Outcome::Completed { .. }),
                    ) = (c, out)
                    {
                        if let Some(arr) = self.aux.groups.arrivals.remove(&(res, ordinal)) {
                            self.sync.completions.extend(arr);
                        }
                    }
                }
                Ok(Step::Blocked(_)) => {
                    // Not applicable yet after all; keep it at the back.
                    self.sync.completions.push_back(c);
                    break;
                }
                Err(e) => {
                    return Err(sched_error(
                        ExecErrorKind::Protocol(e.clone()),
                        self.kernel_index,
                        WarpId(u32::MAX),
                        SiteId::NONE,
                        format!("completion {c:?}: {e:?}"),
                    ))
                }
            }
        }
        Ok(any)
    }

    /// Fire one pending async op (payload, then targets), emitting
    /// observer events. Exposed for tests.
    pub fn fire_completion(&mut self, index: usize, arena: &mut Arena, observer: &mut dyn Observer) -> Result<(), ExecError> {
        let Some(op) = self.sync.async_ops.remove(index) else {
            return Err(sched_error(ExecErrorKind::Internal, self.kernel_index, WarpId(u32::MAX), SiteId::NONE, "no such async op".into()));
        };
        let meta = self.aux.async_meta.remove(&op.id);
        let proxy = meta.as_ref().map(|m| m.proxy).unwrap_or_default();
        let lane = meta.as_ref().map(|m| m.lane).unwrap_or(ALL_LANES);
        let kernel = self.kernel_index;
        let src_err = |e: String| sched_error(ExecErrorKind::OutOfBounds, kernel, op.source.warp, op.source.site, e);
        let mut reads: Vec<(AllocId, ByteSpan)> = Vec::new();
        let mut writes: Vec<(AllocId, ByteSpan)> = Vec::new();
        let mut rmw = false;
        match &op.payload {
            Payload::None => {}
            Payload::Copy { src, dst, zero_fill } => {
                copy_spans(arena, src, dst).map_err(src_err)?;
                reads.extend(src.iter().copied());
                writes.extend(dst.iter().copied());
                let pattern = meta.as_ref().map(|m| m.fill_pattern.as_slice()).unwrap_or(&[]);
                for &(a, s) in zero_fill {
                    if pattern.is_empty() {
                        arena.fill(support::whole(arena, a), &[s], 0).map_err(|e| src_err(e.to_string()))?;
                    } else {
                        let bytes: Vec<u8> = (0..s.len as usize).map(|i| pattern[i % pattern.len()]).collect();
                        arena.write(support::whole(arena, a), &[s], &bytes).map_err(|e| src_err(e.to_string()))?;
                    }
                    writes.push((a, s));
                }
                if meta.as_ref().is_some_and(|m| m.tf32_round) {
                    // TF32 tensor-map loads round each copied f32 element.
                    for &(a, s) in dst {
                        let al = arena.get_mut(a);
                        let (st, en) = (s.start as usize, s.end() as usize);
                        for w in al.bytes[st..en].chunks_exact_mut(4) {
                            let v = u32::from_le_bytes(w.try_into().unwrap());
                            w.copy_from_slice(&crate::oplib::tma_tf32_round(v).to_le_bytes());
                        }
                    }
                }
            }
            Payload::TcgenCp { src, dst, decompress_bits } => {
                if *decompress_bits != 0 {
                    return Err(sched_error(ExecErrorKind::Unsupported, kernel, op.source.warp, op.source.site, "tcgen05.cp decompression".into()));
                }
                copy_spans(arena, src, dst).map_err(src_err)?;
                reads.extend(src.iter().copied());
                writes.extend(dst.iter().copied());
            }
            Payload::Reduce { op: aop, dtype, src, dst } => {
                let s = gather_bytes(arena, src).map_err(src_err)?;
                let mut d = gather_bytes(arena, dst).map_err(src_err)?;
                crate::interp::handlers::mem::rmw_bytes(*aop, *dtype, &mut d, &s, &[], false)
                    .map_err(|e| sched_error(ExecErrorKind::Op(e.kind), kernel, op.source.warp, op.source.site, e.message))?;
                scatter_bytes(arena, dst, &d).map_err(src_err)?;
                reads.extend(src.iter().copied());
                writes.extend(dst.iter().copied());
                rmw = true;
            }
            Payload::Data { dst, bytes } => {
                scatter_bytes(arena, dst, bytes).map_err(src_err)?;
                writes.extend(dst.iter().copied());
            }
            Payload::ReduceData { op: aop, dtype, dst, bytes } => {
                let mut d = gather_bytes(arena, dst).map_err(src_err)?;
                crate::interp::handlers::mem::rmw_bytes(*aop, *dtype, &mut d, bytes, &[], false)
                    .map_err(|e| sched_error(ExecErrorKind::Op(e.kind), kernel, op.source.warp, op.source.site, e.message))?;
                scatter_bytes(arena, dst, &d).map_err(src_err)?;
                writes.extend(dst.iter().copied());
                rmw = true;
            }
            Payload::TcgenMma(p) => {
                let (r, w) = run_mma(arena, p, tc_arch(self.program.arch.as_deref())).map_err(|e| sched_error(ExecErrorKind::Op(e.kind), kernel, op.source.warp, op.source.site, e.message))?;
                reads.extend(r);
                writes.extend(w);
                rmw = true;
            }
        }
        if self.observing {
            let site = op.source.site;
            let mk = |side, kind| AccessSpec {
                actor: Actor::Async { op: op.id, side },
                site,
                kind,
                sem: Sem::Weak,
                scope: Scope::Gpu,
                atomic: false,
                returns_value: false,
                proxy,
            };
            let window = |arena: &Arena, a: AllocId| match arena.get(a).space {
                Space::Global => Some(Window::Global),
                Space::Shared => Some(Window::SharedCta),
                _ => None,
            };
            let mut acc = Accesses::default();
            acc.items = reads.iter().map(|&(a, s)| (a, window(arena, a), LaneSpan { lane, span: s })).collect();
            support::emit_accesses(observer, &mut self.counters, &mut self.aux, arena, mk(Side::Read, AccessKind::Read), &mut acc);
            acc.items = writes.iter().map(|&(a, s)| (a, window(arena, a), LaneSpan { lane, span: s })).collect();
            let wk = if rmw { AccessKind::Rmw } else { AccessKind::Write };
            support::emit_accesses(observer, &mut self.counters, &mut self.aux, arena, mk(Side::Write, wk), &mut acc);
            for c in &op.signals {
                if let Completion::MbarTx { res, gen, .. } | Completion::MbarArrive { res, gen, .. } = *c {
                    observer.sync(&SyncEvent {
                        kernel: self.kernel_index,
                        actor: Actor::Async { op: op.id, side: Side::Write },
                        seq: 0,
                        site,
                        frames: Vec::new(),
                        lanes: WarpMask::NONE,
                        kind: SyncKind::AsyncComplete { op: op.id, milestone: Side::Write, target: PublishTarget::Phase { obj: res, phase: gen } },
                    });
                }
            }
        }
        self.sync.completions.extend(op.signals.iter().copied());
        let open = self.aux.groups.is_open(op.id);
        let due = self.aux.groups.landed(op.id, open);
        self.sync.completions.extend(due);
        Ok(())
    }

    /// Why a warp ended, for `Observer::warp_done`.
    pub fn end_reason(&self, warp: WarpId) -> Option<WarpEnd> {
        self.ends.get(&warp).copied()
    }

    /// End-of-launch events: blocked-at-exit, warp_done, AllocEnd, end_launch.
    fn finish(&mut self, arena: &mut Arena, observer: &mut dyn Observer, status: &RunStatus) {
        let end = match status {
            RunStatus::Deadlock { .. } => WarpEnd::Deadlocked,
            RunStatus::Incomplete { .. } => WarpEnd::Budget,
            _ => WarpEnd::Error,
        };
        let mut blocked_events = Vec::new();
        for c in &self.ctas {
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
        for c in &self.ctas {
            for w in &c.warps {
                if !self.ends.contains_key(&w.id) {
                    self.ends.insert(w.id, end);
                    observer.warp_done(w.id, end);
                }
            }
        }
        if self.observing {
            let mut ends = Vec::new();
            for c in &self.ctas {
                ends.push(c.ctx.smem);
                ends.push(c.ctx.tmem);
                ends.extend(c.warps.iter().filter_map(|w| w.local));
            }
            for a in ends {
                self.host_event(observer, SyncKind::AllocEnd { alloc: a });
            }
        }
        self.stats.instrs = self.counters.instrs;
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

fn actor_warp(a: Actor) -> WarpId {
    match a {
        Actor::Warp { warp, .. } => warp,
        _ => WarpId(u32::MAX),
    }
}

fn write_param(arena: &mut Arena, params: AllocId, off: u64, bytes: &[u8]) {
    let v = support::whole(arena, params);
    let _ = arena.write(v, &[ByteSpan::new(off, bytes.len() as u64)], bytes);
}

/// Copy concatenated `src` spans onto concatenated `dst` spans (equal
/// totals), carrying validity.
fn copy_spans(arena: &mut Arena, src: &[(AllocId, ByteSpan)], dst: &[(AllocId, ByteSpan)]) -> Result<(), String> {
    let total = |v: &[(AllocId, ByteSpan)]| v.iter().map(|s| s.1.len).sum::<u64>();
    if total(src) != total(dst) {
        return Err(format!("copy length mismatch: {} vs {}", total(src), total(dst)));
    }
    let (mut si, mut so, mut di, mut doff) = (0usize, 0u64, 0usize, 0u64);
    while si < src.len() && di < dst.len() {
        let (sa, ss) = src[si];
        let (da, ds) = dst[di];
        let n = (ss.len - so).min(ds.len - doff);
        if n > 0 {
            arena
                .copy_with_validity((sa, ByteSpan::new(ss.start + so, n)), (da, ds.start + doff))
                .map_err(|e| e.to_string())?;
        }
        so += n;
        doff += n;
        if so == ss.len {
            si += 1;
            so = 0;
        }
        if doff == ds.len {
            di += 1;
            doff = 0;
        }
    }
    Ok(())
}

fn gather_bytes(arena: &Arena, spans: &[(AllocId, ByteSpan)]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for &(a, s) in spans {
        let mut b = vec![0u8; s.len as usize];
        arena.read(support::whole(arena, a), &[s], &mut b).map_err(|e| e.to_string())?;
        out.extend(b);
    }
    Ok(out)
}

fn scatter_bytes(arena: &mut Arena, spans: &[(AllocId, ByteSpan)], bytes: &[u8]) -> Result<(), String> {
    let mut pos = 0usize;
    for &(a, s) in spans {
        let n = s.len as usize;
        if pos + n > bytes.len() {
            return Err("payload shorter than its spans".into());
        }
        arena.write(support::whole(arena, a), &[s], &bytes[pos..pos + n]).map_err(|e| e.to_string())?;
        pos += n;
    }
    Ok(())
}

type Spans = Vec<(AllocId, ByteSpan)>;

/// Target architecture of a program (`Program::arch`).
fn tc_arch(arch: Option<&str>) -> crate::oplib::TcArch {
    match arch {
        Some(a) if a.starts_with("sm_103") => crate::oplib::TcArch::Sm103,
        Some(a) if a.starts_with("sm_107") => crate::oplib::TcArch::Sm107,
        _ => crate::oplib::TcArch::Sm100,
    }
}

/// tcgen05.mma numerics through oplib (`tc_mma_ctas`), recording the spans
/// it touched. `p.smem` / `p.tmem` are indexed by CTA within the issuing
/// group (0 = even CTA of the pair for `cta_group::2`).
fn run_mma(arena: &mut Arena, p: &crate::sync::completion::TcgenMmaPayload, arch: crate::oplib::TcArch) -> crate::oplib::OpResult<(Spans, Spans)> {
    use crate::oplib::OpError;
    use std::cell::RefCell;
    let cell = RefCell::new(arena);
    let reads: RefCell<Spans> = RefCell::new(Vec::new());
    let mut writes: Spans = Vec::new();
    let options = crate::oplib::TcMmaOptions { arch, ti16: p.args.ti16, ..Default::default() };
    let smem = |cta: u32, a: u32, out: &mut [u8]| -> crate::oplib::OpResult {
        let al = *p.smem.get(cta as usize).ok_or_else(|| OpError::invalid("mma smem operand of a CTA outside the group"))?;
        let off = addr::decode_shared(a).1 as u64;
        let ar = cell.borrow();
        let span = ByteSpan::new(off, out.len() as u64);
        ar.read(support::whole(&ar, al), &[span], out).map_err(|e| OpError::invalid(e.to_string()))?;
        reads.borrow_mut().push((al, span));
        Ok(())
    };
    let tmem_of = |cta: u32, lane: u32, col: u32| -> crate::oplib::OpResult<(AllocId, u64)> {
        let al = *p.tmem.get(cta as usize).ok_or_else(|| OpError::invalid("mma tmem operand of a CTA outside the group"))?;
        if lane >= addr::TMEM_LANES || col >= addr::TMEM_COLS {
            return Err(OpError::invalid(format!("tmem cell ({lane}, {col}) out of range")));
        }
        Ok((al, addr::tmem_byte_offset(lane, col)))
    };
    let tmem_read = |cta: u32, lane: u32, col: u32, out: &mut [u8]| -> crate::oplib::OpResult {
        let (al, off) = tmem_of(cta, lane, col)?;
        let ar = cell.borrow();
        let span = ByteSpan::new(off, out.len() as u64);
        ar.read(support::whole(&ar, al), &[span], out).map_err(|e| OpError::invalid(e.to_string()))?;
        reads.borrow_mut().push((al, span));
        Ok(())
    };
    let mut tmem_write = |cta: u32, lane: u32, col: u32, data: &[u8]| -> crate::oplib::OpResult {
        let (al, off) = tmem_of(cta, lane, col)?;
        let mut ar = cell.borrow_mut();
        let span = ByteSpan::new(off, data.len() as u64);
        let v = support::whole(&ar, al);
        ar.write(v, &[span], data).map_err(|e| OpError::invalid(e.to_string()))?;
        writes.push((al, span));
        Ok(())
    };
    crate::oplib::tc_mma_ctas(p, &options, &smem, &tmem_read, &mut tmem_write)?;
    let mut r = reads.into_inner();
    ByteSpanList::coalesce(&mut r);
    ByteSpanList::coalesce(&mut writes);
    Ok((r, writes))
}

struct ByteSpanList;
impl ByteSpanList {
    fn coalesce(v: &mut Spans) {
        v.sort();
        let mut out: Spans = Vec::with_capacity(v.len());
        for (a, s) in v.drain(..) {
            match out.last_mut() {
                Some((la, ls)) if *la == a && s.start <= ls.end() => {
                    let end = ls.end().max(s.end());
                    ls.len = end - ls.start;
                }
                _ => out.push((a, s)),
            }
        }
        *v = out;
    }
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
        outcome.diagnostics.append(&mut sched.aux.diagnostics);
        if status == RunStatus::Completed {
            outcome.sync_leftovers.extend(sched.sync.quiescent());
            if !sched.sync.async_ops.is_empty() {
                let e = sched_error(
                    ExecErrorKind::Internal,
                    k as u32,
                    WarpId(u32::MAX),
                    SiteId::NONE,
                    format!("{} async operations never became ready", sched.sync.async_ops.len()),
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
