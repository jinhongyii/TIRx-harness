//! Interpreter backend: per-warp state, the execution context shared with
//! the codegen backend, and the interpreter loop.
//!
//! The interpreter and the codegen backend differ *only* in dispatch: the
//! interpreter matches on `Instr` in [`handlers::dispatch`]; generated code
//! calls the same [`handlers`] functions with constant arguments. Neither
//! may contain semantics of its own (plan 2.3).
//!
//! # Divergence model
//!
//! Structured SIMT: `WarpState::active` is the current active mask;
//! `frames` is the mask stack. See `Instr` docs for the frame rules.
//! `live` = lanes that exist (partial last warp) and have not exited.
//!
//! # Blocking
//!
//! A handler returning `Flow::Blocked(r)` leaves `pc` unchanged; the
//! scheduler re-runs the same instruction later. Handlers whose instruction
//! has a registration half (named/cluster barrier arrive, mbarrier wait on a
//! token) store the token in `WarpState::resume` before blocking and consume
//! it (set `None`) when they finally complete, so a retry never re-registers.
//!
//! # Spin parking
//!
//! A polling handler that observes "not yet" (`try_wait` false,
//! `test_wait` false) calls [`ExecCtx::note_failed_poll`] (every distinct
//! resource is recorded). `LoopEnd` checks `WarpState::poll`: if the
//! iteration had a failed poll and no instruction with
//! `Instr::is_progress` completed (counted by [`end_instr`]), it returns
//! `Flow::Blocked(resource)` (marker `control::PARKED` in `resume`); the
//! retry at the same `LoopEnd` runs the next iteration. The iteration still
//! counts against the loop budget; every 64 iterations a running loop
//! yields (`Flow::Yield`) so other warps of the CTA run.
//!
//! # Divergent blocking
//!
//! See [`divergent_switch`]: a blocking instruction under a partial mask
//! inside a divergent `If` suspends its arm and runs the complementary
//! arm, so `if lane == 0 { wait } else { arrive }` completes as on hardware
//! with independent thread scheduling.
//!
//! # Memory and events
//!
//! Handlers resolve every lane's location ([`support::resolve`],
//! [`support::resolve_buf`]), touch the [`Arena`] lane by lane in lane
//! order, and emit one `observe::Access` per (instruction, allocation)
//! after the effect. Cross-CTA effects (DSMEM stores/atomics, remote and
//! multicast mbarrier commands) apply synchronously at issue (single arena,
//! single `SyncTable`). Async data effects are `AsyncOp`s per issuing thread
//! that the scheduler lands later.

pub mod aux;
pub mod handlers;
pub(crate) mod support;

pub use aux::LaunchAux;

use crate::arena::{AllocId, Arena, View};
use crate::observe::{AccessSeq, Actor, CtaId, LoopFrame, Observer, WarpId};
use crate::oplib::{OpErrorKind, PtxFn};
use crate::program::{LaunchShape, Operand, Pc, Program, Reg};
use crate::sched::RunConfig;
use crate::site::SiteId;
use crate::sync::{ResourceId, SyncError, SyncTable};
use crate::value::{RegFile, WarpMask, WarpValue};
use std::fmt;

/// Kind of a mask-stack frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    /// Then-branch of an `If`; `taken` = lanes that entered it.
    Then,
    Else,
    /// A loop: `begin` = pc of its `LoopBegin` (site = loop-enter site).
    ///
    /// `outer`: failed polls of the enclosing iteration before the loop
    /// was entered plus those of this loop's finished iterations (merged
    /// back into the enclosing iteration when the loop exits). `spin`: hash
    /// of the warp's register state at the previous `LoopEnd` whose
    /// iteration only failed polls (0 = none); spin parking requires the
    /// next such `LoopEnd` to find the same state (a fixed point).
    Loop { begin: Pc, iteration: u64, broken: WarpMask, continued: WarpMask, outer: PollState, spin: u64 },
}

/// One mask-stack frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskFrame {
    pub kind: FrameKind,
    /// pc of the `If` / `LoopBegin` that pushed the frame.
    pub origin: Pc,
    /// Active mask when the frame was pushed.
    pub entry: WarpMask,
    /// If: lanes that took the then-branch. Loop: lanes still iterating.
    pub taken: WarpMask,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarpStatus {
    Running,
    Blocked(ResourceId),
    Exited,
    Trapped,
    Errored,
}

/// Failed-poll bookkeeping for spin parking (reset by `LoopEnd`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PollState {
    /// Resource of the first failed poll in this iteration.
    pub failed_on: Option<ResourceId>,
    /// Further distinct resources polled without success in this iteration
    /// (a loop polling `A || B` waits on both; `overflow` = more than fit).
    /// The scheduler retries parked warps every round, so these are
    /// informational today; a resource-version skip must wait on all of them.
    pub also: [Option<ResourceId>; 3],
    pub overflow: bool,
    /// The iteration did something with an observable effect (store,
    /// successful sync, async issue).
    pub progressed: bool,
}

/// All per-warp execution state (owned by the scheduler's `CtaState`).
#[derive(Clone, Debug)]
pub struct WarpState {
    pub id: WarpId,
    pub cta: CtaId,
    pub warp_in_cta: u32,
    pub pc: Pc,
    /// Register slots (see `crate::value`).
    pub regs: RegFile,
    pub active: WarpMask,
    pub live: WarpMask,
    pub frames: Vec<MaskFrame>,
    pub status: WarpStatus,
    /// Opaque token kept across retries of one blocking instruction (e.g.
    /// the named-barrier generation registered by `Sync` before `Resume`).
    pub resume: Option<u64>,
    pub poll: PollState,
    /// Per-instruction epoch for observer events (incremented by the
    /// dispatcher before every instruction).
    pub epoch: u64,
    /// Next committed `Protocol` event seq.
    pub sync_seq: u32,
    pub steps: u64,
    /// Local-memory allocation (lane-major), if any.
    pub local: Option<AllocId>,
    /// Register-buffer allocation (lane-major, `Space::Reg`), if any.
    pub regbuf: Option<AllocId>,
    /// Arms of divergent `If`s suspended at a blocking instruction while
    /// the complementary arm runs (structured-SIMT scheduling rule, see
    /// [`divergent_switch`]).
    pub suspended: Vec<Suspension>,
    /// Exact memo of [`WarpState::spin_hash`] (W13): the last hashed state.
    pub spin_memo: Option<Box<SpinMemo>>,
    /// Register slots still to allocate (zeroed) before the warp's first
    /// instruction ([`WarpState::new_deferred`]); 0 once allocated.
    pub pending_slots: usize,
}

/// Incremental state of [`WarpState::spin_hash`] (W13): the hash of every
/// register slot, their wrapping sum, and the slots written since the last
/// call (marked by [`WarpState::reg_mut`] / [`WarpState::reg_write_raw`]).
/// A call re-hashes only the written slots, so a poll-only loop iteration
/// costs O(registers it wrote), not O(register file): a kernel's file can
/// be megabytes per warp.
#[derive(Clone, Debug, Default)]
pub struct SpinMemo {
    slot: Vec<u64>,
    total: u64,
    dirty: Vec<u32>,
    marked: Vec<u64>,
}

impl SpinMemo {
    #[inline]
    fn mark(&mut self, s: u32) {
        let (w, b) = ((s / 64) as usize, s % 64);
        if let Some(x) = self.marked.get_mut(w) {
            if *x >> b & 1 == 0 {
                *x |= 1 << b;
                self.dirty.push(s);
            }
        }
    }
}

/// Hash of register slot `i` holding `v` (four multiply-rotate chains,
/// folded, then a splitmix finalizer).
#[inline]
fn slot_hash(i: u32, v: &WarpValue<u64>) -> u64 {
    const K: u64 = 0x5851_f42d_4c95_7f2d;
    let seed = 0xcbf2_9ce4_8422_2325 ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut acc: [u64; 4] = std::array::from_fn(|k| seed.wrapping_add(k as u64));
    for chunk in v.as_chunks::<4>().0 {
        for (a, &x) in acc.iter_mut().zip(chunk) {
            *a = (a.rotate_left(5) ^ x).wrapping_mul(K);
        }
    }
    let mut h = seed;
    for a in acc {
        h = (h.rotate_left(5) ^ a).wrapping_mul(K);
    }
    splitmix(h)
}

#[inline]
fn splitmix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Register files at least this large are allocated at the warp's first
/// step ([`WarpState::new_deferred`]).
pub const DEFER_REGS_BYTES: usize = 128 << 10;

/// A zeroed register file of `n` slots from one zeroed allocation (W13).
/// `RegFile::new` builds it slot by slot (`vec!` only uses a zeroed
/// allocation for arrays of at most 16 elements), and a corpus kernel's
/// file is tens to hundreds of KiB per warp; `calloc` zeroes it in bulk
/// (or maps fresh zero pages for large files).
fn zeroed_regs(n: usize) -> RegFile {
    if n == 0 {
        return RegFile { regs: Vec::new() };
    }
    let layout = std::alloc::Layout::array::<WarpValue<u64>>(n).expect("register file size");
    // SAFETY: `layout` is the layout of `[WarpValue<u64>; n]` (n > 0); the
    // memory comes from the global allocator, all-zero bytes are a valid
    // `[u64; 32]`, and length == capacity == n.
    unsafe {
        let p = std::alloc::alloc_zeroed(layout) as *mut WarpValue<u64>;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        RegFile { regs: Vec::from_raw_parts(p, n, n) }
    }
}

/// One suspended arm of a divergent `If`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Suspension {
    /// Index in `frames` of the `If` frame.
    pub depth: usize,
    /// The blocked instruction.
    pub pc: Pc,
    pub active: WarpMask,
    /// Frames opened inside the arm (above `depth`).
    pub frames: Vec<MaskFrame>,
    pub resume: Option<u64>,
    pub poll: PollState,
    /// What the suspended instruction blocked on (deadlock evidence when
    /// the warp is resumed at it).
    pub res: ResourceId,
}

impl PollState {
    /// Fold `other`'s polls into `self`.
    pub fn merge(&mut self, other: &PollState) {
        self.progressed |= other.progressed;
        self.overflow |= other.overflow;
        for r in std::iter::once(other.failed_on).chain(other.also.iter().copied()).flatten() {
            match self.failed_on {
                None => self.failed_on = Some(r),
                Some(f) if f == r => {}
                Some(_) => {
                    if !self.also.contains(&Some(r)) {
                        match self.also.iter_mut().find(|x| x.is_none()) {
                            Some(slot) => *slot = Some(r),
                            None => self.overflow = true,
                        }
                    }
                }
            }
        }
    }
}

impl WarpState {
    /// Hash of the state a loop iteration can depend on (registers and
    /// masks); equal hashes at consecutive poll-only `LoopEnd`s mean the
    /// next iteration repeats unless memory or sync state changes.
    ///
    /// Computed incrementally ([`SpinMemo`]): equal to
    /// [`WarpState::spin_hash_uncached`] of the current state (checked on
    /// every call in debug builds). Only equality of two hashes is ever
    /// used (a fixed point of the same loop frame), never the value.
    pub fn spin_hash(&mut self) -> u64 {
        let n = self.regs.regs.len();
        let m = self.spin_memo.get_or_insert_with(Default::default);
        if m.slot.len() != n {
            m.slot = self.regs.regs.iter().enumerate().map(|(i, v)| slot_hash(i as u32, v)).collect();
            m.total = m.slot.iter().fold(0u64, |a, &h| a.wrapping_add(h));
            m.marked = vec![0; n.div_ceil(64)];
            m.dirty.clear();
        } else {
            for &s in &m.dirty {
                let i = s as usize;
                let h = slot_hash(s, &self.regs.regs[i]);
                m.total = m.total.wrapping_sub(m.slot[i]).wrapping_add(h);
                m.slot[i] = h;
                m.marked[i / 64] = 0;
            }
            m.dirty.clear();
        }
        let h = Self::spin_finish(m.total, self.spin_masks());
        debug_assert_eq!(h, self.spin_hash_uncached(), "spin_hash: a register write bypassed WarpState::reg_mut");
        h
    }

    fn spin_masks(&self) -> u64 {
        (self.active.bits() as u64) << 32 | self.live.bits() as u64
    }

    fn spin_finish(total: u64, masks: u64) -> u64 {
        splitmix(total ^ splitmix(masks ^ 0x2545_f491_4f6c_dd1d)) | 1
    }

    /// The spin-parking state hash, computed from scratch.
    pub fn spin_hash_uncached(&self) -> u64 {
        let total = self.regs.regs.iter().enumerate().fold(0u64, |a, (i, v)| a.wrapping_add(slot_hash(i as u32, v)));
        Self::spin_finish(total, self.spin_masks())
    }

    /// Register slot `s` for writing; marks it for [`WarpState::spin_hash`].
    /// Every register write goes through this or [`WarpState::reg_write_raw`].
    #[inline]
    pub fn reg_mut(&mut self, s: u32) -> &mut WarpValue<u64> {
        if let Some(m) = &mut self.spin_memo {
            m.mark(s);
        }
        self.regs.get_mut(s)
    }

    /// [`RegFile::write_raw`] through [`WarpState::reg_mut`]'s marking.
    #[inline]
    pub fn reg_write_raw(&mut self, s: u32, v: &WarpValue<u64>, mask: WarpMask) {
        if let Some(m) = &mut self.spin_memo {
            m.mark(s);
        }
        self.regs.write_raw(s, v, mask);
    }

    pub fn new(id: WarpId, cta: CtaId, warp_in_cta: u32, nslots: usize, live: WarpMask) -> WarpState {
        WarpState {
            id,
            cta,
            warp_in_cta,
            pc: Pc(0),
            regs: zeroed_regs(nslots),
            active: live,
            live,
            frames: Vec::new(),
            status: WarpStatus::Running,
            resume: None,
            poll: PollState::default(),
            epoch: 0,
            sync_seq: 0,
            steps: 0,
            local: None,
            regbuf: None,
            suspended: Vec::new(),
            spin_memo: None,
            pending_slots: 0,
        }
    }

    /// [`WarpState::new`] whose register file, when it is at least
    /// [`DEFER_REGS_BYTES`], is allocated (zeroed) by the warp's first
    /// [`step_warp`] on whichever worker runs it, instead of at admission on
    /// the scheduler thread (W13: a Mega MoE launch admits 2,368 warps of
    /// ~910 KiB each at once). Smaller files stay eager: zeroed at admission
    /// they come from fresh heap whose untouched pages are never faulted,
    /// which first-step zeroing loses (measured 5x slower on rmsnorm).
    pub fn new_deferred(id: WarpId, cta: CtaId, warp_in_cta: u32, nslots: usize, live: WarpMask) -> WarpState {
        if nslots * std::mem::size_of::<WarpValue<u64>>() < DEFER_REGS_BYTES {
            return WarpState::new(id, cta, warp_in_cta, nslots, live);
        }
        let mut w = WarpState::new(id, cta, warp_in_cta, 0, live);
        w.pending_slots = nslots;
        w
    }

    /// Allocate a deferred register file (all zero). The one place a
    /// deferred file is materialized; registers are only read by executing
    /// the warp (`step_warp`), which calls this first.
    #[inline]
    pub fn materialize_regs(&mut self) {
        if self.pending_slots != 0 {
            self.regs = zeroed_regs(std::mem::take(&mut self.pending_slots));
        }
    }

    /// Loop-frame snapshot for `SyncEvent::frames`.
    pub fn loop_frames(&self, program: &Program) -> Vec<LoopFrame> {
        self.frames
            .iter()
            .filter_map(|f| match f.kind {
                FrameKind::Loop { begin, iteration, .. } => {
                    Some(LoopFrame { site: program.site_of(begin), iteration })
                }
                _ => None,
            })
            .collect()
    }
}

/// What a handler tells its dispatcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    /// Continue at `pc + 1`.
    Next,
    /// Continue at the given pc.
    Jump(Pc),
    /// Continue at the given pc, but return to the scheduler first
    /// (quantum boundary, nanosleep).
    Yield(Pc),
    /// Re-execute this pc later.
    Blocked(ResourceId),
    /// No live lanes remain.
    Exit,
}

/// What one scheduling slice of a warp ended with.
#[derive(Clone, Debug, PartialEq)]
pub enum StepResult {
    /// Quantum used up; runnable.
    Continue,
    /// Voluntary yield; runnable.
    Yield,
    Blocked(ResourceId),
    Exit,
    Error(ExecError),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExecErrorKind {
    OutOfBounds,
    Uninit,
    Misaligned,
    BadAddress,
    /// Synchronization protocol violation.
    Protocol(SyncError),
    /// Oplib error (Unsupported -> incomplete, Invalid -> kernel error).
    Op(OpErrorKind),
    /// `trap` / failed assert.
    Trap,
    /// Reached an `Unsupported` instruction or unmodeled operand domain.
    Unsupported,
    /// Collective executed with an illegal active mask (e.g. `.aligned`
    /// with divergent lanes, membermask not a subset of active).
    Divergence,
    /// A warp-collective instruction (`.aligned`, `__syncwarp`/shuffle
    /// membermask, `setmaxnreg`, `grid.sync`, ...) executed by an illegal
    /// subset of the warp (legacy kind `warp_collective_divergence`, W5-13).
    /// `Divergence` stays for non-uniform operands.
    WarpCollectiveDivergence,
    /// Loop / step budget exhausted (incomplete, never success).
    Budget,
    /// Engine bug.
    Internal,
}

/// A runtime error at one instruction.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecError {
    pub kind: ExecErrorKind,
    /// Kernel index within the `Module` (whose `sites` table `site` indexes).
    pub kernel: u32,
    pub warp: WarpId,
    pub pc: Pc,
    pub site: SiteId,
    pub lanes: WarpMask,
    pub message: String,
    /// Structured facts for the report (merged into the run-status
    /// diagnostic / `Finding::attrs`): e.g. `faulting_lanes`, `operation`,
    /// `operands` for ALU faults, `budget` for loop budgets (W11-pin-message
    /// items 1 and 5).
    pub attrs: std::collections::BTreeMap<String, serde_json::Value>,
}

impl fmt::Display for ExecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} at {} (warp {}, {}): {}", self.kind, self.site, self.warp.0, self.pc, self.message)
    }
}

impl std::error::Error for ExecError {}

/// How a declared buffer is bound for the current launch (per CTA for
/// shared-window buffers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufBinding {
    /// Global or param buffer: a view of its allocation.
    View(View),
    /// Offset in the executing CTA's shared window.
    SharedWindow { offset: u32, len: u64 },
    /// Per-lane local array in the warp's local allocation.
    Local { offset: u64, per_lane: u64 },
    /// Per-lane register-space array (`Space::Reg` buffer) in the warp's
    /// register-buffer allocation (byte-backed, starts uninitialized).
    Reg { offset: u64, per_lane: u64 },
    /// Tensor-memory view (`Space::Tmem` buffer, BufferDecl TMEM rule):
    /// 32-bit element `e` at TMEM lane `(e / cols) % 128`, column
    /// `base_col + e % cols`.
    /// `base_reg`: a warp-uniform register holding the view's runtime base
    /// taddr (`lane << 16 | col`), read at each access (added to the
    /// static `base_col`).
    Tmem { base_col: u32, cols: u32, base_reg: Option<Reg> },
    /// Not bound (missing host argument): any access is an error.
    Unbound,
}

/// CTA-level facts handlers need.
#[derive(Clone, Debug)]
pub struct CtaCtx {
    pub id: CtaId,
    /// `ctaid` coordinates.
    pub ctaid: [u32; 3],
    pub cluster: u32,
    pub rank_in_cluster: u32,
    /// Global CTA ids of the cluster, indexed by rank.
    pub cluster_ctas: Vec<CtaId>,
    /// Shared windows of the cluster's CTAs, indexed by rank.
    pub cluster_smem: Vec<AllocId>,
    /// Tensor memory of the cluster's CTAs, indexed by rank.
    pub cluster_tmem: Vec<AllocId>,
    pub smem: AllocId,
    pub tmem: AllocId,
    /// Kernel parameter block (tensor maps).
    pub params: AllocId,
}

/// Launch-wide counters shared by all warps.
#[derive(Clone, Debug, Default)]
pub struct LaunchCounters {
    pub next_access: u64,
    pub instrs: u64,
    /// Instructions with `Instr::is_progress` that completed (not blocked).
    pub progress: u64,
}

impl LaunchCounters {
    pub fn next_access_seq(&mut self) -> AccessSeq {
        let s = AccessSeq(self.next_access);
        self.next_access += 1;
        s
    }
}

/// One operand-type signature of a multi-signature PTX op.
#[derive(Clone)]
pub struct OpVariant {
    pub dst_tys: Vec<crate::dtype::Ty>,
    pub src_tys: Vec<crate::dtype::Ty>,
    pub f: PtxFn,
    pub error: Option<String>,
}

/// Program-derived tables computed once per launch.
pub struct Loaded {
    /// `Program::reg_slot_offsets()`.
    pub slots: Vec<u32>,
    /// `Program::ops[i]` resolved to oplib functions.
    pub ops: Vec<PtxFn>,
    /// `Some(reason)` when `ops[i]` could not be resolved (the slot then
    /// holds a placeholder; executing it fails closed as `Unsupported`).
    pub op_errors: Vec<Option<String>>,
    /// For an op used with more than one operand-type signature: every
    /// signature's resolution (`ops` / `op_errors` hold the first one).
    /// Empty for single-signature ops (the common case).
    pub op_variants: Vec<Vec<OpVariant>>,
    /// `code[pc].is_progress()`, precomputed (spin parking / deadlock).
    pub progress: Vec<bool>,
    /// Byte offset of each host parameter in the launch's param block.
    pub param_offsets: Vec<u64>,
    /// Total param block bytes.
    pub param_bytes: u64,
    /// Bytes of local memory per lane (lane stride of the warp's local
    /// allocation); `BufBinding::Local::offset` is relative to a lane's base.
    pub local_per_lane: u64,
    /// Bytes of register-space buffers per lane (lane stride of the warp's
    /// register-buffer allocation).
    pub reg_per_lane: u64,
    /// The program contains cluster-barrier instructions (lane exits are
    /// then reported to the cluster barrier).
    pub uses_cluster_barrier: bool,
    /// The program contains `setmaxnreg` (aligned `bar.sync` then credits
    /// warpgroup syncs).
    pub uses_setmaxnreg: bool,
    /// The program contains tcgen05 instructions (TMEM is allocated).
    pub uses_tmem: bool,
    /// Engine-side effect of `Program::ops[i]` beyond its oplib function
    /// (CONTRACT_REQUESTS W4-9).
    pub op_effects: Vec<OpEffect>,
}

/// Engine state an otherwise register-only PTX op touches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpEffect {
    None,
    /// `prefetch.L1::32B.valid_addr`: the address must name addressable
    /// global memory (legacy `validate_global_cache_hint_address`).
    ValidGlobalAddr,
    /// `applypriority.async.bulk*` (`completion=bulk_group`): an async
    /// bulk operation of the issuing thread's bulk group.
    BulkGroupOp,
}

impl Loaded {
    /// Resolve the op table and precompute per-pc facts.
    pub fn new(program: &Program) -> Loaded {
        use crate::program::{Instr, ParamKind};
        let slots = program.reg_slot_offsets();
        // Every distinct operand-type signature of each op, in first-use
        // order: an op key used with different register types (e.g.
        // `cvt.s8.s8` into an s8 and into an s32 register) resolves once per
        // signature (W11-2), not once per op name.
        type Sig = (Vec<crate::dtype::Ty>, Vec<crate::dtype::Ty>);
        let mut sigs: Vec<Vec<Sig>> = vec![Vec::new(); program.ops.len()];
        let ty_of = |o: &Operand| match o {
            Operand::Reg(r) => program.regs[r.0 as usize].ty,
            Operand::Const(c) => program.consts[c.0 as usize].ty,
        };
        for ins in &program.code {
            if let Instr::Ptx { op, dsts, srcs, .. } = ins {
                let sig: Sig = (dsts.iter().map(|d| program.regs[d.0 as usize].ty).collect(), srcs.iter().map(ty_of).collect());
                let v = &mut sigs[op.0 as usize];
                if !v.contains(&sig) {
                    v.push(sig);
                }
            }
        }
        let resolve = |key: &crate::program::OpKey, (d, s): &Sig| -> (PtxFn, Option<String>) {
            match crate::oplib::resolve_ptx(key, d, s) {
                Ok(f) => (f, None),
                Err(e) => (PtxFn::new(support::unresolved_ptx), Some(e.message)),
            }
        };
        let mut ops: Vec<PtxFn> = Vec::with_capacity(program.ops.len());
        let mut op_errors = Vec::with_capacity(program.ops.len());
        let mut op_variants = Vec::with_capacity(program.ops.len());
        for (i, key) in program.ops.iter().enumerate() {
            let first = sigs[i].first().cloned().unwrap_or_default();
            let (f, e) = resolve(key, &first);
            ops.push(f);
            op_errors.push(e);
            op_variants.push(if sigs[i].len() > 1 {
                sigs[i].iter().map(|sg| {
                    let (f, e) = resolve(key, sg);
                    OpVariant { dst_tys: sg.0.clone(), src_tys: sg.1.clone(), f, error: e }
                }).collect()
            } else {
                Vec::new()
            });
        }
        let mut param_offsets = Vec::with_capacity(program.host_abi.len());
        let mut off = 0u64;
        for p in &program.host_abi {
            let (size, align) = match p.kind {
                ParamKind::TensorMap => (128, 64),
                _ => (8, 8),
            };
            off = off.div_ceil(align) * align;
            param_offsets.push(off);
            off += size;
        }
        let mut local_per_lane = 0u64;
        for b in &program.buffers {
            if b.space == crate::arena::Space::Local && b.view_of.is_none() {
                let len = b.byte_len.as_ref().and_then(|e| e.eval(&|_| None)).unwrap_or(0).max(0) as u64;
                let align = (b.align as u64).max(1);
                local_per_lane = local_per_lane.div_ceil(align) * align + len;
            }
        }
        // Keep every lane's base 16-byte aligned.
        let local_per_lane = local_per_lane.div_ceil(16) * 16;
        let mut reg_per_lane = 0u64;
        for b in &program.buffers {
            if b.space == crate::arena::Space::Reg && b.view_of.is_none() {
                let len = b.byte_len.as_ref().and_then(|e| e.eval(&|_| None)).unwrap_or(0).max(0) as u64;
                let align = (b.align as u64).max(1);
                reg_per_lane = reg_per_lane.div_ceil(align) * align + len;
            }
        }
        let reg_per_lane = reg_per_lane.div_ceil(16) * 16;
        let has = |f: fn(&Instr) -> bool| program.code.iter().any(f);
        Loaded {
            progress: program.code.iter().map(|i| i.is_progress()).collect(),
            uses_cluster_barrier: has(|i| matches!(i, Instr::ClusterArrive { .. } | Instr::ClusterWait { .. })),
            uses_setmaxnreg: has(|i| matches!(i, Instr::SetMaxNReg { .. })),
            uses_tmem: program.requirements.implicit_tmem
                || program.buffers.iter().any(|b| b.space == crate::arena::Space::Tmem)
                || has(|i| {
                matches!(
                    i,
                    Instr::TcgenAlloc { .. }
                        | Instr::TcgenDealloc { .. }
                        | Instr::TcgenLd(_)
                        | Instr::TcgenSt(_)
                        | Instr::TcgenCp(_)
                        | Instr::TcgenMma(_)
                )
            }),
            op_effects: program
                .ops
                .iter()
                .map(|k| match k.name.as_str() {
                    "tirx.ptx.prefetch_valid_addr" => OpEffect::ValidGlobalAddr,
                    n if n.starts_with("tirx.ptx.applypriority_async_bulk") => OpEffect::BulkGroupOp,
                    _ => OpEffect::None,
                })
                .collect(),
            slots,
            ops,
            op_errors,
            op_variants,
            param_offsets,
            param_bytes: off.div_ceil(8) * 8,
            local_per_lane,
            reg_per_lane,
        }
    }
}

/// Everything a handler may touch. Built by the scheduler for one warp's
/// slice. The codegen backend receives the same struct.
pub struct ExecCtx<'a> {
    pub program: &'a Program,
    pub loaded: &'a Loaded,
    pub launch: &'a LaunchShape,
    pub config: &'a RunConfig,
    pub warp: &'a mut WarpState,
    pub cta: &'a CtaCtx,
    /// `Program::buffers[i]` binding.
    pub buffers: &'a [BufBinding],
    pub arena: &'a mut Arena,
    pub sync: &'a mut SyncTable,
    pub observer: &'a mut dyn Observer,
    /// Cached `observer.enabled()`.
    pub observing: bool,
    pub counters: &'a mut LaunchCounters,
    /// Launch-wide engine bookkeeping that is not protocol state
    /// (async-group membership, declared-word histories, collective
    /// rendezvous, tcgen pipeline order). Owned by the scheduler.
    pub aux: &'a mut LaunchAux,
}

impl<'a> ExecCtx<'a> {
    #[inline]
    pub fn pc(&self) -> Pc {
        self.warp.pc
    }
    #[inline]
    pub fn site(&self) -> SiteId {
        self.program.site_of(self.warp.pc)
    }
    #[inline]
    pub fn active(&self) -> WarpMask {
        self.warp.active
    }
    /// First slot of register `r`.
    #[inline]
    pub fn slot(&self, r: Reg) -> u32 {
        self.loaded.slots[r.0 as usize]
    }
    /// Slot `i` of a register's value.
    #[inline]
    pub fn reg_slot(&self, r: Reg, i: u32) -> &WarpValue<u64> {
        self.warp.regs.get(self.slot(r) + i)
    }
    /// Slot `i` (0 = low 64 bits) of an operand, constants broadcast.
    #[inline]
    pub fn read_slot(&self, o: Operand, i: u32) -> WarpValue<u64> {
        match o {
            Operand::Reg(r) => *self.reg_slot(r, i),
            Operand::Const(c) => {
                let k = self.program.consts[c.0 as usize].bits;
                let w = match i {
                    0 => k as u64,
                    1 => (k >> 64) as u64,
                    _ => 0,
                };
                [w; 32]
            }
        }
    }
    /// Low 64 bits of an operand (values <= 64 bits).
    #[inline]
    pub fn read(&self, o: Operand) -> WarpValue<u64> {
        self.read_slot(o, 0)
    }
    /// Write slot `i` of `r` under the active mask.
    #[inline]
    pub fn write_slot(&mut self, r: Reg, i: u32, v: &WarpValue<u64>) {
        let m = self.warp.active;
        let s = self.slot(r) + i;
        self.warp.reg_write_raw(s, v, m);
    }
    /// Write a <= 64-bit value under the active mask.
    #[inline]
    pub fn write(&mut self, r: Reg, v: &WarpValue<u64>) {
        self.write_slot(r, 0, v);
    }
    /// The current warp instruction as an observer actor.
    #[inline]
    pub fn actor(&self) -> Actor {
        Actor::Warp { warp: self.warp.id, epoch: self.warp.epoch }
    }
    /// Build an error at the current instruction.
    pub fn error(&self, kind: ExecErrorKind, message: impl Into<String>) -> ExecError {
        ExecError {
            kind,
            kernel: self.aux.kernel,
            warp: self.warp.id,
            pc: self.warp.pc,
            site: self.site(),
            lanes: self.warp.active,
            message: message.into(),
            attrs: Default::default(),
        }
    }
    /// Record a failed poll for spin parking.
    #[inline]
    pub fn note_failed_poll(&mut self, r: ResourceId) {
        let p = &mut self.warp.poll;
        match p.failed_on {
            None => p.failed_on = Some(r),
            Some(f) if f == r => {}
            Some(_) => {
                if !p.also.contains(&Some(r)) {
                    match p.also.iter_mut().find(|x| x.is_none()) {
                        Some(slot) => *slot = Some(r),
                        None => p.overflow = true,
                    }
                }
            }
        }
    }
}

/// The signature of one warp slice; the interpreter's is [`step_warp`],
/// the codegen backend exports one per kernel with the same type.
pub type WarpStepFn = for<'a, 'b> fn(&'b mut ExecCtx<'a>, u32) -> StepResult;

/// Run one warp slice with `step`, turning a panic into
/// `ExecErrorKind::Internal` at the current pc: a panic must not unwind into
/// the scheduler's worker threads or the host.
#[inline(always)]
pub fn guard_step(ctx: &mut ExecCtx<'_>, quantum: u32, step: WarpStepFn) -> StepResult {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| step(&mut *ctx, quantum))) {
        Ok(r) => r,
        Err(payload) => StepResult::Error(ctx.error(ExecErrorKind::Internal, format!("panic: {}", panic_message(&*payload)))),
    }
}

/// The text of a caught panic payload.
pub fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// Shared per-instruction prologue (both backends): bump counters and the
/// observer epoch. Call with `ctx.warp.pc` already at the instruction.
#[inline(always)]
pub fn begin_instr(ctx: &mut ExecCtx<'_>) {
    ctx.warp.steps += 1;
    ctx.warp.epoch += 1;
    ctx.counters.instrs += 1;
}

/// Shared per-instruction epilogue (both backends): apply the handler's
/// result to the warp. `progress` is `code[pc].is_progress()` (a constant in
/// generated code, `Loaded::progress[pc]` in the interpreter). Returns
/// `Some` when the slice must end.
#[inline(always)]
pub fn end_instr(ctx: &mut ExecCtx<'_>, pc: Pc, progress: bool, res: handlers::HResult) -> Option<StepResult> {
    match res {
        Ok(Flow::Next) => {
            if progress {
                ctx.warp.poll.progressed = true;
                ctx.counters.progress += 1;
            }
            ctx.warp.pc = Pc(pc.0 + 1);
            None
        }
        Ok(Flow::Jump(t)) => {
            if progress {
                ctx.warp.poll.progressed = true;
                ctx.counters.progress += 1;
            }
            ctx.warp.pc = t;
            None
        }
        Ok(Flow::Yield(t)) => {
            ctx.warp.pc = t;
            Some(StepResult::Yield)
        }
        Ok(Flow::Blocked(r)) => divergent_switch(ctx, pc, r),
        Ok(Flow::Exit) => Some(StepResult::Exit),
        Err(e) => Some(StepResult::Error(e)),
    }
}

/// Structured-SIMT scheduling rule for a blocking instruction executed by
/// a divergent subset of the warp (contract review item 6).
///
/// Hardware with independent thread scheduling lets the other lanes run
/// while some lanes wait (`if lane == 0: wait(b) else: arrive(b)`). The
/// interpreter emulates that per `If`: when an instruction blocks inside an
/// arm of a divergent `If` (innermost first),
/// * if the complementary arm has an `Else` with lanes and has not run, the
///   blocked arm is suspended (pc, mask, inner frames, resume token) and the
///   complementary arm runs now, in the same slice; at its `EndIf` the
///   suspended arm resumes at its blocked instruction;
/// * if the complementary arm is itself suspended, the two swap: the warp
///   returns `Blocked` and its next retry starts with the other arm.
///
/// Otherwise the warp blocks as a whole. Arms without an `Else` cannot be
/// interleaved (the other lanes' continuation lies after the `EndIf`); a
/// deadlock involving a divergent warp is reported as incomplete, never as
/// an error.
pub fn divergent_switch(ctx: &mut ExecCtx<'_>, pc: Pc, r: ResourceId) -> Option<StepResult> {
    use crate::program::Instr;
    let w = &mut *ctx.warp;
    for i in (0..w.frames.len()).rev() {
        let f = w.frames[i];
        if !matches!(f.kind, FrameKind::Then | FrameKind::Else) {
            continue;
        }
        if let Some(k) = w.suspended.iter().position(|s| s.depth == i) {
            let other = w.suspended.swap_remove(k);
            let mine = Suspension {
                depth: i,
                pc,
                active: w.active,
                frames: w.frames[i + 1..].to_vec(),
                resume: w.resume.take(),
                poll: std::mem::take(&mut w.poll),
                res: r,
            };
            w.frames.truncate(i + 1);
            w.frames.extend(other.frames);
            w.frames[i].kind = FrameKind::Else;
            w.active = other.active.and(w.live);
            w.resume = other.resume;
            w.poll = other.poll;
            w.pc = other.pc;
            w.suspended.push(mine);
            // The warp now sits at the other arm's blocking instruction.
            return Some(StepResult::Blocked(other.res));
        }
        if f.kind == FrameKind::Then {
            let Some(Instr::If { else_pc, end_pc, .. }) = ctx.program.code.get(f.origin.0 as usize) else { continue };
            if else_pc == end_pc {
                continue;
            }
            // Lanes of the else arm (entry minus taken, still alive).
            let mut alive = w.live;
            for g in w.frames[..i].iter().rev() {
                if let FrameKind::Loop { broken, continued, .. } = g.kind {
                    alive = alive.and_not(broken.or(continued));
                    break;
                }
            }
            let else_lanes = f.entry.and_not(f.taken).and(alive);
            if else_lanes.is_empty() {
                continue;
            }
            let mine = Suspension {
                depth: i,
                pc,
                active: w.active,
                frames: w.frames[i + 1..].to_vec(),
                resume: w.resume.take(),
                poll: std::mem::take(&mut w.poll),
                res: r,
            };
            w.suspended.push(mine);
            w.frames.truncate(i + 1);
            w.frames[i].kind = FrameKind::Else;
            w.active = else_lanes;
            w.pc = Pc(else_pc.0 + 1);
            return None;
        }
    }
    Some(StepResult::Blocked(r))
}

/// The warp is divergent (partial active mask or a suspended arm).
pub fn is_divergent(w: &WarpState) -> bool {
    w.active != w.live || !w.suspended.is_empty()
}

/// Falling off the end of the code: every live lane exits.
pub fn fall_off_end(ctx: &mut ExecCtx<'_>) -> StepResult {
    ctx.warp.active = ctx.warp.live;
    match handlers::exit(ctx) {
        Ok(_) => StepResult::Exit,
        Err(e) => StepResult::Error(e),
    }
}

/// Interpreter: execute up to `quantum` instructions of `ctx.warp`.
pub fn step_warp(ctx: &mut ExecCtx<'_>, quantum: u32) -> StepResult {
    ctx.warp.materialize_regs();
    let program = ctx.program;
    for _ in 0..quantum {
        let pc = ctx.warp.pc;
        let Some(ins) = program.code.get(pc.0 as usize) else {
            return fall_off_end(ctx);
        };
        begin_instr(ctx);
        let res = handlers::dispatch(ctx, ins);
        let progress = ctx.loaded.progress[pc.0 as usize];
        if let Some(r) = end_instr(ctx, pc, progress, res) {
            return r;
        }
    }
    StepResult::Continue
}
