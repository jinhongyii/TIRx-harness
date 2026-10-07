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
//! `test_wait` false) calls [`ExecCtx::note_failed_poll`]. `LoopEnd` checks
//! `WarpState::poll`: if every poll in the iteration failed and the
//! iteration performed no write/sync with effect, it returns
//! `Flow::Blocked(resource)` instead of spinning (counted against the loop
//! budget as one iteration). `Instr::is_progress` classifies instructions;
//! the dispatcher may set `poll.progressed` from it.

pub mod handlers;

use crate::arena::{AllocId, Arena, View};
use crate::observe::{AccessSeq, Actor, CtaId, LoopFrame, Observer, WarpId};
use crate::oplib::{OpErrorKind, PtxFn};
use crate::program::{LaunchShape, Operand, Pc, Program, Reg};
use crate::sched::{InboxMsg, RunConfig};
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
    Loop { begin: Pc, iteration: u64, broken: WarpMask, continued: WarpMask },
}

/// One mask-stack frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskFrame {
    pub kind: FrameKind,
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
    /// Resource of the last failed poll in this iteration.
    pub failed_on: Option<ResourceId>,
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
    pub epoch: u32,
    /// Next committed `Protocol` event seq.
    pub sync_seq: u32,
    pub steps: u64,
    /// Local-memory allocation (lane-major), if any.
    pub local: Option<AllocId>,
    /// Logical clock for `SpecialReg::Clock*`.
    pub clock: u64,
}

impl WarpState {
    pub fn new(id: WarpId, cta: CtaId, warp_in_cta: u32, nslots: usize, live: WarpMask) -> WarpState {
        WarpState {
            id,
            cta,
            warp_in_cta,
            pc: Pc(0),
            regs: RegFile::new(nslots),
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
            clock: 0,
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
    /// Loop / step budget exhausted (incomplete, never success).
    Budget,
    /// Engine bug.
    Internal,
}

/// A runtime error at one instruction.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecError {
    pub kind: ExecErrorKind,
    pub warp: WarpId,
    pub pc: Pc,
    pub site: SiteId,
    pub lanes: WarpMask,
    pub message: String,
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
}

impl LaunchCounters {
    pub fn next_access_seq(&mut self) -> AccessSeq {
        let s = AccessSeq(self.next_access);
        self.next_access += 1;
        s
    }
}

/// Program-derived tables computed once per launch.
pub struct Loaded {
    /// `Program::reg_slot_offsets()`.
    pub slots: Vec<u32>,
    /// `Program::ops[i]` resolved to oplib functions.
    pub ops: Vec<PtxFn>,
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
    /// Effects on *other* CTAs (remote smem stores, remote mbarrier
    /// arrivals, multicast): delivered at the next inbox drain.
    pub outbox: &'a mut Vec<InboxMsg>,
    pub observer: &'a mut dyn Observer,
    /// Cached `observer.enabled()`.
    pub observing: bool,
    pub counters: &'a mut LaunchCounters,
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
    /// Value of a warp-uniform operand (first active lane), checking all
    /// active lanes agree (`Divergence` error otherwise).
    pub fn read_uniform(&self, o: Operand) -> Result<u64, ExecError> {
        match o {
            Operand::Const(c) => Ok(self.program.consts[c.0 as usize].bits as u64),
            Operand::Reg(r) => {
                let v = self.reg_slot(r, 0);
                let mask = self.active();
                let Some(first) = mask.first() else { return Ok(0) };
                if mask.lanes().any(|l| v[l] != v[first]) {
                    return Err(self.error(ExecErrorKind::Divergence, format!("operand {r} is not warp-uniform")));
                }
                Ok(v[first])
            }
        }
    }
    /// Write slot `i` of `r` under the active mask.
    #[inline]
    pub fn write_slot(&mut self, r: Reg, i: u32, v: &WarpValue<u64>) {
        let m = self.warp.active;
        let s = self.slot(r) + i;
        self.warp.regs.write_raw(s, v, m);
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
            warp: self.warp.id,
            pc: self.warp.pc,
            site: self.site(),
            lanes: self.warp.active,
            message: message.into(),
        }
    }
    /// Record a failed poll for spin parking.
    #[inline]
    pub fn note_failed_poll(&mut self, r: ResourceId) {
        self.warp.poll.failed_on = Some(r);
    }
    /// Record that this iteration had an observable effect.
    #[inline]
    pub fn note_progress(&mut self) {
        self.warp.poll.progressed = true;
    }
}

/// The signature of one warp slice; the interpreter's is [`step_warp`],
/// the codegen backend exports one per kernel with the same type.
pub type WarpStepFn = for<'a, 'b> fn(&'b mut ExecCtx<'a>, u32) -> StepResult;

/// Interpreter: execute up to `quantum` instructions of `ctx.warp`.
pub fn step_warp(ctx: &mut ExecCtx<'_>, quantum: u32) -> StepResult {
    let program = ctx.program;
    for _ in 0..quantum {
        let pc = ctx.warp.pc;
        let Some(ins) = program.code.get(pc.0 as usize) else {
            // Falling off the end = implicit exit.
            return StepResult::Exit;
        };
        ctx.warp.steps += 1;
        ctx.warp.epoch = ctx.warp.epoch.wrapping_add(1);
        ctx.counters.instrs += 1;
        match handlers::dispatch(ctx, ins) {
            Ok(Flow::Next) => ctx.warp.pc = Pc(pc.0 + 1),
            Ok(Flow::Jump(t)) => ctx.warp.pc = t,
            Ok(Flow::Yield(t)) => {
                ctx.warp.pc = t;
                return StepResult::Yield;
            }
            Ok(Flow::Blocked(r)) => return StepResult::Blocked(r),
            Ok(Flow::Exit) => return StepResult::Exit,
            Err(e) => return StepResult::Error(e),
        }
    }
    StepResult::Continue
}
