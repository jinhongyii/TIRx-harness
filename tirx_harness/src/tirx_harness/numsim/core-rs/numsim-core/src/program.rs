//! The `Program` bytecode: the single artifact lowering (W1) produces and
//! both backends execute.
//!
//! # Shape
//!
//! A [`Module`] is a sequence of kernels ([`Program`]) launched in order;
//! kernels bind host arguments by [`ParamSlot::name`]. A [`Program`] is a flat
//! `Vec<Instr>` with *structured* control flow whose jump targets are
//! precomputed [`Pc`]s, plus side tables (consts, sites, regs, buffers, ops,
//! wait-until predicate sub-programs, strings, layouts).
//!
//! JSON uses serde's externally tagged form (`{"Load": {...}}`, `"Else"`), so
//! the Python lowering can emit it directly; binary is postcard.
//!
//! # Contract decisions (see README "Contract decisions")
//!
//! * **Granularity (W1 Q1):** dedicated variants for every family with
//!   engine-visible semantics (memory, mbarrier, barriers, fences, atomics,
//!   async copies, TMA, tcgen05, warp collectives, control flow); one generic
//!   [`Instr::Ptx`] for the pure-register ALU tail (`mov/pack/unpack/cvt/fma/
//!   ex2/setp/lop3/prmt/mma.sync/descriptor encoders/...`), backed by the
//!   interned [`Program::ops`] table and resolved to an oplib function once
//!   at load. Checkers never decode PTX strings.
//! * **Sites in a parallel array.** `code_sites[pc]` is the site of
//!   `code[pc]` (`SiteId::NONE` for pure ALU instructions). This is
//!   equivalent to W1's per-variant `site` field but keeps one uniform rule
//!   for handlers and codegen (`ctx.site()`).
//! * **Operands** are `Reg | Const` (W1); constants are interned in
//!   `Program::consts` and broadcast.
//! * **Wide/vector values are one register** with `Ty{elem, lanes}` (W1 Q4).
//! * **Addresses.** `Load/Store` are buffer-relative (`buf` + element offset,
//!   exact OOB, precise `(alloc, range)`); every other memory-touching
//!   instruction takes an *address value* operand (`AddrOf`, `Cvta`, `Mapa`
//!   produce them) plus an [`AddrSpace`].
//! * **No guard predicate** except on `Ptx` (whose oplib function needs PTX
//!   preserve-dst semantics); everything else uses `If`.
//! * **`may_block` is a method on `Instr`** derived from the variant (W1 Q10).
//! * **Effectful CUDA helpers are lowered** to primitive instructions; only
//!   pure reviewed helpers become `Ptx` ops.
//!
//! # SUPPORTED_OPS.md family -> Instr map
//!
//! | SUPPORTED_OPS family | Instr |
//! | --- | --- |
//! | destination_passing_arithmetic, register_* (arith, minmax, unary, bit ops, lop3, prmt, shf, bfe/bfi, dp2a/dp4a, sad, mul24, mad.wide, clmad, set_packed, spdecompress), register_comparison_selection (setp/set/selp/slct/testp), register_conversion (cvt*), register_move (mov.pack/unpack) | `Ptx` (TIR-level arithmetic: `Unary/Binary/Ternary/Compare/Select/Cast/Mov`) |
//! | pure_scalar conversions/bit casts/float2 helpers, runtime_instr_desc, tcgen05/wgmma descriptor encoders, clusterlaunchcontrol_query_*, validated_cuda_helper (pure helpers), warp_matrix_instruction (mma.sync), warp_collective movmatrix/match, cache_policy (createpolicy), ptx_cache_hint (prefetch/applypriority, ordering only) | `Ptx` |
//! | pure_scalar shfl/any/ballot/reduce/elect, warp_collective (shfl, vote, redux, elect) | `Shfl`, `Vote`, `Redux`, `Elect` |
//! | warp_query (activemask), thread_rank, mov_sreg, clock64 | `ReadSpecial` |
//! | pure_scalar iket_* / printf (synchronization) | `Nop` (deterministic representative) |
//! | raw_memory ld/st/ld.v/ld.v256/ldu/readonly, memory ldg | `Load`/`Store` (buffer form) or `LoadAddr`/`StoreAddr` |
//! | raw_memory ldmatrix, memory stmatrix | `LdMatrix`, `StMatrix` |
//! | raw_memory discard | `Discard` |
//! | atomic_bulk_memory atom/red/atomic_add/atomic_cas (scalar, vec, half, b128, bitbucket) | `Atom` |
//! | atomic_bulk_memory cp.async.bulk / cp.reduce.async.bulk (g2s, s2g, s2c, multicast) | `BulkCopy` |
//! | atomic_bulk_memory st.bulk | `StBulk` |
//! | atomic_bulk_memory cp_async_mbarrier_arrive | `CpAsyncMbarArrive` |
//! | async_copy cp.async ca/cg (+src_size/ignore_src) | `CpAsync` |
//! | async_copy commit/wait (cp.async and bulk) | `AsyncCommit`, `AsyncWait` |
//! | async_copy st.async / red.async | `StAsync` |
//! | raw_tma (all tensor g2s/s2g/reduce/prefetch, im2col, multicast, overrides) | `Tma` |
//! | ptx_address cvta/isspacep/mapa/getctarank, cvta_generic_to_shared, smem_addr_from_uint64 | `Cvta`, `Isspacep`, `Mapa`, `GetCtaRank` |
//! | ptx_address_wrapper (addr), `address_of` | `AddrOf` |
//! | raw_tensor_map_replace | `TensorMapReplace` |
//! | raw_tensor_map_fence | `Fence{TensormapRelease/Acquire}`, `TensorMapCopyFence` |
//! | synchronization bar/barrier sync/arrive/red, cta_sync, warpgroup_sync, syncthreads_and/or, cta_reduce | `Barrier` |
//! | synchronization warp_sync, bar_warp_sync | `WarpSync` |
//! | synchronization barrier.cluster arrive/wait, cluster_sync | `ClusterArrive`, `ClusterWait` |
//! | synchronization grid_sync | `GridSync` |
//! | synchronization mbarrier_* | `MbarInit`, `MbarInval`, `MbarArrive`, `MbarTx`, `MbarTestWait`, `MbarWait`, `MbarQuery` |
//! | synchronization fence/fence_proxy/fence_mbarrier_init/thread_fence | `Fence` |
//! | synchronization setmaxnreg | `SetMaxNReg` |
//! | synchronization griddepcontrol | `GridDepControl` |
//! | synchronization clusterlaunchcontrol_try_cancel | `ClcTryCancel` |
//! | synchronization nano_sleep | `Nop` |
//! | assertion trap_when_assert_failed, AssertStmt | `Assert` |
//! | wait_until | `WaitUntil` + `Program::preds` |
//! | tcgen_control alloc/dealloc/relinquish/commit/wait/fence | `TcgenAlloc`, `TcgenDealloc`, `TcgenRelinquish`, `TcgenCommit`, `TcgenWait`, `Fence{Tcgen05*}` |
//! | raw_tcgen ld/st (red, split, spcompress) | `TcgenLd`, `TcgenSt` |
//! | raw_tcgen_copy | `TcgenCp` |
//! | raw_tcgen_mma (all variants) | `TcgenMma` |
//! | CUDA Tile Primitives | `Tile` (provisional; W1 prefers TVM dispatch to PTX-level IR) |
//! | architecture_rejection (wgmma), unreviewed_target_ptx (fabric, multimem) | not representable; lowering rejects (`Unsupported` in non-strict mode) |

use crate::arena::Space;
use crate::dtype::{Dtype, Ty};
use crate::site::{SiteId, SiteInfo};
use crate::sync::async_group::Domain;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Serialization format version of [`Module`]/[`Program`]. Bump on any
/// change to the types in this file.
pub const FORMAT_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Indices
// ---------------------------------------------------------------------------

macro_rules! index_type {
    ($(#[$m:meta])* $name:ident, $prefix:literal) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub struct $name(pub u32);
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($prefix, "{}"), self.0)
            }
        }
    };
}

index_type!(
    /// A per-warp register (index into `Program::regs`).
    Reg, "r");
index_type!(
    /// A declared buffer (index into `Program::buffers`).
    Buf, "b");
index_type!(
    /// An instruction index into `Program::code`.
    Pc, "@");
index_type!(
    /// Index into `Program::consts`.
    ConstId, "k");
index_type!(
    /// Index into `Program::strings`.
    StrId, "str");
index_type!(
    /// Index into `Program::layouts`.
    LayoutId, "L");
index_type!(
    /// Index into `Program::preds` (wait_until sub-programs).
    PredId, "P");
index_type!(
    /// Index into `Program::ops` (generic `Ptx` op table).
    OpId, "op");
index_type!(
    /// Index into `Program::host_abi`.
    ParamId, "param");

/// A source operand: a register or an interned constant (broadcast to all lanes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Operand {
    Reg(Reg),
    Const(ConstId),
}

impl From<Reg> for Operand {
    fn from(r: Reg) -> Operand {
        Operand::Reg(r)
    }
}

impl From<ConstId> for Operand {
    fn from(c: ConstId) -> Operand {
        Operand::Const(c)
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Reg(r) => write!(f, "{r}"),
            Operand::Const(c) => write!(f, "{c}"),
        }
    }
}

/// An interned constant: raw bits of `ty` (floats as bit patterns; vector
/// lanes packed, element 0 low). Up to 128 bits; wider vector constants are
/// built with `Ptx` pack ops.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Const {
    pub ty: Ty,
    pub bits: u128,
}

// ---------------------------------------------------------------------------
// Memory vocabulary
// ---------------------------------------------------------------------------

/// State space named by an instruction (PTX spelling). Allocations live in
/// [`Space`]; `Generic` and `SharedCluster` resolve at execution time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AddrSpace {
    Generic,
    Global,
    /// `.shared::cta` (u32 window offset of the executing CTA).
    Shared,
    /// `.shared::cluster` (u32, see `arena::addr`).
    SharedCluster,
    Local,
    Param,
    Const,
    /// Tensor memory (`taddr` = lane<<16 | column).
    Tmem,
}

impl AddrSpace {
    pub const fn name(self) -> &'static str {
        match self {
            AddrSpace::Generic => "generic",
            AddrSpace::Global => "global",
            AddrSpace::Shared => "shared",
            AddrSpace::SharedCluster => "shared::cluster",
            AddrSpace::Local => "local",
            AddrSpace::Param => "param",
            AddrSpace::Const => "const",
            AddrSpace::Tmem => "tmem",
        }
    }
}

/// Memory-ordering semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Sem {
    #[default]
    Weak,
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    Sc,
    /// `.volatile` (treated as relaxed.sys by checkers).
    Volatile,
    Mmio,
}

/// Memory-ordering scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum Scope {
    Cta,
    Cluster,
    #[default]
    Gpu,
    Sys,
}

/// Memory proxy an access goes through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Proxy {
    #[default]
    Generic,
    /// TMA, cp.async.bulk, tcgen05, st.async.
    Async,
    /// TMA reading a tensor map.
    TensorMap,
    /// `ld.global.nc` / readonly proxy.
    ReadOnly,
    /// tcgen05 access to tensor memory.
    Tcgen,
}

/// Cache operator (ordering-only).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum CacheOp {
    #[default]
    Default,
    Ca,
    Cg,
    Cs,
    Lu,
    Cv,
    Wb,
    Wt,
}

/// Eviction priority (ordering-only).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Evict {
    #[default]
    Normal,
    First,
    Last,
    Unchanged,
    NoAllocate,
}

/// Memory-instruction modifiers. Only `nc` (proxy) and `uniform` (ldu
/// requires warp-uniform addresses) change engine behaviour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct MemMods {
    pub cache: CacheOp,
    pub evict: Evict,
    /// `.L2::64B/128B/256B`, 0 = none.
    pub l2_prefetch: u16,
    /// `.L2::cache_hint` policy operand.
    pub policy: Option<Operand>,
    pub nc: bool,
    pub uniform: bool,
}

// ---------------------------------------------------------------------------
// TIR-level ALU vocabulary (the PTX ALU tail is `Instr::Ptx`)
// ---------------------------------------------------------------------------

/// Rounding modifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Rounding {
    /// TIR/C semantics: round-to-nearest-even for float results,
    /// truncation toward zero for float->int.
    #[default]
    Default,
    Rn,
    Rz,
    Rm,
    Rp,
    Rna,
    Rs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnOp {
    Neg,
    Abs,
    /// Bitwise not (logical not for `Pred`).
    Not,
    Sqrt,
    Rsqrt,
    Exp,
    Exp2,
    Log,
    Log2,
    Sin,
    Cos,
    Tanh,
    Floor,
    Ceil,
    Round,
    Trunc,
    Popcount,
    Clz,
    IsNan,
    IsInf,
    IsFinite,
}

/// TIR binary ops. Integer division/modulo follow TIR (`Div`/`Mod` truncate
/// like C; `FloorDiv`/`FloorMod` floor).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    FloorDiv,
    FloorMod,
    Min,
    Max,
    And,
    Or,
    Xor,
    Shl,
    /// Arithmetic for signed, logical for unsigned.
    Shr,
    Pow,
    Atan2,
    Copysign,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TerOp {
    /// Fused multiply-add, single rounding.
    Fma,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Special registers (`ReadSpecial`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecialReg {
    LaneId,
    /// `tid / 32` within the CTA.
    WarpInCta,
    /// `warp_in_cta / 4`.
    WarpgroupInCta,
    /// Linear thread index within the CTA (`thread_rank`).
    ThreadInCta,
    Tid(Axis),
    NTid(Axis),
    CtaId(Axis),
    NCtaId(Axis),
    /// Linear CTA index in the grid.
    CtaLinear,
    ClusterId(Axis),
    NClusterId(Axis),
    ClusterLinear,
    ClusterCtaId(Axis),
    ClusterNCtaId(Axis),
    ClusterCtaRank,
    ClusterNCtaRank,
    LaneMaskEq,
    LaneMaskLt,
    LaneMaskLe,
    LaneMaskGt,
    LaneMaskGe,
    ActiveMask,
    SmId,
    NSmId,
    GridId,
    /// Deterministic representative: per-warp monotone counter.
    Clock,
    Clock64,
    GlobalTimer,
    DynamicSmemSize,
    TotalSmemSize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Axis {
    X,
    Y,
    Z,
}

// ---------------------------------------------------------------------------
// Warp collectives
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShflMode {
    Idx,
    Up,
    Down,
    Bfly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoteMode {
    All,
    Any,
    Uni,
    Ballot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReduxOp {
    Add,
    Min,
    Max,
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatrixShape {
    M8N8,
    M8N16,
    M16N8,
    M16N16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatrixFmt {
    B16,
    B8,
    /// `.b8x16.b6x16_p32`
    B6x16P32,
    /// `.b8x16.b4x16_p64`
    B4x16P64,
    /// `.s8.s4`
    S8S4,
}

// ---------------------------------------------------------------------------
// Async / TMA / sync vocabulary
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AtomOp {
    Add,
    Min,
    Max,
    Inc,
    Dec,
    And,
    Or,
    Xor,
    Exch,
    Cas,
}

/// How a bulk/tensor async copy reports completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BulkCompletion {
    /// `.mbarrier::complete_tx::bytes [mbar]`.
    Mbarrier { mbar: Operand, space: AddrSpace },
    /// `.bulk_group`.
    Group,
}

/// `cp.async.bulk` / `cp.reduce.async.bulk` (non-tensor).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BulkCopyArgs {
    pub dst: Operand,
    pub dst_space: AddrSpace,
    pub src: Operand,
    pub src_space: AddrSpace,
    pub size: Operand,
    pub completion: BulkCompletion,
    /// `.multicast::cluster` CTA mask.
    pub multicast: Option<Operand>,
    pub reduce: Option<(AtomOp, Dtype)>,
    pub mods: MemMods,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TmaDir {
    /// global -> shared
    Load,
    /// shared -> global
    Store,
    /// shared -(op)-> global
    Reduce(AtomOp),
    /// `cp.async.bulk.prefetch.tensor` (ordering only; still validates the map).
    Prefetch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TmaMode {
    Tile,
    Im2col,
    Im2colW,
    Im2colW128,
    Im2colNoOffs,
    TileGather4,
    TileScatter4,
}

/// A tensor-map field (`tensormap.replace`, per-instruction overrides).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TmapField {
    GlobalAddress,
    Rank,
    BoxDim,
    GlobalDim,
    GlobalStride,
    ElementStride,
    ElemType,
    InterleaveLayout,
    SwizzleMode,
    SwizzleAtomicity,
    FillMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TmapOverride {
    pub field: TmapField,
    pub ord: Option<u8>,
    pub value: Operand,
    /// `_b8`/`_b16` spelling width, 0 = n/a.
    pub elem_bits: u8,
}

/// `cp.async.bulk.tensor` / `cp.reduce.async.bulk.tensor` / tensor prefetch.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TmaArgs {
    pub dir: TmaDir,
    pub mode: TmaMode,
    /// Address of the 128-byte tensor map (`AddrOf` of a param/global/shared buffer).
    pub tmap: Operand,
    pub tmap_space: AddrSpace,
    /// Innermost first, 1..=5.
    pub coords: Vec<Operand>,
    pub im2col_offsets: Vec<Operand>,
    /// Shared-memory side address.
    pub smem: Operand,
    pub smem_space: AddrSpace,
    pub completion: BulkCompletion,
    pub multicast: Option<Operand>,
    /// `.cta_group::1/2`, 0 = unspecified.
    pub cta_group: u8,
    pub overrides: Vec<TmapOverride>,
    pub mods: MemMods,
}

/// `st.async` / `red.async` into a (remote) CTA's shared memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StAsyncArgs {
    pub ty: Ty,
    pub value: Operand,
    pub addr: Operand,
    pub mbar: Operand,
    pub red: Option<AtomOp>,
    pub sem: Sem,
    pub scope: Scope,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BarRedOp {
    Popc,
    And,
    Or,
}

/// What a named-barrier instruction does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BarKind {
    /// `bar.sync` / `barrier.sync` / `__syncthreads`.
    Sync,
    /// `bar.arrive`.
    Arrive,
    /// `bar.red.{popc,and,or}` (also syncthreads_and/or, cta_reduce):
    /// reduces `pred` and writes `dst`.
    Red { op: BarRedOp, pred: Operand, dst: Reg },
}

/// mbarrier wait phase argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhaseArg {
    /// State token from an earlier arrive.
    State(Operand),
    /// `.parity`.
    Parity(Operand),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WaitKind {
    Test,
    Try,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TxOp {
    Expect,
    Complete,
}

/// `mbarrier.arrive` family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MbarArriveArgs {
    pub mbar: Operand,
    pub space: AddrSpace,
    /// None = 1.
    pub count: Option<Operand>,
    pub expect_tx: Option<Operand>,
    pub drop: bool,
    pub no_complete: bool,
    pub sem: Sem,
    pub scope: Scope,
    pub multicast: Option<Operand>,
    /// State-token destination (None = sink).
    pub state: Option<Reg>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MbarQueryOp {
    PendingCount { state: Operand },
    /// `layout_v1` is the layout the kernel declared for this barrier
    /// (`true` = `.layout::v1`, 511-arrival limit); the query reports
    /// whether the live object matches (W3-4).
    CheckLayout { mbar: Operand, space: AddrSpace, layout_v1: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FenceKind {
    /// membar / fence.sc / fence.acq_rel / __threadfence*.
    Thread,
    MbarrierInit,
    ProxyAsync(Option<AddrSpace>),
    ProxyAlias,
    TensormapRelease,
    TensormapAcquire { addr: Operand, space: AddrSpace },
    Tcgen05Before,
    Tcgen05After,
}

// ---------------------------------------------------------------------------
// tcgen05 vocabulary
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TcShape {
    S32x32b,
    S16x64b,
    S16x128b,
    S16x256b,
    S16x32bx2 { split_off: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TcgenLdArgs {
    pub dsts: Vec<Reg>,
    pub taddr: Operand,
    pub shape: TcShape,
    pub num: u16,
    pub pack: bool,
    /// `.red.{min,max}`: op and reduced destinations.
    pub red: Option<(ReduxOp, Vec<Reg>)>,
    pub spcompress: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TcgenStArgs {
    pub srcs: Vec<Operand>,
    pub taddr: Operand,
    pub shape: TcShape,
    pub num: u16,
    pub unpack: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TcgenCpArgs {
    pub taddr: Operand,
    pub sdesc: Operand,
    /// Shape as (rows, bits): `.128x256b` = (128, 256), `.4x256b`, ...
    pub rows: u16,
    pub bits: u16,
    /// Multicast code (`warpx2::02_13` etc.), 0 = none.
    pub multicast: u8,
    /// 0 none, 6 = b6x16_p32, 4 = b4x16_p64.
    pub decompress_bits: u8,
    pub cta_group: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TcA {
    /// Shared-memory descriptor (`_ss`).
    Smem(Operand),
    /// Tensor-memory address (`_ts`).
    Tmem(Operand),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TcMmaKind {
    F16,
    Tf32,
    F8f6f4,
    I8,
    MxF8f6f4,
    MxF4,
    MxF4Nvf4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum CollectorOp {
    #[default]
    None,
    Fill,
    Use,
    LastUse,
    Discard,
}

/// `tcgen05.mma` (all ss/ts/ws/sp/block-scale/collector/lut/ashift forms).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TcgenMmaArgs {
    pub kind: TcMmaKind,
    pub cta_group: u8,
    pub d: Operand,
    pub a: TcA,
    pub b_desc: Operand,
    pub idesc: Operand,
    pub enable_input_d: Operand,
    pub ws: bool,
    pub ws_b_buffer: u8,
    /// (scale_A taddr, scale_B taddr, block size 16/32).
    pub block_scale: Option<(Operand, Operand, u8)>,
    pub scale_input_d: Option<Operand>,
    pub sparse_meta: Option<Operand>,
    pub disable_output_lane: Vec<Operand>,
    pub collector_a: CollectorOp,
    pub collector_b: CollectorOp,
    pub ashift: bool,
    /// `lut_b` / `ti16` qualifiers, interpreted by oplib.
    pub variant: Option<StrId>,
}

// ---------------------------------------------------------------------------
// Tile vocabulary (provisional, W1 B.7)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecScope {
    Thread,
    Warp,
    Warpgroup,
    Cta,
    Cluster,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TileOp {
    Add,
    Sub,
    Mul,
    Fdiv,
    Fma,
    Max,
    Min,
    Maximum,
    Exp,
    Exp2,
    Log2,
    Sqrt,
    Reciprocal,
    Silu,
    Cast,
    Copy,
    CopyAsync,
    Fill,
    Zero,
    Sum,
    Gemm,
    GemmAsync,
    PermuteLayout,
}

/// A tile region: buffer + base element offset + element map.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TileArg {
    Region { buf: Buf, base: Operand, map: LayoutId },
    Frag { first: Reg, map: LayoutId },
    Scalar(Operand),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileArgs {
    pub op: TileOp,
    pub scope: ExecScope,
    /// Destination first, then sources in TIRx call order.
    pub args: Vec<TileArg>,
    pub axes: Vec<u8>,
    pub completion: Option<BulkCompletion>,
}

/// Element map computed at lowering time (W1 B.7): `entries[lane * slots +
/// slot]` = element offset relative to the region base, or -1 (none).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileLayout {
    pub lanes: u32,
    pub slots: u32,
    pub entries: Vec<i64>,
}

// ---------------------------------------------------------------------------
// Instr
// ---------------------------------------------------------------------------

/// One instruction.
///
/// # Control flow (structured; targets precomputed)
///
/// Masks: `active` (current), per-frame `entry` mask, and `live` (not
/// exited). All frame restores intersect with `live` and with the innermost
/// loop's non-broken lanes.
/// * `If{cond, else_pc, end_pc, elect}`: push; `active &= cond`; if empty jump
///   to `else_pc` (the `Else`, or `end_pc` when there is none). `elect`
///   marks a condition derived from `elect.sync` (checkers attribute the
///   region to the elected lane).
/// * `Else{end_pc}`: `active = entry & !taken`; if empty jump to `end_pc`.
/// * `EndIf`: pop.
/// * `LoopBegin{end_pc}`: push a loop frame (entry mask, iteration 0,
///   empty break/continue masks). Its site is the loop-enter event.
/// * `LoopIf{cond, end_pc}`: `active &= cond`; lanes leaving stay inactive
///   until the frame pops; if none remain pop and jump to `end_pc + 1`.
/// * `Break`: active lanes leave the innermost loop (`break |= active;
///   active = 0`). `Continue`: `cont |= active; active = 0` until `LoopEnd`.
/// * `LoopEnd{head_pc}`: `active |= cont`; iteration += 1 (budget ->
///   incomplete at the LoopBegin site); quantum yield; spin parking; jump to
///   `head_pc` (the instruction after `LoopBegin` that starts the condition
///   computation).
/// * `Exit`: active lanes retire permanently (removed from `live`).
///
/// # Blocking
///
/// Instructions with [`Instr::may_block`] may return `Flow::Blocked`; the
/// scheduler re-executes the same pc. Lowering never duplicates them and
/// they are the only effectful work in their instruction.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Instr {
    // ----- control -----
    /// No effect (profiling markers, printf, nanosleep representatives).
    Nop,
    If { cond: Operand, else_pc: Pc, end_pc: Pc, elect: bool },
    Else { end_pc: Pc },
    EndIf,
    LoopBegin { end_pc: Pc },
    LoopIf { cond: Operand, end_pc: Pc },
    LoopEnd { head_pc: Pc },
    Break,
    Continue,
    Exit,
    /// Error finding for every active lane where `cond` is false.
    Assert { cond: Operand, msg: Option<StrId> },
    /// Fail closed (incomplete) if any lane reaches it (`strict=False` lowering).
    Unsupported { reason: StrId },

    // ----- registers / TIR arithmetic -----
    Mov { dst: Reg, src: Operand },
    ReadSpecial { dst: Reg, sreg: SpecialReg },
    /// Read a scalar host parameter (uniform).
    ReadParam { dst: Reg, slot: ParamId },
    Unary { op: UnOp, ty: Ty, dst: Reg, a: Operand },
    Binary { op: BinOp, ty: Ty, dst: Reg, a: Operand, b: Operand },
    Ternary { op: TerOp, ty: Ty, dst: Reg, a: Operand, b: Operand, c: Operand },
    /// Predicate result.
    Compare { op: CmpOp, ty: Ty, dst: Reg, a: Operand, b: Operand },
    /// `cond ? a : b` (TIR `Select` / `if_then_else`).
    Select { ty: Ty, dst: Reg, cond: Operand, a: Operand, b: Operand },
    /// TIR `Cast` (C semantics by default; `reinterpret` uses `Mov`).
    Cast { from: Ty, to: Ty, dst: Reg, src: Operand, rnd: Rounding, sat: bool },
    /// Generic pure PTX / CUDA-helper op from `Program::ops`.
    /// `pred`: PTX guard; lanes where it is false do not execute and keep
    /// their destinations if `keep_dst`, else get a zero representative.
    Ptx { op: OpId, dsts: Vec<Reg>, srcs: Vec<Operand>, pred: Option<Operand>, keep_dst: bool },
    /// Dynamic index into a register-promoted local array `base..base+len`
    /// (all same type). OOB = error finding.
    LoadRegIndexed { dst: Reg, base: Reg, len: u32, idx: Operand },
    StoreRegIndexed { base: Reg, len: u32, idx: Operand, value: Operand },

    // ----- warp collectives -----
    Shfl { mode: ShflMode, ty: Ty, dst: Reg, dst_pred: Option<Reg>, src: Operand, lane: Operand, clamp: Operand, membermask: Operand },
    Vote { mode: VoteMode, dst: Reg, pred: Operand, membermask: Operand },
    Redux { op: ReduxOp, ty: Ty, dst: Reg, src: Operand, membermask: Operand },
    /// `elect.sync`: `dst_pred` = 1 in the elected lane.
    Elect { dst_pred: Reg, dst_lane: Option<Reg>, membermask: Operand },
    WarpSync { membermask: Operand },
    LdMatrix { dsts: Vec<Reg>, addr: Operand, space: AddrSpace, shape: MatrixShape, num: u8, trans: bool, fmt: MatrixFmt },
    StMatrix { srcs: Vec<Operand>, addr: Operand, space: AddrSpace, shape: MatrixShape, num: u8, trans: bool },

    // ----- memory -----
    /// `dst = buf[offset]`; `offset` in elements of `buffers[buf].dtype`
    /// (byte address = offset * elem bytes); `ty` may differ (reinterpret,
    /// vector loads). Space from the buffer declaration.
    Load { ty: Ty, dst: Reg, buf: Buf, offset: Operand, sem: Sem, scope: Scope, mods: MemMods },
    Store { ty: Ty, buf: Buf, offset: Operand, value: Operand, sem: Sem, scope: Scope, mods: MemMods },
    /// Raw-address load (`addr` value in `space`'s encoding).
    LoadAddr { ty: Ty, dst: Reg, addr: Operand, space: AddrSpace, sem: Sem, scope: Scope, mods: MemMods },
    StoreAddr { ty: Ty, addr: Operand, space: AddrSpace, value: Operand, sem: Sem, scope: Scope, mods: MemMods },
    /// `dst` (u64 generic) = address of `buf[offset]`.
    AddrOf { dst: Reg, buf: Buf, offset: Operand },
    /// `atom` (dst Some) / `red` / bitbucket (dst None). Vector `ty` = one
    /// RMW per lane-element. `cmp` only for Cas.
    Atom { op: AtomOp, ty: Ty, dst: Option<Reg>, addr: Operand, space: AddrSpace, value: Operand, cmp: Option<Operand>, sem: Sem, scope: Scope, ftz: bool },
    StBulk { addr: Operand, space: AddrSpace, size: Operand },
    /// Contents become undefined (validity cleared).
    Discard { addr: Operand, space: AddrSpace, size: u32 },
    /// `cvta`: `to_generic` = space -> generic.
    Cvta { dst: Reg, src: Operand, space: AddrSpace, to_generic: bool },
    Isspacep { dst: Reg, src: Operand, space: AddrSpace },
    Mapa { dst: Reg, src: Operand, rank: Operand, space: AddrSpace },
    GetCtaRank { dst: Reg, src: Operand, space: AddrSpace },

    // ----- async copies -----
    CpAsync { dst: Operand, src: Operand, cp_size: u8, src_size: Option<Operand>, ignore_src: Option<Operand>, mods: MemMods },
    /// `cp.async.commit_group` / `cp.async.bulk.commit_group`.
    AsyncCommit { domain: Domain },
    /// `*.wait_group{.read} n` (`cp.async.wait_all` = commit + wait 0).
    AsyncWait { domain: Domain, n: u32, read: bool },
    CpAsyncMbarArrive { mbar: Operand, space: AddrSpace, noinc: bool },
    BulkCopy(BulkCopyArgs),
    Tma(Box<TmaArgs>),
    StAsync(StAsyncArgs),
    TensorMapReplace { tmap: Operand, space: AddrSpace, field: TmapField, ord: Option<u8>, value: Operand },
    TensorMapCopyFence { dst: Operand, src: Operand, size: u32, scope: Scope },

    // ----- synchronization -----
    Barrier { kind: BarKind, id: Operand, count: Option<Operand>, aligned: bool },
    ClusterArrive { sem: Sem, aligned: bool },
    ClusterWait { acquire: bool, aligned: bool },
    GridSync,
    MbarInit { mbar: Operand, space: AddrSpace, count: Operand, layout_v1: bool },
    MbarInval { mbar: Operand, space: AddrSpace },
    MbarArrive(MbarArriveArgs),
    MbarTx { op: TxOp, mbar: Operand, space: AddrSpace, bytes: Operand, multicast: Option<Operand>, scope: Scope },
    /// Non-blocking `test_wait` / `try_wait`; `dst` = ready predicate.
    MbarTestWait { kind: WaitKind, mbar: Operand, space: AddrSpace, phase: PhaseArg, sem: Sem, scope: Scope, dst: Option<Reg> },
    /// Blocking wait (`cuda.mbarrier_wait*`).
    MbarWait { mbar: Operand, space: AddrSpace, phase: PhaseArg, sem: Sem, scope: Scope },
    MbarQuery { dst: Reg, op: MbarQueryOp },
    Fence { kind: FenceKind, sem: Sem, scope: Scope },
    SetMaxNReg { inc: bool, count: u32 },
    /// Block until predicate `pred` accepts the word at `addr`; then `dst`
    /// holds the accepted value. `captures` are snapshotted at issue.
    WaitUntil { dst: Reg, addr: Operand, ty: Ty, space: AddrSpace, sem: Sem, scope: Scope, pred: PredId, captures: Vec<Reg> },
    GridDepControl { launch_dependents: bool },
    ClcTryCancel { resp: Operand, mbar: Operand, multicast: bool },

    // ----- tcgen05 -----
    /// Writes the allocated taddr to shared memory at `dst`.
    TcgenAlloc { dst: Operand, ncols: Operand, cta_group: u8, exclusive: bool },
    TcgenDealloc { taddr: Operand, ncols: Operand, cta_group: u8, exclusive: bool },
    TcgenRelinquish { cta_group: u8 },
    TcgenCommit { mbar: Operand, space: AddrSpace, cta_group: u8, multicast: Option<Operand> },
    TcgenLd(Box<TcgenLdArgs>),
    TcgenSt(Box<TcgenStArgs>),
    /// `wait::ld` (`st = false`) / `wait::st`.
    TcgenWait { st: bool },
    TcgenCp(TcgenCpArgs),
    TcgenMma(Box<TcgenMmaArgs>),

    // ----- tile (provisional) -----
    Tile(Box<TileArgs>),
}

impl Instr {
    /// Stable family name (handler name for codegen, profiles, printing).
    pub fn family(&self) -> &'static str {
        use Instr::*;
        match self {
            Nop => "nop",
            If { .. } => "if",
            Else { .. } => "else",
            EndIf => "endif",
            LoopBegin { .. } => "loop",
            LoopIf { .. } => "loop_if",
            LoopEnd { .. } => "loop_end",
            Break => "break",
            Continue => "continue",
            Exit => "exit",
            Assert { .. } => "assert",
            Unsupported { .. } => "unsupported",
            Mov { .. } => "mov",
            ReadSpecial { .. } => "read_special",
            ReadParam { .. } => "read_param",
            Unary { .. } => "unary",
            Binary { .. } => "binary",
            Ternary { .. } => "ternary",
            Compare { .. } => "compare",
            Select { .. } => "select",
            Cast { .. } => "cast",
            Ptx { .. } => "ptx",
            LoadRegIndexed { .. } => "load_reg_indexed",
            StoreRegIndexed { .. } => "store_reg_indexed",
            Shfl { .. } => "shfl",
            Vote { .. } => "vote",
            Redux { .. } => "redux",
            Elect { .. } => "elect",
            WarpSync { .. } => "warp_sync",
            LdMatrix { .. } => "ldmatrix",
            StMatrix { .. } => "stmatrix",
            Load { .. } => "load",
            Store { .. } => "store",
            LoadAddr { .. } => "load_addr",
            StoreAddr { .. } => "store_addr",
            AddrOf { .. } => "addr_of",
            Atom { .. } => "atom",
            StBulk { .. } => "st_bulk",
            Discard { .. } => "discard",
            Cvta { .. } => "cvta",
            Isspacep { .. } => "isspacep",
            Mapa { .. } => "mapa",
            GetCtaRank { .. } => "getctarank",
            CpAsync { .. } => "cp_async",
            AsyncCommit { .. } => "async_commit",
            AsyncWait { .. } => "async_wait",
            CpAsyncMbarArrive { .. } => "cp_async_mbar_arrive",
            BulkCopy(_) => "bulk_copy",
            Tma(_) => "tma",
            StAsync(_) => "st_async",
            TensorMapReplace { .. } => "tensormap_replace",
            TensorMapCopyFence { .. } => "tensormap_cp_fence",
            Barrier { .. } => "barrier",
            ClusterArrive { .. } => "cluster_arrive",
            ClusterWait { .. } => "cluster_wait",
            GridSync => "grid_sync",
            MbarInit { .. } => "mbar_init",
            MbarInval { .. } => "mbar_inval",
            MbarArrive(_) => "mbar_arrive",
            MbarTx { .. } => "mbar_tx",
            MbarTestWait { .. } => "mbar_test_wait",
            MbarWait { .. } => "mbar_wait",
            MbarQuery { .. } => "mbar_query",
            Fence { .. } => "fence",
            SetMaxNReg { .. } => "setmaxnreg",
            WaitUntil { .. } => "wait_until",
            GridDepControl { .. } => "griddepcontrol",
            ClcTryCancel { .. } => "clc_try_cancel",
            TcgenAlloc { .. } => "tcgen_alloc",
            TcgenDealloc { .. } => "tcgen_dealloc",
            TcgenRelinquish { .. } => "tcgen_relinquish",
            TcgenCommit { .. } => "tcgen_commit",
            TcgenLd(_) => "tcgen_ld",
            TcgenSt(_) => "tcgen_st",
            TcgenWait { .. } => "tcgen_wait",
            TcgenCp(_) => "tcgen_cp",
            TcgenMma(_) => "tcgen_mma",
            Tile(_) => "tile",
        }
    }

    /// May return `Flow::Blocked` (scheduling point). Derived from the variant.
    pub fn may_block(&self) -> bool {
        use Instr::*;
        match self {
            Barrier { kind, .. } => !matches!(kind, BarKind::Arrive),
            ClusterWait { .. }
            | GridSync
            | MbarWait { .. }
            | AsyncWait { .. }
            | TcgenAlloc { .. }
            | TcgenWait { .. }
            | WaitUntil { .. }
            | WarpSync { .. } => true,
            SetMaxNReg { inc, .. } => *inc,
            Tile(t) => matches!(t.op, TileOp::Gemm | TileOp::Copy | TileOp::Sum) || t.scope != ExecScope::Thread,
            _ => false,
        }
    }

    /// True if executing it is "progress" for spin parking (anything that
    /// writes memory, commits a sync transition or issues async work). Pure
    /// register ops and failed polls are not progress.
    pub fn is_progress(&self) -> bool {
        use Instr::*;
        !matches!(
            self,
            Nop | If { .. }
                | Else { .. }
                | EndIf
                | LoopBegin { .. }
                | LoopIf { .. }
                | LoopEnd { .. }
                | Break
                | Continue
                | Mov { .. }
                | ReadSpecial { .. }
                | ReadParam { .. }
                | Unary { .. }
                | Binary { .. }
                | Ternary { .. }
                | Compare { .. }
                | Select { .. }
                | Cast { .. }
                | Ptx { .. }
                | LoadRegIndexed { .. }
                | StoreRegIndexed { .. }
                | Shfl { .. }
                | Vote { .. }
                | Redux { .. }
                | Elect { .. }
                | Load { .. }
                | LoadAddr { .. }
                | AddrOf { .. }
                | Cvta { .. }
                | Isspacep { .. }
                | Mapa { .. }
                | GetCtaRank { .. }
                | MbarTestWait { .. }
                | MbarQuery { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// Program-level tables
// ---------------------------------------------------------------------------

/// Register declaration.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegDecl {
    pub ty: Ty,
    pub name: Option<String>,
    /// Static hint: every write is warp-uniform. Engines must be correct
    /// when ignoring it.
    pub uniform: bool,
}

/// A small expression over scalar host parameters (dynamic extents).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DimExpr {
    Const(i64),
    Param(ParamId),
    Add(Box<DimExpr>, Box<DimExpr>),
    Sub(Box<DimExpr>, Box<DimExpr>),
    Mul(Box<DimExpr>, Box<DimExpr>),
    FloorDiv(Box<DimExpr>, Box<DimExpr>),
    CeilDiv(Box<DimExpr>, Box<DimExpr>),
    Min(Box<DimExpr>, Box<DimExpr>),
    Max(Box<DimExpr>, Box<DimExpr>),
}

impl DimExpr {
    /// Evaluate with `param(slot)` giving scalar parameter values.
    pub fn eval(&self, param: &dyn Fn(ParamId) -> Option<i64>) -> Option<i64> {
        use DimExpr::*;
        let bin = |a: &DimExpr, b: &DimExpr| Some((a.eval(param)?, b.eval(param)?));
        Some(match self {
            Const(v) => *v,
            Param(p) => param(*p)?,
            Add(a, b) => bin(a, b).map(|(x, y)| x.checked_add(y))??,
            Sub(a, b) => bin(a, b).map(|(x, y)| x.checked_sub(y))??,
            Mul(a, b) => bin(a, b).map(|(x, y)| x.checked_mul(y))??,
            FloorDiv(a, b) => {
                let (x, y) = bin(a, b)?;
                if y == 0 {
                    return None;
                }
                x.div_euclid(y) - if y < 0 && x.rem_euclid(y) != 0 { 1 } else { 0 }
            }
            CeilDiv(a, b) => {
                let (x, y) = bin(a, b)?;
                if y == 0 {
                    return None;
                }
                -((-x).div_euclid(y))
            }
            Min(a, b) => bin(a, b).map(|(x, y)| x.min(y))?,
            Max(a, b) => bin(a, b).map(|(x, y)| x.max(y))?,
        })
    }
}

/// A declared buffer.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferDecl {
    pub name: String,
    pub space: Space,
    pub dtype: Ty,
    pub shape: Vec<DimExpr>,
    /// Element strides (row-major when empty).
    pub strides: Vec<DimExpr>,
    /// Host parameter backing this buffer (global buffers, tensor maps).
    pub param_slot: Option<ParamId>,
    /// Byte offset of the buffer in its backing: shared-window offset for
    /// shared buffers; offset within `view_of` for views; 0 otherwise.
    pub base: u64,
    /// Total bytes; None = taken from the bound host argument.
    pub byte_len: Option<DimExpr>,
    pub align: u32,
    /// This buffer is a view (DeclBuffer) of another buffer's storage.
    pub view_of: Option<Buf>,
    /// Lowering hint: the buffer holds declared synchronization words
    /// (an `AddrOf` of it reaches a `WaitUntil`). The engine emits
    /// `DeclareWord` for it at launch begin and keeps write history. A
    /// `WaitUntil` on an undeclared word declares it on first use (history
    /// then starts at that point; checkers must treat earlier writes as
    /// unknown).
    pub sync_words: bool,
}

/// Interned generic op: canonical op name + canonical modifier tuple
/// (`tirx.ptx.cvt`, `["rn", "f16x2", "f32"]`). Resolved to an oplib function
/// at load (`oplib::resolve_ptx`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OpKey {
    pub name: String,
    pub mods: Vec<String>,
}

/// A wait_until predicate sub-program (W1 B.8): straight-line code in
/// `code[start..end]` (placed after the main body), allowed instructions:
/// `Mov/Unary/Binary/Ternary/Compare/Select/Cast/Ptx/Load/LoadAddr/
/// LoadRegIndexed`. The engine writes the candidate word into `arg`, runs
/// the range, reads the predicate from `result`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PredProgram {
    pub arg: Reg,
    pub start: Pc,
    pub end: Pc,
    pub result: Reg,
    /// The range contains loads: history verdicts depend on current memory
    /// (racecheck must not treat the earliest accepted write as exact).
    pub reads_memory: bool,
}

/// Launch topology.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Launch {
    /// Grid in CTAs (linear grids use `[n, 1, 1]`).
    pub grid: [DimExpr; 3],
    /// Cluster shape in CTAs (`[1,1,1]` = no clusters).
    pub cluster: [u32; 3],
    /// Threads per CTA (non-multiple of 32: last warp partial).
    pub block: [u32; 3],
    /// Static shared bytes (lowering's pool layout).
    pub static_smem_bytes: u32,
    pub dyn_smem_bytes: DimExpr,
    pub min_blocks_per_sm: Option<u32>,
    pub cooperative: bool,
    /// Per-thread register budget at launch (setmaxnreg `Configure`), 0 = default.
    pub regs_per_thread: u32,
}

/// Concrete launch shape after evaluating `DimExpr`s.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LaunchShape {
    pub grid: [u32; 3],
    pub cluster: [u32; 3],
    pub block: [u32; 3],
    pub smem_bytes: u32,
}

impl LaunchShape {
    pub fn threads_per_cta(&self) -> u32 {
        self.block[0] * self.block[1] * self.block[2]
    }
    pub fn warps_per_cta(&self) -> u32 {
        self.threads_per_cta().div_ceil(32)
    }
    pub fn num_ctas(&self) -> u32 {
        self.grid[0] * self.grid[1] * self.grid[2]
    }
    pub fn ctas_per_cluster(&self) -> u32 {
        self.cluster[0] * self.cluster[1] * self.cluster[2]
    }
    pub fn num_clusters(&self) -> u32 {
        self.num_ctas() / self.ctas_per_cluster().max(1)
    }
    pub fn num_warps(&self) -> u32 {
        self.num_ctas() * self.warps_per_cta()
    }
    /// Live-lane mask of warp `w` of a CTA (partial last warp).
    pub fn warp_lanes(&self, w: u32) -> crate::value::WarpMask {
        let t = self.threads_per_cta();
        crate::value::WarpMask::first_n(t.saturating_sub(w * 32).min(32))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParamKind {
    Buffer,
    Pointer,
    Scalar,
    TensorMap,
}

/// Host-prelude tensor-map encoding facts (`tensormap_encode_tiled`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorMapSpec {
    pub dtype: Dtype,
    pub rank: u8,
    /// Innermost first.
    pub global_dim: Vec<DimExpr>,
    /// Byte strides of dims 1.. .
    pub global_stride: Vec<DimExpr>,
    pub box_dim: Vec<u32>,
    pub element_stride: Vec<u32>,
    pub interleave: u8,
    pub swizzle: u8,
    pub l2_promotion: u8,
    pub oob_fill: u8,
    /// Byte offset into the base buffer for the global address.
    pub base_offset: DimExpr,
}

/// One host parameter, in kernel signature order.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParamSlot {
    /// Canonical name (binding key, unique within a Module).
    pub name: String,
    pub local_name: String,
    pub aliases: Vec<String>,
    pub kind: ParamKind,
    pub dtype: Option<Ty>,
    pub shape: Vec<DimExpr>,
    /// Tensor maps encoded by the host prelude.
    pub tensor_map: Option<TensorMapSpec>,
    /// Implicit tensor map: the buffer parameter it describes.
    pub implicit_base: Option<ParamId>,
    /// Buffer declared for this parameter (Buffer kind: global buffer;
    /// TensorMap kind: 128-byte Param-space buffer usable with `AddrOf`).
    pub buf: Option<Buf>,
}

/// Feature requirements a program declares (engines reject what they lack).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Requirements {
    pub implicit_tmem: bool,
    pub dynamic_tmem_lifecycle: bool,
    pub readonly_proxy: bool,
    pub grid_dependency: bool,
    pub raw_tensor_map_registry: bool,
}

/// One kernel.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Program {
    pub name: String,
    pub code: Vec<Instr>,
    /// `code_sites[pc]` = site of `code[pc]` (`SiteId::NONE` allowed).
    pub code_sites: Vec<SiteId>,
    pub consts: Vec<Const>,
    pub strings: Vec<String>,
    pub sites: Vec<SiteInfo>,
    pub regs: Vec<RegDecl>,
    pub buffers: Vec<BufferDecl>,
    pub ops: Vec<OpKey>,
    pub preds: Vec<PredProgram>,
    pub layouts: Vec<TileLayout>,
    pub topology: Launch,
    pub host_abi: Vec<ParamSlot>,
    /// Target arch (`sm_100a`, ...).
    pub arch: Option<String>,
    pub requirements: Requirements,
    /// Lowering's collected unsupported reasons (`site#N kind: reason`);
    /// non-empty only for `strict=False` lowering.
    pub unsupported: Vec<String>,
}

/// Kernels launched in order, sharing host bindings by name.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Module {
    pub format_version: u32,
    pub kernels: Vec<Program>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProgramError {
    Decode(String),
    Version { found: u32, expected: u32 },
    Invalid { pc: Option<u32>, message: String },
}

impl fmt::Display for ProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProgramError::Decode(m) => write!(f, "program decode error: {m}"),
            ProgramError::Version { found, expected } => {
                write!(f, "program format version {found}, expected {expected}")
            }
            ProgramError::Invalid { pc: Some(pc), message } => write!(f, "invalid program at @{pc}: {message}"),
            ProgramError::Invalid { pc: None, message } => write!(f, "invalid program: {message}"),
        }
    }
}

impl std::error::Error for ProgramError {}

impl Module {
    pub fn new(kernels: Vec<Program>) -> Module {
        Module { format_version: FORMAT_VERSION, kernels }
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        postcard::to_stdvec(self).expect("Module is serializable")
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Module, ProgramError> {
        let m: Module = postcard::from_bytes(bytes).map_err(|e| ProgramError::Decode(e.to_string()))?;
        m.check_version()?;
        Ok(m)
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("Module is serializable")
    }
    pub fn from_json(s: &str) -> Result<Module, ProgramError> {
        let m: Module = serde_json::from_str(s).map_err(|e| ProgramError::Decode(e.to_string()))?;
        m.check_version()?;
        Ok(m)
    }
    fn check_version(&self) -> Result<(), ProgramError> {
        if self.format_version != FORMAT_VERSION {
            return Err(ProgramError::Version { found: self.format_version, expected: FORMAT_VERSION });
        }
        Ok(())
    }
}

impl Program {
    /// Empty program: one CTA of `threads` threads.
    pub fn empty(name: &str, threads: u32) -> Program {
        Program {
            name: name.to_string(),
            code: Vec::new(),
            code_sites: Vec::new(),
            consts: Vec::new(),
            strings: Vec::new(),
            sites: Vec::new(),
            regs: Vec::new(),
            buffers: Vec::new(),
            ops: Vec::new(),
            preds: Vec::new(),
            layouts: Vec::new(),
            topology: Launch {
                grid: [DimExpr::Const(1), DimExpr::Const(1), DimExpr::Const(1)],
                cluster: [1, 1, 1],
                block: [threads, 1, 1],
                static_smem_bytes: 0,
                dyn_smem_bytes: DimExpr::Const(0),
                min_blocks_per_sm: None,
                cooperative: false,
                regs_per_thread: 0,
            },
            host_abi: Vec::new(),
            arch: None,
            requirements: Requirements::default(),
            unsupported: Vec::new(),
        }
    }

    pub fn site_of(&self, pc: Pc) -> SiteId {
        self.code_sites.get(pc.0 as usize).copied().unwrap_or(SiteId::NONE)
    }

    /// First register slot of each register, plus the total slot count as
    /// the last element (`len = regs.len() + 1`). See `crate::value`.
    pub fn reg_slot_offsets(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.regs.len() + 1);
        let mut acc = 0u32;
        for r in &self.regs {
            out.push(acc);
            acc += r.ty.slots();
        }
        out.push(acc);
        out
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        postcard::to_stdvec(self).expect("Program is serializable")
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Program, ProgramError> {
        postcard::from_bytes(bytes).map_err(|e| ProgramError::Decode(e.to_string()))
    }

    /// Structural validation: parallel arrays, table indices, control-flow
    /// nesting/targets and predicate sub-program ranges. Backends may assume
    /// a validated program.
    pub fn validate(&self) -> Result<(), ProgramError> {
        let err = |pc: usize, m: String| ProgramError::Invalid { pc: Some(pc as u32), message: m };
        if self.code_sites.len() != self.code.len() {
            return Err(ProgramError::Invalid {
                pc: None,
                message: format!("code_sites has {} entries for {} instrs", self.code_sites.len(), self.code.len()),
            });
        }
        let n = self.code.len();
        let is = |pc: Pc, f: fn(&Instr) -> bool| (pc.0 as usize) < n && f(&self.code[pc.0 as usize]);
        // Predicate ranges are excluded from the structured walk.
        let mut in_pred = vec![false; n];
        for (i, p) in self.preds.iter().enumerate() {
            if p.start.0 > p.end.0 || p.end.0 as usize > n {
                return Err(ProgramError::Invalid { pc: None, message: format!("pred {i}: bad range") });
            }
            for pc in p.start.0..p.end.0 {
                in_pred[pc as usize] = true;
                let ok = matches!(
                    self.code[pc as usize],
                    Instr::Mov { .. }
                        | Instr::Unary { .. }
                        | Instr::Binary { .. }
                        | Instr::Ternary { .. }
                        | Instr::Compare { .. }
                        | Instr::Select { .. }
                        | Instr::Cast { .. }
                        | Instr::Ptx { .. }
                        | Instr::Load { .. }
                        | Instr::LoadAddr { .. }
                        | Instr::LoadRegIndexed { .. }
                );
                if !ok {
                    return Err(err(pc as usize, format!("instruction not allowed in predicate {i}")));
                }
            }
        }
        let nregs = self.regs.len() as u32;
        let op_ok = |o: &Operand| match o {
            Operand::Reg(r) => r.0 < nregs,
            Operand::Const(c) => (c.0 as usize) < self.consts.len(),
        };
        enum Open {
            If,
            Else,
            Loop,
        }
        let mut stack: Vec<Open> = Vec::new();
        for (pc, ins) in self.code.iter().enumerate() {
            let s = self.code_sites[pc];
            if !s.is_none() && s.0 as usize >= self.sites.len() {
                return Err(err(pc, format!("site {s} out of range")));
            }
            if in_pred[pc] {
                continue;
            }
            match ins {
                Instr::If { cond, else_pc, end_pc, .. } => {
                    if !op_ok(cond) {
                        return Err(err(pc, "bad If cond".into()));
                    }
                    if !is(*end_pc, |i| matches!(i, Instr::EndIf))
                        || !(is(*else_pc, |i| matches!(i, Instr::Else { .. })) || else_pc == end_pc)
                    {
                        return Err(err(pc, "bad If targets".into()));
                    }
                    stack.push(Open::If);
                }
                Instr::Else { end_pc } => {
                    if !matches!(stack.pop(), Some(Open::If)) {
                        return Err(err(pc, "Else without If".into()));
                    }
                    if !is(*end_pc, |i| matches!(i, Instr::EndIf)) {
                        return Err(err(pc, "Else end_pc is not an EndIf".into()));
                    }
                    stack.push(Open::Else);
                }
                Instr::EndIf => {
                    if !matches!(stack.pop(), Some(Open::If) | Some(Open::Else)) {
                        return Err(err(pc, "EndIf without If".into()));
                    }
                }
                Instr::LoopBegin { end_pc } => {
                    if !is(*end_pc, |i| matches!(i, Instr::LoopEnd { .. })) {
                        return Err(err(pc, "LoopBegin end_pc is not a LoopEnd".into()));
                    }
                    stack.push(Open::Loop);
                }
                Instr::LoopIf { cond, end_pc } => {
                    if !op_ok(cond) || !is(*end_pc, |i| matches!(i, Instr::LoopEnd { .. })) {
                        return Err(err(pc, "bad LoopIf".into()));
                    }
                }
                Instr::Break | Instr::Continue => {
                    if !stack.iter().any(|o| matches!(o, Open::Loop)) {
                        return Err(err(pc, "Break/Continue outside a loop".into()));
                    }
                }
                Instr::LoopEnd { head_pc } => {
                    if !matches!(stack.pop(), Some(Open::Loop)) {
                        return Err(err(pc, "LoopEnd without LoopBegin".into()));
                    }
                    if head_pc.0 as usize >= pc {
                        return Err(err(pc, "LoopEnd head_pc must precede it".into()));
                    }
                }
                Instr::Mov { dst, src } => {
                    if dst.0 >= nregs || !op_ok(src) {
                        return Err(err(pc, "bad Mov operands".into()));
                    }
                }
                Instr::Load { dst, buf, offset, .. } => {
                    if dst.0 >= nregs || buf.0 as usize >= self.buffers.len() || !op_ok(offset) {
                        return Err(err(pc, "bad Load operands".into()));
                    }
                }
                Instr::Store { buf, offset, value, .. } => {
                    if buf.0 as usize >= self.buffers.len() || !op_ok(offset) || !op_ok(value) {
                        return Err(err(pc, "bad Store operands".into()));
                    }
                }
                Instr::Ptx { op, .. } if op.0 as usize >= self.ops.len() => {
                    return Err(err(pc, "Ptx op out of range".into()));
                }
                Instr::WaitUntil { pred, .. } if pred.0 as usize >= self.preds.len() => {
                    return Err(err(pc, "WaitUntil pred out of range".into()));
                }
                Instr::ReadParam { slot, .. } if slot.0 as usize >= self.host_abi.len() => {
                    return Err(err(pc, "ReadParam slot out of range".into()));
                }
                _ => {}
            }
        }
        if !stack.is_empty() {
            return Err(ProgramError::Invalid { pc: None, message: "unterminated control-flow frame".into() });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pretty printer
// ---------------------------------------------------------------------------

fn lc<T: fmt::Debug>(x: T) -> String {
    format!("{x:?}").to_lowercase()
}

fn sem_scope(sem: Sem, scope: Scope) -> String {
    match sem {
        Sem::Weak => String::new(),
        s => format!(".{}.{}", lc(s), lc(scope)),
    }
}

impl fmt::Display for Instr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Instr::*;
        match self {
            If { cond, else_pc, end_pc, elect } => {
                write!(f, "If {cond}{} else={else_pc} end={end_pc}", if *elect { " elect" } else { "" })
            }
            Else { end_pc } => write!(f, "Else end={end_pc}"),
            LoopBegin { end_pc } => write!(f, "LoopBegin end={end_pc}"),
            LoopIf { cond, end_pc } => write!(f, "LoopIf {cond} end={end_pc}"),
            LoopEnd { head_pc } => write!(f, "LoopEnd head={head_pc}"),
            Mov { dst, src } => write!(f, "Mov {dst} <- {src}"),
            ReadSpecial { dst, sreg } => write!(f, "ReadSpecial {dst} <- {sreg:?}"),
            ReadParam { dst, slot } => write!(f, "ReadParam {dst} <- {slot}"),
            Unary { op, ty, dst, a } => write!(f, "Unary {op:?}.{ty} {dst} <- {a}"),
            Binary { op, ty, dst, a, b } => write!(f, "Binary {op:?}.{ty} {dst} <- {a}, {b}"),
            Ternary { op, ty, dst, a, b, c } => write!(f, "Ternary {op:?}.{ty} {dst} <- {a}, {b}, {c}"),
            Compare { op, ty, dst, a, b } => write!(f, "Compare {op:?}.{ty} {dst} <- {a}, {b}"),
            Select { ty, dst, cond, a, b } => write!(f, "Select.{ty} {dst} <- {cond} ? {a} : {b}"),
            Cast { from, to, dst, src, .. } => write!(f, "Cast {to}.{from} {dst} <- {src}"),
            Ptx { op, dsts, srcs, pred, .. } => {
                if let Some(p) = pred {
                    write!(f, "@{p} ")?;
                }
                write!(f, "Ptx {op}")?;
                for (i, d) in dsts.iter().enumerate() {
                    write!(f, "{}{d}", if i == 0 { " " } else { ", " })?;
                }
                f.write_str(" <-")?;
                for (i, s) in srcs.iter().enumerate() {
                    write!(f, "{}{s}", if i == 0 { " " } else { ", " })?;
                }
                Ok(())
            }
            Load { ty, dst, buf, offset, sem, scope, .. } => {
                write!(f, "Load{} {ty} {dst} <- {buf}[{offset}]", sem_scope(*sem, *scope))
            }
            Store { ty, buf, offset, value, sem, scope, .. } => {
                write!(f, "Store{} {ty} {buf}[{offset}] <- {value}", sem_scope(*sem, *scope))
            }
            LoadAddr { ty, dst, addr, space, sem, scope, .. } => {
                write!(f, "LoadAddr{} {} {ty} {dst} <- [{addr}]", sem_scope(*sem, *scope), space.name())
            }
            StoreAddr { ty, addr, space, value, sem, scope, .. } => {
                write!(f, "StoreAddr{} {} {ty} [{addr}] <- {value}", sem_scope(*sem, *scope), space.name())
            }
            AddrOf { dst, buf, offset } => write!(f, "AddrOf {dst} <- {buf}[{offset}]"),
            Barrier { kind, id, count, .. } => {
                write!(f, "Barrier {kind:?} id={id}")?;
                if let Some(c) = count {
                    write!(f, " count={c}")?;
                }
                Ok(())
            }
            MbarInit { mbar, count, .. } => write!(f, "MbarInit [{mbar}] count={count}"),
            MbarArrive(a) => {
                write!(f, "MbarArrive [{}] {}", a.mbar, a.space.name())?;
                if let Some(tx) = a.expect_tx {
                    write!(f, " expect_tx={tx}")?;
                }
                Ok(())
            }
            MbarWait { mbar, phase, .. } => write!(f, "MbarWait [{mbar}] {phase:?}"),
            other => {
                let dbg = format!("{other:?}");
                f.write_str(&dbg)
            }
        }
    }
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let t = &self.topology;
        writeln!(f, "kernel {} grid={:?} cluster={:?} block={:?}", self.name, t.grid, t.cluster, t.block)?;
        for (i, k) in self.consts.iter().enumerate() {
            writeln!(f, "  k{i} = {} {:#x}", k.ty, k.bits)?;
        }
        for (i, p) in self.host_abi.iter().enumerate() {
            writeln!(f, "  param{i} {}: {:?}", p.name, p.kind)?;
        }
        for (i, b) in self.buffers.iter().enumerate() {
            writeln!(f, "  b{i} {}: {:?} {} base={}", b.name, b.space, b.dtype, b.base)?;
        }
        for (i, o) in self.ops.iter().enumerate() {
            writeln!(f, "  op{i} = {} {:?}", o.name, o.mods)?;
        }
        let mut depth = 0usize;
        for (pc, ins) in self.code.iter().enumerate() {
            if matches!(ins, Instr::Else { .. } | Instr::EndIf | Instr::LoopEnd { .. }) {
                depth = depth.saturating_sub(1);
            }
            let site = self.code_sites.get(pc).copied().unwrap_or(SiteId::NONE);
            let at = if site.is_none() { String::new() } else { format!("  @{site}") };
            writeln!(f, "{pc:04}  {}{ins}{at}", "  ".repeat(depth))?;
            if matches!(ins, Instr::If { .. } | Instr::Else { .. } | Instr::LoopBegin { .. }) {
                depth += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim_expr_eval() {
        let e = DimExpr::CeilDiv(Box::new(DimExpr::Param(ParamId(0))), Box::new(DimExpr::Const(128)));
        assert_eq!(e.eval(&|_| Some(1000)), Some(8));
        let e = DimExpr::FloorDiv(Box::new(DimExpr::Const(-7)), Box::new(DimExpr::Const(2)));
        assert_eq!(e.eval(&|_| None), Some(-4));
        assert_eq!(DimExpr::Param(ParamId(0)).eval(&|_| None), None);
    }

    #[test]
    fn instr_size_is_bounded() {
        assert!(std::mem::size_of::<Instr>() <= 128, "Instr is {} bytes", std::mem::size_of::<Instr>());
    }

    #[test]
    fn json_is_externally_tagged() {
        let j = serde_json::to_string(&Instr::Else { end_pc: Pc(3) }).unwrap();
        assert_eq!(j, r#"{"Else":{"end_pc":3}}"#);
        assert_eq!(serde_json::to_string(&Instr::EndIf).unwrap(), r#""EndIf""#);
        let j = serde_json::to_string(&Operand::Const(ConstId(2))).unwrap();
        assert_eq!(j, r#"{"Const":2}"#);
    }
}
