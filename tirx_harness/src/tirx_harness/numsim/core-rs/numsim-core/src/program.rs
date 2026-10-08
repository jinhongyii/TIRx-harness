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
pub const FORMAT_VERSION: u32 = 3;
/// Oldest format still loaded (README decision 15 transition).
pub const MIN_FORMAT_VERSION: u32 = 2;

/// Serde rule for `Option` fields of program types: the field must be
/// present (JSON `null` for `None`). Every program struct also has
/// `deny_unknown_fields`, so a misspelled or omitted field is a decode
/// error instead of a silent default (contract review item 7).
pub(crate) fn required<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(d)
}

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
#[serde(deny_unknown_fields)]
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
/// built with `Ptx` pack ops. Signed values are stored as two's complement
/// *masked to `ty.bits()`* (e.g. `-1i32` = `0xffff_ffff`); `validate`
/// rejects bits above `ty.bits()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
#[serde(deny_unknown_fields)]
pub enum Scope {
    Cta,
    Cluster,
    #[default]
    Gpu,
    Sys,
}

/// Memory proxy an access goes through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct MemMods {
    pub cache: CacheOp,
    pub evict: Evict,
    /// `.L2::64B/128B/256B`, 0 = none.
    pub l2_prefetch: u16,
    /// `.L2::cache_hint` policy operand.
    #[serde(deserialize_with = "required")]
    pub policy: Option<Operand>,
    pub nc: bool,
    pub uniform: bool,
}

// ---------------------------------------------------------------------------
// TIR-level ALU vocabulary (the PTX ALU tail is `Instr::Ptx`)
// ---------------------------------------------------------------------------

/// Rounding modifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub enum UnOp {
    Neg,
    Abs,
    /// Logical not (`prim.Not`): operand and result are `Pred`.
    Not,
    /// Bitwise not (`prim.BitwiseNot`, `~x`) on integer / `Pred` types.
    BitNot,
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub enum TerOp {
    /// Fused multiply-add, single rounding.
    Fma,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
    /// `%nwarpid`: maximum warps per SM (deterministic representative).
    NWarpId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Axis {
    X,
    Y,
    Z,
}

// ---------------------------------------------------------------------------
// Warp collectives
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ShflMode {
    Idx,
    Up,
    Down,
    Bfly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum VoteMode {
    All,
    Any,
    Uni,
    Ballot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ReduxOp {
    Add,
    Min,
    Max,
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum MatrixShape {
    M8N8,
    M8N16,
    M16N8,
    M16N16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub enum BulkCompletion {
    /// `.mbarrier::complete_tx::bytes [mbar]`.
    Mbarrier { mbar: Operand, space: AddrSpace },
    /// `.bulk_group`.
    Group,
}

/// `cp.async.bulk` / `cp.reduce.async.bulk` (non-tensor).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BulkCopyArgs {
    pub dst: Operand,
    pub dst_space: AddrSpace,
    pub src: Operand,
    pub src_space: AddrSpace,
    pub size: Operand,
    pub completion: BulkCompletion,
    /// `.multicast::cluster` CTA mask.
    #[serde(deserialize_with = "required")]
    pub multicast: Option<Operand>,
    #[serde(deserialize_with = "required")]
    pub reduce: Option<(AtomOp, Dtype)>,
    /// `.cp_mask` 16-bit byte mask (s2g): byte `i` of every 16-byte chunk is
    /// written only if bit `i` is set.
    #[serde(deserialize_with = "required")]
    pub byte_mask: Option<Operand>,
    /// `.ignore_oob` (g2s) with its byte counts; `None` = no `.ignore_oob`.
    #[serde(deserialize_with = "required")]
    pub ignore_oob: Option<IgnoreOob>,
    /// `_report` forms (layout::v1 barriers): validity inspection mode whose
    /// result is OR-ed into the completion mbarrier's primary-phase report
    /// predicate. There is no register destination: the report is read back
    /// with `MbarTestWait.report`.
    #[serde(deserialize_with = "required")]
    pub report: Option<ReportMode>,
    pub mods: MemMods,
}

/// `.ignore_oob` byte counts of a bulk copy: the first `ignore_bytes_left`
/// and last `ignore_bytes_right` bytes of the source range are not read and
/// the corresponding destination bytes are left unchanged (tx still counts
/// the full size). A `None` count is 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IgnoreOob {
    #[serde(deserialize_with = "required")]
    pub ignore_bytes_left: Option<Operand>,
    #[serde(deserialize_with = "required")]
    pub ignore_bytes_right: Option<Operand>,
}

/// Validity-inspection mode of `_report` copy forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ReportMode {
    /// `.per_element::ff`: every copied byte is inspected.
    PerElementFf,
    /// `.per_16bytes` without its pattern (legacy lowering form): fails
    /// closed. Superseded by `Per16BytesPattern` (W2-8).
    Per16Bytes,
    /// `.per_16bytes::<hex>` (W2-8): the lowest-addressed copied element of
    /// each 16-byte source chunk (16-byte aligned in the source address
    /// space) is compared with `pattern`; any equal element sets the
    /// report bit. `bits` = element width: 32, 16, 8 or 4 (the number of
    /// hex digits x 4; a 4-bit element is the low nibble of its byte).
    Per16BytesPattern { pattern: u32, bits: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub enum TmapField {
    GlobalAddress,
    Rank,
    BoxDim,
    GlobalDim,
    /// Global stride of dimension `ord`. In the
    /// `override_global_dim_stride_*` TMA forms this carries the *lower*
    /// stride operand of dimension `ord`; the shared upper operand is a
    /// separate `GlobalStrideUpper` override.
    GlobalStride,
    /// Upper stride operand of the `override_global_dim_stride_*` TMA forms
    /// (`ord: null`; one per instruction, applies to every overridden
    /// stride). Combined with the per-dimension `GlobalStride` lower parts
    /// by oplib exactly as legacy `override_tensor_map(lower_stride[],
    /// upper_stride)`.
    GlobalStrideUpper,
    ElementStride,
    ElemType,
    InterleaveLayout,
    SwizzleMode,
    SwizzleAtomicity,
    FillMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TmapOverride {
    pub field: TmapField,
    #[serde(deserialize_with = "required")]
    pub ord: Option<u8>,
    pub value: Operand,
    /// `_b8`/`_b16` spelling width, 0 = n/a.
    pub elem_bits: u8,
}

/// `cp.async.bulk.tensor` / `cp.reduce.async.bulk.tensor` / tensor prefetch.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    #[serde(deserialize_with = "required")]
    pub multicast: Option<Operand>,
    /// `.cta_group::1/2`, 0 = unspecified.
    pub cta_group: u8,
    pub overrides: Vec<TmapOverride>,
    /// `_report` forms; see [`BulkCopyArgs::report`].
    #[serde(deserialize_with = "required")]
    pub report: Option<ReportMode>,
    pub mods: MemMods,
}

/// `st.async` / `red.async` into a (remote) CTA's shared memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StAsyncArgs {
    pub ty: Ty,
    pub value: Operand,
    pub addr: Operand,
    /// Completion mbarrier (`complete_tx`). `None` for the
    /// `st.async.release` / `red.async.release` forms without an mbarrier:
    /// the write is then ordered only by its `.release` semantics (`sem`).
    #[serde(deserialize_with = "required")]
    pub mbar: Option<Operand>,
    #[serde(deserialize_with = "required")]
    pub red: Option<AtomOp>,
    pub sem: Sem,
    pub scope: Scope,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum BarRedOp {
    Popc,
    And,
    Or,
}

/// What a named-barrier instruction does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum BarKind {
    /// `bar.sync` / `barrier.sync` / `__syncthreads`.
    Sync,
    /// `bar.arrive`.
    Arrive,
    /// `bar.red.{popc,and,or}` (also syncthreads_and/or, cta_reduce):
    /// reduces `pred` and writes `dst`.
    Red {
        op: BarRedOp,
        pred: Operand,
        dst: Reg,
    },
}

/// mbarrier wait phase argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum PhaseArg {
    /// State token from an earlier arrive.
    State(Operand),
    /// `.parity`.
    Parity(Operand),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum WaitKind {
    Test,
    Try,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum TxOp {
    Expect,
    Complete,
}

/// `mbarrier.arrive` family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MbarArriveArgs {
    pub mbar: Operand,
    pub space: AddrSpace,
    /// None = 1.
    #[serde(deserialize_with = "required")]
    pub count: Option<Operand>,
    #[serde(deserialize_with = "required")]
    pub expect_tx: Option<Operand>,
    pub drop: bool,
    pub no_complete: bool,
    pub sem: Sem,
    pub scope: Scope,
    #[serde(deserialize_with = "required")]
    pub multicast: Option<Operand>,
    /// State-token destination (None = sink).
    #[serde(deserialize_with = "required")]
    pub state: Option<Reg>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum MbarQueryOp {
    PendingCount {
        state: Operand,
    },
    /// `layout_v1` is the layout the kernel declared for this barrier
    /// (`true` = `.layout::v1`, 511-arrival limit); the query reports
    /// whether the live object matches (W3-4).
    CheckLayout {
        mbar: Operand,
        space: AddrSpace,
        layout_v1: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum FenceKind {
    /// membar / fence.sc / fence.acq_rel / __threadfence*.
    Thread,
    MbarrierInit,
    ProxyAsync(Option<AddrSpace>),
    ProxyAlias,
    TensormapRelease,
    TensormapAcquire {
        addr: Operand,
        space: AddrSpace,
    },
    Tcgen05Before,
    Tcgen05After,
}

// ---------------------------------------------------------------------------
// tcgen05 vocabulary
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum TcShape {
    S32x32b,
    S16x64b,
    S16x128b,
    S16x256b,
    S16x32bx2 { split_off: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TcgenLdArgs {
    pub dsts: Vec<Reg>,
    pub taddr: Operand,
    /// Warp-uniform. Row (TMEM lane) offset added to `taddr`'s lane field, signed; the
    /// effective lane is `(taddr >> 16) + row` mod 2^16 (legacy
    /// `raw_tcgen05_address`). Emit const 0 when the source has none.
    pub row: Operand,
    /// Column offset added to `taddr`'s column field, same rules.
    pub col: Operand,
    pub shape: TcShape,
    pub num: u16,
    pub pack: bool,
    /// `.red.{min,max}`: op and reduced destinations.
    #[serde(deserialize_with = "required")]
    pub red: Option<(ReduxOp, Vec<Reg>)>,
    /// `.red` modifiers: `.abs` (reduce absolute values) and `.NaN`
    /// (propagate NaN); only meaningful with `red`.
    pub red_abs: bool,
    pub red_nan: bool,
    /// `.spcompress` (sparse-compressed destination).
    pub spcompress: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TcgenStArgs {
    pub srcs: Vec<Operand>,
    pub taddr: Operand,
    /// Warp-uniform. Row (TMEM lane) offset added to `taddr`'s lane field, signed; the
    /// effective lane is `(taddr >> 16) + row` mod 2^16 (legacy
    /// `raw_tcgen05_address`). Emit const 0 when the source has none.
    pub row: Operand,
    /// Column offset added to `taddr`'s column field, same rules.
    pub col: Operand,
    pub shape: TcShape,
    pub num: u16,
    pub unpack: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TcgenCpArgs {
    pub taddr: Operand,
    /// Read in the issuing lane. Row (TMEM lane) offset added to `taddr`'s lane field, signed; the
    /// effective lane is `(taddr >> 16) + row` mod 2^16 (legacy
    /// `raw_tcgen05_address`). Emit const 0 when the source has none.
    pub row: Operand,
    /// Column offset added to `taddr`'s column field, same rules.
    pub col: Operand,
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
#[serde(deny_unknown_fields)]
pub enum TcA {
    /// Shared-memory descriptor (`_ss`).
    Smem(Operand),
    /// Tensor-memory address (`_ts`).
    Tmem(Operand),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum TcMmaKind {
    F16,
    Tf32,
    F8f6f4,
    I8,
    MxF8f6f4,
    MxF4,
    MxF4Nvf4,
    /// `.kind::i16` table-indexed forms (`tcgen05_mma*_ti16_*`).
    Ti16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
    #[serde(deserialize_with = "required")]
    pub block_scale: Option<(Operand, Operand, u8)>,
    #[serde(deserialize_with = "required")]
    pub scale_input_d: Option<Operand>,
    #[serde(deserialize_with = "required")]
    pub sparse_meta: Option<Operand>,
    pub disable_output_lane: Vec<Operand>,
    pub collector_a: CollectorOp,
    pub collector_b: CollectorOp,
    pub ashift: bool,
    /// `_lut_b` forms (`tcgen05_mma*_lut_b_*`): B through a lookup table.
    /// Orthogonal to `kind` (block-scaled `lut_b` forms exist).
    pub lut_b: bool,
    /// LUT table address for `lut_b` forms: a *tensor-memory* address value
    /// (`taddr = lane<<16 | column`; TVM types the operand `addr@tmem`).
    /// Must be `Some` iff `lut_b` (checked by `validate`); the engine fails
    /// closed if it does not name a column inside a live TMEM allocation.
    #[serde(deserialize_with = "required")]
    pub lut_b_addr: Option<Operand>,
}

// ---------------------------------------------------------------------------
// Tile vocabulary (provisional, W1 B.7)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ExecScope {
    Thread,
    Warp,
    Warpgroup,
    Cta,
    Cluster,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub enum TileArg {
    Region {
        buf: Buf,
        base: Operand,
        map: LayoutId,
    },
    Frag {
        first: Reg,
        map: LayoutId,
    },
    Scalar(Operand),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TileArgs {
    pub op: TileOp,
    pub scope: ExecScope,
    /// Destination first, then sources in TIRx call order.
    pub args: Vec<TileArg>,
    pub axes: Vec<u8>,
    #[serde(deserialize_with = "required")]
    pub completion: Option<BulkCompletion>,
}

/// Element map computed at lowering time (W1 B.7): `entries[lane * slots +
/// slot]` = element offset relative to the region base, or -1 (none).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub enum Instr {
    // ----- control -----
    /// No effect (profiling markers, printf, nanosleep representatives).
    Nop,
    If {
        cond: Operand,
        else_pc: Pc,
        end_pc: Pc,
        elect: bool,
    },
    Else {
        end_pc: Pc,
    },
    EndIf,
    LoopBegin {
        end_pc: Pc,
    },
    LoopIf {
        cond: Operand,
        end_pc: Pc,
    },
    LoopEnd {
        head_pc: Pc,
    },
    Break,
    Continue,
    Exit,
    /// Error finding for every active lane where `cond` is false.
    Assert {
        cond: Operand,
        #[serde(deserialize_with = "required")]
        msg: Option<StrId>,
    },
    /// Fail closed (incomplete) if any lane reaches it (`strict=False` lowering).
    Unsupported {
        reason: StrId,
    },

    // ----- registers / TIR arithmetic -----
    Mov {
        dst: Reg,
        src: Operand,
    },
    ReadSpecial {
        dst: Reg,
        sreg: SpecialReg,
    },
    /// Read a scalar host parameter (uniform).
    ReadParam {
        dst: Reg,
        slot: ParamId,
    },
    Unary {
        op: UnOp,
        ty: Ty,
        dst: Reg,
        a: Operand,
    },
    Binary {
        op: BinOp,
        ty: Ty,
        dst: Reg,
        a: Operand,
        b: Operand,
    },
    Ternary {
        op: TerOp,
        ty: Ty,
        dst: Reg,
        a: Operand,
        b: Operand,
        c: Operand,
    },
    /// Predicate result.
    Compare {
        op: CmpOp,
        ty: Ty,
        dst: Reg,
        a: Operand,
        b: Operand,
    },
    /// `cond ? a : b` (TIR `Select` / `if_then_else`).
    Select {
        ty: Ty,
        dst: Reg,
        cond: Operand,
        a: Operand,
        b: Operand,
    },
    /// TIR `Cast` (C semantics by default; `reinterpret` uses `Mov`).
    Cast {
        from: Ty,
        to: Ty,
        dst: Reg,
        src: Operand,
        rnd: Rounding,
        sat: bool,
    },
    /// Generic pure PTX / CUDA-helper op from `Program::ops`.
    /// `pred`: PTX guard; lanes where it is false do not execute and keep
    /// their destinations if `keep_dst`, else get a zero representative.
    Ptx {
        op: OpId,
        dsts: Vec<Reg>,
        srcs: Vec<Operand>,
        #[serde(deserialize_with = "required")]
        pred: Option<Operand>,
        keep_dst: bool,
    },
    /// Dynamic index into a register-promoted local array `base..base+len`
    /// (all same type). OOB = error finding.
    LoadRegIndexed {
        dst: Reg,
        base: Reg,
        len: u32,
        idx: Operand,
    },
    StoreRegIndexed {
        base: Reg,
        len: u32,
        idx: Operand,
        value: Operand,
    },

    // ----- warp collectives -----
    Shfl {
        mode: ShflMode,
        ty: Ty,
        dst: Reg,
        #[serde(deserialize_with = "required")]
        dst_pred: Option<Reg>,
        src: Operand,
        lane: Operand,
        clamp: Operand,
        membermask: Operand,
    },
    Vote {
        mode: VoteMode,
        dst: Reg,
        pred: Operand,
        membermask: Operand,
    },
    Redux {
        op: ReduxOp,
        ty: Ty,
        dst: Reg,
        src: Operand,
        membermask: Operand,
    },
    /// `elect.sync`: `dst_pred` = 1 in the elected lane.
    Elect {
        dst_pred: Reg,
        #[serde(deserialize_with = "required")]
        dst_lane: Option<Reg>,
        membermask: Operand,
    },
    WarpSync {
        membermask: Operand,
    },
    LdMatrix {
        dsts: Vec<Reg>,
        addr: Operand,
        space: AddrSpace,
        shape: MatrixShape,
        num: u8,
        trans: bool,
        fmt: MatrixFmt,
    },
    StMatrix {
        srcs: Vec<Operand>,
        addr: Operand,
        space: AddrSpace,
        shape: MatrixShape,
        num: u8,
        trans: bool,
    },

    // ----- memory -----
    /// `dst = buf[offset]`. `offset` counts elements of the buffer's
    /// element type `buffers[buf].dtype.elem`, so the access starts at
    /// *bit* `offset * elem.bits()` of the buffer ([`BufferDecl::bit_offset`]);
    /// for byte-sized elements that is byte `offset * elem.bits()/8`. For
    /// sub-byte elements (fp4, fp6, u4, u6) the bit offset must be
    /// byte-aligned and `ty.bits()` a multiple of 8, else the access is a
    /// `Misaligned` error (no sub-byte read-modify-write). `ty` may differ
    /// from the buffer dtype (reinterpret, vector loads). Space from the
    /// buffer declaration.
    Load {
        ty: Ty,
        dst: Reg,
        buf: Buf,
        offset: Operand,
        sem: Sem,
        scope: Scope,
        mods: MemMods,
    },
    /// `buf[offset] = value`; offset rules as `Load`.
    Store {
        ty: Ty,
        buf: Buf,
        offset: Operand,
        value: Operand,
        sem: Sem,
        scope: Scope,
        mods: MemMods,
    },
    /// Raw-address load (`addr` value in `space`'s encoding).
    LoadAddr {
        ty: Ty,
        dst: Reg,
        addr: Operand,
        space: AddrSpace,
        sem: Sem,
        scope: Scope,
        mods: MemMods,
    },
    StoreAddr {
        ty: Ty,
        addr: Operand,
        space: AddrSpace,
        value: Operand,
        sem: Sem,
        scope: Scope,
        mods: MemMods,
    },
    /// `dst` (u64 generic) = address of `buf[offset]`.
    AddrOf {
        dst: Reg,
        buf: Buf,
        offset: Operand,
    },
    /// `atom` (dst Some) / `red` / bitbucket (dst None). Vector `ty` = one
    /// RMW per lane-element. `cmp` only for Cas.
    Atom {
        op: AtomOp,
        ty: Ty,
        #[serde(deserialize_with = "required")]
        dst: Option<Reg>,
        addr: Operand,
        space: AddrSpace,
        value: Operand,
        #[serde(deserialize_with = "required")]
        cmp: Option<Operand>,
        sem: Sem,
        scope: Scope,
        ftz: bool,
    },
    StBulk {
        addr: Operand,
        space: AddrSpace,
        size: Operand,
    },
    /// Contents become undefined (validity cleared).
    Discard {
        addr: Operand,
        space: AddrSpace,
        size: u32,
    },
    /// `cvta`: `to_generic` = space -> generic.
    Cvta {
        dst: Reg,
        src: Operand,
        space: AddrSpace,
        to_generic: bool,
    },
    Isspacep {
        dst: Reg,
        src: Operand,
        space: AddrSpace,
    },
    Mapa {
        dst: Reg,
        src: Operand,
        rank: Operand,
        space: AddrSpace,
    },
    GetCtaRank {
        dst: Reg,
        src: Operand,
        space: AddrSpace,
    },

    // ----- async copies -----
    CpAsync {
        dst: Operand,
        src: Operand,
        cp_size: u8,
        #[serde(deserialize_with = "required")]
        src_size: Option<Operand>,
        #[serde(deserialize_with = "required")]
        ignore_src: Option<Operand>,
        mods: MemMods,
    },
    /// `cp.async.commit_group` / `cp.async.bulk.commit_group`.
    AsyncCommit {
        domain: Domain,
    },
    /// `*.wait_group{.read} n` (`cp.async.wait_all` = commit + wait 0).
    AsyncWait {
        domain: Domain,
        n: u32,
        read: bool,
    },
    CpAsyncMbarArrive {
        mbar: Operand,
        space: AddrSpace,
        noinc: bool,
    },
    BulkCopy(BulkCopyArgs),
    Tma(Box<TmaArgs>),
    StAsync(StAsyncArgs),
    TensorMapReplace {
        tmap: Operand,
        space: AddrSpace,
        field: TmapField,
        #[serde(deserialize_with = "required")]
        ord: Option<u8>,
        value: Operand,
    },
    TensorMapCopyFence {
        dst: Operand,
        src: Operand,
        size: u32,
        scope: Scope,
    },

    // ----- synchronization -----
    Barrier {
        kind: BarKind,
        id: Operand,
        #[serde(deserialize_with = "required")]
        count: Option<Operand>,
        aligned: bool,
    },
    ClusterArrive {
        sem: Sem,
        aligned: bool,
    },
    ClusterWait {
        acquire: bool,
        aligned: bool,
    },
    GridSync,
    MbarInit {
        mbar: Operand,
        space: AddrSpace,
        count: Operand,
        layout_v1: bool,
    },
    MbarInval {
        mbar: Operand,
        space: AddrSpace,
    },
    MbarArrive(MbarArriveArgs),
    MbarTx {
        op: TxOp,
        mbar: Operand,
        space: AddrSpace,
        bytes: Operand,
        #[serde(deserialize_with = "required")]
        multicast: Option<Operand>,
        scope: Scope,
    },
    /// Non-blocking `test_wait` / `try_wait`; `dst` = ready predicate.
    MbarTestWait {
        kind: WaitKind,
        mbar: Operand,
        space: AddrSpace,
        phase: PhaseArg,
        sem: Sem,
        scope: Scope,
        #[serde(deserialize_with = "required")]
        dst: Option<Reg>,
        /// `_report` forms (layout::v1 copy-report / conditional parity,
        /// sync-semantics.md §2.4): the barrier's report predicate, taken
        /// from the same physical snapshot as `dst`.
        #[serde(deserialize_with = "required")]
        report: Option<Reg>,
        /// `_report_value` forms: the reported value from that snapshot.
        #[serde(deserialize_with = "required")]
        report_value: Option<Reg>,
    },
    /// Blocking wait (`cuda.mbarrier_wait*`; no report forms exist).
    MbarWait {
        mbar: Operand,
        space: AddrSpace,
        phase: PhaseArg,
        sem: Sem,
        scope: Scope,
    },
    MbarQuery {
        dst: Reg,
        op: MbarQueryOp,
    },
    Fence {
        kind: FenceKind,
        sem: Sem,
        scope: Scope,
    },
    SetMaxNReg {
        inc: bool,
        count: u32,
    },
    /// Block until predicate `pred` accepts the word at `addr`; then `dst`
    /// holds the accepted value. `captures` are snapshotted at issue.
    WaitUntil {
        dst: Reg,
        addr: Operand,
        ty: Ty,
        space: AddrSpace,
        sem: Sem,
        scope: Scope,
        pred: PredId,
        captures: Vec<Reg>,
    },
    GridDepControl {
        launch_dependents: bool,
    },
    ClcTryCancel {
        resp: Operand,
        mbar: Operand,
        multicast: bool,
    },

    // ----- tcgen05 -----
    /// Writes the allocated taddr to shared memory at `dst`.
    TcgenAlloc {
        dst: Operand,
        ncols: Operand,
        cta_group: u8,
        exclusive: bool,
    },
    TcgenDealloc {
        taddr: Operand,
        ncols: Operand,
        cta_group: u8,
        exclusive: bool,
    },
    TcgenRelinquish {
        cta_group: u8,
    },
    TcgenCommit {
        mbar: Operand,
        space: AddrSpace,
        cta_group: u8,
        #[serde(deserialize_with = "required")]
        multicast: Option<Operand>,
        /// `.sync_restrict::*` form (the arrive is restricted to the
        /// issuing CTA's / cluster's shared memory window).
        sync_restrict: bool,
        /// `.multicast::cluster.width::N` (`tcgen05_commit_multicast_width`).
        #[serde(deserialize_with = "required")]
        multicast_width: Option<u8>,
    },
    TcgenLd(Box<TcgenLdArgs>),
    TcgenSt(Box<TcgenStArgs>),
    /// `wait::ld` (`st = false`) / `wait::st`.
    TcgenWait {
        st: bool,
    },
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
            // setmaxnreg (inc and dec) is a warpgroup rendezvous;
            // cta_group::2 dealloc/relinquish rendezvous with the peer CTA.
            SetMaxNReg { .. } => true,
            TcgenDealloc { cta_group, .. } | TcgenRelinquish { cta_group } => *cta_group == 2,
            Tile(t) => {
                matches!(t.op, TileOp::Gemm | TileOp::Copy | TileOp::Sum)
                    || t.scope != ExecScope::Thread
            }
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
#[serde(deny_unknown_fields)]
pub struct RegDecl {
    pub ty: Ty,
    #[serde(deserialize_with = "required")]
    pub name: Option<String>,
    /// Static hint: every write is warp-uniform. Engines must be correct
    /// when ignoring it.
    pub uniform: bool,
}

/// A small expression over scalar host parameters (dynamic extents).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
///
/// **TMEM buffers (ruling).** A `Space::Tmem` buffer (a TIR `DeclBuffer` view
/// of tensor memory) is addressed densely: element `offset` lives at TMEM
/// lane `(offset / cols) % 128`, column `base_col + offset % cols`, i.e. the
/// tcgen05 32-lane-per-warp datapath view with 32-bit columns. `Load` and
/// `Store` on it are executed as the equivalent `tcgen05.ld` / `tcgen05.st`
/// (`32x32b`) by a warp whose active lanes each address the TMEM lane of
/// their own sub-partition (`warp_in_cta % 4`); any access that is not
/// expressible that way (lane outside the warp's sub-partition, element not
/// 32-bit aligned, a column outside a live allocation) fails closed
/// (`Unsupported`/`Misaligned`), never silently. `AddrOf` and raw
/// `LoadAddr`/`StoreAddr` on TMEM are not allowed.
///
/// *Sub-word cells.* For 8- and 16-bit element types, `per_cell = 32 /
/// bits` consecutive elements share one 32-bit cell: element `offset` lives
/// in cell `offset / per_cell` (addressed by the lane/column rule above,
/// with `offset` replaced by the cell index) at bit offset
/// `(offset % per_cell) * bits` within the cell. A `Store` of a sub-word
/// element is a read-modify-write of its cell. Other widths (sub-byte,
/// 64-bit, vectors spanning cells) fail closed.
///
/// *Replicated views* (a TIR TMEM layout that maps one logical element to
/// several lanes/columns) are not representable: lowering emits
/// `Unsupported { reason: "tmem_replicated_view: <buffer>" }`, which fails
/// closed as incomplete if reached.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BufferDecl {
    pub name: String,
    pub space: Space,
    pub dtype: Ty,
    pub shape: Vec<DimExpr>,
    /// Element strides (row-major when empty).
    pub strides: Vec<DimExpr>,
    /// Host parameter backing this buffer (global buffers, tensor maps).
    #[serde(deserialize_with = "required")]
    pub param_slot: Option<ParamId>,
    /// Byte offset of the buffer in its backing: shared-window offset for
    /// shared buffers; offset within `view_of` for views; 0 otherwise.
    pub base: u64,
    /// Runtime base (TMEM views over a dynamic `allocated_addr`): when set,
    /// the buffer starts at the value of this register (in the space's
    /// address encoding, e.g. a TMEM `taddr`), read when each access
    /// executes and required to be warp-uniform; `base` must then be 0.
    #[serde(deserialize_with = "required")]
    pub base_reg: Option<Reg>,
    /// Total bytes; None = taken from the bound host argument.
    #[serde(deserialize_with = "required")]
    pub byte_len: Option<DimExpr>,
    pub align: u32,
    /// This buffer is a view (DeclBuffer) of another buffer's storage.
    #[serde(deserialize_with = "required")]
    pub view_of: Option<Buf>,
    /// Lowering hint: the buffer holds declared synchronization words
    /// (an `AddrOf` of it reaches a `WaitUntil`). The engine emits
    /// `DeclareWord` for it at launch begin and keeps write history. A
    /// `WaitUntil` on an undeclared word declares it on first use (history
    /// then starts at that point; checkers must treat earlier writes as
    /// unknown).
    pub sync_words: bool,
}

/// Interned generic op: canonical op name + canonical modifier tuple.
/// Each modifier SHOULD be spelled `"slot=token"` (TVM PTX-table slot name,
/// e.g. `["rnd=rn", "dtype=f16x2", "src=f32"]`) so the key does not depend on
/// slot order (W4-2); bare tokens are still accepted by `oplib::resolve_ptx`
/// and assigned to slots in table order. Resolved once at load.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
    #[serde(deserialize_with = "required")]
    pub min_blocks_per_sm: Option<u32>,
    pub cooperative: bool,
    /// Per-thread register budget at launch (setmaxnreg `Configure`), 0 = default.
    pub regs_per_thread: u32,
}

/// Concrete launch shape after evaluating `DimExpr`s.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub enum ParamKind {
    Buffer,
    Pointer,
    Scalar,
    TensorMap,
    /// A shape variable of buffer parameter `buffer` (axis `axis`, 0 =
    /// outermost). No host value is bound: the binder takes it from the
    /// bound array's shape. Replaces name matching on `<buf>.shape<axis>`.
    ImplicitShape {
        buffer: ParamId,
        axis: u8,
    },
}

/// Host-prelude tensor-map encoding facts (`tensormap_encode_tiled`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorMapSpec {
    pub dtype: Dtype,
    pub rank: u8,
    /// Innermost first.
    pub global_dim: Vec<DimExpr>,
    /// Byte strides of dims 1.. .
    pub global_stride: Vec<DimExpr>,
    /// Box extents, innermost first. `DimExpr` so host-prologue runtime
    /// values are expressible; evaluated at bind time like `global_dim`.
    pub box_dim: Vec<DimExpr>,
    /// Element (traversal) strides, innermost first; `DimExpr` likewise.
    pub element_stride: Vec<DimExpr>,
    pub interleave: u8,
    pub swizzle: u8,
    pub l2_promotion: u8,
    pub oob_fill: u8,
    /// Byte offset into the base buffer for the global address.
    pub base_offset: DimExpr,
    /// Raw `CUtensorMapDataType` the host requested when it differs from
    /// what `dtype` implies (e.g. 11 = TFLOAT32, 13 = 16U4_ALIGN8B,
    /// 14 = 16U4_ALIGN16B, 15 = 16U6_ALIGN16B). The engine/oplib decides
    /// the semantics and fails closed on values it does not model; lowering
    /// passes it through unchanged instead of guessing.
    #[serde(deserialize_with = "required")]
    pub force_cu_dtype: Option<u8>,
}

/// One host parameter, in kernel signature order.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParamSlot {
    /// Canonical name (binding key, unique within a Module).
    pub name: String,
    pub local_name: String,
    pub aliases: Vec<String>,
    pub kind: ParamKind,
    #[serde(deserialize_with = "required")]
    pub dtype: Option<Ty>,
    pub shape: Vec<DimExpr>,
    /// Tensor maps encoded by the host prelude. When set, the host binds no
    /// value for this slot: the engine encodes the 128-byte map at bind time
    /// from `implicit_base`'s allocation (its engine VA + `base_offset`) and
    /// this spec (W8-3).
    #[serde(deserialize_with = "required")]
    pub tensor_map: Option<TensorMapSpec>,
    /// Implicit tensor map: the buffer parameter it describes.
    #[serde(deserialize_with = "required")]
    pub implicit_base: Option<ParamId>,
    /// Buffer declared for this parameter (Buffer kind: global buffer;
    /// TensorMap kind: 128-byte Param-space buffer usable with `AddrOf`).
    #[serde(deserialize_with = "required")]
    pub buf: Option<Buf>,
}

/// Feature requirements a program declares (engines reject what they lack).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Requirements {
    pub implicit_tmem: bool,
    pub dynamic_tmem_lifecycle: bool,
    pub readonly_proxy: bool,
    pub grid_dependency: bool,
    pub raw_tensor_map_registry: bool,
}

/// One kernel.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    #[serde(deserialize_with = "required")]
    pub arch: Option<String>,
    pub requirements: Requirements,
    /// Lowering's collected unsupported reasons (`site#N kind: reason`);
    /// non-empty only for `strict=False` lowering.
    pub unsupported: Vec<String>,
}

/// Kernels launched in order, sharing host bindings by name.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
            ProgramError::Invalid {
                pc: Some(pc),
                message,
            } => write!(f, "invalid program at @{pc}: {message}"),
            ProgramError::Invalid { pc: None, message } => write!(f, "invalid program: {message}"),
        }
    }
}

impl std::error::Error for ProgramError {}

impl Module {
    pub fn new(kernels: Vec<Program>) -> Module {
        Module {
            format_version: FORMAT_VERSION,
            kernels,
        }
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        postcard::to_stdvec(self).expect("Module is serializable")
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Module, ProgramError> {
        let m: Module =
            postcard::from_bytes(bytes).map_err(|e| ProgramError::Decode(e.to_string()))?;
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
        // Format 2 (pre-decision 15) is still accepted: the only change is
        // `SiteInfo.buffers`, derived as `[buffer]` when absent.
        if self.format_version != FORMAT_VERSION && self.format_version != MIN_FORMAT_VERSION {
            return Err(ProgramError::Version {
                found: self.format_version,
                expected: FORMAT_VERSION,
            });
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
        self.code_sites
            .get(pc.0 as usize)
            .copied()
            .unwrap_or(SiteId::NONE)
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

    /// Full structural validation. Backends may assume a validated program;
    /// lowering output that fails here is a lowering bug. Checks:
    /// * parallel arrays (`code_sites`) and every nested index: registers
    ///   (including `base..base+len` ranges), consts, buffers, strings,
    ///   layouts, ops, preds, params, sites, `DimExpr` params;
    /// * `Ty` invariants (`lanes >= 1`, `bits <= MAX_VALUE_BITS`) everywhere
    ///   and `Const` bits fit `ty.bits()` (and <= 128);
    /// * every destination's value type fits its register (`bits` and
    ///   therefore `slots`), so no write spills into the next register;
    /// * structured control flow: every target is the matching instruction
    ///   of *its own* frame (`If.else_pc/end_pc`, `Else.end_pc`,
    ///   `LoopBegin.end_pc`, `LoopIf.end_pc`, `LoopEnd.head_pc`),
    ///   `LoopIf` is directly inside its loop, `Break`/`Continue` inside a
    ///   loop, frames balanced;
    /// * `PredProgram` ranges are disjoint, cover exactly the tail
    ///   `code[main_end..]`, contain only allowed instructions, and the main
    ///   body ends in `Exit` or `Unsupported` so it cannot fall through.
    pub fn validate(&self) -> Result<(), ProgramError> {
        let at = |pc: usize, m: String| ProgramError::Invalid {
            pc: Some(pc as u32),
            message: m,
        };
        let glob = |m: String| ProgramError::Invalid {
            pc: None,
            message: m,
        };
        let n = self.code.len();
        if self.code_sites.len() != n {
            return Err(glob(format!(
                "code_sites has {} entries for {n} instrs",
                self.code_sites.len()
            )));
        }
        let nregs = self.regs.len();
        let ty_ok = |t: Ty| t.lanes >= 1 && t.bits() <= crate::dtype::MAX_VALUE_BITS;
        // ---- tables ----
        for (i, r) in self.regs.iter().enumerate() {
            if !ty_ok(r.ty) {
                return Err(glob(format!("r{i}: invalid type {}", r.ty)));
            }
        }
        for (i, k) in self.consts.iter().enumerate() {
            if !ty_ok(k.ty)
                || k.ty.bits() > 128
                || (k.ty.bits() < 128 && k.bits >> k.ty.bits() != 0)
            {
                return Err(glob(format!(
                    "k{i}: bits {:#x} do not fit {}",
                    k.bits, k.ty
                )));
            }
        }
        let np = self.host_abi.len();
        let mut dim_err: Option<String> = None;
        let mut check_dim = |e: &DimExpr, what: &str| {
            fn walk(e: &DimExpr, np: usize) -> bool {
                use DimExpr::*;
                match e {
                    Const(_) => true,
                    Param(p) => (p.0 as usize) < np,
                    Add(a, b)
                    | Sub(a, b)
                    | Mul(a, b)
                    | FloorDiv(a, b)
                    | CeilDiv(a, b)
                    | Min(a, b)
                    | Max(a, b) => walk(a, np) && walk(b, np),
                }
            }
            if dim_err.is_none() && !walk(e, np) {
                dim_err = Some(format!("{what}: DimExpr parameter out of range"));
            }
        };
        let t = &self.topology;
        for (i, d) in t.grid.iter().enumerate() {
            check_dim(d, &format!("grid[{i}]"));
        }
        check_dim(&t.dyn_smem_bytes, "dyn_smem_bytes");
        for (i, b) in self.buffers.iter().enumerate() {
            if !ty_ok(b.dtype) {
                return Err(glob(format!("b{i}: invalid dtype {}", b.dtype)));
            }
            if b.base_reg.is_some_and(|r| r.0 as usize >= nregs)
                || (b.base_reg.is_some() && b.base != 0)
            {
                return Err(glob(format!(
                    "b{i}: base_reg out of range or combined with a nonzero base"
                )));
            }
            if b.param_slot.is_some_and(|p| p.0 as usize >= np)
                || b.view_of
                    .is_some_and(|v| v.0 as usize >= self.buffers.len() || v.0 as usize == i)
            {
                return Err(glob(format!("b{i}: param_slot/view_of out of range")));
            }
            for d in b.shape.iter().chain(&b.strides).chain(b.byte_len.as_ref()) {
                check_dim(d, &format!("b{i}"));
            }
        }
        for (i, slot) in self.host_abi.iter().enumerate() {
            if let ParamKind::ImplicitShape { buffer, .. } = slot.kind {
                let ok = self
                    .host_abi
                    .get(buffer.0 as usize)
                    .is_some_and(|b| b.kind == ParamKind::Buffer);
                if !ok {
                    return Err(glob(format!(
                        "param{i}: ImplicitShape.buffer is not a Buffer slot"
                    )));
                }
            }
            if slot.dtype.is_some_and(|t| !ty_ok(t))
                || slot.buf.is_some_and(|b| b.0 as usize >= self.buffers.len())
                || slot.implicit_base.is_some_and(|p| p.0 as usize >= np)
            {
                return Err(glob(format!("param{i}: dtype/buf/implicit_base invalid")));
            }
            for d in &slot.shape {
                check_dim(d, &format!("param{i}"));
            }
            if let Some(m) = &slot.tensor_map {
                for d in m
                    .global_dim
                    .iter()
                    .chain(&m.global_stride)
                    .chain(&m.box_dim)
                    .chain(&m.element_stride)
                    .chain(std::iter::once(&m.base_offset))
                {
                    check_dim(d, &format!("param{i}.tensor_map"));
                }
            }
        }
        if let Some(e) = dim_err {
            return Err(glob(e));
        }
        for (i, l) in self.layouts.iter().enumerate() {
            if l.entries.len() as u64 != l.lanes as u64 * l.slots as u64 {
                return Err(glob(format!("L{i}: entries length != lanes * slots")));
            }
        }
        // ---- predicate sub-programs at the tail ----
        let mut ranges: Vec<(u32, u32, usize)> = self
            .preds
            .iter()
            .enumerate()
            .map(|(i, p)| (p.start.0, p.end.0, i))
            .collect();
        ranges.sort_unstable();
        let main_end = ranges.first().map_or(n, |r| r.0 as usize);
        let mut cursor = main_end;
        for &(start, end, i) in &ranges {
            if start as usize != cursor || end < start || end as usize > n {
                return Err(glob(format!(
                    "pred {i}: ranges must be disjoint and tile code[{main_end}..]"
                )));
            }
            cursor = end as usize;
            let p = &self.preds[i];
            if p.arg.0 as usize >= nregs || p.result.0 as usize >= nregs {
                return Err(glob(format!("pred {i}: arg/result register out of range")));
            }
        }
        if cursor != n {
            return Err(glob(format!(
                "code[{cursor}..{n}] is neither main body nor a predicate"
            )));
        }
        if main_end > 0
            && !self.preds.is_empty()
            && !matches!(
                self.code[main_end - 1],
                Instr::Exit | Instr::Unsupported { .. }
            )
        {
            return Err(at(
                main_end - 1,
                "main body must end in Exit/Unsupported before predicate code".into(),
            ));
        }
        for pc in main_end..n {
            let ok = matches!(
                self.code[pc],
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
                return Err(at(
                    pc,
                    "instruction not allowed in a predicate sub-program".into(),
                ));
            }
        }
        // ---- per-instruction references ----
        for (pc, ins) in self.code.iter().enumerate() {
            if let Instr::TcgenMma(a) = ins {
                if a.lut_b != a.lut_b_addr.is_some() {
                    return Err(at(pc, "TcgenMma: lut_b_addr must be set iff lut_b".into()));
                }
            }
            let s = self.code_sites[pc];
            if !s.is_none() && s.0 as usize >= self.sites.len() {
                return Err(at(pc, format!("site {s} out of range")));
            }
            let mut bad: Option<String> = None;
            ins.refs(&mut |r| {
                if bad.is_some() {
                    return;
                }
                let reg_ok = |x: Reg| (x.0 as usize) < nregs;
                bad = match r {
                    Ref::Use(Operand::Reg(x)) | Ref::Def(x, None) if !reg_ok(x) => {
                        Some(format!("{x} out of range"))
                    }
                    Ref::Use(Operand::Const(k)) if k.0 as usize >= self.consts.len() => {
                        Some(format!("{k} out of range"))
                    }
                    Ref::Def(x, Some(ty)) => {
                        if !reg_ok(x) {
                            Some(format!("{x} out of range"))
                        } else if !ty_ok(ty) || ty.bits() > self.regs[x.0 as usize].ty.bits() {
                            Some(format!(
                                "{ty} does not fit {x}: {}",
                                self.regs[x.0 as usize].ty
                            ))
                        } else {
                            None
                        }
                    }
                    Ref::Regs(base, len) if len == 0 || base.0 as usize + len as usize > nregs => {
                        Some(format!("register range {base}+{len} out of range"))
                    }
                    Ref::Ty(ty) if !ty_ok(ty) => Some(format!("invalid type {ty}")),
                    Ref::Buf(b) if b.0 as usize >= self.buffers.len() => {
                        Some(format!("{b} out of range"))
                    }
                    Ref::Str(x) if x.0 as usize >= self.strings.len() => {
                        Some(format!("{x} out of range"))
                    }
                    Ref::Layout(x) if x.0 as usize >= self.layouts.len() => {
                        Some(format!("{x} out of range"))
                    }
                    Ref::Op(x) if x.0 as usize >= self.ops.len() => {
                        Some(format!("{x} out of range"))
                    }
                    Ref::Pred(x) if x.0 as usize >= self.preds.len() => {
                        Some(format!("{x} out of range"))
                    }
                    Ref::Param(x) if x.0 as usize >= np => Some(format!("{x} out of range")),
                    _ => None,
                };
            });
            if let Instr::LoadRegIndexed { dst, base, .. } = ins {
                if bad.is_none()
                    && self.regs[base.0 as usize].ty.bits() > self.regs[dst.0 as usize].ty.bits()
                {
                    bad = Some(format!("{dst} cannot hold an element of {base}"));
                }
            }
            if let Some(m) = bad {
                return Err(at(pc, m));
            }
        }
        // ---- structured control flow (main body only) ----
        enum Open {
            If {
                else_pc: Pc,
                end_pc: Pc,
            },
            Else {
                end_pc: Pc,
            },
            Loop {
                begin: Pc,
                end_pc: Pc,
                cond_seen: bool,
            },
        }
        let mut stack: Vec<Open> = Vec::new();
        for (pc, ins) in self.code[..main_end].iter().enumerate() {
            let here = Pc(pc as u32);
            match ins {
                Instr::If {
                    else_pc, end_pc, ..
                } => {
                    if else_pc.0 <= here.0 || end_pc.0 < else_pc.0 || end_pc.0 as usize >= main_end
                    {
                        return Err(at(pc, "bad If targets".into()));
                    }
                    stack.push(Open::If {
                        else_pc: *else_pc,
                        end_pc: *end_pc,
                    });
                }
                Instr::Else { end_pc } => match stack.pop() {
                    Some(Open::If {
                        else_pc,
                        end_pc: if_end,
                    }) if else_pc == here && if_end == *end_pc => {
                        stack.push(Open::Else { end_pc: *end_pc })
                    }
                    _ => return Err(at(pc, "Else does not match its If (else_pc/end_pc)".into())),
                },
                Instr::EndIf => match stack.pop() {
                    Some(Open::If { else_pc, end_pc }) if else_pc == here && end_pc == here => {}
                    Some(Open::Else { end_pc }) if end_pc == here => {}
                    _ => return Err(at(pc, "EndIf does not match its If/Else".into())),
                },
                Instr::LoopBegin { end_pc } => {
                    if end_pc.0 <= here.0 || end_pc.0 as usize >= main_end {
                        return Err(at(pc, "bad LoopBegin end_pc".into()));
                    }
                    stack.push(Open::Loop {
                        begin: here,
                        end_pc: *end_pc,
                        cond_seen: false,
                    });
                }
                Instr::LoopIf { end_pc, .. } => match stack.last_mut() {
                    Some(Open::Loop {
                        end_pc: e,
                        cond_seen,
                        ..
                    }) if e == end_pc && !*cond_seen => *cond_seen = true,
                    _ => return Err(at(
                        pc,
                        "LoopIf must be directly inside its own loop (once), end_pc = its LoopEnd"
                            .into(),
                    )),
                },
                Instr::Break | Instr::Continue => {
                    if !stack.iter().any(|o| matches!(o, Open::Loop { .. })) {
                        return Err(at(pc, "Break/Continue outside a loop".into()));
                    }
                }
                Instr::LoopEnd { head_pc } => match stack.pop() {
                    Some(Open::Loop {
                        begin,
                        end_pc,
                        cond_seen,
                    }) if end_pc == here && cond_seen => {
                        if head_pc.0 <= begin.0 || head_pc.0 >= here.0 {
                            return Err(at(pc, "LoopEnd head_pc must lie inside its loop".into()));
                        }
                    }
                    _ => {
                        return Err(at(
                            pc,
                            "LoopEnd does not match its LoopBegin (or the loop has no LoopIf)"
                                .into(),
                        ))
                    }
                },
                _ => {}
            }
        }
        if !stack.is_empty() {
            return Err(glob("unterminated control-flow frame".into()));
        }
        Ok(())
    }
}

/// One reference made by an instruction (see [`Instr::refs`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ref {
    /// Operand read.
    Use(Operand),
    /// Register written, with the value type written when known.
    Def(Reg, Option<Ty>),
    /// A register range `base..base+len` (promoted local arrays).
    Regs(Reg, u32),
    Ty(Ty),
    Buf(Buf),
    Str(StrId),
    Layout(LayoutId),
    Op(OpId),
    Pred(PredId),
    Param(ParamId),
}

impl Instr {
    /// Visit every register, constant, table index and type the instruction
    /// references (used by `validate`; usable by printers and analyses).
    pub fn refs(&self, f: &mut dyn FnMut(Ref)) {
        use Instr::*;
        use Ref::*;
        fn opt(o: &Option<Operand>, f: &mut dyn FnMut(Ref)) {
            if let Some(o) = o {
                f(Use(*o));
            }
        }
        fn mods(m: &MemMods, f: &mut dyn FnMut(Ref)) {
            opt(&m.policy, f);
        }
        fn completion(c: &BulkCompletion, f: &mut dyn FnMut(Ref)) {
            if let BulkCompletion::Mbarrier { mbar, .. } = c {
                f(Use(*mbar));
            }
        }
        fn phase(p: &PhaseArg, f: &mut dyn FnMut(Ref)) {
            match p {
                PhaseArg::State(o) | PhaseArg::Parity(o) => f(Use(*o)),
            }
        }
        let pred_ty = |t: crate::dtype::Ty| crate::dtype::Ty::vector(Dtype::Pred, t.lanes);
        match self {
            Nop
            | Else { .. }
            | EndIf
            | LoopBegin { .. }
            | LoopEnd { .. }
            | Break
            | Continue
            | Exit
            | GridSync => {}
            AsyncCommit { .. } | AsyncWait { .. } | ClusterArrive { .. } | ClusterWait { .. } => {}
            SetMaxNReg { .. }
            | GridDepControl { .. }
            | TcgenRelinquish { .. }
            | TcgenWait { .. } => {}
            If { cond, .. } | LoopIf { cond, .. } => f(Use(*cond)),
            Assert { cond, msg } => {
                f(Use(*cond));
                if let Some(m) = msg {
                    f(Str(*m));
                }
            }
            Unsupported { reason } => f(Str(*reason)),
            Mov { dst, src } => {
                f(Def(*dst, None));
                f(Use(*src));
            }
            ReadSpecial { dst, .. } => f(Def(*dst, None)),
            ReadParam { dst, slot } => {
                f(Def(*dst, None));
                f(Param(*slot));
            }
            Unary { op, ty, dst, a } => {
                f(Ty(*ty));
                let out = if matches!(op, UnOp::IsNan | UnOp::IsInf | UnOp::IsFinite) {
                    pred_ty(*ty)
                } else {
                    *ty
                };
                f(Def(*dst, Some(out)));
                f(Use(*a));
            }
            Binary { ty, dst, a, b, .. } => {
                f(Ty(*ty));
                f(Def(*dst, Some(*ty)));
                f(Use(*a));
                f(Use(*b));
            }
            Ternary {
                ty, dst, a, b, c, ..
            } => {
                f(Ty(*ty));
                f(Def(*dst, Some(*ty)));
                f(Use(*a));
                f(Use(*b));
                f(Use(*c));
            }
            Compare { ty, dst, a, b, .. } => {
                f(Ty(*ty));
                f(Def(*dst, Some(pred_ty(*ty))));
                f(Use(*a));
                f(Use(*b));
            }
            Select {
                ty,
                dst,
                cond,
                a,
                b,
            } => {
                f(Ty(*ty));
                f(Def(*dst, Some(*ty)));
                f(Use(*cond));
                f(Use(*a));
                f(Use(*b));
            }
            Cast {
                from, to, dst, src, ..
            } => {
                f(Ty(*from));
                f(Def(*dst, Some(*to)));
                f(Use(*src));
            }
            Ptx {
                op,
                dsts,
                srcs,
                pred,
                ..
            } => {
                f(Op(*op));
                dsts.iter().for_each(|d| f(Def(*d, None)));
                srcs.iter().for_each(|s| f(Use(*s)));
                opt(pred, f);
            }
            LoadRegIndexed {
                dst,
                base,
                len,
                idx,
            } => {
                f(Def(*dst, None));
                f(Regs(*base, *len));
                f(Use(*idx));
            }
            StoreRegIndexed {
                base,
                len,
                idx,
                value,
            } => {
                f(Regs(*base, *len));
                f(Use(*idx));
                f(Use(*value));
            }
            Shfl {
                ty,
                dst,
                dst_pred,
                src,
                lane,
                clamp,
                membermask,
                ..
            } => {
                f(Ty(*ty));
                f(Def(*dst, Some(*ty)));
                if let Some(p) = dst_pred {
                    f(Def(*p, None));
                }
                [src, lane, clamp, membermask]
                    .iter()
                    .for_each(|o| f(Use(**o)));
            }
            Vote {
                dst,
                pred,
                membermask,
                ..
            } => {
                f(Def(*dst, None));
                f(Use(*pred));
                f(Use(*membermask));
            }
            Redux {
                ty,
                dst,
                src,
                membermask,
                ..
            } => {
                f(Ty(*ty));
                f(Def(*dst, Some(*ty)));
                f(Use(*src));
                f(Use(*membermask));
            }
            Elect {
                dst_pred,
                dst_lane,
                membermask,
            } => {
                f(Def(*dst_pred, None));
                if let Some(l) = dst_lane {
                    f(Def(*l, None));
                }
                f(Use(*membermask));
            }
            WarpSync { membermask } => f(Use(*membermask)),
            LdMatrix { dsts, addr, .. } => {
                dsts.iter().for_each(|d| f(Def(*d, None)));
                f(Use(*addr));
            }
            StMatrix { srcs, addr, .. } => {
                srcs.iter().for_each(|s| f(Use(*s)));
                f(Use(*addr));
            }
            Load {
                ty,
                dst,
                buf,
                offset,
                mods: m,
                ..
            } => {
                f(Def(*dst, Some(*ty)));
                f(Buf(*buf));
                f(Use(*offset));
                mods(m, f);
            }
            Store {
                ty,
                buf,
                offset,
                value,
                mods: m,
                ..
            } => {
                f(Ty(*ty));
                f(Buf(*buf));
                f(Use(*offset));
                f(Use(*value));
                mods(m, f);
            }
            LoadAddr {
                ty,
                dst,
                addr,
                mods: m,
                ..
            } => {
                f(Def(*dst, Some(*ty)));
                f(Use(*addr));
                mods(m, f);
            }
            StoreAddr {
                ty,
                addr,
                value,
                mods: m,
                ..
            } => {
                f(Ty(*ty));
                f(Use(*addr));
                f(Use(*value));
                mods(m, f);
            }
            AddrOf { dst, buf, offset } => {
                f(Def(*dst, Some(crate::dtype::Ty::U64)));
                f(Buf(*buf));
                f(Use(*offset));
            }
            Atom {
                ty,
                dst,
                addr,
                value,
                cmp,
                ..
            } => {
                f(Ty(*ty));
                if let Some(d) = dst {
                    f(Def(*d, Some(*ty)));
                }
                f(Use(*addr));
                f(Use(*value));
                opt(cmp, f);
            }
            StBulk { addr, size, .. } => {
                f(Use(*addr));
                f(Use(*size));
            }
            Discard { addr, .. } => f(Use(*addr)),
            Cvta { dst, src, .. } | Isspacep { dst, src, .. } | GetCtaRank { dst, src, .. } => {
                f(Def(*dst, None));
                f(Use(*src));
            }
            Mapa { dst, src, rank, .. } => {
                f(Def(*dst, None));
                f(Use(*src));
                f(Use(*rank));
            }
            CpAsync {
                dst,
                src,
                src_size,
                ignore_src,
                mods: m,
                ..
            } => {
                f(Use(*dst));
                f(Use(*src));
                opt(src_size, f);
                opt(ignore_src, f);
                mods(m, f);
            }
            CpAsyncMbarArrive { mbar, .. } => f(Use(*mbar)),
            BulkCopy(a) => {
                [a.dst, a.src, a.size].iter().for_each(|o| f(Use(*o)));
                completion(&a.completion, f);
                opt(&a.multicast, f);
                opt(&a.byte_mask, f);
                if let Some(io) = &a.ignore_oob {
                    opt(&io.ignore_bytes_left, f);
                    opt(&io.ignore_bytes_right, f);
                }
                mods(&a.mods, f);
            }
            Tma(a) => {
                f(Use(a.tmap));
                a.coords
                    .iter()
                    .chain(&a.im2col_offsets)
                    .for_each(|o| f(Use(*o)));
                f(Use(a.smem));
                completion(&a.completion, f);
                opt(&a.multicast, f);
                a.overrides.iter().for_each(|o| f(Use(o.value)));
                mods(&a.mods, f);
            }
            StAsync(a) => {
                f(Ty(a.ty));
                [a.value, a.addr].iter().for_each(|o| f(Use(*o)));
                opt(&a.mbar, f);
            }
            TensorMapReplace { tmap, value, .. } => {
                f(Use(*tmap));
                f(Use(*value));
            }
            TensorMapCopyFence { dst, src, .. } => {
                f(Use(*dst));
                f(Use(*src));
            }
            Barrier {
                kind, id, count, ..
            } => {
                if let BarKind::Red { pred, dst, .. } = kind {
                    f(Use(*pred));
                    f(Def(*dst, None));
                }
                f(Use(*id));
                opt(count, f);
            }
            MbarInit { mbar, count, .. } => {
                f(Use(*mbar));
                f(Use(*count));
            }
            MbarInval { mbar, .. } => f(Use(*mbar)),
            MbarArrive(a) => {
                f(Use(a.mbar));
                opt(&a.count, f);
                opt(&a.expect_tx, f);
                opt(&a.multicast, f);
                if let Some(s) = a.state {
                    f(Def(s, None));
                }
            }
            MbarTx {
                mbar,
                bytes,
                multicast,
                ..
            } => {
                f(Use(*mbar));
                f(Use(*bytes));
                opt(multicast, f);
            }
            MbarTestWait {
                mbar,
                phase: p,
                dst,
                report,
                report_value,
                ..
            } => {
                f(Use(*mbar));
                phase(p, f);
                for d in [dst, report, report_value].into_iter().flatten() {
                    f(Def(*d, None));
                }
            }
            MbarWait { mbar, phase: p, .. } => {
                f(Use(*mbar));
                phase(p, f);
            }
            MbarQuery { dst, op } => {
                f(Def(*dst, None));
                match op {
                    MbarQueryOp::PendingCount { state } => f(Use(*state)),
                    MbarQueryOp::CheckLayout { mbar, .. } => f(Use(*mbar)),
                }
            }
            Fence { kind, .. } => {
                if let FenceKind::TensormapAcquire { addr, .. } = kind {
                    f(Use(*addr));
                }
            }
            WaitUntil {
                dst,
                addr,
                ty,
                pred,
                captures,
                ..
            } => {
                f(Def(*dst, Some(*ty)));
                f(Use(*addr));
                f(Pred(*pred));
                captures.iter().for_each(|c| f(Use(Operand::Reg(*c))));
            }
            ClcTryCancel { resp, mbar, .. } => {
                f(Use(*resp));
                f(Use(*mbar));
            }
            TcgenAlloc { dst, ncols, .. } => {
                f(Use(*dst));
                f(Use(*ncols));
            }
            TcgenDealloc { taddr, ncols, .. } => {
                f(Use(*taddr));
                f(Use(*ncols));
            }
            TcgenCommit {
                mbar, multicast, ..
            } => {
                f(Use(*mbar));
                opt(multicast, f);
            }
            TcgenLd(a) => {
                a.dsts.iter().for_each(|d| f(Def(*d, None)));
                f(Use(a.taddr));
                f(Use(a.row));
                f(Use(a.col));
                if let Some((_, regs)) = &a.red {
                    regs.iter().for_each(|d| f(Def(*d, None)));
                }
            }
            TcgenSt(a) => {
                a.srcs.iter().for_each(|s| f(Use(*s)));
                f(Use(a.taddr));
                f(Use(a.row));
                f(Use(a.col));
            }
            TcgenCp(a) => {
                f(Use(a.taddr));
                f(Use(a.row));
                f(Use(a.col));
                f(Use(a.sdesc));
            }
            TcgenMma(a) => {
                let ta = match a.a {
                    TcA::Smem(o) | TcA::Tmem(o) => o,
                };
                [a.d, ta, a.b_desc, a.idesc, a.enable_input_d]
                    .iter()
                    .for_each(|o| f(Use(*o)));
                if let Some((x, y, _)) = a.block_scale {
                    f(Use(x));
                    f(Use(y));
                }
                opt(&a.scale_input_d, f);
                opt(&a.sparse_meta, f);
                a.disable_output_lane.iter().for_each(|o| f(Use(*o)));
                opt(&a.lut_b_addr, f);
            }
            Tile(t) => {
                for arg in &t.args {
                    match *arg {
                        TileArg::Region { buf, base, map } => {
                            f(Buf(buf));
                            f(Use(base));
                            f(Layout(map));
                        }
                        TileArg::Frag { first, map } => {
                            f(Use(Operand::Reg(first)));
                            f(Layout(map));
                        }
                        TileArg::Scalar(o) => f(Use(o)),
                    }
                }
                if let Some(c) = &t.completion {
                    completion(c, f);
                }
            }
        }
    }
}

impl BufferDecl {
    /// Bit offset of element `offset` (in units of `dtype.elem`) from the
    /// buffer base: `offset * dtype.elem.bits()`. See `Instr::Load` for the
    /// sub-byte alignment rule.
    pub fn bit_offset(&self, offset: i64) -> Option<i64> {
        offset.checked_mul(self.dtype.elem.bits() as i64)
    }
    /// Byte offset of element `offset`, or `None` if it is not byte-aligned
    /// (sub-byte element at an odd position) or overflows.
    pub fn byte_offset(&self, offset: i64) -> Option<i64> {
        let bits = self.bit_offset(offset)?;
        (bits % 8 == 0).then_some(bits / 8)
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
            If {
                cond,
                else_pc,
                end_pc,
                elect,
            } => {
                write!(
                    f,
                    "If {cond}{} else={else_pc} end={end_pc}",
                    if *elect { " elect" } else { "" }
                )
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
            Ternary {
                op,
                ty,
                dst,
                a,
                b,
                c,
            } => write!(f, "Ternary {op:?}.{ty} {dst} <- {a}, {b}, {c}"),
            Compare { op, ty, dst, a, b } => write!(f, "Compare {op:?}.{ty} {dst} <- {a}, {b}"),
            Select {
                ty,
                dst,
                cond,
                a,
                b,
            } => write!(f, "Select.{ty} {dst} <- {cond} ? {a} : {b}"),
            Cast {
                from, to, dst, src, ..
            } => write!(f, "Cast {to}.{from} {dst} <- {src}"),
            Ptx {
                op,
                dsts,
                srcs,
                pred,
                ..
            } => {
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
            Load {
                ty,
                dst,
                buf,
                offset,
                sem,
                scope,
                ..
            } => {
                write!(
                    f,
                    "Load{} {ty} {dst} <- {buf}[{offset}]",
                    sem_scope(*sem, *scope)
                )
            }
            Store {
                ty,
                buf,
                offset,
                value,
                sem,
                scope,
                ..
            } => {
                write!(
                    f,
                    "Store{} {ty} {buf}[{offset}] <- {value}",
                    sem_scope(*sem, *scope)
                )
            }
            LoadAddr {
                ty,
                dst,
                addr,
                space,
                sem,
                scope,
                ..
            } => {
                write!(
                    f,
                    "LoadAddr{} {} {ty} {dst} <- [{addr}]",
                    sem_scope(*sem, *scope),
                    space.name()
                )
            }
            StoreAddr {
                ty,
                addr,
                space,
                value,
                sem,
                scope,
                ..
            } => {
                write!(
                    f,
                    "StoreAddr{} {} {ty} [{addr}] <- {value}",
                    sem_scope(*sem, *scope),
                    space.name()
                )
            }
            AddrOf { dst, buf, offset } => write!(f, "AddrOf {dst} <- {buf}[{offset}]"),
            Barrier {
                kind, id, count, ..
            } => {
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
        writeln!(
            f,
            "kernel {} grid={:?} cluster={:?} block={:?}",
            self.name, t.grid, t.cluster, t.block
        )?;
        for (i, k) in self.consts.iter().enumerate() {
            writeln!(f, "  k{i} = {} {:#x}", k.ty, k.bits)?;
        }
        for (i, p) in self.host_abi.iter().enumerate() {
            writeln!(f, "  param{i} {}: {:?}", p.name, p.kind)?;
        }
        for (i, b) in self.buffers.iter().enumerate() {
            writeln!(
                f,
                "  b{i} {}: {:?} {} base={}",
                b.name, b.space, b.dtype, b.base
            )?;
        }
        for (i, o) in self.ops.iter().enumerate() {
            writeln!(f, "  op{i} = {} {:?}", o.name, o.mods)?;
        }
        let mut depth = 0usize;
        for (pc, ins) in self.code.iter().enumerate() {
            if matches!(
                ins,
                Instr::Else { .. } | Instr::EndIf | Instr::LoopEnd { .. }
            ) {
                depth = depth.saturating_sub(1);
            }
            let site = self.code_sites.get(pc).copied().unwrap_or(SiteId::NONE);
            let at = if site.is_none() {
                String::new()
            } else {
                format!("  @{site}")
            };
            writeln!(f, "{pc:04}  {}{ins}{at}", "  ".repeat(depth))?;
            if matches!(
                ins,
                Instr::If { .. } | Instr::Else { .. } | Instr::LoopBegin { .. }
            ) {
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
        let e = DimExpr::CeilDiv(
            Box::new(DimExpr::Param(ParamId(0))),
            Box::new(DimExpr::Const(128)),
        );
        assert_eq!(e.eval(&|_| Some(1000)), Some(8));
        let e = DimExpr::FloorDiv(Box::new(DimExpr::Const(-7)), Box::new(DimExpr::Const(2)));
        assert_eq!(e.eval(&|_| None), Some(-4));
        assert_eq!(DimExpr::Param(ParamId(0)).eval(&|_| None), None);
    }

    #[test]
    fn instr_size_is_bounded() {
        assert!(
            std::mem::size_of::<Instr>() <= 128,
            "Instr is {} bytes",
            std::mem::size_of::<Instr>()
        );
    }

    fn base() -> Program {
        let mut p = Program::empty("t", 32);
        p.regs.push(RegDecl {
            ty: Ty::U32,
            name: None,
            uniform: false,
        });
        p.regs.push(RegDecl {
            ty: Ty::PRED,
            name: None,
            uniform: false,
        });
        p.consts.push(Const {
            ty: Ty::U32,
            bits: 7,
        });
        p
    }

    fn with_code(mut p: Program, code: Vec<Instr>) -> Program {
        p.code_sites = vec![SiteId::NONE; code.len()];
        p.code = code;
        p
    }

    #[test]
    fn validate_rejects_corrupt_programs() {
        let k = Operand::Const(ConstId(0));
        // A 128-bit write into a 32-bit register.
        let p = with_code(
            base(),
            vec![Instr::Binary {
                op: BinOp::Add,
                ty: Ty::vector(Dtype::U32, 4),
                dst: Reg(0),
                a: k,
                b: k,
            }],
        );
        assert!(p.validate().is_err());
        // lanes = 0.
        let p = with_code(
            base(),
            vec![Instr::Binary {
                op: BinOp::Add,
                ty: Ty::vector(Dtype::U32, 0),
                dst: Reg(0),
                a: k,
                b: k,
            }],
        );
        assert!(p.validate().is_err());
        // Const wider than its type.
        let mut p = base();
        p.consts[0].bits = 1 << 40;
        assert!(p.validate().is_err());
        // Else pointing at another frame's EndIf.
        let c = Operand::Reg(Reg(1));
        let p = with_code(
            base(),
            vec![
                Instr::If {
                    cond: c,
                    else_pc: Pc(1),
                    end_pc: Pc(4),
                    elect: false,
                },
                Instr::Else { end_pc: Pc(3) },
                Instr::If {
                    cond: c,
                    else_pc: Pc(3),
                    end_pc: Pc(3),
                    elect: false,
                },
                Instr::EndIf,
                Instr::EndIf,
            ],
        );
        assert!(p.validate().is_err());
        // Main body falling through into predicate code.
        let mut p = with_code(
            base(),
            vec![
                Instr::Nop,
                Instr::Mov {
                    dst: Reg(0),
                    src: k,
                },
            ],
        );
        p.preds.push(PredProgram {
            arg: Reg(0),
            start: Pc(1),
            end: Pc(2),
            result: Reg(1),
            reads_memory: false,
        });
        assert!(p.validate().is_err());
        let mut p = with_code(
            base(),
            vec![
                Instr::Exit,
                Instr::Mov {
                    dst: Reg(0),
                    src: k,
                },
            ],
        );
        p.preds.push(PredProgram {
            arg: Reg(0),
            start: Pc(1),
            end: Pc(2),
            result: Reg(1),
            reads_memory: false,
        });
        assert!(p.validate().is_ok());
        // Out-of-range string.
        let p = with_code(base(), vec![Instr::Unsupported { reason: StrId(3) }]);
        assert!(p.validate().is_err());
    }

    #[test]
    fn serde_is_strict() {
        let ok = r#"{"Assert":{"cond":{"Reg":0},"msg":null}}"#;
        assert!(serde_json::from_str::<Instr>(ok).is_ok());
        let missing = r#"{"Assert":{"cond":{"Reg":0}}}"#;
        assert!(serde_json::from_str::<Instr>(missing).is_err());
        let unknown = r#"{"Assert":{"cond":{"Reg":0},"msg":null,"mesage":null}}"#;
        assert!(serde_json::from_str::<Instr>(unknown).is_err());
        assert!(serde_json::from_str::<Ty>(r#"{"elem":"F32","lanes":1,"x":0}"#).is_err());
    }

    #[test]
    fn sub_byte_offsets() {
        let b = BufferDecl {
            name: "a".into(),
            space: Space::Shared,
            dtype: Ty::scalar(Dtype::E2M1),
            shape: vec![],
            strides: vec![],
            param_slot: None,
            base: 0,
            base_reg: None,
            byte_len: None,
            align: 16,
            view_of: None,
            sync_words: false,
        };
        assert_eq!(b.byte_offset(6), Some(3));
        assert_eq!(b.byte_offset(5), None);
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
