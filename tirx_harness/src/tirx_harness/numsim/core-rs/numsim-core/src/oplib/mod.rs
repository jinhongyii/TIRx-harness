//! OpLib: pure numerics. No engine state, no observers, no memory.
//!
//! # Conventions (contract for W4 and every caller)
//!
//! * Functions are pure: inputs are register lane cells
//!   (`&WarpValue<u64>`, encoding per [`crate::value`]) or typed lanes
//!   (`&WarpValue<T: Scalar>`), plus the active `WarpMask`. Outputs are
//!   written only for lanes in the mask (`out: &mut WarpValue<u64>`); other
//!   lanes are left untouched.
//! * Bit-exactness: results must match the GPU bit for bit, including
//!   NaN payloads/canonical NaN, denormal flushing (`ftz`), saturation and
//!   all rounding modes. Approximate ops (`.approx`, `ex2`, `tanh`, `rcp
//!   .approx`) use a documented deterministic representative.
//! * Errors are [`OpError`]: `Unsupported` for valid PTX we do not model
//!   (fails closed as incomplete), `Invalid` for operand values PTX calls
//!   undefined/illegal.
//! * Dispatch on `Ty`/`Dtype` happens *inside* these functions with
//!   `#[inline]` so the codegen backend's constant types fold.
//! * The op registry ([`registry`]) is the single source of
//!   `SUPPORTED_OPS.md` ([`render_supported_ops_md`]).

mod ptx;
mod registry;
mod simd;
pub mod supported_ops;
mod tc;
mod tir;
mod tma;
mod warp;
#[cfg(test)]
mod tests;

use crate::dtype::{Dtype, Ty};
use crate::program::{BinOp, CmpOp, OpKey, ReduxOp, Rounding, ShflMode, TerOp, TmapField, UnOp};
use crate::value::{WarpMask, WarpValue};
use std::fmt;

// ---------------------------------------------------------------------------
// Scalar
// ---------------------------------------------------------------------------

/// A Rust value type with a fixed [`Dtype`] and a bit-exact register-lane
/// encoding (low `DTYPE.bits()` bits of a u64, zero-extended).
pub trait Scalar: Copy + Default + PartialEq + fmt::Debug + Send + Sync + 'static {
    const DTYPE: Dtype;
    /// Decode from a lane cell (ignores bits above `DTYPE.bits()`).
    fn from_bits(bits: u64) -> Self;
    /// Encode to a lane cell (zero-extended).
    fn to_bits(self) -> u64;
}

/// Floating-point formats: exact widening to f64 and correctly rounded
/// narrowing (the bit-exact conversion hooks used by `cvt`).
pub trait FloatScalar: Scalar {
    /// Exact value as f64 (every format here up to f32 is exactly
    /// representable; for f64 this is the identity). NaN maps to a NaN.
    fn to_f64(self) -> f64;
    /// Round an f64 to this format (`sat` = saturate to finite). Must not
    /// double-round: implementations round from the exact f64 bits.
    fn from_f64(x: f64, rnd: Rounding, sat: bool) -> Self;
}

macro_rules! int_scalar {
    ($t:ty, $d:expr, $u:ty) => {
        impl Scalar for $t {
            const DTYPE: Dtype = $d;
            #[inline]
            fn from_bits(bits: u64) -> Self {
                bits as $u as $t
            }
            #[inline]
            fn to_bits(self) -> u64 {
                self as $u as u64
            }
        }
    };
}

int_scalar!(u8, Dtype::U8, u8);
int_scalar!(u16, Dtype::U16, u16);
int_scalar!(u32, Dtype::U32, u32);
int_scalar!(u64, Dtype::U64, u64);
int_scalar!(i8, Dtype::S8, u8);
int_scalar!(i16, Dtype::S16, u16);
int_scalar!(i32, Dtype::S32, u32);
int_scalar!(i64, Dtype::S64, u64);

impl Scalar for bool {
    const DTYPE: Dtype = Dtype::Pred;
    #[inline]
    fn from_bits(bits: u64) -> Self {
        bits & 1 != 0
    }
    #[inline]
    fn to_bits(self) -> u64 {
        self as u64
    }
}

impl Scalar for f32 {
    const DTYPE: Dtype = Dtype::F32;
    #[inline]
    fn from_bits(bits: u64) -> Self {
        f32::from_bits(bits as u32)
    }
    #[inline]
    fn to_bits(self) -> u64 {
        f32::to_bits(self) as u64
    }
}

impl Scalar for f64 {
    const DTYPE: Dtype = Dtype::F64;
    #[inline]
    fn from_bits(bits: u64) -> Self {
        f64::from_bits(bits)
    }
    #[inline]
    fn to_bits(self) -> u64 {
        f64::to_bits(self)
    }
}

macro_rules! storage_float {
    ($(#[$m:meta])* $name:ident, $repr:ty, $d:expr) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        #[repr(transparent)]
        pub struct $name(pub $repr);
        impl Scalar for $name {
            const DTYPE: Dtype = $d;
            #[inline]
            fn from_bits(bits: u64) -> Self {
                $name((bits & ((1u64 << $d.bits()) - 1)) as $repr)
            }
            #[inline]
            fn to_bits(self) -> u64 {
                self.0 as u64
            }
        }
    };
}

storage_float!(
    /// IEEE binary16 bits.
    F16, u16, Dtype::F16);
storage_float!(
    /// bfloat16 bits.
    BF16, u16, Dtype::BF16);
storage_float!(
    /// float8 e4m3fn bits.
    E4M3, u8, Dtype::E4M3);
storage_float!(
    /// float8 e5m2 bits.
    E5M2, u8, Dtype::E5M2);
storage_float!(
    /// unsigned e8m0 scale bits.
    UE8M0, u8, Dtype::UE8M0);
storage_float!(
    /// float6 e2m3 bits (low 6 bits).
    E2M3, u8, Dtype::E2M3);
storage_float!(
    /// float6 e3m2 bits (low 6 bits).
    E3M2, u8, Dtype::E3M2);
storage_float!(
    /// float4 e2m1 bits (low 4 bits).
    E2M1, u8, Dtype::E2M1);

impl FloatScalar for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(x: f64, rnd: Rounding, sat: bool) -> Self {
        tir::f32_from_f64(x, rnd, sat)
    }
}

impl FloatScalar for f64 {
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64(x: f64, _rnd: Rounding, _sat: bool) -> Self {
        x
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpErrorKind {
    /// Valid PTX outside the modeled domain: fail closed (incomplete).
    Unsupported,
    /// Operand value PTX calls illegal/undefined: kernel error.
    Invalid,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpError {
    pub kind: OpErrorKind,
    pub message: String,
}

impl OpError {
    /// An `Unsupported` error: the op or modifier is outside what v2 models.
    pub fn unsupported(m: impl Into<String>) -> OpError {
        OpError { kind: OpErrorKind::Unsupported, message: m.into() }
    }
    /// An `Invalid` error: an operand value PTX calls illegal or undefined.
    pub fn invalid(m: impl Into<String>) -> OpError {
        OpError { kind: OpErrorKind::Invalid, message: m.into() }
    }
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for OpError {}

/// Lift a `numsim-oplib` error (plain message) into the contract error.
/// Messages naming an unmodeled/unsupported form fail closed as
/// `Unsupported`; everything else is an operand error (`Invalid`). Reserved
/// bits set in an operand are always an operand error, even when the legacy
/// text says "unsupported" (`raw tcgen05.cp descriptor uses unsupported
/// reserved/base/LBO-mode bits`: legacy reported it as an error).
impl From<numsim_oplib::types::OpError> for OpError {
    fn from(error: numsim_oplib::types::OpError) -> OpError {
        let message = error.0;
        let lower = message.to_ascii_lowercase();
        if lower.contains("reserved") {
            OpError::invalid(message)
        } else if lower.contains("unsupported")
            || lower.contains("unmodeled")
            || lower.contains("not modeled")
            || lower.contains("no legacy")
            || lower.contains("cannot parse")
        {
            OpError::unsupported(message)
        } else {
            OpError::invalid(message)
        }
    }
}

pub type OpResult<T = ()> = Result<T, OpError>;

// ---------------------------------------------------------------------------
// TIR-level ALU entry points (Instr::Unary/Binary/Ternary/Compare/Select/Cast)
//
// Values <= 64 bits: one slot per operand. Wide values (vector Ty > 64
// bits) pass all slots: `a[i]` is slot i.
// ---------------------------------------------------------------------------

/// `Instr::Unary` (TIR semantics; vector `ty` = element-wise).
#[inline]
pub fn unary(op: UnOp, ty: Ty, a: &[WarpValue<u64>], out: &mut [WarpValue<u64>], mask: WarpMask) -> OpResult {
    tir::unary(op, ty, a, out, mask)
}

/// `Instr::Binary`.
#[inline]
pub fn binary(
    op: BinOp,
    ty: Ty,
    a: &[WarpValue<u64>],
    b: &[WarpValue<u64>],
    out: &mut [WarpValue<u64>],
    mask: WarpMask,
) -> OpResult {
    tir::binary(op, ty, a, b, out, mask)
}

/// `Instr::Ternary`.
#[inline]
pub fn ternary(
    op: TerOp,
    ty: Ty,
    a: &[WarpValue<u64>],
    b: &[WarpValue<u64>],
    c: &[WarpValue<u64>],
    out: &mut [WarpValue<u64>],
    mask: WarpMask,
) -> OpResult {
    tir::ternary(op, ty, a, b, c, out, mask)
}

/// `Instr::Compare`: lanes (within `mask`) where `a op b` holds.
#[inline]
pub fn compare(op: CmpOp, ty: Ty, a: &[WarpValue<u64>], b: &[WarpValue<u64>], mask: WarpMask) -> OpResult<WarpMask> {
    tir::compare(op, ty, a, b, mask)
}

/// `Instr::Cast` (TIR/C semantics with optional rounding/saturation).
#[inline]
pub fn cast(
    from: Ty,
    to: Ty,
    rnd: Rounding,
    sat: bool,
    src: &[WarpValue<u64>],
    out: &mut [WarpValue<u64>],
    mask: WarpMask,
) -> OpResult {
    tir::cast(from, to, rnd, sat, src, out, mask)
}

/// Single-value conversion of raw bits (tile ops, TMA/MMA element paths).
pub fn convert_bits(from: Ty, to: Ty, rnd: Rounding, sat: bool, src: u128) -> OpResult<u128> {
    tir::convert_bits(from, to, rnd, sat, src)
}

// ---------------------------------------------------------------------------
// Generic PTX ops (Instr::Ptx)
// ---------------------------------------------------------------------------

/// Operands of one `Instr::Ptx` execution, flattened to slots.
pub struct PtxIo<'a> {
    /// Destination slots, all destinations concatenated in order.
    pub dsts: &'a mut [WarpValue<u64>],
    pub dst_tys: &'a [Ty],
    /// Source slots, all sources concatenated in order (consts broadcast).
    pub srcs: &'a [WarpValue<u64>],
    pub src_tys: &'a [Ty],
    /// Lanes that execute (active & guard predicate). Pure warp-collective
    /// ops (movmatrix, match) read all lanes of `srcs` but write only these.
    pub mask: WarpMask,
}

/// A resolved generic op, resolved once at program load: either a plain fn
/// item (parameterless hot forms) or a closure that carries the parsed
/// modifiers/operand layout. `Copy` and cheap to call; invoke with
/// [`PtxFn::call`].
#[derive(Clone, Copy)]
pub struct PtxFn {
    imp: PtxImpl,
}

/// A parameterized op body (interned and leaked once per distinct
/// `(OpKey, tys)` per process, so `PtxFn` stays `Copy`).
pub type PtxOp = dyn Fn(&mut PtxIo<'_>) -> OpResult + Send + Sync;

#[derive(Clone, Copy)]
enum PtxImpl {
    Direct(fn(&mut PtxIo<'_>) -> OpResult),
    Data(&'static PtxOp),
}

impl PtxFn {
    /// Wrap a plain function (no captured data).
    pub const fn new(f: fn(&mut PtxIo<'_>) -> OpResult) -> PtxFn {
        PtxFn { imp: PtxImpl::Direct(f) }
    }
    /// Wrap a `'static` closure.
    pub const fn from_static(op: &'static PtxOp) -> PtxFn {
        PtxFn { imp: PtxImpl::Data(op) }
    }
    /// Run the op over `io`.
    #[inline]
    pub fn call(&self, io: &mut PtxIo<'_>) -> OpResult {
        match self.imp {
            PtxImpl::Direct(f) => f(io),
            PtxImpl::Data(op) => op(io),
        }
    }
    /// Whether this is a plain fn item (no captured data).
    pub fn is_direct(&self) -> bool {
        matches!(self.imp, PtxImpl::Direct(_))
    }
    /// Identity (same fn item or same interned closure).
    pub fn same(&self, other: &PtxFn) -> bool {
        match (self.imp, other.imp) {
            (PtxImpl::Direct(a), PtxImpl::Direct(b)) => a as usize == b as usize,
            (PtxImpl::Data(a), PtxImpl::Data(b)) => std::ptr::addr_eq(a as *const PtxOp, b as *const PtxOp),
            _ => false,
        }
    }
}

impl From<fn(&mut PtxIo<'_>) -> OpResult> for PtxFn {
    fn from(f: fn(&mut PtxIo<'_>) -> OpResult) -> PtxFn {
        PtxFn::new(f)
    }
}

impl fmt::Debug for PtxFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.imp {
            PtxImpl::Direct(p) => write!(f, "PtxFn::Direct({:#x})", p as usize),
            PtxImpl::Data(op) => write!(f, "PtxFn::Data({:p})", op as *const PtxOp),
        }
    }
}

/// Resolve an interned `OpKey` (op name + canonical modifiers) for the given
/// operand types, once at program load. Unknown ops/modifiers are
/// `Unsupported` (fail closed); this is also the acceptance table lowering
/// mirrors.
pub fn resolve_ptx(key: &OpKey, dst_tys: &[Ty], src_tys: &[Ty]) -> OpResult<PtxFn> {
    ptx::resolve(key, dst_tys, src_tys)
}

/// Every op name `resolve_ptx` recognises (sorted). Individual modifier/type
/// forms of a listed name may still be rejected; use it for coverage reports
/// and lowering acceptance checks.
pub fn ptx_op_names() -> Vec<&'static str> {
    ptx::known_ops()
}

// ---------------------------------------------------------------------------
// Warp collectives
// ---------------------------------------------------------------------------

/// `shfl.sync` with legacy validation: `active` executes, each lane's
/// `membermask` names the participants. Returns (values written for `active`
/// lanes, lanes whose source was in range). A source lane that is not an
/// active participant is `Invalid` ("warp shuffle reads a non-participant
/// lane"), as is a lane missing from its own membermask.
#[inline]
pub fn shfl_sync(
    mode: ShflMode,
    src: &WarpValue<u64>,
    lane: &WarpValue<u64>,
    clamp: &WarpValue<u64>,
    membermask: &WarpValue<u64>,
    active: WarpMask,
) -> OpResult<(WarpValue<u64>, WarpMask)> {
    warp::shfl_sync(mode, src, lane, clamp, membermask, active)
}

/// `shfl.sync` (32-bit payload slots): (values, lanes whose source was in range).
/// Infallible compatibility form of [`shfl_sync`] (`members` = active =
/// participants); a non-member source reads its value with predicate false.
/// Prefer `shfl_sync`.
#[inline]
pub fn shfl(
    mode: ShflMode,
    src: &WarpValue<u64>,
    lane: &WarpValue<u64>,
    clamp: &WarpValue<u64>,
    members: WarpMask,
) -> (WarpValue<u64>, WarpMask) {
    warp::shfl(mode, src, lane, clamp, members)
}

/// `redux.sync` over `members`.
#[inline]
pub fn redux(op: ReduxOp, ty: Ty, src: &WarpValue<u64>, members: WarpMask) -> OpResult<u64> {
    warp::redux(op, ty, src, members)
}

// ---------------------------------------------------------------------------
// Tensor maps, descriptors, address generation
// ---------------------------------------------------------------------------

/// Im2col pixel bounding box of a tensor map (`cuTensorMapEncodeIm2col`
/// corners; `wide` = `CU_TENSOR_MAP_IM2COL_WIDE` maps for `im2col::w`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Im2colBox {
    /// Lower corner per spatial dim (D, H, W order as in the image: dims 1..).
    pub lower: [i16; 3],
    pub upper: [i16; 3],
    pub wide: bool,
}

/// Decoded CUtensorMap. The 128-byte in-memory encoding is NumSim's own
/// (hardware's is opaque); host ABI and `tensormap.replace` both go through
/// [`TensorMapDesc::encode`]/[`TensorMapDesc::decode`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TensorMapDesc {
    pub global_address: u64,
    pub rank: u8,
    pub elem: Option<Dtype>,
    /// Elements per dimension (innermost first).
    pub global_dim: [u64; 5],
    /// Byte strides of dims 1.. (`global_stride[i]` = dim `i + 1`; [4] = 0).
    pub global_stride: [u64; 5],
    pub box_dim: [u32; 5],
    pub element_stride: [u32; 5],
    /// 0 none, 1 = 16B, 2 = 32B.
    pub interleave: u8,
    /// 0 none, 1 = 32B, 2 = 64B, 3 = 128B, 4 = 128B/32B atoms,
    /// 5 = 128B/32B atoms + 8B flip, 6 = 128B/64B atoms, 7 = 96B.
    pub swizzle: u8,
    pub l2_promotion: u8,
    /// 0 = zero fill, 1 = NaN request-zero FMA fill.
    pub oob_fill: u8,
    /// Swizzle atomicity override (`tensormap.replace.swizzle_atomicity`
    /// numbering: 0 = 16B/default, 1 = 32B, 2 = 32B + 8B flip, 3 = 64B).
    /// 0 keeps the atomicity implied by `swizzle`; decode only sets it for
    /// combinations `swizzle` codes 4..6 cannot name (e.g. 64B width with 32B
    /// atoms, an intermediate `tensormap.replace` state).
    pub swizzle_atomicity: u8,
    /// Im2col bounding box; `None` = tiled map.
    pub im2col: Option<Im2colBox>,
    /// `elem == E2M1` only: the shared-memory layout is the 16-byte-aligned
    /// padded one (`CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B`, one FP4 element
    /// per byte pair slot) instead of the packed `16U4_ALIGN8B` (false).
    pub fp4_padded: bool,
    /// `elem == F32 / TF32` only: the flush-to-zero data type
    /// (`CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ` / `TFLOAT32_FTZ`,
    /// `tensormap.replace .elemtype` 8 / 12). Copies move the same bytes.
    pub elem_ftz: bool,
}

impl TensorMapDesc {
    pub const BYTES: usize = 128;
    /// Encode; an unencodable map (field out of descriptor range, no element
    /// type, ...) is an error.
    pub fn try_encode(&self) -> OpResult<[u8; 128]> {
        tma::try_encode(self)
    }
    /// [`Self::try_encode`], with unencodable maps encoded as all zeros (which
    /// [`Self::decode`] rejects, so every use fails closed).
    pub fn encode(&self) -> [u8; 128] {
        tma::encode(self)
    }
    /// Decode a 128-byte TensorMap payload; errors on any unencodable or inconsistent field.
    pub fn decode(bytes: &[u8; 128]) -> OpResult<TensorMapDesc> {
        tma::decode(bytes)
    }
    /// `tensormap.replace` (one field). `GlobalStrideUpper` only exists in
    /// per-instruction overrides: use [`TensorMapDesc::apply_overrides`].
    pub fn replace(&mut self, field: TmapField, ord: Option<u8>, value: u64) -> OpResult {
        tma::replace(self, field, ord, value)
    }
    /// All per-instruction overrides of one TMA instruction at once (legacy
    /// `override_tensor_map(dims[], lower_stride[], upper_stride)`):
    /// `GlobalDim` (per ord) and `GlobalStride` lower operands (per ord) are
    /// combined with the shared `GlobalStrideUpper` nibbles as
    /// `stride[ord] = (lower | upper_nibble(ord) << 32) << 4`; every other
    /// field is applied as `replace`. Call this instead of looping `replace`.
    pub fn apply_overrides(&mut self, overrides: &[(TmapField, Option<u8>, u64)]) -> OpResult {
        tma::apply_overrides(self, overrides)
    }
}

/// Fill of OOB box elements in a TMA load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TmaFill {
    #[default]
    Zero,
    /// `CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA`: every 16 bits of an
    /// OOB element read as the PTX OOB NaN `0x7ff7`.
    NanRequestZeroFma,
}

/// Byte-level plan of one TMA transfer: matched element runs between the
/// global tensor and the (swizzled) shared box, plus OOB-filled smem runs.
/// Global spans are *virtual addresses*; smem spans are window offsets.
/// `global` and `smem` have equal totals and pair byte-for-byte in
/// concatenation order (load: global -> smem; store/reduce: smem -> global).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TmaPlan {
    pub global: Vec<crate::arena::ByteSpan>,
    pub smem: Vec<crate::arena::ByteSpan>,
    /// Load only: smem bytes of OOB elements, filled per `fill`.
    pub smem_oob_fill: Vec<crate::arena::ByteSpan>,
    /// Transaction bytes (`complete_tx`; the full box for loads).
    pub bytes: u64,
    pub fill: TmaFill,
    /// Byte pattern written repeatedly over each `smem_oob_fill` span from its
    /// first byte (spans start on element boundaries). Empty = zeros.
    pub fill_pattern: Vec<u8>,
    /// Load of a TF32 map: each copied 4-byte element (not the OOB fill) is
    /// rounded f32 -> tf32 on landing (NaN -> `0x7fffe000`); see
    /// [`tma_tf32_round`].
    pub tf32_round: bool,
    /// Store of a sub-byte map (FP4 packed/padded, U6): masked partial-byte
    /// global writes, applied after the byte spans:
    /// `g = (g & !(mask << target_shift)) | (((s >> source_shift) & mask) << target_shift)`
    /// with `s` the shared byte at `smem` and `g` the global byte at `global`.
    pub global_bits: Vec<TmaBitFragment>,
}

/// One masked sub-byte TMA store write (see [`TmaPlan::global_bits`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TmaBitFragment {
    /// Global virtual address of the destination byte.
    pub global: u64,
    /// Shared-window offset of the source byte.
    pub smem: u64,
    pub source_shift: u8,
    pub target_shift: u8,
    pub mask: u8,
}

/// Direction of a TMA plan (`cp.reduce.async.bulk.tensor` plans as `Store`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TmaPlanDir {
    #[default]
    Load,
    Store,
}

/// Address generation for `cp.async.bulk.tensor` (tile / gather4 / scatter4
/// / im2col modes, swizzle, OOB), with the direction's checks. `coords`
/// innermost first (gather4/scatter4: `[col, row0..row3]`).
pub fn tma_plan_dir(
    map: &TensorMapDesc,
    dir: TmaPlanDir,
    mode: crate::program::TmaMode,
    coords: &[i64],
    im2col_offsets: &[i64],
    smem_offset: u64,
) -> OpResult<TmaPlan> {
    tma::plan(map, dir, mode, coords, im2col_offsets, smem_offset)
}

/// `cp.async.bulk.prefetch.tensor`: cache residency only, so no plan
/// (legacy `execute_tma_cache_hint`). Only the instruction's rank must
/// match the descriptor's (gather4: a rank-2 map, `[col, row0..row3]`). A
/// shared layout no transfer can use (e.g. a swizzled 16B interleave) is
/// therefore still prefetchable.
pub fn tma_prefetch_check(map: &TensorMapDesc, mode: crate::program::TmaMode, coords: &[i64]) -> OpResult {
    let rank = match mode {
        crate::program::TmaMode::TileGather4 if coords.len() == 5 => 2,
        _ => coords.len(),
    };
    if usize::from(map.rank) != rank {
        return Err(OpError::invalid(format!(
            "cp.async.bulk.prefetch.tensor rank specialization {rank} does not match descriptor rank {}",
            map.rank
        )));
    }
    Ok(())
}

/// Legacy entry: [`tma_plan_dir`] with `Load`, except the store-only modes
/// `Im2colNoOffs` / `TileScatter4`, which plan as `Store`.
pub fn tma_plan(
    map: &TensorMapDesc,
    mode: crate::program::TmaMode,
    coords: &[i64],
    im2col_offsets: &[i64],
    smem_offset: u64,
) -> OpResult<TmaPlan> {
    use crate::program::TmaMode;
    let dir = match mode {
        TmaMode::Im2colNoOffs | TmaMode::TileScatter4 => TmaPlanDir::Store,
        _ => TmaPlanDir::Load,
    };
    tma::plan(map, dir, mode, coords, im2col_offsets, smem_offset)
}

/// TMA's f32 -> tf32 landing conversion for `TmaPlan::tf32_round` (one
/// little-endian 4-byte element).
pub fn tma_tf32_round(bits: u32) -> u32 {
    numsim_oplib::tma::tma_f32_to_tf32(f32::from_bits(bits)).to_bits()
}

/// `cp.reduce.async.bulk.tensor`: is `.redOp` defined for the TensorMap
/// element type? The PTX table, as legacy `RawTmaReductionOp::resolve`:
/// `.add` u32/s32/u64/f32(tf32)/f16/bf16; `.min/.max` u32/s32/u64/s64/f16/
/// bf16; `.inc/.dec` u32; `.and/.or/.xor` any 32- or 64-bit type.
/// Undefined pairs are `Invalid`.
pub fn tma_reduce_valid(op: crate::program::AtomOp, dtype: Dtype) -> OpResult<()> {
    use crate::program::AtomOp as A;
    use Dtype as D;
    let ok = match op {
        A::Add => matches!(dtype, D::U32 | D::S32 | D::U64 | D::F32 | D::TF32 | D::F16 | D::BF16),
        A::Min | A::Max => matches!(dtype, D::U32 | D::S32 | D::U64 | D::S64 | D::F16 | D::BF16),
        A::Inc | A::Dec => dtype == D::U32,
        A::And | A::Or | A::Xor => matches!(dtype.bits(), 32 | 64),
        A::Exch | A::Cas => false,
    };
    if ok {
        Ok(())
    } else {
        Err(OpError::invalid(format!(
            "cp.reduce.async.bulk.tensor operation {op:?} is invalid for TensorMap dtype {dtype:?}"
        )))
    }
}

/// Decoded tcgen05/wgmma shared-memory matrix descriptor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SmemDesc {
    pub start: u32,
    pub lbo: u32,
    pub sbo: u32,
    pub base_offset: u8,
    pub lbo_mode: u8,
    pub swizzle: u8,
    pub version: u8,
}

/// Decode a shared-memory matrix descriptor with SM100 field widths (byte-valued
/// start/LBO/SBO, swizzle code 0..4). No numerics; errors on invalid descriptors.
pub fn decode_smem_desc(desc: u64) -> OpResult<SmemDesc> {
    tc::decode_smem_desc(desc)
}

/// tcgen05 instruction descriptor fields (idesc).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct InstrDesc {
    pub m: u16,
    pub n: u16,
    pub a: Option<Dtype>,
    pub b: Option<Dtype>,
    pub d: Option<Dtype>,
    pub a_major_mn: bool,
    pub b_major_mn: bool,
    pub negate_a: bool,
    pub negate_b: bool,
    pub sparse: bool,
    pub max_shift: u8,
    pub scale_type: Option<Dtype>,
}

/// Decode an idesc valid for `cta_group::1` or `::2` (SM100).
pub fn decode_instr_desc(idesc: u32, kind: crate::program::TcMmaKind) -> OpResult<InstrDesc> {
    tc::decode_instr_desc(idesc, kind)
}

/// Decode an idesc for one CTA group (1 or 2; SM100; `.ws` shapes accepted
/// for `cta_group::1`).
pub fn decode_instr_desc_for(idesc: u32, kind: crate::program::TcMmaKind, cta_group: u8) -> OpResult<InstrDesc> {
    tc::decode_instr_desc_for(idesc, kind, cta_group)
}

/// Target architecture of a tcgen05 instruction (descriptor field widths,
/// f8f6f4 K=64, LUT-B, SFA layouts, scale formats).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TcArch {
    #[default]
    Sm100,
    Sm103,
    Sm107,
}

/// Instruction facts of a `tcgen05.mma` that `TcgenMmaPayload` does not carry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TcMmaOptions {
    pub arch: TcArch,
    /// `kind::i8` with the `.ti16` (s1z4m11) operand spelling.
    pub ti16: bool,
    /// `.lut_b`: TMEM address (taddr, `lane << 16 | column`) of the lookup
    /// table (f8f6f4 / mxf8f6f4) — the value of `TcgenMmaArgs::lut_b_addr`
    /// (`addr@tmem`). The table is read through `tmem_read`, never smem.
    pub lut_b: Option<u32>,
    /// `.ws` zero-column-mask descriptor. `None` falls back to the payload's
    /// `disable_output_lane` words (`[lo]` or `[lo, hi]`, as lowering passes
    /// the mask operand there), else 0.
    pub zero_col_mask: Option<u64>,
    /// Block-scale `.block16/.block32` spelling (fixed vectors) rather than
    /// `.scale_vec::NX` (only differs for SM103/SM107 K=96/128).
    pub fixed_vectors: bool,
}

impl TcMmaOptions {
    /// Parse a `TcgenMmaArgs::variant` string: tokens separated by `.`, `,`,
    /// `;` or spaces among `ti16`, `fixed_vectors`/`block16`/`block32`,
    /// `sm_100[a]`, `sm_103[a]`, `sm_107[a]` (`lut_b` needs its address and
    /// `zero_col_mask` its value, so they are set as fields).
    pub fn parse_variant(variant: &str) -> OpResult<TcMmaOptions> {
        tc::parse_variant(variant)
    }
}

/// `smem(cta, addr, buf)` of [`tc_mma_ctas`].
pub type TcSmemRead<'a> = &'a dyn Fn(u32, u32, &mut [u8]) -> OpResult;
/// `tmem_read(cta, lane, col, buf)` of [`tc_mma_ctas`].
pub type TcTmemRead<'a> = &'a dyn Fn(u32, u32, u32, &mut [u8]) -> OpResult;
/// `tmem_write(cta, lane, col, bytes)` of [`tc_mma_ctas`].
pub type TcTmemWrite<'a> = &'a mut dyn FnMut(u32, u32, u32, &[u8]) -> OpResult;

/// tcgen05.mma numerics: reads A/B (and scales, metadata, LUT) through the
/// closures, reads and writes the D tile in tensor memory. No engine state.
///
/// `cta` is the CTA index within the issuing group: 0 for `cta_group::1`
/// (the issuing CTA), 0/1 = even/odd CTA of the pair for `cta_group::2`.
/// `smem(cta, addr, buf)` reads that CTA's shared window at a window byte
/// address; `tmem_read(cta, lane, col, buf)` / `tmem_write(cta, lane, col,
/// bytes)` access TMEM from cell (taddr `lane << 16 | col`): a buffer of
/// more than 4 bytes covers the following cells of the same lane (cell
/// `col + i` = bytes `4 * i ..`; a run never crosses a lane, W4-16).
/// `.ashift` shifts A's TMEM rows after the product (writes through
/// `tmem_write`). Collector qualifiers do not change numerics; the engine
/// tracks their state with [`tc_collector_transition`].
pub fn tc_mma_ctas(
    payload: &crate::sync::completion::TcgenMmaPayload,
    options: &TcMmaOptions,
    smem: TcSmemRead<'_>,
    tmem_read: TcTmemRead<'_>,
    tmem_write: TcTmemWrite<'_>,
) -> OpResult {
    tc::tc_mma_ctas(payload, options, smem, tmem_read, tmem_write)
}

/// Single-CTA wrapper of [`tc_mma_ctas`] (default options; `cta_group::2`
/// is an error because the closures reach one CTA).
pub fn tc_mma(
    payload: &crate::sync::completion::TcgenMmaPayload,
    smem: &dyn Fn(u32, &mut [u8]) -> OpResult,
    tmem_read: &dyn Fn(u32, u32, &mut [u8]) -> OpResult,
    tmem_write: &mut dyn FnMut(u32, u32, &[u8]) -> OpResult,
) -> OpResult {
    tc::tc_mma(payload, smem, tmem_read, tmem_write)
}

/// Per-issuing-lane collector buffer state transition (bit 0 = A, bits 1..5
/// = B0..B3; `fill` sets, `use` requires, `lastuse` requires + clears,
/// `discard` clears). `use`/`lastuse` of an unfilled slot is an error.
pub fn tc_collector_transition(
    state: u8,
    collector_a: crate::program::CollectorOp,
    collector_b: crate::program::CollectorOp,
    b_buffer: u8,
) -> OpResult<u8> {
    tc::collector_transition(state, collector_a, collector_b, b_buffer)
}

// ---------------------------------------------------------------------------
// Data-movement maps: tcgen05.ld/st/cp, ldmatrix/stmatrix (W4-7)
// ---------------------------------------------------------------------------

mod mem;

pub use mem::{
    ldmatrix_fragments, ldmatrix_plan, stmatrix_plan, stmatrix_writes, tcgen_cp_decode,
    tcgen_cp_plan, tcgen_ld_dst_count, tcgen_ld_reduce, tcgen_ld_spcompress, tcgen_ldst_map,
    tcgen_ldst_registers, LdMatrixPlan, MatrixAccess, StMatrixPlan, TcgenCpPlan, TcgenCpWord,
    TcgenCellRun, TcgenLdRed, TcgenLdstMap, TcgenLdstPiece,
};

// ---------------------------------------------------------------------------
// Op registry -> SUPPORTED_OPS.md
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fidelity {
    Modeled,
    DeterministicRepresentative,
    OrderingOnly,
    ExactProtocol,
    Rejected,
}

impl Fidelity {
    /// The SUPPORTED_OPS.md spelling of this fidelity.
    pub const fn name(self) -> &'static str {
        match self {
            Fidelity::Modeled => "modeled",
            Fidelity::DeterministicRepresentative => "deterministic_representative",
            Fidelity::OrderingOnly => "ordering_only",
            Fidelity::ExactProtocol => "exact_protocol",
            Fidelity::Rejected => "rejected",
        }
    }
}

/// One TIRx op as lowering accepts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpEntry {
    /// `tirx.ptx.ld`, `tirx.cuda.printf`, `tirx.tile.gemm`, ...
    pub name: &'static str,
    /// SUPPORTED_OPS family column.
    pub family: &'static str,
    pub fidelity: Fidelity,
    pub notes: &'static str,
    /// `Instr::family()` it lowers to ("" for rejected ops).
    pub instr: &'static str,
}

/// The registry: every TIRx op of the legacy `SUPPORTED_OPS.md` (generated
/// from `numsim-oplib`'s op table), with the `Instr::family()` it lowers to.
pub fn registry() -> &'static [OpEntry] {
    registry::entries()
}

/// Render SUPPORTED_OPS.md from registry entries. With [`registry()`] this
/// reproduces the legacy file byte-for-byte (NumSim ABI v38 header, CUDA/PTX
/// table then tile table, each sorted by name).
pub fn render_supported_ops_md(entries: &[OpEntry]) -> String {
    registry::render(entries)
}

