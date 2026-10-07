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
        let _ = (x, rnd, sat);
        unimplemented!("W4: f32::from_f64 with rounding modes")
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
    pub fn unsupported(m: impl Into<String>) -> OpError {
        OpError { kind: OpErrorKind::Unsupported, message: m.into() }
    }
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
    let _ = (op, ty, a, out, mask);
    unimplemented!("W4: oplib::unary")
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
    let _ = (op, ty, a, b, out, mask);
    unimplemented!("W4: oplib::binary")
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
    let _ = (op, ty, a, b, c, out, mask);
    unimplemented!("W4: oplib::ternary")
}

/// `Instr::Compare`: lanes (within `mask`) where `a op b` holds.
#[inline]
pub fn compare(op: CmpOp, ty: Ty, a: &[WarpValue<u64>], b: &[WarpValue<u64>], mask: WarpMask) -> OpResult<WarpMask> {
    let _ = (op, ty, a, b, mask);
    unimplemented!("W4: oplib::compare")
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
    let _ = (from, to, rnd, sat, src, out, mask);
    unimplemented!("W4: oplib::cast")
}

/// Single-value conversion of raw bits (tile ops, TMA/MMA element paths).
pub fn convert_bits(from: Ty, to: Ty, rnd: Rounding, sat: bool, src: u128) -> OpResult<u128> {
    let _ = (from, to, rnd, sat, src);
    unimplemented!("W4: oplib::convert_bits")
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

/// A resolved generic op.
pub type PtxFn = fn(&mut PtxIo<'_>) -> OpResult;

/// Resolve an interned `OpKey` (op name + canonical modifiers) for the given
/// operand types, once at program load. Unknown ops/modifiers are
/// `Unsupported` (fail closed); this is also the acceptance table lowering
/// mirrors.
pub fn resolve_ptx(key: &OpKey, dst_tys: &[Ty], src_tys: &[Ty]) -> OpResult<PtxFn> {
    match key.name.as_str() {
        // Trivial examples; W4 owns the table.
        "tirx.cuda.float_as_uint" | "tirx.cuda.uint_as_float" => Ok(ptx_bitcast32),
        _ => {
            let _ = (dst_tys, src_tys);
            Err(OpError::unsupported(format!("no oplib implementation for {} {:?}", key.name, key.mods)))
        }
    }
}

fn ptx_bitcast32(io: &mut PtxIo<'_>) -> OpResult {
    for l in io.mask.lanes() {
        io.dsts[0][l] = io.srcs[0][l] & 0xffff_ffff;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Warp collectives
// ---------------------------------------------------------------------------

/// `shfl.sync` (32-bit payload slots): (values, lanes whose source was in range).
#[inline]
pub fn shfl(
    mode: ShflMode,
    src: &WarpValue<u64>,
    lane: &WarpValue<u64>,
    clamp: &WarpValue<u64>,
    members: WarpMask,
) -> (WarpValue<u64>, WarpMask) {
    let _ = (mode, src, lane, clamp, members);
    unimplemented!("W4: oplib::shfl")
}

/// `redux.sync` over `members`.
#[inline]
pub fn redux(op: ReduxOp, ty: Ty, src: &WarpValue<u64>, members: WarpMask) -> OpResult<u64> {
    let _ = (op, ty, src, members);
    unimplemented!("W4: oplib::redux")
}

// ---------------------------------------------------------------------------
// Tensor maps, descriptors, address generation
// ---------------------------------------------------------------------------

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
    /// Byte strides of dims 1.. (dim 0 is contiguous).
    pub global_stride: [u64; 5],
    pub box_dim: [u32; 5],
    pub element_stride: [u32; 5],
    pub interleave: u8,
    /// 0 none, 1 = 32B, 2 = 64B, 3 = 128B, 4+ = 128B atom variants.
    pub swizzle: u8,
    pub l2_promotion: u8,
    /// 0 = zero fill, 1 = NaN request-zero FMA fill.
    pub oob_fill: u8,
}

impl TensorMapDesc {
    pub const BYTES: usize = 128;
    pub fn encode(&self) -> [u8; 128] {
        unimplemented!("W4: TensorMapDesc::encode")
    }
    pub fn decode(bytes: &[u8; 128]) -> OpResult<TensorMapDesc> {
        let _ = bytes;
        unimplemented!("W4: TensorMapDesc::decode")
    }
    /// `tensormap.replace` / per-instruction override.
    pub fn replace(&mut self, field: TmapField, ord: Option<u8>, value: u64) -> OpResult {
        let _ = (field, ord, value);
        unimplemented!("W4: TensorMapDesc::replace")
    }
}

/// Byte-level plan of one TMA transfer: matched element runs between the
/// global tensor and the (swizzled) shared box, plus OOB-filled smem runs.
/// Global spans are *virtual addresses*; smem spans are window offsets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TmaPlan {
    pub global: Vec<crate::arena::ByteSpan>,
    pub smem: Vec<crate::arena::ByteSpan>,
    pub smem_oob_fill: Vec<crate::arena::ByteSpan>,
    pub bytes: u64,
}

/// Address generation for `cp.async.bulk.tensor` (tile / im2col modes,
/// swizzle, OOB). `coords` innermost first.
pub fn tma_plan(
    map: &TensorMapDesc,
    mode: crate::program::TmaMode,
    coords: &[i64],
    im2col_offsets: &[i64],
    smem_offset: u64,
) -> OpResult<TmaPlan> {
    let _ = (map, mode, coords, im2col_offsets, smem_offset);
    unimplemented!("W4: oplib::tma_plan")
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

pub fn decode_smem_desc(desc: u64) -> OpResult<SmemDesc> {
    let _ = desc;
    unimplemented!("W4: decode_smem_desc")
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

pub fn decode_instr_desc(idesc: u32, kind: crate::program::TcMmaKind) -> OpResult<InstrDesc> {
    let _ = (idesc, kind);
    unimplemented!("W4: decode_instr_desc")
}

/// tcgen05.mma numerics: reads A/B (and scales) through the closures, reads
/// and writes the D tile in tensor memory through `tmem`. No engine state.
pub fn tc_mma(
    payload: &crate::sync::completion::TcgenMmaPayload,
    smem: &dyn Fn(u32, &mut [u8]) -> OpResult,
    tmem_read: &dyn Fn(u32, u32, &mut [u8]) -> OpResult,
    tmem_write: &mut dyn FnMut(u32, u32, &[u8]) -> OpResult,
) -> OpResult {
    let _ = (payload, smem, tmem_read, tmem_write);
    unimplemented!("W4: oplib::tc_mma")
}

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

/// The registry (W4 fills; lowering checks membership).
pub static OPS: &[OpEntry] = &[];

pub fn registry() -> &'static [OpEntry] {
    OPS
}

/// Render SUPPORTED_OPS.md from registry entries (stable order: by name).
pub fn render_supported_ops_md(entries: &[OpEntry]) -> String {
    let mut ops: Vec<&OpEntry> = entries.iter().filter(|e| !e.name.starts_with("tirx.tile.")).collect();
    ops.sort_by_key(|e| e.name);
    let mut tiles: Vec<&OpEntry> = entries.iter().filter(|e| e.name.starts_with("tirx.tile.")).collect();
    tiles.sort_by_key(|e| e.name);
    let mut s = String::from("# NumSim Engine Operation Support\n\n");
    s.push_str("This file is generated from NumSim's operation registry and is checked by tests.\n\n");
    s.push_str("## TIRx CUDA/PTX Ops\n\n| Operation | Family | Fidelity | Notes |\n| --- | --- | --- | --- |\n");
    for e in ops {
        s.push_str(&format!("| `{}` | {} | {} | {} |\n", e.name, e.family, e.fidelity.name(), e.notes));
    }
    s.push_str("\n## CUDA Tile Primitives\n\n| Operation | Fidelity | Notes |\n| --- | --- | --- |\n");
    for e in tiles {
        s.push_str(&format!("| `{}` | {} | {} |\n", e.name, e.fidelity.name(), e.notes));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_roundtrip() {
        assert_eq!(i32::from_bits((-5i32).to_bits()), -5);
        assert_eq!((-1i8).to_bits(), 0xff);
        assert_eq!(<f32 as Scalar>::from_bits(Scalar::to_bits(1.5f32)), 1.5);
        assert_eq!(E2M1::from_bits(0xff), E2M1(0xf));
        let key = OpKey { name: "tirx.cuda.float_as_uint".into(), mods: vec![] };
        assert!(resolve_ptx(&key, &[Ty::U32], &[Ty::F32]).is_ok());
    }

    #[test]
    fn render_md() {
        let e = [
            OpEntry { name: "tirx.ptx.ld", family: "raw_memory", fidelity: Fidelity::Modeled, notes: "", instr: "ld" },
            OpEntry { name: "tirx.tile.add", family: "modeled", fidelity: Fidelity::Modeled, notes: "", instr: "tile" },
        ];
        let md = render_supported_ops_md(&e);
        assert!(md.contains("| `tirx.ptx.ld` | raw_memory | modeled |  |"));
        assert!(md.contains("| `tirx.tile.add` | modeled |  |"));
    }
}
