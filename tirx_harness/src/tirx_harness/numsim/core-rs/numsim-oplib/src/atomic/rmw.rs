//! Atomic read-modify-write numerics: `old` + operand -> new stored value.
//!
//! Ported from legacy `engine-rs/src/runtime/io.rs` (`RawAtomicOperation`,
//! `RawAtomicScalar::apply_atomic` for i32/i64/u32/u64/U64x2/f32/f64, and the
//! update closures of `raw_atomic_add_{fp16,bf16,fp16x2,bf16x2}_physical_ptr_warp`,
//! `raw_atomic_half_vector_physical_ptr_warp`,
//! `raw_atomic_add_f32x{2,4}_physical_ptr_warp`, `raw_atomic_cas_physical_ptr_warp`).
//! The memory plumbing (alignment checks, ordering, per-lane byte access) stays
//! in the engine; these functions are what each lane's atomic computes.  Every
//! atomic returns `old` to the caller (`atom`) or discards it (`red`); both
//! share these update rules.

use crate::cvt::formats::{bf16_bits_to_f32, f32_to_bf16_bits, f32_to_fp16_bits, fp16_bits_to_f32};
use crate::scalar::{add_f32, add_f32_ftz, ptx_max_f32, ptx_min_f32, F32RoundingMode, U64x2};
use crate::types::{OpError, OpResult};

/// PTX/CUDA atomic operation (legacy `RawAtomicOperation`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomicOp {
    Add,
    /// `atom.add.noftz.f32` (PTX 9.4): round-to-nearest without flushing.
    AddNoFtz,
    BitAnd,
    BitOr,
    BitXor,
    Exchange,
    /// `.inc`: wraps to zero once `old >= operand`.
    Increment,
    /// `.dec`: reloads `operand` when `old == 0 || old > operand`.
    Decrement,
    Minimum,
    Maximum,
}

/// PTX state space of the atomic's target (only `.f32` add depends on it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomicSpace {
    Global,
    Shared,
}

fn unsupported<T>(operation: AtomicOp) -> OpResult<T> {
    Err(OpError::message(format!(
        "atomic operation {operation:?} is invalid for this scalar type"
    )))
}

/// `atom/red.{add,min,max}.s32`.
pub fn atomic_i32(operation: AtomicOp, old: i32, operand: i32) -> OpResult<i32> {
    match operation {
        AtomicOp::Add => Ok(old.wrapping_add(operand)),
        AtomicOp::Minimum => Ok(old.min(operand)),
        AtomicOp::Maximum => Ok(old.max(operand)),
        _ => unsupported(operation),
    }
}

/// `atom/red.{min,max}.s64`.
pub fn atomic_i64(operation: AtomicOp, old: i64, operand: i64) -> OpResult<i64> {
    match operation {
        AtomicOp::Minimum => Ok(old.min(operand)),
        AtomicOp::Maximum => Ok(old.max(operand)),
        _ => unsupported(operation),
    }
}

/// `atom/red.{add,and,or,xor,exch,inc,dec,min,max}.{u32,b32}`.
pub fn atomic_u32(operation: AtomicOp, old: u32, operand: u32) -> OpResult<u32> {
    match operation {
        AtomicOp::Add => Ok(old.wrapping_add(operand)),
        AtomicOp::BitAnd => Ok(old & operand),
        AtomicOp::BitOr => Ok(old | operand),
        AtomicOp::BitXor => Ok(old ^ operand),
        AtomicOp::Exchange => Ok(operand),
        AtomicOp::Increment => Ok(if old >= operand { 0 } else { old + 1 }),
        AtomicOp::Decrement => Ok(if old == 0 || old > operand {
            operand
        } else {
            old - 1
        }),
        AtomicOp::Minimum => Ok(old.min(operand)),
        AtomicOp::Maximum => Ok(old.max(operand)),
        AtomicOp::AddNoFtz => unsupported(operation),
    }
}

/// `atom/red.{add,and,or,xor,exch,min,max}.{u64,b64}`.
pub fn atomic_u64(operation: AtomicOp, old: u64, operand: u64) -> OpResult<u64> {
    match operation {
        AtomicOp::Add => Ok(old.wrapping_add(operand)),
        AtomicOp::BitAnd => Ok(old & operand),
        AtomicOp::BitOr => Ok(old | operand),
        AtomicOp::BitXor => Ok(old ^ operand),
        AtomicOp::Exchange => Ok(operand),
        AtomicOp::Minimum => Ok(old.min(operand)),
        AtomicOp::Maximum => Ok(old.max(operand)),
        _ => unsupported(operation),
    }
}

/// `atom.exch.b128`.
pub fn atomic_u64x2(operation: AtomicOp, _old: U64x2, operand: U64x2) -> OpResult<U64x2> {
    match operation {
        AtomicOp::Exchange => Ok(operand),
        _ => unsupported(operation),
    }
}

/// `atom/red.add{.noftz}.f32` and CUDA `atomicAdd(float*)`.
///
/// A global `.add` flushes subnormal inputs and result (measured); `.noftz`
/// rounds to nearest without flushing; a shared `.add` is a plain IEEE add.
pub fn atomic_f32(
    operation: AtomicOp,
    old: f32,
    operand: f32,
    space: AtomicSpace,
) -> OpResult<f32> {
    match operation {
        AtomicOp::Add if space == AtomicSpace::Global => {
            Ok(add_f32_ftz(old, operand, F32RoundingMode::Nearest))
        }
        AtomicOp::AddNoFtz => Ok(add_f32(old, operand, F32RoundingMode::Nearest)),
        AtomicOp::Add => Ok(old + operand),
        _ => unsupported(operation),
    }
}

/// `atom/red.add.f64` and CUDA `atomicAdd(double*)`.
pub fn atomic_f64(operation: AtomicOp, old: f64, operand: f64) -> OpResult<f64> {
    match operation {
        AtomicOp::Add => Ok(old + operand),
        _ => unsupported(operation),
    }
}

/// CUDA `atomicAdd(__half*)` / `atom.add.noftz.f16` on raw payloads.
pub fn atomic_add_f16(old: u16, operand: u16) -> u16 {
    f32_to_fp16_bits(fp16_bits_to_f32(old) + fp16_bits_to_f32(operand))
}

/// CUDA `atomicAdd(__nv_bfloat16*)` / `atom.add.noftz.bf16` on raw payloads.
pub fn atomic_add_bf16(old: u16, operand: u16) -> u16 {
    f32_to_bf16_bits(bf16_bits_to_f32(old) + bf16_bits_to_f32(operand))
}

/// CUDA `atomicAdd(__half2*)`: each 16-bit component is added independently
/// (element 0 in the low half).
pub fn atomic_add_f16x2(old: u32, operand: u32) -> u32 {
    let lane = |shift: u32| {
        u32::from(atomic_add_f16(
            (old >> shift) as u16,
            (operand >> shift) as u16,
        ))
    };
    lane(0) | (lane(16) << 16)
}

/// CUDA `atomicAdd(__nv_bfloat162*)`.
pub fn atomic_add_bf16x2(old: u32, operand: u32) -> u32 {
    let lane = |shift: u32| {
        u32::from(atomic_add_bf16(
            (old >> shift) as u16,
            (operand >> shift) as u16,
        ))
    };
    lane(0) | (lane(16) << 16)
}

/// One element of `atom/red.{add,min,max}.noftz.{f16,bf16}{x2}{.vN}`.
///
/// `min`/`max` follow `ptx_{min,max}_f32` without `.ftz`/`.NaN` propagation.
pub fn atomic_half(operation: AtomicOp, old: u16, operand: u16, bf16: bool) -> OpResult<u16> {
    type Codec = (fn(u16) -> f32, fn(f32) -> u16);
    let (decode, encode): Codec = if bf16 {
        (bf16_bits_to_f32, f32_to_bf16_bits)
    } else {
        (fp16_bits_to_f32, f32_to_fp16_bits)
    };
    let (old, operand) = (decode(old), decode(operand));
    let result = match operation {
        AtomicOp::Add => old + operand,
        AtomicOp::Minimum => ptx_min_f32(old, operand, false, false),
        AtomicOp::Maximum => ptx_max_f32(old, operand, false, false),
        _ => return unsupported(operation),
    };
    Ok(encode(result))
}

/// Half-vector atomics (2/4/8 elements): each element updates independently.
pub fn atomic_half_vector<const N: usize>(
    operation: AtomicOp,
    old: [u16; N],
    operand: [u16; N],
    bf16: bool,
) -> OpResult<[u16; N]> {
    if !matches!(N, 2 | 4 | 8) {
        return Err(OpError::message(
            "half-vector atomics require global memory and 2/4/8 half elements",
        ));
    }
    let mut result = [0_u16; N];
    for index in 0..N {
        result[index] = atomic_half(operation, old[index], operand[index], bf16)?;
    }
    Ok(result)
}

/// One `.f32` component of a global vector add (`noftz` selects `.noftz`).
fn vector_f32_component(old: f32, operand: f32, noftz: bool) -> f32 {
    let operation = if noftz {
        AtomicOp::AddNoFtz
    } else {
        AtomicOp::Add
    };
    atomic_f32(operation, old, operand, AtomicSpace::Global).expect("add is valid for f32")
}

/// CUDA `atomicAdd(float2*)` / `atom.add{.noftz}.v2.f32` (global only);
/// component 0 is the low 32 bits.
pub fn atomic_add_f32x2(old: u64, operand: u64, noftz: bool) -> u64 {
    let lane = |shift: u32| {
        u64::from(
            vector_f32_component(
                f32::from_bits((old >> shift) as u32),
                f32::from_bits((operand >> shift) as u32),
                noftz,
            )
            .to_bits(),
        ) << shift
    };
    lane(0) | lane(32)
}

/// CUDA `atomicAdd(float4*)` / `atom.add{.noftz}.v4.f32` (global only).
pub fn atomic_add_f32x4(old: [f32; 4], operand: [f32; 4], noftz: bool) -> [f32; 4] {
    std::array::from_fn(|index| vector_f32_component(old[index], operand[index], noftz))
}

/// `atom.cas.b{16,32,64,128}` / CUDA `atomicCAS`: whole-value bitwise
/// compare; returns the value stored afterwards (the caller returns `old`).
pub fn atomic_cas_bytes(old: &[u8], compare: &[u8], value: &[u8]) -> Vec<u8> {
    if old == compare {
        value.to_vec()
    } else {
        old.to_vec()
    }
}

/// [`atomic_cas_bytes`] for payloads up to 64 bits.
pub fn atomic_cas_u64(old: u64, compare: u64, value: u64) -> u64 {
    if old == compare {
        value
    } else {
        old
    }
}
