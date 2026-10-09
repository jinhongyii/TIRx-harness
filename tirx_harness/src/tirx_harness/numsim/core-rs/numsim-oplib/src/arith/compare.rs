//! Register comparison, selection and classification: `setp`, `set`
//! (scalar, packed-half and PTX 9.4 packed-integer), `selp`, `slct`, `testp`.
//!
//! Moved from legacy `engine-rs/src/runtime/instructions/reg_compare.rs` and
//! the `setp`/`SetPacked`/`selp` parts of `reg.rs`
//! (`compare_packed_words`, `signed_packed_lane`, `select_markers!`).

use super::half::{decode_bf16, decode_f16};
use super::BoolOp;
use crate::scalar;
use crate::types::{OpError, OpResult};

/// PTX comparison operator (`CmpOp`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Equ,
    Neu,
    Ltu,
    Leu,
    Gtu,
    Geu,
    Num,
    Nan,
}

/// One compared value, already decoded according to the PTX source type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CompareAtom {
    /// `.b16/.b32/.b64`: only `eq`/`ne` are defined.
    Bits(u64),
    /// `.s8/.s16/.s32/.s64`: ordered relations only.
    Signed(i64),
    /// `.u8/.u16/.u32/.u64`: ordered relations only.
    Unsigned(u64),
    /// Any floating source, widened exactly to f64.
    Float(f64),
}

fn invalid(op: CmpOp, what: &str) -> OpError {
    OpError::message(format!(
        "comparison {op:?} is not defined for {what} sources"
    ))
}

/// Evaluate one PTX comparison. Fails closed on an operator the source type
/// does not define, or on mixed atom kinds.
pub fn compare_atom(op: CmpOp, lhs: CompareAtom, rhs: CompareAtom) -> OpResult<bool> {
    Ok(match (lhs, rhs) {
        (CompareAtom::Bits(lhs), CompareAtom::Bits(rhs)) => match op {
            CmpOp::Eq => lhs == rhs,
            CmpOp::Ne => lhs != rhs,
            _ => return Err(invalid(op, "bit-size")),
        },
        (CompareAtom::Signed(lhs), CompareAtom::Signed(rhs)) => {
            ordered_integer(op, lhs.cmp(&rhs)).ok_or_else(|| invalid(op, "signed integer"))?
        }
        (CompareAtom::Unsigned(lhs), CompareAtom::Unsigned(rhs)) => {
            ordered_integer(op, lhs.cmp(&rhs)).ok_or_else(|| invalid(op, "unsigned integer"))?
        }
        (CompareAtom::Float(lhs), CompareAtom::Float(rhs)) => compare_float(op, lhs, rhs),
        _ => {
            return Err(OpError::message(
                "comparison sources have different PTX types",
            ))
        }
    })
}

fn ordered_integer(op: CmpOp, order: std::cmp::Ordering) -> Option<bool> {
    use std::cmp::Ordering::*;
    Some(match op {
        CmpOp::Eq => order == Equal,
        CmpOp::Ne => order != Equal,
        CmpOp::Lt => order == Less,
        CmpOp::Le => order != Greater,
        CmpOp::Gt => order == Greater,
        CmpOp::Ge => order != Less,
        _ => return None,
    })
}

/// Floating comparison: ordered relations are false on NaN, `u`-suffixed
/// relations are true on NaN.
pub fn compare_float(op: CmpOp, lhs: f64, rhs: f64) -> bool {
    let unordered = lhs.is_nan() || rhs.is_nan();
    match op {
        CmpOp::Eq => !unordered && lhs == rhs,
        CmpOp::Ne => !unordered && lhs != rhs,
        CmpOp::Lt => !unordered && lhs < rhs,
        CmpOp::Le => !unordered && lhs <= rhs,
        CmpOp::Gt => !unordered && lhs > rhs,
        CmpOp::Ge => !unordered && lhs >= rhs,
        CmpOp::Equ => unordered || lhs == rhs,
        CmpOp::Neu => unordered || lhs != rhs,
        CmpOp::Ltu => unordered || lhs < rhs,
        CmpOp::Leu => unordered || lhs <= rhs,
        CmpOp::Gtu => unordered || lhs > rhs,
        CmpOp::Geu => unordered || lhs >= rhs,
        CmpOp::Num => !unordered,
        CmpOp::Nan => unordered,
    }
}

/// `.f32` source atom; `.ftz` flushes subnormal inputs (sign-preserving).
pub fn f32_atom(value: f32, ftz: bool) -> CompareAtom {
    let value = if ftz {
        scalar::flush_subnormal_f32(value)
    } else {
        value
    };
    CompareAtom::Float(value.into())
}

/// `.f16` source atom; `.ftz` flushes subnormal inputs (sign-preserving).
pub fn f16_atom(bits: u16, ftz: bool) -> CompareAtom {
    let bits = if ftz {
        scalar::flush_subnormal_f16_bits(bits)
    } else {
        bits
    };
    CompareAtom::Float(f64::from(decode_f16(bits)))
}

/// `.bf16` source atom (no `.ftz` form).
pub fn bf16_atom(bits: u16) -> CompareAtom {
    CompareAtom::Float(f64::from(decode_bf16(bits)))
}

/// `setp.CmpOp{.ftz}.f32`.
pub fn setp_f32(op: CmpOp, lhs: f32, rhs: f32, ftz: bool) -> bool {
    let (lhs, rhs) = if ftz {
        (
            scalar::flush_subnormal_f32(lhs),
            scalar::flush_subnormal_f32(rhs),
        )
    } else {
        (lhs, rhs)
    };
    compare_float(op, lhs.into(), rhs.into())
}

/// `setp.CmpOp.f64`.
pub fn setp_f64(op: CmpOp, lhs: f64, rhs: f64) -> bool {
    compare_float(op, lhs, rhs)
}

/// `setp.CmpOp{.ftz}.f16`.
pub fn setp_f16(op: CmpOp, lhs: u16, rhs: u16, ftz: bool) -> bool {
    compare_atom(op, f16_atom(lhs, ftz), f16_atom(rhs, ftz))
        .expect("float atoms accept every CmpOp")
}

/// `setp.CmpOp.bf16`.
pub fn setp_bf16(op: CmpOp, lhs: u16, rhs: u16) -> bool {
    compare_atom(op, bf16_atom(lhs), bf16_atom(rhs)).expect("float atoms accept every CmpOp")
}

/// `setp.CmpOp.s{8,16,32,64}` (sources sign-extended to i64).
pub fn setp_signed(op: CmpOp, lhs: i64, rhs: i64) -> OpResult<bool> {
    compare_atom(op, CompareAtom::Signed(lhs), CompareAtom::Signed(rhs))
}

/// `setp.CmpOp.u{8,16,32,64}` (sources zero-extended to u64).
pub fn setp_unsigned(op: CmpOp, lhs: u64, rhs: u64) -> OpResult<bool> {
    compare_atom(op, CompareAtom::Unsigned(lhs), CompareAtom::Unsigned(rhs))
}

/// `setp.{eq,ne}.b{16,32,64}`.
pub fn setp_bits(op: CmpOp, lhs: u64, rhs: u64) -> OpResult<bool> {
    compare_atom(op, CompareAtom::Bits(lhs), CompareAtom::Bits(rhs))
}

/// Two-lane comparison mask of `.f16x2` sources: bit 0 = low half, bit 1 =
/// high half.
pub fn compare_mask_f16x2(op: CmpOp, lhs: u32, rhs: u32, ftz: bool) -> u8 {
    u8::from(setp_f16(op, lhs as u16, rhs as u16, ftz))
        | (u8::from(setp_f16(op, (lhs >> 16) as u16, (rhs >> 16) as u16, ftz)) << 1)
}

/// Two-lane comparison mask of `.bf16x2` sources.
pub fn compare_mask_bf16x2(op: CmpOp, lhs: u32, rhs: u32) -> u8 {
    u8::from(setp_bf16(op, lhs as u16, rhs as u16))
        | (u8::from(setp_bf16(op, (lhs >> 16) as u16, (rhs >> 16) as u16)) << 1)
}

/// Combine each mask bit with predicate `c` (`set/setp.CmpOp.BoolOp`).
pub fn combine_mask(bool_op: BoolOp, mask: u8, predicate: bool) -> u8 {
    u8::from(bool_op.apply(mask & 1 != 0, predicate))
        | (u8::from(bool_op.apply(mask & 2 != 0, predicate)) << 1)
}

/// The `p|q` pair of two-predicate `setp`: a scalar source (`lanes == 1`)
/// yields the comparison and its complement; a packed-half source
/// (`lanes == 2`) yields the low- and high-lane comparisons.
pub fn predicate_pair(mask: u8, lanes: usize) -> (bool, bool) {
    let first = mask & 1 != 0;
    (first, if lanes == 1 { !first } else { mask & 2 != 0 })
}

/// Destination type of a value-producing `set`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetDst {
    U32,
    S32,
    F32,
    U16,
    S16,
    F16,
    Bf16,
    F16x2,
    Bf16x2,
}

/// Encode a comparison mask as the `set` destination bit pattern (in the low
/// bits of the returned `u32`). `lanes` is 2 for packed-half sources.
///
/// For a `.u32/.s32` destination with packed-half sources each half is all
/// ones or zero; `.f32/.f16/.bf16` destinations produce 1.0 or 0.0.
pub fn set_encode(dst: SetDst, mask: u8, lanes: usize) -> u32 {
    let lane16 = |bit: u8, one: u16| u32::from(if mask & bit != 0 { one } else { 0 });
    match dst {
        SetDst::U32 | SetDst::S32 => {
            if lanes == 2 {
                lane16(1, u16::MAX) | (lane16(2, u16::MAX) << 16)
            } else if mask & 1 != 0 {
                u32::MAX
            } else {
                0
            }
        }
        SetDst::F32 => {
            if mask & 1 != 0 {
                1.0_f32.to_bits()
            } else {
                0.0_f32.to_bits()
            }
        }
        SetDst::U16 | SetDst::S16 => lane16(1, u16::MAX),
        SetDst::F16 => lane16(1, 0x3c00),
        SetDst::Bf16 => lane16(1, 0x3f80),
        SetDst::F16x2 => lane16(1, 0x3c00) | (lane16(2, 0x3c00) << 16),
        SetDst::Bf16x2 => lane16(1, 0x3f80) | (lane16(2, 0x3f80) << 16),
    }
}

/// Lane type of PTX 9.4 packed-integer `set.CmpOp.{u8,s8,u16,s16}x{4,2}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedIntLane {
    U8,
    S8,
    U16,
    S16,
}

#[inline(always)]
fn signed_packed_lane(value: u32, width: u32) -> i32 {
    let shift = 32 - width;
    ((value << shift) as i32) >> shift
}

/// PTX 9.4 packed-integer `set`: each true lane becomes all ones.
pub fn set_packed(op: CmpOp, lane: PackedIntLane, lhs: u32, rhs: u32) -> OpResult<u32> {
    let (width, signed) = match lane {
        PackedIntLane::U8 => (8, false),
        PackedIntLane::S8 => (8, true),
        PackedIntLane::U16 => (16, false),
        PackedIntLane::S16 => (16, true),
    };
    let lane_mask = (1_u32 << width) - 1;
    let atom = |bits: u32| {
        if signed {
            CompareAtom::Signed(i64::from(signed_packed_lane(bits, width)))
        } else {
            CompareAtom::Unsigned(u64::from(bits))
        }
    };
    let mut packed = 0_u32;
    for index in 0..(32 / width) {
        let shift = index * width;
        if compare_atom(
            op,
            atom((lhs >> shift) & lane_mask),
            atom((rhs >> shift) & lane_mask),
        )? {
            packed |= lane_mask << shift;
        }
    }
    Ok(packed)
}

/// `selp.type d, a, b, c`: `c ? a : b`.
pub fn selp<T>(predicate: bool, on_true: T, on_false: T) -> T {
    if predicate {
        on_true
    } else {
        on_false
    }
}

/// `slct.dtype.s32 d, a, b, c`: `c >= 0 ? a : b`.
pub fn slct_s32<T>(a: T, b: T, c: i32) -> T {
    if c >= 0 {
        a
    } else {
        b
    }
}

/// `slct{.ftz}.dtype.f32 d, a, b, c`: `c >= 0.0 ? a : b` (NaN selects `b`).
pub fn slct_f32<T>(a: T, b: T, c: f32, ftz: bool) -> T {
    let c = if ftz {
        scalar::flush_subnormal_f32(c)
    } else {
        c
    };
    if c >= 0.0 {
        a
    } else {
        b
    }
}

/// `testp` classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TestpClass {
    Finite,
    Infinite,
    Number,
    NotANumber,
    /// PTX `.normal` includes zero.
    Normal,
    Subnormal,
}

/// `testp.op.f32`.
pub fn testp_f32(class: TestpClass, value: f32) -> bool {
    match class {
        TestpClass::Finite => value.is_finite(),
        TestpClass::Infinite => value.is_infinite(),
        TestpClass::Number => !value.is_nan(),
        TestpClass::NotANumber => value.is_nan(),
        TestpClass::Normal => value == 0.0 || value.is_normal(),
        TestpClass::Subnormal => value.is_subnormal(),
    }
}

/// `testp.op.f64`.
pub fn testp_f64(class: TestpClass, value: f64) -> bool {
    match class {
        TestpClass::Finite => value.is_finite(),
        TestpClass::Infinite => value.is_infinite(),
        TestpClass::Number => !value.is_nan(),
        TestpClass::NotANumber => value.is_nan(),
        TestpClass::Normal => value == 0.0 || value.is_normal(),
        TestpClass::Subnormal => value.is_subnormal(),
    }
}
