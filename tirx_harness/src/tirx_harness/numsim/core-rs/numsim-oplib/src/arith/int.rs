//! Integer arithmetic: add/sub/mul/mad (lo/hi/wide/sat), mul24/mad24, sad,
//! dp2a/dp4a, packed 16x2 add/min/max, integer min/max/relu, div/rem, neg/abs.
//!
//! Moved from legacy `engine-rs/src/runtime/instructions/reg.rs`
//! (`integer_arithmetic_variants!`, `add_16x2`, `minmax_16x2`, `sad_variant!`,
//! `mul_wide_variant!`, `mad_wide_variant!`, `mul24_*`, `dp4a_bits`,
//! `dp2a_bits`, `integer_div_variant!`, `*integer_rem_variant!`, `neg`/`abs`).

use crate::types::{OpError, OpResult};

/// PTX integer register carrier (`.s16/.s32/.s64/.u16/.u32/.u64`).
pub trait PtxInteger: Copy + Ord + std::fmt::Display {
    fn wrapping_add(self, rhs: Self) -> Self;
    fn wrapping_sub(self, rhs: Self) -> Self;
    fn wrapping_mul(self, rhs: Self) -> Self;
    /// Upper half of the double-width product (`mul.hi`).
    fn mul_hi(self, rhs: Self) -> Self;
    fn checked_div(self, rhs: Self) -> Option<Self>;
    fn checked_rem(self, rhs: Self) -> Option<Self>;
    fn is_negative(self) -> bool;
    fn is_zero(self) -> bool;
}

/// Signed PTX integer carrier (`neg`/`abs`).
pub trait PtxSignedInteger: PtxInteger {
    fn wrapping_neg(self) -> Self;
    fn wrapping_abs(self) -> Self;
}

macro_rules! ptx_integer {
    ($scalar:ty, $wide:ty, $negative:expr) => {
        impl PtxInteger for $scalar {
            fn wrapping_add(self, rhs: Self) -> Self {
                <$scalar>::wrapping_add(self, rhs)
            }
            fn wrapping_sub(self, rhs: Self) -> Self {
                <$scalar>::wrapping_sub(self, rhs)
            }
            fn wrapping_mul(self, rhs: Self) -> Self {
                <$scalar>::wrapping_mul(self, rhs)
            }
            fn mul_hi(self, rhs: Self) -> Self {
                ((self as $wide * rhs as $wide) >> <$scalar>::BITS) as $scalar
            }
            fn checked_div(self, rhs: Self) -> Option<Self> {
                <$scalar>::checked_div(self, rhs)
            }
            fn checked_rem(self, rhs: Self) -> Option<Self> {
                <$scalar>::checked_rem(self, rhs)
            }
            fn is_negative(self) -> bool {
                let negative: fn($scalar) -> bool = $negative;
                negative(self)
            }
            fn is_zero(self) -> bool {
                self == 0
            }
        }
    };
}

ptx_integer!(i16, i32, |value| value < 0);
ptx_integer!(i32, i64, |value| value < 0);
ptx_integer!(i64, i128, |value| value < 0);
ptx_integer!(u16, u32, |_value| false);
ptx_integer!(u32, u64, |_value| false);
ptx_integer!(u64, u128, |_value| false);

macro_rules! ptx_signed_integer {
    ($($scalar:ty),+) => {
        $(
            impl PtxSignedInteger for $scalar {
                fn wrapping_neg(self) -> Self {
                    <$scalar>::wrapping_neg(self)
                }
                fn wrapping_abs(self) -> Self {
                    <$scalar>::wrapping_abs(self)
                }
            }
        )+
    };
}

ptx_signed_integer!(i16, i32, i64);

/// `add.{s,u}{16,32,64}` (wrapping).
pub fn add_int<T: PtxInteger>(lhs: T, rhs: T) -> T {
    lhs.wrapping_add(rhs)
}

/// `sub.{s,u}{16,32,64}` (wrapping).
pub fn sub_int<T: PtxInteger>(lhs: T, rhs: T) -> T {
    lhs.wrapping_sub(rhs)
}

/// `mul.lo.{s,u}{16,32,64}`.
pub fn mul_lo_int<T: PtxInteger>(lhs: T, rhs: T) -> T {
    lhs.wrapping_mul(rhs)
}

/// `mul.hi.{s,u}{16,32,64}`.
pub fn mul_hi_int<T: PtxInteger>(lhs: T, rhs: T) -> T {
    lhs.mul_hi(rhs)
}

/// `mad.lo.{s,u}{16,32,64}`.
pub fn mad_lo_int<T: PtxInteger>(a: T, b: T, c: T) -> T {
    a.wrapping_mul(b).wrapping_add(c)
}

/// `mad.hi.{s,u}{16,32,64}`.
pub fn mad_hi_int<T: PtxInteger>(a: T, b: T, c: T) -> T {
    a.mul_hi(b).wrapping_add(c)
}

/// `min.{s,u}{16,32,64}`.
pub fn min_int<T: PtxInteger>(lhs: T, rhs: T) -> T {
    lhs.min(rhs)
}

/// `max.{s,u}{16,32,64}`.
pub fn max_int<T: PtxInteger>(lhs: T, rhs: T) -> T {
    lhs.max(rhs)
}

/// `add.sat.s32`.
pub fn add_sat_s32(lhs: i32, rhs: i32) -> i32 {
    lhs.saturating_add(rhs)
}

/// `sub.sat.s32`.
pub fn sub_sat_s32(lhs: i32, rhs: i32) -> i32 {
    lhs.saturating_sub(rhs)
}

/// `mad.hi.sat.s32`.
pub fn mad_hi_sat_s32(a: i32, b: i32, c: i32) -> i32 {
    a.mul_hi(b).saturating_add(c)
}

/// `add.{s,u}16x2`: signed and unsigned packed addition have identical
/// wrapping bits.
pub fn add_16x2(lhs: u32, rhs: u32) -> u32 {
    u32::from((lhs as u16).wrapping_add(rhs as u16))
        | (u32::from(((lhs >> 16) as u16).wrapping_add((rhs >> 16) as u16)) << 16)
}

/// `min/max{.relu}.{s,u}16x2`, lane order low half first.
pub fn minmax_16x2(a: u32, b: u32, signed: bool, relu: bool, maximum: bool) -> u32 {
    let mut result = 0;
    for shift in [0, 16] {
        let decode = |bits: u32| {
            if signed {
                (bits >> shift) as i16 as i32
            } else {
                (bits >> shift) as u16 as i32
            }
        };
        let (a, b) = (decode(a), decode(b));
        let value = if maximum { a.max(b) } else { a.min(b) };
        let value = if relu { value.max(0) } else { value };
        result |= u32::from(value as u16) << shift;
    }
    result
}

/// `min.relu.s32`.
pub fn min_relu_s32(a: i32, b: i32) -> i32 {
    a.min(b).max(0)
}

/// `max.relu.s32`.
pub fn max_relu_s32(a: i32, b: i32) -> i32 {
    a.max(b).max(0)
}

/// `sad.{s,u}{16,32,64}`: `c + |a - b|` with wrapping arithmetic.
pub fn sad_int<T: PtxInteger>(a: T, b: T, c: T) -> T {
    let difference = if a < b {
        b.wrapping_sub(a)
    } else {
        a.wrapping_sub(b)
    };
    c.wrapping_add(difference)
}

/// `mul.wide.s16`.
pub fn mul_wide_s16(lhs: i16, rhs: i16) -> i32 {
    i32::from(lhs) * i32::from(rhs)
}

/// `mul.wide.u16`.
pub fn mul_wide_u16(lhs: u16, rhs: u16) -> u32 {
    u32::from(lhs) * u32::from(rhs)
}

/// `mul.wide.s32`.
pub fn mul_wide_s32(lhs: i32, rhs: i32) -> i64 {
    i64::from(lhs) * i64::from(rhs)
}

/// `mul.wide.u32`.
pub fn mul_wide_u32(lhs: u32, rhs: u32) -> u64 {
    u64::from(lhs) * u64::from(rhs)
}

/// `mad.wide.s16`.
pub fn mad_wide_s16(lhs: i16, rhs: i16, addend: i32) -> i32 {
    (i32::from(lhs) * i32::from(rhs)).wrapping_add(addend)
}

/// `mad.wide.u16`.
pub fn mad_wide_u16(lhs: u16, rhs: u16, addend: u32) -> u32 {
    (u32::from(lhs) * u32::from(rhs)).wrapping_add(addend)
}

/// `mad.wide.s32`.
pub fn mad_wide_s32(lhs: i32, rhs: i32, addend: i64) -> i64 {
    (i64::from(lhs) * i64::from(rhs)).wrapping_add(addend)
}

/// `mad.wide.u32`.
pub fn mad_wide_u32(lhs: u32, rhs: u32, addend: u64) -> u64 {
    (u64::from(lhs) * u64::from(rhs)).wrapping_add(addend)
}

fn signed_24(value: i32) -> i64 {
    i64::from(((value as u32 & 0x00ff_ffff) << 8) as i32 >> 8)
}

/// `mul24.{lo,hi}.s32` (`high` selects `.hi`: product bits 16..47).
pub fn mul24_s32(lhs: i32, rhs: i32, high: bool) -> i32 {
    let product = signed_24(lhs) * signed_24(rhs);
    if high {
        (product >> 16) as i32
    } else {
        product as i32
    }
}

/// `mul24.{lo,hi}.u32`.
pub fn mul24_u32(lhs: u32, rhs: u32, high: bool) -> u32 {
    let product = u64::from(lhs & 0x00ff_ffff) * u64::from(rhs & 0x00ff_ffff);
    if high {
        (product >> 16) as u32
    } else {
        product as u32
    }
}

/// `mad24.{lo,hi}.s32`.
pub fn mad24_s32(lhs: i32, rhs: i32, addend: i32, high: bool) -> i32 {
    mul24_s32(lhs, rhs, high).wrapping_add(addend)
}

/// `mad24.{lo,hi}.u32`.
pub fn mad24_u32(lhs: u32, rhs: u32, addend: u32, high: bool) -> u32 {
    mul24_u32(lhs, rhs, high).wrapping_add(addend)
}

/// `mad24.hi.sat.s32`.
pub fn mad24_hi_sat_s32(lhs: i32, rhs: i32, addend: i32) -> i32 {
    (i64::from(mul24_s32(lhs, rhs, true)) + i64::from(addend))
        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn packed_element(word: u32, index: usize, width: usize, signed: bool) -> i64 {
    let mask = (1_u32 << width) - 1;
    let raw = (word >> (index * width)) & mask;
    if signed && raw & (1_u32 << (width - 1)) != 0 {
        i64::from(raw) - (1_i64 << width)
    } else {
        i64::from(raw)
    }
}

/// `dp4a.atype.btype` on raw 32-bit carriers; the result carrier is `.u32`
/// when both inputs are unsigned and `.s32` (same bits) otherwise.
pub fn dp4a(a: u32, b: u32, c: u32, signed_a: bool, signed_b: bool) -> u32 {
    let accumulator = if signed_a || signed_b {
        i64::from(c as i32)
    } else {
        i64::from(c)
    };
    (0..4).fold(accumulator, |sum, index| {
        sum + packed_element(a, index, 8, signed_a) * packed_element(b, index, 8, signed_b)
    }) as u32
}

/// `dp2a.{lo,hi}.atype.btype` on raw 32-bit carriers.
pub fn dp2a(a: u32, b: u32, c: u32, signed_a: bool, signed_b: bool, high: bool) -> u32 {
    let accumulator = if signed_a || signed_b {
        i64::from(c as i32)
    } else {
        i64::from(c)
    };
    let first_byte = if high { 2 } else { 0 };
    (0..2).fold(accumulator, |sum, index| {
        sum + packed_element(a, index, 16, signed_a)
            * packed_element(b, first_byte + index, 8, signed_b)
    }) as u32
}

/// `div.{s,u}{16,32,64}`; division by zero and signed overflow fail closed.
pub fn div_int<T: PtxInteger>(lhs: T, rhs: T) -> OpResult<T> {
    lhs.checked_div(rhs).ok_or_else(|| {
        OpError::message(format!(
            "integer div has an undefined operand: {lhs} / {rhs}"
        ))
    })
}

/// `rem.{s,u}{16,32,64}`.
///
/// PTX 9.4 explicitly leaves negative remainder machine-specific: the
/// quotient may round either toward zero or toward negative infinity. Those
/// rules differ exactly for a non-integral, negative quotient, so a nonzero
/// remainder with operands of unlike sign fails closed.
pub fn rem_int<T: PtxInteger>(lhs: T, rhs: T) -> OpResult<T> {
    let remainder = lhs.checked_rem(rhs).ok_or_else(|| {
        OpError::message(format!(
            "integer rem has an undefined operand: {lhs} % {rhs}"
        ))
    })?;
    if !remainder.is_zero() && lhs.is_negative() != rhs.is_negative() {
        return Err(OpError::message(format!(
            "integer rem has a machine-specific negative operand: {lhs} % {rhs}"
        )));
    }
    Ok(remainder)
}

/// `neg.s{16,32,64}` (wrapping: `neg(MIN) == MIN`).
pub fn neg_int<T: PtxSignedInteger>(value: T) -> T {
    value.wrapping_neg()
}

/// `abs.s{16,32,64}` (wrapping: `abs(MIN) == MIN`).
pub fn abs_int<T: PtxSignedInteger>(value: T) -> T {
    value.wrapping_abs()
}
