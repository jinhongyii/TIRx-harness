//! Directed-rounding f32/f64 arithmetic, division, sqrt, FMA, saturation.
#![allow(unused_imports)]
use crate::cvt::formats::*;
use crate::cvt::*;
use crate::scalar::*;
use crate::types::OpError;
use std::cmp::Ordering;
pub(crate) fn compare_f32_to_exact_fma(rounded: f32, exact: ExactFmaValue) -> Ordering {
    debug_assert!(!rounded.is_nan());
    if rounded == f32::INFINITY {
        return Ordering::Greater;
    }
    if rounded == f32::NEG_INFINITY {
        return Ordering::Less;
    }
    ExactFmaValue::addend(rounded).compare(exact)
}

pub(crate) fn round_exact_zero_sum_f32(
    rounded: f32,
    lhs: f32,
    rhs: f32,
    mode: F32RoundingMode,
) -> f32 {
    debug_assert_eq!(rounded, 0.0);
    let both_positive_zero = lhs.to_bits() == 0 && rhs.to_bits() == 0;
    if mode == F32RoundingMode::Down && !both_positive_zero {
        -0.0
    } else {
        rounded
    }
}

pub(crate) fn round_f32_from_exact_fma(
    rounded: f32,
    exact: ExactFmaValue,
    lhs: f32,
    rhs: f32,
    mode: F32RoundingMode,
) -> f32 {
    if exact.is_zero() {
        return round_exact_zero_sum_f32(rounded, lhs, rhs, mode);
    }
    let comparison = compare_f32_to_exact_fma(rounded, exact);
    round_f32_candidate(rounded, comparison, exact.negative, mode)
}

pub(crate) fn compare_f32_to_exact(rounded: f32, exact: ExactF32Value) -> Ordering {
    debug_assert!(!rounded.is_nan());
    if rounded == f32::INFINITY {
        return Ordering::Greater;
    }
    if rounded == f32::NEG_INFINITY {
        return Ordering::Less;
    }
    if exact.is_zero() {
        return if rounded == 0.0 {
            Ordering::Equal
        } else if rounded.is_sign_negative() {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    if rounded == 0.0 {
        return if exact.negative {
            Ordering::Greater
        } else {
            Ordering::Less
        };
    }
    let rounded = ExactF32Value::from_f32(rounded);
    if rounded.negative != exact.negative {
        return if rounded.negative {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let magnitude_order = rounded.magnitude.compare(exact.magnitude);
    if rounded.negative {
        magnitude_order.reverse()
    } else {
        magnitude_order
    }
}

pub(crate) fn round_f32_from_exact_sum(
    rounded: f32,
    exact: ExactF32Value,
    lhs: f32,
    rhs: f32,
    mode: F32RoundingMode,
) -> f32 {
    if exact.is_zero() {
        return round_exact_zero_sum_f32(rounded, lhs, rhs, mode);
    }
    let comparison = compare_f32_to_exact(rounded, exact);
    round_f32_candidate(rounded, comparison, exact.negative, mode)
}

pub(crate) fn round_f32_candidate(
    rounded: f32,
    comparison: Ordering,
    negative: bool,
    mode: F32RoundingMode,
) -> f32 {
    match rounding_adjustment(comparison, negative, mode) {
        Ordering::Less => next_f32(rounded, false),
        Ordering::Greater => next_f32(rounded, true),
        Ordering::Equal => rounded,
    }
}

/// PTX `add{.rnd}.f32` without `.ftz`: binary32 sum rounded by `mode` (RN/RZ/RM/RP),
/// corrected from the exact sum. Subnormals kept; NaN result pinned by
/// [`pin_nan2_f32`] (first NaN operand quieted, else `0xffc0_0000`).
pub fn add_f32(lhs: f32, rhs: f32, mode: F32RoundingMode) -> f32 {
    let rounded = pin_nan2_f32(lhs, rhs, lhs + rhs);
    if mode == F32RoundingMode::Nearest || !lhs.is_finite() || !rhs.is_finite() {
        return rounded;
    }
    let exact = exact_f32_sum(ExactF32Value::from_f32(lhs), ExactF32Value::from_f32(rhs));
    round_f32_from_exact_sum(rounded, exact, lhs, rhs, mode)
}

/// PTX `sub{.rnd}.f32` without `.ftz`: binary32 `lhs - rhs` rounded by `mode`
/// (RN/RZ/RM/RP). Subnormals kept; NaN result pinned by [`pin_nan2_f32`].
pub fn sub_f32(lhs: f32, rhs: f32, mode: F32RoundingMode) -> f32 {
    let rounded = pin_nan2_f32(lhs, rhs, lhs - rhs);
    if mode == F32RoundingMode::Nearest || !lhs.is_finite() || !rhs.is_finite() {
        return rounded;
    }
    let rhs = -rhs;
    let exact = exact_f32_sum(ExactF32Value::from_f32(lhs), ExactF32Value::from_f32(rhs));
    round_f32_from_exact_sum(rounded, exact, lhs, rhs, mode)
}

/// PTX `mul{.rnd}.f32` without `.ftz`: binary32 product rounded by `mode` from the
/// exact binary64 product. Subnormals kept; NaN result pinned by [`pin_nan2_f32`].
pub fn mul_f32(lhs: f32, rhs: f32, mode: F32RoundingMode) -> f32 {
    round_f32_from_exact(
        pin_nan2_f32(lhs, rhs, lhs * rhs),
        (lhs as f64) * (rhs as f64),
        mode,
    )
}

/// [`add_f32`] with `.ftz`: subnormal inputs and results flush to sign-preserving zero.
pub fn add_f32_ftz(lhs: f32, rhs: f32, mode: F32RoundingMode) -> f32 {
    flush_subnormal_f32(add_f32(
        flush_subnormal_f32(lhs),
        flush_subnormal_f32(rhs),
        mode,
    ))
}

/// [`sub_f32`] with `.ftz`: subnormal inputs and results flush to sign-preserving zero.
pub fn sub_f32_ftz(lhs: f32, rhs: f32, mode: F32RoundingMode) -> f32 {
    flush_subnormal_f32(sub_f32(
        flush_subnormal_f32(lhs),
        flush_subnormal_f32(rhs),
        mode,
    ))
}

/// [`mul_f32`] with `.ftz`: subnormal inputs flush to zero; a result that is tiny
/// before rounding (exact |product| < `f32::MIN_POSITIVE`) flushes to signed zero.
pub fn mul_f32_ftz(lhs: f32, rhs: f32, mode: F32RoundingMode) -> f32 {
    let lhs = flush_subnormal_f32(lhs);
    let rhs = flush_subnormal_f32(rhs);
    flush_f32_result(mul_f32(lhs, rhs, mode), || {
        (f64::from(lhs) * f64::from(rhs)).abs() < f64::from(f32::MIN_POSITIVE)
    })
}

/// At the smallest normal result, use the instruction's tininess rule;
/// ordinary subnormal outputs are identified by their rounded representation.
pub(crate) fn flush_f32_result(rounded: f32, is_tiny: impl FnOnce() -> bool) -> f32 {
    if rounded.abs() == f32::MIN_POSITIVE && is_tiny() {
        0.0_f32.copysign(rounded)
    } else {
        flush_subnormal_f32(rounded)
    }
}

/// Binary32 `lhs / rhs`, round-to-nearest-even, no FTZ. NaN payload is the host
/// division's (x86 `divss`: first NaN operand quieted, else `0xffc0_0000`); not pinned.
pub fn div_f32_rn(lhs: f32, rhs: f32) -> f32 {
    lhs / rhs
}

/// PTX `div.{rnd}{.ftz}.f32` (IEEE division): RN from the host quotient, RZ/RM/RP by
/// exact comparison of the candidate. `ftz` flushes subnormal inputs and tiny results
/// to signed zero. NaN payload as in [`div_f32_rn`].
pub fn ptx_div_f32(lhs: f32, rhs: f32, mode: F32RoundingMode, ftz: bool) -> f32 {
    let lhs = if ftz { flush_subnormal_f32(lhs) } else { lhs };
    let rhs = if ftz { flush_subnormal_f32(rhs) } else { rhs };
    let nearest = div_f32_rn(lhs, rhs);
    let rounded = if mode == F32RoundingMode::Nearest
        || !lhs.is_finite()
        || !rhs.is_finite()
        || lhs == 0.0
        || rhs == 0.0
    {
        nearest
    } else {
        let comparison =
            compare_division_candidate(f64::from(nearest), f64::from(lhs), f64::from(rhs));
        round_f32_candidate(
            nearest,
            comparison,
            lhs.is_sign_negative() != rhs.is_sign_negative(),
            mode,
        )
    };
    if ftz {
        flush_f32_result(rounded, || {
            compare_division_candidate(
                f64::from(f32::MIN_POSITIVE),
                f64::from(lhs.abs()),
                f64::from(rhs.abs()),
            ) == Ordering::Greater
        })
    } else {
        rounded
    }
}

/// PTX full-range and limited-range division approximations. In the normal
/// bounded domain the nearest quotient is a stable in-bound representative.
/// The limited-range instruction additionally has prescribed large-divisor
/// zeros/NaNs. It does not require a rounded FP32 reciprocal intermediate.
pub fn ptx_div_approx_f32(lhs: f32, rhs: f32, ftz: bool, full: bool) -> f32 {
    let lhs = if ftz { flush_subnormal_f32(lhs) } else { lhs };
    let rhs = if ftz { flush_subnormal_f32(rhs) } else { rhs };
    let result = if !full && rhs.is_finite() && rhs.abs() > f32::from_bits(0x7e80_0000) {
        lhs * 0.0_f32.copysign(rhs)
    } else {
        div_f32_rn(lhs, rhs)
    };
    if ftz {
        flush_f32_result(result, || {
            compare_division_candidate(
                f64::from(f32::MIN_POSITIVE),
                f64::from(lhs.abs()),
                f64::from(rhs.abs()),
            ) == Ordering::Greater
        })
    } else {
        result
    }
}

/// PTX 1.11.20 gross reciprocal: ignore both low words, honor canonical NaN
/// and FTZ, and choose nearest-even at the reduced precision for finite values.
pub fn ptx_rcp_approx_ftz_f64(value: f64) -> f64 {
    gross_f64_approx(value, |value| 1.0 / value)
}

/// PTX `rsqrt.approx.ftz.f64`: same contract as [`ptx_rcp_approx_ftz_f64`] (low input
/// word ignored, NaN -> `0x7fff_ffff_0000_0000`, FTZ, RN to the high word) for `1/sqrt(x)`.
pub fn ptx_rsqrt_approx_ftz_f64(value: f64) -> f64 {
    gross_f64_approx(value, |value| 1.0 / value.sqrt())
}

/// Shared input/output contract of the two PTX 1.11.20 approximations.
pub(crate) fn gross_f64_approx(value: f64, operation: impl FnOnce(f64) -> f64) -> f64 {
    let value = f64::from_bits(value.to_bits() & 0xffff_ffff_0000_0000);
    if value.is_nan() {
        return f64::from_bits(0x7fff_ffff_0000_0000);
    }
    let value = if value.is_subnormal() {
        0.0_f64.copysign(value)
    } else {
        value
    };
    let reciprocal = operation(value);
    if reciprocal.is_nan() {
        return f64::from_bits(0x7fff_ffff_0000_0000);
    }
    let bits = reciprocal.to_bits();
    let upper = bits >> 32;
    let discarded = bits & 0xffff_ffff;
    let increment = discarded > 0x8000_0000 || (discarded == 0x8000_0000 && upper & 1 != 0);
    let rounded = f64::from_bits((upper + u64::from(increment)) << 32);
    if rounded.is_subnormal() {
        0.0_f64.copysign(rounded)
    } else {
        rounded
    }
}

/// PTX `neg.ftz.f32`: sign-bit flip after flushing a subnormal input to signed zero.
/// NaN payload kept (sign flipped), no quieting.
pub fn ptx_neg_ftz_f32(value: f32) -> f32 {
    let value = flush_subnormal_f32(value);
    flush_subnormal_f32(f32::from_bits(value.to_bits() ^ 0x8000_0000))
}

/// PTX `neg{.ftz}.f16` on raw binary16 bits: sign-bit flip; `ftz` flushes a subnormal
/// input to signed zero first. NaN payload kept, no quieting.
pub fn ptx_neg_f16_bits(value: u16, ftz: bool) -> u16 {
    let value = if ftz {
        flush_subnormal_f16_bits(value)
    } else {
        value
    };
    value ^ 0x8000
}

/// PTX `neg{.ftz}.f16x2`: [`ptx_neg_f16_bits`] on each binary16 half (low half = bits 0..16).
pub fn ptx_neg_f16x2_bits(value: u32, ftz: bool) -> u32 {
    u32::from(ptx_neg_f16_bits(value as u16, ftz))
        | (u32::from(ptx_neg_f16_bits((value >> 16) as u16, ftz)) << 16)
}

/// PTX `sqrt.{rnd}{.ftz}.f32`: RN from the host root, RZ/RM/RP by exact square
/// comparison. `ftz` flushes subnormal input/result to signed zero. `sqrt(-0) = -0`;
/// negative or NaN input gives the host NaN (NaN operand quieted, else `0xffc0_0000`).
pub fn ptx_sqrt_f32(value: f32, mode: F32RoundingMode, ftz: bool) -> f32 {
    let value = if ftz {
        flush_subnormal_f32(value)
    } else {
        value
    };
    let nearest = value.sqrt();
    let rounded = if mode == F32RoundingMode::Nearest
        || !value.is_finite()
        || value <= 0.0
        || nearest.is_nan()
    {
        nearest
    } else {
        // A binary32 square has at most 48 significant bits, so binary64
        // compares the rounded candidate's square with the input exactly.
        let comparison = ((nearest as f64) * (nearest as f64)).total_cmp(&(value as f64));
        match mode {
            F32RoundingMode::Down | F32RoundingMode::Zero if comparison == Ordering::Greater => {
                next_f32(nearest, false)
            }
            F32RoundingMode::Up if comparison == Ordering::Less => next_f32(nearest, true),
            F32RoundingMode::Nearest
            | F32RoundingMode::Down
            | F32RoundingMode::Up
            | F32RoundingMode::Zero => nearest,
        }
    };
    if ftz {
        flush_subnormal_f32(rounded)
    } else {
        rounded
    }
}

pub(crate) fn positive_f64_dyadic(value: f64) -> (u128, i32) {
    debug_assert!(value.is_finite() && value > 0.0);
    let bits = value.to_bits();
    let exponent_field = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    let (significand, exponent) = if exponent_field == 0 {
        (fraction, -1074)
    } else {
        ((1_u64 << 52) | fraction, exponent_field - 1023 - 52)
    };
    let trailing = significand.trailing_zeros();
    (
        u128::from(significand >> trailing),
        exponent + trailing as i32,
    )
}

pub(crate) fn compare_positive_dyadics(
    lhs_significand: u128,
    lhs_exponent: i32,
    rhs_significand: u128,
    rhs_exponent: i32,
) -> Ordering {
    let lhs_bits = (u128::BITS - lhs_significand.leading_zeros()) as i32;
    let rhs_bits = (u128::BITS - rhs_significand.leading_zeros()) as i32;
    match (lhs_bits + lhs_exponent).cmp(&(rhs_bits + rhs_exponent)) {
        Ordering::Equal => {}
        order => return order,
    }
    if lhs_exponent >= rhs_exponent {
        let shift = u32::try_from(lhs_exponent - rhs_exponent).unwrap();
        lhs_significand
            .checked_shl(shift)
            .expect("equal-magnitude dyadics fit after alignment")
            .cmp(&rhs_significand)
    } else {
        let shift = u32::try_from(rhs_exponent - lhs_exponent).unwrap();
        lhs_significand.cmp(
            &rhs_significand
                .checked_shl(shift)
                .expect("equal-magnitude dyadics fit after alignment"),
        )
    }
}

pub(crate) fn compare_f64_square_to_input(root: f64, input: f64) -> Ordering {
    let (root_significand, root_exponent) = positive_f64_dyadic(root);
    let square = root_significand * root_significand;
    let trailing = square.trailing_zeros();
    let square = square >> trailing;
    let square_exponent = 2 * root_exponent + trailing as i32;
    let (input_significand, input_exponent) = positive_f64_dyadic(input);
    compare_positive_dyadics(square, square_exponent, input_significand, input_exponent)
}

/// PTX `sqrt.{rnd}.f64`: RN from the host root, RZ/RM/RP by exact dyadic square
/// comparison. Subnormals kept; `sqrt(-0) = -0`; negative/NaN input gives the host NaN.
pub fn ptx_sqrt_f64(value: f64, mode: F32RoundingMode) -> f64 {
    let nearest = value.sqrt();
    if mode == F32RoundingMode::Nearest || !value.is_finite() || value <= 0.0 || nearest.is_nan() {
        return nearest;
    }
    match (mode, compare_f64_square_to_input(nearest, value)) {
        (F32RoundingMode::Down | F32RoundingMode::Zero, Ordering::Greater) => nearest.next_down(),
        (F32RoundingMode::Up, Ordering::Less) => nearest.next_up(),
        (
            F32RoundingMode::Nearest
            | F32RoundingMode::Down
            | F32RoundingMode::Up
            | F32RoundingMode::Zero,
            _,
        ) => nearest,
    }
}

/// Binary32 fused multiply-add, RN, no FTZ; NaN pinned as in [`host_fma_f32`].
pub fn fma_f32_rn(lhs: f32, rhs: f32, addend: f32) -> f32 {
    host_fma_f32(lhs, rhs, addend)
}

/// Pinned NaN of a host binary32 `+ - * /` whose result is NaN: the first
/// NaN among `(lhs, rhs)`, quieted, else the x86 default NaN `0xffc0_0000`
/// (unoptimized x86 `addss lhs, rhs`; optimized builds may commute the
/// operands, so the payload is pinned instead of left to codegen).
#[inline]
pub fn pin_nan2_f32(lhs: f32, rhs: f32, result: f32) -> f32 {
    if !result.is_nan() {
        return result;
    }
    for value in [lhs, rhs] {
        if value.is_nan() {
            return f32::from_bits(value.to_bits() | 0x0040_0000);
        }
    }
    f32::from_bits(0xffc0_0000)
}

/// Binary64 [`pin_nan2_f32`] (default NaN `0xfff8_0000_0000_0000`).
#[inline]
pub fn pin_nan2_f64(lhs: f64, rhs: f64, result: f64) -> f64 {
    if !result.is_nan() {
        return result;
    }
    for value in [lhs, rhs] {
        if value.is_nan() {
            return f64::from_bits(value.to_bits() | 0x0008_0000_0000_0000);
        }
    }
    f64::from_bits(0xfff8_0000_0000_0000)
}

/// Host binary32 FMA with a pinned NaN result. A non-NaN result is the
/// exactly rounded fused value, which every FMA implementation agrees on.
/// The NaN payload, though, depends on which operand order the code
/// generator picks for `vfmadd` (LLVM may commute operands in optimized
/// builds), so it is pinned here to what legacy's host `fmaf` (glibc's x86
/// FMA path) returns: the first NaN among `(rhs, lhs, addend)`, quieted;
/// otherwise (an invalid `inf * 0` or `inf - inf`) the x86 default NaN
/// `0xffc0_0000`.
#[inline]
pub fn host_fma_f32(lhs: f32, rhs: f32, addend: f32) -> f32 {
    let value = lhs.mul_add(rhs, addend);
    if value.is_nan() {
        fma_nan_f32(lhs, rhs, addend)
    } else {
        value
    }
}

/// The pinned NaN of [`host_fma_f32`] (call only when the result is NaN).
#[inline]
pub fn fma_nan_f32(lhs: f32, rhs: f32, addend: f32) -> f32 {
    for value in [rhs, lhs, addend] {
        if value.is_nan() {
            return f32::from_bits(value.to_bits() | 0x0040_0000);
        }
    }
    f32::from_bits(0xffc0_0000)
}

/// Binary64 [`host_fma_f32`]: first NaN among `(rhs, lhs, addend)`,
/// quieted, else `0xfff8_0000_0000_0000`.
#[inline]
pub fn host_fma_f64(lhs: f64, rhs: f64, addend: f64) -> f64 {
    let value = lhs.mul_add(rhs, addend);
    if value.is_nan() {
        fma_nan_f64(lhs, rhs, addend)
    } else {
        value
    }
}

/// The pinned NaN of [`host_fma_f64`] (call only when the result is NaN).
#[inline]
pub fn fma_nan_f64(lhs: f64, rhs: f64, addend: f64) -> f64 {
    for value in [rhs, lhs, addend] {
        if value.is_nan() {
            return f64::from_bits(value.to_bits() | 0x0008_0000_0000_0000);
        }
    }
    f64::from_bits(0xfff8_0000_0000_0000)
}

/// Correct a nearest binary64 FMA by comparison with its exact dyadic result.
/// No thread-local rounding environment is changed, including on overflow.
pub fn fma_f64(lhs: f64, rhs: f64, addend: f64, mode: F32RoundingMode) -> f64 {
    let rounded = host_fma_f64(lhs, rhs, addend);
    if mode == F32RoundingMode::Nearest
        || !lhs.is_finite()
        || !rhs.is_finite()
        || !addend.is_finite()
    {
        return rounded;
    }
    let exact = exact_fma_sum(
        ExactFmaValue::product_f64(lhs, rhs),
        ExactFmaValue::addend_f64(addend),
    );
    if exact.is_zero() {
        let positive_zero_product =
            (lhs == 0.0 || rhs == 0.0) && lhs.is_sign_negative() == rhs.is_sign_negative();
        return if mode == F32RoundingMode::Down && !(positive_zero_product && addend.to_bits() == 0)
        {
            -0.0
        } else {
            rounded
        };
    }
    let comparison = if rounded == f64::INFINITY {
        Ordering::Greater
    } else if rounded == f64::NEG_INFINITY {
        Ordering::Less
    } else {
        ExactFmaValue::addend_f64(rounded).compare(exact)
    };
    round_f64_candidate(rounded, comparison, exact.negative, mode)
}

pub(crate) fn round_f64_candidate(
    rounded: f64,
    comparison: Ordering,
    negative: bool,
    mode: F32RoundingMode,
) -> f64 {
    match rounding_adjustment(comparison, negative, mode) {
        Ordering::Less => rounded.next_down(),
        Ordering::Greater => rounded.next_up(),
        Ordering::Equal => rounded,
    }
}

pub(crate) fn rounding_adjustment(
    comparison: Ordering,
    negative: bool,
    mode: F32RoundingMode,
) -> Ordering {
    match mode {
        F32RoundingMode::Down if comparison == Ordering::Greater => Ordering::Less,
        F32RoundingMode::Up if comparison == Ordering::Less => Ordering::Greater,
        F32RoundingMode::Zero if !negative && comparison == Ordering::Greater => Ordering::Less,
        F32RoundingMode::Zero if negative && comparison == Ordering::Less => Ordering::Greater,
        _ => Ordering::Equal,
    }
}

/// PTX `add{.rnd}.f64`: RN uses [`cuda_f64_add`]'s NaN selection (rhs NaN first,
/// `inf - inf` -> `0xfff8_0000_0000_0000`); RZ/RM/RP go through the exact [`fma_f64`]
/// with a unit multiplier. Subnormals kept.
pub fn add_f64(lhs: f64, rhs: f64, mode: F32RoundingMode) -> f64 {
    if mode == F32RoundingMode::Nearest || !lhs.is_finite() || !rhs.is_finite() {
        cuda_f64_add(lhs, rhs)
    } else {
        fma_f64(lhs, 1.0, rhs, mode)
    }
}

/// PTX `sub{.rnd}.f64`: RN host difference with NaN pinned by [`pin_nan2_f64`];
/// RZ/RM/RP via exact [`fma_f64`]. Subnormals kept.
pub fn sub_f64(lhs: f64, rhs: f64, mode: F32RoundingMode) -> f64 {
    if mode == F32RoundingMode::Nearest || !lhs.is_finite() || !rhs.is_finite() {
        pin_nan2_f64(lhs, rhs, lhs - rhs)
    } else {
        fma_f64(lhs, 1.0, -rhs, mode)
    }
}

/// PTX `mul{.rnd}.f64`: RN host product with NaN pinned by [`pin_nan2_f64`];
/// RZ/RM/RP via exact [`fma_f64`] with a signed zero addend (keeps `-0`). Subnormals kept.
pub fn mul_f64(lhs: f64, rhs: f64, mode: F32RoundingMode) -> f64 {
    if mode == F32RoundingMode::Nearest || !lhs.is_finite() || !rhs.is_finite() {
        pin_nan2_f64(lhs, rhs, lhs * rhs)
    } else {
        // Same-signed zero addition preserves the product's sign, including -0.
        let zero = f64::from_bits((lhs.to_bits() ^ rhs.to_bits()) & (1_u64 << 63));
        fma_f64(lhs, rhs, zero, mode)
    }
}

/// PTX `div.{rnd}.f64`: RN host quotient, RZ/RM/RP by exact comparison of the
/// candidate. Subnormals kept; NaN payload is the host division's (not pinned).
pub fn div_f64(lhs: f64, rhs: f64, mode: F32RoundingMode) -> f64 {
    let rounded = lhs / rhs;
    if mode == F32RoundingMode::Nearest
        || !lhs.is_finite()
        || !rhs.is_finite()
        || lhs == 0.0
        || rhs == 0.0
    {
        return rounded;
    }
    let comparison = compare_division_candidate(rounded, lhs, rhs);
    round_f64_candidate(
        rounded,
        comparison,
        lhs.is_sign_negative() != rhs.is_sign_negative(),
        mode,
    )
}

pub(crate) fn compare_division_candidate(rounded: f64, lhs: f64, rhs: f64) -> Ordering {
    debug_assert!(lhs.is_finite() && rhs.is_finite() && lhs != 0.0 && rhs != 0.0);
    // Compare |candidate| * |denominator| with |numerator| exactly. The
    // product has at most 106 significand bits; no floating division residual
    // or thread-local rounding state is needed, even at exponent extremes.
    let magnitude_comparison = if rounded.is_infinite() {
        Ordering::Greater
    } else if rounded == 0.0 {
        Ordering::Less
    } else {
        let (candidate, candidate_exp) = positive_f64_dyadic(rounded.abs());
        let (denominator, denominator_exp) = positive_f64_dyadic(rhs.abs());
        let (numerator, numerator_exp) = positive_f64_dyadic(lhs.abs());
        compare_positive_dyadics(
            candidate * denominator,
            candidate_exp + denominator_exp,
            numerator,
            numerator_exp,
        )
    };
    let negative = lhs.is_sign_negative() != rhs.is_sign_negative();
    if negative {
        magnitude_comparison.reverse()
    } else {
        magnitude_comparison
    }
}

/// PTX `fma.{rnd}.f32` without `.ftz`: exact fused result rounded by `mode`
/// (RN from the host FMA, others corrected from the exact sum). NaN pinned as in
/// [`host_fma_f32`]; subnormals kept.
pub fn fma_f32(lhs: f32, rhs: f32, addend: f32, mode: F32RoundingMode) -> f32 {
    let rounded = host_fma_f32(lhs, rhs, addend);
    if mode == F32RoundingMode::Nearest
        || !lhs.is_finite()
        || !rhs.is_finite()
        || !addend.is_finite()
    {
        return rounded;
    }
    let exact = exact_fma_sum(
        ExactFmaValue::product(lhs, rhs),
        ExactFmaValue::addend(addend),
    );
    round_f32_from_exact_fma(rounded, exact, lhs * rhs, addend, mode)
}

/// [`fma_f32`] with `.ftz`: subnormal inputs flush to signed zero; a result whose exact
/// value is below `f32::MIN_POSITIVE` in magnitude flushes to signed zero.
pub fn fma_f32_ftz(lhs: f32, rhs: f32, addend: f32, mode: F32RoundingMode) -> f32 {
    let lhs = flush_subnormal_f32(lhs);
    let rhs = flush_subnormal_f32(rhs);
    let addend = flush_subnormal_f32(addend);
    flush_f32_result(fma_f32(lhs, rhs, addend, mode), || {
        let exact = exact_fma_sum(
            ExactFmaValue::product(lhs, rhs),
            ExactFmaValue::addend(addend),
        );
        exact
            .magnitude
            .compare(ExactFmaValue::addend(f32::MIN_POSITIVE).magnitude)
            == Ordering::Less
    })
}

pub(crate) fn saturate_float<T: PartialOrd + From<u8>>(value: T) -> T {
    // PTX saturation maps NaNs and both signs of zero to positive zero.
    if value.partial_cmp(&T::from(0)) != Some(Ordering::Greater) {
        T::from(0)
    } else if value > T::from(1) {
        T::from(1)
    } else {
        value
    }
}

/// PTX `.sat` on binary32: clamp to `[0, 1]`; NaN and both zeros map to `+0`.
pub fn ptx_saturate_f32(value: f32) -> f32 {
    saturate_float(value)
}

/// PTX `.sat` on binary64: clamp to `[0, 1]`; NaN and both zeros map to `+0`.
pub fn ptx_saturate_f64(value: f64) -> f64 {
    saturate_float(value)
}
