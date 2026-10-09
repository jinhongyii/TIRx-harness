//! Scalar `.f32` / `.f64` arithmetic with PTX rounding, `.ftz`, `.sat`,
//! `.NaN`, `.abs` and `.xorsign` modifiers, plus approximate transcendentals.
//!
//! Moved from legacy `engine-rs/src/runtime/instructions/reg.rs`
//! (`f32_*_entry`, `F32DivideMode`, `apply_f32_clamp`, `f32_minmax_entry`,
//! `rcp/sqrt/sin/cos/exp2/lg2/rsqrt/tanh/neg/abs/copysign` variants and the
//! `f64_arithmetic_variant!` family). Numerics delegate to `crate::scalar`.

use crate::scalar::{self, F32RoundingMode};

/// Division / reciprocal mode of `div.f32`: an IEEE rounding, `.approx` or
/// `.full`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F32DivMode {
    Rounded(F32RoundingMode),
    Approx,
    Full,
}

/// PTX `.sat` clamp on an `.f32` result (NaN and both zeros map to `+0`).
#[inline(always)]
pub fn apply_f32_clamp(value: f32, saturate: bool) -> f32 {
    if saturate {
        scalar::ptx_saturate_f32(value)
    } else {
        value
    }
}

/// `add{.rnd}{.ftz}{.sat}.f32`.
pub fn add_f32(lhs: f32, rhs: f32, mode: F32RoundingMode, ftz: bool, sat: bool) -> f32 {
    let value = if ftz {
        scalar::add_f32_ftz(lhs, rhs, mode)
    } else {
        scalar::add_f32(lhs, rhs, mode)
    };
    apply_f32_clamp(value, sat)
}

/// `sub{.rnd}{.ftz}{.sat}.f32`.
pub fn sub_f32(lhs: f32, rhs: f32, mode: F32RoundingMode, ftz: bool, sat: bool) -> f32 {
    let value = if ftz {
        scalar::sub_f32_ftz(lhs, rhs, mode)
    } else {
        scalar::sub_f32(lhs, rhs, mode)
    };
    apply_f32_clamp(value, sat)
}

/// `mul{.rnd}{.ftz}{.sat}.f32`.
pub fn mul_f32(lhs: f32, rhs: f32, mode: F32RoundingMode, ftz: bool, sat: bool) -> f32 {
    let value = if ftz {
        scalar::mul_f32_ftz(lhs, rhs, mode)
    } else {
        scalar::mul_f32(lhs, rhs, mode)
    };
    apply_f32_clamp(value, sat)
}

/// `fma.rnd{.ftz}{.sat}.f32` and `mad.rnd{.ftz}{.sat}.f32` (both fused).
pub fn fma_f32(a: f32, b: f32, c: f32, mode: F32RoundingMode, ftz: bool, sat: bool) -> f32 {
    let value = if ftz {
        scalar::fma_f32_ftz(a, b, c, mode)
    } else {
        scalar::fma_f32(a, b, c, mode)
    };
    apply_f32_clamp(value, sat)
}

/// `div{.rnd,.approx,.full}{.ftz}.f32`.
pub fn div_f32(lhs: f32, rhs: f32, mode: F32DivMode, ftz: bool) -> f32 {
    match mode {
        F32DivMode::Rounded(round) => scalar::ptx_div_f32(lhs, rhs, round, ftz),
        F32DivMode::Approx => scalar::ptx_div_approx_f32(lhs, rhs, ftz, false),
        F32DivMode::Full => scalar::ptx_div_approx_f32(lhs, rhs, ftz, true),
    }
}

/// `rcp.rnd{.ftz}.f32` (IEEE-rounded reciprocal).
pub fn rcp_f32(value: f32, mode: F32RoundingMode, ftz: bool) -> f32 {
    scalar::ptx_div_f32(1.0, value, mode, ftz)
}

/// `rcp.approx{.ftz}.f32`.
pub fn rcp_approx_f32(value: f32, ftz: bool) -> f32 {
    if ftz {
        scalar::ptx_rcp_approx_ftz_f32(value)
    } else {
        scalar::ptx_rcp_approx_f32(value)
    }
}

/// `rcp.rnd.f64`.
pub fn rcp_f64(value: f64, mode: F32RoundingMode) -> f64 {
    scalar::div_f64(1.0, value, mode)
}

/// `sqrt.rnd{.ftz}.f32`; `sqrt.approx{.ftz}.f32` uses `mode = Nearest`
/// (the correctly rounded value is the deterministic representative).
pub fn sqrt_f32(value: f32, mode: F32RoundingMode, ftz: bool) -> f32 {
    scalar::ptx_sqrt_f32(value, mode, ftz)
}

/// `ex2.approx{.ftz}.f32`.
pub fn ex2_approx_f32(value: f32, ftz: bool) -> f32 {
    if ftz {
        scalar::ptx_exp2_approx_ftz_f32(value)
    } else {
        scalar::ptx_exp2_approx_f32(value)
    }
}

/// `lg2.approx{.ftz}.f32`.
pub fn lg2_approx_f32(value: f32, ftz: bool) -> f32 {
    if ftz {
        scalar::ptx_lg2_approx_ftz_f32(value)
    } else {
        scalar::det::log2_f32(value)
    }
}

/// `lg2.f64` representative (host `log2`).
pub fn lg2_f64(value: f64) -> f64 {
    scalar::det::log2_f64(value)
}

/// `rsqrt.approx{.ftz}.f32`.
pub fn rsqrt_approx_f32(value: f32, ftz: bool) -> f32 {
    if ftz {
        scalar::ptx_rsqrt_approx_ftz_f32(value)
    } else {
        scalar::ptx_rsqrt_approx_f32(value)
    }
}

/// Legacy `rsqrt.f64` marker: `1 / sqrt(x)` in binary64.
pub fn rsqrt_f64(value: f64) -> f64 {
    1.0 / value.sqrt()
}

/// `neg{.ftz}.f32`.
pub fn neg_f32(value: f32, ftz: bool) -> f32 {
    if ftz {
        scalar::ptx_neg_ftz_f32(value)
    } else {
        -value
    }
}

/// `neg.f64`.
pub fn neg_f64(value: f64) -> f64 {
    -value
}

/// `abs{.ftz}.f32`.
pub fn abs_f32(value: f32, ftz: bool) -> f32 {
    if ftz {
        scalar::flush_subnormal_f32(value).abs()
    } else {
        value.abs()
    }
}

/// `abs.f64`: PTX passes NaNs through unchanged, unlike its f32/half forms.
pub fn abs_f64(value: f64) -> f64 {
    if value.is_nan() {
        value
    } else {
        value.abs()
    }
}

/// `copysign.f32 d, a, b`: magnitude of `b`, sign of `a` (PTX operand order).
pub fn copysign_f32(sign: f32, magnitude: f32) -> f32 {
    f32::from_bits((magnitude.to_bits() & 0x7fff_ffff) | (sign.to_bits() & 0x8000_0000))
}

/// `copysign.f64 d, a, b`.
pub fn copysign_f64(sign: f64, magnitude: f64) -> f64 {
    f64::from_bits(
        (magnitude.to_bits() & 0x7fff_ffff_ffff_ffff) | (sign.to_bits() & 0x8000_0000_0000_0000),
    )
}

/// `min/max{.ftz}{.NaN}{.abs}{.xorsign.abs}.f32` with two or three sources.
///
/// `.xorsign` uses the sign of the first two sources' XOR and is not applied
/// to a NaN result.
pub fn minmax_f32<const N: usize>(
    args: [f32; N],
    ftz: bool,
    propagate_nan: bool,
    absolute: bool,
    xor_sign: bool,
    maximum: bool,
) -> f32 {
    let operation = if maximum {
        scalar::ptx_max_f32
    } else {
        scalar::ptx_min_f32
    };
    let sign = (args[0].to_bits() ^ args[1].to_bits()) & 0x8000_0000;
    let values = args.map(|value| if absolute { value.abs() } else { value });
    let result = values[1..].iter().fold(values[0], |acc, &next| {
        operation(acc, next, ftz, propagate_nan)
    });
    if xor_sign && !result.is_nan() {
        f32::from_bits((result.to_bits() & 0x7fff_ffff) | sign)
    } else {
        result
    }
}

/// `max{.ftz}{.NaN}.f32` with two sources.
pub fn max_f32(lhs: f32, rhs: f32, ftz: bool, propagate_nan: bool) -> f32 {
    minmax_f32([lhs, rhs], ftz, propagate_nan, false, false, true)
}

/// `min{.ftz}{.NaN}.f32` with two sources.
pub fn min_f32(lhs: f32, rhs: f32, ftz: bool, propagate_nan: bool) -> f32 {
    minmax_f32([lhs, rhs], ftz, propagate_nan, false, false, false)
}
