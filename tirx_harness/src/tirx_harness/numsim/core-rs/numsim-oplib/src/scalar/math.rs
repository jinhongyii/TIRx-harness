//! Host-rule elementwise math builtins (`tirx.log1p`, `tirx.sigmoid`).
//!
//! Legacy (`frontend-rs/src/emit/pure.rs` `log1p` / `sigmoid`) accepted only
//! `float32 -> float32` and computed with host Rust: `x.ln_1p()` (the C
//! library's `log1pf`) and `1.0 / (1.0 + (-x).exp())`. These kernels keep that
//! value path, except that `log1p`/`exp` come from the pure-Rust `libm` crate
//! ([`super::det`], delta D14) so the bits do not depend on the host's C
//! library, and pin what host code leaves to the library or the code
//! generator:
//!
//! * `log1p`: NaN in -> the same NaN, quieted; `x < -1` (incl. `-inf`) -> the
//!   x86 default NaN; `x == -1` -> `-inf`; `±0` -> `±0`; `+inf` -> `+inf`;
//!   otherwise `libm::log1pf` / `libm::log1p` (< 1 ulp).
//! * `sigmoid`: NaN in -> that NaN with its sign flipped (the `-x` of the
//!   formula), quieted; otherwise the formula in binary32 with round to
//!   nearest at each step (`exp` from `libm`, see [`super::det`]).
//! * f16 / bf16 (an extension: legacy rejected them) decode to f32, use the
//!   f32 kernel, and round back to the half format with RNE (the TIR half
//!   rule: `cvt::f32_to_fp16_bits` / `f32_to_bf16_bits`).

use crate::cvt::{bf16_bits_to_f32, f32_to_bf16_bits, f32_to_fp16_bits, fp16_bits_to_f32};

const F32_DEFAULT_NAN: u32 = 0xffc0_0000;
const F64_DEFAULT_NAN: u64 = 0xfff8_0000_0000_0000;

/// `tirx.log1p` on binary32 (see the module docs).
pub fn log1p_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::from_bits(x.to_bits() | 0x0040_0000);
    }
    if x < -1.0 {
        return f32::from_bits(F32_DEFAULT_NAN);
    }
    if x == -1.0 {
        return f32::NEG_INFINITY;
    }
    if x == 0.0 || x == f32::INFINITY {
        return x;
    }
    super::det::ln_1p_f32(x)
}

/// `tirx.log1p` on binary64.
pub fn log1p_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::from_bits(x.to_bits() | 0x0008_0000_0000_0000);
    }
    if x < -1.0 {
        return f64::from_bits(F64_DEFAULT_NAN);
    }
    if x == -1.0 {
        return f64::NEG_INFINITY;
    }
    if x == 0.0 || x == f64::INFINITY {
        return x;
    }
    super::det::ln_1p_f64(x)
}

/// `tirx.sigmoid` on binary32: `1 / (1 + exp(-x))` (see the module docs).
pub fn sigmoid_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::from_bits((x.to_bits() ^ 0x8000_0000) | 0x0040_0000);
    }
    1.0_f32 / (1.0_f32 + super::det::exp_f32(-x))
}

/// `tirx.sigmoid` on binary64.
pub fn sigmoid_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::from_bits((x.to_bits() ^ (1 << 63)) | 0x0008_0000_0000_0000);
    }
    1.0_f64 / (1.0_f64 + super::det::exp_f64(-x))
}

/// A binary32 kernel applied to a half (`bf16 = false`: f16) bit pattern.
pub fn half_unary(bits: u16, bf16: bool, f: impl Fn(f32) -> f32) -> u16 {
    if bf16 {
        f32_to_bf16_bits(f(bf16_bits_to_f32(bits)))
    } else {
        f32_to_fp16_bits(f(fp16_bits_to_f32(bits)))
    }
}

// ---------------------------------------------------------------------------
// v2-only math builtins (`tirx.erf`, `tirx.exp10`, `tirx.log10`,
// `tirx.nearbyint`). Legacy had no lowering for them. Values come from the
// pure-Rust `libm` crate (machine independent; < 1 ulp), and NaNs are
// pinned: a NaN input is returned quieted, and an invalid operation
// (`log10` of a negative number) gives the x86 default NaN.
// `nearbyint` rounds to the nearest integer, ties to even (the default
// rounding mode), and is exact.
// ---------------------------------------------------------------------------

#[inline]
fn quiet32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() | 0x0040_0000)
}

#[inline]
fn quiet64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() | 0x0008_0000_0000_0000)
}

/// `tirx.erf` on binary32.
pub fn erf_f32(x: f32) -> f32 {
    if x.is_nan() {
        quiet32(x)
    } else {
        libm::erff(x)
    }
}

/// `tirx.erf` on binary64.
pub fn erf_f64(x: f64) -> f64 {
    if x.is_nan() {
        quiet64(x)
    } else {
        libm::erf(x)
    }
}

/// `tirx.exp10` on binary32: computed in binary64 (`libm::exp10`) and
/// rounded once (`libm::exp10f` is off by more than one ulp near -4).
pub fn exp10_f32(x: f32) -> f32 {
    if x.is_nan() {
        quiet32(x)
    } else {
        libm::exp10(f64::from(x)) as f32
    }
}

/// `tirx.exp10` on binary64.
pub fn exp10_f64(x: f64) -> f64 {
    if x.is_nan() {
        quiet64(x)
    } else {
        libm::exp10(x)
    }
}

/// `tirx.log10` on binary32: `log10(-0) = log10(+0) = -inf`, negative ->
/// default NaN.
pub fn log10_f32(x: f32) -> f32 {
    if x.is_nan() {
        quiet32(x)
    } else if x == 0.0 {
        f32::NEG_INFINITY
    } else if x < 0.0 {
        f32::from_bits(F32_DEFAULT_NAN)
    } else {
        libm::log10f(x)
    }
}

/// `tirx.log10` on binary64.
pub fn log10_f64(x: f64) -> f64 {
    if x.is_nan() {
        quiet64(x)
    } else if x == 0.0 {
        f64::NEG_INFINITY
    } else if x < 0.0 {
        f64::from_bits(F64_DEFAULT_NAN)
    } else {
        libm::log10(x)
    }
}

/// `tirx.nearbyint` on binary32 (ties to even; signed zeros kept).
pub fn nearbyint_f32(x: f32) -> f32 {
    if x.is_nan() {
        quiet32(x)
    } else {
        x.round_ties_even()
    }
}

/// `tirx.nearbyint` on binary64.
pub fn nearbyint_f64(x: f64) -> f64 {
    if x.is_nan() {
        quiet64(x)
    } else {
        x.round_ties_even()
    }
}
