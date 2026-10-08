//! Host-rule elementwise math builtins (`tirx.log1p`, `tirx.sigmoid`).
//!
//! Legacy (`frontend-rs/src/emit/pure.rs` `log1p` / `sigmoid`) accepted only
//! `float32 -> float32` and computed with host Rust: `x.ln_1p()` (the C
//! library's `log1pf`) and `1.0 / (1.0 + (-x).exp())`. These kernels keep that
//! value path and pin what host code leaves to the C library or the code
//! generator:
//!
//! * `log1p`: NaN in -> the same NaN, quieted; `x < -1` (incl. `-inf`) -> the
//!   x86 default NaN; `x == -1` -> `-inf`; `±0` -> `±0`; `+inf` -> `+inf`;
//!   otherwise the C library's `log1pf` / `log1p` (glibc: < 1 ulp).
//! * `sigmoid`: NaN in -> that NaN with its sign flipped (the `-x` of the
//!   formula), quieted; otherwise the formula in binary32 with round to
//!   nearest at each step (`exp` from the C library).
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
    x.ln_1p()
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
    x.ln_1p()
}

/// `tirx.sigmoid` on binary32: `1 / (1 + exp(-x))` (see the module docs).
pub fn sigmoid_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::from_bits((x.to_bits() ^ 0x8000_0000) | 0x0040_0000);
    }
    1.0_f32 / (1.0_f32 + (-x).exp())
}

/// `tirx.sigmoid` on binary64.
pub fn sigmoid_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::from_bits((x.to_bits() ^ (1 << 63)) | 0x0008_0000_0000_0000);
    }
    1.0_f64 / (1.0_f64 + (-x).exp())
}

/// A binary32 kernel applied to a half (`bf16 = false`: f16) bit pattern.
pub fn half_unary(bits: u16, bf16: bool, f: impl Fn(f32) -> f32) -> u16 {
    if bf16 {
        f32_to_bf16_bits(f(bf16_bits_to_f32(bits)))
    } else {
        f32_to_fp16_bits(f(fp16_bits_to_f32(bits)))
    }
}
