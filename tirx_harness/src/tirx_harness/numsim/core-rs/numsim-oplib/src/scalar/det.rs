//! Host-independent transcendental functions: the pure-Rust `libm` crate
//! (pinned in `Cargo.toml`) instead of Rust `std`, whose `f32::exp`,
//! `f64::ln`, ... call the system C library. glibc's results differ by an
//! ulp between versions, so corpus output bits would depend on the host's
//! glibc (W4, after the CI runner divergence audit; numsim-behaviour-deltas
//! D14). Every NumSim path that evaluates one of these functions on kernel
//! data goes through this module.
//!
//! The binary32 functions evaluate the binary64 `libm` routine on the exact
//! widening and round once to binary32. That is the correctly rounded result
//! unless the binary64 value lies within its own error of a binary32
//! rounding boundary; the exhaustive binary32 check (delta D14) records how
//! often that happens. `libm`'s binary32 routines are up to one ulp off
//! (e.g. `log2f(6)`), which the device-intrinsic representative tests in
//! `tests/numsim/runtime/test_scalar_control.py` reject.
//!
//! NaN results keep the previous (glibc, x86-64) bits where glibc is
//! consistent: a unary function of a NaN returns that NaN quieted (checked
//! for every binary32 NaN and sampled binary64 NaNs; `libm` alone returns a
//! signaling NaN unquieted). For `pow`/`atan2`, glibc's signaling-NaN
//! results are inconsistent (e.g. it clears the sign in `powf(sNaN, 1)`);
//! here a signaling NaN in the first operand, else in the second, is returned
//! quieted, and every other input (quiet NaNs included) takes `libm`.

/// `exp` (binary32).
#[inline]
pub fn exp_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::from_bits(x.to_bits() | 0x0040_0000);
    }
    libm::exp(f64::from(x)) as f32
}
/// `exp2` (binary32).
#[inline]
pub fn exp2_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::from_bits(x.to_bits() | 0x0040_0000);
    }
    libm::exp2(f64::from(x)) as f32
}
/// Natural log (binary32).
#[inline]
pub fn ln_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::from_bits(x.to_bits() | 0x0040_0000);
    }
    libm::log(f64::from(x)) as f32
}
/// `log2` (binary32).
#[inline]
pub fn log2_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::from_bits(x.to_bits() | 0x0040_0000);
    }
    libm::log2(f64::from(x)) as f32
}
/// `log1p` (binary32).
#[inline]
pub fn ln_1p_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::from_bits(x.to_bits() | 0x0040_0000);
    }
    libm::log1p(f64::from(x)) as f32
}
/// `sin` (binary32).
#[inline]
pub fn sin_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::from_bits(x.to_bits() | 0x0040_0000);
    }
    libm::sin(f64::from(x)) as f32
}
/// `cos` (binary32).
#[inline]
pub fn cos_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::from_bits(x.to_bits() | 0x0040_0000);
    }
    libm::cos(f64::from(x)) as f32
}
/// `tanh` (binary32).
#[inline]
pub fn tanh_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::from_bits(x.to_bits() | 0x0040_0000);
    }
    libm::tanh(f64::from(x)) as f32
}
/// `pow` (binary32).
#[inline]
pub fn pow_f32(x: f32, y: f32) -> f32 {
    if let Some(nan) = signaling_f32(x).or_else(|| signaling_f32(y)) {
        return nan;
    }
    libm::pow(f64::from(x), f64::from(y)) as f32
}
/// `atan2(y, x)` (binary32).
#[inline]
pub fn atan2_f32(y: f32, x: f32) -> f32 {
    if let Some(nan) = signaling_f32(y).or_else(|| signaling_f32(x)) {
        return nan;
    }
    libm::atan2(f64::from(y), f64::from(x)) as f32
}

/// `exp` (binary64).
#[inline]
pub fn exp_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::from_bits(x.to_bits() | 0x0008_0000_0000_0000);
    }
    libm::exp(x)
}
/// `exp2` (binary64).
#[inline]
pub fn exp2_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::from_bits(x.to_bits() | 0x0008_0000_0000_0000);
    }
    libm::exp2(x)
}
/// Natural log (binary64).
#[inline]
pub fn ln_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::from_bits(x.to_bits() | 0x0008_0000_0000_0000);
    }
    libm::log(x)
}
/// `log2` (binary64).
#[inline]
pub fn log2_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::from_bits(x.to_bits() | 0x0008_0000_0000_0000);
    }
    libm::log2(x)
}
/// `log1p` (binary64).
#[inline]
pub fn ln_1p_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::from_bits(x.to_bits() | 0x0008_0000_0000_0000);
    }
    libm::log1p(x)
}
/// `sin` (binary64).
#[inline]
pub fn sin_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::from_bits(x.to_bits() | 0x0008_0000_0000_0000);
    }
    libm::sin(x)
}
/// `cos` (binary64).
#[inline]
pub fn cos_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::from_bits(x.to_bits() | 0x0008_0000_0000_0000);
    }
    libm::cos(x)
}
/// `tanh` (binary64).
#[inline]
pub fn tanh_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::from_bits(x.to_bits() | 0x0008_0000_0000_0000);
    }
    libm::tanh(x)
}
/// `pow` (binary64).
#[inline]
pub fn pow_f64(x: f64, y: f64) -> f64 {
    if let Some(nan) = signaling_f64(x).or_else(|| signaling_f64(y)) {
        return nan;
    }
    libm::pow(x, y)
}
/// `atan2(y, x)` (binary64).
#[inline]
pub fn atan2_f64(y: f64, x: f64) -> f64 {
    if let Some(nan) = signaling_f64(y).or_else(|| signaling_f64(x)) {
        return nan;
    }
    libm::atan2(y, x)
}

/// `Some(quieted x)` when `x` is a signaling NaN.
#[inline]
fn signaling_f32(x: f32) -> Option<f32> {
    (x.is_nan() && x.to_bits() & 0x0040_0000 == 0).then(|| f32::from_bits(x.to_bits() | 0x0040_0000))
}

/// `Some(quieted x)` when `x` is a signaling NaN.
#[inline]
fn signaling_f64(x: f64) -> Option<f64> {
    (x.is_nan() && x.to_bits() & 0x0008_0000_0000_0000 == 0).then(|| f64::from_bits(x.to_bits() | 0x0008_0000_0000_0000))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unary NaNs come back quieted with their payload and sign (glibc's
    /// x86-64 bits); `pow`/`atan2` quiet the first signaling operand.
    #[test]
    fn nan_inputs_are_quieted_deterministically() {
        for bits in [0x7f80_0001_u32, 0xff80_0005, 0x7fc0_1234, 0xffa0_0001] {
            let x = f32::from_bits(bits);
            let quiet = bits | 0x0040_0000;
            for f in [exp_f32, exp2_f32, ln_f32, log2_f32, ln_1p_f32, sin_f32, cos_f32, tanh_f32] {
                assert_eq!(f(x).to_bits(), quiet, "{bits:#x}");
            }
        }
        let s = f32::from_bits(0xffa0_0001);
        assert_eq!(pow_f32(1.0, s).to_bits(), 0xffe0_0001);
        assert_eq!(pow_f32(s, 0.0).to_bits(), 0xffe0_0001);
        assert_eq!(atan2_f32(f32::from_bits(0x7fc0_1234), s).to_bits(), 0xffe0_0001);
        assert_eq!(pow_f32(f32::from_bits(0x7fc0_1234), 0.0), 1.0, "quiet NaN keeps C99 pow(NaN, 0) = 1");
        assert_eq!(exp_f64(f64::from_bits(0x7ff0_0000_0000_0001)).to_bits(), 0x7ff8_0000_0000_0001);
    }

    /// Values are the binary64 `libm` result rounded once.
    #[test]
    fn values_are_libm() {
        for x in [-3.5_f32, -0.0, 0.25, 1.0, 7.75, 88.0] {
            assert_eq!(exp_f32(x).to_bits(), (libm::exp(f64::from(x)) as f32).to_bits());
            assert_eq!(tanh_f32(x).to_bits(), (libm::tanh(f64::from(x)) as f32).to_bits());
            assert_eq!(atan2_f32(x, 2.0).to_bits(), (libm::atan2(f64::from(x), 2.0) as f32).to_bits());
        }
        // Correctly rounded where `libm::log2f` is one ulp off (the CI
        // finding): log2(6) and log2(12).
        assert_eq!(log2_f32(6.0).to_bits(), (6.0_f64.log2() as f32).to_bits());
        assert_eq!(log2_f32(12.0).to_bits(), (12.0_f64.log2() as f32).to_bits());
        assert_ne!(log2_f32(6.0).to_bits(), libm::log2f(6.0).to_bits());
    }
}
