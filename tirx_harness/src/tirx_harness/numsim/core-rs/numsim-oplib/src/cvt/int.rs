//! Integer-side PTX `cvt` forms: integer<->integer, float->integer and
//! integer->float, as plain functions over runtime parameters.
//!
//! Ported from the legacy `engine-rs/src/runtime/instructions/reg.rs` cvt
//! section (`cvt_variant!`, `integer_cvt_variant!`, `integer_cvt_destinations!`,
//! `cvt_float_to_int_*`, `cvt_int_to_float_*`).  The numeric cores live in
//! `cvt::ptx`; this module only binds the per-type NaN, saturation and
//! exactness rules the legacy marker macros encoded.

use crate::cvt::formats::bf16_bits_to_f32;
use crate::cvt::ptx::{
    ptx_cvt_integer_to_bf16, ptx_cvt_integer_to_f16, ptx_cvt_integer_to_f32,
    ptx_cvt_integer_to_f64, ptx_cvt_integral_f32, ptx_cvt_integral_f64, PtxFloatRounding,
    PtxIntegerRounding,
};
use crate::scalar::cuda_fp16_bits_to_f32;

/// A PTX integer register type (`.s8` .. `.u64`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntKind {
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
}

impl IntKind {
    /// Bit width of the type.
    pub const fn bits(self) -> u32 {
        match self {
            Self::S8 | Self::U8 => 8,
            Self::S16 | Self::U16 => 16,
            Self::S32 | Self::U32 => 32,
            Self::S64 | Self::U64 => 64,
        }
    }

    /// Whether the type is signed.
    pub const fn signed(self) -> bool {
        matches!(self, Self::S8 | Self::S16 | Self::S32 | Self::S64)
    }

    /// Smallest representable value.
    pub const fn min(self) -> i128 {
        if self.signed() {
            -(1_i128 << (self.bits() - 1))
        } else {
            0
        }
    }

    /// Largest representable value.
    pub const fn max(self) -> i128 {
        if self.signed() {
            (1_i128 << (self.bits() - 1)) - 1
        } else {
            (1_i128 << self.bits()) - 1
        }
    }

    /// Mask selecting the type's payload bits.
    pub const fn mask(self) -> u64 {
        if self.bits() == 64 {
            u64::MAX
        } else {
            (1_u64 << self.bits()) - 1
        }
    }

    /// Value of the low `bits()` bits of `payload`, sign- or zero-extended.
    pub const fn value(self, payload: u64) -> i128 {
        let shift = 64 - self.bits();
        if self.signed() {
            (((payload << shift) as i64) >> shift) as i128
        } else {
            ((payload << shift) >> shift) as i128
        }
    }

    /// Raw payload (low `bits()` bits) of a value; wraps like Rust `as`.
    pub const fn payload(self, value: i128) -> u64 {
        (value as u64) & self.mask()
    }

    /// Parse a PTX type suffix such as `s32` or `u8`.
    pub fn from_ptx(name: &str) -> Option<Self> {
        Some(match name {
            "s8" => Self::S8,
            "s16" => Self::S16,
            "s32" => Self::S32,
            "s64" => Self::S64,
            "u8" => Self::U8,
            "u16" => Self::U16,
            "u32" => Self::U32,
            "u64" => Self::U64,
            _ => return None,
        })
    }
}

/// `cvt{.sat}.<int>.<int>`: plain forms wrap (truncate / sign-extend, as Rust
/// `as`); `.sat` clamps to the destination range first.
pub fn cvt_int_to_int(payload: u64, src: IntKind, dst: IntKind, sat: bool) -> u64 {
    let value = src.value(payload);
    let value = if sat {
        value.clamp(dst.min(), dst.max())
    } else {
        value
    };
    dst.payload(value)
}

/// NaN answer of `cvt.irnd.<int>.<float>`.
///
/// A non-`.f64` source answers zero unless the destination is 64-bit (then
/// `1 << 63`); an `.f64` source answers `1 << (BitWidth(dst) - 1)`.
pub const fn cvt_float_to_int_nan(dst: IntKind, f64_source: bool) -> u64 {
    if f64_source || dst.bits() == 64 {
        1_u64 << (dst.bits() - 1)
    } else {
        0
    }
}

fn saturate_integral(value: i128, dst: IntKind) -> u64 {
    dst.payload(value.clamp(dst.min(), dst.max()))
}

/// `cvt.irnd{.ftz}{.sat}.<int>.f32`.
///
/// Float-to-integer conversion always saturates in PTX, so `.sat` is inert
/// (the legacy lowering bound both spellings to one specialization).
pub fn cvt_f32_to_int(
    value: f32,
    rounding: PtxIntegerRounding,
    ftz: bool,
    sat: bool,
    dst: IntKind,
) -> u64 {
    let _ = sat;
    if value.is_nan() {
        return cvt_float_to_int_nan(dst, false);
    }
    saturate_integral(ptx_cvt_integral_f32(value, rounding, ftz) as i128, dst)
}

/// `cvt.irnd{.sat}.<int>.f64`.
pub fn cvt_f64_to_int(value: f64, rounding: PtxIntegerRounding, sat: bool, dst: IntKind) -> u64 {
    let _ = sat;
    if value.is_nan() {
        return cvt_float_to_int_nan(dst, true);
    }
    saturate_integral(ptx_cvt_integral_f64(value, rounding) as i128, dst)
}

/// `cvt.irnd{.sat}.<int>.f16` (source widens exactly; no `.ftz`).
pub fn cvt_f16_to_int(bits: u16, rounding: PtxIntegerRounding, sat: bool, dst: IntKind) -> u64 {
    cvt_f32_to_int(cuda_fp16_bits_to_f32(bits), rounding, false, sat, dst)
}

/// `cvt.irnd{.sat}.<int>.bf16` (no 8-bit destination; ptxas rejects it).
pub fn cvt_bf16_to_int(bits: u16, rounding: PtxIntegerRounding, sat: bool, dst: IntKind) -> u64 {
    cvt_f32_to_int(bf16_bits_to_f32(bits), rounding, false, sat, dst)
}

fn magnitude_and_sign(payload: u64, src: IntKind) -> (u64, bool) {
    let value = src.value(payload);
    (value.unsigned_abs() as u64, value < 0)
}

/// Whether every `src` value is exactly representable with `significand_bits`.
/// For exact pairs the rounding axis does not exist (legacy `variant::Exact`),
/// so every `.frnd` spelling binds nearest-even.
const fn exact_pair(src: IntKind, significand_bits: u32) -> bool {
    src.bits() <= significand_bits
}

fn effective(src: IntKind, significand_bits: u32, rounding: PtxFloatRounding) -> PtxFloatRounding {
    if exact_pair(src, significand_bits) {
        PtxFloatRounding::NearestEven
    } else {
        rounding
    }
}

/// `cvt.frnd{.ftz}.f32.<int>` (`.ftz` is inert for integer sources).
pub fn cvt_int_to_f32(payload: u64, src: IntKind, rounding: PtxFloatRounding) -> f32 {
    let (magnitude, negative) = magnitude_and_sign(payload, src);
    ptx_cvt_integer_to_f32(magnitude, negative, effective(src, 24, rounding))
}

/// `cvt.frnd.f64.<int>`.
pub fn cvt_int_to_f64(payload: u64, src: IntKind, rounding: PtxFloatRounding) -> f64 {
    let (magnitude, negative) = magnitude_and_sign(payload, src);
    ptx_cvt_integer_to_f64(magnitude, negative, effective(src, 53, rounding))
}

/// `cvt.frnd.f16.<int>`; only the 8-bit sources are exact.
pub fn cvt_int_to_f16(payload: u64, src: IntKind, rounding: PtxFloatRounding) -> u16 {
    let (magnitude, negative) = magnitude_and_sign(payload, src);
    ptx_cvt_integer_to_f16(magnitude, negative, effective(src, 8, rounding))
}

/// `cvt.frnd.bf16.<int>` (16-bit and wider sources only).
pub fn cvt_int_to_bf16(payload: u64, src: IntKind, rounding: PtxFloatRounding) -> u16 {
    let (magnitude, negative) = magnitude_and_sign(payload, src);
    ptx_cvt_integer_to_bf16(magnitude, negative, rounding)
}

/// `cvt.frnd.sat.{f16,f32,f64}.<int>`: post-conversion `[0, 1]` saturation of an
/// integer is exactly `value > 0`, for every rounding and `.ftz` spelling.
/// Returns `true` when the saturated result is one.
pub fn cvt_int_sat_is_one(payload: u64, src: IntKind) -> bool {
    src.value(payload) > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_to_integer_nan_depends_on_both_widths() {
        let rz = PtxIntegerRounding::Zero;
        assert_eq!(cvt_f32_to_int(f32::NAN, rz, false, false, IntKind::U8), 0);
        assert_eq!(cvt_f32_to_int(f32::NAN, rz, false, false, IntKind::U32), 0);
        assert_eq!(
            cvt_f32_to_int(f32::NAN, rz, false, false, IntKind::U64),
            1 << 63
        );
        assert_eq!(cvt_f64_to_int(f64::NAN, rz, false, IntKind::U8), 0x80);
        assert_eq!(
            cvt_f64_to_int(f64::NAN, rz, false, IntKind::U32),
            0x8000_0000
        );
        assert_eq!(cvt_f64_to_int(f64::NAN, rz, false, IntKind::S16), 0x8000);
    }

    #[test]
    fn float_to_integer_saturates_and_rounds() {
        let rn = PtxIntegerRounding::NearestEven;
        assert_eq!(cvt_f32_to_int(2.5, rn, false, false, IntKind::S32), 2);
        assert_eq!(
            cvt_f32_to_int(-2.5, rn, false, false, IntKind::S32),
            0xffff_fffe
        );
        assert_eq!(
            cvt_f32_to_int(1e20, rn, false, false, IntKind::S32),
            0x7fff_ffff
        );
        assert_eq!(cvt_f32_to_int(-1.0, rn, false, true, IntKind::U16), 0);
        assert_eq!(
            cvt_f32_to_int(f32::INFINITY, rn, false, false, IntKind::U64),
            u64::MAX
        );
        let rp = PtxIntegerRounding::PositiveInfinity;
        let tiny = f32::from_bits(1);
        assert_eq!(cvt_f32_to_int(tiny, rp, false, false, IntKind::S32), 1);
        assert_eq!(cvt_f32_to_int(tiny, rp, true, false, IntKind::S32), 0);
    }

    #[test]
    fn integer_to_integer_wraps_or_clamps() {
        assert_eq!(
            cvt_int_to_int(0xff, IntKind::S8, IntKind::U32, false),
            0xffff_ffff
        );
        assert_eq!(cvt_int_to_int(0xff, IntKind::S8, IntKind::U32, true), 0);
        assert_eq!(
            cvt_int_to_int(u64::MAX, IntKind::U64, IntKind::S64, true),
            i64::MAX as u64
        );
        assert_eq!(
            cvt_int_to_int(0x1_23, IntKind::U32, IntKind::U8, false),
            0x23
        );
        assert_eq!(
            cvt_int_to_int(0x1_23, IntKind::U32, IntKind::U8, true),
            0xff
        );
    }

    #[test]
    fn exact_integer_pairs_ignore_the_rounding_axis() {
        for rounding in [
            PtxFloatRounding::NearestEven,
            PtxFloatRounding::Zero,
            PtxFloatRounding::NegativeInfinity,
            PtxFloatRounding::PositiveInfinity,
        ] {
            assert_eq!(cvt_int_to_f32(0x8000, IntKind::S16, rounding), -32768.0);
            assert_eq!(cvt_int_to_f16(0x80, IntKind::S8, rounding), 0xd800);
        }
        // 2^24 + 1 is not exact in f32: directed modes differ.
        assert_eq!(
            cvt_int_to_f32(16_777_217, IntKind::U32, PtxFloatRounding::PositiveInfinity),
            16_777_218.0
        );
        assert!(cvt_int_sat_is_one(5, IntKind::S32));
        assert!(!cvt_int_sat_is_one(0xffff_ffff, IntKind::S32));
    }
}
