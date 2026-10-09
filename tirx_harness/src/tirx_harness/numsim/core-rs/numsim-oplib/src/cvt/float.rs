//! Scalar float<->float PTX `cvt` forms as plain functions.
//!
//! Ported from the legacy `engine-rs/src/runtime/instructions/reg.rs` cvt
//! section: the plain `cvt_variant!` / `exact_cvt_variant!` float rows,
//! `cvt_f64_to_f32_rounding!`, `cvt_same_size_integral!`, the `.ftz`/`.sat`
//! spellings, `cvt_f32_half_forms!`, `cvt_half_cross_rounding!`,
//! `cvt_narrow_modifiers!`, the tf32 `.rna` rows and the scalar `.pzo`
//! destinations (`scalar_pzo_destination!`), plus the `decode_*`/`encode_*`
//! register codecs (reg.rs:2573-2587).

use crate::cvt::formats::{bf16_bits_to_f32, f32_to_bf16_bits};
use crate::cvt::ptx::{
    ptx_cvt_f32_to_bf16, ptx_cvt_f32_to_f16, ptx_cvt_f32_to_f32, ptx_cvt_f32_to_f64,
    ptx_cvt_f32_to_tf32, ptx_cvt_f64_to_f32, ptx_cvt_f64_to_low, ptx_cvt_integral_f32,
    ptx_cvt_integral_f32_to_f32, ptx_cvt_integral_f64_to_f64, ptx_cvt_low_to_f64, ptx_cvt_pzo_u16,
    ptx_cvt_pzo_u32, ptx_cvt_widen_to_f32, PtxFloatRounding, PtxIntegerRounding,
};
use crate::scalar::{
    cuda_canonicalize_nan_f32, cuda_f32_to_fp16_bits, cuda_fp16_bits_to_f32, flush_subnormal_f32,
    ptx_saturate_f32, ptx_saturate_f64, LowPrecisionFormat,
};

/// Register decode of an `.f16` payload (legacy `decode_f16`).
pub fn reg_decode_f16(bits: u16) -> f32 {
    cuda_fp16_bits_to_f32(bits)
}

/// Register encode to `.f16` (legacy `encode_f16`).
pub fn reg_encode_f16(value: f32) -> u16 {
    cuda_f32_to_fp16_bits(value)
}

/// Register decode of a `.bf16` payload (legacy `decode_bf16`).
pub fn reg_decode_bf16(bits: u16) -> f32 {
    bf16_bits_to_f32(bits)
}

/// Register encode to `.bf16`, NaN-canonicalizing first (legacy `encode_bf16`).
pub fn reg_encode_bf16(value: f32) -> u16 {
    f32_to_bf16_bits(cuda_canonicalize_nan_f32(value))
}

/// Decode a 16-bit float payload of `format`.
pub fn reg_decode_half(bits: u16, format: LowPrecisionFormat) -> f32 {
    match format {
        LowPrecisionFormat::F16 => reg_decode_f16(bits),
        LowPrecisionFormat::Bf16 => reg_decode_bf16(bits),
    }
}

/// Encode an `f32` to a 16-bit float payload of `format`.
pub fn reg_encode_half(value: f32, format: LowPrecisionFormat) -> u16 {
    match format {
        LowPrecisionFormat::F16 => reg_encode_f16(value),
        LowPrecisionFormat::Bf16 => reg_encode_bf16(value),
    }
}

/// `ptx_cvt_f32_to_{f16,bf16}` selected by format.
pub fn ptx_cvt_f32_to_half(
    value: f32,
    format: LowPrecisionFormat,
    rounding: PtxFloatRounding,
    relu: bool,
    satfinite: bool,
) -> u16 {
    match format {
        LowPrecisionFormat::F16 => ptx_cvt_f32_to_f16(value, rounding, relu, satfinite),
        LowPrecisionFormat::Bf16 => ptx_cvt_f32_to_bf16(value, rounding, relu, satfinite),
    }
}

/// `cvt{.irnd}{.ftz}{.sat}.f32.f32`.
///
/// Without `.irnd`/`.ftz`/`.sat` this is the bit-exact identity; `.ftz` alone
/// flushes and canonicalizes NaN; `.irnd` rounds to an integral value.
pub fn cvt_f32_to_f32(value: f32, irnd: Option<PtxIntegerRounding>, ftz: bool, sat: bool) -> f32 {
    let result = match irnd {
        Some(mode) => ptx_cvt_integral_f32_to_f32(value, mode, ftz),
        None if ftz => ptx_cvt_f32_to_f32(value, true),
        None => value,
    };
    if sat {
        ptx_saturate_f32(result)
    } else {
        result
    }
}

/// `cvt{.irnd}{.sat}.f64.f64`.
pub fn cvt_f64_to_f64(value: f64, irnd: Option<PtxIntegerRounding>, sat: bool) -> f64 {
    let result = match irnd {
        Some(mode) => ptx_cvt_integral_f64_to_f64(value, mode),
        None => value,
    };
    if sat {
        ptx_saturate_f64(result)
    } else {
        result
    }
}

/// `cvt{.ftz}{.sat}.f64.f32`.
pub fn cvt_f32_to_f64(value: f32, ftz: bool, sat: bool) -> f64 {
    let result = if ftz {
        ptx_cvt_f32_to_f64(value, true)
    } else {
        f64::from(value)
    };
    if sat {
        ptx_saturate_f64(result)
    } else {
        result
    }
}

/// `cvt.frnd{.ftz}{.sat}.f32.f64`.
pub fn cvt_f64_to_f32(value: f64, rounding: PtxFloatRounding, ftz: bool, sat: bool) -> f32 {
    let result = ptx_cvt_f64_to_f32(value, rounding, ftz);
    if sat {
        ptx_saturate_f32(result)
    } else {
        result
    }
}

/// `cvt{.irnd}{.sat}.f16.f16` / `cvt{.irnd}.bf16.bf16`.
///
/// The plain same-type form still canonicalizes NaN; integral rounding of a
/// widened half stays exactly representable, so encoding cannot round twice.
pub fn cvt_half_to_half(
    bits: u16,
    format: LowPrecisionFormat,
    irnd: Option<PtxIntegerRounding>,
    sat: bool,
) -> u16 {
    let value = reg_decode_half(bits, format);
    match (irnd, sat) {
        (None, false) => {
            ptx_cvt_f32_to_half(value, format, PtxFloatRounding::NearestEven, false, false)
        }
        (None, true) => reg_encode_half(ptx_saturate_f32(value), format),
        (Some(mode), false) => reg_encode_half(ptx_cvt_integral_f32(value, mode, false), format),
        (Some(mode), true) => reg_encode_half(
            ptx_saturate_f32(ptx_cvt_integral_f32(value, mode, false)),
            format,
        ),
    }
}

/// `cvt{.ftz}{.sat}.f32.{f16,bf16}`.
pub fn cvt_half_to_f32(bits: u16, format: LowPrecisionFormat, ftz: bool, sat: bool) -> f32 {
    let value = reg_decode_half(bits, format);
    let value = if ftz {
        ptx_cvt_widen_to_f32(value, true)
    } else {
        value
    };
    if sat {
        ptx_saturate_f32(value)
    } else {
        value
    }
}

/// `cvt{.sat}.f64.{f16,bf16}`.
pub fn cvt_half_to_f64(bits: u16, format: LowPrecisionFormat, sat: bool) -> f64 {
    if sat {
        ptx_saturate_f64(f64::from(reg_decode_half(bits, format)))
    } else {
        ptx_cvt_low_to_f64(bits, format)
    }
}

/// Modifiers of one `cvt.<frnd>.{f16,bf16}.f32` spelling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HalfNarrowing {
    pub ftz: bool,
    pub sat: bool,
    pub relu: bool,
    pub satfinite: bool,
    pub pzo: bool,
}

/// `cvt.frnd{.ftz}{.sat}.{f16,bf16}.f32` and
/// `cvt.frnd2{.relu}{.satfinite}{.pzo}.{f16,bf16}.f32`.
///
/// `.ftz` flushes the input; `.sat` clamps the decoded result to `[0, 1]`;
/// `.relu`/`.satfinite` belong to the narrowing core; `.pzo` clears a
/// negative-zero result after conversion.
pub fn cvt_f32_to_half(
    value: f32,
    format: LowPrecisionFormat,
    rounding: PtxFloatRounding,
    modifiers: HalfNarrowing,
) -> u16 {
    let input = if modifiers.ftz {
        flush_subnormal_f32(value)
    } else {
        value
    };
    let mut result =
        ptx_cvt_f32_to_half(input, format, rounding, modifiers.relu, modifiers.satfinite);
    if modifiers.sat {
        result = reg_encode_half(ptx_saturate_f32(reg_decode_half(result, format)), format);
    }
    if modifiers.pzo {
        result = ptx_cvt_pzo_u16(result);
    }
    result
}

/// `cvt.{rn,rz,rna}{.relu}{.satfinite}{.pzo}.tf32.f32`.
pub fn cvt_f32_to_tf32(
    value: f32,
    rounding: PtxFloatRounding,
    relu: bool,
    satfinite: bool,
    pzo: bool,
) -> u32 {
    let result = ptx_cvt_f32_to_tf32(value, rounding, relu, satfinite);
    if pzo {
        ptx_cvt_pzo_u32(result)
    } else {
        result
    }
}

/// `cvt.frnd{.sat}.{f16,bf16}.f64`: rounds directly from the f64 significand.
pub fn cvt_f64_to_half(
    value: f64,
    format: LowPrecisionFormat,
    rounding: PtxFloatRounding,
    sat: bool,
) -> u16 {
    let result = ptx_cvt_f64_to_low(value, rounding, format);
    if sat {
        reg_encode_half(ptx_saturate_f32(reg_decode_half(result, format)), format)
    } else {
        result
    }
}

/// `cvt.frnd.bf16.f16` / `cvt.frnd.f16.bf16`: the source widens exactly to
/// f32, so the narrowing is the only rounding.
pub fn cvt_half_cross(bits: u16, source: LowPrecisionFormat, rounding: PtxFloatRounding) -> u16 {
    let value = reg_decode_half(bits, source);
    let destination = match source {
        LowPrecisionFormat::F16 => LowPrecisionFormat::Bf16,
        LowPrecisionFormat::Bf16 => LowPrecisionFormat::F16,
    };
    ptx_cvt_f32_to_half(value, destination, rounding, false, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Legacy reg.rs `bf16_narrowing_markers_agree_on_every_nan_encoding`.
    #[test]
    fn bf16_narrowing_markers_agree_on_every_nan_encoding() {
        for bits in [
            0x7fc0_0000_u32,
            0xffc0_0000,
            0x7f80_0001,
            0xff80_0001,
            0x7fff_ffff,
        ] {
            let value = f32::from_bits(bits);
            assert_eq!(reg_encode_bf16(value), 0x7fff, "encode_bf16({bits:#010x})");
            assert_eq!(
                reg_encode_bf16(value),
                ptx_cvt_f32_to_bf16(value, PtxFloatRounding::NearestEven, false, false),
                "bf16 narrowing markers disagree on {bits:#010x}"
            );
        }
    }

    /// The cvt half of legacy reg.rs
    /// `nymph_low_precision_forms_have_ptx_shaped_numeric_semantics`.
    #[test]
    fn f32_f16_round_trip_ties_to_even() {
        let f16 = cvt_f32_to_half(
            f32::from_bits(0x3f80_0400), // 1 + 2^-11, an f16 tie
            LowPrecisionFormat::F16,
            PtxFloatRounding::NearestEven,
            HalfNarrowing::default(),
        );
        assert_eq!(f16, 0x3c00);
        assert_eq!(
            cvt_half_to_f32(f16, LowPrecisionFormat::F16, false, false).to_bits(),
            1.0_f32.to_bits()
        );
    }

    #[test]
    fn same_type_half_canonicalizes_nan_and_saturates() {
        assert_eq!(
            cvt_half_to_half(0x7e01, LowPrecisionFormat::F16, None, false),
            0x7fff
        );
        assert_eq!(
            cvt_half_to_half(0x4100, LowPrecisionFormat::F16, None, true),
            0x3c00
        );
        assert_eq!(
            cvt_half_to_half(
                0x3e00,
                LowPrecisionFormat::F16,
                Some(PtxIntegerRounding::NearestEven),
                false
            ),
            0x4000
        );
    }

    #[test]
    fn pzo_clears_only_negative_zero_results() {
        let rz = PtxFloatRounding::Zero;
        let pzo = HalfNarrowing {
            pzo: true,
            ..HalfNarrowing::default()
        };
        assert_eq!(cvt_f32_to_half(-1e-30, LowPrecisionFormat::F16, rz, pzo), 0);
        assert_eq!(
            cvt_f32_to_half(-1.0, LowPrecisionFormat::Bf16, rz, pzo),
            0xbf80
        );
        assert_eq!(cvt_f32_to_tf32(-1e-45, rz, false, false, true), 0);
    }
}
