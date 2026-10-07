//! Packed narrow-float PTX `cvt` forms as plain functions over runtime flags.
//!
//! Ported from the legacy `engine-rs/src/runtime/instructions/reg.rs` packed
//! cvt section: `PackedNarrowFormat`, `RoundExponentUp`, `SaturateFinite`,
//! `ClampNegative`, `CvtNarrowZero` (`.pzo`), `F32CvtDestination`,
//! `packed_half_destination!`, `packed_narrow_*_variant!`,
//! `packed_half_rs_variant!`, `s2f6_cvt_variant!`, `packed_e8m0_*`,
//! `ptx94_narrow_*`, `ue5m3_unsaturated_variant!`.  The marker axes became
//! `bool`/enum parameters; the numeric bodies are the `cvt::pack`/`cvt::ptx`
//! cores, still selected by const generics internally.

use crate::cvt::float::{ptx_cvt_f32_to_half, reg_decode_bf16, reg_decode_f16};
use crate::cvt::formats::{bf16_bits_to_f32, fp16_bits_to_f32};
use crate::cvt::formats::{
    NarrowFloatFormat, FLOAT4_E2M1, FLOAT6_E2M3, FLOAT6_E3M2, FLOAT8_E4M3, FLOAT8_E5M2,
    FLOAT8_UE5M3,
};
use crate::cvt::pack::*;
use crate::cvt::ptx::{ptx_cvt_f32_to_bf16, ptx_cvt_f32_to_f16};
use crate::cvt::ptx::{ptx_cvt_half_rs, ptx_cvt_pzo_narrow_x2, ptx_cvt_pzo_u16, PtxFloatRounding};
use crate::scalar::LowPrecisionFormat;
use crate::types::{OpError, OpResult};

/// Element format of a packed narrow-float `cvt` operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NarrowKind {
    E4m3,
    E5m2,
    E2m1,
    E2m3,
    E3m2,
    Ue5m3,
}

impl NarrowKind {
    /// The codec format of one element.
    pub const fn format(self) -> NarrowFloatFormat {
        match self {
            Self::E4m3 => FLOAT8_E4M3,
            Self::E5m2 => FLOAT8_E5M2,
            Self::E2m1 => FLOAT4_E2M1,
            Self::E2m3 => FLOAT6_E2M3,
            Self::E3m2 => FLOAT6_E3M2,
            Self::Ue5m3 => FLOAT8_UE5M3,
        }
    }
}

/// Source of a two-element packing conversion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PairSource {
    /// Two `.f32` primaries; `high` lands in the upper field.
    F32 { high: f32, low: f32 },
    /// One `.f16x2` register (upper half is the high element).
    F16x2(u32),
    /// One `.bf16x2` register.
    Bf16x2(u32),
}

impl PairSource {
    /// Exact widening of both elements to `f32` (register codecs).
    pub fn widened(self) -> (f32, f32) {
        match self {
            Self::F32 { high, low } => (high, low),
            Self::F16x2(bits) => (
                reg_decode_f16((bits >> 16) as u16),
                reg_decode_f16(bits as u16),
            ),
            Self::Bf16x2(bits) => (
                reg_decode_bf16((bits >> 16) as u16),
                reg_decode_bf16(bits as u16),
            ),
        }
    }

    /// `.scaled::n1::ue8m0` preprocessing of both elements.
    pub fn scaled_n1(self, scale: u8) -> (f32, f32) {
        match self {
            Self::F32 { high, low } => (
                ptx_cvt_scaled_n1_f32(high, scale),
                ptx_cvt_scaled_n1_f32(low, scale),
            ),
            Self::F16x2(bits) => (
                ptx_cvt_scaled_n1_f16((bits >> 16) as u16, scale),
                ptx_cvt_scaled_n1_f16(bits as u16, scale),
            ),
            Self::Bf16x2(bits) => (
                ptx_cvt_scaled_n1_bf16((bits >> 16) as u16, scale),
                ptx_cvt_scaled_n1_bf16(bits as u16, scale),
            ),
        }
    }
}

/// `cvt.rn.satfinite{.relu}.{e4m3,e5m2,e2m1}x2.<src>` (pre-PTX 9.4 encoder).
pub fn cvt_pack_narrow_x2_rn(source: PairSource, kind: NarrowKind, relu: bool) -> u16 {
    let (high, low) = source.widened();
    if relu {
        ptx_cvt_pack_narrow_x2::<true>(high, low, kind.format())
    } else {
        ptx_cvt_pack_narrow_x2::<false>(high, low, kind.format())
    }
}

/// PTX 9.4 `cvt.{rn,rz,rp}.satfinite{.relu}{.pzo}{.scaled::n1::ue8m0}.<narrow>x2.<src>`.
///
/// `scale_n1` selects the n1-scaled spelling.  `.pzo` (signed formats only)
/// clears each negative-zero field after conversion.
pub fn cvt_pack_narrow_x2_rounded(
    source: PairSource,
    kind: NarrowKind,
    rounding: PtxFloatRounding,
    relu: bool,
    pzo: bool,
    scale_n1: Option<u8>,
) -> u16 {
    let (high, low) = match scale_n1 {
        Some(scale) => source.scaled_n1(scale),
        None => source.widened(),
    };
    let format = kind.format();
    let bits = if relu {
        ptx_cvt_pack_narrow_x2_rounded::<true>(high, low, rounding, format)
    } else {
        ptx_cvt_pack_narrow_x2_rounded::<false>(high, low, rounding, format)
    };
    if pzo {
        ptx_cvt_pzo_narrow_x2(bits, format)
    } else {
        bits
    }
}

/// `cvt.{rn,rz,rp}{.scaled::n1::ue8m0}.ue5m3x2.<src>` without `.satfinite`.
///
/// The scaled spelling pre-divides by unit scale (flushing source subnormals)
/// and hands the real scale to the unsaturated encoder, as legacy did.
pub fn cvt_pack_ue5m3x2_unsaturated(
    source: PairSource,
    rounding: PtxFloatRounding,
    scale_n1: Option<u8>,
) -> u16 {
    let ((high, low), scale) = match scale_n1 {
        Some(scale) => (source.scaled_n1(127), scale),
        None => (source.widened(), 127),
    };
    ptx_cvt_pack_ue5m3x2_unsaturated(high, low, rounding, scale)
}

/// `cvt.rs{.relu}.satfinite.<narrow>x4.f32 d, {a, b, e, f}, rbits`.
pub fn cvt_pack_narrow_x4_rs(values: [f32; 4], rbits: u32, kind: NarrowKind, relu: bool) -> u32 {
    let format = kind.format();
    let randoms = ptx_cvt_rs_randoms(rbits, format);
    if relu {
        ptx_cvt_pack_narrow_x4::<true>(values, randoms, format)
    } else {
        ptx_cvt_pack_narrow_x4::<false>(values, randoms, format)
    }
}

/// `cvt.rn{.relu}.f16x2.<narrow>x2`.
pub fn cvt_unpack_narrow_x2_f16x2(bits: u16, kind: NarrowKind, relu: bool) -> u32 {
    if relu {
        ptx_cvt_unpack_narrow_x2_f16x2::<true>(bits, kind.format())
    } else {
        ptx_cvt_unpack_narrow_x2_f16x2::<false>(bits, kind.format())
    }
}

/// `cvt.rn{.relu}{.satfinite}{.scaled::n2::ue8m0}.bf16x2.<narrow>x2`.
pub fn cvt_unpack_narrow_x2_bf16x2(
    bits: u16,
    kind: NarrowKind,
    relu: bool,
    satfinite: bool,
    scale_n2: Option<u16>,
) -> u32 {
    let format = kind.format();
    match (scale_n2, relu, satfinite) {
        (None, false, false) => ptx_cvt_unpack_narrow_x2_bf16x2::<false, false>(bits, format),
        (None, false, true) => ptx_cvt_unpack_narrow_x2_bf16x2::<false, true>(bits, format),
        (None, true, false) => ptx_cvt_unpack_narrow_x2_bf16x2::<true, false>(bits, format),
        (None, true, true) => ptx_cvt_unpack_narrow_x2_bf16x2::<true, true>(bits, format),
        (Some(s), false, false) => ptx_cvt_unpack_scaled_bf16x2::<false, false>(bits, s, format),
        (Some(s), false, true) => ptx_cvt_unpack_scaled_bf16x2::<false, true>(bits, s, format),
        (Some(s), true, false) => ptx_cvt_unpack_scaled_bf16x2::<true, false>(bits, s, format),
        (Some(s), true, true) => ptx_cvt_unpack_scaled_bf16x2::<true, true>(bits, s, format),
    }
}

/// Default `.s2f6x2` scale (two unit UE8M0 bytes) when no `.scaled` operand.
pub const S2F6_UNIT_SCALE: u16 = 0x7f7f;

/// `cvt.rn.satfinite{.relu}{.scaled::n2::ue8m0}.s2f6x2.{f32,bf16x2}`.
/// `.f16x2` sources do not exist for this destination.
pub fn cvt_pack_s2f6x2(source: PairSource, scale_n2: Option<u16>, relu: bool) -> u16 {
    let (high, low) = source.widened();
    ptx_cvt_pack_s2f6x2(high, low, scale_n2.unwrap_or(S2F6_UNIT_SCALE), relu)
}

/// `cvt.rn{.relu}{.satfinite}{.scaled::n2::ue8m0}.bf16x2.s2f6x2`.
pub fn cvt_unpack_s2f6x2(bits: u16, scale_n2: Option<u16>, relu: bool, satfinite: bool) -> u32 {
    ptx_cvt_unpack_s2f6x2(bits, scale_n2.unwrap_or(S2F6_UNIT_SCALE), relu, satfinite)
}

/// `cvt.{rz,rp}{.satfinite}.ue8m0x2.{f32,bf16x2}` (`round_up` = `.rp`).
pub fn cvt_pack_ue8m0x2(source: PairSource, round_up: bool, satfinite: bool) -> OpResult<u16> {
    Ok(match source {
        PairSource::F32 { high, low } => match (round_up, satfinite) {
            (false, false) => ptx_cvt_pack_e8m0x2_f32::<false, false>(high, low),
            (false, true) => ptx_cvt_pack_e8m0x2_f32::<false, true>(high, low),
            (true, false) => ptx_cvt_pack_e8m0x2_f32::<true, false>(high, low),
            (true, true) => ptx_cvt_pack_e8m0x2_f32::<true, true>(high, low),
        },
        PairSource::Bf16x2(bits) => match (round_up, satfinite) {
            (false, false) => ptx_cvt_pack_e8m0x2_bf16x2::<false, false>(bits),
            (false, true) => ptx_cvt_pack_e8m0x2_bf16x2::<false, true>(bits),
            (true, false) => ptx_cvt_pack_e8m0x2_bf16x2::<true, false>(bits),
            (true, true) => ptx_cvt_pack_e8m0x2_bf16x2::<true, true>(bits),
        },
        PairSource::F16x2(_) => {
            return Err(OpError::message("cvt.ue8m0x2 has no .f16x2 source form"))
        }
    })
}

/// `cvt.rn.bf16x2.ue8m0x2`.
pub fn cvt_unpack_ue8m0x2(bits: u16) -> u32 {
    ptx_cvt_unpack_e8m0x2_bf16x2(bits)
}

/// `cvt.{rn,rz}{.relu}{.satfinite}{.pzo}.{f16x2,bf16x2}.f32 d, a, b`.
pub fn cvt_f32_pair_to_half2(
    high: f32,
    low: f32,
    format: LowPrecisionFormat,
    rounding: PtxFloatRounding,
    relu: bool,
    satfinite: bool,
    pzo: bool,
) -> u32 {
    let convert = |value| {
        let bits = ptx_cvt_f32_to_half(value, format, rounding, relu, satfinite);
        if pzo {
            ptx_cvt_pzo_u16(bits)
        } else {
            bits
        }
    };
    (u32::from(convert(high)) << 16) | u32::from(convert(low))
}

/// `cvt.rs{.relu}{.satfinite}.{f16x2,bf16x2}.f32 d, a, b, rbits`.
///
/// Each element takes its own random halfword (`a` the upper); `.f16x2` uses
/// 13 random bits per element and requires the reserved bits to be zero.
pub fn cvt_f32_pair_to_half2_rs(
    high: f32,
    low: f32,
    rbits: u32,
    format: LowPrecisionFormat,
    relu: bool,
    satfinite: bool,
) -> OpResult<u32> {
    let random_bits = match format {
        LowPrecisionFormat::F16 => 13,
        LowPrecisionFormat::Bf16 => 16,
    };
    let mask = ((1_u64 << random_bits) - 1) as u32;
    if rbits & !(mask | (mask << 16)) != 0 {
        return Err(OpError::message(
            "cvt.rs.f16x2 requires zero reserved rbits",
        ));
    }
    let convert = |value: f32, random: u16| match format {
        LowPrecisionFormat::F16 => ptx_cvt_half_rs::<13>(
            value,
            random,
            relu,
            satfinite,
            ptx_cvt_f32_to_f16,
            fp16_bits_to_f32,
        ),
        LowPrecisionFormat::Bf16 => ptx_cvt_half_rs::<16>(
            value,
            random,
            relu,
            satfinite,
            ptx_cvt_f32_to_bf16,
            bf16_bits_to_f32,
        ),
    };
    Ok((u32::from(convert(high, (rbits >> 16) as u16)) << 16)
        | u32::from(convert(low, rbits as u16)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Legacy reg.rs `ptx94_packed_cvt_specializations_preserve_bits_and_operand_order`.
    #[test]
    fn ptx94_packed_cvt_specializations_preserve_bits_and_operand_order() {
        let e2m3 = cvt_pack_narrow_x2_rounded(
            PairSource::F32 {
                high: 2.125,
                low: 2.375,
            },
            NarrowKind::E2m3,
            PtxFloatRounding::Zero,
            false,
            false,
            Some(128),
        );
        // Divide by two first: RZ maps 1.0625 -> 1.0 (0x08) and
        // 1.1875 -> 1.125 (0x09), in padded upper/lower byte fields.
        assert_eq!(e2m3, 0x0809);
        let ue5 = cvt_pack_narrow_x2_rounded(
            PairSource::F16x2(0x4040_40c0),
            NarrowKind::Ue5m3,
            PtxFloatRounding::Zero,
            false,
            false,
            Some(128),
        );
        assert_eq!(ue5, 0x7879);
        assert_eq!(
            cvt_unpack_narrow_x2_f16x2(0x78fe, NarrowKind::Ue5m3, false),
            0x3c00_7c00
        );
    }

    #[test]
    fn half_rs_rejects_reserved_bits_for_f16_only() {
        let f16 = LowPrecisionFormat::F16;
        assert!(cvt_f32_pair_to_half2_rs(1.0, 1.0, 0x0000_2000, f16, false, false).is_err());
        assert!(cvt_f32_pair_to_half2_rs(
            1.0,
            1.0,
            0xffff_ffff,
            LowPrecisionFormat::Bf16,
            false,
            false
        )
        .is_ok());
        assert_eq!(
            cvt_f32_pair_to_half2_rs(2.0, 1.0, 0, f16, false, false).unwrap(),
            0x4000_3c00
        );
    }

    #[test]
    fn packed_half_pair_places_a_in_the_upper_half() {
        let packed = cvt_f32_pair_to_half2(
            2.0,
            1.0,
            LowPrecisionFormat::F16,
            PtxFloatRounding::NearestEven,
            false,
            false,
            false,
        );
        assert_eq!(packed, 0x4000_3c00);
    }
}
