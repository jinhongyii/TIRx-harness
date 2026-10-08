//! Packed-lane moves, cvt.pack forms, narrow-float pack/unpack, packed f32x2 and half min/max.
#![allow(unused_imports)]
use crate::cvt::formats::*;
use crate::cvt::*;
use crate::scalar::*;
use crate::types::OpError;
use std::cmp::Ordering;
/// Pack two binary32 values bit-exactly into a `float2` b64 (`x` in bits 0..32); no numerics.
pub fn make_float2(x: f32, y: f32) -> u64 {
    x.to_bits() as u64 | ((y.to_bits() as u64) << 32)
}

/// Low (`x`) binary32 lane of a packed `float2`, bit-exact (NaN payload kept).
pub fn float2_x(value: u64) -> f32 {
    f32::from_bits(value as u32)
}

/// High (`y`) binary32 lane of a packed `float2`, bit-exact (NaN payload kept).
pub fn float2_y(value: u64) -> f32 {
    f32::from_bits((value >> 32) as u32)
}

/// PTX vector `mov.b64`: the first b16 lane occupies the least-significant bits.
pub fn ptx_mov_pack_b16x4(values: [u16; 4]) -> u64 {
    values
        .into_iter()
        .enumerate()
        .fold(0_u64, |packed, (lane, value)| {
            packed | (u64::from(value) << (lane * 16))
        })
}

/// Inverse of [`ptx_mov_pack_b16x4`] in PTX register-group order.
pub fn ptx_mov_unpack_b16x4(value: u64) -> [u16; 4] {
    std::array::from_fn(|lane| (value >> (lane * 16)) as u16)
}

/// PTX vector `mov.b128`: four b32 lanes ordered from least to most significant.
pub fn ptx_mov_pack_b32x4(values: [u32; 4]) -> U64x2 {
    [
        u64::from(values[0]) | (u64::from(values[1]) << 32),
        u64::from(values[2]) | (u64::from(values[3]) << 32),
    ]
}

/// Inverse of [`ptx_mov_pack_b32x4`] in PTX register-group order.
pub fn ptx_mov_unpack_b32x4(value: U64x2) -> [u32; 4] {
    [
        value[0] as u32,
        (value[0] >> 32) as u32,
        value[1] as u32,
        (value[1] >> 32) as u32,
    ]
}

/// PTX vector `mov.b128` split into its low then high b64 lane.
pub fn ptx_mov_unpack_b64x2(value: U64x2) -> [u64; 2] {
    value
}

/// PTX `cvt.pack.sat`: saturate two s32 values and pack `b` below `a`.
///
/// For 2/4/8-bit fields, the remaining high bits come from the low bits of
/// `c`. The closed v2 variants instantiate only the four widths admitted by
/// the ISA, with `c = 0` for the 16-bit syntax line that has no c operand.
pub fn ptx_cvt_pack<const BITS: u32, const SIGNED: bool>(a: i32, b: i32, c: u32) -> u32 {
    debug_assert!(matches!(BITS, 2 | 4 | 8 | 16));
    let (minimum, maximum) = if SIGNED {
        (-(1_i32 << (BITS - 1)), (1_i32 << (BITS - 1)) - 1)
    } else {
        (0, ((1_u32 << BITS) - 1) as i32)
    };
    let mask = (1_u32 << BITS) - 1;
    let field = |value: i32| value.clamp(minimum, maximum) as u32 & mask;
    let packed = field(b) | (field(a) << BITS);
    if BITS == 16 {
        packed
    } else {
        packed | (c << (2 * BITS))
    }
}

/// `cvt.rn.bf16x2.f32`-style pack: each f32 to bf16 with RN-even (overflow to inf,
/// subnormals kept); any NaN becomes canonical `0x7fff`. `lhs` goes in bits 0..16.
pub fn pack_bf16x2(lhs: f32, rhs: f32) -> u32 {
    let encode = |value: f32| {
        if value.is_nan() {
            0x7fff_u16
        } else {
            f32_to_bf16_bits(value)
        }
    };
    encode(lhs) as u32 | ((encode(rhs) as u32) << 16)
}

/// Exact widening of a bf16x2 word into a `float2` (low half -> `x`); NaN payloads kept.
pub fn unpack_bf16x2(value: u32) -> u64 {
    make_float2(
        bf16_bits_to_f32(value as u16),
        bf16_bits_to_f32((value >> 16) as u16),
    )
}

/// `.relu` clamps a negative operand — including negative zero — to `+0` before
/// conversion; it deliberately leaves NaN alone, which is the measured hardware
/// behaviour rather than a "NaN is not positive" reading.
pub(crate) fn clamp_negative<const CLAMP: bool>(value: f32) -> f32 {
    if CLAMP && value.is_sign_negative() && !value.is_nan() {
        0.0_f32
    } else {
        value
    }
}

/// PTX `cvt.rn.satfinite{.relu}.<narrow>x2.{f32,f16x2,bf16x2}` element packing.
///
/// `high` becomes the upper element and `low` the lower one, as the ISA
/// specifies for both the two-`f32` and the one-packed-source spellings.
pub fn ptx_cvt_pack_narrow_x2<const CLAMP_NEGATIVE: bool>(
    high: f32,
    low: f32,
    format: NarrowFloatFormat,
) -> u16 {
    let encode = |value: f32| {
        f32_to_narrow_float_bits_rn_satfinite(clamp_negative::<CLAMP_NEGATIVE>(value), format)
    };
    (u16::from(encode(high)) << format.storage_bits) | u16::from(encode(low))
}

/// PTX `cvt.{rn,rz,rp}.satfinite{.relu}.<narrow>x2` element packing.
///
/// The unsigned UE5M3 destination converts a source's magnitude, matching the
/// ISA's unsigned narrow-float encoding.  Directed modes are derived from the
/// reviewed nearest-even encoder and the exactly decoded adjacent code, so
/// normal/subnormal boundaries and ties have one canonical implementation.
pub fn ptx_cvt_pack_narrow_x2_rounded<const CLAMP_NEGATIVE: bool>(
    high: f32,
    low: f32,
    rounding: PtxFloatRounding,
    format: NarrowFloatFormat,
) -> u16 {
    let encode = |value: f32| {
        ptx_cvt_narrow_bits_satfinite(clamp_negative::<CLAMP_NEGATIVE>(value), rounding, format)
    };
    (u16::from(encode(high)) << format.storage_bits) | u16::from(encode(low))
}

/// UE5M3 without saturation, following CUDA 13.4's `cuda_fp8` contract.
/// Sign is discarded before rounding. NaN/Inf become NaN; finite overflow
/// rounds to MAX_NORM for RZ, or to NaN for RN/RP when rounding crosses MAX_NORM.
/// `scale` is the UE8M0 divisor (127 means one). Scaled callers first flush
/// source subnormals using their source format's existing n1 preprocessing.
pub fn ptx_cvt_pack_ue5m3x2_unsaturated(
    high: f32,
    low: f32,
    rounding: PtxFloatRounding,
    scale: u8,
) -> u16 {
    let format = FLOAT8_UE5M3;
    let decode = |code| f64::from(narrow_float_bits_to_f32_checked(code, format).unwrap());
    let max = decode(format.max_finite_code);
    let midpoint = max + (max - decode(format.max_finite_code - 1)) / 2.0;
    let divisor = f64::from(float8_e8m0fnu_bits_to_f32(scale));
    let encode = |value: f32| {
        // Power-of-two scaling is exact in f64. Do not turn finite scaled
        // overflow into an artificial f32 infinity: RZ must still produce MAX.
        let magnitude = f64::from(value).abs() / divisor;
        if !magnitude.is_finite()
            || matches!(rounding, PtxFloatRounding::PositiveInfinity) && magnitude > max
            || matches!(rounding, PtxFloatRounding::NearestEven) && magnitude > midpoint
        {
            format.nan_code
        } else if magnitude > max {
            // The highest finite code is even, so RN's exact midpoint stays here.
            format.max_finite_code
        } else {
            ptx_cvt_narrow_bits_satfinite(magnitude as f32, rounding, format)
        }
    };
    (u16::from(encode(high)) << format.storage_bits) | u16::from(encode(low))
}

/// Apply PTX `.scaled::n1::ue8m0` preprocessing to one binary32 primary.
///
/// The n1 grammar uses one scale for both destination elements and flushes a
/// source subnormal to positive zero before division by that power of two.
pub fn ptx_cvt_scaled_n1_f32(value: f32, scale: u8) -> f32 {
    let value = if value.is_subnormal() { 0.0 } else { value };
    value / float8_e8m0fnu_bits_to_f32(scale)
}

/// Apply PTX `.scaled::n1::ue8m0` preprocessing to one binary16 primary.
pub fn ptx_cvt_scaled_n1_f16(bits: u16, scale: u8) -> f32 {
    let flushed = if is_subnormal_f16_bits(bits) { 0 } else { bits };
    fp16_bits_to_f32(flushed) / float8_e8m0fnu_bits_to_f32(scale)
}

/// Apply PTX `.scaled::n1::ue8m0` preprocessing to one bfloat16 primary.
pub fn ptx_cvt_scaled_n1_bf16(bits: u16, scale: u8) -> f32 {
    let flushed = if bits & 0x7f80 == 0 && bits & 0x007f != 0 {
        0
    } else {
        bits
    };
    bf16_bits_to_f32(flushed) / float8_e8m0fnu_bits_to_f32(scale)
}

pub(crate) fn ptx_cvt_narrow_bits_satfinite(
    value: f32,
    rounding: PtxFloatRounding,
    format: NarrowFloatFormat,
) -> u8 {
    let nearest = f32_to_narrow_float_bits_rn_satfinite(value, format);
    if matches!(rounding, PtxFloatRounding::NearestEven) || !value.is_finite() || value == 0.0 {
        return nearest;
    }

    let sign_mask = format.sign_mask();
    let sign = nearest & sign_mask;
    let nearest_magnitude = nearest & !sign_mask;
    let magnitude = value.abs();
    let nearest_value = narrow_float_bits_to_f32_checked(nearest_magnitude, format)
        .expect("satfinite narrow conversion never selects a NaN code");
    let toward_zero = if nearest_value > magnitude {
        nearest_magnitude.saturating_sub(1)
    } else {
        nearest_magnitude
    };

    match rounding {
        PtxFloatRounding::Zero => sign | toward_zero,
        PtxFloatRounding::PositiveInfinity if format.signed && value.is_sign_negative() => {
            sign | toward_zero
        }
        PtxFloatRounding::PositiveInfinity => {
            let truncated = narrow_float_bits_to_f32_checked(toward_zero, format)
                .expect("a finite magnitude's predecessor is finite");
            let upward = if truncated < magnitude {
                toward_zero.saturating_add(1).min(format.max_finite_code)
            } else {
                toward_zero
            };
            sign | upward
        }
        PtxFloatRounding::NearestEven => nearest,
        PtxFloatRounding::NearestAway | PtxFloatRounding::NegativeInfinity => {
            unreachable!("unsupported packed narrow-float rounding mode")
        }
    }
}

/// PTX `cvt.rs{.relu}.satfinite.<narrow>x4.f32` element packing.
///
/// `values` are the four primaries in PTX operand order, the first in the most
/// significant field, each paired with its own sixteen stochastic-rounding bits
/// from [`ptx_cvt_rs_randoms`].
pub fn ptx_cvt_pack_narrow_x4<const CLAMP_NEGATIVE: bool>(
    values: [f32; 4],
    randoms: [u16; 4],
    format: NarrowFloatFormat,
) -> u32 {
    let mut packed = 0_u32;
    for (index, (value, random)) in values.into_iter().zip(randoms).enumerate() {
        let code =
            f32_to_narrow_float_bits_rs(clamp_negative::<CLAMP_NEGATIVE>(value), random, format);
        packed |= u32::from(code) << (format.storage_bits * (3 - index as u32));
    }
    packed
}

/// Split one `rbits` operand into the four elements' sixteen random bits.
///
/// Established on an NVIDIA B200: each *pair* of primaries shares one sixteen
/// bit field, the first primary of the pair reading it bit-reversed and the
/// second as written. The eight-bit destinations take the two contiguous
/// halfwords — `(a, b)` from `rbits[31:16]` and `(e, f)` from `rbits[15:0]`.
/// `.e2m1x4`, whose result is itself only sixteen bits wide, instead gathers
/// each field from one byte of each halfword: `(a, b)` from bytes 3 and 1,
/// `(e, f)` from bytes 2 and 0.
pub fn ptx_cvt_rs_randoms(rbits: u32, format: NarrowFloatFormat) -> [u16; 4] {
    let (high, low) = if format.storage_bits == 8 {
        ((rbits >> 16) as u16, rbits as u16)
    } else {
        (
            ((rbits >> 8) & 0xff) as u16 | ((((rbits >> 24) & 0xff) as u16) << 8),
            (rbits & 0xff) as u16 | ((((rbits >> 16) & 0xff) as u16) << 8),
        )
    };
    [high.reverse_bits(), high, low.reverse_bits(), low]
}

/// PTX `cvt.rn{.relu}.f16x2.<narrow>x2`.
///
/// Every NaN encoding widens to the canonical `0x7fff` payload with the sign
/// dropped, and `.relu` does not zero it.
pub fn ptx_cvt_unpack_narrow_x2_f16x2<const CLAMP_NEGATIVE: bool>(
    value: u16,
    format: NarrowFloatFormat,
) -> u32 {
    let convert = |code: u8| -> u16 {
        match narrow_float_bits_to_f32_checked(code, format) {
            None => WIDE_CANONICAL_NAN,
            Some(decoded) if CLAMP_NEGATIVE && decoded.is_sign_negative() => 0,
            Some(decoded) => f32_to_fp16_bits(decoded),
        }
    };
    let (high, low) = split_packed_pair(value, format);
    (u32::from(convert(high)) << 16) | u32::from(convert(low))
}

/// PTX `cvt.rn{.relu}{.satfinite}.bf16x2.<narrow>x2`.
///
/// `.satfinite` is observable only for `.e5m2x2`, whose infinities become the
/// signed greatest finite bfloat16; `.e4m3x2` and `.e2m1x2` have no infinities,
/// so the qualifier is numerically inert there.
pub fn ptx_cvt_unpack_narrow_x2_bf16x2<const CLAMP_NEGATIVE: bool, const SATURATE_FINITE: bool>(
    value: u16,
    format: NarrowFloatFormat,
) -> u32 {
    let convert = |code: u8| -> u16 {
        match narrow_float_bits_to_f32_checked(code, format) {
            None => WIDE_CANONICAL_NAN,
            Some(decoded) if CLAMP_NEGATIVE && decoded.is_sign_negative() => 0,
            Some(decoded) if SATURATE_FINITE && decoded.is_infinite() => {
                saturate_bf16(decoded.is_sign_negative())
            }
            Some(decoded) => f32_to_bf16_bits(decoded),
        }
    };
    let (high, low) = split_packed_pair(value, format);
    (u32::from(convert(high)) << 16) | u32::from(convert(low))
}

/// PTX `cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.<narrow>x2`.
///
/// The scale operand is a `ue8m0x2`: each result half is its own element
/// multiplied by the power of two named by the *corresponding* scale byte.
/// A `0xff` scale byte is the E8M0 NaN and makes that half a NaN whatever the
/// element is; `.relu` does not zero it. `.satfinite` clamps a half that
/// *rounds* to an infinity, so it also catches a finite product that overflows
/// bfloat16, not only an infinite element.
///
/// The product is formed in binary32 and narrowed once. That is exact, not a
/// convenience: an element carries at most four significant bits and the
/// factor is an exact power of two in `2^-127 ..= 2^127`. Every product within
/// binary32's range is exact; a product outside it has already overflowed
/// bfloat16. `every_reachable_scaled_product_is_exact_or_already_overflows_bf16`
/// pins that over all 197625 reachable pairs.
pub fn ptx_cvt_unpack_scaled_bf16x2<const CLAMP_NEGATIVE: bool, const SATURATE_FINITE: bool>(
    value: u16,
    scale: u16,
    format: NarrowFloatFormat,
) -> u32 {
    let convert = |code: u8, scale_code: u8| -> u16 {
        let Some(decoded) = narrow_float_bits_to_f32_checked(code, format) else {
            return WIDE_CANONICAL_NAN;
        };
        if scale_code == E8M0_NAN_CODE {
            return WIDE_CANONICAL_NAN;
        }
        let scaled = decoded * float8_e8m0fnu_bits_to_f32(scale_code);
        if CLAMP_NEGATIVE && scaled.is_sign_negative() {
            return 0;
        }
        let rounded = f32_to_bf16_bits(scaled);
        if SATURATE_FINITE && rounded & 0x7fff == BF16_INFINITY {
            return saturate_bf16(rounded & 0x8000 != 0);
        }
        rounded
    };
    let (high, low) = split_packed_pair(value, format);
    (u32::from(convert(high, (scale >> 8) as u8)) << 16) | u32::from(convert(low, scale as u8))
}

/// S2F6 is a signed two's-complement byte in units of 1/64, not FP6.
/// Each input has its own UE8M0 scale; NaN maps to positive MAX_NORM.
pub fn ptx_cvt_pack_s2f6x2(high: f32, low: f32, scale: u16, relu: bool) -> u16 {
    let convert = |value: f32, scale: u8| {
        let value = value / float8_e8m0fnu_bits_to_f32(scale);
        if value.is_nan() {
            return 127_u8;
        }
        let value = if relu { value.max(0.0) } else { value };
        // Power-of-two scaling is exact near every fixed-point midpoint;
        // values overflowing binary32 already require fixed-point saturation.
        (value * 64.0).round_ties_even().clamp(-128.0, 127.0) as i8 as u8
    };
    (u16::from(convert(high, (scale >> 8) as u8)) << 8) | u16::from(convert(low, scale as u8))
}

/// Decode each signed byte exactly, scale it, then use the shared BF16 codec.
pub fn ptx_cvt_unpack_s2f6x2(value: u16, scale: u16, relu: bool, satfinite: bool) -> u32 {
    let convert = |code: u8, scale: u8| {
        let decoded = f32::from(code as i8) / 64.0;
        let scaled = decoded * float8_e8m0fnu_bits_to_f32(scale);
        ptx_cvt_f32_to_bf16(scaled, PtxFloatRounding::NearestEven, relu, satfinite)
    };
    (u32::from(convert((value >> 8) as u8, (scale >> 8) as u8)) << 16)
        | u32::from(convert(value as u8, scale as u8))
}

/// PTX `cvt.{rz,rp}{.satfinite}.ue8m0x2.f32`, one exponent per primary.
pub fn ptx_cvt_pack_e8m0x2_f32<const ROUND_UP: bool, const SATURATE: bool>(
    high: f32,
    low: f32,
) -> u16 {
    let encode = f32_to_float8_e8m0fnu_bits_rounded::<ROUND_UP, SATURATE>;
    (u16::from(encode(high)) << 8) | u16::from(encode(low))
}

/// PTX `cvt.{rz,rp}{.satfinite}.ue8m0x2.bf16x2`.
///
/// Widening bfloat16 to binary32 is exact, subnormals included, so this reuses
/// the binary32 exponent encoder rather than repeating it.
pub fn ptx_cvt_pack_e8m0x2_bf16x2<const ROUND_UP: bool, const SATURATE: bool>(value: u32) -> u16 {
    ptx_cvt_pack_e8m0x2_f32::<ROUND_UP, SATURATE>(
        bf16_bits_to_f32((value >> 16) as u16),
        bf16_bits_to_f32(value as u16),
    )
}

/// PTX `cvt.rn.bf16x2.ue8m0x2`, widening each exponent to a bfloat16.
///
/// Every finite decode is an exact power of two in `2^-127 ..= 2^127`, so the
/// binary32 intermediate is exact and the narrowing rounds once — the least of
/// them lands mid-subnormal in bfloat16 and still converts exactly.
pub fn ptx_cvt_unpack_e8m0x2_bf16x2(value: u16) -> u32 {
    let convert = |code: u8| -> u16 {
        if code == E8M0_NAN_CODE {
            WIDE_CANONICAL_NAN
        } else {
            f32_to_bf16_bits(float8_e8m0fnu_bits_to_f32(code))
        }
    };
    (u32::from(convert((value >> 8) as u8)) << 16) | u32::from(convert(value as u8))
}

/// Split one packed pair into its upper and lower element codes.
pub(crate) fn split_packed_pair(value: u16, format: NarrowFloatFormat) -> (u8, u8) {
    let mask = (1_u16 << format.width_bits) - 1;
    (
        ((value >> format.storage_bits) & mask) as u8,
        (value & mask) as u8,
    )
}

pub(crate) fn saturate_bf16(negative: bool) -> u16 {
    if negative {
        0x8000 | BF16_MAX_FINITE
    } else {
        BF16_MAX_FINITE
    }
}

/// The binary16 and bfloat16 payload every narrow-float NaN widens to.
pub(crate) const WIDE_CANONICAL_NAN: u16 = 0x7fff;

/// Greatest finite bfloat16 magnitude, without the sign bit.
pub(crate) const BF16_MAX_FINITE: u16 = 0x7f7f;

/// bfloat16 infinity, without the sign bit.
pub(crate) const BF16_INFINITY: u16 = 0x7f80;

/// The single E8M0 NaN encoding.
pub(crate) const E8M0_NAN_CODE: u8 = 0xff;

/// PTX `min.bf16x2` per half: selects an existing operand by ordered encoding
/// (`-0 < +0`, no rounding, subnormals kept); one NaN loses, two NaNs give `0x7fff`.
pub fn hmin2_bf16(lhs: u32, rhs: u32) -> u32 {
    low_minmax2(lhs, rhs, LowPrecisionFormat::Bf16, false)
}

/// PTX `min.f16x2` per half, no FTZ; NaN handling as in [`hmin2_bf16`].
pub fn hmin2_f16(lhs: u32, rhs: u32) -> u32 {
    low_minmax2(lhs, rhs, LowPrecisionFormat::F16, false)
}

/// PTX `max.bf16x2` per half (`+0 > -0`); NaN handling as in [`hmin2_bf16`].
pub fn hmax2_bf16(lhs: u32, rhs: u32) -> u32 {
    low_minmax2(lhs, rhs, LowPrecisionFormat::Bf16, true)
}

/// PTX `max.f16x2` per half, no FTZ; NaN handling as in [`hmin2_bf16`].
pub fn hmax2_f16(lhs: u32, rhs: u32) -> u32 {
    low_minmax2(lhs, rhs, LowPrecisionFormat::F16, true)
}

pub(crate) fn low_minmax2(lhs: u32, rhs: u32, format: LowPrecisionFormat, maximum: bool) -> u32 {
    let component = |shift| {
        u32::from(low_minmax(
            (lhs >> shift) as u16,
            (rhs >> shift) as u16,
            format,
            false,
            false,
            false,
            maximum,
        ))
    };
    component(0) | (component(16) << 16)
}

/// Min/max selects an existing half value; comparing ordered encodings avoids
/// host floating-point conversion or rounding, including for BF16 subnormals.
pub fn low_minmax(
    lhs: u16,
    rhs: u16,
    format: LowPrecisionFormat,
    ftz: bool,
    propagate_nan: bool,
    xor_sign: bool,
    maximum: bool,
) -> u16 {
    let sign = (lhs ^ rhs) & 0x8000;
    let prepare = |bits| {
        let bits = if ftz {
            format.flush_subnormal(bits)
        } else {
            bits
        };
        if xor_sign {
            bits & 0x7fff
        } else {
            bits
        }
    };
    let (lhs, rhs) = (prepare(lhs), prepare(rhs));
    let (lhs_nan, rhs_nan) = (
        lhs & 0x7fff > format.infinity(),
        rhs & 0x7fff > format.infinity(),
    );
    if (lhs_nan && rhs_nan) || (propagate_nan && (lhs_nan || rhs_nan)) {
        return WIDE_CANONICAL_NAN;
    }
    let order = |bits: u16| {
        if bits & 0x8000 != 0 {
            !bits
        } else {
            bits ^ 0x8000
        }
    };
    // A NaN operand loses to the other (rhs when both are NaN).
    let result = if lhs_nan || (!rhs_nan && (order(lhs) > order(rhs)) != maximum) {
        rhs
    } else {
        lhs
    };
    if xor_sign {
        (result & 0x7fff) | sign
    } else {
        result
    }
}

/// Four f32 to E4M3FN bytes ([`f32_to_float8_e4m3fn_bits`]: RN-even, satfinite,
/// NaN to `0x7f`, inf to signed max finite); `x` in bits 0..8.
pub fn fp8x4_e4m3_from_float4(x: f32, y: f32, z: f32, w: f32) -> u32 {
    f32_to_float8_e4m3fn_bits(x) as u32
        | ((f32_to_float8_e4m3fn_bits(y) as u32) << 8)
        | ((f32_to_float8_e4m3fn_bits(z) as u32) << 16)
        | ((f32_to_float8_e4m3fn_bits(w) as u32) << 24)
}

pub(crate) fn pack_f32x2(low: f32, high: f32) -> u64 {
    u64::from(low.to_bits()) | (u64::from(high.to_bits()) << 32)
}

pub(crate) fn pack_f32x2_result(low: f32, high: f32) -> u64 {
    pack_f32x2(
        cuda_canonicalize_nan_f32(low),
        cuda_canonicalize_nan_f32(high),
    )
}

/// PTX `add{.rnd}{.ftz}.f32x2`: [`add_f32`]/[`add_f32_ftz`] per lane (low = bits
/// 0..32); any NaN lane result becomes the canonical `0x7fff_ffff`.
pub fn add_f32x2(lhs: u64, rhs: u64, mode: F32RoundingMode, ftz: bool) -> u64 {
    let lhs_x = f32::from_bits(lhs as u32);
    let lhs_y = f32::from_bits((lhs >> 32) as u32);
    let rhs_x = f32::from_bits(rhs as u32);
    let rhs_y = f32::from_bits((rhs >> 32) as u32);
    let operation = if ftz { add_f32_ftz } else { add_f32 };
    pack_f32x2_result(operation(lhs_x, rhs_x, mode), operation(lhs_y, rhs_y, mode))
}

/// PTX `fma{.rnd}{.ftz}.f32x2`: [`fma_f32`]/[`fma_f32_ftz`] per lane; NaN lanes canonical `0x7fff_ffff`.
pub fn fma_f32x2(lhs: u64, rhs: u64, addend: u64, mode: F32RoundingMode, ftz: bool) -> u64 {
    let lhs_x = f32::from_bits(lhs as u32);
    let lhs_y = f32::from_bits((lhs >> 32) as u32);
    let rhs_x = f32::from_bits(rhs as u32);
    let rhs_y = f32::from_bits((rhs >> 32) as u32);
    let addend_x = f32::from_bits(addend as u32);
    let addend_y = f32::from_bits((addend >> 32) as u32);
    let operation = if ftz { fma_f32_ftz } else { fma_f32 };
    pack_f32x2_result(
        operation(lhs_x, rhs_x, addend_x, mode),
        operation(lhs_y, rhs_y, addend_y, mode),
    )
}

/// PTX `sub{.rnd}{.ftz}.f32x2`: [`sub_f32`]/[`sub_f32_ftz`] per lane; NaN lanes canonical `0x7fff_ffff`.
pub fn sub_f32x2(lhs: u64, rhs: u64, mode: F32RoundingMode, ftz: bool) -> u64 {
    let lhs_low = f32::from_bits(lhs as u32);
    let lhs_high = f32::from_bits((lhs >> 32) as u32);
    let rhs_low = f32::from_bits(rhs as u32);
    let rhs_high = f32::from_bits((rhs >> 32) as u32);
    let operation = if ftz { sub_f32_ftz } else { sub_f32 };
    pack_f32x2_result(
        operation(lhs_low, rhs_low, mode),
        operation(lhs_high, rhs_high, mode),
    )
}

/// PTX `mul{.rnd}{.ftz}.f32x2`: [`mul_f32`]/[`mul_f32_ftz`] per lane; NaN lanes canonical `0x7fff_ffff`.
pub fn mul_f32x2(lhs: u64, rhs: u64, mode: F32RoundingMode, ftz: bool) -> u64 {
    let lhs_low = f32::from_bits(lhs as u32);
    let lhs_high = f32::from_bits((lhs >> 32) as u32);
    let rhs_low = f32::from_bits(rhs as u32);
    let rhs_high = f32::from_bits((rhs >> 32) as u32);
    let operation = if ftz { mul_f32_ftz } else { mul_f32 };
    pack_f32x2_result(
        operation(lhs_low, rhs_low, mode),
        operation(lhs_high, rhs_high, mode),
    )
}
