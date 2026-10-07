//! Scalar PTX `cvt` numeric cores (integer/float rounding, narrowing, tf32).
#![allow(unused_imports)]
use crate::cvt::formats::*;
use crate::cvt::*;
use crate::scalar::*;
use crate::types::OpError;
use std::cmp::Ordering;
// ---------------------------------------------------------------------------
// Scalar PTX `cvt` numeric cores.
//
// Rules follow the PTX ISA section "Data Movement and Conversion Instructions:
// cvt". Comments distinguish measured NaN/FTZ behavior where the specification
// leaves representation details open; compiler defects are not ISA semantics.
// ---------------------------------------------------------------------------

/// PTX `.irnd` integer rounding modifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PtxIntegerRounding {
    /// `.rni` — nearest integer, ties to even.
    NearestEven,
    /// `.rzi` — nearest integer toward zero.
    Zero,
    /// `.rmi` — nearest integer toward negative infinity.
    NegativeInfinity,
    /// `.rpi` — nearest integer toward positive infinity.
    PositiveInfinity,
}

/// PTX `.frnd` / `.frnd2` floating-point rounding modifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PtxFloatRounding {
    /// `.rn` — nearest, ties to even.
    NearestEven,
    /// `.rna` — nearest, ties away from zero.  Only `cvt.rna.tf32.f32` spells it.
    NearestAway,
    /// `.rz` — toward zero.
    Zero,
    /// `.rm` — toward negative infinity.
    NegativeInfinity,
    /// `.rp` — toward positive infinity.
    PositiveInfinity,
}

/// The `f32` NaN every measured `cvt` form that canonicalizes produces.
///
/// This is the same all-ones payload `cuda_canonical_nan_f32` already uses.
pub(crate) const PTX_CVT_CANONICAL_NAN_F32: u32 = 0x7fff_ffff;

/// The `.f16` / `.bf16` NaN every measured narrowing `cvt` form produces.
///
/// Hardware drops the sign and the payload: `cvt.rn.bf16.f32`,
/// `cvt.rz.bf16.f32`, and the `.f16` forms all answer `0x7fff` for `+NaN`,
/// `-NaN` and signalling NaN, with or without `.relu` / `.satfinite`.
pub(crate) const PTX_CVT_CANONICAL_NAN_NARROW: u16 = 0x7fff;

/// One tf32 unit in the last place, in `f32` bit positions.
pub(crate) const TF32_ULP: u32 = 0x2000;

/// Round `value` to an integral `f32`, applying `.ftz` to the source first.
///
/// `.ftz` is observable here: without it `cvt.rmi.f32.f32` of the smallest
/// negative subnormal answers `-1.0` and `cvt.rpi.f32.f32` of the smallest
/// positive subnormal answers `1.0`; with it both answer signed zero.
pub fn ptx_cvt_integral_f32(value: f32, rounding: PtxIntegerRounding, ftz: bool) -> f32 {
    let value = if ftz {
        flush_subnormal_f32(value)
    } else {
        value
    };
    match rounding {
        PtxIntegerRounding::NearestEven => value.round_ties_even(),
        PtxIntegerRounding::Zero => value.trunc(),
        PtxIntegerRounding::NegativeInfinity => value.floor(),
        PtxIntegerRounding::PositiveInfinity => value.ceil(),
    }
}

/// `cvt.irnd{.ftz}.f32.f32`: integral rounding inside `f32`.
///
/// Measured: every NaN input answers `0x7fffffff`, with and without `.ftz`.
pub fn ptx_cvt_integral_f32_to_f32(value: f32, rounding: PtxIntegerRounding, ftz: bool) -> f32 {
    if value.is_nan() {
        return f32::from_bits(PTX_CVT_CANONICAL_NAN_F32);
    }
    ptx_cvt_integral_f32(value, rounding, ftz)
}

/// Round `value` to an integral `f64`.  `.ftz` is illegal without an `f32`.
pub fn ptx_cvt_integral_f64(value: f64, rounding: PtxIntegerRounding) -> f64 {
    match rounding {
        PtxIntegerRounding::NearestEven => value.round_ties_even(),
        PtxIntegerRounding::Zero => value.trunc(),
        PtxIntegerRounding::NegativeInfinity => value.floor(),
        PtxIntegerRounding::PositiveInfinity => value.ceil(),
    }
}

/// `cvt.irnd.f64.f64`: integral rounding inside `f64`.
///
/// Measured: unlike the `f32` form this preserves the NaN sign and payload and
/// only quiets a signalling NaN.
pub fn ptx_cvt_integral_f64_to_f64(value: f64, rounding: PtxIntegerRounding) -> f64 {
    if value.is_nan() {
        return f64::from_bits(value.to_bits() | 0x0008_0000_0000_0000);
    }
    ptx_cvt_integral_f64(value, rounding)
}

/// `cvt{.ftz}.f32.f32`: the identity conversion, which `.ftz` makes observable.
///
/// Measured: `.ftz` flushes a subnormal source to signed zero *and*
/// canonicalizes NaN; without it the value passes through bit-for-bit.
pub fn ptx_cvt_f32_to_f32(value: f32, ftz: bool) -> f32 {
    if !ftz {
        return value;
    }
    if value.is_nan() {
        return f32::from_bits(PTX_CVT_CANONICAL_NAN_F32);
    }
    flush_subnormal_f32(value)
}

/// Round the exact integer `magnitude` to `mantissa_bits` of significand.
///
/// Returns `(head, exponent)` with `magnitude ~ head << exponent` and
/// `head < 1 << mantissa_bits`, so the caller can rebuild the value exactly.
/// `away_from_zero` selects between truncation and the next magnitude up,
/// which is how `.rz` / `.rm` / `.rp` differ for `u64`/`s64` sources whose
/// value needs more bits than the destination format has.
pub(crate) fn round_integer_magnitude(magnitude: u64, mantissa_bits: u32, away_from_zero: bool) -> (u64, u32) {
    if magnitude == 0 {
        return (0, 0);
    }
    let significant = u64::BITS - magnitude.leading_zeros();
    if significant <= mantissa_bits {
        return (magnitude, 0);
    }
    let mut exponent = significant - mantissa_bits;
    let mut head = magnitude >> exponent;
    if away_from_zero && magnitude & ((1u64 << exponent) - 1) != 0 {
        head += 1;
        if head == 1u64 << mantissa_bits {
            head >>= 1;
            exponent += 1;
        }
    }
    (head, exponent)
}

pub(crate) fn rounds_away_from_zero(rounding: PtxFloatRounding, negative: bool) -> bool {
    match rounding {
        PtxFloatRounding::Zero => false,
        PtxFloatRounding::NegativeInfinity => negative,
        PtxFloatRounding::PositiveInfinity => !negative,
        // The nearest modes never reach here; their callers answer first.
        PtxFloatRounding::NearestEven | PtxFloatRounding::NearestAway => {
            unreachable!("nearest rounding does not select a directed magnitude")
        }
    }
}

/// `cvt.frnd.f32.{u,s}*`: integer to `f32` with a directed rounding modifier.
///
/// `.rn` is Rust's `as`, which is round-to-nearest-even.  The directed modes
/// are computed from the exact integer magnitude rather than from the
/// nearest-rounded value, because for a 64-bit source the two differ by more
/// than one `f32` ulp.
pub fn ptx_cvt_integer_to_f32(magnitude: u64, negative: bool, rounding: PtxFloatRounding) -> f32 {
    // `.frnd` is {rn, rz, rm, rp}; `.rna` has no integer-to-float spelling, so
    // it fails closed here rather than silently aliasing onto `.rn`.
    assert!(
        rounding != PtxFloatRounding::NearestAway,
        "PTX has no cvt.rna integer-to-f32 form"
    );
    if rounding == PtxFloatRounding::NearestEven {
        let nearest = magnitude as f32;
        return if negative { -nearest } else { nearest };
    }
    let (head, exponent) = round_integer_magnitude(
        magnitude,
        f32::MANTISSA_DIGITS,
        rounds_away_from_zero(rounding, negative),
    );
    let scale = f32::from_bits((127 + exponent) << 23);
    let result = (head as f32) * scale;
    if negative {
        -result
    } else {
        result
    }
}

/// `cvt.frnd.f64.{u,s}*`: integer to `f64` with a directed rounding modifier.
pub fn ptx_cvt_integer_to_f64(magnitude: u64, negative: bool, rounding: PtxFloatRounding) -> f64 {
    assert!(
        rounding != PtxFloatRounding::NearestAway,
        "PTX has no cvt.rna integer-to-f64 form"
    );
    if rounding == PtxFloatRounding::NearestEven {
        let nearest = magnitude as f64;
        return if negative { -nearest } else { nearest };
    }
    let (head, exponent) = round_integer_magnitude(
        magnitude,
        f64::MANTISSA_DIGITS,
        rounds_away_from_zero(rounding, negative),
    );
    let scale = f64::from_bits(((1023 + exponent) as u64) << 52);
    let result = (head as f64) * scale;
    if negative {
        -result
    } else {
        result
    }
}

pub(crate) fn ptx_cvt_integer_to_low(
    magnitude: u64,
    negative: bool,
    rounding: PtxFloatRounding,
    format: LowPrecisionFormat,
) -> u16 {
    if rounding == PtxFloatRounding::NearestEven {
        return encode_exact_low(negative, &[magnitude], 0, format, false);
    }
    let away = rounds_away_from_zero(rounding, negative);
    let (head, exponent) =
        round_integer_magnitude(magnitude, format.fraction_bits() as u32 + 1, away);
    // The directed result already has the target precision. Encoding it is
    // exact except for F16 exponent overflow, which still needs its direction.
    let bits = encode_exact_low(negative, &[head], exponent as i32, format, false);
    if !away && narrow_is_infinite(bits, format.infinity()) {
        narrow_step_toward_zero(bits)
    } else {
        bits
    }
}

pub fn ptx_cvt_integer_to_f16(
    magnitude: u64,
    negative: bool,
    rounding: PtxFloatRounding,
) -> u16 {
    ptx_cvt_integer_to_low(magnitude, negative, rounding, LowPrecisionFormat::F16)
}

pub fn ptx_cvt_integer_to_bf16(
    magnitude: u64,
    negative: bool,
    rounding: PtxFloatRounding,
) -> u16 {
    ptx_cvt_integer_to_low(magnitude, negative, rounding, LowPrecisionFormat::Bf16)
}

/// `cvt.frnd{.ftz}.f32.f64`: narrowing with a directed rounding modifier.
///
/// Measured: a NaN source keeps its sign, is quieted, and carries its high
/// payload bits down. At the normal/subnormal boundary, measured `.ftz`
/// behavior detects tininess after rounding to 24-bit precision without
/// restricting the exponent, not after gradual-underflow rounding.
pub fn ptx_cvt_f64_to_f32(value: f64, rounding: PtxFloatRounding, ftz: bool) -> f32 {
    if value.is_nan() {
        let bits = value.to_bits();
        let sign = ((bits >> 32) as u32) & 0x8000_0000;
        let payload = ((bits >> 29) as u32) & 0x003f_ffff;
        return f32::from_bits(sign | 0x7fc0_0000 | payload);
    }
    let nearest = value as f32;
    let exact = f64::from(nearest) == value;
    let rounded = if exact {
        nearest
    } else {
        match rounding {
            PtxFloatRounding::NearestEven | PtxFloatRounding::NearestAway => nearest,
            PtxFloatRounding::Zero => {
                if value > 0.0 {
                    if f64::from(nearest) > value {
                        nearest.next_down()
                    } else {
                        nearest
                    }
                } else if f64::from(nearest) < value {
                    nearest.next_up()
                } else {
                    nearest
                }
            }
            PtxFloatRounding::NegativeInfinity => {
                if f64::from(nearest) > value {
                    nearest.next_down()
                } else {
                    nearest
                }
            }
            PtxFloatRounding::PositiveInfinity => {
                if f64::from(nearest) < value {
                    nearest.next_up()
                } else {
                    nearest
                }
            }
        }
    };
    if ftz {
        flush_f32_result(rounded, || {
            // Only the MIN_NORMAL boundary enters here. Exact scaling moves
            // it into the normal range so the existing converter retains all
            // 24 significand bits before the tininess comparison.
            ptx_cvt_f64_to_f32(value * 2.0, rounding, false).abs() < 2.0 * f32::MIN_POSITIVE
        })
    } else {
        rounded
    }
}

/// `cvt{.ftz}.f32.f16` / `cvt{.ftz}.f32.bf16`: widening with the `.ftz` axis.
///
/// Measured: `.ftz` on these forms canonicalizes NaN to `0x7fffffff` and
/// flushes a subnormal `f32` result, which only bf16 sources can produce.
pub fn ptx_cvt_widen_to_f32(value: f32, ftz: bool) -> f32 {
    if !ftz {
        return value;
    }
    if value.is_nan() {
        return f32::from_bits(PTX_CVT_CANONICAL_NAN_F32);
    }
    flush_subnormal_f32(value)
}

/// `cvt{.ftz}.f64.f32`: widening, where `.ftz` still applies to the source.
pub fn ptx_cvt_f32_to_f64(value: f32, ftz: bool) -> f64 {
    f64::from(ptx_cvt_widen_to_f32(value, ftz))
}

/// One step of the `.f16` / `.bf16` magnitude toward zero.
pub(crate) fn narrow_step_toward_zero(bits: u16) -> u16 {
    let sign = bits & 0x8000;
    let magnitude = bits & 0x7fff;
    if magnitude == 0 {
        bits
    } else {
        sign | (magnitude - 1)
    }
}

/// Round to `.f16` or `.bf16`. Directed rounding adjusts the nearest
/// encoding by at most one magnitude ULP, including zero and finite overflow.
/// The comparison retains the source precision; the nearest payload must have
/// been rounded directly from that source, not through a narrower intermediate.
pub(crate) fn ptx_cvt_narrow_rounded(
    value: f64,
    rounding: PtxFloatRounding,
    nearest: u16,
    decode: fn(u16) -> f32,
) -> u16 {
    match rounding {
        PtxFloatRounding::NearestEven => return nearest,
        PtxFloatRounding::Zero
        | PtxFloatRounding::NegativeInfinity
        | PtxFloatRounding::PositiveInfinity => {}
        PtxFloatRounding::NearestAway => {
            unreachable!("half CVT has no .rna spelling")
        }
    }
    let decoded = f64::from(decode(nearest));
    let away = rounds_away_from_zero(rounding, value.is_sign_negative());
    if away && decoded.abs() < value.abs() {
        nearest + 1
    } else if !away && decoded.abs() > value.abs() {
        narrow_step_toward_zero(nearest)
    } else {
        nearest
    }
}

/// FP64 narrowing preserves the NaN sign/high payload and quiets it (B200).
/// Finite inputs reuse the exact dyadic codec, avoiding FP32 double rounding.
pub fn ptx_cvt_f64_to_low(
    value: f64,
    rounding: PtxFloatRounding,
    format: LowPrecisionFormat,
) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 48) as u16) & 0x8000;
    let fraction_bits = format.fraction_bits();
    if value.is_nan() {
        let payload = ((bits >> (52 - fraction_bits)) as u16) & ((1 << fraction_bits) - 1);
        return sign | format.infinity() | payload | (1 << (fraction_bits - 1));
    }
    if value.is_infinite() {
        return sign | format.infinity();
    }
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let significand = (bits & ((1_u64 << 52) - 1)) | (u64::from(exponent != 0) << 52);
    let nearest = encode_exact_low(
        sign != 0,
        &[significand],
        exponent.max(1) - 1023 - 52,
        format,
        false,
    );
    let decode = match format {
        LowPrecisionFormat::F16 => fp16_bits_to_f32,
        LowPrecisionFormat::Bf16 => bf16_bits_to_f32,
    };
    ptx_cvt_narrow_rounded(value, rounding, nearest, decode)
}

/// Half-to-FP64 is exact; NaNs retain sign/payload and become quiet (B200).
pub fn ptx_cvt_low_to_f64(bits: u16, format: LowPrecisionFormat) -> f64 {
    if bits & 0x7fff > format.infinity() {
        let fraction_bits = format.fraction_bits();
        let payload = u64::from(bits & ((1 << fraction_bits) - 1)) << (52 - fraction_bits);
        return f64::from_bits((u64::from(bits & 0x8000) << 48) | 0x7ff8_0000_0000_0000 | payload);
    }
    f64::from(decode_low(bits, format))
}

pub(crate) fn narrow_is_infinite(bits: u16, exponent_mask: u16) -> bool {
    bits & 0x7fff == exponent_mask
}

/// Shared body of `cvt.frnd2{.relu}{.satfinite}.{f16,bf16}.f32`.
pub(crate) fn ptx_cvt_narrow(
    value: f32,
    rounding: PtxFloatRounding,
    relu: bool,
    satfinite: bool,
    encode: fn(f32) -> u16,
    decode: fn(u16) -> f32,
    infinity: u16,
) -> u16 {
    if value.is_nan() {
        // Measured: `.relu` and `.satfinite` both leave this canonical NaN.
        return PTX_CVT_CANONICAL_NAN_NARROW;
    }
    let mut bits = ptx_cvt_narrow_rounded(f64::from(value), rounding, encode(value), decode);
    if satfinite && narrow_is_infinite(bits, infinity) {
        bits = narrow_step_toward_zero(bits);
    }
    if relu && bits & 0x8000 != 0 {
        bits = 0;
    }
    bits
}

/// `cvt.frnd2{.relu}{.satfinite}.f16.f32`.
pub fn ptx_cvt_f32_to_f16(
    value: f32,
    rounding: PtxFloatRounding,
    relu: bool,
    satfinite: bool,
) -> u16 {
    ptx_cvt_narrow(
        value,
        rounding,
        relu,
        satfinite,
        f32_to_fp16_bits,
        fp16_bits_to_f32,
        0x7c00,
    )
}

/// `cvt.frnd2{.relu}{.satfinite}.bf16.f32`.
///
/// This is a separate specialization from `Cvt<F32, Bf16, Rn>`, which the tile
/// lowering emits for bf16 elementwise and reduction rounding, but the two
/// agree on NaN: `encode_bf16` canonicalizes to `0x7fffffff` before narrowing,
/// so it also answers the `0x7fff` this family measured on hardware.
pub fn ptx_cvt_f32_to_bf16(
    value: f32,
    rounding: PtxFloatRounding,
    relu: bool,
    satfinite: bool,
) -> u16 {
    ptx_cvt_narrow(
        value,
        rounding,
        relu,
        satfinite,
        f32_to_bf16_bits,
        bf16_bits_to_f32,
        0x7f80,
    )
}

/// Half stochastic rounding reuses the RZ codec and its adjacent value.
///
/// Comparing the discarded fraction plus the supplied random fraction with
/// one is exactly the PTX carry test. All operands are binary32 values or
/// powers of two; binary64 keeps this comparison exact even at half subnormal
/// boundaries. There is no random generator or mutable rounding state.
pub fn ptx_cvt_half_rs<const RANDOM_BITS: u32>(
    value: f32,
    random: u16,
    relu: bool,
    satfinite: bool,
    convert: fn(f32, PtxFloatRounding, bool, bool) -> u16,
    decode: fn(u16) -> f32,
) -> u16 {
    let truncated = convert(value, PtxFloatRounding::Zero, relu, satfinite);
    if !value.is_finite() || (relu && value.is_sign_negative()) {
        return truncated;
    }
    let magnitude = truncated & 0x7fff;
    let base = f64::from(decode(magnitude));
    let next = f64::from(decode(magnitude + 1));
    if satfinite && next.is_infinite() {
        return truncated;
    }
    let step = if next.is_infinite() {
        // Infinity's rounding boundary is the next value at this precision,
        // not an infinite distance from MAX_NORM.
        base - f64::from(decode(magnitude - 1))
    } else {
        next - base
    };
    let units = f64::from(1_u32 << RANDOM_BITS);
    let carry = (f64::from(value.abs()) - base) * units >= step * (units - f64::from(random));
    truncated + u16::from(carry)
}

/// PTX 9.4 `.pzo` post-processing for an f16/bf16 conversion result.
///
/// The qualifier acts after conversion: only a negative-zero result loses its
/// sign. Applying this to the input would miss negative values that round to
/// zero, so callers normalize the encoded destination instead.
pub fn ptx_cvt_pzo_u16(bits: u16) -> u16 {
    if bits & 0x7fff == 0 {
        0
    } else {
        bits
    }
}

/// PTX 9.4 `.pzo` post-processing for a tf32 conversion result.
pub fn ptx_cvt_pzo_u32(bits: u32) -> u32 {
    if bits & 0x7fff_ffff == 0 {
        0
    } else {
        bits
    }
}

/// PTX 9.4 `.pzo` post-processing for two packed signed narrow floats.
///
/// Six-bit formats occupy padded byte fields while E2M1 uses packed nibbles,
/// so the format owns both the sign-bit position and the field stride. Only a
/// field whose encoded magnitude is zero loses its sign.
pub fn ptx_cvt_pzo_narrow_x2(bits: u16, format: NarrowFloatFormat) -> u16 {
    debug_assert!(format.signed);
    let field_mask = (1_u16 << format.storage_bits) - 1;
    let value_mask = (1_u16 << format.width_bits) - 1;
    let sign_mask = u16::from(format.sign_mask());
    let normalize = |field: u16| {
        if field & value_mask == sign_mask {
            field & !sign_mask
        } else {
            field
        }
    };
    let low = normalize(bits & field_mask);
    let high = normalize((bits >> format.storage_bits) & field_mask);
    (high << format.storage_bits) | low
}

/// `cvt.rna{.satfinite}.tf32.f32` and `cvt.frnd2{.satfinite}{.relu}.tf32.f32`.
///
/// The destination is a `.b32` register holding the source's sign and exponent
/// with a 10-bit fraction; the low 13 bits are always zero.  Measured
/// behaviour that the ISA prose does not state:
///
/// * `.rn` and `.rz` answer the canonical tf32 NaN `0x7fffe000` for every NaN
///   input, dropping the sign.
/// * `.rna` instead rounds a NaN arithmetically, so a signalling NaN whose
///   payload lives in the discarded bits becomes infinity; the carry saturates
///   at `0x7fffe000` rather than reaching the sign bit.
/// * `.satfinite` clamps by stepping one tf32 ulp down from any result whose
///   exponent field is all ones, which is why `.rna.satfinite` of a quiet NaN
///   is `0x7fbfe000` rather than a NaN-class constant.
pub fn ptx_cvt_f32_to_tf32(
    value: f32,
    rounding: PtxFloatRounding,
    relu: bool,
    satfinite: bool,
) -> u32 {
    let source = value.to_bits();
    let sign = source & 0x8000_0000;
    let magnitude = source & 0x7fff_ffff;
    if rounding != PtxFloatRounding::NearestAway && value.is_nan() {
        // Measured: `.relu` and `.satfinite` both leave this canonical NaN.
        return 0x7fff_e000;
    }
    let mut bits = match rounding {
        PtxFloatRounding::NearestAway => {
            let carried = (magnitude + 0x1000) & !(TF32_ULP - 1);
            sign | carried.min(0x7fff_e000)
        }
        PtxFloatRounding::Zero => sign | (magnitude & !(TF32_ULP - 1)),
        PtxFloatRounding::NearestEven => f32_to_tf32(value).to_bits(),
        // The tf32 line admits only {.rna, .rn, .rz}; the directed modes fail
        // closed rather than silently answering `.rn`.
        PtxFloatRounding::NegativeInfinity | PtxFloatRounding::PositiveInfinity => {
            unreachable!("PTX cvt.tf32.f32 admits only .rna, .rn and .rz")
        }
    };
    if satfinite && bits & 0x7f80_0000 == 0x7f80_0000 {
        bits = (bits & 0x8000_0000) | ((bits & 0x7fff_ffff) - TF32_ULP);
    }
    if relu && bits & 0x8000_0000 != 0 {
        bits = 0;
    }
    bits
}

