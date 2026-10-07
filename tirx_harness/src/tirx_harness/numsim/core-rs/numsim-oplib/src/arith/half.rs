//! Half-precision (`.f16`, `.bf16`, packed `x2`) arithmetic and the PTX 9.4
//! mixed-precision forms that pair half sources with `.f32`/`.f32x2`.
//!
//! Moved from legacy `engine-rs/src/runtime/instructions/reg.rs`
//! (`HalfRegister`, `HalfClamp`, `half_arithmetic_variant!`,
//! `half_minmax_variant!`, `low_precision_binary_variant!`, half `neg`/`abs`,
//! `MixedF32`, `MixedF32x2`, `MixedF32x2Down`, `mixed_low_mul_word`,
//! `decode_f16`/`encode_f16`/`decode_bf16`/`encode_bf16`),
//! `packed_mul.rs`, and `packed_fma.rs`.

use super::float::apply_f32_clamp;
use crate::cvt::{self, PtxFloatRounding};
use crate::scalar::{self, F32RoundingMode, LowPrecisionFormat};

/// Widen binary16 bits to f32 (CUDA `__half2float`).
pub fn decode_f16(bits: u16) -> f32 {
    scalar::cuda_fp16_bits_to_f32(bits)
}

/// Narrow f32 to binary16 bits, RN, with CUDA's canonical NaN `0x7fff`.
pub fn encode_f16(value: f32) -> u16 {
    scalar::cuda_f32_to_fp16_bits(value)
}

/// Widen bfloat16 bits to f32.
pub fn decode_bf16(bits: u16) -> f32 {
    cvt::bf16_bits_to_f32(bits)
}

/// Narrow f32 to bfloat16 bits, RN. Canonicalizes NaN to `0x7fffffff` first,
/// so every NaN lands on `0x7fff` -- the value `cvt.rn.bf16.f32` was measured
/// to produce on a B200 for every NaN encoding.
pub fn encode_bf16(value: f32) -> u16 {
    cvt::f32_to_bf16_bits(scalar::cuda_canonicalize_nan_f32(value))
}

/// Apply `op` to each 16-bit component of `N` packed `x2` words (low first).
pub fn map_half2<const N: usize>(args: [u32; N], operation: impl Fn([u16; N]) -> u16) -> u32 {
    let mut result = 0_u32;
    for component in 0..2 {
        result |=
            u32::from(operation(args.map(|x| (x >> (component * 16)) as u16))) << (component * 16);
    }
    result
}

/// Result clamp of same-precision half arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalfClamp {
    None,
    /// `.sat`: NaN and negatives map to `+0`; values above 1.0 map to 1.0.
    Sat,
    /// `.relu`: negatives map to `+0`, NaN maps to canonical `0x7fff`.
    Relu,
}

/// Apply a [`HalfClamp`] to one half result.
pub fn apply_half_clamp(bits: u16, format: LowPrecisionFormat, clamp: HalfClamp) -> u16 {
    match clamp {
        HalfClamp::None => bits,
        HalfClamp::Sat => {
            if bits & 0x7fff > format.infinity() || bits & 0x8000 != 0 {
                0
            } else {
                bits.min(0x3c00)
            }
        }
        HalfClamp::Relu => {
            if bits & 0x7fff > format.infinity() {
                0x7fff
            } else if bits & 0x8000 != 0 {
                0
            } else {
                bits
            }
        }
    }
}

/// `add.rn{.ftz}{.sat}.{f16,bf16}`.
pub fn add_half(a: u16, b: u16, format: LowPrecisionFormat, ftz: bool, clamp: HalfClamp) -> u16 {
    apply_half_clamp(scalar::low_add_rn(a, b, format, false, ftz), format, clamp)
}

/// `sub.rn{.ftz}{.sat}.{f16,bf16}`.
pub fn sub_half(a: u16, b: u16, format: LowPrecisionFormat, ftz: bool, clamp: HalfClamp) -> u16 {
    apply_half_clamp(scalar::low_add_rn(a, b, format, true, ftz), format, clamp)
}

/// `mul.rn{.ftz}{.sat}.{f16,bf16}`.
pub fn mul_half(a: u16, b: u16, format: LowPrecisionFormat, ftz: bool, clamp: HalfClamp) -> u16 {
    apply_half_clamp(scalar::low_mul_rn(a, b, format, ftz), format, clamp)
}

/// `fma.rn{.ftz}{.sat,.relu}{.oob}.{f16,bf16}`. With `.oob`, a PTX OOB NaN in
/// either multiplicand yields `+0`; a NaN addend remains NaN.
pub fn fma_half(
    a: u16,
    b: u16,
    c: u16,
    format: LowPrecisionFormat,
    ftz: bool,
    clamp: HalfClamp,
    oob: bool,
) -> u16 {
    let value = if oob && [a, b].iter().any(|x| x & 0x7fff == scalar::PTX_OOB_NAN) {
        0
    } else {
        scalar::low_fma_rn(a, b, c, format, ftz)
    };
    apply_half_clamp(value, format, clamp)
}

/// `add.rn{.ftz}{.sat}.{f16x2,bf16x2}`.
pub fn add_half2(a: u32, b: u32, format: LowPrecisionFormat, ftz: bool, clamp: HalfClamp) -> u32 {
    map_half2([a, b], |[a, b]| add_half(a, b, format, ftz, clamp))
}

/// `sub.rn{.ftz}{.sat}.{f16x2,bf16x2}`.
pub fn sub_half2(a: u32, b: u32, format: LowPrecisionFormat, ftz: bool, clamp: HalfClamp) -> u32 {
    map_half2([a, b], |[a, b]| sub_half(a, b, format, ftz, clamp))
}

/// `mul.rn{.ftz}{.sat}.{f16x2,bf16x2}`.
pub fn mul_half2(a: u32, b: u32, format: LowPrecisionFormat, ftz: bool, clamp: HalfClamp) -> u32 {
    map_half2([a, b], |[a, b]| mul_half(a, b, format, ftz, clamp))
}

/// `fma.rn{.ftz}{.sat,.relu}{.oob}.{f16x2,bf16x2}`.
pub fn fma_half2(
    a: u32,
    b: u32,
    c: u32,
    format: LowPrecisionFormat,
    ftz: bool,
    clamp: HalfClamp,
    oob: bool,
) -> u32 {
    map_half2([a, b, c], |[a, b, c]| {
        fma_half(a, b, c, format, ftz, clamp, oob)
    })
}

/// Legacy packed `mul.rn.f16x2` marker (component-wise `mul_f16_bits_rn`).
pub fn mul_f16x2_rn(lhs: u32, rhs: u32) -> u32 {
    map_half2([lhs, rhs], |[a, b]| scalar::mul_f16_bits_rn(a, b))
}

/// Legacy packed `mul.rn.bf16x2` marker.
pub fn mul_bf16x2_rn(lhs: u32, rhs: u32) -> u32 {
    map_half2([lhs, rhs], |[a, b]| scalar::mul_bf16_bits_rn(a, b))
}

/// Legacy packed `fma.rn.f16x2` marker.
pub fn fma_f16x2_rn(lhs: u32, rhs: u32, addend: u32) -> u32 {
    map_half2([lhs, rhs, addend], |[a, b, c]| {
        scalar::fma_f16_bits_rn(a, b, c)
    })
}

/// Legacy packed `fma.rn.bf16x2` marker.
pub fn fma_bf16x2_rn(lhs: u32, rhs: u32, addend: u32) -> u32 {
    map_half2([lhs, rhs, addend], |[a, b, c]| {
        scalar::fma_bf16_bits_rn(a, b, c)
    })
}

/// `min/max{.ftz}{.NaN}{.xorsign.abs}.{f16,bf16}`: selects an existing value
/// by ordered-encoding comparison (no host rounding).
pub fn minmax_half(
    a: u16,
    b: u16,
    format: LowPrecisionFormat,
    ftz: bool,
    propagate_nan: bool,
    xor_sign: bool,
    maximum: bool,
) -> u16 {
    cvt::low_minmax(a, b, format, ftz, propagate_nan, xor_sign, maximum)
}

/// Packed `x2` form of [`minmax_half`].
pub fn minmax_half2(
    a: u32,
    b: u32,
    format: LowPrecisionFormat,
    ftz: bool,
    propagate_nan: bool,
    xor_sign: bool,
    maximum: bool,
) -> u32 {
    map_half2([a, b], |[a, b]| {
        minmax_half(a, b, format, ftz, propagate_nan, xor_sign, maximum)
    })
}

/// Legacy plain `max.f16` marker: widen, CUDA `fmaxf`, narrow.
pub fn max_f16_widened(lhs: u16, rhs: u16) -> u16 {
    encode_f16(scalar::cuda_f32_max(decode_f16(lhs), decode_f16(rhs)))
}

/// Legacy plain `min.f16` marker.
pub fn min_f16_widened(lhs: u16, rhs: u16) -> u16 {
    encode_f16(scalar::cuda_f32_min(decode_f16(lhs), decode_f16(rhs)))
}

/// Legacy plain `max.bf16` marker.
pub fn max_bf16_widened(lhs: u16, rhs: u16) -> u16 {
    encode_bf16(scalar::cuda_f32_max(decode_bf16(lhs), decode_bf16(rhs)))
}

/// Legacy plain `min.bf16` marker.
pub fn min_bf16_widened(lhs: u16, rhs: u16) -> u16 {
    encode_bf16(scalar::cuda_f32_min(decode_bf16(lhs), decode_bf16(rhs)))
}

/// `neg.bf16` (sign flip, NaN included).
pub fn neg_bf16(value: u16) -> u16 {
    value ^ 0x8000
}

/// `neg.bf16x2`.
pub fn neg_bf16x2(value: u32) -> u32 {
    value ^ 0x8000_8000
}

/// `abs{.ftz}.f16`.
pub fn abs_f16(value: u16, ftz: bool) -> u16 {
    if ftz {
        scalar::flush_subnormal_f16_bits(value) & 0x7fff
    } else {
        value & 0x7fff
    }
}

/// `abs{.ftz}.f16x2`.
pub fn abs_f16x2(value: u32, ftz: bool) -> u32 {
    if ftz {
        u32::from(scalar::flush_subnormal_f16_bits(value as u16) & 0x7fff)
            | (u32::from(scalar::flush_subnormal_f16_bits((value >> 16) as u16) & 0x7fff) << 16)
    } else {
        value & 0x7fff_7fff
    }
}

/// `abs.bf16`.
pub fn abs_bf16(value: u16) -> u16 {
    value & 0x7fff
}

/// `abs.bf16x2`.
pub fn abs_bf16x2(value: u32) -> u32 {
    value & 0x7fff_7fff
}

/// `add.rnd{.sat}.f32.{f16,bf16} d, a, c`: `a` is the half source.
pub fn add_mixed_f32(
    low: u16,
    format: LowPrecisionFormat,
    addend: f32,
    mode: F32RoundingMode,
    sat: bool,
) -> f32 {
    apply_f32_clamp(
        scalar::add_f32(scalar::decode_low(low, format), addend, mode),
        sat,
    )
}

/// `sub.rnd{.sat}.f32.{f16,bf16} d, a, c`.
pub fn sub_mixed_f32(
    low: u16,
    format: LowPrecisionFormat,
    subtrahend: f32,
    mode: F32RoundingMode,
    sat: bool,
) -> f32 {
    apply_f32_clamp(
        scalar::sub_f32(scalar::decode_low(low, format), subtrahend, mode),
        sat,
    )
}

/// `fma.rnd{.sat}.f32.{f16,bf16} d, a, b, c` (half `a`, `b`; f32 `c`).
pub fn fma_mixed_f32(
    lhs: u16,
    rhs: u16,
    format: LowPrecisionFormat,
    addend: f32,
    mode: F32RoundingMode,
    sat: bool,
) -> f32 {
    apply_f32_clamp(
        scalar::fma_f32(
            scalar::decode_low(lhs, format),
            scalar::decode_low(rhs, format),
            addend,
            mode,
        ),
        sat,
    )
}

fn mixed_x2_decode(bits: u16, format: LowPrecisionFormat) -> f32 {
    match format {
        LowPrecisionFormat::F16 => scalar::cuda_fp16_bits_to_f32(bits),
        LowPrecisionFormat::Bf16 => cvt::bf16_bits_to_f32(bits),
    }
}

fn mixed_x2_source(packed: u32, format: LowPrecisionFormat) -> (f32, f32) {
    (
        mixed_x2_decode(packed as u16, format),
        mixed_x2_decode((packed >> 16) as u16, format),
    )
}

fn mixed_x2_result(low: f32, high: f32) -> u64 {
    cvt::make_float2(
        scalar::cuda_canonicalize_nan_f32(low),
        scalar::cuda_canonicalize_nan_f32(high),
    )
}

/// PTX 9.4 `add.rnd.f32x2.{f16x2,bf16x2} d, a, c` (packed half `a`).
pub fn add_mixed_f32x2(
    packed: u32,
    format: LowPrecisionFormat,
    addend: u64,
    mode: F32RoundingMode,
) -> u64 {
    let (low, high) = mixed_x2_source(packed, format);
    mixed_x2_result(
        scalar::add_f32(low, cvt::float2_x(addend), mode),
        scalar::add_f32(high, cvt::float2_y(addend), mode),
    )
}

/// PTX 9.4 `sub.rnd.f32x2.{f16x2,bf16x2} d, a, c`.
pub fn sub_mixed_f32x2(
    packed: u32,
    format: LowPrecisionFormat,
    subtrahend: u64,
    mode: F32RoundingMode,
) -> u64 {
    let (low, high) = mixed_x2_source(packed, format);
    mixed_x2_result(
        scalar::sub_f32(low, cvt::float2_x(subtrahend), mode),
        scalar::sub_f32(high, cvt::float2_y(subtrahend), mode),
    )
}

/// PTX 9.4 `fma.rnd.f32x2.{f16x2,bf16x2} d, a, b, c`.
pub fn fma_mixed_f32x2(
    packed: u32,
    format: LowPrecisionFormat,
    multiplier: u64,
    addend: u64,
    mode: F32RoundingMode,
) -> u64 {
    let (low, high) = mixed_x2_source(packed, format);
    mixed_x2_result(
        scalar::fma_f32(low, cvt::float2_x(multiplier), cvt::float2_x(addend), mode),
        scalar::fma_f32(high, cvt::float2_y(multiplier), cvt::float2_y(addend), mode),
    )
}

/// Arithmetic operation of the `.f32x2` -> packed-half narrowing forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixedDownOp {
    Add,
    Sub,
    Mul,
}

fn mixed_x2_down_lane(lhs: f32, rhs: f32, op: MixedDownOp, dst: LowPrecisionFormat) -> u16 {
    // PTX 9.4 requires `.ftz` on the f16x2 destination spelling. SM107
    // hardware was unavailable while this model was added, so this follows
    // PTX's general `.ftz` contract: flush subnormal inputs and result, while
    // retaining the sign of zero. A future SM107 oracle should lock this down.
    let ftz = matches!(dst, LowPrecisionFormat::F16);
    let flush = |value: f32| {
        if ftz {
            scalar::flush_subnormal_f32(value)
        } else {
            value
        }
    };
    let (lhs, rhs) = (flush(lhs), flush(rhs));
    let operation = match op {
        MixedDownOp::Add => scalar::add_f32,
        MixedDownOp::Sub => scalar::sub_f32,
        MixedDownOp::Mul => scalar::mul_f32,
    };
    // The destination formats are subsets of f32. Rounding the exact
    // operation toward zero to f32 and then toward zero again therefore
    // equals one direct destination-format RZ conversion.
    let value = flush(operation(lhs, rhs, F32RoundingMode::Zero));
    match dst {
        LowPrecisionFormat::F16 => scalar::flush_subnormal_f16_bits(cvt::ptx_cvt_f32_to_f16(
            value,
            PtxFloatRounding::Zero,
            false,
            false,
        )),
        LowPrecisionFormat::Bf16 => {
            cvt::ptx_cvt_f32_to_bf16(value, PtxFloatRounding::Zero, false, false)
        }
    }
}

/// PTX 9.4 `{add,sub,mul}.rz{.ftz}.{f16x2,bf16x2}.f32x2 d, a, b`; the `f16x2`
/// destination carries the mandatory `.ftz`, `bf16x2` preserves subnormals.
pub fn mixed_f32x2_down(lhs: u64, rhs: u64, op: MixedDownOp, dst: LowPrecisionFormat) -> u32 {
    u32::from(mixed_x2_down_lane(
        cvt::float2_x(lhs),
        cvt::float2_x(rhs),
        op,
        dst,
    )) | (u32::from(mixed_x2_down_lane(
        cvt::float2_y(lhs),
        cvt::float2_y(rhs),
        op,
        dst,
    )) << 16)
}

/// PTX 9.4 `mul.rn.bf16x2.f16x2 d, a(bf16x2), b(f16x2)`: converts `b` to bf16
/// first, then performs bf16 RN multiplication.
pub fn mul_bf16x2_f16x2(lhs: u32, rhs: u32) -> u32 {
    map_half2([lhs, rhs], |[a, b]| {
        scalar::mul_bf16_bits_rn(a, encode_bf16(decode_f16(b)))
    })
}

/// PTX 9.4 `mul.rn.f16x2.bf16x2 d, a(f16x2), b(bf16x2)`.
pub fn mul_f16x2_bf16x2(lhs: u32, rhs: u32) -> u32 {
    map_half2([lhs, rhs], |[a, b]| {
        scalar::mul_f16_bits_rn(a, encode_f16(decode_bf16(b)))
    })
}
