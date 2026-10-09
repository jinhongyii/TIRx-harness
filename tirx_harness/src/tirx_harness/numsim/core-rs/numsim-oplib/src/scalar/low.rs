//! binary16 / bfloat16 arithmetic rounded from the exact result.
#![allow(unused_imports)]
use crate::cvt::formats::*;
use crate::cvt::*;
use crate::scalar::*;
use crate::types::OpError;
use std::cmp::Ordering;
#[derive(Clone, Copy)]
pub enum LowPrecisionFormat {
    F16,
    Bf16,
}

impl LowPrecisionFormat {
    pub(crate) const fn fraction_bits(self) -> usize {
        match self {
            Self::F16 => 10,
            Self::Bf16 => 7,
        }
    }

    pub(crate) const fn minimum_normal_exponent(self) -> i32 {
        match self {
            Self::F16 => -14,
            Self::Bf16 => -126,
        }
    }

    pub(crate) const fn maximum_normal_exponent(self) -> i32 {
        match self {
            Self::F16 => 15,
            Self::Bf16 => 127,
        }
    }

    pub(crate) const fn exponent_bias(self) -> i32 {
        match self {
            Self::F16 => 15,
            Self::Bf16 => 127,
        }
    }

    /// Raw bits of `+inf` in this format (f16 `0x7c00`, bf16 `0x7f80`).
    pub const fn infinity(self) -> u16 {
        match self {
            Self::F16 => 0x7c00,
            Self::Bf16 => 0x7f80,
        }
    }

    /// Raw bits of `+1.0` in this format (f16 `0x3c00`, bf16 `0x3f80`): the
    /// `.sat` upper clamp.
    pub const fn one(self) -> u16 {
        match self {
            Self::F16 => 0x3c00,
            Self::Bf16 => 0x3f80,
        }
    }

    /// True for a subnormal encoding of this format (zero exponent field,
    /// nonzero fraction). No numerics: a pure bit test.
    pub const fn is_subnormal(self, bits: u16) -> bool {
        let fraction = (1_u16 << self.fraction_bits()) - 1;
        bits & self.infinity() == 0 && bits & fraction != 0
    }

    /// FTZ of raw bits in this format: a subnormal becomes the zero of the same
    /// sign; every other encoding (zeros, normals, inf, NaN with its payload) is
    /// returned unchanged. The format decides the subnormal test, so f16 and
    /// bf16 bits cannot be flushed with the other format's rule.
    pub const fn flush_subnormal(self, bits: u16) -> u16 {
        if self.is_subnormal(bits) {
            bits & 0x8000
        } else {
            bits
        }
    }

    pub(crate) fn encode_host_result(self, value: f32) -> u16 {
        match self {
            Self::F16 => cuda_f32_to_fp16_bits(value),
            Self::Bf16 => f32_to_bf16_bits(cuda_canonicalize_nan_f32(value)),
        }
    }
}

pub(crate) fn highest_set_bit(words: &[u64]) -> Option<usize> {
    words.iter().rposition(|word| *word != 0).map(|index| {
        index * u64::BITS as usize + (u64::BITS - 1 - words[index].leading_zeros()) as usize
    })
}

pub(crate) fn bit_is_set(words: &[u64], bit: usize) -> bool {
    words
        .get(bit / u64::BITS as usize)
        .is_some_and(|word| word & (1_u64 << (bit % u64::BITS as usize)) != 0)
}

pub(crate) fn any_bit_below(words: &[u64], bit: usize) -> bool {
    let whole_words = bit / u64::BITS as usize;
    if words.iter().take(whole_words).any(|word| *word != 0) {
        return true;
    }
    let partial = bit % u64::BITS as usize;
    partial != 0
        && words
            .get(whole_words)
            .is_some_and(|word| word & ((1_u64 << partial) - 1) != 0)
}

pub(crate) fn low_u64_after_shift(words: &[u64], shift: usize) -> u64 {
    let word_index = shift / u64::BITS as usize;
    let bit_index = shift % u64::BITS as usize;
    let low = words.get(word_index).copied().unwrap_or(0) >> bit_index;
    if bit_index == 0 {
        low
    } else {
        low | words
            .get(word_index + 1)
            .copied()
            .unwrap_or(0)
            .wrapping_shl((u64::BITS as usize - bit_index) as u32)
    }
}

pub(crate) fn round_words_right_even(words: &[u64], shift: usize) -> u64 {
    if shift == 0 {
        return words.first().copied().unwrap_or(0);
    }
    let truncated = low_u64_after_shift(words, shift);
    let halfway = bit_is_set(words, shift - 1);
    let below_halfway = any_bit_below(words, shift - 1);
    truncated + u64::from(halfway && (below_halfway || truncated & 1 != 0))
}

/// Round an exact signed dyadic integer to one scalar f16/bf16 payload.
/// `words` is a little-endian magnitude in units of `2^unit_exponent`.
pub(crate) fn encode_exact_low(
    negative: bool,
    words: &[u64],
    unit_exponent: i32,
    format: LowPrecisionFormat,
    ftz: bool,
) -> u16 {
    let sign = if negative { 0x8000 } else { 0 };
    let Some(highest) = highest_set_bit(words) else {
        return sign;
    };
    let fraction_bits = format.fraction_bits();
    let mut exponent = unit_exponent + highest as i32;
    // Half FTZ detects tininess before rounding, including values that
    // would otherwise round up to the smallest normal.
    if ftz && exponent < format.minimum_normal_exponent() {
        return sign;
    }
    if exponent > format.maximum_normal_exponent() {
        return sign | format.infinity();
    }

    if exponent >= format.minimum_normal_exponent() {
        let mut significand = if highest >= fraction_bits {
            round_words_right_even(words, highest - fraction_bits)
        } else {
            // Exact inputs such as small integers may need normalization to
            // the left, with no discarded bits and therefore no rounding.
            words[0] << (fraction_bits - highest)
        };
        if significand == 1_u64 << (fraction_bits + 1) {
            significand >>= 1;
            exponent += 1;
        }
        if exponent > format.maximum_normal_exponent() {
            return sign | format.infinity();
        }
        let exponent_field = (exponent + format.exponent_bias()) as u16;
        let fraction_mask = (1_u16 << fraction_bits) - 1;
        return sign | (exponent_field << fraction_bits) | (significand as u16 & fraction_mask);
    }

    let quantum_exponent = format.minimum_normal_exponent() - fraction_bits as i32;
    let shift = (quantum_exponent - unit_exponent) as usize;
    let subnormal = round_words_right_even(words, shift);
    sign | subnormal as u16
}

/// Exact widening of raw f16/bf16 bits to binary32. f16 NaN becomes the CUDA
/// canonical `0x7fff_ffff`; bf16 NaN keeps its payload (bits shifted left 16).
pub fn decode_low(bits: u16, format: LowPrecisionFormat) -> f32 {
    match format {
        LowPrecisionFormat::F16 => cuda_fp16_bits_to_f32(bits),
        LowPrecisionFormat::Bf16 => bf16_bits_to_f32(bits),
    }
}

/// PTX `add/sub.rn{.ftz}` on one f16/bf16 payload: exact sum rounded once to the
/// target format (RN-even, overflow to inf). Any NaN result is canonical `0x7fff`.
/// `ftz` flushes subnormal inputs and results tiny before rounding, using
/// `format`'s own subnormal test ([`LowPrecisionFormat::flush_subnormal`]).
pub fn low_add_rn(
    lhs: u16,
    rhs: u16,
    format: LowPrecisionFormat,
    subtract: bool,
    ftz: bool,
) -> u16 {
    let (lhs, rhs) = if ftz {
        (format.flush_subnormal(lhs), format.flush_subnormal(rhs))
    } else {
        (lhs, rhs)
    };
    let lhs = decode_low(lhs, format);
    let rhs = decode_low(rhs, format);
    let host = pin_nan2_f32(lhs, rhs, if subtract { lhs - rhs } else { lhs + rhs });
    if !lhs.is_finite() || !rhs.is_finite() {
        return format.encode_host_result(host);
    }
    let rhs = if subtract { -rhs } else { rhs };
    let exact = exact_f32_sum(ExactF32Value::from_f32(lhs), ExactF32Value::from_f32(rhs));
    if exact.is_zero() {
        return format.encode_host_result(host);
    }
    encode_exact_low(exact.negative, &exact.magnitude.0, -149, format, ftz)
}

/// PTX `fma.rn{.ftz}` on one f16/bf16 payload: exact `lhs*rhs+addend` rounded once
/// (RN-even, overflow to inf); NaN result canonical `0x7fff`. `ftz` as in
/// [`low_add_rn`] (format-aware subnormal test).
pub fn low_fma_rn(lhs: u16, rhs: u16, addend: u16, format: LowPrecisionFormat, ftz: bool) -> u16 {
    let (lhs, rhs, addend) = if ftz {
        (
            format.flush_subnormal(lhs),
            format.flush_subnormal(rhs),
            format.flush_subnormal(addend),
        )
    } else {
        (lhs, rhs, addend)
    };
    let lhs = decode_low(lhs, format);
    let rhs = decode_low(rhs, format);
    let addend = decode_low(addend, format);
    let host = host_fma_f32(lhs, rhs, addend);
    if !lhs.is_finite() || !rhs.is_finite() || !addend.is_finite() {
        return format.encode_host_result(host);
    }
    let exact = exact_fma_sum(
        ExactFmaValue::product(lhs, rhs),
        ExactFmaValue::addend(addend),
    );
    if exact.is_zero() {
        return format.encode_host_result(host);
    }
    encode_exact_low(exact.negative, &exact.magnitude.0, -298, format, ftz)
}

/// `sub.rn.f16` on raw bits, no FTZ (see [`low_add_rn`]).
pub fn sub_f16_bits_rn(lhs: u16, rhs: u16) -> u16 {
    low_add_rn(lhs, rhs, LowPrecisionFormat::F16, true, false)
}

/// `sub.rn.f16x2`: [`sub_f16_bits_rn`] per half (low half = bits 0..16).
pub fn sub_f16x2_bits_rn(lhs: u32, rhs: u32) -> u32 {
    u32::from(sub_f16_bits_rn(lhs as u16, rhs as u16))
        | (u32::from(sub_f16_bits_rn((lhs >> 16) as u16, (rhs >> 16) as u16)) << 16)
}

/// `mul.rn.f16` on raw bits, no FTZ (see [`low_mul_rn`]).
pub fn mul_f16_bits_rn(lhs: u16, rhs: u16) -> u16 {
    low_mul_rn(lhs, rhs, LowPrecisionFormat::F16, false)
}

/// PTX `mul.rn{.ftz}` on one f16/bf16 payload: an exact FMA with a same-signed zero
/// addend, so `-0` products survive; rounding/NaN/`ftz` as in [`low_fma_rn`].
pub fn low_mul_rn(lhs: u16, rhs: u16, format: LowPrecisionFormat, ftz: bool) -> u16 {
    // A same-sign zero addend preserves multiplication's signed zero.
    low_fma_rn(lhs, rhs, (lhs ^ rhs) & 0x8000, format, ftz)
}

/// TMA's 16-bit OOB-NaN payload; FMA tests its magnitude, not its sign.
pub const PTX_OOB_NAN: u16 = 0x7ff7;

/// `fma.rn.f16` on raw bits, no FTZ (see [`low_fma_rn`]).
pub fn fma_f16_bits_rn(lhs: u16, rhs: u16, addend: u16) -> u16 {
    low_fma_rn(lhs, rhs, addend, LowPrecisionFormat::F16, false)
}

/// `add.rn.bf16` on raw bits (see [`low_add_rn`]).
pub fn add_bf16_bits_rn(lhs: u16, rhs: u16) -> u16 {
    low_add_rn(lhs, rhs, LowPrecisionFormat::Bf16, false, false)
}

/// `add.rn.bf16x2`: [`add_bf16_bits_rn`] per half (low half = bits 0..16).
pub fn add_bf16x2_bits_rn(lhs: u32, rhs: u32) -> u32 {
    u32::from(add_bf16_bits_rn(lhs as u16, rhs as u16))
        | (u32::from(add_bf16_bits_rn((lhs >> 16) as u16, (rhs >> 16) as u16)) << 16)
}

/// `sub.rn.bf16` on raw bits (see [`low_add_rn`]).
pub fn sub_bf16_bits_rn(lhs: u16, rhs: u16) -> u16 {
    low_add_rn(lhs, rhs, LowPrecisionFormat::Bf16, true, false)
}

/// `sub.rn.bf16x2`: [`sub_bf16_bits_rn`] per half (low half = bits 0..16).
pub fn sub_bf16x2_bits_rn(lhs: u32, rhs: u32) -> u32 {
    u32::from(sub_bf16_bits_rn(lhs as u16, rhs as u16))
        | (u32::from(sub_bf16_bits_rn((lhs >> 16) as u16, (rhs >> 16) as u16)) << 16)
}

/// `mul.rn.bf16` on raw bits (see [`low_mul_rn`]).
pub fn mul_bf16_bits_rn(lhs: u16, rhs: u16) -> u16 {
    low_mul_rn(lhs, rhs, LowPrecisionFormat::Bf16, false)
}

/// `fma.rn.bf16` on raw bits (see [`low_fma_rn`]).
pub fn fma_bf16_bits_rn(lhs: u16, rhs: u16, addend: u16) -> u16 {
    low_fma_rn(lhs, rhs, addend, LowPrecisionFormat::Bf16, false)
}
