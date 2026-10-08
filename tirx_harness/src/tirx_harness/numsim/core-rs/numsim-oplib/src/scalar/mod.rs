//! Bit-exact scalar arithmetic oracles shared by every NumSim op family.
//!
//! Moved from the legacy engine `scalar.rs`. Split by concern:
//! `exact` (exact-sum machinery), `low` (binary16/bfloat16 arithmetic),
//! `round` (directed-rounding f32/f64 arithmetic, sqrt, div, fma).
//! Conversions (`cvt`) and packed lane helpers live in `crate::cvt`.

mod exact;
mod low;
mod math;
mod round;
#[cfg(test)]
mod tests;

pub(crate) use exact::*;
pub use low::*;
pub use math::*;
pub use round::*;

use crate::cvt::formats::*;
use crate::types::OpError;

pub type F32x4 = [f32; 4];
pub type U64x2 = [u64; 2];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F32RoundingMode {
    Nearest,
    Down,
    Up,
    Zero,
}

pub trait RuntimeScalar: Copy + Clone {
    const BYTE_LEN: usize;

    fn zero() -> Self;
    fn decode_le(bytes: &[u8]) -> Result<Self, OpError>;
    fn encode_le_into(self, target: &mut [u8]);

    fn encode_le(self) -> Vec<u8> {
        let mut bytes = vec![0; Self::BYTE_LEN];
        self.encode_le_into(&mut bytes);
        bytes
    }
}

macro_rules! impl_runtime_scalar {
    ($rust_type:ty, $byte_len:expr) => {
        impl RuntimeScalar for $rust_type {
            const BYTE_LEN: usize = $byte_len;

            fn zero() -> Self {
                0 as $rust_type
            }

            fn decode_le(bytes: &[u8]) -> Result<Self, OpError> {
                let encoded: [u8; $byte_len] = bytes.try_into().map_err(|_| {
                    OpError::message(format!(
                        "expected {} bytes for {}",
                        $byte_len,
                        stringify!($rust_type)
                    ))
                })?;
                Ok(<$rust_type>::from_le_bytes(encoded))
            }

            fn encode_le_into(self, target: &mut [u8]) {
                target.copy_from_slice(&self.to_le_bytes());
            }
        }
    };
}

impl_runtime_scalar!(i8, 1);
impl_runtime_scalar!(i16, 2);
impl_runtime_scalar!(i32, 4);
impl_runtime_scalar!(i64, 8);
impl_runtime_scalar!(u8, 1);
impl_runtime_scalar!(u16, 2);
impl_runtime_scalar!(u32, 4);
impl_runtime_scalar!(u64, 8);
impl_runtime_scalar!(f32, 4);
impl_runtime_scalar!(f64, 8);

impl RuntimeScalar for F32x4 {
    const BYTE_LEN: usize = 16;

    fn zero() -> Self {
        [0.0_f32; 4]
    }

    fn decode_le(bytes: &[u8]) -> Result<Self, OpError> {
        if bytes.len() != Self::BYTE_LEN {
            return Err(OpError::message(format!(
                "expected {} bytes for F32x4, got {}",
                Self::BYTE_LEN,
                bytes.len(),
            )));
        }
        let mut values = [0.0_f32; 4];
        for (index, value) in values.iter_mut().enumerate() {
            let start = index * 4;
            *value = f32::from_le_bytes([
                bytes[start],
                bytes[start + 1],
                bytes[start + 2],
                bytes[start + 3],
            ]);
        }
        Ok(values)
    }

    fn encode_le_into(self, target: &mut [u8]) {
        for (index, value) in self.into_iter().enumerate() {
            let start = index * 4;
            target[start..start + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
}

impl RuntimeScalar for U64x2 {
    const BYTE_LEN: usize = 16;

    fn zero() -> Self {
        [0_u64; 2]
    }

    fn decode_le(bytes: &[u8]) -> Result<Self, OpError> {
        if bytes.len() != Self::BYTE_LEN {
            return Err(OpError::message(format!(
                "expected {} bytes for U64x2, got {}",
                Self::BYTE_LEN,
                bytes.len(),
            )));
        }
        let mut values = [0_u64; 2];
        for (index, value) in values.iter_mut().enumerate() {
            let start = index * 8;
            *value = u64::from_le_bytes(
                bytes[start..start + 8]
                    .try_into()
                    .expect("U64x2 component width was checked"),
            );
        }
        Ok(values)
    }

    fn encode_le_into(self, target: &mut [u8]) {
        for (index, value) in self.into_iter().enumerate() {
            let start = index * 8;
            target[start..start + 8].copy_from_slice(&value.to_le_bytes());
        }
    }
}

impl RuntimeScalar for bool {
    const BYTE_LEN: usize = 1;

    fn zero() -> Self {
        false
    }

    fn decode_le(bytes: &[u8]) -> Result<Self, OpError> {
        match bytes {
            [0] => Ok(false),
            [1] => Ok(true),
            [value] => Err(OpError::message(format!(
                "invalid physical bool byte {value}"
            ))),
            _ => Err(OpError::message("expected one byte for bool")),
        }
    }

    fn encode_le_into(self, target: &mut [u8]) {
        target[0] = u8::from(self);
    }
}

/// Python-style floor division of i64 (rounds toward -inf). Errors on division by zero
/// and on `i64::MIN / -1` overflow.
#[inline]
pub fn floor_div_i64(lhs: i64, rhs: i64) -> Result<i64, OpError> {
    let quotient = lhs
        .checked_div(rhs)
        .ok_or_else(|| OpError::message(format!("invalid floor division: {lhs} // {rhs}")))?;
    let remainder = lhs
        .checked_rem(rhs)
        .ok_or_else(|| OpError::message(format!("invalid floor remainder: {lhs} % {rhs}")))?;
    Ok(if remainder != 0 && ((remainder < 0) != (rhs < 0)) {
        quotient - 1
    } else {
        quotient
    })
}

/// Python-style floor modulo of i64 (result takes the divisor's sign); errors as
/// [`floor_div_i64`], and the final multiply-subtract wraps (`1 mod i64::MIN` is exact).
#[inline]
pub fn floor_mod_i64(lhs: i64, rhs: i64) -> Result<i64, OpError> {
    let quotient = floor_div_i64(lhs, rhs)?;
    // Wrapping: e.g. `1 mod i64::MIN` has quotient -1 and `-1 * MIN`
    // overflows although the result (`MIN + 1`) is representable. Legacy
    // release builds wrapped the same way; debug builds panicked.
    Ok(lhs.wrapping_sub(quotient.wrapping_mul(rhs)))
}

/// PTX `fns.b32`: bit position of the `offset`-th set bit of `mask` counting from
/// `base` (upward for positive, downward for negative, `0` tests `base` itself);
/// `0xffff_ffff` when there is none. `base` must be < 32.
pub fn ptx_fns_b32(mask: u32, base: u32, offset: i32) -> u32 {
    debug_assert!(base < 32);
    if offset == 0 {
        return if mask & (1_u32 << base) != 0 {
            base
        } else {
            u32::MAX
        };
    }
    let mut position = base as i32;
    let mut remaining = offset.unsigned_abs() - 1;
    let increment = if offset > 0 { 1_i32 } else { -1_i32 };
    while (0..32).contains(&position) {
        if mask & (1_u32 << position) != 0 {
            if remaining == 0 {
                return position as u32;
            }
            remaining -= 1;
        }
        position += increment;
    }
    u32::MAX
}

/// FTZ for binary32: a subnormal becomes the zero of the same sign; others unchanged.
pub fn flush_subnormal_f32(value: f32) -> f32 {
    if value.is_subnormal() {
        f32::from_bits(value.to_bits() & 0x8000_0000)
    } else {
        value
    }
}

/// Whether raw binary16 bits encode a subnormal (exponent 0, fraction non-zero).
pub const fn is_subnormal_f16_bits(bits: u16) -> bool {
    bits & 0x7c00 == 0 && bits & 0x03ff != 0
}

/// FTZ for raw binary16 bits: a subnormal becomes the zero of the same sign.
/// Uses the f16 encoding; not valid for bf16 bits.
pub const fn flush_subnormal_f16_bits(bits: u16) -> u16 {
    if is_subnormal_f16_bits(bits) {
        bits & 0x8000
    } else {
        bits
    }
}

pub(crate) fn cuda_canonical_nan_f32() -> f32 {
    f32::from_bits(0x7fff_ffff)
}

/// Replace any NaN with the CUDA canonical binary32 NaN `0x7fff_ffff`; others unchanged.
pub fn cuda_canonicalize_nan_f32(value: f32) -> f32 {
    if value.is_nan() {
        cuda_canonical_nan_f32()
    } else {
        value
    }
}

/// CUDA `__float2half_rn`: binary32 to binary16, RN-even (overflow to inf, binary32
/// subnormals to signed zero); any NaN becomes the canonical `0x7fff`.
pub fn cuda_f32_to_fp16_bits(value: f32) -> u16 {
    if value.is_nan() {
        0x7fff
    } else {
        f32_to_fp16_bits(value)
    }
}

/// CUDA `__half2float`: exact binary16 widening; any NaN becomes `0x7fff_ffff`.
pub fn cuda_fp16_bits_to_f32(bits: u16) -> f32 {
    cuda_canonicalize_nan_f32(fp16_bits_to_f32(bits))
}

/// CUDA binary32 add, RN, no FTZ; any NaN result is the canonical `0x7fff_ffff`.
pub fn cuda_f32_add(lhs: f32, rhs: f32) -> f32 {
    let result = lhs + rhs;
    if result.is_nan() {
        cuda_canonical_nan_f32()
    } else {
        result
    }
}

pub(crate) fn cuda_round_fp16(value: f32) -> f32 {
    cuda_fp16_bits_to_f32(cuda_f32_to_fp16_bits(value))
}

pub(crate) fn cuda_round_bf16(value: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(cuda_canonicalize_nan_f32(value)))
}

/// One fp16 warp-reduce step on f32 carriers: round both to binary16 (RN), add in
/// binary32, round the sum to binary16; NaN canonical.
pub fn cuda_reduce_fp16_add(lhs: f32, rhs: f32) -> f32 {
    cuda_round_fp16(cuda_f32_add(cuda_round_fp16(lhs), cuda_round_fp16(rhs)))
}

/// One fp16 max step: operands rounded to binary16 (RN, NaN canonical), then PTX
/// default `max` ([`cuda_f32_max`]): one NaN loses to the other operand, two NaNs give
/// the canonical NaN, `max(-0, +0) = +0`. The same rule as the engine's TIR `Max` on
/// f16, which the v2 butterfly reduction lowers to.
pub fn cuda_reduce_fp16_max(lhs: f32, rhs: f32) -> f32 {
    cuda_round_fp16(cuda_f32_max(cuda_round_fp16(lhs), cuda_round_fp16(rhs)))
}

/// One fp16 min step: as [`cuda_reduce_fp16_max`] with [`cuda_f32_min`]
/// (`min(-0, +0) = -0`).
pub fn cuda_reduce_fp16_min(lhs: f32, rhs: f32) -> f32 {
    cuda_round_fp16(cuda_f32_min(cuda_round_fp16(lhs), cuda_round_fp16(rhs)))
}

/// One bf16 warp-reduce step on f32 carriers: round both to bfloat16 (RN), add in
/// binary32, round the sum to bfloat16; NaN canonical.
pub fn cuda_reduce_bf16_add(lhs: f32, rhs: f32) -> f32 {
    cuda_round_bf16(cuda_f32_add(cuda_round_bf16(lhs), cuda_round_bf16(rhs)))
}

/// One bf16 max step: operands rounded to bfloat16 (RN, NaN canonical), then PTX
/// default `max` ([`cuda_f32_max`], NaN and signed-zero rules as in
/// [`cuda_reduce_fp16_max`]).
pub fn cuda_reduce_bf16_max(lhs: f32, rhs: f32) -> f32 {
    cuda_round_bf16(cuda_f32_max(cuda_round_bf16(lhs), cuda_round_bf16(rhs)))
}

/// One bf16 min step: as [`cuda_reduce_bf16_max`] with [`cuda_f32_min`].
pub fn cuda_reduce_bf16_min(lhs: f32, rhs: f32) -> f32 {
    cuda_round_bf16(cuda_f32_min(cuda_round_bf16(lhs), cuda_round_bf16(rhs)))
}

/// CUDA `fmaxf`: one NaN loses to the other operand, two NaNs give `0x7fff_ffff`;
/// `max(-0, +0) = +0`. No rounding, no FTZ.
pub fn cuda_f32_max(lhs: f32, rhs: f32) -> f32 {
    match (lhs.is_nan(), rhs.is_nan()) {
        (true, true) => cuda_canonical_nan_f32(),
        (true, false) => rhs,
        (false, true) => lhs,
        (false, false) if lhs == 0.0 && rhs == 0.0 => {
            if lhs.is_sign_positive() || rhs.is_sign_positive() {
                0.0
            } else {
                -0.0
            }
        }
        (false, false) if lhs > rhs => lhs,
        (false, false) => rhs,
    }
}

/// CUDA `fminf`: one NaN loses to the other operand, two NaNs give `0x7fff_ffff`;
/// `min(-0, +0) = -0`. No rounding, no FTZ.
pub fn cuda_f32_min(lhs: f32, rhs: f32) -> f32 {
    match (lhs.is_nan(), rhs.is_nan()) {
        (true, true) => cuda_canonical_nan_f32(),
        (true, false) => rhs,
        (false, true) => lhs,
        (false, false) if lhs == 0.0 && rhs == 0.0 => {
            if lhs.is_sign_negative() || rhs.is_sign_negative() {
                -0.0
            } else {
                0.0
            }
        }
        (false, false) if lhs < rhs => lhs,
        (false, false) => rhs,
    }
}

pub(crate) fn quiet_f64_nan(value: f64) -> f64 {
    f64::from_bits(value.to_bits() | 0x0008_0000_0000_0000)
}

/// CUDA binary64 add, RN, no FTZ: a NaN `rhs` (else `lhs`) is returned quieted with its
/// payload; `inf + -inf` gives `0xfff8_0000_0000_0000`.
pub fn cuda_f64_add(lhs: f64, rhs: f64) -> f64 {
    if rhs.is_nan() {
        quiet_f64_nan(rhs)
    } else if lhs.is_nan() {
        quiet_f64_nan(lhs)
    } else if lhs.is_infinite()
        && rhs.is_infinite()
        && lhs.is_sign_negative() != rhs.is_sign_negative()
    {
        f64::from_bits(0xfff8_0000_0000_0000)
    } else {
        lhs + rhs
    }
}

/// Canonical binary64 NaN `0x7fff_ffff_ffff_ffff`: the f64 analogue of the CUDA
/// canonical `0x7fff_ffff`, returned by `max`/`min` when both operands are NaN.
pub const CUDA_CANONICAL_NAN_F64_BITS: u64 = 0x7fff_ffff_ffff_ffff;

/// CUDA `fmax` / PTX `max.f64`: one NaN loses to the other operand; two NaNs give the
/// canonical NaN [`CUDA_CANONICAL_NAN_F64_BITS`] (the [`cuda_f32_max`] rule);
/// `max(-0, +0) = +0`.
pub fn cuda_f64_max(lhs: f64, rhs: f64) -> f64 {
    match (lhs.is_nan(), rhs.is_nan()) {
        (true, true) => f64::from_bits(CUDA_CANONICAL_NAN_F64_BITS),
        (true, false) => rhs,
        (false, true) => lhs,
        (false, false) if lhs == 0.0 && rhs == 0.0 => {
            if lhs.is_sign_positive() || rhs.is_sign_positive() {
                0.0
            } else {
                -0.0
            }
        }
        (false, false) if lhs > rhs => lhs,
        (false, false) => rhs,
    }
}

/// CUDA `fmin` / PTX `min.f64`: one NaN loses; two NaNs give the canonical NaN
/// [`CUDA_CANONICAL_NAN_F64_BITS`]; `min(-0, +0) = -0`.
pub fn cuda_f64_min(lhs: f64, rhs: f64) -> f64 {
    match (lhs.is_nan(), rhs.is_nan()) {
        (true, true) => f64::from_bits(CUDA_CANONICAL_NAN_F64_BITS),
        (true, false) => rhs,
        (false, true) => lhs,
        (false, false) if lhs == 0.0 && rhs == 0.0 => {
            if lhs.is_sign_negative() || rhs.is_sign_negative() {
                -0.0
            } else {
                0.0
            }
        }
        (false, false) if lhs < rhs => lhs,
        (false, false) => rhs,
    }
}

/// PTX `ex2.approx.f32` representative: libm binary64 `exp2` rounded once to binary32
/// (RN), no FTZ. Not a GPU polynomial; a NaN result is pinned by [`pin_nan1_f32`]
/// (NaN input quieted, payload and sign kept; delta D8).
pub fn ptx_exp2_approx_f32(value: f32) -> f32 {
    // Use the software f64 implementation as a target-independent canonical
    // representative, then round once to binary32. This deliberately does not
    // replay any GPU architecture's approximation polynomial.
    pin_nan1_f32(value, libm::exp2(value as f64) as f32)
}

/// [`ptx_exp2_approx_f32`] with `.ftz`: subnormal input and result flush to signed zero.
pub fn ptx_exp2_approx_ftz_f32(value: f32) -> f32 {
    let value = flush_subnormal_f32(value);
    flush_subnormal_f32(ptx_exp2_approx_f32(value))
}

/// PTX `sin.approx{.ftz}.f32` representative: libm binary64 `sin` rounded to binary32;
/// `ftz` flushes subnormal input/result. NaN input gives it back quieted, `±inf` the
/// default NaN `0xffc0_0000` ([`pin_nan1_f32`], delta D8).
pub fn ptx_sin_approx_f32(value: f32, ftz: bool) -> f32 {
    let value = if ftz {
        flush_subnormal_f32(value)
    } else {
        value
    };
    // A high-accuracy software sine is a stable representative inside PTX's
    // architecture-dependent approximation bound.
    let result = pin_nan1_f32(value, libm::sin(value as f64) as f32);
    if ftz {
        flush_subnormal_f32(result)
    } else {
        result
    }
}

/// PTX `cos.approx{.ftz}.f32` representative: libm binary64 `cos` rounded to binary32;
/// `ftz` flushes subnormal input/result. NaN input gives it back quieted, `±inf` the
/// default NaN `0xffc0_0000` ([`pin_nan1_f32`], delta D8).
pub fn ptx_cos_approx_f32(value: f32, ftz: bool) -> f32 {
    let value = if ftz {
        flush_subnormal_f32(value)
    } else {
        value
    };
    // As with sine, do not pretend to reproduce one GPU's approximation
    // polynomial; keep one target-independent value within the PTX contract.
    let result = pin_nan1_f32(value, libm::cos(value as f64) as f32);
    if ftz {
        flush_subnormal_f32(result)
    } else {
        result
    }
}

/// PTX `ex2.approx.ftz.bf16x2`: [`ptx_exp2_approx_ftz_bf16`] per half (low = bits 0..16).
pub fn ptx_exp2_approx_ftz_bf16x2(value: u32) -> u32 {
    u32::from(ptx_exp2_approx_ftz_bf16(value as u16))
        | (u32::from(ptx_exp2_approx_ftz_bf16((value >> 16) as u16)) << 16)
}

/// PTX `ex2.approx.ftz.bf16`: bf16 widened exactly, [`ptx_exp2_approx_ftz_f32`], RN to
/// bfloat16; a subnormal bf16 result flushes to signed zero, NaN becomes `0x7fff`.
pub fn ptx_exp2_approx_ftz_bf16(value: u16) -> u16 {
    let result = ptx_exp2_approx_ftz_f32(bf16_bits_to_f32(value));
    let encoded = f32_to_bf16_bits(cuda_canonicalize_nan_f32(result));
    if encoded & 0x7f80 == 0 {
        encoded & 0x8000
    } else {
        encoded
    }
}

/// PTX `ex2.approx.f16`: exact widening (NaN canonical), [`ptx_exp2_approx_f32`], then
/// RN to binary16 (subnormals kept, NaN `0x7fff`).
pub fn ptx_exp2_approx_f16(value: u16) -> u16 {
    cuda_f32_to_fp16_bits(ptx_exp2_approx_f32(cuda_fp16_bits_to_f32(value)))
}

/// PTX `ex2.approx.f16x2`: [`ptx_exp2_approx_f16`] per half (low = bits 0..16).
pub fn ptx_exp2_approx_f16x2(value: u32) -> u32 {
    u32::from(ptx_exp2_approx_f16(value as u16))
        | (u32::from(ptx_exp2_approx_f16((value >> 16) as u16)) << 16)
}

/// PTX `lg2.approx.ftz.f32` representative: subnormal input flushed (so `lg2(±subnormal)
/// = -inf`), libm binary64 `log2` rounded to binary32; negative input gives the default
/// NaN `0xffc0_0000`, a NaN input itself quieted ([`pin_nan1_f32`], delta D8).
pub fn ptx_lg2_approx_ftz_f32(value: f32) -> f32 {
    let value = flush_subnormal_f32(value);
    // As with the exp2 adapter, use a target-independent software value as the
    // canonical representative instead of replaying an architecture-specific
    // approximation polynomial.
    flush_subnormal_f32(pin_nan1_f32(value, libm::log2(value as f64) as f32))
}

/// PTX `tanh.approx.f32` representative: libm binary64 `tanh` rounded to binary32, no
/// FTZ; a NaN input is returned quieted ([`pin_nan1_f32`], delta D8).
pub fn ptx_tanh_approx_f32(value: f32) -> f32 {
    // Use one target-independent software representative rather than replaying
    // an architecture-specific tanh.approx polynomial.
    pin_nan1_f32(value, libm::tanh(value as f64) as f32)
}

/// PTX `tanh.approx.f16`: exact widening, [`ptx_tanh_approx_f32`], RN to binary16
/// (NaN `0x7fff`).
pub fn ptx_tanh_approx_f16(value: u16) -> u16 {
    cuda_f32_to_fp16_bits(ptx_tanh_approx_f32(cuda_fp16_bits_to_f32(value)))
}

/// PTX `tanh.approx.f16x2`: [`ptx_tanh_approx_f16`] per half (low = bits 0..16).
pub fn ptx_tanh_approx_f16x2(value: u32) -> u32 {
    u32::from(ptx_tanh_approx_f16(value as u16))
        | (u32::from(ptx_tanh_approx_f16((value >> 16) as u16)) << 16)
}

/// PTX `tanh.approx.bf16`: exact widening, [`ptx_tanh_approx_f32`], RN to bfloat16
/// (subnormals kept, NaN `0x7fff`).
pub fn ptx_tanh_approx_bf16(value: u16) -> u16 {
    f32_to_bf16_bits(cuda_canonicalize_nan_f32(ptx_tanh_approx_f32(
        bf16_bits_to_f32(value),
    )))
}

/// PTX `tanh.approx.bf16x2`: [`ptx_tanh_approx_bf16`] per half (low = bits 0..16).
pub fn ptx_tanh_approx_bf16x2(value: u32) -> u32 {
    u32::from(ptx_tanh_approx_bf16(value as u16))
        | (u32::from(ptx_tanh_approx_bf16((value >> 16) as u16)) << 16)
}

/// PTX `rsqrt.approx.ftz.f32` representative: `1 / sqrt(x)` in binary32 (two RN roundings),
/// subnormal input/result flushed; `rsqrt(±0) = ±inf`; negative input gives the default
/// NaN `0xffc0_0000`, a NaN input itself quieted ([`pin_nan1_f32`], delta D8).
pub fn ptx_rsqrt_approx_ftz_f32(value: f32) -> f32 {
    let value = flush_subnormal_f32(value);
    flush_subnormal_f32(pin_nan1_f32(value, 1.0_f32 / value.sqrt()))
}

/// PTX `rsqrt.approx.f32` representative: `1 / sqrt(x)` in binary32, no FTZ; NaN as in
/// [`ptx_rsqrt_approx_ftz_f32`].
pub fn ptx_rsqrt_approx_f32(value: f32) -> f32 {
    pin_nan1_f32(value, 1.0_f32 / value.sqrt())
}

/// PTX `rcp.approx.ftz.f32` representative: binary32 `1 / x` (RN), subnormal input and
/// result flushed to signed zero; a NaN input is returned quieted ([`pin_nan1_f32`],
/// delta D8).
pub fn ptx_rcp_approx_ftz_f32(value: f32) -> f32 {
    let value = flush_subnormal_f32(value);
    flush_subnormal_f32(pin_nan1_f32(value, 1.0_f32 / value))
}

/// PTX `rcp.approx.f32` representative: binary32 `1 / x` (RN), no FTZ; NaN as in
/// [`ptx_rcp_approx_ftz_f32`].
pub fn ptx_rcp_approx_f32(value: f32) -> f32 {
    pin_nan1_f32(value, 1.0_f32 / value)
}

/// PTX `max{.ftz}{.NaN}.f32`: [`cuda_f32_max`]; `propagate_nan` returns `0x7fff_ffff` if
/// either operand is NaN; `ftz` flushes subnormal inputs and the result.
pub fn ptx_max_f32(lhs: f32, rhs: f32, ftz: bool, propagate_nan: bool) -> f32 {
    let lhs = if ftz { flush_subnormal_f32(lhs) } else { lhs };
    let rhs = if ftz { flush_subnormal_f32(rhs) } else { rhs };
    let result = if propagate_nan && (lhs.is_nan() || rhs.is_nan()) {
        cuda_canonical_nan_f32()
    } else {
        cuda_f32_max(lhs, rhs)
    };
    if ftz {
        flush_subnormal_f32(result)
    } else {
        result
    }
}

/// PTX `min{.ftz}{.NaN}.f32`: [`cuda_f32_min`]; `propagate_nan` returns `0x7fff_ffff` if
/// either operand is NaN; `ftz` flushes subnormal inputs and the result.
pub fn ptx_min_f32(lhs: f32, rhs: f32, ftz: bool, propagate_nan: bool) -> f32 {
    let lhs = if ftz { flush_subnormal_f32(lhs) } else { lhs };
    let rhs = if ftz { flush_subnormal_f32(rhs) } else { rhs };
    let result = if propagate_nan && (lhs.is_nan() || rhs.is_nan()) {
        cuda_canonical_nan_f32()
    } else {
        cuda_f32_min(lhs, rhs)
    };
    if ftz {
        flush_subnormal_f32(result)
    } else {
        result
    }
}

/// CUDA `__ffs` on a u32: 1-based index of the least-significant set bit, or 0
/// (frontend `emit/pure.rs` `cuda_ffs_u32`).
pub fn cuda_ffs_u32(value: u32) -> i32 {
    if value == 0 {
        0
    } else {
        value.trailing_zeros() as i32 + 1
    }
}

/// Bindings for CUDA helper intrinsics the frontend lowers straight to scalar
/// and packed-lane oracles (`frontend-rs/src/emit/pure.rs`).
pub(crate) const BINDINGS: &[crate::registry::Binding] = &[
    crate::registry::Binding {
        op: "tirx.cuda.bfloat1622float2",
        function: "cvt::unpack_bf16x2",
    },
    crate::registry::Binding {
        op: "tirx.cuda.bfloat162float",
        function: "cvt::bf16_bits_to_f32",
    },
    crate::registry::Binding {
        op: "tirx.cuda.bfloat162float",
        function: "scalar::cuda_canonicalize_nan_f32",
    },
    crate::registry::Binding {
        op: "tirx.cuda.half2float",
        function: "scalar::cuda_fp16_bits_to_f32",
    },
    crate::registry::Binding {
        op: "tirx.cuda.fadd2_rn",
        function: "cvt::add_f32x2",
    },
    crate::registry::Binding {
        op: "tirx.cuda.fmul2_rn",
        function: "cvt::mul_f32x2",
    },
    crate::registry::Binding {
        op: "tirx.cuda.fdividef",
        function: "scalar::div_f32_rn",
    },
    crate::registry::Binding {
        op: "tirx.cuda.ffs_u32",
        function: "scalar::cuda_ffs_u32",
    },
    crate::registry::Binding {
        op: "tirx.cuda.float22bfloat162_rn",
        function: "cvt::pack_bf16x2",
    },
    crate::registry::Binding {
        op: "tirx.cuda.float22bfloat162_rn_from_float2",
        function: "cvt::pack_bf16x2",
    },
    crate::registry::Binding {
        op: "tirx.cuda.float22half2",
        function: "scalar::cuda_f32_to_fp16_bits",
    },
    crate::registry::Binding {
        op: "tirx.cuda.half8tofloat8",
        function: "scalar::cuda_fp16_bits_to_f32",
    },
    crate::registry::Binding {
        op: "tirx.cuda.float8tohalf8",
        function: "scalar::cuda_f32_to_fp16_bits",
    },
    crate::registry::Binding {
        op: "tirx.cuda.make_float2",
        function: "cvt::make_float2",
    },
    crate::registry::Binding {
        op: "tirx.cuda.float2_x",
        function: "cvt::float2_x",
    },
    crate::registry::Binding {
        op: "tirx.cuda.float2_y",
        function: "cvt::float2_y",
    },
    crate::registry::Binding {
        op: "tirx.cuda.fp8x4_e4m3_from_float4",
        function: "cvt::fp8x4_e4m3_from_float4",
    },
];
