//! Element conversions for `Instr::Cast` / `convert_bits` and the
//! `FloatScalar::from_f64` hook for `f32`.
//!
//! `Rounding::Default` without `sat` reproduces what the legacy NumSim
//! frontend emitted for a TIR `Cast` (`emit/expr.rs::cast_atom`): low-precision
//! floats are carried as `f32`, so a cast *to* f16/bf16/fp8 is
//! `encode(x as f32)` (double rounding from f64 / wide ints, as legacy), a cast
//! *from* them decodes to `f32` first; float->int is Rust `as` (truncation
//! toward zero, saturating, NaN -> 0); int->float is Rust `as` (RNE);
//! int->int wraps/sign-extends; `bool` is `x != 0` / 0-1.
//!
//! Explicit rounding (`Rn/Rz/Rm/Rp/Rna`) is a single correctly rounded step
//! from the exact source value (PTX `cvt` semantics); `sat` saturates float
//! results to finite (`.satfinite`) and integer results to the destination
//! range (`.sat`). `Rs` needs random bits and is `Unsupported`.

use super::super::{OpError, OpResult};
use super::elem::{int_range, int_value, mask128};
use crate::dtype::Dtype;
use crate::program::Rounding;
use numsim_oplib::cvt::{self as cvt, PtxFloatRounding};
use numsim_oplib::scalar::LowPrecisionFormat;

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// Exact value of a narrow OCP format, `None` for formats without a scalar
/// codec here.
#[inline]
fn narrow_format(d: Dtype) -> Option<cvt::NarrowFloatFormat> {
    Some(match d {
        Dtype::E5M2 => cvt::FLOAT8_E5M2,
        Dtype::E2M3 => cvt::FLOAT6_E2M3,
        Dtype::E3M2 => cvt::FLOAT6_E3M2,
        Dtype::E2M1 => cvt::FLOAT4_E2M1,
        Dtype::UE5M3 => cvt::FLOAT8_UE5M3,
        _ => return None,
    })
}

/// Decode a float dtype whose values all fit `f32` exactly (everything but
/// F64) the way legacy did (`*_bits_to_f32`). `None`: no codec.
#[inline]
pub(super) fn decode_f32(d: Dtype, bits: u64) -> Option<f32> {
    Some(match d {
        Dtype::F32 | Dtype::TF32 => f32::from_bits(bits as u32),
        Dtype::F16 => cvt::fp16_bits_to_f32(bits as u16),
        Dtype::BF16 => cvt::bf16_bits_to_f32(bits as u16),
        Dtype::E4M3 => cvt::float8_e4m3fn_bits_to_f32(bits as u8),
        Dtype::UE8M0 => cvt::float8_e8m0fnu_bits_to_f32(bits as u8),
        _ => {
            let format = narrow_format(d)?;
            cvt::narrow_float_bits_to_f32_checked(bits as u8, format).unwrap_or(f32::NAN)
        }
    })
}

/// Exact value of any float dtype as f64 (NaN stays NaN). `None`: no codec.
#[inline]
pub(super) fn decode_f64(d: Dtype, bits: u64) -> Option<f64> {
    if d == Dtype::F64 {
        return Some(f64::from_bits(bits));
    }
    decode_f32(d, bits).map(f64::from)
}

// ---------------------------------------------------------------------------
// Rounding helpers
// ---------------------------------------------------------------------------

#[inline]
fn ptx_rounding(rnd: Rounding) -> Option<PtxFloatRounding> {
    Some(match rnd {
        Rounding::Default | Rounding::Rn => PtxFloatRounding::NearestEven,
        Rounding::Rz => PtxFloatRounding::Zero,
        Rounding::Rm => PtxFloatRounding::NegativeInfinity,
        Rounding::Rp => PtxFloatRounding::PositiveInfinity,
        Rounding::Rna | Rounding::Rs => return None,
    })
}

/// Round-to-nearest, ties away from zero, built from the directed modes of a
/// binary format: `round(v, mode)` returns raw bits (sign in `sign_bit`),
/// `decode` their exact value. Requires `v` finite.
fn round_ties_away(v: f64, sign_bit: u64, round: impl Fn(PtxFloatRounding) -> u64, decode: impl Fn(u64) -> f64) -> u64 {
    let toward = round(PtxFloatRounding::Zero);
    let away = round(if v < 0.0 {
        PtxFloatRounding::NegativeInfinity
    } else {
        PtxFloatRounding::PositiveInfinity
    });
    if toward == away {
        return toward;
    }
    let low = decode(toward).abs();
    let high = decode(away).abs();
    let mid = if high.is_finite() {
        (low + high) / 2.0
    } else {
        // Overflow boundary: greatest finite + half its ulp.
        let below = decode((toward & !sign_bit) - 1).abs();
        low + (low - below) / 2.0
    };
    if v.abs() >= mid {
        away
    } else {
        toward
    }
}

/// `FloatScalar::from_f64` for `f32`: one correctly rounded step from the
/// exact f64 (`Default`/`Rn` nearest-even, `Rz/Rm/Rp` directed, `Rna` ties
/// away), NaN keeps sign and high payload (quieted). `sat` clamps an
/// infinite result to the greatest finite magnitude. `Rs` (no f32 form
/// exists; no random bits here) rounds to nearest-even.
pub(in crate::oplib) fn f32_from_f64(x: f64, rnd: Rounding, sat: bool) -> f32 {
    let r = if x.is_nan() {
        cvt::ptx_cvt_f64_to_f32(x, PtxFloatRounding::NearestEven, false)
    } else if rnd == Rounding::Rna {
        let bits = round_ties_away(
            x,
            0x8000_0000,
            |m| u64::from(cvt::ptx_cvt_f64_to_f32(x, m, false).to_bits()),
            |b| f64::from(f32::from_bits(b as u32)),
        );
        f32::from_bits(bits as u32)
    } else {
        let mode = ptx_rounding(rnd).unwrap_or(PtxFloatRounding::NearestEven);
        cvt::ptx_cvt_f64_to_f32(x, mode, false)
    };
    if sat && r.is_infinite() {
        f32::MAX.copysign(r)
    } else {
        r
    }
}

fn low_format(d: Dtype) -> LowPrecisionFormat {
    if d == Dtype::F16 {
        LowPrecisionFormat::F16
    } else {
        LowPrecisionFormat::Bf16
    }
}

/// f64 -> f16/bf16 bits, single rounding per `rnd` (not Rs).
fn low_from_f64(x: f64, d: Dtype, rnd: Rounding, sat: bool) -> u16 {
    let format = low_format(d);
    let decode = |b: u64| match format {
        LowPrecisionFormat::F16 => f64::from(cvt::fp16_bits_to_f32(b as u16)),
        LowPrecisionFormat::Bf16 => f64::from(cvt::bf16_bits_to_f32(b as u16)),
    };
    let r = if rnd == Rounding::Rna && !x.is_nan() && x.is_finite() {
        round_ties_away(x, 0x8000, |m| u64::from(cvt::ptx_cvt_f64_to_low(x, m, format)), decode) as u16
    } else {
        let mode = ptx_rounding(rnd).unwrap_or(PtxFloatRounding::NearestEven);
        cvt::ptx_cvt_f64_to_low(x, mode, format)
    };
    satfinite_low(r, d, sat)
}

#[inline]
fn satfinite_low(bits: u16, d: Dtype, sat: bool) -> u16 {
    let (inf, max) = if d == Dtype::F16 { (0x7c00, 0x7bff) } else { (0x7f80, 0x7f7f) };
    if sat && bits & 0x7fff == inf {
        (bits & 0x8000) | max
    } else {
        bits
    }
}

#[inline]
fn satfinite_f32(x: f32, sat: bool) -> f32 {
    if sat && x.is_infinite() {
        f32::MAX.copysign(x)
    } else {
        x
    }
}

#[inline]
fn satfinite_f64(x: f64, sat: bool) -> f64 {
    if sat && x.is_infinite() {
        f64::MAX.copysign(x)
    } else {
        x
    }
}

/// Integer -> f64 rounded per `rnd` (`odd` = round-to-odd, used as an
/// innocuous intermediate for a narrower second rounding).
fn int_to_f64(v: i128, rnd: Rounding, odd: bool) -> f64 {
    let neg = v < 0;
    let m = v.unsigned_abs();
    if m < (1u128 << 53) {
        let f = m as f64;
        return if neg { -f } else { f };
    }
    let len = 128 - m.leading_zeros();
    let s = len - 53;
    let mut q = m >> s;
    let rem = m & mask128(s);
    let half = 1u128 << (s - 1);
    let inc = if odd {
        false
    } else {
        match rnd {
            Rounding::Rz => false,
            Rounding::Rp => rem != 0 && !neg,
            Rounding::Rm => rem != 0 && neg,
            Rounding::Rna => rem >= half,
            _ => rem > half || (rem == half && q & 1 == 1),
        }
    };
    if odd && rem != 0 {
        q |= 1;
    }
    if inc {
        q += 1;
    }
    let f = (q as f64) * 2f64.powi(s as i32);
    if neg {
        -f
    } else {
        f
    }
}

/// Float (exact f64) -> integer of `to`, rounding the value per `rnd`
/// (Default = truncate, like C / Rust `as`), saturating, NaN -> 0.
fn float_to_int(v: f64, to: Dtype, rnd: Rounding) -> u128 {
    if v.is_nan() {
        return 0;
    }
    let r = match rnd {
        Rounding::Default | Rounding::Rz => v.trunc(),
        Rounding::Rn => v.round_ties_even(),
        Rounding::Rna => v.round(),
        Rounding::Rm => v.floor(),
        Rounding::Rp | Rounding::Rs => v.ceil(),
    };
    let (lo, hi) = int_range(to);
    // f64 `as i128` saturates; every int range here fits i128.
    let i = (r as i128).clamp(lo, hi);
    (i as u128) & mask128(to.bits())
}

/// Encode an f32 value into a narrow (fp8/fp6/fp4) dtype, RN satfinite (the
/// CUDA `__nv_fp8_*`/`__nv_fp4_*` constructor semantics). E4M3 uses the
/// legacy codec, UE8M0 the legacy nearest-power-of-two codec.
fn narrow_from_f32(y: f32, to: Dtype) -> OpResult<u128> {
    Ok(match to {
        Dtype::E4M3 => u128::from(cvt::f32_to_float8_e4m3fn_bits(y)),
        Dtype::UE8M0 => u128::from(cvt::f32_to_float8_e8m0fnu_bits(y)),
        Dtype::E5M2 | Dtype::E2M3 | Dtype::E3M2 | Dtype::E2M1 => {
            let format = narrow_format(to).expect("narrow format");
            u128::from(cvt::f32_to_narrow_float_bits_rn_satfinite(y, format)) & mask128(to.bits())
        }
        _ => return Err(OpError::unsupported(format!("cast to {to}"))),
    })
}

fn is_narrow(d: Dtype) -> bool {
    matches!(d, Dtype::E4M3 | Dtype::E5M2 | Dtype::UE8M0 | Dtype::E2M3 | Dtype::E3M2 | Dtype::E2M1)
}

/// Value of the source element.
enum Src {
    Int(i128),
    Float(f64),
}

/// Convert one element `x` (raw bits of `from`) to raw bits of `to`.
pub(super) fn cast_elem(from: Dtype, to: Dtype, rnd: Rounding, sat: bool, x: u128) -> OpResult<u128> {
    let unsupported = || OpError::unsupported(format!("cast {from} -> {to} (rnd {rnd:?}, sat {sat})"));
    if rnd == Rounding::Rs {
        return Err(unsupported());
    }
    // A carried f32 (legacy f32 carrier of a TIR half expression) casts as
    // the f32 it is; casting back to f16/bf16 rounds it.
    if matches!(from, Dtype::F16 | Dtype::BF16) {
        if let Some(v) = super::alu::half_carry(x) {
            return cast_elem(Dtype::F32, to, rnd, sat, u128::from(v.to_bits()));
        }
    }
    if from == to {
        // Same type: the value is already representable.
        return Ok(match to {
            Dtype::F32 => u128::from(satfinite_f32(f32::from_bits(x as u32), sat).to_bits()),
            Dtype::F64 => u128::from(satfinite_f64(f64::from_bits(x as u64), sat).to_bits()),
            Dtype::F16 | Dtype::BF16 => u128::from(satfinite_low(x as u16, to, sat)),
            _ => x & mask128(to.bits()),
        });
    }
    if from == Dtype::B128 || to == Dtype::B128 {
        return Err(unsupported());
    }
    // ---- source value
    let src = if from == Dtype::Pred {
        Src::Int((x & 1) as i128)
    } else if from.is_int() {
        Src::Int(int_value(from, x))
    } else {
        Src::Float(decode_f64(from, x as u64).ok_or_else(unsupported)?)
    };
    // ---- bool destination: x != 0 (NaN != 0)
    if to == Dtype::Pred {
        if sat {
            return Err(unsupported());
        }
        return Ok(match src {
            Src::Int(v) => (v != 0) as u128,
            Src::Float(f) => (f != 0.0) as u128,
        });
    }
    // ---- integer destination
    if to.is_int() {
        let (lo, hi) = int_range(to);
        return match src {
            Src::Int(v) => {
                if rnd != Rounding::Default {
                    return Err(unsupported());
                }
                let v = if sat { v.clamp(lo, hi) } else { v };
                Ok((v as u128) & mask128(to.bits()))
            }
            Src::Float(f) => Ok(float_to_int(f, to, rnd)),
        };
    }
    // ---- float destination
    if rnd == Rounding::Default {
        // Legacy: everything except f64 travels as f32.
        let wide = match src {
            Src::Int(v) => {
                if to == Dtype::F64 {
                    Src::Float(v as f64)
                } else {
                    Src::Int(v)
                }
            }
            s => s,
        };
        return match to {
            Dtype::F64 => match wide {
                Src::Float(f) => Ok(u128::from(satfinite_f64(f, sat).to_bits())),
                Src::Int(v) => Ok(u128::from(satfinite_f64(v as f64, sat).to_bits())),
            },
            _ => {
                let y: f32 = match wide {
                    Src::Int(v) => v as f32,
                    Src::Float(f) => {
                        if from == Dtype::F64 {
                            f as f32
                        } else {
                            // Exact (and the legacy decoded f32 itself).
                            decode_f32(from, x as u64).ok_or_else(unsupported)?
                        }
                    }
                };
                match to {
                    Dtype::F32 => Ok(u128::from(satfinite_f32(y, sat).to_bits())),
                    Dtype::F16 => Ok(u128::from(satfinite_low(cvt::f32_to_fp16_bits(y), to, sat))),
                    Dtype::BF16 => Ok(u128::from(satfinite_low(cvt::f32_to_bf16_bits(y), to, sat))),
                    d if is_narrow(d) => narrow_from_f32(y, d),
                    _ => Err(unsupported()),
                }
            }
        };
    }
    // Explicit rounding: one rounding step from the exact value.
    match to {
        Dtype::F64 => {
            let f = match src {
                Src::Int(v) => int_to_f64(v, rnd, false),
                Src::Float(f) => f,
            };
            Ok(u128::from(satfinite_f64(f, sat).to_bits()))
        }
        Dtype::F32 | Dtype::F16 | Dtype::BF16 => {
            // Round-to-odd to 53 bits is innocuous before a <= 24-bit rounding.
            let f = match src {
                Src::Int(v) => int_to_f64(v, rnd, true),
                Src::Float(f) => f,
            };
            Ok(if to == Dtype::F32 {
                u128::from(f32_from_f64(f, rnd, sat).to_bits())
            } else {
                u128::from(low_from_f64(f, to, rnd, sat))
            })
        }
        Dtype::UE8M0 if matches!(rnd, Rounding::Rz | Rounding::Rp) => {
            let Src::Float(f) = src else { return Err(unsupported()) };
            if from == Dtype::F64 {
                return Err(unsupported());
            }
            let y = f as f32;
            Ok(u128::from(match (rnd == Rounding::Rp, sat) {
                (false, false) => cvt::f32_to_float8_e8m0fnu_bits_rounded::<false, false>(y),
                (false, true) => cvt::f32_to_float8_e8m0fnu_bits_rounded::<false, true>(y),
                (true, false) => cvt::f32_to_float8_e8m0fnu_bits_rounded::<true, false>(y),
                (true, true) => cvt::f32_to_float8_e8m0fnu_bits_rounded::<true, true>(y),
            }))
        }
        Dtype::E4M3 | Dtype::E2M3 | Dtype::E3M2 | Dtype::E2M1 | Dtype::E5M2 if rnd == Rounding::Rn => {
            // PTX only has `cvt.rn.satfinite` for these (from f32/f16/bf16
            // values); e5m2 has infinities, so require `sat` for it.
            if to == Dtype::E5M2 && !sat {
                return Err(unsupported());
            }
            let Src::Float(f) = src else { return Err(unsupported()) };
            if from == Dtype::F64 {
                return Err(unsupported());
            }
            narrow_from_f32(f as f32, to)
        }
        _ => Err(unsupported()),
    }
}
