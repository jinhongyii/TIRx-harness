//! Per-element TIR ALU kernels. Values are raw element bits (`u128`, low
//! `dtype.bits()` significant). Semantics follow the legacy NumSim frontend
//! (`frontend-rs/src/{tables.rs, emit/expr.rs, emit/pure.rs}`), i.e. host
//! Rust executing C/TVM semantics:
//!
//! * integers: wrapping `+ - *` at the type width; `Div`/`Mod` truncate and
//!   are `checked_*` (zero divisor and `MIN / -1` are errors -> `Invalid`);
//!   `FloorDiv`/`FloorMod` floor (signed via `floor_div_i64` on the i64
//!   widening then wrapped back, unsigned `checked_*`); shifts are Rust
//!   `wrapping_shl/shr` (count taken modulo the width), `Shr` arithmetic for
//!   signed; `Min/Max` ordered by signedness.
//! * f32/f64: host IEEE round-to-nearest operators; `exp exp2 ln log2 sin
//!   cos tanh pow atan2` from the pure-Rust `libm` crate
//!   (`numsim_oplib::scalar::det`, delta D14: std would call the host's C
//!   library); `rsqrt = 1/sqrt`, `Round` = C `roundf`,
//!   ties away). `Min/Max` = `cuda_f32_min/max` / `cuda_f64_min/max`
//!   (NaN-ignoring, -0 < +0: device `min.f32/.f64`, which is also what the
//!   legacy tile reductions used; legacy scalar TIR f64 used Rust
//!   `f64::min/max`, whose signed-zero result is unspecified).
//! * f16/bf16: legacy carried them as f32: decode, compute in f32, round
//!   back with the legacy codec (RNE; for `+ - * /` and `sqrt` this equals the
//!   correctly rounded result since f32 has > 2p+2 bits). `Fma` is a single
//!   rounding (`numsim_oplib::scalar::low_fma_rn`). `Neg/Abs` are sign-bit ops.
//! * `Pred`: logical `And/Or/Xor/Not`; ordering compares are unsupported (as
//!   legacy). `B128`: bitwise `And/Or/Xor/Not`, `Eq/Ne`.

use super::super::{OpError, OpResult};
use super::convert::decode_f64;
use super::elem::{int_value, mask128};
use numsim_oplib::scalar::det;
use crate::dtype::Dtype;
use crate::program::{BinOp, CmpOp, TerOp, UnOp};
use numsim_oplib::cvt;
use numsim_oplib::scalar as sc;

#[inline]
fn f32v(x: u128) -> f32 {
    f32::from_bits(x as u32)
}
#[inline]
fn f64v(x: u128) -> f64 {
    f64::from_bits(x as u64)
}
#[inline]
fn of32(x: f32) -> u128 {
    u128::from(x.to_bits())
}
#[inline]
fn of64(x: f64) -> u128 {
    u128::from(x.to_bits())
}

/// f16/bf16 <-> f32 with the legacy codecs.
#[inline]
pub(super) fn half_dec(d: Dtype, x: u128) -> f32 {
    if d == Dtype::F16 {
        cvt::fp16_bits_to_f32(x as u16)
    } else {
        cvt::bf16_bits_to_f32(x as u16)
    }
}
#[inline]
fn half_enc(d: Dtype, y: f32) -> u128 {
    u128::from(if d == Dtype::F16 {
        cvt::f32_to_fp16_bits(y)
    } else {
        cvt::f32_to_bf16_bits(y)
    })
}

fn unsupported_un(op: UnOp, d: Dtype) -> OpError {
    OpError::unsupported(format!("unary {op:?} on {d}"))
}

/// f32 math shared by f32 and (via f32) f16/bf16.
#[inline]
fn f32_unary(op: UnOp, x: f32) -> Option<f32> {
    Some(match op {
        UnOp::Neg => -x,
        UnOp::Abs => x.abs(),
        UnOp::Sqrt => x.sqrt(),
        UnOp::Rsqrt => 1.0_f32 / x.sqrt(),
        UnOp::Exp => det::exp_f32(x),
        UnOp::Exp2 => det::exp2_f32(x),
        UnOp::Log => det::ln_f32(x),
        UnOp::Log2 => det::log2_f32(x),
        UnOp::Sin => det::sin_f32(x),
        UnOp::Cos => det::cos_f32(x),
        UnOp::Tanh => det::tanh_f32(x),
        UnOp::Floor => x.floor(),
        UnOp::Ceil => x.ceil(),
        UnOp::Round => x.round(),
        UnOp::Trunc => x.trunc(),
        _ => return None,
    })
}

#[inline]
fn f64_unary(op: UnOp, x: f64) -> Option<f64> {
    Some(match op {
        UnOp::Neg => -x,
        UnOp::Abs => x.abs(),
        UnOp::Sqrt => x.sqrt(),
        UnOp::Rsqrt => 1.0_f64 / x.sqrt(),
        UnOp::Exp => det::exp_f64(x),
        UnOp::Exp2 => det::exp2_f64(x),
        UnOp::Log => det::ln_f64(x),
        UnOp::Log2 => det::log2_f64(x),
        UnOp::Sin => det::sin_f64(x),
        UnOp::Cos => det::cos_f64(x),
        UnOp::Tanh => det::tanh_f64(x),
        UnOp::Floor => x.floor(),
        UnOp::Ceil => x.ceil(),
        UnOp::Round => x.round(),
        UnOp::Trunc => x.trunc(),
        _ => return None,
    })
}

/// `true` if the op yields a predicate (0/1) instead of a value of `ty`.
#[inline]
pub(super) fn unary_yields_pred(op: UnOp) -> bool {
    matches!(op, UnOp::IsNan | UnOp::IsInf | UnOp::IsFinite)
}

pub(super) fn unary(op: UnOp, d: Dtype, x: u128) -> OpResult<u128> {
    let w = d.bits();
    let m = mask128(w);
    if unary_yields_pred(op) {
        let (nan, inf) = if d.is_int() || d == Dtype::Pred {
            (false, false)
        } else {
            let f = decode_f64(d, x as u64).ok_or_else(|| unsupported_un(op, d))?;
            (f.is_nan(), f.is_infinite())
        };
        return Ok(match op {
            UnOp::IsNan => nan,
            UnOp::IsInf => inf,
            _ => !nan && !inf,
        } as u128);
    }
    match d {
        Dtype::Pred => match op {
            UnOp::Not | UnOp::BitNot => Ok((x & 1) ^ 1),
            _ => Err(unsupported_un(op, d)),
        },
        Dtype::B128 => match op {
            UnOp::BitNot | UnOp::Not => Ok(!x),
            _ => Err(unsupported_un(op, d)),
        },
        _ if d.is_int() => {
            let v = int_value(d, x);
            Ok(match op {
                UnOp::Neg => (v.wrapping_neg() as u128) & m,
                UnOp::Abs => (v.wrapping_abs() as u128) & m,
                // `BitNot` is `~x`; `Not` on an integer (pre-contract-batch
                // lowering spelled `~x` as `Not`) keeps the same meaning.
                UnOp::BitNot | UnOp::Not => !x & m,
                UnOp::Popcount => u128::from((x & m).count_ones()),
                UnOp::Clz => u128::from((x & m).leading_zeros() - (128 - w)),
                UnOp::Floor | UnOp::Ceil | UnOp::Round | UnOp::Trunc => x & m,
                _ => return Err(unsupported_un(op, d)),
            })
        }
        Dtype::F32 => match op {
            UnOp::Neg => Ok((x ^ 0x8000_0000) & m),
            _ => f32_unary(op, f32v(x)).map(of32).ok_or_else(|| unsupported_un(op, d)),
        },
        Dtype::F64 => match op {
            UnOp::Neg => Ok((x ^ (1 << 63)) & m),
            _ => f64_unary(op, f64v(x)).map(of64).ok_or_else(|| unsupported_un(op, d)),
        },
        Dtype::F16 | Dtype::BF16 => match op {
            UnOp::Neg => Ok((x ^ 0x8000) & m),
            UnOp::Abs => Ok(x & 0x7fff),
            _ => f32_unary(op, half_dec(d, x)).map(|y| half_enc(d, y)).ok_or_else(|| unsupported_un(op, d)),
        },
        _ => Err(unsupported_un(op, d)),
    }
}

fn unsupported_bin(op: BinOp, d: Dtype) -> OpError {
    OpError::unsupported(format!("binary {op:?} on {d}"))
}

#[inline]
fn f32_binary(op: BinOp, a: f32, b: f32) -> Option<f32> {
    Some(match op {
        BinOp::Add => sc::pin_nan2_f32(a, b, a + b),
        BinOp::Sub => sc::pin_nan2_f32(a, b, a - b),
        BinOp::Mul => sc::pin_nan2_f32(a, b, a * b),
        BinOp::Div => sc::pin_nan2_f32(a, b, a / b),
        BinOp::Min => sc::cuda_f32_min(a, b),
        BinOp::Max => sc::cuda_f32_max(a, b),
        BinOp::Pow => det::pow_f32(a, b),
        BinOp::Atan2 => det::atan2_f32(a, b),
        BinOp::Copysign => a.copysign(b),
        _ => return None,
    })
}

#[inline]
fn f64_binary(op: BinOp, a: f64, b: f64) -> Option<f64> {
    Some(match op {
        BinOp::Add => sc::pin_nan2_f64(a, b, a + b),
        BinOp::Sub => sc::pin_nan2_f64(a, b, a - b),
        BinOp::Mul => sc::pin_nan2_f64(a, b, a * b),
        BinOp::Div => sc::pin_nan2_f64(a, b, a / b),
        BinOp::Min => sc::cuda_f64_min(a, b),
        BinOp::Max => sc::cuda_f64_max(a, b),
        BinOp::Pow => det::pow_f64(a, b),
        BinOp::Atan2 => det::atan2_f64(a, b),
        BinOp::Copysign => a.copysign(b),
        _ => return None,
    })
}

fn int_binary(op: BinOp, d: Dtype, x: u128, y: u128) -> OpResult<u128> {
    let w = d.bits();
    let m = mask128(w);
    let signed = d.is_signed_int();
    let (a, b) = (int_value(d, x), int_value(d, y));
    let (lo, _) = super::elem::int_range(d);
    let wrap = |v: i128| (v as u128) & m;
    let shift = (y as u32) & (w - 1);
    Ok(match op {
        BinOp::Add => wrap(a.wrapping_add(b)),
        BinOp::Sub => wrap(a.wrapping_sub(b)),
        BinOp::Mul => wrap(a.wrapping_mul(b)),
        BinOp::Div | BinOp::Mod => {
            let what = if op == BinOp::Div { "division" } else { "remainder" };
            if b == 0 || (signed && a == lo && b == -1) {
                return Err(OpError::invalid(format!("invalid truncating {what}: {a} / {b} ({d})")));
            }
            wrap(if op == BinOp::Div { a / b } else { a % b })
        }
        BinOp::FloorDiv | BinOp::FloorMod => {
            if signed {
                // Legacy: `floor_*_i64(a as i64, b as i64)? as T`.
                let (a, b) = (a as i64, b as i64);
                let r = if op == BinOp::FloorDiv {
                    sc::floor_div_i64(a, b)?
                } else {
                    sc::floor_mod_i64(a, b)?
                };
                wrap(r as i128)
            } else {
                if b == 0 {
                    return Err(OpError::invalid(format!("invalid unsigned floor division by zero ({d})")));
                }
                wrap(if op == BinOp::FloorDiv { a / b } else { a % b })
            }
        }
        BinOp::Min => wrap(a.min(b)),
        BinOp::Max => wrap(a.max(b)),
        BinOp::And => x & y & m,
        BinOp::Or => (x | y) & m,
        BinOp::Xor => (x ^ y) & m,
        BinOp::Shl => ((x & m) << shift) & m,
        BinOp::Shr => {
            if signed {
                wrap(a >> shift)
            } else {
                (x & m) >> shift
            }
        }
        BinOp::Pow | BinOp::Atan2 | BinOp::Copysign => return Err(unsupported_bin(op, d)),
    })
}

pub(super) fn binary(op: BinOp, d: Dtype, x: u128, y: u128) -> OpResult<u128> {
    match d {
        Dtype::Pred => match op {
            BinOp::And => Ok(x & y & 1),
            BinOp::Or => Ok((x | y) & 1),
            BinOp::Xor => Ok((x ^ y) & 1),
            _ => Err(unsupported_bin(op, d)),
        },
        Dtype::B128 => match op {
            BinOp::And => Ok(x & y),
            BinOp::Or => Ok(x | y),
            BinOp::Xor => Ok(x ^ y),
            _ => Err(unsupported_bin(op, d)),
        },
        _ if d.is_int() => int_binary(op, d, x, y),
        Dtype::F32 => f32_binary(op, f32v(x), f32v(y)).map(of32).ok_or_else(|| unsupported_bin(op, d)),
        Dtype::F64 => f64_binary(op, f64v(x), f64v(y)).map(of64).ok_or_else(|| unsupported_bin(op, d)),
        Dtype::F16 | Dtype::BF16 => f32_binary(op, half_dec(d, x), half_dec(d, y))
            .map(|r| half_enc(d, r))
            .ok_or_else(|| unsupported_bin(op, d)),
        _ => Err(unsupported_bin(op, d)),
    }
}

pub(super) fn ternary(op: TerOp, d: Dtype, x: u128, y: u128, z: u128) -> OpResult<u128> {
    match op {
        TerOp::Fma => match d {
            Dtype::F32 => Ok(of32(sc::fma_f32_rn(f32v(x), f32v(y), f32v(z)))),
            Dtype::F64 => Ok(of64(sc::host_fma_f64(f64v(x), f64v(y), f64v(z)))),
            Dtype::F16 => Ok(u128::from(sc::fma_f16_bits_rn(x as u16, y as u16, z as u16))),
            Dtype::BF16 => Ok(u128::from(sc::fma_bf16_bits_rn(x as u16, y as u16, z as u16))),
            _ if d.is_int() => {
                let (a, b, c) = (int_value(d, x), int_value(d, y), int_value(d, z));
                Ok((a.wrapping_mul(b).wrapping_add(c) as u128) & mask128(d.bits()))
            }
            _ => Err(OpError::unsupported(format!("ternary {op:?} on {d}"))),
        },
    }
}

pub(super) fn compare(op: CmpOp, d: Dtype, x: u128, y: u128) -> OpResult<bool> {
    use std::cmp::Ordering;
    let ord: Option<Ordering> = match d {
        Dtype::Pred | Dtype::B128 => {
            let (a, b) = if d == Dtype::Pred { (x & 1, y & 1) } else { (x, y) };
            return match op {
                CmpOp::Eq => Ok(a == b),
                CmpOp::Ne => Ok(a != b),
                _ => Err(OpError::unsupported(format!("compare {op:?} on {d}"))),
            };
        }
        _ if d.is_int() => Some(int_value(d, x).cmp(&int_value(d, y))),
        _ => {
            let f = |v: u128| match d {
                Dtype::F16 | Dtype::BF16 => Ok(f64::from(half_dec(d, v))),
                _ => decode_f64(d, v as u64).ok_or_else(|| OpError::unsupported(format!("compare on {d}"))),
            };
            f(x)?.partial_cmp(&f(y)?)
        }
    };
    Ok(match ord {
        None => op == CmpOp::Ne,
        Some(o) => match op {
            CmpOp::Eq => o == Ordering::Equal,
            CmpOp::Ne => o != Ordering::Equal,
            CmpOp::Lt => o == Ordering::Less,
            CmpOp::Le => o != Ordering::Greater,
            CmpOp::Gt => o == Ordering::Greater,
            CmpOp::Ge => o != Ordering::Less,
        },
    })
}
