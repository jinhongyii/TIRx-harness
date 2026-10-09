//! Integer arithmetic: add/sub/mul/mad (lo, hi, sat, wide, 24-bit, packed
//! 16x2), sad, div/rem, neg/abs, dp2a/dp4a and integer min/max (legacy
//! `integer_arithmetic_variants!`, `IntegerArithmetic`, `MulWide`,
//! `MadWide`, `Mul24`, `Mad24`, `Dp2a`, `Dp4a`, `IntegerMinMax`).
//!
//! Sources are read at the PTX width from the low bits of their carriers;
//! signed results are sign-extended into wider carriers by `Operands::put`.

use super::{flat, flat_exact, int_type, map, try_map, Md};
use crate::oplib::ptx::{Operands, Resolved};
use crate::oplib::{OpResult, PtxIo};
use numsim_oplib::arith;

/// Expand `$body` with `$t` bound to the Rust carrier of `(bits, signed)`
/// (16/32/64 only) and `$v` converting a raw source to `$t`.
macro_rules! with_int {
    ($m:expr, $bits:expr, $signed:expr, |$t:ident| $body:expr) => {
        match ($bits, $signed) {
            (16, false) => {
                type $t = u16;
                $body
            }
            (16, true) => {
                type $t = i16;
                $body
            }
            (32, false) => {
                type $t = u32;
                $body
            }
            (32, true) => {
                type $t = i32;
                $body
            }
            (64, false) => {
                type $t = u64;
                $body
            }
            (64, true) => {
                type $t = i64;
                $body
            }
            _ => $m.unsupported("integer width"),
        }
    };
}

pub(super) fn resolve(op: &str, m: &Md, ops: &Operands) -> OpResult<Resolved> {
    let ty = m.get("type");
    let sat = m.has("sat");
    match op {
        "add_int" | "sub_int" => {
            let add = op == "add_int";
            if ty == "u16x2" || ty == "s16x2" {
                if sat || !add {
                    return m.unsupported("packed 16x2 form");
                }
                return map::<2, _>(m, ops, 32, false, |[a, b]| {
                    u64::from(arith::add_16x2(a as u32, b as u32))
                });
            }
            let (bits, signed) = int_type(m, ty)?;
            if sat {
                if ty != "s32" {
                    return m.unsupported(".sat applies only to .s32");
                }
                return map::<2, _>(m, ops, 32, true, move |[a, b]| {
                    let (a, b) = (a as u32 as i32, b as u32 as i32);
                    (if add {
                        arith::add_sat_s32(a, b)
                    } else {
                        arith::sub_sat_s32(a, b)
                    }) as u32 as u64
                });
            }
            if add
                && bits == 32
                && (if signed {
                    flat_exact(ops, 1, 2, 32)
                } else {
                    flat(ops, 1, 2, 32)
                })
            {
                return Ok(Resolved::Direct(add_u32_direct));
            }
            with_int!(m, bits, signed, |T| map::<2, _>(
                m,
                ops,
                bits,
                signed,
                move |[a, b]| {
                    let (a, b) = (a as T, b as T);
                    (if add {
                        arith::add_int::<T>(a, b)
                    } else {
                        arith::sub_int::<T>(a, b)
                    }) as u64
                }
            ))
        }
        "mul_int" | "mad_int" => {
            let (bits, signed) = int_type(m, ty)?;
            let high = m.get("mode") == "hi";
            if op == "mul_int" {
                return with_int!(m, bits, signed, |T| map::<2, _>(
                    m,
                    ops,
                    bits,
                    signed,
                    move |[a, b]| {
                        let (a, b) = (a as T, b as T);
                        (if high {
                            arith::mul_hi_int::<T>(a, b)
                        } else {
                            arith::mul_lo_int::<T>(a, b)
                        }) as u64
                    }
                ));
            }
            if sat {
                if !(high && ty == "s32") {
                    return m.unsupported(".sat applies only to mad.hi.s32");
                }
                return map::<3, _>(m, ops, 32, true, |[a, b, c]| {
                    arith::mad_hi_sat_s32(a as u32 as i32, b as u32 as i32, c as u32 as i32) as u32
                        as u64
                });
            }
            with_int!(m, bits, signed, |T| map::<3, _>(
                m,
                ops,
                bits,
                signed,
                move |[a, b, c]| {
                    let (a, b, c) = (a as T, b as T, c as T);
                    (if high {
                        arith::mad_hi_int::<T>(a, b, c)
                    } else {
                        arith::mad_lo_int::<T>(a, b, c)
                    }) as u64
                }
            ))
        }
        "mul_wide" | "mad_wide" => {
            let mad = op == "mad_wide";
            let (bits, signed) = int_type(m, ty)?;
            let wide = bits * 2;
            match (ty, mad) {
                ("s16", false) => map::<2, _>(m, ops, wide, true, |[a, b]| {
                    arith::mul_wide_s16(a as i16, b as i16) as u32 as u64
                }),
                ("u16", false) => map::<2, _>(m, ops, wide, false, |[a, b]| {
                    u64::from(arith::mul_wide_u16(a as u16, b as u16))
                }),
                ("s32", false) => map::<2, _>(m, ops, wide, true, |[a, b]| {
                    arith::mul_wide_s32(a as i32, b as i32) as u64
                }),
                ("u32", false) => map::<2, _>(m, ops, wide, false, |[a, b]| {
                    arith::mul_wide_u32(a as u32, b as u32)
                }),
                ("s16", true) => map::<3, _>(m, ops, wide, true, |[a, b, c]| {
                    arith::mad_wide_s16(a as i16, b as i16, c as i32) as u32 as u64
                }),
                ("u16", true) => map::<3, _>(m, ops, wide, false, |[a, b, c]| {
                    u64::from(arith::mad_wide_u16(a as u16, b as u16, c as u32))
                }),
                ("s32", true) => map::<3, _>(m, ops, wide, true, |[a, b, c]| {
                    arith::mad_wide_s32(a as i32, b as i32, c as i64) as u64
                }),
                ("u32", true) => map::<3, _>(m, ops, wide, false, |[a, b, c]| {
                    arith::mad_wide_u32(a as u32, b as u32, c)
                }),
                _ => {
                    let _ = signed;
                    m.unsupported("wide type")
                }
            }
        }
        "mul24" | "mad24" => {
            let high = m.get("mode") == "hi";
            let signed = ty == "s32";
            if op == "mul24" {
                return if signed {
                    map::<2, _>(m, ops, 32, true, move |[a, b]| {
                        arith::mul24_s32(a as i32, b as i32, high) as u32 as u64
                    })
                } else {
                    map::<2, _>(m, ops, 32, false, move |[a, b]| {
                        u64::from(arith::mul24_u32(a as u32, b as u32, high))
                    })
                };
            }
            if sat {
                if !(high && signed) {
                    return m.unsupported(".sat applies only to mad24.hi.s32");
                }
                return map::<3, _>(m, ops, 32, true, |[a, b, c]| {
                    arith::mad24_hi_sat_s32(a as i32, b as i32, c as i32) as u32 as u64
                });
            }
            if signed {
                map::<3, _>(m, ops, 32, true, move |[a, b, c]| {
                    arith::mad24_s32(a as i32, b as i32, c as i32, high) as u32 as u64
                })
            } else {
                map::<3, _>(m, ops, 32, false, move |[a, b, c]| {
                    u64::from(arith::mad24_u32(a as u32, b as u32, c as u32, high))
                })
            }
        }
        "sad" => {
            let (bits, signed) = int_type(m, ty)?;
            with_int!(m, bits, signed, |T| map::<3, _>(
                m,
                ops,
                bits,
                signed,
                |[a, b, c]| { arith::sad_int::<T>(a as T, b as T, c as T) as u64 }
            ))
        }
        "div" | "rem" => {
            let (bits, signed) = int_type(m, ty)?;
            let div = op == "div";
            with_int!(m, bits, signed, |T| try_map::<2, _>(
                m,
                ops,
                bits,
                signed,
                move |[a, b]| {
                    let (a, b) = (a as T, b as T);
                    Ok((if div {
                        arith::div_int::<T>(a, b)?
                    } else {
                        arith::rem_int::<T>(a, b)?
                    }) as u64)
                }
            ))
        }
        "neg_int" | "abs" => {
            let neg = op == "neg_int";
            match ty {
                "s16" => map::<1, _>(m, ops, 16, true, move |[a]| {
                    (if neg {
                        arith::neg_int(a as i16)
                    } else {
                        arith::abs_int(a as i16)
                    }) as u16 as u64
                }),
                "s32" => map::<1, _>(m, ops, 32, true, move |[a]| {
                    (if neg {
                        arith::neg_int(a as i32)
                    } else {
                        arith::abs_int(a as i32)
                    }) as u32 as u64
                }),
                "s64" => map::<1, _>(m, ops, 64, true, move |[a]| {
                    (if neg {
                        arith::neg_int(a as i64)
                    } else {
                        arith::abs_int(a as i64)
                    }) as u64
                }),
                other => m.unsupported(format!("type `{other}`")),
            }
        }
        "dp2a" | "dp4a" => {
            let signed_a = m.get("atype") == "s32";
            let signed_b = m.get("btype") == "s32";
            let signed = signed_a || signed_b;
            if op == "dp4a" {
                map::<3, _>(m, ops, 32, signed, move |[a, b, c]| {
                    u64::from(arith::dp4a(
                        a as u32, b as u32, c as u32, signed_a, signed_b,
                    ))
                })
            } else {
                let high = m.get("mode") == "hi";
                map::<3, _>(m, ops, 32, signed, move |[a, b, c]| {
                    u64::from(arith::dp2a(
                        a as u32, b as u32, c as u32, signed_a, signed_b, high,
                    ))
                })
            }
        }
        "max" | "min" => {
            let maximum = op == "max";
            let relu = m.has("relu");
            for flag in ["ftz", "nan", "xorsign", "abs"] {
                if m.has(flag) {
                    return m.unsupported(format!(".{flag} on an integer min/max"));
                }
            }
            if ty == "u16x2" || ty == "s16x2" {
                let signed = ty == "s16x2";
                if relu && !signed {
                    return m.unsupported(".relu.u16x2");
                }
                return map::<2, _>(m, ops, 32, false, move |[a, b]| {
                    u64::from(arith::minmax_16x2(
                        a as u32, b as u32, signed, relu, maximum,
                    ))
                });
            }
            if relu {
                if ty != "s32" {
                    return m.unsupported(".relu applies only to .s32 and .s16x2");
                }
                return map::<2, _>(m, ops, 32, true, move |[a, b]| {
                    let (a, b) = (a as i32, b as i32);
                    (if maximum {
                        arith::max_relu_s32(a, b)
                    } else {
                        arith::min_relu_s32(a, b)
                    }) as u32 as u64
                });
            }
            let (bits, signed) = int_type(m, ty)?;
            with_int!(m, bits, signed, |T| map::<2, _>(
                m,
                ops,
                bits,
                signed,
                move |[a, b]| {
                    let (a, b) = (a as T, b as T);
                    (if maximum {
                        arith::max_int::<T>(a, b)
                    } else {
                        arith::min_int::<T>(a, b)
                    }) as u64
                }
            ))
        }
        _ => m.unsupported("not an integer ALU op"),
    }
}

/// `add.{u32,s32}`: wrapping 32-bit add (validated by `flat`/`flat_exact`).
fn add_u32_direct(io: &mut PtxIo<'_>) -> OpResult {
    for lane in io.mask.lanes() {
        io.dsts[0][lane] =
            u64::from((io.srcs[0][lane] as u32).wrapping_add(io.srcs[1][lane] as u32));
    }
    Ok(())
}
