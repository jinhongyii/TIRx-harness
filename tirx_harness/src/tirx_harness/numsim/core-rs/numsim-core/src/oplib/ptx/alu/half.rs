//! `.f16/.bf16{x2}` arithmetic, min/max, neg/abs, approximate `ex2`/`tanh`,
//! and the PTX 9.4 mixed packed-half / `.f32x2` vector forms (legacy
//! `HalfArithmetic`, `HalfMinMax`, half `neg`/`abs`/`exp2`/`tanh`,
//! `MixedF32x2`, `MixedF32x2Down`, `MixedLowMul`).

use super::{half_format, map, rounding, Md};
use crate::oplib::ptx::{Operands, Resolved};
use crate::oplib::OpResult;
use numsim_oplib::arith::{self, HalfClamp, MixedDownOp};
use numsim_oplib::scalar::{self, LowPrecisionFormat};

pub(super) fn resolve(op: &str, m: &Md, ops: &Operands) -> OpResult<Resolved> {
    let ftz = m.has("ftz");
    match op {
        "add_half" | "sub_half" | "mul_half" | "fma_half" => arithmetic(op, m, ops),
        "max" | "min" => {
            let (format, packed) = half_format(m, m.get("type"))?;
            let maximum = op == "max";
            let nan = m.has("nan");
            let xorsign = m.has("xorsign");
            if m.has("relu") {
                return m.unsupported("relu on a half min/max (legacy dropped it)");
            }
            if m.has("abs") && !xorsign {
                return m
                    .unsupported(".abs without .xorsign on a half min/max (legacy dropped it)");
            }
            if ftz && matches!(format, LowPrecisionFormat::Bf16) {
                return m.unsupported(".ftz.bf16 min/max");
            }
            if packed {
                map::<2, _>(m, ops, 32, false, move |[a, b]| {
                    u64::from(arith::minmax_half2(
                        a as u32, b as u32, format, ftz, nan, xorsign, maximum,
                    ))
                })
            } else {
                map::<2, _>(m, ops, 16, false, move |[a, b]| {
                    u64::from(arith::minmax_half(
                        a as u16, b as u16, format, ftz, nan, xorsign, maximum,
                    ))
                })
            }
        }
        "neg_half" | "abs_half" => {
            let neg = op == "neg_half";
            let ty = m.get("type");
            if ftz && !matches!(ty, "f16" | "f16x2") {
                return m.unsupported(".ftz on a bf16 neg/abs");
            }
            match (ty, neg) {
                ("f16", true) => map::<1, _>(m, ops, 16, false, move |[a]| {
                    let a = a as u16;
                    u64::from(if ftz {
                        scalar::ptx_neg_f16_bits(a, true)
                    } else {
                        a ^ 0x8000
                    })
                }),
                ("f16x2", true) => map::<1, _>(m, ops, 32, false, move |[a]| {
                    let a = a as u32;
                    u64::from(if ftz {
                        scalar::ptx_neg_f16x2_bits(a, true)
                    } else {
                        a ^ 0x8000_8000
                    })
                }),
                ("bf16", true) => map::<1, _>(m, ops, 16, false, |[a]| {
                    u64::from(arith::neg_bf16(a as u16))
                }),
                ("bf16x2", true) => map::<1, _>(m, ops, 32, false, |[a]| {
                    u64::from(arith::neg_bf16x2(a as u32))
                }),
                ("f16", false) => map::<1, _>(m, ops, 16, false, move |[a]| {
                    u64::from(arith::abs_f16(a as u16, ftz))
                }),
                ("f16x2", false) => map::<1, _>(m, ops, 32, false, move |[a]| {
                    u64::from(arith::abs_f16x2(a as u32, ftz))
                }),
                ("bf16", false) => map::<1, _>(m, ops, 16, false, |[a]| {
                    u64::from(arith::abs_bf16(a as u16))
                }),
                _ => map::<1, _>(m, ops, 32, false, |[a]| {
                    u64::from(arith::abs_bf16x2(a as u32))
                }),
            }
        }
        "ex2_half" => {
            let ty = m.get("type");
            // PTX: `.ftz` is mandatory for bf16 and absent for f16.
            if ftz != ty.starts_with("bf16") {
                return m.unsupported("ex2.approx half form has no NumSim representative");
            }
            match ty {
                "f16" => map::<1, _>(m, ops, 16, false, |[a]| {
                    u64::from(scalar::ptx_exp2_approx_f16(a as u16))
                }),
                "f16x2" => map::<1, _>(m, ops, 32, false, |[a]| {
                    u64::from(scalar::ptx_exp2_approx_f16x2(a as u32))
                }),
                "bf16" => map::<1, _>(m, ops, 16, false, |[a]| {
                    u64::from(scalar::ptx_exp2_approx_ftz_bf16(a as u16))
                }),
                _ => map::<1, _>(m, ops, 32, false, |[a]| {
                    u64::from(scalar::ptx_exp2_approx_ftz_bf16x2(a as u32))
                }),
            }
        }
        "tanh_half" => match m.get("type") {
            "f16" => map::<1, _>(m, ops, 16, false, |[a]| {
                u64::from(scalar::ptx_tanh_approx_f16(a as u16))
            }),
            "f16x2" => map::<1, _>(m, ops, 32, false, |[a]| {
                u64::from(scalar::ptx_tanh_approx_f16x2(a as u32))
            }),
            "bf16" => map::<1, _>(m, ops, 16, false, |[a]| {
                u64::from(scalar::ptx_tanh_approx_bf16(a as u16))
            }),
            _ => map::<1, _>(m, ops, 32, false, |[a]| {
                u64::from(scalar::ptx_tanh_approx_bf16x2(a as u32))
            }),
        },
        "add_mixed_vec_up" | "sub_mixed_vec_up" => {
            let round = rounding(m, m.get("rnd"))?;
            let (format, _) = half_format(m, m.get("atype"))?;
            if op == "add_mixed_vec_up" {
                map::<2, _>(m, ops, 64, false, move |[a, c]| {
                    arith::add_mixed_f32x2(a as u32, format, c, round)
                })
            } else {
                map::<2, _>(m, ops, 64, false, move |[a, c]| {
                    arith::sub_mixed_f32x2(a as u32, format, c, round)
                })
            }
        }
        "fma_mixed_vec" => {
            let round = rounding(m, m.get("rnd"))?;
            let (format, _) = half_format(m, m.get("atype"))?;
            map::<3, _>(m, ops, 64, false, move |[a, b, c]| {
                arith::fma_mixed_f32x2(a as u32, format, b, c, round)
            })
        }
        "add_mixed_vec_down_f16"
        | "add_mixed_vec_down_bf16"
        | "sub_mixed_vec_down_f16"
        | "sub_mixed_vec_down_bf16"
        | "mul_mixed_vec_down_f16"
        | "mul_mixed_vec_down_bf16" => {
            // The table pins `.rz`, `.ftz` (f16 only), `dtype`, `atype`, `ctype`.
            let kind = match &op[..3] {
                "add" => MixedDownOp::Add,
                "sub" => MixedDownOp::Sub,
                _ => MixedDownOp::Mul,
            };
            let dst = if op.ends_with("_f16") {
                LowPrecisionFormat::F16
            } else {
                LowPrecisionFormat::Bf16
            };
            map::<2, _>(m, ops, 32, false, move |[a, c]| {
                u64::from(arith::mixed_f32x2_down(a, c, kind, dst))
            })
        }
        "mul_mixed_vec_bf16_f16" => map::<2, _>(m, ops, 32, false, |[a, c]| {
            u64::from(arith::mul_bf16x2_f16x2(a as u32, c as u32))
        }),
        "mul_mixed_vec_f16_bf16" => map::<2, _>(m, ops, 32, false, |[a, c]| {
            u64::from(arith::mul_f16x2_bf16x2(a as u32, c as u32))
        }),
        _ => m.unsupported("not a half ALU op"),
    }
}

/// Same-precision `add/sub/mul/fma.rn{.ftz}{.sat,.relu}{.oob}` (always RN).
fn arithmetic(op: &str, m: &Md, ops: &Operands) -> OpResult<Resolved> {
    let (format, packed) = half_format(m, m.get("type"))?;
    let ftz = m.has("ftz");
    let oob = m.has("oob");
    let clamp = match (m.has("sat"), m.has("relu")) {
        (true, true) => return m.unsupported(".sat with .relu"),
        (true, false) => HalfClamp::Sat,
        (false, true) => HalfClamp::Relu,
        (false, false) => HalfClamp::None,
    };
    let bits = if packed { 32 } else { 16 };
    macro_rules! binary {
        ($scalar:path, $pair:path) => {
            if packed {
                map::<2, _>(m, ops, bits, false, move |[a, b]| {
                    u64::from($pair(a as u32, b as u32, format, ftz, clamp))
                })
            } else {
                map::<2, _>(m, ops, bits, false, move |[a, b]| {
                    u64::from($scalar(a as u16, b as u16, format, ftz, clamp))
                })
            }
        };
    }
    match op {
        "add_half" => binary!(arith::add_half, arith::add_half2),
        "sub_half" => binary!(arith::sub_half, arith::sub_half2),
        "mul_half" => binary!(arith::mul_half, arith::mul_half2),
        _ => {
            if packed {
                map::<3, _>(m, ops, bits, false, move |[a, b, c]| {
                    u64::from(arith::fma_half2(
                        a as u32, b as u32, c as u32, format, ftz, clamp, oob,
                    ))
                })
            } else {
                map::<3, _>(m, ops, bits, false, move |[a, b, c]| {
                    u64::from(arith::fma_half(
                        a as u16, b as u16, c as u16, format, ftz, clamp, oob,
                    ))
                })
            }
        }
    }
}
