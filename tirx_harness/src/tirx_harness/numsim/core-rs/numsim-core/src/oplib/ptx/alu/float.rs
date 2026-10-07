//! `.f32` / `.f64` / `.f32x2` arithmetic, f32 mixed-precision sources,
//! division, reciprocal, roots, approximate transcendentals, neg/abs/copysign
//! and float min/max (legacy `F32Arithmetic`, `F64Arithmetic`,
//! `F32x2Arithmetic`, `MixedF32`, `F32MinMax` and the unary f32 variants).

use super::{bits_f32, f32_of, flat, half_format, map, rounding, Md};
use crate::oplib::ptx::{Operands, Resolved};
use crate::oplib::{OpResult, PtxIo};
use numsim_oplib::arith::{self, F32DivMode};
use numsim_oplib::cvt;
use numsim_oplib::scalar::{self, F32RoundingMode};

const RN: F32RoundingMode = F32RoundingMode::Nearest;

pub(super) fn resolve(op: &str, m: &Md, ops: &Operands) -> OpResult<Resolved> {
    let ty = m.get("type");
    let ftz = m.has("ftz");
    match op {
        "add" | "sub" | "mul" | "fma" | "mad_f" => arithmetic(op, m, ops),
        "div_f" => div(m, ops),
        "copysign" => match ty {
            "f32" => map::<2, _>(m, ops, 32, false, |[a, b]| {
                bits_f32(arith::copysign_f32(f32_of(a), f32_of(b)))
            }),
            _ => map::<2, _>(m, ops, 64, false, |[a, b]| {
                arith::copysign_f64(f64::from_bits(a), f64::from_bits(b)).to_bits()
            }),
        },
        "neg" | "abs_f" => {
            let neg = op == "neg";
            match (ty, ftz) {
                ("f32", _) => {
                    if !neg && flat(ops, 1, 1, 32) {
                        return Ok(Resolved::Direct(if ftz {
                            abs_f32_direct::<true>
                        } else {
                            abs_f32_direct::<false>
                        }));
                    }
                    map::<1, _>(m, ops, 32, false, move |[a]| {
                        let a = f32_of(a);
                        bits_f32(if neg {
                            arith::neg_f32(a, ftz)
                        } else {
                            arith::abs_f32(a, ftz)
                        })
                    })
                }
                ("f64", false) => map::<1, _>(m, ops, 64, false, move |[a]| {
                    let a = f64::from_bits(a);
                    (if neg {
                        arith::neg_f64(a)
                    } else {
                        arith::abs_f64(a)
                    })
                    .to_bits()
                }),
                _ => m.unsupported(".ftz.f64"),
            }
        }
        "rcp" => rcp(m, ops),
        "sqrt" => {
            let mode = m.get("mode");
            match (ty, mode) {
                ("f32", "approx") => map::<1, _>(m, ops, 32, false, move |[a]| {
                    bits_f32(arith::sqrt_f32(f32_of(a), RN, ftz))
                }),
                ("f32", _) => {
                    let round = rounding(m, mode)?;
                    map::<1, _>(m, ops, 32, false, move |[a]| {
                        bits_f32(arith::sqrt_f32(f32_of(a), round, ftz))
                    })
                }
                ("f64", "approx") => m.unsupported("sqrt.approx.f64"),
                ("f64", _) if ftz => m.unsupported("sqrt.rnd.ftz.f64"),
                _ => {
                    let round = rounding(m, mode)?;
                    map::<1, _>(m, ops, 64, false, move |[a]| {
                        scalar::ptx_sqrt_f64(f64::from_bits(a), round).to_bits()
                    })
                }
            }
        }
        "rsqrt" => match ty {
            "f32" => map::<1, _>(m, ops, 32, false, move |[a]| {
                bits_f32(arith::rsqrt_approx_f32(f32_of(a), ftz))
            }),
            _ if ftz => map::<1, _>(m, ops, 64, false, |[a]| {
                scalar::ptx_rsqrt_approx_ftz_f64(f64::from_bits(a)).to_bits()
            }),
            _ => map::<1, _>(m, ops, 64, false, |[a]| {
                arith::rsqrt_f64(f64::from_bits(a)).to_bits()
            }),
        },
        "sin" => map::<1, _>(m, ops, 32, false, move |[a]| {
            bits_f32(scalar::ptx_sin_approx_f32(f32_of(a), ftz))
        }),
        "cos" => map::<1, _>(m, ops, 32, false, move |[a]| {
            bits_f32(scalar::ptx_cos_approx_f32(f32_of(a), ftz))
        }),
        "ex2" => {
            if flat(ops, 1, 1, 32) {
                return Ok(Resolved::Direct(if ftz {
                    ex2_direct::<true>
                } else {
                    ex2_direct::<false>
                }));
            }
            map::<1, _>(m, ops, 32, false, move |[a]| {
                bits_f32(arith::ex2_approx_f32(f32_of(a), ftz))
            })
        }
        "lg2" => map::<1, _>(m, ops, 32, false, move |[a]| {
            bits_f32(arith::lg2_approx_f32(f32_of(a), ftz))
        }),
        "tanh" => map::<1, _>(m, ops, 32, false, |[a]| {
            bits_f32(scalar::ptx_tanh_approx_f32(f32_of(a)))
        }),
        "max" | "min" | "max3" | "min3" => minmax(op, m, ops),
        _ => m.unsupported("not a float ALU op"),
    }
}

// ---------------------------------------------------------------------------
// add / sub / mul / fma / mad
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arith {
    Add,
    Sub,
    Mul,
    Fma,
}

fn arithmetic(op: &str, m: &Md, ops: &Operands) -> OpResult<Resolved> {
    let kind = match op {
        "add" => Arith::Add,
        "sub" => Arith::Sub,
        "mul" => Arith::Mul,
        _ => Arith::Fma,
    };
    let rnd_token = m.get("rnd");
    let round = rounding(m, rnd_token)?;
    let ftz = m.has("ftz");
    let sat = m.has("sat");
    let ty = m.get("type");
    let srctype = m.get("srctype");

    if !srctype.is_empty() {
        // `add/sub/fma.rnd{.sat}.f32.{f16,bf16}` (legacy `MixedF32`).
        if ty != "f32" {
            return m.unsupported("mixed-precision sources require an f32 destination");
        }
        if ftz {
            return m.unsupported("mixed-precision form cannot use ftz");
        }
        let (format, _) = half_format(m, srctype)?;
        return match kind {
            Arith::Add => map::<2, _>(m, ops, 32, false, move |[a, c]| {
                bits_f32(arith::add_mixed_f32(
                    a as u16,
                    format,
                    f32_of(c),
                    round,
                    sat,
                ))
            }),
            Arith::Sub => map::<2, _>(m, ops, 32, false, move |[a, c]| {
                bits_f32(arith::sub_mixed_f32(
                    a as u16,
                    format,
                    f32_of(c),
                    round,
                    sat,
                ))
            }),
            Arith::Fma => map::<3, _>(m, ops, 32, false, move |[a, b, c]| {
                bits_f32(arith::fma_mixed_f32(
                    a as u16,
                    b as u16,
                    format,
                    f32_of(c),
                    round,
                    sat,
                ))
            }),
            Arith::Mul => m.unsupported("mul has no mixed-precision form"),
        };
    }

    match ty {
        "f32" => {
            let hot_round = rnd_token.is_empty() || rnd_token == "rn";
            let arity = if kind == Arith::Fma { 3 } else { 2 };
            if hot_round && !sat && flat(ops, 1, arity, 32) {
                let direct: crate::oplib::PtxFn = match (kind, ftz) {
                    (Arith::Add, false) => add_f32_direct::<false>,
                    (Arith::Add, true) => add_f32_direct::<true>,
                    (Arith::Sub, false) => sub_f32_direct::<false>,
                    (Arith::Sub, true) => sub_f32_direct::<true>,
                    (Arith::Mul, false) => mul_f32_direct::<false>,
                    (Arith::Mul, true) => mul_f32_direct::<true>,
                    (Arith::Fma, false) => fma_f32_direct::<false>,
                    (Arith::Fma, true) => fma_f32_direct::<true>,
                };
                return Ok(Resolved::Direct(direct));
            }
            match kind {
                Arith::Add => map::<2, _>(m, ops, 32, false, move |[a, b]| {
                    bits_f32(arith::add_f32(f32_of(a), f32_of(b), round, ftz, sat))
                }),
                Arith::Sub => map::<2, _>(m, ops, 32, false, move |[a, b]| {
                    bits_f32(arith::sub_f32(f32_of(a), f32_of(b), round, ftz, sat))
                }),
                Arith::Mul => map::<2, _>(m, ops, 32, false, move |[a, b]| {
                    bits_f32(arith::mul_f32(f32_of(a), f32_of(b), round, ftz, sat))
                }),
                Arith::Fma => map::<3, _>(m, ops, 32, false, move |[a, b, c]| {
                    bits_f32(arith::fma_f32(
                        f32_of(a),
                        f32_of(b),
                        f32_of(c),
                        round,
                        ftz,
                        sat,
                    ))
                }),
            }
        }
        "f32x2" => {
            if sat {
                return m.unsupported("f32x2 form cannot use sat");
            }
            match kind {
                Arith::Add => map::<2, _>(m, ops, 64, false, move |[a, b]| {
                    cvt::add_f32x2(a, b, round, ftz)
                }),
                Arith::Sub => map::<2, _>(m, ops, 64, false, move |[a, b]| {
                    cvt::sub_f32x2(a, b, round, ftz)
                }),
                Arith::Mul => map::<2, _>(m, ops, 64, false, move |[a, b]| {
                    cvt::mul_f32x2(a, b, round, ftz)
                }),
                Arith::Fma => map::<3, _>(m, ops, 64, false, move |[a, b, c]| {
                    cvt::fma_f32x2(a, b, c, round, ftz)
                }),
            }
        }
        "f64" => {
            if ftz || sat {
                return m.unsupported("f64 form cannot use ftz or sat");
            }
            let f = f64::from_bits;
            match kind {
                Arith::Add => map::<2, _>(m, ops, 64, false, move |[a, b]| {
                    scalar::add_f64(f(a), f(b), round).to_bits()
                }),
                Arith::Sub => map::<2, _>(m, ops, 64, false, move |[a, b]| {
                    scalar::sub_f64(f(a), f(b), round).to_bits()
                }),
                Arith::Mul => map::<2, _>(m, ops, 64, false, move |[a, b]| {
                    scalar::mul_f64(f(a), f(b), round).to_bits()
                }),
                Arith::Fma => map::<3, _>(m, ops, 64, false, move |[a, b, c]| {
                    scalar::fma_f64(f(a), f(b), f(c), round).to_bits()
                }),
            }
        }
        other => m.unsupported(format!("arithmetic type `{other}`")),
    }
}

fn div(m: &Md, ops: &Operands) -> OpResult<Resolved> {
    let ftz = m.has("ftz");
    let mode = m.get("mode");
    match m.get("type") {
        "f32" => {
            let mode = match mode {
                "approx" => F32DivMode::Approx,
                "full" => F32DivMode::Full,
                other => F32DivMode::Rounded(rounding(m, other)?),
            };
            map::<2, _>(m, ops, 32, false, move |[a, b]| {
                bits_f32(arith::div_f32(f32_of(a), f32_of(b), mode, ftz))
            })
        }
        _ if ftz || mode == "approx" || mode == "full" => {
            m.unsupported("no deterministic div.f64 representative")
        }
        _ => {
            let round = rounding(m, mode)?;
            map::<2, _>(m, ops, 64, false, move |[a, b]| {
                scalar::div_f64(f64::from_bits(a), f64::from_bits(b), round).to_bits()
            })
        }
    }
}

fn rcp(m: &Md, ops: &Operands) -> OpResult<Resolved> {
    let ftz = m.has("ftz");
    let mode = m.get("mode");
    match (m.get("type"), mode) {
        ("f32", "approx") => {
            if flat(ops, 1, 1, 32) {
                return Ok(Resolved::Direct(if ftz {
                    rcp_approx_direct::<true>
                } else {
                    rcp_approx_direct::<false>
                }));
            }
            map::<1, _>(m, ops, 32, false, move |[a]| {
                bits_f32(arith::rcp_approx_f32(f32_of(a), ftz))
            })
        }
        ("f32", _) => {
            let round = rounding(m, mode)?;
            map::<1, _>(m, ops, 32, false, move |[a]| {
                bits_f32(arith::rcp_f32(f32_of(a), round, ftz))
            })
        }
        // Legacy `F64Arithmetic<Approx>`: the PTX form's `.ftz` is mandatory.
        (_, "approx") => map::<1, _>(m, ops, 64, false, |[a]| {
            scalar::ptx_rcp_approx_ftz_f64(f64::from_bits(a)).to_bits()
        }),
        _ if ftz => m.unsupported("rcp.rnd.ftz.f64"),
        _ => {
            let round = rounding(m, mode)?;
            map::<1, _>(m, ops, 64, false, move |[a]| {
                arith::rcp_f64(f64::from_bits(a), round).to_bits()
            })
        }
    }
}

fn minmax(op: &str, m: &Md, ops: &Operands) -> OpResult<Resolved> {
    let maximum = op.starts_with("max");
    let ftz = m.has("ftz");
    let nan = m.has("nan");
    let abs = m.has("abs");
    let xorsign = m.has("xorsign");
    if m.get("type") == "f64" {
        if ftz || nan || abs || xorsign || m.has("relu") {
            return m.unsupported("min/max.f64 accepts no modifiers");
        }
        return map::<2, _>(m, ops, 64, false, move |[a, b]| {
            let (a, b) = (f64::from_bits(a), f64::from_bits(b));
            (if maximum {
                scalar::cuda_f64_max(a, b)
            } else {
                scalar::cuda_f64_min(a, b)
            })
            .to_bits()
        });
    }
    if m.has("relu") {
        return m.unsupported("relu on a float min/max (legacy dropped it)");
    }
    let three = op.ends_with('3');
    if !abs && !xorsign && flat(ops, 1, if three { 3 } else { 2 }, 32) {
        macro_rules! pick {
            ($n:literal) => {
                match (maximum, ftz, nan) {
                    (true, false, false) => minmax_direct::<$n, true, false, false>,
                    (true, false, true) => minmax_direct::<$n, true, false, true>,
                    (true, true, false) => minmax_direct::<$n, true, true, false>,
                    (true, true, true) => minmax_direct::<$n, true, true, true>,
                    (false, false, false) => minmax_direct::<$n, false, false, false>,
                    (false, false, true) => minmax_direct::<$n, false, false, true>,
                    (false, true, false) => minmax_direct::<$n, false, true, false>,
                    (false, true, true) => minmax_direct::<$n, false, true, true>,
                }
            };
        }
        let direct: crate::oplib::PtxFn = if three { pick!(3) } else { pick!(2) };
        return Ok(Resolved::Direct(direct));
    }
    if three {
        map::<3, _>(m, ops, 32, false, move |[a, b, c]| {
            bits_f32(arith::minmax_f32(
                [f32_of(a), f32_of(b), f32_of(c)],
                ftz,
                nan,
                abs,
                xorsign,
                maximum,
            ))
        })
    } else {
        map::<2, _>(m, ops, 32, false, move |[a, b]| {
            bits_f32(arith::minmax_f32(
                [f32_of(a), f32_of(b)],
                ftz,
                nan,
                abs,
                xorsign,
                maximum,
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// Direct hot forms (validated by `flat`: one slot per operand, f32-capable
// destination carrier).
// ---------------------------------------------------------------------------

#[inline(always)]
fn src(io: &PtxIo<'_>, i: usize, lane: usize) -> f32 {
    f32::from_bits(io.srcs[i][lane] as u32)
}

fn add_f32_direct<const FTZ: bool>(io: &mut PtxIo<'_>) -> OpResult {
    for lane in io.mask.lanes() {
        io.dsts[0][lane] = bits_f32(arith::add_f32(
            src(io, 0, lane),
            src(io, 1, lane),
            RN,
            FTZ,
            false,
        ));
    }
    Ok(())
}

fn sub_f32_direct<const FTZ: bool>(io: &mut PtxIo<'_>) -> OpResult {
    for lane in io.mask.lanes() {
        io.dsts[0][lane] = bits_f32(arith::sub_f32(
            src(io, 0, lane),
            src(io, 1, lane),
            RN,
            FTZ,
            false,
        ));
    }
    Ok(())
}

fn mul_f32_direct<const FTZ: bool>(io: &mut PtxIo<'_>) -> OpResult {
    for lane in io.mask.lanes() {
        io.dsts[0][lane] = bits_f32(arith::mul_f32(
            src(io, 0, lane),
            src(io, 1, lane),
            RN,
            FTZ,
            false,
        ));
    }
    Ok(())
}

fn fma_f32_direct<const FTZ: bool>(io: &mut PtxIo<'_>) -> OpResult {
    for lane in io.mask.lanes() {
        let value = arith::fma_f32(
            src(io, 0, lane),
            src(io, 1, lane),
            src(io, 2, lane),
            RN,
            FTZ,
            false,
        );
        io.dsts[0][lane] = bits_f32(value);
    }
    Ok(())
}

fn ex2_direct<const FTZ: bool>(io: &mut PtxIo<'_>) -> OpResult {
    for lane in io.mask.lanes() {
        io.dsts[0][lane] = bits_f32(arith::ex2_approx_f32(src(io, 0, lane), FTZ));
    }
    Ok(())
}

fn rcp_approx_direct<const FTZ: bool>(io: &mut PtxIo<'_>) -> OpResult {
    for lane in io.mask.lanes() {
        io.dsts[0][lane] = bits_f32(arith::rcp_approx_f32(src(io, 0, lane), FTZ));
    }
    Ok(())
}

fn abs_f32_direct<const FTZ: bool>(io: &mut PtxIo<'_>) -> OpResult {
    for lane in io.mask.lanes() {
        io.dsts[0][lane] = bits_f32(arith::abs_f32(src(io, 0, lane), FTZ));
    }
    Ok(())
}

fn minmax_direct<const N: usize, const MAX: bool, const FTZ: bool, const NAN: bool>(
    io: &mut PtxIo<'_>,
) -> OpResult {
    for lane in io.mask.lanes() {
        let args: [f32; N] = std::array::from_fn(|i| src(io, i, lane));
        io.dsts[0][lane] = bits_f32(arith::minmax_f32(args, FTZ, NAN, false, false, MAX));
    }
    Ok(())
}
