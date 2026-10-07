//! Logic and bit manipulation: and/or/xor/not/cnot, lop3 (+ predicate
//! forms), brev/clz/popc, bfe/bfi/bfind/bmsk, clmad, shf/shl/shr, szext,
//! prmt, fns (legacy `bit_variants!`, `Lop3`, `Lop3Bool`, `bfe/bfi/bfind`
//! variants, `Bmsk`, `Shf`, `Szext`, `shift_variant!`, `Prmt`, `fns_spec`).

use super::{flat, int_type, lanes, map, mask, try_map, Md};
use crate::oplib::ptx::{Operands, Resolved};
use crate::oplib::{OpError, OpResult, PtxIo};
use numsim_oplib::arith::{self, BoolOp, PrmtMode};

/// `immLut` arrives as a register; PTX's domain is one byte.
fn lut(value: u64) -> OpResult<u8> {
    u8::try_from(value)
        .map_err(|_| OpError::invalid(format!("lop3 immLut {value:#x} is outside 0..=255")))
}

pub(super) fn resolve(op: &str, m: &Md, ops: &Operands) -> OpResult<Resolved> {
    let ty = m.get("type");
    match op {
        "and" | "or" | "xor" | "not" => {
            let bits = if ty == "pred" { 1 } else { int_type(m, ty)?.0 };
            let all = mask(bits);
            match op {
                "and" => map::<2, _>(m, ops, bits, false, move |[a, b]| arith::and(a, b) & all),
                "or" => map::<2, _>(m, ops, bits, false, move |[a, b]| arith::or(a, b) & all),
                "xor" => map::<2, _>(m, ops, bits, false, move |[a, b]| arith::xor(a, b) & all),
                _ => map::<1, _>(m, ops, bits, false, move |[a]| arith::not(a) & all),
            }
        }
        "cnot" => match ty {
            "b16" => map::<1, _>(m, ops, 16, false, |[a]| {
                u64::from(arith::cnot_b16(a as u16))
            }),
            "b32" => map::<1, _>(m, ops, 32, false, |[a]| {
                u64::from(arith::cnot_b32(a as u32))
            }),
            _ => map::<1, _>(m, ops, 64, false, |[a]| arith::cnot_b64(a)),
        },
        "brev" => match ty {
            "b32" => map::<1, _>(m, ops, 32, false, |[a]| {
                u64::from(arith::brev_b32(a as u32))
            }),
            _ => map::<1, _>(m, ops, 64, false, |[a]| arith::brev_b64(a)),
        },
        "clz" | "popc" => {
            let clz = op == "clz";
            match ty {
                "b32" => map::<1, _>(m, ops, 32, false, move |[a]| {
                    u64::from(if clz {
                        arith::clz_b32(a as u32)
                    } else {
                        arith::popc_b32(a as u32)
                    })
                }),
                _ => map::<1, _>(m, ops, 32, false, move |[a]| {
                    u64::from(if clz {
                        arith::clz_b64(a)
                    } else {
                        arith::popc_b64(a)
                    })
                }),
            }
        }
        "bfe" => match ty {
            "u32" => try_map::<3, _>(m, ops, 32, false, |[a, b, c]| {
                Ok(u64::from(arith::bfe_u32(a as u32, b as u32, c as u32)?))
            }),
            "u64" => try_map::<3, _>(m, ops, 64, false, |[a, b, c]| {
                Ok(arith::bfe_u64(a, b as u32, c as u32)?)
            }),
            "s32" => try_map::<3, _>(m, ops, 32, true, |[a, b, c]| {
                Ok(arith::bfe_s32(a as i32, b as u32, c as u32)? as u32 as u64)
            }),
            _ => try_map::<3, _>(m, ops, 64, true, |[a, b, c]| {
                Ok(arith::bfe_s64(a as i64, b as u32, c as u32)? as u64)
            }),
        },
        "bfi" => match ty {
            "b32" => try_map::<4, _>(m, ops, 32, false, |[a, b, c, d]| {
                Ok(u64::from(arith::bfi_b32(
                    a as u32, b as u32, c as u32, d as u32,
                )?))
            }),
            _ => try_map::<4, _>(m, ops, 64, false, |[a, b, c, d]| {
                Ok(arith::bfi_b64(a, b, c as u32, d as u32)?)
            }),
        },
        "bfind" => {
            let shift = m.has("shiftamt");
            match ty {
                "u32" => map::<1, _>(m, ops, 32, false, move |[a]| {
                    u64::from(arith::bfind_u32(a as u32, shift))
                }),
                "u64" => map::<1, _>(m, ops, 32, false, move |[a]| {
                    u64::from(arith::bfind_u64(a, shift))
                }),
                "s32" => map::<1, _>(m, ops, 32, false, move |[a]| {
                    u64::from(arith::bfind_s32(a as i32, shift))
                }),
                _ => map::<1, _>(m, ops, 32, false, move |[a]| {
                    u64::from(arith::bfind_s64(a as i64, shift))
                }),
            }
        }
        "bmsk" => {
            let clamp = m.get("mode") == "clamp";
            map::<2, _>(m, ops, 32, false, move |[a, b]| {
                u64::from(arith::bmsk_b32(a as u32, b as u32, clamp))
            })
        }
        "clmad" => {
            if m.get("mode") == "hi" {
                map::<3, _>(m, ops, 64, false, |[a, b, c]| arith::clmad_hi(a, b, c))
            } else {
                map::<3, _>(m, ops, 64, false, |[a, b, c]| arith::clmad_lo(a, b, c))
            }
        }
        "shf" => {
            let left = m.get("dir") == "l";
            let clamp = m.get("mode") == "clamp";
            map::<3, _>(m, ops, 32, false, move |[a, b, c]| {
                u64::from(arith::shf_b32(a as u32, b as u32, c as u32, left, clamp))
            })
        }
        "shl" => match ty {
            "b16" => map::<2, _>(m, ops, 16, false, |[a, b]| {
                u64::from(arith::shl(a as u16, b as u32))
            }),
            "b32" => {
                if flat(ops, 1, 2, 32) {
                    return Ok(Resolved::Direct(shl_b32_direct));
                }
                map::<2, _>(m, ops, 32, false, |[a, b]| {
                    u64::from(arith::shl(a as u32, b as u32))
                })
            }
            _ => map::<2, _>(m, ops, 64, false, |[a, b]| arith::shl(a, b as u32)),
        },
        "shr" => match ty {
            "b16" | "u16" => map::<2, _>(m, ops, 16, false, |[a, b]| {
                u64::from(arith::shr(a as u16, b as u32))
            }),
            "s16" => map::<2, _>(m, ops, 16, true, |[a, b]| {
                arith::shr(a as i16, b as u32) as u16 as u64
            }),
            "b32" | "u32" => map::<2, _>(m, ops, 32, false, |[a, b]| {
                u64::from(arith::shr(a as u32, b as u32))
            }),
            "s32" => map::<2, _>(m, ops, 32, true, |[a, b]| {
                arith::shr(a as i32, b as u32) as u32 as u64
            }),
            "b64" | "u64" => map::<2, _>(m, ops, 64, false, |[a, b]| arith::shr(a, b as u32)),
            _ => map::<2, _>(m, ops, 64, true, |[a, b]| {
                arith::shr(a as i64, b as u32) as u64
            }),
        },
        "szext" => {
            let clamp = m.get("mode") == "clamp";
            if ty == "s32" {
                map::<2, _>(m, ops, 32, true, move |[a, b]| {
                    arith::szext_s32(a as i32, b as u32, clamp) as u32 as u64
                })
            } else {
                map::<2, _>(m, ops, 32, false, move |[a, b]| {
                    u64::from(arith::szext_u32(a as u32, b as u32, clamp))
                })
            }
        }
        "prmt" => {
            let mode = match m.get("mode") {
                "" => PrmtMode::Generic,
                "f4e" => PrmtMode::F4e,
                "b4e" => PrmtMode::B4e,
                "rc8" => PrmtMode::Rc8,
                "ecl" => PrmtMode::Ecl,
                "ecr" => PrmtMode::Ecr,
                "rc16" => PrmtMode::Rc16,
                other => return m.unsupported(format!("prmt mode `{other}`")),
            };
            map::<3, _>(m, ops, 32, false, move |[a, b, c]| {
                u64::from(arith::prmt_b32(a as u32, b as u32, c as u32, mode))
            })
        }
        "fns" => try_map::<3, _>(m, ops, 32, false, |[a, b, c]| {
            Ok(u64::from(arith::fns_b32(a as u32, b as u32, c as i32)?))
        }),
        "lop3" => try_map::<4, _>(m, ops, 32, false, |[a, b, c, l]| {
            Ok(u64::from(arith::lop3_b32(
                a as u32,
                b as u32,
                c as u32,
                lut(l)?,
            )))
        }),
        "lop3_bool" | "lop3_bool_sink" => {
            let bool_op = if m.get("boolop") == "and" {
                BoolOp::And
            } else {
                BoolOp::Or
            };
            let eval = move |[a, b, c, l, q]: [u64; 5]| -> OpResult<(u32, bool)> {
                Ok(arith::lop3_bool_b32(
                    a as u32,
                    b as u32,
                    c as u32,
                    lut(l)?,
                    bool_op,
                    q != 0,
                )?)
            };
            // The sink form's `d` is a literal `_` (no register); accept a
            // scratch `d` too.
            if op == "lop3_bool_sink" && ops.dst_tys.len() == 1 {
                lanes::<5, 1, _>(m, ops, [(1, false)], move |args| {
                    eval(args).map(|(_, p)| [u64::from(p)])
                })
            } else {
                lanes::<5, 2, _>(m, ops, [(32, false), (1, false)], move |args| {
                    eval(args).map(|(d, p)| [u64::from(d), u64::from(p)])
                })
            }
        }
        _ => m.unsupported("not a bit ALU op"),
    }
}

/// `shl.b32` with PTX clamping (validated by `flat`).
fn shl_b32_direct(io: &mut PtxIo<'_>) -> OpResult {
    for lane in io.mask.lanes() {
        io.dsts[0][lane] = u64::from(arith::shl(io.srcs[0][lane] as u32, io.srcs[1][lane] as u32));
    }
    Ok(())
}
