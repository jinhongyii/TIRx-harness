//! Comparison, selection and classification: `setp` / `set` (scalar,
//! packed-half, `.BoolOp`, two-predicate `p|q`), PTX 9.4 packed-integer
//! `set`, `selp`, `slct`, `testp` (legacy `reg_compare.rs`: `Setp`,
//! `SetpBool`, `SetpPair`, `SetpPairBool`, `Set`, `SetBool`, `SetPacked`,
//! `Slct`, `Testp`, and `selp`).
//!
//! Every comparison/source/destination legality check the legacy trait
//! bounds expressed (`ValidComparison`, `SubnormalMode`, `SetDestination`)
//! is checked at resolve time, so the per-lane kernels cannot fail.

use super::{int_type, lanes, map, mask, sext, Md};
use crate::oplib::ptx::{Operands, Resolved};
use crate::oplib::OpResult;
use numsim_oplib::arith::{self, BoolOp, CmpOp, CompareAtom, PackedIntLane, SetDst, TestpClass};

/// How one compared PTX source decodes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Src {
    Bits(u32),
    Signed(u32),
    Unsigned(u32),
    F32,
    F64,
    F16,
    Bf16,
    F16x2,
    Bf16x2,
}

impl Src {
    fn parse(m: &Md, token: &str) -> OpResult<Src> {
        Ok(match token {
            "b16" | "b32" | "b64" => Src::Bits(int_type(m, token)?.0),
            "u16" | "u32" | "u64" => Src::Unsigned(int_type(m, token)?.0),
            "s16" | "s32" | "s64" => Src::Signed(int_type(m, token)?.0),
            "f32" => Src::F32,
            "f64" => Src::F64,
            "f16" => Src::F16,
            "bf16" => Src::Bf16,
            "f16x2" => Src::F16x2,
            "bf16x2" => Src::Bf16x2,
            other => return m.unsupported(format!("compare source type `{other}`")),
        })
    }
    fn lanes(self) -> usize {
        if matches!(self, Src::F16x2 | Src::Bf16x2) {
            2
        } else {
            1
        }
    }
    fn atom(self, value: u64, ftz: bool) -> CompareAtom {
        match self {
            Src::Bits(bits) => CompareAtom::Bits(value & mask(bits)),
            Src::Unsigned(bits) => CompareAtom::Unsigned(value & mask(bits)),
            Src::Signed(bits) => CompareAtom::Signed(sext(value, bits)),
            Src::F32 => arith::f32_atom(f32::from_bits(value as u32), ftz),
            Src::F64 => CompareAtom::Float(f64::from_bits(value)),
            Src::F16 | Src::F16x2 => arith::f16_atom(value as u16, ftz),
            Src::Bf16 | Src::Bf16x2 => arith::bf16_atom(value as u16),
        }
    }
    /// Comparison mask: bit 0 = (low) lane, bit 1 = high half of an x2 source.
    fn mask(self, cmp: CmpOp, a: u64, b: u64, ftz: bool) -> u8 {
        let one = |a: u64, b: u64| {
            // Validated at resolve time (`check`): never fails.
            u8::from(
                arith::compare_atom(cmp, self.atom(a, ftz), self.atom(b, ftz)).unwrap_or(false),
            )
        };
        if self.lanes() == 2 {
            one(a & 0xffff, b & 0xffff) | (one((a >> 16) & 0xffff, (b >> 16) & 0xffff) << 1)
        } else {
            one(a, b)
        }
    }
}

/// Parse `cmp` (`lo/ls/hi/hs` alias `lt/le/gt/ge`, legacy
/// `PTX_COMPARE_MARKERS`) and check it and `.ftz` against the source type.
fn check(m: &Md, src: Src) -> OpResult<(CmpOp, bool)> {
    let cmp = match m.get("cmp") {
        "eq" => CmpOp::Eq,
        "ne" => CmpOp::Ne,
        "lt" | "lo" => CmpOp::Lt,
        "le" | "ls" => CmpOp::Le,
        "gt" | "hi" => CmpOp::Gt,
        "ge" | "hs" => CmpOp::Ge,
        "equ" => CmpOp::Equ,
        "neu" => CmpOp::Neu,
        "ltu" => CmpOp::Ltu,
        "leu" => CmpOp::Leu,
        "gtu" => CmpOp::Gtu,
        "geu" => CmpOp::Geu,
        "num" => CmpOp::Num,
        "nan" => CmpOp::Nan,
        other => return m.unsupported(format!("comparison `{other}`")),
    };
    let ordered = matches!(
        cmp,
        CmpOp::Eq | CmpOp::Ne | CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge
    );
    let valid = match src {
        Src::Bits(_) => matches!(cmp, CmpOp::Eq | CmpOp::Ne),
        Src::Signed(_) | Src::Unsigned(_) => ordered,
        _ => true,
    };
    if !valid {
        return m.unsupported("comparison is not defined for this source type");
    }
    let ftz = m.has("ftz");
    if ftz && !matches!(src, Src::F32 | Src::F16 | Src::F16x2) {
        return m.unsupported(".ftz on this source type");
    }
    Ok((cmp, ftz))
}

fn bool_op(m: &Md) -> OpResult<BoolOp> {
    Ok(match m.get("boolop") {
        "and" => BoolOp::And,
        "or" => BoolOp::Or,
        "xor" => BoolOp::Xor,
        other => return m.unsupported(format!("boolop `{other}`")),
    })
}

/// `set` destination encoding, legal only for the legacy `SetDestination`
/// (dst, src) pairs. Returns (encoding, bits, signed).
fn set_dst(m: &Md, src: Src) -> OpResult<(SetDst, u32, bool)> {
    let half = matches!(src, Src::F16 | Src::Bf16 | Src::F16x2 | Src::Bf16x2);
    let packed = src.lanes() == 2;
    let (dst, bits, signed, ok) = match m.get("dtype") {
        "u32" => (SetDst::U32, 32, false, true),
        "s32" => (SetDst::S32, 32, true, true),
        "f32" => (SetDst::F32, 32, false, !half),
        "u16" => (SetDst::U16, 16, false, matches!(src, Src::F16 | Src::Bf16)),
        "s16" => (SetDst::S16, 16, true, matches!(src, Src::F16 | Src::Bf16)),
        "f16" => (SetDst::F16, 16, false, !packed && src != Src::Bf16),
        "bf16" => (SetDst::Bf16, 16, false, !packed && src != Src::Bf16),
        "f16x2" => (SetDst::F16x2, 32, false, src == Src::F16x2),
        "bf16x2" => (SetDst::Bf16x2, 32, false, src == Src::Bf16x2),
        other => return m.unsupported(format!("set destination `{other}`")),
    };
    if !ok {
        return m.unsupported("set destination/source pair has no legacy encoding");
    }
    Ok((dst, bits, signed))
}

pub(super) fn resolve(op: &str, m: &Md, ops: &Operands) -> OpResult<Resolved> {
    match op {
        "setp" | "setp_half" | "setp_bool" | "setp_half_bool" | "setp_pq" | "setp_half_pq"
        | "setp_bool_pq" | "setp_half_bool_pq" => {
            let src = Src::parse(m, m.get("type"))?;
            let (cmp, ftz) = check(m, src)?;
            let pair = op.ends_with("_pq");
            let combine = op.contains("bool");
            let lanes_n = src.lanes();
            if !pair && lanes_n == 2 {
                return m.unsupported("single-predicate setp on a packed source");
            }
            match (pair, combine) {
                (false, false) => map::<2, _>(m, ops, 1, false, move |[a, b]| {
                    u64::from(src.mask(cmp, a, b, ftz) & 1)
                }),
                (false, true) => {
                    let bop = bool_op(m)?;
                    map::<3, _>(m, ops, 1, false, move |[a, b, c]| {
                        u64::from(bop.apply(src.mask(cmp, a, b, ftz) & 1 != 0, c != 0))
                    })
                }
                (true, false) => lanes::<2, 2, _>(m, ops, [(1, false); 2], move |[a, b]| {
                    let (p, q) = arith::predicate_pair(src.mask(cmp, a, b, ftz), lanes_n);
                    Ok([u64::from(p), u64::from(q)])
                }),
                (true, true) => {
                    let bop = bool_op(m)?;
                    lanes::<3, 2, _>(m, ops, [(1, false); 2], move |[a, b, c]| {
                        let (p, q) = arith::predicate_pair(src.mask(cmp, a, b, ftz), lanes_n);
                        Ok([
                            u64::from(bop.apply(p, c != 0)),
                            u64::from(bop.apply(q, c != 0)),
                        ])
                    })
                }
            }
        }
        "set" | "set_half" | "set_bool" | "set_half_bool" => {
            let src = Src::parse(m, m.get("stype"))?;
            let (cmp, ftz) = check(m, src)?;
            let (dst, bits, signed) = set_dst(m, src)?;
            let lanes_n = src.lanes();
            if op.ends_with("bool") {
                let bop = bool_op(m)?;
                map::<3, _>(m, ops, bits, signed, move |[a, b, c]| {
                    let mask = arith::combine_mask(bop, src.mask(cmp, a, b, ftz), c != 0);
                    u64::from(arith::set_encode(dst, mask, lanes_n))
                })
            } else {
                map::<2, _>(m, ops, bits, signed, move |[a, b]| {
                    u64::from(arith::set_encode(dst, src.mask(cmp, a, b, ftz), lanes_n))
                })
            }
        }
        "set_packed" => {
            let lane = match m.get("type") {
                "u8x4" => PackedIntLane::U8,
                "s8x4" => PackedIntLane::S8,
                "u16x2" => PackedIntLane::U16,
                _ => PackedIntLane::S16,
            };
            let (cmp, _) = check(m, Src::Unsigned(32))?;
            // Every ordered relation is defined for packed integer lanes.
            map::<2, _>(m, ops, 32, false, move |[a, b]| {
                u64::from(arith::set_packed(cmp, lane, a as u32, b as u32).unwrap_or(0))
            })
        }
        "selp" => {
            let (bits, signed) = match m.get("type") {
                "f32" => (32, false),
                "f64" => (64, false),
                other => int_type(m, other)?,
            };
            map::<3, _>(m, ops, bits, signed, |[a, b, c]| arith::selp(c != 0, a, b))
        }
        "slct" => {
            let (bits, signed) = match m.get("dtype") {
                "f32" => (32, false),
                "f64" => (64, false),
                other => int_type(m, other)?,
            };
            let ftz = m.has("ftz");
            if m.get("ctype") == "s32" {
                if ftz {
                    return m.unsupported(".ftz with an .s32 selector");
                }
                map::<3, _>(m, ops, bits, signed, |[a, b, c]| {
                    arith::slct_s32(a, b, c as u32 as i32)
                })
            } else {
                map::<3, _>(m, ops, bits, signed, move |[a, b, c]| {
                    arith::slct_f32(a, b, f32::from_bits(c as u32), ftz)
                })
            }
        }
        "testp" => {
            let class = match m.get("op") {
                "finite" => TestpClass::Finite,
                "infinite" => TestpClass::Infinite,
                "number" => TestpClass::Number,
                "notanumber" => TestpClass::NotANumber,
                "normal" => TestpClass::Normal,
                _ => TestpClass::Subnormal,
            };
            if m.get("type") == "f32" {
                map::<1, _>(m, ops, 1, false, move |[a]| {
                    u64::from(arith::testp_f32(class, f32::from_bits(a as u32)))
                })
            } else {
                map::<1, _>(m, ops, 1, false, move |[a]| {
                    u64::from(arith::testp_f64(class, f64::from_bits(a)))
                })
            }
        }
        _ => m.unsupported("not a compare ALU op"),
    }
}
