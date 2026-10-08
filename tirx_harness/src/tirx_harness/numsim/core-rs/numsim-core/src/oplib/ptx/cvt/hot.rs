//! Monomorphic fn items for the corpus-hot `cvt` forms (no per-lane
//! spelling dispatch). Each calls exactly the leaf function
//! `CvtSpelling::execute` would call for that spelling; `cvt/tests.rs`
//! `hot_forms_match_the_spelling_dispatch` checks bit equality.

use super::super::{DirectFn, Operands};
use crate::oplib::{OpResult, PtxIo};
use numsim_oplib::cvt::{
    cvt_f32_pair_to_half2, cvt_f32_to_half, cvt_f32_to_int, cvt_half_to_f32, CvtRounding, CvtSpelling, CvtType,
    HalfNarrowing, IntKind,
};
use numsim_oplib::cvt::{PtxFloatRounding, PtxIntegerRounding};
use numsim_oplib::scalar::LowPrecisionFormat;

const fn fmt(bf16: bool) -> LowPrecisionFormat {
    if bf16 {
        LowPrecisionFormat::Bf16
    } else {
        LowPrecisionFormat::F16
    }
}

/// `cvt.{rn,rz}{.relu}{.satfinite}.{f16x2,bf16x2}.f32 d, a, b` (a = upper).
fn pair_to_half2<const BF16: bool, const RZ: bool, const RELU: bool, const SATF: bool>(io: &mut PtxIo<'_>) -> OpResult {
    if !BF16 && !RZ {
        // `ptx_cvt_narrow` for .rn.f16: NaN -> 0x7fff, else the RNE encoding,
        // then .satfinite (inf -> max finite) and .relu (negative -> +0).
        let fix = |v: f32, bits: u16| -> u16 {
            if v.is_nan() {
                return 0x7fff;
            }
            let mut bits = bits;
            if SATF && bits & 0x7fff == 0x7c00 {
                bits -= 1;
            }
            if RELU && bits & 0x8000 != 0 {
                bits = 0;
            }
            bits
        };
        let hi: [f32; 32] = std::array::from_fn(|l| f32::from_bits(io.srcs[0][l] as u32));
        let lo: [f32; 32] = std::array::from_fn(|l| f32::from_bits(io.srcs[1][l] as u32));
        let (eh, el) = (crate::oplib::simd::f32_to_f16_rne(&hi), crate::oplib::simd::f32_to_f16_rne(&lo));
        for lane in io.mask.lanes() {
            io.dsts[0][lane] = (u64::from(fix(hi[lane], eh[lane])) << 16) | u64::from(fix(lo[lane], el[lane]));
        }
        return Ok(());
    }
    let rnd = if RZ { PtxFloatRounding::Zero } else { PtxFloatRounding::NearestEven };
    for lane in io.mask.lanes() {
        let (hi, lo) = (f32::from_bits(io.srcs[0][lane] as u32), f32::from_bits(io.srcs[1][lane] as u32));
        io.dsts[0][lane] = u64::from(cvt_f32_pair_to_half2(hi, lo, fmt(BF16), rnd, RELU, SATF, false));
    }
    Ok(())
}

/// `cvt.rn.{f16,bf16}.f32 d, a` (no other modifiers).
fn f32_to_half_rn<const BF16: bool>(io: &mut PtxIo<'_>) -> OpResult {
    let m = HalfNarrowing::default();
    for lane in io.mask.lanes() {
        let v = f32::from_bits(io.srcs[0][lane] as u32);
        io.dsts[0][lane] = u64::from(cvt_f32_to_half(v, fmt(BF16), PtxFloatRounding::NearestEven, m));
    }
    Ok(())
}

/// `cvt.f32.{f16,bf16} d, a` (no modifiers).
fn half_to_f32<const BF16: bool>(io: &mut PtxIo<'_>) -> OpResult {
    for lane in io.mask.lanes() {
        let v = cvt_half_to_f32(io.srcs[0][lane] as u16, fmt(BF16), false, false);
        io.dsts[0][lane] = u64::from(v.to_bits());
    }
    Ok(())
}

const fn irnd(code: u8) -> PtxIntegerRounding {
    match code {
        0 => PtxIntegerRounding::NearestEven,
        1 => PtxIntegerRounding::Zero,
        2 => PtxIntegerRounding::NegativeInfinity,
        _ => PtxIntegerRounding::PositiveInfinity,
    }
}

/// `cvt.{rni,rzi,rmi,rpi}{.ftz}{.sat}.{s32,u32}.f32 d, a` into an exact
/// 32-bit carrier.
fn f32_to_int32<const MODE: u8, const FTZ: bool, const SAT: bool, const SIGNED: bool>(io: &mut PtxIo<'_>) -> OpResult {
    let kind = if SIGNED { IntKind::S32 } else { IntKind::U32 };
    if !FTZ {
        // Round per mode, then Rust's saturating `as` (NaN -> 0): the PTX
        // float->int rule (`.sat` is inert; checked against the leaf).
        for lane in io.mask.lanes() {
            let v = f32::from_bits(io.srcs[0][lane] as u32);
            let r = match MODE {
                0 => v.round_ties_even(),
                1 => v.trunc(),
                2 => v.floor(),
                _ => v.ceil(),
            };
            io.dsts[0][lane] = if SIGNED { u64::from(r as i32 as u32) } else { u64::from(r as u32) };
        }
        return Ok(());
    }
    let _ = SAT;
    for lane in io.mask.lanes() {
        let v = f32::from_bits(io.srcs[0][lane] as u32);
        io.dsts[0][lane] = cvt_f32_to_int(v, irnd(MODE), FTZ, SAT, kind) & 0xffff_ffff;
    }
    Ok(())
}


/// The hot fn item for `parsed` with these operand carriers, if any.
pub(super) fn select(parsed: &CvtSpelling, ops: &Operands) -> Option<DirectFn> {
    let one_slot = ops.src_tys.iter().all(|t| t.slots() == 1) && ops.dst_tys.len() == 1 && ops.dst_tys[0].slots() == 1;
    if !one_slot || parsed.pack.is_some() || parsed.scale.is_some() || parsed.pzo {
        return None;
    }
    let dst_bits = ops.dst_tys[0].bits();
    use CvtType as T;
    match (parsed.src, parsed.dst) {
        (T::F32, T::F16x2 | T::Bf16x2) if !parsed.ftz && !parsed.sat && ops.src_tys.len() == 2 && dst_bits >= 32 => {
            let bf16 = parsed.dst == T::Bf16x2;
            let rz = match parsed.rounding {
                Some(CvtRounding::Rn) => false,
                Some(CvtRounding::Rz) => true,
                _ => return None,
            };
            let (relu, satf) = (parsed.relu, parsed.satfinite);
            Some(match (bf16, rz, relu, satf) {
                (false, false, false, false) => pair_to_half2::<false, false, false, false>,
                (false, false, true, false) => pair_to_half2::<false, false, true, false>,
                (false, false, false, true) => pair_to_half2::<false, false, false, true>,
                (false, false, true, true) => pair_to_half2::<false, false, true, true>,
                (false, true, false, false) => pair_to_half2::<false, true, false, false>,
                (false, true, true, false) => pair_to_half2::<false, true, true, false>,
                (false, true, false, true) => pair_to_half2::<false, true, false, true>,
                (false, true, true, true) => pair_to_half2::<false, true, true, true>,
                (true, false, false, false) => pair_to_half2::<true, false, false, false>,
                (true, false, true, false) => pair_to_half2::<true, false, true, false>,
                (true, false, false, true) => pair_to_half2::<true, false, false, true>,
                (true, false, true, true) => pair_to_half2::<true, false, true, true>,
                (true, true, false, false) => pair_to_half2::<true, true, false, false>,
                (true, true, true, false) => pair_to_half2::<true, true, true, false>,
                (true, true, false, true) => pair_to_half2::<true, true, false, true>,
                (true, true, true, true) => pair_to_half2::<true, true, true, true>,
            })
        }
        (T::F32, T::F16 | T::Bf16)
            if parsed.rounding == Some(CvtRounding::Rn)
                && !(parsed.ftz || parsed.sat || parsed.relu || parsed.satfinite)
                && ops.src_tys.len() == 1
                && dst_bits >= 16 =>
        {
            Some(if parsed.dst == T::Bf16 { f32_to_half_rn::<true> } else { f32_to_half_rn::<false> })
        }
        (T::F16 | T::Bf16, T::F32)
            if parsed.rounding.is_none()
                && !(parsed.ftz || parsed.sat || parsed.relu || parsed.satfinite)
                && ops.src_tys.len() == 1
                && dst_bits >= 32 =>
        {
            Some(if parsed.src == T::Bf16 { half_to_f32::<true> } else { half_to_f32::<false> })
        }
        (T::F32, T::Int(kind @ (IntKind::S32 | IntKind::U32)))
            if !(parsed.relu || parsed.satfinite) && ops.src_tys.len() == 1 && dst_bits == 32 =>
        {
            let mode = match parsed.rounding? {
                CvtRounding::Rni => 0u8,
                CvtRounding::Rzi => 1,
                CvtRounding::Rmi => 2,
                CvtRounding::Rpi => 3,
                _ => return None,
            };
            let signed = kind == IntKind::S32;
            macro_rules! pick {
                ($m:expr) => {
                    match (parsed.ftz, parsed.sat, signed) {
                        (false, false, false) => f32_to_int32::<$m, false, false, false> as DirectFn,
                        (false, false, true) => f32_to_int32::<$m, false, false, true>,
                        (false, true, false) => f32_to_int32::<$m, false, true, false>,
                        (false, true, true) => f32_to_int32::<$m, false, true, true>,
                        (true, false, false) => f32_to_int32::<$m, true, false, false>,
                        (true, false, true) => f32_to_int32::<$m, true, false, true>,
                        (true, true, false) => f32_to_int32::<$m, true, true, false>,
                        (true, true, true) => f32_to_int32::<$m, true, true, true>,
                    }
                };
            }
            Some(match mode {
                0 => pick!(0),
                1 => pick!(1),
                2 => pick!(2),
                _ => pick!(3),
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Ty;
    use crate::value::{WarpMask, WarpValue};
    use numsim_oplib::cvt::CvtOperands;

    #[test]
    fn hot_forms_match_the_spelling_dispatch() {
        let mut s = 0x0bad_5eed_1234_5678u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let specials = [0u64, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0xffc1_2345, 1, 0x7f7f_ffff, 0x4f00_0000, 0xcf00_0001, 0x7c00, 0xfc01, 0x3c01];
        let mut forms = Vec::new();
        for r in ["rn", "rz"] {
            for relu in ["", ".relu"] {
                for satf in ["", ".satfinite"] {
                    for d in ["f16x2", "bf16x2"] {
                        forms.push((format!("cvt.{r}{relu}{satf}.{d}.f32"), 2usize, Ty::U32));
                    }
                }
            }
        }
        forms.push(("cvt.rn.f16.f32".into(), 1, Ty::U16));
        forms.push(("cvt.rn.bf16.f32".into(), 1, Ty::U16));
        forms.push(("cvt.f32.f16".into(), 1, Ty::U32));
        forms.push(("cvt.f32.bf16".into(), 1, Ty::U32));
        for r in ["rni", "rzi", "rmi", "rpi"] {
            for ftz in ["", ".ftz"] {
                for sat in ["", ".sat"] {
                    for d in ["s32", "u32"] {
                        forms.push((format!("cvt.{r}{ftz}{sat}.{d}.f32"), 1, Ty::U32));
                    }
                }
            }
        }
        for (spelling, nsrc, dst) in forms {
            let parsed = CvtSpelling::parse(&spelling).unwrap();
            let ops = Operands::new(&[dst], &vec![Ty::U32; nsrc]);
            let f = select(&parsed, &ops).unwrap_or_else(|| panic!("{spelling} is not hot"));
            for _ in 0..64 {
                let srcs: Vec<WarpValue<u64>> = (0..nsrc)
                    .map(|_| std::array::from_fn(|l| if l % 3 == 0 { specials[(next() % 13) as usize] } else { next() & 0xffff_ffff }))
                    .collect();
                let mut dsts = vec![[0u64; 32]];
                let mut io = PtxIo { dsts: &mut dsts, dst_tys: &[dst], srcs: &srcs, src_tys: &vec![Ty::U32; nsrc], mask: WarpMask::ALL };
                f(&mut io).unwrap();
                for l in 0..32 {
                    let mut o = CvtOperands::unary(srcs[0][l]);
                    if nsrc == 2 {
                        o.b = srcs[1][l];
                    }
                    let want = parsed.execute(&o).unwrap();
                    assert_eq!(dsts[0][l], want, "{spelling} lane {l} srcs {:x?}", srcs.iter().map(|s| s[l]).collect::<Vec<_>>());
                }
            }
        }
    }
}
