//! Bit-exact cases for the TIR ALU and conversions.

use super::super::{OpErrorKind, OpResult};
use super::*;
use crate::value::{splat, WARP_SIZE};

fn s(v: u64) -> Vec<WarpValue<u64>> {
    vec![splat(v)]
}

fn bin(op: BinOp, ty: Ty, a: u64, b: u64) -> OpResult<u64> {
    let mut out = s(0);
    binary(op, ty, &s(a), &s(b), &mut out, WarpMask::ALL)?;
    Ok(out[0][0])
}

fn un(op: UnOp, ty: Ty, a: u64) -> OpResult<u64> {
    let mut out = s(0);
    unary(op, ty, &s(a), &mut out, WarpMask::ALL)?;
    Ok(out[0][0])
}

fn cmp(op: CmpOp, ty: Ty, a: u64, b: u64) -> bool {
    compare(op, ty, &s(a), &s(b), WarpMask::lane(0)).unwrap().contains(0)
}

fn cv(from: Ty, to: Ty, rnd: Rounding, sat: bool, x: u64) -> OpResult<u64> {
    let mut out = s(0);
    cast(from, to, rnd, sat, &s(x), &mut out, WarpMask::ALL)?;
    Ok(out[0][0])
}

fn kind<T: std::fmt::Debug>(r: OpResult<T>) -> OpErrorKind {
    r.unwrap_err().kind
}

fn f(x: f32) -> u64 {
    x.to_bits() as u64
}
fn d(x: f64) -> u64 {
    x.to_bits()
}
const D: Rounding = Rounding::Default;

#[test]
fn f32_f64_arith_is_host_ieee() {
    for (a, b) in [(1.5f32, 2.25f32), (1e-40, 3.0), (f32::MAX, 2.0), (0.1, 0.2)] {
        assert_eq!(bin(BinOp::Add, Ty::F32, f(a), f(b)).unwrap(), f(a + b));
        assert_eq!(bin(BinOp::Sub, Ty::F32, f(a), f(b)).unwrap(), f(a - b));
        assert_eq!(bin(BinOp::Mul, Ty::F32, f(a), f(b)).unwrap(), f(a * b));
        assert_eq!(bin(BinOp::Div, Ty::F32, f(a), f(b)).unwrap(), f(a / b));
    }
    assert_eq!(bin(BinOp::Add, Ty::F64, d(0.1), d(0.2)).unwrap(), d(0.1 + 0.2));
    // f32/f64 min/max: cuda (NaN-ignoring, -0 < +0), as legacy tile reductions.
    assert_eq!(bin(BinOp::Min, Ty::F32, f(f32::NAN), f(2.0)).unwrap(), f(2.0));
    assert_eq!(bin(BinOp::Max, Ty::F32, f(-0.0), f(0.0)).unwrap(), f(0.0));
    assert_eq!(bin(BinOp::Min, Ty::F32, f(0.0), f(-0.0)).unwrap(), f(-0.0));
    assert_eq!(bin(BinOp::Max, Ty::F64, d(f64::NAN), d(-3.0)).unwrap(), d(-3.0));
    assert_eq!(bin(BinOp::Min, Ty::F64, d(-0.0), d(0.0)).unwrap(), d(-0.0));
    assert_eq!(bin(BinOp::Min, Ty::F64, d(0.0), d(-0.0)).unwrap(), d(-0.0));
    assert_eq!(bin(BinOp::Max, Ty::F64, d(-0.0), d(0.0)).unwrap(), d(0.0));
    assert_eq!(bin(BinOp::Max, Ty::F64, d(0.0), d(-0.0)).unwrap(), d(0.0));
    assert_eq!(bin(BinOp::Copysign, Ty::F32, f(2.0), f(-0.0)).unwrap(), f(-2.0));
    assert_eq!(kind(bin(BinOp::FloorMod, Ty::F32, f(1.0), f(2.0))), OpErrorKind::Unsupported);
    assert_eq!(kind(bin(BinOp::And, Ty::F32, f(1.0), f(2.0))), OpErrorKind::Unsupported);
}

#[test]
fn half_arith_and_vectors() {
    // float16x2: (1 + 1, 2 + 1) and RNE tie 1 + 2^-11 -> 1.
    let a = 0x4000_3c00u64;
    let b = 0x3c00_3c00u64;
    assert_eq!(bin(BinOp::Add, Ty::F16X2, a, b).unwrap(), 0x4200_4000);
    assert_eq!(bin(BinOp::Add, Ty::F16, 0x3c00, 0x1000).unwrap(), 0x3c00);
    assert_eq!(bin(BinOp::Add, Ty::F16, 0x3c00, 0x1001).unwrap(), 0x3c01);
    // bf16x2 multiply: 1.5 * 2 = 3, -1 * 0.5 = -0.5.
    assert_eq!(bin(BinOp::Mul, Ty::BF16X2, 0xbf80_3fc0, 0x3f00_4000).unwrap(), 0xbf00_4040);
    assert_eq!(un(UnOp::Neg, Ty::F16, 0x3c00).unwrap(), 0xbc00);
    assert_eq!(un(UnOp::Abs, Ty::BF16, 0xbf80).unwrap(), 0x3f80);
    // f16 fma, single rounding: (1+2^-10)^2 - 1 = 2^-9 + 2^-20 is exactly
    // half an f16 ulp above 2^-9 -> ties to even 2^-9.
    let x = 0x3c01u64;
    let mut out = s(0);
    ternary(TerOp::Fma, Ty::F16, &s(x), &s(x), &s(0xbc00), &mut out, WarpMask::ALL).unwrap();
    assert_eq!(out[0][0], 0x1800); // 2^-9
    // uint32x4 (two slots) element-wise add with per-element wrap.
    let ty = Ty::vector(Dtype::U32, 4);
    let a = vec![splat(0xffff_ffff_0000_0001u64), splat(0x0000_0003_0000_0002)];
    let b = vec![splat(0x0000_0001_0000_0010u64), splat(0x0000_0004_0000_0020)];
    let mut out = vec![splat(0), splat(0)];
    binary(BinOp::Add, ty, &a, &b, &mut out, WarpMask::ALL).unwrap();
    assert_eq!((out[0][5], out[1][5]), (0x0000_0000_0000_0011, 0x0000_0007_0000_0022));
    // Division by zero in one element of a vector is invalid.
    let z = vec![splat(0x0000_0000_0000_0001u64), splat(1)];
    assert_eq!(kind(binary(BinOp::Div, ty, &a, &z, &mut out, WarpMask::ALL)), OpErrorKind::Invalid);
}

#[test]
fn wide_b128_and_pred() {
    let ty = Ty::B128;
    let a = vec![splat(0xf0f0u64), splat(0x1)];
    let b = vec![splat(0xff00u64), splat(0x3)];
    let mut out = vec![splat(0), splat(0)];
    binary(BinOp::And, ty, &a, &b, &mut out, WarpMask::ALL).unwrap();
    assert_eq!((out[0][0], out[1][0]), (0xf000, 0x1));
    unary(UnOp::Not, ty, &a, &mut out, WarpMask::ALL).unwrap();
    assert_eq!((out[0][0], out[1][0]), (!0xf0f0u64, !1u64));
    assert_eq!(kind(binary(BinOp::Add, ty, &a, &b, &mut out, WarpMask::ALL)), OpErrorKind::Unsupported);
    assert!(compare(CmpOp::Ne, ty, &a, &b, WarpMask::ALL).unwrap().is_all());
    // Too few slots for a wide value is an operand error.
    assert_eq!(kind(binary(BinOp::And, ty, &a[..1], &b, &mut out, WarpMask::ALL)), OpErrorKind::Invalid);
    // Pred: logical ops, 0/1.
    assert_eq!(bin(BinOp::And, Ty::PRED, 1, 1).unwrap(), 1);
    assert_eq!(bin(BinOp::Xor, Ty::PRED, 1, 1).unwrap(), 0);
    assert_eq!(un(UnOp::Not, Ty::PRED, 0).unwrap(), 1);
    assert_eq!(kind(bin(BinOp::Add, Ty::PRED, 1, 1)), OpErrorKind::Unsupported);
    assert_eq!(kind(compare(CmpOp::Lt, Ty::PRED, &s(0), &s(1), WarpMask::ALL)), OpErrorKind::Unsupported);
    assert!(cmp(CmpOp::Eq, Ty::PRED, 1, 1));
}

#[test]
fn integer_semantics() {
    let s32 = |v: i32| v as u32 as u64;
    let s64 = |v: i64| v as u64;
    // Wrapping at the type width; registers keep own-width two's complement.
    assert_eq!(bin(BinOp::Add, Ty::U8, 200, 100).unwrap(), 44);
    assert_eq!(bin(BinOp::Mul, Ty::scalar(Dtype::S8), 0x10, 0x10).unwrap(), 0);
    assert_eq!(bin(BinOp::Sub, Ty::S32, s32(0), s32(1)).unwrap(), 0xffff_ffff);
    // Truncating Div/Mod; errors on zero and MIN / -1.
    assert_eq!(bin(BinOp::Div, Ty::S32, s32(-7), s32(2)).unwrap(), s32(-3));
    assert_eq!(bin(BinOp::Mod, Ty::S32, s32(-7), s32(2)).unwrap(), s32(-1));
    assert_eq!(kind(bin(BinOp::Div, Ty::S32, s32(1), 0)), OpErrorKind::Invalid);
    assert_eq!(kind(bin(BinOp::Mod, Ty::U32, 1, 0)), OpErrorKind::Invalid);
    assert_eq!(kind(bin(BinOp::Div, Ty::S32, s32(i32::MIN), s32(-1))), OpErrorKind::Invalid);
    // Floor division signs.
    for (a, b, q, r) in [(-7, 2, -4, 1), (7, -2, -4, -1), (-7, -2, 3, -1), (7, 2, 3, 1), (-8, 2, -4, 0)] {
        assert_eq!(bin(BinOp::FloorDiv, Ty::S32, s32(a), s32(b)).unwrap(), s32(q), "{a} // {b}");
        assert_eq!(bin(BinOp::FloorMod, Ty::S32, s32(a), s32(b)).unwrap(), s32(r), "{a} % {b}");
        assert_eq!(bin(BinOp::FloorDiv, Ty::S64, s64(a as i64), s64(b as i64)).unwrap(), s64(q as i64));
    }
    assert_eq!(kind(bin(BinOp::FloorDiv, Ty::S32, s32(1), 0)), OpErrorKind::Invalid);
    assert_eq!(kind(bin(BinOp::FloorMod, Ty::U32, 1, 0)), OpErrorKind::Invalid);
    // Legacy widens to i64 for floor ops: s32 MIN // -1 wraps; s64 errors.
    assert_eq!(bin(BinOp::FloorDiv, Ty::S32, s32(i32::MIN), s32(-1)).unwrap(), s32(i32::MIN));
    assert_eq!(kind(bin(BinOp::FloorDiv, Ty::S64, s64(i64::MIN), s64(-1))), OpErrorKind::Invalid);
    assert_eq!(bin(BinOp::FloorDiv, Ty::U32, 7, 2).unwrap(), 3);
    // Shifts: arithmetic for signed, count modulo width (wrapping_shl/shr).
    assert_eq!(bin(BinOp::Shr, Ty::S32, s32(-8), 1).unwrap(), s32(-4));
    assert_eq!(bin(BinOp::Shr, Ty::U32, 0x8000_0000, 31).unwrap(), 1);
    assert_eq!(bin(BinOp::Shl, Ty::U32, 1, 33).unwrap(), 2);
    assert_eq!(bin(BinOp::Shl, Ty::U8, 0x81, 1).unwrap(), 0x02);
    // Min/Max by signedness.
    assert_eq!(bin(BinOp::Min, Ty::S32, s32(-1), 1).unwrap(), s32(-1));
    assert_eq!(bin(BinOp::Min, Ty::U32, 0xffff_ffff, 1).unwrap(), 1);
    assert!(cmp(CmpOp::Lt, Ty::S32, s32(-1), 0));
    assert!(cmp(CmpOp::Gt, Ty::U32, 0xffff_ffff, 0));
    // Unary.
    assert_eq!(un(UnOp::Neg, Ty::S32, s32(5)).unwrap(), s32(-5));
    assert_eq!(un(UnOp::Not, Ty::U8, 0x0f).unwrap(), 0xf0);
    assert_eq!(un(UnOp::BitNot, Ty::U32, 0x0f).unwrap(), 0xffff_fff0);
    assert_eq!(un(UnOp::BitNot, Ty::U64, 0).unwrap(), u64::MAX);
    assert_eq!(un(UnOp::BitNot, Ty::PRED, 1).unwrap(), 0);
    assert_eq!(un(UnOp::Popcount, Ty::U32, 0xf0f0).unwrap(), 8);
    assert_eq!(un(UnOp::Clz, Ty::U32, 1).unwrap(), 31);
    assert_eq!(un(UnOp::Clz, Ty::U64, 1).unwrap(), 63);
    assert_eq!(kind(un(UnOp::Sqrt, Ty::S32, 4)), OpErrorKind::Unsupported);
    // 4-bit ints wrap at 4 bits.
    assert_eq!(bin(BinOp::Add, Ty::scalar(Dtype::S4), 0x7, 0x1).unwrap(), 0x8);
}

#[test]
fn masked_lanes_untouched_and_unsupported_fails_closed() {
    let a: WarpValue<u64> = std::array::from_fn(|l| l as u64);
    let mut out = vec![splat(0xdead_u64)];
    binary(BinOp::Add, Ty::U32, &[a], &s(10), &mut out, WarpMask(0b101)).unwrap();
    assert_eq!(&out[0][..4], &[10, 0xdead, 12, 0xdead]);
    assert_eq!(out[0][WARP_SIZE - 1], 0xdead);
    // Unsupported is reported even with an empty mask.
    assert_eq!(kind(unary(UnOp::Popcount, Ty::F32, &s(0), &mut out, WarpMask::NONE)), OpErrorKind::Unsupported);
    // ... but value errors only for active lanes.
    assert!(binary(BinOp::Div, Ty::U32, &s(1), &s(0), &mut out, WarpMask::NONE).is_ok());
}

#[test]
fn float_unary_and_compare() {
    assert_eq!(un(UnOp::Rsqrt, Ty::F32, f(4.0)).unwrap(), f(0.5));
    assert_eq!(un(UnOp::Exp, Ty::F32, f(1.0)).unwrap(), f(1.0f32.exp()));
    assert_eq!(un(UnOp::Log2, Ty::F32, f(8.0)).unwrap(), f(3.0));
    assert_eq!(un(UnOp::Round, Ty::F32, f(2.5)).unwrap(), f(3.0)); // C roundf
    assert_eq!(un(UnOp::Round, Ty::F32, f(-2.5)).unwrap(), f(-3.0));
    assert_eq!(un(UnOp::Floor, Ty::F64, d(-1.5)).unwrap(), d(-2.0));
    assert_eq!(un(UnOp::Neg, Ty::F32, f(0.0)).unwrap(), f(-0.0));
    assert_eq!(un(UnOp::Sqrt, Ty::F16, 0x4400).unwrap(), 0x4000); // sqrt(4) = 2
    // Is* write predicates (0/1), also per element for vectors.
    assert_eq!(un(UnOp::IsNan, Ty::F32, f(f32::NAN)).unwrap(), 1);
    assert_eq!(un(UnOp::IsFinite, Ty::F32, f(f32::INFINITY)).unwrap(), 0);
    assert_eq!(un(UnOp::IsInf, Ty::F16X2, 0x7c00_3c00).unwrap(), 0x0100);
    assert_eq!(un(UnOp::IsNan, Ty::scalar(Dtype::E4M3), 0x7f).unwrap(), 1);
    // NaN compares false except Ne.
    let nan = f(f32::NAN);
    assert!(!cmp(CmpOp::Eq, Ty::F32, nan, nan));
    assert!(cmp(CmpOp::Ne, Ty::F32, nan, nan));
    assert!(!cmp(CmpOp::Le, Ty::F32, nan, f(1.0)));
    assert!(cmp(CmpOp::Eq, Ty::F32, f(0.0), f(-0.0)));
    assert!(cmp(CmpOp::Lt, Ty::BF16, 0xbf80, 0x3f80));
    assert!(cmp(CmpOp::Ge, Ty::scalar(Dtype::E4M3), 0x38, 0x38));
    // Fma single rounding: x = 1 + 2^-12, x*x - 1 = 2^-11 + 2^-24.
    let x = 1.0f32 + 2f32.powi(-12);
    let mut out = s(0);
    ternary(TerOp::Fma, Ty::F32, &s(f(x)), &s(f(x)), &s(f(-1.0)), &mut out, WarpMask::ALL).unwrap();
    assert_eq!(out[0][0], f(2f32.powi(-11) + 2f32.powi(-24)));
    assert_ne!(f(x * x - 1.0), out[0][0]);
}

#[test]
fn casts_follow_legacy_c_semantics() {
    let (f32t, f16t, bf16t) = (Ty::F32, Ty::F16, Ty::BF16);
    // f32 -> f16 RNE ties.
    assert_eq!(cv(f32t, f16t, D, false, f(1.0 + 2f32.powi(-11))).unwrap(), 0x3c00);
    assert_eq!(cv(f32t, f16t, D, false, f(1.0 + 3.0 * 2f32.powi(-11))).unwrap(), 0x3c02);
    assert_eq!(cv(f32t, f16t, D, false, f(70000.0)).unwrap(), 0x7c00);
    assert_eq!(cv(f32t, f16t, D, true, f(70000.0)).unwrap(), 0x7bff);
    // f32 -> bf16 tie to even.
    assert_eq!(cv(f32t, bf16t, D, false, f(1.0 + 2f32.powi(-8))).unwrap(), 0x3f80);
    assert_eq!(cv(f32t, bf16t, D, false, f(1.0 + 3.0 * 2f32.powi(-8))).unwrap(), 0x3f82);
    // Legacy f64 -> f16 goes through f32 (double rounding); explicit Rn does not.
    let x = 1.0 + 2f64.powi(-11) + 2f64.powi(-40);
    assert_eq!(cv(Ty::F64, f16t, D, false, d(x)).unwrap(), 0x3c00);
    assert_eq!(cv(Ty::F64, f16t, Rounding::Rn, false, d(x)).unwrap(), 0x3c01);
    // f32 -> s32: truncation, saturation, NaN -> 0 (Rust `as`).
    let s32 = |v: i32| v as u32 as u64;
    assert_eq!(cv(f32t, Ty::S32, D, false, f(-2.7)).unwrap(), s32(-2));
    assert_eq!(cv(f32t, Ty::S32, D, false, f(3.9)).unwrap(), 3);
    assert_eq!(cv(f32t, Ty::S32, D, false, f(f32::NAN)).unwrap(), 0);
    assert_eq!(cv(f32t, Ty::S32, D, false, f(1e10)).unwrap(), s32(i32::MAX));
    assert_eq!(cv(f32t, Ty::U32, D, false, f(-1.5)).unwrap(), 0);
    assert_eq!(cv(f16t, Ty::U8, D, false, 0x5bf8).unwrap(), 255); // 255.0
    // Explicit integer rounding.
    assert_eq!(cv(f32t, Ty::S32, Rounding::Rn, false, f(2.5)).unwrap(), 2);
    assert_eq!(cv(f32t, Ty::S32, Rounding::Rna, false, f(2.5)).unwrap(), 3);
    assert_eq!(cv(f32t, Ty::S32, Rounding::Rm, false, f(-2.5)).unwrap(), s32(-3));
    assert_eq!(cv(f32t, Ty::S32, Rounding::Rp, false, f(2.1)).unwrap(), 3);
    // int -> float.
    assert_eq!(cv(Ty::S32, f32t, D, false, 16_777_217).unwrap(), f(16_777_216.0));
    assert_eq!(cv(Ty::S32, f32t, Rounding::Rp, false, 16_777_217).unwrap(), f(16_777_218.0));
    assert_eq!(cv(Ty::U64, f32t, Rounding::Rz, false, u64::MAX).unwrap(), 0x5f7f_ffff);
    assert_eq!(cv(Ty::U64, f32t, Rounding::Rn, false, u64::MAX).unwrap(), 0x5f80_0000);
    assert_eq!(cv(Ty::S64, Ty::F64, Rounding::Rz, false, (-(1i64 << 53) - 1) as u64).unwrap(), d(-(2f64.powi(53))));
    assert_eq!(cv(Ty::S32, f16t, D, false, s32(-2)).unwrap(), 0xc000);
    // int -> int: wrap / sign-extend, or clamp with sat.
    assert_eq!(cv(Ty::S32, Ty::U8, D, false, 300).unwrap(), 44);
    assert_eq!(cv(Ty::S32, Ty::U8, D, true, 300).unwrap(), 255);
    assert_eq!(cv(Ty::scalar(Dtype::S8), Ty::S32, D, false, 0xff).unwrap(), 0xffff_ffff);
    assert_eq!(cv(Ty::scalar(Dtype::S8), Ty::U64, D, false, 0x80).unwrap(), (-128i64) as u64);
    // bool.
    assert_eq!(cv(f32t, Ty::PRED, D, false, f(f32::NAN)).unwrap(), 1);
    assert_eq!(cv(f32t, Ty::PRED, D, false, f(-0.0)).unwrap(), 0);
    assert_eq!(cv(Ty::PRED, f32t, D, false, 1).unwrap(), f(1.0));
    assert_eq!(cv(Ty::PRED, Ty::S32, D, false, 1).unwrap(), 1);
    // fp8 / fp4.
    let e4m3 = Ty::scalar(Dtype::E4M3);
    assert_eq!(cv(f32t, e4m3, D, false, f(1.0)).unwrap(), 0x38);
    assert_eq!(cv(f32t, e4m3, D, false, f(448.0)).unwrap(), 0x7e);
    assert_eq!(cv(f32t, e4m3, D, false, f(1000.0)).unwrap(), 0x7e);
    assert_eq!(cv(f32t, e4m3, D, false, f(-1000.0)).unwrap(), 0xfe);
    assert_eq!(cv(f32t, e4m3, D, false, f(f32::NAN)).unwrap(), 0x7f);
    assert_eq!(cv(f32t, e4m3, D, false, f(1.0625)).unwrap(), 0x38); // tie -> even
    assert_eq!(cv(e4m3, f32t, D, false, 0xb8).unwrap(), f(-1.0));
    let e2m1 = Ty::scalar(Dtype::E2M1);
    assert_eq!(cv(f32t, e2m1, D, false, f(5.0)).unwrap(), 0x6); // tie 4|6 -> 4 (even code)
    assert_eq!(cv(f32t, e2m1, D, false, f(-7.0)).unwrap(), 0xf);
    assert_eq!(cv(e2m1, f32t, D, false, 0xf).unwrap(), f(-6.0));
    assert_eq!(cv(f32t, Ty::scalar(Dtype::E5M2), D, false, f(1.0)).unwrap(), 0x3c);
    // Unsupported forms.
    assert_eq!(kind(cv(f32t, f16t, Rounding::Rs, false, 0)), OpErrorKind::Unsupported);
    assert_eq!(kind(convert_bits(Ty::B128, Ty::U32, D, false, 0)), OpErrorKind::Unsupported);
    assert_eq!(kind(cv(f32t, Ty::scalar(Dtype::S2F6), D, false, 0)), OpErrorKind::Unsupported);
    assert_eq!(kind(cv(Ty::F64, e4m3, Rounding::Rn, true, 0)), OpErrorKind::Unsupported);
    // Same type: identity (incl. NaN payloads).
    assert_eq!(cv(f16t, f16t, D, false, 0x7c01).unwrap(), 0x7c01);
    // Vector cast float16x2 -> float32x2 (one slot -> one slot).
    assert_eq!(
        cv(Ty::F16X2, Ty::vector(Dtype::F32, 2), D, false, 0xc000_3c00).unwrap(),
        f(1.0) | (f(-2.0) << 32)
    );
}

#[test]
fn six_bit_elements_straddle_slots() {
    // float6_e3m2 x16 = 96 bits (2 slots); element 10 spans bits 60..66.
    let from = Ty::vector(Dtype::E3M2, 16);
    let to = Ty::vector(Dtype::F16, 16);
    let mut v = [0u64; 4];
    for e in 0..16 {
        let code = match e {
            10 => 0x10u128, // 2.0
            11 => 0x2c,     // -1.0
            _ => 0x0c,      // 1.0
        };
        elem::put(&mut v, e, 6, code);
    }
    let src = vec![splat(v[0]), splat(v[1])];
    let mut out = vec![splat(0u64); 4];
    cast(from, to, D, false, &src, &mut out, WarpMask::ALL).unwrap();
    let r = [out[0][7], out[1][7], out[2][7], out[3][7]];
    for e in 0..16 {
        let want = match e {
            10 => 0x4000,
            11 => 0xbc00,
            _ => 0x3c00,
        };
        assert_eq!(elem::get(&r, e, 16), want, "element {e}");
    }
}

#[test]
fn f32_from_f64_each_rounding() {
    let one = 1.0f32;
    let up = 1.0 + f32::EPSILON;
    let tie = 1.0 + 2f64.powi(-24);
    let cases: [(Rounding, f32, f32); 7] = [
        (Rounding::Default, one, -one),
        (Rounding::Rn, one, -one),
        (Rounding::Rna, up, -up),
        (Rounding::Rz, one, -one),
        (Rounding::Rm, one, -up),
        (Rounding::Rp, up, -one),
        (Rounding::Rs, one, -one),
    ];
    for (rnd, pos, neg) in cases {
        assert_eq!(f32_from_f64(tie, rnd, false).to_bits(), pos.to_bits(), "{rnd:?}");
        assert_eq!(f32_from_f64(-tie, rnd, false).to_bits(), neg.to_bits(), "{rnd:?} neg");
    }
    // Just above the tie: nearest modes go up.
    let above = tie + 2f64.powi(-40);
    assert_eq!(f32_from_f64(above, Rounding::Rn, false), up);
    assert_eq!(f32_from_f64(above, Rounding::Rna, false), up);
    assert_eq!(f32_from_f64(above, Rounding::Rz, false), one);
    // Overflow and saturation.
    assert_eq!(f32_from_f64(1e39, Rounding::Rn, false), f32::INFINITY);
    assert_eq!(f32_from_f64(1e39, Rounding::Rz, false), f32::MAX);
    assert_eq!(f32_from_f64(-1e39, Rounding::Rp, false), -f32::MAX);
    assert_eq!(f32_from_f64(1e39, Rounding::Rn, true), f32::MAX);
    let max_tie = f64::from(f32::MAX) + 2f64.powi(103);
    assert_eq!(f32_from_f64(max_tie, Rounding::Rna, false), f32::INFINITY);
    assert_eq!(f32_from_f64(max_tie - 2f64.powi(80), Rounding::Rna, false), f32::MAX);
    // Exact values and NaN payload (sign + high bits, quiet).
    assert_eq!(f32_from_f64(0.5, Rounding::Rp, false), 0.5);
    let nan = f64::from_bits(0xfff4_0000_2000_0000);
    assert_eq!(f32_from_f64(nan, Rounding::Rz, false).to_bits(), 0xffe0_0001);
    // Subnormal result.
    assert_eq!(f32_from_f64(2f64.powi(-149) * 1.5, Rounding::Rz, false).to_bits(), 1);
    assert_eq!(f32_from_f64(2f64.powi(-149) * 1.5, Rounding::Rn, false).to_bits(), 2);
    assert_eq!(<f32 as super::super::FloatScalar>::from_f64(tie, Rounding::Rp, false), up);
}

#[test]
fn convert_bits_single_values() {
    assert_eq!(convert_bits(Ty::F32, Ty::BF16, D, false, 0x3f80_0000).unwrap(), 0x3f80);
    assert_eq!(
        convert_bits(Ty::F16X2, Ty::vector(Dtype::F32, 2), D, false, 0x4000_3c00).unwrap(),
        u128::from(f(1.0)) | (u128::from(f(2.0)) << 32)
    );
    let f32x4 = Ty::vector(Dtype::F32, 4);
    let e4m3x4 = Ty::vector(Dtype::E4M3, 4);
    let src = [1.0f32, -2.0, 0.5, 448.0]
        .iter()
        .enumerate()
        .fold(0u128, |acc, (i, x)| acc | (u128::from(x.to_bits()) << (32 * i)));
    assert_eq!(convert_bits(f32x4, e4m3x4, D, false, src).unwrap(), 0x7e30_c038);
    assert_eq!(convert_bits(Ty::F32, Ty::F16, Rounding::Rz, false, u128::from(f(70000.0))).unwrap(), 0x7bff);
    assert_eq!(convert_bits(Ty::F32, Ty::F16, Rounding::Rna, false, u128::from(f(1.0 + 2f32.powi(-11)))).unwrap(), 0x3c01);
    assert_eq!(convert_bits(Ty::B128, Ty::B128, D, false, u128::MAX).unwrap(), u128::MAX);
    assert_eq!(kind(convert_bits(Ty::vector(Dtype::F32, 8), Ty::vector(Dtype::F32, 8), D, false, 0)), OpErrorKind::Unsupported);
    assert_eq!(kind(convert_bits(Ty::F16X2, Ty::F32, D, false, 0)), OpErrorKind::Invalid);
}

#[test]
fn half_registers_are_plain_zero_extended_bits() {
    // Each half op rounds (RNE); chains in f32 are lowering's job (D1).
    let sum = bin(BinOp::Add, Ty::F16, 0x3c00, 0x1000).unwrap(); // 1 + 2^-11 -> tie -> 1
    assert_eq!(sum, 0x3c00);
    assert_eq!(bin(BinOp::Add, Ty::F16, sum, 0x1000).unwrap(), 0x3c00);
    assert_eq!(cv(Ty::F16, Ty::F32, Rounding::Default, false, sum).unwrap(), f(1.0));
}

/// Bit-exact equivalence of every monomorphic fast path (`fast.rs`) with the
/// generic element path over random and special operands and partial masks.
#[test]
fn fast_paths_match_the_generic_path() {
    use super::{binary_generic, cast_generic, compare_generic, ternary_generic, unary_generic};
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let specials: [u64; 16] = [
        0, 1, 0xffff_ffff_ffff_ffff, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_0001, 0xffc0_0000,
        0x7ff8_0000_0000_0001, 0x8000_0000_0000_0000, 0x7c00, 0xfc01, 0x7f7f_ffff, 0x0000_0001_0000_0000, 0x80, 0x7fff,
    ];
    let mut operand = |ty: Ty| -> Vec<WarpValue<u64>> {
        let m = if ty.bits() >= 64 { u64::MAX } else { (1u64 << ty.bits()) - 1 };
        vec![std::array::from_fn(|l| (if l % 4 == 0 { specials[(next() % 16) as usize] } else { next() }) & m)]
    };
    let ints = [Dtype::U8, Dtype::U16, Dtype::U32, Dtype::U64, Dtype::S8, Dtype::S16, Dtype::S32, Dtype::S64];
    let floats = [Dtype::F32, Dtype::F64, Dtype::F16, Dtype::BF16];
    let all: Vec<Dtype> = ints.iter().chain(floats.iter()).copied().chain([Dtype::Pred]).collect();
    let mask = WarpMask(0xdead_beef);
    let same = |what: String, f: OpResult, g: OpResult, fo: &[WarpValue<u64>], go: &[WarpValue<u64>]| {
        match (&f, &g) {
            (Ok(()), Ok(())) => assert_eq!(fo, go, "{what}"),
            (Err(x), Err(y)) => assert_eq!(x, y, "{what}"),
            _ => panic!("{what}: fast {f:?} generic {g:?}"),
        }
    };
    for _round in 0..64 {
        for &d in &all {
            let ty = Ty::scalar(d);
            for op in [
                BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Mod, BinOp::FloorDiv, BinOp::FloorMod,
                BinOp::Min, BinOp::Max, BinOp::And, BinOp::Or, BinOp::Xor, BinOp::Shl, BinOp::Shr,
            ] {
                let (a, b, init) = (operand(ty), operand(ty), operand(ty));
                if fast::binary(op, ty, &a, &b, &mut init.clone(), mask).is_none() {
                    continue;
                }
                let (mut fo, mut go) = (init.clone(), init.clone());
                let f = fast::binary(op, ty, &a, &b, &mut fo, mask).unwrap();
                let g = binary_generic(op, ty, &a, &b, &mut go, mask);
                same(format!("binary {op:?} {d}"), f, g, &fo, &go);
            }
            for op in [UnOp::Neg, UnOp::Abs, UnOp::Not, UnOp::BitNot, UnOp::Sqrt, UnOp::Exp, UnOp::Exp2, UnOp::Log, UnOp::Log2, UnOp::Rsqrt] {
                let (a, init) = (operand(ty), operand(ty));
                let (mut fo, mut go) = (init.clone(), init.clone());
                let Some(f) = fast::unary(op, ty, &a, &mut fo, mask) else { continue };
                let g = unary_generic(op, ty, &a, &mut go, mask);
                same(format!("unary {op:?} {d}"), f, g, &fo, &go);
            }
            {
                let (a, b, c, init) = (operand(ty), operand(ty), operand(ty), operand(ty));
                let (mut fo, mut go) = (init.clone(), init.clone());
                if let Some(f) = fast::ternary(TerOp::Fma, ty, &a, &b, &c, &mut fo, mask) {
                    let g = ternary_generic(TerOp::Fma, ty, &a, &b, &c, &mut go, mask);
                    same(format!("fma {d}"), f, g, &fo, &go);
                }
            }
            for op in [CmpOp::Eq, CmpOp::Ne, CmpOp::Lt, CmpOp::Le, CmpOp::Gt, CmpOp::Ge] {
                let (a, b) = (operand(ty), operand(ty));
                let Some(f) = fast::compare(op, ty, &a, &b, mask) else { continue };
                assert_eq!(f, compare_generic(op, ty, &a, &b, mask), "compare {op:?} {d}");
            }
            for &to in &all {
                let to_ty = Ty::scalar(to);
                let (a, init) = (operand(ty), operand(to_ty));
                let (mut fo, mut go) = (init.clone(), init.clone());
                let Some(f) = fast::cast(ty, to_ty, Rounding::Default, false, &a, &mut fo, mask) else { continue };
                let g = cast_generic(ty, to_ty, Rounding::Default, false, &a, &mut go, mask);
                same(format!("cast {d} -> {to}"), f, g, &fo, &go);
            }
        }
    }
}
