//! Bit-exact ALU cases through `resolve_ptx` (expected values from the
//! ported legacy tests in `numsim-oplib/src/{arith,scalar}/tests.rs` where
//! available), plus fail-closed checks.

use crate::dtype::{Dtype, Ty};
use crate::oplib::{resolve_ptx, OpErrorKind, OpResult, PtxIo};
use crate::program::OpKey;
use crate::value::{WarpMask, WarpValue};

const POISON: u64 = 0xdead_beef_dead_beef;
const S16: Ty = Ty::scalar(Dtype::S16);

fn key(name: &str, mods: &[&str]) -> OpKey {
    OpKey {
        name: format!("tirx.ptx.{name}"),
        mods: mods.iter().map(|m| m.to_string()).collect(),
    }
}

/// Run on lanes 0 and 7 with identical source slots; return lane 0's
/// destination slots (checking lane 7 agrees and inactive lanes are untouched).
fn run(
    name: &str,
    mods: &[&str],
    dst_tys: &[Ty],
    src_tys: &[Ty],
    srcs: &[u64],
) -> OpResult<Vec<u64>> {
    let f = resolve_ptx(&key(name, mods), dst_tys, src_tys)?;
    let dst_slots: usize = dst_tys.iter().map(|t| t.slots() as usize).sum();
    let src_slots: usize = src_tys.iter().map(|t| t.slots() as usize).sum();
    assert_eq!(src_slots, srcs.len(), "{name}: source slot count");
    let src_vals: Vec<WarpValue<u64>> = srcs.iter().map(|&v| [v; 32]).collect();
    let mut dst_vals: Vec<WarpValue<u64>> = vec![[POISON; 32]; dst_slots];
    let mut io = PtxIo {
        dsts: &mut dst_vals,
        dst_tys,
        srcs: &src_vals,
        src_tys,
        mask: WarpMask(1 | (1 << 7)),
    };
    f(&mut io)?;
    for slot in &dst_vals {
        assert_eq!(slot[0], slot[7], "{name}: lanes disagree");
        assert_eq!(slot[1], POISON, "{name}: inactive lane written");
    }
    Ok(dst_vals.iter().map(|slot| slot[0]).collect())
}

fn ok(name: &str, mods: &[&str], dst_tys: &[Ty], src_tys: &[Ty], srcs: &[u64]) -> Vec<u64> {
    run(name, mods, dst_tys, src_tys, srcs).unwrap_or_else(|e| panic!("{name} {mods:?}: {e}"))
}

fn one(name: &str, mods: &[&str], dst: Ty, src_tys: &[Ty], srcs: &[u64]) -> u64 {
    ok(name, mods, &[dst], src_tys, srcs)[0]
}

fn kind(result: OpResult<Vec<u64>>) -> OpErrorKind {
    result.expect_err("expected an error").kind
}

fn f(v: f32) -> u64 {
    u64::from(v.to_bits())
}

fn d(v: f64) -> u64 {
    v.to_bits()
}

fn f2(x: f32, y: f32) -> u64 {
    f(x) | (f(y) << 32)
}

const F32: Ty = Ty::F32;
const U32: Ty = Ty::U32;
const U16: Ty = Ty::U16;
const U64: Ty = Ty::U64;
const S32: Ty = Ty::S32;
const PRED: Ty = Ty::PRED;

#[test]
fn f32_f64_and_f32x2_arithmetic() {
    let half_ulp = f32::from_bits(0x3380_0000);
    // Rounding, saturation, both modifier spellings.
    assert_eq!(
        one(
            "add",
            &["rnd=rp", "type=f32"],
            F32,
            &[F32, F32],
            &[f(1.0), f(half_ulp)]
        ),
        0x3f80_0001
    );
    assert_eq!(
        one(
            "add",
            &["rp", "f32"],
            F32,
            &[F32, F32],
            &[f(1.0), f(half_ulp)]
        ),
        0x3f80_0001
    );
    assert_eq!(
        one(
            "add",
            &["rp", "sat", "f32"],
            F32,
            &[F32, F32],
            &[f(1.0), f(half_ulp)]
        ),
        f(1.0)
    );
    // Hot direct forms (no rnd = RN).
    assert_eq!(
        one("add", &["f32"], F32, &[F32, F32], &[f(1.0), f(half_ulp)]),
        f(1.0)
    );
    assert_eq!(
        one("mul", &["type=f32"], F32, &[F32, F32], &[f(3.0), f(4.0)]),
        f(12.0)
    );
    assert_eq!(
        one(
            "mul",
            &["rnd=rn", "sat=sat", "type=f32"],
            F32,
            &[F32, F32],
            &[f(f32::NAN), f(1.0)]
        ),
        f(0.0)
    );
    assert_eq!(
        one(
            "sub",
            &["ftz", "f32"],
            F32,
            &[F32, F32],
            &[f(f32::from_bits(1)), f(0.0)]
        ),
        f(0.0)
    );
    assert_eq!(
        one(
            "fma",
            &["rn", "f32"],
            F32,
            &[F32, F32, F32],
            &[f(1.5), f(2.0), f(0.25)]
        ),
        f(3.25)
    );
    assert_eq!(
        one(
            "fma",
            &["rn", "ftz", "f32"],
            F32,
            &[F32, F32, F32],
            &[f(1.5), f(2.0), f(0.25)]
        ),
        f(3.25)
    );
    assert_eq!(
        one(
            "mad_f",
            &["rz", "f32"],
            F32,
            &[F32, F32, F32],
            &[f(1.5), f(2.0), f(0.25)]
        ),
        f(3.25)
    );
    // Result into a wider carrier is zero-extended.
    assert_eq!(
        one("mul", &["f32"], U64, &[F32, F32], &[f(-1.0), f(1.0)]),
        f(-1.0)
    );
    // f64.
    assert_eq!(
        one(
            "add",
            &["rn", "f64"],
            Ty::F64,
            &[Ty::F64, Ty::F64],
            &[d(1.5), d(2.25)]
        ),
        d(3.75)
    );
    assert_eq!(
        one(
            "fma",
            &["rn", "f64"],
            Ty::F64,
            &[Ty::F64; 3],
            &[d(2.0), d(3.0), d(1.0)]
        ),
        d(7.0)
    );
    assert_eq!(
        one(
            "div_f",
            &["rz", "f64"],
            Ty::F64,
            &[Ty::F64; 2],
            &[d(1.0), d(4.0)]
        ),
        d(0.25)
    );
    assert_eq!(
        kind(run(
            "add",
            &["rn", "ftz", "f64"],
            &[Ty::F64],
            &[Ty::F64; 2],
            &[0, 0]
        )),
        OpErrorKind::Unsupported
    );
    // f32x2 (lane order preserved).
    assert_eq!(
        one(
            "mul",
            &["rn", "ftz", "f32x2"],
            U64,
            &[U64, U64],
            &[f2(2.0, 3.0), f2(4.0, 5.0)]
        ),
        f2(8.0, 15.0)
    );
    assert_eq!(
        one(
            "fma",
            &["rn", "f32x2"],
            U64,
            &[U64; 3],
            &[f2(2.0, 3.0), f2(4.0, 5.0), f2(1.0, 2.0)]
        ),
        f2(9.0, 17.0)
    );
    assert_eq!(
        kind(run("add", &["sat", "f32x2"], &[U64], &[U64, U64], &[0, 0])),
        OpErrorKind::Unsupported
    );
    // Mixed f32 <- bf16 source.
    let half_bf16 = 0x3f00;
    assert_eq!(
        one(
            "add",
            &["rn", "f32", "bf16"],
            F32,
            &[U16, F32],
            &[half_bf16, f(4.0)]
        ),
        f(4.5)
    );
    assert_eq!(
        one(
            "sub",
            &["rn", "sat", "f32", "bf16"],
            F32,
            &[U16, F32],
            &[half_bf16, f(4.0)]
        ),
        f(0.0)
    );
    assert_eq!(
        kind(run(
            "add",
            &["rn", "ftz", "f32", "bf16"],
            &[F32],
            &[U16, F32],
            &[0, 0]
        )),
        OpErrorKind::Unsupported
    );
    // div / copysign / neg / abs / min / max.
    assert_eq!(
        one(
            "div_f",
            &["mode=rz", "type=f32"],
            F32,
            &[F32, F32],
            &[f(1.0), f(4.0)]
        ),
        f(0.25)
    );
    assert_eq!(
        one("copysign", &["f32"], F32, &[F32, F32], &[f(-1.0), f(2.0)]),
        f(-2.0)
    );
    assert_eq!(one("neg", &["f32"], F32, &[F32], &[f(2.0)]), f(-2.0));
    assert_eq!(
        one(
            "abs_f",
            &["ftz", "f32"],
            F32,
            &[F32],
            &[f(-f32::from_bits(1))]
        ),
        f(0.0)
    );
    assert_eq!(
        one("abs_f", &["f64"], Ty::F64, &[Ty::F64], &[d(-f64::NAN)]),
        d(-f64::NAN)
    );
    assert_eq!(
        one(
            "max3",
            &["abs", "f32"],
            F32,
            &[F32; 3],
            &[f(-3.0), f(2.0), f(-5.0)]
        ),
        f(5.0)
    );
    assert_eq!(
        one(
            "max3",
            &["f32"],
            F32,
            &[F32; 3],
            &[f(-3.0), f(2.0), f(-5.0)]
        ),
        f(2.0)
    );
    assert_eq!(
        one(
            "max",
            &["xorsign", "abs", "f32"],
            F32,
            &[F32; 2],
            &[f(-3.0), f(2.0)]
        ),
        f(-3.0)
    );
    assert_eq!(
        one("max", &["f32"], F32, &[F32; 2], &[f(f32::NAN), f(1.0)]),
        f(1.0)
    );
    assert!(f32::from_bits(one(
        "max",
        &["NaN", "f32"],
        F32,
        &[F32; 2],
        &[f(f32::NAN), f(1.0)]
    ) as u32)
    .is_nan());
    assert_eq!(
        one("min", &["f64"], Ty::F64, &[Ty::F64; 2], &[d(1.0), d(-2.0)]),
        d(-2.0)
    );
    assert_eq!(
        kind(run("max", &["relu", "f32"], &[F32], &[F32; 2], &[0, 0])),
        OpErrorKind::Unsupported
    );
}

#[test]
fn approximate_transcendentals() {
    // Canonical representatives from `scalar/tests.rs`.
    assert_eq!(
        one("ex2", &["approx", "ftz", "f32"], F32, &[F32], &[f(0.1)]),
        0x3f89_2fdf
    );
    assert_eq!(one("ex2", &["approx", "f32"], F32, &[F32], &[f(-149.0)]), 1);
    assert_eq!(
        one("ex2", &["approx", "ftz", "f32"], F32, &[F32], &[f(-149.0)]),
        0
    );
    assert_eq!(
        one("rcp", &["approx", "ftz", "f32"], F32, &[F32], &[f(-3.75)]),
        0xbe88_8889
    );
    assert_eq!(one("rcp", &["rn", "f32"], F32, &[F32], &[f(4.0)]), f(0.25));
    assert_eq!(
        one("rcp", &["rn", "f64"], Ty::F64, &[Ty::F64], &[d(4.0)]),
        d(0.25)
    );
    assert_eq!(
        one("rsqrt", &["approx", "ftz", "f32"], F32, &[F32], &[f(0.25)]),
        f(2.0)
    );
    assert_eq!(one("sqrt", &["rn", "f32"], F32, &[F32], &[f(16.0)]), f(4.0));
    assert_eq!(
        one("sqrt", &["rz", "f64"], Ty::F64, &[Ty::F64], &[d(16.0)]),
        d(4.0)
    );
    assert_eq!(
        one("tanh", &["approx", "f32"], F32, &[F32], &[f(1.0)]),
        f(0.761_594_2)
    );
    assert_eq!(
        one("lg2", &["approx", "f32"], F32, &[F32], &[f(8.0)]),
        f(3.0)
    );
    assert_eq!(
        one("sin", &["approx", "f32"], F32, &[F32], &[f(0.0)]),
        f(0.0)
    );
    assert_eq!(
        one("cos", &["approx", "ftz", "f32"], F32, &[F32], &[f(0.0)]),
        f(1.0)
    );
    assert_eq!(
        one(
            "ex2_half",
            &["approx", "f16x2"],
            U32,
            &[U32],
            &[0x3c00_0000]
        ),
        0x4000_3c00
    );
    assert_eq!(
        one(
            "ex2_half",
            &["approx", "f16x2"],
            U32,
            &[U32],
            &[0xce00_cb00]
        ),
        0x0001_0400
    );
    // bf16 requires .ftz, f16 forbids it.
    assert_eq!(
        kind(run("ex2_half", &["approx", "bf16"], &[U16], &[U16], &[0])),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        kind(run(
            "ex2_half",
            &["approx", "ftz", "f16"],
            &[U16],
            &[U16],
            &[0]
        )),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        kind(run(
            "sqrt",
            &["approx", "f64"],
            &[Ty::F64],
            &[Ty::F64],
            &[0]
        )),
        OpErrorKind::Unsupported
    );
}

#[test]
fn half_and_mixed_vector_arithmetic() {
    // RN-even ties (legacy low-precision tests).
    assert_eq!(
        one(
            "add_half",
            &["rn", "f16"],
            U16,
            &[U16, U16],
            &[0x3c00, 0x1000]
        ),
        0x3c00
    );
    assert_eq!(
        one("add_half", &["bf16"], U16, &[U16, U16], &[0x3f80, 0x3b80]),
        0x3f80
    );
    assert_eq!(
        one(
            "fma_half",
            &["rn", "f16"],
            U16,
            &[U16; 3],
            &[0x3e00, 0x4000, 0x3800]
        ),
        0x4300
    );
    assert_eq!(
        one(
            "fma_half",
            &["rn", "f16x2"],
            U32,
            &[U32; 3],
            &[0x3e00_3e00, 0x4000_4000, 0x3800_3800]
        ),
        0x4300_4300
    );
    // relu clamps a negative result to +0; sat clamps 2.0 to 1.0.
    assert_eq!(
        one(
            "fma_half",
            &["rn", "relu", "f16"],
            U16,
            &[U16; 3],
            &[0xbc00, 0x3c00, 0x0000]
        ),
        0
    );
    assert_eq!(
        one(
            "mul_half",
            &["rn", "sat", "f16"],
            U16,
            &[U16; 2],
            &[0x4000, 0x3c00]
        ),
        0x3c00
    );
    assert_eq!(
        kind(run(
            "fma_half",
            &["rn", "sat", "relu", "f16"],
            &[U16],
            &[U16; 3],
            &[0, 0, 0]
        )),
        OpErrorKind::Unsupported
    );
    // neg / abs / min / max.
    assert_eq!(
        one("neg_half", &["f16x2"], U32, &[U32], &[0x3c00_bc00]),
        0xbc00_3c00
    );
    assert_eq!(
        one("abs_half", &["bf16x2"], U32, &[U32], &[0xbf80_3f80]),
        0x3f80_3f80
    );
    assert_eq!(
        kind(run("neg_half", &["ftz", "bf16"], &[U16], &[U16], &[0])),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        one(
            "max",
            &["f16x2"],
            U32,
            &[U32; 2],
            &[0x3c00_4000, 0x4000_3c00]
        ),
        0x4000_4000
    );
    assert_eq!(
        one("min", &["bf16"], U16, &[U16; 2], &[0x3f80, 0xbf80]),
        0xbf80
    );
    assert_eq!(
        kind(run("max", &["abs", "f16"], &[U16], &[U16; 2], &[0, 0])),
        OpErrorKind::Unsupported
    );
    assert_eq!(one("tanh_half", &["approx", "f16"], U16, &[U16], &[0]), 0);
    // PTX 9.4 mixed vectors (legacy expected bits).
    let f16_one_negative_one = 0xbc00_3c00;
    let half_ulp = f2(2.0_f32.powi(-24), -(2.0_f32.powi(-24)));
    let up = ["rnd=rm", "dtype=f32x2", "atype=f16x2", "ctype=f32x2"];
    assert_eq!(
        one(
            "add_mixed_vec_up",
            &up,
            U64,
            &[U32, U64],
            &[f16_one_negative_one, half_ulp]
        ),
        0xbf80_0001_3f80_0000
    );
    assert_eq!(
        one(
            "add_mixed_vec_up",
            &["f32x2", "f16x2", "f32x2"],
            U64,
            &[U32, U64],
            &[f16_one_negative_one, half_ulp]
        ),
        0xbf80_0000_3f80_0000
    );
    assert_eq!(
        one(
            "sub_mixed_vec_up",
            &["rn", "f32x2", "bf16x2", "f32x2"],
            U64,
            &[U32, U64],
            &[0x4000_3f80, f2(0.5, 1.0)]
        ),
        f2(0.5, 1.0)
    );
    assert_eq!(
        one(
            "fma_mixed_vec",
            &["rz", "f32x2", "bf16x2", "f32x2", "f32x2"],
            U64,
            &[U32, U64, U64],
            &[0x4040_4040, f2(2.0, 4.0), f2(1.0, 1.0)]
        ),
        f2(7.0, 13.0)
    );
    let (lhs, rhs) = (f2(4.0, 8.0), f2(0.5, 1.0));
    let down_f16 = ["rz", "ftz", "f16x2", "f32x2", "f32x2"];
    assert_eq!(
        one(
            "add_mixed_vec_down_f16",
            &down_f16,
            U32,
            &[U64, U64],
            &[lhs, rhs]
        ),
        0x4880_4480
    );
    assert_eq!(
        one(
            "sub_mixed_vec_down_bf16",
            &["rz", "bf16x2", "f32x2", "f32x2"],
            U32,
            &[U64, U64],
            &[lhs, rhs]
        ),
        0x40e0_4060
    );
    assert_eq!(
        one(
            "mul_mixed_vec_down_f16",
            &["ftz", "rz", "f16x2", "f32x2", "f32x2"],
            U32,
            &[U64, U64],
            &[lhs, rhs]
        ),
        0x4800_4000
    );
    assert_eq!(
        kind(run(
            "add_mixed_vec_down_f16",
            &["rz", "f16x2", "f32x2", "f32x2"],
            &[U32],
            &[U64, U64],
            &[0, 0]
        )),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        one(
            "mul_mixed_vec_bf16_f16",
            &["bf16x2", "bf16x2", "f16x2"],
            U32,
            &[U32, U32],
            &[0x3dcd_3dcd, 0x3003_3003]
        ),
        0x3c4d_3c4d
    );
    assert_eq!(
        one(
            "mul_mixed_vec_f16_bf16",
            &["f16x2", "f16x2", "bf16x2"],
            U32,
            &[U32, U32],
            &[0x0e6b_0e6b, 0x4d08_4d08]
        ),
        0x7c00_7c00
    );
}

#[test]
fn integer_arithmetic() {
    let m = |v: i32| u64::from(v as u32);
    assert_eq!(
        one("add_int", &["s32"], S32, &[S32, S32], &[m(i32::MAX), 2]),
        m(i32::MIN + 1)
    );
    assert_eq!(
        one("add_int", &["u32"], U32, &[U32, U32], &[0xffff_ffff, 2]),
        1
    );
    assert_eq!(
        one(
            "add_int",
            &["sat", "s32"],
            S32,
            &[S32, S32],
            &[m(i32::MAX), 2]
        ),
        m(i32::MAX)
    );
    assert_eq!(
        one(
            "sub_int",
            &["sat", "s32"],
            S32,
            &[S32, S32],
            &[m(i32::MIN), 2]
        ),
        m(i32::MIN)
    );
    assert_eq!(
        kind(run("add_int", &["sat", "u32"], &[U32], &[U32; 2], &[0, 0])),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        one(
            "add_int",
            &["u16x2"],
            U32,
            &[U32, U32],
            &[0xffff_0001, 0x0001_ffff]
        ),
        0
    );
    // s32 result into an s64 carrier is sign-extended; s16 source read from the low bits.
    assert_eq!(
        one("add_int", &["s32"], Ty::S64, &[S32, S32], &[m(-3), 1]),
        (-2i64) as u64
    );
    assert_eq!(
        one("add_int", &["s16"], S16, &[S16, S16], &[0xffff, 0xffff]),
        0xfffe
    );
    assert_eq!(
        one("mul_int", &["hi", "u32"], U32, &[U32; 2], &[0xffff_ffff, 2]),
        1
    );
    assert_eq!(
        one("mul_int", &["lo", "s16"], S16, &[S16; 2], &[0x7fff, 2]),
        0xfffe
    );
    assert_eq!(
        one(
            "mad_int",
            &["hi", "sat", "s32"],
            S32,
            &[S32; 3],
            &[m(i32::MAX), 2, m(i32::MAX)]
        ),
        m(i32::MAX)
    );
    assert_eq!(
        kind(run(
            "mad_int",
            &["lo", "sat", "s32"],
            &[S32],
            &[S32; 3],
            &[0, 0, 0]
        )),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        one(
            "mul_wide",
            &["wide", "s16"],
            S32,
            &[S16, S16],
            &[(-30_000i16) as u16 as u64, 2]
        ),
        m(-60_000)
    );
    assert_eq!(
        one(
            "mad_wide",
            &["wide", "u16"],
            U32,
            &[U16, U16, U32],
            &[0xffff, 0xffff, 0xffff_ffff]
        ),
        4_294_836_224
    );
    assert_eq!(
        one(
            "mul_wide",
            &["wide", "u32"],
            U64,
            &[U32, U32],
            &[0xffff_ffff, 0xffff_ffff]
        ),
        0xffff_fffe_0000_0001
    );
    assert_eq!(
        one("mul24", &["hi", "s32"], S32, &[S32; 2], &[0x0080_0000, 2]),
        m(-256)
    );
    assert_eq!(
        one(
            "mad24",
            &["hi", "sat", "s32"],
            S32,
            &[S32; 3],
            &[0x007f_ffff, 0x007f_ffff, m(i32::MAX)]
        ),
        m(i32::MAX)
    );
    assert_eq!(
        one(
            "sad",
            &["s32"],
            S32,
            &[S32; 3],
            &[m(i32::MIN), m(i32::MAX), 7]
        ),
        6
    );
    assert_eq!(one("div", &["s32"], S32, &[S32; 2], &[m(-7), 3]), m(-2));
    assert_eq!(
        kind(run("div", &["u32"], &[U32], &[U32; 2], &[1, 0])),
        OpErrorKind::Invalid
    );
    assert_eq!(one("rem", &["s32"], S32, &[S32; 2], &[m(-7), m(-3)]), m(-1));
    assert_eq!(one("neg_int", &["s16"], S16, &[S16], &[0x8000]), 0x8000);
    assert_eq!(one("abs", &["s32"], S32, &[S32], &[m(-5)]), 5);
    assert_eq!(
        one(
            "dp4a",
            &["u32", "u32"],
            U32,
            &[U32; 3],
            &[0x0403_0201, 0x0403_0201, 5]
        ),
        35
    );
    assert_eq!(
        one(
            "dp4a",
            &["u32", "s32"],
            S32,
            &[U32, S32, S32],
            &[0x0403_0201, 0xfc03_fe01, 5]
        ),
        m(-5)
    );
    assert_eq!(
        one(
            "dp2a",
            &["hi", "s32", "s32"],
            S32,
            &[S32; 3],
            &[0xfffd_0002, 0x07fa_0504, 10]
        ),
        m(-23)
    );
    assert_eq!(
        one(
            "max",
            &["s16x2"],
            U32,
            &[U32; 2],
            &[0x8000_0005, 0x0001_0003]
        ),
        0x0001_0005
    );
    assert_eq!(
        one(
            "min",
            &["relu", "s16x2"],
            U32,
            &[U32; 2],
            &[0x8000_0005, 0x0001_0003]
        ),
        0x0000_0003
    );
    assert_eq!(
        one("max", &["relu", "s32"], S32, &[S32; 2], &[m(-5), m(-3)]),
        0
    );
    assert_eq!(
        one(
            "min",
            &["s64"],
            Ty::S64,
            &[Ty::S64; 2],
            &[(-5i64) as u64, 3]
        ),
        (-5i64) as u64
    );
    assert_eq!(
        one("max", &["u64"], U64, &[U64; 2], &[u64::MAX, 3]),
        u64::MAX
    );
    assert_eq!(
        kind(run("max", &["relu", "u32"], &[U32], &[U32; 2], &[0, 0])),
        OpErrorKind::Unsupported
    );
}

#[test]
fn logic_and_bit_manipulation() {
    let (a, b, c) = (0xf0f0_0f0f_u64, 0xcccc_3333_u64, 0xaaaa_5555_u64);
    assert_eq!(
        one("lop3", &["b32"], U32, &[U32; 4], &[a, b, c, 0x80]),
        a & b & c
    );
    assert_eq!(
        one("lop3", &["b32"], U32, &[U32; 4], &[a, b, c, 0x1a]),
        ((a & b) | c) ^ a
    );
    assert_eq!(
        kind(run("lop3", &["b32"], &[U32], &[U32; 4], &[a, b, c, 0x100])),
        OpErrorKind::Invalid
    );
    assert_eq!(
        ok(
            "lop3_bool",
            &["or", "b32"],
            &[U32, PRED],
            &[U32, U32, U32, U32, PRED],
            &[a, b, c, 0, 1]
        ),
        vec![0, 1]
    );
    assert_eq!(
        ok(
            "lop3_bool",
            &["and", "b32"],
            &[U32, PRED],
            &[U32, U32, U32, U32, PRED],
            &[a, b, c, 0xff, 1]
        ),
        vec![0xffff_ffff, 1]
    );
    assert_eq!(
        ok(
            "lop3_bool_sink",
            &["and", "b32"],
            &[PRED],
            &[U32, U32, U32, U32, PRED],
            &[a, b, c, 0, 1]
        ),
        vec![0]
    );
    assert_eq!(one("and", &["pred"], PRED, &[PRED; 2], &[1, 0]), 0);
    assert_eq!(one("not", &["pred"], PRED, &[PRED], &[0]), 1);
    assert_eq!(one("xor", &["b32"], U32, &[U32; 2], &[0xf0, 0x33]), 0xc3);
    assert_eq!(one("not", &["b16"], U16, &[U16], &[0]), 0xffff);
    assert_eq!(
        one("or", &["b64"], U64, &[U64; 2], &[1 << 40, 1]),
        (1 << 40) | 1
    );
    assert_eq!(one("cnot", &["b32"], U32, &[U32], &[0]), 1);
    assert_eq!(one("brev", &["b32"], U32, &[U32], &[1]), 0x8000_0000);
    assert_eq!(one("clz", &["b64"], U32, &[U64], &[1]), 63);
    assert_eq!(one("popc", &["b32"], U32, &[U32], &[0xf0f0]), 8);
    // Clamped shifts, logical .b shr, arithmetic .s shr.
    assert_eq!(
        one("shl", &["b32"], U32, &[U32, U32], &[0xdead_beef, 32]),
        0
    );
    assert_eq!(one("shl", &["b32"], U32, &[U32, U32], &[1, 4]), 16);
    assert_eq!(one("shl", &["b16"], U16, &[U16, U32], &[0x8001, 1]), 2);
    assert_eq!(
        one("shr", &["b32"], U32, &[U32, U32], &[0x8000_0000, 4]),
        0x0800_0000
    );
    assert_eq!(
        one("shr", &["s32"], S32, &[S32, U32], &[0x8000_0000, 4]),
        0xf800_0000
    );
    assert_eq!(
        one(
            "shr",
            &["s32"],
            S32,
            &[S32, U32],
            &[u64::from((-3i32) as u32), 64]
        ),
        0xffff_ffff
    );
    assert_eq!(
        one("bfe", &["s32"], S32, &[S32, U32, U32], &[0x80, 7, 1]),
        0xffff_ffff
    );
    assert_eq!(
        kind(run("bfe", &["u32"], &[U32], &[U32; 3], &[1, 256, 0])),
        OpErrorKind::Invalid
    );
    assert_eq!(one("bfi", &["b32"], U32, &[U32; 4], &[0xf, 0, 4, 4]), 0xf0);
    assert_eq!(
        one("bfind", &["shiftamt", "u32"], U32, &[U32], &[0x8000_0000]),
        0
    );
    assert_eq!(
        one("bfind", &["s64"], U32, &[U64], &[u64::MAX]),
        u64::from(u32::MAX)
    );
    assert_eq!(
        one(
            "bmsk",
            &["clamp", "b32"],
            U32,
            &[U32; 2],
            &[31, u64::from(u32::MAX)]
        ),
        0x8000_0000
    );
    assert_eq!(one("bmsk", &["wrap", "b32"], U32, &[U32; 2], &[33, 34]), 6);
    assert_eq!(
        one(
            "shf",
            &["l", "clamp", "b32"],
            U32,
            &[U32; 3],
            &[0x0123_4567, 0x89ab_cdef, 32]
        ),
        0x0123_4567
    );
    assert_eq!(
        one("szext", &["clamp", "s32"], S32, &[S32, U32], &[0b1000, 4]),
        u64::from((-8i32) as u32)
    );
    assert_eq!(one("clmad", &["lo", "u64"], U64, &[U64; 3], &[3, 3, 0]), 5);
    let (pa, pb) = (0x4433_2211, 0x8877_6655);
    assert_eq!(
        one("prmt", &["b32"], U32, &[U32; 3], &[pa, pb, 0x000f]),
        0x1111_11ff
    );
    assert_eq!(
        one("prmt", &["b32", "b4e"], U32, &[U32; 3], &[pa, pb, 0]),
        0x6677_8811
    );
    assert_eq!(
        one(
            "prmt",
            &["type=b32", "mode=rc16"],
            U32,
            &[U32; 3],
            &[pa, pb, 1]
        ),
        0x4433_4433
    );
    assert_eq!(
        one("fns", &["b32"], U32, &[U32, U32, S32], &[0xff, 0, 1]),
        0
    );
    assert_eq!(
        kind(run(
            "fns",
            &["b32"],
            &[U32],
            &[U32, U32, S32],
            &[u64::from(u32::MAX), 32, 1]
        )),
        OpErrorKind::Invalid
    );
}

#[test]
fn compare_select_and_classify() {
    assert_eq!(
        one("setp", &["lt", "f32"], PRED, &[F32; 2], &[f(-0.0), f(1.0)]),
        1
    );
    assert_eq!(
        one(
            "setp",
            &["ne", "f32"],
            PRED,
            &[F32; 2],
            &[f(f32::NAN), f(1.0)]
        ),
        0
    );
    assert_eq!(
        one(
            "setp",
            &["neu", "f32"],
            PRED,
            &[F32; 2],
            &[f(f32::NAN), f(1.0)]
        ),
        1
    );
    assert_eq!(
        one("setp", &["lt", "s32"], PRED, &[S32; 2], &[0xffff_ffff, 0]),
        1
    );
    assert_eq!(
        one("setp", &["lo", "u32"], PRED, &[U32; 2], &[0xffff_ffff, 0]),
        0
    );
    assert_eq!(one("setp", &["eq", "b32"], PRED, &[U32; 2], &[5, 5]), 1);
    assert_eq!(
        kind(run("setp", &["lt", "b32"], &[PRED], &[U32; 2], &[0, 0])),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        kind(run("setp", &["ltu", "s32"], &[PRED], &[S32; 2], &[0, 0])),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        kind(run(
            "setp",
            &["eq", "ftz", "f64"],
            &[PRED],
            &[Ty::F64; 2],
            &[0, 0]
        )),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        one(
            "setp_half",
            &["eq", "ftz", "f16"],
            PRED,
            &[U16; 2],
            &[0x8001, 0]
        ),
        1
    );
    assert_eq!(
        one("setp_half", &["eq", "f16"], PRED, &[U16; 2], &[0x8001, 0]),
        0
    );
    assert_eq!(
        one(
            "setp_bool",
            &["lt", "xor", "f32"],
            PRED,
            &[F32, F32, PRED],
            &[f(0.0), f(1.0), 1]
        ),
        0
    );
    assert_eq!(
        ok("setp_pq", &["lt", "u32"], &[PRED; 2], &[U32; 2], &[1, 2]),
        vec![1, 0]
    );
    assert_eq!(
        ok(
            "setp_half_pq",
            &["lt", "f16x2"],
            &[PRED; 2],
            &[U32; 2],
            &[0x3c00_4000, 0x4000_3c00]
        ),
        vec![0, 1]
    );
    assert_eq!(
        ok(
            "setp_half_bool_pq",
            &["lt", "or", "f16x2"],
            &[PRED; 2],
            &[U32, U32, PRED],
            &[0x3c00_4000, 0x4000_3c00, 1]
        ),
        vec![1, 1]
    );
    assert_eq!(
        ok(
            "setp_bool_pq",
            &["eq", "and", "s32"],
            &[PRED; 2],
            &[S32, S32, PRED],
            &[3, 3, 1]
        ),
        vec![1, 0]
    );
    // set destination encodings.
    assert_eq!(
        one(
            "set_half",
            &["lt", "u32", "f16x2"],
            U32,
            &[U32; 2],
            &[0x3c00_4000, 0x4000_3c00]
        ),
        0xffff_0000
    );
    assert_eq!(
        one(
            "set_half",
            &["lt", "f16x2", "f16x2"],
            U32,
            &[U32; 2],
            &[0x3c00_4000, 0x4000_3c00]
        ),
        0x3c00_0000
    );
    assert_eq!(
        one("set", &["eq", "f32", "s32"], F32, &[S32; 2], &[4, 4]),
        f(1.0)
    );
    assert_eq!(
        one(
            "set",
            &["eq", "dtype=s32", "stype=f32"],
            Ty::S64,
            &[F32; 2],
            &[f(1.0), f(1.0)]
        ),
        u64::MAX
    );
    assert_eq!(
        one(
            "set_bool",
            &["eq", "xor", "u32", "u32"],
            U32,
            &[U32, U32, PRED],
            &[1, 1, 1]
        ),
        0
    );
    assert_eq!(
        one(
            "set_half_bool",
            &["gt", "or", "bf16", "f32"],
            U16,
            &[F32, F32, PRED],
            &[f(0.0), f(1.0), 1]
        ),
        0x3f80
    );
    assert_eq!(
        kind(run(
            "set_half",
            &["eq", "f16", "bf16"],
            &[U16],
            &[U16; 2],
            &[0, 0]
        )),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        one(
            "set_packed",
            &["lt", "s8x4"],
            U32,
            &[U32; 2],
            &[0x807f_ff00, 0x0080_00ff]
        ),
        0xff00_ff00
    );
    assert_eq!(
        one(
            "set_packed",
            &["lo", "u8x4"],
            U32,
            &[U32; 2],
            &[0x807f_ff00, 0x0080_00ff]
        ),
        0x00ff_00ff
    );
    // selp / slct / testp.
    assert_eq!(
        one(
            "selp",
            &["f32"],
            F32,
            &[F32, F32, PRED],
            &[f(1.0), f(2.0), 0]
        ),
        f(2.0)
    );
    assert_eq!(
        one("selp", &["s16"], S32, &[S16, S16, PRED], &[0xffff, 0, 1]),
        0xffff_ffff
    );
    assert_eq!(
        one(
            "slct",
            &["b32", "f32"],
            U32,
            &[U32, U32, F32],
            &[1, 2, f(-0.0)]
        ),
        1
    );
    assert_eq!(
        one(
            "slct",
            &["b32", "f32"],
            U32,
            &[U32, U32, F32],
            &[1, 2, f(f32::NAN)]
        ),
        2
    );
    assert_eq!(
        one(
            "slct",
            &["s32", "s32"],
            S32,
            &[S32, S32, S32],
            &[1, 2, 0xffff_ffff]
        ),
        2
    );
    assert_eq!(
        kind(run(
            "slct",
            &["ftz", "b32", "s32"],
            &[U32],
            &[U32, U32, S32],
            &[0, 0, 0]
        )),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        one("testp", &["normal", "f32"], PRED, &[F32], &[f(-0.0)]),
        1
    );
    assert_eq!(
        one("testp", &["subnormal", "f64"], PRED, &[Ty::F64], &[1]),
        1
    );
}

#[test]
fn moves_and_packs() {
    assert_eq!(
        one("mov", &["b32"], U32, &[U32], &[0x1234_5678]),
        0x1234_5678
    );
    assert_eq!(one("mov", &["f32"], F32, &[F32], &[f(1.5)]), f(1.5));
    assert_eq!(one("mov", &["b64"], U64, &[U64], &[u64::MAX]), u64::MAX);
    assert_eq!(one("mov", &["s16"], S32, &[S16], &[0x8000]), 0xffff_8000);
    assert_eq!(one("mov", &["pred"], PRED, &[PRED], &[1]), 1);
    assert_eq!(
        one(
            "mov_pack_b32x2",
            &["b64"],
            U64,
            &[U32, U32],
            &[0x0123_4567, 0x89ab_cdef]
        ),
        0x89ab_cdef_0123_4567
    );
    assert_eq!(
        ok(
            "mov_unpack_b32x2",
            &["b64"],
            &[U32, U32],
            &[U64],
            &[0x89ab_cdef_0123_4567]
        ),
        vec![0x0123_4567, 0x89ab_cdef]
    );
    assert_eq!(
        one(
            "mov_pack_b16x2",
            &["b32"],
            U32,
            &[U16, U16],
            &[0x1111, 0x2222]
        ),
        0x2222_1111
    );
    assert_eq!(
        ok(
            "mov_unpack_b16x2",
            &["b32"],
            &[U16, U16],
            &[U32],
            &[0x2222_1111]
        ),
        vec![0x1111, 0x2222]
    );
    assert_eq!(
        one(
            "mov_pack_b16x4",
            &["b64"],
            U64,
            &[U16; 4],
            &[0x0123, 0x4567, 0x89ab, 0xcdef]
        ),
        0xcdef_89ab_4567_0123
    );
    assert_eq!(
        ok(
            "mov_unpack_b16x4",
            &["b64"],
            &[U16; 4],
            &[U64],
            &[0xcdef_89ab_4567_0123]
        ),
        vec![0x0123, 0x4567, 0x89ab, 0xcdef]
    );
    let packed = vec![0x89ab_cdef_0123_4567, 0x7654_3210_fedc_ba98];
    assert_eq!(
        ok(
            "mov_pack_b32x4",
            &["b128"],
            &[Ty::B128],
            &[U32; 4],
            &[0x0123_4567, 0x89ab_cdef, 0xfedc_ba98, 0x7654_3210]
        ),
        packed
    );
    assert_eq!(
        ok(
            "mov_unpack_b32x4",
            &["b128"],
            &[U32; 4],
            &[Ty::B128],
            &packed
        ),
        vec![0x0123_4567, 0x89ab_cdef, 0xfedc_ba98, 0x7654_3210]
    );
    assert_eq!(
        ok("mov_pack_b64x2", &["b128"], &[Ty::B128], &[U64; 2], &packed),
        packed
    );
    assert_eq!(
        ok(
            "mov_unpack_b64x2",
            &["b128"],
            &[U64; 2],
            &[Ty::B128],
            &packed
        ),
        packed
    );
}

#[test]
fn sparse_and_createpolicy() {
    // PTX low-bit-first spdecompress example (legacy test).
    let mods = ["elemsize=b8", "idxsize=b4", "spfactor=sp::2:4", "num=x2"];
    assert_eq!(
        ok(
            "spdecompress",
            &mods,
            &[U32; 2],
            &[U32; 2],
            &[0x3121, 0x0605_03f9]
        ),
        vec![0x0003_f900, 0x0600_0500]
    );
    let bad = ["b8", "b4", "sp::2:4", "x2"];
    assert_eq!(
        kind(run(
            "spdecompress",
            &bad,
            &[U32; 2],
            &[U32; 2],
            &[0x3121 | 0xf, 0]
        )),
        OpErrorKind::Invalid
    );
    // spcompress round-trips through spdecompress (legacy test).
    let compressed = ok(
        "spcompress",
        &["b8", "b2", "sp::2:4", "x1"],
        &[U32; 2],
        &[U32, U32, U32],
        &[0x0401_0302, 0x0102_0807, 0],
    );
    let dense = ok(
        "spdecompress",
        &["b8", "b2", "sp::2:4", "x2"],
        &[U32; 2],
        &[U32; 2],
        &compressed,
    );
    assert_eq!(dense, vec![0x0400_0300, 0x0000_0807]);
    // createpolicy: opaque zero policy, validated inputs.
    let fraction = ["fractional", "L2::evict_last", "b64"];
    assert_eq!(
        one("createpolicy_fraction", &fraction, U64, &[F32], &[f(0.5)]),
        0
    );
    assert_eq!(
        kind(run(
            "createpolicy_fraction",
            &fraction,
            &[U64],
            &[F32],
            &[f(0.0)]
        )),
        OpErrorKind::Invalid
    );
    assert_eq!(
        one(
            "createpolicy_fractional",
            &["fractional", "L2::evict_last", "L2::evict_first", "b64"],
            U64,
            &[],
            &[]
        ),
        0
    );
    assert_eq!(
        one("createpolicy_cvt", &["cvt", "L2", "b64"], U64, &[U64], &[7]),
        0
    );
    let range = ["range", "global", "L2::evict_first", "b64"];
    assert_eq!(
        one(
            "createpolicy_range",
            &range,
            U64,
            &[U64, U32, U32],
            &[0x1000, 16, 32]
        ),
        0
    );
    assert_eq!(
        kind(run(
            "createpolicy_range",
            &range,
            &[U64],
            &[U64, U32, U32],
            &[0x1000, 64, 32]
        )),
        OpErrorKind::Invalid
    );
}

#[test]
fn fail_closed_on_unknown_ops_and_modifiers() {
    // Unknown op name: no family owns it.
    assert_eq!(
        kind(run("not_an_op", &[], &[U32], &[U32], &[0])),
        OpErrorKind::Unsupported
    );
    // Unknown token / unknown slot / out-of-domain token / missing required slot.
    assert_eq!(
        kind(run("add", &["rna", "f32"], &[F32], &[F32; 2], &[0, 0])),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        kind(run(
            "add",
            &["rnd=rna", "type=f32"],
            &[F32],
            &[F32; 2],
            &[0, 0]
        )),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        kind(run(
            "add",
            &["bogus=1", "type=f32"],
            &[F32],
            &[F32; 2],
            &[0, 0]
        )),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        kind(run("fma", &["f32"], &[F32], &[F32; 3], &[0, 0, 0])),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        kind(run("add", &["f32", "f32"], &[F32], &[F32; 2], &[0, 0])),
        OpErrorKind::Unsupported
    );
    // Wrong operand arity.
    assert_eq!(
        kind(run("add", &["f32"], &[F32], &[F32], &[0])),
        OpErrorKind::Unsupported
    );
    // cvt / helpers are not ALU-owned.
    assert!(super::NAMES.iter().all(|n| !n.starts_with("tirx.ptx.cvt")));
    assert_eq!(super::NAMES.len(), 110);
}

#[test]
fn hot_forms_are_direct_and_stable() {
    // Direct forms resolve to the same fn item every time, independent of
    // the interning table; boxed forms intern per (key, tys).
    let a = resolve_ptx(&key("mul", &["f32"]), &[F32], &[F32, F32]).unwrap();
    let b = resolve_ptx(&key("mul", &["rnd=rn", "type=f32"]), &[F32], &[F32, F32]).unwrap();
    assert_eq!(a as usize, b as usize);
    let narrow = resolve_ptx(&key("mul", &["f32"]), &[U16], &[F32, F32]).unwrap();
    assert_ne!(a as usize, narrow as usize);
}
