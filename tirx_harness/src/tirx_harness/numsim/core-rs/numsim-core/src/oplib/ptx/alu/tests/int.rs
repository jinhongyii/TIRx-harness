//! Integer arithmetic.

use super::*;

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
