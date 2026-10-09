//! Logic and bit manipulation.

use super::*;

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
