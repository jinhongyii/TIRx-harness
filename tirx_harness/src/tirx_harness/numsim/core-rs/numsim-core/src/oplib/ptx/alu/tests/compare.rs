//! Compare, select and classify.

use super::*;

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
