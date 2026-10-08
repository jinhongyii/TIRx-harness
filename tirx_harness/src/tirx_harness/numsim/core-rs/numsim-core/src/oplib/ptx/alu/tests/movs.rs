//! Moves, packs, sparse (de)compression and createpolicy.

use super::*;

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
