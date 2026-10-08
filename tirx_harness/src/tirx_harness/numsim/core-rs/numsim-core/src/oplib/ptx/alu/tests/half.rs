//! Half-precision and mixed-vector arithmetic.

use super::*;

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
