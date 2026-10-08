//! f32/f64/f32x2 arithmetic and approximate transcendentals.

use super::*;

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
