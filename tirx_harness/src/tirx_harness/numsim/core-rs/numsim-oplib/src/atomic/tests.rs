//! Tests for the atomic family.
//!
//! * `bulk_*`: legacy `engine-rs/src/memory.rs` tests
//!   `deferred_reduction_variants_match_ptx_scalar_semantics` and
//!   `deferred_add_f32_reduction_flushes_subnormal_inputs_and_outputs`.
//! * `atom_and_red_*`: the numeric sequence of legacy
//!   `runtime/instructions/mem.rs` `atom_and_red_share_the_atomic_core_...`.
//! * `float_atomics_*`: the operand sets of
//!   `tests/numsim/microtests/cases/float_atomics.py` (GPU-paired there, with
//!   no recorded values) with their expected results written out.

use super::*;
use crate::cvt::formats::{f32_to_bf16_bits, f32_to_fp16_bits};

#[test]
fn bulk_reduction_variants_match_ptx_scalar_semantics() {
    let f16 = |value: f32| f32_to_fp16_bits(value).to_le_bytes().to_vec();
    let bf16 = |value: f32| f32_to_bf16_bits(value).to_le_bytes().to_vec();
    let cases = vec![
        (
            BulkReduction::AddU32,
            7_u32.to_le_bytes().to_vec(),
            9_u32.to_le_bytes().to_vec(),
            16_u32.to_le_bytes().to_vec(),
        ),
        (
            BulkReduction::AddI32,
            (-7_i32).to_le_bytes().to_vec(),
            3_i32.to_le_bytes().to_vec(),
            (-4_i32).to_le_bytes().to_vec(),
        ),
        (
            BulkReduction::AddU64,
            (u64::MAX - 2).to_le_bytes().to_vec(),
            5_u64.to_le_bytes().to_vec(),
            2_u64.to_le_bytes().to_vec(),
        ),
        (
            BulkReduction::AddF32Ftz,
            1.5_f32.to_le_bytes().to_vec(),
            2.25_f32.to_le_bytes().to_vec(),
            3.75_f32.to_le_bytes().to_vec(),
        ),
        (BulkReduction::AddF16, f16(1.5), f16(2.0), f16(3.5)),
        (BulkReduction::AddBf16, bf16(1.5), bf16(2.0), bf16(3.5)),
        (
            BulkReduction::MinU32,
            7_u32.to_le_bytes().to_vec(),
            9_u32.to_le_bytes().to_vec(),
            7_u32.to_le_bytes().to_vec(),
        ),
        (
            BulkReduction::MinI32,
            (-7_i32).to_le_bytes().to_vec(),
            3_i32.to_le_bytes().to_vec(),
            (-7_i32).to_le_bytes().to_vec(),
        ),
        (
            BulkReduction::MinU64,
            11_u64.to_le_bytes().to_vec(),
            5_u64.to_le_bytes().to_vec(),
            5_u64.to_le_bytes().to_vec(),
        ),
        (
            BulkReduction::MinI64,
            (-11_i64).to_le_bytes().to_vec(),
            (-5_i64).to_le_bytes().to_vec(),
            (-11_i64).to_le_bytes().to_vec(),
        ),
        (BulkReduction::MinF16, f16(1.5), f16(-2.0), f16(-2.0)),
        (BulkReduction::MinBf16, bf16(1.5), bf16(-2.0), bf16(-2.0)),
        (
            BulkReduction::MaxU32,
            7_u32.to_le_bytes().to_vec(),
            9_u32.to_le_bytes().to_vec(),
            9_u32.to_le_bytes().to_vec(),
        ),
        (
            BulkReduction::MaxI32,
            (-7_i32).to_le_bytes().to_vec(),
            3_i32.to_le_bytes().to_vec(),
            3_i32.to_le_bytes().to_vec(),
        ),
        (
            BulkReduction::MaxU64,
            11_u64.to_le_bytes().to_vec(),
            5_u64.to_le_bytes().to_vec(),
            11_u64.to_le_bytes().to_vec(),
        ),
        (
            BulkReduction::MaxI64,
            (-11_i64).to_le_bytes().to_vec(),
            (-5_i64).to_le_bytes().to_vec(),
            (-5_i64).to_le_bytes().to_vec(),
        ),
        (BulkReduction::MaxF16, f16(1.5), f16(-2.0), f16(1.5)),
        (BulkReduction::MaxBf16, bf16(1.5), bf16(-2.0), bf16(1.5)),
        (
            BulkReduction::IncU32,
            2_u32.to_le_bytes().to_vec(),
            4_u32.to_le_bytes().to_vec(),
            3_u32.to_le_bytes().to_vec(),
        ),
        (
            BulkReduction::DecU32,
            0_u32.to_le_bytes().to_vec(),
            4_u32.to_le_bytes().to_vec(),
            4_u32.to_le_bytes().to_vec(),
        ),
        (
            BulkReduction::AndB32,
            0xf0f0_00ff_u32.to_le_bytes().to_vec(),
            0x0ff0_f00f_u32.to_le_bytes().to_vec(),
            0x00f0_000f_u32.to_le_bytes().to_vec(),
        ),
        (
            BulkReduction::AndB64,
            0xf0f0_00ff_f0f0_00ff_u64.to_le_bytes().to_vec(),
            0x0ff0_f00f_0ff0_f00f_u64.to_le_bytes().to_vec(),
            0x00f0_000f_00f0_000f_u64.to_le_bytes().to_vec(),
        ),
        (
            BulkReduction::OrB32,
            0xf0f0_00ff_u32.to_le_bytes().to_vec(),
            0x0ff0_f00f_u32.to_le_bytes().to_vec(),
            0xfff0_f0ff_u32.to_le_bytes().to_vec(),
        ),
        (
            BulkReduction::OrB64,
            0xf0f0_00ff_f0f0_00ff_u64.to_le_bytes().to_vec(),
            0x0ff0_f00f_0ff0_f00f_u64.to_le_bytes().to_vec(),
            0xfff0_f0ff_fff0_f0ff_u64.to_le_bytes().to_vec(),
        ),
        (
            BulkReduction::XorB32,
            0xf0f0_00ff_u32.to_le_bytes().to_vec(),
            0x0ff0_f00f_u32.to_le_bytes().to_vec(),
            0xff00_f0f0_u32.to_le_bytes().to_vec(),
        ),
        (
            BulkReduction::XorB64,
            0xf0f0_00ff_f0f0_00ff_u64.to_le_bytes().to_vec(),
            0x0ff0_f00f_0ff0_f00f_u64.to_le_bytes().to_vec(),
            0xff00_f0f0_ff00_f0f0_u64.to_le_bytes().to_vec(),
        ),
    ];

    for (operation, current, source, expected) in cases {
        assert_eq!(
            operation.apply(&current, &source),
            expected,
            "reduction {operation:?}"
        );
    }
}

#[test]
fn bulk_add_f32_reduction_flushes_subnormal_inputs_and_outputs() {
    let smallest_normal = f32::MIN_POSITIVE;
    let smallest_subnormal = f32::from_bits(1);
    assert_eq!(
        BulkReduction::AddF32Ftz
            .apply(
                &smallest_normal.to_le_bytes(),
                &(-smallest_subnormal).to_le_bytes()
            )
            .as_slice(),
        smallest_normal.to_le_bytes()
    );
    assert_eq!(
        BulkReduction::AddF32Ftz
            .apply(
                &smallest_subnormal.to_le_bytes(),
                &smallest_subnormal.to_le_bytes()
            )
            .as_slice(),
        0.0_f32.to_le_bytes()
    );
}

#[test]
fn atom_and_red_share_the_atomic_core() {
    let stored = 7_u32;
    let after_add = atomic_u32(AtomicOp::Add, stored, 5).unwrap();
    assert_eq!(after_add, 12);
    let after_xor = atomic_u32(AtomicOp::BitXor, after_add, 3).unwrap();
    assert_eq!(after_xor, 15);
    assert_eq!(atomic_cas_u64(u64::from(after_xor), 15, 99), 99);
    assert_eq!(atomic_cas_u64(16, 15, 99), 16);
    assert_eq!(atomic_cas_bytes(&[1, 2], &[1, 2], &[3, 4]), vec![3, 4]);
}

#[test]
fn integer_atomics_follow_ptx_inc_dec_and_reject_invalid_types() {
    assert_eq!(atomic_u32(AtomicOp::Increment, 4, 4).unwrap(), 0);
    assert_eq!(atomic_u32(AtomicOp::Increment, 2, 4).unwrap(), 3);
    assert_eq!(atomic_u32(AtomicOp::Decrement, 0, 4).unwrap(), 4);
    assert_eq!(atomic_u32(AtomicOp::Decrement, 9, 4).unwrap(), 4);
    assert_eq!(atomic_u32(AtomicOp::Decrement, 3, 4).unwrap(), 2);
    assert_eq!(atomic_i32(AtomicOp::Add, i32::MAX, 1).unwrap(), i32::MIN);
    assert_eq!(atomic_u64(AtomicOp::Add, u64::MAX, 2).unwrap(), 1);
    assert!(atomic_i64(AtomicOp::Add, 1, 1).is_err());
    assert!(atomic_u64(AtomicOp::Increment, 1, 1).is_err());
    assert!(atomic_f64(AtomicOp::Minimum, 1.0, 1.0).is_err());
    assert_eq!(
        atomic_u64x2(AtomicOp::Exchange, [1, 2], [3, 4]).unwrap(),
        [3, 4]
    );
}

#[test]
fn float_atomics_scalar_half_global_and_shared() {
    // cuda_f16 / cuda_bf16: 0 + smallest subnormal keeps the subnormal.
    assert_eq!(atomic_add_f16(0x0000, 0x0001), 0x0001);
    assert_eq!(atomic_add_bf16(0x0000, 0x0001), 0x0001);
    assert_eq!(
        atomic_add_f16(f32_to_fp16_bits(1.5), f32_to_fp16_bits(2.0)),
        f32_to_fp16_bits(3.5)
    );
    assert_eq!(
        atomic_add_bf16(f32_to_bf16_bits(1.5), f32_to_bf16_bits(2.0)),
        f32_to_bf16_bits(3.5)
    );
}

#[test]
fn float_atomics_f32_flushes_only_in_global_memory() {
    let tiny = f32::from_bits(1);
    let global = atomic_f32(AtomicOp::Add, 0.0, tiny, AtomicSpace::Global).unwrap();
    let shared = atomic_f32(AtomicOp::Add, 0.0, tiny, AtomicSpace::Shared).unwrap();
    assert_eq!(global.to_bits(), 0);
    assert_eq!(shared.to_bits(), 1);
    let noftz = atomic_f32(AtomicOp::AddNoFtz, 0.0, tiny, AtomicSpace::Global).unwrap();
    assert_eq!(noftz.to_bits(), 1);
    assert!(atomic_f32(AtomicOp::Maximum, 0.0, 1.0, AtomicSpace::Global).is_err());
}

#[test]
fn float_atomics_f64_rounds_ties_to_even() {
    let one = f64::from_bits(0x3FF0_0000_0000_0000);
    let half_ulp = f64::from_bits(0x3CA0_0000_0000_0000);
    assert_eq!(
        atomic_f64(AtomicOp::Add, one, half_ulp).unwrap().to_bits(),
        one.to_bits()
    );
}

#[test]
fn float_atomics_packed_half_pairs_add_per_component() {
    assert_eq!(atomic_add_f16x2(0, 0x8001_0001), 0x8001_0001);
    assert_eq!(atomic_add_bf16x2(0, 0x8001_0001), 0x8001_0001);
    assert_eq!(
        atomic_half_vector(AtomicOp::Add, [0, 0], [0x0001, 0x8001], false).unwrap(),
        [0x0001, 0x8001]
    );
    assert_eq!(
        atomic_half_vector(
            AtomicOp::Maximum,
            [0x3c00, 0xbc00, 0, 0],
            [0x4000, 0xc000, 0, 0x8000],
            false
        )
        .unwrap(),
        [0x4000, 0xbc00, 0, 0]
    );
    assert_eq!(
        atomic_half(AtomicOp::Minimum, 0x3f80, 0xbf80, true).unwrap(),
        0xbf80
    );
    assert!(atomic_half(AtomicOp::Exchange, 0, 0, true).is_err());
}

#[test]
fn float_atomics_f32_vectors_flush_per_component() {
    // cuda_f32x2_global: [2^-149 + 0, 1 + 2^-24] -> [0 (FTZ), 1 (tie to even)].
    let old = 0x3F80_0000_0000_0001_u64;
    let operand = 0x3380_0000_0000_0000_u64;
    assert_eq!(atomic_add_f32x2(old, operand, false), 0x3F80_0000_0000_0000);
    assert_eq!(atomic_add_f32x2(old, operand, true), 0x3F80_0000_0000_0001);
    // cuda_f32x4_global.
    let bits = |values: [u32; 4]| values.map(f32::from_bits);
    let result = atomic_add_f32x4(
        bits([0x0000_0001, 0x8000_0001, 0x3F80_0000, 0xBF80_0000]),
        bits([0x0000_0000, 0x8000_0000, 0x3380_0000, 0xB380_0000]),
        false,
    );
    assert_eq!(
        result.map(f32::to_bits),
        [0x0000_0000, 0x8000_0000, 0x3F80_0000, 0xBF80_0000]
    );
}
