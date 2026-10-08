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
