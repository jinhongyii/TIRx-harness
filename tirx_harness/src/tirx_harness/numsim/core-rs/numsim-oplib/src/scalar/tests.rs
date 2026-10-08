//! Legacy `scalar.rs` unit tests, moved verbatim.
#![allow(unused_imports)]
use crate::cvt::formats::*;
use crate::cvt::*;
use crate::scalar::*;
use crate::types::OpError;
use std::cmp::Ordering;

#[test]
fn floor_division_matches_python_sign_rules() {
    assert_eq!(floor_div_i64(7, 3).unwrap(), 2);
    assert_eq!(floor_div_i64(-7, 3).unwrap(), -3);
    assert_eq!(floor_div_i64(7, -3).unwrap(), -3);
    assert_eq!(floor_mod_i64(-7, 3).unwrap(), 2);
    assert_eq!(floor_mod_i64(7, -3).unwrap(), -2);
}

#[test]
fn fns_walks_set_bits_in_both_directions() {
    let mask = 0b10110_u32;
    assert_eq!(ptx_fns_b32(mask, 1, 1), 1);
    assert_eq!(ptx_fns_b32(mask, 1, 2), 2);
    assert_eq!(ptx_fns_b32(mask, 4, -2), 2);
    assert_eq!(ptx_fns_b32(mask, 0, -1), u32::MAX);
}

#[test]
fn packed_f32_operations_preserve_lane_order() {
    let lhs = 2.0_f32.to_bits() as u64 | ((3.0_f32.to_bits() as u64) << 32);
    let rhs = 4.0_f32.to_bits() as u64 | ((5.0_f32.to_bits() as u64) << 32);
    let addend = 1.0_f32.to_bits() as u64 | ((2.0_f32.to_bits() as u64) << 32);
    let product = mul_f32x2(lhs, rhs, F32RoundingMode::Nearest, true);
    assert_eq!(f32::from_bits(product as u32), 8.0);
    assert_eq!(f32::from_bits((product >> 32) as u32), 15.0);
    let sum = add_f32x2(lhs, rhs, F32RoundingMode::Nearest, false);
    assert_eq!(f32::from_bits(sum as u32), 6.0);
    assert_eq!(f32::from_bits((sum >> 32) as u32), 8.0);
    let fused = fma_f32x2(lhs, rhs, addend, F32RoundingMode::Nearest, false);
    assert_eq!(f32::from_bits(fused as u32), 9.0);
    assert_eq!(f32::from_bits((fused >> 32) as u32), 17.0);
}

#[test]
fn ptx_move_vector_layout_is_low_lane_first_and_bit_exact() {
    let b16 = [0x0123_u16, 0x4567, 0x89ab, 0xcdef];
    assert_eq!(ptx_mov_pack_b16x4(b16), 0xcdef_89ab_4567_0123);
    assert_eq!(ptx_mov_unpack_b16x4(0xcdef_89ab_4567_0123), b16);

    let b32 = [0x0123_4567_u32, 0x89ab_cdef, 0xfedc_ba98, 0x7654_3210];
    let packed = [0x89ab_cdef_0123_4567, 0x7654_3210_fedc_ba98];
    assert_eq!(ptx_mov_pack_b32x4(b32), packed);
    assert_eq!(ptx_mov_unpack_b32x4(packed), b32);
    assert_eq!(ptx_mov_unpack_b64x2(packed), packed);
}

#[test]
fn ptx_cvt_pack_saturates_fields_places_b_low_and_preserves_c_high_bits() {
    assert_eq!(ptx_cvt_pack::<16, false>(70_000, -3, 0), 0xffff_0000);
    assert_eq!(ptx_cvt_pack::<16, true>(40_000, -40_000, 0), 0x7fff_8000);
    assert_eq!(ptx_cvt_pack::<8, false>(300, -1, 0x89ab_cdef), 0xcdef_ff00);
    assert_eq!(ptx_cvt_pack::<8, true>(200, -200, 0x0123_4567), 0x4567_7f80);
    assert_eq!(ptx_cvt_pack::<4, false>(17, -1, 0xdead_beef), 0xadbe_eff0);
    assert_eq!(ptx_cvt_pack::<4, true>(8, -9, 0x1234_5678), 0x3456_7878);
    assert_eq!(ptx_cvt_pack::<2, false>(4, -1, 0xfedc_ba98), 0xedcb_a98c);
    assert_eq!(ptx_cvt_pack::<2, true>(2, -3, 0x0123_4567), 0x123_45676);
}

#[test]
fn low_precision_rounding_uses_the_exact_result_not_an_f32_intermediate() {
    // Both values are just above an exact target-format midpoint.  An
    // intermediate f32 loses the final bit and would tie-to-even downward.
    let f16_exact = [(1_u64 << 31) | (1_u64 << 20) | 1];
    assert_eq!(
        encode_exact_low(false, &f16_exact, -30, LowPrecisionFormat::F16, false),
        0x4001
    );
    assert_eq!(f32_to_fp16_bits(2.0 + 2.0_f32.powi(-10)), 0x4000);

    let bf16_exact = [(1_u64 << 60) | (1_u64 << 52) | 1];
    assert_eq!(
        encode_exact_low(false, &bf16_exact, -60, LowPrecisionFormat::Bf16, false),
        0x3f81
    );
    assert_eq!(f32_to_bf16_bits(1.0 + 2.0_f32.powi(-8)), 0x3f80);
}

#[test]
fn ptx_ftz_helpers_flush_inputs_and_outputs_with_sign() {
    let positive_subnormal = f32::from_bits(0x0000_0001);
    let negative_subnormal = f32::from_bits(0x8000_0001);
    let largest_normal = f32::MAX;

    assert_eq!(
        add_f32_ftz(
            positive_subnormal,
            positive_subnormal,
            F32RoundingMode::Nearest
        )
        .to_bits(),
        0.0_f32.to_bits()
    );
    assert_eq!(
        add_f32_ftz(
            negative_subnormal,
            negative_subnormal,
            F32RoundingMode::Nearest
        )
        .to_bits(),
        (-0.0_f32).to_bits()
    );
    assert_eq!(
        ptx_exp2_approx_ftz_f32(negative_subnormal).to_bits(),
        1.0_f32.to_bits()
    );
    assert_eq!(ptx_exp2_approx_ftz_f32(-149.0).to_bits(), 0.0_f32.to_bits());
    assert_eq!(ptx_exp2_approx_f32(-149.0).to_bits(), 1);
    assert_eq!(
        ptx_rcp_approx_ftz_f32(positive_subnormal).to_bits(),
        f32::INFINITY.to_bits()
    );
    assert_eq!(
        ptx_rcp_approx_ftz_f32(negative_subnormal).to_bits(),
        f32::NEG_INFINITY.to_bits()
    );
    assert_eq!(
        ptx_rsqrt_approx_ftz_f32(positive_subnormal).to_bits(),
        f32::INFINITY.to_bits()
    );
    assert_eq!(
        ptx_rsqrt_approx_ftz_f32(negative_subnormal).to_bits(),
        f32::NEG_INFINITY.to_bits()
    );
    assert_eq!(
        ptx_rcp_approx_ftz_f32(largest_normal).to_bits(),
        0.0_f32.to_bits()
    );
}

#[test]
fn ptx_f16x2_exp2_preserves_component_order_and_subnormal_results() {
    assert_eq!(ptx_exp2_approx_f16x2(0x3c00_0000), 0x4000_3c00);
    assert_eq!(ptx_exp2_approx_f16x2(0xce00_cb00), 0x0001_0400);
    assert_eq!(ptx_exp2_approx_f16x2(0x0001_fc00), 0x3c00_0000);
}

#[test]
fn packed_f32_arithmetic_applies_directed_rounding_and_ftz() {
    let one_up = f32::from_bits(1.0_f32.to_bits() + 1);
    let three_quarter_ulp = 3.0_f32 * 2.0_f32.powi(-25);
    let tiny_increment = 2.0_f32.powi(-100);
    assert_eq!(
        fma_f32(1.0, 1.0, three_quarter_ulp, F32RoundingMode::Nearest),
        one_up
    );
    assert_eq!(
        fma_f32(1.0, 1.0, three_quarter_ulp, F32RoundingMode::Zero),
        1.0
    );
    assert_eq!(
        fma_f32(1.0, 1.0, tiny_increment, F32RoundingMode::Up),
        one_up
    );

    let smallest_normal = f32::from_bits(0x0080_0000);
    let smallest_subnormal = f32::from_bits(1);
    assert_eq!(
        add_f32_ftz(smallest_normal, smallest_subnormal, F32RoundingMode::Zero).to_bits(),
        smallest_normal.to_bits()
    );
    assert_eq!(
        mul_f32_ftz(smallest_normal, 0.5, F32RoundingMode::Zero).to_bits(),
        0.0_f32.to_bits()
    );
    assert_eq!(
        fma_f32_ftz(
            smallest_subnormal,
            2.0_f32.powi(126),
            0.0,
            F32RoundingMode::Zero
        )
        .to_bits(),
        0.0_f32.to_bits()
    );
}

#[test]
fn cuda_f32_oracles_canonicalize_nan_and_signed_zero() {
    let nan_a = f32::from_bits(0x7fc0_0001);
    let nan_b = f32::from_bits(0xffc0_0002);
    let canonical_nan = 0x7fff_ffff;

    assert_eq!(cuda_f32_add(nan_a, 1.0).to_bits(), canonical_nan);
    assert_eq!(cuda_f32_max(nan_a, nan_b).to_bits(), canonical_nan);
    assert_eq!(cuda_f32_min(nan_a, nan_b).to_bits(), canonical_nan);
    assert_eq!(cuda_f32_max(nan_a, 3.0), 3.0);
    assert_eq!(cuda_f32_max(3.0, nan_a), 3.0);
    assert_eq!(cuda_f32_min(nan_a, -3.0), -3.0);
    assert_eq!(cuda_f32_min(-3.0, nan_a), -3.0);
    assert_eq!(cuda_f32_max(-0.0, 0.0).to_bits(), 0.0_f32.to_bits());
    assert_eq!(cuda_f32_max(0.0, -0.0).to_bits(), 0.0_f32.to_bits());
    assert_eq!(cuda_f32_min(-0.0, 0.0).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(cuda_f32_min(0.0, -0.0).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(cuda_f32_max(-0.0, -0.0).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(cuda_f32_min(0.0, 0.0).to_bits(), 0.0_f32.to_bits());
}

#[test]
fn cuda_f64_minmax_match_nan_and_signed_zero_rules() {
    let nan_a = f64::from_bits(0x7ff8_0000_0000_0001);
    let nan_b = f64::from_bits(0xfff8_0000_0000_0002);

    assert_eq!(cuda_f64_max(nan_a, nan_b).to_bits(), nan_b.to_bits());
    assert_eq!(cuda_f64_min(nan_a, nan_b).to_bits(), nan_b.to_bits());
    assert_eq!(cuda_f64_max(nan_a, 3.0), 3.0);
    assert_eq!(cuda_f64_max(3.0, nan_a), 3.0);
    assert_eq!(cuda_f64_min(nan_a, -3.0), -3.0);
    assert_eq!(cuda_f64_min(-3.0, nan_a), -3.0);
    assert_eq!(cuda_f64_max(-0.0, 0.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(cuda_f64_max(0.0, -0.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(cuda_f64_min(-0.0, 0.0).to_bits(), (-0.0_f64).to_bits());
    assert_eq!(cuda_f64_min(0.0, -0.0).to_bits(), (-0.0_f64).to_bits());
    assert_eq!(cuda_f64_max(-0.0, -0.0).to_bits(), (-0.0_f64).to_bits());
    assert_eq!(cuda_f64_min(0.0, 0.0).to_bits(), 0.0_f64.to_bits());
}

#[test]
fn cuda_f64_add_matches_nan_selection_and_invalid_infinity() {
    let nan_a = f64::from_bits(0x7ff0_0000_0000_1234);
    let nan_b = f64::from_bits(0xfff8_0000_0000_5678);
    let quiet_nan_a = f64::from_bits(0x7ff8_0000_0000_1234);

    assert_eq!(cuda_f64_add(nan_a, 1.0).to_bits(), quiet_nan_a.to_bits());
    assert_eq!(cuda_f64_add(1.0, nan_b).to_bits(), nan_b.to_bits());
    assert_eq!(cuda_f64_add(nan_a, nan_b).to_bits(), nan_b.to_bits());
    assert_eq!(
        cuda_f64_add(f64::INFINITY, f64::NEG_INFINITY).to_bits(),
        0xfff8_0000_0000_0000
    );
}

#[test]
fn cuda_packed_ops_canonicalize_nan_components() {
    let lhs = 0x7fe1_2345_ffc5_4321_u64;
    let rhs = 0xffd2_468a_7fa1_3579_u64;
    let canonical_pair = 0x7fff_ffff_7fff_ffff_u64;
    assert_eq!(
        add_f32x2(lhs, rhs, F32RoundingMode::Nearest, true),
        canonical_pair
    );
    assert_eq!(
        mul_f32x2(lhs, rhs, F32RoundingMode::Nearest, true),
        canonical_pair
    );
    assert_eq!(pack_bf16x2(f32::NAN, -f32::NAN), 0x7fff_7fff);
    assert_eq!(hmin2_bf16(0xffc2_7fc1, 0x7fc4_7fe3), 0x7fff_7fff);
    assert_eq!(hmax2_bf16(0xffc2_7fc1, 0x7fc4_7fe3), 0x7fff_7fff);

    let smallest_normal = f32::from_bits(0x0080_0000);
    let negative_half_normal = f32::from_bits(0x8040_0000);
    assert_eq!(
        add_f32x2(
            pack_f32x2(smallest_normal, smallest_normal),
            pack_f32x2(negative_half_normal, negative_half_normal),
            F32RoundingMode::Nearest,
            true,
        ),
        pack_f32x2(smallest_normal, smallest_normal),
    );
    assert_eq!(
        mul_f32x2(
            pack_f32x2(smallest_normal, smallest_normal),
            pack_f32x2(0.5, 0.5),
            F32RoundingMode::Nearest,
            true,
        ),
        pack_f32x2(0.0, 0.0),
    );

    let positive_zero_pair = pack_bf16x2(0.0, 0.0);
    let negative_zero_pair = pack_bf16x2(-0.0, -0.0);
    assert_eq!(
        hmin2_bf16(positive_zero_pair, negative_zero_pair),
        negative_zero_pair
    );
    assert_eq!(
        hmin2_bf16(negative_zero_pair, positive_zero_pair),
        negative_zero_pair
    );
    assert_eq!(
        hmax2_bf16(positive_zero_pair, negative_zero_pair),
        positive_zero_pair
    );
    assert_eq!(
        hmax2_bf16(negative_zero_pair, positive_zero_pair),
        positive_zero_pair
    );
}

#[test]
fn ptx_approximate_helpers_have_stable_canonical_representatives() {
    let inputs = [0.1_f32, -3.75_f32, 17.125_f32, 1.234_567_f32];
    let exp2_bits = [0x3f89_2fdf, 0x3d98_37f0, 0x480b_95c2, 0x4016_994f];
    let reciprocal_bits = [0x4120_0000, 0xbe88_8889, 0x3d6f_2eb7, 0x3f4f_5c32];

    for ((input, expected_exp2), expected_reciprocal) in
        inputs.into_iter().zip(exp2_bits).zip(reciprocal_bits)
    {
        assert_eq!(ptx_exp2_approx_ftz_f32(input).to_bits(), expected_exp2);
        assert_eq!(ptx_rcp_approx_ftz_f32(input).to_bits(), expected_reciprocal);
    }
    assert_eq!(ptx_rsqrt_approx_ftz_f32(0.25).to_bits(), 2.0_f32.to_bits());
    assert_eq!(ptx_rsqrt_approx_ftz_f32(4.0).to_bits(), 0.5_f32.to_bits());
    assert_eq!(
        ptx_tanh_approx_f32(1.0).to_bits(),
        0.761_594_2_f32.to_bits()
    );
    assert_eq!(
        ptx_tanh_approx_f32(-1.0).to_bits(),
        (-0.761_594_2_f32).to_bits()
    );
}

#[test]
fn packed_scalar_abi_helpers_preserve_component_order() {
    let pair = make_float2(1.25, -2.5);
    assert_eq!(float2_x(pair), 1.25);
    assert_eq!(float2_y(pair), -2.5);

    let bf16_pair = pack_bf16x2(1.0, 2.0);
    let unpacked = unpack_bf16x2(bf16_pair);
    assert_eq!(float2_x(unpacked), 1.0);
    assert_eq!(float2_y(unpacked), 2.0);

    let other = pack_bf16x2(3.0, -1.0);
    let minimum = unpack_bf16x2(hmin2_bf16(bf16_pair, other));
    let maximum = unpack_bf16x2(hmax2_bf16(bf16_pair, other));
    assert_eq!((float2_x(minimum), float2_y(minimum)), (1.0, -1.0));
    assert_eq!((float2_x(maximum), float2_y(maximum)), (3.0, 2.0));

    assert_eq!(
        fp8x4_e4m3_from_float4(1.0, 2.0, 3.0, 4.0),
        f32_to_float8_e4m3fn_bits(1.0) as u32
            | ((f32_to_float8_e4m3fn_bits(2.0) as u32) << 8)
            | ((f32_to_float8_e4m3fn_bits(3.0) as u32) << 16)
            | ((f32_to_float8_e4m3fn_bits(4.0) as u32) << 24)
    );
}

#[test]
fn directed_f32_rounding_handles_ties_signs_and_overflow() {
    let half_ulp_at_one = f32::from_bits(0x3380_0000);
    let one_up = f32::from_bits(1.0_f32.to_bits() + 1);
    assert_eq!(add_f32(1.0, half_ulp_at_one, F32RoundingMode::Nearest), 1.0);
    assert_eq!(add_f32(1.0, half_ulp_at_one, F32RoundingMode::Down), 1.0);
    assert_eq!(add_f32(1.0, half_ulp_at_one, F32RoundingMode::Up), one_up);
    assert_eq!(add_f32(1.0, half_ulp_at_one, F32RoundingMode::Zero), 1.0);

    let minus_one_down = f32::from_bits((-1.0_f32).to_bits() + 1);
    assert_eq!(
        add_f32(-1.0, -half_ulp_at_one, F32RoundingMode::Down),
        minus_one_down
    );
    assert_eq!(add_f32(-1.0, -half_ulp_at_one, F32RoundingMode::Up), -1.0);
    assert_eq!(add_f32(-1.0, -half_ulp_at_one, F32RoundingMode::Zero), -1.0);

    assert_eq!(mul_f32(f32::MAX, 2.0, F32RoundingMode::Down), f32::MAX);
    assert_eq!(mul_f32(f32::MAX, 2.0, F32RoundingMode::Up), f32::INFINITY);
    assert_eq!(sub_f32(one_up, half_ulp_at_one, F32RoundingMode::Down), 1.0);
}

#[test]
fn directed_add_sub_detect_increments_below_f64_precision() {
    let two_to_minus_100 = f32::from_bits(0x0d80_0000);
    let one_up = f32::from_bits(1.0_f32.to_bits() + 1);
    let one_down = f32::from_bits(1.0_f32.to_bits() - 1);
    let minus_one_down = f32::from_bits((-1.0_f32).to_bits() + 1);

    assert_eq!(add_f32(1.0, two_to_minus_100, F32RoundingMode::Up), one_up);
    assert_eq!(add_f32(1.0, two_to_minus_100, F32RoundingMode::Down), 1.0);
    assert_eq!(add_f32(1.0, two_to_minus_100, F32RoundingMode::Zero), 1.0);

    assert_eq!(
        add_f32(-1.0, -two_to_minus_100, F32RoundingMode::Down),
        minus_one_down
    );
    assert_eq!(add_f32(-1.0, -two_to_minus_100, F32RoundingMode::Up), -1.0);
    assert_eq!(
        add_f32(-1.0, -two_to_minus_100, F32RoundingMode::Zero),
        -1.0
    );

    assert_eq!(
        sub_f32(1.0, two_to_minus_100, F32RoundingMode::Down),
        one_down
    );
    assert_eq!(sub_f32(1.0, two_to_minus_100, F32RoundingMode::Up), 1.0);
    assert_eq!(
        sub_f32(1.0, two_to_minus_100, F32RoundingMode::Zero),
        one_down
    );
}

#[test]
fn directed_add_sub_preserve_cancellation_and_subnormal_results() {
    let smallest_normal = f32::from_bits(0x0080_0000);
    let largest_subnormal = f32::from_bits(0x007f_ffff);
    let smallest_subnormal = f32::from_bits(1);
    let modes = [
        F32RoundingMode::Nearest,
        F32RoundingMode::Down,
        F32RoundingMode::Up,
        F32RoundingMode::Zero,
    ];

    for mode in modes {
        let cancellation_zero = if mode == F32RoundingMode::Down {
            -0.0_f32
        } else {
            0.0_f32
        };
        assert_eq!(
            sub_f32(smallest_normal, largest_subnormal, mode),
            smallest_subnormal
        );
        assert_eq!(
            sub_f32(largest_subnormal, smallest_normal, mode),
            -smallest_subnormal
        );
        assert_eq!(
            add_f32(1.0, -1.0, mode).to_bits(),
            cancellation_zero.to_bits()
        );
        assert_eq!(
            sub_f32(1.0, 1.0, mode).to_bits(),
            cancellation_zero.to_bits()
        );
        assert_eq!(
            fma_f32(1.0, 1.0, -1.0, mode).to_bits(),
            cancellation_zero.to_bits()
        );
        assert_eq!(add_f32(0.0, 0.0, mode).to_bits(), 0.0_f32.to_bits());
        assert_eq!(add_f32(-0.0, -0.0, mode).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(sub_f32(-0.0, 0.0, mode).to_bits(), (-0.0_f32).to_bits());
    }
}

#[test]
fn directed_add_sub_preserve_special_values_and_overflow() {
    assert_eq!(add_f32(f32::MAX, f32::MAX, F32RoundingMode::Down), f32::MAX);
    assert_eq!(add_f32(f32::MAX, f32::MAX, F32RoundingMode::Zero), f32::MAX);
    assert_eq!(
        add_f32(f32::MAX, f32::MAX, F32RoundingMode::Up),
        f32::INFINITY
    );
    assert_eq!(
        add_f32(-f32::MAX, -f32::MAX, F32RoundingMode::Down),
        f32::NEG_INFINITY
    );
    assert_eq!(
        add_f32(-f32::MAX, -f32::MAX, F32RoundingMode::Up),
        -f32::MAX
    );
    assert_eq!(
        add_f32(-f32::MAX, -f32::MAX, F32RoundingMode::Zero),
        -f32::MAX
    );

    for mode in [
        F32RoundingMode::Nearest,
        F32RoundingMode::Down,
        F32RoundingMode::Up,
        F32RoundingMode::Zero,
    ] {
        assert_eq!(add_f32(f32::INFINITY, 1.0, mode), f32::INFINITY);
        assert_eq!(sub_f32(f32::NEG_INFINITY, 1.0, mode), f32::NEG_INFINITY);
        assert!(add_f32(f32::NAN, 1.0, mode).is_nan());
        assert!(add_f32(f32::INFINITY, f32::NEG_INFINITY, mode).is_nan());
    }
}

#[test]
fn nearest_division_and_fma_use_native_f32_operations() {
    assert_eq!(div_f32_rn(7.0, 2.0), 3.5);
    assert_eq!(fma_f32_rn(2.0, 4.0, 1.0), 9.0);
}

#[test]
fn runtime_scalars_round_trip_little_endian_bytes() {
    assert_eq!(u32::decode_le(&42_u32.encode_le()).unwrap(), 42);
    assert!(bool::decode_le(&true.encode_le()).unwrap());
    let vector = [1.0_f32, -2.0, 3.5, 0.0];
    assert_eq!(F32x4::decode_le(&vector.encode_le()).unwrap(), vector);
}

#[test]
fn packed_float8_pack_places_the_first_operand_in_the_upper_byte() {
    assert_eq!(
        ptx_cvt_pack_narrow_x2::<false>(1.0, 2.0, FLOAT8_E4M3),
        0x3840
    );
    // .relu clamps negatives, negative zero, and -inf, but not NaN.
    assert_eq!(
        ptx_cvt_pack_narrow_x2::<false>(-0.0, f32::NEG_INFINITY, FLOAT8_E4M3),
        0x80fe
    );
    assert_eq!(
        ptx_cvt_pack_narrow_x2::<true>(-0.0, f32::NEG_INFINITY, FLOAT8_E4M3),
        0x0000
    );
    assert_eq!(
        ptx_cvt_pack_narrow_x2::<true>(f32::NAN, -f32::NAN, FLOAT8_E5M2),
        0x7f7f
    );
}

#[test]
fn ptx94_narrow_rounding_and_padded_lane_placement_are_bit_exact() {
    let pack = |high, low, rounding, format| {
        ptx_cvt_pack_narrow_x2_rounded::<false>(high, low, rounding, format)
    };

    // E2M3 uses six payload bits in each byte.  1.0625 is exactly halfway
    // between codes 0x08 and 0x09; RN chooses even and RZ truncates.
    assert_eq!(
        pack(1.0625, -1.1875, PtxFloatRounding::NearestEven, FLOAT6_E2M3,),
        0x082a
    );
    assert_eq!(
        pack(1.0625, -1.1875, PtxFloatRounding::Zero, FLOAT6_E2M3,),
        0x0829
    );
    assert_eq!(
        pack(1.0, 28.0, PtxFloatRounding::NearestEven, FLOAT6_E3M2,),
        0x0c1f
    );

    // UE5M3 has no sign bit.  Its code 0x78 is 1.0 and 0x79 is 1.125.
    assert_eq!(
        pack(1.0625, 1.1875, PtxFloatRounding::NearestEven, FLOAT8_UE5M3,),
        0x787a
    );
    assert_eq!(
        pack(1.0625, 1.1875, PtxFloatRounding::Zero, FLOAT8_UE5M3,),
        0x7879
    );
    assert_eq!(
        pack(
            1.0625,
            1.1875,
            PtxFloatRounding::PositiveInfinity,
            FLOAT8_UE5M3,
        ),
        0x797a
    );
    assert_eq!(
        pack(
            f32::NAN,
            f32::INFINITY,
            PtxFloatRounding::Zero,
            FLOAT8_UE5M3,
        ),
        0xfffe
    );
}

#[test]
fn ptx94_n1_scaling_flushes_source_subnormals_to_positive_zero() {
    let negative_f32_subnormal = f32::from_bits(0x8000_0001);
    assert_eq!(
        ptx_cvt_scaled_n1_f32(negative_f32_subnormal, 127).to_bits(),
        0
    );
    assert_eq!(ptx_cvt_scaled_n1_f16(0x8001, 127).to_bits(), 0);
    assert_eq!(ptx_cvt_scaled_n1_bf16(0x8001, 127).to_bits(), 0);

    // UE8M0 code 128 is 2 and code 126 is 1/2.
    assert_eq!(ptx_cvt_scaled_n1_f32(2.0, 128), 1.0);
    assert_eq!(ptx_cvt_scaled_n1_f16(0x3c00, 126), 2.0);
    assert_eq!(ptx_cvt_scaled_n1_bf16(0x3f80, 126), 2.0);
    assert!(ptx_cvt_scaled_n1_f32(1.0, 0xff).is_nan());
}

/// The `.rs` random-bit split, pinned directly rather than only through
/// the goldens.
///
/// This is the one place the wave's least obvious fact lives, and the
/// checked-in golden vectors separate it from the plausible alternatives
/// on only a handful of values — 13 of the 1536 `.rs` entries, and none at
/// all for `e2m1x4.f32.rs.relu`. The vectors below were measured
/// independently on an NVIDIA B200 through an exact carry-condition
/// read-out, and they discriminate every alternative reading by
/// construction: an eight-bit field, unreversed pairs, and a low-order
/// placement each produce different values here.
#[test]
fn stochastic_rounding_splits_rbits_into_four_sixteen_bit_fields() {
    // Eight-bit destinations take the two contiguous halfwords: (a, b)
    // from rbits[31:16] and (e, f) from rbits[15:0], the first of each
    // pair bit-reversed.
    assert_eq!(
        ptx_cvt_rs_randoms(0x1234_5678, FLOAT8_E4M3),
        [0x2c48, 0x1234, 0x1e6a, 0x5678]
    );
    assert_eq!(
        ptx_cvt_rs_randoms(0x1234_5678, FLOAT8_E5M2),
        [0x2c48, 0x1234, 0x1e6a, 0x5678]
    );
    // `.e2m1x4`, whose result is itself only sixteen bits wide, instead
    // gathers each field from one byte of each halfword: (a, b) from bytes
    // 3 and 1, (e, f) from bytes 2 and 0.
    assert_eq!(
        ptx_cvt_rs_randoms(0x1234_5678, FLOAT4_E2M1),
        [0x6a48, 0x1256, 0x1e2c, 0x3478]
    );
    // All-ones and all-zeros are fixed points of every candidate rule, so
    // they only guard against a field being dropped entirely.
    for format in [FLOAT8_E4M3, FLOAT4_E2M1] {
        assert_eq!(ptx_cvt_rs_randoms(0, format), [0, 0, 0, 0]);
        assert_eq!(
            ptx_cvt_rs_randoms(u32::MAX, format),
            [0xffff, 0xffff, 0xffff, 0xffff]
        );
    }
    // One bit at a time: bit 31 is the most significant bit of `b`'s field
    // for an eight-bit destination and the least significant bit of `a`'s
    // reversed field, and both formats agree on that pair.
    assert_eq!(
        ptx_cvt_rs_randoms(1 << 31, FLOAT8_E4M3),
        [0x0001, 0x8000, 0, 0]
    );
    assert_eq!(
        ptx_cvt_rs_randoms(1 << 31, FLOAT4_E2M1),
        [0x0001, 0x8000, 0, 0]
    );
    // Bit 8 separates the two formats: it belongs to (e, f) for an
    // eight-bit destination and to (a, b) for `.e2m1x4`.
    assert_eq!(
        ptx_cvt_rs_randoms(1 << 8, FLOAT8_E4M3),
        [0, 0, 0x0080, 0x0100]
    );
    assert_eq!(
        ptx_cvt_rs_randoms(1 << 8, FLOAT4_E2M1),
        [0x8000, 0x0001, 0, 0]
    );
}

// The expectations below are B200 measurements (sm_100a, driver 595.58.03,
// CUDA 13.2); the Python microtests carry the full recorded tables.

#[test]
fn ptx_cvt_integer_rounding_follows_the_named_direction() {
    let minus_subnormal = f32::from_bits(0x8000_0001);
    assert_eq!(
        ptx_cvt_integral_f32(2.5, PtxIntegerRounding::NearestEven, false),
        2.0
    );
    assert_eq!(
        ptx_cvt_integral_f32(1.5, PtxIntegerRounding::NearestEven, false),
        2.0
    );
    assert_eq!(
        ptx_cvt_integral_f32(1.5, PtxIntegerRounding::Zero, false),
        1.0
    );
    assert_eq!(
        ptx_cvt_integral_f32(-0.5, PtxIntegerRounding::NegativeInfinity, false),
        -1.0
    );
    assert_eq!(
        ptx_cvt_integral_f32(0.5, PtxIntegerRounding::PositiveInfinity, false),
        1.0
    );
    // `.ftz` is what makes a subnormal round to signed zero instead of -1.
    assert_eq!(
        ptx_cvt_integral_f32(minus_subnormal, PtxIntegerRounding::NegativeInfinity, false),
        -1.0
    );
    assert_eq!(
        ptx_cvt_integral_f32(minus_subnormal, PtxIntegerRounding::NegativeInfinity, true).to_bits(),
        0x8000_0000
    );
    // Same-size float-to-float rounding canonicalizes NaN in f32 but keeps
    // the payload in f64.
    assert_eq!(
        ptx_cvt_integral_f32_to_f32(f32::NAN, PtxIntegerRounding::Zero, false).to_bits(),
        0x7fff_ffff
    );
    assert_eq!(
        ptx_cvt_integral_f64_to_f64(
            f64::from_bits(0x7ff0_0000_0000_0001),
            PtxIntegerRounding::Zero
        )
        .to_bits(),
        0x7ff8_0000_0000_0001
    );
}

// TODO(W4-cvt): restore against the pure float->int cvt kernel.
#[test]
fn ptx_cvt_float_to_integer_nan_depends_on_both_widths() {
    // NaN results belong to the float->int cvt forms, keyed by both widths.
    let rzi = PtxIntegerRounding::Zero;
    assert_eq!(cvt_f32_to_int(f32::NAN, rzi, false, false, IntKind::U8), 0);
    assert_eq!(cvt_f32_to_int(f32::NAN, rzi, false, false, IntKind::U32), 0);
    assert_eq!(
        cvt_f32_to_int(f32::NAN, rzi, false, false, IntKind::U64),
        1 << 63
    );
    assert_eq!(cvt_f64_to_int(f64::NAN, rzi, false, IntKind::U8), 0x80);
    assert_eq!(
        cvt_f64_to_int(f64::NAN, rzi, false, IntKind::U32),
        0x8000_0000
    );
    assert_eq!(
        ptx_cvt_unary("cvt.rzi.u8.f64", f64::NAN.to_bits()).unwrap(),
        0x80
    );
}

#[test]
fn ptx_cvt_integer_to_float_rounds_from_the_exact_magnitude() {
    for rounding in [
        PtxFloatRounding::NearestEven,
        PtxFloatRounding::Zero,
        PtxFloatRounding::NegativeInfinity,
        PtxFloatRounding::PositiveInfinity,
    ] {
        assert_eq!(ptx_cvt_integer_to_f16(0, false, rounding), 0);
        assert_eq!(ptx_cvt_integer_to_f16(1, true, rounding), 0xbc00);
        assert_eq!(ptx_cvt_integer_to_bf16(1, false, rounding), 0x3f80);
    }
    assert_eq!(
        ptx_cvt_integer_to_bf16(
            (1_u64 << 60) + (1_u64 << 52) + 1,
            false,
            PtxFloatRounding::NearestEven
        ),
        0x5d81
    );
    assert_eq!(
        ptx_cvt_integer_to_f16(u64::MAX, true, PtxFloatRounding::PositiveInfinity),
        0xfbff
    );
    assert_eq!(
        ptx_cvt_integer_to_f16(u64::MAX, true, PtxFloatRounding::NegativeInfinity),
        0xfc00
    );
    // 2^24 + 1 needs 25 significand bits, so the directed modes differ.
    assert_eq!(
        ptx_cvt_integer_to_f32(16_777_217, false, PtxFloatRounding::Zero),
        16_777_216.0
    );
    assert_eq!(
        ptx_cvt_integer_to_f32(16_777_217, false, PtxFloatRounding::PositiveInfinity),
        16_777_218.0
    );
    assert_eq!(
        ptx_cvt_integer_to_f32(16_777_217, true, PtxFloatRounding::NegativeInfinity),
        -16_777_218.0
    );
    assert_eq!(
        ptx_cvt_integer_to_f32(u64::MAX, false, PtxFloatRounding::Zero),
        18_446_742_974_197_923_840.0
    );
    assert_eq!(
        ptx_cvt_integer_to_f32(u64::MAX, false, PtxFloatRounding::PositiveInfinity),
        18_446_744_073_709_551_616.0
    );
    // i64::MIN is exactly representable, so every mode agrees.
    for rounding in [
        PtxFloatRounding::Zero,
        PtxFloatRounding::NegativeInfinity,
        PtxFloatRounding::PositiveInfinity,
    ] {
        assert_eq!(
            ptx_cvt_integer_to_f64(1u64 << 63, true, rounding),
            -9_223_372_036_854_775_808.0
        );
    }
    assert_eq!(
        ptx_cvt_integer_to_f64((1u64 << 53) + 1, false, PtxFloatRounding::Zero),
        9_007_199_254_740_992.0
    );
    assert_eq!(
        ptx_cvt_integer_to_f64((1u64 << 53) + 1, false, PtxFloatRounding::PositiveInfinity),
        9_007_199_254_740_994.0
    );
}

#[test]
fn packed_float8_unpack_matches_hardware_special_values() {
    // Upper byte to upper half; every NaN widens to 0x7fff, sign dropped.
    assert_eq!(
        ptx_cvt_unpack_narrow_x2_f16x2::<false>(0x7f38, FLOAT8_E4M3),
        0x7fff_3c00
    );
    assert_eq!(
        ptx_cvt_unpack_narrow_x2_f16x2::<true>(0xff80, FLOAT8_E4M3),
        0x7fff_0000
    );
    // e5m2 infinities widen exactly; .relu still zeroes the negative one.
    assert_eq!(
        ptx_cvt_unpack_narrow_x2_f16x2::<false>(0x7cfc, FLOAT8_E5M2),
        0x7c00_fc00
    );
    assert_eq!(
        ptx_cvt_unpack_narrow_x2_f16x2::<true>(0x7cfc, FLOAT8_E5M2),
        0x7c00_0000
    );
    // .satfinite is observable only where the source has infinities.
    assert_eq!(
        ptx_cvt_unpack_narrow_x2_bf16x2::<false, false>(0x7cfc, FLOAT8_E5M2),
        0x7f80_ff80
    );
    assert_eq!(
        ptx_cvt_unpack_narrow_x2_bf16x2::<false, true>(0x7cfc, FLOAT8_E5M2),
        0x7f7f_ff7f
    );
    assert_eq!(
        ptx_cvt_unpack_narrow_x2_bf16x2::<false, false>(0x7efe, FLOAT8_E4M3),
        ptx_cvt_unpack_narrow_x2_bf16x2::<false, true>(0x7efe, FLOAT8_E4M3),
    );
}

#[test]
fn ptx_cvt_f64_to_f32_saturates_and_flushes_per_modifier() {
    use PtxFloatRounding::{NearestEven, NegativeInfinity, PositiveInfinity, Zero};

    // Full-precision rounding below MIN_NORMAL has half the spacing of
    // F32 gradual underflow. Cover both ties and the immediate neighbors.
    let normal = 0x3810_0000_0000_0000_u64;
    for (distance, nearest, away) in [
        (0, true, true),
        (1, true, true),
        ((1 << 28) - 1, true, true),
        (1 << 28, true, true),
        ((1 << 28) + 1, false, true),
        ((1 << 29) - 1, false, true),
        (1 << 29, false, false),
        ((1 << 29) + 1, false, false),
    ] {
        for negative in [false, true] {
            let sign = if negative { 0x8000_0000 } else { 0 };
            let value = f64::from_bits(normal - distance) * if negative { -1.0 } else { 1.0 };
            for (mode, keep) in [
                (NearestEven, nearest),
                (Zero, distance == 0),
                (
                    NegativeInfinity,
                    if negative { away } else { distance == 0 },
                ),
                (
                    PositiveInfinity,
                    if negative { distance == 0 } else { away },
                ),
            ] {
                assert_eq!(
                    ptx_cvt_f64_to_f32(value, mode, true).to_bits(),
                    sign | if keep { 0x0080_0000 } else { 0 },
                    "distance={distance}, negative={negative}, mode={mode:?}"
                );
            }
        }
    }
    let tiny = f64::from_bits(0x0000_0000_0000_0001);
    assert_eq!(
        ptx_cvt_f64_to_f32(tiny, PtxFloatRounding::PositiveInfinity, false).to_bits(),
        0x0000_0001
    );
    assert_eq!(
        ptx_cvt_f64_to_f32(tiny, PtxFloatRounding::PositiveInfinity, true).to_bits(),
        0x0000_0000
    );
    assert_eq!(
        ptx_cvt_f64_to_f32(f64::MAX, PtxFloatRounding::Zero, false),
        f32::MAX
    );
    assert_eq!(
        ptx_cvt_f64_to_f32(f64::MAX, PtxFloatRounding::PositiveInfinity, false),
        f32::INFINITY
    );
    assert_eq!(
        ptx_cvt_f64_to_f32(f64::NAN, PtxFloatRounding::NearestEven, false).to_bits(),
        0x7fc0_0000
    );
}

#[test]
fn ptx_cvt_narrowing_applies_satfinite_then_relu() {
    // Without `.satfinite` an overflowing `.rn` result is infinity; with it
    // the result is the destination's largest finite value.
    assert_eq!(
        ptx_cvt_f32_to_f16(f32::MAX, PtxFloatRounding::NearestEven, false, false),
        0x7c00
    );
    assert_eq!(
        ptx_cvt_f32_to_f16(f32::MAX, PtxFloatRounding::NearestEven, false, true),
        0x7bff
    );
    // `.rz` truncates, so 65520 lands on MAX_NORM instead of infinity.
    assert_eq!(
        ptx_cvt_f32_to_f16(65_520.0, PtxFloatRounding::NearestEven, false, false),
        0x7c00
    );
    assert_eq!(
        ptx_cvt_f32_to_f16(65_520.0, PtxFloatRounding::Zero, false, false),
        0x7bff
    );
    assert_eq!(
        ptx_cvt_f32_to_f16(
            f32::NEG_INFINITY,
            PtxFloatRounding::NearestEven,
            true,
            false
        ),
        0
    );
    // Every NaN answers the canonical narrow NaN, `.relu` included.
    assert_eq!(
        ptx_cvt_f32_to_f16(f32::NAN, PtxFloatRounding::NearestEven, true, true),
        0x7fff
    );
    assert_eq!(
        ptx_cvt_f32_to_bf16(f32::NAN, PtxFloatRounding::NearestEven, false, false),
        0x7fff
    );
    assert_eq!(
        ptx_cvt_f32_to_bf16(f32::MAX, PtxFloatRounding::Zero, false, false),
        0x7f7f
    );
    assert_eq!(
        ptx_cvt_f32_to_bf16(f32::MAX, PtxFloatRounding::NearestEven, false, false),
        0x7f80
    );
}
#[test]
fn ptx_cvt_pzo_changes_only_negative_zero_results() {
    assert_eq!(ptx_cvt_pzo_u16(0x8000), 0);
    assert_eq!(ptx_cvt_pzo_u16(0), 0);
    assert_eq!(ptx_cvt_pzo_u16(0xbc00), 0xbc00);
    assert_eq!(ptx_cvt_pzo_u16(0xffff), 0xffff);

    assert_eq!(ptx_cvt_pzo_u32(0x8000_0000), 0);
    assert_eq!(ptx_cvt_pzo_u32(0), 0);
    assert_eq!(ptx_cvt_pzo_u32(0xbf80_0000), 0xbf80_0000);
    assert_eq!(ptx_cvt_pzo_u32(0xffff_e000), 0xffff_e000);

    // `.pzo` is post-conversion: a negative value that narrows to -0 is
    // normalized even though its f32 input was not itself zero.
    let tiny_f16 = ptx_cvt_f32_to_f16(-f32::from_bits(1), PtxFloatRounding::Zero, false, false);
    assert_eq!(tiny_f16, 0x8000);
    assert_eq!(ptx_cvt_pzo_u16(tiny_f16), 0);
    let tiny_tf32 = ptx_cvt_f32_to_tf32(-f32::from_bits(1), PtxFloatRounding::Zero, false, false);
    assert_eq!(tiny_tf32, 0x8000_0000);
    assert_eq!(ptx_cvt_pzo_u32(tiny_tf32), 0);

    assert_eq!(ptx_cvt_pzo_narrow_x2(0x8080, FLOAT8_E4M3), 0);
    assert_eq!(ptx_cvt_pzo_narrow_x2(0x807f, FLOAT8_E4M3), 0x007f);
    assert_eq!(ptx_cvt_pzo_narrow_x2(0x2020, FLOAT6_E2M3), 0);
    assert_eq!(ptx_cvt_pzo_narrow_x2(0x0088, FLOAT4_E2M1), 0);
}

#[test]
fn ptx_cvt_tf32_rounds_ties_away_only_for_rna() {
    let tie = f32::from_bits(0x3f80_1000);
    assert_eq!(
        ptx_cvt_f32_to_tf32(tie, PtxFloatRounding::NearestEven, false, false),
        0x3f80_0000
    );
    assert_eq!(
        ptx_cvt_f32_to_tf32(tie, PtxFloatRounding::NearestAway, false, false),
        0x3f80_2000
    );
    assert_eq!(
        ptx_cvt_f32_to_tf32(tie, PtxFloatRounding::Zero, false, false),
        0x3f80_0000
    );
    // `.rn`/`.rz` canonicalize NaN; `.rna` rounds it arithmetically, which
    // turns a payload-in-the-discarded-bits NaN into infinity.
    assert_eq!(
        ptx_cvt_f32_to_tf32(f32::NAN, PtxFloatRounding::Zero, false, true),
        0x7fff_e000
    );
    assert_eq!(
        ptx_cvt_f32_to_tf32(
            f32::from_bits(0x7f80_0001),
            PtxFloatRounding::NearestAway,
            false,
            false
        ),
        0x7f80_0000
    );
    assert_eq!(
        ptx_cvt_f32_to_tf32(
            f32::from_bits(0x7fc0_0000),
            PtxFloatRounding::NearestAway,
            false,
            true
        ),
        0x7fbf_e000
    );
    assert_eq!(
        ptx_cvt_f32_to_tf32(f32::MAX, PtxFloatRounding::NearestEven, false, true),
        0x7f7f_e000
    );
    assert_eq!(
        ptx_cvt_f32_to_tf32(-f32::MAX, PtxFloatRounding::NearestEven, true, false),
        0
    );
}

/// The pinned host-FMA NaN (legacy glibc `fmaf` on x86 FMA hardware): first
/// NaN among (rhs, lhs, addend), quieted; else the x86 default NaN.
#[test]
fn host_fma_nan_is_pinned() {
    let f = f32::from_bits;
    let (qa, sb) = (f(0x7fc0_1234), f(0xffa0_0001));
    assert_eq!(host_fma_f32(qa, sb, 1.0).to_bits(), 0xffe0_0001);
    assert_eq!(host_fma_f32(sb, qa, 1.0).to_bits(), 0x7fc0_1234);
    assert_eq!(host_fma_f32(1.0, 2.0, sb).to_bits(), 0xffe0_0001);
    assert_eq!(host_fma_f32(f32::INFINITY, 0.0, 1.0).to_bits(), 0xffc0_0000);
    assert_eq!(
        host_fma_f32(f32::INFINITY, 1.0, f32::NEG_INFINITY).to_bits(),
        0xffc0_0000
    );
    assert_eq!(host_fma_f32(1.5, 2.0, 0.25), 3.25);
    let d = f64::from_bits;
    assert_eq!(
        host_fma_f64(d(0x7ff8_0000_0000_1234), d(0xfff4_0000_0000_0001), 1.0).to_bits(),
        0xfffc_0000_0000_0001
    );
    assert_eq!(
        host_fma_f64(0.0, f64::INFINITY, 1.0).to_bits(),
        0xfff8_0000_0000_0000
    );
}

/// On x86-64 Linux the pinned rule is what the C library's `fmaf`/`fma`
/// return (legacy called them through `mul_add`).
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn host_fma_nan_matches_the_c_library() {
    extern "C" {
        fn fmaf(a: f32, b: f32, c: f32) -> f32;
        fn fma(a: f64, b: f64, c: f64) -> f64;
    }
    if !std::arch::is_x86_feature_detected!("fma") {
        return; // glibc's software path propagates NaNs differently
    }
    let vals: [u32; 10] = [
        0x7fc0_1234,
        0xffa0_0001,
        0x7f80_0001,
        0xffc0_0000,
        0x3f80_0000,
        0x7f80_0000,
        0xff80_0000,
        0,
        0x8000_0000,
        0x4000_0000,
    ];
    for &x in &vals {
        for &y in &vals {
            for &z in &vals {
                let (a, b, c) = (f32::from_bits(x), f32::from_bits(y), f32::from_bits(z));
                let libc = unsafe {
                    fmaf(
                        std::hint::black_box(a),
                        std::hint::black_box(b),
                        std::hint::black_box(c),
                    )
                };
                assert_eq!(
                    host_fma_f32(a, b, c).to_bits(),
                    libc.to_bits(),
                    "{x:#x} {y:#x} {z:#x}"
                );
                let (a, b, c) = (f64::from(a), f64::from(b), f64::from(c));
                let libc = unsafe {
                    fma(
                        std::hint::black_box(a),
                        std::hint::black_box(b),
                        std::hint::black_box(c),
                    )
                };
                assert_eq!(
                    host_fma_f64(a, b, c).to_bits(),
                    libc.to_bits(),
                    "f64 {x:#x} {y:#x} {z:#x}"
                );
            }
        }
    }
}

fn ulp_distance(a: f32, b: f32) -> u32 {
    let key = |v: f32| {
        let bits = v.to_bits() as i32;
        if bits < 0 {
            i32::MIN - bits
        } else {
            bits
        }
    };
    key(a).abs_diff(key(b))
}

/// `tirx.log1p` special values are pinned; finite values stay within one
/// binary32 ulp of an independent binary64 reference (`ln_1p` in f64,
/// rounded once), and the legacy test's sweep matches it.
#[test]
fn log1p_special_values_and_accuracy() {
    let f = f32::from_bits;
    assert_eq!(log1p_f32(f(0x7fc0_1234)).to_bits(), 0x7fc0_1234);
    assert_eq!(log1p_f32(f(0xffa0_0001)).to_bits(), 0xffe0_0001);
    assert_eq!(log1p_f32(-2.0).to_bits(), 0xffc0_0000);
    assert_eq!(log1p_f32(f32::NEG_INFINITY).to_bits(), 0xffc0_0000);
    assert_eq!(log1p_f32(-1.0), f32::NEG_INFINITY);
    assert_eq!(log1p_f32(-0.0).to_bits(), 0x8000_0000);
    assert_eq!(log1p_f32(0.0).to_bits(), 0);
    assert_eq!(log1p_f32(f32::INFINITY), f32::INFINITY);
    assert_eq!(log1p_f32(f(1)).to_bits(), 1, "tiny x: log1p(x) = x");
    assert_eq!(log1p_f64(-1.0), f64::NEG_INFINITY);
    assert_eq!(log1p_f64(-3.0).to_bits(), 0xfff8_0000_0000_0000);
    // Legacy test_gate_intrinsics sweep, plus a wide log-spaced sweep.
    let mut inputs: Vec<f32> = (0..32)
        .map(|i| -0.875 + i as f32 * (8.875 / 31.0))
        .collect();
    let mut x = 1e-30_f32;
    while x < 1e30 {
        inputs.extend([x, -x.min(0.999_99)]);
        x *= 1.37;
    }
    for x in inputs {
        let reference = (x as f64).ln_1p() as f32;
        assert!(
            ulp_distance(log1p_f32(x), reference) <= 1,
            "log1p({x:e}) = {} vs {reference}",
            log1p_f32(x)
        );
    }
}

/// `tirx.sigmoid`: the legacy formula in binary32, NaN pinned.
#[test]
fn sigmoid_special_values_and_accuracy() {
    let f = f32::from_bits;
    assert_eq!(sigmoid_f32(f(0x7fc0_1234)).to_bits(), 0xffc0_1234);
    assert_eq!(sigmoid_f32(0.0), 0.5);
    assert_eq!(sigmoid_f32(f32::INFINITY), 1.0);
    assert_eq!(sigmoid_f32(f32::NEG_INFINITY), 0.0);
    assert_eq!(sigmoid_f32(-200.0), 0.0);
    for i in 0..200 {
        let x = -20.0 + i as f32 * 0.2;
        let reference = (1.0 / (1.0 + (-(x as f64)).exp())) as f32;
        assert!(ulp_distance(sigmoid_f32(x), reference) <= 2, "sigmoid({x})");
    }
    // Half formats go through f32 with one RNE rounding back.
    assert_eq!(half_unary(0x0000, false, sigmoid_f32), 0x3800); // 0.5 in f16
                                                                // bf16 1.0 = 0x3f80; log1p(1) = 0.693147 -> bf16 0x3f31 (RNE).
    assert_eq!(half_unary(0x3f80, true, log1p_f32), 0x3f31);
}

/// v2-only math builtins: within 1 ulp of a binary64 reference rounded once,
/// pinned NaNs and edge values; `nearbyint` exact (ties to even).
#[test]
fn erf_exp10_log10_nearbyint_definitions() {
    let f = f32::from_bits;
    for g in [
        erf_f32 as fn(f32) -> f32,
        exp10_f32,
        log10_f32,
        nearbyint_f32,
    ] {
        assert_eq!(g(f(0x7fc0_1234)).to_bits(), 0x7fc0_1234);
        assert_eq!(g(f(0xffa0_0001)).to_bits(), 0xffe0_0001);
    }
    assert_eq!(log10_f32(-1.0).to_bits(), 0xffc0_0000);
    assert_eq!(log10_f32(-0.0), f32::NEG_INFINITY);
    assert_eq!(log10_f32(1000.0), 3.0);
    assert_eq!(exp10_f32(2.0), 100.0);
    assert_eq!(erf_f32(0.0).to_bits(), 0);
    assert_eq!(erf_f32(-0.0).to_bits(), 0x8000_0000);
    assert_eq!(erf_f32(f32::INFINITY), 1.0);
    for (x, want) in [
        (0.5_f32, 0.0_f32),
        (1.5, 2.0),
        (2.5, 2.0),
        (-0.5, -0.0),
        (-1.5, -2.0),
        (1e30, 1e30),
    ] {
        assert_eq!(nearbyint_f32(x).to_bits(), want.to_bits(), "nearbyint({x})");
    }
    for i in 0..400 {
        let x = -4.0 + i as f32 * 0.02;
        assert!(
            ulp_distance(erf_f32(x), libm::erf(f64::from(x)) as f32) <= 1,
            "erf({x}) vs the binary64 erf"
        );
        assert!(
            ulp_distance(exp10_f32(x), 10f64.powf(f64::from(x)) as f32) <= 1,
            "exp10({x})"
        );
        let y = 1e-30 * 1.37_f32.powi(i % 200);
        assert!(
            ulp_distance(log10_f32(y), (f64::from(y)).log10() as f32) <= 1,
            "log10({y})"
        );
    }
    assert_eq!(log10_f64(-2.0).to_bits(), 0xfff8_0000_0000_0000);
    assert_eq!(nearbyint_f64(2.5), 2.0);
}
