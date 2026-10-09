//! Unit tests ported from legacy `reg.rs` (`mod tests`) and `reg_compare.rs`,
//! rewritten against the pure per-lane kernels.

use super::*;
use crate::cvt::{make_float2, PtxFloatRounding};
use crate::scalar::{F32RoundingMode, LowPrecisionFormat};

#[test]
fn lop3_truth_table_uses_ptx_a_b_c_row_order() {
    let a = 0xf0f0_0f0f_u32;
    let b = 0xcccc_3333_u32;
    let c = 0xaaaa_5555_u32;
    assert_eq!(lop3_b32(a, b, c, 0x00), 0);
    assert_eq!(lop3_b32(a, b, c, 0xff), u32::MAX);
    assert_eq!(lop3_b32(a, b, c, 0x80), a & b & c);
    assert_eq!(lop3_b32(a, b, c, 0xfe), a | b | c);
    assert_eq!(lop3_b32(a, b, c, 0x40), a & b & !c);
    assert_eq!(lop3_b32(a, b, c, 0x1a), ((a & b) | c) ^ a);
    assert_eq!(
        lop3_bool_b32(a, b, c, 0x00, BoolOp::Or, true).unwrap(),
        (0, true)
    );
    assert_eq!(
        lop3_bool_b32(a, b, c, 0x00, BoolOp::And, true).unwrap(),
        (0, false)
    );
    assert!(lop3_bool_b32(a, b, c, 0x00, BoolOp::Xor, true).is_err());
}

#[test]
fn packed_comparison_reuses_scalar_signedness_semantics() {
    let lhs = 0x807f_ff00;
    let rhs = 0x0080_00ff;
    assert_eq!(
        set_packed(CmpOp::Lt, PackedIntLane::S8, lhs, rhs).unwrap(),
        0xff00_ff00
    );
    assert_eq!(
        set_packed(CmpOp::Lt, PackedIntLane::U8, lhs, rhs).unwrap(),
        0x00ff_00ff
    );
    assert!(set_packed(CmpOp::Ltu, PackedIntLane::U8, lhs, rhs).is_err());
}

#[test]
fn ptx_94_bit_helpers_cover_control_boundaries() {
    assert_eq!(bit_field_extract(0xffff_ffff, 0, 0, 32, true), 0);
    assert_eq!(
        bit_field_extract(0x8000_0000, 31, 1, 32, true),
        u32::MAX as u64
    );
    assert_eq!(
        bit_field_extract(0x8000_0000, 32, 1, 32, true),
        u32::MAX as u64
    );
    assert_eq!(bit_field_extract(1, 255, 255, 32, false), 0);

    assert_eq!(bit_field_insert(0, 0xdead_beef, 32, 1, 32), 0xdead_beef);
    assert_eq!(bit_field_insert(1, 0, 255, 255, 32), 0);

    assert_eq!(most_significant_non_sign_bit(0, 32, false), u32::MAX);
    assert_eq!(
        most_significant_non_sign_bit(u32::MAX as u64, 32, true),
        u32::MAX
    );
    assert_eq!(most_significant_non_sign_bit(0xffff_fffe, 32, true), 0);
    assert_eq!(most_significant_non_sign_bit(0x8000_0000, 32, false), 31);

    assert_eq!(bmsk_b32(32, 1, true), 0);
    assert_eq!(bmsk_b32(31, u32::MAX, true), 0x8000_0000);
    assert_eq!(bmsk_b32(33, 34, false), 0x0000_0006);
    assert_eq!(bmsk_b32(0, 32, false), 0);

    let low = 0x0123_4567;
    let high = 0x89ab_cdef;
    assert_eq!(shf_b32(low, high, 0, true, true), high);
    assert_eq!(shf_b32(low, high, 0, false, true), low);
    assert_eq!(shf_b32(low, high, 32, true, true), low);
    assert_eq!(shf_b32(low, high, 32, false, true), high);
    assert_eq!(shf_b32(low, high, 32, true, false), high);
    assert_eq!(shf_b32(low, high, 32, false, false), low);

    assert_eq!(szext_u32(u32::MAX, 0, true), 0);
    assert_eq!(szext_u32(u32::MAX, 33, true), u32::MAX);
    assert_eq!(szext_u32(u32::MAX, 33, false), 1);
    assert_eq!(szext_s32(0b1000, 4, true), -8);
    assert_eq!(szext_s32(-1, 0, false), 0);
    assert_eq!(szext_s32(i32::MIN, 32, true), i32::MIN);
    assert_eq!(szext_s32(0b10, 33, false), 0);

    // Active-lane control validation (legacy `check_bit_field_controls`).
    assert!(bfe_u32(1, 256, 0).is_err());
    assert!(bfi_b32(1, 0, 0, 256).is_err());
    assert_eq!(bfe_s32(0x80, 7, 1).unwrap(), -1);
    assert_eq!(bfind_u32(0x8000_0000, true), 0);
    assert_eq!(bfind_s64(-1, true), u32::MAX);
}

#[test]
fn spdecompress_matches_the_ptx_low_bit_first_example() {
    let shape = SpDecompressShape {
        elem_bits: 8,
        index_bits: 4,
        src: 2,
        dst: 4,
        num: 2,
    };
    let output = spdecompress(shape, &[0x3121], &[0x0605_03f9]).unwrap();
    assert_eq!(output, vec![0x0003_f900, 0x0600_0500]);
}

#[test]
fn spdecompress_supports_the_128_register_output_boundary() {
    let shape = SpDecompressShape {
        elem_bits: 8,
        index_bits: 4,
        src: 4,
        dst: 16,
        num: 32,
    };
    let output = spdecompress(shape, &[0; 16], &[0xa5a5_a5a5; 32]).unwrap();
    assert_eq!(output.len(), 128);
    assert_eq!(output[0], 0xa5);
    assert_eq!(output[124], 0xa5);
    assert_eq!(output[127], 0);
}

#[test]
fn spdecompress_rejects_out_of_range_metadata_and_shapes() {
    let shape = SpDecompressShape {
        elem_bits: 8,
        index_bits: 4,
        src: 1,
        dst: 2,
        num: 2,
    };
    spdecompress(shape, &[0], &[0x2211]).unwrap();
    let error = spdecompress(shape, &[2], &[0x2211]).unwrap_err();
    assert!(error
        .to_string()
        .contains("metadata index 2 is outside 0..2"));
    let bad = SpDecompressShape { num: 1, ..shape };
    assert!(spdecompress(bad, &[0], &[0]).is_err());
    assert!(spdecompress(shape, &[0, 0], &[0x2211]).is_err());
}

#[test]
fn spcompress_round_trips_through_spdecompress() {
    // Two 2:4 groups of u8: keep the two largest magnitudes of each.
    let data = [0x0401_0302_u32, 0x0102_0807];
    let shape = SpCompressShape {
        elem_bits: 8,
        index_bits: 2,
        num: 1,
    };
    let output = spcompress(shape, &data, 0).unwrap();
    assert_eq!(output.len(), shape.output_registers());
    let dense = spdecompress(
        SpDecompressShape {
            elem_bits: 8,
            index_bits: 2,
            src: 2,
            dst: 4,
            num: 2,
        },
        &output[..1],
        &output[1..],
    )
    .unwrap();
    assert_eq!(dense, vec![0x0400_0300, 0x0000_0807]);
    assert_eq!(sparse_pair_indices([1.0, f32::NAN, 3.0, 2.0], true), [1, 2]);
}

#[test]
fn static_register_variants_execute_one_lane_wise_instruction() {
    assert_eq!(add_int(i32::MAX, 2), i32::MIN + 1);
    assert_eq!(xor(0xf0_u32, 0x33), 0xc3);
    assert!(!and(true, false));
    assert_eq!(not(0_u16), u16::MAX);
}

#[test]
fn ptx_integer_arithmetic_variants_match_instruction_width_semantics() {
    assert_eq!(div_int(-7_i32, 3).unwrap(), -2);
    assert_eq!(rem_int(-7_i32, -3).unwrap(), -1);
    assert_eq!(sad_int(i32::MIN, i32::MAX, 7), 6);
    assert_eq!(mul_wide_s16(-30_000, 2), -60_000);
    assert_eq!(mad_wide_u16(u16::MAX, u16::MAX, u32::MAX), 4_294_836_224);

    assert_eq!(mul24_s32(0x0080_0000, 2, false), -16_777_216);
    assert_eq!(mul24_s32(0x0080_0000, 2, true), -256);
    assert_eq!(mul24_u32(0xffff_ffff, 2, true), 0x1ff);
    assert_eq!(
        mad24_hi_sat_s32(0x007f_ffff, 0x007f_ffff, i32::MAX),
        i32::MAX
    );

    assert_eq!(dp4a(0x0403_0201, 0x0403_0201, 5, false, false), 35);
    assert_eq!(dp4a(0x0403_0201, 0xfc03_fe01, 5, false, true) as i32, -5);
    assert_eq!(
        dp2a(0xfffd_0002, 0x07fa_0504, 10, true, true, false) as i32,
        3
    );
    assert_eq!(
        dp2a(0xfffd_0002, 0x07fa_0504, 10, true, true, true) as i32,
        -23
    );
    assert_eq!(neg_int(i16::MIN), i16::MIN);

    assert_eq!(mul_hi_int(u32::MAX, 2), 1);
    assert_eq!(mad_hi_sat_s32(i32::MAX, 2, i32::MAX), i32::MAX);
    assert_eq!(add_16x2(0xffff_0001, 0x0001_ffff), 0x0000_0000);
    assert_eq!(
        minmax_16x2(0x8000_0005, 0x0001_0003, true, false, true),
        0x0001_0005
    );
    assert_eq!(
        minmax_16x2(0x8000_0005, 0x0001_0003, true, true, false),
        0x0000_0003
    );
    assert_eq!(
        minmax_16x2(0x8000_0005, 0x0001_0003, false, false, true),
        0x8000_0005
    );
}

#[test]
fn integer_division_fails_closed() {
    assert_eq!(div_int(8_i16, 2).unwrap(), 4);

    let zero = div_int(1_u32, 0).unwrap_err();
    assert!(zero.to_string().contains(": 1 / 0"));

    let overflow = rem_int(i64::MIN, -1).unwrap_err();
    assert!(overflow.to_string().contains(": -9223372036854775808 % -1"));

    let machine_specific = rem_int(-5_i32, 2).unwrap_err();
    assert!(machine_specific
        .to_string()
        .contains("machine-specific negative operand"));
    assert_eq!(rem_int(7_u16, 3).unwrap(), 1);
}

#[test]
fn bit_size_shifts_are_logical_and_clamp_out_of_range_amounts() {
    // Zero fill, not sign fill: the same bit pattern read as i32 would keep
    // its leading ones.
    assert_eq!(shr(0x8000_0000_u32, 4), 0x0800_0000);
    assert_eq!(shr(i32::MIN, 4), -0x0800_0000);
    assert_eq!(shr(0x8000_0000_0000_0000_u64, 4), 0x0800_0000_0000_0000);

    // Clamped, not wrapped: a modulo-N amount would return the operand
    // unchanged for a shift of exactly N.
    assert_eq!(shr(0xdead_beef_u32, 32), 0);
    assert_eq!(shr(u64::MAX, 100), 0);
    assert_eq!(shl(0xdead_beef_u32, 32), 0);
    assert_eq!(shr(-3_i32, 64), -1);
    assert_eq!(shr(3_i64, 64), 0);
}

#[test]
fn nymph_low_precision_forms_have_ptx_shaped_numeric_semantics() {
    // 1 + 2^-11 is exactly halfway between two f16 values; RN-even returns
    // 1. The bf16 case is the analogous 1 + 2^-8 tie.
    let f16 = LowPrecisionFormat::F16;
    let bf16 = LowPrecisionFormat::Bf16;
    assert_eq!(
        add_half(0x3c00, 0x1000, f16, false, HalfClamp::None),
        0x3c00
    );
    assert_eq!(
        add_half(0x3f80, 0x3b80, bf16, false, HalfClamp::None),
        0x3f80
    );
    // 1.5 * 2 + 0.5 = 3.5
    assert_eq!(
        fma_half(0x3e00, 0x4000, 0x3800, f16, false, HalfClamp::None, false),
        0x4300
    );

    assert!(setp_f32(CmpOp::Lt, -0.0, 1.0, false));
    assert!(!setp_bf16(CmpOp::Ne, 0x7fc1, 0x3f80));
}

#[test]
fn ptx_94_mixed_vector_arithmetic_is_bit_exact_and_lane_ordered() {
    let f16 = LowPrecisionFormat::F16;
    let bf16 = LowPrecisionFormat::Bf16;
    let f16_one_negative_one = 0xbc00_3c00_u32;
    let half_ulp = make_float2(2.0_f32.powi(-24), -2.0_f32.powi(-24));

    // The two packed lanes deliberately have opposite signs. A halfway
    // addition distinguishes all four f32 rounding modes.
    let add = |mode| add_mixed_f32x2(f16_one_negative_one, f16, half_ulp, mode);
    assert_eq!(add(F32RoundingMode::Nearest), 0xbf80_0000_3f80_0000);
    assert_eq!(add(F32RoundingMode::Zero), 0xbf80_0000_3f80_0000);
    assert_eq!(add(F32RoundingMode::Down), 0xbf80_0001_3f80_0000);
    assert_eq!(add(F32RoundingMode::Up), 0xbf80_0000_3f80_0001);

    assert_eq!(
        sub_mixed_f32x2(
            0x4000_3f80,
            bf16,
            make_float2(0.5, 1.0),
            F32RoundingMode::Nearest
        ),
        make_float2(0.5, 1.0)
    );

    // 3 * (1 + 2^-23) - 3 is 3 * 2^-23. Rounding the product before the
    // subtraction would instead produce 4 * 2^-23.
    assert_eq!(
        fma_mixed_f32x2(
            0x4200_4200,
            f16,
            make_float2(1.0 + f32::EPSILON, 1.0 + f32::EPSILON),
            make_float2(-3.0, -3.0),
            F32RoundingMode::Nearest,
        ),
        0x34c0_0000_34c0_0000
    );
    assert_eq!(
        fma_mixed_f32x2(
            0x4040_4040,
            bf16,
            make_float2(2.0, 4.0),
            make_float2(1.0, 1.0),
            F32RoundingMode::Zero,
        ),
        make_float2(7.0, 13.0)
    );

    let wide_lhs = make_float2(4.0, 8.0);
    let wide_rhs = make_float2(0.5, 1.0);
    let down = |op, dst| mixed_f32x2_down(wide_lhs, wide_rhs, op, dst);
    assert_eq!(down(MixedDownOp::Add, f16), 0x4880_4480);
    assert_eq!(down(MixedDownOp::Add, bf16), 0x4110_4090);
    assert_eq!(down(MixedDownOp::Sub, f16), 0x4700_4300);
    assert_eq!(down(MixedDownOp::Sub, bf16), 0x40e0_4060);
    assert_eq!(down(MixedDownOp::Mul, f16), 0x4800_4000);
    assert_eq!(down(MixedDownOp::Mul, bf16), 0x4100_4000);

    // `.rz.ftz.f16x2` flushes low-precision subnormal results to signed zero.
    // The bf16 destination spelling has no `.ftz` and preserves both signs of
    // its smallest subnormal.
    assert_eq!(
        mixed_f32x2_down(
            make_float2(2.0_f32.powi(-24), -2.0_f32.powi(-24)),
            0,
            MixedDownOp::Add,
            f16
        ),
        0x8000_0000
    );
    assert_eq!(
        mixed_f32x2_down(0x8001_0000_0001_0000, 0, MixedDownOp::Add, bf16),
        0x8001_0001
    );

    // The unlike-format multiply first converts the right input to the
    // destination format, then performs low-precision RN multiplication.
    assert_eq!(mul_bf16x2_f16x2(0x3dcd_3dcd, 0x3003_3003), 0x3c4d_3c4d);
    assert_eq!(mul_f16x2_bf16x2(0x0e6b_0e6b, 0x4d08_4d08), 0x7c00_7c00);
}

/// The tile lowering's bf16 narrowing marker and the PTX-spelled
/// `cvt.frnd2` family must not disagree on NaN.
#[test]
fn bf16_narrowing_markers_agree_on_every_nan_encoding() {
    for bits in [
        0x7fc0_0000_u32, // quiet NaN
        0xffc0_0000,     // negative quiet NaN
        0x7f80_0001,     // signalling NaN
        0xff80_0001,     // negative signalling NaN
        0x7fff_ffff,     // all-ones payload
    ] {
        let value = f32::from_bits(bits);
        assert_eq!(encode_bf16(value), 0x7fff, "encode_bf16({bits:#010x})");
        assert_eq!(
            encode_bf16(value),
            crate::cvt::ptx_cvt_f32_to_bf16(value, PtxFloatRounding::NearestEven, false, false),
            "bf16 narrowing markers disagree on {bits:#010x}"
        );
    }
}

/// Mixed-precision part of the gated scalar test
/// `packed_scalar_abi_helpers_preserve_component_order` (legacy drove
/// `reg::add::<MixedF32<Bf16>>`).
#[test]
fn mixed_bf16_f32_add_widens_exactly() {
    let half = crate::cvt::f32_to_bf16_bits(0.5);
    let sum = add_mixed_f32(
        half,
        LowPrecisionFormat::Bf16,
        4.0,
        F32RoundingMode::Nearest,
        false,
    );
    assert_eq!(sum, 4.5);
    assert_eq!(
        sub_mixed_f32(
            half,
            LowPrecisionFormat::Bf16,
            4.0,
            F32RoundingMode::Nearest,
            true
        ),
        0.0
    );
}

#[test]
fn ordered_float_ne_is_false_for_nan_but_unordered_ne_is_true() {
    let nan = CompareAtom::Float(f64::NAN);
    let one = CompareAtom::Float(1.0);
    assert!(!compare_atom(CmpOp::Ne, nan, one).unwrap());
    assert!(compare_atom(CmpOp::Neu, nan, one).unwrap());
    assert!(!compare_atom(CmpOp::Num, nan, one).unwrap());
    assert!(compare_atom(CmpOp::Nan, nan, one).unwrap());
    assert!(compare_atom(CmpOp::Lt, CompareAtom::Bits(0), CompareAtom::Bits(1)).is_err());
    assert!(compare_atom(CmpOp::Eq, CompareAtom::Bits(0), CompareAtom::Float(0.0)).is_err());
}

#[test]
fn half_ftz_is_sign_preserving_and_testp_zero_is_normal() {
    assert_eq!(crate::scalar::flush_subnormal_f16_bits(0x0001), 0x0000);
    assert_eq!(crate::scalar::flush_subnormal_f16_bits(0x8001), 0x8000);
    assert!(testp_f32(TestpClass::Normal, -0.0));
    assert!(!testp_f32(TestpClass::Subnormal, -0.0));
    // -subnormal vs +0 compares equal only under `.ftz`.
    assert!(setp_f16(CmpOp::Eq, 0x8001, 0x0000, true));
    assert!(!setp_f16(CmpOp::Eq, 0x8001, 0x0000, false));
}

#[test]
fn set_encodings_and_predicate_pairs_follow_destination_type() {
    let mask = compare_mask_f16x2(CmpOp::Lt, 0x3c00_4000, 0x4000_3c00, false);
    assert_eq!(mask, 0b10);
    assert_eq!(set_encode(SetDst::U32, mask, 2), 0xffff_0000);
    assert_eq!(set_encode(SetDst::F16x2, mask, 2), 0x3c00_0000);
    assert_eq!(set_encode(SetDst::U32, 1, 1), u32::MAX);
    assert_eq!(set_encode(SetDst::F32, 1, 1), 1.0_f32.to_bits());
    assert_eq!(set_encode(SetDst::Bf16, 1, 1), 0x3f80);
    assert_eq!(predicate_pair(1, 1), (true, false));
    assert_eq!(predicate_pair(0b10, 2), (false, true));
    assert_eq!(combine_mask(BoolOp::Xor, 0b01, true), 0b10);
    assert_eq!(slct_f32(1, 2, -0.0, false), 1);
    assert_eq!(slct_f32(1, 2, f32::NAN, false), 2);
    assert_eq!(slct_s32(1, 2, -1), 2);
}

#[test]
fn prmt_modes_select_from_the_b_a_byte_pair() {
    let a = 0x4433_2211;
    let b = 0x8877_6655;
    assert_eq!(prmt_b32(a, b, 0x3210, PrmtMode::Generic), a);
    assert_eq!(prmt_b32(a, b, 0x7654, PrmtMode::Generic), b);
    assert_eq!(prmt_b32(a, b, 0x000f, PrmtMode::Generic), 0x1111_11ff);
    assert_eq!(prmt_b32(a, b, 1, PrmtMode::F4e), 0x5544_3322);
    assert_eq!(prmt_b32(a, b, 0, PrmtMode::B4e), 0x6677_8811);
    assert_eq!(prmt_b32(a, b, 2, PrmtMode::Rc8), 0x3333_3333);
    assert_eq!(prmt_b32(a, b, 1, PrmtMode::Rc16), 0x4433_4433);
    assert!(fns_b32(u32::MAX, 32, 1).is_err());
}

#[test]
fn f32_modifiers_compose_rounding_ftz_and_saturation() {
    let half_ulp_at_one = f32::from_bits(0x3380_0000);
    assert_eq!(
        add_f32(1.0, half_ulp_at_one, F32RoundingMode::Up, false, false),
        f32::from_bits(0x3f80_0001)
    );
    assert_eq!(
        add_f32(1.0, half_ulp_at_one, F32RoundingMode::Up, false, true),
        1.0
    );
    assert_eq!(
        mul_f32(f32::NAN, 1.0, F32RoundingMode::Nearest, false, true),
        0.0
    );
    assert_eq!(
        div_f32(1.0, 4.0, F32DivMode::Rounded(F32RoundingMode::Zero), false),
        0.25
    );
    assert_eq!(
        minmax_f32([-3.0, 2.0, -5.0], false, false, true, false, true),
        5.0
    );
    assert_eq!(
        minmax_f32([-3.0, 2.0], false, false, true, true, true).to_bits(),
        (-3.0_f32).to_bits()
    );
    assert!(max_f32(f32::NAN, 1.0, false, true).is_nan());
    assert_eq!(max_f32(f32::NAN, 1.0, false, false), 1.0);
    assert_eq!(copysign_f32(-1.0, 2.0), -2.0);
    assert!(abs_f64(-f64::NAN).is_sign_negative());
    assert_eq!(
        apply_half_clamp(0xbc00, LowPrecisionFormat::F16, HalfClamp::Relu),
        0
    );
    assert_eq!(
        apply_half_clamp(0x7e00, LowPrecisionFormat::F16, HalfClamp::Relu),
        0x7fff
    );
    assert_eq!(
        apply_half_clamp(0x4000, LowPrecisionFormat::F16, HalfClamp::Sat),
        0x3c00
    );
}

/// PTX `add/sub/mul.f32` NaN results are pinned (first NaN of (a, b),
/// quieted; invalid -> 0xffc00000), so optimized and unoptimized builds give
/// the same bits (the `--release` run of this test checks the former).
#[test]
fn ptx_add_sub_mul_nan_bits_are_build_independent() {
    use crate::scalar::F32RoundingMode as M;
    let f = f32::from_bits;
    let (qa, sb) = (f(0x7fc0_1234), f(0xffa0_0001));
    for mode in [M::Nearest, M::Zero, M::Down, M::Up] {
        for ftz in [false, true] {
            let add = |a, b| {
                add_f32(
                    std::hint::black_box(a),
                    std::hint::black_box(b),
                    mode,
                    ftz,
                    false,
                )
                .to_bits()
            };
            let sub = |a, b| {
                sub_f32(
                    std::hint::black_box(a),
                    std::hint::black_box(b),
                    mode,
                    ftz,
                    false,
                )
                .to_bits()
            };
            let mul = |a, b| {
                mul_f32(
                    std::hint::black_box(a),
                    std::hint::black_box(b),
                    mode,
                    ftz,
                    false,
                )
                .to_bits()
            };
            for op in [&add as &dyn Fn(f32, f32) -> u32, &sub, &mul] {
                assert_eq!(op(qa, sb), 0x7fc0_1234, "{mode:?} ftz {ftz}");
                assert_eq!(op(sb, qa), 0xffe0_0001, "{mode:?} ftz {ftz}");
                assert_eq!(op(1.0, sb), 0xffe0_0001, "{mode:?} ftz {ftz}");
            }
            assert_eq!(add(f32::INFINITY, f32::NEG_INFINITY), 0xffc0_0000);
            assert_eq!(sub(f32::INFINITY, f32::INFINITY), 0xffc0_0000);
            assert_eq!(mul(f32::INFINITY, 0.0), 0xffc0_0000);
        }
    }
}

/// FTZ and `.sat` take the operand format, so bf16 bits are never judged by the
/// f16 encoding. `0x0300` is an f16 subnormal but a bf16 normal (exponent 6).
#[test]
fn low_precision_ftz_and_sat_are_format_aware() {
    use crate::scalar::{low_add_rn, low_fma_rn, low_mul_rn};
    let (f16, bf16) = (LowPrecisionFormat::F16, LowPrecisionFormat::Bf16);
    // Subnormal tests per format.
    assert!(f16.is_subnormal(0x0300) && !bf16.is_subnormal(0x0300));
    assert!(f16.is_subnormal(0x8001) && bf16.is_subnormal(0x8001));
    assert!(!f16.is_subnormal(0x0400) && bf16.is_subnormal(0x0040));
    assert_eq!(bf16.flush_subnormal(0x807f), 0x8000);
    assert_eq!(bf16.flush_subnormal(0x0080), 0x0080);
    assert_eq!(bf16.flush_subnormal(0x7fc1), 0x7fc1); // NaN payload untouched
                                                      // add/sub/mul/fma: a bf16 normal survives FTZ; f16 bits with the same
                                                      // pattern flush to +0.
    assert_eq!(low_add_rn(0x0300, 0x0000, bf16, false, true), 0x0300);
    assert_eq!(low_add_rn(0x0300, 0x0000, f16, false, true), 0x0000);
    assert_eq!(low_mul_rn(0x0300, 0x3f80, bf16, true), 0x0300);
    assert_eq!(low_fma_rn(0x0300, 0x3f80, 0x0000, bf16, true), 0x0300);
    // A bf16 subnormal input does flush.
    assert_eq!(low_add_rn(0x0001, 0x0000, bf16, false, true), 0x0000);
    assert_eq!(low_add_rn(0x0001, 0x0000, bf16, false, false), 0x0001);
    // -0 * 1 + +0 is +0 under RN.
    assert_eq!(low_fma_rn(0x8040, 0x3f80, 0x0000, bf16, true), 0x0000);
    assert_eq!(low_mul_rn(0x8040, 0x3f80, bf16, true), 0x8000);
    // .sat clamps at each format's 1.0.
    assert_eq!(apply_half_clamp(0x4000, bf16, HalfClamp::Sat), 0x3f80);
    assert_eq!(apply_half_clamp(0x3f00, bf16, HalfClamp::Sat), 0x3f00); // 0.5 kept
    assert_eq!(apply_half_clamp(0x4000, f16, HalfClamp::Sat), 0x3c00);
    assert_eq!(apply_half_clamp(0x3800, f16, HalfClamp::Sat), 0x3800);
    assert_eq!(apply_half_clamp(0x7fc1, bf16, HalfClamp::Sat), 0x0000);
    assert_eq!(apply_half_clamp(0xbf80, bf16, HalfClamp::Sat), 0x0000);
    // min/max FTZ: bf16 0x0300 > 0x0200, both normal in bf16.
    assert_eq!(
        crate::cvt::low_minmax(0x0300, 0x0200, bf16, true, false, false, true),
        0x0300
    );
    assert_eq!(
        crate::cvt::low_minmax(0x0300, 0x0200, f16, true, false, false, true),
        0x0000
    );
}
