//! Tests ported from legacy `runtime/tcgen_ops.rs` `mod tests` (descriptor decoders).
use super::*;
use crate::tcgen05::narrow::NarrowFormat;
use crate::tcgen05::smem_desc::MatrixDescriptorLayout;

fn mxf4_family_descriptor(ue8m0: bool, sfa_id: u32) -> u32 {
    (1 << 7) | (1 << 10) | ((8 >> 3) << 17) | (u32::from(ue8m0) << 23) | (8 << 24) | (sfa_id << 29)
}

fn mxf8f6f4_descriptor(m: u32, n: u32) -> u32 {
    ((n >> 3) << 17) | (1 << 23) | ((m >> 4) << 24)
}

#[test]
fn mxf8f6f4_descriptor_contract_accepts_cta1_eight_column_granularity() {
    for n in [8, 24] {
        let decoded = decode_mxf8f6f4(
            mxf8f6f4_descriptor(128, n),
            1,
            MatrixDescriptorLayout::Sm100,
        )
        .unwrap();
        assert_eq!((decoded.m, decoded.n), (128, n as usize));
    }
}

#[test]
fn mxf8f6f4_descriptor_accepts_both_cta2_accumulator_layouts() {
    for m in [128, 256] {
        for n in [16, 256] {
            let decoded =
                decode_mxf8f6f4(mxf8f6f4_descriptor(m, n), 2, MatrixDescriptorLayout::Sm100)
                    .unwrap();
            assert_eq!((decoded.m, decoded.n), (m as usize, n as usize));
        }
    }
    for m in [64, 144, 384] {
        assert!(
            decode_mxf8f6f4(mxf8f6f4_descriptor(m, 32), 2, MatrixDescriptorLayout::Sm100,).is_err()
        );
    }
}

#[test]
fn mxf8f6f4_descriptor_contract_reports_invalid_m_as_geometry() {
    let error = match decode_mxf8f6f4(
        mxf8f6f4_descriptor(144, 16),
        1,
        MatrixDescriptorLayout::Sm100,
    ) {
        Ok(_) => panic!("invalid M descriptor decoded successfully"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("requires M=128"));
}

#[test]
fn mxf8f6f4_descriptor_decodes_e4m3_transpose_and_enforces_b_granularity() {
    let descriptor = mxf8f6f4_descriptor(128, 16) | (1 << 15) | (1 << 16);
    let decoded = decode_mxf8f6f4(descriptor, 1, MatrixDescriptorLayout::Sm100).unwrap();
    assert!(decoded.transpose_a);
    assert!(decoded.transpose_b);

    let error = decode_mxf8f6f4(
        mxf8f6f4_descriptor(128, 8) | (1 << 16),
        1,
        MatrixDescriptorLayout::Sm100,
    )
    .unwrap_err();
    assert!(error.to_string().contains("N in 16..=256 by 16"));
}

#[test]
fn mxf8f6f4_descriptor_rejects_transposed_packed_e2m1() {
    let descriptor = mxf8f6f4_descriptor(128, 16) | (5 << 7) | (1 << 15);
    let error = decode_mxf8f6f4(descriptor, 1, MatrixDescriptorLayout::Sm100).unwrap_err();
    assert!(error
        .to_string()
        .contains("transpose A requires an 8-bit operand"));
}

#[test]
fn mxf4_family_descriptor_requires_the_scale_type_its_kind_spells() {
    let nvf4 = decode_mxf4(
        mxf4_family_descriptor(false, 0),
        Mxf4ScaleSpelling::Ue4m3Vec4x,
        MatrixDescriptorLayout::Sm100,
    )
    .unwrap();
    assert_eq!((nvf4.m, nvf4.n, nvf4.sfa_id, nvf4.sfb_id), (128, 8, 0, 0));

    // The same bits are an `.kind::mxf4` descriptor with the wrong scale type.
    let error = decode_mxf4(
        mxf4_family_descriptor(false, 0),
        Mxf4ScaleSpelling::Ue8m0Vec2x,
        MatrixDescriptorLayout::Sm100,
    )
    .unwrap_err();
    assert!(error.to_string().contains("must encode UE8M0 scales"));

    let error = decode_mxf4(
        mxf4_family_descriptor(true, 0),
        Mxf4ScaleSpelling::Ue4m3Vec4x,
        MatrixDescriptorLayout::Sm100,
    )
    .unwrap_err();
    assert!(error.to_string().contains("must encode UE4M3 scales"));

    let nvf4_ue8m0 = decode_mxf4(
        mxf4_family_descriptor(true, 0),
        Mxf4ScaleSpelling::Ue8m0Vec4x,
        MatrixDescriptorLayout::Sm100,
    )
    .unwrap();
    assert_eq!(
        (
            nvf4_ue8m0.m,
            nvf4_ue8m0.n,
            nvf4_ue8m0.sfa_id,
            nvf4_ue8m0.sfb_id
        ),
        (128, 8, 0, 0)
    );

    assert_eq!(
        block_mxf4_shape(
            mxf4_family_descriptor(false, 0),
            Mxf4ScaleSpelling::Ue4m3Vec4x,
            1,
            MatrixDescriptorLayout::Sm100,
            false,
        )
        .unwrap(),
        (128, 8, 64)
    );
    assert_eq!(
        block_mxf4_shape(
            mxf4_family_descriptor(true, 0),
            Mxf4ScaleSpelling::Ue8m0Vec4x,
            1,
            MatrixDescriptorLayout::Sm100,
            false,
        )
        .unwrap(),
        (128, 8, 64)
    );
}

#[test]
fn fp4_new_scale_types_require_sm107_and_reject_reserved_encoding() {
    for (encoding, scale) in [
        (0, Mxf4ScaleSpelling::Ue4m3Vec2x),
        (2, Mxf4ScaleSpelling::Ue5m3Vec2x),
        (2, Mxf4ScaleSpelling::Ue5m3Vec4x),
    ] {
        let descriptor = mxf4_family_descriptor(false, 0) | (encoding << 23);
        let error = decode_mxf4(descriptor, scale, MatrixDescriptorLayout::Sm100).unwrap_err();
        assert!(error.to_string().contains("require SM107"));
        decode_mxf4(descriptor | (1 << 12), scale, MatrixDescriptorLayout::Sm107).unwrap();
    }
    let reserved = mxf4_family_descriptor(false, 0) | (3 << 23) | (1 << 12);
    assert!(decode_mxf4(
        reserved,
        mxf4nvf4_vec4x_scale(reserved),
        MatrixDescriptorLayout::Sm107,
    )
    .is_err());
}

#[test]
fn scale_vec_4x_rejects_the_nonzero_scale_factor_ids_2x_allows() {
    // `.scale_vec::2X` selects one byte pair, so SFA ID 2 is legal there.
    decode_mxf4(
        mxf4_family_descriptor(true, 2),
        Mxf4ScaleSpelling::Ue8m0Vec2x,
        MatrixDescriptorLayout::Sm100,
    )
    .unwrap();

    // `.scale_vec::4X` spends all four bytes, so the ID must be 0.
    let error = decode_mxf4(
        mxf4_family_descriptor(false, 2),
        Mxf4ScaleSpelling::Ue4m3Vec4x,
        MatrixDescriptorLayout::Sm100,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("scale_vec::4X requires SFA/SFB IDs 0"));

    let error = decode_mxf4(
        mxf4_family_descriptor(true, 2),
        Mxf4ScaleSpelling::Ue8m0Vec4x,
        MatrixDescriptorLayout::Sm100,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("scale_vec::4X requires SFA/SFB IDs 0"));
}

#[test]
fn fp4_k_encoding_and_sparsity_version_follow_the_authored_architecture() {
    use MatrixDescriptorLayout::{Sm100, Sm103, Sm107};
    for (layout, k, k_bits, id) in [
        (Sm100, 64, 0, 0),
        (Sm103, 96, 1 << 31, 0),
        (Sm103, 96, 1 << 31, 2),
        (Sm107, 128, 1 << 3, 0),
    ] {
        let bits = mxf4_family_descriptor(false, id) | k_bits | (u32::from(layout == Sm107) << 12);
        let decode = |value, arch| decode_mxf4(value, Mxf4ScaleSpelling::Ue4m3Vec4x, arch);
        let block = decode(bits, layout).unwrap();
        assert_eq!((block.k, block.block_elements), (k, 16));
        if k == 64 {
            let vector =
                decode_mxf4_for_cta_group(bits, Mxf4ScaleSpelling::Ue4m3Vec4x, 1, layout, true)
                    .unwrap();
            let fields = |d: Mxf4Instr| {
                (
                    d.sfa_lanes,
                    d.m,
                    d.n,
                    d.k,
                    d.sfa_id,
                    d.sfb_id,
                    d.negate_a,
                    d.negate_b,
                    d.block_elements,
                )
            };
            assert_eq!(fields(block), fields(vector));
        }
        assert!(decode(bits ^ (1 << 12), layout).is_err());
        if k != 64 {
            assert!(decode(bits, Sm100).is_err());
        }
        if k == 128 {
            assert!(decode(bits & !(1 << 12), Sm103).is_err());
        }
    }
    let invalid = mxf4_family_descriptor(false, 0) | (1 << 12) | (1 << 31) | (1 << 3);
    assert!(decode_mxf4(invalid, Mxf4ScaleSpelling::Ue4m3Vec4x, Sm107,).is_err());
}

#[test]
fn b16_half_and_ws_descriptors_validate_the_actual_geometry() {
    let half = f8f6f4_descriptor(0, 0, 0, 128, 64);
    assert!(decode_b16(half, false, false, 1, false, false).is_ok());
    assert!(decode_b16(half | (1 << 7), true, false, 1, false, false,).is_err());
    for sparse in [false, true] {
        let mixed = f8f6f4_descriptor(1, 1, 0, 128, 16) | if sparse { 4 } else { 0 };
        for a_bf16 in [false, true] {
            let decode = |bits| decode_b16(bits, a_bf16, false, 1, false, sparse).unwrap_err();
            assert!(decode(mixed)
                .to_string()
                .contains("requires matching F16/BF16"));
        }
    }
    for sparse in [false, true] {
        let flags = if sparse { 4 } else { 0 };
        for n in [8, 16, 24, 32, 48, 256] {
            for (cta_group, m) in [(1, 64), (1, 128), (2, 128), (2, 256)] {
                // PTX ISA shape table (delta L4).
                let isa = match (cta_group, m) {
                    (1, 64) => n % 8 == 0,
                    (1, _) => n % 16 == 0,
                    _ => n % 32 == 0,
                };
                let descriptor = f8f6f4_descriptor(1, 0, 0, m, n) | flags;
                assert_eq!(
                    decode_b16(descriptor, false, false, cta_group, false, sparse,).is_ok(),
                    isa,
                    "b16 cta{cta_group} M{m} N{n}"
                );
                let tf32 = f8f6f4_descriptor(1, 2, 2, m, n) | flags;
                assert_eq!(
                    decode_tf32(tf32, cta_group as usize, false, sparse,).is_ok(),
                    isa,
                    "tf32 cta{cta_group} M{m} N{n}"
                );
            }
            let ti16 = f8f6f4_descriptor(2, 3, 3, 128, n) | flags;
            assert_eq!(
                crate::tcgen05::integer::integer_shape(
                    crate::tcgen05::integer::IntegerKind::Ti16,
                    ti16,
                    1,
                    false,
                    sparse,
                )
                .is_ok(),
                n % 16 == 0,
            );
        }
        for m in [32, 64, 128] {
            for n in [64, 128, 256] {
                let ws = f8f6f4_descriptor(1, 0, 0, m, n) | (1 << 30) | flags;
                assert_eq!(
                    decode_b16(ws, false, false, 1, true, sparse,).is_ok(),
                    !sparse || n != 256,
                );
                let tf32 = f8f6f4_descriptor(1, 2, 2, m, n) | (1 << 30) | flags;
                assert_eq!(
                    decode_tf32(tf32, 1, true, sparse).is_ok(),
                    !sparse || n != 256,
                );
            }
        }
    }
    let narrow = f8f6f4_descriptor(1, 0, 0, 64, 8);
    assert!(decode_b16(narrow, false, false, 1, false, false).is_ok());
    assert!(decode_b16(narrow, false, false, 1, true, false).is_ok());
}

#[test]
fn f8f6f4_ws_geometry_and_diagnostics_match_ptx_table_48() {
    for sparse in [false, true] {
        for m in [16, 32, 64, 128, 256] {
            for n in (8..=264).step_by(8) {
                let descriptor = f8f6f4_descriptor(1, 0, 0, m, n) | if sparse { 4 } else { 0 };
                let result = decode_f8f6f4(
                    descriptor,
                    NarrowFormat::E4M3,
                    NarrowFormat::E4M3,
                    false,
                    1,
                    false,
                    true,
                    sparse,
                );
                let valid =
                    matches!(m, 32 | 64 | 128) && (matches!(n, 64 | 128) || (!sparse && n == 256));
                assert_eq!(result.is_ok(), valid, "M={m}, N={n}, sparse={sparse}");
                if let Err(error) = result {
                    let message = error.to_string();
                    let n_shapes = if sparse {
                        "{64, 128}"
                    } else {
                        "{64, 128, 256}"
                    };
                    assert!(message.contains(".ws cta_group=1, M in {32, 64, 128}"));
                    assert!(message.contains(&format!("N in {n_shapes}")));
                    assert!(message.contains(&format!("got M={m}, N={n}")));
                    assert!(!message.contains("by 8"));
                }
            }
        }
    }
}

#[test]
fn f8f6f4_descriptor_decoding_is_pinned_to_the_specialized_operand_types() {
    // The V specialization names the operand types; the descriptor must
    // agree with it, or the call is a form the engine never modeled.
    let descriptor = f8f6f4_descriptor(1, 1, 0, 128, 16);
    let decoded = decode_f8f6f4(
        descriptor,
        NarrowFormat::E5M2,
        NarrowFormat::E4M3,
        false,
        1,
        false,
        false,
        false,
    )
    .unwrap();
    assert_eq!((decoded.m, decoded.n), (128, 16));

    for (a_format, b_format, d_f16) in [
        (NarrowFormat::E4M3, NarrowFormat::E4M3, false),
        (NarrowFormat::E5M2, NarrowFormat::E5M2, false),
        (NarrowFormat::E5M2, NarrowFormat::E4M3, true),
    ] {
        let error = decode_f8f6f4(
            descriptor, a_format, b_format, d_f16, 1, false, false, false,
        )
        .unwrap_err();
        assert!(
            format!("{error}").contains("must encode dense"),
            "unexpected error {error}"
        );
    }

    // A float16 destination is format 0 in bits 4-5, float32 is format 1.
    let f16_descriptor = f8f6f4_descriptor(0, 0, 1, 64, 8);
    let decoded = decode_f8f6f4(
        f16_descriptor,
        NarrowFormat::E4M3,
        NarrowFormat::E5M2,
        true,
        1,
        false,
        false,
        false,
    )
    .unwrap();
    assert_eq!((decoded.m, decoded.n), (64, 8));
    assert!(decode_f8f6f4(
        f16_descriptor,
        NarrowFormat::E4M3,
        NarrowFormat::E5M2,
        false,
        1,
        false,
        false,
        false,
    )
    .is_err());
}

#[test]
fn f8f6f4_cta2_descriptor_distinguishes_sm100_and_sm107_k_widths() {
    // Canonical bmm_fp8_rubin tactic 1: M=256, N=128, K=64, F32 D,
    // E5M2 A/B, MN-major B. SM107's K=64 selector is bit 29.
    let descriptor = f8f6f4_descriptor(1, 1, 1, 256, 128) | (1 << 16) | (1 << 29);
    let decoded = decode_f8f6f4(
        descriptor,
        NarrowFormat::E5M2,
        NarrowFormat::E5M2,
        false,
        2,
        true,
        false,
        false,
    )
    .unwrap();
    assert_eq!((decoded.m, decoded.n, decoded.k), (256, 128, 64));
    assert!(!decoded.transpose_a);
    assert!(decoded.transpose_b);

    let sm100_error = decode_f8f6f4(
        descriptor,
        NarrowFormat::E5M2,
        NarrowFormat::E5M2,
        false,
        2,
        false,
        false,
        false,
    )
    .unwrap_err();
    assert!(sm100_error.to_string().contains("must encode dense"));

    let k32_descriptor = descriptor & !(1 << 29);
    for supports_k64 in [false, true] {
        let decoded = decode_f8f6f4(
            k32_descriptor,
            NarrowFormat::E5M2,
            NarrowFormat::E5M2,
            false,
            2,
            supports_k64,
            false,
            false,
        )
        .unwrap();
        assert_eq!(decoded.k, 32);
    }

    let f16 = f8f6f4_descriptor(0, 1, 1, 256, 128) | (1 << 29);
    let decoded = decode_f8f6f4(
        f16,
        NarrowFormat::E5M2,
        NarrowFormat::E5M2,
        true,
        2,
        true,
        false,
        false,
    )
    .unwrap();
    assert_eq!((decoded.m, decoded.n, decoded.k), (256, 128, 64));
}

#[test]
fn f8f6f4_cta1_mn_major_b_requires_sixteen_column_granularity() {
    for n in [8, 16, 24, 256] {
        let descriptor = f8f6f4_descriptor(1, 1, 0, 128, n) | (1 << 16);
        let result = decode_f8f6f4(
            descriptor,
            NarrowFormat::E5M2,
            NarrowFormat::E4M3,
            false,
            1,
            false,
            false,
            false,
        );
        assert_eq!(result.is_ok(), n % 16 == 0);
    }
}

#[test]
fn f8f6f4_cta2_n_granularity_follows_b_major() {
    // PTX ISA shape table (delta L4): cta_group::2 takes N % 32 for either
    // B major; N = 16 is not a hardware shape.
    let decode = |bits| {
        decode_f8f6f4(
            bits,
            NarrowFormat::E5M2,
            NarrowFormat::E5M2,
            false,
            2,
            false,
            false,
            false,
        )
    };
    for transpose in [0, 1 << 16] {
        let n16 = f8f6f4_descriptor(1, 1, 1, 256, 16) | transpose;
        let error = decode(n16).unwrap_err();
        assert!(
            error.to_string().contains("N in 32..=256 by 32"),
            "unexpected error {error}"
        );
        let decoded = decode(f8f6f4_descriptor(1, 1, 1, 256, 32) | transpose).unwrap();
        assert_eq!(
            (decoded.m, decoded.n, decoded.transpose_b),
            (256, 32, transpose != 0)
        );
    }
}

/// Exhaustive grid over the PTX ISA dense shape table (delta L4): every
/// (cta_group, M, N) for f16/bf16, tf32 and f8f6f4 decodes exactly when the
/// ISA lists it.
#[test]
fn dense_shapes_match_the_ptx_isa_table() {
    for cta_group in [1_usize, 2] {
        for m in (16..=256).step_by(16) {
            for n in (8..=256).step_by(8) {
                let isa = match (cta_group, m) {
                    (1, 64) => n % 8 == 0,
                    (1, 128) => n % 16 == 0,
                    (2, 128 | 256) => n % 32 == 0,
                    _ => false,
                };
                let (m32, n32) = (m as u32, n as u32);
                let b16 = decode_b16(
                    f8f6f4_descriptor(1, 0, 0, m32, n32),
                    false,
                    false,
                    cta_group as u32,
                    false,
                    false,
                )
                .is_ok();
                let tf32 = decode_tf32(
                    f8f6f4_descriptor(1, 2, 2, m32, n32),
                    cta_group,
                    false,
                    false,
                )
                .is_ok();
                let f8 = decode_f8f6f4(
                    f8f6f4_descriptor(1, 0, 0, m32, n32),
                    NarrowFormat::E4M3,
                    NarrowFormat::E4M3,
                    false,
                    cta_group as u32,
                    false,
                    false,
                    false,
                )
                .is_ok();
                assert_eq!((b16, tf32, f8), (isa, isa, isa), "cta{cta_group} M{m} N{n}");
                for (kind, k) in [("f16", 16), ("tf32", 8), ("f8f6f4", 32)] {
                    let encoded = crate::tcgen05::encode::validate_tcgen05_instruction_shape(
                        kind,
                        cta_group as i64,
                        m as i64,
                        n as i64,
                        k,
                        false,
                    );
                    assert_eq!(
                        encoded.is_ok(),
                        isa,
                        "encode {kind} cta{cta_group} M{m} N{n}"
                    );
                }
            }
        }
    }
}

/// Encode one dense `kind::f8f6f4` instruction descriptor the way the
/// frontend does, so the decoder is checked against a real bit layout.
fn f8f6f4_descriptor(d_format: u32, a_format: u32, b_format: u32, m: u32, n: u32) -> u32 {
    (d_format << 4) | (a_format << 7) | (b_format << 10) | ((n >> 3) << 17) | ((m >> 4) << 24)
}

/// W11-7: the shape a runtime descriptor encodes is compared with the declared
/// MMA shape; the reproducer's flipped N bit (bit 17) turns N=16 into N=24.
#[test]
fn runtime_descriptor_shape_must_match_the_declared_mma() {
    // kind::mxf8f6f4 M=128 N=16 K=32: M field 8 at bits 24.., N field 2 at 17.., UE8M0 bit 23.
    let mx = (8_u32 << 24) | (2 << 17) | (1 << 23);
    let declared = MmaShape {
        m: 128,
        n: 16,
        k: 32,
    };
    assert_eq!(idesc_shape(mx, IdescFamily::Mxf8f6f4), declared);
    check_declared_shape(mx, IdescFamily::Mxf8f6f4, declared).unwrap();
    let flipped = mx ^ (1 << 17);
    assert_eq!(idesc_shape(flipped, IdescFamily::Mxf8f6f4).n, 24);
    let err = check_declared_shape(flipped, IdescFamily::Mxf8f6f4, declared)
        .unwrap_err()
        .0;
    assert!(
        err.contains("M=128 N=24 K=32") && err.contains("M=128 N=16 K=32"),
        "{err}"
    );
    // Bit 26 is not part of the mxf8f6f4 M field; bit 31 selects K=64; sparse doubles K.
    assert_eq!(idesc_shape(mx | (1 << 26), IdescFamily::Mxf8f6f4), declared);
    assert_eq!(idesc_shape(mx | (1 << 31), IdescFamily::Mxf8f6f4).k, 64);
    assert_eq!(idesc_shape(mx | 4, IdescFamily::Mxf8f6f4).k, 64);
    // Dense kinds: f16 K=16, tf32 K=8, f8f6f4 K=32 (64 with bit 29); M/N errors either way.
    let dense = (4_u32 << 24) | (8 << 17);
    assert_eq!(
        idesc_shape(dense, IdescFamily::Dense { k: 16 }),
        MmaShape {
            m: 64,
            n: 64,
            k: 16
        }
    );
    assert_eq!(
        idesc_shape(dense | (1 << 29), IdescFamily::Dense { k: 32 }).k,
        64
    );
    assert_eq!(
        idesc_shape(dense | (1 << 29), IdescFamily::Dense { k: 16 }).k,
        16
    );
    assert!(check_declared_shape(
        dense,
        IdescFamily::Dense { k: 8 },
        MmaShape {
            m: 128,
            n: 64,
            k: 8
        }
    )
    .is_err());
    // kind::mxf4: M = bits 27..29 x 128, K 64 / 96 (bit 31) / 128 (bit 3).
    let mxf4 = (1_u32 << 27) | (32 << 17);
    assert_eq!(
        idesc_shape(mxf4, IdescFamily::Mxf4),
        MmaShape {
            m: 128,
            n: 256,
            k: 64
        }
    );
    assert_eq!(idesc_shape(mxf4 | (1 << 31), IdescFamily::Mxf4).k, 96);
    assert_eq!(idesc_shape(mxf4 | (1 << 3), IdescFamily::Mxf4).k, 128);
    assert!(check_declared_shape(
        mxf4,
        IdescFamily::Mxf4,
        MmaShape {
            m: 256,
            n: 256,
            k: 64
        }
    )
    .is_err());
}
