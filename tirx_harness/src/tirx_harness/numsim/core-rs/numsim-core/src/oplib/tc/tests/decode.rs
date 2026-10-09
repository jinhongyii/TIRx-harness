//! Descriptor decoders.

use super::*;

// ---------------------------------------------------------------------------
// Descriptor decoders
// ---------------------------------------------------------------------------

#[test]
fn smem_desc_round_trips_the_encoder() {
    for (swizzle, code) in [(0_i64, 0_u8), (1, 1), (2, 2), (3, 3), (4, 4)] {
        let bits = encode_matrix_descriptor(0x1a40, 0x20, 0x40, swizzle);
        let desc = decode_smem_desc(bits).unwrap();
        assert_eq!((desc.start, desc.lbo, desc.sbo), (0x1a40, 0x200, 0x400));
        assert_eq!(
            (desc.swizzle, desc.version, desc.base_offset, desc.lbo_mode),
            (code, 1, 0, 0)
        );
    }
    assert!(decode_smem_desc(0).is_err(), "version 0 is invalid");
    assert!(
        decode_smem_desc(encode_matrix_descriptor(0, 1, 1, 0) | (1 << 49)).is_err(),
        "base offset"
    );
}

#[test]
fn instr_desc_round_trips_the_encoders() {
    let d = decode_instr_desc(
        dense_idesc("float32", "bfloat16", "bfloat16", 128, 64, 16),
        TcMmaKind::F16,
    )
    .unwrap();
    assert_eq!(
        (d.m, d.n, d.a, d.b, d.d),
        (
            128,
            64,
            Some(Dtype::BF16),
            Some(Dtype::BF16),
            Some(Dtype::F32)
        )
    );
    let d = decode_instr_desc(
        dense_idesc("float16", "float16", "float16", 64, 8, 16),
        TcMmaKind::F16,
    )
    .unwrap();
    assert_eq!(
        (d.m, d.n, d.a, d.d),
        (64, 8, Some(Dtype::F16), Some(Dtype::F16))
    );
    let neg = encode_dense_instr_descriptor_fields(
        "float32", "float16", "float16", 128, 32, 16, true, false, 1, true, true, false, false,
    )
    .unwrap() as u32;
    let d = decode_instr_desc(neg, TcMmaKind::F16).unwrap();
    assert!(d.a_major_mn && !d.b_major_mn && d.negate_a && d.negate_b);
    let d = decode_instr_desc(
        dense_idesc("float32", "tf32", "tf32", 128, 16, 8),
        TcMmaKind::Tf32,
    )
    .unwrap();
    assert_eq!(
        (d.m, d.n, d.a, d.d),
        (128, 16, Some(Dtype::TF32), Some(Dtype::F32))
    );
    let d = decode_instr_desc(
        dense_idesc("float32", "float8_e4m3fn", "float8_e5m2", 128, 32, 32),
        TcMmaKind::F8f6f4,
    )
    .unwrap();
    assert_eq!(
        (d.a, d.b, d.d),
        (Some(Dtype::E4M3), Some(Dtype::E5M2), Some(Dtype::F32))
    );
    let d = decode_instr_desc(
        dense_idesc("float32", "float4_e2m1fn", "float6_e3m2fn", 64, 16, 32),
        TcMmaKind::F8f6f4,
    )
    .unwrap();
    assert_eq!((d.m, d.a, d.b), (64, Some(Dtype::E2M1), Some(Dtype::E3M2)));
    let d = decode_instr_desc(
        dense_idesc("int32", "int8", "uint8", 128, 32, 32),
        TcMmaKind::I8,
    )
    .unwrap();
    assert_eq!(
        (d.a, d.b, d.d),
        (Some(Dtype::S8), Some(Dtype::U8), Some(Dtype::S32))
    );
    // CTA-pair shapes are accepted (no cta_group in the contract).
    let d = decode_instr_desc(
        encode_dense_instr_descriptor_fields(
            "float32", "bfloat16", "bfloat16", 256, 256, 16, false, false, 2, false, false, false,
            false,
        )
        .unwrap() as u32,
        TcMmaKind::F16,
    )
    .unwrap();
    assert_eq!((d.m, d.n), (256, 256));

    let mx = |a: &str, b: &str, sf: &str, m: i64, n: i64, k: i64| {
        encode_block_scaled_instr_descriptor_fields(
            "float32", a, b, sf, sf, m, n, k, false, false, 1, false, false, false,
        )
        .unwrap() as u32
    };
    let d = decode_instr_desc(
        mx(
            "float8_e4m3fn",
            "float6_e2m3fn",
            "float8_e8m0fnu",
            128,
            64,
            32,
        ),
        TcMmaKind::MxF8f6f4,
    )
    .unwrap();
    assert_eq!(
        (d.m, d.n, d.a, d.b, d.scale_type),
        (
            128,
            64,
            Some(Dtype::E4M3),
            Some(Dtype::E2M3),
            Some(Dtype::UE8M0)
        )
    );
    let d = decode_instr_desc(
        mx(
            "float4_e2m1fn",
            "float4_e2m1fn",
            "float8_e8m0fnu",
            128,
            64,
            64,
        ),
        TcMmaKind::MxF4,
    )
    .unwrap();
    assert_eq!(
        (d.m, d.a, d.scale_type),
        (128, Some(Dtype::E2M1), Some(Dtype::UE8M0))
    );
    let d = decode_instr_desc(
        mx(
            "float4_e2m1fn",
            "float4_e2m1fn",
            "float8_e4m3fn",
            128,
            64,
            64,
        ),
        TcMmaKind::MxF4Nvf4,
    )
    .unwrap();
    assert_eq!(d.scale_type, Some(Dtype::UE4M3));

    // Kind/descriptor disagreement and reserved bits are rejected.
    assert!(decode_instr_desc(
        dense_idesc("float32", "tf32", "tf32", 128, 16, 8),
        TcMmaKind::F16
    )
    .is_err());
    assert!(decode_instr_desc(
        dense_idesc("float32", "bfloat16", "bfloat16", 128, 16, 16) | (1 << 23),
        TcMmaKind::F16
    )
    .is_err());
    assert!(decode_instr_desc(0, TcMmaKind::I8).is_err());
}
