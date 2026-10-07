//! Tests ported from `engine-rs/src/runtime/tensor_map.rs` (image codec,
//! replacement fields, overrides, reduction mapping).

use super::*;
use crate::tma::tensor_map::{TensorMapLayout, TensorMapSpec};

fn u32_map(global_shape: Vec<usize>, global_strides: Vec<usize>, box_shape: Vec<usize>) -> TensorMapLayout {
    let rank = global_shape.len();
    TensorMapLayout::new(
        TensorMapSpec::tiled(
            global_shape,
            global_strides,
            box_shape,
            vec![1; rank],
            TensorMapElementType::U32,
            None,
            None,
            TensorMapFillMode::Zero,
        ),
        0,
        1 << 20,
    )
    .unwrap()
}

#[test]
fn replacement_fields_use_ptx_encodings_and_preserve_other_image_fields() {
    let map = u32_map(vec![8, 8], vec![32], vec![8, 8]);
    let original = map.to_image(1, 0);
    for (field, limit) in [
        ("box_dim", MAX_BOX_DIMENSION),
        ("element_stride", MAX_ELEMENT_STRIDE),
    ] {
        for index in 0..5 {
            for value in [1, 2, limit] {
                let mut image = original.clone();
                image.replace_field(field, Some(index), value).unwrap();
                let mut expected = original.clone();
                if field == "box_dim" {
                    expected.box_shape[index] = value;
                } else {
                    expected.element_strides[index] = value;
                }
                assert_eq!(TensorMapImage::decode(&image.encode().unwrap()).unwrap(), expected);
            }
        }
        for (index, value) in [(0, 0), (0, limit + 1), (0, usize::MAX), (5, 1)] {
            let mut image = original.clone();
            assert!(image.replace_field(field, Some(index), value).is_err());
            assert_eq!(image, original);
        }
    }
    for (code, dtype) in [
        (0, TensorMapElementType::U8),
        (1, TensorMapElementType::U16),
        (2, TensorMapElementType::U32),
        (3, TensorMapElementType::I32),
        (4, TensorMapElementType::U64),
        (5, TensorMapElementType::I64),
        (6, TensorMapElementType::F16),
        (7, TensorMapElementType::F32),
        (8, TensorMapElementType::F32Ftz),
        (9, TensorMapElementType::F64),
        (10, TensorMapElementType::Bf16),
        (11, TensorMapElementType::Tf32),
        (12, TensorMapElementType::Tf32Ftz),
    ] {
        let mut image = original.clone();
        image.replace_field("elemtype", None, code).unwrap();
        let expected = TensorMapImage {
            element_type: dtype,
            ..original.clone()
        };
        assert_eq!(TensorMapImage::decode(&image.encode().unwrap()).unwrap(), expected);
    }
    for (field, value) in [
        ("rank", 4),
        ("swizzle_mode", 3),
        ("swizzle_mode", 4),
        ("fill_mode", 1),
        ("interleave_layout", 0),
        ("interleave_layout", 1),
        ("interleave_layout", 2),
    ] {
        let mut image = original.clone();
        image.replace_field(field, None, value).unwrap();
        let mut expected = original.clone();
        match field {
            "rank" => expected.rank = 5,
            "swizzle_mode" => expected.swizzle_bytes = Some(if value == 4 { 96 } else { 128 }),
            "fill_mode" => expected.fill_mode = TensorMapFillMode::OobNan,
            "interleave_layout" => {
                expected.interleave_bytes = match value {
                    0 => None,
                    1 => Some(16),
                    2 => Some(32),
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        }
        assert_eq!(TensorMapImage::decode(&image.encode().unwrap()).unwrap(), expected);
    }
    for (field, value) in [
        ("rank", 5),
        ("elemtype", 13),
        ("elemtype", 16),
        ("swizzle_mode", 5),
        ("fill_mode", 2),
        ("interleave_layout", 3),
    ] {
        let mut image = original.clone();
        assert!(image.replace_field(field, None, value).is_err());
        assert_eq!(image, original);
    }
}

#[test]
fn private_image_round_trips_documented_field_maxima() {
    let maximum_stride = usize::try_from(MAX_GLOBAL_STRIDE - 16).unwrap();
    let maximum_dimension = usize::try_from(MAX_GLOBAL_DIMENSION).unwrap();
    let image = TensorMapImage {
        allocation_id: u64::MAX,
        im2col: None,
        base_byte_offset: usize::MAX,
        host_address: true,
        rank: 5,
        physical_global_shape: [maximum_dimension; 5],
        physical_global_strides: [0, 16, maximum_stride, 32],
        box_shape: [MAX_BOX_DIMENSION; 5],
        element_strides: [MAX_ELEMENT_STRIDE; 5],
        element_type: TensorMapElementType::U32x2,
        interleave_bytes: None,
        fp4_shared_layout: None,
        swizzle_bytes: Some(128),
        swizzle_atomicity: SwizzleAtomicity::B16,
        fill_mode: TensorMapFillMode::Zero,
    };

    let encoded = image.encode().unwrap();
    assert_eq!(encoded.len(), TENSOR_MAP_PAYLOAD_BYTES);
    assert_eq!(u32::from_le_bytes(encoded[16..20].try_into().unwrap()), 0);
    assert_eq!(TensorMapImage::decode(&encoded).unwrap(), image);
}

#[test]
fn override_decodes_all_four_stride_nibbles_without_mutating_source() {
    let original = TensorMapLayout::new(
        TensorMapSpec::tiled(
            vec![4, 1, 1, 1, 1],
            vec![16; 4],
            vec![4, 1, 1, 1, 1],
            vec![1; 5],
            TensorMapElementType::F32,
            None,
            None,
            TensorMapFillMode::Zero,
        ),
        0,
        128 * 1024,
    )
    .unwrap();
    validate_override_address(0, 128 * 1024).unwrap();
    assert!(validate_override_address(8, 128 * 1024).is_err());
    assert!(validate_override_address(0, 128 * 1024 - 1).is_err());
    let source = original.to_image(1, 0);
    let mut image = source.clone();
    image
        .apply_overrides(5, &[4, 1, 1, 1, 1], &[1, 2, 3, 4], 0x4321, &[0; 5])
        .unwrap();
    // The legacy override re-materializes; the descriptor must still be legal
    // structurally (strides are not validated against a span here).
    for axis in 0..4 {
        assert_eq!(
            image.physical_global_strides[axis],
            (((axis + 1) << 32) | (axis + 1)) << 4
        );
    }
    let overridden = image.materialize(5, 128 * 1024, 0).unwrap();
    assert_eq!(overridden.physical_global_strides, image.physical_global_strides);
    assert_eq!(source.physical_global_strides, [16; 4]);
    assert_eq!(original.physical_global_strides, [16; 4]);
    let mut bad = source.clone();
    assert!(bad
        .apply_overrides(5, &[4, 1, 1, 1, 1], &[1, 2, 3, 4], 0x14321, &[0; 5])
        .is_err());
    assert!(bad
        .apply_overrides(5, &[4, 1, 1, 1, 1], &[1, 2, 3, 4], 0, &[1, 0, 0, 0, 0])
        .unwrap_err()
        .to_string()
        .contains("zero coordinates"));
    assert!(bad.apply_overrides(5, &[], &[1], 0, &[0; 5]).is_err());
}

#[test]
fn private_image_write_leaves_descriptor_tail_untouched() {
    let mut descriptor = vec![0xa5; TENSOR_MAP_DESCRIPTOR_BYTES];
    let image = TensorMapImage {
        allocation_id: 7,
        im2col: None,
        base_byte_offset: 16,
        host_address: false,
        rank: 1,
        physical_global_shape: [16, 1, 1, 1, 1],
        physical_global_strides: [0; 4],
        box_shape: [16, 1, 1, 1, 1],
        element_strides: [1; 5],
        element_type: TensorMapElementType::U8,
        interleave_bytes: None,
        fp4_shared_layout: None,
        swizzle_bytes: None,
        swizzle_atomicity: SwizzleAtomicity::B16,
        fill_mode: TensorMapFillMode::Zero,
    };

    image.write_descriptor(&mut descriptor).unwrap();

    assert_eq!(TensorMapImage::decode_descriptor(&descriptor).unwrap(), image);
    assert_eq!(
        &descriptor[TENSOR_MAP_PAYLOAD_BYTES..],
        &[0xa5; TENSOR_MAP_DESCRIPTOR_BYTES - TENSOR_MAP_PAYLOAD_BYTES][..],
    );
    // A non-zero tail is not a canonical (discoverable) descriptor ...
    assert_eq!(TensorMapImage::decode_candidate(&descriptor), None);
    // ... while the zero-tailed canonical encoding is.
    descriptor[TENSOR_MAP_PAYLOAD_BYTES..].fill(0);
    assert_eq!(TensorMapImage::decode_candidate(&descriptor), Some(image.clone()));
    let mut noisy = descriptor.clone();
    noisy[20] = 2; // inactive global dimension 1 != 1
    assert_eq!(TensorMapImage::decode_candidate(&noisy), None);
    assert!(validate_descriptor_address(7, 128, 128).is_ok());
    assert!(validate_descriptor_address(7, 64, 128)
        .unwrap_err()
        .to_string()
        .contains("must be 128-byte aligned"));
    assert!(validate_descriptor_address(7, 0, 127).is_err());
}

#[test]
fn im2col_image_extension_round_trips() {
    let image = TensorMapImage {
        allocation_id: 3,
        im2col: Some(TensorMapIm2col {
            lower: [-1, -2, 0],
            upper: [3, 0, 0],
            wide: true,
        }),
        base_byte_offset: 0,
        host_address: false,
        rank: 4,
        physical_global_shape: [64, 8, 8, 2, 1],
        physical_global_strides: [128, 1024, 8192, 0],
        box_shape: [64, 1000, 1, 1, 1],
        element_strides: [1; 5],
        element_type: TensorMapElementType::F16,
        interleave_bytes: None,
        fp4_shared_layout: None,
        swizzle_bytes: Some(128),
        swizzle_atomicity: SwizzleAtomicity::B32,
        fill_mode: TensorMapFillMode::OobNan,
    };
    let encoded = image.encode().unwrap();
    assert_eq!(encoded.len(), 80);
    assert_eq!(tensor_map_payload_bytes(encoded[63]), 80);
    assert_eq!(TensorMapImage::decode(&encoded).unwrap(), image);
    let mut bad = encoded.clone();
    bad[77] = 1;
    assert_eq!(
        TensorMapImage::decode(&bad).unwrap_err().to_string(),
        "invalid im2col TensorMap extension"
    );
}

#[test]
fn reduction_mapping_covers_every_valid_ptx_operation_type_pair() {
    use RawTmaReductionOp as Op;
    use TensorMapElementType as T;
    use TmaReduction as R;
    let cases = [
        (Op::Add, T::U32, R::AddU32),
        (Op::Add, T::I32, R::AddI32),
        (Op::Add, T::U64, R::AddU64),
        (Op::Add, T::F32, R::AddF32),
        (Op::Add, T::F16, R::AddF16),
        (Op::Add, T::Bf16, R::AddBf16),
        (Op::Min, T::U32, R::MinU32),
        (Op::Min, T::I32, R::MinI32),
        (Op::Min, T::U64, R::MinU64),
        (Op::Min, T::I64, R::MinI64),
        (Op::Min, T::F16, R::MinF16),
        (Op::Min, T::Bf16, R::MinBf16),
        (Op::Max, T::U32, R::MaxU32),
        (Op::Max, T::I32, R::MaxI32),
        (Op::Max, T::U64, R::MaxU64),
        (Op::Max, T::I64, R::MaxI64),
        (Op::Max, T::F16, R::MaxF16),
        (Op::Max, T::Bf16, R::MaxBf16),
        (Op::Inc, T::U32, R::IncU32),
        (Op::Dec, T::U32, R::DecU32),
        (Op::And, T::U32, R::AndB32),
        (Op::And, T::U64, R::AndB64),
        (Op::Or, T::U32, R::OrB32),
        (Op::Or, T::U64, R::OrB64),
        (Op::Xor, T::U32, R::XorB32),
        (Op::Xor, T::U64, R::XorB64),
    ];
    for (operation, element_type, expected) in cases {
        assert_eq!(
            operation.resolve(element_type).unwrap(),
            expected,
            "operation={operation:?}, dtype={element_type}"
        );
    }
}

#[test]
fn reduction_mapping_rejects_invalid_ptx_operation_type_pairs() {
    for (operation, element_type) in [
        (RawTmaReductionOp::Add, TensorMapElementType::I64),
        (RawTmaReductionOp::Min, TensorMapElementType::F32),
        (RawTmaReductionOp::Max, TensorMapElementType::F32),
        (RawTmaReductionOp::Inc, TensorMapElementType::I32),
        (RawTmaReductionOp::And, TensorMapElementType::U16),
    ] {
        let error = operation.resolve(element_type).unwrap_err();
        assert!(
            error.to_string().contains("is invalid for TensorMap dtype"),
            "operation={operation:?}, dtype={element_type}: {error}"
        );
    }
}
