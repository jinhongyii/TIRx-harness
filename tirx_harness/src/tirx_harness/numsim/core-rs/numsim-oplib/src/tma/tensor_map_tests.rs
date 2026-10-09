//! Tests ported from `engine-rs/src/runtime/tensor_map.rs` (descriptor
//! validation, geometry, coordinate helpers).

use super::*;
use crate::tma::descriptor::TensorMapIm2col;

pub(crate) fn element_type_for_bits(element_bits: usize) -> TensorMapElementType {
    match element_bits {
        4 => TensorMapElementType::Float4E2M1Fn,
        8 => TensorMapElementType::U8,
        16 => TensorMapElementType::U16,
        32 => TensorMapElementType::U32,
        64 => TensorMapElementType::U64,
        _ => TensorMapElementType::Bool,
    }
}

/// Legacy `try_make_map`: a map over a view of `view_byte_len` bytes at
/// absolute address 0 (the legacy allocation was observed at 0).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_make_map(
    view_byte_len: usize,
    global_shape: Vec<usize>,
    global_strides: Vec<usize>,
    box_shape: Vec<usize>,
    element_strides: Vec<usize>,
    element_bits: usize,
    fp4_shared_layout: Option<Fp4SharedLayout>,
    swizzle_bytes: Option<usize>,
) -> OpResult<TensorMapLayout> {
    TensorMapLayout::new(
        TensorMapSpec {
            element_bits,
            ..TensorMapSpec::tiled(
                global_shape,
                global_strides,
                box_shape,
                element_strides,
                element_type_for_bits(element_bits),
                fp4_shared_layout,
                swizzle_bytes,
                TensorMapFillMode::Zero,
            )
        },
        0,
        view_byte_len,
    )
}

pub(crate) fn make_map(
    view_byte_len: usize,
    global_shape: Vec<usize>,
    global_strides: Vec<usize>,
    box_shape: Vec<usize>,
    element_bits: usize,
    fp4_shared_layout: Option<Fp4SharedLayout>,
    swizzle_bytes: Option<usize>,
) -> TensorMapLayout {
    let rank = global_shape.len();
    try_make_map(
        view_byte_len,
        global_shape,
        global_strides,
        box_shape,
        vec![1; rank],
        element_bits,
        fp4_shared_layout,
        swizzle_bytes,
    )
    .unwrap()
}

#[test]
fn fp4_geometry_and_transaction_accounting_match_shared_layout() {
    let error = TensorMapLayout::new(
        TensorMapSpec {
            interleave_bytes: Some(16),
            ..TensorMapSpec::tiled(
                vec![8, 3, 2],
                vec![128, 384],
                vec![4, 1, 1],
                vec![1; 3],
                TensorMapElementType::Float4E2M1Fn,
                Some(Fp4SharedLayout::Align16Padded),
                None,
                TensorMapFillMode::Zero,
            )
        },
        0,
        128 * 6,
    )
    .expect_err("padded interleave remains unmodeled");
    assert_eq!(
        error,
        analysis_incomplete("tma_padded_fp4_interleave_unmodeled")
    );
    for bytes in [16, 32] {
        let geometry = tensor_map_geometry_from_metadata(
            &[3, 2, 1],
            bytes * 8,
            Some(Fp4SharedLayout::Align8Packed),
        )
        .unwrap();
        assert_eq!(geometry.packed_elements, 1);
        assert_eq!(geometry.unit_bytes, bytes);
        assert_eq!(geometry.unit_stride_bytes, bytes);
        assert_eq!(geometry.inner_units, 3);
        assert_eq!(geometry.inner_row_bytes, 3 * bytes);
        assert_eq!(geometry.outer_count, 2);
    }
    let align8 = make_map(
        128,
        vec![128, 2],
        vec![64],
        vec![128, 2],
        4,
        Some(Fp4SharedLayout::Align8Packed),
        Some(64),
    );
    assert_eq!(
        align8.geometry().unwrap(),
        TensorMapGeometry {
            packed_elements: 2,
            unit_bytes: 1,
            unit_stride_bytes: 1,
            inner_units: 64,
            inner_row_bytes: 64,
            outer_count: 2,
        }
    );
    assert_eq!(tensor_map_transaction_bytes(1).unwrap(), 1);

    let align16 = make_map(
        128,
        vec![128, 2],
        vec![64],
        vec![128, 2],
        4,
        Some(Fp4SharedLayout::Align16Padded),
        Some(128),
    );
    assert_eq!(
        align16.geometry().unwrap(),
        TensorMapGeometry {
            packed_elements: 16,
            unit_bytes: 8,
            unit_stride_bytes: 16,
            inner_units: 8,
            inner_row_bytes: 128,
            outer_count: 2,
        }
    );
    assert_eq!(tensor_map_transaction_bytes(8).unwrap(), 8);
}

#[test]
fn rank_coordinates_and_swizzle_are_physical_layout_operations() {
    let rank3 = make_map(
        96,
        vec![4, 3, 2],
        vec![16, 48],
        vec![4, 3, 2],
        32,
        None,
        None,
    );
    let outer = rank3.outer_coordinates(5);
    assert_eq!(outer, vec![0, 2, 1]);
    let coordinates = rank3.global_coordinates(&[0, 0, 0], 2, &outer).unwrap();
    assert_eq!(coordinates, vec![2, 2, 1]);
    assert_eq!(rank3.global_byte_offset(&coordinates).unwrap(), (88, 0));

    let swizzled = make_map(256, vec![8, 8], vec![32], vec![8, 8], 32, None, Some(32));
    assert_eq!(swizzled.shared_byte_offset(4, 0, 32, 0).unwrap(), 144);

    let align16 = make_map(
        128,
        vec![128, 2],
        vec![64],
        vec![128, 2],
        4,
        Some(Fp4SharedLayout::Align16Padded),
        Some(128),
    );
    assert_eq!(align16.shared_byte_offset(1, 0, 128, 0).unwrap(), 144);
    assert_eq!(align16.shared_byte_offset(1, 16, 128, 0).unwrap(), 128);
    assert!(!rank3.coordinates_in_bounds(&[4, 0, 0]));
    assert!(!rank3.coordinates_in_bounds(&[0, -1, 0]));
    assert!(rank3.global_byte_offset(&[0, 3, 0]).is_err());
}

#[test]
fn element_stride_controls_traversal_count_and_coordinates() {
    let tensor_map = try_make_map(
        16 * 5,
        vec![16, 5],
        vec![16],
        vec![16, 5],
        vec![7, 2],
        8,
        None,
        None,
    )
    .unwrap();

    assert_eq!(tensor_map.element_strides, vec![1, 2]);
    assert_eq!(tensor_map.traversal_shape, vec![16, 3]);
    assert_eq!(
        tensor_map.geometry().unwrap(),
        TensorMapGeometry {
            packed_elements: 1,
            unit_bytes: 1,
            unit_stride_bytes: 1,
            inner_units: 16,
            inner_row_bytes: 16,
            outer_count: 3,
        }
    );
    let outer = tensor_map.outer_coordinates(2);
    assert_eq!(outer, vec![0, 2]);
    assert_eq!(
        tensor_map.global_coordinates(&[0, 0], 7, &outer).unwrap(),
        vec![7, 4]
    );
}

#[test]
fn descriptor_validation_rejects_hardware_illegal_ranges() {
    let invalid_box = try_make_map(
        512,
        vec![16, 2],
        vec![16],
        vec![16, 257],
        vec![1, 1],
        8,
        None,
        None,
    )
    .err()
    .unwrap();
    assert!(invalid_box.to_string().contains("1..=256"));

    let invalid_element_stride = try_make_map(
        32,
        vec![16, 2],
        vec![16],
        vec![16, 2],
        vec![1, 9],
        8,
        None,
        None,
    )
    .err()
    .unwrap();
    assert!(invalid_element_stride.to_string().contains("1..=8"));

    let invalid_global_stride = try_make_map(
        32,
        vec![16, 2],
        vec![24],
        vec![16, 2],
        vec![1, 1],
        8,
        None,
        None,
    )
    .err()
    .unwrap();
    assert!(invalid_global_stride
        .to_string()
        .contains("multiples of 16 below 2^40"));

    let oversized_global_stride = try_make_map(
        32,
        vec![16, 2],
        vec![1_usize << 40],
        vec![16, 2],
        vec![1, 1],
        8,
        None,
        None,
    )
    .err()
    .unwrap();
    assert!(oversized_global_stride
        .to_string()
        .contains("multiples of 16 below 2^40"));

    let overlapping_global_stride = try_make_map(
        64,
        vec![16, 2, 2],
        vec![16, 16],
        vec![16, 2, 2],
        vec![1, 1, 1],
        8,
        None,
        None,
    )
    .err()
    .unwrap();
    assert!(overlapping_global_stride
        .to_string()
        .contains("overlaps the prior 32-byte span"));

    let invalid_inner_transfer =
        try_make_map(16, vec![16], vec![], vec![15], vec![1], 8, None, None)
            .err()
            .unwrap();
    assert!(invalid_inner_transfer
        .to_string()
        .contains("multiple of 16 bytes"));

    let invalid_packed_shape = try_make_map(
        64,
        vec![64, 2],
        vec![32],
        vec![128, 1],
        vec![1, 1],
        4,
        Some(Fp4SharedLayout::Align16Padded),
        Some(128),
    )
    .err()
    .unwrap();
    assert!(invalid_packed_shape
        .to_string()
        .contains("global dimension zero to be a multiple of 128"));

    if usize::BITS > 32 {
        let invalid_global_dimension = try_make_map(
            16,
            vec![(1_u64 << 32) as usize + 1],
            vec![],
            vec![16],
            vec![1],
            8,
            None,
            None,
        )
        .err()
        .unwrap();
        assert!(invalid_global_dimension
            .to_string()
            .contains("at most 2^32"));
    }

    // View/address checks that the legacy engine derived from its BufferView.
    let short_view = try_make_map(
        31,
        vec![16, 2],
        vec![16],
        vec![16, 2],
        vec![1, 1],
        8,
        None,
        None,
    )
    .err()
    .unwrap();
    assert_eq!(
        short_view.to_string(),
        "TensorMap requires 32 global bytes, but its view has 31"
    );
    let misaligned = TensorMapLayout::new(
        TensorMapSpec::tiled(
            vec![16],
            vec![],
            vec![16],
            vec![1],
            TensorMapElementType::U8,
            None,
            None,
            TensorMapFillMode::Zero,
        ),
        8,
        16,
    )
    .unwrap_err();
    assert_eq!(
        misaligned.to_string(),
        "TensorMap global address must be 16-byte aligned"
    );
}

#[test]
fn descriptor_accepts_permuted_nonoverlapping_outer_axes() {
    let tensor_map = try_make_map(
        64 * 2 * 128 * 2,
        vec![128, 64, 2],
        vec![512, 256],
        vec![64, 64, 1],
        vec![1, 1, 1],
        16,
        None,
        Some(128),
    )
    .unwrap();

    assert_eq!(tensor_map.global_strides, vec![512, 256]);
}

#[test]
fn image_materializes_with_allocation_relative_view() {
    let map = make_map(256, vec![8, 8], vec![32], vec![8, 8], 32, None, Some(32));
    let image = map.to_image(9, 64);
    let materialized = image.materialize(2, 64 + 256, 0).unwrap();
    assert_eq!(materialized.global_shape, map.global_shape);
    assert_eq!(materialized.to_image(9, 64), image);
    assert!(image.materialize(3, 320, 0).is_err());
    assert!(image.materialize(2, 63, 0).is_err());
    assert!(image.materialize(2, 64 + 255, 0).is_err());
    // A misaligned allocation base fails the 16-byte address rule.
    assert!(image.materialize(2, 320, 8).is_err());
    let mut host = image.clone();
    host.restore_host_address(0x1000).unwrap();
    assert_eq!(
        (host.allocation_id, host.base_byte_offset, host.host_address),
        (0x1040, 0, true)
    );
    assert!(host.materialize(2, 320, 0).is_err());
    host.relocate(9, 64);
    assert_eq!(host, image);
}

#[test]
fn swizzle_direction_and_u6_origin_rules() {
    let spec = TensorMapSpec {
        swizzle_atomicity: SwizzleAtomicity::B32Flip8,
        ..TensorMapSpec::tiled(
            vec![64, 2],
            vec![128],
            vec![64, 2],
            vec![1, 1],
            TensorMapElementType::U16,
            None,
            Some(128),
            TensorMapFillMode::Zero,
        )
    };
    let flip = TensorMapLayout::new(spec.clone(), 0, 256).unwrap();
    flip.validate_swizzle_direction(true).unwrap();
    assert!(flip.validate_swizzle_direction(false).is_err());
    let b64 = TensorMapLayout::new(
        TensorMapSpec {
            swizzle_atomicity: SwizzleAtomicity::B64,
            ..spec.clone()
        },
        0,
        256,
    )
    .unwrap();
    b64.validate_swizzle_direction(false).unwrap();
    assert_eq!(
        b64.validate_swizzle_direction(true).unwrap_err(),
        analysis_incomplete("tma_64b_atomicity_load_unmodeled")
    );
    assert!(TensorMapLayout::new(
        TensorMapSpec {
            swizzle_bytes: Some(64),
            ..spec.clone()
        },
        0,
        256
    )
    .unwrap_err()
    .to_string()
    .contains("requires 128B swizzle"));
    let u6 = TensorMapLayout::new(
        TensorMapSpec::tiled(
            vec![128, 1],
            vec![96],
            vec![128, 1],
            vec![1, 1],
            TensorMapElementType::U6,
            None,
            None,
            TensorMapFillMode::Zero,
        ),
        0,
        96,
    )
    .unwrap();
    assert!(u6.validate_u6_origin(&[64, 0]).is_err());
    u6.validate_u6_origin(&[128, 0]).unwrap();
    let wide = TensorMapLayout::new(
        TensorMapSpec {
            box_shape: vec![64, 32],
            im2col: Some(TensorMapIm2col {
                lower: [0; 3],
                upper: [0; 3],
                wide: true,
            }),
            ..TensorMapSpec::tiled(
                vec![64, 8, 2],
                vec![128, 1024],
                vec![64, 32],
                vec![1, 1, 1],
                TensorMapElementType::U16,
                None,
                Some(128),
                TensorMapFillMode::Zero,
            )
        },
        0,
        2048,
    )
    .unwrap();
    assert_eq!(wide.traversal_shape, vec![64, 32]);
    assert_eq!(wide.transfer_template.payload_len, 128);
}
