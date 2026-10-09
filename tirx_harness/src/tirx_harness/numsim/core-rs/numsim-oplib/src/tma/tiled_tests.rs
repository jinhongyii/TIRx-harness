//! Tests ported from `engine-rs/src/runtime/tensor_map.rs` (tensor copy
//! round trips, OOB fill, FP4 codecs, element strides, reductions). Engine
//! memory is replaced by byte vectors and access footprints by the planned
//! byte runs.

use std::collections::BTreeSet;

use super::*;
use crate::tma::bulk::multicast_target_ctas;
use crate::tma::tensor_map::tests::{make_map, try_make_map};
use crate::tma::tensor_map::TensorMapSpec;

fn run_bytes(runs: &[ByteRun]) -> BTreeSet<usize> {
    runs.iter()
        .flat_map(|run| run.byte_offset..run.byte_offset + run.byte_len)
        .collect()
}

#[test]
fn tensor_copy_round_trips_rank_dtype_swizzle_and_pointer_phase_matrix() {
    let cases = [
        (1_usize, 8_usize, None, 0_usize),
        (2, 16, Some(32), 1),
        (3, 32, Some(64), 3),
        (4, 64, Some(128), 7),
        (5, 8, Some(128), 5),
    ];

    for (rank, element_bits, swizzle_bytes, pointer_phase) in cases {
        let inner_elements = 128 / element_bits;
        let mut shape = vec![inner_elements];
        shape.extend(std::iter::repeat_n(2, rank - 1));
        let mut global_strides = Vec::with_capacity(rank.saturating_sub(1));
        let mut byte_len = 16_usize;
        for &extent in shape.iter().skip(1) {
            global_strides.push(byte_len);
            byte_len *= extent;
        }
        let source_bytes = (0..byte_len)
            .map(|index| (index as u8).wrapping_mul(29).wrapping_add(17))
            .collect::<Vec<_>>();
        let source_map = TensorMapLayout::new(
            TensorMapSpec::tiled(
                shape.clone(),
                global_strides.clone(),
                shape.clone(),
                vec![1; rank],
                crate::tma::tensor_map::tests::element_type_for_bits(element_bits),
                None,
                swizzle_bytes,
                TensorMapFillMode::Zero,
            ),
            32,
            byte_len,
        )
        .unwrap();

        let pointer_base = pointer_phase * 128;
        let outer_count = 1_usize << rank.saturating_sub(1);
        let row_stride = swizzle_bytes.unwrap_or(16);
        let shared_byte_len = pointer_base + outer_count * row_stride;
        let origin = vec![0_i64; rank];
        let plan = plan_tiled_g2s(&source_map, &origin, pointer_base).unwrap();
        assert_eq!(plan.bytes_per_target().unwrap() as usize, byte_len);
        assert_eq!(run_bytes(&plan.source_runs), (0..byte_len).collect());

        let mut actual_shared = vec![0_u8; shared_byte_len];
        execute_g2s(
            &source_map,
            &plan,
            &source_bytes,
            &mut actual_shared,
            pointer_base,
            0,
        )
        .unwrap();
        let mut expected_shared = vec![0_u8; shared_byte_len];
        let mut expected_shared_footprint = BTreeSet::new();
        for outer in 0..outer_count {
            for byte_in_row in 0..16 {
                let relative = if let Some(swizzle_bytes) = swizzle_bytes {
                    let groups = swizzle_bytes / 16;
                    let row_shift = match groups {
                        2 => 2,
                        4 => 1,
                        8 => 0,
                        _ => unreachable!(),
                    };
                    let atom = ((outer >> row_shift) + pointer_phase) % groups;
                    outer * swizzle_bytes + atom * 16 + byte_in_row
                } else {
                    outer * 16 + byte_in_row
                };
                expected_shared[pointer_base + relative] = source_bytes[outer * 16 + byte_in_row];
                expected_shared_footprint.insert(relative);
            }
        }
        assert_eq!(actual_shared, expected_shared);
        assert_eq!(run_bytes(&plan.destination_runs), expected_shared_footprint);

        let store = plan_tiled_s2g(&source_map, &origin, pointer_base).unwrap();
        let mut round_trip = vec![0_u8; byte_len];
        execute_s2g_copy(&store, &actual_shared, pointer_base, &mut round_trip).unwrap();
        assert_eq!(round_trip, source_bytes);
    }
}

#[test]
fn tensor_copy_oob_nan_fill_and_source_footprint_match_scalar_oracle() {
    for (element_bits, element_type) in [
        (16_usize, TensorMapElementType::F16),
        (32, TensorMapElementType::F32),
        (64, TensorMapElementType::F64),
    ] {
        let element_bytes = element_bits / 8;
        let element_count = 16 / element_bytes;
        let source_bytes = (0..16)
            .map(|index| (index as u8).wrapping_mul(11).wrapping_add(3))
            .collect::<Vec<_>>();
        let tensor_map = TensorMapLayout::new(
            TensorMapSpec::tiled(
                vec![element_count],
                vec![],
                vec![element_count],
                vec![1],
                element_type,
                None,
                None,
                TensorMapFillMode::OobNan,
            ),
            0,
            16,
        )
        .unwrap();
        let origin = [i64::try_from(element_count / 2).unwrap()];
        let plan = plan_tiled_g2s(&tensor_map, &origin, 0).unwrap();
        let mut actual = vec![0_u8; 16];
        execute_g2s(&tensor_map, &plan, &source_bytes, &mut actual, 0, 0).unwrap();
        let valid_bytes = 8_usize;
        let mut expected = source_bytes[valid_bytes..].to_vec();
        expected.extend(std::iter::repeat_n(0x7ff7_u16.to_le_bytes(), valid_bytes / 2).flatten());
        assert_eq!(actual, expected);
        assert_eq!(run_bytes(&plan.source_runs), (valid_bytes..16).collect());
    }
}

#[test]
fn fp4_tensor_store_preserves_odd_origin_nibbles() {
    let tensor_map = TensorMapLayout::new(
        TensorMapSpec::tiled(
            vec![32, 1],
            vec![16],
            vec![32, 1],
            vec![1, 1],
            TensorMapElementType::Float4E2M1Fn,
            Some(Fp4SharedLayout::Align8Packed),
            None,
            TensorMapFillMode::Zero,
        ),
        0,
        16,
    )
    .unwrap();
    let source_bytes = (0..16)
        .map(|index| (index as u8).wrapping_mul(19).wrapping_add(0x21))
        .collect::<Vec<_>>();

    let plan = plan_tiled_s2g(&tensor_map, &[1, 0], 0).unwrap();
    assert!(plan.destination_runs.is_empty());
    let mut actual = vec![0_u8; 16];
    execute_s2g_copy(&plan, &source_bytes, 0, &mut actual).unwrap();
    let mut expected = vec![0_u8; 16];
    for local_element in 0..31 {
        let source_nibble = (source_bytes[local_element / 2] >> ((local_element % 2) * 4)) & 0x0f;
        let global_element = local_element + 1;
        expected[global_element / 2] |= source_nibble << ((global_element % 2) * 4);
    }
    assert_eq!(actual, expected);
}

#[test]
fn tensor_copy_multicast_delivers_identical_bytes_and_distinct_footprints() {
    let source_bytes = (0_u8..16).map(|value| value ^ 0x5a).collect::<Vec<_>>();
    let tensor_map = make_map(16, vec![16], vec![], vec![16], 8, None, None);
    let targets = multicast_target_ctas(true, 0b11, 2, 0).unwrap();
    assert_eq!(targets, vec![0, 1]);
    let plan = plan_tiled_g2s(&tensor_map, &[0], 0).unwrap();
    assert_eq!(plan.bytes_per_target().unwrap(), 16);
    for _ in &targets {
        let mut shared = vec![0_u8; 16];
        execute_g2s(&tensor_map, &plan, &source_bytes, &mut shared, 0, 0).unwrap();
        assert_eq!(shared, source_bytes);
    }
    assert_eq!(run_bytes(&plan.destination_runs), (0..16).collect());
}

#[test]
fn tensor_reduction_applies_each_in_bounds_element_once() {
    let initial = [10_u32, 20, 30, 40];
    let tensor_map = make_map(16, vec![4], vec![], vec![4], 32, None, None);
    let contributions = [1_u32, 2, 3, 99];
    let contribution_bytes = contributions
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let plan = plan_tiled_s2g(&tensor_map, &[1], 0).unwrap();
    let payload =
        gather_payload(&contribution_bytes, 0, &plan.source_runs, plan.payload_len).unwrap();
    let (reduction, elements) =
        s2g_reduction_elements(&tensor_map, &plan, &payload, RawTmaReductionOp::Add).unwrap();
    assert_eq!(reduction, TmaReduction::AddU32);
    let mut global = initial
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    for element in &elements {
        let current = &mut global[element.byte_offset..element.byte_offset + 4];
        let updated = reduction.apply(current, &element.source);
        current.copy_from_slice(&updated);
    }
    let actual = global
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    assert_eq!(actual, vec![10, 21, 32, 43]);
}

#[test]
fn tensor_copy_observes_outer_element_stride_in_both_directions() {
    let source_bytes = (0_u8..80).collect::<Vec<_>>();
    let source_map = try_make_map(
        80,
        vec![16, 5],
        vec![16],
        vec![16, 5],
        vec![1, 2],
        8,
        None,
        None,
    )
    .unwrap();
    let plan = plan_tiled_g2s(&source_map, &[0, 0], 0).unwrap();
    assert_eq!(plan.bytes_per_target().unwrap(), 48);
    let mut shared = vec![0_u8; 48];
    execute_g2s(&source_map, &plan, &source_bytes, &mut shared, 0, 0).unwrap();

    let expected_payload = [
        &source_bytes[0..16],
        &source_bytes[32..48],
        &source_bytes[64..80],
    ]
    .concat();
    assert_eq!(shared, expected_payload);
    let expected_footprint = [0_usize, 32, 64]
        .into_iter()
        .flat_map(|row| row..row + 16)
        .collect::<BTreeSet<_>>();
    assert_eq!(run_bytes(&plan.source_runs), expected_footprint);

    let store = plan_tiled_s2g(&source_map, &[0, 0], 0).unwrap();
    let mut destination = vec![0_u8; 80];
    execute_s2g_copy(&store, &shared, 0, &mut destination).unwrap();
    let mut expected_destination = vec![0_u8; 80];
    expected_destination[0..16].copy_from_slice(&expected_payload[0..16]);
    expected_destination[32..48].copy_from_slice(&expected_payload[16..32]);
    expected_destination[64..80].copy_from_slice(&expected_payload[32..48]);
    assert_eq!(destination, expected_destination);
}

#[test]
fn repeated_read_payloads_zero_fill_out_of_bounds_units() {
    let global = (10_u8..42).collect::<Vec<_>>();
    let tensor_map = make_map(32, vec![16, 2], vec![16], vec![16, 4], 8, None, None);
    let template = &tensor_map.transfer_template;
    for (origin, expected) in [([0, 0], 12), ([0, 2], 0)] {
        let source = template.bind_global(&tensor_map, &origin).unwrap();
        let (payload, _) = materialize_g2s_payload(
            &global,
            tensor_map.element_type,
            &source,
            template.payload_len,
            template.geometry.unit_bytes,
            tensor_map.fill_mode,
            0,
        )
        .unwrap();
        assert_eq!(payload.len(), 64);
        assert_eq!(payload[2], expected);
        assert_eq!(&payload[32..], &[0; 32]);
    }
}

#[test]
fn fp4_read_codec_matches_packed_and_padded_shared_layouts() {
    let read = |global: &[u8], map: &TensorMapLayout| {
        let template = &map.transfer_template;
        let source = template.bind_global(map, &[0, 0]).unwrap();
        materialize_g2s_payload(
            global,
            map.element_type,
            &source,
            template.payload_len,
            template.geometry.unit_bytes,
            map.fill_mode,
            0,
        )
        .unwrap()
        .0
    };
    let align8 = make_map(
        16,
        vec![32, 1],
        vec![16],
        vec![32, 1],
        4,
        Some(Fp4SharedLayout::Align8Packed),
        None,
    );
    assert_eq!(
        read(&[vec![0x21, 0x43], vec![0; 14]].concat(), &align8)[..1],
        [0x21]
    );

    let packed = vec![0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe];
    let align16 = make_map(
        64,
        vec![128, 1],
        vec![64],
        vec![128, 1],
        4,
        Some(Fp4SharedLayout::Align16Padded),
        None,
    );
    assert_eq!(
        &read(&[packed.clone(), vec![0; 56]].concat(), &align16)[..8],
        packed.as_slice()
    );
}

#[test]
fn tf32_maps_round_in_bounds_data_and_canonicalize_nan() {
    let values = [
        1.0_f32 + f32::EPSILON,
        f32::from_bits(0x7fc0_0001),
        3.0,
        -0.0,
    ];
    let global = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let map = TensorMapLayout::new(
        TensorMapSpec::tiled(
            vec![4],
            vec![],
            vec![4],
            vec![1],
            TensorMapElementType::Tf32,
            None,
            None,
            TensorMapFillMode::OobNan,
        ),
        0,
        16,
    )
    .unwrap();
    let plan = plan_tiled_g2s(&map, &[2], 0).unwrap();
    let mut shared = vec![0_u8; 16];
    execute_g2s(&map, &plan, &global, &mut shared, 0, 0).unwrap();
    let words = shared
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect::<Vec<_>>();
    assert_eq!(
        words,
        [
            3.0_f32.to_bits(),
            (-0.0_f32).to_bits(),
            0x7ff7_7ff7,
            0x7ff7_7ff7
        ]
    );
    assert_eq!(tma_f32_to_tf32(values[1]).to_bits(), 0x7fff_e000);
    assert_eq!(tma_f32_to_tf32(values[0]), 1.0);
}

#[test]
fn gather4_reads_four_rows_into_consecutive_box_rows() {
    let global = (0..128).map(|i| i as u8).collect::<Vec<_>>();
    let map = make_map(128, vec![16, 8], vec![16], vec![16, 1], 8, None, None);
    let plan = plan_gather4_g2s(&map, 0, &[3, 0, 7, 9], 0).unwrap();
    assert_eq!(plan.payload_len, 64);
    let mut shared = vec![0xaa_u8; 64];
    execute_g2s(&map, &plan, &global, &mut shared, 0, 0).unwrap();
    assert_eq!(&shared[..16], &global[48..64]);
    assert_eq!(&shared[16..32], &global[..16]);
    assert_eq!(&shared[32..48], &global[112..128]);
    assert_eq!(&shared[48..], &[0; 16]); // row 9 is OOB
    assert!(plan_gather4_g2s(&map, 0, &[0, 1, 2], 0).is_err());
}
