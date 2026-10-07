//! Tests ported from legacy `runtime/tcgen_ops.rs` `mod tests` and the
//! `zero_column_mask_shift_contract` test.
use super::*;

fn whole(len: usize, virtual_base: usize) -> SharedWindow {
    SharedWindow::whole(virtual_base, len)
}

#[test]
fn zero_column_mask_shift_contract() {
    let shifted = 2_u64 << 56;
    assert!(ColumnMask::new(shifted, 128, 64, 0).is_err());
    assert_eq!(
        ColumnMask::new(shifted, 128, 64, 1 << 30)
            .unwrap()
            .source_column(5),
        Some(7)
    );
    assert!(ColumnMask::new(17 << 56, 32, 64, 3 << 30).is_err());
    assert!(ColumnMask::new(1 << 36, 128, 64, 0).is_err());
    assert!(ColumnMask::new(0, 32, 0, 0).is_err());
    let saturated = (1 << 39) | (2 << 40) | (3 << 48) | (1 << 32) | 255;
    let mask = ColumnMask::new(saturated, 128, 64, 0).unwrap();
    assert_eq!(
        (0..8)
            .map(|i| mask.source_column(i).is_none())
            .collect::<Vec<_>>(),
        [true, false, false, false, false, true, true, true]
    );
}

#[test]
fn mxf8f6f4_mn_major_offset_matches_ptx_canonical_layout() {
    let source = whole(4096, 0);
    let descriptor = MatrixDescriptor {
        absolute_leading_address: false,
        start_address: 0,
        leading_byte_offset: 0,
        stride_byte_offset: 1024,
        swizzle_bits: 3,
        swizzle_atom_bytes: 16,
        swizzle_xor_shift: 3,
    };
    assert_eq!(
        byte8_matrix_byte_offset(source, descriptor, 0, 0, true).unwrap(),
        0
    );
    assert_eq!(
        byte8_matrix_byte_offset(source, descriptor, 0, 1, true).unwrap(),
        144
    );
    assert_eq!(
        byte8_matrix_byte_offset(source, descriptor, 127, 31, true).unwrap(),
        3983
    );
}

#[test]
fn fp4_absolute_ldo_maps_a_k96_straddle_and_rejects_other_consumers() {
    use MatrixDescriptorLayout::{Sm100, Sm103, Sm107};
    let source = whole(65536, 0);
    let bits = 0x4010404000000000_u64 | 6 | ((32768_u64 >> 4) << 16);
    let descriptor = decode_packed_matrix_descriptor(bits, Sm103, 48, false).unwrap();
    for row in [0, 1, 7, 8, 127] {
        for column in 0..48 {
            let chunk = if column < 32 {
                96 + column
            } else {
                32768 + column - 32
            };
            let linear = chunk + (row % 8) * 128 + (row / 8) * 1024;
            let expected = linear ^ (((linear >> 7) & 7) << 4);
            assert_eq!(
                shared_byte_offset(source, descriptor, row, column, 1).unwrap(),
                expected
            );
        }
    }
    assert!(decode_matrix_descriptor_for_layout(bits, Sm103).is_err());
    for (arch, k) in [(Sm100, 96), (Sm103, 64), (Sm107, 128)] {
        assert!(decode_packed_matrix_descriptor(bits, arch, k / 2, false).is_err());
    }
    for invalid in [
        bits ^ (6_u64 << 61),
        bits | (1 << 49),
        bits | (1 << 53),
        bits | (1 << 16),
    ] {
        assert!(decode_packed_matrix_descriptor(invalid, Sm103, 48, false).is_err());
    }
}

#[test]
fn sm107_matrix_descriptor_accepts_bit14_but_sm100_and_bit15_fail_closed() {
    let base = (1_u64 << 46) | (2_u64 << 61);
    let extended = base | (1_u64 << 14) | (1_u64 << 30);
    let decoded =
        decode_matrix_descriptor_for_layout(extended, MatrixDescriptorLayout::Sm107).unwrap();
    assert_eq!(decoded.start_address, 1 << 18);
    assert_eq!(decoded.leading_byte_offset, 1 << 18);
    assert!(decode_matrix_descriptor_for_layout(extended, MatrixDescriptorLayout::Sm100).is_err());
    for reserved_bit in [15, 31] {
        assert!(decode_matrix_descriptor_for_layout(
            base | (1_u64 << reserved_bit),
            MatrixDescriptorLayout::Sm107,
        )
        .is_err());
    }
    assert_eq!(decode_matrix_descriptor(base).unwrap().swizzle_bits, 3);
}

#[test]
fn shared_descriptor_swizzles_the_absolute_virtual_address() {
    let source = whole(512, 128);
    let descriptor = MatrixDescriptor {
        absolute_leading_address: false,
        start_address: 128,
        leading_byte_offset: 0,
        stride_byte_offset: 512,
        swizzle_bits: 2,
        swizzle_atom_bytes: 16,
        swizzle_xor_shift: 3,
    };
    assert_eq!(shared_byte_offset(source, descriptor, 0, 0, 2).unwrap(), 16);
    assert_eq!(shared_byte_offset(source, descriptor, 0, 16, 2).unwrap(), 0);
    assert_eq!(
        shared_byte_offset(source, descriptor, 2, 0, 2).unwrap(),
        160
    );
    let unified_source = whole(656, 0);
    let crossing_descriptor = MatrixDescriptor {
        start_address: 144,
        ..descriptor
    };
    assert_eq!(
        shared_byte_offset(unified_source, crossing_descriptor, 0, 0, 2).unwrap(),
        128
    );
}

#[test]
fn access_view_window_bounds_are_enforced() {
    let descriptor = MatrixDescriptor {
        absolute_leading_address: false,
        start_address: 0,
        leading_byte_offset: 0,
        stride_byte_offset: 0,
        swizzle_bits: 0,
        swizzle_atom_bytes: 16,
        swizzle_xor_shift: 3,
    };
    let view = SharedWindow {
        virtual_base: 0,
        view_offset: 4,
        view_len: 8,
        backing_byte_len: 16,
        access_view: true,
    };
    assert_eq!(shared_byte_offset(view, descriptor, 0, 4, 2).unwrap(), 4);
    assert!(shared_byte_offset(view, descriptor, 0, 8, 2)
        .unwrap_err()
        .to_string()
        .contains("exceeds selected shared view"));
}

#[test]
fn tile_gemm_bf16_snapshot_decode_reverses_inner_swizzle() {
    let rows = 8;
    let columns = 64;
    let logical = (0..rows * columns)
        .map(|value| (value % 16) as f32)
        .collect::<Vec<_>>();
    let mut snapshot = vec![0_u8; logical.len() * 2];
    for (unswizzled, value) in logical.iter().enumerate() {
        let quotient = unswizzled >> 3;
        let physical = ((quotient ^ ((quotient & 56) >> 3)) << 3) | (unswizzled & 7);
        snapshot[physical * 2..physical * 2 + 2]
            .copy_from_slice(&crate::cvt::f32_to_bf16_bits(*value).to_le_bytes());
    }
    assert_eq!(
        decode_tile_gemm_bf16_snapshot(
            &snapshot,
            rows,
            columns,
            TileGemmBf16OperandLayout::new(64, 3, 56, 3)
        )
        .unwrap(),
        logical
    );
}

#[test]
fn tile_gemm_bf16_snapshot_decode_reorders_column_atoms() {
    let rows = 8;
    let columns = 128;
    let mut snapshot = vec![0_u8; rows * columns * 2];
    let mut logical = vec![0.0_f32; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let unswizzled = (column / 64) * rows * 64 + row * 64 + column % 64;
            let quotient = unswizzled >> 3;
            let physical = ((quotient ^ ((quotient & 56) >> 3)) << 3) | (unswizzled & 7);
            let value = ((row * columns + column) % 16) as f32;
            logical[row * columns + column] = value;
            snapshot[physical * 2..physical * 2 + 2]
                .copy_from_slice(&crate::cvt::f32_to_bf16_bits(value).to_le_bytes());
        }
    }
    let decoded = decode_tile_gemm_bf16_snapshot(
        &snapshot,
        rows,
        columns,
        TileGemmBf16OperandLayout::new(64, 3, 56, 3),
    )
    .unwrap();
    assert_eq!(decoded, logical);
}

#[test]
fn tile_gemm_fp8_swizzle_maps_a_non_power_of_two_row_count_bijectively() {
    let mut physical =
        tile_gemm_operand_physical_elements(120, 128, TileGemmOperandLayout::new(128, 4, 56, 3))
            .unwrap();
    assert_eq!(physical.len(), 120 * 128);
    physical.sort_unstable();
    assert_eq!(physical, (0..120 * 128).collect::<Vec<_>>());
}
