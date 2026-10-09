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

/// Swizzle modes of the tcgen05 matrix descriptor (bits 61..63).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RefMode {
    None,
    B32,
    B64,
    B128,
    B128Atom32,
}

impl RefMode {
    const ALL: [Self; 5] = [
        Self::None,
        Self::B32,
        Self::B64,
        Self::B128,
        Self::B128Atom32,
    ];

    fn layout_type(self) -> u64 {
        match self {
            Self::None => 0,
            Self::B32 => 6,
            Self::B64 => 4,
            Self::B128 => 2,
            Self::B128Atom32 => 1,
        }
    }

    /// Bytes of one swizzled row (the repeating pattern is 8 such rows).
    fn width(self) -> usize {
        match self {
            Self::None => 16,
            Self::B32 => 32,
            Self::B64 => 64,
            Self::B128 | Self::B128Atom32 => 128,
        }
    }

    /// CuTe `Swizzle<B, M, S>` of an absolute shared byte address: XOR bits
    /// `[M, M+B)` with bits `[M+S, M+S+B)` (32B: 4^7, 64B: [4,6)^[7,9),
    /// 128B: [4,7)^[7,10), 128B with 32B atoms: [5,7)^[7,9)).
    fn swizzle(self, address: usize) -> usize {
        let (b, m, s) = match self {
            Self::None => return address,
            Self::B32 => (1, 4, 3),
            Self::B64 => (2, 4, 3),
            Self::B128 => (3, 4, 3),
            Self::B128Atom32 => (2, 5, 2),
        };
        address ^ (((address >> (m + s)) & ((1 << b) - 1)) << m)
    }
}

/// Independent reference for one element of a descriptor-addressed operand,
/// written from the PTX ISA canonical layouts (tcgen05 "Shared Memory Matrix
/// Layout"), in bytes with T = 16 bytes:
///
/// * K-major, none:     ((8,m),(T,2k)) : ((1T,SBO),(1,LBO))
/// * K-major, W-byte:   ((8,m),(T,2k)) : ((W,SBO),(1,T))       W = 32/64/128
/// * MN-major, none:    ((T,1,m),(8,k)) : ((1,T,SBO),(1T,LBO))
/// * MN-major, W-byte:  ((T,W/T,m),(8,k)) : ((1,T,LBO),(W,SBO))
/// * MN-major, 128B with 32B atoms (TF32): as W = 128 with 4 K rows per atom
///
/// followed by the swizzle of the absolute address. `mn` is the M/N index,
/// `k` the K index, both in elements of `element_bytes`.
#[allow(clippy::too_many_arguments)] // a flat reference formula reads best unpacked
fn reference_matrix_address(
    mode: RefMode,
    k_major: bool,
    element_bytes: usize,
    start: usize,
    lbo: usize,
    sbo: usize,
    mn: usize,
    k: usize,
) -> usize {
    let width = mode.width();
    let unswizzled = if k_major {
        let k_byte = k * element_bytes;
        if mode == RefMode::None {
            start + (mn % 8) * 16 + (mn / 8) * sbo + (k_byte / 16) * lbo + k_byte % 16
        } else {
            assert!(k_byte < width, "K-major swizzled reference covers one row");
            start + (mn % 8) * width + (mn / 8) * sbo + k_byte
        }
    } else {
        let mn_byte = mn * element_bytes;
        let k_rows = if mode == RefMode::B128Atom32 { 4 } else { 8 };
        let (mn_outer, k_outer) = if mode == RefMode::None {
            (sbo, lbo)
        } else {
            (lbo, sbo)
        };
        start
            + mn_byte % width
            + (mn_byte / width) * mn_outer
            + (k % k_rows) * width
            + (k / k_rows) * k_outer
    };
    mode.swizzle(unswizzled)
}

fn encode_reference_descriptor(mode: RefMode, start: usize, lbo: usize, sbo: usize) -> u64 {
    assert!(start.is_multiple_of(16) && lbo.is_multiple_of(16) && sbo.is_multiple_of(16));
    ((start as u64 >> 4) & 0x3fff)
        | (((lbo as u64 >> 4) & 0x3fff) << 16)
        | (((sbo as u64 >> 4) & 0x3fff) << 32)
        | (1 << 46)
        | (mode.layout_type() << 61)
}

/// Every (mode x K-/MN-major x element width) the corpus feeds tcgen05.mma —
/// b16 (f16/bf16), 8-bit (fp8 and the 8-bit-container fp6/fp4 forms) and TF32 —
/// over several descriptor starts (period-aligned, a K-step advance inside the
/// pattern, and an unaligned start, whose swizzle still keys on the absolute
/// address), LBO/SBO pairs and window bases, element by element against the
/// independent PTX reference above.
#[test]
fn matrix_byte_offset_matches_ptx_canonical_layouts_exhaustively() {
    let backing = 1 << 16;
    let mut checked = 0_usize;
    for mode in RefMode::ALL {
        let width = mode.width();
        for k_major in [true, false] {
            for element_bytes in [1_usize, 2, 4] {
                // PTX: MN-major TF32 is only the 128B/32B-atom layout, which in
                // turn is TF32-MN-major only.
                let tf32_mn = !k_major && element_bytes == 4;
                if tf32_mn != (mode == RefMode::B128Atom32) {
                    continue;
                }
                let strides: &[(usize, usize)] = match (k_major, mode) {
                    (true, RefMode::None) => &[(128, 256), (2048, 128)],
                    (true, _) => &[(16, 8 * width), (16, 16 * width)],
                    (false, RefMode::None) => &[(128, 256), (512, 2048)],
                    (false, _) => &[(8 * width, 16 * width), (32 * width, 8 * width)],
                };
                let (mn_extent, k_extent) = if k_major {
                    (
                        32,
                        if mode == RefMode::None { 64 } else { width } / element_bytes,
                    )
                } else {
                    (2 * width / element_bytes, 16)
                };
                for start in [0_usize, 1024, 1024 + 32, 2048 + 64, 4096 + 16] {
                    if mode == RefMode::B128Atom32 && start % 32 != 0 {
                        continue; // the decoder requires 32B alignment here
                    }
                    for &(lbo, sbo) in strides {
                        for virtual_base in [0_usize, 0x4000] {
                            let bits =
                                encode_reference_descriptor(mode, start + virtual_base, lbo, sbo);
                            let descriptor = decode_matrix_descriptor(bits).unwrap();
                            let window = whole(backing, virtual_base);
                            for mn in 0..mn_extent {
                                for k in 0..k_extent {
                                    let expected = reference_matrix_address(
                                        mode,
                                        k_major,
                                        element_bytes,
                                        start + virtual_base,
                                        lbo,
                                        sbo,
                                        mn,
                                        k,
                                    ) - virtual_base;
                                    let actual = match element_bytes {
                                        1 => byte8_matrix_byte_offset(
                                            window, descriptor, mn, k, !k_major,
                                        ),
                                        2 => b16_matrix_byte_offset(
                                            window, descriptor, mn, k, !k_major,
                                        ),
                                        _ => matrix_byte_offset(
                                            window,
                                            descriptor,
                                            mn,
                                            k,
                                            !k_major,
                                            element_bytes,
                                        ),
                                    }
                                    .unwrap();
                                    assert_eq!(
                                        actual, expected,
                                        "{mode:?} k_major={k_major} bytes={element_bytes} \
                                         start={start} lbo={lbo} sbo={sbo} base={virtual_base} \
                                         mn={mn} k={k}"
                                    );
                                    checked += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 100_000, "{checked}");
}

/// A buffer placed off its swizzle period (the v2 bug W9's valid-shape ports
/// hit: SWIZZLE_32B operands at shared offset 64) is read by the descriptor
/// with the absolute-address XOR, which differs from a buffer-relative
/// swizzle — so the allocator, not the descriptor math, must align it.
#[test]
fn unaligned_swizzled_base_differs_from_buffer_relative_swizzle() {
    let mode = RefMode::B32;
    let base = 64;
    let descriptor =
        decode_matrix_descriptor(encode_reference_descriptor(mode, base, 16, 256)).unwrap();
    let window = whole(1 << 13, 0);
    let mut differing_rows = Vec::new();
    for row in 0..8 {
        let absolute = b16_matrix_byte_offset(window, descriptor, row, 0, false).unwrap();
        let relative = base + mode.swizzle(row * 32);
        if absolute != relative {
            differing_rows.push(row);
        }
    }
    // Row bit 1 flips K chunk 0 <-> 1 (the observed "k ^ 8 when (i//2)%2 differs").
    assert_eq!(differing_rows, [2, 3, 6, 7]);
}

/// `SharedOffsets::offset` is `Some` only where `shared_byte_offset` is `Ok`
/// with the same value, over random descriptors (every swizzle, LDO 0 and
/// not, absolute LDO), windows (access views, short backings, bases above
/// and below the descriptor address), rows, columns and access sizes; and it
/// serves the ordinary in-bounds case.
#[test]
fn shared_offsets_agree_with_shared_byte_offset() {
    let mut seed = 0x2545_f491_u64;
    let mut next = move |bound: usize| {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((seed >> 33) as usize) % bound
    };
    let (mut served, mut ok_total, mut checked) = (0, 0, 0);
    for _ in 0..4000 {
        let swizzle_bits = next(4);
        let descriptor = MatrixDescriptor {
            start_address: next(1 << 16) & !15,
            leading_byte_offset: [0, 16, 128, 4096][next(4)] + 16 * next(2),
            absolute_leading_address: next(8) == 0,
            stride_byte_offset: [0, 256, 1024, 8192][next(4)],
            swizzle_bits,
            swizzle_atom_bytes: [16, 16, 32, 48][next(4)],
            swizzle_xor_shift: [3, 4, 7][next(3)],
        };
        let backing = 1 << (12 + next(6));
        let view_offset = if next(2) == 0 { 0 } else { next(backing) };
        let source = SharedWindow {
            virtual_base: next(1 << 16),
            view_offset,
            view_len: next(backing - view_offset + 1),
            backing_byte_len: backing,
            access_view: next(2) == 0,
        };
        let offsets = SharedOffsets::new(source, descriptor);
        for _ in 0..32 {
            let row = next(300);
            let column = next(160);
            let access = 1 + next(16);
            let direct = shared_byte_offset(source, descriptor, row, column, access);
            let fast = offsets.offset(row, column, access);
            if let Some(value) = fast {
                assert_eq!(direct.as_ref().ok(), Some(&value), "{descriptor:?} {source:?} row {row} column {column} access {access}");
                served += 1;
            }
            ok_total += usize::from(direct.is_ok());
            checked += 1;
        }
    }
    assert!(checked > 0 && ok_total > 1000, "{ok_total} Ok of {checked}");
    assert!(served * 2 > ok_total, "served {served} of {ok_total} Ok offsets");
}
