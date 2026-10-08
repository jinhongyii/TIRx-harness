// Hand references index fragments by (register, lane) like the PTX tables.
#![allow(clippy::needless_range_loop)]

use std::collections::HashSet;

use super::*;
use crate::arena::ByteSpan;
use crate::dtype::Dtype;
use crate::oplib::{OpErrorKind, TcArch};
use crate::program::{MatrixFmt, MatrixShape, ReduxOp, TcShape};
use numsim_oplib::tcgen05::encode::encode_matrix_descriptor;

// ---------------------------------------------------------------------------
// tcgen05.ld / st
// ---------------------------------------------------------------------------

/// `(lane, register) -> (tmem lane, column)` of an unpacked map.
fn cell(map: &TcgenLdstMap, register: usize, lane: usize) -> (u32, u32) {
    let piece = map.pieces(register, lane)[0];
    assert_eq!((piece.cell_byte, piece.reg_byte, piece.len), (0, 0, 4));
    (piece.tmem_lane, piece.column)
}

/// Every (register, lane) hits a distinct cell of `rows x columns` per
/// `.num` repeat (the shape is a bijection onto its tile).
fn assert_bijection(map: &TcgenLdstMap, base_lane: u32, base_col: u32, rows: u32, cols: u32) {
    let mut seen = HashSet::new();
    for register in 0..map.registers {
        for lane in 0..32 {
            let (l, c) = cell(map, register, lane);
            assert!((base_lane..base_lane + rows).contains(&l), "lane {l}");
            assert!(seen.insert((l, c)), "duplicate cell ({l}, {c})");
            let _ = c - base_col;
        }
    }
    assert_eq!(seen.len() as u32, rows * cols);
}

#[test]
fn ld_st_32x32b_maps_lane_to_row_and_register_to_column() {
    for taddr_lane in [0_u32, 32] {
        let map = tcgen_ldst_map(TcShape::S32x32b, 4, false, 5, taddr_lane << 16 | 10).unwrap();
        assert_eq!(map.registers, 4);
        for register in 0..4 {
            for lane in 0..32 {
                assert_eq!(
                    cell(&map, register, lane),
                    (32 + lane as u32, 10 + register as u32)
                );
            }
        }
    }
    // Warp 1's subpartition is lanes 32..64.
    let error = tcgen_ldst_map(TcShape::S32x32b, 1, false, 1, 64 << 16).unwrap_err();
    assert_eq!(error.kind, OpErrorKind::Invalid);
    assert!(
        tcgen_ldst_map(TcShape::S32x32b, 3, false, 0, 0).is_err(),
        "x3"
    );
    assert!(
        tcgen_ldst_map(TcShape::S16x256b, 64, false, 0, 0).is_err(),
        "16x256b caps at x32"
    );
    assert!(tcgen_ldst_map(TcShape::S32x32b, 1, false, 0, 511).is_ok());
    assert!(
        tcgen_ldst_map(TcShape::S32x32b, 2, false, 0, 511).is_err(),
        "past column 512"
    );
}

#[test]
fn ld_st_16xnb_shapes_follow_the_ptx_fragments() {
    // 16x64b: lane t holds row t/4 + 8*(t&1), column ((t>>1)&1) + 2*reg.
    let map = tcgen_ldst_map(TcShape::S16x64b, 2, false, 0, 4).unwrap();
    for (lane, row, col) in [
        (0, 0, 4),
        (1, 8, 4),
        (2, 0, 5),
        (3, 8, 5),
        (4, 1, 4),
        (31, 15, 5),
    ] {
        assert_eq!(cell(&map, 0, lane), (row, col), "lane {lane}");
    }
    assert_eq!(cell(&map, 1, 2), (0, 7));
    assert_bijection(&map, 0, 4, 16, 4);
    // 16x128b: two registers per repeat, rows split by register parity.
    let map = tcgen_ldst_map(TcShape::S16x128b, 1, false, 0, 0).unwrap();
    assert_eq!(map.registers, 2);
    assert_eq!((cell(&map, 0, 5), cell(&map, 1, 5)), ((1, 1), (9, 1)));
    assert_bijection(&map, 0, 0, 16, 4);
    // 16x256b: four registers per repeat over 8 columns.
    let map = tcgen_ldst_map(TcShape::S16x256b, 2, false, 2, 64 << 16).unwrap();
    assert_eq!(map.registers, 8);
    assert_eq!(cell(&map, 3, 6), (64 + 1 + 8, 1 + 4));
    assert_eq!(cell(&map, 4, 0), (64, 8));
    assert_bijection(&map, 64, 0, 16, 16);
    // 16x32bx2: lanes 16..32 read the second half at taddr + split_off.
    let map = tcgen_ldst_map(TcShape::S16x32bx2 { split_off: 64 }, 2, false, 0, 3).unwrap();
    assert_eq!(cell(&map, 1, 7), (7, 4));
    assert_eq!(cell(&map, 1, 23), (7, 3 + 64 + 1));
}

#[test]
fn pack16_splits_a_register_over_two_columns() {
    let map = tcgen_ldst_map(TcShape::S32x32b, 2, true, 0, 8).unwrap();
    assert_eq!(map.pieces_per_register, 2);
    let pieces = map.pieces(1, 3);
    assert_eq!(
        pieces,
        &[
            TcgenLdstPiece {
                tmem_lane: 3,
                column: 10,
                cell_byte: 0,
                reg_byte: 0,
                len: 2
            },
            TcgenLdstPiece {
                tmem_lane: 3,
                column: 11,
                cell_byte: 0,
                reg_byte: 2,
                len: 2
            },
        ]
    );
    assert_eq!(map.all().len(), 2 * 32 * 2);
}

#[test]
fn ld_variants_red_and_spcompress() {
    assert_eq!(
        tcgen_ld_dst_count(TcShape::S16x256b, 2, false, false, None).unwrap(),
        8
    );
    assert_eq!(
        tcgen_ld_dst_count(TcShape::S32x32b, 4, false, true, None).unwrap(),
        5
    );
    assert_eq!(
        tcgen_ld_dst_count(TcShape::S32x32b, 64, false, false, Some((true, false))).unwrap(),
        34
    );
    assert!(tcgen_ld_dst_count(TcShape::S16x64b, 4, false, true, None).is_err());
    assert!(
        tcgen_ld_dst_count(TcShape::S32x32b, 4, true, true, None).is_err(),
        "red needs unpacked"
    );
    assert!(tcgen_ld_dst_count(TcShape::S32x32b, 2, false, false, Some((true, true))).is_err());

    let f = |v: f32| v.to_bits();
    let abs_max = TcgenLdRed::new(ReduxOp::Max, Dtype::F32, true, false).unwrap();
    assert_eq!(
        tcgen_ld_reduce(abs_max, &[f(-4.0), f(3.0)]).unwrap(),
        f(4.0)
    );
    let nan_min = TcgenLdRed::new(ReduxOp::Min, Dtype::F32, false, true).unwrap();
    assert!(
        f32::from_bits(tcgen_ld_reduce(nan_min, &[f(1.0), f32::NAN.to_bits()]).unwrap()).is_nan()
    );
    let min = TcgenLdRed::new(ReduxOp::Min, Dtype::F32, false, false).unwrap();
    assert_eq!(
        tcgen_ld_reduce(min, &[f(1.0), f32::NAN.to_bits(), f(-2.0)]).unwrap(),
        f(-2.0)
    );
    let s32 = TcgenLdRed::new(ReduxOp::Max, Dtype::S32, false, false).unwrap();
    assert_eq!(
        tcgen_ld_reduce(s32, &[(-3_i32) as u32, 2, (-7_i32) as u32]).unwrap(),
        2
    );
    let u32_min = TcgenLdRed::new(ReduxOp::Min, Dtype::U32, false, false).unwrap();
    assert_eq!(tcgen_ld_reduce(u32_min, &[5, 3, 9]).unwrap(), 3);
    assert!(tcgen_ld_reduce(u32_min, &[5]).is_err());
    assert!(TcgenLdRed::new(ReduxOp::Add, Dtype::U32, false, false).is_err());
    assert!(TcgenLdRed::new(ReduxOp::Max, Dtype::U32, true, false).is_err());
    assert!(TcgenLdRed::new(ReduxOp::Max, Dtype::F16, false, false).is_err());

    // Keep the two largest magnitudes of each group of four.
    let values = [1.0_f32, -5.0, 3.0, 0.5, 2.0, 0.0, -1.0, 8.0].map(f32::to_bits);
    let (out, valid) = tcgen_ld_spcompress(&values, &[true; 8], true, true).unwrap();
    assert_eq!(out[0], (1 | 2 << 2) | (3 << 2) << 4);
    assert_eq!(out[1..], [f(-5.0), f(3.0), f(2.0), f(8.0)]);
    assert!(valid.iter().all(|v| *v));
    let mut partly = [true; 8];
    partly[5] = false;
    let (_, valid) = tcgen_ld_spcompress(&values, &partly, true, true).unwrap();
    assert_eq!(valid, vec![false, true, true, false, false]);
    assert!(tcgen_ld_spcompress(&values[..6], &[true; 6], true, false).is_err());
}

// ---------------------------------------------------------------------------
// tcgen05.cp
// ---------------------------------------------------------------------------

/// K-major no-swizzle descriptor at 0x400: 16B atoms, LBO 128 (next atom),
/// SBO 256 (next 8-row group).
fn cp_desc() -> u64 {
    encode_matrix_descriptor(0x400, 128 >> 4, 256 >> 4, 0)
}

fn src(row: usize, word: usize) -> u64 {
    0x400
        + (row % 8) as u64 * 16
        + (row / 8) as u64 * 256
        + (word / 4) as u64 * 128
        + (word % 4) as u64 * 4
}

#[test]
fn cp_128x256b_walks_the_descriptor() {
    let plan = tcgen_cp_plan(128, 256, 0, 0, cp_desc(), 20, 1, TcArch::Sm100).unwrap();
    assert_eq!(plan.words.len(), 128 * 8);
    assert_eq!((plan.lane_end, plan.column_end), (128, 28));
    for (index, word) in plan.words.iter().enumerate() {
        let (row, w) = (index / 8, index % 8);
        assert_eq!(word.src, ByteSpan::new(src(row, w), 4));
        assert_eq!(
            (word.lanes(), word.column),
            (&[row as u32][..], 20 + w as u32)
        );
    }
    let (spans, cells) = plan.pairs();
    assert_eq!((spans.len(), cells.len()), (1024, 1024));
    // Swizzled descriptors XOR the 16B atom index with address bits 7.. (128B).
    let swizzled = encode_matrix_descriptor(0x400, 1, 1024 >> 4, 3);
    let plan = tcgen_cp_plan(128, 128, 0, 0, swizzled, 0, 2, TcArch::Sm100).unwrap();
    // Row 1, word 0: unswizzled 0x400 + 128 -> atom 0 ^ 1.
    assert_eq!(plan.words[4].src, ByteSpan::new(0x400 + 128 + 16, 4));
}

#[test]
fn cp_multicast_replicates_rows_across_warps() {
    let lanes = |rows, multicast, row: usize| {
        let plan = tcgen_cp_plan(rows, 128, multicast, 0, cp_desc(), 0, 1, TcArch::Sm100).unwrap();
        assert_eq!(plan.words.len(), rows as usize * 4);
        plan.words[row * 4].lanes().to_vec()
    };
    assert_eq!(lanes(32, 3, 5), vec![5, 37, 69, 101]);
    assert_eq!(lanes(64, 1, 3), vec![3, 67]);
    assert_eq!(lanes(64, 1, 40), vec![40, 104]);
    assert_eq!(lanes(64, 2, 3), vec![3, 35]);
    assert_eq!(lanes(64, 2, 40), vec![72, 104]);
    let plan = tcgen_cp_plan(4, 256, 0, 0, cp_desc(), 0, 1, TcArch::Sm100).unwrap();
    assert_eq!(plan.words[3 * 8].lanes(), &[96]);
    assert_eq!(plan.pairs().0.len(), 32);
    let multicast = tcgen_cp_plan(32, 128, 3, 0, cp_desc(), 0, 1, TcArch::Sm100).unwrap();
    assert_eq!(multicast.pairs().0.len(), 32 * 4 * 4);
    // Illegal shape/multicast pairs, lane overflow, cta_group, decompression.
    for (rows, bits, multicast) in [(32, 128, 0), (128, 128, 3), (64, 128, 0), (64, 256, 1)] {
        assert!(tcgen_cp_plan(rows, bits, multicast, 0, cp_desc(), 0, 1, TcArch::Sm100).is_err());
    }
    assert!(tcgen_cp_plan(128, 128, 0, 0, cp_desc(), 1 << 16, 1, TcArch::Sm100).is_err());
    assert!(tcgen_cp_plan(128, 128, 0, 5, cp_desc(), 0, 1, TcArch::Sm100).is_err());
    assert!(tcgen_cp_plan(128, 128, 0, 0, cp_desc(), 0, 3, TcArch::Sm100).is_err());
    assert!(
        tcgen_cp_plan(128, 128, 0, 0, 0, 0, 1, TcArch::Sm100).is_err(),
        "descriptor version"
    );
}

#[test]
fn cp_decompression_reads_packed_sources() {
    // b6x16_p32: 3-byte sources at 16-byte atoms, bytes 3*w of each atom.
    let plan = tcgen_cp_plan(128, 128, 0, 6, cp_desc(), 0, 1, TcArch::Sm100).unwrap();
    assert_eq!(plan.words[1].src, ByteSpan::new(0x400 + 3, 3));
    assert_eq!(plan.decompress_bits, 6);
    // b4x16_p64: 2-byte sources.
    let plan = tcgen_cp_plan(128, 128, 0, 4, cp_desc(), 0, 1, TcArch::Sm100).unwrap();
    assert_eq!(plan.words[2].src, ByteSpan::new(0x400 + 4, 2));
    // Decoding: b6 packs four 6-bit codes little-endian; b4 nibbles << 2.
    let b6 = 0x3f_u32 | 0x15 << 6 | 0x2a << 12 | 0x01 << 18;
    assert_eq!(
        tcgen_cp_decode(&b6.to_le_bytes()[..3], 6).unwrap(),
        [0x3f, 0x15, 0x2a, 0x01]
    );
    assert_eq!(
        tcgen_cp_decode(&[0x21, 0xf3], 4).unwrap(),
        [0x04, 0x08, 0x0c, 0x3c]
    );
    assert_eq!(tcgen_cp_decode(&[1, 2, 3, 4], 0).unwrap(), [1, 2, 3, 4]);
    assert!(tcgen_cp_decode(&[1, 2, 3], 4).is_err(), "width mismatch");
}

// ---------------------------------------------------------------------------
// ldmatrix / stmatrix
// ---------------------------------------------------------------------------

/// 32 provider rows of 16 bytes; lane p's row pointer is `p * 16`.
fn memory() -> Vec<u8> {
    (0..32 * 16).map(|i| (i * 7 + 3) as u8).collect()
}

fn reader(
    memory: &[u8],
) -> impl FnMut(usize, usize, usize) -> crate::oplib::OpResult<Vec<u8>> + '_ {
    move |provider, delta, len| {
        Ok(memory[provider * 16 + delta..provider * 16 + delta + len].to_vec())
    }
}

#[test]
fn ldmatrix_b16_m8n8_plain_and_transposed() {
    let memory = memory();
    let element = |m: usize, r: usize, c: usize| {
        let at = m * 128 + r * 16 + c * 2;
        u16::from_le_bytes([memory[at], memory[at + 1]]) as u32
    };
    for trans in [false, true] {
        let plan = ldmatrix_plan(MatrixShape::M8N8, 2, trans, MatrixFmt::B16).unwrap();
        assert_eq!(
            (plan.registers, plan.providers, plan.row_bytes),
            (2, 16, 16)
        );
        let fragments = ldmatrix_fragments(&plan, |p| Ok(p as u64 * 16), reader(&memory)).unwrap();
        for m in 0..2 {
            for lane in 0..32 {
                let (r, c) = (lane / 4, lane % 4 * 2);
                let expect = if trans {
                    element(m, c, r) | element(m, c + 1, r) << 16
                } else {
                    element(m, r, c) | element(m, r, c + 1) << 16
                };
                assert_eq!(fragments[m][lane], expect);
            }
        }
    }
}

#[test]
fn ldmatrix_b8_family() {
    let memory = memory();
    // m16n16.b8.trans x1: two registers; element e of register g in lane t
    // is row (g/2)*16 + (t%4)*4 + e, column (g%2)*8 + t/4.
    let plan = ldmatrix_plan(MatrixShape::M16N16, 1, true, MatrixFmt::B8).unwrap();
    assert_eq!(
        (plan.registers, plan.providers, plan.source_bits),
        (2, 16, 8)
    );
    let fragments = ldmatrix_fragments(&plan, |p| Ok(p as u64 * 16), reader(&memory)).unwrap();
    for g in 0..2 {
        for t in 0..32 {
            let expect = (0..4).fold(0_u32, |acc, e| {
                let row = (g / 2) * 16 + (t % 4) * 4 + e;
                acc | u32::from(memory[row * 16 + (g % 2) * 8 + t / 4]) << (8 * e)
            });
            assert_eq!(fragments[g][t], expect, "register {g} lane {t}");
        }
    }
    // m8n16.b8x16.b6x16_p32 x1: 6-bit codes packed in 12-byte rows.
    let code = |row: usize, k: usize| ((row * 5 + k * 3) % 64) as u32;
    let mut packed = vec![0_u8; 32 * 16];
    for row in 0..8 {
        let bits = (0..16).fold(0_u128, |acc, k| acc | u128::from(code(row, k)) << (6 * k));
        packed[row * 16..row * 16 + 12].copy_from_slice(&bits.to_le_bytes()[..12]);
    }
    let plan = ldmatrix_plan(MatrixShape::M8N16, 1, false, MatrixFmt::B6x16P32).unwrap();
    assert_eq!(plan.row_bytes, 12);
    let fragments = ldmatrix_fragments(&plan, |p| Ok(p as u64 * 16), reader(&packed)).unwrap();
    for t in 0..32 {
        let expect = (0..4).fold(0_u32, |acc, e| acc | code(t / 4, t % 4 * 4 + e) << (8 * e));
        assert_eq!(fragments[0][t], expect);
    }
    // Unaligned b8 rows are invalid.
    assert!(ldmatrix_fragments(&plan, |p| Ok(p as u64 * 16 + 4), reader(&packed)).is_err());
    // m8n16.s8.s4: nibbles sign-extend into bytes.
    let mut nibbles = vec![0_u8; 32 * 16];
    nibbles[0] = 0xf7; // k0 = 7, k1 = -1
    let plan = ldmatrix_plan(MatrixShape::M8N16, 1, false, MatrixFmt::S8S4).unwrap();
    let fragments = ldmatrix_fragments(&plan, |p| Ok(p as u64 * 16), reader(&nibbles)).unwrap();
    assert_eq!(fragments[0][0] & 0xffff, 0xff07);
    // Footprint form.
    assert_eq!(plan.accesses(5, |p| Ok(p as u64 * 16)).unwrap().len(), 4);
    let b16 = ldmatrix_plan(MatrixShape::M8N8, 4, true, MatrixFmt::B16).unwrap();
    assert_eq!(b16.accesses(5, |_| Ok(0)).unwrap().len(), 8);
    // Illegal forms.
    for (shape, num, trans, fmt) in [
        (MatrixShape::M16N16, 4, true, MatrixFmt::B8),
        (MatrixShape::M16N16, 1, false, MatrixFmt::B8),
        (MatrixShape::M8N16, 1, true, MatrixFmt::S8S4),
        (MatrixShape::M8N8, 3, false, MatrixFmt::B16),
        (MatrixShape::M16N8, 1, false, MatrixFmt::B16),
    ] {
        assert!(
            ldmatrix_plan(shape, num, trans, fmt).is_err(),
            "{shape:?} {fmt:?}"
        );
    }
}

#[test]
fn stmatrix_inverts_ldmatrix_and_writes_m16n8_b8() {
    let memory = memory();
    for trans in [false, true] {
        let load = ldmatrix_plan(MatrixShape::M8N8, 4, trans, MatrixFmt::B16).unwrap();
        let fragments = ldmatrix_fragments(&load, |p| Ok(p as u64 * 16), reader(&memory)).unwrap();
        let store = stmatrix_plan(MatrixShape::M8N8, 4, trans).unwrap();
        let writes = stmatrix_writes(&store, &fragments, |p| Ok(p as u64 * 16)).unwrap();
        let mut written = vec![0_u8; 32 * 16];
        for (provider, delta, bytes) in writes {
            written[provider * 16 + delta..provider * 16 + delta + bytes.len()]
                .copy_from_slice(&bytes);
        }
        assert_eq!(written, memory);
    }
    let store = stmatrix_plan(MatrixShape::M16N8, 1, true).unwrap();
    let writes = stmatrix_writes(&store, &[[0x4433_2211_u32; 32]], |p| Ok(p as u64 * 16)).unwrap();
    assert_eq!(writes.len(), 128);
    assert_eq!(
        writes[..3],
        [(0, 0, vec![0x11]), (1, 0, vec![0x22]), (0, 8, vec![0x33])]
    );
    let error = stmatrix_writes(&store, &[[0; 32]], |p| Ok(p as u64 * 8)).unwrap_err();
    assert!(error.message.contains("16-byte alignment"));
    assert!(
        stmatrix_writes(&store, &[[0; 32], [0; 32]], |_| Ok(0)).is_err(),
        "register count"
    );
    assert!(stmatrix_plan(MatrixShape::M16N8, 1, false).is_err());
    assert!(stmatrix_plan(MatrixShape::M16N16, 1, true).is_err());
    // Closure errors keep their kind.
    let error = stmatrix_writes(
        &stmatrix_plan(MatrixShape::M8N8, 1, false).unwrap(),
        &[[0; 32]],
        |_| Err(crate::oplib::OpError::unsupported("remote window")),
    )
    .unwrap_err();
    assert_eq!(error.kind, OpErrorKind::Unsupported);
}
