use super::super::{
    tma_plan, tma_plan_dir, tma_tf32_round, Im2colBox, OpErrorKind, TensorMapDesc, TmaFill,
    TmaPlan, TmaPlanDir,
};
use super::L2_PROMOTION_BYTE;
use crate::arena::ByteSpan;
use crate::dtype::Dtype;
use crate::program::{TmaMode, TmapField};
use numsim_oplib::tma::{shared_byte_offset, SwizzleAtomicity};

const VA: u64 = 0x7f00_0000_1000;

fn desc2d(elem: Dtype, dims: [u64; 2], stride: u64, boxes: [u32; 2], swizzle: u8) -> TensorMapDesc {
    TensorMapDesc {
        global_address: VA,
        rank: 2,
        elem: Some(elem),
        global_dim: [dims[0], dims[1], 1, 1, 1],
        global_stride: [stride, 0, 0, 0, 0],
        box_dim: [boxes[0], boxes[1], 1, 1, 1],
        element_stride: [1; 5],
        interleave: 0,
        swizzle,
        l2_promotion: 0,
        oob_fill: 0,
        ..TensorMapDesc::default()
    }
}

/// Byte-level pairing of a plan: (global VA, smem offset) per copied byte.
fn pairs(plan: &TmaPlan) -> Vec<(u64, u64)> {
    let flat = |spans: &[ByteSpan]| {
        spans
            .iter()
            .flat_map(|s| s.start..s.end())
            .collect::<Vec<_>>()
    };
    let global = flat(&plan.global);
    let smem = flat(&plan.smem);
    assert_eq!(
        global.len(),
        smem.len(),
        "matched spans must have equal totals"
    );
    global.into_iter().zip(smem).collect()
}

#[test]
fn encode_decode_round_trip() {
    let mut desc = desc2d(Dtype::BF16, [64, 32], 128, [64, 8], 3);
    desc.l2_promotion = 2;
    desc.oob_fill = 1;
    let bytes = desc.encode();
    assert_eq!(bytes[L2_PROMOTION_BYTE], 2);
    assert_eq!(TensorMapDesc::decode(&bytes).unwrap(), desc);

    for (swizzle, elem) in [
        (0, Dtype::U8),
        (1, Dtype::F16),
        (2, Dtype::F32),
        (4, Dtype::E4M3),
        (5, Dtype::S8),
        (6, Dtype::F64),
        (7, Dtype::U16),
    ] {
        let desc = desc2d(elem, [256, 7], 512, [16, 3], swizzle);
        assert_eq!(
            TensorMapDesc::decode(&desc.encode()).unwrap(),
            desc,
            "swizzle {swizzle}"
        );
    }
    let mut desc5 = TensorMapDesc {
        global_address: VA,
        rank: 5,
        elem: Some(Dtype::S32),
        global_dim: [8, 3, 4, 5, 6],
        global_stride: [32, 96, 384, 1920, 0],
        box_dim: [8, 2, 2, 2, 2],
        element_stride: [1, 2, 1, 1, 3],
        interleave: 1,
        swizzle: 0,
        l2_promotion: 4,
        oob_fill: 0,
        ..TensorMapDesc::default()
    };
    assert_eq!(TensorMapDesc::decode(&desc5.encode()).unwrap(), desc5);
    // Unused axes with zero fillers encode as the canonical 1.
    desc5.rank = 1;
    desc5.interleave = 0;
    let mut sparse = desc5.clone();
    sparse.global_dim[4] = 0;
    sparse.box_dim[4] = 0;
    sparse.element_stride[4] = 0;
    let decoded = TensorMapDesc::decode(&sparse.encode()).unwrap();
    assert_eq!(
        (
            decoded.global_dim[4],
            decoded.box_dim[4],
            decoded.element_stride[4]
        ),
        (1, 1, 1)
    );
}

#[test]
fn decode_validates_and_unencodable_maps_fail_closed() {
    let good = desc2d(Dtype::F16, [64, 32], 128, [64, 8], 0).encode();
    let mut bad_magic = good;
    bad_magic[63] = 0;
    assert_eq!(
        TensorMapDesc::decode(&bad_magic).unwrap_err().kind,
        OpErrorKind::Invalid
    );
    let mut bad_rank = good;
    bad_rank[59] &= !0b111;
    assert_eq!(
        TensorMapDesc::decode(&bad_rank).unwrap_err().kind,
        OpErrorKind::Invalid
    );
    let mut bad_dtype = good;
    bad_dtype[59] = (bad_dtype[59] & 0b111) | (31 << 3);
    assert_eq!(
        TensorMapDesc::decode(&bad_dtype).unwrap_err().kind,
        OpErrorKind::Invalid
    );
    assert!(TensorMapDesc::decode(&[0; 128]).is_err());

    // Out-of-range or unmappable fields encode to zeros, which never decode.
    let mut zero_stride = desc2d(Dtype::F16, [64, 32], 129, [64, 8], 0);
    assert_eq!(zero_stride.encode(), [0; 128]);
    zero_stride.global_stride[0] = 128;
    zero_stride.elem = Some(Dtype::E5M2);
    assert_eq!(zero_stride.encode(), [0; 128]);
    zero_stride.elem = None;
    assert_eq!(zero_stride.encode(), [0; 128]);
}

#[test]
fn replace_follows_tensormap_replace() {
    let mut desc = desc2d(Dtype::F16, [64, 32], 128, [64, 8], 3);
    desc.replace(TmapField::GlobalDim, Some(1), 77).unwrap();
    assert_eq!(desc.global_dim[1], 77);
    desc.replace(TmapField::GlobalStride, Some(0), 256).unwrap();
    assert_eq!(desc.global_stride[0], 256);
    desc.replace(TmapField::BoxDim, Some(1), 16).unwrap();
    assert_eq!(desc.box_dim[1], 16);
    desc.replace(TmapField::ElementStride, Some(1), 2).unwrap();
    assert_eq!(desc.element_stride[1], 2);
    desc.replace(TmapField::Rank, None, 2).unwrap();
    assert_eq!(desc.rank, 3);
    desc.replace(TmapField::ElemType, None, 10).unwrap();
    assert_eq!(desc.elem, Some(Dtype::BF16));
    desc.replace(TmapField::SwizzleAtomicity, None, 1).unwrap();
    assert_eq!(desc.swizzle, 4);
    desc.replace(TmapField::SwizzleAtomicity, None, 0).unwrap();
    desc.replace(TmapField::SwizzleMode, None, 1).unwrap();
    assert_eq!(desc.swizzle, 1);
    desc.replace(TmapField::SwizzleMode, None, 4).unwrap();
    assert_eq!(desc.swizzle, 7);
    desc.replace(TmapField::FillMode, None, 1).unwrap();
    assert_eq!(desc.oob_fill, 1);
    desc.replace(TmapField::InterleaveLayout, None, 2).unwrap();
    assert_eq!(desc.interleave, 2);
    desc.replace(TmapField::GlobalAddress, None, VA + 4096)
        .unwrap();
    assert_eq!(desc.global_address, VA + 4096);

    assert!(desc
        .replace(TmapField::GlobalAddress, None, VA + 8)
        .is_err());
    assert!(desc.replace(TmapField::BoxDim, None, 4).is_err());
    assert!(desc.replace(TmapField::BoxDim, Some(0), 257).is_err());
    assert!(desc.replace(TmapField::FillMode, None, 2).is_err());
    assert_eq!(
        desc.replace(TmapField::ElemType, None, 13)
            .unwrap_err()
            .kind,
        OpErrorKind::Unsupported
    );
}

#[test]
fn tiled_box_fully_in_bounds() {
    let desc = desc2d(Dtype::F16, [64, 32], 128, [16, 4], 0);
    let plan = tma_plan(&desc, TmaMode::Tile, &[8, 2], &[], 256).unwrap();
    assert_eq!(plan.bytes, 16 * 4 * 2);
    assert!(plan.smem_oob_fill.is_empty());
    let expected_global = (0..4)
        .map(|row| ByteSpan::new(VA + (2 + row) * 128 + 16, 32))
        .collect::<Vec<_>>();
    assert_eq!(plan.global, expected_global);
    assert_eq!(plan.smem, vec![ByteSpan::new(256, 128)]);
}

#[test]
fn tiled_box_partially_out_of_bounds() {
    let desc = desc2d(Dtype::F16, [64, 32], 128, [16, 4], 0);
    let plan = tma_plan(&desc, TmaMode::Tile, &[56, 30], &[], 0).unwrap();
    assert_eq!(plan.bytes, 128);
    assert_eq!(
        plan.global,
        vec![
            ByteSpan::new(VA + 30 * 128 + 112, 16),
            ByteSpan::new(VA + 31 * 128 + 112, 16)
        ]
    );
    assert_eq!(plan.smem, vec![ByteSpan::new(0, 16), ByteSpan::new(32, 16)]);
    assert_eq!(
        plan.smem_oob_fill,
        vec![ByteSpan::new(16, 16), ByteSpan::new(48, 80)]
    );
    // Fully outside: everything is fill.
    let plan = tma_plan(&desc, TmaMode::Tile, &[-16, 0], &[], 0).unwrap();
    assert!(plan.global.is_empty() && plan.smem.is_empty());
    assert_eq!(plan.smem_oob_fill, vec![ByteSpan::new(0, 128)]);
    assert_eq!(
        (plan.fill, plan.fill_pattern.is_empty()),
        (TmaFill::Zero, true)
    );
    // NaN-request-zero-FMA fill: every 16 bits of an OOB element read 0x7ff7.
    let mut nan = desc.clone();
    nan.oob_fill = 1;
    let plan = tma_plan(&nan, TmaMode::Tile, &[56, 30], &[], 0).unwrap();
    assert_eq!(plan.fill, TmaFill::NanRequestZeroFma);
    assert_eq!(plan.fill_pattern, vec![0xf7, 0x7f]);
    assert_eq!(
        plan.smem_oob_fill,
        vec![ByteSpan::new(16, 16), ByteSpan::new(48, 80)]
    );
    assert!(!plan.tf32_round);
}

#[test]
fn swizzled_128b_box_matches_tma_swizzle() {
    let desc = desc2d(Dtype::BF16, [128, 16], 256, [64, 8], 3);
    let smem_offset = 1024 + 128 * 3; // non-zero phase of the 1024B pattern
    let plan = tma_plan(&desc, TmaMode::Tile, &[64, 4], &[], smem_offset).unwrap();
    assert_eq!(plan.bytes, 64 * 8 * 2);
    let pairs = pairs(&plan);
    assert_eq!(pairs.len(), 1024);
    for (global, smem) in pairs {
        let relative = global - VA;
        let row = (relative / 256) as usize - 4;
        let byte = (relative % 256) as usize - 128;
        let expected = shared_byte_offset(
            Some(128),
            SwizzleAtomicity::B16,
            row,
            byte,
            128,
            smem_offset as usize,
        )
        .unwrap();
        assert_eq!(smem, smem_offset + expected as u64, "row {row} byte {byte}");
    }
    // The pattern really permutes 16B chunks (row 1 chunk 0 is not at 128).
    assert_ne!(
        shared_byte_offset(Some(128), SwizzleAtomicity::B16, 1, 0, 128, 1024).unwrap(),
        128
    );
}

#[test]
fn im2col_pixels_walk_the_spatial_axis() {
    // (C=8, W=10, N=2) f16, box 8 channels x 4 pixels.
    let desc = TensorMapDesc {
        global_address: VA,
        rank: 3,
        elem: Some(Dtype::F16),
        global_dim: [8, 10, 2, 1, 1],
        global_stride: [16, 160, 0, 0, 0],
        box_dim: [8, 4, 1, 1, 1],
        element_stride: [1; 5],
        ..TensorMapDesc::default()
    };
    let plan = tma_plan(&desc, TmaMode::Im2col, &[0, 2, 1], &[1], 64).unwrap();
    assert_eq!(plan.bytes, 4 * 16);
    assert_eq!(plan.global, vec![ByteSpan::new(VA + 160 + 3 * 16, 64)]);
    assert_eq!(plan.smem, vec![ByteSpan::new(64, 64)]);
    // The pixel walk wraps W into the next image (zero bounding box):
    // pixels (w=8,n=0), (9,0), (0,1), (1,1) are contiguous in this packing.
    let plan = tma_plan(&desc, TmaMode::Im2col, &[0, 8, 0], &[0], 0).unwrap();
    assert_eq!(plan.global, vec![ByteSpan::new(VA + 8 * 16, 64)]);
    // With offset 1 the pixels read w=9, w=10 (outside W: zero fill), then
    // n=1 w=1..2.
    let plan = tma_plan(&desc, TmaMode::Im2col, &[0, 8, 0], &[1], 0).unwrap();
    assert_eq!(
        plan.global,
        vec![
            ByteSpan::new(VA + 9 * 16, 16),
            ByteSpan::new(VA + 160 + 16, 32)
        ]
    );
    assert_eq!(plan.smem, vec![ByteSpan::new(0, 16), ByteSpan::new(32, 32)]);
    assert_eq!(plan.smem_oob_fill, vec![ByteSpan::new(16, 16)]);
}

#[test]
fn invalid_modes_and_tf32_rounding() {
    let desc = desc2d(Dtype::F16, [64, 32], 128, [16, 4], 0);
    // scatter4 needs a one-row box; gather4 is load-only, scatter4 store-only.
    assert_eq!(
        tma_plan(&desc, TmaMode::TileScatter4, &[0, 0, 1, 2, 3], &[], 0)
            .unwrap_err()
            .kind,
        OpErrorKind::Invalid
    );
    let row_map = desc2d(Dtype::U8, [64, 32], 64, [32, 1], 0);
    for (dir, mode) in [
        (TmaPlanDir::Store, TmaMode::TileGather4),
        (TmaPlanDir::Load, TmaMode::TileScatter4),
        (TmaPlanDir::Load, TmaMode::Im2colNoOffs),
    ] {
        let error = tma_plan_dir(&row_map, dir, mode, &[0, 0, 1, 2, 3], &[], 0).unwrap_err();
        assert_eq!(error.kind, OpErrorKind::Invalid, "{dir:?} {mode:?}");
    }
    assert_eq!(
        tma_plan(&desc, TmaMode::Tile, &[0], &[], 0)
            .unwrap_err()
            .kind,
        OpErrorKind::Invalid
    );
    // TF32 loads round on landing; stores copy bytes.
    let tf32 = desc2d(Dtype::TF32, [64, 32], 256, [16, 4], 0);
    assert!(
        tma_plan(&tf32, TmaMode::Tile, &[0, 0], &[], 0)
            .unwrap()
            .tf32_round
    );
    assert!(
        !tma_plan_dir(&tf32, TmaPlanDir::Store, TmaMode::Tile, &[0, 0], &[], 0)
            .unwrap()
            .tf32_round
    );
    assert_eq!(
        tma_tf32_round(0x3f80_1fff),
        0x3f80_2000,
        "RNA to 10 mantissa bits"
    );
    assert_eq!(
        tma_tf32_round(0x7fc0_0001),
        0x7fff_e000,
        "NaN canonicalizes"
    );
    let mut misaligned = desc.clone();
    misaligned.global_address += 8;
    assert_eq!(
        tma_plan(&misaligned, TmaMode::Tile, &[0, 0], &[], 0)
            .unwrap_err()
            .kind,
        OpErrorKind::Invalid
    );
}

#[test]
fn tiled_store_skips_oob_and_rejects_negative_coords() {
    let desc = desc2d(Dtype::F16, [64, 32], 128, [16, 4], 0);
    let plan = tma_plan_dir(&desc, TmaPlanDir::Store, TmaMode::Tile, &[56, 30], &[], 512).unwrap();
    assert_eq!(
        plan.global,
        vec![
            ByteSpan::new(VA + 30 * 128 + 112, 16),
            ByteSpan::new(VA + 31 * 128 + 112, 16)
        ]
    );
    assert_eq!(
        plan.smem,
        vec![ByteSpan::new(512, 16), ByteSpan::new(512 + 32, 16)]
    );
    assert!(plan.smem_oob_fill.is_empty());
    assert_eq!(plan.bytes, 128);
    assert_eq!(
        tma_plan_dir(&desc, TmaPlanDir::Store, TmaMode::Tile, &[-8, 0], &[], 0)
            .unwrap_err()
            .kind,
        OpErrorKind::Invalid
    );
    // 8B-flip atomicity is load-only.
    let mut flip = desc2d(Dtype::F16, [64, 32], 128, [64, 4], 5);
    assert!(tma_plan(&flip, TmaMode::Tile, &[0, 0], &[], 0).is_ok());
    assert!(tma_plan_dir(&flip, TmaPlanDir::Store, TmaMode::Tile, &[0, 0], &[], 0).is_err());
    flip.swizzle = 3;
    assert!(tma_plan_dir(&flip, TmaPlanDir::Store, TmaMode::Tile, &[0, 0], &[], 0).is_ok());
}

#[test]
fn scatter4_store_mirrors_gather4() {
    let desc = desc2d(Dtype::U8, [64, 32], 64, [32, 1], 0);
    let coords = [16, 3, 9, 1, 31];
    let store = tma_plan_dir(
        &desc,
        TmaPlanDir::Store,
        TmaMode::TileScatter4,
        &coords,
        &[],
        0,
    )
    .unwrap();
    let load = tma_plan(&desc, TmaMode::TileGather4, &coords, &[], 0).unwrap();
    assert_eq!(
        (store.global.clone(), store.smem.clone(), store.bytes),
        (load.global, load.smem, load.bytes)
    );
    // The legacy wrapper plans scatter4 as a store.
    assert_eq!(
        tma_plan(&desc, TmaMode::TileScatter4, &coords, &[], 0).unwrap(),
        store
    );
    // OOB rows are skipped; negative rows are invalid.
    let partial = tma_plan_dir(
        &desc,
        TmaPlanDir::Store,
        TmaMode::TileScatter4,
        &[48, 0, 40, 2, 3],
        &[],
        0,
    )
    .unwrap();
    assert_eq!(
        partial.global,
        vec![
            ByteSpan::new(VA + 48, 16),
            ByteSpan::new(VA + 2 * 64 + 48, 16),
            ByteSpan::new(VA + 3 * 64 + 48, 16)
        ]
    );
    assert_eq!(
        partial.smem,
        vec![
            ByteSpan::new(0, 16),
            ByteSpan::new(64, 16),
            ByteSpan::new(96, 16)
        ]
    );
    assert!(tma_plan_dir(
        &desc,
        TmaPlanDir::Store,
        TmaMode::TileScatter4,
        &[0, -1, 0, 0, 0],
        &[],
        0
    )
    .is_err());
}

#[test]
fn im2col_corners_round_trip_and_bound_the_walk() {
    // (C=8, W=10, N=2) f16, 8 channels x 4 pixels, W padded by one on each side.
    let desc = TensorMapDesc {
        global_address: VA,
        rank: 3,
        elem: Some(Dtype::F16),
        global_dim: [8, 10, 2, 1, 1],
        global_stride: [16, 160, 0, 0, 0],
        box_dim: [8, 4, 1, 1, 1],
        element_stride: [1; 5],
        im2col: Some(Im2colBox {
            lower: [-1, 0, 0],
            upper: [1, 0, 0],
            wide: false,
        }),
        ..TensorMapDesc::default()
    };
    let bytes = desc.try_encode().unwrap();
    assert_eq!(TensorMapDesc::decode(&bytes).unwrap(), desc);
    // Start at w = -1 (padding: zero fill), then w = 0..2.
    let plan = tma_plan(&desc, TmaMode::Im2col, &[0, -1, 0], &[0], 0).unwrap();
    assert_eq!(plan.global, vec![ByteSpan::new(VA, 48)]);
    assert_eq!(plan.smem, vec![ByteSpan::new(16, 48)]);
    assert_eq!(plan.smem_oob_fill, vec![ByteSpan::new(0, 16)]);
    // The walk wraps at w = W + upper = 11 back to lower = -1 of image 1.
    let plan = tma_plan(&desc, TmaMode::Im2col, &[0, 9, 0], &[0], 0).unwrap();
    // Pixels: w=9, w=10 (pad), image 1 w=-1 (pad), w=0; (0,9) and (1,0) are
    // adjacent in global memory.
    assert_eq!(plan.global, vec![ByteSpan::new(VA + 9 * 16, 32)]);
    assert_eq!(plan.smem, vec![ByteSpan::new(0, 16), ByteSpan::new(48, 16)]);
    assert_eq!(plan.smem_oob_fill, vec![ByteSpan::new(16, 32)]);
    // A wide instruction on a non-wide map is invalid; a tiled one too.
    assert!(tma_plan(&desc, TmaMode::Im2colW, &[0, 0, 0], &[0, 0], 0).is_err());
    assert!(tma_plan(&desc, TmaMode::Tile, &[0, 0, 0], &[], 0).is_err());
    // im2col_no_offs store over the same walk.
    let mut inside = desc.clone();
    inside.im2col = Some(Im2colBox::default());
    let store = tma_plan_dir(
        &inside,
        TmaPlanDir::Store,
        TmaMode::Im2colNoOffs,
        &[0, 2, 1],
        &[],
        0,
    )
    .unwrap();
    assert_eq!(store.global, vec![ByteSpan::new(VA + 160 + 32, 64)]);
    assert_eq!(store.smem, vec![ByteSpan::new(0, 64)]);
}

#[test]
fn swizzle_atomicity_intermediate_states_and_try_encode() {
    let mut desc = desc2d(Dtype::F16, [64, 32], 128, [64, 8], 4);
    // 128B/32B atoms -> 64B width keeps 32B atomicity (invalid until fixed).
    desc.replace(TmapField::SwizzleMode, None, 2).unwrap();
    assert_eq!((desc.swizzle, desc.swizzle_atomicity), (2, 1));
    let bytes = desc.encode();
    assert_eq!(TensorMapDesc::decode(&bytes).unwrap(), desc);
    assert!(tma_plan(&desc, TmaMode::Tile, &[0, 0], &[], 0).is_err());
    desc.replace(TmapField::SwizzleAtomicity, None, 0).unwrap();
    assert_eq!((desc.swizzle, desc.swizzle_atomicity), (2, 0));
    desc.replace(TmapField::SwizzleMode, None, 3).unwrap();
    desc.replace(TmapField::SwizzleAtomicity, None, 3).unwrap();
    assert_eq!((desc.swizzle, desc.swizzle_atomicity), (6, 0));
    // Conflicting explicit atomicity on a 4..6 code, and range errors.
    desc.swizzle_atomicity = 1;
    assert_eq!(desc.try_encode().unwrap_err().kind, OpErrorKind::Invalid);
    assert_eq!(desc.encode(), [0; 128]);
    let mut wide = desc2d(Dtype::F16, [64, 32], 129, [64, 8], 3);
    assert!(wide.try_encode().is_err());
    wide.global_stride[0] = 128;
    assert!(wide.try_encode().is_ok());
}

#[test]
fn gather4_reads_four_rows() {
    let desc = desc2d(Dtype::U8, [64, 32], 64, [32, 1], 0);
    let plan = tma_plan(&desc, TmaMode::TileGather4, &[16, 3, 9, 1, 31], &[], 0).unwrap();
    assert_eq!(plan.bytes, 128);
    let rows = [3_u64, 9, 1, 31];
    assert_eq!(
        plan.global,
        rows.iter()
            .map(|r| ByteSpan::new(VA + r * 64 + 16, 32))
            .collect::<Vec<_>>()
    );
    assert_eq!(plan.smem, vec![ByteSpan::new(0, 128)]);
}

#[test]
fn fp4_padded_and_u6_element_types_round_trip() {
    let mut ends = Vec::new();
    for (elem, padded) in [(Dtype::E2M1, false), (Dtype::E2M1, true), (Dtype::U6, false)] {
        let mut d = desc2d(elem, [256, 4], 128, [128, 4], 3);
        d.fp4_padded = padded;
        let bytes = d.try_encode().unwrap_or_else(|e| panic!("{elem:?} padded={padded}: {e}"));
        let back = TensorMapDesc::decode(&bytes).unwrap();
        assert_eq!((back.elem, back.fp4_padded), (Some(elem), padded));
        assert_eq!(back.try_encode().unwrap(), bytes);
        if elem == Dtype::E2M1 {
            // The legacy planner accepts both FP4 shared layouts for loads:
            // 256 payload bytes either way; the padded layout spreads each
            // 8-byte unit over a 16-byte slot, doubling the shared extent.
            let plan = tma_plan(&d, TmaMode::Tile, &[0, 0], &[], 1024).unwrap();
            assert_eq!(plan.bytes, 256, "padded={padded}");
            let copied: u64 = plan.smem.iter().map(|s| s.len).sum();
            assert_eq!(copied, 256, "padded={padded}");
            ends.push(plan.smem.iter().map(|s| s.end()).max().unwrap());
        }
    }
    assert!(ends[1] > ends[0], "padded FP4 spreads over a larger shared extent: {ends:?}");
    let mut bad = desc2d(Dtype::F16, [64, 4], 128, [64, 4], 0);
    bad.fp4_padded = true;
    assert!(bad.try_encode().is_err());
}

/// Apply a store plan to byte arrays the way the engine does: byte spans
/// pairwise in concatenation order, then the masked sub-byte fragments.
fn apply_store(plan: &TmaPlan, shared: &[u8], global: &mut [u8], va: u64) {
    let src: Vec<u8> = plan.smem.iter().flat_map(|s| shared[s.start as usize..s.end() as usize].to_vec()).collect();
    let mut at = 0usize;
    for g in &plan.global {
        let off = (g.start - va) as usize;
        global[off..off + g.len as usize].copy_from_slice(&src[at..at + g.len as usize]);
        at += g.len as usize;
    }
    assert_eq!(at, src.len());
    for f in &plan.global_bits {
        let mask = f.mask << f.target_shift;
        let s = (shared[f.smem as usize] >> f.source_shift) & f.mask;
        let g = &mut global[(f.global - va) as usize];
        *g = (*g & !mask) | ((s << f.target_shift) & mask);
    }
}

#[test]
fn fp4_tma_store_matches_the_legacy_planner_packed_and_padded() {
    use numsim_oplib::tma::{execute_s2g_copy, plan_tiled_s2g};
    // Packed FP4 stores write every element as a masked nibble (legacy
    // `append_s2g_bits`); the box is partially out of bounds on the right
    // (cols 192..320 of 256) and the bottom (rows 2..6 of 4).
    {
        let padded = false;
        let mut d = desc2d(Dtype::E2M1, [256, 4], 128, [128, 4], 3);
        d.fp4_padded = padded;
        let smem_offset = 2048u64;
        let plan = tma_plan_dir(&d, TmaPlanDir::Store, TmaMode::Tile, &[192, 2], &[], smem_offset).unwrap();
        assert!(!plan.global_bits.is_empty(), "padded={padded}: expected sub-byte fragments");
        // Legacy reference over the same layout.
        let mut image = super::desc_to_image(&d).unwrap();
        image.host_address = false;
        image.allocation_id = 0;
        let layout = image.materialize(image.rank, usize::MAX, VA).unwrap();
        let reference = plan_tiled_s2g(&layout, &[192, 2], smem_offset as usize).unwrap();
        let shared: Vec<u8> = (0..16384u32).map(|i| (i.wrapping_mul(37) ^ (i >> 3)) as u8).collect();
        let global_len = 128 * 4;
        let mut want = vec![0xa5u8; global_len];
        execute_s2g_copy(&reference, &shared, smem_offset as usize, &mut want).unwrap();
        let mut got = vec![0xa5u8; global_len];
        apply_store(&plan, &shared, &mut got, VA);
        assert_eq!(got, want, "padded={padded}");
        assert_ne!(got, vec![0xa5u8; global_len], "the store wrote something");
    }
    // The 16-byte-aligned padded FP4 layout has no shared-to-global copy
    // (PTX; legacy rejected it the same way).
    let mut d = desc2d(Dtype::E2M1, [256, 4], 128, [128, 4], 3);
    d.fp4_padded = true;
    let err = tma_plan_dir(&d, TmaPlanDir::Store, TmaMode::Tile, &[0, 0], &[], 0).unwrap_err();
    assert!(err.message.contains("padded FP4"), "{err}");
}

#[test]
fn tma_reduce_validation_matches_the_legacy_ptx_table() {
    use crate::program::AtomOp as A;
    use numsim_oplib::tma::{RawTmaReductionOp as R, TensorMapElementType as T};
    let pairs: [(A, R); 8] = [
        (A::Add, R::Add),
        (A::Min, R::Min),
        (A::Max, R::Max),
        (A::Inc, R::Inc),
        (A::Dec, R::Dec),
        (A::And, R::And),
        (A::Or, R::Or),
        (A::Xor, R::Xor),
    ];
    let types: [(Dtype, T); 11] = [
        (Dtype::U8, T::U8),
        (Dtype::U16, T::U16),
        (Dtype::U32, T::U32),
        (Dtype::S32, T::I32),
        (Dtype::U64, T::U64),
        (Dtype::S64, T::I64),
        (Dtype::F16, T::F16),
        (Dtype::BF16, T::Bf16),
        (Dtype::F32, T::F32),
        (Dtype::F64, T::F64),
        (Dtype::TF32, T::Tf32),
    ];
    for (op, raw) in pairs {
        for (dtype, elem) in types {
            assert_eq!(
                super::super::tma_reduce_valid(op, dtype).is_ok(),
                raw.resolve(elem).is_ok(),
                "{op:?} {dtype:?}"
            );
        }
    }
    for op in [A::Exch, A::Cas] {
        assert!(super::super::tma_reduce_valid(op, Dtype::U32).is_err());
    }
}

/// The translation cache returns the direct planner's plan for interior and
/// OOB boxes alike (loads and stores, every swizzle, ranks 1..3).
#[test]
fn cached_tiled_plans_equal_direct_plans() {
    use super::super::{TmaPlanDir, TensorMapDesc};
    let mut seed = 0x9e37_79b9_u64;
    let mut next = move |bound: u64| {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (seed >> 33) % bound
    };
    let mut checked = 0;
    for (elem, bytes) in [(Dtype::U8, 1_u64), (Dtype::BF16, 2), (Dtype::F32, 4), (Dtype::F64, 8)] {
        for swizzle in 0..=3_u8 {
            for rank in 1..=3_u8 {
                let inner_box = (if swizzle == 0 { 64 } else { 16 << swizzle }) / bytes;
                let mut map = TensorMapDesc {
                    global_address: 0x1_0000_0000 + 0x100 * next(16),
                    rank,
                    elem: Some(elem),
                    global_dim: [512, 64, 8, 1, 1],
                    global_stride: [512 * bytes + 256, (512 * bytes + 256) * 64, 0, 0, 0],
                    box_dim: [inner_box as u32, 8, 2, 1, 1],
                    element_stride: [1, 1, 1, 1, 1],
                    swizzle,
                    ..Default::default()
                };
                for i in usize::from(rank)..5 {
                    map.global_dim[i] = 1;
                    map.box_dim[i] = 1;
                }
                for dir in [TmaPlanDir::Load, TmaPlanDir::Store] {
                    for _ in 0..12 {
                        let coords: Vec<i64> = (0..usize::from(rank))
                            .map(|i| next(map.global_dim[i] + 8) as i64 - 4)
                            .collect();
                        let smem = 0x400 * next(4);
                        let direct = super::plan_uncached(&map, dir, TmaMode::Tile, &coords, &[], smem);
                        let cached = super::plan(&map, dir, TmaMode::Tile, &coords, &[], smem);
                        assert_eq!(
                            cached.map_err(|e| e.to_string()),
                            direct.map_err(|e| e.to_string()),
                            "{elem:?} swizzle {swizzle} rank {rank} {dir:?} {coords:?}"
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 0);
}

/// `tensormap.replace .elemtype` 8 / 12 (f32.ftz / tf32.ftz) are
/// representable (`elem_ftz`), round-trip through the image, and plan like
/// their non-FTZ types (TF32 maps keep the landing tf32 rounding).
#[test]
fn ftz_element_types_are_representable() {
    let mut desc = desc2d(Dtype::F32, [64, 8], 256, [16, 4], 0);
    for (code, elem, tf32) in [(8_u64, Dtype::F32, false), (12, Dtype::TF32, true)] {
        desc.replace(TmapField::ElemType, None, code).unwrap();
        assert_eq!((desc.elem, desc.elem_ftz), (Some(elem), true), "elemtype {code}");
        let bytes = desc.try_encode().unwrap();
        let back = TensorMapDesc::decode(&bytes).unwrap();
        assert_eq!((back.elem, back.elem_ftz), (Some(elem), true));
        let plan = tma_plan(&desc, TmaMode::Tile, &[0, 0], &[], 0).unwrap();
        let mut plain = desc.clone();
        plain.elem_ftz = false;
        let reference = tma_plan(&plain, TmaMode::Tile, &[0, 0], &[], 0).unwrap();
        assert_eq!((plan.global, plan.smem, plan.tf32_round), (reference.global, reference.smem, tf32));
    }
    desc.replace(TmapField::ElemType, None, 7).unwrap();
    assert_eq!((desc.elem, desc.elem_ftz), (Some(Dtype::F32), false));
    let mut bad = desc2d(Dtype::F16, [64, 8], 128, [16, 4], 0);
    bad.elem_ftz = true;
    assert!(bad.try_encode().is_err());
}

/// `tensormap.replace` of a per-dimension field at an ordinal outside the
/// descriptor's slots (5 dimensions, 4 strides) is an operand error
/// (`Invalid`), never a panic. Ordinals between the rank and the last slot
/// are kept (legacy; kernels rewrite every slot before raising the rank).
#[test]
fn replace_rejects_ordinals_outside_the_descriptor_slots() {
    let base = desc2d(Dtype::F32, [64, 8], 256, [16, 4], 0);
    for (field, first_bad) in [
        (TmapField::BoxDim, 5_u8),
        (TmapField::GlobalDim, 5),
        (TmapField::ElementStride, 5),
        (TmapField::GlobalStride, 4),
    ] {
        for ord in [first_bad, first_bad + 1, 200, 255] {
            let mut desc = base.clone();
            let result = std::panic::catch_unwind(move || desc.replace(field, Some(ord), 16).map(|_| desc));
            let result = result.unwrap_or_else(|_| panic!("{field:?}[{ord}] panicked"));
            match result {
                Err(e) => assert_eq!(e.kind, OpErrorKind::Invalid, "{field:?}[{ord}]: {e}"),
                Ok(d) => panic!("{field:?}[{ord}] succeeded: {d:?}"),
            }
        }
    }
    // Inside the slots but beyond rank 2: stored, and live once the rank grows.
    let mut desc = base.clone();
    desc.replace(TmapField::GlobalDim, Some(2), 3).unwrap();
    desc.replace(TmapField::BoxDim, Some(2), 3).unwrap();
    desc.replace(TmapField::GlobalStride, Some(1), 4096).unwrap();
    assert_eq!((desc.global_dim[2], desc.box_dim[2], desc.global_stride[1]), (3, 3, 4096));
    desc.replace(TmapField::Rank, None, 2).unwrap();
    assert_eq!(desc.rank, 3);
}

/// W12-gaps 2: TMA `.override::*` operand rules (legacy `with_overrides`). The
/// override address needs 128 KiB of accessible memory, and a dimension/stride
/// override requires zero tensor coordinates; both are `Invalid`.
#[test]
fn tma_overrides_reject_short_address_window_and_nonzero_coordinates() {
    let base = desc2d(Dtype::F32, [8, 8], 32, [4, 2], 0);
    let window_end = 0x9000_0000_u64 + 128 * 1024 + 16;
    let accessible = |va: u64, len: u64| va >= 0x9000_0000 && va + len <= window_end;
    let address = |va| (TmapField::GlobalAddress, None, va);
    let dims = [
        (TmapField::GlobalDim, Some(0), 4),
        (TmapField::GlobalDim, Some(1), 2),
        (TmapField::GlobalStride, Some(0), 2),
        (TmapField::GlobalStrideUpper, None, 0),
    ];
    let with = |extra: &[(TmapField, Option<u8>, u64)], va| {
        let mut v = vec![address(va)];
        v.extend_from_slice(extra);
        v
    };
    // In-window address (exactly 128 KiB remain, and 16 more), zero coordinates: accepted.
    for va in [0x9000_0000_u64, 0x9000_0010] {
        let mut d = base.clone();
        d.apply_overrides(&with(&dims, va), &[0, 0], &accessible).unwrap();
        assert_eq!(d.global_address, va);
        assert_eq!(d.global_dim[..2], [4, 2]);
    }
    // 32 bytes too far: fewer than 128 KiB accessible.
    let mut d = base.clone();
    let err = d.apply_overrides(&with(&dims, 0x9000_0020), &[0, 0], &accessible).unwrap_err();
    assert_eq!(err.kind, OpErrorKind::Invalid, "{err:?}");
    assert!(err.to_string().contains("128 KiB"), "{err}");
    // Address-only override: the window rule still applies.
    let mut d = base.clone();
    assert!(d.apply_overrides(&with(&[], 0x9000_0020), &[3, 1], &accessible).is_err());
    // Address-only override with nonzero coordinates: allowed (no attribute override).
    let mut d = base.clone();
    d.apply_overrides(&with(&[], 0x9000_0000), &[3, 1], &accessible).unwrap();
    // Dimension/stride override with a nonzero coordinate: rejected, either axis.
    for coords in [[1, 0], [0, 1], [-1, 0]] {
        let mut d = base.clone();
        let err = d.apply_overrides(&with(&dims, 0x9000_0000), &coords, &accessible).unwrap_err();
        assert_eq!(err.kind, OpErrorKind::Invalid, "{err:?}");
        assert!(err.to_string().contains("zero coordinates"), "{err}");
    }
}
