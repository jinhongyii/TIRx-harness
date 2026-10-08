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
