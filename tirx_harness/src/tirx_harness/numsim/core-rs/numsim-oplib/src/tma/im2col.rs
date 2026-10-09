//! Im2col TMA: descriptor bounding-box validation, pixel-origin walk
//! (`im2col`, `im2col::w`, `im2col::w::128` incl. wHalo/wOffset and spatial
//! offsets), and load/store transfer planning.
//!
//! Legacy source: `engine-rs/src/runtime/tensor_map.rs`
//! (`RuntimeTensorMap::new_with_layout` im2col checks,
//! `RuntimeTensorMap::im2col_origins`, `im2col_transfer_runs`,
//! `RawTmaG2cTransferPlan::im2col`, `RawTmaS2gTransferPlan::im2col`) and
//! `engine-rs/src/runtime/instructions/async_copy.rs` (`tma_store_mode`).

use super::descriptor::{Fp4SharedLayout, TensorMapIm2col};
use super::tensor_map::TensorMapLayout;
use super::tiled::{
    append_s2g_bits, expand_u6_source_runs, g2s_layout_checks, ByteRun, G2sPlan, S2gPlan,
};
use crate::types::{OpError, OpResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Im2colMode {
    /// `.im2col`: spatial offsets, rank-2 spatial dims.
    Spatial,
    /// `.im2col::w`.
    Wide,
    /// `.im2col::w::128`.
    Wide128,
}

/// Store-mode operand of the raw TMA store ABI: 0 tiled, 1 im2col, 2 im2col::w.
pub fn tma_store_mode(mode: u32) -> OpResult<Option<Im2colMode>> {
    match mode {
        0 => Ok(None),
        1 => Ok(Some(Im2colMode::Spatial)),
        2 => Ok(Some(Im2colMode::Wide)),
        _ => Err(OpError::message("invalid TensorMap store mode")),
    }
}

/// Descriptor-time im2col checks (corner ranges per rank, unused corners,
/// wide-mode interleave/swizzle restrictions).
pub(crate) fn validate_im2col_bounds(
    config: &TensorMapIm2col,
    global_shape: &[usize],
    box_shape: &[usize],
    interleave_bytes: Option<usize>,
    swizzle_bytes: Option<usize>,
) -> OpResult<()> {
    let rank = global_shape.len();
    if rank < 3 || box_shape[0] > 256 || box_shape[1] > 1024 {
        return Err(OpError::message(
            "im2col requires rank 3..5, 1..256 channels and 1..1024 pixels",
        ));
    }
    let spatial_rank = if config.wide { 1 } else { rank - 2 };
    let bits = if config.wide || rank == 3 {
        16
    } else if rank == 4 {
        8
    } else {
        5
    };
    for axis in 0..3 {
        let lo = i64::from(config.lower[axis]);
        let hi = i64::from(config.upper[axis]);
        if axis >= spatial_rank {
            if lo != 0 || hi != 0 {
                return Err(OpError::message("im2col has nonzero unused corners"));
            }
        } else if lo < -(1 << (bits - 1))
            || hi < -(1 << (bits - 1))
            || lo >= (1 << (bits - 1))
            || hi >= (1 << (bits - 1))
            || lo >= global_shape[axis + usize::from(interleave_bytes.is_none())] as i64 + hi
        {
            return Err(OpError::message("im2col has invalid bounding-box corners"));
        }
    }
    if config.wide && interleave_bytes.is_some() {
        return Err(OpError::message("wide im2col does not support interleave"));
    }
    if config.wide && !matches!(swizzle_bytes, Some(64 | 96 | 128)) {
        return Err(OpError::message(
            "wide im2col requires 64B, 96B or 128B swizzle",
        ));
    }
    Ok(())
}

/// Global coordinates of every pixel the instruction loads, in payload order
/// (main pixels first, then the halo plane; `w::128` interleaves halo chunks).
/// `info` is the spatial offsets (`Spatial`) or `[wHalo, wOffset]` (wide).
pub fn im2col_origins(
    map: &TensorMapLayout,
    origin: &[i64],
    mode: Im2colMode,
    info: &[i64],
) -> OpResult<Vec<Vec<i64>>> {
    let layout = map
        .im2col
        .as_ref()
        .ok_or_else(|| OpError::message("im2col instruction requires an im2col TensorMap"))?;
    let rank = map.rank();
    let wide = mode != Im2colMode::Spatial;
    if origin.len() != rank || wide != layout.wide {
        return Err(OpError::message(
            "im2col instruction/descriptor layout mismatch",
        ));
    }
    let spatial = if wide { 1 } else { rank - 2 };
    let first_spatial = usize::from(map.interleave_bytes.is_none());
    let (halo, shift) = if wide {
        if info.len() != 2
            || !(0..32).contains(&info[1])
            || !(0..if mode == Im2colMode::Wide128 { 32 } else { 512 }).contains(&info[0])
        {
            return Err(OpError::message("invalid im2col wHalo/wOffset"));
        }
        (info[0] as usize, info[1])
    } else {
        let bits = match rank {
            3 => 16,
            4 => 8,
            _ => 5,
        };
        if info.len() != spatial || info.iter().any(|value| !(0..1_i64 << bits).contains(value)) {
            return Err(OpError::message("invalid im2col spatial offsets"));
        }
        (0, 0)
    };
    let pixels = if mode == Im2colMode::Wide128 {
        128
    } else {
        map.box_shape[1]
    };
    if mode == Im2colMode::Wide128 && map.swizzle_bytes == Some(96) {
        return Err(OpError::message(
            "im2col::w::128 does not support 96B swizzle",
        ));
    }
    let lower = (0..spatial)
        .map(|axis| i64::from(layout.lower[axis]) + shift)
        .collect::<Vec<_>>();
    let upper = (0..spatial)
        .map(|axis| {
            map.global_shape[axis + first_spatial] as i64 + i64::from(layout.upper[axis]) + shift
        })
        .collect::<Vec<_>>();
    let mut cursor = origin.to_vec();
    cursor[first_spatial] = cursor[first_spatial]
        .checked_add(shift)
        .ok_or_else(|| OpError::message("im2col coordinate overflow"))?;
    if (0..spatial).any(|axis| {
        (!wide && cursor[axis + first_spatial] < lower[axis])
            || cursor[axis + first_spatial] >= upper[axis]
    }) {
        return Err(OpError::message(
            "im2col filter origin is outside its traversal bounds",
        ));
    }
    let advance = |cursor: &mut Vec<i64>| -> OpResult<()> {
        for axis in 0..spatial {
            cursor[axis + first_spatial] = cursor[axis + first_spatial]
                .checked_add(map.element_strides[axis + first_spatial] as i64)
                .ok_or_else(|| OpError::message("im2col coordinate overflow"))?;
            if cursor[axis + first_spatial] < upper[axis] {
                return Ok(());
            }
            cursor[axis + first_spatial] = lower[axis];
        }
        cursor[rank - 1] = cursor[rank - 1]
            .checked_add(map.element_strides[rank - 1] as i64)
            .ok_or_else(|| OpError::message("im2col batch coordinate overflow"))?;
        Ok(())
    };
    let chunks = if mode == Im2colMode::Wide128 { 4 } else { 1 };
    let mut origins = vec![Vec::new(); pixels + chunks * halo];
    for pixel in 0..pixels {
        let mut address = cursor.clone();
        if !wide {
            for axis in 0..spatial {
                address[axis + first_spatial] = address[axis + first_spatial]
                    .checked_add(info[axis])
                    .ok_or_else(|| OpError::message("im2col offset overflow"))?;
            }
        }
        origins[pixel] = address;
        advance(&mut cursor)?;
        if (pixel + 1) % (pixels / chunks) == 0 {
            let chunk = (pixel + 1) / (pixels / chunks) - 1;
            let mut halo_cursor = cursor.clone();
            for index in 0..halo {
                // w::128 appends an interleaved halo plane after its
                // 128 main pixels, rather than appending to each chunk.
                origins[pixels + index * chunks + chunk] = halo_cursor.clone();
                advance(&mut halo_cursor)?;
            }
        }
    }
    Ok(origins)
}

/// Both directions use the same pixel walk, channel packing and swizzle.
/// Returns `(global runs, shared runs)`; shared runs are relative to the
/// shared pointer. Sub-byte stores skip the global runs (bit fragments).
pub fn im2col_transfer_runs(
    map: &TensorMapLayout,
    origins: &[Vec<i64>],
    absolute_base: usize,
    write_shared: bool,
) -> OpResult<(Vec<ByteRun>, Vec<ByteRun>)> {
    let template = &map.transfer_template;
    let geometry = template.geometry;
    let mut global_runs = Vec::new();
    let mut shared_runs = Vec::new();
    for (pixel, origin) in origins.iter().enumerate() {
        let payload_base = pixel * template.payload_len;
        if write_shared || !matches!(map.transfer_element_bits(), 4 | 6) {
            for mut run in template.bind_global(map, origin)? {
                run.payload_offset += payload_base;
                global_runs.push(run);
            }
        }
        for unit in 0..geometry.inner_units {
            // A 32B channel slice crosses two independently swizzled atoms.
            let atom_bytes = geometry.unit_bytes.min(16);
            for atom in (0..geometry.unit_bytes).step_by(atom_bytes) {
                let offset = map.shared_byte_offset(
                    pixel,
                    unit * geometry.unit_stride_bytes + atom,
                    geometry.inner_row_bytes,
                    absolute_base,
                )?;
                shared_runs.push(ByteRun {
                    byte_offset: offset,
                    payload_offset: payload_base + unit * geometry.unit_bytes + atom,
                    byte_len: atom_bytes,
                });
            }
        }
    }
    Ok((global_runs, shared_runs))
}

/// Plan an im2col global-to-shared load.
pub fn plan_im2col_g2s(
    map: &TensorMapLayout,
    origin: &[i64],
    mode: Im2colMode,
    info: &[i64],
    shared_absolute_base: usize,
) -> OpResult<G2sPlan> {
    let origins = im2col_origins(map, origin, mode, info)?;
    let mut geometry = g2s_layout_checks(map, origin.first().copied().unwrap_or(0))?;
    geometry.outer_count = origins.len();
    let (source_runs, destination_runs) =
        im2col_transfer_runs(map, &origins, shared_absolute_base, true)?;
    Ok(G2sPlan {
        geometry,
        source_runs,
        destination_runs,
        payload_len: origins.len() * map.transfer_template.payload_len,
        fill_mode: map.fill_mode,
    })
}

/// Plan an im2col (`im2col_no_offs` / `im2col::w`) shared-to-global store or
/// reduction.
pub fn plan_im2col_s2g(
    map: &TensorMapLayout,
    origin: &[i64],
    mode: Im2colMode,
    shared_absolute_base: usize,
) -> OpResult<S2gPlan> {
    let bounds = map
        .im2col
        .as_ref()
        .ok_or_else(|| OpError::message("im2col store requires an im2col TensorMap"))?;
    if origin.iter().any(|value| *value < 0)
        || bounds.lower.iter().any(|value| *value < 0)
        || bounds.upper.iter().any(|value| *value > 0)
    {
        return Err(OpError::message(
            "im2col store requires nonnegative coordinates and bounds inside the tensor",
        ));
    }
    if map.fp4_shared_layout == Some(Fp4SharedLayout::Align16Padded) {
        return Err(OpError::message(
            "align16 padded FP4 TensorMap does not support shared-to-global Tensor Copy",
        ));
    }
    let count = if mode == Im2colMode::Spatial {
        map.rank() - 2
    } else {
        2
    };
    let mut origins = im2col_origins(map, origin, mode, &vec![0; count])?;
    if mode == Im2colMode::Wide {
        // Once a pixel leaves the tensor, no subsequent pixel is read
        // from shared memory or written/reduced to global memory.
        let valid = origins
            .iter()
            .take_while(|coordinates| map.coordinates_in_bounds(coordinates))
            .count();
        origins.truncate(valid);
    }
    map.validate_swizzle_direction(false)?;
    let (destination_runs, mut source_runs) =
        im2col_transfer_runs(map, &origins, shared_absolute_base, false)?;
    let geometry = map.transfer_template.geometry;
    let mut pixel_bytes = map.transfer_template.payload_len;
    let unit_bytes = if map.element_bits == 6 {
        pixel_bytes = expand_u6_source_runs(&mut source_runs, pixel_bytes)?;
        16
    } else {
        geometry.unit_bytes.min(16)
    };
    let mut destination_bits = Vec::new();
    if matches!(map.transfer_element_bits(), 4 | 6) {
        for (pixel, origin) in origins.iter().enumerate() {
            map.validate_u6_origin(origin)?;
            for unit in 0..geometry.inner_units {
                let mut coordinates = origin.clone();
                coordinates[0] = coordinates[0]
                    .checked_add((unit * geometry.packed_elements) as i64)
                    .ok_or_else(|| OpError::message("sub-byte TensorMap coordinate overflow"))?;
                append_s2g_bits(
                    map,
                    &coordinates,
                    pixel * pixel_bytes + unit * unit_bytes,
                    &mut destination_bits,
                )?;
            }
        }
    }
    Ok(S2gPlan {
        source_runs,
        destination_runs,
        destination_bits,
        unit_bytes,
        payload_len: origins.len() * pixel_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tma::descriptor::{TensorMapElementType, TensorMapFillMode};
    use crate::tma::swizzle::SwizzleAtomicity;
    use crate::tma::tensor_map::TensorMapSpec;
    use crate::tma::tiled::{execute_g2s, execute_s2g_copy};

    /// NHWC rank-4 map: C=16 u8 channels, W=4, H=3, N=2.
    fn nhwc_map(lower: [i16; 3], upper: [i16; 3], pixels: usize) -> TensorMapLayout {
        TensorMapLayout::new(
            TensorMapSpec {
                global_shape: vec![16, 4, 3, 2],
                global_strides: vec![16, 64, 192],
                box_shape: vec![16, pixels],
                element_strides: vec![1; 4],
                element_bits: 8,
                element_type: TensorMapElementType::U8,
                fp4_shared_layout: None,
                swizzle_bytes: None,
                swizzle_atomicity: SwizzleAtomicity::B16,
                fill_mode: TensorMapFillMode::Zero,
                interleave_bytes: None,
                im2col: Some(TensorMapIm2col {
                    lower,
                    upper,
                    wide: false,
                }),
            },
            0,
            384,
        )
        .unwrap()
    }

    #[test]
    fn spatial_walk_wraps_w_then_h_then_n_and_applies_offsets() {
        let map = nhwc_map([0, 0, 0], [-1, -1, 0], 8);
        // Traversal bounds W in [0, 3), H in [0, 2); offsets (1, 1).
        let origins = im2col_origins(&map, &[0, 0, 0, 0], Im2colMode::Spatial, &[1, 1]).unwrap();
        let expected = [
            [0, 1, 1, 0],
            [0, 2, 1, 0],
            [0, 3, 1, 0],
            [0, 1, 2, 0],
            [0, 2, 2, 0],
            [0, 3, 2, 0],
            [0, 1, 1, 1],
            [0, 2, 1, 1],
        ];
        assert_eq!(
            origins,
            expected.iter().map(|o| o.to_vec()).collect::<Vec<_>>()
        );
        assert!(im2col_origins(&map, &[0, 3, 0, 0], Im2colMode::Spatial, &[0, 0]).is_err());
        assert!(im2col_origins(&map, &[0, 0, 0, 0], Im2colMode::Spatial, &[256, 0]).is_err());
        assert!(im2col_origins(&map, &[0, 0, 0, 0], Im2colMode::Wide, &[0, 0]).is_err());
    }

    #[test]
    fn descriptor_rejects_illegal_corners() {
        let error = TensorMapLayout::new(
            TensorMapSpec {
                im2col: Some(TensorMapIm2col {
                    lower: [0, 0, 1],
                    upper: [0, 0, 0],
                    wide: false,
                }),
                ..nhwc_spec()
            },
            0,
            384,
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "im2col has nonzero unused corners");
        let error = TensorMapLayout::new(
            TensorMapSpec {
                im2col: Some(TensorMapIm2col {
                    lower: [128, 0, 0],
                    upper: [0, 0, 0],
                    wide: false,
                }),
                ..nhwc_spec()
            },
            0,
            384,
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "im2col has invalid bounding-box corners");
        let error = TensorMapLayout::new(
            TensorMapSpec {
                im2col: Some(TensorMapIm2col {
                    lower: [0, 0, 0],
                    upper: [0, 0, 0],
                    wide: true,
                }),
                ..nhwc_spec()
            },
            0,
            384,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "wide im2col requires 64B, 96B or 128B swizzle"
        );
    }

    fn nhwc_spec() -> TensorMapSpec {
        TensorMapSpec {
            global_shape: vec![16, 4, 3, 2],
            global_strides: vec![16, 64, 192],
            box_shape: vec![16, 4],
            element_strides: vec![1; 4],
            element_bits: 8,
            element_type: TensorMapElementType::U8,
            fp4_shared_layout: None,
            swizzle_bytes: None,
            swizzle_atomicity: SwizzleAtomicity::B16,
            fill_mode: TensorMapFillMode::Zero,
            interleave_bytes: None,
            im2col: None,
        }
    }

    #[test]
    fn padded_pixels_read_zero_and_store_round_trips_in_bounds_pixels() {
        // Lower corner -1 pads one pixel on the left of each W row.
        let map = nhwc_map([-1, 0, 0], [0, 0, 0], 5);
        let global = (0..384)
            .map(|i| (i as u8).wrapping_mul(7))
            .collect::<Vec<_>>();
        let plan = plan_im2col_g2s(&map, &[0, -1, 0, 0], Im2colMode::Spatial, &[0, 0], 0).unwrap();
        assert_eq!(plan.payload_len, 80);
        assert_eq!(plan.geometry.outer_count, 5);
        let mut shared = vec![0xee_u8; 80];
        execute_g2s(&map, &plan, &global, &mut shared, 0, 0).unwrap();
        // Pixel 0 is W=-1 (zero fill), then W=0..3 of H=0.
        assert_eq!(&shared[..16], &[0; 16]);
        assert_eq!(&shared[16..80], &global[..64]);

        let store_map = nhwc_map([0, 0, 0], [0, 0, 0], 4);
        let store = plan_im2col_s2g(&store_map, &[0, 0, 1, 0], Im2colMode::Spatial, 0).unwrap();
        let mut destination = vec![0_u8; 384];
        execute_s2g_copy(&store, &shared[16..], 0, &mut destination).unwrap();
        assert_eq!(&destination[64..128], &global[..64]);
        assert!(destination[..64]
            .iter()
            .chain(&destination[128..])
            .all(|b| *b == 0));
        assert!(tma_store_mode(3).is_err());
        assert_eq!(tma_store_mode(2).unwrap(), Some(Im2colMode::Wide));
    }
}
