//! Pure address walks behind the checker footprints of tcgen05 instructions.
//!
//! Legacy source: `engine-rs/src/runtime/tcgen_ops.rs`
//! (`raw_tcgen05_cta1_shared_matrix_accesses`, `raw_tcgen05_f8_shared_matrix_accesses`,
//! `raw_tcgen05_b16/tf32_shared_*footprints` walks, `raw_tcgen05_cta1_accumulator_accesses`,
//! `raw_tcgen05_mxf8_scale_accesses`, `raw_tcgen05_cta{1,2}_scale_accesses`,
//! `raw_tcgen05_lut_b_tmem_footprints`, `raw_tcgen05_sparse_metadata_footprints`,
//! `raw_tcgen05_cta2_*_tmem_footprints`).
//!
//! Legacy returned engine tuples `(provenance_lane, execution_lane, cta,
//! lane, byte, column, bytes)`; here the caller decorates [`TmemAccess`] /
//! `(offset, bytes)` with lanes and resolves the source buffer.

use super::layouts::{
    block_scale_address, dense_tmem_cells, layout_f_lane, lut_b_location, packed_tmem_a_cells,
    scale_chunk, sparse_metadata_location, tmem_address, DenseTmemLayout, ScaleLayout,
    SparseMetadataLayout, CTA1_PACKED_A_COLUMNS,
};
use super::narrow::NarrowFormat;
use super::smem_desc::{
    byte8_matrix_byte_offset, lut_b_row_accesses, masked_row, matrix_byte_offset,
    narrow_shared_atom_accesses, shared_byte_offset, ColumnMask, MatrixDescriptor, SharedWindow,
};
use crate::types::{OpError, OpResult};

/// One TMEM byte-range access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TmemAccess {
    pub cta: Option<usize>,
    pub lane: usize,
    pub byte: usize,
    pub column: usize,
    pub bytes: usize,
}

impl TmemAccess {
    pub const fn cell(cta: Option<usize>, lane: usize, column: usize) -> Self {
        Self {
            cta,
            lane,
            byte: 0,
            column,
            bytes: 4,
        }
    }
}

/// `pair_base` of a CTA2 op, failing when the pair leaves the cluster.
pub fn cta_pair_base(
    cta_id_in_cluster: usize,
    ctas_per_cluster: usize,
    message: &str,
) -> OpResult<usize> {
    let pair_base = cta_id_in_cluster & !1_usize;
    if pair_base + 1 >= ctas_per_cluster {
        return Err(OpError::message(message.to_owned()));
    }
    Ok(pair_base)
}

/// 16-byte atoms of a padded K32 row (legacy `raw_tcgen05_cta1_shared_matrix_accesses`).
pub fn cta1_shared_matrix_accesses(
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    rows: usize,
    k: usize,
) -> OpResult<Vec<(usize, usize)>> {
    let mut accesses = Vec::with_capacity(rows * (k / 32));
    for row in 0..rows {
        for atom in 0..(k / 32) {
            accesses.push((
                shared_byte_offset(source, descriptor, row, atom * 16, 16)?,
                16,
            ));
        }
    }
    Ok(accesses)
}

/// Narrow-operand shared accesses of one CTA (legacy `raw_tcgen05_f8_shared_matrix_accesses`).
#[allow(clippy::too_many_arguments)]
pub fn f8_shared_matrix_accesses(
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    rows: usize,
    k_extent: usize,
    format: NarrowFormat,
    transpose: bool,
    mask: Option<ColumnMask>,
    lut_b: bool,
    padded_atoms: bool,
) -> OpResult<Vec<(usize, usize)>> {
    let mut accesses = Vec::new();
    if lut_b {
        super::instr_desc::validate_lut_b(Some(0), k_extent, format, transpose)?;
        for row in 0..rows {
            lut_b_row_accesses(source, descriptor, row, |offset, bytes| {
                accesses.push((offset, bytes));
                Ok(())
            })?;
        }
        return Ok(accesses);
    }
    if transpose {
        if format.format().width_bits != 8 {
            return Err(OpError::message(
                "MN-major narrow MMA footprints require 8-bit elements",
            ));
        }
        for row in 0..rows {
            let Some(row) = masked_row(mask, row) else {
                continue;
            };
            for k in 0..k_extent {
                accesses.push((
                    byte8_matrix_byte_offset(source, descriptor, row, k, true)?,
                    1,
                ));
            }
        }
        return Ok(accesses);
    }
    if k_extent % 16 != 0 {
        return Err(OpError::message(format!(
            "raw f8f6f4 footprint K={k_extent} is not a multiple of 16"
        )));
    }
    for row in 0..rows {
        let Some(row) = masked_row(mask, row) else {
            continue;
        };
        for atom in 0..k_extent / 16 {
            narrow_shared_atom_accesses(
                source,
                descriptor,
                row,
                k_extent,
                format,
                atom,
                |offset, bytes| {
                    accesses.push((offset, bytes));
                    Ok(())
                },
                padded_atoms,
            )?;
        }
    }
    Ok(accesses)
}

/// Element-wise shared accesses of a 16/32-bit operand (legacy
/// `raw_tcgen05_b16_shared_masked_footprints` / `tf32_shared_footprints` walk).
pub fn element_shared_accesses(
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    rows: usize,
    columns: usize,
    transpose: bool,
    element_bytes: usize,
    mask: Option<ColumnMask>,
) -> OpResult<Vec<(usize, usize)>> {
    let mut accesses = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or_else(|| OpError::message("raw MMA footprint shape overflow"))?,
    );
    for row in 0..rows {
        let Some(row) = masked_row(mask, row) else {
            continue;
        };
        for column in 0..columns {
            accesses.push((
                matrix_byte_offset(source, descriptor, row, column, transpose, element_bytes)?,
                element_bytes,
            ));
        }
    }
    Ok(accesses)
}

/// Dense accumulator cells (legacy `raw_tcgen05_dense_f32_tmem_footprints`).
pub fn dense_tmem_accesses(
    destination_address: u32,
    m: usize,
    n: usize,
    layout: DenseTmemLayout,
    disable_output_lane: [u32; 4],
    cta: Option<usize>,
) -> OpResult<Vec<TmemAccess>> {
    Ok(
        dense_tmem_cells(destination_address, m, n, layout, Some(disable_output_lane))?
            .into_iter()
            .map(|(_, _, lane, column)| TmemAccess::cell(cta, lane, column))
            .collect(),
    )
}

/// CTA-pair accumulator cells (legacy `raw_tcgen05_cta2_layout_tmem_footprints`).
pub fn cta2_layout_tmem_accesses(
    pair_base: usize,
    destination_address: u32,
    m: usize,
    n: usize,
    layout: DenseTmemLayout,
    disable_output_lane: [u32; 8],
) -> OpResult<Vec<TmemAccess>> {
    if !matches!(m, 128 | 256) {
        return Err(OpError::message(
            "invalid CTA2 TMEM shape or missing paired CTA",
        ));
    }
    let mut accesses = Vec::with_capacity(m * n);
    for cta in 0..2 {
        accesses.extend(dense_tmem_accesses(
            destination_address,
            m / 2,
            n,
            layout,
            std::array::from_fn(|i| disable_output_lane[cta * 4 + i]),
            Some(pair_base + cta),
        )?);
    }
    Ok(accesses)
}

/// Packed TMEM A cells (legacy `raw_tcgen05_packed_tmem_a_column_footprints`).
pub fn packed_tmem_a_accesses(
    address: u32,
    m: usize,
    layout: DenseTmemLayout,
    columns: usize,
    cta: Option<usize>,
) -> OpResult<Vec<TmemAccess>> {
    Ok(packed_tmem_a_cells(address, m, layout, columns)?
        .into_iter()
        .map(|(_, _, _, lane, column)| TmemAccess::cell(cta, lane, column))
        .collect())
}

/// Whole-row accumulator spans of a block-scaled CTA1 MMA (legacy
/// `raw_tcgen05_cta1_accumulator_accesses`).
pub fn cta1_accumulator_accesses(
    cta: usize,
    destination_address: u32,
    m: usize,
    n: usize,
    layout_f: bool,
    disable_output_lane: [u32; 4],
) -> OpResult<Vec<TmemAccess>> {
    let (base_lane, base_column) = tmem_address(destination_address, 0, 0)?;
    let mut accesses = Vec::with_capacity(m);
    for row in 0..m {
        let lane_delta = if layout_f { layout_f_lane(row)? } else { row };
        let lane = base_lane
            .checked_add(lane_delta)
            .ok_or_else(|| OpError::message("raw TCGEN destination lane overflow"))?;
        if lane >= 128 {
            return Err(OpError::message(format!(
                "raw TCGEN destination lane {lane} is outside 128 lanes"
            )));
        }
        if super::layouts::lane_disabled(&disable_output_lane, lane) {
            continue;
        }
        accesses.push(TmemAccess {
            cta: Some(cta),
            lane,
            byte: 0,
            column: base_column,
            bytes: n
                .checked_mul(4)
                .ok_or_else(|| OpError::message("raw TCGEN destination width overflow"))?,
        });
    }
    Ok(accesses)
}

pub fn mxf8_scale_accesses(
    address: u32,
    scale_id: usize,
    rows: usize,
    layout: ScaleLayout,
    target_cta: usize,
) -> OpResult<Vec<TmemAccess>> {
    let mut accesses = Vec::with_capacity(rows * layout.replicas());
    for row in 0..rows {
        for replica in 0..layout.replicas() {
            let (lane, column) = layout.location(address, row, replica)?;
            accesses.push(TmemAccess {
                cta: Some(target_cta),
                lane,
                byte: scale_id,
                column,
                bytes: 1,
            });
        }
    }
    Ok(accesses)
}

fn scale_row_accesses(
    accesses: &mut Vec<TmemAccess>,
    cta: usize,
    address: u32,
    scale_id: usize,
    scale_bytes: usize,
    lane: usize,
    matrix_row: usize,
    matrix_rows: usize,
    lanes_per_column: usize,
    column_message: &'static str,
) -> OpResult<()> {
    let mut vector = 0;
    while vector < scale_bytes {
        let (chunk_address, byte) =
            scale_chunk(address, scale_id, vector, matrix_rows, lanes_per_column)?;
        let (_, base_column) = block_scale_address(chunk_address)?;
        let chunk_bytes = (scale_bytes - vector).min(4 - byte);
        let column = base_column
            .checked_add(matrix_row / lanes_per_column)
            .ok_or_else(|| OpError::message(column_message))?;
        accesses.push(TmemAccess {
            cta: Some(cta),
            lane,
            byte,
            column,
            bytes: chunk_bytes,
        });
        vector += chunk_bytes;
    }
    Ok(())
}

/// Scale-byte runs of a CTA1 block-scaled operand (legacy `raw_tcgen05_cta1_scale_accesses`).
pub fn cta1_scale_accesses(
    cta: usize,
    address: u32,
    scale_id: usize,
    scale_bytes: usize,
    rows: usize,
    lanes_per_column: usize,
) -> OpResult<Vec<TmemAccess>> {
    let (base_lane, _) = block_scale_address(address)?;
    let mut accesses = Vec::with_capacity(rows);
    for row in 0..rows {
        let lane = base_lane
            .checked_add(row % lanes_per_column)
            .ok_or_else(|| OpError::message("raw TCGEN scale lane overflow"))?;
        if lane >= 128 {
            return Err(OpError::message(format!(
                "raw TCGEN scale location lane={lane}, byte={scale_id} is outside TMEM"
            )));
        }
        scale_row_accesses(
            &mut accesses,
            cta,
            address,
            scale_id,
            scale_bytes,
            lane,
            row,
            rows,
            lanes_per_column,
            "raw TCGEN scale column overflow",
        )?;
    }
    Ok(accesses)
}

/// Scale-byte runs of a CTA-pair block-scaled operand (legacy `raw_tcgen05_cta2_scale_accesses`).
#[allow(clippy::too_many_arguments)]
pub fn cta2_scale_accesses(
    pair_base: usize,
    address: u32,
    scale_id: usize,
    scale_bytes: usize,
    rows_per_cta: usize,
    rows_are_joint: bool,
    lanes_per_column: usize,
) -> OpResult<Vec<TmemAccess>> {
    let (base_lane, _) = block_scale_address(address)?;
    let mut accesses = Vec::with_capacity(rows_per_cta * 2);
    for target_offset in 0..2 {
        let target_cta = pair_base + target_offset;
        for row in 0..rows_per_cta {
            let matrix_row = if rows_are_joint {
                target_offset
                    .checked_mul(rows_per_cta)
                    .and_then(|value| value.checked_add(row))
                    .ok_or_else(|| OpError::message("raw TCGEN cta2 scale row overflow"))?
            } else {
                row
            };
            let lane = base_lane
                .checked_add(matrix_row % lanes_per_column)
                .ok_or_else(|| OpError::message("raw TCGEN cta2 scale lane overflow"))?;
            if lane >= 128 {
                return Err(OpError::message(format!(
                    "raw TCGEN cta2 scale location lane={lane}, byte={scale_id} is outside TMEM"
                )));
            }
            scale_row_accesses(
                &mut accesses,
                target_cta,
                address,
                scale_id,
                scale_bytes,
                lane,
                matrix_row,
                rows_per_cta * if rows_are_joint { 2 } else { 1 },
                lanes_per_column,
                "raw TCGEN cta2 scale column overflow",
            )?;
        }
    }
    Ok(accesses)
}

/// LUT-B lookup words of each CTA in the group (legacy `raw_tcgen05_lut_b_tmem_footprints`).
pub fn lut_b_tmem_accesses(
    first_cta: usize,
    address: u32,
    rows: usize,
    cta_group: usize,
) -> OpResult<Vec<TmemAccess>> {
    if !matches!(cta_group, 1 | 2) || rows == 0 || rows % 8 != 0 {
        return Err(OpError::message("invalid LUT-B geometry"));
    }
    let mut accesses = Vec::with_capacity(rows / 8 * 2 * cta_group);
    for cta in first_cta..first_cta + cta_group {
        for group in 0..rows / 8 {
            for word in 0..2 {
                let (row, col) = lut_b_location(address, group, word)?;
                accesses.push(TmemAccess::cell(Some(cta), row, col));
            }
        }
    }
    Ok(accesses)
}

/// Distinct metadata cells of a sparse MMA (legacy `raw_tcgen05_sparse_metadata_footprints`).
pub fn sparse_metadata_accesses(
    first_cta: usize,
    address: u32,
    metadata_layout: SparseMetadataLayout,
    rows: usize,
    layout: DenseTmemLayout,
    cta_group: usize,
) -> OpResult<Vec<TmemAccess>> {
    let mut locations = std::collections::BTreeSet::new();
    for bank in 0..layout.packed_a_banks() {
        for row in 0..rows {
            let physical_row = layout
                .packed_a_location(bank, row, 0, rows, CTA1_PACKED_A_COLUMNS)?
                .0;
            for chunk in 0..metadata_layout.chunks() {
                let (lane, column, _) =
                    sparse_metadata_location(address, metadata_layout, physical_row, chunk)?;
                locations.insert((lane, column));
            }
        }
    }
    Ok((0..cta_group)
        .flat_map(|cta| {
            locations.iter().copied().map(move |(lane, column)| {
                TmemAccess::cell(
                    if cta_group == 2 {
                        Some(first_cta + cta)
                    } else {
                        None
                    },
                    lane,
                    column,
                )
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp4_scale_word_continuations_match_ptx_byte_layout_and_footprints() {
        let accesses = cta2_scale_accesses(0, 16, 2, 6, 128, true, 32).unwrap();
        assert_eq!(accesses.len(), 512);
        for cta in 0..2 {
            for row in 0..128 {
                let pair = &accesses[(cta * 128 + row) * 2..][..2];
                let lane = row % 32;
                let column = 16 + (cta * 128 + row) / 32;
                let access = |byte, column, bytes| TmemAccess {
                    cta: Some(cta),
                    lane,
                    byte,
                    column,
                    bytes,
                };
                assert_eq!(pair[0], access(2, column, 2));
                assert_eq!(pair[1], access(0, column + 8, 4));
            }
        }
        for (id, count, stride) in [(0, 4, 4), (0, 6, 4), (2, 6, 4), (0, 8, 8)] {
            for vector in 0..count {
                let expected = (16 + ((id + vector) / 4 * stride) as u32, (id + vector) % 4);
                assert_eq!(
                    scale_chunk(16, id, vector, stride * 32, 32).unwrap(),
                    expected
                );
            }
        }
        assert!(scale_chunk(16, 0, 4, 0, 32).is_err());
    }
}
