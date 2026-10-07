//! Operand gathers: descriptor walk + element decode (+ block scales), with
//! all memory access supplied by the caller.
//!
//! Legacy source: `engine-rs/src/runtime/tcgen_ops.rs`
//! (`raw_tcgen05_gather_mxf4_matrix`, `raw_tcgen05_gather_mxf4_cta2_matrix`,
//! `raw_tcgen05_gather_sparse_mxf4_e8m0_matrix`, `raw_tcgen05_gather_lut_b`,
//! `raw_tcgen05_gather_f8_shared_matrix`, `raw_tcgen05_gather_tf32_shared_matrix`,
//! `raw_tcgen05_gather_b16_shared_*`, `raw_tcgen05_gather_packed_tmem_a_*`,
//! `raw_tcgen05_gather_scaled_tmem_a`, `raw_tcgen05_gather_mxf8f6f4_matrix`,
//! `raw_tcgen05_read_dense_tmem`, `raw_tcgen05_scatter_dense`).
//!
//! Each gather covers ONE CTA; legacy CTA-pair gathers are the concatenation
//! of per-CTA results in `first_cta..first_cta + cta_group` order. Closures:
//! - `read_shared(offset, buf)`: bytes at a window-relative offset (that CTA);
//! - `read_tmem_byte(lane, column, byte)` / `read_tmem_word(lane, column)`.
//! Closures are invoked in exactly the legacy read order.

use super::layouts::{
    block_scale_address, dense_tmem_cells, lut_b_location, packed_tmem_a_cells, scale_chunk,
    scale_columns, sparse_mxf4_metadata_location, DenseTmemLayout,
};
use super::narrow::{tf32_payload_to_f32, CellDtype, NarrowFormat};
use super::scale::{decode_ue8m0_scale, read_block_scale, ScaleDecoder};
use super::smem_desc::{
    b16_matrix_byte_offset, byte8_matrix_byte_offset, lut_b_row_accesses, masked_row,
    matrix_byte_offset, narrow_shared_atom_accesses, shared_byte_offset, ColumnMask,
    MatrixDescriptor, SharedWindow,
};
use crate::cvt::float4_e2m1fn_bits_to_f32;
use crate::types::{OpError, OpResult};

/// Shared reader: `(window-relative offset, destination bytes)`.
pub trait SharedRead: FnMut(usize, &mut [u8]) -> OpResult<()> {}
impl<T: FnMut(usize, &mut [u8]) -> OpResult<()>> SharedRead for T {}
/// TMEM byte reader: `(lane, column, byte_in_cell) -> byte`.
pub trait TmemByteRead: FnMut(usize, usize, usize) -> OpResult<u8> {}
impl<T: FnMut(usize, usize, usize) -> OpResult<u8>> TmemByteRead for T {}
/// TMEM word reader: `(lane, column) -> little-endian u32 cell`.
pub trait TmemWordRead: FnMut(usize, usize) -> OpResult<u32> {}
impl<T: FnMut(usize, usize) -> OpResult<u32>> TmemWordRead for T {}

fn push_e2m1_bytes(values: &mut Vec<f32>, bytes: &[u8], scale_value: f32, negate: bool) {
    for &packed in bytes {
        for nibble_index in 0..2 {
            let bits = if nibble_index == 0 {
                packed & 0xf
            } else {
                packed >> 4
            };
            let mut value = float4_e2m1fn_bits_to_f32(bits) * scale_value;
            if negate {
                value = -value;
            }
            values.push(value);
        }
    }
}

/// Block-scaled E2M1 rows of one CTA (legacy `raw_tcgen05_gather_mxf4_matrix`
/// for CTA1 with `scale_row_base = 0, scale_rows = rows`; the per-target body
/// of `raw_tcgen05_gather_mxf4_cta2_matrix` with `scale_row_base =
/// target_offset * rows` (joint) or `0`, and `scale_rows` the joint row count).
#[allow(clippy::too_many_arguments)]
pub fn gather_mxf4_rows(
    read_shared: &mut impl SharedRead,
    read_scale: &mut impl TmemByteRead,
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    rows: usize,
    scale_row_base: usize,
    scale_rows: usize,
    scale_address: u32,
    scale_id: usize,
    decode: ScaleDecoder,
    negate: bool,
    k: usize,
    block_elements: usize,
    lanes_per_column: usize,
) -> OpResult<Vec<f32>> {
    let atom_bytes = block_elements / 2;
    let mut values = Vec::with_capacity(
        rows.checked_mul(k)
            .ok_or_else(|| OpError::message("raw mxf4 gather shape overflow"))?,
    );
    for row in 0..rows {
        let matrix_row = scale_row_base
            .checked_add(row)
            .ok_or_else(|| OpError::message("raw mxf4 cta2 scale row overflow"))?;
        let scales = (0..(k / (2 * atom_bytes)))
            .map(|vector_index| {
                read_block_scale(
                    read_scale,
                    scale_address,
                    scale_id,
                    matrix_row,
                    vector_index,
                    decode,
                    scale_rows,
                    lanes_per_column,
                )
            })
            .collect::<OpResult<Vec<_>>>()?;
        for (atom_index, scale_value) in scales.into_iter().enumerate() {
            let mut sub = 0;
            while sub < atom_bytes {
                let column = atom_index * atom_bytes + sub;
                let chunk_bytes = (atom_bytes - sub)
                    .min(16)
                    .min(descriptor.swizzle_atom_bytes - column % descriptor.swizzle_atom_bytes);
                let source_offset =
                    shared_byte_offset(source, descriptor, row, column, chunk_bytes)?;
                let mut packed_storage = [0_u8; 16];
                let packed_bytes = &mut packed_storage[..chunk_bytes];
                read_shared(source_offset, packed_bytes)?;
                push_e2m1_bytes(&mut values, packed_bytes, scale_value, negate);
                sub += chunk_bytes;
            }
        }
    }
    Ok(values)
}

/// `(first_column, column_count)` of the scale window one CTA of a CTA2 FP4
/// gather must own (legacy validation in `raw_tcgen05_gather_mxf4_cta2_matrix`).
#[allow(clippy::too_many_arguments)]
pub fn mxf4_cta2_scale_column_window(
    scale_address: u32,
    scale_id: usize,
    k: usize,
    block_elements: usize,
    rows_per_cta: usize,
    scale_rows_are_joint: bool,
    target_offset: usize,
    lanes_per_column: usize,
) -> OpResult<(usize, usize)> {
    let atom_bytes = block_elements / 2;
    let scale_rows = rows_per_cta * if scale_rows_are_joint { 2 } else { 1 };
    let first_scale_row = if scale_rows_are_joint {
        target_offset
            .checked_mul(rows_per_cta)
            .ok_or_else(|| OpError::message("raw mxf4 cta2 scale row overflow"))?
    } else {
        0
    };
    let last_scale_row = first_scale_row
        .checked_add(rows_per_cta - 1)
        .ok_or_else(|| OpError::message("raw mxf4 cta2 scale row overflow"))?;
    let (_, scale_base_column) = block_scale_address(scale_address)?;
    let first_scale_column = scale_base_column
        .checked_add(first_scale_row / lanes_per_column)
        .ok_or_else(|| OpError::message("raw mxf4 cta2 scale column overflow"))?;
    let (last_address, _) = scale_chunk(
        scale_address,
        scale_id,
        k / (2 * atom_bytes) - 1,
        scale_rows,
        lanes_per_column,
    )?;
    let (_, last_base_column) = block_scale_address(last_address)?;
    let last_scale_column = last_base_column
        .checked_add(last_scale_row / lanes_per_column)
        .ok_or_else(|| OpError::message("raw mxf4 cta2 scale column overflow"))?;
    Ok((
        first_scale_column,
        last_scale_column - first_scale_column + 1,
    ))
}

/// Sparse `mxf4` UE8M0 rows: packed A (64 columns) or dense B (128 columns).
#[allow(clippy::too_many_arguments)]
pub fn gather_sparse_mxf4_e8m0_rows(
    read_shared: &mut impl SharedRead,
    read_scale: &mut impl TmemByteRead,
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    rows: usize,
    columns: usize,
    scale_address: u32,
    scale_id: usize,
    negate: bool,
) -> OpResult<Vec<f32>> {
    if !matches!(columns, 64 | 128) {
        return Err(OpError::message(
            "sparse mxf4 gather columns must be packed-A 64 or dense-B 128",
        ));
    }
    let mut values = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or_else(|| OpError::message("sparse mxf4 gather shape overflow"))?,
    );
    for row in 0..rows {
        let scales = [
            read_block_scale(
                read_scale,
                scale_address,
                scale_id,
                row,
                0,
                decode_ue8m0_scale,
                0,
                32,
            )?,
            read_block_scale(
                read_scale,
                scale_address,
                scale_id,
                row,
                1,
                decode_ue8m0_scale,
                0,
                32,
            )?,
        ];
        for element in 0..columns {
            let byte_in_row = element / 2;
            let source_offset = shared_byte_offset(source, descriptor, row, byte_in_row, 1)?;
            let mut packed = [0_u8; 1];
            read_shared(source_offset, &mut packed)?;
            let bits = if element % 2 == 0 {
                packed[0] & 0xf
            } else {
                packed[0] >> 4
            };
            let mut value =
                float4_e2m1fn_bits_to_f32(bits) * scales[usize::from(element >= columns / 2)];
            if negate {
                value = -value;
            }
            values.push(value);
        }
    }
    Ok(values)
}

/// Sparse `mxf4` metadata code for `(row, chunk)` from a TMEM word reader.
pub fn sparse_mxf4_metadata_code(
    read_word: &mut impl TmemWordRead,
    address: u32,
    row: usize,
    chunk: usize,
) -> OpResult<u8> {
    let (lane, column, nibble) = sparse_mxf4_metadata_location(address, row, chunk)?;
    Ok(((read_word(lane, column)? >> (4 * nibble)) & 0xf) as u8)
}

/// Decode one LUT-B row: 64 3-bit indices from `segment` of the 48-byte
/// compressed row, looked up in the 8-entry E4M3 table.
pub fn decode_lut_b_row(
    compressed: &[u8; 48],
    lookup: &[u8; 8],
    segment: usize,
    negate: bool,
) -> [f32; 64] {
    std::array::from_fn(|k| {
        let bit = segment * 24 * 8 + k * 3;
        let byte = bit / 8;
        let mut index = u16::from(compressed[byte]) >> (bit % 8);
        if bit % 8 > 5 {
            index |= u16::from(compressed[byte + 1]) << (8 - bit % 8);
        }
        let value = NarrowFormat::E4M3.decode_value(lookup[usize::from(index & 7)]);
        if negate {
            -value
        } else {
            value
        }
    })
}

/// LUT-B rows of one CTA (legacy per-CTA body of `raw_tcgen05_gather_lut_b`).
#[allow(clippy::too_many_arguments)]
pub fn gather_lut_b_rows(
    read_shared: &mut impl SharedRead,
    read_word: &mut impl TmemWordRead,
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    segment: usize,
    lookup_address: u32,
    rows: usize,
    negate: bool,
) -> OpResult<Vec<f32>> {
    let mut values = Vec::with_capacity(rows * 64);
    for group in 0..rows / 8 {
        let mut lookup = [0_u8; 8];
        for word in 0..2 {
            let (row, column) = lut_b_location(lookup_address, group, word)?;
            lookup[word * 4..word * 4 + 4].copy_from_slice(&read_word(row, column)?.to_le_bytes());
        }
        for row in group * 8..group * 8 + 8 {
            let mut compressed = [0_u8; 48];
            let mut done = 0;
            lut_b_row_accesses(source, descriptor, row, |offset, count| {
                read_shared(offset, &mut compressed[done..done + count])?;
                done += count;
                Ok(())
            })?;
            values.extend(decode_lut_b_row(&compressed, &lookup, segment, negate));
        }
    }
    Ok(values)
}

/// Geometry check of `raw_tcgen05_gather_f8_shared_matrix`.
pub fn validate_f8_gather(
    cta_group: usize,
    k_extent: usize,
    format: NarrowFormat,
    transpose: bool,
) -> OpResult<()> {
    if !matches!(cta_group, 1 | 2)
        || k_extent % 16 != 0
        || (transpose && format.format().width_bits != 8)
    {
        return Err(OpError::message(
            "invalid narrow MMA gather geometry/transpose",
        ));
    }
    Ok(())
}

/// Narrow (f8/f6/f4) rows of one CTA; masked-out rows read as zero.
#[allow(clippy::too_many_arguments)]
pub fn gather_f8_rows(
    read_shared: &mut impl SharedRead,
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    rows: usize,
    k_extent: usize,
    format: NarrowFormat,
    negate: bool,
    transpose: bool,
    mask: Option<ColumnMask>,
    padded_atoms: bool,
) -> OpResult<Vec<f32>> {
    let mut values = Vec::with_capacity(rows * k_extent);
    for row in 0..rows {
        let Some(row) = masked_row(mask, row) else {
            values.resize(values.len() + k_extent, 0.0);
            continue;
        };
        for atom in 0..k_extent / 16 {
            let decoded = if transpose {
                let mut decoded = [0.0; 16];
                for (i, value) in decoded.iter_mut().enumerate() {
                    let offset =
                        byte8_matrix_byte_offset(source, descriptor, row, atom * 16 + i, true)?;
                    let mut bits = [0_u8];
                    read_shared(offset, &mut bits)?;
                    *value = format.decode_value(bits[0]);
                }
                decoded
            } else {
                let mut bits = [0_u8; 16];
                let mut done = 0;
                narrow_shared_atom_accesses(
                    source,
                    descriptor,
                    row,
                    k_extent,
                    format,
                    atom,
                    |offset, count| {
                        read_shared(offset, &mut bits[done..done + count])?;
                        done += count;
                        Ok(())
                    },
                    padded_atoms,
                )?;
                format.decode_shared_atom(bits)
            };
            values.extend(decoded.into_iter().map(|v| if negate { -v } else { v }));
        }
    }
    Ok(values)
}

/// TF32 rows of one CTA (storage bits, truncated mantissa).
#[allow(clippy::too_many_arguments)]
pub fn gather_tf32_rows(
    read_shared: &mut impl SharedRead,
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    rows: usize,
    columns: usize,
    transpose: bool,
    negate: bool,
    mask: Option<ColumnMask>,
) -> OpResult<Vec<f32>> {
    let mut values = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or_else(|| OpError::message("raw TF32 gather shape overflow"))?,
    );
    for row in 0..rows {
        let Some(row) = masked_row(mask, row) else {
            values.resize(values.len() + columns, 0.0);
            continue;
        };
        for k in 0..columns {
            let source_offset = matrix_byte_offset(source, descriptor, row, k, transpose, 4)?;
            let mut bytes = [0_u8; 4];
            read_shared(source_offset, &mut bytes)?;
            let mut value = tf32_payload_to_f32(u32::from_le_bytes(bytes));
            if negate {
                value = -value;
            }
            values.push(value);
        }
    }
    Ok(values)
}

/// 16-bit rows of one CTA with a caller decoder; masked rows are `default`.
#[allow(clippy::too_many_arguments)]
pub fn gather_b16_rows_with<Scalar: Default>(
    read_shared: &mut impl SharedRead,
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    rows: usize,
    columns: usize,
    transpose: bool,
    mask: Option<ColumnMask>,
    decode: impl Fn(u16) -> OpResult<Scalar>,
) -> OpResult<Vec<Scalar>> {
    let mut values = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or_else(|| OpError::message("raw sparse b16 gather shape overflow"))?,
    );
    for row in 0..rows {
        let Some(row) = masked_row(mask, row) else {
            values.extend(std::iter::repeat_with(Scalar::default).take(columns));
            continue;
        };
        for column in 0..columns {
            let offset = b16_matrix_byte_offset(source, descriptor, row, column, transpose)?;
            let mut bytes = [0_u8; 2];
            read_shared(offset, &mut bytes)?;
            values.push(decode(u16::from_le_bytes(bytes))?);
        }
    }
    Ok(values)
}

/// Packed TMEM A words of one CTA, decoded per word (legacy
/// `raw_tcgen05_gather_packed_tmem_a_columns`). Output is bank-major.
pub fn gather_packed_tmem_a<Scalar, const ELEMENTS_PER_WORD: usize>(
    read_word: &mut impl TmemWordRead,
    address: u32,
    rows: usize,
    layout: DenseTmemLayout,
    columns: usize,
    decode: impl Fn(u32) -> OpResult<[Scalar; ELEMENTS_PER_WORD]>,
) -> OpResult<Vec<Scalar>> {
    let cells = packed_tmem_a_cells(address, rows, layout, columns)?;
    let mut values = Vec::with_capacity(cells.len() * ELEMENTS_PER_WORD);
    for (_, _, _, lane, column) in cells {
        for value in decode(read_word(lane, column)?)? {
            values.push(value);
        }
    }
    Ok(values)
}

/// Interleave two CTAs' packed A preserving N-selected banks across the pair
/// (legacy `raw_tcgen05_gather_packed_tmem_a_cta2_with`). `local[cta]` is each
/// CTA's [`gather_packed_tmem_a`] result for `m / 2` rows.
pub fn merge_cta2_packed_a<Scalar: Copy>(
    local: &[Vec<Scalar>; 2],
    m: usize,
    columns: usize,
    elements_per_word: usize,
    layout: DenseTmemLayout,
) -> Vec<Scalar> {
    let bank_size = m / 2 * columns * elements_per_word;
    let mut values = Vec::with_capacity(m * columns * elements_per_word * layout.packed_a_banks());
    for bank in 0..layout.packed_a_banks() {
        for cta in local {
            values.extend_from_slice(&cta[bank * bank_size..(bank + 1) * bank_size]);
        }
    }
    values
}

/// Geometry check and TMEM column needs of `raw_tcgen05_gather_scaled_tmem_a`:
/// returns `(rows_per_cta, layout, scale_columns)`.
pub fn scaled_tmem_a_geometry(
    m: usize,
    k: usize,
    columns: usize,
    cta_group: usize,
    groups_fit_cluster: bool,
    scale_id: usize,
    block_elements: usize,
    lanes_per_column: usize,
) -> OpResult<(usize, DenseTmemLayout, usize)> {
    let rows = m / cta_group;
    if !(rows == 128 || (cta_group == 2 && rows == 64))
        || columns == 0
        || !k.is_multiple_of(block_elements)
        || !groups_fit_cluster
    {
        return Err(OpError::message("invalid block-scale TMEM A geometry"));
    }
    let scale_columns = scale_columns(rows, scale_id, k / block_elements, lanes_per_column)?;
    let layout = super::layouts::cta1_dense_tmem_layout(rows, true)?;
    Ok((rows, layout, scale_columns))
}

/// One CTA's block-scaled TMEM A (legacy per-CTA body of
/// `raw_tcgen05_gather_scaled_tmem_a`): reads that CTA's packed words and
/// scale bytes and writes rows into `result` (`banks * m * k`, bank-major,
/// CTA `cta_index` owns rows `cta_index * rows..`).
#[allow(clippy::too_many_arguments)]
pub fn gather_scaled_tmem_a_cta(
    read_word: &mut impl TmemWordRead,
    read_scale: &mut impl TmemByteRead,
    result: &mut [f32],
    cta_index: usize,
    address: u32,
    m: usize,
    k: usize,
    rows: usize,
    columns: usize,
    layout: DenseTmemLayout,
    scale_address: u32,
    scale_id: usize,
    block_elements: usize,
    scale_decode: ScaleDecoder,
    negate: bool,
    decode: &impl Fn(&[u32]) -> OpResult<Vec<f32>>,
    lanes_per_column: usize,
) -> OpResult<()> {
    let words = gather_packed_tmem_a(read_word, address, rows, layout, columns, |word| Ok([word]))?;
    for (row, words) in words.chunks_exact(columns).enumerate() {
        let bank = row / rows;
        let row = row % rows;
        let mut values = decode(words)?;
        if values.len() != k {
            return Err(OpError::message(
                "scaled TMEM decoder returned the wrong K extent",
            ));
        }
        for (block, values) in values.chunks_exact_mut(block_elements).enumerate() {
            let scale = read_block_scale(
                read_scale,
                scale_address,
                scale_id,
                row,
                block,
                scale_decode,
                rows,
                lanes_per_column,
            )?;
            for value in values {
                *value *= scale;
                if negate {
                    *value = -*value;
                }
            }
        }
        let start = (bank * m + cta_index * rows + row) * k;
        result[start..start + k].copy_from_slice(&values);
    }
    Ok(())
}

/// Read an `m x n` accumulator window (legacy `raw_tcgen05_read_dense_tmem`).
/// `cell_valid(lane, column)` reports full byte validity; a disabled lane
/// with an invalid cell reads as `disabled_value` instead of erroring.
#[allow(clippy::too_many_arguments)]
pub fn read_dense_window<Scalar: Copy>(
    cell_valid: &mut impl FnMut(usize, usize) -> OpResult<bool>,
    read_cell: &mut impl FnMut(usize, usize) -> OpResult<[u8; 4]>,
    destination_address: u32,
    m: usize,
    n: usize,
    layout: DenseTmemLayout,
    decode: impl Fn([u8; 4]) -> Scalar,
    disabled_value: Scalar,
    disable_output_lane: [u32; 4],
) -> OpResult<Vec<Scalar>> {
    let cells = dense_tmem_cells(destination_address, m, n, layout, None)?;
    let mut values = Vec::with_capacity(cells.len());
    for (_, _, lane, column) in cells {
        let disabled = super::layouts::lane_disabled(&disable_output_lane, lane);
        if disabled && !cell_valid(lane, column)? {
            values.push(disabled_value);
            continue;
        }
        values.push(decode(read_cell(lane, column)?));
    }
    Ok(values)
}

/// Write an `m x n` product over the window, skipping disabled lanes
/// (legacy `raw_tcgen05_scatter_dense`).
#[allow(clippy::too_many_arguments)]
pub fn scatter_dense_window<Scalar: Copy>(
    write_cell: &mut impl FnMut(usize, usize, [u8; 4]) -> OpResult<()>,
    destination_address: u32,
    m: usize,
    n: usize,
    layout: DenseTmemLayout,
    encode: impl Fn(Scalar) -> [u8; 4],
    disable_output_lane: [u32; 4],
    output_values: &[Scalar],
) -> OpResult<()> {
    for (row, col, lane, column) in
        dense_tmem_cells(destination_address, m, n, layout, Some(disable_output_lane))?
    {
        write_cell(lane, column, encode(output_values[row * n + col]))?;
    }
    Ok(())
}

/// Cell map of a CTA-pair accumulator window: `(cta_offset, lane, column,
/// output_index)` per enabled cell (legacy slow paths of
/// `raw_tcgen05_read_cta2_tmem` / `raw_tcgen05_scatter_cta2`; their F32
/// fast paths touch the same cells). `disable = None` keeps disabled cells.
pub fn cta2_window_cells(
    destination_address: u32,
    m: usize,
    n: usize,
    layout: DenseTmemLayout,
    disable_output_lane: Option<[u32; 8]>,
) -> OpResult<Vec<(usize, usize, usize, usize)>> {
    if m != 128 && m != 256 {
        return Err(OpError::message(format!(
            "raw cta_group=2 destination requires M=128 or 256, got {m}"
        )));
    }
    let (base_lane, base_col) = super::layouts::tmem_address(destination_address, 0, 0)?;
    let rows_per_cta = m / 2;
    layout.physical_columns(n)?;
    let mut cells = Vec::with_capacity(m * n);
    for target_offset in 0..2 {
        for row in 0..rows_per_cta {
            for col in 0..n {
                let (lane_delta, physical_col) = layout.location(row, col, rows_per_cta, n)?;
                let lane = base_lane
                    .checked_add(lane_delta)
                    .ok_or_else(|| OpError::message("raw cta_group=2 destination lane overflow"))?;
                if lane >= 128 {
                    return Err(OpError::message(format!(
                        "raw cta_group=2 destination lane {lane} is outside 128 lanes"
                    )));
                }
                if disable_output_lane.is_some_and(|mask| {
                    ((mask[target_offset * 4 + lane / 32] >> (lane % 32)) & 1) != 0
                }) {
                    continue;
                }
                let column = base_col.checked_add(physical_col).ok_or_else(|| {
                    OpError::message("raw cta_group=2 destination column overflow")
                })?;
                cells.push((
                    target_offset,
                    lane,
                    column,
                    (target_offset * rows_per_cta + row) * n + col,
                ));
            }
        }
    }
    Ok(cells)
}

/// `(base_lane, base_col)` of a block-scaled whole-lane destination (legacy
/// `RawMmaWindow::scatter_lane_cells` bounds check).
pub fn lane_cells_window(destination_address: u32, m: usize) -> OpResult<(usize, usize)> {
    let (base_lane, base_col) = super::layouts::tmem_address(destination_address, 0, 0)?;
    let end_lane = base_lane
        .checked_add(m)
        .ok_or_else(|| OpError::message("raw mxf4 destination lane overflow"))?;
    if end_lane > 128 {
        return Err(OpError::message(format!(
            "raw mxf4 destination lanes [{base_lane}, {end_lane}) exceed 128 TMEM lanes"
        )));
    }
    Ok((base_lane, base_col))
}

/// Encode one accumulator row for a whole-lane store.
pub fn encode_lane_row(cell_dtype: CellDtype, row_values: &[f32]) -> Vec<[u8; 4]> {
    row_values
        .iter()
        .map(|&value| cell_dtype.encode(value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_b_row_decode_reads_three_bit_indices_across_bytes() {
        let mut compressed = [0_u8; 48];
        // indices 0..8 repeating: pack 3-bit fields little-endian.
        let mut bits: u128 = 0;
        for k in 0..42 {
            bits |= ((k % 8) as u128) << (3 * k);
        }
        compressed[..16].copy_from_slice(&bits.to_le_bytes());
        let lookup = [0x00, 0x38, 0x40, 0x44, 0x48, 0x4c, 0x50, 0xb8];
        let row = decode_lut_b_row(&compressed, &lookup, 0, false);
        for k in 0..42 {
            assert_eq!(row[k], NarrowFormat::E4M3.decode_value(lookup[k % 8]));
        }
    }

    #[test]
    fn dense_window_scatter_then_read_round_trips() {
        let mut cells = std::collections::HashMap::new();
        let values = (0..64 * 8).map(|i| i as f32).collect::<Vec<_>>();
        scatter_dense_window(
            &mut |lane, column, bytes| {
                cells.insert((lane, column), bytes);
                Ok(())
            },
            0,
            64,
            8,
            DenseTmemLayout::F,
            |v: f32| v.to_le_bytes(),
            [0; 4],
            &values,
        )
        .unwrap();
        assert!(cells.contains_key(&(32, 0)) && !cells.contains_key(&(16, 0)));
        let read = read_dense_window(
            &mut |_, _| Ok(true),
            &mut |lane, column| Ok(cells[&(lane, column)]),
            0,
            64,
            8,
            DenseTmemLayout::F,
            f32::from_le_bytes,
            f32::NAN,
            [0; 4],
        )
        .unwrap();
        assert_eq!(read, values);
    }
}
