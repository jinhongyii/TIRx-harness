//! Tiled TMA transfer planning: box -> (global byte runs, shared byte runs)
//! iteration, OOB predicate and fill, sub-byte store fragments, gather4, and
//! pure payload movement over byte slices.
//!
//! Legacy source: `engine-rs/src/runtime/tensor_map.rs`
//! (`TensorMapTransferTemplate::{compile, shared_program, bind_global}`,
//! `RawTmaG2cTransferPlan::{new, gather4}`, `raw_tma_g2c_layout`,
//! `raw_tma_gather4_layout`, `RawTmaS2gTransferPlan::{new, execute}`,
//! `append_s2g_bits`, `expand_u6_source_runs`,
//! `materialize_raw_tma_g2c_payload`, `tma_f32_to_tf32`).
//!
//! Offsets: global run `byte_offset`s are relative to the TensorMap view
//! start; shared run `byte_offset`s are relative to the shared pointer of the
//! issuing lane (the caller adds that pointer's offset and bounds-checks
//! `[0, extent)`); `payload_offset`s index the transfer payload.

use std::sync::Arc;

use super::bulk::copy_report_matches_runs;
use super::descriptor::{
    Fp4SharedLayout, RawTmaReductionOp, TensorMapElementType, TensorMapFillMode, TmaReduction,
};
use super::swizzle::{shared_byte_offset, SwizzleAtomicity};
use super::tensor_map::{tensor_map_geometry_from_metadata, TensorMapGeometry, TensorMapLayout};
use crate::cvt::f32_to_tf32;
use crate::scalar::PTX_OOB_NAN;
use crate::types::{OpError, OpResult};

/// One contiguous copy between memory bytes and payload bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ByteRun {
    pub byte_offset: usize,
    pub payload_offset: usize,
    pub byte_len: usize,
}

/// Shared-memory runs for one 128-byte pointer phase.
#[derive(Clone, Debug)]
pub struct SharedRunProgram {
    pub runs: Arc<[ByteRun]>,
    /// Bytes past the pointer that the program touches.
    pub byte_extent: usize,
}

#[derive(Clone, Debug)]
struct RowTemplate {
    coordinate_deltas: Box<[i64]>,
    global_outer_byte_delta: usize,
    payload_offset: usize,
}

/// Origin-independent part of a tiled transfer, compiled once per map.
#[derive(Clone, Debug)]
pub struct TransferTemplate {
    pub geometry: TensorMapGeometry,
    rows: Arc<[RowTemplate]>,
    source_row_order: Arc<[usize]>,
    max_coordinate_deltas: Box<[i64]>,
    shared_programs: Box<[SharedRunProgram]>,
    pub payload_len: usize,
}

fn push_merged(runs: &mut Vec<ByteRun>, run: ByteRun, overflow: &'static str) -> OpResult<()> {
    if let Some(previous) = runs.last_mut() {
        let bytes_contiguous = previous.byte_offset.checked_add(previous.byte_len) == Some(run.byte_offset);
        let payload_contiguous =
            previous.payload_offset.checked_add(previous.byte_len) == Some(run.payload_offset);
        if bytes_contiguous && payload_contiguous {
            previous.byte_len = previous
                .byte_len
                .checked_add(run.byte_len)
                .ok_or_else(|| OpError::message(overflow))?;
            return Ok(());
        }
    }
    runs.push(run);
    Ok(())
}

impl TransferTemplate {
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        traversal_shape: &[usize],
        element_strides: &[usize],
        global_strides: &[usize],
        element_bits: usize,
        fp4_shared_layout: Option<Fp4SharedLayout>,
        swizzle_bytes: Option<usize>,
        swizzle_atomicity: SwizzleAtomicity,
    ) -> OpResult<Self> {
        let geometry =
            tensor_map_geometry_from_metadata(traversal_shape, element_bits, fp4_shared_layout)?;
        let unit_count = geometry
            .outer_count
            .checked_mul(geometry.inner_units)
            .ok_or_else(|| OpError::message("TensorMap transfer unit count overflow"))?;
        let payload_len = unit_count
            .checked_mul(geometry.unit_bytes)
            .ok_or_else(|| OpError::message("TensorMap transfer payload size overflow"))?;
        let row_payload_len = geometry
            .inner_units
            .checked_mul(geometry.unit_bytes)
            .ok_or_else(|| OpError::message("TensorMap row payload size overflow"))?;

        let max_coordinate_deltas = traversal_shape
            .iter()
            .zip(element_strides)
            .map(|(extent, stride)| {
                extent
                    .checked_sub(1)
                    .and_then(|value| value.checked_mul(*stride))
                    .and_then(|value| i64::try_from(value).ok())
                    .ok_or_else(|| OpError::message("TensorMap coordinate delta overflow"))
            })
            .collect::<Result<Box<[_]>, _>>()?;

        let mut rows = Vec::with_capacity(geometry.outer_count);
        for outer_linear in 0..geometry.outer_count {
            let mut linear = outer_linear;
            let mut coordinate_deltas = Vec::with_capacity(traversal_shape.len().saturating_sub(1));
            let mut global_outer_byte_delta = 0_usize;
            for axis in 1..traversal_shape.len() {
                let coordinate = linear % traversal_shape[axis];
                linear /= traversal_shape[axis];
                let delta = coordinate
                    .checked_mul(element_strides[axis])
                    .ok_or_else(|| OpError::message("TensorMap coordinate delta overflow"))?;
                global_outer_byte_delta = global_outer_byte_delta
                    .checked_add(
                        delta
                            .checked_mul(global_strides[axis - 1])
                            .ok_or_else(|| OpError::message("TensorMap stride offset overflow"))?,
                    )
                    .ok_or_else(|| OpError::message("TensorMap byte offset overflow"))?;
                coordinate_deltas.push(
                    i64::try_from(delta)
                        .map_err(|_| OpError::message("TensorMap coordinate delta overflow"))?,
                );
            }
            rows.push(RowTemplate {
                coordinate_deltas: coordinate_deltas.into_boxed_slice(),
                global_outer_byte_delta,
                payload_offset: outer_linear
                    .checked_mul(row_payload_len)
                    .ok_or_else(|| OpError::message("TensorMap payload offset overflow"))?,
            });
        }
        let mut source_row_order = (0..rows.len()).collect::<Vec<_>>();
        source_row_order.sort_unstable_by_key(|row| {
            (rows[*row].global_outer_byte_delta, rows[*row].payload_offset)
        });

        let phase_count = swizzle_bytes.map_or(1, |bytes| {
            if bytes == 96 {
                2
            } else {
                bytes / swizzle_atomicity.bytes()
            }
        });
        let mut shared_programs = Vec::with_capacity(phase_count);
        for pointer_phase in 0..phase_count {
            let absolute_base = pointer_phase
                .checked_mul(128)
                .ok_or_else(|| OpError::message("TensorMap shared pointer phase overflow"))?;
            let mut runs = Vec::<ByteRun>::new();
            let mut byte_extent = 0_usize;
            for outer_linear in 0..geometry.outer_count {
                for inner_unit in 0..geometry.inner_units {
                    let unit_index = outer_linear
                        .checked_mul(geometry.inner_units)
                        .and_then(|base| base.checked_add(inner_unit))
                        .ok_or_else(|| OpError::message("TensorMap transfer index overflow"))?;
                    let payload_offset = unit_index
                        .checked_mul(geometry.unit_bytes)
                        .ok_or_else(|| OpError::message("TensorMap payload offset overflow"))?;
                    // Interleave slices split at 16B atoms. Accepted 8B-flip
                    // layouts already have transfer units of at most 8B.
                    let atom_bytes = geometry.unit_bytes.min(16);
                    for atom in (0..geometry.unit_bytes).step_by(atom_bytes) {
                        let payload_offset = payload_offset + atom;
                        let byte_offset = shared_byte_offset(
                            swizzle_bytes,
                            swizzle_atomicity,
                            outer_linear,
                            inner_unit * geometry.unit_stride_bytes + atom,
                            geometry.inner_row_bytes,
                            absolute_base,
                        )?;
                        byte_extent = byte_extent.max(
                            byte_offset
                                .checked_add(atom_bytes)
                                .ok_or_else(|| OpError::message("TensorMap shared extent overflow"))?,
                        );
                        push_merged(
                            &mut runs,
                            ByteRun {
                                byte_offset,
                                payload_offset,
                                byte_len: atom_bytes,
                            },
                            "TensorMap shared run size overflow",
                        )?;
                    }
                }
            }
            shared_programs.push(SharedRunProgram {
                runs: Arc::from(runs),
                byte_extent,
            });
        }

        Ok(Self {
            geometry,
            rows: Arc::from(rows),
            source_row_order: Arc::from(source_row_order),
            max_coordinate_deltas,
            shared_programs: shared_programs.into_boxed_slice(),
            payload_len,
        })
    }

    /// Shared runs for a destination whose absolute byte address is
    /// `absolute_base` (only `(absolute_base / 128) % phases` matters).
    pub fn shared_program(&self, absolute_base: usize) -> &SharedRunProgram {
        let phase = if self.shared_programs.len() == 1 {
            0
        } else {
            (absolute_base / 128) % self.shared_programs.len()
        };
        &self.shared_programs[phase]
    }

    /// Global runs of the in-bounds part of the box at `origin`. Rows and
    /// inner elements outside the tensor are skipped (their payload bytes
    /// keep the OOB fill).
    pub fn bind_global(&self, map: &TensorMapLayout, origin: &[i64]) -> OpResult<Vec<ByteRun>> {
        if origin.len() != map.global_shape.len() {
            return Err(OpError::message(format!(
                "TensorMap expected {} coordinates, got {}",
                map.global_shape.len(),
                origin.len()
            )));
        }
        map.validate_u6_origin(origin)?;
        for (&coordinate, &max_delta) in origin.iter().zip(&self.max_coordinate_deltas) {
            coordinate
                .checked_add(max_delta)
                .ok_or_else(|| OpError::message("TensorMap coordinate overflow"))?;
        }

        let extent = map.traversal_shape[0] as i128;
        let origin_inner = i128::from(origin[0]);
        let global_extent = map.global_shape[0] as i128;
        let step = map.element_strides[0] as i128;
        let ceil_div = |value: i128| -(-value).div_euclid(step);
        let start = ceil_div(-origin_inner).clamp(0, extent);
        let end = ceil_div(global_extent - origin_inner).clamp(0, extent);
        let packed = self.geometry.packed_elements as i128;
        if start % packed != 0 || end % packed != 0 {
            return Err(OpError::message(
                "TensorMap valid FP4 interval is not aligned to a packed transfer unit",
            ));
        }
        if end <= start {
            return Ok(Vec::new());
        }
        let transfer_bits = map.transfer_element_bits() as i128;
        let mut inner_runs = Vec::new();
        let first_unit = (start / packed) as usize;
        let end_unit = (end / packed) as usize;
        let units_per_run = if step == 1 { end_unit - first_unit } else { 1 };
        for unit in (first_unit..end_unit).step_by(units_per_run) {
            let coordinate = origin_inner + unit as i128 * packed * step;
            let byte_offset = usize::try_from(coordinate * transfer_bits / 8)
                .map_err(|_| OpError::message("TensorMap inner byte offset overflow"))?;
            inner_runs.push(ByteRun {
                byte_offset,
                payload_offset: unit
                    .checked_mul(self.geometry.unit_bytes)
                    .ok_or_else(|| OpError::message("TensorMap inner payload offset overflow"))?,
                byte_len: units_per_run
                    .checked_mul(self.geometry.unit_bytes)
                    .ok_or_else(|| OpError::message("TensorMap inner run size overflow"))?,
            });
        }
        let mut origin_outer_byte_offset = 0_i128;
        for axis in 1..origin.len() {
            origin_outer_byte_offset = origin_outer_byte_offset
                .checked_add(
                    i128::from(origin[axis])
                        .checked_mul(map.global_strides[axis - 1] as i128)
                        .ok_or_else(|| OpError::message("TensorMap stride offset overflow"))?,
                )
                .ok_or_else(|| OpError::message("TensorMap byte offset overflow"))?;
        }

        let mut runs = Vec::<ByteRun>::new();
        for &row_index in self.source_row_order.iter() {
            let row = &self.rows[row_index];
            let mut in_bounds = true;
            for axis in 1..origin.len() {
                let coordinate = origin[axis]
                    .checked_add(row.coordinate_deltas[axis - 1])
                    .ok_or_else(|| OpError::message("TensorMap coordinate overflow"))?;
                if coordinate < 0
                    || usize::try_from(coordinate).map_or(true, |value| value >= map.global_shape[axis])
                {
                    in_bounds = false;
                    break;
                }
            }
            if !in_bounds {
                continue;
            }
            for inner in &inner_runs {
                let byte_offset = usize::try_from(
                    origin_outer_byte_offset
                        .checked_add(row.global_outer_byte_delta as i128)
                        .and_then(|value| value.checked_add(inner.byte_offset as i128))
                        .ok_or_else(|| OpError::message("TensorMap byte offset overflow"))?,
                )
                .map_err(|_| OpError::message("TensorMap byte offset overflow"))?;
                let payload_offset = row
                    .payload_offset
                    .checked_add(inner.payload_offset)
                    .ok_or_else(|| OpError::message("TensorMap payload offset overflow"))?;
                push_merged(
                    &mut runs,
                    ByteRun {
                        byte_offset,
                        payload_offset,
                        byte_len: inner.byte_len,
                    },
                    "TensorMap source run size overflow",
                )?;
            }
        }
        Ok(runs)
    }
}

/// Global-to-shared (load) transfer plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct G2sPlan {
    pub geometry: TensorMapGeometry,
    /// Global runs relative to the TensorMap view.
    pub source_runs: Vec<ByteRun>,
    /// Shared runs relative to the destination pointer.
    pub destination_runs: Vec<ByteRun>,
    pub payload_len: usize,
    pub fill_mode: TensorMapFillMode,
}

impl G2sPlan {
    /// Bytes delivered to each destination CTA (`complete_tx` count).
    pub fn bytes_per_target(&self) -> OpResult<u64> {
        u64::try_from(self.payload_len)
            .map_err(|_| OpError::message("raw TMA delivered-byte count overflow"))
    }

    /// Bytes past the destination pointer that the plan writes.
    pub fn destination_extent(&self) -> usize {
        runs_extent(&self.destination_runs)
    }
}

pub fn runs_extent(runs: &[ByteRun]) -> usize {
    runs.iter()
        .map(|run| run.byte_offset + run.byte_len)
        .max()
        .unwrap_or(0)
}

/// Shared checks of every global-to-shared plan (legacy `raw_tma_g2c_layout`
/// minus lane/pointer/multicast plumbing). Multicast masks are validated by
/// [`super::bulk::multicast_target_ctas`].
pub(crate) fn g2s_layout_checks(map: &TensorMapLayout, inner_origin: i64) -> OpResult<TensorMapGeometry> {
    let geometry = map.geometry()?;
    if let Some(required_alignment) = map.fp4_origin_alignment()? {
        if inner_origin.rem_euclid(required_alignment) != 0 {
            return Err(OpError::message(format!(
                "FP4 TensorMap inner origin must be a multiple of {required_alignment}"
            )));
        }
    }
    map.validate_swizzle_direction(true)?;
    Ok(geometry)
}

/// Plan a tiled `cp.async.bulk.tensor` global-to-shared load.
/// `shared_absolute_base` is the absolute byte address of the destination.
pub fn plan_tiled_g2s(
    map: &TensorMapLayout,
    origin: &[i64],
    shared_absolute_base: usize,
) -> OpResult<G2sPlan> {
    if map.im2col.is_some() {
        return Err(OpError::message("tiled load requires a tiled TensorMap"));
    }
    let geometry = g2s_layout_checks(map, origin.first().copied().unwrap_or(0))?;
    let template = &map.transfer_template;
    let source_runs = template.bind_global(map, origin)?;
    let destination_runs = template.shared_program(shared_absolute_base).runs.to_vec();
    Ok(G2sPlan {
        geometry,
        source_runs,
        destination_runs,
        payload_len: template.payload_len,
        fill_mode: map.fill_mode,
    })
}

/// Plan a `cp.async.bulk.tensor.2d.tile::gather4` load of four rows.
pub fn plan_gather4_g2s(
    map: &TensorMapLayout,
    column: i64,
    rows: &[i64],
    shared_absolute_base: usize,
) -> OpResult<G2sPlan> {
    if map.global_shape.len() != 2 || map.box_shape.len() != 2 || map.box_shape[1] != 1 {
        return Err(OpError::message(
            "TensorMap gather4 requires a rank-2 map with outer box extent one",
        ));
    }
    if rows.len() != 4 {
        return Err(OpError::message(format!(
            "TensorMap gather4 requires exactly four rows, got {}",
            rows.len()
        )));
    }
    let geometry = map.geometry()?;
    if let Some(required_alignment) = map.fp4_origin_alignment()? {
        if column.rem_euclid(required_alignment) != 0 {
            return Err(OpError::message(format!(
                "FP4 TensorMap gather4 inner origin must be a multiple of {required_alignment}"
            )));
        }
    }
    if geometry.outer_count != 1 {
        return Err(OpError::message(
            "TensorMap gather4 source bounding box must contain one row",
        ));
    }
    map.validate_swizzle_direction(true)?;
    let template = &map.transfer_template;
    let payload_len = template
        .payload_len
        .checked_mul(4)
        .ok_or_else(|| OpError::message("gather4 payload size overflow"))?;
    let mut source_runs = Vec::new();
    let mut destination_runs = Vec::new();
    for (row, &row_coordinate) in rows.iter().enumerate() {
        let payload_base = row * template.payload_len;
        for mut run in template.bind_global(map, &[column, row_coordinate])? {
            run.payload_offset += payload_base;
            source_runs.push(run);
        }
        for unit in 0..geometry.inner_units {
            let box_offset = map.shared_byte_offset(
                row,
                unit * geometry.unit_stride_bytes,
                geometry.inner_row_bytes,
                shared_absolute_base,
            )?;
            destination_runs.push(ByteRun {
                byte_offset: box_offset,
                payload_offset: payload_base + unit * geometry.unit_bytes,
                byte_len: geometry.unit_bytes,
            });
        }
    }
    let mut geometry = geometry;
    geometry.outer_count = 4;
    Ok(G2sPlan {
        geometry,
        source_runs,
        destination_runs,
        payload_len,
        fill_mode: map.fill_mode,
    })
}

/// TMA's TF32 conversion canonicalizes input NaNs (SM100 GPU bit oracle).
/// OOB fill is generated separately and must not pass through conversion.
pub fn tma_f32_to_tf32(value: f32) -> f32 {
    if value.is_nan() {
        f32::from_bits(0x7fff_e000)
    } else {
        f32_to_tf32(value)
    }
}

fn slice_at<'a>(bytes: &'a [u8], offset: usize, len: usize, label: &str) -> OpResult<&'a [u8]> {
    offset
        .checked_add(len)
        .and_then(|end| bytes.get(offset..end))
        .ok_or_else(|| {
            OpError::message(format!(
                "{label} bytes {offset}..{} exceed {} available bytes",
                offset.saturating_add(len),
                bytes.len()
            ))
        })
}

fn slice_at_mut<'a>(
    bytes: &'a mut [u8],
    offset: usize,
    len: usize,
    label: &str,
) -> OpResult<&'a mut [u8]> {
    let available = bytes.len();
    offset
        .checked_add(len)
        .and_then(|end| bytes.get_mut(offset..end))
        .ok_or_else(|| {
            OpError::message(format!(
                "{label} bytes {offset}..{} exceed {available} available bytes",
                offset.saturating_add(len)
            ))
        })
}

/// Build the load payload from the TensorMap view bytes `global`: OOB fill
/// (zero or PTX OOB-NaN), in-bounds source bytes, then TF32 rounding of
/// in-bounds data for TF32 maps. Also evaluates the `.report` pattern on the
/// raw source bytes (`report_pattern = 0` disables it).
pub fn materialize_g2s_payload(
    global: &[u8],
    element_type: TensorMapElementType,
    source_runs: &[ByteRun],
    payload_len: usize,
    unit_bytes: usize,
    fill_mode: TensorMapFillMode,
    report_pattern: u32,
) -> OpResult<(Vec<u8>, bool)> {
    let mut payload = vec![0_u8; payload_len];
    if fill_mode == TensorMapFillMode::OobNan {
        if unit_bytes < 2 || !unit_bytes.is_multiple_of(2) {
            return Err(OpError::message(
                "TensorMap OOB-NaN fill requires an even floating-point element width",
            ));
        }
        for unit in payload.chunks_exact_mut(unit_bytes) {
            for chunk in unit.chunks_exact_mut(2) {
                chunk.copy_from_slice(&PTX_OOB_NAN.to_le_bytes());
            }
        }
    }
    for run in source_runs {
        let source = slice_at(global, run.byte_offset, run.byte_len, "TensorMap global source")?;
        payload[run.payload_offset..run.payload_offset + run.byte_len].copy_from_slice(source);
    }
    let reported = copy_report_matches_runs(
        report_pattern,
        source_runs.iter().map(|run| {
            (
                run.byte_offset,
                &payload[run.payload_offset..run.payload_offset + run.byte_len],
            )
        }),
    )?;
    // FTZ descriptor types affect tensor reductions, not copies. Convert only
    // in-bounds TF32 source data; hardware leaves OOB-NaN fill bits untouched.
    if matches!(
        element_type,
        TensorMapElementType::Tf32 | TensorMapElementType::Tf32Ftz
    ) {
        for run in source_runs {
            for element in
                payload[run.payload_offset..run.payload_offset + run.byte_len].chunks_exact_mut(4)
            {
                let raw: [u8; 4] = element.try_into().expect("four-byte TMA float element");
                element.copy_from_slice(&tma_f32_to_tf32(f32::from_le_bytes(raw)).to_le_bytes());
            }
        }
    }
    Ok((payload, reported))
}

/// Execute a load plan entirely over byte slices: `global` is the TensorMap
/// view, `shared` the destination CTA's shared bytes with the destination
/// pointer at `shared_offset`. Returns whether `.report` matched.
pub fn execute_g2s(
    map: &TensorMapLayout,
    plan: &G2sPlan,
    global: &[u8],
    shared: &mut [u8],
    shared_offset: usize,
    report_pattern: u32,
) -> OpResult<bool> {
    let (payload, reported) = materialize_g2s_payload(
        global,
        map.element_type,
        &plan.source_runs,
        plan.payload_len,
        plan.geometry.unit_bytes,
        plan.fill_mode,
        report_pattern,
    )?;
    scatter_payload(shared, shared_offset, &plan.destination_runs, &payload)?;
    Ok(reported)
}

/// Copy `payload` bytes to `memory[base + run.byte_offset ..]` for each run.
pub fn scatter_payload(memory: &mut [u8], base: usize, runs: &[ByteRun], payload: &[u8]) -> OpResult<()> {
    for run in runs {
        let offset = base
            .checked_add(run.byte_offset)
            .ok_or_else(|| OpError::message("TensorMap shared offset overflow"))?;
        slice_at_mut(memory, offset, run.byte_len, "TensorMap destination")?
            .copy_from_slice(&payload[run.payload_offset..run.payload_offset + run.byte_len]);
    }
    Ok(())
}

/// Gather `memory[base + run.byte_offset ..]` into a payload of `payload_len`.
pub fn gather_payload(memory: &[u8], base: usize, runs: &[ByteRun], payload_len: usize) -> OpResult<Vec<u8>> {
    let mut payload = vec![0_u8; payload_len];
    for run in runs {
        let offset = base
            .checked_add(run.byte_offset)
            .ok_or_else(|| OpError::message("TensorMap shared offset overflow"))?;
        payload[run.payload_offset..run.payload_offset + run.byte_len]
            .copy_from_slice(slice_at(memory, offset, run.byte_len, "TensorMap source")?);
    }
    Ok(payload)
}

/// Sub-byte (FP4/U6) store fragment: `mask`ed bits of one payload byte
/// written into one global byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BitFragment {
    pub byte_offset: usize,
    pub payload_offset: usize,
    pub source_shift: usize,
    pub target_shift: usize,
    pub mask: u8,
}

pub(crate) fn append_s2g_bits(
    map: &TensorMapLayout,
    coordinates: &[i64],
    payload_offset: usize,
    fragments: &mut Vec<BitFragment>,
) -> OpResult<()> {
    let mut coordinates = coordinates.to_vec();
    let bits = map.element_bits;
    let packed_elements = map.transfer_template.geometry.packed_elements;
    for packed_index in 0..packed_elements {
        if packed_index != 0 {
            coordinates[0] = coordinates[0]
                .checked_add(1)
                .ok_or_else(|| OpError::message("sub-byte TensorMap coordinate overflow"))?;
        }
        if !map.coordinates_in_bounds(&coordinates) {
            continue;
        }
        let (byte_offset, target_shift) = map.global_byte_offset(&coordinates)?;
        // U4 input is nibble-packed; U6 input has one value per shared byte.
        let source_bit = packed_index * if bits == 6 { 8 } else { bits };
        let mut consumed = 0;
        while consumed < bits {
            let target_bit = target_shift + consumed;
            let width = (bits - consumed).min(8 - target_bit % 8);
            fragments.push(BitFragment {
                byte_offset: byte_offset + target_bit / 8,
                payload_offset: payload_offset + source_bit / 8,
                source_shift: source_bit % 8 + consumed,
                target_shift: target_bit % 8,
                mask: ((1_u16 << width) - 1) as u8,
            });
            consumed += width;
        }
    }
    Ok(())
}

/// The U6 load template names twelve payload bytes per sixteen-byte atom.
/// Stores instead read all sixteen bytes (one low-six-bit value in each).
pub(crate) fn expand_u6_source_runs(runs: &mut [ByteRun], payload_len: usize) -> OpResult<usize> {
    let expand = |bytes: usize| {
        debug_assert!(bytes.is_multiple_of(12));
        (bytes / 12)
            .checked_mul(16)
            .ok_or_else(|| OpError::message("U6 TensorMap source payload overflow"))
    };
    for run in runs {
        debug_assert_eq!(run.byte_len, 12);
        run.payload_offset = expand(run.payload_offset)?;
        run.byte_len = 16;
    }
    expand(payload_len)
}

/// Shared-to-global (store / reduce) transfer plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S2gPlan {
    /// Shared runs relative to the source pointer.
    pub source_runs: Vec<ByteRun>,
    /// Whole-byte global runs relative to the TensorMap view.
    pub destination_runs: Vec<ByteRun>,
    /// Masked sub-byte global writes (FP4/U6 maps).
    pub destination_bits: Vec<BitFragment>,
    /// Access-unit size used for footprints (16 for U6).
    pub unit_bytes: usize,
    pub payload_len: usize,
}

impl S2gPlan {
    pub fn source_extent(&self) -> usize {
        runs_extent(&self.source_runs)
    }
}

/// Plan a tiled `cp.async.bulk.tensor` / `cp.reduce.async.bulk.tensor`
/// shared-to-global transfer.
pub fn plan_tiled_s2g(
    map: &TensorMapLayout,
    origin: &[i64],
    shared_absolute_base: usize,
) -> OpResult<S2gPlan> {
    // PTX tensor-copy direction restrictions: tiled shared-to-global
    // copies/reductions cannot start at negative tensor coordinates.
    if origin.iter().any(|coordinate| *coordinate < 0) {
        return Err(OpError::message(
            "tiled TMA store requires nonnegative starting coordinates",
        ));
    }
    if map.im2col.is_some() {
        return Err(OpError::message("tiled store requires a tiled TensorMap"));
    }
    if map.fp4_shared_layout == Some(Fp4SharedLayout::Align16Padded) {
        return Err(OpError::message(
            "align16 padded FP4 TensorMap does not support shared-to-global Tensor Copy",
        ));
    }
    let template = &map.transfer_template;
    let geometry = template.geometry;
    map.validate_swizzle_direction(false)?;
    let mut source_runs = template.shared_program(shared_absolute_base).runs.to_vec();

    let mut payload_len = template.payload_len;
    let unit_bytes = if map.element_bits == 6 {
        payload_len = expand_u6_source_runs(&mut source_runs, payload_len)?;
        16
    } else {
        geometry.unit_bytes.min(16)
    };
    let mut destination_bits = Vec::new();
    let destination_runs = if matches!(map.transfer_element_bits(), 4 | 6) {
        if origin.len() != map.global_shape.len() {
            return Err(OpError::message(format!(
                "TensorMap expected {} coordinates, got {}",
                map.global_shape.len(),
                origin.len()
            )));
        }
        map.validate_u6_origin(origin)?;
        let mut coordinates = vec![0_i64; map.global_shape.len()];
        for outer_linear in 0..geometry.outer_count {
            let outer = map.outer_coordinates(outer_linear);
            for inner_unit in 0..geometry.inner_units {
                let payload_offset = outer_linear
                    .checked_mul(geometry.inner_units)
                    .and_then(|base| base.checked_add(inner_unit))
                    .and_then(|unit| unit.checked_mul(unit_bytes))
                    .ok_or_else(|| OpError::message("TensorMap payload offset overflow"))?;
                let inner_element = inner_unit * geometry.packed_elements;
                map.global_coordinates_into(origin, inner_element, &outer, &mut coordinates)?;
                append_s2g_bits(map, &coordinates, payload_offset, &mut destination_bits)?;
            }
        }
        Vec::new()
    } else {
        template.bind_global(map, origin)?
    };

    Ok(S2gPlan {
        source_runs,
        destination_runs,
        destination_bits,
        unit_bytes,
        payload_len,
    })
}

/// Apply a plain (non-reducing) store plan to the TensorMap view bytes.
pub fn apply_s2g_copy(global: &mut [u8], plan: &S2gPlan, payload: &[u8]) -> OpResult<()> {
    scatter_payload(global, 0, &plan.destination_runs, payload)?;
    for fragment in &plan.destination_bits {
        let mask = fragment.mask << fragment.target_shift;
        let source_bits = (payload[fragment.payload_offset] >> fragment.source_shift) & fragment.mask;
        let byte = slice_at_mut(global, fragment.byte_offset, 1, "TensorMap destination")?;
        byte[0] = (byte[0] & !mask) | ((source_bits << fragment.target_shift) & mask);
    }
    Ok(())
}

/// Execute a plain store plan over byte slices (`shared` holds the source
/// CTA's shared bytes with the source pointer at `shared_offset`).
pub fn execute_s2g_copy(plan: &S2gPlan, shared: &[u8], shared_offset: usize, global: &mut [u8]) -> OpResult<()> {
    let payload = gather_payload(shared, shared_offset, &plan.source_runs, plan.payload_len)?;
    apply_s2g_copy(global, plan, &payload)
}

/// One element-wise reduction `global[byte_offset..] op= source`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TmaReductionElement {
    pub byte_offset: usize,
    pub source: Vec<u8>,
}

/// Resolve the reduction and split a store plan's destination runs into
/// one reduction per element (legacy `RawTmaS2gTransferPlan::execute` with a
/// reduction). Apply each element with `TmaReduction::apply` (atomic family).
pub fn s2g_reduction_elements(
    map: &TensorMapLayout,
    plan: &S2gPlan,
    payload: &[u8],
    operation: RawTmaReductionOp,
) -> OpResult<(TmaReduction, Vec<TmaReductionElement>)> {
    let reduction = operation.resolve(map.element_type)?;
    let element_bytes = map.element_bits / 8;
    let mut elements = Vec::new();
    for run in &plan.destination_runs {
        let bytes = &payload[run.payload_offset..run.payload_offset + run.byte_len];
        for element_offset in (0..run.byte_len).step_by(element_bytes) {
            elements.push(TmaReductionElement {
                byte_offset: run.byte_offset + element_offset,
                source: bytes[element_offset..element_offset + element_bytes].to_vec(),
            });
        }
    }
    Ok((reduction, elements))
}

#[cfg(test)]
#[path = "tiled_tests.rs"]
mod tests;
