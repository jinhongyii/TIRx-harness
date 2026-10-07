//! TMEM address decoding and the lane/column layouts tcgen05 instructions use.
//!
//! Legacy source: `engine-rs/src/runtime/tcgen_ops.rs` (`raw_tcgen05_address`,
//! `raw_tcgen05_block_scale_address`, `RawTcgenLdstShape`,
//! `raw_tcgen05_ldst_location`, tcgen05.cp shapes / destination lanes / word
//! decoding / b6 unpack, `RawTcgenDenseTmemLayout`, Layout F lane, CTA2
//! banking, scale-factor chunks and `RawTcgenScaleLayout`, sparse metadata
//! locations, LUT-B location) and `runtime/instructions/tcgen05.rs`
//! (ld/st register counts and legal `.num` values, cp shape codes).

use crate::types::{OpError, OpResult};

/// Decode a TMEM address `(lane << 16) | column` with signed offsets, wrapping
/// each field mod 2^16 (legacy `raw_tcgen05_address`).
pub fn tmem_address(address: u32, row_offset: i64, col_offset: i64) -> OpResult<(usize, usize)> {
    let row = (i64::from((address >> 16) & 0xffff) + row_offset).rem_euclid(1_i64 << 16);
    let col = (i64::from(address & 0xffff) + col_offset).rem_euclid(1_i64 << 16);
    Ok((
        usize::try_from(row).map_err(|_| OpError::message("negative TCGEN row"))?,
        usize::try_from(col).map_err(|_| OpError::message("negative TCGEN column"))?,
    ))
}

/// Physical location of a block-scale TMEM operand. Bits 30-31 carry the
/// runtime SFA/SFB sub-column selector and are not part of the lane.
pub fn block_scale_address(address: u32) -> OpResult<(usize, usize)> {
    tmem_address(address & !0xc000_0000_u32, 0, 0)
}

/// `tirx.cuda.get_tmem_addr` (legacy `runtime/tmem.rs::get_tmem_addr`).
pub fn get_tmem_addr(encoded: u32, row_offset: i32, column_offset: u32) -> u32 {
    let row = ((encoded >> 16) & 0xffff).wrapping_add(row_offset as u32) & 0xffff;
    let column = (encoded & 0xffff).wrapping_add(column_offset) & 0xffff;
    (row << 16) | column
}

/// Number of 32-bit TMEM columns a byte span starting at `byte_in_cell`
/// touches (legacy `runtime/tmem.rs::tmem_access_column_count`).
pub fn tmem_access_column_count(byte_in_cell: usize, access_bytes: usize) -> OpResult<usize> {
    byte_in_cell
        .checked_add(access_bytes)
        .and_then(|bytes| bytes.checked_add(3))
        .map(|bytes| bytes / 4)
        .ok_or_else(|| OpError::message("TMEM access column span overflow"))
}

/// First CTA of a `cta_group` (`cta & !(cta_group - 1)`).
pub fn cta_group_first(cta_id_in_cluster: usize, cta_group: usize) -> usize {
    cta_id_in_cluster & !(cta_group - 1)
}

// ---------------------------------------------------------------------------
// tcgen05.ld / tcgen05.st
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LdstShape {
    /// `.16x32bx2` with its `immHalfSplitoff`.
    Shape16x32bx2(usize),
    Shape16x64b,
    Shape16x128b,
    Shape32x32b,
    Shape16x256b,
}

impl LdstShape {
    /// Registers per `.num` repeat (legacy `LdstShape::REGISTERS_PER_NUM`).
    pub const fn registers_per_num(self) -> usize {
        match self {
            Self::Shape16x32bx2(_) | Self::Shape16x64b | Self::Shape32x32b => 1,
            Self::Shape16x128b => 2,
            Self::Shape16x256b => 4,
        }
    }

    /// Legal `.num` values (legacy `ValidNum` impls).
    pub const fn valid_num(self, num: usize) -> bool {
        let max = match self {
            Self::Shape16x32bx2(_) | Self::Shape16x64b | Self::Shape32x32b => 128,
            Self::Shape16x128b => 64,
            Self::Shape16x256b => 32,
        };
        num.is_power_of_two() && num <= max
    }
}

/// TMEM `(lane, column)` an ld/st register touches (legacy `raw_tcgen05_ldst_location`).
///
/// `warp_id_in_cta` replaces the legacy `WarpContext`.
#[allow(clippy::too_many_arguments)]
pub fn ldst_location(
    warp_id_in_cta: usize,
    address: u32,
    row_offset: i64,
    col_offset: i64,
    shape: LdstShape,
    packed: bool,
    register_index: usize,
    execution_lane: usize,
) -> OpResult<(usize, usize)> {
    let (base_row, base_col) = tmem_address(address, row_offset, col_offset)?;
    let warp_in_group = warp_id_in_cta % 4;
    let (row_delta, col_delta) = match shape {
        LdstShape::Shape16x32bx2(_) => (execution_lane % 16, register_index),
        LdstShape::Shape16x64b => (
            (execution_lane >> 2)
                .checked_add(8 * (execution_lane & 1))
                .ok_or_else(|| OpError::message("tcgen05.16x64b row overflow"))?,
            ((execution_lane >> 1) & 1)
                .checked_add(2 * register_index)
                .ok_or_else(|| OpError::message("tcgen05.16x64b column overflow"))?,
        ),
        LdstShape::Shape16x128b => (
            (execution_lane >> 2)
                .checked_add(8 * (register_index & 1))
                .ok_or_else(|| OpError::message("tcgen05.16x128b row overflow"))?,
            (execution_lane & 3)
                .checked_add(4 * (register_index >> 1))
                .ok_or_else(|| OpError::message("tcgen05.16x128b column overflow"))?,
        ),
        LdstShape::Shape32x32b => (execution_lane, register_index),
        LdstShape::Shape16x256b => (
            (execution_lane >> 2)
                .checked_add(8 * ((register_index >> 1) & 1))
                .ok_or_else(|| OpError::message("tcgen05.16x256b row overflow"))?,
            (register_index & 1)
                .checked_add(2 * (execution_lane & 3))
                .and_then(|value| value.checked_add(8 * (register_index >> 2)))
                .ok_or_else(|| OpError::message("tcgen05.16x256b column overflow"))?,
        ),
    };
    let warp_lane_base = warp_in_group * 32;
    // Tile lowering passes a warp-relative lane; hand-written PTX may pass the
    // absolute 0..127 TMEM lane. Normalize both before the fragment mapping.
    let normalized_base_row = if base_row < 32 {
        warp_lane_base
            .checked_add(base_row)
            .ok_or_else(|| OpError::message("raw TCGEN warp-relative row overflow"))?
    } else if (warp_lane_base..warp_lane_base + 32).contains(&base_row) {
        base_row
    } else {
        return Err(OpError::message(format!(
            "raw TCGEN base row {base_row} is outside warp {warp_in_group}'s accessible TMEM lanes {warp_lane_base}..{}",
            warp_lane_base + 32
        )));
    };
    let row = normalized_base_row
        .checked_add(row_delta)
        .ok_or_else(|| OpError::message("raw TCGEN row overflow"))?;
    let physical_col_delta = if packed {
        col_delta
            .checked_mul(2)
            .ok_or_else(|| OpError::message("raw TCGEN packed column overflow"))?
    } else {
        col_delta
    };
    // The second half starts at taddr + immHalfSplitoff, independently of
    // the repeat count and pack/unpack width of each half.
    let half_offset = match shape {
        LdstShape::Shape16x32bx2(offset) => (execution_lane / 16)
            .checked_mul(offset)
            .ok_or_else(|| OpError::message("raw TCGEN half-split overflow"))?,
        _ => 0,
    };
    let column = base_col
        .checked_add(physical_col_delta)
        .and_then(|column| column.checked_add(half_offset))
        .ok_or_else(|| OpError::message("raw TCGEN column overflow"))?;
    if row >= 128 {
        return Err(OpError::message(format!(
            "raw TCGEN row {row} is outside 128 TMEM lanes"
        )));
    }
    let accessible_start = warp_lane_base;
    let accessible_end = accessible_start + 32;
    if !(accessible_start..accessible_end).contains(&row) {
        return Err(OpError::message(format!(
            "raw TCGEN row {row} is outside warp {warp_in_group}'s accessible TMEM lanes {accessible_start}..{accessible_end}"
        )));
    }
    Ok((row, column))
}

// ---------------------------------------------------------------------------
// tcgen05.cp
// ---------------------------------------------------------------------------

/// PTX shape code of each tcgen05.cp form (legacy `CpShape::CODE`).
pub mod cp_shape {
    pub const WARPX4_32X128B: u8 = 0;
    pub const WARPX2_02_13_64X128B: u8 = 1;
    pub const SHAPE_128X128B: u8 = 2;
    pub const SHAPE_128X256B: u8 = 3;
    pub const SHAPE_4X256B: u8 = 4;
    pub const WARPX2_01_23_64X128B: u8 = 5;
}

/// `(source rows, 32-bit words per row)` of a cp shape code.
pub fn cp_rows_words(shape: u8) -> OpResult<(usize, usize)> {
    match shape {
        0 => Ok((32, 4)),
        1 => Ok((64, 4)),
        2 => Ok((128, 4)),
        3 => Ok((128, 8)),
        4 => Ok((4, 8)),
        5 => Ok((64, 4)),
        _ => Err(OpError::message(format!(
            "raw tcgen05.cp shape code {shape} is invalid"
        ))),
    }
}

/// Up to four destination lanes of one source row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpDestinationLanes {
    values: [usize; 4],
    len: usize,
}

impl CpDestinationLanes {
    const fn one(first: usize) -> Self {
        Self {
            values: [first, 0, 0, 0],
            len: 1,
        }
    }

    const fn two(first: usize, second: usize) -> Self {
        Self {
            values: [first, second, 0, 0],
            len: 2,
        }
    }

    const fn four(values: [usize; 4]) -> Self {
        Self { values, len: 4 }
    }

    pub const fn as_slice(&self) -> &[usize] {
        self.values.split_at(self.len).0
    }
}

pub fn cp_destination_lanes(shape: u8, source_row: usize) -> OpResult<CpDestinationLanes> {
    match shape {
        0 => Ok(CpDestinationLanes::four([
            source_row,
            32 + source_row,
            64 + source_row,
            96 + source_row,
        ])),
        1 => {
            let half = source_row / 32;
            let row = source_row % 32;
            Ok(CpDestinationLanes::two(
                half * 32 + row,
                (half + 2) * 32 + row,
            ))
        }
        2 | 3 => Ok(CpDestinationLanes::one(source_row)),
        4 => Ok(CpDestinationLanes::one(
            source_row
                .checked_mul(32)
                .ok_or_else(|| OpError::message("raw tcgen05.cp 4x256b lane overflow"))?,
        )),
        5 => {
            let half = source_row / 32;
            let row = source_row % 32;
            let base = half
                .checked_mul(64)
                .ok_or_else(|| OpError::message("raw tcgen05.cp multicast overflow"))?;
            Ok(CpDestinationLanes::two(base + row, base + 32 + row))
        }
        _ => Err(OpError::message(format!(
            "raw tcgen05.cp shape code {shape} is invalid"
        ))),
    }
}

pub fn unpack_b6(packed: [u8; 3]) -> [u8; 4] {
    let bits = u32::from(packed[0]) | (u32::from(packed[1]) << 8) | (u32::from(packed[2]) << 16);
    [
        (bits & 0x3f) as u8,
        ((bits >> 6) & 0x3f) as u8,
        ((bits >> 12) & 0x3f) as u8,
        ((bits >> 18) & 0x3f) as u8,
    ]
}

/// Expand one (possibly b4/b6-compressed) source word into the 4-byte cell.
pub fn cp_decode_word(
    packed: &[u8; 4],
    source_byte_len: usize,
    decompress: u8,
) -> OpResult<[u8; 4]> {
    match decompress {
        0 if source_byte_len == 4 => Ok(*packed),
        1 if source_byte_len == 2 => Ok([
            (packed[0] & 0x0f) << 2,
            (packed[0] >> 4) << 2,
            (packed[1] & 0x0f) << 2,
            (packed[1] >> 4) << 2,
        ]),
        2 if source_byte_len == 3 => Ok(unpack_b6([packed[0], packed[1], packed[2]])),
        _ => Err(OpError::message(format!(
            "raw tcgen05.cp decompression code {decompress} has invalid source width {source_byte_len}"
        ))),
    }
}

/// Target CTAs of a tcgen05.cp (legacy cta_group check in `raw_tcgen05_cp`).
pub fn cp_target_ctas(
    cta_id_in_cluster: usize,
    ctas_per_cluster: usize,
    cta_group: usize,
) -> OpResult<Vec<usize>> {
    let mut targets = vec![cta_id_in_cluster];
    if cta_group == 2 {
        let peer = cta_id_in_cluster ^ 1;
        if peer >= ctas_per_cluster {
            return Err(OpError::message(
                "raw tcgen05.cp cta_group=2 has no paired CTA",
            ));
        }
        targets.push(peer);
    } else if cta_group != 1 {
        return Err(OpError::message(format!(
            "raw tcgen05.cp cta_group must be 1 or 2, got {cta_group}"
        )));
    }
    Ok(targets)
}

/// Every destination `(source_row, word, lane, column)` of a tcgen05.cp, with
/// the legacy lane-range check; returns also `(lane_end, column_end)`.
pub fn cp_destination_cells(
    address: u32,
    row_offset: i64,
    col_offset: i64,
    shape: u8,
) -> OpResult<(Vec<(usize, usize, usize, usize)>, usize, usize)> {
    let (base_row, base_col) = tmem_address(address, row_offset, col_offset)?;
    let (rows, words) = cp_rows_words(shape)?;
    let column_end = base_col
        .checked_add(words)
        .ok_or_else(|| OpError::message("raw tcgen05.cp TMEM column overflow"))?;
    let mut lane_end = 0;
    let mut cells = Vec::with_capacity(rows * words);
    for source_row in 0..rows {
        let lanes = cp_destination_lanes(shape, source_row)?;
        let mut absolute = [0_usize; 4];
        for (slot, &destination_lane) in lanes.as_slice().iter().enumerate() {
            let lane = base_row
                .checked_add(destination_lane)
                .ok_or_else(|| OpError::message("raw tcgen05.cp TMEM lane overflow"))?;
            if lane >= 128 {
                return Err(OpError::message(format!(
                    "raw tcgen05.cp TMEM lane {lane} is outside 128 lanes"
                )));
            }
            lane_end = lane_end.max(lane + 1);
            absolute[slot] = lane;
        }
        for word in 0..words {
            for &lane in &absolute[..lanes.as_slice().len()] {
                cells.push((source_row, word, lane, base_col + word));
            }
        }
    }
    Ok((cells, lane_end, column_end))
}

// ---------------------------------------------------------------------------
// Dense accumulator / packed-A TMEM layouts
// ---------------------------------------------------------------------------

/// F16 K=16 and FP8 K=32 occupy the same eight 32-bit words per CTA1 row.
pub const CTA1_PACKED_A_COLUMNS: usize = 8;

pub fn layout_f_lane(row: usize) -> OpResult<usize> {
    if row >= 64 {
        return Err(OpError::message(format!(
            "raw TCGEN Layout F row {row} is outside 64 rows"
        )));
    }
    (row / 16)
        .checked_mul(32)
        .and_then(|value| value.checked_add(row % 16))
        .ok_or_else(|| OpError::message("raw TCGEN Layout F lane overflow"))
}

/// Physical TMEM organization selected by a dense CTA1 MMA (PTX 9.7.17.10.5:
/// M=128 Layout D, M=64 `.ws` Layout E, M=64 non-`.ws` Layout F, M=32 `.ws` G).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseTmemLayout {
    D,
    E,
    F,
    G,
}

pub fn cta1_dense_tmem_layout(m: usize, weight_stationary: bool) -> OpResult<DenseTmemLayout> {
    match (m, weight_stationary) {
        (128, _) => Ok(DenseTmemLayout::D),
        (64, true) => Ok(DenseTmemLayout::E),
        (64, false) => Ok(DenseTmemLayout::F),
        (32, true) => Ok(DenseTmemLayout::G),
        _ => Err(OpError::message(format!(
            "raw CTA1 dense TMEM layout has unsupported M={m}"
        ))),
    }
}

impl DenseTmemLayout {
    pub fn physical_columns(self, logical_columns: usize) -> OpResult<usize> {
        match self {
            Self::D | Self::F => Ok(logical_columns),
            Self::E | Self::G if logical_columns.is_multiple_of(self.packed_a_banks()) => {
                Ok(logical_columns / self.packed_a_banks())
            }
            Self::E | Self::G => Err(OpError::message(format!(
                "raw TCGEN banked layout requires a divisible column count, got {logical_columns}"
            ))),
        }
    }

    pub fn location(
        self,
        row: usize,
        column: usize,
        rows: usize,
        columns: usize,
    ) -> OpResult<(usize, usize)> {
        if row >= rows || column >= columns {
            return Err(OpError::message(
                "raw dense TMEM logical coordinate is outside its matrix",
            ));
        }
        match self {
            Self::D => Ok((row, column)),
            Self::F => Ok((layout_f_lane(row)?, column)),
            Self::E | Self::G => {
                let bank_rows = 128 / self.packed_a_banks();
                if rows != bank_rows {
                    return Err(OpError::message(format!(
                        "raw TCGEN banked layout requires {bank_rows} rows, got {rows}"
                    )));
                }
                let bank_columns = self.physical_columns(columns)?;
                let lane = row
                    .checked_add(bank_rows * (column / bank_columns))
                    .ok_or_else(|| OpError::message("raw TCGEN Layout E lane overflow"))?;
                Ok((lane, column % bank_columns))
            }
        }
    }

    pub fn packed_a_banks(self) -> usize {
        match self {
            Self::E => 2,
            Self::G => 4,
            _ => 1,
        }
    }

    pub fn packed_a_location(
        self,
        bank: usize,
        row: usize,
        packed_column: usize,
        rows: usize,
        columns: usize,
    ) -> OpResult<(usize, usize)> {
        if bank >= self.packed_a_banks() {
            return Err(OpError::message(
                "raw packed TMEM A bank is outside its datapath layout",
            ));
        }
        if matches!(self, Self::E | Self::G) {
            let bank_rows = 128 / self.packed_a_banks();
            if rows != bank_rows {
                return Err(OpError::message(format!(
                    "raw TCGEN banked A layout requires {bank_rows} rows, got {rows}"
                )));
            }
            let lane = row
                .checked_add(bank_rows * bank)
                .ok_or_else(|| OpError::message("raw TCGEN Layout E A lane overflow"))?;
            return Ok((lane, packed_column));
        }
        self.location(row, packed_column, rows, columns)
    }
}

/// F32 CTA-pair accumulators use two 64-row banks for M=128, one for M=256.
pub fn cta2_columns_per_bank(m: usize, n: usize) -> usize {
    if m == 128 {
        n / 2
    } else {
        n
    }
}

/// Whether `lane` is masked off by a `disable_output_lane` word array.
pub fn lane_disabled(disable_output_lane: &[u32], lane: usize) -> bool {
    ((disable_output_lane[lane / 32] >> (lane % 32)) & 1) != 0
}

/// Absolute `(row, col, lane, column)` of each enabled cell of a dense
/// accumulator window; the cell walk shared by reads, scatters and footprints
/// (legacy `raw_tcgen05_dense_f32_tmem_footprints` / `raw_tcgen05_scatter_dense`).
/// `disable_output_lane = None` keeps disabled cells (reads handle them).
pub fn dense_tmem_cells(
    destination_address: u32,
    m: usize,
    n: usize,
    layout: DenseTmemLayout,
    disable_output_lane: Option<[u32; 4]>,
) -> OpResult<Vec<(usize, usize, usize, usize)>> {
    let (base_lane, base_column) = tmem_address(destination_address, 0, 0)?;
    let mut cells = Vec::with_capacity(
        m.checked_mul(n)
            .ok_or_else(|| OpError::message("raw TCGEN destination shape overflow"))?,
    );
    for row in 0..m {
        for column in 0..n {
            let (lane_delta, column_delta) = layout.location(row, column, m, n)?;
            let lane = base_lane
                .checked_add(lane_delta)
                .ok_or_else(|| OpError::message("raw TCGEN destination lane overflow"))?;
            if lane >= 128 {
                return Err(OpError::message(format!(
                    "raw TCGEN destination lane {lane} is outside 128 lanes"
                )));
            }
            if disable_output_lane.is_some_and(|mask| lane_disabled(&mask, lane)) {
                continue;
            }
            let allocated_column = base_column
                .checked_add(column_delta)
                .ok_or_else(|| OpError::message("raw TCGEN destination column overflow"))?;
            cells.push((row, column, lane, allocated_column));
        }
    }
    Ok(cells)
}

/// Absolute `(bank, row, packed_column, lane, column)` of each packed TMEM A
/// word (legacy `raw_tcgen05_gather_packed_tmem_a_columns` / footprints).
pub fn packed_tmem_a_cells(
    address: u32,
    rows: usize,
    layout: DenseTmemLayout,
    columns: usize,
) -> OpResult<Vec<(usize, usize, usize, usize, usize)>> {
    let (base_lane, base_col) = tmem_address(address, 0, 0)?;
    let capacity = rows
        .checked_mul(columns)
        .and_then(|value| value.checked_mul(layout.packed_a_banks()))
        .ok_or_else(|| OpError::message("raw packed TMEM A shape overflow"))?;
    let mut cells = Vec::with_capacity(capacity);
    for bank in 0..layout.packed_a_banks() {
        for row in 0..rows {
            for packed_column in 0..columns {
                let (lane_delta, column_delta) =
                    layout.packed_a_location(bank, row, packed_column, rows, columns)?;
                let lane = base_lane
                    .checked_add(lane_delta)
                    .ok_or_else(|| OpError::message("raw packed TMEM A lane overflow"))?;
                if lane >= 128 {
                    return Err(OpError::message(format!(
                        "raw packed TMEM A lane {lane} is outside 128 lanes"
                    )));
                }
                let column = base_col
                    .checked_add(column_delta)
                    .ok_or_else(|| OpError::message("raw packed TMEM A column overflow"))?;
                cells.push((bank, row, packed_column, lane, column));
            }
        }
    }
    Ok(cells)
}

// ---------------------------------------------------------------------------
// Block-scale factor layouts
// ---------------------------------------------------------------------------

/// Four-byte scale words occupy four columns per 128 rows in the 32-lane
/// layout, or one column in SM107's 128-lane SFA layout. Returns the address
/// of the word holding `scale_id + vector_index` and the byte inside it.
pub fn scale_chunk(
    address: u32,
    scale_id: usize,
    vector_index: usize,
    matrix_rows: usize,
    lanes_per_column: usize,
) -> OpResult<(u32, usize)> {
    if !matches!(lanes_per_column, 32 | 128) {
        return Err(OpError::message("scale layout requires 32 or 128 lanes"));
    }
    let column_stride = matrix_rows.div_ceil(128) * (128 / lanes_per_column);
    let byte = scale_id
        .checked_add(vector_index)
        .ok_or_else(|| OpError::message("raw TCGEN scale byte overflow"))?;
    let word = byte / 4;
    if word != 0 && column_stride == 0 {
        return Err(OpError::message(
            "raw TCGEN scale byte exceeds its TMEM word",
        ));
    }
    let offset = word
        .checked_mul(column_stride)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| OpError::message("raw TCGEN scale column overflow"))?;
    let (lane, column) = block_scale_address(address)?;
    let next_column = column
        .checked_add(offset as usize)
        .filter(|v| *v <= 0xffff)
        .ok_or_else(|| OpError::message("raw TCGEN scale column overflow"))?;
    Ok((((lane as u32) << 16) | next_column as u32, byte % 4))
}

/// Number of TMEM columns `count` scale vectors of `rows` rows span.
pub fn scale_columns(
    rows: usize,
    scale_id: usize,
    count: usize,
    lanes_per_column: usize,
) -> OpResult<usize> {
    let (last_address, _) = scale_chunk(0, scale_id, count - 1, rows, lanes_per_column)?;
    Ok(last_address as usize + rows.div_ceil(lanes_per_column))
}

/// `(lane, column, byte)` of the scale for `matrix_row`/`vector_index`
/// (the location half of legacy `raw_tcgen05_read_block_scale`).
pub fn block_scale_location(
    address: u32,
    scale_id: usize,
    matrix_row: usize,
    vector_index: usize,
    matrix_rows: usize,
    lanes_per_column: usize,
) -> OpResult<(usize, usize, usize)> {
    let (address, byte_in_cell) = scale_chunk(
        address,
        scale_id,
        vector_index,
        matrix_rows,
        lanes_per_column,
    )?;
    let (base_lane, base_col) = block_scale_address(address)?;
    let lane = base_lane
        .checked_add(matrix_row % lanes_per_column)
        .ok_or_else(|| OpError::message("raw TCGEN scale lane overflow"))?;
    let column = base_col
        .checked_add(matrix_row / lanes_per_column)
        .ok_or_else(|| OpError::message("raw TCGEN scale column overflow"))?;
    if lane >= 128 || byte_in_cell >= 4 {
        return Err(OpError::message(format!(
            "raw TCGEN scale location lane={lane}, byte={byte_in_cell} is outside TMEM"
        )));
    }
    Ok((lane, column, byte_in_cell))
}

/// SM100 block scales repeat across 32-lane TMEM partitions. M128 CTA2
/// splits SFB's N dimension between the lower and upper pair of partitions.
/// See CUTLASS tmem_sf_frg (ScaleFactorDuplicated4by1 / Duplicated2by2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleLayout {
    Replicated,
    SplitB { rows_per_half: usize },
}

impl ScaleLayout {
    pub fn replicas(self) -> usize {
        match self {
            Self::Replicated => 4,
            Self::SplitB { .. } => 2,
        }
    }

    pub fn location(self, address: u32, row: usize, replica: usize) -> OpResult<(usize, usize)> {
        let (base_lane, base_column) = block_scale_address(address)?;
        let (partition, row) = match self {
            Self::Replicated => (replica, row),
            Self::SplitB { rows_per_half } => {
                (2 * (row / rows_per_half) + replica, row % rows_per_half)
            }
        };
        let lane = base_lane + partition * 32 + row % 32;
        let column = base_column + row / 32;
        if lane >= 128 || column > 0xffff {
            return Err(OpError::message("raw TCGEN scale replica is outside TMEM"));
        }
        Ok((lane, column))
    }
}

pub fn mxf8_scale_layout(m: usize, n: usize, cta_group: usize, is_b: bool) -> ScaleLayout {
    if is_b && cta_group == 2 && m == 128 {
        ScaleLayout::SplitB {
            rows_per_half: n / 2,
        }
    } else {
        ScaleLayout::Replicated
    }
}

// ---------------------------------------------------------------------------
// Sparse metadata and LUT-B
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseMetadataLayout {
    B16 { selector: usize },
    Narrow { k: usize },
}

impl SparseMetadataLayout {
    pub fn chunks(self) -> usize {
        match self {
            Self::B16 { .. } => 8,
            Self::Narrow { k } => k / 4,
        }
    }

    /// Metadata columns a sparse MMA must own.
    pub fn columns(self) -> usize {
        match self {
            Self::B16 { .. } => 2,
            Self::Narrow { k } => k / 32,
        }
    }
}

/// `(lane, column, nibble)` of one 4-bit sparse metadata code.
pub fn sparse_metadata_location(
    address: u32,
    metadata_layout: SparseMetadataLayout,
    row: usize,
    chunk: usize,
) -> OpResult<(usize, usize, usize)> {
    let (base_lane, base_column) = tmem_address(address, 0, 0)?;
    let selector = match metadata_layout {
        SparseMetadataLayout::Narrow { k } => {
            if !matches!(k, 64 | 128) || chunk >= k / 4 {
                return Err(OpError::message(
                    "sparse narrow metadata chunk exceeds its K extent",
                ));
            }
            let lane = base_lane
                .checked_add(row)
                .ok_or_else(|| OpError::message("sparse metadata lane overflow"))?;
            let column = base_column
                .checked_add(chunk / 8)
                .ok_or_else(|| OpError::message("sparse metadata column overflow"))?;
            if lane >= 128 {
                return Err(OpError::message("sparse metadata lane exceeds TMEM"));
            }
            return Ok((lane, column, chunk % 8));
        }
        SparseMetadataLayout::B16 { selector } => selector,
    };
    let row_in_partition = row % 32;
    let lane = base_lane
        .checked_add((row / 32) * 32)
        .and_then(|value| value.checked_add(row_in_partition % 8))
        .and_then(|value| value.checked_add(16 * (row_in_partition / 16)))
        .and_then(|value| value.checked_add(8 * (chunk / 4)))
        .ok_or_else(|| OpError::message("sparse f16 metadata lane overflow"))?;
    let column = base_column
        .checked_add(selector)
        .ok_or_else(|| OpError::message("sparse f16 metadata column overflow"))?;
    if lane >= 128 {
        return Err(OpError::message(format!(
            "sparse f16 metadata lane {lane} is outside TMEM"
        )));
    }
    let nibble = 4 * ((row_in_partition % 16) / 8) + chunk % 4;
    Ok((lane, column, nibble))
}

/// Extract a nibble from a metadata word.
pub fn metadata_nibble(word: u32, nibble: usize) -> u8 {
    ((word >> (4 * nibble)) & 0xf) as u8
}

/// Location of the sparse `mxf4` metadata word (legacy
/// `raw_tcgen05_sparse_mxf4_metadata_code`): `(lane, column, nibble)`.
pub fn sparse_mxf4_metadata_location(
    address: u32,
    row: usize,
    chunk: usize,
) -> OpResult<(usize, usize, usize)> {
    let (base_lane, base_column) = tmem_address(address, 0, 0)?;
    let lane = base_lane
        .checked_add(row)
        .ok_or_else(|| OpError::message("sparse mxf4 metadata lane overflow"))?;
    let column = base_column
        .checked_add(chunk / 8)
        .ok_or_else(|| OpError::message("sparse mxf4 metadata column overflow"))?;
    if lane >= 128 {
        return Err(OpError::message("sparse mxf4 metadata exceeds TMEM lanes"));
    }
    Ok((lane, column, chunk % 8))
}

/// Sparse metadata must start on an even column in the destination's lanes.
pub fn validate_sparse_metadata_address(metadata: u32, destination_address: u32) -> OpResult<()> {
    let (metadata_lane, metadata_column) = tmem_address(metadata, 0, 0)?;
    if metadata_column % 2 != 0 || metadata_lane != tmem_address(destination_address, 0, 0)?.0 {
        return Err(OpError::message(
            "sparse floating MMA metadata requires two-column alignment and matching datapath lanes",
        ));
    }
    Ok(())
}

pub fn lut_b_location(address: u32, group: usize, word: usize) -> OpResult<(usize, usize)> {
    let (base_row, base_column) = tmem_address(address, 0, 0)?;
    if address & 1 != 0 {
        return Err(OpError::message(
            "tcgen05.mma LUT-B address must be two-column aligned",
        ));
    }
    let row = base_row
        .checked_add(group)
        .filter(|row| *row < 128)
        .ok_or_else(|| OpError::message("tcgen05.mma LUT-B row is outside TMEM"))?;
    let column = base_column
        .checked_add(word)
        .filter(|column| *column < 65536)
        .ok_or_else(|| OpError::message("tcgen05.mma LUT-B column overflow"))?;
    Ok((row, column))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_ldst_accepts_absolute_tmem_lane_addresses_for_each_warp() {
        for warp in 0..4 {
            let address = u32::try_from(warp * 32).unwrap() << 16 | 7;
            let location =
                ldst_location(warp, address, 0, 0, LdstShape::Shape32x32b, false, 2, 3).unwrap();
            assert_eq!(location, (warp * 32 + 3, 9));
        }
    }

    #[test]
    fn raw_ldst_normalizes_a_warp_relative_tmem_lane_address() {
        for warp in 0..4 {
            let location =
                ldst_location(warp, 7_u32 << 16, 0, 0, LdstShape::Shape32x32b, false, 2, 3)
                    .unwrap();
            assert_eq!(location, (warp * 32 + 10, 2));
        }
    }

    #[test]
    fn raw_ldst_rejects_an_absolute_lane_address_owned_by_another_warp() {
        let error =
            ldst_location(2, 32_u32 << 16, 0, 0, LdstShape::Shape32x32b, false, 0, 0).unwrap_err();
        assert!(error.to_string().contains("accessible TMEM lanes 64..96"));
    }

    #[test]
    fn raw_cp_4x256b_places_each_source_row_in_one_warp_lane() {
        for (row, lane) in [(0, 0), (1, 32), (2, 64), (3, 96)] {
            assert_eq!(cp_destination_lanes(4, row).unwrap().as_slice(), [lane]);
        }
    }

    #[test]
    fn raw_cp_b6_decompression_preserves_the_six_bit_codes() {
        let values = [1_u32, 2, 31, 63];
        let bits = values[0] | (values[1] << 6) | (values[2] << 12) | (values[3] << 18);
        let packed = [bits as u8, (bits >> 8) as u8, (bits >> 16) as u8];
        assert_eq!(unpack_b6(packed), [1, 2, 31, 63]);
    }

    #[test]
    fn get_tmem_addr_wraps_each_field() {
        assert_eq!(get_tmem_addr(0x0001_fffe, -2, 3), 0xffff_0001);
    }

    #[test]
    fn ldst_num_tables_match_the_variant_impls() {
        assert!(LdstShape::Shape16x256b.valid_num(32));
        assert!(!LdstShape::Shape16x256b.valid_num(64));
        assert!(LdstShape::Shape16x128b.valid_num(64));
        assert!(!LdstShape::Shape16x128b.valid_num(128));
        assert!(LdstShape::Shape16x32bx2(4).valid_num(128));
        assert!(!LdstShape::Shape32x32b.valid_num(3));
        assert_eq!(LdstShape::Shape16x256b.registers_per_num(), 4);
    }
}
