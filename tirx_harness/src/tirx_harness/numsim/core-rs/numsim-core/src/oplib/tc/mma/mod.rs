//! `tcgen05.mma` numerics recomposed from `numsim_oplib::tcgen05` pieces,
//! ported form by form from the legacy drivers in
//! `engine-rs/src/runtime/tcgen_ops.rs` (`raw_tcgen05_mma_float`,
//! `raw_tcgen05_sparse_float_tail`, `raw_tcgen05_mma_f8f6f4_cta{1,2}`,
//! `raw_tcgen05_mma_block_scale_mxf4`,
//! `raw_tcgen05_mma_sp_block_scale_mxf4_e8m0_ss_cta1`,
//! `raw_tcgen05_mma_block_scale_mxf8f6f4`), `tcgen_ops/integer.rs`
//! (`raw_tcgen05_mma_integer`), `matrix_ops.rs` (`raw_tcgen05_shift`, the
//! `.ashift` tail) and `instructions/tcgen05.rs` (form checks).
//!
//! Address conventions: `smem(cta, addr, buf)` reads CTA `cta`'s shared
//! window at a window byte address (descriptor start addresses are window
//! addresses; the window is one flat region based at 0, so the swizzle XOR
//! sees the window address bits as the legacy virtual base did);
//! `tmem_read(cta, lane, col, buf)` / `tmem_write(cta, lane, col, bytes)`
//! access one 32-bit TMEM cell. `cta` is the index within the CTA group
//! (legacy `first_cta + cta`); CTA-pair gathers concatenate per-CTA results
//! in CTA order. Lifecycle/allocation checks (`validate_raw_tcgen05_tmem_*`)
//! are the engine's.

mod block_scale;
mod dense;

use std::cell::RefCell;

use super::super::{OpError, OpResult, TcArch, TcMmaOptions, TcSmemRead, TcTmemRead, TcTmemWrite};
use super::mxf4_spellings;
use crate::program::{TcA, TcMmaKind};
use crate::sync::completion::TcgenMmaPayload;
use numsim_oplib::mma::mma_f32_abt_increasing_k;
use numsim_oplib::tcgen05::gather::{
    cta2_window_cells, gather_b16_rows_with, gather_f8_rows, gather_lut_b_rows, gather_mxf4_rows,
    gather_packed_tmem_a, gather_scaled_tmem_a_cta, gather_sparse_mxf4_e8m0_rows, gather_tf32_rows,
    merge_cta2_packed_a, scaled_tmem_a_geometry,
    sparse_mxf4_metadata_code, validate_f8_gather,
};
use numsim_oplib::tcgen05::instr_desc::{
    decode_f8f6f4, decode_mxf4_for_cta_group, decode_mxf8f6f4, decode_sparse_mxf4,
    f8_tmem_a_address, tf32_family_input_scale, validate_lut_b, validate_tmem_a_transpose,
    FloatKind,
};
use numsim_oplib::tcgen05::integer::{
    gather_integer_rows, integer_mma_accumulate, integer_shape, IntegerKind,
};
use numsim_oplib::tcgen05::layouts::{
    cta1_dense_tmem_layout, lane_disabled, metadata_nibble, tmem_address, mxf8_scale_layout, sparse_metadata_location,
    validate_sparse_metadata_address, DenseTmemLayout, ScaleLayout, SparseMetadataLayout,
    CTA1_PACKED_A_COLUMNS,
};
use numsim_oplib::tcgen05::mma::{
    expand_sparse_2of4, expand_sparse_mxf4_a, mma_dense_tail, sparse_float_mma,
};
use numsim_oplib::tcgen05::narrow::{
    decode_b16, decode_b16_word, decode_e2m1_word, tf32_payload_to_f32, CellDtype, NarrowFormat,
};
use numsim_oplib::tcgen05::scale::{
    apply_row_scales, check_joint_scales, decode_ue8m0_scale, mxf8_scale_values, read_block_scale,
};
use numsim_oplib::tcgen05::smem_desc::{
    decode_matrix_descriptor, decode_matrix_descriptor_for_layout, decode_packed_matrix_descriptor,
    b16_matrix_byte_offset, f8_b_descriptor, masked_row, ColumnMask, MatrixDescriptor,
    MatrixDescriptorLayout, SharedWindow,
};
use numsim_oplib::types::{OpError as LibError, OpResult as LibResult};

use block_scale::{mxf4_mma, mxf8f6f4_mma};
use dense::{f8f6f4_mma, float_mma, integer_mma};

/// The whole 32-bit shared-window address space; `smem` bounds-checks.
const WINDOW: SharedWindow = SharedWindow::whole(0, 1 << 32);

type TmemWrite<'w> = TcTmemWrite<'w>;

fn descriptor_layout(arch: TcArch) -> MatrixDescriptorLayout {
    match arch {
        TcArch::Sm100 => MatrixDescriptorLayout::Sm100,
        TcArch::Sm103 => MatrixDescriptorLayout::Sm103,
        TcArch::Sm107 => MatrixDescriptorLayout::Sm107,
    }
}

// ---------------------------------------------------------------------------
// Closure bridge
// ---------------------------------------------------------------------------

/// Bridges the contract closures into `numsim_oplib` closures, keeping the
/// caller's error (kind and message) instead of round-tripping it through a
/// plain `numsim_oplib` message.
struct Io<'a> {
    smem: TcSmemRead<'a>,
    tmem_read: TcTmemRead<'a>,
    stash: RefCell<Option<OpError>>,
}

impl Io<'_> {
    fn fail(&self, error: OpError) -> LibError {
        let message = error.message.clone();
        self.stash.borrow_mut().get_or_insert(error);
        LibError::message(format!("tc_mma operand access failed: {message}"))
    }

    fn lift(&self, error: LibError) -> OpError {
        self.stash
            .borrow_mut()
            .take()
            .unwrap_or_else(|| error.into())
    }

    /// Run a `numsim_oplib` step, restoring a stashed closure error.
    fn lib<T>(&self, result: LibResult<T>) -> OpResult<T> {
        result.map_err(|error| self.lift(error))
    }

    fn index(&self, value: usize, what: &str) -> LibResult<u32> {
        u32::try_from(value).map_err(|_| {
            self.fail(OpError::invalid(format!(
                "tc_mma {what} {value} exceeds u32"
            )))
        })
    }

    fn shared(&self, cta: usize) -> impl FnMut(usize, &mut [u8]) -> LibResult<()> + '_ {
        move |offset, buf| {
            let address = self.index(offset, "shared address")?;
            (self.smem)(cta as u32, address, buf).map_err(|error| self.fail(error))
        }
    }

    fn cell(&self, cta: usize, lane: usize, column: usize) -> LibResult<[u8; 4]> {
        let mut bytes = [0_u8; 4];
        (self.tmem_read)(
            cta as u32,
            self.index(lane, "TMEM lane")?,
            self.index(column, "TMEM column")?,
            &mut bytes,
        )
        .map_err(|error| self.fail(error))?;
        Ok(bytes)
    }

    fn words(&self, cta: usize) -> impl FnMut(usize, usize) -> LibResult<u32> + '_ {
        move |lane, column| Ok(u32::from_le_bytes(self.cell(cta, lane, column)?))
    }

    fn bytes(&self, cta: usize) -> impl FnMut(usize, usize, usize) -> LibResult<u8> + '_ {
        move |lane, column, byte| {
            let cell = self.cell(cta, lane, column)?;
            cell.get(byte)
                .copied()
                .ok_or_else(|| self.fail(OpError::invalid("TMEM byte index outside a cell")))
        }
    }

    /// One 4-bit sparse metadata code (legacy `raw_tcgen05_sparse_metadata_code`).
    fn metadata_code(
        &self,
        cta: usize,
        address: u32,
        layout: SparseMetadataLayout,
        row: usize,
        chunk: usize,
    ) -> LibResult<u8> {
        let (lane, column, nibble) = sparse_metadata_location(address, layout, row, chunk)?;
        Ok(metadata_nibble(
            u32::from_le_bytes(self.cell(cta, lane, column)?),
            nibble,
        ))
    }

    /// Concatenate a per-CTA gather over the group.
    fn per_cta<T>(
        &self,
        cta_group: usize,
        mut gather: impl FnMut(usize) -> LibResult<Vec<T>>,
    ) -> OpResult<Vec<T>> {
        // One CTA: its gather is the result (no second copy).
        let mut values = self.lib(gather(0))?;
        for cta in 1..cta_group {
            values.extend(self.lib(gather(cta))?);
        }
        Ok(values)
    }

    /// Packed TMEM A of the group (legacy `raw_tcgen05_gather_packed_tmem_a_columns`
    /// for CTA1, `raw_tcgen05_gather_packed_tmem_a_cta2_with` for the pair).
    #[allow(clippy::too_many_arguments)]
    fn tmem_a<S: Copy, const E: usize>(
        &self,
        cta_group: usize,
        address: u32,
        m: usize,
        layout: DenseTmemLayout,
        columns: usize,
        decode: impl Fn(u32) -> LibResult<[S; E]>,
    ) -> OpResult<Vec<S>> {
        if cta_group == 1 {
            return self.lib(gather_packed_tmem_a(
                &mut self.words(0),
                address,
                m,
                layout,
                columns,
                &decode,
            ));
        }
        if !matches!(m, 128 | 256) {
            return Err(OpError::invalid(
                "invalid CTA2 packed A shape or missing paired CTA",
            ));
        }
        let local = [0, 1].map(|cta| {
            gather_packed_tmem_a(
                &mut self.words(cta),
                address,
                m / 2,
                layout,
                columns,
                &decode,
            )
        });
        let [first, second] = local;
        let local = [self.lib(first)?, self.lib(second)?];
        Ok(merge_cta2_packed_a(&local, m, columns, E, layout))
    }
}

/// One call per run of consecutive columns of one TMEM lane (perf,
/// W2-21): `cells` are `(cta, lane, column, index)` in walk order; runs are
/// maximal stretches of that order with the same CTA and lane and columns
/// increasing by one. The closures see exactly the cells of the walk.
fn cell_runs(cells: &[(usize, usize, usize, usize)]) -> impl Iterator<Item = &[(usize, usize, usize, usize)]> {
    let mut rest = cells;
    std::iter::from_fn(move || {
        let (&first, _) = rest.split_first()?;
        let mut len = 1;
        while let Some(&(cta, lane, column, _)) = rest.get(len) {
            if cta != first.0 || lane != first.1 || column != first.2 + len {
                break;
            }
            len += 1;
        }
        let (run, tail) = rest.split_at(len);
        rest = tail;
        Some(run)
    })
}

fn check_cell(io: &Io<'_>, lane: usize, column: usize) -> LibResult<(u32, u32)> {
    let lane = io.index(lane, "TMEM lane")?;
    let column = io.index(column, "TMEM column")?;
    if lane >= crate::arena::addr::TMEM_LANES || column >= crate::arena::addr::TMEM_COLS {
        return Err(io.fail(OpError::invalid(format!("tmem cell ({lane}, {column}) out of range"))));
    }
    Ok((lane, column))
}

/// `first..first + len` when `indices` is that consecutive range.
fn contiguous(indices: &[usize]) -> Option<std::ops::Range<usize>> {
    let (&first, _) = indices.split_first()?;
    let end = first.checked_add(indices.len())?;
    (indices.iter().enumerate().all(|(i, &index)| index == first + i)).then_some(first..end)
}

/// Read one run of cells (`indices` = value slots of `column..`).
fn read_run<T>(
    io: &Io<'_>,
    buf: &mut Vec<u8>,
    (cta, lane, column): (usize, usize, usize),
    indices: &[usize],
    values: &mut [T],
    decode: &impl Fn([u8; 4]) -> T,
) -> LibResult<()> {
    check_cell(io, lane, column + indices.len() - 1)?;
    let (lane, column) = check_cell(io, lane, column)?;
    buf.clear();
    buf.resize(indices.len() * 4, 0);
    (io.tmem_read)(cta as u32, lane, column, buf).map_err(|error| io.fail(error))?;
    let cells = buf.as_chunks::<4>().0;
    if let Some(slots) = contiguous(indices).and_then(|r| values.get_mut(r)) {
        // Every streamed run is a contiguous index range: no per-cell indirection.
        for (slot, bytes) in slots.iter_mut().zip(cells) {
            *slot = decode(*bytes);
        }
    } else {
        for (&index, bytes) in indices.iter().zip(cells) {
            values[index] = decode(*bytes);
        }
    }
    Ok(())
}

/// Write one run of cells.
fn write_run<T: Copy>(
    io: &Io<'_>,
    tmem_write: TcTmemWrite<'_>,
    buf: &mut Vec<u8>,
    (cta, lane, column): (usize, usize, usize),
    indices: &[usize],
    values: &[T],
    encode: &impl Fn(T) -> [u8; 4],
) -> LibResult<()> {
    check_cell(io, lane, column + indices.len() - 1)?;
    let (lane, column) = check_cell(io, lane, column)?;
    buf.clear();
    buf.resize(indices.len() * 4, 0);
    let cells = buf.as_chunks_mut::<4>().0;
    if let Some(slots) = contiguous(indices).and_then(|r| values.get(r)) {
        for (cell, &value) in cells.iter_mut().zip(slots) {
            *cell = encode(value);
        }
    } else {
        for (&index, cell) in indices.iter().zip(cells) {
            *cell = encode(values[index]);
        }
    }
    tmem_write(cta as u32, lane, column, buf).map_err(|error| io.fail(error))
}

/// Runs of a `(cta, lane, column, index)` cell list (CTA pairs).
fn list_runs(
    cells: &[(usize, usize, usize, usize)],
    mut run: impl FnMut((usize, usize, usize), &[usize]) -> LibResult<()>,
) -> LibResult<()> {
    let mut indices = Vec::new();
    for chunk in cell_runs(cells) {
        indices.clear();
        indices.extend(chunk.iter().map(|cell| cell.3));
        run((chunk[0].0, chunk[0].1, chunk[0].2), &indices)?;
    }
    Ok(())
}

/// The CTA1 dense window walk of `dense_tmem_cells` (same order, same
/// checks and messages), streamed as runs of consecutive columns of one
/// lane: `run(lane, column, indices)` with `indices[i] = row * n + col` of
/// the cell at `column + i`. No per-cell vectors (perf, W2-21).
fn cta1_runs(
    taddr: u32,
    m: usize,
    n: usize,
    layout: DenseTmemLayout,
    mask: Option<[u32; 4]>,
    mut run: impl FnMut(usize, usize, &[usize]) -> LibResult<()>,
) -> LibResult<()> {
    let (base_lane, base_column) = tmem_address(taddr, 0, 0)?;
    m.checked_mul(n)
        .ok_or_else(|| LibError::message("raw TCGEN destination shape overflow"))?;
    let mut indices: Vec<usize> = Vec::with_capacity(n);
    if layout == DenseTmemLayout::D && n > 0 {
        // Layout D: row `r` is lane `base + r`, columns `base..base + n`.
        layout.location(m - 1, n - 1, m, n)?;
        for row in 0..m {
            let lane = base_lane
                .checked_add(row)
                .ok_or_else(|| LibError::message("raw TCGEN destination lane overflow"))?;
            if lane >= 128 {
                return Err(LibError::message(format!(
                    "raw TCGEN destination lane {lane} is outside 128 lanes"
                )));
            }
            if mask.is_some_and(|mask| lane_disabled(&mask, lane)) {
                continue;
            }
            base_column
                .checked_add(n - 1)
                .ok_or_else(|| LibError::message("raw TCGEN destination column overflow"))?;
            indices.clear();
            indices.extend(row * n..row * n + n);
            run(lane, base_column, &indices)?;
        }
        return Ok(());
    }
    let mut current: Option<(usize, usize)> = None;
    for row in 0..m {
        for col in 0..n {
            let (lane_delta, column_delta) = layout.location(row, col, m, n)?;
            let lane = base_lane
                .checked_add(lane_delta)
                .ok_or_else(|| LibError::message("raw TCGEN destination lane overflow"))?;
            if lane >= 128 {
                return Err(LibError::message(format!(
                    "raw TCGEN destination lane {lane} is outside 128 lanes"
                )));
            }
            if mask.is_some_and(|mask| lane_disabled(&mask, lane)) {
                continue;
            }
            let column = base_column
                .checked_add(column_delta)
                .ok_or_else(|| LibError::message("raw TCGEN destination column overflow"))?;
            match current {
                Some((l, c)) if l == lane && c + indices.len() == column => {}
                _ => {
                    if let Some((l, c)) = current {
                        run(l, c, &indices)?;
                    }
                    indices.clear();
                    current = Some((lane, column));
                }
            }
            indices.push(row * n + col);
        }
    }
    if let Some((l, c)) = current {
        run(l, c, &indices)?;
    }
    Ok(())
}

/// The CTA-pair window walk of `cta2_window_cells` (same order, same checks
/// and messages), streamed as runs: Layout D (every canonical accumulator)
/// is one run per (CTA, row) without building the `m * n` cell list (perf,
/// W4 profile: the list walk was ~12% of `fp16_bf16_gemm`). Other layouts
/// take the cell list.
fn cta2_runs(
    taddr: u32,
    m: usize,
    n: usize,
    layout: DenseTmemLayout,
    mask: Option<[u32; 8]>,
    mut run: impl FnMut((usize, usize, usize), &[usize]) -> LibResult<()>,
) -> LibResult<()> {
    if layout != DenseTmemLayout::D || n == 0 || !(m == 128 || m == 256) {
        let cells = cta2_window_cells(taddr, m, n, layout, mask)?;
        return list_runs(&cells, run);
    }
    let (base_lane, base_column) = tmem_address(taddr, 0, 0)?;
    let rows_per_cta = m / 2;
    layout.physical_columns(n)?;
    layout.location(rows_per_cta - 1, n - 1, rows_per_cta, n)?;
    let mut indices: Vec<usize> = Vec::with_capacity(n);
    for target in 0..2 {
        for row in 0..rows_per_cta {
            let lane = base_lane
                .checked_add(row)
                .ok_or_else(|| LibError::message("raw cta_group=2 destination lane overflow"))?;
            if lane >= 128 {
                return Err(LibError::message(format!(
                    "raw cta_group=2 destination lane {lane} is outside 128 lanes"
                )));
            }
            if mask.is_some_and(|mask| ((mask[target * 4 + lane / 32] >> (lane % 32)) & 1) != 0) {
                continue;
            }
            base_column
                .checked_add(n - 1)
                .ok_or_else(|| LibError::message("raw cta_group=2 destination column overflow"))?;
            let first = (target * rows_per_cta + row) * n;
            indices.clear();
            indices.extend(first..first + n);
            run((target, lane, base_column), &indices)?;
        }
    }
    Ok(())
}

/// `gather_b16_rows_with` with K-major rows read 16 bytes (8 elements) at
/// a time where the descriptor layout keeps them contiguous, which every
/// canonical K-major layout does within a 16-byte core-matrix row (perf,
/// W2-21). Same values, same bytes read, in the same row order; any other
/// shape takes the per-element path.
#[allow(clippy::too_many_arguments)]
fn gather_b16_chunked<Scalar: Default + Copy>(
    read_shared: &mut impl FnMut(usize, &mut [u8]) -> LibResult<()>,
    descriptor: MatrixDescriptor,
    rows: usize,
    columns: usize,
    transpose: bool,
    mask: Option<ColumnMask>,
    decode: impl Fn(u16) -> LibResult<Scalar>,
) -> LibResult<Vec<Scalar>> {
    if transpose || !columns.is_multiple_of(8) {
        return gather_b16_rows_with(read_shared, WINDOW, descriptor, rows, columns, transpose, mask, decode);
    }
    let mut values = Vec::with_capacity(
        rows.checked_mul(columns)
            .ok_or_else(|| LibError::message("raw sparse b16 gather shape overflow"))?,
    );
    let mut chunk = [0_u8; 16];
    for row in 0..rows {
        let Some(row) = masked_row(mask, row) else {
            values.extend(std::iter::repeat_with(Scalar::default).take(columns));
            continue;
        };
        for first in (0..columns).step_by(8) {
            // A 16-byte-aligned start whose 8th element sits 14 bytes on is
            // one core-matrix row: swizzling permutes whole 16-byte units.
            let start = b16_matrix_byte_offset(WINDOW, descriptor, row, first, false)?;
            let contiguous = start % 16 == 0
                && b16_matrix_byte_offset(WINDOW, descriptor, row, first + 7, false)? == start + 14;
            if contiguous {
                read_shared(start, &mut chunk)?;
                for pair in chunk.as_chunks::<2>().0 {
                    values.push(decode(u16::from_le_bytes(*pair))?);
                }
            } else {
                for column in first..first + 8 {
                    let offset = b16_matrix_byte_offset(WINDOW, descriptor, row, column, false)?;
                    let mut bytes = [0_u8; 2];
                    read_shared(offset, &mut bytes)?;
                    values.push(decode(u16::from_le_bytes(bytes))?);
                }
            }
        }
    }
    Ok(values)
}

fn write_cell(
    io: &Io<'_>,
    tmem_write: TcTmemWrite<'_>,
    cta: usize,
    lane: usize,
    column: usize,
    bytes: &[u8],
) -> LibResult<()> {
    let lane = io.index(lane, "TMEM lane")?;
    let column = io.index(column, "TMEM column")?;
    tmem_write(cta as u32, lane, column, bytes).map_err(|error| io.fail(error))
}

// ---------------------------------------------------------------------------
// Accumulator window
// ---------------------------------------------------------------------------

/// The D tile: one CTA's dense window (legacy `RawMmaDestination::Dense` /
/// `LaneCells`) or a CTA pair (`Cta2`, `cta2_window_cells`).
struct Window {
    taddr: u32,
    m: usize,
    n: usize,
    layout: DenseTmemLayout,
    /// `4 * cta_group` disable-output-lane words.
    mask: [u32; 8],
    cta_group: usize,
}

impl Window {
    fn mask4(&self) -> [u32; 4] {
        [self.mask[0], self.mask[1], self.mask[2], self.mask[3]]
    }

    fn read<T: Copy>(
        &self,
        io: &Io<'_>,
        decode: impl Fn([u8; 4]) -> T,
        disabled: T,
    ) -> OpResult<Vec<T>> {
        // Legacy `read_dense_window` with an always-valid cell reads every
        // cell, disabled lanes included (`disabled` is never used).
        let _ = disabled;
        let mut values = vec![disabled; self.m * self.n];
        let mut buf = Vec::new();
        if self.cta_group == 1 {
            io.lib(cta1_runs(self.taddr, self.m, self.n, self.layout, None, |lane, column, indices| {
                read_run(io, &mut buf, (0, lane, column), indices, &mut values, &decode)
            }))?;
        } else {
            io.lib(cta2_runs(self.taddr, self.m, self.n, self.layout, None, |at, indices| {
                read_run(io, &mut buf, at, indices, &mut values, &decode)
            }))?;
        }
        Ok(values)
    }

    fn write<T: Copy>(
        &self,
        io: &Io<'_>,
        tmem_write: TmemWrite<'_>,
        encode: impl Fn(T) -> [u8; 4],
        values: &[T],
    ) -> OpResult {
        let mut buf = Vec::new();
        if self.cta_group == 1 {
            io.lib(cta1_runs(self.taddr, self.m, self.n, self.layout, Some(self.mask4()), |lane, column, indices| {
                write_run(io, tmem_write, &mut buf, (0, lane, column), indices, values, &encode)
            }))
        } else {
            io.lib(cta2_runs(self.taddr, self.m, self.n, self.layout, Some(self.mask), |at, indices| {
                write_run(io, tmem_write, &mut buf, at, indices, values, &encode)
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// Form
// ---------------------------------------------------------------------------

/// Instruction-level facts shared by every kind.
struct Form {
    cta_group: usize,
    /// `.ws` zero-column mask (`Some` iff `.ws`).
    ws_mask: Option<u64>,
    /// Sparse metadata TMEM address (`.sp`).
    metadata: Option<u32>,
    /// `4 * cta_group` output-lane mask words (zeros for `.ws`).
    mask: [u32; 8],
    a_in_tmem: bool,
}

fn check_form(payload: &TcgenMmaPayload, options: &TcMmaOptions) -> OpResult<Form> {
    let args = &payload.args;
    let kind = args.kind;
    let cta_group = match args.cta_group {
        0 | 1 => 1,
        2 => 2,
        other => {
            return Err(OpError::invalid(format!(
                "tcgen05.mma cta_group {other} is invalid"
            )))
        }
    };
    let block_scaled = matches!(
        kind,
        TcMmaKind::MxF8f6f4 | TcMmaKind::MxF4 | TcMmaKind::MxF4Nvf4
    );
    if block_scaled != payload.scale_taddrs.is_some() || block_scaled != args.block_scale.is_some()
    {
        return Err(OpError::invalid(format!(
            "tcgen05.mma.kind::{kind:?} {} block-scale operands",
            if block_scaled {
                "requires"
            } else {
                "does not take"
            }
        )));
    }
    if payload.scale_input_d.is_some() && !matches!(kind, TcMmaKind::F16 | TcMmaKind::Tf32) {
        return Err(OpError::invalid(format!(
            "scale-input-d is only valid for kind::f16/tf32, not {kind:?}"
        )));
    }
    if args.ws && (block_scaled || cta_group != 1) {
        return Err(OpError::invalid(format!(
            "tcgen05.mma.ws is only valid for cta_group::1 non-block-scaled kinds, not {kind:?}"
        )));
    }
    if args.ashift && (block_scaled || !matches!(args.a, TcA::Tmem(_))) {
        return Err(OpError::invalid(
            "tcgen05.mma.ashift requires a non-block-scaled MMA with TMEM A",
        ));
    }
    if options.ti16 && !matches!(kind, TcMmaKind::I8 | TcMmaKind::Ti16) {
        return Err(OpError::invalid(".ti16 is a kind::i8 operand spelling"));
    }
    if options.lut_b.is_some() && !matches!(kind, TcMmaKind::F8f6f4 | TcMmaKind::MxF8f6f4) {
        return Err(OpError::invalid(".lut_b is only valid for f8f6f4 kinds"));
    }
    let metadata = payload.sparse_meta;
    if args.sparse_meta.is_some() != metadata.is_some() {
        return Err(OpError::invalid(
            "tcgen05.mma.sp metadata operand and resolved address disagree",
        ));
    }
    let words = &payload.disable_output_lane;
    let mut mask = [0_u32; 8];
    let ws_mask = if args.ws {
        Some(match options.zero_col_mask {
            Some(bits) => bits,
            None => match words.as_slice() {
                [] => 0,
                [low] => u64::from(*low),
                [low, high] => u64::from(*low) | u64::from(*high) << 32,
                other => {
                    return Err(OpError::invalid(format!(
                        "tcgen05.mma.ws zero-column mask has {} words",
                        other.len()
                    )))
                }
            },
        })
    } else {
        if block_scaled && words.iter().any(|word| *word != 0) {
            return Err(OpError::invalid(
                "tcgen05.mma.block_scale has no disable_output_lane operand",
            ));
        }
        match words.len() {
            0 => {}
            len if len == 4 * cta_group => mask[..len].copy_from_slice(words),
            len => {
                return Err(OpError::invalid(format!(
                    "cta_group::{cta_group} disable_output_lane needs {} words, got {len}",
                    4 * cta_group
                )))
            }
        }
        None
    };
    Ok(Form {
        cta_group,
        ws_mask,
        metadata,
        mask,
        a_in_tmem: matches!(args.a, TcA::Tmem(_)),
    })
}

fn tmem_a_address(a: u64) -> OpResult<u32> {
    u32::try_from(a).map_err(|_| OpError::invalid("tcgen05.mma TMEM A address exceeds u32"))
}

fn column_mask(
    io: &Io<'_>,
    form: &Form,
    m: usize,
    n: usize,
    idesc: u32,
) -> OpResult<Option<ColumnMask>> {
    form.ws_mask
        .map(|bits| io.lib(ColumnMask::new(bits, m, n, idesc)))
        .transpose()
}

/// `raw_tcgen05_shift::<COLUMNS>` (`.ashift`): in every 32-lane partition of
/// each CTA's TMEM, rows 0..31 of the `columns` A columns take rows 1..32;
/// row 31 keeps its value. TMEM allocations span all 128 lanes.
fn shift_a(
    io: &Io<'_>,
    tmem_write: TmemWrite<'_>,
    address: u32,
    columns: usize,
    cta_group: usize,
) -> OpResult {
    let base_row = ((address >> 16) & 0xffff) as usize;
    let base_col = (address & 0xffff) as usize;
    if !base_row.is_multiple_of(32) || base_row >= 128 {
        return Err(OpError::invalid(
            "tcgen05.shift row address must name one aligned warp",
        ));
    }
    for cta in 0..cta_group {
        for partition in (0..128).step_by(32) {
            let mut rows = vec![[[0_u8; 4]; 16]; 31];
            for (row, cells) in rows.iter_mut().enumerate() {
                for (column, cell) in cells.iter_mut().take(columns).enumerate() {
                    *cell = io.lib(io.cell(cta, partition + row + 1, base_col + column))?;
                }
            }
            for (row, cells) in rows.iter().enumerate() {
                for (column, cell) in cells.iter().take(columns).enumerate() {
                    io.lib(write_cell(
                        io,
                        tmem_write,
                        cta,
                        partition + row,
                        base_col + column,
                        cell,
                    ))?;
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

pub(super) fn run(
    payload: &TcgenMmaPayload,
    options: &TcMmaOptions,
    smem: TcSmemRead<'_>,
    tmem_read: TcTmemRead<'_>,
    tmem_write: TcTmemWrite<'_>,
) -> OpResult {
    let form = check_form(payload, options)?;
    let io = Io {
        smem,
        tmem_read,
        stash: RefCell::new(None),
    };
    match payload.args.kind {
        TcMmaKind::F16 | TcMmaKind::Tf32 => float_mma(&io, payload, options, &form, tmem_write),
        TcMmaKind::F8f6f4 if form.metadata.is_some() => {
            if options.lut_b.is_some() {
                return Err(OpError::invalid(".lut_b has no sparse kind::f8f6f4 form"));
            }
            float_mma(&io, payload, options, &form, tmem_write)
        }
        TcMmaKind::F8f6f4 => f8f6f4_mma(&io, payload, options, &form, tmem_write),
        TcMmaKind::I8 | TcMmaKind::Ti16 => integer_mma(&io, payload, options, &form, tmem_write),
        TcMmaKind::MxF4 | TcMmaKind::MxF4Nvf4 => mxf4_mma(&io, payload, options, &form, tmem_write),
        TcMmaKind::MxF8f6f4 => mxf8f6f4_mma(&io, payload, options, &form, tmem_write),
    }
}

#[cfg(test)]
mod chunk_tests {
    use super::*;
    use numsim_oplib::tcgen05::encode::encode_matrix_descriptor;

    /// The chunked K-major b16 gather returns the per-element gather's
    /// values and reads exactly the same bytes, for every swizzle mode.
    #[test]
    fn chunked_b16_gather_matches_the_per_element_gather() {
        let smem: Vec<u8> = (0..1 << 16).map(|i: u32| (i.wrapping_mul(2_654_435_761) >> 24) as u8).collect();
        for swizzle in 0..=6_i64 {
            for (rows, columns) in [(64, 16), (128, 64), (8, 8), (16, 24)] {
                for transpose in [false, true] {
                    let Ok(descriptor) = decode_matrix_descriptor(encode_matrix_descriptor(0x2000, 8, 64, swizzle)) else {
                        continue;
                    };
                    let run = |chunked: bool| {
                        let mut read: Vec<usize> = Vec::new();
                        let mut fetch = |offset: usize, buf: &mut [u8]| -> LibResult<()> {
                            buf.copy_from_slice(&smem[offset..offset + buf.len()]);
                            read.extend(offset..offset + buf.len());
                            Ok(())
                        };
                        let decode = |bits: u16| Ok(u32::from(bits));
                        let values = if chunked {
                            gather_b16_chunked(&mut fetch, descriptor, rows, columns, transpose, None, decode)
                        } else {
                            gather_b16_rows_with(&mut fetch, WINDOW, descriptor, rows, columns, transpose, None, decode)
                        };
                        read.sort_unstable();
                        read.dedup();
                        (values.map_err(|e| e.to_string()), read)
                    };
                    assert_eq!(run(true), run(false), "swizzle {swizzle} {rows}x{columns} transpose {transpose}");
                }
            }
        }
    }

    /// The streamed CTA-pair walk emits exactly the runs of the cell-list walk
    /// (same order, CTA, lane, column and indices) and the same errors.
    #[test]
    fn cta2_runs_match_the_cell_list_walk() {
        type Runs = Vec<((usize, usize, usize), Vec<usize>)>;
        type Sink<'a> = &'a mut dyn FnMut((usize, usize, usize), &[usize]) -> LibResult<()>;
        type Walk<'a> = &'a dyn Fn(Sink<'_>) -> LibResult<()>;
        let collect = |f: Walk<'_>| -> Result<Runs, String> {
            let mut out = Vec::new();
            f(&mut |at, idx| {
                out.push((at, idx.to_vec()));
                Ok(())
            })
            .map_err(|e| format!("{e:?}"))?;
            Ok(out)
        };
        let masks = [None, Some([0; 8]), Some([0x8000_0001, 0, 0xffff_ffff, 0, 0, 0x10, 0, 0x8000_0000])];
        for taddr in [0_u32, 16, 0x0040_0020, 0x0050_0000] {
            for m in [128_usize, 256, 64] {
                for n in [8_usize, 32, 256] {
                    for mask in masks {
                        let streamed = collect(&|run| cta2_runs(taddr, m, n, DenseTmemLayout::D, mask, run));
                        let listed = collect(&|run| {
                            let cells = cta2_window_cells(taddr, m, n, DenseTmemLayout::D, mask)?;
                            list_runs(&cells, run)
                        });
                        assert_eq!(streamed, listed, "taddr={taddr:#x} m={m} n={n} mask={mask:?}");
                    }
                }
            }
        }
    }
}
