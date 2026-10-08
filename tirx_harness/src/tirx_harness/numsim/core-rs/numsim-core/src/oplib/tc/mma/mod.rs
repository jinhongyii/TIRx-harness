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
    merge_cta2_packed_a, read_dense_window, scaled_tmem_a_geometry, scatter_dense_window,
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
    cta1_dense_tmem_layout, metadata_nibble, mxf8_scale_layout, sparse_metadata_location,
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
    f8_b_descriptor, ColumnMask, MatrixDescriptor, MatrixDescriptorLayout, SharedWindow,
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
        let mut values = Vec::new();
        for cta in 0..cta_group {
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
        if self.cta_group == 1 {
            return io.lib(read_dense_window(
                &mut |_, _| Ok(true),
                &mut |lane, column| io.cell(0, lane, column),
                self.taddr,
                self.m,
                self.n,
                self.layout,
                decode,
                disabled,
                self.mask4(),
            ));
        }
        let cells = io.lib(cta2_window_cells(
            self.taddr,
            self.m,
            self.n,
            self.layout,
            None,
        ))?;
        let mut values = vec![disabled; self.m * self.n];
        for (cta, lane, column, index) in cells {
            values[index] = decode(io.lib(io.cell(cta, lane, column))?);
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
        if self.cta_group == 1 {
            let result = scatter_dense_window(
                &mut |lane, column, bytes: [u8; 4]| {
                    write_cell(io, tmem_write, 0, lane, column, &bytes)
                },
                self.taddr,
                self.m,
                self.n,
                self.layout,
                encode,
                self.mask4(),
                values,
            );
            return io.lib(result);
        }
        let cells = io.lib(cta2_window_cells(
            self.taddr,
            self.m,
            self.n,
            self.layout,
            Some(self.mask),
        ))?;
        for (cta, lane, column, index) in cells {
            let bytes = encode(values[index]);
            io.lib(write_cell(io, tmem_write, cta, lane, column, &bytes))?;
        }
        Ok(())
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
