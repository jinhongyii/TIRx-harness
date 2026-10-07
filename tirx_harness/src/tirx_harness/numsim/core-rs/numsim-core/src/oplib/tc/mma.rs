//! `tcgen05.mma` numerics recomposed from `numsim_oplib::tcgen05` pieces.
//!
//! Address conventions: `smem(addr, buf)` reads shared bytes at a
//! shared-window byte address (descriptor start addresses are window
//! addresses, so the window is modeled as one flat region based at 0 and the
//! swizzle XOR sees the window address bits, as the legacy virtual base did);
//! `tmem_read(lane, col, buf)` / `tmem_write(lane, col, bytes)` access one
//! 32-bit TMEM cell (taddr `lane << 16 | col`, `arena::addr::tmem_addr`).

use std::cell::RefCell;

use super::super::{OpError, OpResult};
use super::mxf4_spellings;
use crate::program::{CollectorOp, TcA, TcMmaKind};
use crate::sync::completion::TcgenMmaPayload;
use numsim_oplib::mma::mma_f32_abt_increasing_k;
use numsim_oplib::tcgen05::gather::{
    gather_b16_rows_with, gather_f8_rows, gather_mxf4_rows, gather_packed_tmem_a,
    gather_scaled_tmem_a_cta, gather_tf32_rows, read_dense_window, scaled_tmem_a_geometry,
    scatter_dense_window, validate_f8_gather,
};
use numsim_oplib::tcgen05::instr_desc::{
    decode_f8f6f4, decode_mxf4_for_cta_group, decode_mxf8f6f4, tf32_family_input_scale,
    validate_tmem_a_transpose, FloatKind,
};
use numsim_oplib::tcgen05::integer::{
    gather_integer_rows, integer_mma_accumulate, integer_shape, IntegerKind,
};
use numsim_oplib::tcgen05::layouts::{
    cta1_dense_tmem_layout, mxf8_scale_layout, DenseTmemLayout, CTA1_PACKED_A_COLUMNS,
};
use numsim_oplib::tcgen05::mma::mma_dense_tail;
use numsim_oplib::tcgen05::narrow::{
    decode_b16, decode_b16_word, decode_e2m1_word, tf32_payload_to_f32, CellDtype, NarrowFormat,
};
use numsim_oplib::tcgen05::scale::{apply_row_scales, decode_ue8m0_scale, mxf8_scale_values};
use numsim_oplib::tcgen05::smem_desc::{
    decode_matrix_descriptor, decode_packed_matrix_descriptor, MatrixDescriptor,
    MatrixDescriptorLayout, SharedWindow,
};
use numsim_oplib::types::{OpError as LibError, OpResult as LibResult};

const SM100: MatrixDescriptorLayout = MatrixDescriptorLayout::Sm100;

/// The whole 32-bit shared-window address space; `smem` bounds-checks.
const WINDOW: SharedWindow = SharedWindow::whole(0, 1 << 32);

/// Bridges the contract closures into `numsim_oplib` closures, keeping the
/// caller's error (kind and message) instead of round-tripping it through a
/// plain `numsim_oplib` message.
struct Io<'a> {
    smem: &'a dyn Fn(u32, &mut [u8]) -> OpResult,
    tmem_read: &'a dyn Fn(u32, u32, &mut [u8]) -> OpResult,
    stash: RefCell<Option<OpError>>,
}

impl<'a> Io<'a> {
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

    fn shared(&self) -> impl FnMut(usize, &mut [u8]) -> LibResult<()> + '_ {
        move |offset, buf| {
            let address = self.index(offset, "shared address")?;
            (self.smem)(address, buf).map_err(|error| self.fail(error))
        }
    }

    fn cell(&self, lane: usize, column: usize) -> LibResult<[u8; 4]> {
        let mut bytes = [0_u8; 4];
        (self.tmem_read)(
            self.index(lane, "TMEM lane")?,
            self.index(column, "TMEM column")?,
            &mut bytes,
        )
        .map_err(|error| self.fail(error))?;
        Ok(bytes)
    }

    fn words(&self) -> impl FnMut(usize, usize) -> LibResult<u32> + '_ {
        move |lane, column| Ok(u32::from_le_bytes(self.cell(lane, column)?))
    }

    fn bytes(&self) -> impl FnMut(usize, usize, usize) -> LibResult<u8> + '_ {
        move |lane, column, byte| {
            let cell = self.cell(lane, column)?;
            cell.get(byte)
                .copied()
                .ok_or_else(|| self.fail(OpError::invalid("TMEM byte index outside a cell")))
        }
    }

    fn cells(&self) -> impl FnMut(usize, usize) -> LibResult<[u8; 4]> + '_ {
        move |lane, column| self.cell(lane, column)
    }
}

/// Accumulator window of one CTA.
struct Window {
    taddr: u32,
    m: usize,
    n: usize,
    layout: DenseTmemLayout,
    mask: [u32; 4],
}

impl Window {
    fn read<T: Copy>(
        &self,
        io: &Io<'_>,
        decode: impl Fn([u8; 4]) -> T,
        disabled: T,
    ) -> OpResult<Vec<T>> {
        io.lib(read_dense_window(
            &mut |_, _| Ok(true),
            &mut io.cells(),
            self.taddr,
            self.m,
            self.n,
            self.layout,
            decode,
            disabled,
            self.mask,
        ))
    }

    fn write<T: Copy>(
        &self,
        io: &Io<'_>,
        tmem_write: &mut dyn FnMut(u32, u32, &[u8]) -> OpResult,
        encode: impl Fn(T) -> [u8; 4],
        values: &[T],
    ) -> OpResult {
        let result = scatter_dense_window(
            &mut |lane, column, bytes: [u8; 4]| {
                let lane = io.index(lane, "TMEM lane")?;
                let column = io.index(column, "TMEM column")?;
                tmem_write(lane, column, &bytes).map_err(|error| io.fail(error))
            },
            self.taddr,
            self.m,
            self.n,
            self.layout,
            encode,
            self.mask,
            values,
        );
        io.lib(result)
    }
}

/// Fail closed on every form outside the modeled set (see the module docs of
/// `oplib::tc`).
fn check_form(payload: &TcgenMmaPayload) -> OpResult {
    let args = &payload.args;
    let kind = args.kind;
    match args.cta_group {
        0 | 1 => {}
        2 => {
            return Err(OpError::unsupported(format!(
                "tcgen05.mma.cta_group::2.kind::{kind:?}: CTA-pair operands are not modeled (tc_mma has one CTA's smem/tmem accessors)"
            )))
        }
        other => return Err(OpError::invalid(format!("tcgen05.mma cta_group {other} is invalid"))),
    }
    let unsupported = |form: &str| {
        Err(OpError::unsupported(format!(
            "tcgen05.mma{form}.kind::{kind:?} is not modeled"
        )))
    };
    if args.ws {
        return unsupported(".ws");
    }
    if args.sparse_meta.is_some() || payload.sparse_meta.is_some() {
        return unsupported(".sp");
    }
    if args.collector_a != CollectorOp::None || args.collector_b != CollectorOp::None {
        return unsupported(".collector");
    }
    if args.ashift {
        return unsupported(".ashift");
    }
    if args.variant.is_some() {
        return unsupported(" (lut_b/ti16 variant)");
    }
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
    if block_scaled && payload.disable_output_lane.iter().any(|word| *word != 0) {
        return unsupported(".block_scale with disable_output_lane");
    }
    Ok(())
}

fn output_mask(payload: &TcgenMmaPayload) -> OpResult<[u32; 4]> {
    match payload.disable_output_lane.as_slice() {
        [] => Ok([0; 4]),
        [a, b, c, d] => Ok([*a, *b, *c, *d]),
        other => Err(OpError::invalid(format!(
            "cta_group::1 disable_output_lane needs 4 words, got {}",
            other.len()
        ))),
    }
}

/// D tile: read input D (when enabled), run the increasing-K tail, write D.
#[allow(clippy::too_many_arguments)]
fn float_tail(
    io: &Io<'_>,
    tmem_write: &mut dyn FnMut(u32, u32, &[u8]) -> OpResult,
    window: &Window,
    cell: CellDtype,
    k: usize,
    a: &[f32],
    b: &[f32],
    enable_input_d: bool,
    a_in_tmem: bool,
    scale: impl FnOnce() -> OpResult<f32>,
) -> OpResult {
    let input_d = if enable_input_d {
        Some(window.read(io, |bytes| cell.decode(bytes), f32::NAN)?)
    } else {
        None
    };
    let scale = scale()?;
    let output = io.lib(mma_dense_tail(
        window.m,
        window.n,
        k,
        a,
        b,
        input_d.as_deref().map(|values| (values, scale)),
        a_in_tmem,
        window.layout,
    ))?;
    window.write(io, tmem_write, |value| cell.encode(value), &output)
}

fn smem_desc(io: &Io<'_>, bits: u64) -> OpResult<MatrixDescriptor> {
    io.lib(decode_matrix_descriptor(bits))
}

fn tmem_a_address(a: u64) -> OpResult<u32> {
    u32::try_from(a).map_err(|_| OpError::invalid("tcgen05.mma TMEM A address exceeds u32"))
}

/// `kind::f16` / `kind::tf32` (legacy `raw_tcgen05_mma_float`, dense CTA1).
fn float_mma(
    io: &Io<'_>,
    payload: &TcgenMmaPayload,
    tmem_write: &mut dyn FnMut(u32, u32, &[u8]) -> OpResult,
) -> OpResult {
    let idesc = payload.idesc;
    let kind = match payload.args.kind {
        TcMmaKind::Tf32 => FloatKind::Tf32,
        _ => FloatKind::B16 {
            a_bf16: (idesc >> 7) & 7 == 1,
            b_bf16: (idesc >> 10) & 7 == 1,
        },
    };
    let instruction = io.lib(kind.instruction(idesc, 1, false, false))?;
    let k = kind.packed_k(idesc);
    let (m, n) = (instruction.m, instruction.n);
    let b_descriptor = smem_desc(io, payload.b_desc)?;
    let layout = io.lib(cta1_dense_tmem_layout(m, false))?;
    let gather_shared = |descriptor, rows, transpose, negate, is_b: bool| -> OpResult<Vec<f32>> {
        io.lib(match kind {
            FloatKind::Tf32 => gather_tf32_rows(
                &mut io.shared(),
                WINDOW,
                descriptor,
                rows,
                k,
                transpose,
                negate,
                None,
            ),
            FloatKind::B16 { a_bf16, b_bf16 } => {
                let bf16 = if is_b { b_bf16 } else { a_bf16 };
                gather_b16_rows_with(
                    &mut io.shared(),
                    WINDOW,
                    descriptor,
                    rows,
                    k,
                    transpose,
                    None,
                    |bits| Ok(decode_b16(bits, bf16, negate)),
                )
            }
            FloatKind::SparseNarrow { .. } => {
                Err(LibError::message("unreachable sparse narrow kind"))
            }
        })
    };
    let a_in_tmem = matches!(payload.args.a, TcA::Tmem(_));
    let a = if a_in_tmem {
        io.lib(validate_tmem_a_transpose(instruction.transpose_a))?;
        let address = tmem_a_address(payload.a)?;
        let negate = instruction.negate_a;
        io.lib(match kind {
            FloatKind::Tf32 => gather_packed_tmem_a(
                &mut io.words(),
                address,
                m,
                layout,
                CTA1_PACKED_A_COLUMNS,
                |word| {
                    let value = tf32_payload_to_f32(word);
                    Ok([if negate { -value } else { value }])
                },
            ),
            FloatKind::B16 { a_bf16, .. } => gather_packed_tmem_a(
                &mut io.words(),
                address,
                m,
                layout,
                CTA1_PACKED_A_COLUMNS,
                |word| Ok(decode_b16_word(word, a_bf16, negate)),
            ),
            FloatKind::SparseNarrow { .. } => {
                Err(LibError::message("unreachable sparse narrow kind"))
            }
        })?
    } else {
        let descriptor = smem_desc(io, payload.a)?;
        gather_shared(
            descriptor,
            m,
            instruction.transpose_a,
            instruction.negate_a,
            false,
        )?
    };
    let b = gather_shared(
        b_descriptor,
        n,
        instruction.transpose_b,
        instruction.negate_b,
        true,
    )?;
    let window = Window {
        taddr: payload.d_taddr,
        m,
        n,
        layout,
        mask: output_mask(payload)?,
    };
    let scale_input_d = payload.scale_input_d.unwrap_or(0) as usize;
    float_tail(
        io,
        tmem_write,
        &window,
        instruction.cell_dtype,
        k,
        &a,
        &b,
        payload.enable_input_d,
        a_in_tmem,
        || io.lib(tf32_family_input_scale(scale_input_d)),
    )
}

/// `kind::f8f6f4` (legacy `raw_tcgen05_mma_f8f6f4_cta1`, no LUT-B / ws).
fn f8f6f4_mma(
    io: &Io<'_>,
    payload: &TcgenMmaPayload,
    tmem_write: &mut dyn FnMut(u32, u32, &[u8]) -> OpResult,
) -> OpResult {
    let idesc = payload.idesc;
    let a_format = io.lib(NarrowFormat::decode((idesc >> 7) & 7, "A"))?;
    let b_format = io.lib(NarrowFormat::decode((idesc >> 10) & 7, "B"))?;
    let d_f16 = (idesc >> 4) & 3 == 0;
    let instruction = io.lib(decode_f8f6f4(
        idesc,
        a_format,
        b_format,
        d_f16,
        1,
        SM100.supports_f8f6f4_k64(),
        false,
        false,
    ))?;
    let (m, n, k) = (instruction.m, instruction.n, instruction.k);
    let b_descriptor = smem_desc(io, payload.b_desc)?;
    let layout = io.lib(cta1_dense_tmem_layout(m, false))?;
    let a_in_tmem = matches!(payload.args.a, TcA::Tmem(_));
    let a = if a_in_tmem {
        io.lib(validate_tmem_a_transpose(instruction.transpose_a))?;
        let address = tmem_a_address(payload.a)?;
        let negate = instruction.negate_a;
        io.lib(gather_packed_tmem_a(
            &mut io.words(),
            address,
            m,
            layout,
            k / 4,
            |word| {
                Ok(a_format
                    .decode_tmem_word(word)?
                    .map(|value| if negate { -value } else { value }))
            },
        ))?
    } else {
        let descriptor = smem_desc(io, payload.a)?;
        io.lib(validate_f8_gather(1, k, a_format, instruction.transpose_a))?;
        io.lib(gather_f8_rows(
            &mut io.shared(),
            WINDOW,
            descriptor,
            m,
            k,
            a_format,
            instruction.negate_a,
            instruction.transpose_a,
            None,
            false,
        ))?
    };
    io.lib(validate_f8_gather(1, k, b_format, instruction.transpose_b))?;
    let b = io.lib(gather_f8_rows(
        &mut io.shared(),
        WINDOW,
        b_descriptor,
        n,
        k,
        b_format,
        instruction.negate_b,
        instruction.transpose_b,
        None,
        false,
    ))?;
    // A `.f16` destination rounds once on store (legacy `RawMmaCellDtype`).
    let window = Window {
        taddr: payload.d_taddr,
        m,
        n,
        layout,
        mask: output_mask(payload)?,
    };
    float_tail(
        io,
        tmem_write,
        &window,
        CellDtype::from_half(d_f16),
        k,
        &a,
        &b,
        payload.enable_input_d,
        a_in_tmem,
        || Ok(1.0),
    )
}

/// `kind::i8` (legacy `raw_tcgen05_mma_integer`, CTA1 dense): exact i64
/// accumulation, low 32 bits stored (clamped to s32 with `.satfinite`, bit 3).
fn integer_mma(
    io: &Io<'_>,
    payload: &TcgenMmaPayload,
    tmem_write: &mut dyn FnMut(u32, u32, &[u8]) -> OpResult,
) -> OpResult {
    let idesc = payload.idesc;
    let kind = IntegerKind::I8;
    let (m, n, transpose_a, transpose_b) = io.lib(integer_shape(kind, idesc, 1, false, false))?;
    let k = kind.packed_k();
    let layout = io.lib(cta1_dense_tmem_layout(m, false))?;
    let b_descriptor = smem_desc(io, payload.b_desc)?;
    let decode_a = |bits| kind.decode(bits, (idesc >> 7) & 7, false);
    let decode_b = |bits| kind.decode(bits, (idesc >> 10) & 7, false);
    let a = match payload.args.a {
        TcA::Smem(_) => {
            let descriptor = smem_desc(io, payload.a)?;
            io.lib(gather_integer_rows(
                &mut io.shared(),
                kind,
                WINDOW,
                descriptor,
                m,
                k,
                transpose_a,
                None,
                decode_a,
            ))?
        }
        TcA::Tmem(_) => {
            io.lib(validate_tmem_a_transpose(transpose_a))?;
            let address = tmem_a_address(payload.a)?;
            io.lib(gather_packed_tmem_a(
                &mut io.words(),
                address,
                m,
                layout,
                CTA1_PACKED_A_COLUMNS,
                |word: u32| {
                    Ok([
                        decode_a(word as u8 as u16)?,
                        decode_a((word >> 8) as u8 as u16)?,
                        decode_a((word >> 16) as u8 as u16)?,
                        decode_a((word >> 24) as u16)?,
                    ])
                },
            ))?
        }
    };
    let b = io.lib(gather_integer_rows(
        &mut io.shared(),
        kind,
        WINDOW,
        b_descriptor,
        n,
        k,
        transpose_b,
        None,
        decode_b,
    ))?;
    let window = Window {
        taddr: payload.d_taddr,
        m,
        n,
        layout,
        mask: output_mask(payload)?,
    };
    let mut output = if payload.enable_input_d {
        window
            .read(io, i32::from_le_bytes, 0)?
            .into_iter()
            .map(i64::from)
            .collect()
    } else {
        vec![0_i64; m * n]
    };
    io.lib(integer_mma_accumulate(m, n, k, &a, &b, &mut output))?;
    let saturate = idesc & 8 != 0;
    window.write(
        io,
        tmem_write,
        |value: i64| {
            let value = if saturate {
                value.clamp(i64::from(i32::MIN), i64::from(i32::MAX))
            } else {
                value
            };
            (value as i32).to_le_bytes()
        },
        &output,
    )
}

/// `kind::mxf4` / `kind::mxf4nvf4` block-scale, CTA1 (legacy
/// `raw_tcgen05_mma_block_scale_mxf4`, SM100 descriptor widths).
fn mxf4_mma(
    io: &Io<'_>,
    payload: &TcgenMmaPayload,
    tmem_write: &mut dyn FnMut(u32, u32, &[u8]) -> OpResult,
) -> OpResult {
    let idesc = payload.idesc;
    let block = payload.args.block_scale.map(|(_, _, block)| block);
    let spelling = mxf4_spellings(payload.args.kind, idesc, block)?[0];
    let instruction = io.lib(decode_mxf4_for_cta_group(idesc, spelling, 1, SM100, false))?;
    let (m, n, k) = (instruction.m, instruction.n, instruction.k);
    let (sfa, sfb) = payload
        .scale_taddrs
        .ok_or_else(|| OpError::invalid("block-scale MMA has no scale addresses"))?;
    let b_descriptor = io.lib(decode_packed_matrix_descriptor(
        payload.b_desc,
        SM100,
        k / 2,
        false,
    ))?;
    let a = match payload.args.a {
        TcA::Smem(_) => {
            let descriptor = io.lib(decode_packed_matrix_descriptor(
                payload.a,
                SM100,
                k / 2,
                false,
            ))?;
            io.lib(gather_mxf4_rows(
                &mut io.shared(),
                &mut io.bytes(),
                WINDOW,
                descriptor,
                m,
                0,
                m,
                sfa,
                instruction.sfa_id,
                spelling.decoder(),
                instruction.negate_a,
                k,
                instruction.block_elements,
                instruction.sfa_lanes,
            ))?
        }
        TcA::Tmem(_) => {
            let address = tmem_a_address(payload.a)?;
            let columns = k / 8;
            let (rows, layout, _) = io.lib(scaled_tmem_a_geometry(
                m,
                k,
                columns,
                1,
                true,
                instruction.sfa_id,
                instruction.block_elements,
                instruction.sfa_lanes,
            ))?;
            let mut values = vec![0.0_f32; layout.packed_a_banks() * m * k];
            io.lib(gather_scaled_tmem_a_cta(
                &mut io.words(),
                &mut io.bytes(),
                &mut values,
                0,
                address,
                m,
                k,
                rows,
                columns,
                layout,
                sfa,
                instruction.sfa_id,
                instruction.block_elements,
                spelling.decoder(),
                instruction.negate_a,
                &|words: &[u32]| {
                    Ok(words
                        .iter()
                        .flat_map(|word| decode_e2m1_word(*word))
                        .collect())
                },
                instruction.sfa_lanes,
            ))?;
            values
        }
    };
    let b = io.lib(gather_mxf4_rows(
        &mut io.shared(),
        &mut io.bytes(),
        WINDOW,
        b_descriptor,
        n,
        0,
        n,
        sfb,
        instruction.sfb_id,
        spelling.decoder(),
        instruction.negate_b,
        k,
        instruction.block_elements,
        32,
    ))?;
    // Legacy `LaneCells`: Layout D, F32 cells, no output-lane mask.
    let window = Window {
        taddr: payload.d_taddr,
        m,
        n,
        layout: DenseTmemLayout::D,
        mask: [0; 4],
    };
    let input_d = if payload.enable_input_d {
        Some(window.read(io, |bytes| CellDtype::F32.decode(bytes), f32::NAN)?)
    } else {
        None
    };
    let output = io.lib(mma_f32_abt_increasing_k(
        m,
        n,
        k,
        &a,
        &b,
        input_d.as_deref().map(|values| (values, 1.0)),
    ))?;
    window.write(io, tmem_write, |value: f32| value.to_le_bytes(), &output)
}

/// `kind::mxf8f6f4` block-scale `scale_vec::1X`, CTA1 dense (legacy
/// `raw_tcgen05_mma_block_scale_mxf8f6f4`, SM100, no LUT-B / sparse).
fn mxf8f6f4_mma(
    io: &Io<'_>,
    payload: &TcgenMmaPayload,
    tmem_write: &mut dyn FnMut(u32, u32, &[u8]) -> OpResult,
) -> OpResult {
    let idesc = payload.idesc;
    if let Some((_, _, block)) = payload.args.block_scale {
        if block != 32 {
            return Err(OpError::invalid(format!(
                "kind::mxf8f6f4 block size must be 32, got {block}"
            )));
        }
    }
    let instruction = io.lib(decode_mxf8f6f4(idesc, 1, SM100))?;
    if instruction.sparse {
        return Err(OpError::unsupported(
            "tcgen05.mma.sp.kind::mxf8f6f4 (descriptor sparsity bit) is not modeled",
        ));
    }
    let (m, n, k) = (instruction.m, instruction.n, instruction.k);
    let packed_k = instruction.packed_k();
    let (sfa, sfb) = payload
        .scale_taddrs
        .ok_or_else(|| OpError::invalid("block-scale MMA has no scale addresses"))?;
    let layout = io.lib(cta1_dense_tmem_layout(m, true))?;
    let transposed = instruction.transpose_a || instruction.transpose_b;
    let b_row_bytes = k / 16
        * if instruction.padded_atoms() {
            16
        } else {
            instruction.b_format.shared_atom_stride(k)
        };
    let b_descriptor = io.lib(decode_packed_matrix_descriptor(
        payload.b_desc,
        SM100,
        b_row_bytes,
        transposed,
    ))?;
    let a = match payload.args.a {
        TcA::Smem(_) => {
            let row_bytes = packed_k / 16 * instruction.a_format.shared_atom_stride(packed_k);
            let descriptor = io.lib(decode_packed_matrix_descriptor(
                payload.a, SM100, row_bytes, transposed,
            ))?;
            io.lib(validate_f8_gather(
                1,
                packed_k,
                instruction.a_format,
                instruction.transpose_a,
            ))?;
            let mut values = io.lib(gather_f8_rows(
                &mut io.shared(),
                WINDOW,
                descriptor,
                m,
                packed_k,
                instruction.a_format,
                false,
                instruction.transpose_a,
                None,
                instruction.padded_atoms(),
            ))?;
            if instruction.sfa_lanes != 32 {
                return Err(OpError::unsupported(
                    "128-lane SFA layout (SM107) is not modeled",
                ));
            }
            let scales = io.lib(mxf8_scale_values(
                &mut io.bytes(),
                sfa,
                instruction.sfa_id,
                m,
                mxf8_scale_layout(m, n, 1, false),
            ))?;
            apply_row_scales(&mut values, packed_k, &scales, 0, instruction.negate_a);
            values
        }
        TcA::Tmem(_) => {
            io.lib(validate_tmem_a_transpose(instruction.transpose_a))?;
            let address = tmem_a_address(payload.a)?;
            let columns = io.lib(instruction.a_format.block_tmem_columns(packed_k))?;
            let (rows, a_layout, _) = io.lib(scaled_tmem_a_geometry(
                m,
                packed_k,
                columns,
                1,
                true,
                instruction.sfa_id,
                packed_k,
                instruction.sfa_lanes,
            ))?;
            let mut values = vec![0.0_f32; a_layout.packed_a_banks() * m * packed_k];
            let format = instruction.a_format;
            io.lib(gather_scaled_tmem_a_cta(
                &mut io.words(),
                &mut io.bytes(),
                &mut values,
                0,
                address,
                m,
                packed_k,
                rows,
                columns,
                a_layout,
                sfa,
                instruction.sfa_id,
                packed_k,
                decode_ue8m0_scale,
                instruction.negate_a,
                &|words: &[u32]| format.decode_block_tmem_row(words, packed_k),
                instruction.sfa_lanes,
            ))?;
            values
        }
    };
    io.lib(validate_f8_gather(
        1,
        k,
        instruction.b_format,
        instruction.transpose_b,
    ))?;
    let mut b = io.lib(gather_f8_rows(
        &mut io.shared(),
        WINDOW,
        b_descriptor,
        n,
        k,
        instruction.b_format,
        false,
        instruction.transpose_b,
        None,
        instruction.padded_atoms(),
    ))?;
    let scales = io.lib(mxf8_scale_values(
        &mut io.bytes(),
        sfb,
        instruction.sfb_id,
        n,
        mxf8_scale_layout(m, n, 1, true),
    ))?;
    apply_row_scales(&mut b, k, &scales, 0, instruction.negate_b);
    let window = Window {
        taddr: payload.d_taddr,
        m,
        n,
        layout: DenseTmemLayout::D,
        mask: [0; 4],
    };
    let a_in_tmem = matches!(payload.args.a, TcA::Tmem(_));
    float_tail(
        io,
        tmem_write,
        &window,
        CellDtype::F32,
        k,
        &a,
        &b,
        payload.enable_input_d,
        a_in_tmem && layout.packed_a_banks() > 1,
        || Ok(1.0),
    )
}

pub(super) fn run(
    payload: &TcgenMmaPayload,
    smem: &dyn Fn(u32, &mut [u8]) -> OpResult,
    tmem_read: &dyn Fn(u32, u32, &mut [u8]) -> OpResult,
    tmem_write: &mut dyn FnMut(u32, u32, &[u8]) -> OpResult,
) -> OpResult {
    check_form(payload)?;
    let io = Io {
        smem,
        tmem_read,
        stash: RefCell::new(None),
    };
    match payload.args.kind {
        TcMmaKind::F16 | TcMmaKind::Tf32 => float_mma(&io, payload, tmem_write),
        TcMmaKind::F8f6f4 => f8f6f4_mma(&io, payload, tmem_write),
        TcMmaKind::I8 => integer_mma(&io, payload, tmem_write),
        TcMmaKind::MxF4 | TcMmaKind::MxF4Nvf4 => mxf4_mma(&io, payload, tmem_write),
        TcMmaKind::MxF8f6f4 => mxf8f6f4_mma(&io, payload, tmem_write),
    }
}
