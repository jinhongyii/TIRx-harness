//! Block-scaled `tcgen05.mma` drivers: kind::mxf4/mxf4nvf4 (+ the legacy
//! sparse kind::mxf4 form) and kind::mxf8f6f4 (+ `.sp`, `.lut_b`).

use super::*;

// ---------------------------------------------------------------------------
// kind::mxf4 / mxf4nvf4 block-scale (raw_tcgen05_mma_block_scale_mxf4 and
// raw_tcgen05_mma_sp_block_scale_mxf4_e8m0_ss_cta1)
// ---------------------------------------------------------------------------

fn sparse_mxf4_mma(
    io: &Io<'_>,
    payload: &TcgenMmaPayload,
    form: &Form,
    metadata: u32,
    tmem_write: TmemWrite<'_>,
) -> OpResult {
    if payload.args.kind != TcMmaKind::MxF4 || form.cta_group != 1 || form.a_in_tmem {
        return Err(OpError::unsupported(format!(
            "tcgen05.mma.sp.kind::{:?}.block_scale cta_group::{} {} A: only the legacy \
             kind::mxf4 UE8M0 SS cta_group::1 sparse form is modeled",
            payload.args.kind,
            form.cta_group,
            if form.a_in_tmem { "TMEM" } else { "shared" }
        )));
    }
    if payload
        .args
        .block_scale
        .is_some_and(|(_, _, block)| block != 32)
    {
        return Err(OpError::invalid(
            "sparse kind::mxf4 uses block32 UE8M0 scales",
        ));
    }
    let instruction = io.lib(decode_sparse_mxf4(payload.idesc))?;
    let (sfa, sfb) = payload
        .scale_taddrs
        .ok_or_else(|| OpError::invalid("block-scale MMA has no scale addresses"))?;
    let a_descriptor = io.lib(decode_matrix_descriptor(payload.a))?;
    let b_descriptor = io.lib(decode_matrix_descriptor(payload.b_desc))?;
    let packed_a = io.lib(gather_sparse_mxf4_e8m0_rows(
        &mut io.shared(0),
        &mut io.bytes(0),
        WINDOW,
        a_descriptor,
        128,
        64,
        sfa,
        instruction.sfa_id,
        instruction.negate_a,
    ))?;
    let a = io.lib(expand_sparse_mxf4_a(&packed_a, |row, chunk| {
        sparse_mxf4_metadata_code(&mut io.words(0), metadata, row, chunk)
    }))?;
    let b = io.lib(gather_sparse_mxf4_e8m0_rows(
        &mut io.shared(0),
        &mut io.bytes(0),
        WINDOW,
        b_descriptor,
        instruction.n,
        128,
        sfb,
        instruction.sfb_id,
        instruction.negate_b,
    ))?;
    let window = Window {
        taddr: payload.d_taddr,
        m: 128,
        n: instruction.n,
        layout: DenseTmemLayout::D,
        mask: [0; 8],
        cta_group: 1,
    };
    let input_d = if payload.enable_input_d {
        Some(window.read_f32(io)?)
    } else {
        None
    };
    let output = io.lib(mma_f32_abt_increasing_k(
        128,
        instruction.n,
        128,
        &a,
        &b,
        input_d.as_deref().map(|values| (values, 1.0)),
    ))?;
    window.write_f32(io, tmem_write, &output)
}

pub(super) fn mxf4_mma(
    io: &Io<'_>,
    payload: &TcgenMmaPayload,
    options: &TcMmaOptions,
    form: &Form,
    tmem_write: TmemWrite<'_>,
) -> OpResult {
    if let Some(metadata) = form.metadata {
        return sparse_mxf4_mma(io, payload, form, metadata, tmem_write);
    }
    let idesc = payload.idesc;
    let cg = form.cta_group;
    let arch = descriptor_layout(options.arch);
    let block = payload.args.block_scale.map(|(_, _, block)| block);
    let spelling = mxf4_spellings(payload.args.kind, idesc, block)?[0];
    let instruction = io.lib(decode_mxf4_for_cta_group(
        idesc,
        spelling,
        cg,
        arch,
        options.fixed_vectors,
    ))?;
    let (m, n, k) = (instruction.m, instruction.n, instruction.k);
    let (sfa, sfb) = payload
        .scale_taddrs
        .ok_or_else(|| OpError::invalid("block-scale MMA has no scale addresses"))?;
    let b_descriptor = io.lib(decode_packed_matrix_descriptor(
        payload.b_desc,
        arch,
        k / 2,
        false,
    ))?;
    let a = if form.a_in_tmem {
        let address = tmem_a_address(payload.a)?;
        let columns = k / 8;
        let (rows, layout, _) = io.lib(scaled_tmem_a_geometry(
            m,
            k,
            columns,
            cg,
            true,
            (instruction.sfa_id, instruction.block_elements, instruction.sfa_lanes),
        ))?;
        let mut values = vec![0.0_f32; layout.packed_a_banks() * m * k];
        for cta in 0..cg {
            io.lib(gather_scaled_tmem_a_cta(
                &mut io.words(cta),
                &mut io.bytes(cta),
                &mut values,
                cta,
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
        }
        values
    } else {
        let descriptor = io.lib(decode_packed_matrix_descriptor(
            payload.a,
            arch,
            k / 2,
            false,
        ))?;
        let rows = m / cg;
        io.per_cta(cg, |cta| {
            gather_mxf4_rows(
                &mut io.shared(cta),
                &mut io.bytes(cta),
                WINDOW,
                descriptor,
                rows,
                0,
                rows,
                sfa,
                instruction.sfa_id,
                spelling.decoder(),
                instruction.negate_a,
                k,
                instruction.block_elements,
                instruction.sfa_lanes,
            )
        })?
    };
    // SFB rows are joint across the pair: CTA c reads rows c*n/2.. of the
    // whole-N scale matrix from its own TMEM.
    let rows = n / cg;
    let b = io.per_cta(cg, |cta| {
        gather_mxf4_rows(
            &mut io.shared(cta),
            &mut io.bytes(cta),
            WINDOW,
            b_descriptor,
            rows,
            cta * rows,
            n,
            sfb,
            instruction.sfb_id,
            spelling.decoder(),
            instruction.negate_b,
            k,
            instruction.block_elements,
            32,
        )
    })?;
    let layout = if cg == 1 {
        DenseTmemLayout::D
    } else {
        io.lib(cta1_dense_tmem_layout(m / 2, true))?
    };
    let window = Window {
        taddr: payload.d_taddr,
        m,
        n,
        layout,
        mask: [0; 8],
        cta_group: cg,
    };
    let input_d = if payload.enable_input_d {
        Some(window.read_f32(io)?)
    } else {
        None
    };
    // Legacy `RawMmaTail::run`: the plain increasing-K core.
    let output = io.lib(mma_f32_abt_increasing_k(
        m,
        n,
        k,
        &a,
        &b,
        input_d.as_deref().map(|values| (values, 1.0)),
    ))?;
    window.write_f32(io, tmem_write, &output)
}

// ---------------------------------------------------------------------------
// kind::mxf8f6f4 block-scale (raw_tcgen05_mma_block_scale_mxf8f6f4)
// ---------------------------------------------------------------------------

/// Legacy `raw_tcgen05_gather_mxf8f6f4_matrix`: narrow (or LUT-B) rows of
/// every CTA, then one UE8M0 scale per row (replicated SM100 layout, or the
/// SM107 128-lane SFA layout), SFB copies checked equal across the pair.
#[allow(clippy::too_many_arguments)]
/// `numsim_oplib::tcgen05::scale::mxf8_scale_values` over the MMA's TMEM
/// reads, without its per-call location and byte vectors (perf): the same
/// location checks first (every row and replica, before any read), the same
/// one-cell reads in the same (row, replica) order, then the same replica
/// agreement check and UE8M0 decode per row. Same values and errors.
fn read_mxf8_scales(
    io: &Io<'_>,
    cta: usize,
    address: u32,
    scale_id: usize,
    rows: usize,
    layout: ScaleLayout,
) -> OpResult<Vec<f32>> {
    thread_local! {
        static SCRATCH: std::cell::RefCell<ScaleScratch> = std::cell::RefCell::new(ScaleScratch::default());
    }
    if scale_id >= 4 {
        return Err(io.lift(LibError::message("raw TCGEN scale byte is outside TMEM")));
    }
    let replicas = layout.replicas();
    if let Some(values) = replicated_scales_fast(io, cta, address, scale_id, rows, layout)? {
        return Ok(values);
    }
    SCRATCH.with(|scratch| {
        let scratch = &mut *scratch.borrow_mut();
        // Every location first, before any read (as `mxf8_scale_locations`).
        scratch.locations.clear();
        for row in 0..rows {
            for replica in 0..replicas {
                scratch.locations.push(io.lib(layout.location(address, row, replica))?);
            }
        }
        scratch.bytes.clear();
        scratch.bytes.resize(scratch.locations.len(), 0);
        scratch.served.clear();
        scratch.served.resize(scratch.locations.len(), false);
        // Window pass (no observable effect besides order-free read notes):
        // when a lane's cells form one contiguous column range, read it in one
        // piece from the window; every other cell keeps its own read below.
        if io.windows.is_some() {
            window_lane_runs(io, cta, scale_id, scratch);
        }
        // The remaining cells, one read each, in (row, replica) order.
        for (index, &(lane, column)) in scratch.locations.iter().enumerate() {
            if !scratch.served[index] {
                scratch.bytes[index] = io.lib(io.cell(cta, lane, column))?[scale_id];
            }
        }
        let mut values = Vec::with_capacity(rows);
        for copies in scratch.bytes.chunks_exact(replicas) {
            if copies.iter().any(|&bits| bits != copies[0]) {
                return Err(io.lift(LibError::message("raw TCGEN block-scale replicas disagree")));
            }
            values.push(io.lib(decode_ue8m0_scale(copies[0]))?);
        }
        Ok(values)
    })
}

/// [`read_mxf8_scales`] for the replicated layout when no location can fail:
/// `ScaleLayout::location` puts `(row, replica)` at lane `base_lane + 32 *
/// replica + row % 32`, column `base_column + row / 32`, and fails only past
/// lane 127 or column 0xffff, so checking the largest lane and column up front
/// is the same as checking every location. Each lane's cells (rows `l`,
/// `l + 32`, ...) are consecutive columns: read once from the window when it
/// serves the run, else per cell in (row, replica) order as the general path.
/// `None` = take the general path (another layout, a failing location, or
/// no windows).
fn replicated_scales_fast(
    io: &Io<'_>,
    cta: usize,
    address: u32,
    scale_id: usize,
    rows: usize,
    layout: ScaleLayout,
) -> OpResult<Option<Vec<f32>>> {
    thread_local! {
        static FAST: std::cell::RefCell<(Vec<u8>, Vec<bool>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
    }
    if !matches!(layout, ScaleLayout::Replicated) || rows == 0 || io.windows.is_none() {
        return Ok(None);
    }
    let replicas = layout.replicas();
    let Ok((base_lane, base_column)) = block_scale_address(address) else { return Ok(None) };
    let lanes_used = rows.min(32);
    if base_lane + (replicas - 1) * 32 + lanes_used > 128 || base_column + (rows - 1) / 32 > 0xffff {
        return Ok(None);
    }
    FAST.with(|fast| {
        let (bytes, served) = &mut *fast.borrow_mut();
        bytes.clear();
        bytes.resize(rows * replicas, 0);
        served.clear();
        served.resize(rows * replicas, false);
        let mut run = [0_u8; 4 * 64];
        // Every `(lane, replica)` run as `Io::try_window_tmem` would serve it,
        // with the window set borrowed once (perf, W4).
        let mut serve = |window: Option<&TcWindow<'_>>, reads: Option<&RefCell<TcWindowReads>>| {
            let Some(window) = window else { return };
            for l in 0..lanes_used {
                let count = (rows - l).div_ceil(32);
                if count > 64 {
                    continue;
                }
                for replica in 0..replicas {
                    let lane = base_lane + 32 * replica + l;
                    let piece = &mut run[..4 * count];
                    let in_lane = lane < addr::TMEM_LANES as usize
                        && base_column < addr::TMEM_COLS as usize
                        && base_column * 4 + piece.len() <= addr::TMEM_COLS as usize * 4;
                    let offset = addr::tmem_byte_offset(lane as u32, base_column as u32);
                    if in_lane && window.try_read(offset, piece) {
                        if let Some(reads) = reads {
                            reads.borrow_mut().push((TcSpace::Tmem, cta as u32, offset, piece.len() as u64));
                        }
                        for t in 0..count {
                            let index = (l + 32 * t) * replicas + replica;
                            bytes[index] = run[4 * t + scale_id];
                            served[index] = true;
                        }
                    }
                }
            }
        };
        if let Some(windows) = io.windows {
            windows.with_window_dyn(TcSpace::Tmem, cta as u32, &mut serve);
        }
        for row in 0..rows {
            for replica in 0..replicas {
                let index = row * replicas + replica;
                if !served[index] {
                    let (lane, column) = (base_lane + 32 * replica + row % 32, base_column + row / 32);
                    bytes[index] = io.lib(io.cell(cta, lane, column))?[scale_id];
                }
            }
        }
        let mut values = Vec::with_capacity(rows);
        for copies in bytes.chunks_exact(replicas) {
            if copies.iter().any(|&bits| bits != copies[0]) {
                return Err(io.lift(LibError::message("raw TCGEN block-scale replicas disagree")));
            }
            values.push(io.lib(decode_ue8m0_scale(copies[0]))?);
        }
        Ok(Some(values))
    })
}

/// Reusable buffers of [`read_mxf8_scales`].
#[derive(Default)]
struct ScaleScratch {
    locations: Vec<(usize, usize)>,
    bytes: Vec<u8>,
    served: Vec<bool>,
    /// Per TMEM lane: (first column, last column, cell count, start of the
    /// lane's cells in `run`, or `usize::MAX` when the window did not serve it).
    lanes: Vec<(usize, usize, usize, usize)>,
    run: Vec<u8>,
}

/// Serve scale cells from the TMEM window one lane at a time: a lane whose
/// requested cells are exactly the columns `first..=last` (no gaps, no
/// repeats) is one window read of those cells, the same bytes the per-cell
/// reads would take. A lane the window cannot serve (out of range, invalid
/// bytes, no window) is left to the per-cell reads.
fn window_lane_runs(io: &Io<'_>, cta: usize, scale_id: usize, scratch: &mut ScaleScratch) {
    const UNUSED: (usize, usize, usize, usize) = (usize::MAX, 0, 0, usize::MAX);
    scratch.lanes.clear();
    scratch.lanes.resize(addr::TMEM_LANES as usize, UNUSED);
    for &(lane, column) in &scratch.locations {
        let Some(entry) = scratch.lanes.get_mut(lane) else { return };
        *entry = (entry.0.min(column), entry.1.max(column), entry.2 + 1, usize::MAX);
    }
    scratch.run.clear();
    for lane in 0..scratch.lanes.len() {
        let (first, last, count, _) = scratch.lanes[lane];
        if count == 0 || last - first + 1 != count {
            continue;
        }
        let start = scratch.run.len();
        scratch.run.resize(start + count * 4, 0);
        if io.try_window_tmem(cta as u32, lane as u32, first as u32, &mut scratch.run[start..]) {
            scratch.lanes[lane].3 = start;
        } else {
            scratch.run.truncate(start);
        }
    }
    for (index, &(lane, column)) in scratch.locations.iter().enumerate() {
        let (first, _, _, start) = scratch.lanes[lane];
        if start != usize::MAX {
            scratch.bytes[index] = scratch.run[start + (column - first) * 4 + scale_id];
            scratch.served[index] = true;
        }
    }
}

fn gather_mxf8f6f4(
    io: &Io<'_>,
    cg: usize,
    descriptor: MatrixDescriptor,
    rows_per_cta: usize,
    k: usize,
    format: NarrowFormat,
    transpose: bool,
    joint: bool,
    scale_address: u32,
    scale_id: usize,
    scale_layout: ScaleLayout,
    negate: bool,
    lanes_per_column: usize,
    lut_b: Option<(u32, usize)>,
    padded_atoms: bool,
) -> OpResult<Vec<f32>> {
    let mut values = match lut_b {
        Some((lookup, segment)) => io.per_cta(cg, |cta| {
            gather_lut_b_rows(
                &mut io.shared(cta),
                &mut io.words(cta),
                WINDOW,
                descriptor,
                segment,
                lookup,
                rows_per_cta,
                false,
            )
        })?,
        None => {
            io.lib(validate_f8_gather(cg, k, format, transpose))?;
            io.per_cta(cg, |cta| {
                io.shared_with(cta, |mut read| gather_f8_rows(
                    &mut read,
                    WINDOW,
                    descriptor,
                    rows_per_cta,
                    k,
                    format,
                    false,
                    transpose,
                    None,
                    padded_atoms,
                ))
            })?
        }
    };
    let mut joint_scales: Option<Vec<f32>> = None;
    for (cta, chunk) in values.chunks_exact_mut(rows_per_cta * k).enumerate() {
        let rows = rows_per_cta * if joint { cg } else { 1 };
        let scales = if lanes_per_column == 32 {
            read_mxf8_scales(io, cta, scale_address, scale_id, rows, scale_layout)?
        } else {
            (0..rows)
                .map(|row| {
                    io.lib(read_block_scale(
                        &mut io.bytes(cta),
                        scale_address,
                        scale_id,
                        row,
                        0,
                        decode_ue8m0_scale,
                        rows,
                        lanes_per_column,
                    ))
                })
                .collect::<OpResult<Vec<_>>>()?
        };
        if joint {
            io.lib(check_joint_scales(joint_scales.as_deref(), &scales))?;
            joint_scales = Some(scales.clone());
        }
        apply_row_scales(
            chunk,
            k,
            &scales,
            if joint { cta * rows_per_cta } else { 0 },
            negate,
        );
    }
    Ok(values)
}

pub(super) fn mxf8f6f4_mma(
    io: &Io<'_>,
    payload: &TcgenMmaPayload,
    options: &TcMmaOptions,
    form: &Form,
    tmem_write: TmemWrite<'_>,
) -> OpResult {
    let idesc = payload.idesc;
    let cg = form.cta_group;
    let arch = descriptor_layout(options.arch);
    if let Some((_, _, block)) = payload.args.block_scale {
        if block != 32 {
            return Err(OpError::invalid(format!(
                "kind::mxf8f6f4 block size must be 32, got {block}"
            )));
        }
    }
    let instruction = io.lib(decode_mxf8f6f4(idesc, cg, arch))?;
    if instruction.sparse != form.metadata.is_some() {
        return Err(OpError::invalid(
            "MXF8F6F4 sparsity bit and metadata operand disagree",
        ));
    }
    if let Some(metadata) = form.metadata {
        io.lib(validate_sparse_metadata_address(metadata, payload.d_taddr))?;
    }
    let (m, n, k) = (instruction.m, instruction.n, instruction.k);
    let packed_k = instruction.packed_k();
    let (sfa, sfb) = payload
        .scale_taddrs
        .ok_or_else(|| OpError::invalid("block-scale MMA has no scale addresses"))?;
    let layout = io.lib(cta1_dense_tmem_layout(m / cg, true))?;
    let lut_b = options.lut_b;
    io.lib(validate_lut_b(
        lut_b,
        k,
        instruction.b_format,
        instruction.transpose_b,
    ))?;
    let transposed = instruction.transpose_a || instruction.transpose_b;
    let b_descriptor = if lut_b.is_some() {
        io.lib(f8_b_descriptor(payload.b_desc, arch, lut_b))?
    } else {
        let row_bytes = k / 16
            * if instruction.padded_atoms() {
                16
            } else {
                instruction.b_format.shared_atom_stride(k)
            };
        io.lib(decode_packed_matrix_descriptor(
            payload.b_desc,
            arch,
            row_bytes,
            transposed,
        ))?
    };
    let a = if form.a_in_tmem {
        io.lib(validate_tmem_a_transpose(instruction.transpose_a))?;
        let address = tmem_a_address(payload.a)?;
        let columns = io.lib(instruction.a_format.block_tmem_columns(packed_k))?;
        let (rows, a_layout, _) = io.lib(scaled_tmem_a_geometry(
            m,
            packed_k,
            columns,
            cg,
            true,
            (instruction.sfa_id, packed_k, instruction.sfa_lanes),
        ))?;
        let mut values = vec![0.0_f32; a_layout.packed_a_banks() * m * packed_k];
        let format = instruction.a_format;
        for cta in 0..cg {
            io.lib(gather_scaled_tmem_a_cta(
                &mut io.words(cta),
                &mut io.bytes(cta),
                &mut values,
                cta,
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
        }
        values
    } else {
        let row_bytes = packed_k / 16 * instruction.a_format.shared_atom_stride(packed_k);
        let descriptor = io.lib(decode_packed_matrix_descriptor(
            payload.a, arch, row_bytes, transposed,
        ))?;
        gather_mxf8f6f4(
            io,
            cg,
            descriptor,
            m / cg,
            packed_k,
            instruction.a_format,
            instruction.transpose_a,
            false,
            sfa,
            instruction.sfa_id,
            mxf8_scale_layout(m, n, cg, false),
            instruction.negate_a,
            instruction.sfa_lanes,
            None,
            instruction.padded_atoms(),
        )?
    };
    let b = gather_mxf8f6f4(
        io,
        cg,
        b_descriptor,
        n / cg,
        k,
        instruction.b_format,
        instruction.transpose_b,
        true,
        sfb,
        instruction.sfb_id,
        mxf8_scale_layout(m, n, cg, true),
        instruction.negate_b,
        32,
        lut_b.map(|address| (address, ((payload.b_desc >> 53) & 1) as usize)),
        instruction.padded_atoms(),
    )?;
    // CTA1 is legacy `LaneCells` (Layout D); the pair uses `layout`.
    let window = Window {
        taddr: payload.d_taddr,
        m,
        n,
        layout: if cg == 1 { DenseTmemLayout::D } else { layout },
        mask: [0; 8],
        cta_group: cg,
    };
    let input_d = if payload.enable_input_d {
        Some(window.read_f32(io)?)
    } else {
        None
    };
    let input = input_d.as_deref().map(|values| (values, 1.0));
    let output = match form.metadata {
        Some(metadata) => {
            let kind = FloatKind::SparseNarrow {
                a_format: instruction.a_format,
                b_format: instruction.b_format,
                descriptor_layout: arch,
            };
            let metadata_layout = SparseMetadataLayout::Narrow { k };
            io.lib(sparse_float_mma(
                m,
                n,
                k,
                &a,
                &b,
                input,
                kind,
                layout,
                cg,
                |cta, row, chunk| io.metadata_code(cta, metadata, metadata_layout, row, chunk),
            ))?
        }
        None => io.lib(mma_dense_tail(
            (m, n, k),
            &a,
            &b,
            input,
            form.a_in_tmem,
            layout,
        ))?,
    };
    window.write_f32(io, tmem_write, &output)
}
