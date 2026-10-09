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
        static BYTES: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    if scale_id >= 4 {
        return Err(io.lift(LibError::message("raw TCGEN scale byte is outside TMEM")));
    }
    let replicas = layout.replicas();
    for row in 0..rows {
        for replica in 0..replicas {
            io.lib(layout.location(address, row, replica))?;
        }
    }
    BYTES.with(|bytes| {
        let mut bytes = bytes.borrow_mut();
        bytes.clear();
        for row in 0..rows {
            for replica in 0..replicas {
                let (lane, column) = io.lib(layout.location(address, row, replica))?;
                bytes.push(io.lib(io.cell(cta, lane, column))?[scale_id]);
            }
        }
        let mut values = Vec::with_capacity(rows);
        for copies in bytes.chunks_exact(replicas) {
            if copies.iter().any(|&bits| bits != copies[0]) {
                return Err(io.lift(LibError::message("raw TCGEN block-scale replicas disagree")));
            }
            values.push(io.lib(decode_ue8m0_scale(copies[0]))?);
        }
        Ok(values)
    })
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
                gather_f8_rows(
                    &mut io.shared(cta),
                    WINDOW,
                    descriptor,
                    rows_per_cta,
                    k,
                    format,
                    false,
                    transpose,
                    None,
                    padded_atoms,
                )
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
