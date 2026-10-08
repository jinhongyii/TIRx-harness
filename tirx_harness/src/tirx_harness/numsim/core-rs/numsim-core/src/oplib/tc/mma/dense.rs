//! Dense-family `tcgen05.mma` drivers: kind::f16/tf32 (+ `.sp`, `.ws`,
//! `.ashift`, CTA pairs), dense kind::f8f6f4 (+ `.ws`, `.lut_b`, `.ashift`)
//! and kind::i8 / `.ti16` (+ `.sp`, `.ws`, `.ashift`).

use super::*;

// ---------------------------------------------------------------------------
// kind::f16 / kind::tf32 / sparse kind::f8f6f4 (raw_tcgen05_mma_float)
// ---------------------------------------------------------------------------

pub(super) fn float_mma(
    io: &Io<'_>,
    payload: &TcgenMmaPayload,
    options: &TcMmaOptions,
    form: &Form,
    tmem_write: TmemWrite<'_>,
) -> OpResult {
    let idesc = payload.idesc;
    let cg = form.cta_group;
    let sparse = form.metadata.is_some();
    let kind = match payload.args.kind {
        TcMmaKind::Tf32 => FloatKind::Tf32,
        TcMmaKind::F16 => FloatKind::B16 {
            a_bf16: (idesc >> 7) & 7 == 1,
            b_bf16: (idesc >> 10) & 7 == 1,
        },
        _ => FloatKind::SparseNarrow {
            a_format: io.lib(NarrowFormat::decode((idesc >> 7) & 7, "A"))?,
            b_format: io.lib(NarrowFormat::decode((idesc >> 10) & 7, "B"))?,
            descriptor_layout: descriptor_layout(options.arch),
        },
    };
    let instruction = io.lib(kind.instruction(idesc, cg, form.ws_mask.is_some(), sparse))?;
    let packed_k = kind.packed_k(idesc);
    let (m, n) = (instruction.m, instruction.n);
    if payload.args.ashift {
        if cg == 1 && m != 128 {
            return Err(OpError::invalid(
                "tcgen05.mma.ashift requires M=128 for cta_group::1",
            ));
        }
        if cg == 2 && m != 256 && sparse {
            // Legacy `analysis_incomplete`: PTX lists M128/M256, but sparse
            // M128 traps in SM100 probes.
            return Err(OpError::unsupported(
                "tcgen_sparse_m128_ashift_unmodeled: tcgen05.mma.sp.ashift cta_group::2 M=128",
            ));
        }
    }
    let b_descriptor = io.lib(decode_matrix_descriptor_for_layout(
        payload.b_desc,
        kind.descriptor_layout(),
    ))?;
    let layout = io.lib(cta1_dense_tmem_layout(
        m / cg,
        form.ws_mask.is_some() || (cg == 2 && !sparse),
    ))?;
    let gather_shared = |descriptor: MatrixDescriptor,
                         rows: usize,
                         columns: usize,
                         transpose: bool,
                         negate: bool,
                         mask: Option<ColumnMask>,
                         is_b: bool|
     -> OpResult<Vec<f32>> {
        match kind {
            FloatKind::SparseNarrow {
                a_format, b_format, ..
            } => {
                let format = if is_b { b_format } else { a_format };
                io.lib(validate_f8_gather(cg, columns, format, transpose))?;
                io.per_cta(cg, |cta| {
                    gather_f8_rows(
                        &mut io.shared(cta),
                        WINDOW,
                        descriptor,
                        rows,
                        columns,
                        format,
                        negate,
                        transpose,
                        mask,
                        true,
                    )
                })
            }
            FloatKind::Tf32 => io.per_cta(cg, |cta| {
                gather_tf32_rows(
                    &mut io.shared(cta),
                    WINDOW,
                    descriptor,
                    rows,
                    columns,
                    transpose,
                    negate,
                    mask,
                )
            }),
            FloatKind::B16 { a_bf16, b_bf16 } => {
                let bf16 = if is_b { b_bf16 } else { a_bf16 };
                // Legacy CTA2 B16 gathers take no column mask (.ws is CTA1).
                let mask = if cg == 1 { mask } else { None };
                io.per_cta(cg, |cta| {
                    gather_b16_chunked(
                        &mut io.shared(cta),
                        descriptor,
                        rows,
                        columns,
                        transpose,
                        mask,
                        |bits| Ok(decode_b16(bits, bf16, negate)),
                    )
                })
            }
        }
    };
    let a = if form.a_in_tmem {
        io.lib(validate_tmem_a_transpose(instruction.transpose_a))?;
        let address = tmem_a_address(payload.a)?;
        let columns = kind.tmem_a_columns(idesc);
        let negate = instruction.negate_a;
        let sign = move |value: f32| if negate { -value } else { value };
        match kind {
            FloatKind::SparseNarrow { a_format, .. } => {
                io.tmem_a(cg, address, m, layout, columns, |word| {
                    Ok(a_format.decode_tmem_word(word)?.map(sign))
                })?
            }
            FloatKind::Tf32 => {
                io.tmem_a(cg, address, m, layout, CTA1_PACKED_A_COLUMNS, |word| {
                    Ok([sign(tf32_payload_to_f32(word))])
                })?
            }
            FloatKind::B16 { a_bf16, .. } => {
                io.tmem_a(cg, address, m, layout, CTA1_PACKED_A_COLUMNS, |word| {
                    Ok(decode_b16_word(word, a_bf16, negate))
                })?
            }
        }
    } else {
        let descriptor = io.lib(decode_matrix_descriptor_for_layout(
            payload.a,
            kind.descriptor_layout(),
        ))?;
        gather_shared(
            descriptor,
            m / cg,
            packed_k,
            instruction.transpose_a,
            instruction.negate_a,
            None,
            false,
        )?
    };
    let k = packed_k * if sparse { 2 } else { 1 };
    if let Some(metadata) = form.metadata {
        io.lib(validate_sparse_metadata_address(metadata, payload.d_taddr))?;
    }
    let mask = column_mask(io, form, m, n, idesc)?;
    let b = gather_shared(
        b_descriptor,
        n / cg,
        k,
        instruction.transpose_b,
        instruction.negate_b,
        mask,
        true,
    )?;
    let window = Window {
        taddr: payload.d_taddr,
        m,
        n,
        layout,
        mask: form.mask,
        cta_group: cg,
    };
    let cell = instruction.cell_dtype;
    let input_d = if payload.enable_input_d {
        Some(window.read(io, |bytes| cell.decode(bytes), f32::NAN)?)
    } else {
        None
    };
    let scale = io.lib(tf32_family_input_scale(
        payload.scale_input_d.unwrap_or(0) as usize
    ))?;
    let input = input_d.as_deref().map(|values| (values, scale));
    let output = match form.metadata {
        Some(metadata) => {
            let metadata_layout = kind.metadata_layout(idesc);
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
    window.write(io, tmem_write, |value| cell.encode(value), &output)?;
    if payload.args.ashift {
        let columns = if kind.tmem_a_columns(idesc) == 16 {
            16
        } else {
            8
        };
        shift_a(io, tmem_write, tmem_a_address(payload.a)?, columns, cg)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// dense kind::f8f6f4 (raw_tcgen05_mma_f8f6f4_cta{1,2})
// ---------------------------------------------------------------------------

pub(super) fn f8f6f4_mma(
    io: &Io<'_>,
    payload: &TcgenMmaPayload,
    options: &TcMmaOptions,
    form: &Form,
    tmem_write: TmemWrite<'_>,
) -> OpResult {
    let idesc = payload.idesc;
    let cg = form.cta_group;
    let arch = descriptor_layout(options.arch);
    let a_format = io.lib(NarrowFormat::decode((idesc >> 7) & 7, "A"))?;
    let b_format = io.lib(NarrowFormat::decode((idesc >> 10) & 7, "B"))?;
    let d_f16 = (idesc >> 4) & 3 == 0;
    let instruction = io.lib(decode_f8f6f4(
        idesc,
        a_format,
        b_format,
        d_f16,
        cg as u32,
        arch.supports_f8f6f4_k64(),
        form.ws_mask.is_some(),
        false,
    ))?;
    let (m, n, k) = (instruction.m, instruction.n, instruction.k);
    let lut_b = options.lut_b;
    io.lib(validate_lut_b(lut_b, k, b_format, instruction.transpose_b))?;
    let b_descriptor = io.lib(f8_b_descriptor(payload.b_desc, arch, lut_b))?;
    let layout = if cg == 1 {
        io.lib(cta1_dense_tmem_layout(m, form.ws_mask.is_some()))?
    } else {
        io.lib(cta1_dense_tmem_layout(m / 2, true))?
    };
    if payload.args.ashift && !matches!(m, 128 | 256) {
        return Err(OpError::invalid(
            "tcgen05.mma.ashift requires TMEM A and M=128/256",
        ));
    }
    let a = if form.a_in_tmem {
        let address = io.lib(f8_tmem_a_address(payload.a, instruction.transpose_a))?;
        let negate = instruction.negate_a;
        io.tmem_a(cg, address, m, layout, k / 4, |word| {
            Ok(a_format
                .decode_tmem_word(word)?
                .map(|value| if negate { -value } else { value }))
        })?
    } else {
        let descriptor = io.lib(decode_matrix_descriptor_for_layout(payload.a, arch))?;
        io.lib(validate_f8_gather(cg, k, a_format, instruction.transpose_a))?;
        io.per_cta(cg, |cta| {
            gather_f8_rows(
                &mut io.shared(cta),
                WINDOW,
                descriptor,
                m / cg,
                k,
                a_format,
                instruction.negate_a,
                instruction.transpose_a,
                None,
                false,
            )
        })?
    };
    let mask = column_mask(io, form, m, n, idesc)?;
    let b = match lut_b {
        Some(lookup) => {
            let segment = ((payload.b_desc >> 53) & 1) as usize;
            io.per_cta(cg, |cta| {
                gather_lut_b_rows(
                    &mut io.shared(cta),
                    &mut io.words(cta),
                    WINDOW,
                    b_descriptor,
                    segment,
                    lookup,
                    n / cg,
                    instruction.negate_b,
                )
            })?
        }
        None => {
            io.lib(validate_f8_gather(cg, k, b_format, instruction.transpose_b))?;
            io.per_cta(cg, |cta| {
                gather_f8_rows(
                    &mut io.shared(cta),
                    WINDOW,
                    b_descriptor,
                    n / cg,
                    k,
                    b_format,
                    instruction.negate_b,
                    instruction.transpose_b,
                    mask,
                    false,
                )
            })?
        }
    };
    // A `.f16` destination rounds once on store (legacy `RawMmaCellDtype`).
    let cell = CellDtype::from_half(d_f16);
    let window = Window {
        taddr: payload.d_taddr,
        m,
        n,
        layout,
        mask: form.mask,
        cta_group: cg,
    };
    let input_d = if payload.enable_input_d {
        Some(window.read(io, |bytes| cell.decode(bytes), f32::NAN)?)
    } else {
        None
    };
    let output = io.lib(mma_dense_tail(
        (m, n, k),
        &a,
        &b,
        input_d.as_deref().map(|values| (values, 1.0)),
        form.a_in_tmem,
        layout,
    ))?;
    window.write(io, tmem_write, |value| cell.encode(value), &output)?;
    if payload.args.ashift {
        let address = io.lib(f8_tmem_a_address(payload.a, false))?;
        shift_a(io, tmem_write, address, if k == 64 { 16 } else { 8 }, cg)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// kind::i8 / .ti16 (raw_tcgen05_mma_integer)
// ---------------------------------------------------------------------------

pub(super) fn integer_mma(
    io: &Io<'_>,
    payload: &TcgenMmaPayload,
    options: &TcMmaOptions,
    form: &Form,
    tmem_write: TmemWrite<'_>,
) -> OpResult {
    let idesc = payload.idesc;
    let cg = form.cta_group;
    let kind = if options.ti16 {
        IntegerKind::Ti16
    } else {
        IntegerKind::I8
    };
    let sparse = form.metadata.is_some();
    let (m, n, transpose_a, transpose_b) = io.lib(integer_shape(
        kind,
        idesc,
        cg,
        form.ws_mask.is_some(),
        sparse,
    ))?;
    if payload.args.ashift {
        if cg == 1 && m != 128 {
            return Err(OpError::invalid(
                "tcgen05.mma.ashift requires M=128 for cta_group::1",
            ));
        }
        if cg == 2 && m != 256 && sparse {
            return Err(OpError::unsupported(
                "tcgen_sparse_m128_ashift_unmodeled: tcgen05.mma.sp.ashift cta_group::2 M=128",
            ));
        }
    }
    let packed_k = kind.packed_k();
    let k = packed_k * if sparse { 2 } else { 1 };
    let mask = column_mask(io, form, m, n, idesc)?;
    let rows = m / cg;
    let layout = if cg == 1 {
        io.lib(cta1_dense_tmem_layout(m, form.ws_mask.is_some()))?
    } else {
        io.lib(cta1_dense_tmem_layout(rows, !sparse))?
    };
    let b_descriptor = io.lib(decode_matrix_descriptor(payload.b_desc))?;
    let decode_a = |bits| kind.decode(bits, (idesc >> 7) & 7, idesc & (1 << 13) != 0);
    let decode_b = |bits| kind.decode(bits, (idesc >> 10) & 7, idesc & (1 << 14) != 0);
    let a = if form.a_in_tmem {
        io.lib(validate_tmem_a_transpose(transpose_a))?;
        let address = tmem_a_address(payload.a)?;
        match kind {
            IntegerKind::Ti16 => io.tmem_a(
                cg,
                address,
                m,
                layout,
                CTA1_PACKED_A_COLUMNS,
                |word: u32| Ok([decode_a(word as u16)?, decode_a((word >> 16) as u16)?]),
            )?,
            IntegerKind::I8 => io.tmem_a(
                cg,
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
            )?,
        }
    } else {
        let descriptor = io.lib(decode_matrix_descriptor(payload.a))?;
        io.per_cta(cg, |cta| {
            gather_integer_rows(
                &mut io.shared(cta),
                kind,
                WINDOW,
                descriptor,
                rows,
                packed_k,
                transpose_a,
                None,
                decode_a,
            )
        })?
    };
    let a = match form.metadata {
        Some(metadata) => {
            io.lib(validate_sparse_metadata_address(metadata, payload.d_taddr).map_err(|_| {
                LibError::message(
                    "sparse TI16 metadata requires two-column alignment and matching datapath lanes",
                )
            }))?;
            let metadata_layout = SparseMetadataLayout::B16 {
                selector: (idesc & 1) as usize,
            };
            let mut expanded = Vec::new();
            for cta in 0..cg {
                // CTA2 sparse layouts have one A bank per CTA; CTA1 .ws
                // expands its N-selected banks together.
                let packed = if cg == 1 {
                    &a[..]
                } else {
                    &a[cta * rows * packed_k..(cta + 1) * rows * packed_k]
                };
                expanded.extend(io.lib(expand_sparse_2of4(
                    packed,
                    rows,
                    layout,
                    k,
                    |row, chunk| io.metadata_code(cta, metadata, metadata_layout, row, chunk),
                ))?);
            }
            expanded
        }
        None => a,
    };
    let b = io.per_cta(cg, |cta| {
        gather_integer_rows(
            &mut io.shared(cta),
            kind,
            WINDOW,
            b_descriptor,
            n / cg,
            k,
            transpose_b,
            mask,
            decode_b,
        )
    })?;
    let window = Window {
        taddr: payload.d_taddr,
        m,
        n,
        layout,
        mask: form.mask,
        cta_group: cg,
    };
    let mut output: Vec<i64> = if payload.enable_input_d {
        window
            .read(io, i32::from_le_bytes, 0)?
            .into_iter()
            .map(i64::from)
            .collect()
    } else {
        vec![0; m * n]
    };
    io.lib(integer_mma_accumulate(m, n, k, &a, &b, &mut output))?;
    // Accumulate exactly, then keep the low 32 bits (`.satfinite`, I8 bit 3,
    // clamps to s32).
    let saturate = kind == IntegerKind::I8 && idesc & 8 != 0;
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
    )?;
    if payload.args.ashift {
        shift_a(io, tmem_write, tmem_a_address(payload.a)?, 8, cg)?;
    }
    Ok(())
}
