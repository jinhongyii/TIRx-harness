//! tcgen05 descriptor decoders and MMA numerics.
//!
//! Decoders delegate to `numsim_oplib::tcgen05::{smem_desc, instr_desc,
//! integer}` (SM100 field widths). [`tc_mma`] recomposes the legacy drivers
//! (`engine-rs/src/runtime/tcgen_ops.rs`: `raw_tcgen05_mma_float`,
//! `raw_tcgen05_mma_f8f6f4_cta1`, `raw_tcgen05_mma_block_scale_mxf4`,
//! `raw_tcgen05_mma_block_scale_mxf8f6f4`; `tcgen_ops/integer.rs`
//! `raw_tcgen05_mma_integer`) from the `numsim_oplib::tcgen05` pieces in the
//! legacy order: decode/validate -> gather A -> gather B -> read D -> numeric
//! tail -> scatter D. See the [`mma`](numsim_oplib::mma) module docs for the
//! accumulation contract (per-element increasing-K binary32 FMA chain seeded
//! with `D * scale` or `+0`).
//!
//! Modeled (`tc_mma_ctas`): every legacy form: `cta_group::1/2`, A from
//! shared memory or TMEM, `enable_input_d`, `scale_input_d`,
//! `disable_output_lane`; kind::{f16, tf32, f8f6f4, i8 (+ `.ti16`)} dense,
//! `.sp`, `.ws` (zero-column mask), `.ashift`, `.lut_b` (SM107); block-scaled
//! kind::{mxf8f6f4 (+ `.sp`, `.lut_b`), mxf4, mxf4nvf4}, and the legacy sparse
//! kind::mxf4 SS CTA1 form. Collectors do not change numerics
//! (`tc_collector_transition` tracks their state). Still fail closed (as
//! legacy did): `.sp.ashift` CTA-pair M=128, sparse block-scaled FP4 beyond
//! the SS CTA1 UE8M0 form.

mod mma;

use super::{
    InstrDesc, OpError, OpResult, SmemDesc, TcArch, TcMmaOptions, TcSmemRead, TcTmemRead,
    TcTmemWrite,
};
use crate::dtype::Dtype;
use crate::program::{CollectorOp, TcMmaKind};
use numsim_oplib::tcgen05::instr_desc::{
    decode_b16, decode_f8f6f4, decode_mxf4_for_cta_group, decode_mxf8f6f4, decode_sparse_mxf4,
    decode_tf32, mxf4nvf4_vec2x_scale, mxf4nvf4_vec4x_scale, Mxf4ScaleSpelling,
};
use numsim_oplib::tcgen05::integer::{integer_shape, IntegerKind};
use numsim_oplib::tcgen05::narrow::NarrowFormat;
use numsim_oplib::tcgen05::smem_desc::{decode_matrix_descriptor, MatrixDescriptorLayout};

/// Decode a shared-memory matrix descriptor with SM100 field widths.
///
/// `start`, `lbo`, `sbo` are byte values (fields `<< 4`); `swizzle` uses the
/// `tcgen05_encode_matrix_descriptor` numbering (0 none, 1 = 32B, 2 = 64B,
/// 3 = 128B, 4 = 128B with 32B atoms); `version` is bits 46-47 (always 1);
/// `base_offset` / `lbo_mode` are always 0 because the SM100 decoder rejects
/// nonzero base-offset / LBO-mode bits.
pub(super) fn decode_smem_desc(desc: u64) -> OpResult<SmemDesc> {
    let decoded = decode_matrix_descriptor(desc)?;
    let swizzle = match (decoded.swizzle_bits, decoded.swizzle_atom_bytes) {
        (0, _) => 0,
        (1, 16) => 1,
        (2, 16) => 2,
        (3, 16) => 3,
        (2, 32) => 4,
        other => {
            return Err(OpError::invalid(format!(
                "matrix descriptor swizzle {other:?} has no code"
            )))
        }
    };
    let field = |value: usize, what: &str| {
        u32::try_from(value)
            .map_err(|_| OpError::invalid(format!("matrix descriptor {what} overflow")))
    };
    Ok(SmemDesc {
        start: field(decoded.start_address, "start")?,
        lbo: field(decoded.leading_byte_offset, "LBO")?,
        sbo: field(decoded.stride_byte_offset, "SBO")?,
        base_offset: ((desc >> 49) & 0x7) as u8,
        lbo_mode: ((desc >> 52) & 1) as u8,
        swizzle,
        version: ((desc >> 46) & 0x3) as u8,
    })
}

fn narrow_dtype(format: NarrowFormat) -> Dtype {
    match format {
        NarrowFormat::E4M3 => Dtype::E4M3,
        NarrowFormat::E5M2 => Dtype::E5M2,
        NarrowFormat::E2M3 => Dtype::E2M3,
        NarrowFormat::E3M2 => Dtype::E3M2,
        NarrowFormat::E2M1 => Dtype::E2M1,
    }
}

/// The FP4 scale spelling implied by `kind` and the descriptor's scale-type
/// bits. `block` is the block size (16 or 32) when known.
pub(crate) fn mxf4_spellings(
    kind: TcMmaKind,
    idesc: u32,
    block: Option<u8>,
) -> OpResult<Vec<Mxf4ScaleSpelling>> {
    Ok(match (kind, block) {
        (TcMmaKind::MxF4, None | Some(32)) => vec![Mxf4ScaleSpelling::Ue8m0Vec2x],
        (TcMmaKind::MxF4Nvf4, Some(16)) => vec![mxf4nvf4_vec4x_scale(idesc)],
        (TcMmaKind::MxF4Nvf4, Some(32)) => vec![mxf4nvf4_vec2x_scale(idesc)],
        (TcMmaKind::MxF4Nvf4, None) => {
            vec![mxf4nvf4_vec4x_scale(idesc), mxf4nvf4_vec2x_scale(idesc)]
        }
        (kind, block) => {
            return Err(OpError::invalid(format!(
                "{kind:?} block-scale block size {block:?} is invalid"
            )))
        }
    })
}

fn scale_dtype(spelling: Mxf4ScaleSpelling) -> Dtype {
    match spelling.scale_format_bit() {
        0 => Dtype::UE4M3,
        2 => Dtype::UE5M3,
        _ => Dtype::UE8M0,
    }
}

/// Decode one idesc for one CTA group and `.ws` choice (SM100).
fn decode_one(idesc: u32, kind: TcMmaKind, cta: usize, ws: bool) -> OpResult<InstrDesc> {
    let ws_shift = ((idesc >> 30) & 3) as u8;
    let bit = |n: u32| idesc & (1 << n) != 0;
    let sparse = bit(2);
    let common = InstrDesc {
        a_major_mn: bit(15),
        b_major_mn: bit(16),
        negate_a: bit(13),
        negate_b: bit(14),
        sparse,
        max_shift: if ws { ws_shift } else { 0 },
        ..InstrDesc::default()
    };
    let shape = |m: usize, n: usize| -> OpResult<(u16, u16)> {
        Ok((
            u16::try_from(m).map_err(|_| OpError::invalid("idesc M overflow"))?,
            u16::try_from(n).map_err(|_| OpError::invalid("idesc N overflow"))?,
        ))
    };
    Ok(match kind {
        TcMmaKind::F16 => {
            let a_bf16 = (idesc >> 7) & 7 == 1;
            let b_bf16 = (idesc >> 10) & 7 == 1;
            let decoded = decode_b16(idesc, a_bf16, b_bf16, cta as u32, ws, sparse)?;
            let (m, n) = shape(decoded.m, decoded.n)?;
            let ab = if a_bf16 { Dtype::BF16 } else { Dtype::F16 };
            let d = if bit(4) { Dtype::F32 } else { Dtype::F16 };
            InstrDesc {
                m,
                n,
                a: Some(ab),
                b: Some(ab),
                d: Some(d),
                ..common
            }
        }
        TcMmaKind::Tf32 => {
            let decoded = decode_tf32(idesc, cta, ws, sparse)?;
            let (m, n) = shape(decoded.m, decoded.n)?;
            InstrDesc {
                m,
                n,
                a: Some(Dtype::TF32),
                b: Some(Dtype::TF32),
                d: Some(Dtype::F32),
                ..common
            }
        }
        TcMmaKind::F8f6f4 => {
            let a = NarrowFormat::decode((idesc >> 7) & 7, "A")?;
            let b = NarrowFormat::decode((idesc >> 10) & 7, "B")?;
            let d_f16 = (idesc >> 4) & 3 == 0;
            let decoded = decode_f8f6f4(idesc, a, b, d_f16, cta as u32, false, ws, sparse)?;
            let (m, n) = shape(decoded.m, decoded.n)?;
            let d = if d_f16 { Dtype::F16 } else { Dtype::F32 };
            InstrDesc {
                m,
                n,
                a: Some(narrow_dtype(a)),
                b: Some(narrow_dtype(b)),
                d: Some(d),
                ..common
            }
        }
        TcMmaKind::I8 | TcMmaKind::Ti16 => {
            // Format 3/3 is the `.ti16` (s1z4m11) operand spelling; it has no
            // `Dtype`, so `a`/`b` stay `None`.
            let ti16 = (idesc >> 7) & 7 == 3;
            let integer = if ti16 {
                IntegerKind::Ti16
            } else {
                IntegerKind::I8
            };
            let (m, n, _, _) = integer_shape(integer, idesc, cta, ws, sparse)?;
            let (m, n) = shape(m, n)?;
            let int = |format: u32| match format {
                1 => Some(Dtype::S8),
                0 => Some(Dtype::U8),
                _ => None,
            };
            InstrDesc {
                m,
                n,
                a: int((idesc >> 7) & 7),
                b: int((idesc >> 10) & 7),
                d: Some(Dtype::S32),
                // kind::i8 reserves bits 13/14 (checked above); TI16 negates.
                negate_a: ti16 && bit(13),
                negate_b: ti16 && bit(14),
                ..common
            }
        }
        TcMmaKind::MxF8f6f4 => {
            if ws {
                return Err(OpError::invalid("block-scaled MMA has no .ws form"));
            }
            let decoded = decode_mxf8f6f4(idesc, cta, MatrixDescriptorLayout::Sm100)?;
            let (m, n) = shape(decoded.m, decoded.n)?;
            InstrDesc {
                m,
                n,
                a: Some(narrow_dtype(decoded.a_format)),
                b: Some(narrow_dtype(decoded.b_format)),
                d: Some(Dtype::F32),
                sparse: decoded.sparse,
                scale_type: Some(Dtype::UE8M0),
                ..common
            }
        }
        TcMmaKind::MxF4 | TcMmaKind::MxF4Nvf4 => {
            if ws {
                return Err(OpError::invalid("block-scaled MMA has no .ws form"));
            }
            if sparse && kind == TcMmaKind::MxF4 && cta == 1 {
                let decoded = decode_sparse_mxf4(idesc)?;
                let (m, n) = shape(decoded.m, decoded.n)?;
                return Ok(InstrDesc {
                    m,
                    n,
                    a: Some(Dtype::E2M1),
                    b: Some(Dtype::E2M1),
                    d: Some(Dtype::F32),
                    scale_type: Some(Dtype::UE8M0),
                    ..common
                });
            }
            let mut last = None;
            for spelling in mxf4_spellings(kind, idesc, None)? {
                match decode_mxf4_for_cta_group(
                    idesc,
                    spelling,
                    cta,
                    MatrixDescriptorLayout::Sm100,
                    false,
                ) {
                    Ok(decoded) => {
                        let (m, n) = shape(decoded.m, decoded.n)?;
                        return Ok(InstrDesc {
                            m,
                            n,
                            a: Some(Dtype::E2M1),
                            b: Some(Dtype::E2M1),
                            d: Some(Dtype::F32),
                            sparse: false,
                            scale_type: Some(scale_dtype(spelling)),
                            ..common
                        });
                    }
                    Err(error) => last = Some(OpError::from(error)),
                }
            }
            return Err(
                last.unwrap_or_else(|| OpError::invalid("mxf4 descriptor has no scale spelling"))
            );
        }
    })
}

/// Decode for one CTA group; `cta_group::1` also accepts `.ws` shapes
/// (dense first). `max_shift` is reported only for a `.ws` decode.
pub(super) fn decode_instr_desc_for(
    idesc: u32,
    kind: TcMmaKind,
    cta_group: u8,
) -> OpResult<InstrDesc> {
    let cta = match cta_group {
        1 | 2 => usize::from(cta_group),
        other => {
            return Err(OpError::invalid(format!(
                "tcgen05 cta_group {other} is invalid"
            )))
        }
    };
    match decode_one(idesc, kind, cta, false) {
        Ok(value) => Ok(value),
        Err(first)
            if cta == 1
                && !matches!(
                    kind,
                    TcMmaKind::MxF8f6f4 | TcMmaKind::MxF4 | TcMmaKind::MxF4Nvf4
                ) =>
        {
            decode_one(idesc, kind, cta, true).map_err(|_| first)
        }
        Err(first) => Err(first),
    }
}

/// Decode an idesc valid for `cta_group::1` or `::2` (SM100); the first
/// group that accepts wins, else the `::1` error.
pub(super) fn decode_instr_desc(idesc: u32, kind: TcMmaKind) -> OpResult<InstrDesc> {
    match decode_instr_desc_for(idesc, kind, 1) {
        Ok(value) => Ok(value),
        Err(first) => decode_instr_desc_for(idesc, kind, 2).map_err(|_| first),
    }
}

pub(super) fn parse_variant(variant: &str) -> OpResult<TcMmaOptions> {
    let mut options = TcMmaOptions::default();
    for token in variant
        .split(|c: char| c == '.' || c == ',' || c == ';' || c.is_whitespace())
        .filter(|token| !token.is_empty())
    {
        match token.trim_start_matches(':') {
            "ti16" | "kind::ti16" => options.ti16 = true,
            "fixed_vectors"
            | "block16"
            | "block32"
            | "block_size::block16"
            | "block_size::block32" => options.fixed_vectors = true,
            "sm_100" | "sm_100a" | "sm_100f" => options.arch = TcArch::Sm100,
            "sm_103" | "sm_103a" | "sm_103f" => options.arch = TcArch::Sm103,
            "sm_107" | "sm_107a" | "sm_107f" => options.arch = TcArch::Sm107,
            other => {
                return Err(OpError::unsupported(format!(
                    "tcgen05.mma variant token {other:?} is not modeled"
                )))
            }
        }
    }
    Ok(options)
}

/// Legacy `collector_transition` (`instructions/tcgen05.rs`).
pub(super) fn collector_transition(
    state: u8,
    collector_a: CollectorOp,
    collector_b: CollectorOp,
    b_buffer: u8,
) -> OpResult<u8> {
    if b_buffer > 3 {
        return Err(OpError::invalid(format!(
            "tcgen05.mma collector B buffer b{b_buffer} is invalid"
        )));
    }
    let (mut fill, mut require, mut discard) = (0_u8, 0_u8, 0_u8);
    for (op, bit) in [(collector_a, 1_u8), (collector_b, 2_u8 << b_buffer)] {
        match op {
            CollectorOp::None => {}
            CollectorOp::Fill => fill |= bit,
            CollectorOp::Use => require |= bit,
            CollectorOp::LastUse => {
                require |= bit;
                discard |= bit;
            }
            CollectorOp::Discard => discard |= bit,
        }
    }
    if fill & (require | discard) != 0 {
        return Err(OpError::invalid("invalid TCGEN collector transition"));
    }
    if state & require != require {
        return Err(OpError::invalid(format!(
            "tcgen05.mma collector use/lastuse requires a valid previous fill (missing slots {:#x})",
            require & !state
        )));
    }
    Ok((state | fill) & !discard)
}

pub(super) fn tc_mma_ctas(
    payload: &crate::sync::completion::TcgenMmaPayload,
    options: &TcMmaOptions,
    smem: TcSmemRead<'_>,
    tmem_read: TcTmemRead<'_>,
    tmem_write: TcTmemWrite<'_>,
    io: Option<&super::TcMmaIo<'_>>,
) -> OpResult {
    // The program's form (`TcMmaKind::Ti16`, `TcgenMmaArgs::lut_b`) selects the
    // form; the options carry what the args cannot (arch, LUT taddr, ...).
    let mut options = *options;
    options.ti16 |= payload.args.kind == TcMmaKind::Ti16;
    if payload.args.lut_b && options.lut_b.is_none() {
        return Err(OpError::unsupported(
            "tcgen05.mma .lut_b needs the lookup-table taddr in TcMmaOptions::lut_b",
        ));
    }
    if !payload.args.lut_b {
        options.lut_b = None;
    }
    mma::run(payload, &options, smem, tmem_read, tmem_write, io)
}

pub(super) fn tc_mma(
    payload: &crate::sync::completion::TcgenMmaPayload,
    smem: &dyn Fn(u32, &mut [u8]) -> OpResult,
    tmem_read: &dyn Fn(u32, u32, &mut [u8]) -> OpResult,
    tmem_write: &mut dyn FnMut(u32, u32, &[u8]) -> OpResult,
) -> OpResult {
    if payload.args.cta_group == 2 {
        return Err(OpError::unsupported(
            "tc_mma reaches one CTA; cta_group::2 needs tc_mma_ctas",
        ));
    }
    let only_cta0 = |cta: u32| {
        if cta == 0 {
            Ok(())
        } else {
            Err(OpError::invalid(
                "tc_mma single-CTA accessor asked for CTA 1",
            ))
        }
    };
    let smem2 = |cta: u32, address: u32, buf: &mut [u8]| {
        only_cta0(cta)?;
        smem(address, buf)
    };
    let read2 = |cta: u32, lane: u32, col: u32, buf: &mut [u8]| {
        only_cta0(cta)?;
        tmem_read(lane, col, buf)
    };
    let mut write2 = |cta: u32, lane: u32, col: u32, bytes: &[u8]| {
        only_cta0(cta)?;
        tmem_write(lane, col, bytes)
    };
    mma::run(
        payload,
        &TcMmaOptions::default(),
        &smem2,
        &read2,
        &mut write2,
        None,
    )
}

#[cfg(test)]
mod tests;
