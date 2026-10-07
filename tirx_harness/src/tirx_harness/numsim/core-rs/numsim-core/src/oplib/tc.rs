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
//! Modeled: `cta_group::1` dense `kind::{f16, tf32, f8f6f4, i8}` with A from
//! shared memory or TMEM, `enable_input_d`, `scale_input_d` (f16/tf32),
//! `disable_output_lane`; and `cta_group::1` dense
//! `kind::{mxf8f6f4, mxf4, mxf4nvf4}.block_scale` (SM100 field widths).
//! Everything else fails closed as `Unsupported` naming the form.

mod mma;

use super::{InstrDesc, OpError, OpResult, SmemDesc};
use crate::dtype::Dtype;
use crate::program::TcMmaKind;
use numsim_oplib::tcgen05::instr_desc::{
    decode_b16, decode_f8f6f4, decode_mxf4_for_cta_group, decode_mxf8f6f4, decode_tf32,
    mxf4nvf4_vec2x_scale, mxf4nvf4_vec4x_scale, Mxf4ScaleSpelling,
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

/// Try `cta_group::1`, then `::2` (the contract passes no CTA group); the
/// first decoder that accepts wins, else the `::1` error.
fn either_cta_group<T>(decode: impl Fn(usize) -> numsim_oplib::types::OpResult<T>) -> OpResult<T> {
    match decode(1) {
        Ok(value) => Ok(value),
        Err(first) => decode(2).map_err(|_| OpError::from(first)),
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

/// Decode and validate a tcgen05 instruction descriptor for `kind` (SM100;
/// accepted if valid for `cta_group::1` or `::2`).
pub(super) fn decode_instr_desc(idesc: u32, kind: TcMmaKind) -> OpResult<InstrDesc> {
    let ws_shift = ((idesc >> 30) & 3) as u8;
    let bit = |n: u32| idesc & (1 << n) != 0;
    let common = InstrDesc {
        a_major_mn: bit(15),
        b_major_mn: bit(16),
        negate_a: bit(13),
        negate_b: bit(14),
        sparse: bit(2),
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
            let bf16 = (idesc >> 7) & 7 == 1;
            let decoded =
                either_cta_group(|cta| decode_b16(idesc, bf16, bf16, cta as u32, false, bit(2)))?;
            let (m, n) = shape(decoded.m, decoded.n)?;
            let ab = if bf16 { Dtype::BF16 } else { Dtype::F16 };
            let d = if bit(4) { Dtype::F32 } else { Dtype::F16 };
            InstrDesc {
                m,
                n,
                a: Some(ab),
                b: Some(ab),
                d: Some(d),
                max_shift: ws_shift,
                ..common
            }
        }
        TcMmaKind::Tf32 => {
            let decoded = either_cta_group(|cta| decode_tf32(idesc, cta, false, bit(2)))?;
            let (m, n) = shape(decoded.m, decoded.n)?;
            InstrDesc {
                m,
                n,
                a: Some(Dtype::TF32),
                b: Some(Dtype::TF32),
                d: Some(Dtype::F32),
                max_shift: ws_shift,
                ..common
            }
        }
        TcMmaKind::F8f6f4 => {
            let a = NarrowFormat::decode((idesc >> 7) & 7, "A")?;
            let b = NarrowFormat::decode((idesc >> 10) & 7, "B")?;
            let d_f16 = (idesc >> 4) & 3 == 0;
            let decoded = either_cta_group(|cta| {
                decode_f8f6f4(idesc, a, b, d_f16, cta as u32, false, false, bit(2))
            })?;
            let (m, n) = shape(decoded.m, decoded.n)?;
            let d = if d_f16 { Dtype::F16 } else { Dtype::F32 };
            InstrDesc {
                m,
                n,
                a: Some(narrow_dtype(a)),
                b: Some(narrow_dtype(b)),
                d: Some(d),
                max_shift: ws_shift,
                ..common
            }
        }
        TcMmaKind::I8 => {
            let (m, n, _, _) =
                either_cta_group(|cta| integer_shape(IntegerKind::I8, idesc, cta, false, false))?;
            let (m, n) = shape(m, n)?;
            let int = |format: u32| if format == 1 { Dtype::S8 } else { Dtype::U8 };
            InstrDesc {
                m,
                n,
                a: Some(int((idesc >> 7) & 7)),
                b: Some(int((idesc >> 10) & 7)),
                d: Some(Dtype::S32),
                // Bits 13/14 are reserved for kind::i8 (checked above).
                negate_a: false,
                negate_b: false,
                max_shift: ws_shift,
                ..common
            }
        }
        TcMmaKind::MxF8f6f4 => {
            let decoded =
                either_cta_group(|cta| decode_mxf8f6f4(idesc, cta, MatrixDescriptorLayout::Sm100))?;
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
            let mut last = None;
            for spelling in mxf4_spellings(kind, idesc, None)? {
                match either_cta_group(|cta| {
                    decode_mxf4_for_cta_group(
                        idesc,
                        spelling,
                        cta,
                        MatrixDescriptorLayout::Sm100,
                        false,
                    )
                }) {
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
                    Err(error) => last = Some(error),
                }
            }
            return Err(
                last.unwrap_or_else(|| OpError::invalid("mxf4 descriptor has no scale spelling"))
            );
        }
    })
}

pub(super) fn tc_mma(
    payload: &crate::sync::completion::TcgenMmaPayload,
    smem: &dyn Fn(u32, &mut [u8]) -> OpResult,
    tmem_read: &dyn Fn(u32, u32, &mut [u8]) -> OpResult,
    tmem_write: &mut dyn FnMut(u32, u32, &[u8]) -> OpResult,
) -> OpResult {
    mma::run(payload, smem, tmem_read, tmem_write)
}

#[cfg(test)]
mod tests;
