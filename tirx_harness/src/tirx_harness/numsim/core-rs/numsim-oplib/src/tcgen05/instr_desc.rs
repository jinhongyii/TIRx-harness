//! tcgen05.mma instruction-descriptor decoders, one struct per kind.
//!
//! Legacy source: `engine-rs/src/runtime/tcgen_ops.rs`
//! (`RawTcgenDenseInstructionDescriptor`, `RawTcgenF8f6f4InstructionDescriptor`,
//! `decode_raw_tcgen_{f8f6f4,tf32,b16,mxf4,sparse_mxf4,mxf8f6f4}_instruction_descriptor`,
//! `RawTcgenMxf4ScaleSpelling`, `raw_tcgen05_sfa_lanes`, `RawTcgenFloatKind`,
//! `raw_tcgen05_*_shape`, `raw_tcgen05_validate_lut_b`,
//! `raw_tcgen05_validate_tmem_a_transpose`, `raw_tcgen05_input_scale`).
//! The runtime-descriptor packing itself lives in
//! `crate::codec::tcgen_runtime_instruction_descriptor`.
//! Error texts are verbatim.

use super::layouts::{lut_b_location, SparseMetadataLayout};
use super::narrow::{CellDtype, NarrowFormat};
use super::scale::{decode_ue4m3_scale, decode_ue5m3_scale, decode_ue8m0_scale, ScaleDecoder};
use super::smem_desc::MatrixDescriptorLayout;
use crate::types::{OpError, OpResult};

/// Dense kind::f16 / kind::tf32 (and sparse narrow re-expressed) descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseInstr {
    pub cell_dtype: CellDtype,
    pub m: usize,
    pub n: usize,
    pub negate_a: bool,
    pub negate_b: bool,
    pub transpose_a: bool,
    pub transpose_b: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F8f6f4Instr {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub negate_a: bool,
    pub negate_b: bool,
    pub transpose_a: bool,
    pub transpose_b: bool,
}

/// Decode one dense or sparse `kind::f8f6f4` instruction descriptor.
///
/// The operand types are fixed by the caller, so the descriptor bits are
/// *checked* against them: bits 4-5 hold the D format (0 = f16, 1 = f32) and
/// bits 7-9 / 10-12 the A / B narrow formats.
#[allow(clippy::too_many_arguments)]
pub fn decode_f8f6f4(
    descriptor: u32,
    a_format: NarrowFormat,
    b_format: NarrowFormat,
    d_f16: bool,
    cta_group: u32,
    supports_k64: bool,
    weight_stationary: bool,
    sparse: bool,
) -> OpResult<F8f6f4Instr> {
    if !matches!(cta_group, 1 | 2) {
        return Err(OpError::message(format!(
            "raw f8f6f4 tcgen05.mma cta_group must be 1 or 2, got {cta_group}"
        )));
    }
    let d_format = u32::from(!d_f16);
    // Bit 29 selects SM107 K=64; WS independently owns bits 30..31.
    let reserved_high_mask = (if weight_stationary { 0 } else { 0x3_u32 << 30 })
        | (if supports_k64 { 0 } else { 1_u32 << 29 });
    let k = if supports_k64 && descriptor & (1_u32 << 29) != 0 {
        64
    } else {
        32
    };
    // Table 48 restricts dense K64 to FP8, not sparse K128.
    if k == 64
        && !sparse
        && (a_format.format().width_bits != 8 || b_format.format().width_bits != 8)
    {
        return Err(OpError::message(
            "dense f8f6f4 K=64 requires E4M3/E5M2 operands",
        ));
    }
    if descriptor & 0xf != if sparse { 4 } else { 0 }
        || descriptor & (1_u32 << 6) != 0
        || descriptor & (1_u32 << 23) != 0
        || descriptor & reserved_high_mask != 0
        || ((descriptor >> 4) & 0x3) != d_format
        || NarrowFormat::decode((descriptor >> 7) & 7, "A")? != a_format
        || NarrowFormat::decode((descriptor >> 10) & 7, "B")? != b_format
    {
        let d_name = if d_f16 { "F16" } else { "F32" };
        let density = if sparse {
            "sparse selector-zero"
        } else {
            "dense"
        };
        return Err(OpError::message(format!(
            "raw f8f6f4 tcgen05.mma descriptor must encode {density} {d_name}/{a_format:?}/{b_format:?}"
        )));
    }
    if (descriptor & (1 << 15) != 0 && a_format.format().width_bits != 8)
        || (descriptor & (1 << 16) != 0 && b_format.format().width_bits != 8)
    {
        return Err(OpError::message(
            "raw f8f6f4 MN-major operands must have 8-bit elements",
        ));
    }
    let m = usize::try_from((descriptor >> 24) & 0x1f)
        .map_err(|_| OpError::message("raw f8f6f4 M conversion failed"))?
        .checked_mul(16)
        .ok_or_else(|| OpError::message("raw f8f6f4 M overflow"))?;
    let n = usize::try_from((descriptor >> 17) & 0x3f)
        .map_err(|_| OpError::message("raw f8f6f4 N conversion failed"))?
        .checked_mul(8)
        .ok_or_else(|| OpError::message("raw f8f6f4 N overflow"))?;
    let b_mn_major = descriptor & (1_u32 << 16) != 0;
    // PTX ISA shape table (delta L4): CTA1 M=64 N%8 (N%16 for 8-bit
    // MN-major B, 9.7.17.10.1), CTA1 M=128 N%16, CTA2 N%32.
    let cta1_n_granularity = if b_mn_major || m == 128 { 16 } else { 8 };
    let cta2_n_granularity = 32;
    let valid_geometry = match (cta_group, weight_stationary) {
        (1, true) => matches!(m, 32 | 64 | 128) && (matches!(n, 64 | 128) || (!sparse && n == 256)),
        (1, false) => {
            matches!(m, 64 | 128)
                && (cta1_n_granularity..=256).contains(&n)
                && n % cta1_n_granularity == 0
        }
        (2, false) => {
            matches!(m, 128 | 256)
                && (cta2_n_granularity..=256).contains(&n)
                && n % cta2_n_granularity == 0
        }
        _ => false,
    };
    // Table 48's extended-K M restriction is for dense K=64, not sparse K=128.
    if k == 64 && !sparse && !weight_stationary && m != 128 * cta_group as usize {
        return Err(OpError::message("dense f8f6f4 K=64 requires M=128 per CTA"));
    }
    if !valid_geometry {
        let requirement = if weight_stationary {
            let n_shapes = if sparse {
                "{64, 128}"
            } else {
                "{64, 128, 256}"
            };
            format!(".ws cta_group=1, M in {{32, 64, 128}} and N in {n_shapes}")
        } else if cta_group == 1 {
            format!("M in {{64, 128}} and N in {cta1_n_granularity}..=256 by {cta1_n_granularity}")
        } else {
            let b_major = if b_mn_major { "MN-major" } else { "K-major" };
            format!(
                "M in {{128, 256}} and N in {cta2_n_granularity}..=256 by {cta2_n_granularity} for {b_major} B"
            )
        };
        return Err(OpError::message(format!(
            "raw f8f6f4 tcgen05.mma cta_group={cta_group} requires {requirement}, got M={m}, N={n}"
        )));
    }
    Ok(F8f6f4Instr {
        m,
        n,
        k,
        negate_a: descriptor & (1_u32 << 13) != 0,
        negate_b: descriptor & (1_u32 << 14) != 0,
        transpose_a: descriptor & (1_u32 << 15) != 0,
        transpose_b: descriptor & (1_u32 << 16) != 0,
    })
}

/// `(m, n, k, transpose_a, transpose_b)` of a CTA1 f8f6f4 MMA.
pub fn f8f6f4_cta1_shape(
    descriptor: u32,
    a_format: NarrowFormat,
    b_format: NarrowFormat,
    d_f16: bool,
    weight_stationary: bool,
    descriptor_layout: MatrixDescriptorLayout,
) -> OpResult<(usize, usize, usize, bool, bool)> {
    let i = decode_f8f6f4(
        descriptor,
        a_format,
        b_format,
        d_f16,
        1,
        descriptor_layout.supports_f8f6f4_k64(),
        weight_stationary,
        false,
    )?;
    Ok((i.m, i.n, i.k, i.transpose_a, i.transpose_b))
}

/// `(m, n, k, transpose_a, transpose_b)` of a CTA2 f8f6f4 MMA.
pub fn f8f6f4_cta2_shape(
    descriptor: u32,
    a_format: NarrowFormat,
    b_format: NarrowFormat,
    d_f16: bool,
    descriptor_layout: MatrixDescriptorLayout,
) -> OpResult<(usize, usize, usize, bool, bool)> {
    let i = decode_f8f6f4(
        descriptor,
        a_format,
        b_format,
        d_f16,
        2,
        descriptor_layout.supports_f8f6f4_k64(),
        false,
        false,
    )?;
    Ok((i.m, i.n, i.k, i.transpose_a, i.transpose_b))
}

/// PTX Table 48 gives F16/BF16 and TF32 the same M/N contract.
pub fn valid_b16_tf32_shape(
    cta_group: usize,
    m: usize,
    n: usize,
    weight_stationary: bool,
    sparse: bool,
) -> bool {
    match (cta_group, m, weight_stationary) {
        (1, 32, true) => matches!(n, 64 | 128) || (!sparse && n == 256),
        (1, 64, true) if sparse => matches!(n, 64 | 128),
        // PTX ISA shape table (delta L4): M=64 N%8, M=128 N%16, CTA2 N%32.
        (1, 64, _) => (8..=256).contains(&n) && n.is_multiple_of(8),
        (1, 128, false) => (16..=256).contains(&n) && n.is_multiple_of(16),
        (1, 128, true) => matches!(n, 64 | 128) || (!sparse && n == 256),
        (2, 128 | 256, false) => (32..=256).contains(&n) && n.is_multiple_of(32),
        _ => false,
    }
}

pub fn decode_tf32(
    descriptor: u32,
    cta_group: usize,
    weight_stationary: bool,
    sparse: bool,
) -> OpResult<DenseInstr> {
    if (descriptor & 4 != 0) != sparse
        || descriptor & (if sparse { 0xa } else { 0xf }) != 0
        || descriptor & (1_u32 << 6) != 0
        || descriptor & (1_u32 << 23) != 0
        || descriptor
            & (if weight_stationary {
                1_u32 << 29
            } else {
                0x7_u32 << 29
            })
            != 0
        || ((descriptor >> 4) & 0x3) != 1
        || ((descriptor >> 7) & 0x7) != 2
        || ((descriptor >> 10) & 0x7) != 2
    {
        return Err(OpError::message(
            "raw TF32 tcgen05.mma descriptor must encode F32/TF32/TF32 with matching sparsity and a valid selector",
        ));
    }
    let m = usize::try_from((descriptor >> 24) & 0x1f)
        .map_err(|_| OpError::message("raw TF32 M conversion failed"))?
        .checked_mul(16)
        .ok_or_else(|| OpError::message("raw TF32 M overflow"))?;
    let n = usize::try_from((descriptor >> 17) & 0x3f)
        .map_err(|_| OpError::message("raw TF32 N conversion failed"))?
        .checked_mul(8)
        .ok_or_else(|| OpError::message("raw TF32 N overflow"))?;
    if !valid_b16_tf32_shape(cta_group, m, n, weight_stationary, sparse) {
        return Err(OpError::message(format!(
            "raw TF32 tcgen05.mma invalid cta_group={cta_group}, WS={weight_stationary}, M={m}, N={n} geometry"
        )));
    }
    Ok(DenseInstr {
        cell_dtype: CellDtype::F32,
        m,
        n,
        negate_a: descriptor & (1_u32 << 13) != 0,
        negate_b: descriptor & (1_u32 << 14) != 0,
        transpose_a: descriptor & (1 << 15) != 0,
        transpose_b: descriptor & (1 << 16) != 0,
    })
}

pub fn decode_b16(
    descriptor: u32,
    a_bf16: bool,
    b_bf16: bool,
    cta_group: u32,
    weight_stationary: bool,
    sparse: bool,
) -> OpResult<DenseInstr> {
    let a_format = (descriptor >> 7) & 7;
    let b_format = (descriptor >> 10) & 7;
    if (descriptor & 4 != 0) != sparse
        || descriptor & (if sparse { 0xa } else { 0xf }) != 0
        || descriptor & (1_u32 << 6) != 0
        || descriptor & (1_u32 << 23) != 0
        || descriptor
            & (if weight_stationary {
                1_u32 << 29
            } else {
                0x7_u32 << 29
            })
            != 0
        || ((descriptor >> 4) & 0x3) > 1
        || a_format > 1
        || b_format > 1
    {
        return Err(OpError::message(
            "raw f16 tcgen05.mma descriptor must encode F16-or-F32/F16-or-BF16 with matching sparsity and a valid selector",
        ));
    }
    if descriptor & (1 << 4) == 0 && (a_format != 0 || b_format != 0) {
        return Err(OpError::message(
            "F16 accumulation requires F16 operands; BF16 requires F32",
        ));
    }
    let m = usize::try_from((descriptor >> 24) & 0x1f)
        .map_err(|_| OpError::message("raw f16 M conversion failed"))?
        .checked_mul(16)
        .ok_or_else(|| OpError::message("raw f16 M overflow"))?;
    let n = usize::try_from((descriptor >> 17) & 0x3f)
        .map_err(|_| OpError::message("raw f16 N conversion failed"))?
        .checked_mul(8)
        .ok_or_else(|| OpError::message("raw f16 N overflow"))?;
    if !valid_b16_tf32_shape(cta_group as usize, m, n, weight_stationary, sparse) {
        return Err(OpError::message(format!(
            "raw f16 tcgen05.mma has invalid cta_group={cta_group}, M={m}, N={n} geometry"
        )));
    }
    if a_format != b_format {
        return Err(OpError::message(
            "tcgen05.mma.kind::f16 requires matching F16/BF16 operand types; mixed F16/BF16 is invalid",
        ));
    }
    if a_format != u32::from(a_bf16) || b_format != u32::from(b_bf16) {
        return Err(OpError::message(
            "raw f16 descriptor operand formats do not match its codec specialization",
        ));
    }
    Ok(DenseInstr {
        cell_dtype: CellDtype::from_half(descriptor & (1 << 4) == 0),
        m,
        n,
        negate_a: descriptor & (1_u32 << 13) != 0,
        negate_b: descriptor & (1_u32 << 14) != 0,
        transpose_a: descriptor & (1_u32 << 15) != 0,
        transpose_b: descriptor & (1_u32 << 16) != 0,
    })
}

// ---------------------------------------------------------------------------
// kind::mxf4 / mxf4nvf4
// ---------------------------------------------------------------------------

/// Which `.kind::mxf4`-family block-scale spelling a descriptor carries: fixes
/// the scale format (descriptor bits 23-24), the legal SFA/SFB IDs and the
/// vector count (legacy `RawTcgenMxf4ScaleSpelling`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mxf4ScaleSpelling {
    Ue8m0Vec2x,
    Ue4m3Vec2x,
    Ue5m3Vec2x,
    Ue8m0Vec4x,
    Ue4m3Vec4x,
    Ue5m3Vec4x,
}

impl Mxf4ScaleSpelling {
    /// Diagnostic prefix; also the name the PTX `.kind` qualifier spells.
    pub fn label(self) -> &'static str {
        match self {
            Self::Ue8m0Vec2x => "raw mxf4",
            _ => "raw mxf4nvf4",
        }
    }

    /// Required value of instruction-descriptor bits 23-24 (scale matrix type).
    pub fn scale_format_bit(self) -> u32 {
        match self {
            Self::Ue8m0Vec2x | Self::Ue8m0Vec4x => 1,
            Self::Ue4m3Vec2x | Self::Ue4m3Vec4x => 0,
            Self::Ue5m3Vec2x | Self::Ue5m3Vec4x => 2,
        }
    }

    pub fn scale_format_name(self) -> &'static str {
        match self {
            Self::Ue8m0Vec2x | Self::Ue8m0Vec4x => "UE8M0",
            Self::Ue4m3Vec2x | Self::Ue4m3Vec4x => "UE4M3",
            Self::Ue5m3Vec2x | Self::Ue5m3Vec4x => "UE5M3",
        }
    }

    pub fn vector_count(self) -> usize {
        match self {
            Self::Ue8m0Vec2x | Self::Ue4m3Vec2x | Self::Ue5m3Vec2x => 2,
            Self::Ue8m0Vec4x | Self::Ue4m3Vec4x | Self::Ue5m3Vec4x => 4,
        }
    }

    pub fn scale_id_is_legal(self, scale_id: usize) -> bool {
        match self.vector_count() {
            2 => matches!(scale_id, 0 | 2),
            _ => scale_id == 0,
        }
    }

    pub fn legal_scale_ids(self) -> &'static str {
        match self.vector_count() {
            2 => "scale_vec::2X requires SFA/SFB IDs 0 or 2",
            _ => "scale_vec::4X requires SFA/SFB IDs 0",
        }
    }

    /// How one stored scale byte decodes.
    pub fn decoder(self) -> ScaleDecoder {
        match self {
            Self::Ue8m0Vec2x | Self::Ue8m0Vec4x => decode_ue8m0_scale,
            Self::Ue4m3Vec2x | Self::Ue4m3Vec4x => decode_ue4m3_scale,
            Self::Ue5m3Vec2x | Self::Ue5m3Vec4x => decode_ue5m3_scale,
        }
    }
}

pub fn mxf4nvf4_vec4x_scale(descriptor: u32) -> Mxf4ScaleSpelling {
    match (descriptor >> 23) & 3 {
        0 => Mxf4ScaleSpelling::Ue4m3Vec4x,
        2 => Mxf4ScaleSpelling::Ue5m3Vec4x,
        _ => Mxf4ScaleSpelling::Ue8m0Vec4x,
    }
}

pub fn mxf4nvf4_vec2x_scale(descriptor: u32) -> Mxf4ScaleSpelling {
    match (descriptor >> 23) & 3 {
        0 => Mxf4ScaleSpelling::Ue4m3Vec2x,
        2 => Mxf4ScaleSpelling::Ue5m3Vec2x,
        _ => Mxf4ScaleSpelling::Ue8m0Vec2x,
    }
}

/// PTX Tables 52/53, bit 26 selects the SM107 SFA layout (Figures 238--256).
pub fn sfa_lanes(descriptor: u32, layout: MatrixDescriptorLayout) -> OpResult<usize> {
    if descriptor & (1 << 26) == 0 {
        return Ok(32);
    }
    if layout != MatrixDescriptorLayout::Sm107 {
        return Err(OpError::message("128-lane SFA layout requires SM107"));
    }
    Ok(128)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mxf4Instr {
    pub sfa_lanes: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub sfa_id: usize,
    pub sfb_id: usize,
    pub negate_a: bool,
    pub negate_b: bool,
    pub block_elements: usize,
}

/// CTA1, non-fixed-vector decode (legacy test helper `decode_raw_tcgen_mxf4_instruction_descriptor`).
pub fn decode_mxf4(
    descriptor: u32,
    scale: Mxf4ScaleSpelling,
    descriptor_layout: MatrixDescriptorLayout,
) -> OpResult<Mxf4Instr> {
    decode_mxf4_for_cta_group(descriptor, scale, 1, descriptor_layout, false)
}

pub fn decode_mxf4_for_cta_group(
    descriptor: u32,
    scale: Mxf4ScaleSpelling,
    cta_group: usize,
    descriptor_layout: MatrixDescriptorLayout,
    fixed_vectors: bool,
) -> OpResult<Mxf4Instr> {
    let label = scale.label();
    // PTX 9.4 Table 53: dense MXF4 still requires the architecture's
    // sparsity-version encoding, even though no sparse metadata is read.
    let version = u32::from(descriptor_layout == MatrixDescriptorLayout::Sm107);
    if (descriptor >> 12) & 1 != version {
        return Err(OpError::message(format!(
            "{label} descriptor sparsity version must be v{version} for {descriptor_layout:?}"
        )));
    }
    let k = match ((descriptor >> 3) & 1, descriptor >> 31) {
        (0, 0) => 64,
        (0, 1) if descriptor_layout != MatrixDescriptorLayout::Sm100 => 96,
        (1, 0) if descriptor_layout == MatrixDescriptorLayout::Sm107 => 128,
        _ => {
            return Err(OpError::message(format!(
                "{label} descriptor has unsupported K encoding for {descriptor_layout:?}"
            )))
        }
    };
    let block_elements = if fixed_vectors {
        k / scale.vector_count()
    } else {
        64 / scale.vector_count()
    };
    if !matches!(cta_group, 1 | 2) {
        return Err(OpError::message(format!(
            "{label} tcgen05.mma cta_group must be 1 or 2, got {cta_group}"
        )));
    }
    if descriptor & 0x3 != 0
        || descriptor & (1_u32 << 2) != 0
        || descriptor & (1_u32 << 6) != 0
        || descriptor & (1_u32 << 25) != 0
    {
        return Err(OpError::message(format!(
            "{label} tcgen05.mma descriptor uses sparse or reserved bits ({descriptor:#010x})"
        )));
    }
    if ((descriptor >> 7) & 0x7) != 1 || ((descriptor >> 10) & 0x3) != 1 {
        return Err(OpError::message(format!(
            "{label} tcgen05.mma descriptor must encode E2M1 A and B"
        )));
    }
    if ((descriptor >> 23) & 0x3) != scale.scale_format_bit() {
        return Err(OpError::message(format!(
            "{label} tcgen05.mma descriptor must encode {} scales",
            scale.scale_format_name()
        )));
    }
    if descriptor_layout != MatrixDescriptorLayout::Sm107
        && matches!(
            scale,
            Mxf4ScaleSpelling::Ue4m3Vec2x
                | Mxf4ScaleSpelling::Ue5m3Vec2x
                | Mxf4ScaleSpelling::Ue5m3Vec4x
        )
    {
        return Err(OpError::message(
            "UE5M3 scales and block32 UE4M3 require SM107",
        ));
    }
    if descriptor & ((1_u32 << 15) | (1_u32 << 16)) != 0 {
        return Err(OpError::message(format!(
            "{label} tcgen05.mma transpose descriptors are not implemented"
        )));
    }
    let m = usize::try_from((descriptor >> 27) & 0x3)
        .map_err(|_| OpError::message(format!("{label} M conversion failed")))?
        .checked_mul(128)
        .ok_or_else(|| OpError::message(format!("{label} M overflow")))?;
    let n = usize::try_from((descriptor >> 17) & 0x3f)
        .map_err(|_| OpError::message(format!("{label} N conversion failed")))?
        .checked_mul(8)
        .ok_or_else(|| OpError::message(format!("{label} N overflow")))?;
    let expected_m = 128 * cta_group;
    let n_granularity = 8 * cta_group;
    if m != expected_m || !(n_granularity..=256).contains(&n) || n % n_granularity != 0 {
        return Err(OpError::message(format!(
            "{label} tcgen05.mma cta_group={cta_group} requires M={expected_m} and N in {n_granularity}..=256 by {n_granularity}, got M={m}, N={n}"
        )));
    }
    let sfa_id = usize::try_from((descriptor >> 29) & 0x3)
        .map_err(|_| OpError::message(format!("{label} SFA ID conversion failed")))?;
    let sfb_id = usize::try_from((descriptor >> 4) & 0x3)
        .map_err(|_| OpError::message(format!("{label} SFB ID conversion failed")))?;
    let valid_id = |id| {
        if fixed_vectors {
            scale.scale_id_is_legal(id)
        } else {
            match (k, scale.vector_count()) {
                (96, 2) => id < 4,
                (96, _) => matches!(id, 0 | 2),
                (128, _) => id == 0,
                _ => scale.scale_id_is_legal(id),
            }
        }
    };
    if !valid_id(sfa_id) || !valid_id(sfb_id) {
        if k != 64 {
            return Err(OpError::message(format!(
                "{label} descriptor has unsupported scale IDs {sfa_id}/{sfb_id} for K={k}"
            )));
        }
        return Err(OpError::message(format!(
            "{label} {}, got {sfa_id}/{sfb_id}",
            scale.legal_scale_ids()
        )));
    }
    Ok(Mxf4Instr {
        sfa_lanes: sfa_lanes(descriptor, descriptor_layout)?,
        m,
        n,
        k,
        sfa_id,
        sfb_id,
        negate_a: descriptor & (1_u32 << 13) != 0,
        negate_b: descriptor & (1_u32 << 14) != 0,
        block_elements,
    })
}

/// `(m, n, k)` of a dense block-scaled FP4 MMA.
pub fn block_mxf4_shape(
    descriptor: u32,
    scale: Mxf4ScaleSpelling,
    cta_group: usize,
    descriptor_layout: MatrixDescriptorLayout,
    fixed_vectors: bool,
) -> OpResult<(usize, usize, usize)> {
    let i = decode_mxf4_for_cta_group(
        descriptor,
        scale,
        cta_group,
        descriptor_layout,
        fixed_vectors,
    )?;
    Ok((i.m, i.n, i.k))
}

pub fn decode_sparse_mxf4(descriptor: u32) -> OpResult<Mxf4Instr> {
    if descriptor & 0x3 != 0
        || descriptor & (1_u32 << 2) == 0
        || descriptor & (1_u32 << 3) != 0
        || descriptor & (1_u32 << 6) != 0
        || descriptor & (1_u32 << 12) != 0
        || descriptor & (0x7_u32 << 24) != 0
        || descriptor & (1_u32 << 31) != 0
    {
        return Err(OpError::message(
            "raw sparse mxf4 descriptor must encode sparse K=128 without reserved bits",
        ));
    }
    if ((descriptor >> 7) & 0x7) != 1 || ((descriptor >> 10) & 0x3) != 1 {
        return Err(OpError::message(
            "raw sparse mxf4 descriptor must encode E2M1 A and B",
        ));
    }
    if ((descriptor >> 23) & 0x1) != 1 {
        return Err(OpError::message(
            "raw sparse mxf4 descriptor must encode UE8M0 scales",
        ));
    }
    if descriptor & ((1_u32 << 15) | (1_u32 << 16)) != 0 {
        return Err(OpError::message(
            "raw sparse mxf4 transpose descriptors are not implemented",
        ));
    }
    let m = usize::try_from((descriptor >> 27) & 0x3)
        .map_err(|_| OpError::message("raw sparse mxf4 M conversion failed"))?
        .checked_mul(128)
        .ok_or_else(|| OpError::message("raw sparse mxf4 M overflow"))?;
    let n = usize::try_from((descriptor >> 17) & 0x3f)
        .map_err(|_| OpError::message("raw sparse mxf4 N conversion failed"))?
        .checked_mul(8)
        .ok_or_else(|| OpError::message("raw sparse mxf4 N overflow"))?;
    if m != 128 || !(8..=256).contains(&n) || n % 8 != 0 {
        return Err(OpError::message(format!(
            "raw sparse mxf4 cta_group=1 requires M=128 and N in 8..=256 by 8, got M={m}, N={n}"
        )));
    }
    let sfa_id = usize::try_from((descriptor >> 29) & 0x3)
        .map_err(|_| OpError::message("raw sparse mxf4 SFA ID conversion failed"))?;
    let sfb_id = usize::try_from((descriptor >> 4) & 0x3)
        .map_err(|_| OpError::message("raw sparse mxf4 SFB ID conversion failed"))?;
    if !matches!(sfa_id, 0 | 2) || !matches!(sfb_id, 0 | 2) {
        return Err(OpError::message(format!(
            "raw sparse mxf4 scale_vec::2X requires SFA/SFB IDs 0 or 2, got {sfa_id}/{sfb_id}"
        )));
    }
    Ok(Mxf4Instr {
        sfa_lanes: 32,
        m,
        n,
        k: 128,
        sfa_id,
        sfb_id,
        negate_a: descriptor & (1_u32 << 13) != 0,
        negate_b: descriptor & (1_u32 << 14) != 0,
        block_elements: 64,
    })
}

// ---------------------------------------------------------------------------
// kind::mxf8f6f4
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mxf8f6f4Instr {
    pub sparse: bool,
    pub sfa_lanes: usize,
    pub k: usize,
    pub m: usize,
    pub n: usize,
    pub a_format: NarrowFormat,
    pub b_format: NarrowFormat,
    pub sfa_id: usize,
    pub sfb_id: usize,
    pub negate_a: bool,
    pub negate_b: bool,
    pub transpose_a: bool,
    pub transpose_b: bool,
}

impl Mxf8f6f4Instr {
    pub fn packed_k(self) -> usize {
        self.k / if self.sparse { 2 } else { 1 }
    }
    pub fn padded_atoms(self) -> bool {
        self.packed_k() == 32
    }
}

pub fn decode_mxf8f6f4(
    descriptor: u32,
    cta_group: usize,
    layout: MatrixDescriptorLayout,
) -> OpResult<Mxf8f6f4Instr> {
    if !matches!(cta_group, 1 | 2) {
        return Err(OpError::message(format!(
            "raw mxf8f6f4 cta_group must be 1 or 2, got {cta_group}"
        )));
    }
    if descriptor & 0xb != 0
        || descriptor & (1_u32 << 6) != 0
        || (descriptor & (1_u32 << 31) != 0 && layout != MatrixDescriptorLayout::Sm107)
        || ((descriptor >> 23) & 0x1) != 1
    {
        return Err(OpError::message(
            "raw mxf8f6f4 descriptor requires UE8M0 scales and valid K/reserved bits",
        ));
    }
    let a_format = NarrowFormat::decode((descriptor >> 7) & 0x7, "A")?;
    let b_format = NarrowFormat::decode((descriptor >> 10) & 0x7, "B")?;
    let transpose_a = descriptor & (1_u32 << 15) != 0;
    let transpose_b = descriptor & (1_u32 << 16) != 0;
    if transpose_a && a_format.format().width_bits != 8 {
        return Err(OpError::message(
            "raw mxf8f6f4 transpose A requires an 8-bit operand",
        ));
    }
    if transpose_b && b_format.format().width_bits != 8 {
        return Err(OpError::message(
            "raw mxf8f6f4 transpose B requires an 8-bit operand",
        ));
    }
    let m = usize::try_from(((descriptor & !(1 << 26)) >> 24) & 0x1f)
        .map_err(|_| OpError::message("raw mxf8f6f4 M conversion failed"))?
        .checked_mul(16)
        .ok_or_else(|| OpError::message("raw mxf8f6f4 M overflow"))?;
    let n = usize::try_from((descriptor >> 17) & 0x3f)
        .map_err(|_| OpError::message("raw mxf8f6f4 N conversion failed"))?
        .checked_mul(8)
        .ok_or_else(|| OpError::message("raw mxf8f6f4 N overflow"))?;
    let expected_m = 128 * cta_group;
    let k = if descriptor & (1 << 31) != 0 { 64 } else { 32 };
    let n_granularity = if transpose_b {
        16 * cta_group
    } else {
        8 * cta_group
    };
    // PTX MMA shape table permits CTA2 M=128 only for dense K=32.
    let valid_m = m == expected_m || (cta_group == 2 && m == 128 && k == 32 && descriptor & 4 == 0);
    if !valid_m || !(n_granularity..=256).contains(&n) || n % n_granularity != 0 {
        return Err(OpError::message(format!(
            "raw mxf8f6f4 cta_group={cta_group} requires M={expected_m} (also M=128 for dense CTA2 K=32) and N in {n_granularity}..=256 by {n_granularity}, got M={m}, N={n}"
        )));
    }
    let sparse = descriptor & 4 != 0;
    Ok(Mxf8f6f4Instr {
        sparse,
        sfa_lanes: sfa_lanes(descriptor, layout)?,
        k: k * if sparse { 2 } else { 1 },
        m,
        n,
        a_format,
        b_format,
        sfa_id: usize::try_from((descriptor >> 29) & 0x3)
            .map_err(|_| OpError::message("raw mxf8f6f4 SFA ID conversion failed"))?,
        sfb_id: usize::try_from((descriptor >> 4) & 0x3)
            .map_err(|_| OpError::message("raw mxf8f6f4 SFB ID conversion failed"))?,
        negate_a: descriptor & (1_u32 << 13) != 0,
        negate_b: descriptor & (1_u32 << 14) != 0,
        transpose_a,
        transpose_b,
    })
}

pub fn block_mxf8f6f4_shape(
    descriptor: u32,
    cta_group: usize,
    descriptor_layout: MatrixDescriptorLayout,
) -> OpResult<(usize, usize, usize)> {
    let i = decode_mxf8f6f4(descriptor, cta_group, descriptor_layout)?;
    Ok((i.m, i.n, i.k))
}

// ---------------------------------------------------------------------------
// Dense/sparse floating kinds
// ---------------------------------------------------------------------------

/// Floating dense/sparse MMA kind (legacy `RawTcgenFloatKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatKind {
    Tf32,
    B16 {
        a_bf16: bool,
        b_bf16: bool,
    },
    SparseNarrow {
        a_format: NarrowFormat,
        b_format: NarrowFormat,
        descriptor_layout: MatrixDescriptorLayout,
    },
}

impl FloatKind {
    pub fn metadata_layout(self, descriptor: u32) -> SparseMetadataLayout {
        match self {
            Self::SparseNarrow { .. } => SparseMetadataLayout::Narrow {
                k: self.packed_k(descriptor) * 2,
            },
            _ => SparseMetadataLayout::B16 {
                selector: (descriptor & 1) as usize,
            },
        }
    }

    pub fn descriptor_layout(self) -> MatrixDescriptorLayout {
        match self {
            Self::SparseNarrow {
                descriptor_layout, ..
            } => descriptor_layout,
            _ => MatrixDescriptorLayout::Sm100,
        }
    }

    pub fn instruction(
        self,
        instruction_descriptor: u32,
        cta_group: usize,
        weight_stationary: bool,
        sparse: bool,
    ) -> OpResult<DenseInstr> {
        match self {
            Self::Tf32 => decode_tf32(instruction_descriptor, cta_group, weight_stationary, sparse),
            Self::B16 { a_bf16, b_bf16 } => decode_b16(
                instruction_descriptor,
                a_bf16,
                b_bf16,
                cta_group as u32,
                weight_stationary,
                sparse,
            ),
            Self::SparseNarrow {
                a_format,
                b_format,
                descriptor_layout,
            } => {
                if !sparse || instruction_descriptor & 15 != 4 {
                    return Err(OpError::message(
                        "sparse F8F6F4 requires sparsity and selector zero",
                    ));
                }
                let half = instruction_descriptor & (1 << 4) == 0;
                let value = decode_f8f6f4(
                    instruction_descriptor,
                    a_format,
                    b_format,
                    half,
                    cta_group as u32,
                    descriptor_layout.supports_f8f6f4_k64(),
                    weight_stationary,
                    true,
                )?;
                Ok(DenseInstr {
                    cell_dtype: CellDtype::from_half(half),
                    m: value.m,
                    n: value.n,
                    negate_a: value.negate_a,
                    negate_b: value.negate_b,
                    transpose_a: value.transpose_a,
                    transpose_b: value.transpose_b,
                })
            }
        }
    }

    pub fn packed_k(self, descriptor: u32) -> usize {
        match self {
            Self::Tf32 => 8,
            Self::B16 { .. } => 16,
            Self::SparseNarrow { .. } => {
                if descriptor & (1 << 29) == 0 {
                    32
                } else {
                    64
                }
            }
        }
    }

    pub fn tmem_a_columns(self, descriptor: u32) -> usize {
        match self {
            Self::SparseNarrow { .. } => self.packed_k(descriptor) / 4,
            _ => 8,
        }
    }

    /// Dense K index selected by metadata `code` for packed element `inner`.
    pub fn sparse_index(self, code: u8, inner: usize) -> OpResult<usize> {
        match self {
            Self::Tf32 => match code {
                0x4 => Ok(inner * 2),
                0xe => Ok(inner * 2 + 1),
                _ => Err(OpError::message(format!(
                    "sparse 1:2 TF32 metadata code 0x{code:x} is not a defined index"
                ))),
            },
            Self::B16 { .. } | Self::SparseNarrow { .. } => {
                Ok((inner / 2) * 4 + super::mma::sparse_2of4_indices(code)?[inner % 2])
            }
        }
    }
}

/// `(m, n, transpose_a, transpose_b)` of a floating dense/sparse MMA.
pub fn float_shape(
    kind: FloatKind,
    descriptor: u32,
    cta_group: usize,
    ws: bool,
    sparse: bool,
) -> OpResult<(usize, usize, bool, bool)> {
    let value = kind.instruction(descriptor, cta_group, ws, sparse)?;
    Ok((value.m, value.n, value.transpose_a, value.transpose_b))
}

// ---------------------------------------------------------------------------
// Small shared operand checks
// ---------------------------------------------------------------------------

/// TMEM A must be K-major (legacy `raw_tcgen05_validate_tmem_a_transpose`).
pub fn validate_tmem_a_transpose(transpose: bool) -> OpResult<()> {
    if transpose {
        return Err(OpError::message(
            "tcgen05.mma TMEM A must be K-major; transpose_a is invalid",
        ));
    }
    Ok(())
}

pub fn f8_tmem_a_address(bits: u64, transpose: bool) -> OpResult<u32> {
    validate_tmem_a_transpose(transpose)?;
    u32::try_from(bits).map_err(|_| OpError::message("raw f8f6f4 TMEM A address exceeds u32"))
}

pub fn validate_lut_b(
    lut_b: Option<u32>,
    k: usize,
    b_format: NarrowFormat,
    transpose_b: bool,
) -> OpResult<()> {
    if let Some(address) = lut_b {
        if k != 64 || b_format != NarrowFormat::E4M3 || transpose_b {
            return Err(OpError::message(
                "tcgen05.mma LUT-B requires K=64 and non-transposed E4M3 B",
            ));
        }
        lut_b_location(address, 0, 0)?;
    }
    Ok(())
}

/// `2^-scale_input_d` with the mnemonic's conversion diagnostic.
pub fn input_scale(scale_input_d: usize, conversion_message: &'static str) -> OpResult<f32> {
    Ok(2.0_f32
        .powi(-i32::try_from(scale_input_d).map_err(|_| OpError::message(conversion_message))?))
}

/// The dense floating MMA's `scale-input-d` (range check + `2^-d`).
pub fn tf32_family_input_scale(scale_input_d: usize) -> OpResult<f32> {
    if scale_input_d > 15 {
        return Err(OpError::message("raw TF32 scale-input-d is outside 0..=15"));
    }
    input_scale(scale_input_d, "raw TF32 input scale conversion failed")
}

#[cfg(test)]
#[path = "instr_desc_tests.rs"]
mod tests;
