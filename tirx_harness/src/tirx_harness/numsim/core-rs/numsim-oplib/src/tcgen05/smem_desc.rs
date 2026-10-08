//! Shared-memory matrix descriptors and their byte-offset math.
//!
//! Legacy source: `engine-rs/src/runtime/tcgen_ops.rs` (`RawTcgenMatrixDescriptor`,
//! `decode_raw_tcgen_matrix_descriptor*`, `raw_tcgen05_shared_byte_offset`,
//! `raw_tcgen05_finish_shared_byte_offset`, `raw_tcgen05_*matrix_byte_offset`,
//! `raw_tcgen05_cp_source_span`, `raw_tcgen05_narrow_shared_atom_accesses`,
//! `raw_tcgen05_lut_b_row_accesses`, `raw_tcgen05_f8_b_descriptor`,
//! `RawTcgenColumnMask`, tile-GEMM operand layouts).
//!
//! The legacy code resolved the descriptor address against a `RuntimeBuffer`
//! (shared view). Here the caller passes the resolved window geometry as a
//! [`SharedWindow`]; every offset returned is relative to that window, exactly
//! as the legacy offsets were relative to the selected source view.

use super::narrow::NarrowFormat;
use crate::types::{OpError, OpResult};

/// Which architecture's matrix-descriptor field widths apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixDescriptorLayout {
    Sm100,
    Sm103,
    Sm107,
}

impl MatrixDescriptorLayout {
    /// Whether this architecture encodes the f8f6f4 K=64 form (SM107 only).
    pub fn supports_f8f6f4_k64(self) -> bool {
        matches!(self, Self::Sm107)
    }
}

/// Decoded shared-memory matrix descriptor (legacy `RawTcgenMatrixDescriptor`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixDescriptor {
    pub start_address: usize,
    pub leading_byte_offset: usize,
    pub absolute_leading_address: bool,
    pub stride_byte_offset: usize,
    pub swizzle_bits: usize,
    pub swizzle_atom_bytes: usize,
    pub swizzle_xor_shift: usize,
}

/// Geometry of the shared-memory source the descriptor address resolved to.
///
/// Mirrors the fields legacy read from `runtime_buffer_base(source)`:
/// `virtual_base`, `byte_offset` (view offset), `byte_len` (view length),
/// `backing_byte_len`, and whether the source was an `AccessView` (which adds
/// the view-bounds check).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedWindow {
    pub virtual_base: usize,
    pub view_offset: usize,
    pub view_len: usize,
    pub backing_byte_len: usize,
    pub access_view: bool,
}

impl SharedWindow {
    /// A plain (non-access-view) shared buffer covering its whole backing.
    pub const fn whole(virtual_base: usize, byte_len: usize) -> Self {
        Self {
            virtual_base,
            view_offset: 0,
            view_len: byte_len,
            backing_byte_len: byte_len,
            access_view: false,
        }
    }

    /// The canonical source legacy `raw_tcgen05_shared_source` builds: an
    /// access view over the whole backing at `virtual_base`.
    pub const fn resolved(virtual_base: usize, byte_len: usize) -> Self {
        Self {
            virtual_base,
            view_offset: 0,
            view_len: byte_len,
            backing_byte_len: byte_len,
            access_view: true,
        }
    }
}

/// Decode with SM100 field widths (legacy `decode_raw_tcgen_matrix_descriptor`).
pub fn decode_matrix_descriptor(descriptor: u64) -> OpResult<MatrixDescriptor> {
    decode_matrix_descriptor_for_layout(descriptor, MatrixDescriptorLayout::Sm100)
}

/// Decode a 64-bit shared matrix descriptor with this architecture's field widths
/// (14 bits SM100/103, 15 bits SM107): start/LBO/SBO in bytes (field << 4) and the
/// swizzle mode (bits 61..63). No numerics; errors on a wrong version, reserved
/// bits, a non-zero base offset, an invalid layout type or a misaligned 32B-atom start.
pub fn decode_matrix_descriptor_for_layout(
    descriptor: u64,
    layout: MatrixDescriptorLayout,
) -> OpResult<MatrixDescriptor> {
    let field_bits = match layout {
        MatrixDescriptorLayout::Sm100 | MatrixDescriptorLayout::Sm103 => 14,
        MatrixDescriptorLayout::Sm107 => 15,
    };
    if ((descriptor >> 46) & 0x3) != 1 {
        return Err(OpError::message(
            "raw tcgen05.cp descriptor has an invalid version field",
        ));
    }
    let field_mask = (1_u64 << field_bits) - 1;
    let start_reserved_mask = 0xffff_u64 & !field_mask;
    let ldo_reserved_mask = start_reserved_mask << 16;
    if descriptor & start_reserved_mask != 0
        || descriptor & ldo_reserved_mask != 0
        || descriptor & (((1_u64 << 13) - 1) << 48) != 0
    {
        return Err(OpError::message(
            "raw tcgen05.cp descriptor uses unsupported reserved/base/LBO-mode bits",
        ));
    }
    let layout_type = usize::try_from((descriptor >> 61) & 0x7)
        .map_err(|_| OpError::message("matrix descriptor layout conversion failed"))?;
    let (swizzle_bits, swizzle_atom_bytes, swizzle_xor_shift) = match layout_type {
        0 => (0, 16, 3),
        6 => (1, 16, 3),
        4 => (2, 16, 3),
        2 => (3, 16, 3),
        1 => (2, 32, 2),
        _ => {
            return Err(OpError::message(format!(
                "raw tcgen05.cp descriptor has invalid layout type {layout_type}"
            )));
        }
    };
    let start_address = usize::try_from(descriptor & field_mask)
        .map_err(|_| OpError::message("matrix descriptor start conversion failed"))?
        << 4;
    if swizzle_atom_bytes == 32 && start_address % 32 != 0 {
        return Err(OpError::message(
            "raw tcgen05.cp 128B/32B-atomic descriptor is not 32-byte aligned",
        ));
    }
    Ok(MatrixDescriptor {
        start_address,
        absolute_leading_address: false,
        leading_byte_offset: usize::try_from((descriptor >> 16) & field_mask)
            .map_err(|_| OpError::message("matrix descriptor LDO conversion failed"))?
            << 4,
        stride_byte_offset: usize::try_from((descriptor >> 32) & 0x3fff)
            .map_err(|_| OpError::message("matrix descriptor SDO conversion failed"))?
            << 4,
        swizzle_bits,
        swizzle_atom_bytes,
        swizzle_xor_shift,
    })
}

/// PTX 9.7.18.3.1.2: 48B K-major packed rows may straddle two 128B
/// swizzle chunks. Other consumers still reject bit 52 in the ordinary decoder.
pub fn decode_packed_matrix_descriptor(
    bits: u64,
    layout: MatrixDescriptorLayout,
    row_bytes: usize,
    transpose: bool,
) -> OpResult<MatrixDescriptor> {
    let absolute = bits & (1_u64 << 52) != 0;
    let mut descriptor = decode_matrix_descriptor_for_layout(bits & !(1_u64 << 52), layout)?;
    if absolute
        && (layout == MatrixDescriptorLayout::Sm100
            || row_bytes != 48
            || transpose
            || descriptor.swizzle_bits != 3
            || descriptor.swizzle_atom_bytes != 16
            || descriptor.leading_byte_offset % 128 != 0)
    {
        return Err(OpError::message("absolute LDO requires an SM103/SM107 48B K-major row, 128B/16B swizzle and aligned second chunk"));
    }
    descriptor.absolute_leading_address = absolute;
    Ok(descriptor)
}

/// B-operand descriptor for f8f6f4, honouring LUT-B (legacy `raw_tcgen05_f8_b_descriptor`).
pub fn f8_b_descriptor(
    bits: u64,
    layout: MatrixDescriptorLayout,
    lut_b: Option<u32>,
) -> OpResult<MatrixDescriptor> {
    if lut_b.is_some() {
        if layout != MatrixDescriptorLayout::Sm107 {
            return Err(OpError::message("tcgen05.mma LUT-B requires SM107"));
        }
        decode_packed_matrix_descriptor(bits & !(1_u64 << 53), layout, 48, false)
    } else {
        decode_matrix_descriptor_for_layout(bits, layout)
    }
}

/// Byte offset of `byte_in_row` of K-major `row` (legacy `raw_tcgen05_shared_byte_offset`).
pub fn shared_byte_offset(
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    row: usize,
    byte_in_row: usize,
    access_bytes: usize,
) -> OpResult<usize> {
    let row_stride =
        descriptor
            .swizzle_atom_bytes
            .checked_shl(u32::try_from(descriptor.swizzle_bits).map_err(|_| {
                OpError::message("matrix descriptor swizzle length conversion failed")
            })?)
            .ok_or_else(|| OpError::message("matrix descriptor row stride overflow"))?;
    let atom = byte_in_row / descriptor.swizzle_atom_bytes;
    let byte_in_atom = byte_in_row % descriptor.swizzle_atom_bytes;
    if byte_in_atom
        .checked_add(access_bytes)
        .ok_or_else(|| OpError::message("raw TCGEN access size overflow"))?
        > descriptor.swizzle_atom_bytes
    {
        return Err(OpError::message(
            "raw TCGEN scalar access crosses a descriptor swizzle atom",
        ));
    }
    let column_stride = if descriptor.swizzle_bits == 0 {
        if atom != 0 && descriptor.leading_byte_offset == 0 {
            return Err(OpError::message(
                "non-swizzled raw TCGEN descriptor needs nonzero LDO for multiple 128-bit columns",
            ));
        }
        descriptor
            .leading_byte_offset
            .max(descriptor.swizzle_atom_bytes)
    } else {
        descriptor.swizzle_atom_bytes
    };
    let (start_address, column_offset) = if descriptor.absolute_leading_address
        && descriptor.start_address % 128 + byte_in_row >= 128
    {
        (
            descriptor.leading_byte_offset,
            (descriptor.start_address % 128 + byte_in_row) % 128,
        )
    } else {
        (
            descriptor.start_address,
            atom * column_stride + byte_in_atom,
        )
    };
    let unswizzled = start_address
        .checked_add((row % 8) * row_stride)
        .and_then(|value| value.checked_add((row / 8) * descriptor.stride_byte_offset))
        .and_then(|value| value.checked_add(column_offset))
        .ok_or_else(|| OpError::message("raw TCGEN source address overflow"))?;
    finish_shared_byte_offset(source, descriptor, unswizzled, access_bytes)
}

/// Swizzle an absolute virtual address and make it window-relative
/// (legacy `raw_tcgen05_finish_shared_byte_offset`).
pub fn finish_shared_byte_offset(
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    unswizzled: usize,
    access_bytes: usize,
) -> OpResult<usize> {
    let SharedWindow {
        virtual_base,
        view_offset,
        view_len,
        backing_byte_len,
        access_view,
    } = source;
    let swizzled_virtual = if descriptor.swizzle_bits == 0 {
        unswizzled
    } else {
        let atom_shift = descriptor.swizzle_atom_bytes.trailing_zeros();
        let byte_in_atom = unswizzled & (descriptor.swizzle_atom_bytes - 1);
        let atom_index = unswizzled >> atom_shift;
        let swizzle_mask = (1_usize << descriptor.swizzle_bits) - 1;
        let swizzled_atom = atom_index
            ^ ((atom_index & (swizzle_mask << descriptor.swizzle_xor_shift))
                >> descriptor.swizzle_xor_shift);
        (swizzled_atom << atom_shift) | byte_in_atom
    };
    let swizzled_relative =
        usize::try_from(swizzled_virtual as i128 - virtual_base as i128 + view_offset as i128)
            .map_err(|_| OpError::message("raw TCGEN source precedes its shared backing"))?;
    let access_end = swizzled_relative
        .checked_add(access_bytes)
        .ok_or_else(|| OpError::message("raw TCGEN source end overflow"))?;
    if access_end > backing_byte_len {
        return Err(OpError::message(format!(
            "raw TCGEN source range [{swizzled_relative}, {access_end}) exceeds shared backing {backing_byte_len}"
        )));
    }
    let view_end = view_offset
        .checked_add(view_len)
        .ok_or_else(|| OpError::message("raw TCGEN source view end overflow"))?;
    if access_view && (swizzled_relative < view_offset || access_end > view_end) {
        return Err(OpError::message(format!(
            "raw TCGEN source range [{swizzled_relative}, {access_end}) exceeds selected shared view [{view_offset}, {view_end})"
        )));
    }
    swizzled_relative
        .checked_sub(view_offset)
        .ok_or_else(|| OpError::message("raw TCGEN source precedes its shared view"))
}

/// PTX Table 67: MN-major atoms are 128B x 4 K elements for TF32
/// (32B atomicity), and swizzle-width x 8 K elements for 8/16-bit operands.
/// Numeric gathers and checker footprints must use this same address owner.
pub fn matrix_byte_offset(
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    row: usize,
    column: usize,
    transpose: bool,
    element_bytes: usize,
) -> OpResult<usize> {
    if !transpose {
        let byte_in_row = column
            .checked_mul(element_bytes)
            .ok_or_else(|| OpError::message("raw MMA matrix byte offset overflow"))?;
        return shared_byte_offset(source, descriptor, row, byte_in_row, element_bytes);
    }
    let tf32 = element_bytes == 4;
    if tf32 != (descriptor.swizzle_atom_bytes == 32) {
        return Err(OpError::message(
            "MN-major TF32 requires 128B/32B-atomic swizzling; other operand widths forbid it",
        ));
    }
    let swizzle_bytes = descriptor.swizzle_atom_bytes << descriptor.swizzle_bits;
    let elements_per_swizzle = swizzle_bytes / element_bytes;
    let k_per_atom = if tf32 { 4 } else { 8 };
    // Without swizzling, the LDO/SDO roles are exchanged.
    let (mn_stride, k_stride) = if descriptor.swizzle_bits == 0 {
        (
            descriptor.stride_byte_offset,
            descriptor.leading_byte_offset,
        )
    } else {
        (
            descriptor.leading_byte_offset,
            descriptor.stride_byte_offset,
        )
    };
    let unswizzled = descriptor
        .start_address
        .checked_add((row % elements_per_swizzle) * element_bytes)
        .and_then(|v| v.checked_add((row / elements_per_swizzle).checked_mul(mn_stride)?))
        .and_then(|v| v.checked_add((column % k_per_atom) * swizzle_bytes))
        .and_then(|v| v.checked_add((column / k_per_atom).checked_mul(k_stride)?))
        .ok_or_else(|| OpError::message("raw MMA MN-major source address overflow"))?;
    finish_shared_byte_offset(source, descriptor, unswizzled, element_bytes)
}

/// [`matrix_byte_offset`] for 2-byte (f16/bf16) elements: window-relative byte offset
/// of `(row, column)`, K-major unless `transpose`. No numerics.
pub fn b16_matrix_byte_offset(
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    row: usize,
    column: usize,
    transpose: bool,
) -> OpResult<usize> {
    matrix_byte_offset(source, descriptor, row, column, transpose, 2)
}

/// [`matrix_byte_offset`] for 1-byte elements (FP8, and FP6/FP4 in 8-bit containers).
/// No numerics.
pub fn byte8_matrix_byte_offset(
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    row: usize,
    column: usize,
    transpose: bool,
) -> OpResult<usize> {
    matrix_byte_offset(source, descriptor, row, column, transpose, 1)
}

/// Window-relative byte offset of 32-bit word `word` of K-major `row` (the tcgen05.cp
/// source walk). No numerics; errors as [`shared_byte_offset`].
pub fn shared_word_offset(
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    row: usize,
    word: usize,
) -> OpResult<usize> {
    let byte_in_row = word
        .checked_mul(4)
        .ok_or_else(|| OpError::message("raw TCGEN word offset overflow"))?;
    shared_byte_offset(source, descriptor, row, byte_in_row, 4)
}

/// `(byte_offset, byte_len)` of one tcgen05.cp source word (legacy `raw_tcgen05_cp_source_span`).
pub fn cp_source_span(
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    row: usize,
    word: usize,
    decompress: u8,
) -> OpResult<(usize, usize)> {
    match decompress {
        0 => Ok((shared_word_offset(source, descriptor, row, word)?, 4)),
        1 => {
            let atom = word / 4;
            let word_in_atom = word % 4;
            let byte_in_row = atom
                .checked_mul(16)
                .and_then(|value| value.checked_add(word_in_atom * 2))
                .ok_or_else(|| OpError::message("raw tcgen05.cp b4 source overflow"))?;
            Ok((
                shared_byte_offset(source, descriptor, row, byte_in_row, 2)?,
                2,
            ))
        }
        2 => {
            let atom = word / 4;
            let word_in_atom = word % 4;
            let byte_in_row = atom
                .checked_mul(16)
                .and_then(|value| value.checked_add(word_in_atom * 3))
                .ok_or_else(|| OpError::message("raw tcgen05.cp b6 source overflow"))?;
            Ok((
                shared_byte_offset(source, descriptor, row, byte_in_row, 3)?,
                3,
            ))
        }
        _ => Err(OpError::message(format!(
            "raw tcgen05.cp decompression code {decompress} is invalid"
        ))),
    }
}

/// Visit the `(offset, count)` byte runs of one 16-element narrow atom
/// (legacy `raw_tcgen05_narrow_shared_atom_accesses`).
#[allow(clippy::too_many_arguments)]
pub fn narrow_shared_atom_accesses(
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    row: usize,
    k: usize,
    format: NarrowFormat,
    atom: usize,
    mut visit: impl FnMut(usize, usize) -> OpResult<()>,
    padded_atoms: bool,
) -> OpResult<()> {
    let bytes = format.payload_bytes_per_k16();
    // PTX 9.7.18.10.4.4: K32 pads each group of 16 values to 16 bytes;
    // K64 packs them contiguously. A packed FP6 group can cross a swizzle atom.
    // Ordinary sparse F8/F6/F4 retains 16-byte atoms even though B has K=64.
    // Block-scaled K64 uses contiguous payloads instead; logical K alone is insufficient.
    let start = atom
        * if padded_atoms {
            16
        } else {
            format.shared_atom_stride(k)
        };
    let mut done = 0;
    while done < bytes {
        let column = start + done;
        let count = (bytes - done)
            .min(descriptor.swizzle_atom_bytes - column % descriptor.swizzle_atom_bytes);
        let offset = shared_byte_offset(source, descriptor, row, column, count)?;
        visit(offset, count)?;
        done += count;
    }
    Ok(())
}

/// PTX 9.4 Tables 49 and Figures 279--282: LUT-B reads a full 48-byte
/// compressed K=128 row; descriptor bit 53 selects one 24-byte K=64 segment.
pub fn lut_b_row_accesses(
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    row: usize,
    mut visit: impl FnMut(usize, usize) -> OpResult<()>,
) -> OpResult<()> {
    let mut byte = 0;
    while byte < 48 {
        let count =
            (48 - byte).min(descriptor.swizzle_atom_bytes - byte % descriptor.swizzle_atom_bytes);
        visit(
            shared_byte_offset(source, descriptor, row, byte, count)?,
            count,
        )?;
        byte += count;
    }
    Ok(())
}

/// PTX zero-column descriptor (legacy `RawTcgenColumnMask`). One immutable
/// mapping owns both B addresses and the numeric zero mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnMask {
    bits: u64,
    bank_columns: usize,
}

impl ColumnMask {
    /// Validate a zero-column mask descriptor for an `m`-row (32/64/128), `n`-column B:
    /// reserved bits clear and the column shift within the instruction's maximum. No numerics.
    pub fn new(bits: u64, m: usize, n: usize, instruction: u32) -> OpResult<Self> {
        let maximum_shift = match instruction >> 30 {
            0 => 0,
            code => 4 << code,
        };
        if !matches!(m, 32 | 64 | 128)
            || n == 0
            || !n.is_multiple_of(128 / m)
            || bits & 0xc000_0070_0000_0000 != 0
            || ((bits >> 56) & 63) > if m == 32 { 16 } else { 32 }
            || ((bits >> 56) & 63) > maximum_shift
        {
            return Err(OpError::message(
                "invalid TCGEN zero-column mask descriptor",
            ));
        }
        Ok(Self {
            bits,
            bank_columns: n / (128 / m),
        })
    }

    /// B column that logical `column` reads after the mask's shift, or `None` when the
    /// use/skip span pattern zeroes it (the MMA then multiplies by zero).
    pub fn source_column(self, column: usize) -> Option<usize> {
        let bank = column / self.bank_columns;
        if self.bits & (1 << 39) != 0 {
            let start = ((self.bits >> (8 * bank)) & 255) as usize;
            let first_zero = self.bits & (1 << (32 + bank)) != 0;
            // PTX examples 2--4 and the SM100 oracle: use-span gives the
            // number of B columns, skip-span gives the number of zero columns.
            let used = ((self.bits >> 48) & 255) as usize + 1;
            let zero = ((self.bits >> 40) & 255) as usize + 1;
            // The initial-span counter saturates at one remaining column;
            // SM100's runtime-descriptor oracle covers start >= first span.
            let start = start.min(if first_zero { zero } else { used } - 1);
            let position = (column % self.bank_columns + start) % (used + zero);
            if if first_zero {
                position < zero
            } else {
                position >= used
            } {
                return None;
            }
        }
        Some(column + ((self.bits >> 56) & 63) as usize)
    }
}

/// Map a logical row through an optional column mask (`None` = zero row).
pub fn masked_row(mask: Option<ColumnMask>, row: usize) -> Option<usize> {
    mask.map_or(Some(row), |mask| mask.source_column(row))
}

// ---------------------------------------------------------------------------
// Tile GEMM operand layouts (legacy `TileGemmOperandLayout` & friends).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileGemmOperandLayout {
    pub atom_columns: usize,
    pub per_element_shift: u32,
    pub outer_mask: usize,
    pub atom_shift: u32,
}

impl TileGemmOperandLayout {
    /// Swizzled operand layout: `atom_columns` per atom, XOR of the element-group index
    /// (`>> per_element_shift`) with its `outer_mask` bits shifted down by `atom_shift`.
    pub const fn new(
        atom_columns: usize,
        per_element_shift: u32,
        outer_mask: usize,
        atom_shift: u32,
    ) -> Self {
        Self {
            atom_columns,
            per_element_shift,
            outer_mask,
            atom_shift,
        }
    }
}

pub type TileGemmBf16OperandLayout = TileGemmOperandLayout;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileGemmBf16Descriptor {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a_layout: TileGemmOperandLayout,
    pub b_layout: TileGemmOperandLayout,
    pub reuse_a_as_b: bool,
}

impl TileGemmBf16Descriptor {
    /// Build a tile BF16 GEMM descriptor (`m x n x k`, operand layouts, B aliasing A).
    pub const fn new(
        m: usize,
        n: usize,
        k: usize,
        a_layout: TileGemmOperandLayout,
        b_layout: TileGemmOperandLayout,
        reuse_a_as_b: bool,
    ) -> Self {
        Self {
            m,
            n,
            k,
            a_layout,
            b_layout,
            reuse_a_as_b,
        }
    }

    /// The dense CTA1 descriptor check shared by both tile GEMM entry points.
    pub fn validate_dense_cta1(&self) -> OpResult<()> {
        if !matches!(self.m, 64 | 128)
            || self.n < 64
            || self.k < 64
            || (self.reuse_a_as_b && (self.m != self.n || self.a_layout != self.b_layout))
        {
            return Err(OpError::message(
                "invalid tile BF16 GEMM descriptor for the dense CTA1 path",
            ));
        }
        Ok(())
    }
}

/// Physical element index of every logical `(row, column)` (row-major order) of a
/// swizzled tile GEMM operand. No numerics; errors on an invalid or out-of-range layout.
pub fn tile_gemm_operand_physical_elements(
    rows: usize,
    columns: usize,
    layout: TileGemmOperandLayout,
) -> OpResult<Vec<usize>> {
    let element_count = rows
        .checked_mul(columns)
        .ok_or_else(|| OpError::message("tile GEMM matrix shape overflows usize"))?;
    if rows == 0
        || columns == 0
        || layout.atom_columns == 0
        || !columns.is_multiple_of(layout.atom_columns)
        || layout.per_element_shift >= usize::BITS
        || layout.atom_shift >= usize::BITS
    {
        return Err(OpError::message("invalid tile GEMM operand layout"));
    }
    let element_group = 1_usize << layout.per_element_shift;
    if element_count % element_group != 0 {
        return Err(OpError::message(
            "tile GEMM operand layout does not divide its element domain",
        ));
    }
    let quotient_domain = element_count / element_group;
    if layout.outer_mask >= quotient_domain {
        return Err(OpError::message(
            "tile GEMM operand swizzle exceeds its element domain",
        ));
    }

    let element_mask = element_group - 1;
    let column_tiles = columns / layout.atom_columns;
    let mut physical_elements = Vec::with_capacity(element_count);
    for row in 0..rows {
        for column_tile in 0..column_tiles {
            let physical_tile_base = column_tile
                .checked_mul(rows)
                .and_then(|value| value.checked_mul(layout.atom_columns))
                .and_then(|value| value.checked_add(row * layout.atom_columns))
                .ok_or_else(|| OpError::message("tile GEMM operand tile offset overflow"))?;
            for inner_column in 0..layout.atom_columns {
                let unswizzled = physical_tile_base
                    .checked_add(inner_column)
                    .ok_or_else(|| OpError::message("tile GEMM operand offset overflow"))?;
                let quotient = unswizzled >> layout.per_element_shift;
                let swizzled_quotient =
                    quotient ^ ((quotient & layout.outer_mask) >> layout.atom_shift);
                let physical =
                    (swizzled_quotient << layout.per_element_shift) | (unswizzled & element_mask);
                if physical >= element_count {
                    return Err(OpError::message(
                        "tile GEMM operand swizzle produced an out-of-range element",
                    ));
                }
                physical_elements.push(physical);
            }
        }
    }
    Ok(physical_elements)
}

/// Decode a little-endian BF16 shared snapshot into logical row-major f32 values
/// through [`tile_gemm_operand_physical_elements`]; exact widening, NaN payloads kept.
/// Errors on a size mismatch.
pub fn decode_tile_gemm_bf16_snapshot(
    snapshot: &[u8],
    rows: usize,
    columns: usize,
    layout: TileGemmOperandLayout,
) -> OpResult<Vec<f32>> {
    let physical_elements = tile_gemm_operand_physical_elements(rows, columns, layout)?;
    let expected_bytes = physical_elements
        .len()
        .checked_mul(2)
        .ok_or_else(|| OpError::message("tile BF16 GEMM snapshot size overflows usize"))?;
    if snapshot.len() != expected_bytes {
        return Err(OpError::message(format!(
            "tile BF16 GEMM snapshot has {} bytes, expected {expected_bytes}",
            snapshot.len()
        )));
    }
    physical_elements
        .into_iter()
        .map(|physical| {
            let byte = physical * 2;
            Ok(crate::cvt::bf16_bits_to_f32(u16::from_le_bytes([
                snapshot[byte],
                snapshot[byte + 1],
            ])))
        })
        .collect()
}

#[cfg(test)]
#[path = "smem_desc_tests.rs"]
mod tests;
