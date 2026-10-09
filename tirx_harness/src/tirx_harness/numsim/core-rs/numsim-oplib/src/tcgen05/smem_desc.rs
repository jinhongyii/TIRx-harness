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

/// `(value / atom_bytes, value % atom_bytes)`; a shift and mask for the
/// power-of-two atom sizes descriptors decode to (16 or 32 bytes), the
/// division otherwise (perf: this runs per operand piece). Same results.
#[inline]
pub(crate) fn div_rem_atom(value: usize, atom_bytes: usize) -> (usize, usize) {
    if atom_bytes.is_power_of_two() {
        (value >> atom_bytes.trailing_zeros(), value & (atom_bytes - 1))
    } else {
        (value / atom_bytes, value % atom_bytes)
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
    let (atom, byte_in_atom) = div_rem_atom(byte_in_row, descriptor.swizzle_atom_bytes);
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

/// [`shared_byte_offset`] with the descriptor- and window-derived constants
/// computed once per operand (perf, W4: the K-major gathers call it per
/// 16-byte atom). [`offset`](Self::offset) returns `Some` only when
/// `shared_byte_offset` returns `Ok` with that same value; every other case
/// (any error, any overflow, the absolute-LDO and non-power-of-two-atom
/// layouts) is `None`, and callers then call `shared_byte_offset` itself, so
/// errors and their order are unchanged.
#[derive(Clone, Copy, Debug)]
pub struct SharedOffsets {
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    fast: bool,
    row_stride: usize,
    column_stride: usize,
    atom_shift: u32,
    swizzle_mask: usize,
    view_end: usize,
}

impl SharedOffsets {
    pub fn new(source: SharedWindow, descriptor: MatrixDescriptor) -> Self {
        let atom = descriptor.swizzle_atom_bytes;
        let row_stride = u32::try_from(descriptor.swizzle_bits)
            .ok()
            .and_then(|bits| atom.checked_shl(bits));
        let view_end = source.view_offset.checked_add(source.view_len);
        let fast = !descriptor.absolute_leading_address
            && atom.is_power_of_two()
            && descriptor.swizzle_bits < 16
            && descriptor.swizzle_xor_shift < usize::BITS as usize
            && row_stride.is_some()
            && view_end.is_some();
        let column_stride = if descriptor.swizzle_bits == 0 {
            descriptor.leading_byte_offset.max(atom)
        } else {
            atom
        };
        Self {
            source,
            descriptor,
            fast,
            row_stride: row_stride.unwrap_or(0),
            column_stride,
            atom_shift: atom.trailing_zeros(),
            swizzle_mask: if descriptor.swizzle_bits < 16 { (1_usize << descriptor.swizzle_bits) - 1 } else { 0 },
            view_end: view_end.unwrap_or(0),
        }
    }

    /// `shared_byte_offset(source, descriptor, row, byte_in_row, access_bytes)`
    /// when that is `Ok`, computed without re-deriving the constants;
    /// `None` when the caller must call `shared_byte_offset`.
    #[inline]
    pub fn offset(&self, row: usize, byte_in_row: usize, access_bytes: usize) -> Option<usize> {
        if !self.fast {
            return None;
        }
        let descriptor = &self.descriptor;
        let atom_bytes = descriptor.swizzle_atom_bytes;
        let atom = byte_in_row >> self.atom_shift;
        let byte_in_atom = byte_in_row & (atom_bytes - 1);
        if byte_in_atom.checked_add(access_bytes)? > atom_bytes {
            return None;
        }
        if descriptor.swizzle_bits == 0 && atom != 0 && descriptor.leading_byte_offset == 0 {
            return None;
        }
        let column_offset = atom.checked_mul(self.column_stride)?.checked_add(byte_in_atom)?;
        let unswizzled = descriptor
            .start_address
            .checked_add((row % 8).checked_mul(self.row_stride)?)?
            .checked_add((row / 8).checked_mul(descriptor.stride_byte_offset)?)?
            .checked_add(column_offset)?;
        let swizzled = if descriptor.swizzle_bits == 0 {
            unswizzled
        } else {
            let byte_in_atom = unswizzled & (atom_bytes - 1);
            let atom_index = unswizzled >> self.atom_shift;
            let shift = descriptor.swizzle_xor_shift;
            let swizzled_atom = atom_index ^ ((atom_index & (self.swizzle_mask << shift)) >> shift);
            (swizzled_atom << self.atom_shift) | byte_in_atom
        };
        let source = &self.source;
        let relative = swizzled.checked_add(source.view_offset)?.checked_sub(source.virtual_base)?;
        let end = relative.checked_add(access_bytes)?;
        if end > source.backing_byte_len
            || (source.access_view && (relative < source.view_offset || end > self.view_end))
        {
            return None;
        }
        relative.checked_sub(source.view_offset)
    }
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
            .min(descriptor.swizzle_atom_bytes - div_rem_atom(column, descriptor.swizzle_atom_bytes).1);
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

#[cfg(test)]
#[path = "smem_desc_tests.rs"]
mod tests;
