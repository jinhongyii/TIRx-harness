//! Exact integer tcgen05.mma (`kind::i8`, `kind::ti16` / s1z4m11).
//!
//! Legacy source: `engine-rs/src/runtime/tcgen_ops/integer.rs`
//! (`RawTcgenIntegerKind`, `decode_ti16`, `raw_tcgen05_integer_shape`,
//! `gather_integer_shared`, the accumulate/store half of `raw_tcgen05_mma_integer`).

use super::gather::SharedRead;
use super::smem_desc::{
    b16_matrix_byte_offset, byte8_matrix_byte_offset, masked_row, ColumnMask, MatrixDescriptor,
    SharedWindow,
};
use crate::types::{OpError, OpResult};

/// Legacy `EngineError::analysis_incomplete` kind raised for TI16 transpose.
pub const TI16_TRANSPOSE_UNMODELED: &str = "ti16_transpose_unmodeled";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegerKind {
    Ti16,
    I8,
}

impl IntegerKind {
    pub fn packed_k(self) -> usize {
        match self {
            Self::Ti16 => 16,
            Self::I8 => 32,
        }
    }

    /// Decode one operand element; `format` is the 3-bit descriptor format.
    pub fn decode(self, bits: u16, format: u32, negate: bool) -> OpResult<i32> {
        match self {
            Self::Ti16 => decode_ti16(bits, negate),
            Self::I8 => Ok(if format == 1 {
                i32::from(bits as u8 as i8)
            } else {
                i32::from(bits as u8)
            }),
        }
    }

    pub fn shared_offset(
        self,
        source: SharedWindow,
        descriptor: MatrixDescriptor,
        row: usize,
        column: usize,
        transpose: bool,
    ) -> OpResult<usize> {
        match self {
            Self::Ti16 => b16_matrix_byte_offset(source, descriptor, row, column, transpose),
            Self::I8 => byte8_matrix_byte_offset(source, descriptor, row, column, transpose),
        }
    }

    /// Bytes per element in shared memory.
    pub fn element_bytes(self) -> usize {
        32 / self.packed_k()
    }
}

pub fn decode_ti16(bits: u16, negate: bool) -> OpResult<i32> {
    if bits & 0x7800 != 0 {
        return Err(OpError::message(
            "s1z4m11 operand has nonzero reserved bits",
        ));
    }
    let magnitude = i32::from(bits & 0x7ff);
    Ok(if (bits & 0x8000 != 0) ^ negate {
        -magnitude
    } else {
        magnitude
    })
}

/// `(m, n, transpose_a, transpose_b)`. A TI16 transpose fails with the
/// message `"ti16_transpose_unmodeled requires an unmodeled analysis contract"`
/// (legacy `EngineError::analysis_incomplete(TI16_TRANSPOSE_UNMODELED)`).
pub fn integer_shape(
    kind: IntegerKind,
    descriptor: u32,
    cta_group: usize,
    weight_stationary: bool,
    sparse: bool,
) -> OpResult<(usize, usize, bool, bool)> {
    // PTX 9.4 Table 51: both integer kinds accumulate in S32.
    let mut reserved = if weight_stationary {
        0x2080004f
    } else {
        0xe080004f
    };
    if sparse {
        reserved &= !7;
    }
    if kind == IntegerKind::I8 {
        reserved &= !8;
        reserved |= (1 << 13) | (1 << 14);
    }
    let a_format = (descriptor >> 7) & 7;
    let b_format = (descriptor >> 10) & 7;
    let valid_operands = match kind {
        IntegerKind::Ti16 => a_format == 3 && b_format == 3 && (!sparse || descriptor & 2 == 0),
        IntegerKind::I8 => a_format <= 1 && b_format <= 1 && !sparse,
    };
    if descriptor & reserved != 0
        || (descriptor & 4 != 0) != sparse
        || (descriptor >> 4) & 3 != 2
        || !valid_operands
    {
        return Err(OpError::message(
            "integer MMA descriptor has invalid reserved bits, sparsity or operand formats",
        ));
    }
    let m = ((descriptor >> 24) & 31) as usize * 16;
    let n = ((descriptor >> 17) & 63) as usize * 8;
    let valid = match (cta_group, m, weight_stationary) {
        (1, 32 | 64 | 128, true) => matches!(n, 64 | 128) || (!sparse && n == 256),
        (1, 64 | 128, false) if kind == IntegerKind::I8 => {
            matches!(n, 8 | 24) || ((16..=256).contains(&n) && n.is_multiple_of(16))
        }
        (1, 64, false) => (8..=256).contains(&n) && n.is_multiple_of(8),
        (1, 128, false) => (16..=256).contains(&n) && n.is_multiple_of(16),
        (2, 128 | 256, false) => (32..=256).contains(&n) && n.is_multiple_of(32),
        _ => false,
    };
    if !valid {
        return Err(OpError::message(format!(
            "{kind:?} CTA{cta_group} has invalid M={m}, N={n} geometry",
        )));
    }
    if kind == IntegerKind::Ti16 && descriptor & ((1 << 15) | (1 << 16)) != 0 {
        return Err(OpError::message(format!(
            "{TI16_TRANSPOSE_UNMODELED} requires an unmodeled analysis contract"
        )));
    }
    Ok((
        m,
        n,
        descriptor & (1 << 15) != 0,
        descriptor & (1 << 16) != 0,
    ))
}

/// Integer rows of one CTA; masked rows read as zero.
#[allow(clippy::too_many_arguments)]
pub fn gather_integer_rows(
    read_shared: &mut impl SharedRead,
    kind: IntegerKind,
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    rows: usize,
    columns: usize,
    transpose: bool,
    mask: Option<ColumnMask>,
    decode: impl Fn(u16) -> OpResult<i32>,
) -> OpResult<Vec<i32>> {
    let mut values = Vec::with_capacity(rows * columns);
    let width = kind.element_bytes();
    for row in 0..rows {
        let Some(row) = masked_row(mask, row) else {
            values.resize(values.len() + columns, 0);
            continue;
        };
        for column in 0..columns {
            let offset = kind.shared_offset(source, descriptor, row, column, transpose)?;
            let mut bytes = [0_u8; 2];
            read_shared(offset, &mut bytes[..width])?;
            values.push(decode(u16::from_le_bytes(bytes))?);
        }
    }
    Ok(values)
}

/// `(offset, bytes)` shared accesses of an integer operand on one CTA
/// (legacy `raw_tcgen05_integer_shared_footprints`).
#[allow(clippy::too_many_arguments)]
pub fn integer_shared_accesses(
    kind: IntegerKind,
    source: SharedWindow,
    descriptor: MatrixDescriptor,
    rows: usize,
    columns: usize,
    transpose: bool,
    mask: Option<ColumnMask>,
) -> OpResult<Vec<(usize, usize)>> {
    let width = kind.element_bytes();
    let atom_elements = if transpose { 1 } else { 16 / width };
    let mut accesses = Vec::new();
    for row in 0..rows {
        let Some(row) = masked_row(mask, row) else {
            continue;
        };
        for column in (0..columns).step_by(atom_elements) {
            let offset = kind.shared_offset(source, descriptor, row, column, transpose)?;
            accesses.push((offset, atom_elements * width));
        }
    }
    Ok(accesses)
}

/// Decode packed TMEM A words: TI16 two halves, I8 four bytes.
pub fn decode_integer_tmem_word(
    kind: IntegerKind,
    word: u32,
    decode_a: impl Fn(u16) -> OpResult<i32>,
) -> OpResult<Vec<i32>> {
    Ok(match kind {
        IntegerKind::Ti16 => vec![decode_a(word as u16)?, decode_a((word >> 16) as u16)?],
        IntegerKind::I8 => vec![
            decode_a(word as u8 as u16)?,
            decode_a((word >> 8) as u8 as u16)?,
            decode_a((word >> 16) as u8 as u16)?,
            decode_a((word >> 24) as u16)?,
        ],
    })
}

/// Exact banked integer accumulation: `output` (`m x n` i64, input D or
/// zeros) `+= A * B^T` per N-selected A bank (`banks = a.len() / (m*k)`).
pub fn integer_mma_accumulate(
    m: usize,
    n: usize,
    k: usize,
    a: &[i32],
    b: &[i32],
    output: &mut [i64],
) -> OpResult<()> {
    let banks = a.len() / (m * k);
    let bank_n = n / banks;
    for bank in 0..banks {
        let mut partial = output
            .chunks_exact(n)
            .flat_map(|row| row[bank * bank_n..(bank + 1) * bank_n].iter().copied())
            .collect::<Vec<_>>();
        crate::mma::mma_i32_abt_i64(
            m,
            bank_n,
            k,
            &a[bank * m * k..(bank + 1) * m * k],
            &b[bank * bank_n * k..(bank + 1) * bank_n * k],
            &mut partial,
        )?;
        for (row, partial_row) in output.chunks_exact_mut(n).zip(partial.chunks_exact(bank_n)) {
            row[bank * bank_n..(bank + 1) * bank_n].copy_from_slice(partial_row);
        }
    }
    Ok(())
}

/// Stored s32 cell: I8 with descriptor bit 3 (`.satfinite`) clamps, else wraps.
pub fn integer_store_cell(kind: IntegerKind, descriptor: u32, value: i64) -> [u8; 4] {
    let value = if kind == IntegerKind::I8 && descriptor & 8 != 0 {
        value.clamp(i64::from(i32::MIN), i64::from(i32::MAX))
    } else {
        value
    };
    (value as i32).to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_formats_geometry_and_reserved_bits() {
        for (bits, expected) in [(0, 0), (0x8000, 0), (0x7ff, 2047), (0x87ff, -2047)] {
            assert_eq!(decode_ti16(bits, false).unwrap(), expected);
            assert_eq!(decode_ti16(bits, true).unwrap(), -expected);
        }
        assert!(decode_ti16(0x800, false).is_err());
        let descriptor = (2 << 4) | (3 << 7) | (3 << 10) | (1 << 17) | (4 << 24);
        assert_eq!(
            integer_shape(IntegerKind::Ti16, descriptor, 1, false, false).unwrap(),
            (64, 8, false, false)
        );
        assert!(integer_shape(IntegerKind::Ti16, descriptor | 8, 1, false, false).is_err());
        for bit in [15, 16] {
            let error = integer_shape(IntegerKind::Ti16, descriptor | (1 << bit), 1, false, false)
                .unwrap_err();
            assert!(error.to_string().starts_with(TI16_TRANSPOSE_UNMODELED));
        }
        for (cta, m, n, ws, i8_valid, ti16_valid) in [
            (1, 64, 8, false, true, true),
            (1, 64, 24, false, true, true),
            (1, 128, 8, false, true, false),
            (1, 128, 24, false, true, false),
            (1, 64, 40, false, false, true),
            (1, 128, 40, false, false, false),
            (1, 128, 48, false, true, true),
            (1, 128, 256, false, true, true),
            (1, 128, 264, false, false, false),
            (2, 128, 16, false, false, false),
            (2, 128, 32, false, true, true),
            (2, 256, 24, false, false, false),
            (2, 256, 64, false, true, true),
            (1, 32, 64, true, true, true),
            (1, 128, 24, true, false, false),
            (1, 128, 256, true, true, true),
        ] {
            for (kind, format, valid) in [
                (IntegerKind::I8, 1, i8_valid),
                (IntegerKind::Ti16, 3, ti16_valid),
            ] {
                let descriptor =
                    (2 << 4) | (format << 7) | (format << 10) | ((n / 8) << 17) | ((m / 16) << 24);
                assert_eq!(
                    integer_shape(kind, descriptor, cta, ws, false).is_ok(),
                    valid,
                    "{kind:?} CTA{cta} M{m} N{n} WS={ws}"
                );
            }
        }
    }

    #[test]
    fn integer_store_saturates_only_i8_satfinite() {
        let big = i64::from(i32::MAX) + 5;
        assert_eq!(
            integer_store_cell(IntegerKind::I8, 8, big),
            i32::MAX.to_le_bytes()
        );
        assert_eq!(
            integer_store_cell(IntegerKind::I8, 0, big),
            (big as i32).to_le_bytes()
        );
        assert_eq!(
            integer_store_cell(IntegerKind::Ti16, 8, big),
            (big as i32).to_le_bytes()
        );
    }
}
