//! Element codecs used by tcgen05 MMA operands and accumulators.
//!
//! Legacy source: `engine-rs/src/runtime/tcgen_ops.rs` (`RawTcgenNarrowFormat`,
//! `RawMmaCellDtype`, `raw_tcgen05_tf32_payload_to_f32`).

use crate::cvt::{
    f32_to_fp16_bits, float4_e2m1fn_bits_to_f32, fp16_bits_to_f32,
    narrow_float_bits_to_f32_checked, NarrowFloatFormat, FLOAT4_E2M1, FLOAT6_E2M3, FLOAT6_E3M2,
    FLOAT8_E4M3, FLOAT8_E5M2,
};
use crate::types::{OpError, OpResult};

/// TF32 operands read storage bits directly: the low 13 mantissa bits are
/// dropped, never rounded.
pub fn tf32_payload_to_f32(bits: u32) -> f32 {
    f32::from_bits(bits & 0xffff_e000)
}

/// `f32` bits of [`NarrowFormat::decode_value_direct`] for every 8-bit code of
/// each format (index = `NarrowFormat as usize`), built once.
fn narrow_decode_table() -> &'static [[u32; 256]; 5] {
    static TABLE: std::sync::OnceLock<[[u32; 256]; 5]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let formats = [
            NarrowFormat::E4M3,
            NarrowFormat::E5M2,
            NarrowFormat::E2M3,
            NarrowFormat::E3M2,
            NarrowFormat::E2M1,
        ];
        std::array::from_fn(|f| {
            debug_assert_eq!(formats[f] as usize, f);
            std::array::from_fn(|code| formats[f].decode_value_direct(code as u8).to_bits())
        })
    })
}

/// `kind::f8f6f4` / `mxf8f6f4` narrow operand formats (legacy `RawTcgenNarrowFormat`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NarrowFormat {
    E4M3,
    E5M2,
    E2M3,
    E3M2,
    E2M1,
}

impl NarrowFormat {
    /// Decode the 3-bit instruction-descriptor format field.
    pub fn decode(bits: u32, operand: &str) -> OpResult<Self> {
        match bits {
            0 => Ok(Self::E4M3),
            1 => Ok(Self::E5M2),
            3 => Ok(Self::E2M3),
            4 => Ok(Self::E3M2),
            5 => Ok(Self::E2M1),
            _ => Err(OpError::message(format!(
                "raw mxf8f6f4 {operand} format {bits} is reserved"
            ))),
        }
    }

    /// The narrow-float codec (`E4M3`/`E5M2`/`E2M3`/`E3M2`/`E2M1` bit layout) of this operand format.
    pub const fn format(self) -> NarrowFloatFormat {
        match self {
            Self::E4M3 => FLOAT8_E4M3,
            Self::E5M2 => FLOAT8_E5M2,
            Self::E2M3 => FLOAT6_E2M3,
            Self::E3M2 => FLOAT6_E3M2,
            Self::E2M1 => FLOAT4_E2M1,
        }
    }

    /// Dense K64 (and sparse B K128) packs atoms contiguously; K32 pads to 16 bytes.
    pub fn shared_atom_stride(self, k: usize) -> usize {
        if k >= 64 {
            self.payload_bytes_per_k16()
        } else {
            16
        }
    }

    /// Packed bytes 16 K elements occupy (2 x element width in bits: 16/12/8).
    pub const fn payload_bytes_per_k16(self) -> usize {
        self.format().width_bits as usize * 2
    }

    /// Exact decode of one narrow code (low `width` bits, sign included) to f32; subnormals
    /// and E5M2 infinities exact. Any NaN code gives `f32::NAN` (`0x7fc0_0000`, sign dropped).
    #[inline]
    pub fn decode_value(self, bits: u8) -> f32 {
        // Table lookup (perf, W4 profile: the per-element decode was ~7% of
        // `deepgemm_sm100_fp8_gemm_1d1d`): the table holds exactly the bits
        // of the direct decode for every code, so results are identical. Codes
        // wider than the format index the table modulo 256 like the direct
        // decode's masked fields; callers pass `width`-bit codes.
        f32::from_bits(narrow_decode_table()[self as usize][usize::from(bits)])
    }

    /// The direct (table-free) decode [`decode_value`](Self::decode_value) is built from.
    pub fn decode_value_direct(self, bits: u8) -> f32 {
        narrow_float_bits_to_f32_checked(bits, self.format()).unwrap_or(f32::NAN)
    }

    /// Decode one 16-byte shared atom to 16 values: E2M1 two nibbles per byte (low first),
    /// FP6/FP8 little-endian packed fields; per value as [`decode_value`](Self::decode_value).
    pub fn decode_shared_atom(self, bytes: [u8; 16]) -> [f32; 16] {
        if matches!(self, Self::E2M1) {
            return std::array::from_fn(|i| {
                float4_e2m1fn_bits_to_f32(bytes[i / 2] >> (4 * (i % 2)))
            });
        }
        let width = self.format().width_bits;
        // One table fetch per atom (the `OnceLock` load showed in the profile).
        let table = &narrow_decode_table()[self as usize];
        if width == 8 {
            // FP8: one code per byte (the packed-field extraction below with
            // width 8 is exactly byte `i`).
            return bytes.map(|code| f32::from_bits(table[usize::from(code)]));
        }
        let packed = u128::from_le_bytes(bytes);
        let mask = (1_u128 << width) - 1;
        std::array::from_fn(|i| {
            f32::from_bits(table[((packed >> (i as u32 * width)) & mask) as usize])
        })
    }

    /// Decode one K=32 TMEM A word to four values (byte 0 first): FP4 in bits 2..5, FP6 in
    /// bits 0..5, FP8 the whole byte. Errors on nonzero padding bits; values as `decode_value`.
    pub fn decode_tmem_word(self, word: u32) -> OpResult<[f32; 4]> {
        // PTX K=32 TMEM containers: FP4 uses bits 2..5, FP6 uses
        // bits 0..5, and FP8 consumes the full byte. Numeric formats
        // themselves are shared with register conversions.
        let shift = if matches!(self, Self::E2M1) { 2 } else { 0 };
        let mask = (((1_u16 << self.format().width_bits) - 1) << shift) as u8;
        let mut values = [0.0; 4];
        for (value, bits) in values.iter_mut().zip(word.to_le_bytes()) {
            if bits & !mask != 0 {
                return Err(OpError::message(
                    "block-scale TMEM A has nonzero padding bits",
                ));
            }
            *value = self.decode_value(bits >> shift);
        }
        Ok(values)
    }

    /// 32-bit TMEM columns one block-scaled A row uses: K=32 8, K=64 packed FP6 12.
    /// No numerics; errors for other K/width combinations.
    pub fn block_tmem_columns(self, k: usize) -> OpResult<usize> {
        match (k, self.format().width_bits) {
            (32, _) => Ok(8),
            (64, 6) => Ok(12),
            _ => Err(OpError::message("MXF8F6F4 K=64 TMEM A requires packed FP6")),
        }
    }

    /// Decode one block-scaled TMEM A row of `k` values (K=32 per-byte containers, K=64
    /// packed FP6 three words per 16 values). Errors on wrong length or padding bits.
    pub fn decode_block_tmem_row(self, words: &[u32], k: usize) -> OpResult<Vec<f32>> {
        if words.len() != self.block_tmem_columns(k)? {
            return Err(OpError::message("invalid narrow TMEM row length"));
        }
        let mut values = Vec::with_capacity(k);
        if k == 32 {
            for &word in words {
                values.extend(self.decode_tmem_word(word)?);
            }
        } else {
            // 16 FP6 values occupy exactly three words; the 6-bit fields
            // crossing a word boundary use the same 96-bit decoder as shared A.
            for group in words.as_chunks::<3>().0 {
                let mut bytes = [0_u8; 16];
                for (dst, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(group) {
                    *dst = word.to_le_bytes();
                }
                values.extend(self.decode_shared_atom(bytes));
            }
        }
        Ok(values)
    }
}

/// How one dense TMEM destination cell carries an accumulator element
/// (legacy `RawMmaCellDtype`).
///
/// Both occupy a whole 32-bit TMEM cell; `F16` keeps the value in the low half
/// and writes the high half as zero. Measured on B200: the MMA does not round
/// the accumulator to binary16 between K steps, and the store rounds to nearest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellDtype {
    F32,
    F16,
}

impl CellDtype {
    /// `F16` when the descriptor selects an F16 accumulator, else `F32`.
    pub fn from_half(half: bool) -> Self {
        if half {
            Self::F16
        } else {
            Self::F32
        }
    }

    /// Read a TMEM cell as f32: F32 bit-exact, F16 exact widening of the low half (NaN
    /// payload kept); the high half is ignored.
    #[inline]
    pub fn decode(self, bytes: [u8; 4]) -> f32 {
        match self {
            Self::F32 => f32::from_le_bytes(bytes),
            Self::F16 => fp16_bits_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])),
        }
    }

    /// Write an accumulator value to a TMEM cell: F32 bit-exact; F16 RN-even to binary16
    /// (overflow to inf, NaN quieted keeping its high payload) with the high half zeroed.
    #[inline]
    pub fn encode(self, value: f32) -> [u8; 4] {
        match self {
            Self::F32 => value.to_le_bytes(),
            Self::F16 => {
                let [low, high] = f32_to_fp16_bits(value).to_le_bytes();
                [low, high, 0, 0]
            }
        }
    }
}

/// Decode one 16-bit F16/BF16 operand (and optionally negate).
#[inline]
pub fn decode_b16(bits: u16, bf16: bool, negate: bool) -> f32 {
    let value = if bf16 {
        crate::cvt::bf16_bits_to_f32(bits)
    } else {
        fp16_bits_to_f32(bits)
    };
    if negate {
        -value
    } else {
        value
    }
}

/// Decode a packed TMEM A word of two F16/BF16 halves.
#[inline]
pub fn decode_b16_word(word: u32, bf16: bool, negate: bool) -> [f32; 2] {
    [word as u16, (word >> 16) as u16].map(|bits| decode_b16(bits, bf16, negate))
}

/// Decode a packed TMEM A word of eight E2M1 nibbles (no scale, no negate).
pub fn decode_e2m1_word(word: u32) -> [f32; 8] {
    std::array::from_fn(|i| float4_e2m1fn_bits_to_f32(((word >> (i * 4)) & 15) as u8))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decode table is bit-identical to the direct decode for every
    /// format and all 256 codes (NaN codes included).
    #[test]
    fn decode_table_matches_the_direct_decode_exhaustively() {
        for format in [
            NarrowFormat::E4M3,
            NarrowFormat::E5M2,
            NarrowFormat::E2M3,
            NarrowFormat::E3M2,
            NarrowFormat::E2M1,
        ] {
            for code in 0..=255_u8 {
                assert_eq!(
                    format.decode_value(code).to_bits(),
                    format.decode_value_direct(code).to_bits(),
                    "{format:?} {code:#04x}"
                );
            }
        }
    }

    #[test]
    fn the_f16_destination_codec_uses_the_low_half_and_zeroes_the_upper_half() {
        let stored = CellDtype::F16.encode(1032.0);
        assert_eq!(stored, [0x08, 0x64, 0x00, 0x00]);
        assert_eq!(CellDtype::F16.decode([0x00, 0x64, 0xEF, 0xBE]), 1024.0);
        assert_eq!(CellDtype::F32.decode(1032.0_f32.to_le_bytes()), 1032.0);
        assert_eq!(
            CellDtype::F16.decode(CellDtype::F16.encode(1031.75)),
            1032.0
        );
        assert_eq!(
            CellDtype::F16.decode(CellDtype::F16.encode(1024.25)),
            1024.0
        );
    }

    #[test]
    fn raw_tcgen_tf32_operands_decode_storage_bits_without_rne_conversion() {
        let bits = 1.000_6_f32.to_bits();
        let decoded = tf32_payload_to_f32(bits);
        assert_eq!(decoded.to_bits(), bits & 0xffff_e000);
        assert_ne!(decoded, crate::cvt::f32_to_tf32(f32::from_bits(bits)));
    }

    #[test]
    fn packed_fp4_atoms_preserve_every_nibble_and_signed_zero() {
        const EXPECTED: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        for packed in 0_u16..=255 {
            let mut bytes = [0xff; 16];
            for (i, byte) in bytes[..8].iter_mut().enumerate() {
                *byte = (packed as u8).wrapping_add(i as u8);
            }
            let decoded = NarrowFormat::E2M1.decode_shared_atom(bytes);
            for (i, value) in decoded.iter().enumerate() {
                let byte = (packed as u8).wrapping_add((i / 2) as u8);
                let code = if i % 2 == 0 { byte & 15 } else { byte >> 4 };
                assert_eq!(value.to_bits(), EXPECTED[usize::from(code)].to_bits());
            }
        }
    }
}
