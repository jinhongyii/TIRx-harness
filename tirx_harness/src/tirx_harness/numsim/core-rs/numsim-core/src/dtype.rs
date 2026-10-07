//! Element types.
//!
//! [`Dtype`] is a *scalar element* type; [`Ty`] = element x lanes is a
//! register value type and covers every TIR vector/wide dtype (`float16x2`,
//! `uint32x4`, `uint128`, `float32x8`) as ONE register (see [`crate::value`]).
//!
//! PTX bit types (`.b8`...`.b64`) map to the unsigned types of the same width;
//! the distinction is only a printing concern. `B128` is TIR `uint128`/`int128`
//! (128-bit loads/stores/CAS, CLC responses).

use serde::{Deserialize, Serialize};
use std::fmt;

/// Scalar element type. Bit width via [`Dtype::bits`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Dtype {
    /// Predicate / bool. Stored as 0 or 1 in a register lane; 8 bits in memory.
    Pred,
    U8,
    U16,
    U32,
    U64,
    S8,
    S16,
    S32,
    S64,
    /// 128-bit untyped (b128). Two registers: lo then hi.
    B128,
    F16,
    BF16,
    /// TF32 stored in a 32-bit container (low 13 mantissa bits zero after cvt).
    TF32,
    F32,
    F64,
    /// float8 e4m3fn (finite, NaN = 0x7f/0xff).
    E4M3,
    /// float8 e5m2.
    E5M2,
    /// unsigned e8m0 scale factor (float8_e8m0fnu).
    UE8M0,
    /// unsigned e4m3 scale factor (NVFP4 block scale).
    UE4M3,
    /// unsigned e5m3 (PTX `.ue5m3`).
    UE5M3,
    /// float6 e2m3fn.
    E2M3,
    /// float6 e3m2fn.
    E3M2,
    /// PTX `.s2f6` signed scale format.
    S2F6,
    /// float4 e2m1fn.
    E2M1,
    /// 4-bit integers (mma .s4/.u4; ldmatrix s8.s4).
    U4,
    S4,
}

impl Dtype {
    pub const fn bits(self) -> u32 {
        use Dtype::*;
        match self {
            Pred | U8 | S8 | E4M3 | E5M2 | UE8M0 | UE4M3 => 8,
            U16 | S16 | F16 | BF16 => 16,
            U32 | S32 | F32 | TF32 => 32,
            U64 | S64 | F64 => 64,
            B128 => 128,
            UE5M3 => 8,
            E2M3 | E3M2 | S2F6 => 6,
            E2M1 | U4 | S4 => 4,
        }
    }

    /// Bytes this element occupies in memory (sub-byte types round up to 1;
    /// packed sub-byte memory layouts are described by the instruction).
    pub const fn mem_bytes(self) -> u32 {
        let b = self.bits();
        if b < 8 {
            1
        } else {
            b / 8
        }
    }

    pub const fn is_float(self) -> bool {
        use Dtype::*;
        matches!(
            self,
            F16 | BF16 | TF32 | F32 | F64 | E4M3 | E5M2 | UE8M0 | UE4M3 | UE5M3 | E2M3 | E3M2 | S2F6 | E2M1
        )
    }

    pub const fn is_signed_int(self) -> bool {
        matches!(self, Dtype::S8 | Dtype::S16 | Dtype::S32 | Dtype::S64 | Dtype::S4)
    }

    pub const fn is_int(self) -> bool {
        use Dtype::*;
        matches!(self, U8 | U16 | U32 | U64 | S8 | S16 | S32 | S64 | U4 | S4)
    }

    /// Number of 64-bit registers one element of this type occupies (1, or 2 for B128).
    pub const fn regs(self) -> u32 {
        if self.bits() > 64 {
            2
        } else {
            1
        }
    }

    /// PTX-style suffix (`f32`, `s32`, `e4m3`, ...). Used by the pretty printer.
    pub const fn ptx_name(self) -> &'static str {
        use Dtype::*;
        match self {
            Pred => "pred",
            U8 => "u8",
            U16 => "u16",
            U32 => "u32",
            U64 => "u64",
            S8 => "s8",
            S16 => "s16",
            S32 => "s32",
            S64 => "s64",
            B128 => "b128",
            F16 => "f16",
            BF16 => "bf16",
            TF32 => "tf32",
            F32 => "f32",
            F64 => "f64",
            E4M3 => "e4m3",
            E5M2 => "e5m2",
            UE8M0 => "ue8m0",
            UE4M3 => "ue4m3",
            UE5M3 => "ue5m3",
            E2M3 => "e2m3",
            E3M2 => "e3m2",
            S2F6 => "s2f6",
            E2M1 => "e2m1",
            U4 => "u4",
            S4 => "s4",
        }
    }

    /// Map a TVM dtype name (`dtype_registry.json` spelling) to a [`Dtype`].
    pub fn from_tvm(name: &str) -> Option<Dtype> {
        use Dtype::*;
        Some(match name {
            "bool" => Pred,
            "int8" => S8,
            "uint8" => U8,
            "int16" => S16,
            "uint16" => U16,
            "int32" => S32,
            "uint32" => U32,
            "int64" => S64,
            "uint64" => U64,
            "int128" | "uint128" => B128,
            "float16" => F16,
            "bfloat16" => BF16,
            "float32" => F32,
            "float64" => F64,
            "float8_e4m3fn" => E4M3,
            "float8_e5m2" => E5M2,
            "float8_e8m0fnu" => UE8M0,
            "float6_e2m3fn" => E2M3,
            "float6_e3m2fn" => E3M2,
            "float4_e2m1fn" => E2M1,
            "int4" => S4,
            "uint4" => U4,
            _ => return None,
        })
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.ptx_name())
    }
}

/// A register value type: `lanes` elements of `elem` packed little-endian
/// (element 0 in the low bits).
///
/// This is how vector and wide TIR dtypes are carried (W1 Q4, decided:
/// *one register, Dtype with lanes*): `float16x2` = `Ty{F16, 2}` (32 bits),
/// `uint32x4` = `Ty{U32, 4}` (128 bits), `uint128` = `Ty{B128, 1}`,
/// `float32x8` (ld.v8 / ld_vec256) = `Ty{F32, 8}` (256 bits).
///
/// Invariant: `1 <= lanes`, `elem.bits() * lanes <= 256`. A value occupies
/// [`Ty::slots`] 64-bit register slots (see [`crate::value`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Ty {
    pub elem: Dtype,
    pub lanes: u8,
}

/// Largest register value in bits.
pub const MAX_VALUE_BITS: u32 = 256;

impl Ty {
    pub const fn scalar(elem: Dtype) -> Ty {
        Ty { elem, lanes: 1 }
    }
    pub const fn vector(elem: Dtype, lanes: u8) -> Ty {
        Ty { elem, lanes }
    }
    /// Total payload bits (sub-byte elements packed densely).
    pub const fn bits(self) -> u32 {
        self.elem.bits() * self.lanes as u32
    }
    /// Bytes in memory (sub-byte payloads round up).
    pub const fn mem_bytes(self) -> u32 {
        self.bits().div_ceil(8)
    }
    /// 64-bit register slots this value occupies (1..=4).
    pub const fn slots(self) -> u32 {
        let s = self.bits().div_ceil(64);
        if s == 0 {
            1
        } else {
            s
        }
    }
    pub const fn is_scalar(self) -> bool {
        self.lanes == 1
    }
    /// Parse a TVM dtype string such as `float16x2`, `uint32x4`, `uint128`.
    pub fn from_tvm(name: &str) -> Option<Ty> {
        if let Some(d) = Dtype::from_tvm(name) {
            return Some(Ty::scalar(d));
        }
        let (base, lanes) = name.rsplit_once('x')?;
        let lanes: u8 = lanes.parse().ok()?;
        let ty = Ty::vector(Dtype::from_tvm(base)?, lanes);
        (ty.bits() <= MAX_VALUE_BITS && lanes >= 1).then_some(ty)
    }
    pub const PRED: Ty = Ty::scalar(Dtype::Pred);
    pub const U8: Ty = Ty::scalar(Dtype::U8);
    pub const U16: Ty = Ty::scalar(Dtype::U16);
    pub const U32: Ty = Ty::scalar(Dtype::U32);
    pub const S32: Ty = Ty::scalar(Dtype::S32);
    pub const U64: Ty = Ty::scalar(Dtype::U64);
    pub const S64: Ty = Ty::scalar(Dtype::S64);
    pub const F32: Ty = Ty::scalar(Dtype::F32);
    pub const F64: Ty = Ty::scalar(Dtype::F64);
    pub const F16: Ty = Ty::scalar(Dtype::F16);
    pub const BF16: Ty = Ty::scalar(Dtype::BF16);
    pub const B128: Ty = Ty::scalar(Dtype::B128);
    pub const F16X2: Ty = Ty::vector(Dtype::F16, 2);
    pub const BF16X2: Ty = Ty::vector(Dtype::BF16, 2);
}

impl From<Dtype> for Ty {
    fn from(d: Dtype) -> Ty {
        Ty::scalar(d)
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.lanes == 1 {
            write!(f, "{}", self.elem)
        } else {
            write!(f, "{}x{}", self.elem, self.lanes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn widths() {
        assert_eq!(Ty::F16X2.bits(), 32);
        assert_eq!(Ty::from_tvm("uint32x4"), Some(Ty::vector(Dtype::U32, 4)));
        assert_eq!(Ty::from_tvm("uint32x4").unwrap().slots(), 2);
        assert_eq!(Ty::from_tvm("float32x8").unwrap().slots(), 4);
        assert_eq!(Dtype::B128.regs(), 2);
        assert_eq!(Dtype::from_tvm("float32"), Some(Dtype::F32));
    }
}
