//! Minimal local vocabulary types for OpLib.
//!
//! These intentionally live in ONE small file: the coordinator will swap them
//! for the `numsim-core` contract types later (re-export or `From` impls), and
//! every other OpLib module only names them through `crate::types`.

use std::error::Error;
use std::fmt;

/// Lanes per warp.
pub const WARP_SIZE: usize = 32;

/// One value per lane.
pub type WarpValue<T> = [T; WARP_SIZE];

/// Active-lane bitmask for one warp (bit `i` = lane `i`).
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct WarpMask(pub u32);

impl WarpMask {
    pub const EMPTY: Self = Self(0);
    pub const FULL: Self = Self(u32::MAX);

    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }
    pub const fn bits(self) -> u32 {
        self.0
    }
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
    pub const fn is_full(self) -> bool {
        self.0 == u32::MAX
    }
    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }
    pub const fn contains(self, lane: usize) -> bool {
        lane < WARP_SIZE && (self.0 & (1_u32 << lane)) != 0
    }
    pub const fn first_active(self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            Some(self.0.trailing_zeros() as usize)
        }
    }
    pub fn from_predicate(mut predicate: impl FnMut(usize) -> bool) -> Self {
        let mut bits = 0_u32;
        for lane in 0..WARP_SIZE {
            if predicate(lane) {
                bits |= 1 << lane;
            }
        }
        Self(bits)
    }
    /// Active lanes in ascending order.
    pub fn iter(self) -> impl Iterator<Item = usize> {
        let mut remaining = self.0;
        std::iter::from_fn(move || {
            if remaining == 0 {
                return None;
            }
            let lane = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1;
            Some(lane)
        })
    }
}

/// Element dtypes from `numsim/dtype_registry.json` (TVM spellings).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Dtype {
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    I128,
    U128,
    F16,
    Bf16,
    F32,
    F64,
    F8E3M4,
    F8E4M3,
    F8E4M3B11Fnuz,
    F8E4M3Fn,
    F8E4M3Fnuz,
    F8E5M2,
    F8E5M2Fnuz,
    F8E8M0Fnu,
    F4E2M1Fn,
}

impl Dtype {
    pub const ALL: [Dtype; 24] = [
        Dtype::Bool,
        Dtype::I8,
        Dtype::U8,
        Dtype::I16,
        Dtype::U16,
        Dtype::I32,
        Dtype::U32,
        Dtype::I64,
        Dtype::U64,
        Dtype::I128,
        Dtype::U128,
        Dtype::F16,
        Dtype::Bf16,
        Dtype::F32,
        Dtype::F64,
        Dtype::F8E3M4,
        Dtype::F8E4M3,
        Dtype::F8E4M3B11Fnuz,
        Dtype::F8E4M3Fn,
        Dtype::F8E4M3Fnuz,
        Dtype::F8E5M2,
        Dtype::F8E5M2Fnuz,
        Dtype::F8E8M0Fnu,
        Dtype::F4E2M1Fn,
    ];

    /// Storage width in bits.
    pub const fn bits(self) -> u32 {
        match self {
            Dtype::Bool | Dtype::I8 | Dtype::U8 => 8,
            Dtype::F8E3M4
            | Dtype::F8E4M3
            | Dtype::F8E4M3B11Fnuz
            | Dtype::F8E4M3Fn
            | Dtype::F8E4M3Fnuz
            | Dtype::F8E5M2
            | Dtype::F8E5M2Fnuz
            | Dtype::F8E8M0Fnu => 8,
            Dtype::I16 | Dtype::U16 | Dtype::F16 | Dtype::Bf16 => 16,
            Dtype::I32 | Dtype::U32 | Dtype::F32 => 32,
            Dtype::I64 | Dtype::U64 | Dtype::F64 => 64,
            Dtype::I128 | Dtype::U128 => 128,
            Dtype::F4E2M1Fn => 4,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Dtype::Bool => "bool",
            Dtype::I8 => "int8",
            Dtype::U8 => "uint8",
            Dtype::I16 => "int16",
            Dtype::U16 => "uint16",
            Dtype::I32 => "int32",
            Dtype::U32 => "uint32",
            Dtype::I64 => "int64",
            Dtype::U64 => "uint64",
            Dtype::I128 => "int128",
            Dtype::U128 => "uint128",
            Dtype::F16 => "float16",
            Dtype::Bf16 => "bfloat16",
            Dtype::F32 => "float32",
            Dtype::F64 => "float64",
            Dtype::F8E3M4 => "float8_e3m4",
            Dtype::F8E4M3 => "float8_e4m3",
            Dtype::F8E4M3B11Fnuz => "float8_e4m3b11fnuz",
            Dtype::F8E4M3Fn => "float8_e4m3fn",
            Dtype::F8E4M3Fnuz => "float8_e4m3fnuz",
            Dtype::F8E5M2 => "float8_e5m2",
            Dtype::F8E5M2Fnuz => "float8_e5m2fnuz",
            Dtype::F8E8M0Fnu => "float8_e8m0fnu",
            Dtype::F4E2M1Fn => "float4_e2m1fn",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|dtype| dtype.name() == name)
    }
}

/// Error returned by an OpLib routine whose operands fall outside its domain.
///
/// Mirrors the legacy `EngineError::message` text so diagnostics stay stable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpError(pub String);

impl OpError {
    pub fn message(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for OpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for OpError {}

pub type OpResult<T> = Result<T, OpError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_names_round_trip_and_match_registry_widths() {
        for dtype in Dtype::ALL {
            assert_eq!(Dtype::from_name(dtype.name()), Some(dtype));
        }
        assert_eq!(Dtype::F4E2M1Fn.bits(), 4);
        assert_eq!(Dtype::U128.bits(), 128);
    }

    #[test]
    fn mask_iterates_active_lanes_ascending() {
        let mask = WarpMask::from_bits(0x8000_0005);
        assert_eq!(mask.iter().collect::<Vec<_>>(), [0, 2, 31]);
        assert_eq!(mask.first_active(), Some(0));
        assert_eq!(mask.len(), 3);
        assert!(WarpMask::from_predicate(|lane| lane < 32).is_full());
    }
}
