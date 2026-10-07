//! Element reductions published by asynchronous bulk / tensor reductions
//! (`cp.reduce.async.bulk{.tensor}` to global memory).
//!
//! Ported verbatim from legacy `engine-rs/src/memory.rs`
//! (`DeferredGlobalReduction::{byte_len, apply}`, ~lines 533-745); the deferral
//! and publication machinery stays in the engine.  Note that `MinF16`/`MaxF16`
//! use the CUDA min/max oracles while the `atom` half forms use
//! `ptx_{min,max}_f32` (`rmw::atomic_half`), as legacy did.

use crate::cvt::formats::{bf16_bits_to_f32, f32_to_bf16_bits, f32_to_fp16_bits, fp16_bits_to_f32};
use crate::scalar::{cuda_f32_max, cuda_f32_min, F32RoundingMode};

/// One element-wise reduction of a bulk reduce (legacy `DeferredGlobalReduction`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BulkReduction {
    AddU32,
    AddI32,
    AddU64,
    AddF32,
    AddF32Ftz,
    AddF64,
    AddF16,
    AddBf16,
    MinU32,
    MinI32,
    MinU64,
    MinI64,
    MinF16,
    MinBf16,
    MaxU32,
    MaxI32,
    MaxU64,
    MaxI64,
    MaxF16,
    MaxBf16,
    IncU32,
    DecU32,
    AndB32,
    AndB64,
    OrB32,
    OrB64,
    XorB32,
    XorB64,
}

impl BulkReduction {
    pub fn byte_len(self) -> usize {
        match self {
            Self::AddF16
            | Self::AddBf16
            | Self::MinF16
            | Self::MinBf16
            | Self::MaxF16
            | Self::MaxBf16 => 2,
            Self::AddU32
            | Self::AddI32
            | Self::AddF32
            | Self::AddF32Ftz
            | Self::MinU32
            | Self::MinI32
            | Self::MaxU32
            | Self::MaxI32
            | Self::IncU32
            | Self::DecU32
            | Self::AndB32
            | Self::OrB32
            | Self::XorB32 => 4,
            Self::AddU64
            | Self::AddF64
            | Self::MinU64
            | Self::MinI64
            | Self::MaxU64
            | Self::MaxI64
            | Self::AndB64
            | Self::OrB64
            | Self::XorB64 => 8,
        }
    }

    /// New destination bytes for `current` reduced with `source` (little endian).
    pub fn apply(self, current: &[u8], source: &[u8]) -> Vec<u8> {
        debug_assert_eq!(current.len(), self.byte_len());
        debug_assert_eq!(source.len(), self.byte_len());
        match self {
            Self::AddU32 => u32::from_le_bytes(current.try_into().unwrap())
                .wrapping_add(u32::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::AddI32 => i32::from_le_bytes(current.try_into().unwrap())
                .wrapping_add(i32::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::AddU64 => u64::from_le_bytes(current.try_into().unwrap())
                .wrapping_add(u64::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::AddF32 | Self::AddF32Ftz => {
                let add = if matches!(self, Self::AddF32Ftz) {
                    crate::scalar::add_f32_ftz
                } else {
                    crate::scalar::add_f32
                };
                add(
                    f32::from_le_bytes(current.try_into().unwrap()),
                    f32::from_le_bytes(source.try_into().unwrap()),
                    F32RoundingMode::Nearest,
                )
                .to_le_bytes()
                .to_vec()
            }
            Self::AddF64 => crate::scalar::cuda_f64_add(
                f64::from_le_bytes(current.try_into().unwrap()),
                f64::from_le_bytes(source.try_into().unwrap()),
            )
            .to_le_bytes()
            .to_vec(),
            Self::AddF16 => f32_to_fp16_bits(
                fp16_bits_to_f32(u16::from_le_bytes(current.try_into().unwrap()))
                    + fp16_bits_to_f32(u16::from_le_bytes(source.try_into().unwrap())),
            )
            .to_le_bytes()
            .to_vec(),
            Self::AddBf16 => f32_to_bf16_bits(
                bf16_bits_to_f32(u16::from_le_bytes(current.try_into().unwrap()))
                    + bf16_bits_to_f32(u16::from_le_bytes(source.try_into().unwrap())),
            )
            .to_le_bytes()
            .to_vec(),
            Self::MinU32 => u32::from_le_bytes(current.try_into().unwrap())
                .min(u32::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::MinI32 => i32::from_le_bytes(current.try_into().unwrap())
                .min(i32::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::MinU64 => u64::from_le_bytes(current.try_into().unwrap())
                .min(u64::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::MinI64 => i64::from_le_bytes(current.try_into().unwrap())
                .min(i64::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::MinF16 => f32_to_fp16_bits(cuda_f32_min(
                fp16_bits_to_f32(u16::from_le_bytes(current.try_into().unwrap())),
                fp16_bits_to_f32(u16::from_le_bytes(source.try_into().unwrap())),
            ))
            .to_le_bytes()
            .to_vec(),
            Self::MinBf16 => f32_to_bf16_bits(cuda_f32_min(
                bf16_bits_to_f32(u16::from_le_bytes(current.try_into().unwrap())),
                bf16_bits_to_f32(u16::from_le_bytes(source.try_into().unwrap())),
            ))
            .to_le_bytes()
            .to_vec(),
            Self::MaxU32 => u32::from_le_bytes(current.try_into().unwrap())
                .max(u32::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::MaxI32 => i32::from_le_bytes(current.try_into().unwrap())
                .max(i32::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::MaxU64 => u64::from_le_bytes(current.try_into().unwrap())
                .max(u64::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::MaxI64 => i64::from_le_bytes(current.try_into().unwrap())
                .max(i64::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::MaxF16 => f32_to_fp16_bits(cuda_f32_max(
                fp16_bits_to_f32(u16::from_le_bytes(current.try_into().unwrap())),
                fp16_bits_to_f32(u16::from_le_bytes(source.try_into().unwrap())),
            ))
            .to_le_bytes()
            .to_vec(),
            Self::MaxBf16 => f32_to_bf16_bits(cuda_f32_max(
                bf16_bits_to_f32(u16::from_le_bytes(current.try_into().unwrap())),
                bf16_bits_to_f32(u16::from_le_bytes(source.try_into().unwrap())),
            ))
            .to_le_bytes()
            .to_vec(),
            Self::IncU32 => {
                let old = u32::from_le_bytes(current.try_into().unwrap());
                let limit = u32::from_le_bytes(source.try_into().unwrap());
                (if old >= limit { 0 } else { old + 1 })
                    .to_le_bytes()
                    .to_vec()
            }
            Self::DecU32 => {
                let old = u32::from_le_bytes(current.try_into().unwrap());
                let limit = u32::from_le_bytes(source.try_into().unwrap());
                (if old == 0 || old > limit {
                    limit
                } else {
                    old - 1
                })
                .to_le_bytes()
                .to_vec()
            }
            Self::AndB32 => (u32::from_le_bytes(current.try_into().unwrap())
                & u32::from_le_bytes(source.try_into().unwrap()))
            .to_le_bytes()
            .to_vec(),
            Self::AndB64 => (u64::from_le_bytes(current.try_into().unwrap())
                & u64::from_le_bytes(source.try_into().unwrap()))
            .to_le_bytes()
            .to_vec(),
            Self::OrB32 => (u32::from_le_bytes(current.try_into().unwrap())
                | u32::from_le_bytes(source.try_into().unwrap()))
            .to_le_bytes()
            .to_vec(),
            Self::OrB64 => (u64::from_le_bytes(current.try_into().unwrap())
                | u64::from_le_bytes(source.try_into().unwrap()))
            .to_le_bytes()
            .to_vec(),
            Self::XorB32 => (u32::from_le_bytes(current.try_into().unwrap())
                ^ u32::from_le_bytes(source.try_into().unwrap()))
            .to_le_bytes()
            .to_vec(),
            Self::XorB64 => (u64::from_le_bytes(current.try_into().unwrap())
                ^ u64::from_le_bytes(source.try_into().unwrap()))
            .to_le_bytes()
            .to_vec(),
        }
    }
}
