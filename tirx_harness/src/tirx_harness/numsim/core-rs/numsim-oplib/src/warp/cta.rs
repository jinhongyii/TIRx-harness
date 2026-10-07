//! CTA-level reduction numerics: `tirx.cuda.cta_reduce`,
//! `syncthreads_and/or` and `barrier.red.{popc,and,or}`.
//!
//! Legacy sources: `engine-rs/src/collectives.rs` (`CtaReduceOp`,
//! `CtaReduceElement`, `Fp16Reduce`, `Bf16Reduce`, `CtaReduceValue`,
//! `CtaReduceContribution`, and the publisher closure of
//! `CollectiveHub::cta_reduce_hub`) and
//! `engine-rs/src/runtime/instructions/collective.rs` (warp-count check,
//! `cta_vote_variants!` lane folds, `bar_reduce_variants!` lane folds, and the
//! result conversions).
//!
//! The rendezvous itself (`CollectiveHub`, wakers, occurrence keys, the
//! named-barrier generation that picks `barrier.red` participants, and the
//! scratch-memory traffic) stays in the engine. The engine collects one
//! [`CtaReduceContribution`] per participating warp and calls
//! [`cta_reduce_publish`] exactly once.
//!
//! Deterministic combine order (semantic, fixes float rounding):
//! 1. Each warp reduces its 32 lanes with the full-width butterfly of
//!    [`super::reduce::warp_reduce`] and contributes lane 0's value.
//! 2. The publisher places warp partials in a 32-slot array at index
//!    `global_warp_id % warps_per_cta`; unused slots hold the operation's
//!    identity (0 / type MIN / type MAX; ±inf for floats).
//! 3. It runs a second butterfly over the 32 slots, `delta = 16, 8, 4, 2, 1`,
//!    `slot[i] = combine(prev[i], prev[i ^ delta])`, and returns slot 0.

use std::collections::BTreeMap;

use crate::cvt::{bf16_bits_to_f32, f32_to_bf16_bits, f32_to_fp16_bits, fp16_bits_to_f32};
use crate::scalar::{
    cuda_f32_add, cuda_f32_max, cuda_f32_min, cuda_f64_add, cuda_f64_max, cuda_f64_min,
    cuda_reduce_bf16_add, cuda_reduce_bf16_max, cuda_reduce_bf16_min, cuda_reduce_fp16_add,
    cuda_reduce_fp16_max, cuda_reduce_fp16_min,
};
use crate::types::{OpError, OpResult, WarpMask, WarpValue, WARP_SIZE};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CtaReduceOp {
    Sum,
    Max,
    Min,
}

impl CtaReduceOp {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Max => "max",
            Self::Min => "min",
        }
    }
}

/// Legacy `SynchronizationError::CompletionSourceOperationFailed` rendering.
fn cta_reduce_failure(details: impl Into<String>) -> OpError {
    OpError::message(format!(
        "completion source cta_reduce operation failed: {}",
        details.into()
    ))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CtaReduceValue {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F16(Fp16Reduce),
    Bf16(Bf16Reduce),
    F32(f32),
    F64(f64),
}

/// Scalar that can take part in a CTA reduction.
pub trait CtaReduceElement: Copy {
    fn identity(operation: CtaReduceOp) -> Self;
    fn combine(operation: CtaReduceOp, lhs: Self, rhs: Self) -> Self;
    fn into_cta_reduce_value(self) -> CtaReduceValue;
    fn from_cta_reduce_value(value: CtaReduceValue) -> OpResult<Self>;
}

fn type_mismatch(expected: &str, other: CtaReduceValue) -> OpError {
    cta_reduce_failure(format!(
        "CTA reduction result type mismatch: expected {expected}, got {}",
        other.type_name(),
    ))
}

macro_rules! impl_integer_cta_reduce_element {
    ($rust_type:ty, $variant:ident) => {
        impl CtaReduceElement for $rust_type {
            fn identity(operation: CtaReduceOp) -> Self {
                match operation {
                    CtaReduceOp::Sum => 0,
                    CtaReduceOp::Max => <$rust_type>::MIN,
                    CtaReduceOp::Min => <$rust_type>::MAX,
                }
            }
            fn combine(operation: CtaReduceOp, lhs: Self, rhs: Self) -> Self {
                match operation {
                    CtaReduceOp::Sum => lhs.wrapping_add(rhs),
                    CtaReduceOp::Max => lhs.max(rhs),
                    CtaReduceOp::Min => lhs.min(rhs),
                }
            }
            fn into_cta_reduce_value(self) -> CtaReduceValue {
                CtaReduceValue::$variant(self)
            }
            fn from_cta_reduce_value(value: CtaReduceValue) -> OpResult<Self> {
                match value {
                    CtaReduceValue::$variant(value) => Ok(value),
                    other => Err(type_mismatch(stringify!($rust_type), other)),
                }
            }
        }
    };
}

impl_integer_cta_reduce_element!(i8, I8);
impl_integer_cta_reduce_element!(i16, I16);
impl_integer_cta_reduce_element!(i32, I32);
impl_integer_cta_reduce_element!(i64, I64);
impl_integer_cta_reduce_element!(u8, U8);
impl_integer_cta_reduce_element!(u16, U16);
impl_integer_cta_reduce_element!(u32, U32);
impl_integer_cta_reduce_element!(u64, U64);

/// fp16 partial stored as bits (rounded from its f32 carrier).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fp16Reduce(u16);

impl Fp16Reduce {
    pub fn from_f32(value: f32) -> Self {
        Self(f32_to_fp16_bits(value))
    }
    pub fn to_f32(self) -> f32 {
        fp16_bits_to_f32(self.0)
    }
    pub const fn bits(self) -> u16 {
        self.0
    }
}

/// bf16 partial stored as bits (rounded from its f32 carrier).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16Reduce(u16);

impl Bf16Reduce {
    pub fn from_f32(value: f32) -> Self {
        Self(f32_to_bf16_bits(value))
    }
    pub fn to_f32(self) -> f32 {
        bf16_bits_to_f32(self.0)
    }
    pub const fn bits(self) -> u16 {
        self.0
    }
}

macro_rules! impl_low_precision_cta_reduce_element {
    ($wrapper:ty, $variant:ident, $sum:path, $max:path, $min:path) => {
        impl CtaReduceElement for $wrapper {
            fn identity(operation: CtaReduceOp) -> Self {
                Self::from_f32(match operation {
                    CtaReduceOp::Sum => 0.0,
                    CtaReduceOp::Max => f32::NEG_INFINITY,
                    CtaReduceOp::Min => f32::INFINITY,
                })
            }
            fn combine(operation: CtaReduceOp, lhs: Self, rhs: Self) -> Self {
                let lhs = lhs.to_f32();
                let rhs = rhs.to_f32();
                Self::from_f32(match operation {
                    CtaReduceOp::Sum => $sum(lhs, rhs),
                    CtaReduceOp::Max => $max(lhs, rhs),
                    CtaReduceOp::Min => $min(lhs, rhs),
                })
            }
            fn into_cta_reduce_value(self) -> CtaReduceValue {
                CtaReduceValue::$variant(self)
            }
            fn from_cta_reduce_value(value: CtaReduceValue) -> OpResult<Self> {
                match value {
                    CtaReduceValue::$variant(value) => Ok(value),
                    other => Err(type_mismatch(stringify!($wrapper), other)),
                }
            }
        }
    };
}

impl_low_precision_cta_reduce_element!(
    Fp16Reduce,
    F16,
    cuda_reduce_fp16_add,
    cuda_reduce_fp16_max,
    cuda_reduce_fp16_min
);
impl_low_precision_cta_reduce_element!(
    Bf16Reduce,
    Bf16,
    cuda_reduce_bf16_add,
    cuda_reduce_bf16_max,
    cuda_reduce_bf16_min
);

macro_rules! impl_float_cta_reduce_element {
    ($rust_type:ty, $variant:ident, $add:path, $max:path, $min:path) => {
        impl CtaReduceElement for $rust_type {
            fn identity(operation: CtaReduceOp) -> Self {
                match operation {
                    CtaReduceOp::Sum => 0.0,
                    CtaReduceOp::Max => <$rust_type>::NEG_INFINITY,
                    CtaReduceOp::Min => <$rust_type>::INFINITY,
                }
            }
            fn combine(operation: CtaReduceOp, lhs: Self, rhs: Self) -> Self {
                match operation {
                    CtaReduceOp::Sum => $add(lhs, rhs),
                    CtaReduceOp::Max => $max(lhs, rhs),
                    CtaReduceOp::Min => $min(lhs, rhs),
                }
            }
            fn into_cta_reduce_value(self) -> CtaReduceValue {
                CtaReduceValue::$variant(self)
            }
            fn from_cta_reduce_value(value: CtaReduceValue) -> OpResult<Self> {
                match value {
                    CtaReduceValue::$variant(value) => Ok(value),
                    other => Err(type_mismatch(stringify!($rust_type), other)),
                }
            }
        }
    };
}

impl_float_cta_reduce_element!(f32, F32, cuda_f32_add, cuda_f32_max, cuda_f32_min);
impl_float_cta_reduce_element!(f64, F64, cuda_f64_add, cuda_f64_max, cuda_f64_min);

impl CtaReduceValue {
    pub fn type_name(self) -> &'static str {
        match self {
            Self::I8(_) => "i8",
            Self::I16(_) => "i16",
            Self::I32(_) => "i32",
            Self::I64(_) => "i64",
            Self::U8(_) => "u8",
            Self::U16(_) => "u16",
            Self::U32(_) => "u32",
            Self::U64(_) => "u64",
            Self::F16(_) => "float16",
            Self::Bf16(_) => "bfloat16",
            Self::F32(_) => "f32",
            Self::F64(_) => "f64",
        }
    }

    /// Identity of `operation` in this value's scalar type.
    pub fn identity(self, operation: CtaReduceOp) -> Self {
        match self {
            Self::I8(_) => Self::I8(i8::identity(operation)),
            Self::I16(_) => Self::I16(i16::identity(operation)),
            Self::I32(_) => Self::I32(i32::identity(operation)),
            Self::I64(_) => Self::I64(i64::identity(operation)),
            Self::U8(_) => Self::U8(u8::identity(operation)),
            Self::U16(_) => Self::U16(u16::identity(operation)),
            Self::U32(_) => Self::U32(u32::identity(operation)),
            Self::U64(_) => Self::U64(u64::identity(operation)),
            Self::F16(_) => Self::F16(Fp16Reduce::identity(operation)),
            Self::Bf16(_) => Self::Bf16(Bf16Reduce::identity(operation)),
            Self::F32(_) => Self::F32(f32::identity(operation)),
            Self::F64(_) => Self::F64(f64::identity(operation)),
        }
    }

    /// `combine(self, rhs)`; mixed scalar types are an error.
    pub fn combine(self, operation: CtaReduceOp, rhs: Self) -> OpResult<Self> {
        let result = match (self, rhs) {
            (Self::I8(lhs), Self::I8(rhs)) => Self::I8(i8::combine(operation, lhs, rhs)),
            (Self::I16(lhs), Self::I16(rhs)) => Self::I16(i16::combine(operation, lhs, rhs)),
            (Self::I32(lhs), Self::I32(rhs)) => Self::I32(i32::combine(operation, lhs, rhs)),
            (Self::I64(lhs), Self::I64(rhs)) => Self::I64(i64::combine(operation, lhs, rhs)),
            (Self::U8(lhs), Self::U8(rhs)) => Self::U8(u8::combine(operation, lhs, rhs)),
            (Self::U16(lhs), Self::U16(rhs)) => Self::U16(u16::combine(operation, lhs, rhs)),
            (Self::U32(lhs), Self::U32(rhs)) => Self::U32(u32::combine(operation, lhs, rhs)),
            (Self::U64(lhs), Self::U64(rhs)) => Self::U64(u64::combine(operation, lhs, rhs)),
            (Self::F16(lhs), Self::F16(rhs)) => Self::F16(Fp16Reduce::combine(operation, lhs, rhs)),
            (Self::Bf16(lhs), Self::Bf16(rhs)) => {
                Self::Bf16(Bf16Reduce::combine(operation, lhs, rhs))
            }
            (Self::F32(lhs), Self::F32(rhs)) => Self::F32(f32::combine(operation, lhs, rhs)),
            (Self::F64(lhs), Self::F64(rhs)) => Self::F64(f64::combine(operation, lhs, rhs)),
            (lhs, rhs) => {
                return Err(cta_reduce_failure(format!(
                    "CTA reduction participants disagree on scalar type: {} versus {}",
                    lhs.type_name(),
                    rhs.type_name(),
                )))
            }
        };
        Ok(result)
    }

    pub fn same_type(self, other: Self) -> bool {
        self.type_name() == other.type_name()
    }
}

/// One warp's contribution to a CTA reduction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CtaReduceContribution {
    pub operation: CtaReduceOp,
    pub warp_partial: CtaReduceValue,
}

impl CtaReduceContribution {
    pub fn new<T: CtaReduceElement>(operation: CtaReduceOp, warp_partial: T) -> Self {
        Self {
            operation,
            warp_partial: warp_partial.into_cta_reduce_value(),
        }
    }
}

/// Cross-warp publication stage of every CTA reduction (legacy
/// `CtaReduceHub::cta_reduce_hub` publisher). `inputs` is keyed by global warp
/// id; the lowest key supplies the reference operation and scalar type.
/// See the module docs for the combine order.
pub fn cta_reduce_publish(
    warps_per_cta: usize,
    inputs: &BTreeMap<usize, CtaReduceContribution>,
) -> OpResult<CtaReduceValue> {
    let first = inputs
        .values()
        .next()
        .ok_or_else(|| cta_reduce_failure("CTA reduction has no inputs"))?;
    let operation = first.operation;
    let exemplar = first.warp_partial;
    let mut partials = [exemplar.identity(operation); WARP_SIZE];
    for (global_warp_id, contribution) in inputs {
        if contribution.operation != operation {
            return Err(cta_reduce_failure(format!(
                "CTA reduction participants disagree on operation: {} versus {}",
                operation.name(),
                contribution.operation.name(),
            )));
        }
        if !contribution.warp_partial.same_type(exemplar) {
            return Err(cta_reduce_failure(format!(
                "CTA reduction participants disagree on scalar type: {} versus {}",
                exemplar.type_name(),
                contribution.warp_partial.type_name(),
            )));
        }
        // Legacy indexes a 32-slot array; warps_per_cta > 32 would panic there.
        partials[global_warp_id % warps_per_cta] = contribution.warp_partial;
    }
    for delta in [16_usize, 8, 4, 2, 1] {
        let previous = partials;
        for lane in 0..WARP_SIZE {
            partials[lane] = previous[lane].combine(operation, previous[lane ^ delta])?;
        }
    }
    Ok(partials[0])
}

/// `tirx.cuda.cta_reduce` static warp-count contract.
pub fn validate_cta_reduce_warps(warps: usize, warps_per_cta: usize) -> OpResult<()> {
    if warps == 0 || warps > WARP_SIZE || !warps.is_power_of_two() {
        return Err(OpError::message(format!(
            "CTA reduction warp count must be a power of two in 1..={WARP_SIZE}, got {warps}"
        )));
    }
    if warps_per_cta != warps {
        return Err(OpError::message(format!("cuda_cta_reduce expected {warps} warps")));
    }
    Ok(())
}

/// `syncthreads_and` / `syncthreads_or` operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CtaVoteOp {
    And,
    Or,
}

impl CtaVoteOp {
    /// CTA reduction used to publish the vote (`and` -> min, `or` -> max).
    pub const fn reduce_op(self) -> CtaReduceOp {
        match self {
            Self::And => CtaReduceOp::Min,
            Self::Or => CtaReduceOp::Max,
        }
    }
    /// Full-warp diagnostic label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::And => "cuda_syncthreads_and",
            Self::Or => "cuda_syncthreads_or",
        }
    }
}

/// Warp-local contribution of `syncthreads_and/or`: fold over active lanes.
pub fn cta_vote_local(op: CtaVoteOp, active_mask: WarpMask, predicates: &WarpValue<bool>) -> i32 {
    let mut lanes = active_mask.iter();
    i32::from(match op {
        CtaVoteOp::And => lanes.all(|lane| predicates[lane]),
        CtaVoteOp::Or => lanes.any(|lane| predicates[lane]),
    })
}

/// Result of `syncthreads_and/or` from the published i32 (`!= 0` as i64).
pub fn cta_vote_result(published: CtaReduceValue) -> OpResult<i64> {
    Ok(i64::from(i32::from_cta_reduce_value(published)? != 0))
}

/// `barrier.red` operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BarRedOp {
    /// `.popc.u32`
    Popc,
    /// `.and.pred`
    And,
    /// `.or.pred`
    Or,
}

impl BarRedOp {
    /// CTA reduction used to publish (`popc` -> sum, `and` -> min, `or` -> max).
    pub const fn reduce_op(self) -> CtaReduceOp {
        match self {
            Self::Popc => CtaReduceOp::Sum,
            Self::And => CtaReduceOp::Min,
            Self::Or => CtaReduceOp::Max,
        }
    }
}

/// Warp-local `barrier.red` contribution. Counts all 32 lanes' predicates
/// (the engine requires a full warp first): popc = count, and = count == 32,
/// or = count != 0.
pub fn bar_red_local(op: BarRedOp, predicates: &WarpValue<bool>) -> i32 {
    let count_true: i32 = predicates.iter().map(|value| i32::from(*value)).sum();
    match op {
        BarRedOp::And => i32::from(count_true == 32),
        BarRedOp::Or => i32::from(count_true != 0),
        BarRedOp::Popc => count_true,
    }
}

/// `barrier.red` result from the published i32 (`as u32`, broadcast).
pub fn bar_red_result(published: CtaReduceValue) -> OpResult<u32> {
    Ok(i32::from_cta_reduce_value(published)? as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn publish<T: CtaReduceElement>(
        warps_per_cta: usize,
        operation: CtaReduceOp,
        partials: &[(usize, T)],
    ) -> OpResult<CtaReduceValue> {
        let inputs = partials
            .iter()
            .map(|&(warp, partial)| (warp, CtaReduceContribution::new(operation, partial)))
            .collect();
        cta_reduce_publish(warps_per_cta, &inputs)
    }

    // Ported from collectives.rs `cta_reduce_hub_owns_the_cross_warp_stage`.
    #[test]
    fn cta_reduce_owns_the_cross_warp_stage() {
        let result = publish(2, CtaReduceOp::Sum, &[(0, 32.0_f32), (1, 64.0_f32)]).unwrap();
        assert_eq!(f32::from_cta_reduce_value(result).unwrap(), 96.0);
    }

    // Ported from collectives.rs `cta_reduce_hub_supports_max_and_min_identities`.
    #[test]
    fn cta_reduce_supports_max_and_min_identities() {
        for (operation, expected) in [(CtaReduceOp::Max, 7.0), (CtaReduceOp::Min, -3.0)] {
            let result = publish(2, operation, &[(0, -3.0_f32), (1, 7.0_f32)]).unwrap();
            assert_eq!(f32::from_cta_reduce_value(result).unwrap(), expected);
        }
    }

    // Ported from collectives.rs `cta_reduce_hub_supports_integer_and_float64_scalars`.
    #[test]
    fn cta_reduce_supports_integer_and_float64_scalars() {
        let result = publish(2, CtaReduceOp::Sum, &[(0, 10_i32), (1, 20_i32)]).unwrap();
        assert_eq!(result, CtaReduceValue::I32(30));
        let result = publish(2, CtaReduceOp::Sum, &[(0, 1.25_f64), (1, 2.5_f64)]).unwrap();
        assert_eq!(result, CtaReduceValue::F64(3.75));
    }

    #[test]
    fn float64_cta_reduce_combine_preserves_cuda_zero_and_nan_selection() {
        let nan_a = f64::from_bits(0x7ff8_0000_0000_1234);
        let nan_b = f64::from_bits(0xfff8_0000_0000_5678);
        assert_eq!(
            f64::combine(CtaReduceOp::Max, -0.0, -0.0).to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(f64::combine(CtaReduceOp::Min, 0.0, 0.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(f64::combine(CtaReduceOp::Max, nan_a, nan_b).to_bits(), nan_b.to_bits());
        assert_eq!(f64::combine(CtaReduceOp::Min, nan_a, nan_b).to_bits(), nan_b.to_bits());
    }

    #[test]
    fn cta_reduce_places_partials_by_warp_in_cta_and_rejects_mismatches() {
        // Global warp ids 4..8 in a 4-warp CTA land in slots 0..4.
        let result = publish(4, CtaReduceOp::Max, &[(5, 3_u8), (6, 9_u8)]).unwrap();
        assert_eq!(result, CtaReduceValue::U8(9));
        let mut inputs = BTreeMap::new();
        inputs.insert(0, CtaReduceContribution::new(CtaReduceOp::Sum, 1_i32));
        inputs.insert(1, CtaReduceContribution::new(CtaReduceOp::Max, 1_i32));
        assert_eq!(
            cta_reduce_publish(2, &inputs).unwrap_err().to_string(),
            "completion source cta_reduce operation failed: \
             CTA reduction participants disagree on operation: sum versus max"
        );
        inputs.insert(1, CtaReduceContribution::new(CtaReduceOp::Sum, 1_u32));
        assert!(cta_reduce_publish(2, &inputs)
            .unwrap_err()
            .to_string()
            .contains("disagree on scalar type: i32 versus u32"));
        assert!(cta_reduce_publish(2, &BTreeMap::new()).is_err());
        assert!(i64::from_cta_reduce_value(CtaReduceValue::I32(1))
            .unwrap_err()
            .to_string()
            .contains("expected i64, got i32"));
    }

    #[test]
    fn narrow_partials_round_at_publication() {
        let result = publish(
            2,
            CtaReduceOp::Sum,
            &[(0, Fp16Reduce::from_f32(2048.0)), (1, Fp16Reduce::from_f32(1.0))],
        )
        .unwrap();
        assert_eq!(Fp16Reduce::from_cta_reduce_value(result).unwrap().to_f32(), 2048.0);
        let result = publish(
            2,
            CtaReduceOp::Min,
            &[(0, Bf16Reduce::from_f32(1.5)), (1, Bf16Reduce::from_f32(-2.0))],
        )
        .unwrap();
        assert_eq!(Bf16Reduce::from_cta_reduce_value(result).unwrap().to_f32(), -2.0);
    }

    #[test]
    fn warp_count_and_vote_and_bar_red_folds() {
        assert!(validate_cta_reduce_warps(4, 4).is_ok());
        assert_eq!(
            validate_cta_reduce_warps(3, 3).unwrap_err().to_string(),
            "CTA reduction warp count must be a power of two in 1..=32, got 3"
        );
        assert_eq!(
            validate_cta_reduce_warps(4, 2).unwrap_err().to_string(),
            "cuda_cta_reduce expected 4 warps"
        );
        let predicates: WarpValue<bool> = std::array::from_fn(|lane| lane < 3);
        assert_eq!(cta_vote_local(CtaVoteOp::And, WarpMask::from_bits(0b111), &predicates), 1);
        assert_eq!(cta_vote_local(CtaVoteOp::And, WarpMask::FULL, &predicates), 0);
        assert_eq!(cta_vote_local(CtaVoteOp::Or, WarpMask::FULL, &predicates), 1);
        assert_eq!(bar_red_local(BarRedOp::Popc, &predicates), 3);
        assert_eq!(bar_red_local(BarRedOp::And, &predicates), 0);
        assert_eq!(bar_red_local(BarRedOp::And, &[true; 32]), 1);
        assert_eq!(bar_red_local(BarRedOp::Or, &[false; 32]), 0);
        // Two warps: popc sums, and = min, or = max.
        let popc = publish(2, BarRedOp::Popc.reduce_op(), &[(0, 3_i32), (1, 32_i32)]).unwrap();
        assert_eq!(bar_red_result(popc).unwrap(), 35);
        let and = publish(2, CtaVoteOp::And.reduce_op(), &[(0, 1_i32), (1, 0_i32)]).unwrap();
        assert_eq!(cta_vote_result(and).unwrap(), 0);
        let or = publish(2, CtaVoteOp::Or.reduce_op(), &[(0, 1_i32), (1, 0_i32)]).unwrap();
        assert_eq!(cta_vote_result(or).unwrap(), 1);
    }
}
