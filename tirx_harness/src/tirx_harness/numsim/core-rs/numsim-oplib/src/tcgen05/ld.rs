//! tcgen05.ld variant numerics (`.red`, `.spcompress`) and MMA collector state.
//!
//! Legacy source: `engine-rs/src/runtime/instructions/tcgen05.rs`
//! (`LoadReduction` impls, the validation and per-lane compression body of
//! `execute_ld_entry`, `collector_transition`).

use super::layouts::LdstShape;
use crate::types::{OpError, OpResult};

/// `tcgen05.ld.red` reduction operator over 32-bit register payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LdReduction {
    /// `.f32` min/max, optional `.abs`, optional `.NaN` propagation.
    F32 {
        max: bool,
        abs: bool,
        nan: bool,
    },
    U32 {
        max: bool,
    },
    I32 {
        max: bool,
    },
}

impl LdReduction {
    /// One `tcgen05.ld.red` step: f32 min/max via [`crate::scalar::ptx_min_f32`]/`ptx_max_f32`
    /// (no FTZ; `.abs` clears signs first, `.NaN` gives `0x7fff_ffff` on any NaN, else a
    /// NaN loses), or u32/i32 integer min/max.
    pub fn apply(self, lhs: u32, rhs: u32) -> u32 {
        match self {
            Self::F32 { max, abs, nan } => {
                let value = |bits| {
                    let value = f32::from_bits(bits);
                    if abs {
                        value.abs()
                    } else {
                        value
                    }
                };
                let (lhs, rhs) = (value(lhs), value(rhs));
                if max {
                    crate::scalar::ptx_max_f32(lhs, rhs, false, nan).to_bits()
                } else {
                    crate::scalar::ptx_min_f32(lhs, rhs, false, nan).to_bits()
                }
            }
            Self::U32 { max } => {
                if max {
                    lhs.max(rhs)
                } else {
                    lhs.min(rhs)
                }
            }
            Self::I32 { max } => {
                let (lhs, rhs) = (lhs as i32, rhs as i32);
                (if max { lhs.max(rhs) } else { lhs.min(rhs) }) as u32
            }
        }
    }

    /// Left fold over one lane's loaded values in register order.
    pub fn reduce(self, values: impl IntoIterator<Item = u32>) -> Option<u32> {
        values.into_iter().reduce(|lhs, rhs| self.apply(lhs, rhs))
    }
}

/// Validate a tcgen05.ld variant and return the destination register count
/// (including the trailing reduction register). `compression` is
/// `Some((max, abs))` for `.spcompress`.
pub fn ld_destination_count(
    shape: LdstShape,
    num: usize,
    packed: bool,
    reduction: bool,
    compression: Option<(bool, bool)>,
) -> OpResult<usize> {
    let expected = shape
        .registers_per_num()
        .checked_mul(num)
        .ok_or_else(|| OpError::message("tcgen05.ld register count overflow"))?;
    if reduction
        && (packed
            || num < 2
            || !matches!(shape, LdstShape::Shape32x32b | LdstShape::Shape16x32bx2(_)))
    {
        return Err(OpError::message(
            "tcgen05.ld.red requires an unpacked 32x32b/16x32bx2 load with at least x2",
        ));
    }
    if compression.is_some() && (packed || num < 4 || !matches!(shape, LdstShape::Shape32x32b)) {
        return Err(OpError::message(
            "tcgen05.ld.spcompress requires unpacked 32x32b with at least x4",
        ));
    }
    Ok(if compression.is_some() {
        num.div_ceil(32) + num / 2
    } else {
        expected
    } + usize::from(reduction))
}

/// One lane of `tcgen05.ld.spcompress`: from `num` loaded words (and their
/// validity) keep 2 of each 4 by magnitude order; returns the
/// `num.div_ceil(32)` metadata words followed by `num / 2` kept values, with
/// per-output validity. `pair_indices` is `crate::arith::sparse_pair_indices`.
pub fn spcompress_lane(
    values: &[u32],
    valid: &[bool],
    maximum: bool,
    absolute: bool,
    pair_indices: impl Fn([f32; 4], bool) -> [usize; 2],
) -> (Vec<u32>, Vec<bool>) {
    let num = values.len();
    let metadata_count = num.div_ceil(32);
    let outputs = metadata_count + num / 2;
    let mut output = vec![0_u32; outputs];
    let mut output_valid = vec![true; outputs];
    for group in 0..num / 4 {
        let group_valid = (0..4).all(|i| valid[group * 4 + i]);
        let pair = pair_indices(
            std::array::from_fn(|i| {
                let value = f32::from_bits(values[group * 4 + i]);
                if absolute {
                    value.abs()
                } else {
                    value
                }
            }),
            maximum,
        );
        for (j, index) in pair.into_iter().enumerate() {
            let element = group * 2 + j;
            output[element / 16] |= (index as u32) << ((element % 16) * 2);
            output_valid[element / 16] &= group_valid;
            output[metadata_count + element] = values[group * 4 + index];
            output_valid[metadata_count + element] = group_valid;
        }
    }
    (output, output_valid)
}

/// MMA collector-buffer state transition (`::fill`, `::use`/`::lastuse`).
pub fn collector_transition(state: u8, fill: u8, require: u8, discard: u8) -> OpResult<u8> {
    if (fill | require | discard) & !31 != 0 || fill & (require | discard) != 0 {
        return Err(OpError::message("invalid TCGEN collector transition"));
    }
    if state & require != require {
        return Err(OpError::message(format!(
            "tcgen05.mma collector use/lastuse requires a valid previous fill (missing slots {:#x})",
            require & !state
        )));
    }
    Ok((state | fill) & !discard)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reductions_follow_their_operand_type() {
        let r = LdReduction::I32 { max: true };
        assert_eq!(r.reduce([(-3_i32) as u32, 2, (-7_i32) as u32]), Some(2));
        assert_eq!(LdReduction::U32 { max: false }.reduce([5, 3, 9]), Some(3));
        let abs_max = LdReduction::F32 {
            max: true,
            abs: true,
            nan: false,
        };
        assert_eq!(
            abs_max.reduce([(-4.0_f32).to_bits(), 3.0_f32.to_bits()]),
            Some(4.0_f32.to_bits())
        );
        let nan_min = LdReduction::F32 {
            max: false,
            abs: false,
            nan: true,
        };
        assert!(f32::from_bits(nan_min.apply(f32::NAN.to_bits(), 1.0_f32.to_bits())).is_nan());
    }

    #[test]
    fn ld_variant_register_counts() {
        assert_eq!(
            ld_destination_count(LdstShape::Shape16x256b, 2, false, false, None).unwrap(),
            8
        );
        assert_eq!(
            ld_destination_count(LdstShape::Shape32x32b, 4, false, true, None).unwrap(),
            5
        );
        assert_eq!(
            ld_destination_count(
                LdstShape::Shape32x32b,
                64,
                false,
                false,
                Some((true, false))
            )
            .unwrap(),
            2 + 32
        );
        assert!(ld_destination_count(LdstShape::Shape16x64b, 4, false, true, None).is_err());
        assert!(
            ld_destination_count(LdstShape::Shape32x32b, 2, false, false, Some((true, true)))
                .is_err()
        );
    }

    #[test]
    fn spcompress_keeps_two_of_four_with_metadata() {
        let values = [1.0_f32, -5.0, 3.0, 0.5].map(f32::to_bits);
        let (out, valid) = spcompress_lane(
            &values,
            &[true; 4],
            true,
            true,
            crate::arith::sparse_pair_indices,
        );
        assert_eq!(out.len(), 1 + 2);
        assert_eq!(out[0], 1 | (2 << 2));
        assert_eq!(out[1..], [(-5.0_f32).to_bits(), 3.0_f32.to_bits()]);
        assert!(valid.iter().all(|v| *v));
    }

    #[test]
    fn collector_requires_a_prior_fill() {
        assert_eq!(collector_transition(0, 1, 0, 0).unwrap(), 1);
        assert_eq!(collector_transition(1, 0, 1, 1).unwrap(), 0);
        assert!(collector_transition(0, 0, 1, 0).is_err());
        assert!(collector_transition(0, 1, 1, 0).is_err());
    }
}
