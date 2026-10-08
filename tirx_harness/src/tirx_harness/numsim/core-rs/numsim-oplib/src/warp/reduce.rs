//! CUDA `tirx.cuda.warp_reduce` butterfly reductions.
//!
//! Legacy source: `engine-rs/src/runtime/warp_ops.rs` (`WarpReduceElement`,
//! `warp_reduce`, `warp_reduce_{sum,max,min}{,_fp16,_bf16}`).
//!
//! Combine order is semantic (it fixes float rounding): for
//! `delta = width/2, width/4, ..., 1`, every lane simultaneously computes
//! `combine(prev[lane], prev[group_base + ((lane - group_base) ^ delta)])`,
//! i.e. own value on the left. f16/bf16 variants carry values in f32 and
//! round to the narrow format at every step (`cuda_reduce_{fp16,bf16}_*`).

use super::mask::require_full_warp;
use crate::scalar::{
    cuda_f32_add, cuda_f32_max, cuda_f32_min, cuda_f64_add, cuda_f64_max, cuda_f64_min,
    cuda_reduce_bf16_add, cuda_reduce_bf16_max, cuda_reduce_bf16_min, cuda_reduce_fp16_add,
    cuda_reduce_fp16_max, cuda_reduce_fp16_min,
};
use crate::types::{OpError, OpResult, WarpMask, WarpValue, WARP_SIZE};

/// Element type of a CUDA warp butterfly reduction.
pub trait WarpReduceElement: Copy {
    fn reduce_sum(self, other: Self) -> Self;
    fn reduce_max(self, other: Self) -> Self;
    fn reduce_min(self, other: Self) -> Self;
}

macro_rules! impl_integer_warp_reduce_element {
    ($($rust_type:ty),+ $(,)?) => {
        $(
            impl WarpReduceElement for $rust_type {
                fn reduce_sum(self, other: Self) -> Self {
                    self.wrapping_add(other)
                }
                fn reduce_max(self, other: Self) -> Self {
                    self.max(other)
                }
                fn reduce_min(self, other: Self) -> Self {
                    self.min(other)
                }
            }
        )+
    };
}

impl_integer_warp_reduce_element!(i8, i16, i32, i64, u8, u16, u32, u64);

impl WarpReduceElement for f32 {
    fn reduce_sum(self, other: Self) -> Self {
        cuda_f32_add(self, other)
    }
    fn reduce_max(self, other: Self) -> Self {
        cuda_f32_max(self, other)
    }
    fn reduce_min(self, other: Self) -> Self {
        cuda_f32_min(self, other)
    }
}

impl WarpReduceElement for f64 {
    fn reduce_sum(self, other: Self) -> Self {
        cuda_f64_add(self, other)
    }
    fn reduce_max(self, other: Self) -> Self {
        cuda_f64_max(self, other)
    }
    fn reduce_min(self, other: Self) -> Self {
        cuda_f64_min(self, other)
    }
}

/// Generic butterfly reduction over groups of `width` lanes (full warp only).
pub fn warp_reduce<T: Copy>(
    active_mask: WarpMask,
    values: &WarpValue<T>,
    width: usize,
    combine: impl Fn(T, T) -> T,
) -> OpResult<WarpValue<T>> {
    require_full_warp(active_mask, "cuda_warp_reduce")?;
    if width == 0 || width > WARP_SIZE || !width.is_power_of_two() {
        return Err(OpError::message(format!(
            "cuda_warp_reduce width must be a power of two in 1..={WARP_SIZE}, got {width}"
        )));
    }
    let mut result = *values;
    let mut delta = width / 2;
    while delta > 0 {
        let previous = result;
        for lane in active_mask.lanes() {
            let group_base = (lane / width) * width;
            let source_lane = group_base + ((lane - group_base) ^ delta);
            result[lane] = combine(previous[lane], previous[source_lane]);
        }
        delta >>= 1;
    }
    Ok(result)
}

/// `tirx.cuda.warp_reduce` butterfly sum over `width`-lane groups: integers wrap, f32
/// is [`cuda_f32_add`] (RN, canonical NaN), f64 [`cuda_f64_add`]. Full warp required.
pub fn warp_reduce_sum<T: WarpReduceElement>(
    active_mask: WarpMask,
    values: &WarpValue<T>,
    width: usize,
) -> OpResult<WarpValue<T>> {
    warp_reduce(active_mask, values, width, T::reduce_sum)
}

/// Butterfly max over `width`-lane groups ([`cuda_f32_max`]/[`cuda_f64_max`] for floats).
/// Errors unless all 32 lanes are active and `width` is a power of two <= 32.
pub fn warp_reduce_max<T: WarpReduceElement>(
    active_mask: WarpMask,
    values: &WarpValue<T>,
    width: usize,
) -> OpResult<WarpValue<T>> {
    warp_reduce(active_mask, values, width, T::reduce_max)
}

/// Butterfly min over `width`-lane groups ([`cuda_f32_min`]/[`cuda_f64_min`] for floats).
/// Errors unless all 32 lanes are active and `width` is a power of two <= 32.
pub fn warp_reduce_min<T: WarpReduceElement>(
    active_mask: WarpMask,
    values: &WarpValue<T>,
    width: usize,
) -> OpResult<WarpValue<T>> {
    warp_reduce(active_mask, values, width, T::reduce_min)
}

macro_rules! narrow_reduce_fn {
    ($name:ident, $combine:path) => {
        /// f32-carried narrow-format butterfly reduction (rounds every step).
        pub fn $name(
            active_mask: WarpMask,
            values: &WarpValue<f32>,
            width: usize,
        ) -> OpResult<WarpValue<f32>> {
            warp_reduce(active_mask, values, width, $combine)
        }
    };
}

narrow_reduce_fn!(warp_reduce_sum_fp16, cuda_reduce_fp16_add);
narrow_reduce_fn!(warp_reduce_max_fp16, cuda_reduce_fp16_max);
narrow_reduce_fn!(warp_reduce_min_fp16, cuda_reduce_fp16_min);
narrow_reduce_fn!(warp_reduce_sum_bf16, cuda_reduce_bf16_add);
narrow_reduce_fn!(warp_reduce_max_bf16, cuda_reduce_bf16_max);
narrow_reduce_fn!(warp_reduce_min_bf16, cuda_reduce_bf16_min);

#[cfg(test)]
mod tests {
    use super::super::mask::lanes_below;
    use super::*;

    fn from_fn<T>(f: impl FnMut(usize) -> T) -> WarpValue<T> {
        std::array::from_fn(f)
    }

    #[test]
    fn float64_warp_reductions_preserve_cuda_zero_and_nan_selection() {
        let nan_a = f64::from_bits(0x7ff8_0000_0000_1234);
        let nan_b = f64::from_bits(0xfff8_0000_0000_5678);
        let values = from_fn(|lane| match lane {
            0 => 0.0,
            1 => -0.0,
            4 => nan_a,
            5 => nan_b,
            _ => lane as f64,
        });
        let maximum = warp_reduce_max(WarpMask::ALL, &values, 2).unwrap();
        let minimum = warp_reduce_min(WarpMask::ALL, &values, 2).unwrap();
        assert_eq!(maximum[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(maximum[1].to_bits(), 0.0_f64.to_bits());
        assert_eq!(minimum[0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(minimum[1].to_bits(), (-0.0_f64).to_bits());
        use crate::scalar::CUDA_CANONICAL_NAN_F64_BITS;
        assert_eq!(maximum[4].to_bits(), CUDA_CANONICAL_NAN_F64_BITS);
        assert_eq!(maximum[5].to_bits(), CUDA_CANONICAL_NAN_F64_BITS);
        assert_eq!(minimum[4].to_bits(), CUDA_CANONICAL_NAN_F64_BITS);
        assert_eq!(minimum[5].to_bits(), CUDA_CANONICAL_NAN_F64_BITS);
    }

    #[test]
    fn butterfly_reductions_support_integer_and_float_groups() {
        let integers = from_fn(|lane| lane as u32);
        let sums = warp_reduce_sum(WarpMask::ALL, &integers, 8).unwrap();
        let maxima = warp_reduce_max(WarpMask::ALL, &integers, 8).unwrap();
        let minima = warp_reduce_min(WarpMask::ALL, &integers, 8).unwrap();
        for lane in 0..WARP_SIZE {
            let group = lane / 8;
            assert_eq!(sums[lane], (group * 64 + 28) as u32);
            assert_eq!(maxima[lane], (group * 8 + 7) as u32);
            assert_eq!(minima[lane], (group * 8) as u32);
        }

        let floats = from_fn(|lane| lane as f32 - 16.0);
        let sums = warp_reduce_sum(WarpMask::ALL, &floats, 32).unwrap();
        let maxima = warp_reduce_max(WarpMask::ALL, &floats, 32).unwrap();
        let minima = warp_reduce_min(WarpMask::ALL, &floats, 32).unwrap();
        for lane in 0..WARP_SIZE {
            assert_eq!(sums[lane], -16.0);
            assert_eq!(maxima[lane], 15.0);
            assert_eq!(minima[lane], -16.0);
        }

        let doubles = from_fn(|lane| lane as f64 * 0.5 - 8.0);
        let sums = warp_reduce_sum(WarpMask::ALL, &doubles, 32).unwrap();
        let maxima = warp_reduce_max(WarpMask::ALL, &doubles, 32).unwrap();
        let minima = warp_reduce_min(WarpMask::ALL, &doubles, 32).unwrap();
        for lane in 0..WARP_SIZE {
            assert_eq!(sums[lane], -8.0);
            assert_eq!(maxima[lane], 7.5);
            assert_eq!(minima[lane], -8.0);
        }
    }

    #[test]
    fn butterfly_reductions_require_full_warps_and_power_of_two_widths() {
        let values = [1_u32; 32];
        let error = warp_reduce_sum(lanes_below(16), &values, 8).unwrap_err();
        assert_eq!(
            error.to_string(),
            "cuda_warp_reduce requires all 32 lanes, got mask 0x0000ffff"
        );
        let error = warp_reduce_sum(WarpMask::ALL, &values, 3).unwrap_err();
        assert!(error.to_string().contains("width must be a power of two"));
    }

    #[test]
    fn narrow_reductions_round_every_step() {
        // 2048 + 1 is not representable in fp16; each step rounds back to 2048.
        let values = from_fn(|lane| if lane == 0 { 2048.0 } else { 1.0 });
        let sums = warp_reduce_sum_fp16(WarpMask::ALL, &values, 2).unwrap();
        assert_eq!(sums[0], 2048.0);
        let bf = warp_reduce_max_bf16(WarpMask::ALL, &from_fn(|lane| lane as f32), 32).unwrap();
        assert_eq!(bf[0], 31.0);
    }
}
