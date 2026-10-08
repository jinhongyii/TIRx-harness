//! Lane-parallel kernels with runtime CPU dispatch. Each kernel is
//! bit-identical to its scalar definition: IEEE FMA is exactly rounded, so
//! hardware `vfmadd` and libm `fma` agree on every non-NaN result; which NaN
//! payload propagates differs (LLVM may commute `vfmadd` operands in
//! optimized builds), so NaN lanes take the pinned NaN of
//! `numsim_oplib::scalar::host_fma_f32/f64`. Tests compare both, NaNs included.

use crate::value::WARP_SIZE;

/// `a[l] * b[l] + c[l]` with one rounding (RNE), 32 lanes.
#[inline]
pub(crate) fn fma_f32(a: &[f32; WARP_SIZE], b: &[f32; WARP_SIZE], c: &[f32; WARP_SIZE]) -> [f32; WARP_SIZE] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("fma") {
            // SAFETY: the `fma` feature was detected at runtime.
            return unsafe { fma_f32_fma(a, b, c) };
        }
    }
    fma_f32_scalar(a, b, c)
}

#[inline(always)]
fn fma_f32_scalar(a: &[f32; WARP_SIZE], b: &[f32; WARP_SIZE], c: &[f32; WARP_SIZE]) -> [f32; WARP_SIZE] {
    std::array::from_fn(|l| numsim_oplib::scalar::host_fma_f32(a[l], b[l], c[l]))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma,avx2")]
unsafe fn fma_f32_fma(a: &[f32; WARP_SIZE], b: &[f32; WARP_SIZE], c: &[f32; WARP_SIZE]) -> [f32; WARP_SIZE] {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_storeu_ps};
    let mut r = [0f32; WARP_SIZE];
    for i in (0..WARP_SIZE).step_by(8) {
        // SAFETY: i + 8 <= 32 for every array here.
        unsafe {
            let v = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
                _mm256_loadu_ps(c.as_ptr().add(i)),
            );
            _mm256_storeu_ps(r.as_mut_ptr().add(i), v);
        }
    }
    for l in 0..WARP_SIZE {
        if r[l].is_nan() {
            // The hardware payload depends on the operand order LLVM picks.
            r[l] = numsim_oplib::scalar::fma_nan_f32(a[l], b[l], c[l]);
        }
    }
    r
}

/// `a[l] * b[l] + c[l]` with one rounding (RNE), 32 lanes, binary64.
#[inline]
pub(crate) fn fma_f64(a: &[f64; WARP_SIZE], b: &[f64; WARP_SIZE], c: &[f64; WARP_SIZE]) -> [f64; WARP_SIZE] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("fma") {
            // SAFETY: the `fma` feature was detected at runtime.
            return unsafe { fma_f64_fma(a, b, c) };
        }
    }
    fma_f64_scalar(a, b, c)
}

#[inline(always)]
fn fma_f64_scalar(a: &[f64; WARP_SIZE], b: &[f64; WARP_SIZE], c: &[f64; WARP_SIZE]) -> [f64; WARP_SIZE] {
    std::array::from_fn(|l| numsim_oplib::scalar::host_fma_f64(a[l], b[l], c[l]))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma,avx2")]
unsafe fn fma_f64_fma(a: &[f64; WARP_SIZE], b: &[f64; WARP_SIZE], c: &[f64; WARP_SIZE]) -> [f64; WARP_SIZE] {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_storeu_pd};
    let mut r = [0f64; WARP_SIZE];
    for i in (0..WARP_SIZE).step_by(4) {
        // SAFETY: i + 4 <= 32 for every array here.
        unsafe {
            let v = _mm256_fmadd_pd(
                _mm256_loadu_pd(a.as_ptr().add(i)),
                _mm256_loadu_pd(b.as_ptr().add(i)),
                _mm256_loadu_pd(c.as_ptr().add(i)),
            );
            _mm256_storeu_pd(r.as_mut_ptr().add(i), v);
        }
    }
    for l in 0..WARP_SIZE {
        if r[l].is_nan() {
            r[l] = numsim_oplib::scalar::fma_nan_f64(a[l], b[l], c[l]);
        }
    }
    r
}

/// IEEE binary32 -> binary16, round-to-nearest-even, 32 lanes: identical to
/// `numsim_oplib::cvt::f32_to_fp16_bits` (NaN keeps sign and the top 10
/// payload bits with the quiet bit set; tested on every exponent class).
#[inline]
pub(crate) fn f32_to_f16_rne(a: &[f32; WARP_SIZE]) -> [u16; WARP_SIZE] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("f16c") {
            // SAFETY: the `f16c` feature was detected at runtime.
            return unsafe { f32_to_f16_f16c(a) };
        }
    }
    a.map(numsim_oplib::cvt::f32_to_fp16_bits)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c,avx")]
unsafe fn f32_to_f16_f16c(a: &[f32; WARP_SIZE]) -> [u16; WARP_SIZE] {
    use std::arch::x86_64::{_mm256_cvtps_ph, _mm256_loadu_ps, _mm_storeu_si128, _MM_FROUND_TO_NEAREST_INT};
    let mut r = [0u16; WARP_SIZE];
    for i in (0..WARP_SIZE).step_by(8) {
        // SAFETY: i + 8 <= 32; the 128-bit store writes 8 u16 lanes.
        unsafe {
            let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(a.as_ptr().add(i)));
            _mm_storeu_si128(r.as_mut_ptr().add(i).cast(), h);
        }
    }
    r
}

/// IEEE binary16 -> binary32 (exact), 32 lanes: identical to
/// `numsim_oplib::cvt::fp16_bits_to_f32` (tested on all 65536 inputs; NaN
/// lanes, which hardware would quiet, take the codec).
#[inline]
pub(crate) fn f16_to_f32(a: &[u16; WARP_SIZE]) -> [f32; WARP_SIZE] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("f16c") {
            // SAFETY: the `f16c` feature was detected at runtime.
            return unsafe { f16_to_f32_f16c(a) };
        }
    }
    a.map(numsim_oplib::cvt::fp16_bits_to_f32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c,avx")]
unsafe fn f16_to_f32_f16c(a: &[u16; WARP_SIZE]) -> [f32; WARP_SIZE] {
    use std::arch::x86_64::{_mm256_cvtph_ps, _mm256_storeu_ps, _mm_loadu_si128};
    let mut r = [0f32; WARP_SIZE];
    for i in (0..WARP_SIZE).step_by(8) {
        // SAFETY: i + 8 <= 32; the 128-bit load reads 8 u16 lanes.
        unsafe {
            let v = _mm256_cvtph_ps(_mm_loadu_si128(a.as_ptr().add(i).cast()));
            _mm256_storeu_ps(r.as_mut_ptr().add(i), v);
        }
    }
    // Hardware quiets signalling NaNs; the codec keeps the payload as is.
    for l in 0..WARP_SIZE {
        if a[l] & 0x7c00 == 0x7c00 && a[l] & 0x03ff != 0 {
            r[l] = numsim_oplib::cvt::fp16_bits_to_f32(a[l]);
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardware_fma_matches_libm_including_nans() {
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let special = [0u32, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0x7fa0_0001, 0xffc1_2345, 0x0000_0001, 0x7f7f_ffff];
        for round in 0..2000 {
            let pick = |x: u64, l: usize| if (l + round) % 5 == 0 { special[(x % 9) as usize] } else { x as u32 };
            let a: [f32; 32] = std::array::from_fn(|l| f32::from_bits(pick(next(), l)));
            let b: [f32; 32] = std::array::from_fn(|l| f32::from_bits(pick(next(), l + 1)));
            let c: [f32; 32] = std::array::from_fn(|l| f32::from_bits(pick(next(), l + 2)));
            let fast = fma_f32(&a, &b, &c).map(f32::to_bits);
            let slow = fma_f32_scalar(&a, &b, &c).map(f32::to_bits);
            assert_eq!(fast, slow, "f32 round {round}");
            let a: [f64; 32] = std::array::from_fn(|_| f64::from_bits(next()));
            let b: [f64; 32] = std::array::from_fn(|l| if l % 7 == 0 { f64::NAN } else { f64::from_bits(next()) });
            let c: [f64; 32] = std::array::from_fn(|_| f64::from_bits(next()));
            assert_eq!(fma_f64(&a, &b, &c).map(f64::to_bits), fma_f64_scalar(&a, &b, &c).map(f64::to_bits));
        }
    }

    #[test]
    fn hardware_f16_decode_matches_the_software_codec_exhaustively() {
        for base in (0..=u16::MAX as u32).step_by(32) {
            let a: [u16; 32] = std::array::from_fn(|l| (base + l as u32) as u16);
            assert_eq!(
                f16_to_f32(&a).map(f32::to_bits),
                a.map(|h| numsim_oplib::cvt::fp16_bits_to_f32(h).to_bits()),
                "{base:#x}"
            );
        }
    }

    #[test]
    fn hardware_f16_encode_matches_the_software_codec() {
        // Every f32 exponent, signs, ties, NaN payloads, subnormal ranges.
        let mut words = Vec::new();
        for exp in 0u32..256 {
            for frac in [0u32, 1, 0x1000, 0x0fff, 0x1001, 0x2000, 0x3000, 0x7f_e000, 0x7f_f000, 0x7f_ffff, 0x40_0000, 0x40_0001, 0x55_5555] {
                for sign in [0u32, 1] {
                    words.push(sign << 31 | exp << 23 | frac);
                }
            }
        }
        for chunk in words.chunks(32) {
            let mut a = [0f32; 32];
            for (i, w) in chunk.iter().enumerate() {
                a[i] = f32::from_bits(*w);
            }
            assert_eq!(f32_to_f16_rne(&a), a.map(numsim_oplib::cvt::f32_to_fp16_bits), "{chunk:x?}");
        }
    }
}
