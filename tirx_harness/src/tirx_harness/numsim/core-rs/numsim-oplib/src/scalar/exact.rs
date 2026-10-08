//! Exact (wide-integer) sums and products used to round f32/f64 results.
#![allow(unused_imports)]
use crate::cvt::formats::*;
use crate::cvt::*;
use crate::scalar::*;
use crate::types::OpError;
use std::cmp::Ordering;
pub(crate) fn next_f32(value: f32, upward: bool) -> f32 {
    let terminal = if upward {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };
    if value.is_nan() || value == terminal {
        return value;
    }
    if value == 0.0 {
        return f32::from_bits(if upward { 1 } else { 0x8000_0001 });
    }
    let bits = value.to_bits();
    let next = if (value > 0.0) == upward {
        bits + 1
    } else {
        bits - 1
    };
    f32::from_bits(next)
}

pub(crate) fn round_f32_from_exact(rounded: f32, exact: f64, mode: F32RoundingMode) -> f32 {
    match mode {
        F32RoundingMode::Nearest => rounded,
        F32RoundingMode::Down if (rounded as f64) > exact => next_f32(rounded, false),
        F32RoundingMode::Up if (rounded as f64) < exact => next_f32(rounded, true),
        F32RoundingMode::Zero if exact > 0.0 && (rounded as f64) > exact => {
            next_f32(rounded, false)
        }
        F32RoundingMode::Zero if exact < 0.0 && (rounded as f64) < exact => next_f32(rounded, true),
        F32RoundingMode::Down | F32RoundingMode::Up | F32RoundingMode::Zero => rounded,
    }
}

// Every finite f32 is an integer multiple of 2^-149. Five words cover the
// largest exact sum: two 24-bit significands shifted by at most 253 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExactF32Magnitude(pub(crate) [u64; 5]);

impl ExactF32Magnitude {
    pub(crate) fn from_f32(value: f32) -> Self {
        debug_assert!(value.is_finite());
        let bits = value.to_bits() & 0x7fff_ffff;
        let exponent = ((bits >> 23) & 0xff) as usize;
        let fraction = bits & 0x007f_ffff;
        let significand = if exponent == 0 {
            fraction
        } else {
            (1 << 23) | fraction
        } as u64;
        let shift = if exponent == 0 { 0 } else { exponent - 1 };
        let word_index = shift / 64;
        let bit_index = shift % 64;
        let mut words = [0_u64; 5];
        words[word_index] = significand << bit_index;
        if bit_index != 0 {
            words[word_index + 1] = significand >> (64 - bit_index);
        }
        Self(words)
    }

    pub(crate) fn is_zero(self) -> bool {
        self.0.iter().all(|word| *word == 0)
    }

    pub(crate) fn compare(self, rhs: Self) -> Ordering {
        for index in (0..self.0.len()).rev() {
            match self.0[index].cmp(&rhs.0[index]) {
                Ordering::Equal => {}
                ordering => return ordering,
            }
        }
        Ordering::Equal
    }

    pub(crate) fn add(self, rhs: Self) -> Self {
        let mut words = [0_u64; 5];
        let mut carry = 0_u128;
        for (index, word) in words.iter_mut().enumerate() {
            let sum = self.0[index] as u128 + rhs.0[index] as u128 + carry;
            *word = sum as u64;
            carry = sum >> 64;
        }
        debug_assert_eq!(carry, 0);
        Self(words)
    }

    pub(crate) fn subtract(self, rhs: Self) -> Self {
        debug_assert!(self.compare(rhs) != Ordering::Less);
        let mut words = [0_u64; 5];
        let mut borrow = false;
        for (index, word) in words.iter_mut().enumerate() {
            let (difference, rhs_borrow) = self.0[index].overflowing_sub(rhs.0[index]);
            let (difference, carry_borrow) = difference.overflowing_sub(u64::from(borrow));
            *word = difference;
            borrow = rhs_borrow || carry_borrow;
        }
        debug_assert!(!borrow);
        Self(words)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExactF32Value {
    pub(crate) negative: bool,
    pub(crate) magnitude: ExactF32Magnitude,
}

pub(crate) const EXACT_FMA_WORDS: usize = 10;
// Binary64 products are multiples of 2^-2148 and smaller than 2^2048.
// The same fixed-word arithmetic therefore needs ceil(4196 / 64) words.
pub(crate) const EXACT_F64_FMA_WORDS: usize = 66;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExactFmaMagnitude<const WORDS: usize = EXACT_FMA_WORDS>(pub(crate) [u64; WORDS]);

impl<const WORDS: usize> ExactFmaMagnitude<WORDS> {
    pub(crate) fn from_significand(significand: u64, shift: usize) -> Self {
        let mut words = [0_u64; WORDS];
        if significand == 0 {
            return Self(words);
        }
        let word_index = shift / 64;
        let bit_index = shift % 64;
        debug_assert!(word_index < WORDS);
        words[word_index] = significand << bit_index;
        if bit_index != 0 {
            debug_assert!(word_index + 1 < WORDS);
            words[word_index + 1] = significand >> (64 - bit_index);
        }
        Self(words)
    }
}

impl ExactFmaMagnitude {
    pub(crate) fn f32_parts(value: f32) -> (u64, usize) {
        debug_assert!(value.is_finite());
        let bits = value.to_bits() & 0x7fff_ffff;
        let exponent = ((bits >> 23) & 0xff) as usize;
        let fraction = bits & 0x007f_ffff;
        let significand = if exponent == 0 {
            fraction
        } else {
            (1 << 23) | fraction
        } as u64;
        let shift = if exponent == 0 { 0 } else { exponent - 1 };
        (significand, shift)
    }

    pub(crate) fn from_f32_addend(value: f32) -> Self {
        let (significand, shift) = Self::f32_parts(value);
        Self::from_significand(significand, shift + 149)
    }

    pub(crate) fn from_f32_product(lhs: f32, rhs: f32) -> Self {
        let (lhs_significand, lhs_shift) = Self::f32_parts(lhs);
        let (rhs_significand, rhs_shift) = Self::f32_parts(rhs);
        Self::from_significand(lhs_significand * rhs_significand, lhs_shift + rhs_shift)
    }
}

impl<const WORDS: usize> ExactFmaMagnitude<WORDS> {
    pub(crate) fn is_zero(self) -> bool {
        self.0.iter().all(|word| *word == 0)
    }

    pub(crate) fn compare(self, rhs: Self) -> Ordering {
        for index in (0..WORDS).rev() {
            match self.0[index].cmp(&rhs.0[index]) {
                Ordering::Equal => {}
                ordering => return ordering,
            }
        }
        Ordering::Equal
    }

    pub(crate) fn add(self, rhs: Self) -> Self {
        let mut words = [0_u64; WORDS];
        let mut carry = 0_u128;
        for (index, word) in words.iter_mut().enumerate() {
            let sum = self.0[index] as u128 + rhs.0[index] as u128 + carry;
            *word = sum as u64;
            carry = sum >> 64;
        }
        debug_assert_eq!(carry, 0);
        Self(words)
    }

    pub(crate) fn subtract(self, rhs: Self) -> Self {
        debug_assert!(self.compare(rhs) != Ordering::Less);
        let mut words = [0_u64; WORDS];
        let mut borrow = false;
        for (index, word) in words.iter_mut().enumerate() {
            let (difference, rhs_borrow) = self.0[index].overflowing_sub(rhs.0[index]);
            let (difference, carry_borrow) = difference.overflowing_sub(u64::from(borrow));
            *word = difference;
            borrow = rhs_borrow || carry_borrow;
        }
        debug_assert!(!borrow);
        Self(words)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExactFmaValue<const WORDS: usize = EXACT_FMA_WORDS> {
    pub(crate) negative: bool,
    pub(crate) magnitude: ExactFmaMagnitude<WORDS>,
}

impl ExactFmaValue {
    pub(crate) fn product(lhs: f32, rhs: f32) -> Self {
        Self {
            negative: lhs.is_sign_negative() != rhs.is_sign_negative(),
            magnitude: ExactFmaMagnitude::from_f32_product(lhs, rhs),
        }
    }

    pub(crate) fn addend(value: f32) -> Self {
        Self {
            negative: value.is_sign_negative(),
            magnitude: ExactFmaMagnitude::from_f32_addend(value),
        }
    }
}

impl<const WORDS: usize> ExactFmaValue<WORDS> {
    pub(crate) fn is_zero(self) -> bool {
        self.magnitude.is_zero()
    }

    pub(crate) fn compare(self, rhs: Self) -> Ordering {
        let lhs_negative = self.negative && !self.is_zero();
        let rhs_negative = rhs.negative && !rhs.is_zero();
        if lhs_negative != rhs_negative {
            return if lhs_negative {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        }
        let magnitude_order = self.magnitude.compare(rhs.magnitude);
        if lhs_negative {
            magnitude_order.reverse()
        } else {
            magnitude_order
        }
    }
}

impl ExactFmaValue<EXACT_F64_FMA_WORDS> {
    pub(crate) fn product_f64(lhs: f64, rhs: f64) -> Self {
        let magnitude = if lhs == 0.0 || rhs == 0.0 {
            ExactFmaMagnitude([0; EXACT_F64_FMA_WORDS])
        } else {
            let (a, a_exponent) = positive_f64_dyadic(lhs.abs());
            let (b, b_exponent) = positive_f64_dyadic(rhs.abs());
            let product = a * b;
            let shift = (a_exponent + b_exponent + 2148) as usize;
            ExactFmaMagnitude::from_significand(product as u64, shift).add(
                ExactFmaMagnitude::from_significand((product >> 64) as u64, shift + 64),
            )
        };
        Self {
            negative: lhs.is_sign_negative() != rhs.is_sign_negative(),
            magnitude,
        }
    }

    pub(crate) fn addend_f64(value: f64) -> Self {
        let magnitude = if value == 0.0 {
            ExactFmaMagnitude([0; EXACT_F64_FMA_WORDS])
        } else {
            let (significand, exponent) = positive_f64_dyadic(value.abs());
            ExactFmaMagnitude::from_significand(significand as u64, (exponent + 2148) as usize)
        };
        Self {
            negative: value.is_sign_negative(),
            magnitude,
        }
    }
}

impl ExactF32Value {
    pub(crate) fn from_f32(value: f32) -> Self {
        Self {
            negative: value.is_sign_negative(),
            magnitude: ExactF32Magnitude::from_f32(value),
        }
    }

    pub(crate) fn is_zero(self) -> bool {
        self.magnitude.is_zero()
    }
}

pub(crate) fn exact_f32_sum(lhs: ExactF32Value, rhs: ExactF32Value) -> ExactF32Value {
    if lhs.is_zero() {
        return rhs;
    }
    if rhs.is_zero() {
        return lhs;
    }
    if lhs.negative == rhs.negative {
        return ExactF32Value {
            negative: lhs.negative,
            magnitude: lhs.magnitude.add(rhs.magnitude),
        };
    }
    match lhs.magnitude.compare(rhs.magnitude) {
        Ordering::Greater => ExactF32Value {
            negative: lhs.negative,
            magnitude: lhs.magnitude.subtract(rhs.magnitude),
        },
        Ordering::Less => ExactF32Value {
            negative: rhs.negative,
            magnitude: rhs.magnitude.subtract(lhs.magnitude),
        },
        Ordering::Equal => ExactF32Value {
            negative: false,
            magnitude: ExactF32Magnitude([0; 5]),
        },
    }
}

pub(crate) fn exact_fma_sum<const WORDS: usize>(
    lhs: ExactFmaValue<WORDS>,
    rhs: ExactFmaValue<WORDS>,
) -> ExactFmaValue<WORDS> {
    if lhs.is_zero() {
        return rhs;
    }
    if rhs.is_zero() {
        return lhs;
    }
    if lhs.negative == rhs.negative {
        return ExactFmaValue {
            negative: lhs.negative,
            magnitude: lhs.magnitude.add(rhs.magnitude),
        };
    }
    match lhs.magnitude.compare(rhs.magnitude) {
        Ordering::Greater => ExactFmaValue {
            negative: lhs.negative,
            magnitude: lhs.magnitude.subtract(rhs.magnitude),
        },
        Ordering::Less => ExactFmaValue {
            negative: rhs.negative,
            magnitude: rhs.magnitude.subtract(lhs.magnitude),
        },
        Ordering::Equal => ExactFmaValue {
            negative: false,
            magnitude: ExactFmaMagnitude([0; WORDS]),
        },
    }
}
