//! Register-value packing: one lane's value as up to four 64-bit slots, with
//! `Ty::lanes` elements of `elem.bits()` packed densely, element 0 in the low
//! bits (see `crate::value`). Elements may straddle a slot boundary (6-bit
//! formats); B128 elements are always slot-aligned.

use super::super::{OpError, OpResult};
use crate::dtype::{Dtype, Ty};
use crate::value::WarpValue;

/// One lane's value: `slots()` 64-bit words, little-endian.
pub(super) type Packed = [u64; 4];

/// Mask of the low `bits` bits (`bits <= 128`).
#[inline]
pub(super) const fn mask128(bits: u32) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

/// Check that a slot slice can hold a value of `ty`.
#[inline]
pub(super) fn check_slots(slots: usize, ty: Ty, what: &str) -> OpResult {
    if ty.lanes == 0 || ty.bits() > crate::dtype::MAX_VALUE_BITS {
        return Err(OpError::invalid(format!("{what}: malformed type {ty}")));
    }
    if slots < ty.slots() as usize {
        return Err(OpError::invalid(format!(
            "{what}: {ty} needs {} slots, got {slots}",
            ty.slots()
        )));
    }
    Ok(())
}

/// Read `lane`'s value of `ty` from `slots`.
#[inline]
pub(super) fn load(slots: &[WarpValue<u64>], ty: Ty, lane: usize) -> Packed {
    let mut v = [0u64; 4];
    for (i, w) in v.iter_mut().enumerate().take(ty.slots() as usize) {
        *w = slots[i][lane];
    }
    v
}

/// Write `lane`'s value of `ty` into `slots`.
#[inline]
pub(super) fn store(slots: &mut [WarpValue<u64>], ty: Ty, lane: usize, v: &Packed) {
    for (i, w) in v.iter().enumerate().take(ty.slots() as usize) {
        slots[i][lane] = *w;
    }
}

/// Element `idx` (each `bits` wide) of a packed value, zero-extended.
#[inline]
pub(super) fn get(v: &Packed, idx: usize, bits: u32) -> u128 {
    let off = idx * bits as usize;
    let (word, shift) = (off / 64, off % 64);
    let lo = v[word] as u128;
    let hi = if word + 1 < 4 { (v[word + 1] as u128) << 64 } else { 0 };
    ((lo | hi) >> shift) & mask128(bits)
}

/// Set element `idx` (each `bits` wide) of a packed value.
#[inline]
pub(super) fn put(v: &mut Packed, idx: usize, bits: u32, x: u128) {
    let off = idx * bits as usize;
    let (word, shift) = (off / 64, off % 64);
    let two = word + 1 < 4;
    let mut window = v[word] as u128 | if two { (v[word + 1] as u128) << 64 } else { 0 };
    let m = mask128(bits) << shift;
    window = (window & !m) | ((x << shift) & m);
    v[word] = window as u64;
    if two {
        v[word + 1] = (window >> 64) as u64;
    }
}

/// Sign-extend the low `bits` of `x`.
#[inline]
pub(super) fn sext(x: u128, bits: u32) -> i128 {
    let s = 128 - bits;
    ((x << s) as i128) >> s
}

/// Value range `[min, max]` of an integer dtype.
#[inline]
pub(super) fn int_range(d: Dtype) -> (i128, i128) {
    let w = d.bits();
    if d.is_signed_int() {
        (-(1i128 << (w - 1)), (1i128 << (w - 1)) - 1)
    } else {
        (0, (1i128 << w) - 1)
    }
}

/// Integer value of an int dtype's raw bits (sign-extended for signed).
#[inline]
pub(super) fn int_value(d: Dtype, x: u128) -> i128 {
    if d.is_signed_int() {
        sext(x, d.bits())
    } else {
        (x & mask128(d.bits())) as i128
    }
}
