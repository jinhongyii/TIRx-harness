//! Warp-wide values and the register model.
//!
//! # Register model (contract)
//!
//! * Registers are declared in `Program::regs` with a static [`crate::Ty`]
//!   (lowering is SSA-like: one register per expression result, one per
//!   named scalar / promoted local element).
//! * Storage: each warp owns a [`RegFile`] of 64-bit *slots*; a slot is a
//!   [`WarpValue<u64>`] (one 64-bit cell per lane). Register `r` occupies
//!   `ty.slots()` (1..=4) consecutive slots starting at
//!   `Program::reg_slot_offsets()[r]`. Values <= 64 bits use one slot.
//! * Encoding inside the slots: the value's raw bits, little-endian across
//!   slots (bits 0..64 in the first slot), zero-extended. Signed integers are
//!   stored as two's complement *of their own width* (not sign-extended).
//!   Floats are raw IEEE bits. Vector lanes are packed densely, element 0 in
//!   the low bits (`float16x2`: elem0 = bits 0..16). Predicates are 0/1.
//! * Addresses are ordinary register values: generic/global pointers are
//!   `U64`, shared (`shared::cta` / `shared::cluster`) and tmem addresses are
//!   `U32`; encodings in [`crate::arena::addr`].
//! * Writes are masked by the warp's active mask: inactive lanes keep their
//!   value (this is how divergent `If` merges named state).

use serde::{Deserialize, Serialize};
use std::fmt;

/// Number of lanes per warp.
pub const WARP_SIZE: usize = 32;

/// One value per lane.
pub type WarpValue<T> = [T; WARP_SIZE];

/// A set of lanes. Bit `i` = lane `i`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WarpMask(pub u32);

/// Alias used in observer/checker vocabulary.
pub type LaneMask = WarpMask;

impl WarpMask {
    pub const NONE: WarpMask = WarpMask(0);
    pub const ALL: WarpMask = WarpMask(u32::MAX);

    /// The first `n` lanes (`n <= 32`), e.g. the live lanes of a partial warp.
    pub const fn first_n(n: u32) -> WarpMask {
        if n >= 32 {
            WarpMask::ALL
        } else {
            WarpMask((1u32 << n) - 1)
        }
    }
    pub const fn lane(i: usize) -> WarpMask {
        WarpMask(1u32 << i)
    }
    pub const fn bits(self) -> u32 {
        self.0
    }
    pub const fn contains(self, lane: usize) -> bool {
        (self.0 >> lane) & 1 == 1
    }
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
    pub const fn is_all(self) -> bool {
        self.0 == u32::MAX
    }
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }
    pub const fn first(self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            Some(self.0.trailing_zeros() as usize)
        }
    }
    pub const fn and(self, o: WarpMask) -> WarpMask {
        WarpMask(self.0 & o.0)
    }
    pub const fn or(self, o: WarpMask) -> WarpMask {
        WarpMask(self.0 | o.0)
    }
    pub const fn and_not(self, o: WarpMask) -> WarpMask {
        WarpMask(self.0 & !o.0)
    }
    pub const fn not(self) -> WarpMask {
        WarpMask(!self.0)
    }
    /// Iterate active lanes in ascending order.
    pub fn lanes(self) -> impl Iterator<Item = usize> {
        let mut m = self.0;
        std::iter::from_fn(move || {
            if m == 0 {
                None
            } else {
                let i = m.trailing_zeros() as usize;
                m &= m - 1;
                Some(i)
            }
        })
    }
    /// Build a mask from a predicate register value (lane bit set iff value != 0).
    pub fn from_pred(v: &WarpValue<u64>) -> WarpMask {
        let mut m = 0u32;
        for (i, x) in v.iter().enumerate() {
            if *x != 0 {
                m |= 1 << i;
            }
        }
        WarpMask(m)
    }
}

impl fmt::Debug for WarpMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WarpMask({:#010x})", self.0)
    }
}

impl fmt::Display for WarpMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#010x}", self.0)
    }
}

/// Splat a value to all lanes.
pub fn splat<T: Copy>(v: T) -> WarpValue<T> {
    [v; WARP_SIZE]
}

/// Per-warp register file (see the module docs for the encoding).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegFile {
    pub regs: Vec<WarpValue<u64>>,
}

impl RegFile {
    pub fn new(count: usize) -> RegFile {
        RegFile {
            regs: vec![[0u64; WARP_SIZE]; count],
        }
    }
    #[inline]
    pub fn get(&self, r: u32) -> &WarpValue<u64> {
        &self.regs[r as usize]
    }
    #[inline]
    pub fn get_mut(&mut self, r: u32) -> &mut WarpValue<u64> {
        &mut self.regs[r as usize]
    }
    /// Read a one-slot value as typed lanes (`r` is a *slot* index).
    #[inline]
    pub fn read_as<T: crate::oplib::Scalar>(&self, r: u32) -> WarpValue<T> {
        let raw = self.get(r);
        std::array::from_fn(|i| T::from_bits(raw[i]))
    }
    /// Write typed lanes under `mask`; inactive lanes keep their old value.
    #[inline]
    pub fn write_as<T: crate::oplib::Scalar>(&mut self, r: u32, v: &WarpValue<T>, mask: WarpMask) {
        let dst = self.get_mut(r);
        for lane in mask.lanes() {
            dst[lane] = v[lane].to_bits();
        }
    }
    /// Write raw lane cells under `mask`.
    #[inline]
    pub fn write_raw(&mut self, r: u32, v: &WarpValue<u64>, mask: WarpMask) {
        let dst = self.get_mut(r);
        for lane in mask.lanes() {
            dst[lane] = v[lane];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mask_lanes() {
        let m = WarpMask(0b1010_0001);
        assert_eq!(m.lanes().collect::<Vec<_>>(), vec![0, 5, 7]);
        assert_eq!(WarpMask::first_n(32), WarpMask::ALL);
        assert_eq!(WarpMask::first_n(3).0, 7);
    }
    #[test]
    fn regfile_typed() {
        let mut rf = RegFile::new(2);
        rf.write_as::<f32>(1, &splat(1.5f32), WarpMask::lane(3));
        assert_eq!(rf.get(1)[3], 1.5f32.to_bits() as u64);
        assert_eq!(rf.get(1)[0], 0);
        assert_eq!(rf.read_as::<f32>(1)[3], 1.5);
    }
}
