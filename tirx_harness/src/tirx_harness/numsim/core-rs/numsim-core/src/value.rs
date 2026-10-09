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

pub use numsim_types::value::*;

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
    fn regfile_typed() {
        let mut rf = RegFile::new(2);
        rf.write_as::<f32>(1, &splat(1.5f32), WarpMask::lane(3));
        assert_eq!(rf.get(1)[3], 1.5f32.to_bits() as u64);
        assert_eq!(rf.get(1)[0], 0);
        assert_eq!(rf.read_as::<f32>(1)[3], 1.5);
    }
}
