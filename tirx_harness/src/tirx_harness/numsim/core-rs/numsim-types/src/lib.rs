//! Leaf vocabulary types shared by `numsim-core` (which re-exports them as
//! `numsim_core::{dtype, value}`) and `numsim-oplib`. No dependencies beyond
//! serde, so both crates can use them without a cycle.

pub mod dtype;
pub mod value;

pub use dtype::{Dtype, Ty, MAX_VALUE_BITS};
pub use value::{splat, LaneMask, WarpMask, WarpValue, WARP_SIZE};
