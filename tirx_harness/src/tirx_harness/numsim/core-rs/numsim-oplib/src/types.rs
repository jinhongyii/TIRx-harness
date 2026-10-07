//! Vocabulary types for OpLib.
//!
//! `WarpValue`, `WarpMask`, `WARP_SIZE` and `Dtype`/`Ty` are the contract
//! types from the leaf crate `numsim-types` (re-exported by `numsim-core`).
//! OpLib keeps only its own error type and a few mask conveniences.

use std::error::Error;
use std::fmt;

pub use numsim_types::{Dtype, Ty, WarpMask, WarpValue, WARP_SIZE};

/// Mask helpers OpLib code used before the contract swap.
pub trait WarpMaskExt: Sized {
    fn from_predicate(predicate: impl FnMut(usize) -> bool) -> Self;
}

impl WarpMaskExt for WarpMask {
    fn from_predicate(mut predicate: impl FnMut(usize) -> bool) -> Self {
        let mut bits = 0_u32;
        for lane in 0..WARP_SIZE {
            if predicate(lane) {
                bits |= 1 << lane;
            }
        }
        WarpMask(bits)
    }
}

/// Error returned by an OpLib routine whose operands fall outside its domain.
///
/// Mirrors the legacy `EngineError::message` text so diagnostics stay stable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpError(pub String);

impl OpError {
    pub fn message(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for OpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for OpError {}

pub type OpResult<T> = Result<T, OpError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_iterates_active_lanes_ascending() {
        let mask = WarpMask(0x8000_0005);
        assert_eq!(mask.lanes().collect::<Vec<_>>(), [0, 2, 31]);
        assert_eq!(mask.first(), Some(0));
        assert_eq!(mask.count(), 3);
        assert!(WarpMask::from_predicate(|lane| lane < 32).is_all());
    }
}
