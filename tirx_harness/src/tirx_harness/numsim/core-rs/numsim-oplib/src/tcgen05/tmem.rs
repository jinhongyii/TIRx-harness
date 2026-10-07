//! TMEM allocation rules (pure part of the tcgen05 lifecycle).
//!
//! Legacy source: `engine-rs/src/tcgen.rs` (`TMEM_COLUMN_CAPACITY`,
//! `TMEM_ALLOCATION_GRANULARITY`, `TcgenAllocation`, `valid_columns`,
//! `validate_columns` message, `allocation_interval`). The lifecycle hub
//! (collectives, futures, snapshots) stays in the engine.

use crate::types::{OpError, OpResult};

pub const TMEM_COLUMN_CAPACITY: usize = 512;
pub const TMEM_ALLOCATION_GRANULARITY: usize = 32;
/// TMEM lanes per CTA.
pub const TMEM_LANES: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TcgenAllocation {
    pub base_column: u32,
    pub columns: usize,
}

impl TcgenAllocation {
    pub const fn base_column(self) -> u32 {
        self.base_column
    }
    pub const fn columns(self) -> usize {
        self.columns
    }
}

/// The qualifier changes legal operand sizes, not allocation ownership.
pub fn valid_columns(columns: usize, exclusive: bool, capacity: usize) -> bool {
    (32..=capacity).contains(&columns)
        && if exclusive {
            columns % TMEM_ALLOCATION_GRANULARITY == 0
        } else {
            columns <= TMEM_COLUMN_CAPACITY && columns.is_power_of_two()
        }
}

/// `valid_columns` with the legacy diagnostic text (legacy wraps it in a
/// `TcgenLifecycleError` of kind `InvalidColumns` with an occurrence key).
pub fn validate_columns(columns: usize, exclusive: bool, capacity: usize) -> OpResult<()> {
    if valid_columns(columns, exclusive, capacity) {
        return Ok(());
    }
    let rule = if exclusive {
        "multiple of 32"
    } else {
        "power of two"
    };
    let capacity = if exclusive {
        capacity
    } else {
        capacity.min(TMEM_COLUMN_CAPACITY)
    };
    Err(OpError::message(format!(
        "tcgen05 column count must be a {rule} in 32..={capacity}, got {columns}"
    )))
}

/// First-fit 32-column-aligned free interval; SM placement is not modeled.
pub fn allocation_interval(
    columns: usize,
    capacity: usize,
    allocations: impl Iterator<Item = TcgenAllocation> + Clone,
) -> Option<TcgenAllocation> {
    if !valid_columns(columns, true, capacity) {
        return None;
    }
    (0..=capacity - columns)
        .step_by(TMEM_ALLOCATION_GRANULARITY)
        .find(|&base| {
            allocations.clone().all(|allocation| {
                base + columns <= allocation.base_column as usize
                    || base >= allocation.base_column as usize + allocation.columns
            })
        })
        .map(|base| TcgenAllocation {
            base_column: base as u32,
            columns,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_rules_and_first_fit() {
        assert!(valid_columns(32, false, 512));
        assert!(!valid_columns(96, false, 512));
        assert!(valid_columns(96, true, 512));
        assert!(!valid_columns(16, true, 512));
        assert!(validate_columns(48, false, 512)
            .unwrap_err()
            .to_string()
            .contains("power of two in 32..=512, got 48"));
        let live = [TcgenAllocation {
            base_column: 0,
            columns: 64,
        }];
        assert_eq!(
            allocation_interval(32, 512, live.iter().copied()),
            Some(TcgenAllocation {
                base_column: 64,
                columns: 32
            })
        );
        let full = [TcgenAllocation {
            base_column: 0,
            columns: 512,
        }];
        assert_eq!(allocation_interval(32, 512, full.iter().copied()), None);
    }
}
