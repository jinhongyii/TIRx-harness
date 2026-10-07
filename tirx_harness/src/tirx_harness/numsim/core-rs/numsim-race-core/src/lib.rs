//! `numsim-race-core`: the racecheck core's benchmark and history crate.
//!
//! The core itself moved into the contract crate as
//! `numsim_core::racecheck` (phase 3: it implements `observe::Observer`);
//! its scenario tests live in `numsim-core/tests/racecheck_*.rs`. This crate
//! re-exports it and owns the criterion benchmarks that guard the pruning
//! tricks (packed stamps, exact-hit update, chunk join memo, 16-byte
//! witnesses).

pub use numsim_core::racecheck::*;
