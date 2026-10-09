//! NumSim OpLib: engine-independent numerical and addressing kernels.
//!
//! Every function here is pure: values in, values out. No engine state, no
//! locks, no observer. See `registry` for the op table that renders
//! `SUPPORTED_OPS.md`.

pub mod arith;
pub mod atomic;
pub mod codec;
pub mod cvt;
pub mod fpenv;
pub mod layout;
pub mod mma;
pub mod registry;
pub mod scalar;
pub mod tcgen05;
pub mod tma;
pub mod types;
pub mod warp;
