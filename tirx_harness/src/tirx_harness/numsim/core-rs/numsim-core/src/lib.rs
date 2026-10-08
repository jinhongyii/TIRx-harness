//! `numsim-core`: the single execution path shared by NumSim, Racecheck and
//! Synccheck (see `docs/development/numsim-redesign.md`).
//!
//! This file is owned by the coordinator. Every module is pre-declared so the
//! parallel workers never need to edit it; see `core-rs/README.md` for which
//! worker owns which module.
//!
//! Contract modules (changes go through the coordinator): [`program`],
//! [`dtype`], [`value`], [`site`], [`observe`], [`arena`] (public API),
//! [`sync`] (types and `step` signatures), [`report`], the
//! [`interp::handlers`] signatures and the [`oplib`] trait/registry types.

#![allow(clippy::too_many_arguments)]

pub mod arena;
pub mod dtype;
pub mod interp;
pub mod observe;
pub mod oplib;
pub mod program;
pub mod racecheck;
pub mod report;
pub mod sched;
pub mod site;
pub mod sync;
pub mod synccheck;
pub mod testutil;
pub mod value;

pub use dtype::{Dtype, Ty};
pub use program::{Instr, Module, Program};
pub use value::{WarpMask, WarpValue, WARP_SIZE};
