//! Conversions: dtype codecs (`formats`), packed-lane cvt forms (`pack`),
//! scalar PTX `cvt` numeric cores (`ptx`), plain runtime-parameter cvt forms
//! (`int`, `float`, `narrow`), and the spelling-level dispatcher (`spelling`).

pub mod formats;
mod pack;
mod ptx;

mod bindings;
mod float;
mod int;
mod narrow;
mod spelling;

/// GPU-recorded cvt golden rows (tests, and the `goldens` feature for
/// contract-boundary replays in `numsim-core`).
#[cfg(any(test, feature = "goldens"))]
pub mod goldens;

pub(crate) use bindings::BINDINGS;
pub use float::*;
pub use formats::*;
pub use int::*;
pub use narrow::*;
pub use pack::*;
pub use ptx::*;
pub use spelling::*;
