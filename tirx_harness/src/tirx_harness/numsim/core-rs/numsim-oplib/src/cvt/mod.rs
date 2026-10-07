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

#[cfg(test)]
mod goldens;

pub(crate) use bindings::BINDINGS;
pub use float::*;
pub use formats::*;
pub use int::*;
pub use narrow::*;
pub use pack::*;
pub use ptx::*;
pub use spelling::*;
