//! Codegen backend (W7): a *printer* from `Program` to Rust source in which
//! every instruction is a call to the same `interp::handlers` function the
//! interpreter uses, with register numbers, dtypes, spaces and other fields
//! as constants. No semantics may appear in generated code (plan 2.3).
//!
//! Shape of the generated crate: a `cdylib` depending on `numsim-core`
//! (same version + toolchain as the host), exporting one
//! `interp::WarpStepFn` per kernel under the symbol
//! `{STEP_SYMBOL_PREFIX}{kernel_index}`. The function is a resumable
//! `loop { match ctx.warp.pc.0 { ... } }` that applies `Flow` exactly like
//! `interp::step_warp` (including the quantum count).

use crate::program::Module;
use std::path::{Path, PathBuf};

/// Exported symbol prefix of per-kernel step functions.
pub const STEP_SYMBOL_PREFIX: &str = "numsim_step_kernel_";

#[derive(Clone, Debug, Default)]
pub struct CodegenOptions {
    /// Emit `ctx.counters` / profiling hooks.
    pub profile: bool,
}

/// Print the Rust source (a single `lib.rs`) for `module`.
pub fn print_module(module: &Module, opts: &CodegenOptions) -> String {
    let _ = (module, opts);
    unimplemented!("W7: codegen::print_module")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildError(pub String);

/// Build the printed source into a cdylib under `out_dir` (one rustc
/// invocation) and return its path.
pub fn build(source: &str, out_dir: &Path) -> Result<PathBuf, BuildError> {
    let _ = (source, out_dir);
    unimplemented!("W7: codegen::build")
}
