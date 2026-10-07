//! Python bindings for `numsim-core` (W8). Build with `--features python`.
//!
//! Boundary contract: Python passes a serialized `Module`
//! (`Module::to_bytes`, produced by lowering) plus inputs; Rust returns
//! outputs and a JSON `report::Report`. No engine types cross the boundary.

pub use numsim_core;

#[cfg(feature = "python")]
mod py {
    use pyo3::prelude::*;

    /// Format version of serialized programs this extension accepts.
    #[pyfunction]
    fn program_format_version() -> u32 {
        numsim_core::program::FORMAT_VERSION
    }

    #[pymodule]
    fn numsim_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(program_format_version, m)?)?;
        Ok(())
    }
}
