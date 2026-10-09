//! Bulk-reduction numerics (`cp.reduce.async.bulk*`, TMA reduce): pure
//! "old value + operand -> new value" kernels.
//!
//! Atomic RMW (`atom`/`red`/`atomicAdd`/CAS) has one implementation, the
//! engine's `interp::handlers::mem::rmw_elem`/`rmw_bytes`; the legacy-ported
//! `rmw` kernels that duplicated it were retired after a differential test
//! (numsim-core `oplib/atomic_differential_tests.rs`) found no disagreement.
//!
//! Legacy sources: `engine-rs/src/runtime/io.rs` (`RawAtomicOperation`,
//! `RawAtomicScalar`, `raw_atomic_*_physical_ptr_warp` update closures),
//! `engine-rs/src/memory.rs` (`DeferredGlobalReduction`), and the op tables in
//! `runtime/instructions/{mem,async_copy}.rs` / `runtime/tensor_map.rs`.

mod reduction;
#[cfg(test)]
mod tests;

pub use reduction::*;

use crate::registry::Binding;

const fn b(op: &'static str, function: &'static str) -> Binding {
    Binding { op, function }
}

/// Registry bindings for this family: legacy op name -> OpLib function path.
pub(crate) const BINDINGS: &[Binding] = &[
    b(
        "tirx.ptx.cp_reduce_async_bulk_s2g",
        "atomic::BulkReduction::apply",
    ),
    b(
        "tirx.ptx.cp_reduce_async_bulk_s2g_f32_noftz",
        "atomic::BulkReduction::apply",
    ),
    b(
        "tirx.ptx.cp_reduce_async_bulk_tensor",
        "atomic::BulkReduction::apply",
    ),
];
