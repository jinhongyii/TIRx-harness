//! Atomic RMW and bulk-reduction numerics (pure "old value + operand -> new
//! value" kernels).
//!
//! Legacy sources: `engine-rs/src/runtime/io.rs` (`RawAtomicOperation`,
//! `RawAtomicScalar`, `raw_atomic_*_physical_ptr_warp` update closures),
//! `engine-rs/src/memory.rs` (`DeferredGlobalReduction`), and the op tables in
//! `runtime/instructions/{mem,async_copy}.rs` / `runtime/tensor_map.rs`.

mod reduction;
mod rmw;
#[cfg(test)]
mod tests;

pub use reduction::*;
pub use rmw::*;

use crate::registry::Binding;

const fn b(op: &'static str, function: &'static str) -> Binding {
    Binding { op, function }
}

/// Registry bindings for this family: legacy op name -> OpLib function path.
pub(crate) const BINDINGS: &[Binding] = &[
    b("tirx.cuda.atomic_add", "atomic::atomic_i32"),
    b("tirx.cuda.atomic_add", "atomic::atomic_u32"),
    b("tirx.cuda.atomic_add", "atomic::atomic_u64"),
    b("tirx.cuda.atomic_add", "atomic::atomic_f32"),
    b("tirx.cuda.atomic_add", "atomic::atomic_f64"),
    b("tirx.cuda.atomic_add", "atomic::atomic_add_f16"),
    b("tirx.cuda.atomic_add", "atomic::atomic_add_bf16"),
    b("tirx.cuda.atomic_add", "atomic::atomic_add_f16x2"),
    b("tirx.cuda.atomic_add", "atomic::atomic_add_bf16x2"),
    b("tirx.cuda.atomic_add", "atomic::atomic_add_f32x2"),
    b("tirx.cuda.atomic_add", "atomic::atomic_add_f32x4"),
    b("tirx.cuda.atomic_cas", "atomic::atomic_cas_bytes"),
    b("tirx.cuda.atomic_cas", "atomic::atomic_cas_u64"),
    b("tirx.ptx.atom", "atomic::atomic_i32"),
    b("tirx.ptx.atom", "atomic::atomic_i64"),
    b("tirx.ptx.atom", "atomic::atomic_u32"),
    b("tirx.ptx.atom", "atomic::atomic_u64"),
    b("tirx.ptx.atom", "atomic::atomic_f32"),
    b("tirx.ptx.atom", "atomic::atomic_f64"),
    b("tirx.ptx.atom_bitbucket", "atomic::atomic_u32"),
    b("tirx.ptx.atom_cas", "atomic::atomic_cas_u64"),
    b("tirx.ptx.atom_cas_bitbucket", "atomic::atomic_cas_u64"),
    b("tirx.ptx.atom_exch", "atomic::atomic_u32"),
    b("tirx.ptx.atom_exch", "atomic::atomic_u64"),
    b("tirx.ptx.atom_exch", "atomic::atomic_u64x2"),
    b("tirx.ptx.atom_exch_bitbucket", "atomic::atomic_u64"),
    b("tirx.ptx.atom_f32_noftz", "atomic::atomic_f32"),
    b("tirx.ptx.atom_f32_noftz_bitbucket", "atomic::atomic_f32"),
    b("tirx.ptx.atom_half", "atomic::atomic_half"),
    b("tirx.ptx.atom_half_bitbucket", "atomic::atomic_half"),
    b("tirx.ptx.atom_vec_f32", "atomic::atomic_add_f32x2"),
    b("tirx.ptx.atom_vec_f32", "atomic::atomic_add_f32x4"),
    b(
        "tirx.ptx.atom_vec_f32_bitbucket",
        "atomic::atomic_add_f32x4",
    ),
    b("tirx.ptx.atom_vec_f32_noftz", "atomic::atomic_add_f32x4"),
    b(
        "tirx.ptx.atom_vec_f32_noftz_bitbucket",
        "atomic::atomic_add_f32x4",
    ),
    b("tirx.ptx.atom_vec_half", "atomic::atomic_half_vector"),
    b(
        "tirx.ptx.atom_vec_half_bitbucket",
        "atomic::atomic_half_vector",
    ),
    b("tirx.ptx.red", "atomic::atomic_i32"),
    b("tirx.ptx.red", "atomic::atomic_i64"),
    b("tirx.ptx.red", "atomic::atomic_u32"),
    b("tirx.ptx.red", "atomic::atomic_u64"),
    b("tirx.ptx.red", "atomic::atomic_f32"),
    b("tirx.ptx.red", "atomic::atomic_f64"),
    b("tirx.ptx.red_f32_noftz", "atomic::atomic_f32"),
    b("tirx.ptx.red_half", "atomic::atomic_half"),
    b("tirx.ptx.red_vec_f32", "atomic::atomic_add_f32x4"),
    b("tirx.ptx.red_vec_f32_noftz", "atomic::atomic_add_f32x4"),
    b("tirx.ptx.red_vec_half", "atomic::atomic_half_vector"),
    b("tirx.ptx.red_async", "atomic::atomic_i32"),
    b("tirx.ptx.red_async", "atomic::atomic_u32"),
    b("tirx.ptx.red_async", "atomic::atomic_u64"),
    b("tirx.ptx.red_async_release", "atomic::atomic_u32"),
    b("tirx.ptx.cp_reduce_async_bulk_s2c", "atomic::atomic_u32"),
    b("tirx.ptx.cp_reduce_async_bulk_s2c", "atomic::atomic_i32"),
    b("tirx.ptx.cp_reduce_async_bulk_s2c", "atomic::atomic_u64"),
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
