//! Tile layout and thread-value maps: logical tile coordinates, execution
//! scope partitions, the pure element-map vocabulary (byte / bit / TMEM
//! locations), copy and GEMM operand planning over element maps, and
//! `ldmatrix`/`stmatrix` fragment address maps.
//!
//! Legacy sources: `engine-rs/src/runtime/instructions/tile.rs`,
//! `engine-rs/src/runtime/abi_transport.rs` (element-map types) and
//! `engine-rs/src/runtime/memory_ops.rs` (ldmatrix/stmatrix maps).

pub mod element;
pub mod matrix;
pub mod plan;
pub mod tile;

pub use element::*;
pub use matrix::*;
pub use plan::*;
pub use tile::*;

use crate::registry::Binding;

/// Registry bindings for this family: legacy op name -> OpLib function path.
pub(crate) const BINDINGS: &[Binding] = &[
    Binding { op: "tirx.tile.copy", function: "layout::mapped_sync_copy_plan" },
    Binding { op: "tirx.tile.copy", function: "layout::coordinates_for" },
    Binding { op: "tirx.tile.copy", function: "layout::scope_lanes" },
    Binding { op: "tirx.tile.copy_async", function: "layout::mapped_copy_plan" },
    Binding { op: "tirx.tile.copy_async", function: "layout::mapped_tcgen_elements" },
    Binding { op: "tirx.tile.copy_async", function: "layout::fast_tmem_f32_m64_load" },
    Binding { op: "tirx.tile.copy_async", function: "layout::canonical_32x32b_tmem_rows" },
    Binding { op: "tirx.tile.gemm", function: "layout::map_register_gemm_matrix" },
    Binding { op: "tirx.tile.gemm_async", function: "layout::map_gemm_matrix" },
    Binding { op: "tirx.tile.gemm_async", function: "layout::map_gemm_scale_matrix" },
    Binding { op: "tirx.tile.gemm_async", function: "layout::storage_location" },
    Binding { op: "tirx.ptx.ldmatrix", function: "layout::ldmatrix_b16_fragments" },
    Binding { op: "tirx.ptx.ldmatrix", function: "layout::ldmatrix_lane_accesses" },
    Binding { op: "tirx.ptx.ldmatrix_b8fmt", function: "layout::ldmatrix_b8_fragments" },
    Binding { op: "tirx.ptx.ldmatrix_m16n16_b8", function: "layout::ldmatrix_b8_fragments" },
    Binding { op: "tirx.ptx.ldmatrix_s8_s4", function: "layout::ldmatrix_b8_fragments" },
    Binding { op: "tirx.ptx.stmatrix", function: "layout::stmatrix_writes" },
    Binding { op: "tirx.ptx.stmatrix_m16n8_b8", function: "layout::stmatrix_writes" },
];

#[cfg(test)]
mod tests {
    #[test]
    fn bindings_name_registered_ops() {
        for binding in super::BINDINGS.iter().chain(crate::tma::BINDINGS) {
            assert!(
                crate::registry::OPS.iter().any(|op| op.name == binding.op),
                "unknown op {}",
                binding.op
            );
        }
    }
}
