//! `ldmatrix` / `stmatrix` fragment address maps. (The legacy tile-op layer
//! -- tile coordinates, scope partitions, element maps, copy and GEMM operand
//! planning -- was removed at step 5: v2 lowers `tirx.tile.*` through TVM's
//! dispatch, contract decision 6.)
//!
//! Legacy sources: `engine-rs/src/runtime/instructions/tile.rs`,
//! `engine-rs/src/runtime/abi_transport.rs` (element-map types) and
//! `engine-rs/src/runtime/memory_ops.rs` (ldmatrix/stmatrix maps).

pub mod matrix;

pub use matrix::*;

use crate::registry::Binding;

/// Registry bindings for this family: legacy op name -> OpLib function path.
pub(crate) const BINDINGS: &[Binding] = &[
    Binding {
        op: "tirx.ptx.ldmatrix",
        function: "layout::ldmatrix_b16_fragments",
    },
    Binding {
        op: "tirx.ptx.ldmatrix",
        function: "layout::ldmatrix_lane_accesses",
    },
    Binding {
        op: "tirx.ptx.ldmatrix_b8fmt",
        function: "layout::ldmatrix_b8_fragments",
    },
    Binding {
        op: "tirx.ptx.ldmatrix_m16n16_b8",
        function: "layout::ldmatrix_b8_fragments",
    },
    Binding {
        op: "tirx.ptx.ldmatrix_s8_s4",
        function: "layout::ldmatrix_b8_fragments",
    },
    Binding {
        op: "tirx.ptx.stmatrix",
        function: "layout::stmatrix_writes",
    },
    Binding {
        op: "tirx.ptx.stmatrix_m16n8_b8",
        function: "layout::stmatrix_writes",
    },
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
