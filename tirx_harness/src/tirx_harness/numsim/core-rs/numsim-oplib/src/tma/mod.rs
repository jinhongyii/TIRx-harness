//! TMA (`cp.async.bulk[.tensor]`) addressing: TensorMap descriptor image
//! encode/decode/validation, swizzle, tiled/im2col/gather4 box iteration with
//! OOB handling, and bulk-copy operand rules.
//!
//! Legacy sources: `engine-rs/src/runtime/tensor_map.rs`,
//! `engine-rs/src/runtime/tensor_map_registry.rs` (descriptor address rule
//! only), `engine-rs/src/runtime/memory_ops.rs`, `engine-rs/src/runtime/io.rs`
//! and `engine-rs/src/runtime/instructions/async_copy.rs` (pure helpers).
//!
//! Everything operates on plain values and byte slices. The engine keeps
//! memory resolution, footprints, mbarrier completion, registries and the
//! async pipeline; it calls `plan_*` to obtain byte runs and the `execute_*`
//! / `materialize_*` / `apply_*` helpers (or its own memory backends) to
//! move bytes.

pub mod bulk;
pub mod descriptor;
pub mod im2col;
pub mod swizzle;
pub mod tensor_map;
pub mod tiled;

pub use bulk::*;
pub use descriptor::*;
pub use im2col::*;
pub use swizzle::*;
pub use tensor_map::*;
pub use tiled::*;

use crate::registry::Binding;

macro_rules! bind {
    ($function:literal: $($op:literal),+ $(,)?) => {
        [$(Binding { op: $op, function: $function }),+]
    };
}

const TILED_G2S: &[Binding] = &bind!("tma::plan_tiled_g2s":
    "tirx.ptx.cp_async_bulk_tensor_g2s_cta",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cta_report",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_address",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_global_dim_b8",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_global_dim_b16",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_global_dim_stride_b8",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_global_dim_stride_b16",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_report",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_multicast16",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_multicast32",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_address",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_global_dim_b8",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_global_dim_b16",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_global_dim_stride_b8",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_global_dim_stride_b16",
    "tirx.ptx.cp_async_bulk_tensor_prefetch",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_evict_last",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_address",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_address_evict_last",
    "tirx.tile.copy_async",
);

const IM2COL_G2S: &[Binding] = &bind!("tma::plan_im2col_g2s":
    "tirx.ptx.cp_async_bulk_tensor_g2s_cta_im2col",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_address_im2col",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_im2col",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_address_im2col",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_im2col",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_im2col_evict_last",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_address_im2col",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_address_im2col_evict_last",
);

const TILED_S2G: &[Binding] = &bind!("tma::plan_tiled_s2g":
    "tirx.ptx.cp_async_bulk_tensor_s2g",
    "tirx.ptx.cp_async_bulk_tensor_s2g_override_address",
    "tirx.ptx.cp_async_bulk_tensor_s2g_override_global_dim_b8",
    "tirx.ptx.cp_async_bulk_tensor_s2g_override_global_dim_b16",
    "tirx.ptx.cp_async_bulk_tensor_s2g_override_global_dim_stride_b8",
    "tirx.ptx.cp_async_bulk_tensor_s2g_override_global_dim_stride_b16",
    "tirx.ptx.cp_reduce_async_bulk_tensor",
    "tirx.ptx.cp_reduce_async_bulk_tensor_override_address",
    "tirx.ptx.cp_reduce_async_bulk_tensor_override_global_dim_b8",
    "tirx.ptx.cp_reduce_async_bulk_tensor_override_global_dim_b16",
    "tirx.ptx.cp_reduce_async_bulk_tensor_override_global_dim_stride_b8",
    "tirx.ptx.cp_reduce_async_bulk_tensor_override_global_dim_stride_b16",
    "tirx.tile.copy_async",
);

const IM2COL_S2G: &[Binding] = &bind!("tma::plan_im2col_s2g":
    "tirx.ptx.cp_async_bulk_tensor_s2g_im2col_no_offs_w",
    "tirx.ptx.cp_async_bulk_tensor_s2g_override_address_im2col_no_offs_w",
    "tirx.ptx.cp_reduce_async_bulk_tensor_im2col_no_offs_w",
    "tirx.ptx.cp_reduce_async_bulk_tensor_override_address_im2col_no_offs_w",
);

const OVERRIDES: &[Binding] = &bind!("tma::TensorMapImage::apply_overrides":
    "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_global_dim_b8",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_global_dim_b16",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_global_dim_stride_b8",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_global_dim_stride_b16",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_global_dim_b8",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_global_dim_b16",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_global_dim_stride_b8",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_global_dim_stride_b16",
    "tirx.ptx.cp_async_bulk_tensor_s2g_override_global_dim_b8",
    "tirx.ptx.cp_async_bulk_tensor_s2g_override_global_dim_b16",
    "tirx.ptx.cp_async_bulk_tensor_s2g_override_global_dim_stride_b8",
    "tirx.ptx.cp_async_bulk_tensor_s2g_override_global_dim_stride_b16",
    "tirx.ptx.cp_reduce_async_bulk_tensor_override_global_dim_b8",
    "tirx.ptx.cp_reduce_async_bulk_tensor_override_global_dim_b16",
    "tirx.ptx.cp_reduce_async_bulk_tensor_override_global_dim_stride_b8",
    "tirx.ptx.cp_reduce_async_bulk_tensor_override_global_dim_stride_b16",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_global_dim_b8",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_global_dim_b16",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_global_dim_stride_b8",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_global_dim_stride_b16",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_global_dim_evict_last_b8",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_global_dim_evict_last_b16",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_global_dim_stride_evict_last_b8",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_global_dim_stride_evict_last_b16",
    "tirx.ptx.applypriority_async_bulk_tensor_override_global_dim_b8",
    "tirx.ptx.applypriority_async_bulk_tensor_override_global_dim_b16",
    "tirx.ptx.applypriority_async_bulk_tensor_override_global_dim_stride_b8",
    "tirx.ptx.applypriority_async_bulk_tensor_override_global_dim_stride_b16",
);

const OVERRIDE_ADDRESS: &[Binding] = &bind!("tma::validate_override_address":
    "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_address",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_address",
    "tirx.ptx.cp_async_bulk_tensor_s2g_override_address",
    "tirx.ptx.cp_reduce_async_bulk_tensor_override_address",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_address",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_address_evict_last",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_address_im2col",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_address_im2col_evict_last",
    "tirx.ptx.applypriority_async_bulk_tensor_override_address",
    "tirx.ptx.applypriority_async_bulk_tensor_override_address_im2col",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_address_im2col",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_address_im2col",
    "tirx.ptx.cp_async_bulk_tensor_s2g_override_address_im2col_no_offs_w",
    "tirx.ptx.cp_reduce_async_bulk_tensor_override_address_im2col_no_offs_w",
);

const REPLACE: &[Binding] = &bind!("tma::TensorMapImage::replace_field":
    "tirx.ptx.tensormap_replace_dim",
    "tirx.ptx.tensormap_replace_stride",
    "tirx.ptx.tensormap_replace_elemtype",
    "tirx.ptx.tensormap_replace_fill_mode",
    "tirx.ptx.tensormap_replace_interleave_layout",
    "tirx.ptx.tensormap_replace_rank",
    "tirx.ptx.tensormap_replace_swizzle_mode",
    "tirx.ptx.tensormap_replace_swizzle_mode_sm103a",
    "tirx.ptx.tensormap_replace_address",
);

const MISC: &[Binding] = &[
    Binding { op: "tirx.ptx.cp_reduce_async_bulk_tensor", function: "tma::RawTmaReductionOp::resolve" },
    Binding { op: "tirx.ptx.cp_reduce_async_bulk_tensor", function: "tma::s2g_reduction_elements" },
    Binding { op: "tirx.ptx.cp_async_bulk_tensor_s2g", function: "tma::execute_s2g_copy" },
    Binding { op: "tirx.ptx.cp_async_bulk_tensor_g2s_cta", function: "tma::execute_g2s" },
    Binding { op: "tirx.ptx.cp_async_bulk_tensor_g2s_cta", function: "tma::materialize_g2s_payload" },
    Binding { op: "tirx.ptx.cp_async_bulk_tensor_g2s_cta", function: "tma::shared_byte_offset" },
    Binding { op: "tirx.ptx.cp_async_bulk_tensor_g2s_cta_report", function: "tma::copy_report_matches_runs" },
    Binding { op: "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_multicast16", function: "tma::multicast_target_ctas" },
    Binding { op: "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_multicast32", function: "tma::multicast_target_ctas" },
    Binding { op: "tirx.ptx.cp_async_bulk_g2s_cluster_multicast16", function: "tma::multicast_target_ctas" },
    Binding { op: "tirx.ptx.cp_async_bulk_g2s_cluster_multicast32", function: "tma::multicast_target_ctas" },
    Binding { op: "tirx.ptx.cp_async_bulk_g2s_cta", function: "tma::bulk_byte_len" },
    Binding { op: "tirx.ptx.cp_async_bulk_g2s_cluster", function: "tma::bulk_byte_len" },
    Binding { op: "tirx.ptx.cp_async_bulk_s2g", function: "tma::bulk_issue_byte_len" },
    Binding { op: "tirx.ptx.cp_async_bulk_s2c", function: "tma::bulk_issue_byte_len" },
    Binding { op: "tirx.ptx.cp_reduce_async_bulk_s2g", function: "tma::bulk_issue_byte_len" },
    Binding { op: "tirx.ptx.cp_async_bulk_g2s_cta_report", function: "tma::copy_report_matches" },
    Binding { op: "tirx.ptx.cp_async_bulk_g2s_cluster_report", function: "tma::copy_report_matches" },
    Binding { op: "tirx.ptx.cp_async_bulk_s2g", function: "tma::masked_destination_runs" },
    Binding { op: "tirx.ptx.cp_async_bulk_g2s_cta", function: "tma::ignore_oob_window" },
    Binding { op: "tirx.ptx.st_bulk", function: "tma::st_bulk_byte_count" },
    Binding { op: "tirx.ptx.prefetch_valid_addr", function: "tma::validate_cache_hint_alignment" },
    Binding { op: "tirx.ptx.applypriority_async_bulk", function: "tma::validate_bulk_cache_hint_size" },
    Binding { op: "tirx.ptx.cp_async_bulk_prefetch", function: "tma::validate_bulk_cache_hint_size" },
    Binding { op: "tirx.ptx.cp_async_bulk_prefetch_evict_last", function: "tma::validate_bulk_cache_hint_size" },
    Binding { op: "tirx.ptx.tensormap_cp_fenceproxy", function: "tma::validate_descriptor_address" },
    Binding { op: "tirx.ptx.fence_proxy_tensormap_acquire", function: "tma::validate_descriptor_address" },
    Binding { op: "tirx.ptx.applypriority_async_bulk_tensor", function: "tma::TensorMapImage::decode_descriptor" },
    Binding { op: "tirx.ptx.applypriority_async_bulk_tensor_im2col", function: "tma::TensorMapImage::decode_descriptor" },
    Binding { op: "tirx.ptx.cp_async_bulk_tensor_g2s_cta", function: "tma::TensorMapImage::materialize" },
];

const fn concat<const N: usize>(parts: &[&[Binding]]) -> [Binding; N] {
    let mut out = [Binding { op: "", function: "" }; N];
    let mut index = 0;
    let mut part = 0;
    while part < parts.len() {
        let mut item = 0;
        while item < parts[part].len() {
            out[index] = parts[part][item];
            index += 1;
            item += 1;
        }
        part += 1;
    }
    assert!(index == N, "BINDINGS length mismatch");
    out
}

const ALL_LEN: usize = TILED_G2S.len()
    + IM2COL_G2S.len()
    + TILED_S2G.len()
    + IM2COL_S2G.len()
    + OVERRIDES.len()
    + OVERRIDE_ADDRESS.len()
    + REPLACE.len()
    + MISC.len();

const ALL: [Binding; ALL_LEN] = concat::<ALL_LEN>(&[
    TILED_G2S,
    IM2COL_G2S,
    TILED_S2G,
    IM2COL_S2G,
    OVERRIDES,
    OVERRIDE_ADDRESS,
    REPLACE,
    MISC,
]);

/// Registry bindings for this family: legacy op name -> OpLib function path.
pub(crate) const BINDINGS: &[Binding] = &ALL;
