//! tcgen05 (Blackwell tensor-core / TMEM) pure kernels.
//!
//! Legacy sources: `engine-rs/src/runtime/tcgen_ops.rs`,
//! `runtime/tcgen_ops/integer.rs`, `runtime/instructions/tcgen05.rs`,
//! `runtime/tmem.rs` (`get_tmem_addr`, column-count math) and the pure
//! allocation helpers of `engine-rs/src/tcgen.rs`.
//!
//! Submodules:
//! - [`instr_desc`]: instruction-descriptor decoders per kind;
//! - [`smem_desc`]: shared-memory matrix descriptors, swizzled byte offsets,
//!   zero-column mask, tile-GEMM operand layouts;
//! - [`layouts`]: TMEM addresses, ld/st/cp lane maps, dense/banked accumulator
//!   layouts, scale-factor and sparse-metadata locations;
//! - [`narrow`]: narrow-float / cell codecs; [`scale`]: block-scale decode;
//! - [`gather`]: operand gathers over caller-supplied memory closures;
//! - [`mma`]: sparse expansion, sparse/dense tails, tile GEMM compute;
//! - [`integer`]: `kind::i8` / `kind::ti16`; [`ld`]: ld.red / spcompress /
//!   collector state; [`footprints`]: checker address walks; [`tmem`]:
//!   allocation rules.
//!
//! Engine plumbing (lifecycle validation, TMEM/shared views, ordering,
//! register IO) is not here; see each module's notes.

pub mod encode;
pub mod footprints;
pub mod gather;
pub mod instr_desc;
pub mod integer;
pub mod layouts;
pub mod ld;
pub mod mma;
pub mod narrow;
pub mod scale;
pub mod smem_desc;
pub mod tmem;

use crate::registry::Binding;

const fn b(op: &'static str, function: &'static str) -> Binding {
    Binding { op, function }
}

/// Registry bindings for this family: legacy op name -> OpLib function path.
pub(crate) const BINDINGS: &[Binding] = &[
    b("tirx.cuda.tcgen05_encode_instr_descriptor", "tcgen05::encode::encode_dense_instr_descriptor_fields"),
    b("tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled", "tcgen05::encode::encode_block_scaled_instr_descriptor_fields"),
    b("tirx.cuda.tcgen05_encode_matrix_descriptor", "tcgen05::encode::encode_matrix_descriptor"),
    b("tirx.cuda.get_tmem_addr", "tcgen05::layouts::get_tmem_addr"),
    b(
        "tirx.ptx.tcgen05_alloc",
        "tcgen05::tmem::allocation_interval",
    ),
    b("tirx.ptx.tcgen05_alloc", "tcgen05::tmem::validate_columns"),
    b(
        "tirx.ptx.tcgen05_alloc_exclusive",
        "tcgen05::tmem::allocation_interval",
    ),
    b(
        "tirx.ptx.tcgen05_alloc_exclusive",
        "tcgen05::tmem::validate_columns",
    ),
    b(
        "tirx.ptx.tcgen05_dealloc",
        "tcgen05::tmem::validate_columns",
    ),
    b(
        "tirx.ptx.tcgen05_dealloc_exclusive",
        "tcgen05::tmem::validate_columns",
    ),
    // tcgen05.ld / st
    b("tirx.ptx.tcgen05_ld", "tcgen05::layouts::ldst_location"),
    b("tirx.ptx.tcgen05_ld", "tcgen05::ld::ld_destination_count"),
    b(
        "tirx.ptx.tcgen05_ld_split",
        "tcgen05::layouts::ldst_location",
    ),
    b(
        "tirx.ptx.tcgen05_ld_red",
        "tcgen05::ld::LdReduction::reduce",
    ),
    b(
        "tirx.ptx.tcgen05_ld_red_split",
        "tcgen05::ld::LdReduction::reduce",
    ),
    b(
        "tirx.ptx.tcgen05_ld_spcompress",
        "tcgen05::ld::spcompress_lane",
    ),
    b(
        "tirx.ptx.tcgen05_ld_red_spcompress",
        "tcgen05::ld::spcompress_lane",
    ),
    b(
        "tirx.ptx.tcgen05_ld_red_spcompress",
        "tcgen05::ld::LdReduction::reduce",
    ),
    b("tirx.ptx.tcgen05_st", "tcgen05::layouts::ldst_location"),
    b(
        "tirx.ptx.tcgen05_st_split",
        "tcgen05::layouts::ldst_location",
    ),
    // tcgen05.cp
    b(
        "tirx.ptx.tcgen05_cp",
        "tcgen05::smem_desc::decode_matrix_descriptor_for_layout",
    ),
    b("tirx.ptx.tcgen05_cp", "tcgen05::smem_desc::cp_source_span"),
    b(
        "tirx.ptx.tcgen05_cp",
        "tcgen05::layouts::cp_destination_cells",
    ),
    b("tirx.ptx.tcgen05_cp", "tcgen05::layouts::cp_decode_word"),
    // dense / ws / sparse floating MMA
    b(
        "tirx.ptx.tcgen05_mma_ss",
        "tcgen05::instr_desc::FloatKind::instruction",
    ),
    b(
        "tirx.ptx.tcgen05_mma_ss",
        "tcgen05::instr_desc::decode_f8f6f4",
    ),
    b(
        "tirx.ptx.tcgen05_mma_ss",
        "tcgen05::gather::gather_b16_rows_with",
    ),
    b(
        "tirx.ptx.tcgen05_mma_ss",
        "tcgen05::gather::gather_tf32_rows",
    ),
    b("tirx.ptx.tcgen05_mma_ss", "tcgen05::gather::gather_f8_rows"),
    b("tirx.ptx.tcgen05_mma_ss", "tcgen05::mma::mma_dense_tail"),
    b(
        "tirx.ptx.tcgen05_mma_ts",
        "tcgen05::instr_desc::FloatKind::instruction",
    ),
    b(
        "tirx.ptx.tcgen05_mma_ts",
        "tcgen05::gather::gather_packed_tmem_a",
    ),
    b("tirx.ptx.tcgen05_mma_ts", "tcgen05::mma::mma_dense_tail"),
    b(
        "tirx.ptx.tcgen05_mma_ws_ss",
        "tcgen05::smem_desc::ColumnMask::source_column",
    ),
    b("tirx.ptx.tcgen05_mma_ws_ss", "tcgen05::mma::mma_dense_tail"),
    b(
        "tirx.ptx.tcgen05_mma_ws_ts",
        "tcgen05::smem_desc::ColumnMask::source_column",
    ),
    b("tirx.ptx.tcgen05_mma_ws_ts", "tcgen05::mma::mma_dense_tail"),
    b(
        "tirx.ptx.tcgen05_mma_collector_ab_ss",
        "tcgen05::ld::collector_transition",
    ),
    b(
        "tirx.ptx.tcgen05_mma_collector_ab_ts",
        "tcgen05::ld::collector_transition",
    ),
    b(
        "tirx.ptx.tcgen05_mma_ashift_collector_b_ts",
        "tcgen05::ld::collector_transition",
    ),
    b(
        "tirx.ptx.tcgen05_mma_lut_b_ss",
        "tcgen05::gather::gather_lut_b_rows",
    ),
    b(
        "tirx.ptx.tcgen05_mma_lut_b_ts",
        "tcgen05::gather::gather_lut_b_rows",
    ),
    b(
        "tirx.ptx.tcgen05_mma_sp_collector_ab_ss",
        "tcgen05::mma::sparse_float_mma",
    ),
    b(
        "tirx.ptx.tcgen05_mma_sp_collector_ab_ts",
        "tcgen05::mma::sparse_float_mma",
    ),
    b(
        "tirx.ptx.tcgen05_mma_sp_ashift_collector_b_ts",
        "tcgen05::mma::sparse_float_mma",
    ),
    // integer (ti16 / i8)
    b(
        "tirx.ptx.tcgen05_mma_ti16_ss",
        "tcgen05::integer::integer_shape",
    ),
    b(
        "tirx.ptx.tcgen05_mma_ti16_ss",
        "tcgen05::integer::integer_mma_accumulate",
    ),
    b(
        "tirx.ptx.tcgen05_mma_ti16_ts",
        "tcgen05::integer::integer_mma_accumulate",
    ),
    b(
        "tirx.ptx.tcgen05_mma_ti16_ss_collector_b",
        "tcgen05::integer::integer_mma_accumulate",
    ),
    b(
        "tirx.ptx.tcgen05_mma_ti16_ts_collector_b",
        "tcgen05::integer::integer_mma_accumulate",
    ),
    b(
        "tirx.ptx.tcgen05_mma_ws_ti16_ss",
        "tcgen05::integer::integer_mma_accumulate",
    ),
    b(
        "tirx.ptx.tcgen05_mma_ws_ti16_ts",
        "tcgen05::integer::integer_mma_accumulate",
    ),
    b(
        "tirx.ptx.tcgen05_mma_ws_ti16_ss_mask",
        "tcgen05::integer::integer_mma_accumulate",
    ),
    b(
        "tirx.ptx.tcgen05_mma_ws_ti16_ts_mask",
        "tcgen05::integer::integer_mma_accumulate",
    ),
    b(
        "tirx.ptx.tcgen05_mma_sp_ti16_ss",
        "tcgen05::mma::expand_sparse_2of4",
    ),
    b(
        "tirx.ptx.tcgen05_mma_sp_ti16_ts",
        "tcgen05::mma::expand_sparse_2of4",
    ),
    b(
        "tirx.ptx.tcgen05_mma_sp_ti16_ss_collector_b",
        "tcgen05::mma::expand_sparse_2of4",
    ),
    b(
        "tirx.ptx.tcgen05_mma_sp_ti16_ts_collector_b",
        "tcgen05::mma::expand_sparse_2of4",
    ),
    b(
        "tirx.ptx.tcgen05_mma_ws_sp_ti16_ss",
        "tcgen05::mma::expand_sparse_2of4",
    ),
    b(
        "tirx.ptx.tcgen05_mma_ws_sp_ti16_ts",
        "tcgen05::mma::expand_sparse_2of4",
    ),
    b(
        "tirx.ptx.tcgen05_mma_ws_sp_ti16_ss_mask",
        "tcgen05::mma::expand_sparse_2of4",
    ),
    b(
        "tirx.ptx.tcgen05_mma_ws_sp_ti16_ts_mask",
        "tcgen05::mma::expand_sparse_2of4",
    ),
    // block-scaled (mxf4 / mxf4nvf4 / mxf8f6f4)
    b(
        "tirx.ptx.tcgen05_mma_block_scale_ss",
        "tcgen05::instr_desc::decode_mxf4_for_cta_group",
    ),
    b(
        "tirx.ptx.tcgen05_mma_block_scale_ss",
        "tcgen05::instr_desc::decode_mxf8f6f4",
    ),
    b(
        "tirx.ptx.tcgen05_mma_block_scale_ss",
        "tcgen05::gather::gather_mxf4_rows",
    ),
    b(
        "tirx.ptx.tcgen05_mma_block_scale_ss",
        "tcgen05::scale::mxf8_scale_values",
    ),
    b(
        "tirx.ptx.tcgen05_mma_block_scale_ts",
        "tcgen05::gather::gather_scaled_tmem_a_cta",
    ),
    b(
        "tirx.ptx.tcgen05_mma_block_scale_block_ss",
        "tcgen05::instr_desc::decode_mxf4_for_cta_group",
    ),
    b(
        "tirx.ptx.tcgen05_mma_block_scale_block_ts",
        "tcgen05::gather::gather_scaled_tmem_a_cta",
    ),
    b(
        "tirx.ptx.tcgen05_mma_block_scale_collector_ab_ss",
        "tcgen05::gather::gather_mxf4_rows",
    ),
    b(
        "tirx.ptx.tcgen05_mma_block_scale_collector_ab_ts",
        "tcgen05::gather::gather_scaled_tmem_a_cta",
    ),
    b(
        "tirx.ptx.tcgen05_mma_block_scale_lut_b_ss",
        "tcgen05::gather::gather_lut_b_rows",
    ),
    b(
        "tirx.ptx.tcgen05_mma_block_scale_lut_b_ts",
        "tcgen05::gather::gather_lut_b_rows",
    ),
    b(
        "tirx.ptx.tcgen05_mma_sp_block_scale_collector_ab_ss",
        "tcgen05::instr_desc::decode_sparse_mxf4",
    ),
    b(
        "tirx.ptx.tcgen05_mma_sp_block_scale_collector_ab_ss",
        "tcgen05::gather::gather_sparse_mxf4_e8m0_rows",
    ),
    b(
        "tirx.ptx.tcgen05_mma_sp_block_scale_collector_ab_ss",
        "tcgen05::mma::expand_sparse_mxf4_a",
    ),
    b(
        "tirx.ptx.tcgen05_mma_sp_block_scale_collector_ab_ts",
        "tcgen05::mma::sparse_float_mma",
    ),
    // tile-level GEMM (canonical BF16 CTA1 path)
    b("tirx.tile.gemm", "tcgen05::mma::tile_gemm_bf16_f32"),
    b("tirx.tile.gemm_async", "tcgen05::mma::tile_gemm_bf16_f32"),
];
