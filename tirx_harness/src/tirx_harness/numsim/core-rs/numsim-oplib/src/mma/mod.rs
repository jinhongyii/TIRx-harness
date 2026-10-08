//! Warp-level MMA numerics (`mma.sync`, `mma.sp.sync`) and the dense matmul
//! cores shared with tcgen05.
//!
//! Legacy sources: `engine-rs/src/runtime/matrix_ops.rs`,
//! `engine-rs/src/runtime/instructions/matrix.rs` (ABI variants only; the
//! marker types became the runtime enums in `fragments` / `sparse`),
//! and the generic matmul helpers of `engine-rs/src/runtime/tcgen_ops.rs`.
//!
//! # Accumulation contract
//!
//! **f32 accumulators (every float MMA: mma.sync f16/bf16/tf32/f8, mma.sp,
//! raw tcgen05, typed tile GEMM).** For each output element independently:
//!
//! ```text
//! acc = (input_d present) ? d_in * scale : +0.0      // one binary32 multiply
//! for i in 0..k (increasing): acc = fma_f32(a[i], b[i], acc)   // RNE binary32
//! ```
//!
//! D is the *initial accumulator* of the chain, never a separately rounded
//! addend after the product (`backend::mma_f32_abt_increasing_k`, legacy
//! `tcgen_ops.rs:4634-4694`, comment at 4672-4680; pinned by the ported tests
//! `raw_mma_uses_a_fixed_increasing_k_reduction`,
//! `raw_mma_starts_the_fma_chain_from_scaled_input_d`,
//! `raw_mma_vectorized_columns_are_bitwise_equal_to_scalar_dots`, legacy
//! `tcgen_ops.rs:9046-9115`). `mma.sync` passes `scale = 1.0`
//! (`matrix_ops.rs:249-263`). SIMD only advances independent columns
//! together; it never reassociates a chain (`fpenv::fma_f32_abt_increasing_k`).
//! Operands are decoded exactly to f32 first (f16/bf16/fp8 exact; tf32 via
//! `f32_to_tf32` for mma.sync, `matrix_ops.rs:458-466`). f16 C is widened
//! exactly; an f16 D is the f32 chain result rounded once to f16 RNE
//! (`matrix_ops.rs:331-365`, `pack_f16_registers`). Sparse forms run the same
//! chain over the compressed terms in stored order (`matrix_ops.rs:1351-1374`).
//!
//! **f64** (`mma.sync .f64`): one binary64 FMA chain per element from C in
//! increasing K; `.rn` uses `fpenv::fma_f64_abt_increasing_k`, `.rz/.rm/.rp`
//! use `scalar::fma_f64` per step (`matrix_ops.rs:265-299`).
//!
//! **Integers** (s8/u8/s4/u4/b1, dense and sparse): exact i32 products summed
//! in i64 from C; `.satfinite` clamps to s32, otherwise wraps; `b1.xor` is
//! `popc(a) + sum((1-2a) * b)` (`matrix_ops.rs:532-566`).
//!
//! **The NumPy exception (removed).** Legacy `tile_gemm_bf16_f32_ss_cta1`
//! used `numpy.matmul` in plain runs (delta T1). That tile-GEMM path, and the
//! `MatmulBackend`/`NumpyBackend` choice that served it, went with the tile
//! layer: `tirx.tile.gemm{,_async}` now lowers through TVM dispatch to
//! tcgen05.mma, whose numerics are the increasing-K chain above.
//!
//! ldmatrix/stmatrix/movmatrix lane maps are not in these legacy files
//! (`runtime/memory_ops.rs`, `runtime/instructions/warp.rs`), so they are not
//! here. Block-scaled MMA scale decode/application lives in tcgen05.

pub mod backend;
pub mod fragments;
pub mod sparse;
pub mod sync;

pub use backend::*;
pub use fragments::*;
pub use sparse::*;
pub use sync::*;

use crate::registry::Binding;

const fn bind(op: &'static str, function: &'static str) -> Binding {
    Binding { op, function }
}

/// Registry bindings for this family: legacy op name -> OpLib function path.
pub(crate) const BINDINGS: &[Binding] = &[
    bind("tirx.ptx.mma", "mma::mma_sync_f32_b16"),
    bind("tirx.ptx.mma", "mma::mma_sync_f32_tf32"),
    bind("tirx.ptx.mma", "mma::mma_sync_f32_f8"),
    bind("tirx.ptx.mma", "mma::mma_sync_m8n8k4_f16"),
    bind("tirx.ptx.mma_f16acc", "mma::mma_sync_f16_f16"),
    bind("tirx.ptx.mma_f16acc", "mma::mma_sync_f16_f8"),
    bind("tirx.ptx.mma_f16acc", "mma::mma_sync_m8n8k4_f16"),
    bind("tirx.ptx.mma_f16c_f32d", "mma::mma_sync_m8n8k4_f16"),
    bind("tirx.ptx.mma_f64", "mma::mma_sync_f64"),
    bind("tirx.ptx.mma_int", "mma::mma_sync_packed_integer"),
    bind("tirx.ptx.mma_sp", "mma::mma_sp_sync"),
    bind("tirx.ptx.mma_sp_all", "mma::mma_sp_sync"),
    bind("tirx.ptx.mma_sp_pair", "mma::mma_sp_sync"),
    bind("tirx.ptx.mma_sp_f16acc", "mma::mma_sp_sync"),
    bind("tirx.ptx.mma_sp_f16acc_pair", "mma::mma_sp_sync"),
    bind("tirx.ptx.mma_sp_int_all", "mma::mma_sp_sync"),
    bind("tirx.ptx.mma_sp_int_pair", "mma::mma_sp_sync"),
    bind("tirx.ptx.mma_sp", "mma::sparse_metadata_source_mask"),
    bind("mma.sync", "mma::mma_f32"),
    bind("tirx.tile.gemm_async", "mma::mma_f32_abt_increasing_k"),
];
