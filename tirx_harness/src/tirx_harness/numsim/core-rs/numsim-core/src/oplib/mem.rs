//! Pure address/fragment maps for the data-movement instructions whose
//! layouts live in `numsim-oplib`: `tcgen05.ld` / `tcgen05.st` (lane and
//! register -> TMEM cell), `tcgen05.ld.red` / `.spcompress` numerics,
//! `tcgen05.cp` (shared descriptor -> source spans, destination cells,
//! multicast lanes, b4/b6 decompression), and `ldmatrix` / `stmatrix`
//! fragments beyond `m8n8.b16`.
//!
//! Legacy sources: `engine-rs/src/runtime/tcgen_ops.rs`
//! (`raw_tcgen05_ldst_location`, `raw_tcgen05_{read,ld,st}_register`,
//! `raw_tcgen05_cp{,_footprints,_source_span,_destination_lanes,_decode_word}`),
//! `engine-rs/src/runtime/instructions/tcgen05.rs` (`execute_ld_entry`,
//! `execute_st_entry`, `execute_cp_entry`), `engine-rs/src/runtime/
//! memory_ops.rs` (`raw_ldmatrix_b{16,8}_fragments`, `raw_stmatrix`) and
//! `engine-rs/src/runtime/instructions/mem.rs` (`Ldmatrix<COUNT, TRANSPOSE,
//! SOURCE_BITS, SIGNED>`, `Stmatrix*` variants) with the frontend's form
//! mapping (`frontend-rs/src/emit/{raw_memory,memory_ops}.rs`).
//!
//! Engine concerns (full-warp/issue-lane checks, TMEM allocation/lifecycle
//! validation, byte validity, memory resolution, footprint emission) stay
//! with the caller.

mod matrix;
mod tcgen;

pub use matrix::{
    ldmatrix_fragments, ldmatrix_plan, stmatrix_plan, stmatrix_writes, LdMatrixPlan, MatrixAccess,
    StMatrixPlan,
};
pub use tcgen::{
    tcgen_cp_decode, tcgen_cp_plan, tcgen_ld_dst_count, tcgen_ld_reduce, tcgen_ld_spcompress,
    tcgen_ldst_map, tcgen_ldst_registers, TcgenCpPlan, TcgenCpWord, TcgenLdRed, TcgenLdstMap,
    TcgenLdstPiece,
};

#[cfg(test)]
mod tests;
