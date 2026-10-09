//! Warp / CTA cross-lane math on `WarpValue<T> = [T; 32]` + `WarpMask`.
//!
//! Legacy sources: `engine-rs/src/runtime/warp_ops.rs`,
//! `engine-rs/src/runtime/instructions/warp.rs`,
//! `engine-rs/src/runtime/instructions/collective.rs`,
//! `engine-rs/src/collectives.rs` (CTA reduce combine only) and
//! `engine-rs/src/runtime/sync.rs` (`require_full_warp_sync`).
//!
//! Submodules:
//! - [`mask`]: membermask validation, full-warp/uniform checks, `activemask`.
//! - [`shuffle`]: `shfl.sync` (idx/up/down/bfly, packed `c` operand).
//! - [`sync`]: `vote`, `match`, `redux`, `elect`, `fns`, `movmatrix`.
//! - [`reduce`]: `tirx.cuda.warp_reduce` butterflies.
//! - [`cta`]: `cta_reduce` / `syncthreads_and|or` / `barrier.red` combine.
//!
//! Every function is pure; engine scheduling (rendezvous, collective hubs,
//! occurrence keys, operation begin/finish, scratch traffic) stays outside.

pub mod cta;
pub mod mask;
pub mod reduce;
pub mod shuffle;
pub mod sync;

pub use cta::*;
pub use mask::*;
pub use reduce::*;
pub use shuffle::*;
pub use sync::*;

use crate::registry::Binding;

const fn bind(op: &'static str, function: &'static str) -> Binding {
    Binding { op, function }
}

/// Registry bindings for this family: legacy op name -> OpLib function path.
pub(crate) const BINDINGS: &[Binding] = &[
    bind("tirx.cuda.__activemask", "warp::activemask"),
    bind("tirx.ptx.activemask", "warp::activemask"),
    bind("tirx.cuda.__shfl_sync", "warp::shfl_idx"),
    bind("tirx.cuda.__shfl_up_sync", "warp::shfl_up"),
    bind("tirx.cuda.__shfl_down_sync", "warp::shfl_down"),
    bind("tirx.cuda.__shfl_xor_sync", "warp::shfl_bfly"),
    bind("tirx.ptx.shfl_sync", "warp::shfl_sync"),
    bind("tirx.ptx.shfl_sync", "warp::shuffle_source_mask"),
    bind("tirx.ptx.shfl_sync_p", "warp::shfl_sync"),
    bind("tirx.cuda.any_sync", "warp::vote_any"),
    bind("tirx.cuda.ballot_sync", "warp::vote_ballot"),
    bind("tirx.ptx.vote_sync", "warp::vote_all"),
    bind("tirx.ptx.vote_sync", "warp::vote_any"),
    bind("tirx.ptx.vote_sync", "warp::vote_uni"),
    bind("tirx.ptx.vote_sync_ballot", "warp::vote_ballot"),
    bind("tirx.ptx.match_any_sync", "warp::match_any"),
    bind("tirx.ptx.match_all_sync", "warp::match_all"),
    bind("tirx.ptx.match_all_sync_p", "warp::match_all"),
    bind("tirx.ptx.redux_sync", "warp::redux_sync_u32"),
    bind("tirx.ptx.redux_sync", "warp::redux_sync_i32"),
    bind("tirx.ptx.redux_sync_bitwise", "warp::redux_sync_u32"),
    bind("tirx.ptx.redux_sync_f32", "warp::redux_sync_f32"),
    bind("tirx.cuda.elect_sync", "warp::elect_sync"),
    bind("tirx.ptx.elect_sync", "warp::elect_sync"),
    bind("tirx.ptx.fns", "warp::fns_b32"),
    bind("tirx.ptx.movmatrix", "warp::movmatrix_m8n8_trans_b16"),
    bind("tirx.cuda.warp_reduce", "warp::warp_reduce_sum"),
    bind("tirx.cuda.warp_reduce", "warp::warp_reduce_max"),
    bind("tirx.cuda.warp_reduce", "warp::warp_reduce_min"),
    bind("tirx.cuda.warp_reduce", "warp::warp_reduce_sum_fp16"),
    bind("tirx.cuda.warp_reduce", "warp::warp_reduce_sum_bf16"),
    bind("tirx.cuda.cta_reduce", "warp::validate_cta_reduce_warps"),
    bind("tirx.cuda.cta_reduce", "warp::warp_reduce"),
    bind("tirx.cuda.cta_reduce", "warp::cta_reduce_publish"),
    bind("tirx.cuda.syncthreads_and", "warp::cta_vote_local"),
    bind("tirx.cuda.syncthreads_and", "warp::cta_reduce_publish"),
    bind("tirx.cuda.syncthreads_and", "warp::cta_vote_result"),
    bind("tirx.cuda.syncthreads_or", "warp::cta_vote_local"),
    bind("tirx.cuda.syncthreads_or", "warp::cta_reduce_publish"),
    bind("tirx.cuda.syncthreads_or", "warp::cta_vote_result"),
    bind("tirx.ptx.barrier_red_popc", "warp::bar_red_local"),
    bind("tirx.ptx.barrier_red_popc", "warp::cta_reduce_publish"),
    bind("tirx.ptx.barrier_red_popc", "warp::bar_red_result"),
    bind("tirx.ptx.barrier_red_popc_count", "warp::bar_red_local"),
    bind(
        "tirx.ptx.barrier_red_popc_count",
        "warp::cta_reduce_publish",
    ),
    bind("tirx.ptx.barrier_red_pred", "warp::bar_red_local"),
    bind("tirx.ptx.barrier_red_pred", "warp::cta_reduce_publish"),
    bind("tirx.ptx.barrier_red_pred_count", "warp::bar_red_local"),
    bind(
        "tirx.ptx.barrier_red_pred_count",
        "warp::cta_reduce_publish",
    ),
];

#[cfg(test)]
mod tests {
    use super::BINDINGS;

    #[test]
    fn bindings_name_registered_ops() {
        for binding in BINDINGS {
            assert!(
                crate::registry::op(binding.op).is_some(),
                "unknown op {}",
                binding.op
            );
            assert!(binding.function.starts_with("warp::"));
        }
    }
}
