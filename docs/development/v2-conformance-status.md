---
orphan: true
---

# v2 conformance status

Generated 2026-10-08 at 62c4226 by the W8 sweep: every canonical case
(`tests/numsim/corpus/canonical_cases.py`) x {numsim, racecheck, synccheck} run under
`NUMSIM_IMPL=v2` and compared with the legacy snapshots in `tirx_harness/tests/conformance/`
(v2 may add source anchors to diagnostics legacy recorded without one; see
`snapshot.relax_unanchored`). The engine changes hourly; regenerate before acting on a row.

Status values: **match** (bit-identical normalized snapshot); **differs** (with the
behaviour-delta row that explains it, or the filed issue); **incomplete** (v2 stops with a
fail-closed reason); **crash** (engine runtime error or binder exception); **no oracle**
(legacy fails the case). Issue ids `V2C-*` and `W8-6` are filed in
`tirx_harness/src/tirx_harness/numsim/core-rs/numsim-core/CONTRACT_REQUESTS.md` under
"v2 conformance".

## Summary

| mode | match | differs | incomplete | crash | no oracle |
| --- | --- | --- | --- | --- | --- |
| numsim | 81 | 9 | 1 | 8 | 2 |
| racecheck | 34 | 54 | 1 | 10 | 2 |
| synccheck | 63 | 11 | 16 | 9 | 2 |

## Issues by root cause

| issue | owner | cause | cases (modes) |
| --- | --- | --- | --- |
| V2C-5 | racecheck | racecheck-behaviour-deltas P6 (`AsyncNeverCompleted` incomplete at launch end) -- verify | 35: `bmm_fp8_rubin` (r), `cudnn_sm100_bsa_backward_blk128` (r), `cudnn_sm100_bsa_backward_blk64` (r), `cudnn_sm100_bsa_forward_blk128` (r), `cudnn_sm100_bsa_forward_blk64` (r), `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` (r), `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` (r), `cudnn_sm100_dense_gemm_persistent_swiglu` (r), `cudnn_sm100_dsa_sparse_attention_backward` (r), `cudnn_sm100_gdn2_recompute_f16` (r), `cudnn_sm100_gdn_bprop_f16` (r), `cudnn_sm100_gdn_prefill_f16` (r), `cudnn_sm100_gdn_recompute_f16` (r), `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` (r), `cudnn_sm100_moe_grouped_gemm_dglu_dbias` (r), `cudnn_sm103_flex_attention_forward` (r), `deepgemm_sm100_fp4_mqa_logits` (r), `deepgemm_sm100_fp8_bmm` (r), `deepgemm_sm100_fp8_gemm_1d1d` (r), `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` (r), `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` (r), `deepgemm_sm100_m_grouped_fp8_gemm_masked` (r), `deepgemm_sm100_tf32_hc_prenorm_gemm` (r), `dense_blockscaled_gemm_sm107` (r), `fastcu_nvfp4_gemm_gb300` (r), `flash_attention4` (r), `flash_attention4_fp4` (r), `flash_mla_sparse_fwd` (r), `gdn_cp_prefill_sm100` (r), `gdn_prefill_sm100` (r), `grouped_gemm_masked_rubin` (r), `msa_sparse_atten_fwd_sm100` (r), `nvfp4_gemm` (r), `sparse_flashmla_prefill_head128_phase1` (r), `sparse_flashmla_prefill_head64_phase1` (r) |
| V2C-18 | racecheck | racecheck reports `alias_stale_read` where legacy did not | 13: `cudnn_sm100_flex_attention_backward` (r), `cudnn_sm100_flex_attention_forward_hd256` (r), `cudnn_sm100_gdn2_bprop_f16` (r), `cudnn_sm100_gdn2_prefill_f16` (r), `cudnn_sm100_kda_bprop_f16` (r), `kda_decode_multishape` (r), `msa_prefill_multishape` (r), `selective_state_update_mtp_horizontal` (r), `selective_state_update_mtp_vertical` (r), `selective_state_update_stp_horizontal` (r), `selective_state_update_stp_vertical` (r), `stable_sort_topk_by_value` (r), `vsa_multishape` (r) |
| V2C-31 | synccheck | no result within the sweep timeout (explorer ignores the wall-time limit inside one projection) | 10: `cudnn_sm100_bsa_backward_blk64` (s), `cudnn_sm100_dsa_sparse_attention_backward` (s), `cudnn_sm100_gdn_bprop_f16` (s), `cudnn_sm100_gdn_prefill_f16` (s), `cudnn_sm100_gdn_recompute_f16` (s), `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` (r/s), `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` (r/s), `fp16_bf16_gemm` (n/r/s), `gdn_prefill_sm100` (s), `sparse_flashmla_prefill_head128_small_topk_phase1` (n/r/s) |
| V2C-4 | synccheck | synccheck could not build the fixed sync program | 6: `bmm_fp8_rubin` (s), `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` (s), `deepgemm_sm100_fp8_gemm_1d1d` (s), `fastcu_nvfp4_gemm_gb300` (s), `nvfp4_gemm` (s), `sparse_flashmla_prefill_head128_phase1` (s) |
| V2C-20 | interp + test infra (snapshot projection) | uninitialized reads of local arrays: legacy reports space `register` in a per-thread layout, v2 space `local` in its own layout; footprints are not comparable | 4: `flashinfer_qk_rmsnorm` (n/r/s), `flashinfer_rmsnorm_quant` (n/r/s), `gdn_decode_fp32_mtp_warp` (n/r/s), `selective_state_update_mtp_vertical` (n/s) |
| V2C-22 | interp / oplib | new `uninitialized_read` advisories: reads of bytes the engine never wrote (same cases fail the numeric reference, V2C-22) | 3: `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` (n), `cudnn_sm100_moe_grouped_gemm_dglu_dbias` (n), `deepgemm_sm100_tf32_hc_prenorm_gemm` (n/s) |
| V2C-28 | synccheck | synccheck: fixed sync program model incomplete | 3: `deepgemm_sm100_fp4_mqa_logits` (s), `deepgemm_sm100_fp8_mqa_logits` (s), `msa_prefill_multishape` (s) |
| V2C-14 | sync / synccheck | runtime `RegPool(IncompleteWarpgroup)` on setmaxnreg legacy accepts | 2: `cudnn_sm100_dense_blockscaled_gemm_persistent_amax` (n/r/s), `cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant` (n/r/s) |
| V2C-15 | sync / synccheck | synccheck `Cluster(UnexpectedParticipant)` (cf. sync-behaviour-deltas C1 membership) | 2: `dense_blockscaled_gemm_sm107` (s), `grouped_gemm_masked_rubin` (s) |
| V2C-30 | interp (arena::addr) / lowering | shared::cta address names another CTA rank | 2: `alphamoe_fp8_blockscale_qwen3next` (n/r/s), `flash_attention_backward_sm100` (n/r/s) |
| V2C-33 | lowering | lowering rejects `wait_until` whose destination is not a promoted local | 2: `radix_topk_multi_cta` (n/r), `sm100_fp8_fp4_mega_moe` (n/r/s) |
| V2C-19 | interp | an uninitialized read legacy reports is not reported under `ZeroAndReport` | 1: `cudnn_sm100_gdn_recompute_f16` (n) |
| V2C-32 | oplib | sub-byte (FP4/U4) TMA store fragments not representable in TmaPlan | 1: `blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin` (n/r/s) |
| V2C-34 | racecheck (contract: TMEM span convention) | TMEM byte spans use a different addressing than legacy (lane * 2048 + col * 4); the snapshot's column projection cannot compare them | 1: `msa_sparse_atten_fwd_nvfp4_kv_sm100` (r) |

## Public-API legacy tests under v2

The 440 category-A, `surface = public`, non-corpus tests of
`scripts/numsim-v2/coverage/test_classification.csv` (762 parametrized items) run
unchanged with `NUMSIM_IMPL=v2`: `tests/conftest.py` rebinds the public names of
`tirx_harness.numsim` (`transpile`, `Engine`, `compare`, `run_case`, `CoverageBounds`,
`ResourceLimits`, ...) and `tirx_harness.racecheck`/`synccheck` to `numsim.v2` before
collection.

- legacy: 762 passed, 0 failed (2026-10-08)
- v2: **365 passed, 397 failed** of 762

| failure class | owner | count | example |
| --- | --- | --- | --- |
| synccheck verdict differs from legacy | synccheck / sync | 108 | `tests/numsim/runtime/test_tcgen05_tf32.py::test_tf32_sparse_metadata[case3]` |
| other assertion (numeric or report shape) | triage | 48 | `tests/numsim/runtime/test_scalar_control.py::test_fetch_register_models_logical_coordinates_and_uses_stable_time_tokens` |
| expects legacy error text or a legacy raise site | test port / delta review | 44 | `tests/numsim/runtime/test_flashkda_cuda_helpers.py::test_flashkda_math_helper_semantic_mutations_fail_closed[modified_flashkda_fmaf_rn]` |
| racecheck verdict differs from legacy | racecheck | 42 | `tests/numsim/runtime/test_tma_im2col.py::test_im2col_store_and_reduce[False-False-False]` |
| two kernels declare different parameters with one canonical name (V2C-7) | lowering + contract | 41 | `tests/analysis_tools/synccheck/runtime/test_device_additional_payload_ops.py::test_payload_runtime[scalar_cvt_narrowing]` |
| pins legacy internals (generated Rust text, scheduler poll stats); needs porting (test-migration A-internal) | test port | 34 | `tests/numsim/runtime/test_loop_bounds.py::test_lane_varying_min_extent_and_step_use_masked_native_loop` |
| engine stops fail-closed (unsupported / budget) | interp / oplib | 31 | `tests/numsim/integration/test_gemm_async_artifact.py::test_m64_tcgen_mma_infers_weight_stationary_from_packed_layout_e` |
| lowering rejects the kernel | lowering | 28 | `tests/numsim/integration/test_tcgen_transfer_artifact.py::test_tcgen_cp_bitcasts_uint8_scale_payload_into_float8_tmem` |
| engine runtime error legacy did not raise | interp / sync | 21 | `tests/numsim/runtime/test_scalar_control.py::test_unaligned_named_barrier_recombines_disjoint_lane_paths` |

Files passing completely under v2: 34 of 120.

## Per case

| case | numsim | racecheck | synccheck |
| --- | --- | --- | --- |
| `act_and_mul` | match | match | match |
| `alphamoe_fp8_blockscale_qwen3next` | crash [V2C-30] | crash [V2C-30] | crash [V2C-30] |
| `blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin` | incomplete [V2C-32] | incomplete [V2C-32] | incomplete [V2C-32] |
| `bmm_fp8_rubin` | match | differs [V2C-5] | incomplete [V2C-4] |
| `cudnn_sm100_bsa_backward_blk128` | match | differs [V2C-5] | differs |
| `cudnn_sm100_bsa_backward_blk64` | match | differs [V2C-5] | incomplete [V2C-31] |
| `cudnn_sm100_bsa_forward_blk128` | match | differs [V2C-5] | match |
| `cudnn_sm100_bsa_forward_blk64` | match | differs [V2C-5] | match |
| `cudnn_sm100_bsa_forward_combine_blk64` | match | match | match |
| `cudnn_sm100_csa_compressor_fwd` | match | match | match |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_amax` | crash [V2C-14] | crash [V2C-14] | crash [V2C-14] |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant` | crash [V2C-14] | crash [V2C-14] | crash [V2C-14] |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | match | differs [V2C-5] | incomplete [V2C-4] |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | match | differs [V2C-5] | match |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | match | differs [V2C-5] | match |
| `cudnn_sm100_dsa_sparse_attention_backward` | match | differs [V2C-5] | incomplete [V2C-31] |
| `cudnn_sm100_flex_attention_backward` | match | differs [V2C-18] | match |
| `cudnn_sm100_flex_attention_forward_hd256` | match | differs [V2C-18] | match |
| `cudnn_sm100_gdn2_bprop_f16` | match | differs [V2C-18] | match |
| `cudnn_sm100_gdn2_prefill_f16` | match | differs [V2C-18] | match |
| `cudnn_sm100_gdn2_recompute_f16` | match | differs [V2C-5] | match |
| `cudnn_sm100_gdn_bprop_f16` | match | differs [V2C-5] | incomplete [V2C-31] |
| `cudnn_sm100_gdn_prefill_f16` | match | differs [V2C-5] | incomplete [V2C-31] |
| `cudnn_sm100_gdn_recompute_f16` | differs [V2C-19] | differs [V2C-5] | incomplete [V2C-31] |
| `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` | match | crash [V2C-31] | crash [V2C-31] |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | match | crash [V2C-31] | crash [V2C-31] |
| `cudnn_sm100_kda_bprop_f16` | match | differs [V2C-18] | match |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | differs [V2C-22] | differs [V2C-5] | match |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | differs [V2C-22] | differs [V2C-5] | match |
| `cudnn_sm103_flex_attention_forward` | match | differs [V2C-5] | match |
| `deepgemm_sm100_fp4_mqa_logits` | match | differs [V2C-5] | incomplete [V2C-28] |
| `deepgemm_sm100_fp4_paged_mqa_logits` | match | match | match |
| `deepgemm_sm100_fp8_bmm` | match | differs [V2C-5] | match |
| `deepgemm_sm100_fp8_gemm_1d1d` | match | differs [V2C-5] | incomplete [V2C-4] |
| `deepgemm_sm100_fp8_mqa_logits` | match | match | incomplete [V2C-28] |
| `deepgemm_sm100_fp8_paged_mqa_logits` | match | match | match |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | match | differs [V2C-5] | match |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | match | differs [V2C-5] | match |
| `deepgemm_sm100_m_grouped_fp8_gemm_masked` | match | differs [V2C-5] | match |
| `deepgemm_sm100_tf32_hc_prenorm_gemm` | differs [V2C-22] | differs [V2C-5] | differs [V2C-22] |
| `dense_blockscaled_gemm_sm107` | match | differs [V2C-5] | differs [V2C-15] |
| `fast_topk_clusters` | match | match | match |
| `fastcu_nvfp4_gemm_gb300` | match | differs [V2C-5] | incomplete [V2C-4] |
| `filtered_topk` | match | match | match |
| `flash_attention4` | match | differs [V2C-5] | match |
| `flash_attention4_fp4` | match | differs [V2C-5] | match |
| `flash_attention_backward_sm100` | crash [V2C-30] | crash [V2C-30] | crash [V2C-30] |
| `flash_mla_sparse_fwd` | match | differs [V2C-5] | differs |
| `flashinfer_add_rmsnorm_fp4quant` | match | match | match |
| `flashinfer_fused_add_rmsnorm` | match | match | match |
| `flashinfer_fused_add_rmsnorm_quant` | match | match | match |
| `flashinfer_fused_dit_layernorm` | match | match | match |
| `flashinfer_layernorm` | match | match | match |
| `flashinfer_qk_rmsnorm` | differs [V2C-20] | differs [V2C-20] | differs [V2C-20] |
| `flashinfer_rmsnorm` | match | match | match |
| `flashinfer_rmsnorm_fp4quant` | match | match | match |
| `flashinfer_rmsnorm_quant` | differs [V2C-20] | differs [V2C-20] | differs [V2C-20] |
| `fp16_bf16_gemm` | crash [V2C-31] | crash [V2C-31] | crash [V2C-31] |
| `gdn_cp_prefill_sm100` | match | differs [V2C-5] | match |
| `gdn_decode_bf16_ilp4` | match | match | match |
| `gdn_decode_bf16_wide_vec_mtp` | match | match | match |
| `gdn_decode_bf16_wide_vec_t1` | match | match | match |
| `gdn_decode_fp32_mtp_warp` | differs [V2C-20] | differs [V2C-20] | differs [V2C-20] |
| `gdn_prefill_sm100` | match | differs [V2C-5] | incomplete [V2C-31] |
| `grouped_gemm_masked_rubin` | match | differs [V2C-5] | differs [V2C-15] |
| `kda_backward_packed` | match | differs | match |
| `kda_decode_multishape` | match | differs [V2C-18] | match |
| `kda_forward_portfolio_multishape` | no oracle | no oracle | no oracle |
| `merge_state` | match | match | match |
| `mla_dsv4_multishape` | no oracle | no oracle | no oracle |
| `msa_decode_multishape` | match | match | match |
| `msa_prefill_multishape` | match | differs [V2C-18] | incomplete [V2C-28] |
| `msa_sparse_atten_fwd_combine_sm100` | match | match | match |
| `msa_sparse_atten_fwd_nvfp4_kv_sm100` | match | differs [V2C-34] | match |
| `msa_sparse_atten_fwd_sm100` | match | differs [V2C-5] | match |
| `msa_sparse_prepare_flat_schedule_sm100` | match | match | match |
| `msa_sparse_prepare_fwd_split_atomic_sm100` | match | match | match |
| `mxfp4_quantize` | match | match | match |
| `mxfp8_quantize` | match | match | match |
| `nvfp4_gemm` | match | differs [V2C-5] | incomplete [V2C-4] |
| `nvfp4_quantize` | match | match | match |
| `nvfp4_quantize_per_token` | match | match | match |
| `radix_topk_multi_cta` | crash [V2C-33] | crash [V2C-33] | match |
| `radix_topk_single_cta` | match | match | match |
| `recurrent_kda_decode_grouped` | match | match | match |
| `recurrent_kda_decode_one_warp` | match | match | match |
| `rmsnorm` | match | match | match |
| `selective_state_update_mtp_horizontal` | match | differs [V2C-18] | match |
| `selective_state_update_mtp_simple` | match | match | match |
| `selective_state_update_mtp_vertical` | differs [V2C-20] | differs [V2C-18] | differs [V2C-20] |
| `selective_state_update_stp_horizontal` | match | differs [V2C-18] | match |
| `selective_state_update_stp_simple` | match | match | match |
| `selective_state_update_stp_vertical` | match | differs [V2C-18] | match |
| `silu_and_mul_nvfp4_experts_quantize` | match | match | match |
| `sm100_fp8_fp4_mega_moe` | crash [V2C-33] | crash [V2C-33] | crash [V2C-33] |
| `sparse_flashmla_decode_head64` | differs | differs | differs |
| `sparse_flashmla_prefill_head128_phase1` | match | differs [V2C-5] | incomplete [V2C-4] |
| `sparse_flashmla_prefill_head128_small_topk_phase1` | crash [V2C-31] | crash [V2C-31] | crash [V2C-31] |
| `sparse_flashmla_prefill_head64_phase1` | match | differs [V2C-5] | differs |
| `stable_sort_topk_by_value` | match | differs [V2C-18] | match |
| `vsa_multishape` | match | differs [V2C-18] | match |
