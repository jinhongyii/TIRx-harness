---
orphan: true
---

# v2 conformance status

Generated 2026-10-08 at 62c4226 (corpus sweep) / 9484204 (public-API run) by the W8 sweep: every canonical case
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
- v2: **367 passed, 395 failed** of 762

| failure class | owner | count | example |
| --- | --- | --- | --- |
| synccheck verdict differs from legacy | synccheck / sync | 107 | `tests/numsim/runtime/test_async_release.py::test_async_release_payloads[True-s64]` |
| other assertion (numeric or report shape) | triage | 64 | `tests/numsim/runtime/test_non_tensor_bulk_forms.py::test_raw_bulk_prefetch_preserves_global_memory` |
| pins legacy internals (generated Rust text, scheduler poll stats); needs porting (test-migration A-internal) | test port | 60 | `tests/numsim/runtime/test_loop_bounds.py::test_lane_varying_min_extent_and_step_use_masked_native_loop` |
| expects legacy error text or a legacy raise site | test port / delta review | 44 | `tests/numsim/runtime/test_flashkda_cuda_helpers.py::test_flashkda_math_helper_semantic_mutations_fail_closed[modified_flashkda_fmaf_rn]` |
| racecheck verdict differs from legacy | racecheck | 42 | `tests/numsim/runtime/test_tma_im2col.py::test_im2col_store_and_reduce[False-False-False]` |
| engine stops fail-closed (unsupported / budget) | interp / oplib | 31 | `tests/numsim/integration/test_gemm_async_artifact.py::test_m64_tcgen_mma_infers_weight_stationary_from_packed_layout_e` |
| lowering rejects the kernel | lowering | 28 | `tests/numsim/integration/test_warp_gemm_artifact.py::test_warp_gemm_supports_registered_m16n8_family[bfloat16-8-2-2-3-True-False-1]` |
| engine runtime error legacy did not raise | interp / sync | 19 | `tests/numsim/runtime/test_scalar_control.py::test_unaligned_named_barrier_recombines_disjoint_lane_paths` |

Files passing completely under v2: 34 of 120.

### Triage of the "other assertion" public-API failures (2026-10-08, at 9484204)

Re-run with the current engine: 65 items fall outside the named classes
(the "other" bucket grew because several checker-verdict assertions are
written as `assert verdict == ...` rather than through the report text).
Classes: **delta** = explained by a behaviour-delta row or ruling (test needs
updating), **bug** = v2 is wrong (owner), **internals** = pins legacy
internals or legacy scheduling policy (delete or port to `Program` asserts).

| class | owner | items | node ids (file::test, params collapsed) |
| --- | --- | --- | --- |
| internals | test port (W9) | 4 | `test_non_tensor_bulk_forms::{test_raw_bulk_prefetch_preserves_global_memory, test_bulk_g2s_cluster_accepts_static_true_predicate_as_unconditional, test_bulk_g2s_cluster_dynamic_predicate_transpiles}` (legacy `call_op_names(spec)`); `test_ordering_calls::test_ordering_only_calls_preserve_native_source_order` (`KernelSpec.semantic_requirements`) |
| internals | test port (W9) | 3 | `test_scheduler_polling_artifact::{test_time_slice_does_not_replay_body_side_effects_and_schedules_peer, test_time_slice_uses_configured_reschedule_quantum, test_finite_for_time_slice_schedules_peer_without_pattern_matching}`: assert legacy time-slice/quantum interleavings (schedule policy, not semantics) |
| delta | test port (W9) | 1 | `test_discard::test_discard_indeterminate_read_and_alignment`: reading discarded bytes is `review` under the W8-5 ruling (`ZeroAndReport`), legacy `error` |
| delta | test port (W9) | 1 | `test_mbarrier_maintenance::test_maintenance_active_addresses_still_checked`: kind `oob` -> `out_of_bounds` (racecheck delta P5 / test-migration rename list) plus an `uninitialized_read` review |
| delta | sync (confirm) | 2 | `test_memory_coverage_next::test_no_complete_rejects_exhausted_arrivals[2,3]`: sync delta M6 (typed `NoCompleteWouldComplete`); the test looks for the legacy kind |
| bug | interp | 3 | `test_launch_resource_facts::test_smid_defaults_to_zero_in_execution_and_both_checkers`, `test_fetch_register_domain_oracle::test_every_fetch_register_form_matches_the_logical_topology`, `test_scalar_control::test_fetch_register_models_logical_coordinates_and_uses_stable_time_tokens`: `%smid` (and related fetch registers) report the CTA index instead of legacy's 0 |
| bug / contract | arena::addr (coordinator) | 2 | `test_launch_resource_facts::test_pointer_bits_preserve_binding_address_and_subview_offset` (legacy VAs keep the host pointer's low 8 bits), `test_scalar_control::test_mapa_and_cvta_expose_the_device_validated_integer_bits` (device-validated aperture bits): need a ruling on synthetic address bits |
| bug | oplib | 9 | `test_atomic_f32_noftz::test_atomic_f32_noftz[{atom,red,sink}-{1-shared::cta,2-global,4-}]`: f32 atomics flush/round subnormals |
| bug | oplib | 1 | `test_approximate_f32_contract::test_worker_normalizes_rounding_without_polluting_the_caller`: 1-ulp rounding-mode leak |
| bug | oplib / lowering (tile) | 4 | `test_tile_reduction_variants::{test_warp_collective_reduction_follows_physical_lane_ownership, test_local_float64_reductions_match_b200_edge_bits, test_maxmin_uses_canonical_lexicographic_nan_and_signed_zero_order, test_local_collective_uses_lexicographic_order}`: reduction order / NaN and signed-zero ordering |
| bug | oplib | 2 | `test_tma_u6::test_tma_u6_layout_and_checkers` (U6 TMA layout), `test_non_tensor_bulk_forms::test_ignore_oob_dead_bytes_are_zero_filled_and_require_review_if_read` (ignore_oob fill) |
| bug | interp | 4 | `test_mov_forms::test_mov_aliased_sources_destinations_and_predicate`, `test_memory_sync_coverage::{test_memory_sync_extensions[address_queries-...], test_half_vector_predicate_alias_is_captured_before_either_result_store}`, `test_non_tensor_bulk_forms::test_st_bulk_size_is_evaluated_per_issuing_lane` |
| bug | interp / lowering | 1 | `test_global_alias_artifact::test_global_decl_buffer_alias_reuses_parameter_allocation_bytes`: a global `decl_buffer` view of a parameter reads zeros |
| bug | interp | 4 | `test_copy_multicast32::test_multicast_high_bit_targets_and_bounds[bulk,tensor,commit,im2col]`: out-of-cluster multicast targets are `incomplete`, legacy `error` |
| bug | sync / interp | 1 | `test_sync_runtime_domain_oracle::test_clc_no_work_completion_and_cluster_acquire_wait`: CLC no-work response value |
| bug | sync | 1 | `test_launch_resource_facts::test_exclusive_tmem_uses_cta_local_lifecycle_without_placement`: exclusive 96/576-column lifecycle deadlocks (cf. sync delta T2/T4; 576 needs `Program.arch`) |
| bug | racecheck | 2 | `test_tcgen05_restricted_commit::test_restricted_commit_preserves_full_mma_completion[1-False-False, 1-True-True]`: a race on B after waiting only for A is **not reported** (false negative) |
| bug | synccheck | 19 | `tests/analysis_tools/synccheck/runtime/test_device_additional_payload_ops.py::test_payload_runtime[*]` (16 params), `test_shared_descriptor_choices::test_shared_descriptor_choices_validate_the_consumed_bits[False-select, False-compose]`, `test_reported_tool_regressions::test_reported_instruction_support[red_vec_packed_bf16]`: Synccheck `incomplete` where legacy was clean/error (V2C-4/V2C-28 class) |
| bug | lowering | 1 | `test_host_prelude::test_explicit_tensor_map_override_updates_its_distinct_backing`: the explicit tensor map is named `tensor_map.tmap`, so the caller's `tensor_map` binding is unknown |

### Public-API failures that expect legacy error text or raise sites (to W9)

These 44 fail only because they match legacy messages or expect a raise
where v2 reports a diagnostic; each needs its expectation ported (or a delta
row) by the test-migration owner:

- `tests/numsim/runtime/test_flashkda_cuda_helpers.py::test_flashkda_math_helper_semantic_mutations_fail_closed[modified_flashkda_fmaf_rn]`
- `tests/numsim/integration/test_scheduler_polling_artifact.py::test_finite_for_uses_native_loop_iteration_budget`
- `tests/numsim/runtime/test_flashkda_cuda_helpers.py::test_flashkda_math_helper_semantic_mutations_fail_closed[modified_flashkda_rsqrtf]`
- `tests/numsim/runtime/test_flashkda_cuda_helpers.py::test_flashkda_math_helper_semantic_mutations_fail_closed[modified_flashkda_tanh_approx]`
- `tests/numsim/integration/test_scheduler_polling_artifact.py::test_finite_loop_can_exceed_default_budget_when_configured`
- `tests/numsim/runtime/test_ptx_integer_arithmetic.py::test_ptx_signed_division_overflow_fails_closed[div]`
- `tests/numsim/runtime/test_scalar_control.py::test_floating_trap_rejects_signed_zero[predicate0]`
- `tests/numsim/runtime/test_wait_until.py::test_a_candidate_index_outside_its_local_table_is_an_execution_error`
- `tests/numsim/runtime/test_pointer_slot_arrays.py::test_pointer_array_initialization_and_bounds[oob]`
- `tests/numsim/runtime/test_ptx_integer_arithmetic.py::test_ptx_signed_division_overflow_fails_closed[rem]`
- `tests/numsim/runtime/test_float8_address_shuffle.py::test_scalar_fp8_identity_reinterpret_rejects_256_payload_roundtrip[float8_e4m3fn]`
- `tests/numsim/runtime/test_float8_address_shuffle.py::test_scalar_fp8_identity_reinterpret_rejects_256_payload_roundtrip[float8_e8m0fnu]`
- `tests/numsim/runtime/test_scalar_control.py::test_floating_trap_rejects_signed_zero[predicate1]`
- `tests/numsim/runtime/test_scalar_control.py::test_integer_trap_predicate_uses_cpp_truth_conversion`
- `tests/analysis_tools/shared/test_known_cuda_func_artifact.py::test_modified_gdn_lg2_helper_remains_fail_closed`
- `tests/analysis_tools/shared/test_known_cuda_func_artifact.py::test_modified_known_helper_remains_fail_closed`
- `tests/numsim/runtime/test_ordering_calls.py::test_default_full_mask_warp_sync_rejects_divergent_execution`
- `tests/analysis_tools/shared/test_known_cuda_func_artifact.py::test_modified_packed_fma_helper_remains_fail_closed`
- `tests/numsim/runtime/test_ptx_integer_arithmetic.py::test_ptx_integer_division_by_zero_fails_closed_at_the_faulting_lane[div]`
- `tests/numsim/runtime/test_ptx_integer_arithmetic.py::test_ptx_integer_division_by_zero_fails_closed_at_the_faulting_lane[rem]`
- `tests/numsim/runtime/test_scalar_control.py::test_mbarrier_state_token_rejects_a_generation_older_than_the_previous_one`
- `tests/numsim/runtime/test_ordering_calls.py::test_setmaxnreg_rejects_warp_disagreement_within_one_occurrence`
- `tests/numsim/runtime/test_ordering_calls.py::test_setmaxnreg_requires_explicit_warpgroup_sync_before_a_later_call`
- `tests/numsim/runtime/test_packed_float4_global_views.py::test_direct_one_byte_per_value_float4_array_is_rejected`
- `tests/numsim/runtime/test_dynamic_pure_call_runtime_domains.py::test_if_then_else_mixed_pointer_spaces_fail_closed`
- `tests/numsim/runtime/test_memory_coverage_next.py::test_ldu_rejects_nonuniform_or_misaligned_vector[source.ptr_to([lane])]`
- `tests/numsim/runtime/test_memory_coverage_next.py::test_ldu_rejects_nonuniform_or_misaligned_vector[source.ptr_to([1])]`
- `tests/numsim/runtime/test_flashkda_cuda_helpers.py::test_flashkda_math_helper_dtype_mismatch_fails_closed`
- `tests/numsim/runtime/test_mbarrier_lane_semantics.py::test_blocking_wait_fails_closed_for_mixed_lane_readiness`
- `tests/numsim/integration/test_host_prelude.py::test_host_tensor_map_integer_expressions_fail_closed_on_unregistered_nodes`
- `tests/numsim/integration/test_tmem_artifact.py::test_tmem_runtime_address_without_a_dynamic_lease_is_rejected`
- `tests/numsim/integration/test_warp_ops_artifact.py::test_warp_collectives_reject_invalid_participant_contracts`
- `tests/numsim/runtime/test_memory_coverage_next.py::test_tensor_map_update_requires_release`
- `tests/numsim/runtime/test_non_tensor_bulk_forms.py::test_ignore_oob_rejects_counts_outside_ptx_range[ignored0]`
- `tests/numsim/runtime/test_non_tensor_bulk_forms.py::test_ignore_oob_rejects_counts_outside_ptx_range[ignored1]`
- `tests/numsim/integration/test_tmem_artifact.py::test_tmem_dynamic_lease_rejects_invalid_lifetimes_and_ranges[dynamic_tmem_use_before_alloc-not covered by any live allocation]`
- `tests/numsim/integration/test_tmem_artifact.py::test_tmem_dynamic_lease_rejects_invalid_lifetimes_and_ranges[dynamic_tmem_outside_live_lease-not covered by any live allocation]`
- `tests/numsim/integration/test_tmem_artifact.py::test_tmem_dynamic_lease_rejects_invalid_lifetimes_and_ranges[dynamic_tmem_use_after_dealloc-not covered by any live allocation]`
- `tests/numsim/runtime/test_tile_unary_codegen.py::test_unary_tile_ops_fail_closed_on_unknown_config`
- `tests/numsim/integration/test_tmem_artifact.py::test_tmem_live_allocation_at_kernel_exit_is_rejected`
- `tests/numsim/runtime/test_scalar_control.py::test_cuda_pointer_helpers_check_typed_dereference_alignment`
- `tests/numsim/runtime/test_scalar_control.py::test_cuda_shuffle_rejects_invalid_width`
- `tests/numsim/runtime/test_tile_general_semantics.py::test_float64_directed_rounding_fails_closed`
- `tests/numsim/integration/test_warp_gemm_artifact.py::test_warp_gemm_rejects_layout_that_disagrees_with_instruction_abi`

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
