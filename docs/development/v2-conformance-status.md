---
orphan: true
---

# v2 conformance status

Generated 2026-10-08 at 8fba7e1 + a3c4df9 (fifth sweep); rows re-run at ea185df: R3/X4 deltas, fp16_bf16_gemm, kda_forward_portfolio_multishape by the W8 sweep: every canonical case
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
| numsim | 99 | 0 | 0 | 0 | 2 |
| racecheck | 100 | 0 | 0 | 0 | 1 |
| synccheck | 98 | 0 | 1 | 0 | 2 |

## Issues by root cause

| issue | owner | cause | cases (modes) |
| --- | --- | --- | --- |
| V2C-TF1 (closed) | W1, W12 | Coverage regression: TVM's `TilePrimitiveDispatch` rejected tile ops in 58 public-API kernels that legacy compiled. All 58 are now resolved: lowered through TVM after a legacy-spelling repair or by a v2 tile form (`v2/lowering/tile_forms/`, lowering-inventory Part F), or ruled hardware-invalid (deltas L4–L7, with the tests moved to valid shapes), or blocked only by L1 (replicated TMEM view). | `tests/numsim/v2/tile_forms/legacy_compiled.tsv`; `test_legacy_compiled_tile_forms.py`: 62 pass, 1 strict-xfail (L1) |
| V2C-31 | synccheck | synccheck exhausts its state/transition budget where legacy completed within the same ResourceLimits | 1: `kda_forward_portfolio_multishape` (s) |

## Decisions deferred to after the legacy deletion

| decision | current v2 behaviour | why deferred |
| --- | --- | --- |
| Enforce the hardware cluster-size limit (16 CTAs, non-portable, sm_90/sm_100) | Legacy engine limit: more than 64 CTAs per cluster fails closed at transpile | 14 sweep kernels use 20-CTA clusters that legacy accepted; enforcing needs a delta row and a test migration. See numsim-isa-answers.md, "Cluster size". |

## Cases without a legacy result

| case | legacy | v2 status |
| --- | --- | --- |
| `fp16_bf16_gemm` | Legacy lowering raises `UnsupportedTIRxError: unsupported host statement before tirx.device_entry: Evaluate` (host prelude) in all three modes. | numsim clean, outputs match the reference; synccheck clean (both still skipped: no oracle until the post-deletion regeneration). Racecheck has a v2 oracle now: `racecheck.delta.json`, racecheck-behaviour-deltas B7 (TMEM-teardown handshake, racecheck-semantics §11). |
| `mla_dsv4_multishape` | Same legacy host-prelude `UnsupportedTIRxError`. | numsim clean, outputs match the reference; racecheck clean; synccheck clean since ea185df (M14). Skipped until the post-deletion regeneration from v2. |
| `kda_forward_portfolio_multishape` | Runs since W9's fixture fix (binds `items`/`item_counts` from `packed_schedule`); legacy snapshots regenerated (clean in all modes). | numsim and racecheck match. Synccheck is `incomplete` (`resource_limit`) under the corpus budget (100k states, 180 s, 32 MB diagnostics), where legacy is clean: V2C-31 (synccheck budget). |

## Public-API legacy tests under v2

The 440 category-A, `surface = public`, non-corpus tests of
`scripts/numsim-v2/coverage/test_classification.csv` (762 parametrized items) run
unchanged with `NUMSIM_IMPL=v2`: `tests/conftest.py` rebinds the public names of
`tirx_harness.numsim` (`transpile`, `Engine`, `compare`, `run_case`, `CoverageBounds`,
`ResourceLimits`, ...) and `tirx_harness.racecheck`/`synccheck` to `numsim.v2` before
collection.

- legacy: 762 passed, 0 failed (2026-10-08)
- v2: **551 passed, 211 failed** of 762

| failure class | owner | count | example |
| --- | --- | --- | --- |
| pins legacy internals (generated Rust text, scheduler poll stats); needs porting (test-migration A-internal) | test port | 85 | `tests/analysis_tools/shared/test_known_cuda_func_artifact.py::test_known_combine_int_frac_ex2_is_bit_exact` |
| other assertion (numeric or report shape) | triage | 41 | `tests/numsim/runtime/test_memory_sync_coverage.py::test_memory_sync_extensions[address_queries-inputs5-expected5]` |
| expects legacy error text or a legacy raise site | test port / delta review | 26 | `tests/numsim/integration/test_scheduler_polling_artifact.py::test_finite_for_uses_native_loop_iteration_budget` |
| racecheck verdict differs from legacy | racecheck | 24 | `tests/numsim/runtime/test_non_tensor_bulk_forms.py::test_raw_bulk_s2c_preserves_mapped_remote_cta_ownership` |
| engine runtime error legacy did not raise | interp / sync | 21 | `tests/numsim/runtime/test_layout_lowering_contract.py::test_tmem_tlane_tcol_coordinates_are_observable_through_a_physical_alias` |
| synccheck verdict differs from legacy | synccheck / sync | 7 | `tests/numsim/runtime/test_pointer_slot_arrays.py::test_pointer_array_initialization_and_bounds[uninitialized]` |
| lowering rejects the kernel | lowering | 6 | `tests/numsim/runtime/test_dense_mma_forms.py::test_legacy_m16n8k32_int8_reuses_dense_form_and_engine` |
| engine stops fail-closed (unsupported / budget) | interp / oplib | 1 | `tests/numsim/runtime/test_gate_intrinsics.py::test_gate_intrinsics_match_float32_semantics` |

Files passing completely under v2: 64 of 120.

### Public-API failure triage (fifth sweep, 8fba7e1)

Per function in `scripts/numsim-v2/coverage/v2_public_status.tsv` (refreshed from this run; W9 input).
An item counts once, under the first class its failure message matches. "delta-explained" items need their
expectation ported; "legacy-internals" items pin legacy implementation details and are delete-or-port; the
rest are v2 bugs for the named owner unless a delta row is added.

| class | triage | owner | failing items | functions | example function |
| --- | --- | --- | --- | --- | --- |
| pin-internals | legacy-internals (delete or port) | test port (W9) | 90 | 39 | `tests/analysis_tools/shared/test_known_cuda_func_artifact.py::test_known_combine_int_frac_ex2_is_bit_exact` |
| other-assertion | bug: numeric or report-shape assertion | triage per test | 29 | 25 | `tests/analysis_tools/synccheck/runtime/test_device_additional_payload_ops.py::test_payload_runtime` |
| pin-message | delta-explained or message change (port expectation) | test port (W9) | 25 | 19 | `tests/numsim/integration/test_scheduler_polling_artifact.py::test_finite_for_uses_native_loop_iteration_budget` |
| racecheck-verdict | bug or delta: racecheck verdict differs | racecheck (W5) | 24 | 7 | `tests/numsim/runtime/test_cp_async_mbarrier_visibility.py::test_copy_arrival_retains_earlier_copy_history` |
| engine-stops | bug: engine stops (unsupported / runtime error) | interp / oplib (W2 / W4) | 22 | 19 | `tests/numsim/integration/test_fp8_cta1_descriptor_layout.py::test_fp8_cta1_extended_shared_addresses` |
| synccheck-verdict | bug or delta: synccheck verdict differs | synccheck / sync (W6 / W3) | 14 | 8 | `tests/analysis_tools/synccheck/runtime/test_device_additional_payload_ops.py::test_payload_runtime` |
| lowering-rejects | bug: lowering rejects the kernel | lowering (W1) | 6 | 6 | `tests/numsim/integration/test_tcgen_transfer_artifact.py::test_tcgen_cp_bitcasts_uint8_scale_payload_into_float8_tmem` |
| v2-accepts-legacy-rejection | bug: v2 accepts what legacy rejected | lowering / interp (per test) | 1 | 1 | `tests/numsim/integration/test_reported_layout_regressions.py::test_explicit_shared_strides_still_reject_an_executed_oob_address` |

<details><summary>synccheck-verdict: 8 functions, owner synccheck / sync (W6/W3)</summary>

- `tests/analysis_tools/synccheck/runtime/test_device_additional_payload_ops.py::test_payload_runtime` (other-assertion=1 pin-internals=38 synccheck-verdict=2)
- `tests/numsim/runtime/test_copy_multicast32.py::test_multicast_high_bit_targets_and_bounds` (synccheck-verdict=4)
- `tests/numsim/runtime/test_cp_async_mbarrier_visibility.py::test_copy_arrival_is_not_an_explicit_commit` (synccheck-verdict=1)
- `tests/numsim/runtime/test_mov_forms.py::test_mov_pointer_identity_masks_and_nulls` (synccheck-verdict=1)
- `tests/numsim/runtime/test_pointer_slot_arrays.py::test_pointer_array_initialization_and_bounds` (pass=1 pin-message=1 synccheck-verdict=1)
- `tests/numsim/runtime/test_sync_instruction_predicates.py::test_pending_count_instruction_predicates` (synccheck-verdict=1)
- `tests/numsim/runtime/test_tcgen05_ld_spcompress.py::test_tcgen_load_compression` (synccheck-verdict=3)
- `tests/numsim/runtime/test_tensormap_predicates.py::test_tensor_map_predicate_effects` (synccheck-verdict=1)

</details>

<details><summary>racecheck-verdict: 7 functions, owner racecheck (W5)</summary>

- `tests/numsim/runtime/test_cp_async_mbarrier_visibility.py::test_copy_arrival_retains_earlier_copy_history` (racecheck-verdict=2)
- `tests/numsim/runtime/test_layout_lowering_contract.py::test_physical_buffers_preserve_coordinates_aliases_and_lane_private_storage` (racecheck-verdict=1)
- `tests/numsim/runtime/test_mbarrier_multicast.py::test_mbarrier_multicast32` (pass=2 racecheck-verdict=6)
- `tests/numsim/runtime/test_mbarrier_multicast.py::test_mbarrier_multicast_lane_masks` (racecheck-verdict=1)
- `tests/numsim/runtime/test_memory_sync_coverage.py::test_memory_sync_extensions` (other-assertion=1 pass=5 racecheck-verdict=1)
- `tests/numsim/runtime/test_non_tensor_bulk_forms.py::test_raw_bulk_s2c_preserves_mapped_remote_cta_ownership` (racecheck-verdict=1)
- `tests/numsim/runtime/test_red_async.py::test_red_async` (racecheck-verdict=12)

</details>

<details><summary>engine-stops: 19 functions, owner interp / oplib (W2/W4)</summary>

- `tests/numsim/integration/test_fp8_cta1_descriptor_layout.py::test_fp8_cta1_extended_shared_addresses` (engine-stops=2)
- `tests/numsim/integration/test_gemm_async_artifact.py::test_bf16_m64_tcgen_mma_uses_layout_f` (engine-stops=1)
- `tests/numsim/integration/test_gemm_async_artifact.py::test_cta_group2_banked_a_selects_matching_b_shard` (engine-stops=1)
- `tests/numsim/integration/test_gemm_async_artifact.py::test_dense_gemm_async_reads_tmem_a_and_transposed_b_storage` (engine-stops=1)
- `tests/numsim/integration/test_gemm_async_artifact.py::test_dense_gemm_async_starts_the_fma_chain_from_input_d` (engine-stops=1)
- `tests/numsim/integration/test_gemm_async_artifact.py::test_large_bf16_gemm_async_uses_one_engine_gemm` (engine-stops=1)
- `tests/numsim/integration/test_gemm_async_artifact.py::test_m64_tcgen_mma_infers_weight_stationary_from_packed_layout_e` (engine-stops=1)
- `tests/numsim/integration/test_gemm_async_artifact.py::test_m64_tcgen_mma_uses_layout_f_independently_of_declared_tmem_layout` (engine-stops=1)
- `tests/numsim/integration/test_raw_versus_typed_cta2_mma.py::test_raw_cta2_mma_matches_the_typed_gemm_async_exactly` (engine-stops=3)
- `tests/numsim/integration/test_raw_versus_typed_cta2_mma.py::test_raw_cta2_ts_m128_selects_the_matching_a_lane_bank` (engine-stops=1)
- `tests/numsim/integration/test_tmem_artifact.py::test_tmem_layout_f_maps_rows_to_half_slabs` (engine-stops=1)
- `tests/numsim/runtime/test_gate_intrinsics.py::test_gate_intrinsics_match_float32_semantics` (engine-stops=1)
- `tests/numsim/runtime/test_launch_resource_facts.py::test_exclusive_tmem_uses_cta_local_lifecycle_without_placement` (engine-stops=1)
- `tests/numsim/runtime/test_layout_lowering_contract.py::test_tmem_tlane_tcol_coordinates_are_observable_through_a_physical_alias` (engine-stops=1)
- `tests/numsim/runtime/test_ordering_calls.py::test_deleting_setmaxnreg_keeps_the_numerical_result` (engine-stops=1)
- `tests/numsim/runtime/test_ordering_calls.py::test_setmaxnreg_is_an_ordering_call_not_a_tcgen_lifecycle_call` (engine-stops=1)
- `tests/numsim/runtime/test_ordering_calls.py::test_setmaxnreg_static_expressions_follow_public_parser_and_runtime` (engine-stops=1)
- `tests/numsim/runtime/test_tf32_layout_f_poison.py::test_tf32_layout_f_unwritten_holes_are_zero_filled_and_require_review` (engine-stops=1)
- `tests/numsim/runtime/test_tile_owner_transport.py::test_copy_transports_unique_owners_across_warps` (engine-stops=1)

</details>

<details><summary>other-assertion: 25 functions, owner triage per test</summary>

- `tests/analysis_tools/synccheck/runtime/test_device_additional_payload_ops.py::test_payload_runtime` (other-assertion=1 pin-internals=38 synccheck-verdict=2)
- `tests/numsim/integration/test_global_alias_artifact.py::test_global_decl_buffer_alias_reuses_parameter_allocation_bytes` (other-assertion=1)
- `tests/numsim/integration/test_scheduler_polling_artifact.py::test_finite_for_time_slice_schedules_peer_without_pattern_matching` (other-assertion=1)
- `tests/numsim/integration/test_scheduler_polling_artifact.py::test_time_slice_does_not_replay_body_side_effects_and_schedules_peer` (other-assertion=1)
- `tests/numsim/integration/test_scheduler_polling_artifact.py::test_time_slice_uses_configured_reschedule_quantum` (other-assertion=1)
- `tests/numsim/integration/test_tcgen_transfer_artifact.py::test_tcgen_cp_cta_group2_supports_float16_payloads` (other-assertion=1)
- `tests/numsim/integration/test_tcgen_transfer_artifact.py::test_tcgen_cp_expands_tlane_replicas` (other-assertion=1)
- `tests/numsim/integration/test_tcgen_transfer_artifact.py::test_tcgen_cp_supports_rank3_multi_instruction_layout` (other-assertion=1)
- `tests/numsim/runtime/test_compare_predicates.py::test_compare_instruction_predicates` (other-assertion=2)
- `tests/numsim/runtime/test_cvt_carriers.py::test_cvt_carriers_truncate_extend_and_gate_reads` (other-assertion=3)
- `tests/numsim/runtime/test_discard.py::test_discard_indeterminate_read_and_alignment` (other-assertion=1)
- `tests/numsim/runtime/test_launch_resource_facts.py::test_pointer_bits_preserve_binding_address_and_subview_offset` (other-assertion=1)
- `tests/numsim/runtime/test_mbarrier_maintenance.py::test_maintenance_active_addresses_still_checked` (other-assertion=1)
- `tests/numsim/runtime/test_mbarrier_maintenance.py::test_maintenance_predicates_and_carriers` (other-assertion=1)
- `tests/numsim/runtime/test_memory_coverage_next.py::test_no_complete_rejects_exhausted_arrivals` (other-assertion=2)
- `tests/numsim/runtime/test_memory_sync_coverage.py::test_memory_sync_extensions` (other-assertion=1 pass=5 racecheck-verdict=1)
- `tests/numsim/runtime/test_non_tensor_bulk_forms.py::test_st_bulk_size_is_evaluated_per_issuing_lane` (other-assertion=1)
- `tests/numsim/runtime/test_scalar_control.py::test_mapa_and_cvta_expose_the_device_validated_integer_bits` (other-assertion=1)
- `tests/numsim/runtime/test_sm107_register_predicates.py::test_sm107_register_predicates` (other-assertion=1 pass=1)
- `tests/numsim/runtime/test_tile_general_semantics.py::test_right_aligned_buffer_broadcast_maps_destination_coordinates` (other-assertion=1)
- `tests/numsim/runtime/test_tile_reduction_variants.py::test_local_collective_uses_lexicographic_order` (other-assertion=1)
- `tests/numsim/runtime/test_tile_reduction_variants.py::test_maxmin_uses_canonical_lexicographic_nan_and_signed_zero_order` (other-assertion=1)
- `tests/numsim/runtime/test_tile_reduction_variants.py::test_warp_collective_reduction_follows_physical_lane_ownership` (other-assertion=1)
- `tests/numsim/runtime/test_tile_unary_codegen.py::test_warpgroup_shared_owner_is_independent_of_vector_chunk` (other-assertion=1)
- `tests/numsim/runtime/test_tma_atomicity.py::test_tma_atomicity_direction_restrictions` (other-assertion=1)

</details>

<details><summary>lowering-rejects: 6 functions, owner lowering (W1)</summary>

- `tests/numsim/integration/test_tcgen_transfer_artifact.py::test_tcgen_cp_bitcasts_uint8_scale_payload_into_float8_tmem` (lowering-rejects=1)
- `tests/numsim/integration/test_tcgen_transfer_artifact.py::test_tcgen_cp_cta_group2_reads_and_writes_each_cta_scale_backing` (lowering-rejects=1)
- `tests/numsim/integration/test_tcgen_transfer_artifact.py::test_tcgen_cp_cta_group2_routes_each_pair_in_four_cta_cluster` (lowering-rejects=1)
- `tests/numsim/runtime/test_dense_mma_forms.py::test_legacy_m16n8k32_int8_reuses_dense_form_and_engine` (lowering-rejects=1)
- `tests/numsim/runtime/test_matrix_memory_domain_oracle.py::test_legacy_ldmatrix_x1_domain_matches_independent_fragment_mapping` (lowering-rejects=1)
- `tests/numsim/runtime/test_tile_general_semantics.py::test_mxfp4_uses_ue8m0_scales_over_32_element_vectors` (lowering-rejects=1)

</details>

<details><summary>v2-accepts-legacy-rejection: 1 functions, owner lowering / interp</summary>

- `tests/numsim/integration/test_reported_layout_regressions.py::test_explicit_shared_strides_still_reject_an_executed_oob_address` (v2-accepts-legacy-rejection=1)

</details>

<details><summary>pin-message: 19 functions, owner test port (W9)</summary>

- `tests/numsim/integration/test_scheduler_polling_artifact.py::test_finite_for_uses_native_loop_iteration_budget` (pin-message=1)
- `tests/numsim/integration/test_scheduler_polling_artifact.py::test_finite_loop_can_exceed_default_budget_when_configured` (pin-message=1)
- `tests/numsim/integration/test_tmem_artifact.py::test_tmem_dynamic_lease_rejects_invalid_lifetimes_and_ranges` (pin-message=3)
- `tests/numsim/integration/test_tmem_artifact.py::test_tmem_runtime_address_without_a_dynamic_lease_is_rejected` (pin-message=1)
- `tests/numsim/integration/test_warp_ops_artifact.py::test_warp_collectives_reject_invalid_participant_contracts` (pin-message=1)
- `tests/numsim/runtime/test_dynamic_pure_call_runtime_domains.py::test_if_then_else_mixed_pointer_spaces_fail_closed` (pin-message=1)
- `tests/numsim/runtime/test_mbarrier_lane_semantics.py::test_blocking_wait_fails_closed_for_mixed_lane_readiness` (pin-message=1)
- `tests/numsim/runtime/test_memory_coverage_next.py::test_ldu_rejects_nonuniform_or_misaligned_vector` (pin-message=2)
- `tests/numsim/runtime/test_ordering_calls.py::test_default_full_mask_warp_sync_rejects_divergent_execution` (pin-message=1)
- `tests/numsim/runtime/test_ordering_calls.py::test_setmaxnreg_rejects_warp_disagreement_within_one_occurrence` (pin-message=1)
- `tests/numsim/runtime/test_ordering_calls.py::test_setmaxnreg_requires_explicit_warpgroup_sync_before_a_later_call` (pin-message=1)
- `tests/numsim/runtime/test_pointer_slot_arrays.py::test_pointer_array_initialization_and_bounds` (pass=1 pin-message=1 synccheck-verdict=1)
- `tests/numsim/runtime/test_ptx_integer_arithmetic.py::test_ptx_integer_division_by_zero_fails_closed_at_the_faulting_lane` (pin-message=2)
- `tests/numsim/runtime/test_ptx_integer_arithmetic.py::test_ptx_signed_division_overflow_fails_closed` (pin-message=2)
- `tests/numsim/runtime/test_scalar_control.py::test_cuda_pointer_helpers_check_typed_dereference_alignment` (pin-message=1)
- `tests/numsim/runtime/test_scalar_control.py::test_floating_trap_rejects_signed_zero` (pin-message=2)
- `tests/numsim/runtime/test_scalar_control.py::test_integer_trap_predicate_uses_cpp_truth_conversion` (pin-message=1)
- `tests/numsim/runtime/test_scalar_control.py::test_mbarrier_state_token_rejects_a_generation_older_than_the_previous_one` (pin-message=1)
- `tests/numsim/runtime/test_wait_until.py::test_a_candidate_index_outside_its_local_table_is_an_execution_error` (pin-message=1)

</details>

<details><summary>pin-internals: 39 functions, owner test port (W9): delete or port</summary>

- `tests/analysis_tools/shared/test_known_cuda_func_artifact.py::test_known_combine_int_frac_ex2_is_bit_exact` (pin-internals=1)
- `tests/analysis_tools/shared/test_known_cuda_func_artifact.py::test_known_shl_u32_clamp_is_bit_exact` (pin-internals=1)
- `tests/analysis_tools/synccheck/runtime/test_device_additional_payload_ops.py::test_payload_runtime` (other-assertion=1 pin-internals=38 synccheck-verdict=2)
- `tests/numsim/integration/test_deterministic_reexecution.py::test_repeated_full_launch_keeps_outputs_stats_and_poll_order_deterministic` (pin-internals=1)
- `tests/numsim/integration/test_host_prelude.py::test_dynamic_tensor_map_expressions_run_in_the_loaded_artifact_prologue` (pin-internals=1)
- `tests/numsim/integration/test_large_buffer_artifact.py::test_large_register_buffer_table_is_heap_backed_and_executes` (pin-internals=1)
- `tests/numsim/integration/test_register_layout_artifact.py::test_tcgen_atom_layout_uses_warp_and_lane_owners` (pin-internals=1)
- `tests/numsim/integration/test_register_layout_artifact.py::test_wg_local_layout_uses_tid_in_wg_as_owner_and_m_as_register_index` (pin-internals=1)
- `tests/numsim/integration/test_scheduler_polling_artifact.py::test_effectful_scheduler_loop_claims_ready_root_once` (pin-internals=1)
- `tests/numsim/integration/test_scheduler_polling_artifact.py::test_full_continue_still_reaches_time_slice` (pin-internals=1)
- `tests/numsim/integration/test_scheduler_polling_artifact.py::test_initially_false_while_never_executes_body_or_suspends` (pin-internals=1)
- `tests/numsim/integration/test_scheduler_polling_artifact.py::test_nested_while_inner_break_does_not_escape_time_sliced_outer_loop` (pin-internals=1)
- `tests/numsim/integration/test_scheduler_polling_artifact.py::test_scheduler_progress_loop_cas_mismatch_lane_observes_in_body_write_epoch` (pin-internals=1)
- `tests/numsim/integration/test_scheduler_polling_artifact.py::test_scheduler_progress_loop_suspends_until_continuation_publish` (pin-internals=1)
- `tests/numsim/integration/test_scheduler_polling_artifact.py::test_scheduler_progress_loop_suspends_until_done_publish` (pin-internals=1)
- `tests/numsim/integration/test_scheduler_polling_artifact.py::test_short_while_does_not_suspend` (pin-internals=1)
- `tests/numsim/integration/test_scheduler_polling_artifact.py::test_time_slice_preserves_lane_divergence_and_native_locals` (pin-internals=1)
- `tests/numsim/runtime/test_async_group_domain_oracle.py::test_async_group_wait_counts_complete_on_an_empty_queue` (pin-internals=1)
- `tests/numsim/runtime/test_atomic_domain_oracle.py::test_every_cuda_atomic_add_and_cas_form_executes_with_exact_bits` (pin-internals=1)
- `tests/numsim/runtime/test_atomic_f32_noftz.py::test_atomic_f32_noftz` (pin-internals=12)
- `tests/numsim/runtime/test_byte_pointer_view_alias.py::test_byte_pointer_can_form_a_wider_physical_alias` (pin-internals=1)
- `tests/numsim/runtime/test_fetch_register_domain_oracle.py::test_every_fetch_register_form_matches_the_logical_topology` (pin-internals=1)
- `tests/numsim/runtime/test_float8_address_shuffle.py::test_cuda_four_argument_shfl_sync_uses_implicit_warp_size` (pin-internals=1)
- `tests/numsim/runtime/test_float8_address_shuffle.py::test_float8_address_uses_one_byte_physical_pointer` (pin-internals=1)
- `tests/numsim/runtime/test_loop_bounds.py::test_lane_varying_loop_supports_break_and_continue` (pin-internals=1)
- `tests/numsim/runtime/test_loop_bounds.py::test_lane_varying_min_extent_and_step_use_masked_native_loop` (pin-internals=1)
- `tests/numsim/runtime/test_loop_bounds.py::test_static_single_trip_for_preserves_masks_and_loop_path` (pin-internals=1)
- `tests/numsim/runtime/test_loop_bounds.py::test_static_zero_trip_for_omits_native_loop_scaffolding` (pin-internals=1)
- `tests/numsim/runtime/test_loop_bounds.py::test_uniform_loop_var_preserves_tir_dtype_in_select` (pin-internals=1)
- `tests/numsim/runtime/test_loop_bounds.py::test_v0_style_nested_triangular_bounds` (pin-internals=1)
- `tests/numsim/runtime/test_non_tensor_bulk_forms.py::test_bulk_g2s_cluster_accepts_static_true_predicate_as_unconditional` (pin-internals=1)
- `tests/numsim/runtime/test_non_tensor_bulk_forms.py::test_bulk_g2s_cluster_dynamic_predicate_transpiles` (pin-internals=1)
- `tests/numsim/runtime/test_non_tensor_bulk_forms.py::test_raw_bulk_prefetch_preserves_global_memory` (pin-internals=1)
- `tests/numsim/runtime/test_ordering_calls.py::test_griddep_token_crosses_sequential_kernel_phases` (pin-internals=1)
- `tests/numsim/runtime/test_ordering_calls.py::test_ordering_only_calls_preserve_native_source_order` (pin-internals=1)
- `tests/numsim/runtime/test_wait_until.py::test_a_backoff_is_the_cuda_loops_business_and_not_the_engines` (pin-internals=1)
- `tests/numsim/runtime/test_wait_until.py::test_bit_typed_word_matches_raw_spelling` (pin-internals=1)
- `tests/numsim/runtime/test_wait_until.py::test_packed_wait_matches_raw_spelling` (pin-internals=4)
- `tests/numsim/runtime/test_wait_until.py::test_rendezvous_matches_raw_spelling` (pin-internals=1)

</details>

## Per case

| case | numsim | racecheck | synccheck |
| --- | --- | --- | --- |
| `act_and_mul` | match | match | match |
| `alphamoe_fp8_blockscale_qwen3next` | match | match | match |
| `blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin` | match | match | match |
| `bmm_fp8_rubin` | match | match | match |
| `cudnn_sm100_bsa_backward_blk128` | match | match | match |
| `cudnn_sm100_bsa_backward_blk64` | match | match | match |
| `cudnn_sm100_bsa_forward_blk128` | match | match | match |
| `cudnn_sm100_bsa_forward_blk64` | match | match | match |
| `cudnn_sm100_bsa_forward_combine_blk64` | match | match | match |
| `cudnn_sm100_csa_compressor_fwd` | match | match | match |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_amax` | match | match | match |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant` | match | match | match |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | match | match | match |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | match | match | match |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | match | match | match |
| `cudnn_sm100_dsa_sparse_attention_backward` | match | match | match |
| `cudnn_sm100_flex_attention_backward` | match | match | match |
| `cudnn_sm100_flex_attention_forward_hd256` | match | match | match |
| `cudnn_sm100_gdn2_bprop_f16` | match | match | match |
| `cudnn_sm100_gdn2_prefill_f16` | match | match | match |
| `cudnn_sm100_gdn2_recompute_f16` | match | match | match |
| `cudnn_sm100_gdn_bprop_f16` | match | match | match |
| `cudnn_sm100_gdn_prefill_f16` | match | match | match |
| `cudnn_sm100_gdn_recompute_f16` | match | match | match |
| `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` | match | match | match |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | match | match | match |
| `cudnn_sm100_kda_bprop_f16` | match | match | match |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | match | match | match |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | match | match | match |
| `cudnn_sm103_flex_attention_forward` | match | match | match |
| `deepgemm_sm100_fp4_mqa_logits` | match | match | match |
| `deepgemm_sm100_fp4_paged_mqa_logits` | match | match | match |
| `deepgemm_sm100_fp8_bmm` | match | match | match |
| `deepgemm_sm100_fp8_gemm_1d1d` | match | match | match |
| `deepgemm_sm100_fp8_mqa_logits` | match | match | match |
| `deepgemm_sm100_fp8_paged_mqa_logits` | match | match | match |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | match | match | match |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | match | match | match |
| `deepgemm_sm100_m_grouped_fp8_gemm_masked` | match | match | match |
| `deepgemm_sm100_tf32_hc_prenorm_gemm` | match | match | match |
| `dense_blockscaled_gemm_sm107` | match | match | match |
| `fast_topk_clusters` | match | match | match |
| `fastcu_nvfp4_gemm_gb300` | match | match | match |
| `filtered_topk` | match | match | match |
| `flash_attention4` | match | match | match |
| `flash_attention4_fp4` | match | match | match |
| `flash_attention_backward_sm100` | match | match | match |
| `flash_mla_sparse_fwd` | match | match | match |
| `flashinfer_add_rmsnorm_fp4quant` | match | match | match |
| `flashinfer_fused_add_rmsnorm` | match | match | match |
| `flashinfer_fused_add_rmsnorm_quant` | match | match | match |
| `flashinfer_fused_dit_layernorm` | match | match | match |
| `flashinfer_layernorm` | match | match | match |
| `flashinfer_qk_rmsnorm` | match | match | match |
| `flashinfer_rmsnorm` | match | match | match |
| `flashinfer_rmsnorm_fp4quant` | match | match | match |
| `flashinfer_rmsnorm_quant` | match | match | match |
| `fp16_bf16_gemm` | no oracle | match | no oracle |
| `gdn_cp_prefill_sm100` | match | match | match |
| `gdn_decode_bf16_ilp4` | match | match | match |
| `gdn_decode_bf16_wide_vec_mtp` | match | match | match |
| `gdn_decode_bf16_wide_vec_t1` | match | match | match |
| `gdn_decode_fp32_mtp_warp` | match | match | match |
| `gdn_prefill_sm100` | match | match | match |
| `grouped_gemm_masked_rubin` | match | match | match |
| `kda_backward_packed` | match | match | match |
| `kda_decode_multishape` | match | match | match |
| `kda_forward_portfolio_multishape` | match | match | incomplete [V2C-31] |
| `merge_state` | match | match | match |
| `mla_dsv4_multishape` | no oracle | no oracle | no oracle |
| `msa_decode_multishape` | match | match | match |
| `msa_prefill_multishape` | match | match | match |
| `msa_sparse_atten_fwd_combine_sm100` | match | match | match |
| `msa_sparse_atten_fwd_nvfp4_kv_sm100` | match | match | match |
| `msa_sparse_atten_fwd_sm100` | match | match | match |
| `msa_sparse_prepare_flat_schedule_sm100` | match | match | match |
| `msa_sparse_prepare_fwd_split_atomic_sm100` | match | match | match |
| `mxfp4_quantize` | match | match | match |
| `mxfp8_quantize` | match | match | match |
| `nvfp4_gemm` | match | match | match |
| `nvfp4_quantize` | match | match | match |
| `nvfp4_quantize_per_token` | match | match | match |
| `radix_topk_multi_cta` | match | match | match |
| `radix_topk_single_cta` | match | match | match |
| `recurrent_kda_decode_grouped` | match | match | match |
| `recurrent_kda_decode_one_warp` | match | match | match |
| `rmsnorm` | match | match | match |
| `selective_state_update_mtp_horizontal` | match | match | match |
| `selective_state_update_mtp_simple` | match | match | match |
| `selective_state_update_mtp_vertical` | match | match | match |
| `selective_state_update_stp_horizontal` | match | match | match |
| `selective_state_update_stp_simple` | match | match | match |
| `selective_state_update_stp_vertical` | match | match | match |
| `silu_and_mul_nvfp4_experts_quantize` | match | match | match |
| `sm100_fp8_fp4_mega_moe` | match | match | match |
| `sparse_flashmla_decode_head64` | match | match | match |
| `sparse_flashmla_prefill_head128_phase1` | match | match | match |
| `sparse_flashmla_prefill_head128_small_topk_phase1` | match | match | match |
| `sparse_flashmla_prefill_head64_phase1` | match | match | match |
| `stable_sort_topk_by_value` | match | match | match |
| `vsa_multishape` | match | match | match |
