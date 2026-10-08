---
orphan: true
---

# v2 conformance status

Generated 2026-10-08 at df0e1e4 (sweep 7) by `scripts/numsim-v2/status.py --run -n 16`:
every canonical case (`tests/numsim/corpus/canonical_cases.py`) x {numsim, racecheck,
synccheck} run under `NUMSIM_IMPL=v2` and compared with the legacy snapshots in
`tirx_harness/tests/conformance/` (v2 may add source anchors to diagnostics legacy recorded
without one; see `snapshot.relax_unanchored`). Regenerate with `status.py` before acting on a
row; this page is the per-case view of that run.

Status values: **match** (normalized snapshot equal to the legacy oracle); **match (delta:
rows)** (equal to the `<mode>.delta.json` snapshot that the cited behaviour-delta rows justify;
row ids are file-qualified, `scripts/numsim-v2/delta_rows.py`); **no-oracle** (legacy fails the
case, so there is nothing to compare until the post-deletion regeneration); **fail**.

## Summary

| mode | match | match (delta) | no-oracle | fail |
| --- | --- | --- | --- | --- |
| numsim | 98 | 1 | 2 | 0 |
| racecheck | 86 | 14 | 1 | 0 |
| synccheck | 97 | 2 | 2 | 0 |

Every case with an oracle matches. Open conformance issues: none. Closed since sweep 5:
V2C-31 (`kda_forward_portfolio_multishape` synccheck budget, W6 `singleton_persistent`
rule), the `alphamoe_fp8_blockscale_qwen3next` racecheck footprint (racecheck R3 + T18), and
the `sm100_fp8_fp4_mega_moe` racecheck instance change after engine commit f1d920d
(racecheck T19, X4; deterministic across worker counts). V2C-TF1 (TVM tile-dispatch coverage)
is closed: `tests/numsim/v2/tile_forms/test_legacy_compiled_tile_forms.py` has 63 passing
cases.

At deletion, the 45 base snapshots that match only through `relax_unanchored` are regenerated
from v2 (`pytest tests/conformance --update-snapshots`, trailer `Snapshot-Regen: schema
relax_unanchored projection`; W9's step-5 checklist), after the 17 delta files are folded
(`fold_snapshot_deltas.py`).

## Decisions deferred to after the legacy deletion

| decision | current v2 behaviour | why deferred |
| --- | --- | --- |
| Enforce the hardware cluster-size limit (16 CTAs, non-portable, sm_90/sm_100) | Legacy engine limit: more than 64 CTAs per cluster fails closed at transpile | 14 sweep kernels use 20-CTA clusters that legacy accepted; enforcing needs a delta row and a test migration. See numsim-isa-answers.md, "Cluster size". |

## Cases without a legacy result

| case | legacy | v2 status |
| --- | --- | --- |
| `fp16_bf16_gemm` | Legacy lowering raises `UnsupportedTIRxError` (host prelude: `Evaluate` before `tirx.device_entry`) in all three modes. | numsim clean, outputs match the reference; synccheck clean (skipped: no oracle until the post-deletion regeneration). Racecheck has a v2 oracle: `racecheck.delta.json` (racecheck B7, TMEM-teardown handshake). |
| `mla_dsv4_multishape` | Same legacy host-prelude `UnsupportedTIRxError`. | numsim, racecheck and synccheck clean; skipped until the post-deletion regeneration from v2. |

## Public-API legacy tests under v2

The category-A, `surface = public` tests of `scripts/numsim-v2/coverage/test_classification.csv`
run unchanged with `NUMSIM_IMPL=v2` (`tests/conftest.py` rebinds the public names of
`tirx_harness.numsim` and `tirx_harness.racecheck`/`synccheck` to `numsim.v2`).

- 562 of 761 items pass unchanged
  (323 of 435 functions fully), from
  `scripts/numsim-v2/coverage/v2_public_status.tsv`.
- Counting a function as covered when every v2 copy mapped to it in
  `scripts/numsim-v2/coverage/v2_ports_*.tsv` passes: 760 of 761 items are pass-or-covered
  (test-migration.md, "Current status"). The 4 GPU-only functions skip on CPU.
- The per-function failure classes are in `v2_public_status.tsv` (`v2_status` column); the
  step-5 blockers are listed by `scripts/numsim-v2/retire_legacy.py --dry-run`.

## Per case

| case | numsim | racecheck | synccheck |
| --- | --- | --- | --- |
| `act_and_mul` | match | match | match |
| `alphamoe_fp8_blockscale_qwen3next` | match | match (delta: racecheck R3, racecheck T18) | match |
| `blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin` | match | match | match |
| `bmm_fp8_rubin` | match | match (delta: racecheck B7) | match |
| `cudnn_sm100_bsa_backward_blk128` | match | match | match |
| `cudnn_sm100_bsa_backward_blk64` | match | match | match |
| `cudnn_sm100_bsa_forward_blk128` | match | match | match |
| `cudnn_sm100_bsa_forward_blk64` | match | match | match |
| `cudnn_sm100_bsa_forward_combine_blk64` | match | match | match |
| `cudnn_sm100_csa_compressor_fwd` | match | match | match |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_amax` | match | match | match |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant` | match | match (delta: racecheck B7) | match |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | match | match (delta: racecheck B7) | match |
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
| `deepgemm_sm100_fp8_gemm_1d1d` | match | match (delta: racecheck B7) | match |
| `deepgemm_sm100_fp8_mqa_logits` | match | match | match |
| `deepgemm_sm100_fp8_paged_mqa_logits` | match | match | match |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | match | match | match |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | match | match | match |
| `deepgemm_sm100_m_grouped_fp8_gemm_masked` | match | match | match |
| `deepgemm_sm100_tf32_hc_prenorm_gemm` | match (delta: numsim H5) | match (delta: numsim H5) | match (delta: numsim H5) |
| `dense_blockscaled_gemm_sm107` | match | match | match |
| `fast_topk_clusters` | match | match | match |
| `fastcu_nvfp4_gemm_gb300` | match | match (delta: racecheck B7) | match |
| `filtered_topk` | match | match | match |
| `flash_attention4` | match | match | match |
| `flash_attention4_fp4` | match | match | match |
| `flash_attention_backward_sm100` | match | match (delta: racecheck B7, racecheck X4) | match |
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
| `fp16_bf16_gemm` | no-oracle | match (delta: racecheck B7) | no-oracle |
| `gdn_cp_prefill_sm100` | match | match | match |
| `gdn_decode_bf16_ilp4` | match | match | match |
| `gdn_decode_bf16_wide_vec_mtp` | match | match | match |
| `gdn_decode_bf16_wide_vec_t1` | match | match | match |
| `gdn_decode_fp32_mtp_warp` | match | match | match |
| `gdn_prefill_sm100` | match | match | match |
| `grouped_gemm_masked_rubin` | match | match | match |
| `kda_backward_packed` | match | match (delta: racecheck X4) | match |
| `kda_decode_multishape` | match | match | match |
| `kda_forward_portfolio_multishape` | match | match | match |
| `merge_state` | match | match | match |
| `mla_dsv4_multishape` | no-oracle | no-oracle | no-oracle |
| `msa_decode_multishape` | match | match | match |
| `msa_prefill_multishape` | match | match | match (delta: sync S1) |
| `msa_sparse_atten_fwd_combine_sm100` | match | match | match |
| `msa_sparse_atten_fwd_nvfp4_kv_sm100` | match | match | match |
| `msa_sparse_atten_fwd_sm100` | match | match | match |
| `msa_sparse_prepare_flat_schedule_sm100` | match | match | match |
| `msa_sparse_prepare_fwd_split_atomic_sm100` | match | match | match |
| `mxfp4_quantize` | match | match | match |
| `mxfp8_quantize` | match | match | match |
| `nvfp4_gemm` | match | match (delta: racecheck B7) | match |
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
| `sm100_fp8_fp4_mega_moe` | match | match (delta: racecheck B1, racecheck B7, racecheck R4, racecheck T19, racecheck X4) | match |
| `sparse_flashmla_decode_head64` | match | match | match |
| `sparse_flashmla_prefill_head128_phase1` | match | match (delta: racecheck B7, racecheck X4) | match |
| `sparse_flashmla_prefill_head128_small_topk_phase1` | match | match (delta: racecheck B7) | match |
| `sparse_flashmla_prefill_head64_phase1` | match | match | match |
| `stable_sort_topk_by_value` | match | match | match |
| `vsa_multishape` | match | match | match |
