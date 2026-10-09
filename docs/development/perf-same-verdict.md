# Legacy vs v2 per tool, verdict-identical cases only (engine 6ba4190)

Source: `backend-comparison.json` (`head.cases`, engine wall, min of the samples per cell, same host/session for both columns). A (case, mode) is excluded when a `<mode>.delta.json` existed before step 5 (`git ls-tree 79f04eb^`): legacy and v2 disagreed on verdict or finding set there (racecheck B1/B7/R3/R4/T18/T19/X4, numsim H5, sync S1). The Mega-MoE perf configs are in `backend-comparison.md` (racecheck verdicts differ there; numsim verdicts agree).

## Summary (geometric mean over verdict-identical rows)

| mode | workers | rows | v2 vs legacy (geomean, 6ba4190) | rows where v2 is >10% slower (6ba4190) | v2 now (243f9f4) vs legacy | rows >10% slower now |
| --- | --- | --- | --- | --- | --- | --- |
| numsim | 1 | 30 | 4.12x faster | 1 | 7.49x faster | 0 |
| numsim | 8 | 30 | 3.44x faster | 0 | 6.67x faster | 0 |
| numsim | 32 | 30 | 2.95x faster | 0 | 6.70x faster | 0 |
| racecheck | 1 | 24 | 2.96x faster | 0 | 5.94x faster | 0 |
| racecheck | 8 | 23 | 1.87x faster | 5 | 3.61x faster | 3 |
| racecheck | 32 | 23 | 1.77x faster | 7 | 3.58x faster | 3 |
| synccheck | 1 | 29 | 3.43x faster | 0 | 6.67x faster | 0 |
| synccheck | 8 | 29 | 3.00x faster | 0 | 5.87x faster | 0 |
| synccheck | 32 | 29 | 2.82x faster | 3 | 6.11x faster | 0 |

**v2 now (243f9f4), racecheck and synccheck** (W16, 2026-10-09): private release build of a `git archive 243f9f4` copy; `Engine(max_workers=w, native_loop_iteration_budget=10_000_000)`, fresh engine per sample, every phase via `run_racecheck_phase` / `run_synccheck_phase` (synccheck with the conformance coverage bounds and resource limits), engine wall, min of 3 samples interleaved over 1/8/32 workers; serial racecheck checker (`FORK_JOIN` off), default collector threads (W14's parallel GC not in HEAD). Host: 256 CPUs, 1-minute load 4.3-12.7 during the run. The legacy and 6ba4190 columns are unchanged (legacy is not re-measurable: deleted in 79f04eb). Rows still >10% slower than legacy now: `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` racecheck 8 (1.25x); `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` racecheck 8 (1.23x); `kda_decode_multishape` racecheck 8 (1.66x); `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` racecheck 32 (1.26x); `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` racecheck 32 (1.45x); `kda_decode_multishape` racecheck 32 (1.58x).

**v2 now (243f9f4), numsim** (W13, 2026-10-09):
- Build: private release build of a `git archive 243f9f4` copy.
- Driver: the backend-comparison driver (`bench_backends.py` from 7c7d049, `run --variants interp`), `Engine(max_workers=w)` as in the 6ba4190 column.
- Timing: engine wall, min of 3 samples per worker count; every run matched its conformance oracle and the reference (`compare` ok).
- Host: 256 CPUs; 1-minute load mostly 4.2-14 during the run, up to 21 for `gdn_cp_prefill_sm100` and 30-36 for `gdn_prefill_sm100`.
- No numsim row is >10% slower than legacy now; `kda_backward_packed` (formerly 1.26x slower at 1 worker) is now 5.0x faster (1.01 s -> 202 ms).

## Excluded (verdict/finding delta vs legacy)

- `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` racecheck
- `deepgemm_sm100_fp8_gemm_1d1d` racecheck
- `kda_backward_packed` racecheck
- `sparse_flashmla_prefill_head128_phase1` racecheck

## Mega-MoE perf workloads (verdict-identical rows only)

Separate configuration set (148 SMs, `native_loop_iteration_budget=10_000_000`, one run each, 900 s cap; `backend-comparison.md` "Mega-MoE perf workloads"). Only numsim verdicts agree (both `clean`); every racecheck row differs (legacy `review`, v2 `error` per B7/T19, or v2 timed out at 6ba4190) and is excluded.

| config | mode | workers | legacy | v2 (6ba4190) | v2 vs legacy (6ba4190) | v2 now (243f9f4) | now vs legacy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small | numsim | 1 / 16 / 32 | 3.1 / 1.5 / 1.5 s | 0.6 / 0.4 / 0.4 s | **5.2x / 3.8x / 3.8x faster** | 0.50 / 0.19 / 0.23 s | **6.2x / 7.9x / 6.5x faster** |
| twenty_four_experts | numsim | 1 / 16 / 32 | 4.8 / 1.8 / 1.7 s | 1.7 / 0.7 / 0.7 s | **2.8x / 2.6x / 2.4x faster** | 1.27 / 0.45 / 0.48 s | **3.8x / 4.0x / 3.5x faster** |
| medium | numsim | 1 | 35.7 s | 36.0 s | 1.01x (even) | 22.2 s | **1.61x faster** |
| medium | numsim | 16 / 32 | 5.0 / 3.7 s | 8.2 / 7.7 s | 1.6x / 2.1x slower | 5.75 / 6.92 s | 1.15x / 1.87x slower |
| large | numsim | 1 / 16 / 32 | not measured | 75.1 / 14.0 / 13.6 s | — | 40.4 / 10.3 / 11.2 s | — |
| max config (`test_mega_moe_numsim_max_config`) | numsim | 16 | 185 s (load 40-50) | 366 s (before b36e44e, load 17-21) | 1.98x slower | 271.6 s (load 7-18) | 1.47x slower |

v2 now: `bench_backends.py mega --impls interp` on the 243f9f4 build (148 SMs, `native_loop_iteration_budget=10_000_000`), min of 2 runs per cell (load 4.6-15). The max config is one run of the perf test (its `perf_metrics` engine time). Mega-MoE numsim at 16 and 32 workers stays slower than legacy on medium: the remaining gap is per-partition CPU inflation at 16+ workers and the MMA landing (see engine-review.md).

## numsim

| case | workers | legacy | v2 (6ba4190) | v2 vs legacy (6ba4190) | v2 now (243f9f4) | now vs legacy |
| --- | --- | --- | --- | --- | --- | --- |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | 1 | 209 ms | 40 ms | **5.20x faster** | 27 ms | **7.62x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 1 | 130 ms | 38 ms | **3.43x faster** | 22 ms | **5.92x faster** |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 1 | 233 ms | 35 ms | **6.71x faster** | 19 ms | **12.34x faster** |
| `cudnn_sm100_flex_attention_backward` | 1 | 199 ms | 22 ms | **9.04x faster** | 16 ms | **12.44x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 1 | 158 ms | 133 ms | **1.18x faster** | 36 ms | **4.39x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 1 | 250 ms | 93 ms | **2.68x faster** | 40 ms | **6.21x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 1 | 689 ms | 393 ms | **1.75x faster** | 220 ms | **3.13x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 1 | 357 ms | 137 ms | **2.60x faster** | 41 ms | **8.66x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` | 1 | 65.47 s | 18.68 s | **3.51x faster** | 4.31 s | **15.17x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 1 | 32.18 s | 15.78 s | **2.04x faster** | 3.23 s | **9.96x faster** |
| `cudnn_sm100_kda_bprop_f16` | 1 | 160 ms | 100 ms | **1.60x faster** | 27 ms | **6.02x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 1 | 1.92 s | 266 ms | **7.21x faster** | 167 ms | **11.49x faster** |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 1 | 1.71 s | 326 ms | **5.26x faster** | 117 ms | **14.59x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 1 | 742 ms | 213 ms | **3.48x faster** | 102 ms | **7.25x faster** |
| `deepgemm_sm100_fp8_gemm_1d1d` | 1 | 35 ms | 6 ms | **5.81x faster** | 4 ms | **9.36x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 1 | 605 ms | 64 ms | **9.50x faster** | 39 ms | **15.48x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 1 | 80 ms | 19 ms | **4.13x faster** | 11 ms | **7.39x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 1 | 1.26 s | 202 ms | **6.23x faster** | 111 ms | **11.34x faster** |
| `filtered_topk` | 1 | 70 ms | 7 ms | **9.91x faster** | 7 ms | **10.41x faster** |
| `flash_attention4` | 1 | 8.75 s | 1.11 s | **7.92x faster** | 826 ms | **10.60x faster** |
| `gdn_cp_prefill_sm100` | 1 | 5.31 s | 2.17 s | **2.45x faster** | 1.52 s | **3.50x faster** |
| `gdn_decode_bf16_ilp4` | 1 | 238 ms | 34 ms | **7.04x faster** | 28 ms | **8.58x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 1 | 1.46 s | 311 ms | **4.69x faster** | 252 ms | **5.79x faster** |
| `gdn_prefill_sm100` | 1 | 1.69 s | 237 ms | **7.15x faster** | 230 ms | **7.35x faster** |
| `kda_backward_packed` | 1 | 1.01 s | 1.27 s | 1.26x slower | 202 ms | **5.00x faster** |
| `kda_decode_multishape` | 1 | 178 ms | 55 ms | **3.22x faster** | 47 ms | **3.75x faster** |
| `msa_sparse_atten_fwd_sm100` | 1 | 163 ms | 20 ms | **8.25x faster** | 19 ms | **8.37x faster** |
| `recurrent_kda_decode_one_warp` | 1 | 5.30 s | 532 ms | **9.95x faster** | 652 ms | **8.13x faster** |
| `selective_state_update_stp_simple` | 1 | 33 ms | 6 ms | **5.19x faster** | 5 ms | **6.14x faster** |
| `sparse_flashmla_prefill_head128_phase1` | 1 | 123 ms | 74 ms | **1.66x faster** | 54 ms | **2.29x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | 8 | 165 ms | 75 ms | **2.19x faster** | 30 ms | **5.43x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 8 | 61 ms | 36 ms | **1.72x faster** | 24 ms | **2.52x faster** |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 8 | 78 ms | 31 ms | **2.54x faster** | 19 ms | **4.10x faster** |
| `cudnn_sm100_flex_attention_backward` | 8 | 192 ms | 21 ms | **9.27x faster** | 17 ms | **11.50x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 8 | 155 ms | 135 ms | **1.14x faster** | 36 ms | **4.33x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 8 | 243 ms | 104 ms | **2.33x faster** | 40 ms | **6.13x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 8 | 675 ms | 467 ms | **1.44x faster** | 219 ms | **3.09x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 8 | 382 ms | 145 ms | **2.64x faster** | 41 ms | **9.29x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` | 8 | 13.30 s | 3.94 s | **3.37x faster** | 925 ms | **14.38x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 8 | 8.70 s | 3.49 s | **2.49x faster** | 734 ms | **11.85x faster** |
| `cudnn_sm100_kda_bprop_f16` | 8 | 159 ms | 99 ms | **1.61x faster** | 26 ms | **6.02x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 8 | 1.24 s | 185 ms | **6.70x faster** | 94 ms | **13.23x faster** |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 8 | 656 ms | 120 ms | **5.46x faster** | 47 ms | **13.85x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 8 | 407 ms | 228 ms | **1.78x faster** | 87 ms | **4.70x faster** |
| `deepgemm_sm100_fp8_gemm_1d1d` | 8 | 29 ms | 7 ms | **4.23x faster** | 4 ms | **7.74x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 8 | 323 ms | 96 ms | **3.37x faster** | 42 ms | **7.73x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 8 | 56 ms | 26 ms | **2.13x faster** | 17 ms | **3.23x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 8 | 645 ms | 275 ms | **2.34x faster** | 95 ms | **6.78x faster** |
| `filtered_topk` | 8 | 67 ms | 8 ms | **8.21x faster** | 7 ms | **9.93x faster** |
| `flash_attention4` | 8 | 2.06 s | 303 ms | **6.80x faster** | 250 ms | **8.23x faster** |
| `gdn_cp_prefill_sm100` | 8 | 2.17 s | 447 ms | **4.86x faster** | 319 ms | **6.79x faster** |
| `gdn_decode_bf16_ilp4` | 8 | 97 ms | 11 ms | **9.05x faster** | 8 ms | **11.61x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 8 | 471 ms | 60 ms | **7.90x faster** | 37 ms | **12.57x faster** |
| `gdn_prefill_sm100` | 8 | 1.69 s | 239 ms | **7.06x faster** | 230 ms | **7.35x faster** |
| `kda_backward_packed` | 8 | 905 ms | 917 ms | 1.01x slower | 156 ms | **5.81x faster** |
| `kda_decode_multishape` | 8 | 58 ms | 29 ms | **2.02x faster** | 23 ms | **2.56x faster** |
| `msa_sparse_atten_fwd_sm100` | 8 | 158 ms | 21 ms | **7.49x faster** | 19 ms | **8.19x faster** |
| `recurrent_kda_decode_one_warp` | 8 | 2.11 s | 167 ms | **12.65x faster** | 194 ms | **10.87x faster** |
| `selective_state_update_stp_simple` | 8 | 19 ms | 4 ms | **5.21x faster** | 4 ms | **5.17x faster** |
| `sparse_flashmla_prefill_head128_phase1` | 8 | 123 ms | 77 ms | **1.61x faster** | 53 ms | **2.30x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | 32 | 130 ms | 71 ms | **1.83x faster** | 30 ms | **4.30x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 32 | 60 ms | 50 ms | **1.20x faster** | 23 ms | **2.56x faster** |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 32 | 80 ms | 42 ms | **1.89x faster** | 19 ms | **4.29x faster** |
| `cudnn_sm100_flex_attention_backward` | 32 | 213 ms | 24 ms | **8.78x faster** | 18 ms | **12.14x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 32 | 168 ms | 151 ms | **1.11x faster** | 35 ms | **4.79x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 32 | 249 ms | 109 ms | **2.29x faster** | 41 ms | **6.08x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 32 | 725 ms | 438 ms | **1.66x faster** | 218 ms | **3.33x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 32 | 393 ms | 152 ms | **2.59x faster** | 43 ms | **9.23x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` | 32 | 7.33 s | 3.50 s | **2.10x faster** | 707 ms | **10.37x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 32 | 7.33 s | 3.31 s | **2.21x faster** | 555 ms | **13.21x faster** |
| `cudnn_sm100_kda_bprop_f16` | 32 | 160 ms | 105 ms | **1.52x faster** | 28 ms | **5.74x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 32 | 1.39 s | 255 ms | **5.46x faster** | 122 ms | **11.41x faster** |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 32 | 627 ms | 109 ms | **5.74x faster** | 46 ms | **13.71x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 32 | 395 ms | 320 ms | **1.24x faster** | 85 ms | **4.62x faster** |
| `deepgemm_sm100_fp8_gemm_1d1d` | 32 | 30 ms | 8 ms | **3.68x faster** | 4 ms | **7.97x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 32 | 326 ms | 172 ms | **1.90x faster** | 41 ms | **7.90x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 32 | 56 ms | 38 ms | **1.47x faster** | 16 ms | **3.54x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 32 | 755 ms | 467 ms | **1.62x faster** | 95 ms | **7.92x faster** |
| `filtered_topk` | 32 | 68 ms | 10 ms | **6.80x faster** | 7 ms | **10.25x faster** |
| `flash_attention4` | 32 | 1.29 s | 430 ms | **3.00x faster** | 375 ms | **3.44x faster** |
| `gdn_cp_prefill_sm100` | 32 | 1.43 s | 339 ms | **4.22x faster** | 207 ms | **6.90x faster** |
| `gdn_decode_bf16_ilp4` | 32 | 89 ms | 9 ms | **9.50x faster** | 7 ms | **13.29x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 32 | 384 ms | 42 ms | **9.05x faster** | 28 ms | **13.78x faster** |
| `gdn_prefill_sm100` | 32 | 1.69 s | 246 ms | **6.89x faster** | 202 ms | **8.36x faster** |
| `kda_backward_packed` | 32 | 964 ms | 1.03 s | 1.07x slower | 156 ms | **6.17x faster** |
| `kda_decode_multishape` | 32 | 61 ms | 38 ms | **1.62x faster** | 21 ms | **2.90x faster** |
| `msa_sparse_atten_fwd_sm100` | 32 | 177 ms | 22 ms | **8.13x faster** | 19 ms | **9.12x faster** |
| `recurrent_kda_decode_one_warp` | 32 | 2.21 s | 128 ms | **17.21x faster** | 130 ms | **16.98x faster** |
| `selective_state_update_stp_simple` | 32 | 19 ms | 5 ms | **3.96x faster** | 4 ms | **4.77x faster** |
| `sparse_flashmla_prefill_head128_phase1` | 32 | 127 ms | 80 ms | **1.58x faster** | 54 ms | **2.35x faster** |

## racecheck

| case | workers | legacy | v2 (6ba4190) | v2 vs legacy (6ba4190) | v2 now (243f9f4) | now vs legacy |
| --- | --- | --- | --- | --- | --- | --- |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 1 | 275 ms | 302 ms | 1.10x slower | 127 ms | **2.16x faster** |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 1 | 495 ms | 288 ms | **1.72x faster** | 131 ms | **3.78x faster** |
| `cudnn_sm100_flex_attention_backward` | 1 | 340 ms | 96 ms | **3.54x faster** | 53 ms | **6.36x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 1 | 1.13 s | 229 ms | **4.92x faster** | 79 ms | **14.22x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 1 | 1.72 s | 268 ms | **6.42x faster** | 143 ms | **12.04x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 1 | 5.13 s | 956 ms | **5.37x faster** | 285 ms | **18.03x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 1 | 2.87 s | 304 ms | **9.44x faster** | 140 ms | **20.56x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` | 1 | 121.14 s | 40.30 s | **3.01x faster** | 13.88 s | **8.73x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 1 | 45.73 s | 35.19 s | **1.30x faster** | 13.79 s | **3.32x faster** |
| `cudnn_sm100_kda_bprop_f16` | 1 | 1.17 s | 198 ms | **5.92x faster** | 57 ms | **20.70x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 1 | 1.89 s | 1.22 s | **1.55x faster** | 474 ms | **3.99x faster** |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 1 | 1.57 s | 587 ms | **2.67x faster** | 259 ms | **6.07x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 1 | 1.60 s | 734 ms | **2.18x faster** | 362 ms | **4.42x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 1 | 1.56 s | 432 ms | **3.62x faster** | 183 ms | **8.51x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 1 | 182 ms | 85 ms | **2.14x faster** | 47 ms | **3.91x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 1 | 3.98 s | 2.39 s | **1.67x faster** | 1.07 s | **3.71x faster** |
| `filtered_topk` | 1 | 123 ms | 40 ms | **3.07x faster** | 33 ms | **3.77x faster** |
| `flash_attention4` | 1 | 16.44 s | 3.45 s | **4.76x faster** | 2.13 s | **7.71x faster** |
| `gdn_decode_bf16_ilp4` | 1 | 352 ms | 97 ms | **3.62x faster** | 71 ms | **4.93x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 1 | 2.13 s | 1.36 s | **1.57x faster** | 854 ms | **2.50x faster** |
| `kda_decode_multishape` | 1 | 494 ms | 344 ms | **1.44x faster** | 216 ms | **2.29x faster** |
| `msa_sparse_atten_fwd_sm100` | 1 | 241 ms | 45 ms | **5.34x faster** | 30 ms | **8.12x faster** |
| `recurrent_kda_decode_one_warp` | 1 | 8.94 s | 1.94 s | **4.62x faster** | 1.51 s | **5.90x faster** |
| `selective_state_update_stp_simple` | 1 | 57 ms | 20 ms | **2.80x faster** | 16 ms | **3.61x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 8 | 110 ms | 275 ms | 2.50x slower | 137 ms | 1.25x slower |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 8 | 204 ms | 293 ms | 1.43x slower | 137 ms | **1.49x faster** |
| `cudnn_sm100_flex_attention_backward` | 8 | 350 ms | 102 ms | **3.45x faster** | 50 ms | **7.05x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 8 | 1.14 s | 234 ms | **4.86x faster** | 78 ms | **14.54x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 8 | 1.80 s | 251 ms | **7.17x faster** | 142 ms | **12.68x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 8 | 4.89 s | 795 ms | **6.15x faster** | 286 ms | **17.07x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 8 | 2.79 s | 308 ms | **9.07x faster** | 138 ms | **20.15x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 8 | 9.07 s | 20.05 s | 2.21x slower | 11.20 s | 1.23x slower |
| `cudnn_sm100_kda_bprop_f16` | 8 | 1.17 s | 194 ms | **6.03x faster** | 56 ms | **20.85x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 8 | 950 ms | 1.02 s | 1.07x slower | 383 ms | **2.48x faster** |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 8 | 739 ms | 507 ms | **1.46x faster** | 198 ms | **3.72x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 8 | 871 ms | 737 ms | **1.18x faster** | 413 ms | **2.11x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 8 | 872 ms | 448 ms | **1.95x faster** | 219 ms | **3.98x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 8 | 146 ms | 95 ms | **1.53x faster** | 53 ms | **2.74x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 8 | 2.12 s | 2.21 s | 1.04x slower | 1.17 s | **1.82x faster** |
| `filtered_topk` | 8 | 115 ms | 40 ms | **2.88x faster** | 33 ms | **3.54x faster** |
| `flash_attention4` | 8 | 3.15 s | 2.52 s | **1.25x faster** | 1.52 s | **2.07x faster** |
| `gdn_decode_bf16_ilp4` | 8 | 198 ms | 71 ms | **2.79x faster** | 42 ms | **4.73x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 8 | 784 ms | 1.09 s | 1.38x slower | 654 ms | **1.20x faster** |
| `kda_decode_multishape` | 8 | 123 ms | 312 ms | 2.54x slower | 204 ms | 1.66x slower |
| `msa_sparse_atten_fwd_sm100` | 8 | 230 ms | 46 ms | **4.98x faster** | 30 ms | **7.75x faster** |
| `recurrent_kda_decode_one_warp` | 8 | 3.89 s | 1.49 s | **2.61x faster** | 1.11 s | **3.51x faster** |
| `selective_state_update_stp_simple` | 8 | 33 ms | 15 ms | **2.15x faster** | 11 ms | **2.87x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 32 | 108 ms | 297 ms | 2.74x slower | 136 ms | 1.26x slower |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 32 | 169 ms | 305 ms | 1.80x slower | 144 ms | **1.17x faster** |
| `cudnn_sm100_flex_attention_backward` | 32 | 372 ms | 98 ms | **3.79x faster** | 48 ms | **7.81x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 32 | 1.22 s | 241 ms | **5.06x faster** | 78 ms | **15.63x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 32 | 1.91 s | 252 ms | **7.57x faster** | 142 ms | **13.47x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 32 | 4.96 s | 831 ms | **5.96x faster** | 285 ms | **17.42x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 32 | 2.84 s | 317 ms | **8.95x faster** | 139 ms | **20.49x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 32 | 6.71 s | 18.98 s | 2.83x slower | 9.74 s | 1.45x slower |
| `cudnn_sm100_kda_bprop_f16` | 32 | 1.16 s | 200 ms | **5.81x faster** | 56 ms | **20.73x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 32 | 968 ms | 893 ms | **1.08x faster** | 401 ms | **2.41x faster** |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 32 | 795 ms | 526 ms | **1.51x faster** | 199 ms | **4.00x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 32 | 837 ms | 809 ms | **1.03x faster** | 411 ms | **2.04x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 32 | 925 ms | 531 ms | **1.74x faster** | 215 ms | **4.31x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 32 | 143 ms | 108 ms | **1.32x faster** | 53 ms | **2.70x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 32 | 2.15 s | 2.64 s | 1.23x slower | 1.15 s | **1.87x faster** |
| `filtered_topk` | 32 | 123 ms | 43 ms | **2.90x faster** | 32 ms | **3.79x faster** |
| `flash_attention4` | 32 | 1.64 s | 2.33 s | 1.43x slower | 1.43 s | **1.15x faster** |
| `gdn_decode_bf16_ilp4` | 32 | 171 ms | 69 ms | **2.47x faster** | 37 ms | **4.60x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 32 | 673 ms | 1.04 s | 1.54x slower | 625 ms | **1.08x faster** |
| `kda_decode_multishape` | 32 | 129 ms | 325 ms | 2.52x slower | 204 ms | 1.58x slower |
| `msa_sparse_atten_fwd_sm100` | 32 | 252 ms | 48 ms | **5.24x faster** | 30 ms | **8.53x faster** |
| `recurrent_kda_decode_one_warp` | 32 | 4.98 s | 1.38 s | **3.60x faster** | 987 ms | **5.04x faster** |
| `selective_state_update_stp_simple` | 32 | 32 ms | 17 ms | **1.94x faster** | 12 ms | **2.74x faster** |

## synccheck

| case | workers | legacy | v2 (6ba4190) | v2 vs legacy (6ba4190) | v2 now (243f9f4) | now vs legacy |
| --- | --- | --- | --- | --- | --- | --- |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | 1 | 222 ms | 132 ms | **1.68x faster** | 80 ms | **2.76x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 1 | 152 ms | 90 ms | **1.69x faster** | 41 ms | **3.71x faster** |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 1 | 227 ms | 100 ms | **2.28x faster** | 40 ms | **5.69x faster** |
| `cudnn_sm100_flex_attention_backward` | 1 | 228 ms | 55 ms | **4.18x faster** | 35 ms | **6.60x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 1 | 1.10 s | 154 ms | **7.12x faster** | 41 ms | **26.72x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 1 | 1.64 s | 119 ms | **13.74x faster** | 42 ms | **38.66x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 1 | 4.77 s | 531 ms | **8.98x faster** | 168 ms | **28.33x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 1 | 3.08 s | 189 ms | **16.33x faster** | 35 ms | **88.69x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 1 | 30.57 s | 21.45 s | **1.42x faster** | 5.96 s | **5.13x faster** |
| `cudnn_sm100_kda_bprop_f16` | 1 | 1.03 s | 116 ms | **8.93x faster** | 32 ms | **31.97x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 1 | 1.88 s | 607 ms | **3.09x faster** | 394 ms | **4.77x faster** |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 1 | 1.79 s | 584 ms | **3.07x faster** | 412 ms | **4.34x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 1 | 827 ms | 415 ms | **1.99x faster** | 188 ms | **4.40x faster** |
| `deepgemm_sm100_fp8_gemm_1d1d` | 1 | 41 ms | 29 ms | **1.40x faster** | 18 ms | **2.33x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 1 | 642 ms | 187 ms | **3.44x faster** | 70 ms | **9.13x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 1 | 86 ms | 31 ms | **2.76x faster** | 20 ms | **4.27x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 1 | 1.30 s | 668 ms | **1.95x faster** | 279 ms | **4.65x faster** |
| `filtered_topk` | 1 | 112 ms | 22 ms | **5.14x faster** | 17 ms | **6.72x faster** |
| `flash_attention4` | 1 | 9.46 s | 2.31 s | **4.09x faster** | 1.37 s | **6.93x faster** |
| `gdn_cp_prefill_sm100` | 1 | 15.24 s | 2.73 s | **5.59x faster** | 2.30 s | **6.63x faster** |
| `gdn_decode_bf16_ilp4` | 1 | 235 ms | 44 ms | **5.32x faster** | 41 ms | **5.70x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 1 | 1.45 s | 317 ms | **4.57x faster** | 280 ms | **5.18x faster** |
| `gdn_prefill_sm100` | 1 | 1.68 s | 1.65 s | **1.01x faster** | 1.50 s | **1.12x faster** |
| `kda_backward_packed` | 1 | 1.25 s | 1.31 s | 1.05x slower | 213 ms | **5.88x faster** |
| `kda_decode_multishape` | 1 | 256 ms | 89 ms | **2.87x faster** | 70 ms | **3.65x faster** |
| `msa_sparse_atten_fwd_sm100` | 1 | 195 ms | 30 ms | **6.48x faster** | 15 ms | **13.04x faster** |
| `recurrent_kda_decode_one_warp` | 1 | 5.40 s | 1.03 s | **5.24x faster** | 1.01 s | **5.33x faster** |
| `selective_state_update_stp_simple` | 1 | 38 ms | 9 ms | **4.38x faster** | 8 ms | **4.48x faster** |
| `sparse_flashmla_prefill_head128_phase1` | 1 | 181 ms | 157 ms | **1.16x faster** | 101 ms | **1.79x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | 8 | 148 ms | 135 ms | **1.10x faster** | 84 ms | **1.77x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 8 | 78 ms | 75 ms | **1.04x faster** | 41 ms | **1.89x faster** |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 8 | 97 ms | 79 ms | **1.22x faster** | 41 ms | **2.38x faster** |
| `cudnn_sm100_flex_attention_backward` | 8 | 246 ms | 56 ms | **4.42x faster** | 35 ms | **7.12x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 8 | 1.24 s | 169 ms | **7.35x faster** | 38 ms | **32.25x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 8 | 1.53 s | 126 ms | **12.15x faster** | 46 ms | **33.62x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 8 | 4.82 s | 577 ms | **8.36x faster** | 167 ms | **28.78x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 8 | 2.83 s | 171 ms | **16.62x faster** | 35 ms | **79.87x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 8 | 9.60 s | 5.40 s | **1.78x faster** | 2.42 s | **3.97x faster** |
| `cudnn_sm100_kda_bprop_f16` | 8 | 1.03 s | 123 ms | **8.39x faster** | 32 ms | **31.70x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 8 | 1.46 s | 530 ms | **2.76x faster** | 328 ms | **4.45x faster** |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 8 | 1.21 s | 480 ms | **2.52x faster** | 342 ms | **3.54x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 8 | 503 ms | 372 ms | **1.35x faster** | 188 ms | **2.67x faster** |
| `deepgemm_sm100_fp8_gemm_1d1d` | 8 | 42 ms | 32 ms | **1.33x faster** | 18 ms | **2.39x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 8 | 377 ms | 199 ms | **1.89x faster** | 73 ms | **5.18x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 8 | 62 ms | 46 ms | **1.35x faster** | 24 ms | **2.57x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 8 | 728 ms | 697 ms | **1.05x faster** | 260 ms | **2.80x faster** |
| `filtered_topk` | 8 | 113 ms | 23 ms | **4.93x faster** | 17 ms | **6.78x faster** |
| `flash_attention4` | 8 | 2.77 s | 851 ms | **3.26x faster** | 581 ms | **4.77x faster** |
| `gdn_cp_prefill_sm100` | 8 | 7.41 s | 728 ms | **10.19x faster** | 577 ms | **12.85x faster** |
| `gdn_decode_bf16_ilp4` | 8 | 96 ms | 13 ms | **7.60x faster** | 11 ms | **9.12x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 8 | 506 ms | 91 ms | **5.57x faster** | 73 ms | **6.94x faster** |
| `gdn_prefill_sm100` | 8 | 1.69 s | 1.62 s | **1.04x faster** | 1.51 s | **1.12x faster** |
| `kda_backward_packed` | 8 | 1.16 s | 956 ms | **1.21x faster** | 171 ms | **6.80x faster** |
| `kda_decode_multishape` | 8 | 133 ms | 70 ms | **1.89x faster** | 41 ms | **3.23x faster** |
| `msa_sparse_atten_fwd_sm100` | 8 | 190 ms | 31 ms | **6.12x faster** | 15 ms | **12.70x faster** |
| `recurrent_kda_decode_one_warp` | 8 | 2.08 s | 655 ms | **3.17x faster** | 645 ms | **3.22x faster** |
| `selective_state_update_stp_simple` | 8 | 25 ms | 5 ms | **4.79x faster** | 4 ms | **6.27x faster** |
| `sparse_flashmla_prefill_head128_phase1` | 8 | 181 ms | 151 ms | **1.20x faster** | 101 ms | **1.80x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | 32 | 145 ms | 155 ms | 1.07x slower | 82 ms | **1.76x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 32 | 79 ms | 88 ms | 1.13x slower | 42 ms | **1.89x faster** |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 32 | 98 ms | 86 ms | **1.14x faster** | 39 ms | **2.52x faster** |
| `cudnn_sm100_flex_attention_backward` | 32 | 231 ms | 55 ms | **4.20x faster** | 32 ms | **7.28x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 32 | 1.23 s | 177 ms | **6.94x faster** | 39 ms | **31.71x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 32 | 1.65 s | 133 ms | **12.44x faster** | 42 ms | **39.00x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 32 | 5.31 s | 597 ms | **8.88x faster** | 168 ms | **31.61x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 32 | 2.83 s | 174 ms | **16.24x faster** | 35 ms | **81.49x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 32 | 7.37 s | 4.46 s | **1.65x faster** | 1.94 s | **3.80x faster** |
| `cudnn_sm100_kda_bprop_f16` | 32 | 1.02 s | 126 ms | **8.07x faster** | 33 ms | **31.31x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 32 | 1.45 s | 550 ms | **2.63x faster** | 366 ms | **3.97x faster** |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 32 | 1.66 s | 490 ms | **3.39x faster** | 315 ms | **5.27x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 32 | 536 ms | 592 ms | 1.11x slower | 191 ms | **2.81x faster** |
| `deepgemm_sm100_fp8_gemm_1d1d` | 32 | 43 ms | 33 ms | **1.29x faster** | 18 ms | **2.44x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 32 | 372 ms | 264 ms | **1.41x faster** | 73 ms | **5.07x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 32 | 64 ms | 56 ms | **1.16x faster** | 25 ms | **2.61x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 32 | 719 ms | 968 ms | 1.35x slower | 268 ms | **2.69x faster** |
| `filtered_topk` | 32 | 120 ms | 24 ms | **4.94x faster** | 17 ms | **7.21x faster** |
| `flash_attention4` | 32 | 2.13 s | 954 ms | **2.23x faster** | 684 ms | **3.12x faster** |
| `gdn_cp_prefill_sm100` | 32 | 6.42 s | 607 ms | **10.57x faster** | 363 ms | **17.68x faster** |
| `gdn_decode_bf16_ilp4` | 32 | 81 ms | 11 ms | **7.22x faster** | 7 ms | **12.01x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 32 | 445 ms | 62 ms | **7.15x faster** | 45 ms | **9.83x faster** |
| `gdn_prefill_sm100` | 32 | 1.67 s | 1.62 s | **1.03x faster** | 1.50 s | **1.11x faster** |
| `kda_backward_packed` | 32 | 1.27 s | 1.15 s | **1.11x faster** | 173 ms | **7.35x faster** |
| `kda_decode_multishape` | 32 | 131 ms | 67 ms | **1.95x faster** | 42 ms | **3.09x faster** |
| `msa_sparse_atten_fwd_sm100` | 32 | 200 ms | 32 ms | **6.27x faster** | 15 ms | **13.38x faster** |
| `recurrent_kda_decode_one_warp` | 32 | 1.97 s | 548 ms | **3.59x faster** | 563 ms | **3.50x faster** |
| `selective_state_update_stp_simple` | 32 | 24 ms | 6 ms | **3.83x faster** | 5 ms | **5.32x faster** |
| `sparse_flashmla_prefill_head128_phase1` | 32 | 179 ms | 148 ms | **1.21x faster** | 101 ms | **1.77x faster** |

