# Legacy vs v2 per tool, verdict-identical cases only (engine 6ba4190)

Source: `backend-comparison.json` (`head.cases`, engine wall, min of the samples per cell, same host/session for both columns). A (case, mode) is excluded when a `<mode>.delta.json` existed before step 5 (`git ls-tree 79f04eb^`): legacy and v2 disagreed on verdict or finding set there (racecheck B1/B7/R3/R4/T18/T19/X4, numsim H5, sync S1). The Mega-MoE perf configs are in `backend-comparison.md` (racecheck verdicts differ there; numsim verdicts agree).

## Summary (geometric mean over verdict-identical rows)

| mode | workers | rows | v2 vs legacy (geomean) | rows where v2 is >10% slower |
| --- | --- | --- | --- | --- |
| numsim | 1 | 30 | 4.12x faster | 1 |
| numsim | 8 | 30 | 3.44x faster | 0 |
| numsim | 32 | 30 | 2.95x faster | 0 |
| racecheck | 1 | 24 | 2.96x faster | 0 |
| racecheck | 8 | 23 | 1.87x faster | 5 |
| racecheck | 32 | 23 | 1.77x faster | 7 |
| synccheck | 1 | 29 | 3.43x faster | 0 |
| synccheck | 8 | 29 | 3.00x faster | 0 |
| synccheck | 32 | 29 | 2.82x faster | 3 |

## Excluded (verdict/finding delta vs legacy)

- `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` racecheck
- `deepgemm_sm100_fp8_gemm_1d1d` racecheck
- `kda_backward_packed` racecheck
- `sparse_flashmla_prefill_head128_phase1` racecheck

## numsim

| case | workers | legacy | v2 | v2 vs legacy |
| --- | --- | --- | --- | --- |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | 1 | 209 ms | 40 ms | **5.20x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 1 | 130 ms | 38 ms | **3.43x faster** |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 1 | 233 ms | 35 ms | **6.71x faster** |
| `cudnn_sm100_flex_attention_backward` | 1 | 199 ms | 22 ms | **9.04x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 1 | 158 ms | 133 ms | **1.18x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 1 | 250 ms | 93 ms | **2.68x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 1 | 689 ms | 393 ms | **1.75x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 1 | 357 ms | 137 ms | **2.60x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` | 1 | 65.47 s | 18.68 s | **3.51x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 1 | 32.18 s | 15.78 s | **2.04x faster** |
| `cudnn_sm100_kda_bprop_f16` | 1 | 160 ms | 100 ms | **1.60x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 1 | 1.92 s | 266 ms | **7.21x faster** |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 1 | 1.71 s | 326 ms | **5.26x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 1 | 742 ms | 213 ms | **3.48x faster** |
| `deepgemm_sm100_fp8_gemm_1d1d` | 1 | 35 ms | 6 ms | **5.81x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 1 | 605 ms | 64 ms | **9.50x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 1 | 80 ms | 19 ms | **4.13x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 1 | 1.26 s | 202 ms | **6.23x faster** |
| `filtered_topk` | 1 | 70 ms | 7 ms | **9.91x faster** |
| `flash_attention4` | 1 | 8.75 s | 1.11 s | **7.92x faster** |
| `gdn_cp_prefill_sm100` | 1 | 5.31 s | 2.17 s | **2.45x faster** |
| `gdn_decode_bf16_ilp4` | 1 | 238 ms | 34 ms | **7.04x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 1 | 1.46 s | 311 ms | **4.69x faster** |
| `gdn_prefill_sm100` | 1 | 1.69 s | 237 ms | **7.15x faster** |
| `kda_backward_packed` | 1 | 1.01 s | 1.27 s | 1.26x slower |
| `kda_decode_multishape` | 1 | 178 ms | 55 ms | **3.22x faster** |
| `msa_sparse_atten_fwd_sm100` | 1 | 163 ms | 20 ms | **8.25x faster** |
| `recurrent_kda_decode_one_warp` | 1 | 5.30 s | 532 ms | **9.95x faster** |
| `selective_state_update_stp_simple` | 1 | 33 ms | 6 ms | **5.19x faster** |
| `sparse_flashmla_prefill_head128_phase1` | 1 | 123 ms | 74 ms | **1.66x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | 8 | 165 ms | 75 ms | **2.19x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 8 | 61 ms | 36 ms | **1.72x faster** |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 8 | 78 ms | 31 ms | **2.54x faster** |
| `cudnn_sm100_flex_attention_backward` | 8 | 192 ms | 21 ms | **9.27x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 8 | 155 ms | 135 ms | **1.14x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 8 | 243 ms | 104 ms | **2.33x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 8 | 675 ms | 467 ms | **1.44x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 8 | 382 ms | 145 ms | **2.64x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` | 8 | 13.30 s | 3.94 s | **3.37x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 8 | 8.70 s | 3.49 s | **2.49x faster** |
| `cudnn_sm100_kda_bprop_f16` | 8 | 159 ms | 99 ms | **1.61x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 8 | 1.24 s | 185 ms | **6.70x faster** |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 8 | 656 ms | 120 ms | **5.46x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 8 | 407 ms | 228 ms | **1.78x faster** |
| `deepgemm_sm100_fp8_gemm_1d1d` | 8 | 29 ms | 7 ms | **4.23x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 8 | 323 ms | 96 ms | **3.37x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 8 | 56 ms | 26 ms | **2.13x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 8 | 645 ms | 275 ms | **2.34x faster** |
| `filtered_topk` | 8 | 67 ms | 8 ms | **8.21x faster** |
| `flash_attention4` | 8 | 2.06 s | 303 ms | **6.80x faster** |
| `gdn_cp_prefill_sm100` | 8 | 2.17 s | 447 ms | **4.86x faster** |
| `gdn_decode_bf16_ilp4` | 8 | 97 ms | 11 ms | **9.05x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 8 | 471 ms | 60 ms | **7.90x faster** |
| `gdn_prefill_sm100` | 8 | 1.69 s | 239 ms | **7.06x faster** |
| `kda_backward_packed` | 8 | 905 ms | 917 ms | 1.01x slower |
| `kda_decode_multishape` | 8 | 58 ms | 29 ms | **2.02x faster** |
| `msa_sparse_atten_fwd_sm100` | 8 | 158 ms | 21 ms | **7.49x faster** |
| `recurrent_kda_decode_one_warp` | 8 | 2.11 s | 167 ms | **12.65x faster** |
| `selective_state_update_stp_simple` | 8 | 19 ms | 4 ms | **5.21x faster** |
| `sparse_flashmla_prefill_head128_phase1` | 8 | 123 ms | 77 ms | **1.61x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | 32 | 130 ms | 71 ms | **1.83x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 32 | 60 ms | 50 ms | **1.20x faster** |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 32 | 80 ms | 42 ms | **1.89x faster** |
| `cudnn_sm100_flex_attention_backward` | 32 | 213 ms | 24 ms | **8.78x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 32 | 168 ms | 151 ms | **1.11x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 32 | 249 ms | 109 ms | **2.29x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 32 | 725 ms | 438 ms | **1.66x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 32 | 393 ms | 152 ms | **2.59x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` | 32 | 7.33 s | 3.50 s | **2.10x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 32 | 7.33 s | 3.31 s | **2.21x faster** |
| `cudnn_sm100_kda_bprop_f16` | 32 | 160 ms | 105 ms | **1.52x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 32 | 1.39 s | 255 ms | **5.46x faster** |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 32 | 627 ms | 109 ms | **5.74x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 32 | 395 ms | 320 ms | **1.24x faster** |
| `deepgemm_sm100_fp8_gemm_1d1d` | 32 | 30 ms | 8 ms | **3.68x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 32 | 326 ms | 172 ms | **1.90x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 32 | 56 ms | 38 ms | **1.47x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 32 | 755 ms | 467 ms | **1.62x faster** |
| `filtered_topk` | 32 | 68 ms | 10 ms | **6.80x faster** |
| `flash_attention4` | 32 | 1.29 s | 430 ms | **3.00x faster** |
| `gdn_cp_prefill_sm100` | 32 | 1.43 s | 339 ms | **4.22x faster** |
| `gdn_decode_bf16_ilp4` | 32 | 89 ms | 9 ms | **9.50x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 32 | 384 ms | 42 ms | **9.05x faster** |
| `gdn_prefill_sm100` | 32 | 1.69 s | 246 ms | **6.89x faster** |
| `kda_backward_packed` | 32 | 964 ms | 1.03 s | 1.07x slower |
| `kda_decode_multishape` | 32 | 61 ms | 38 ms | **1.62x faster** |
| `msa_sparse_atten_fwd_sm100` | 32 | 177 ms | 22 ms | **8.13x faster** |
| `recurrent_kda_decode_one_warp` | 32 | 2.21 s | 128 ms | **17.21x faster** |
| `selective_state_update_stp_simple` | 32 | 19 ms | 5 ms | **3.96x faster** |
| `sparse_flashmla_prefill_head128_phase1` | 32 | 127 ms | 80 ms | **1.58x faster** |

## racecheck

| case | workers | legacy | v2 | v2 vs legacy |
| --- | --- | --- | --- | --- |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 1 | 275 ms | 302 ms | 1.10x slower |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 1 | 495 ms | 288 ms | **1.72x faster** |
| `cudnn_sm100_flex_attention_backward` | 1 | 340 ms | 96 ms | **3.54x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 1 | 1.13 s | 229 ms | **4.92x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 1 | 1.72 s | 268 ms | **6.42x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 1 | 5.13 s | 956 ms | **5.37x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 1 | 2.87 s | 304 ms | **9.44x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` | 1 | 121.14 s | 40.30 s | **3.01x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 1 | 45.73 s | 35.19 s | **1.30x faster** |
| `cudnn_sm100_kda_bprop_f16` | 1 | 1.17 s | 198 ms | **5.92x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 1 | 1.89 s | 1.22 s | **1.55x faster** |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 1 | 1.57 s | 587 ms | **2.67x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 1 | 1.60 s | 734 ms | **2.18x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 1 | 1.56 s | 432 ms | **3.62x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 1 | 182 ms | 85 ms | **2.14x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 1 | 3.98 s | 2.39 s | **1.67x faster** |
| `filtered_topk` | 1 | 123 ms | 40 ms | **3.07x faster** |
| `flash_attention4` | 1 | 16.44 s | 3.45 s | **4.76x faster** |
| `gdn_decode_bf16_ilp4` | 1 | 352 ms | 97 ms | **3.62x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 1 | 2.13 s | 1.36 s | **1.57x faster** |
| `kda_decode_multishape` | 1 | 494 ms | 344 ms | **1.44x faster** |
| `msa_sparse_atten_fwd_sm100` | 1 | 241 ms | 45 ms | **5.34x faster** |
| `recurrent_kda_decode_one_warp` | 1 | 8.94 s | 1.94 s | **4.62x faster** |
| `selective_state_update_stp_simple` | 1 | 57 ms | 20 ms | **2.80x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 8 | 110 ms | 275 ms | 2.50x slower |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 8 | 204 ms | 293 ms | 1.43x slower |
| `cudnn_sm100_flex_attention_backward` | 8 | 350 ms | 102 ms | **3.45x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 8 | 1.14 s | 234 ms | **4.86x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 8 | 1.80 s | 251 ms | **7.17x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 8 | 4.89 s | 795 ms | **6.15x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 8 | 2.79 s | 308 ms | **9.07x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 8 | 9.07 s | 20.05 s | 2.21x slower |
| `cudnn_sm100_kda_bprop_f16` | 8 | 1.17 s | 194 ms | **6.03x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 8 | 950 ms | 1.02 s | 1.07x slower |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 8 | 739 ms | 507 ms | **1.46x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 8 | 871 ms | 737 ms | **1.18x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 8 | 872 ms | 448 ms | **1.95x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 8 | 146 ms | 95 ms | **1.53x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 8 | 2.12 s | 2.21 s | 1.04x slower |
| `filtered_topk` | 8 | 115 ms | 40 ms | **2.88x faster** |
| `flash_attention4` | 8 | 3.15 s | 2.52 s | **1.25x faster** |
| `gdn_decode_bf16_ilp4` | 8 | 198 ms | 71 ms | **2.79x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 8 | 784 ms | 1.09 s | 1.38x slower |
| `kda_decode_multishape` | 8 | 123 ms | 312 ms | 2.54x slower |
| `msa_sparse_atten_fwd_sm100` | 8 | 230 ms | 46 ms | **4.98x faster** |
| `recurrent_kda_decode_one_warp` | 8 | 3.89 s | 1.49 s | **2.61x faster** |
| `selective_state_update_stp_simple` | 8 | 33 ms | 15 ms | **2.15x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 32 | 108 ms | 297 ms | 2.74x slower |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 32 | 169 ms | 305 ms | 1.80x slower |
| `cudnn_sm100_flex_attention_backward` | 32 | 372 ms | 98 ms | **3.79x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 32 | 1.22 s | 241 ms | **5.06x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 32 | 1.91 s | 252 ms | **7.57x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 32 | 4.96 s | 831 ms | **5.96x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 32 | 2.84 s | 317 ms | **8.95x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 32 | 6.71 s | 18.98 s | 2.83x slower |
| `cudnn_sm100_kda_bprop_f16` | 32 | 1.16 s | 200 ms | **5.81x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 32 | 968 ms | 893 ms | **1.08x faster** |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 32 | 795 ms | 526 ms | **1.51x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 32 | 837 ms | 809 ms | **1.03x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 32 | 925 ms | 531 ms | **1.74x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 32 | 143 ms | 108 ms | **1.32x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 32 | 2.15 s | 2.64 s | 1.23x slower |
| `filtered_topk` | 32 | 123 ms | 43 ms | **2.90x faster** |
| `flash_attention4` | 32 | 1.64 s | 2.33 s | 1.43x slower |
| `gdn_decode_bf16_ilp4` | 32 | 171 ms | 69 ms | **2.47x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 32 | 673 ms | 1.04 s | 1.54x slower |
| `kda_decode_multishape` | 32 | 129 ms | 325 ms | 2.52x slower |
| `msa_sparse_atten_fwd_sm100` | 32 | 252 ms | 48 ms | **5.24x faster** |
| `recurrent_kda_decode_one_warp` | 32 | 4.98 s | 1.38 s | **3.60x faster** |
| `selective_state_update_stp_simple` | 32 | 32 ms | 17 ms | **1.94x faster** |

## synccheck

| case | workers | legacy | v2 | v2 vs legacy |
| --- | --- | --- | --- | --- |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | 1 | 222 ms | 132 ms | **1.68x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 1 | 152 ms | 90 ms | **1.69x faster** |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 1 | 227 ms | 100 ms | **2.28x faster** |
| `cudnn_sm100_flex_attention_backward` | 1 | 228 ms | 55 ms | **4.18x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 1 | 1.10 s | 154 ms | **7.12x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 1 | 1.64 s | 119 ms | **13.74x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 1 | 4.77 s | 531 ms | **8.98x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 1 | 3.08 s | 189 ms | **16.33x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 1 | 30.57 s | 21.45 s | **1.42x faster** |
| `cudnn_sm100_kda_bprop_f16` | 1 | 1.03 s | 116 ms | **8.93x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 1 | 1.88 s | 607 ms | **3.09x faster** |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 1 | 1.79 s | 584 ms | **3.07x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 1 | 827 ms | 415 ms | **1.99x faster** |
| `deepgemm_sm100_fp8_gemm_1d1d` | 1 | 41 ms | 29 ms | **1.40x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 1 | 642 ms | 187 ms | **3.44x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 1 | 86 ms | 31 ms | **2.76x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 1 | 1.30 s | 668 ms | **1.95x faster** |
| `filtered_topk` | 1 | 112 ms | 22 ms | **5.14x faster** |
| `flash_attention4` | 1 | 9.46 s | 2.31 s | **4.09x faster** |
| `gdn_cp_prefill_sm100` | 1 | 15.24 s | 2.73 s | **5.59x faster** |
| `gdn_decode_bf16_ilp4` | 1 | 235 ms | 44 ms | **5.32x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 1 | 1.45 s | 317 ms | **4.57x faster** |
| `gdn_prefill_sm100` | 1 | 1.68 s | 1.65 s | **1.01x faster** |
| `kda_backward_packed` | 1 | 1.25 s | 1.31 s | 1.05x slower |
| `kda_decode_multishape` | 1 | 256 ms | 89 ms | **2.87x faster** |
| `msa_sparse_atten_fwd_sm100` | 1 | 195 ms | 30 ms | **6.48x faster** |
| `recurrent_kda_decode_one_warp` | 1 | 5.40 s | 1.03 s | **5.24x faster** |
| `selective_state_update_stp_simple` | 1 | 38 ms | 9 ms | **4.38x faster** |
| `sparse_flashmla_prefill_head128_phase1` | 1 | 181 ms | 157 ms | **1.16x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | 8 | 148 ms | 135 ms | **1.10x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 8 | 78 ms | 75 ms | **1.04x faster** |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 8 | 97 ms | 79 ms | **1.22x faster** |
| `cudnn_sm100_flex_attention_backward` | 8 | 246 ms | 56 ms | **4.42x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 8 | 1.24 s | 169 ms | **7.35x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 8 | 1.53 s | 126 ms | **12.15x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 8 | 4.82 s | 577 ms | **8.36x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 8 | 2.83 s | 171 ms | **16.62x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 8 | 9.60 s | 5.40 s | **1.78x faster** |
| `cudnn_sm100_kda_bprop_f16` | 8 | 1.03 s | 123 ms | **8.39x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 8 | 1.46 s | 530 ms | **2.76x faster** |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 8 | 1.21 s | 480 ms | **2.52x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 8 | 503 ms | 372 ms | **1.35x faster** |
| `deepgemm_sm100_fp8_gemm_1d1d` | 8 | 42 ms | 32 ms | **1.33x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 8 | 377 ms | 199 ms | **1.89x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 8 | 62 ms | 46 ms | **1.35x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 8 | 728 ms | 697 ms | **1.05x faster** |
| `filtered_topk` | 8 | 113 ms | 23 ms | **4.93x faster** |
| `flash_attention4` | 8 | 2.77 s | 851 ms | **3.26x faster** |
| `gdn_cp_prefill_sm100` | 8 | 7.41 s | 728 ms | **10.19x faster** |
| `gdn_decode_bf16_ilp4` | 8 | 96 ms | 13 ms | **7.60x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 8 | 506 ms | 91 ms | **5.57x faster** |
| `gdn_prefill_sm100` | 8 | 1.69 s | 1.62 s | **1.04x faster** |
| `kda_backward_packed` | 8 | 1.16 s | 956 ms | **1.21x faster** |
| `kda_decode_multishape` | 8 | 133 ms | 70 ms | **1.89x faster** |
| `msa_sparse_atten_fwd_sm100` | 8 | 190 ms | 31 ms | **6.12x faster** |
| `recurrent_kda_decode_one_warp` | 8 | 2.08 s | 655 ms | **3.17x faster** |
| `selective_state_update_stp_simple` | 8 | 25 ms | 5 ms | **4.79x faster** |
| `sparse_flashmla_prefill_head128_phase1` | 8 | 181 ms | 151 ms | **1.20x faster** |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | 32 | 145 ms | 155 ms | 1.07x slower |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 32 | 79 ms | 88 ms | 1.13x slower |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 32 | 98 ms | 86 ms | **1.14x faster** |
| `cudnn_sm100_flex_attention_backward` | 32 | 231 ms | 55 ms | **4.20x faster** |
| `cudnn_sm100_gdn2_bprop_f16` | 32 | 1.23 s | 177 ms | **6.94x faster** |
| `cudnn_sm100_gdn2_recompute_f16` | 32 | 1.65 s | 133 ms | **12.44x faster** |
| `cudnn_sm100_gdn_bprop_f16` | 32 | 5.31 s | 597 ms | **8.88x faster** |
| `cudnn_sm100_gdn_prefill_f16` | 32 | 2.83 s | 174 ms | **16.24x faster** |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 32 | 7.37 s | 4.46 s | **1.65x faster** |
| `cudnn_sm100_kda_bprop_f16` | 32 | 1.02 s | 126 ms | **8.07x faster** |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 32 | 1.45 s | 550 ms | **2.63x faster** |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 32 | 1.66 s | 490 ms | **3.39x faster** |
| `deepgemm_sm100_fp4_mqa_logits` | 32 | 536 ms | 592 ms | 1.11x slower |
| `deepgemm_sm100_fp8_gemm_1d1d` | 32 | 43 ms | 33 ms | **1.29x faster** |
| `deepgemm_sm100_fp8_mqa_logits` | 32 | 372 ms | 264 ms | **1.41x faster** |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 32 | 64 ms | 56 ms | **1.16x faster** |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 32 | 719 ms | 968 ms | 1.35x slower |
| `filtered_topk` | 32 | 120 ms | 24 ms | **4.94x faster** |
| `flash_attention4` | 32 | 2.13 s | 954 ms | **2.23x faster** |
| `gdn_cp_prefill_sm100` | 32 | 6.42 s | 607 ms | **10.57x faster** |
| `gdn_decode_bf16_ilp4` | 32 | 81 ms | 11 ms | **7.22x faster** |
| `gdn_decode_bf16_wide_vec_mtp` | 32 | 445 ms | 62 ms | **7.15x faster** |
| `gdn_prefill_sm100` | 32 | 1.67 s | 1.62 s | **1.03x faster** |
| `kda_backward_packed` | 32 | 1.27 s | 1.15 s | **1.11x faster** |
| `kda_decode_multishape` | 32 | 131 ms | 67 ms | **1.95x faster** |
| `msa_sparse_atten_fwd_sm100` | 32 | 200 ms | 32 ms | **6.27x faster** |
| `recurrent_kda_decode_one_warp` | 32 | 1.97 s | 548 ms | **3.59x faster** |
| `selective_state_update_stp_simple` | 32 | 24 ms | 6 ms | **3.83x faster** |
| `sparse_flashmla_prefill_head128_phase1` | 32 | 179 ms | 148 ms | **1.21x faster** |

