# OpLib benchmarks

These are the baselines for the criterion benches that guard OpLib
optimizations (CLAUDE.md: every kept optimization has a benchmark).

- `cargo bench -p numsim-oplib --bench kernels` (`numsim-oplib/benches/kernels.rs`)
- `cargo bench -p numsim-core --bench oplib` (`numsim-core/benches/oplib.rs`)

CI builds every bench without running it: `cargo bench --workspace --no-run`
in `.github/workflows/tests.yml`. To smoke-run one iteration of each, use
`cargo bench -p numsim-oplib --bench kernels -- --test` and
`cargo bench -p numsim-core --bench oplib -- --test`.

**Host and load.** AMD EPYC 7763 (256 hardware threads), 2026-10-08, criterion
`--warm-up-time 1 --measurement-time 2`, release profile. The host was **not
idle**: the load average was 35-49, from other workers' builds and tests. Read
the numbers as relative. To judge a change, rerun its bench on the same host
and compare; don't treat these as absolute targets. Each value is criterion's
median estimate.

## MMA: tcgen05.mma numerics

One MMA through `tc_mma_ctas` with engine-like closures (`numsim-core` bench `oplib`). The kinds and shapes are the ones the canonical corpus issues.

| bench | median |
| --- | --- |
| `tc_mma/f16_ss_m128_n256_k16_accumulate` | 401.19 µs |
| `tc_mma/bf16_ss_m64_n256_k16_layout_f` | 344.05 µs |
| `tc_mma/tf32_ss_m128_n256_k8` | 411.55 µs |
| `tc_mma/f8f6f4_e4m3_ss_m128_n256_k32` | 482.96 µs |
| `tc_mma/i8_ss_m128_n256_k32` | 700.76 µs |
| `tc_mma/mxf8f6f4_e4m3_ss_m128_n256_k32` | 478.74 µs |
| `tc_mma/mxf4_e2m1_ss_m128_n256_k64` | 539.26 µs |
| `tc_mma/bf16_ss_cta2_m256_n256_k16_accumulate` | 672 µs |

## MMA: increasing-K chain

`fpenv::fma_*_abt_increasing_k` (`numsim-oplib` bench `kernels`). The `_with_nan` row exercises the NaN re-pin post-pass (delta D8).

| bench | median |
| --- | --- |
| `mma_chain/f32_m128_n256_k16` | 90.168 µs |
| `mma_chain/f32_m128_n256_k32` | 134.19 µs |
| `mma_chain/f32_m64_n256_k16` | 41.786 µs |
| `mma_chain/f32_m16_n8_k16_mma_sync` | 1.1426 µs |
| `mma_chain/f32_m128_n256_k16_with_nan` | 587.72 µs |
| `mma_chain/f64_m16_n8_k4` | 590.65 ns |

## mma.sync

| bench | median |
| --- | --- |
| `mma_sync/m16n8k16_f32_bf16` | 2.7284 µs |

## Warp reductions and collectives

Dispatched tile reductions run on `shfl.sync` and `redux`; `warp_reduce` covers the `cuda_warp_reduce` / `cta_reduce` lanes.

| bench | median |
| --- | --- |
| `warp_reduce/sum_f32_w32` | 243.19 ns |
| `warp_reduce/max_f32_w32` | 728.28 ns |
| `warp_reduce/sum_f64_w8` | 333.23 ns |
| `warp_32_lanes/shfl_sync_bfly` | 216.26 ns |
| `warp_32_lanes/shfl_sync_idx` | 219.76 ns |
| `warp_32_lanes/redux_add_u32` | 87.419 ns |
| `warp_32_lanes/redux_max_f32` | 108.23 ns |

## Dtype conversion

1024 elements per iteration; per-element cost is time / 1024. The 32-lane PTX `cvt` rows are in the dispatch table below.

| bench | median |
| --- | --- |
| `cvt_x1024/f32_to_fp16_rn` | 2.4128 µs |
| `cvt_x1024/f32_to_bf16_rn` | 352.68 ns |
| `cvt_x1024/fp16_to_f32` | 1.3671 µs |
| `cvt_x1024/f32_to_tf32` | 997.43 ns |
| `cvt_x1024/f32_to_e4m3_satfinite` | 4.77 µs |
| `cvt_x1024/f32_to_e2m1_satfinite` | 4.57 µs |

## Host NaN rule (delta D8)

1024 elements per iteration, finite inputs and all-NaN inputs. The 32-lane PTX path is `ptx_32_lanes/*_all_nan`.

| bench | median |
| --- | --- |
| `host_nan_x1024/host_fma_f32_finite` | 3.7965 µs |
| `host_nan_x1024/pinned_add_f32_finite` | 429.91 ns |
| `host_nan_x1024/host_fma_f32_all_nan` | 3.0566 µs |
| `host_nan_x1024/pinned_add_f32_all_nan` | 433.40 ns |

## resolve_ptx dispatch and the ALU tail

`resolve_ptx` itself (load-time cost per op), then one 32-lane call of each hot TIR / PTX form.

| bench | median |
| --- | --- |
| `tir_32_lanes/binary_add_u32` | 29.306 ns |
| `tir_32_lanes/binary_add_f32` | 49.913 ns |
| `tir_32_lanes/binary_mul_f32` | 50.700 ns |
| `tir_32_lanes/binary_add_f16` | 110.29 ns |
| `tir_32_lanes/binary_floordiv_s32` | 71.461 ns |
| `tir_32_lanes/ternary_fma_f32` | 64.421 ns |
| `tir_32_lanes/unary_neg_f32` | 18.846 ns |
| `tir_32_lanes/cast_f32_s32` | 39.931 ns |
| `tir_32_lanes/cast_f32_f16` | 41.754 ns |
| `tir_32_lanes/cast_s32_f32` | 19.431 ns |
| `tir_32_lanes/compare_lt_u32` | 18.031 ns |
| `tir_32_lanes/compare_lt_f32` | 17.056 ns |
| `ptx_32_lanes/mov_pack_b32x2` | 27.867 ns |
| `ptx_32_lanes/mov_unpack_b32x2` | 29.359 ns |
| `ptx_32_lanes/fma_rn_f32` | 60.987 ns |
| `ptx_32_lanes/fma_rn_f32_all_nan` | 76.741 ns |
| `ptx_32_lanes/add_rn_f32_all_nan` | 216.79 ns |
| `ptx_32_lanes/ex2_approx_ftz_f32` | 243.74 ns |
| `ptx_32_lanes/cvt_rn_f16x2_f32` | 63.863 ns |
| `ptx_32_lanes/cvt_rn_bf16x2_f32` | 280.19 ns |
| `ptx_32_lanes/cvt_rzi_s32_f32` | 81.098 ns |
| `ptx_32_lanes/cvt_rn_f32_f16` | 134.30 ns |
| `ptx_32_lanes/encode_instr_descriptor` | 502.00 ns |
| `ptx_32_lanes/encode_matrix_descriptor` | 492.92 ns |
| `ptx_32_lanes/pack_u32x4` | 189.41 ns |
| `dispatch/resolve_ptx_direct_form` | 302.24 ns |
| `dispatch/resolve_ptx_boxed_form_interned` | 909.07 ns |

## TMA plans, descriptors, tcgen05.ld/st maps

| bench | median |
| --- | --- |
| `tma_plan/load_bf16_64x128_sw128` | 566.94 ns |
| `tma_plan/load_bf16_64x128_sw128_oob_uncached` | 966.09 µs |
| `tma_plan/descriptor_decode` | 47.037 ns |
| `tma_plan/descriptor_encode` | 78.578 ns |
| `tcgen_ldst/map_32x32b_x64` | 59.789 ns |

## v2 math builtins (deltas D9, D10)

1024 elements per iteration.

| bench | median |
| --- | --- |
| `math_x1024/log1p_f32` | 11.625 µs |
| `math_x1024/sigmoid_f32` | 7.0341 µs |
| `math_x1024/erf_f32` | 21.004 µs |
| `math_x1024/exp10_f32` | 11.185 µs |
| `math_x1024/log10_f32` | 9.9913 µs |

## Optimizations these benches guard

| change | bench | before | after |
| --- | --- | --- | --- |
| D window and packed TMEM A read as one closure call per lane run; streamed Layout-D walk; 16-byte K-major B16 gathers (W4-16) | `tc_mma/f16_ss_m128_n256_k16_accumulate` | 2.16 ms | 0.31-0.40 ms |
| Translation-cached interior TMA plans (W4-16) | `tma_plan/load_bf16_64x128_sw128` (cache miss: `..._oob_uncached`) | 891 µs | 0.5 µs (miss: ~0.9 ms) |
| Cached `tcgen05.ld/st` maps (W4-16) | `tcgen_ldst/map_32x32b_x64` | 18.5 µs | 54-60 ns |
| Word-level `numsim.pack` (W4-16) | `ptx_32_lanes/pack_u32x4` | 3.20 µs | 0.17-0.19 µs |
| Warp-uniform and cached instruction-descriptor encode (W4-16) | `ptx_32_lanes/encode_instr_descriptor` | 2.64 µs | 0.44-0.50 µs |
| Arithmetic E4M3 encoder (W4-19; the old nearest-code search is the test oracle) | `cvt_x1024/f32_to_e4m3_satfinite` | 693 µs | 4.8 µs |
| Streamed CTA-pair Layout-D accumulator walk (`cta2_runs`; no `m*n` cell list) (W4-perf) | `tc_mma/bf16_ss_cta2_m256_n256_k16_accumulate` | 1.43 ms | 0.67 ms |
| Table decode of tcgen05 narrow operands (fp8/fp6/fp4; bit-identical, tested exhaustively) (W4-perf) | `tcgen05_narrow_decode_x1024_atoms/E4M3_table` (old path: `..._direct`) | 136 µs | 20.4 µs |

End to end (1 worker, interpreter, min of 3 interleaved runs on a loaded host), these
two changes took `fp16_bf16_gemm` from 96 to 76 ms per run and
`deepgemm_sm100_fp8_gemm_1d1d` from 11.5 to 10.0 ms per run.

The NaN re-pin of the MMA chain (delta D8) costs nothing on finite outputs.
When many outputs are NaN, it recomputes each of their chains with the scalar
pinned FMA; that is the `mma_chain/f32_m128_n256_k16_with_nan` row.
