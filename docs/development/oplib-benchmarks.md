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
| AVX2 register-tiled FMA chain (4 rows x 16 columns kept in registers across K, as the AVX-512 path; bit-identical, tested over tile shapes and specials) and branch-free NaN pre-scan (W4-perf2) | `mma_chain/f32_m128_n256_k16` / `_k32` / `f32_m16_n8_k16_mma_sync` | 85.7 / 132.9 / 0.92 µs | 23.4 / 65.5 / 0.26 µs |
| Cross-crate `#[inline]` on cell/operand decoders, one decode-table fetch per atom, contiguous-run fast path in `read_run`/`write_run` (W4-perf2) | `tc_mma/f16_ss_m128_n256_k16_accumulate`, `tc_mma/mxf8f6f4_e4m3_ss_m128_n256_k32`, `tc_mma/bf16_ss_cta2_m256_n256_k16_accumulate` | 457 / 391 / 661 µs | 332 / 249 / 359 µs |

End to end (1 worker, interpreter, min of 3 interleaved runs on a loaded host), the two
W4-perf changes took `fp16_bf16_gemm` from 96 to 76 ms per run and
`deepgemm_sm100_fp8_gemm_1d1d` from 11.5 to 10.0 ms per run. The W4-perf2 round is
within host noise end to end (gemm 69.7 -> 67.4 ms min, while concurrent engine work
moved the same runs), so its criterion rows above are the guard. A thin-LTO,
one-codegen-unit release build measured a further ~10% on `fp16_bf16_gemm`
(67 -> 60 ms) and ~4% on `deepgemm_sm100_fp8_gemm_1d1d`; that is a workspace
profile choice, not adopted here.

The NaN re-pin of the MMA chain (delta D8) costs nothing on finite outputs.
When many outputs are NaN, it recomputes each of their chains with the scalar
pinned FMA; that is the `mma_chain/f32_m128_n256_k16_with_nan` row.

## Measured negative: borrowed-view MMA operand I/O (reverted)

A whole-allocation view API on `tc_mma_ctas` (`TcViews`, landed in 79e643d) let the
MMA read shared operands and read/write TMEM straight from borrowed allocation
slices instead of through the per-piece copying callbacks (about 2,200 calls per
block-scaled MMA on the Mega MoE e24 stream). On oplib's own criterion rows with
cheap callbacks it measured `tc_mma/f16_ss_m128_n256_k16_accumulate` 348 -> 169 µs
and `tc_mma/mxf8f6f4_e4m3_ss_m128_n256_k32` 270 -> 228 µs. Wired into the engine,
a same-moment A/B on the Mega MoE max config and e24 showed no measurable gain
(per MMA 207-271 µs with views off vs 244-262 µs on; the max-config differences
equal host drift). The earlier "callback boundary is half the cost" estimate came
from a stub with zero-filled operands that took cheaper arithmetic paths. The
engine-side per-MMA cost is the per-element work in oplib (operand decode, the
increasing-K FMA chain, accumulator encode), so the API was removed (W4/W2,
2026-10-08). Further gains need fewer per-element operations, not a cheaper I/O
boundary.

## Mega MoE block-scaled MMA: where `tc_mma_ctas` spends its time (W4, 2026-10-09)

Mega MoE medium (`t64_h2048_i1536_e96_k4_g1`, 110,592 `tcgen05.mma.kind::mxf8f6f4.block_scale`
MMAs), private builds with scratch-only phase counters (`Instant` per phase, callback
call counts), host load 8-16. "Before" is d86beb5; "after" adds the in-place F32 D
window read/write, the vector-free scale reads (`read_mxf8_scales`) and the bit-built
E8M0 decode. All three give bit-identical results.

| phase (CPU-s) | before, 1 worker | after, 1 worker | before, 16 workers | after, 16 workers |
| --- | --- | --- | --- | --- |
| `run_mma` (engine wrapper + `tc_mma_ctas`) | 14.59 | 12.45 | 19.97 | 16.17 |
| `tc_mma_ctas` | 11.71 | 9.66 | 16.76 | 13.05 |
| A/B operand gather (`gather_f8_rows`) | 3.74 | 2.87 | 4.85 | 3.90 |
| scale reads (`mxf8_scale_values` / `read_mxf8_scales`) | 3.89 | 3.75 | 5.72 | 4.83 |
| D window read | 2.06 | 1.34 | 3.21 | 1.85 |
| D window write | 1.27 | 1.00 | 1.68 | 1.33 |
| increasing-K FMA chain (`mma_dense_tail`) | 0.53 | 0.49 | 0.73 | 0.69 |
| scale application | 0.10 | 0.10 | 0.14 | 0.12 |

Per MMA the engine callbacks are called about 2,200 times: 544 shared-memory reads,
1,404 TMEM reads (about 1,150 of them single-cell scale reads, every row's scale in
every replica) and 256 TMEM writes. A scale cell read costs about 25 ns, nearly all
inside the engine callback (arena borrow, allocation and overlay lookup, validity
check, copy, read note).

**Finding: about 95% of `tc_mma_ctas` is per-piece operand and accumulator I/O through
the engine callbacks, not arithmetic** (the FMA chain is 4-5%). Medium end to end
moved about 6-8% at 1 worker and within noise at 16. Reading each scale lane as one
run (4x fewer calls; scales 3.70 -> 2.07 s in a prototype) was declined: it changes
the first-error offset order and the uninitialized-span granularity for invalid scale
cells. The next lever is the engine's per-callback cost (resolve the window's
allocation and overlay once per MMA, keep per-cell validity and notes); see the
contract request draft routed by the coordinator.


## Mega MoE block-scaled MMA: TMEM/shared windows plus the `note_all` fix (W4 + W13, 2026-10-09)

Same counters and fixture as above. "Before" is d719a9d. "After" is one batch:
- `TcMmaIo` windows: each piece is read from a borrowed arena window when the piece is in
  bounds and fully valid, or else through that piece's callback. Windows exist only for
  non-overlaid, non-metadata allocations and are dropped before the first TMEM write.
- W13's `note_all` read-note fix: read notes are taken only when observing.
- Window lane-run scale reads, with diagnostics unchanged because any invalid or
  out-of-window cell falls back to the per-cell callback in the original order.
- A shift/mask atom split in `smem_desc` (`div_rem_atom`).
- An FP8 byte-decode fast path and a hoisted negate in `gather_f8_rows`.

Each figure is the median of 3 interleaved runs at 1 worker (host load 7-15). Results are
bit-identical: conformance 304 passed, and output digests of 11 MMA-heavy cases (both Mega MoE
fixtures included) are identical at 1/8/32 workers.

| phase (CPU-s, 1 worker) | before (d719a9d) | after |
| --- | --- | --- |
| `run_mma` (engine wrapper + `tc_mma_ctas`) | 12.66 | 7.42 (1.71x) |
| `tc_mma_ctas` | 9.97 | 7.23 |
| A/B operand gather (`gather_f8_rows`) | 3.26 | 3.12 |
| scale reads (`read_mxf8_scales`) | 3.71 | 1.16 |
| D window read | 1.32 | 1.22 |
| D window write | 1.01 | 1.07 |
| increasing-K FMA chain (`mma_dense_tail`) | 0.48 | 0.48 |
| scale application | 0.11 | 0.10 |

The engine wrapper's share of `run_mma` falls from 2.69 s to 0.19 s, which is the
`note_all` fix. Scale reads drop 3.2x. Gather is now the largest phase.

New bench, `tcgen05_gather/f8_rows_256x32_sw128` (256 rows, K=32, 128B swizzle, slice
reader): 25.5 us before the FP8 decode and negate changes, 19.6 us after. Attribution
experiments on this bench (code temporarily removed, not kept):
- dropping the decode gives 11.9 us, so decode costs about 40%;
- dropping `finish_shared_byte_offset` gives 15.4 us, so the offset finish costs about 20%.

In the Mega MoE run, gather moved only 4%. The per-atom reader dispatch and the offset
finish dominate there, not the decode.

## Sub-byte TMA plan caching and the K-major F8/F6/F4 gather (W4, 2026-10-09)

Both changes are exact. Fixture: Mega MoE medium (`t64_h2048_i1536_e96_k4_g1`), 1 worker,
scratch-only phase counters, median of 3 interleaved runs against c257bfe. Conformance 304
passed. Output digests of 14 cases are identical to c257bfe at 1/8/32 workers. The cases
cover both Mega MoE fixtures, `nvfp4_gemm`, `fastcu_nvfp4_gemm_gb300`, `flash_attention4(_fp4)`,
sm107 block-scaled, `mxfp4_quantize` and the grouped GEMMs.

**TMA plans.** The translation cache (`oplib/tma/cache.rs`) now also serves sub-byte (FP4, U6)
loads whose inner origin is a multiple of 128 elements:
- every origin-dependent check passes there exactly as at coordinate 0: packed FP4 needs
  a multiple of 2, padded FP4 and U6 a multiple of 128;
- `c_0 * bits / 8` is exact, so the plan is the coordinate-0 plan shifted.

Sub-byte stores and other origins are still planned directly. The guard is
`cached_sub_byte_plans_equal_direct_plans`: plan or error equal to the direct planner for FP4
packed/padded and U6, every swizzle, ranks 1-3, loads and stores, aligned and unaligned
origins, at least 150 cache-served plans. A wrong shift or 2-element alignment fails it.

| bench | before | after |
| --- | --- | --- |
| `tma_plan/load_e2m1_128x128_sw128_mega_moe` (new) | 7.90 us | 0.48 us (16.6x) |
| `tma_plan/load_e2m1_128x64_sw128` | 4.00 us | 0.33 us (12x) |
| `tma_plan/load_e2m1_128x128_sw128_unaligned_uncached` (new) | 7.95 us | 7.87 us (direct planner) |

On medium, `tma_plan_dir` falls from 1.36 s to 0.11 s over 168,192 calls. The 55,296 FP4
weight loads were 22 us each.

**Operand gather.** `gather_f8_rows`, K-major with whole 16-value atoms, now does:
- the same shared reads in the same order;
- each offset from `SharedOffsets`, which hoists the descriptor and window constants and
  returns a value only when `shared_byte_offset` would return `Ok` with it; otherwise
  `shared_byte_offset` itself is called and produces the error;
- decode through a per-call `AtomDecoder`: the FP8 table, or a 16-entry E2M1 table built
  from `float4_e2m1fn_bits_to_f32`, with negation as a sign-bit flip;
- writes straight into the output.

Guards:
- `k_major_f8_gather_equals_the_reference_loop`: the pre-change loop is kept as the oracle;
  values are compared as bits, plus error text and the read log, over every format, K,
  padding, negation, swizzle, column masks, short windows and failing reads;
- `atom_decoder_matches_decode_shared_atom`: every byte in every position;
- `shared_offsets_agree_with_shared_byte_offset`: random descriptors and windows.

Mutations (no view check, a wrong E2M1 mask, a wrong row offset) fail them.

| bench | before | after |
| --- | --- | --- |
| `tcgen05_gather/f8_rows_256x32_sw128` | 19.06 us | 9.56 us (2.0x) |
| `tcgen05_gather/f4_rows_256x32_sw128` (new) | 20.92 us | 8.97 us (2.3x) |

| medium phase (CPU-s, 1 worker) | c257bfe | after |
| --- | --- | --- |
| `run_mma` | 8.01 | 5.96 |
| A/B operand gather | 3.38 | 1.81 |
| `tma_plan_dir` | 1.36 | 0.11 |
| process wall | 26.9 s | 22.4 s |

## `tcgen05.cp` plan memo and predicate ALU fast forms (W4, 2026-10-09)

Both changes are exact. Fixture: Mega MoE medium, 1 worker. Conformance 304 passed. Output
digests of 16 cases are identical to 01754cd at 1/8/32 workers (both Mega MoE fixtures, the
MMA-heavy set, `radix_topk_multi_cta`, `filtered_topk`, `gdn_prefill_sm100`,
`kda_decode_multishape`, `flashinfer_rmsnorm_quant`).

**`tcgen_cp_plan`.** It is a pure function of its arguments, so successful plans are memoized
per thread. Errors are not cached. `TcgenCpPlan::pairs` pre-sizes its two vectors.

The guard is `memoized_cp_plans_equal_direct_plans`:
- plan or error equal to the direct planner over every shape, multicast, decompression, arch,
  CTA group, and random descriptors and TMEM addresses;
- every sampled request also runs with each key field varied, so dropping any field from the
  key (8 mutations, one per field) fails it;
- `pairs()` is checked as the word-major, lane-minor expansion of the plan.

| | before | after |
| --- | --- | --- |
| bench `tcgen_cp/plan_pairs_32x128b_warpx4` (new) | 4.54 us | 1.29 us (3.5x) |
| medium handler, per issue (55,296 issues) | 10.8 us | 7.65 us |
| of which oplib plan + pairs | 5.3 us | 2.1 us |

The remaining handler cost is the interp's per-cell span construction (about 4 us per issue).

**ALU (`oplib::binary`/`cast`/`compare` bodies).** The `Pred` And/Or/Xor and `Pred`<->integer
(and F32/F64 -> `Pred`) casts took the generic `u128` path. They now have fast forms with the
generic results: `(x op y) & 1`, `x & 1` and `x != 0`.

| bench (`tir_32_lanes/`, partial mask) | before | after |
| --- | --- | --- |
| `binary_or_pred` | 455 ns | 20.6 ns (22x) |
| `binary_and_pred` | 428 ns | 19.6 ns |
| `cast_pred_u32` | 539 ns | 23.6 ns (23x) |
| `cast_u32_pred` | 473 ns | 22.6 ns (21x) |

The guards are the existing `fast_paths_match_the_generic_path`, which iterates `Pred`, and
the new `w4_fast_forms_match_the_generic_path_on_edges`, which uses junk upper bits.

The medium mix is about 0.54 M such calls per run, so the saving is about 0.24 s at 1 worker.
Whole-run CPU A/B at this size is within host noise (about +/-1 s).

**Measured negatives, not kept:**
- *The integer forms are already at the floor.* S32 add/mul, the S32/U32/U64 casts and the
  S32 compare cost 17-27 ns per 32-lane call, with one `(op, dtype)` match per call.
  The rest of the 200-350 cycles per issue that W13 measured is on the interp side.
- *Branchless blend.* A branchless partial-mask blend made partial-mask ops slower
  (22 -> 34 ns).
- *Floor-division fast path.* A pre-checked check-free floor div/mod was slower than the
  per-active-lane checked loop (75 -> 119 ns).
