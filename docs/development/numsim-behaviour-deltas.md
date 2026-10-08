---
orphan: true
---

# NumSim numerical behaviour deltas vs. legacy

Use this list to review conformance-snapshot diffs in register and memory
values. The new numerics are defined by `core-rs/numsim-oplib/` (bit-exact
kernels moved from the legacy engine) and `core-rs/numsim-core/src/oplib/`
(the contract entry points: TIR ALU, `resolve_ptx`, TMA, tcgen05). A value diff
that matches no row here is a regression.

**Status column.**
- **Confirmed** rows keep legacy behaviour by ruling. They are listed so that a
  future change is caught.
- **Open** rows are questions that a GPU golden should settle. The last column
  says which golden.
- **New** rows add behaviour legacy did not have. Legacy rejected these forms.

## TIR-level arithmetic (`Instr::Unary/Binary/Ternary/Compare/Cast`)

| ID | Status | Change | Legacy | New | What a GPU golden would settle |
| --- | --- | --- | --- | --- | --- |
| D1 | Confirmed (coordinator ruling, 2026-10-07; implemented in lowering 2026-10-08) | Scalar f16/bf16 TIR expression chains | The frontend carried every f16/bf16 TIR value as an `f32` (`expr.rs` `cast_atom`). Arithmetic stayed in f32 across the chain. It rounded only at a cast to half (`f32_to_fp16_bits`) or a memory store. | Matches legacy, done by lowering. W1 (`ir_walk.py` `half_chain`) evaluates half expression trees in `Ty{F32, lanes}` registers and emits one `Cast` back to half where the value leaves the chain: a store, an explicit cast, a call operand or a `Bind`. OpLib half ops (`tir`) round per op (RNE), and half registers always hold plain zero-extended bits. The interim bit-16/bits-32..63 f32 carrier in OpLib was removed. | Whether TVM's CUDA codegen evaluates `half` expression trees in `float` or with `__hadd`-style per-op rounding. Settle with a kernel that computes `(a+b)+c` with `a=1`, `b=c=2^-11` in f16 and stores the result. Legacy and the current model give `0x3c01`; per-op rounding gives `0x3c00`. |
| D2 | Open | `UnOp::Round` | Legacy never emitted `tirx.round`. | Ties away from zero (`f32::round`), on the assumption that TVM's CUDA backend emits `roundf`. | `round(2.5f)` and `round(-0.5f)` on the GPU through `T.round`. `roundf` gives 3.0 and -1.0; `nearbyint`/`rint` gives 2.0 and -0.0. |
| D3 | Open | Unary/binary math legacy never emitted: Sqrt, Exp2, Sin, Cos, Tanh, Floor, Ceil, Trunc, Pow, Atan2, Copysign, f64 math, and f16 math beyond what legacy carried | Legacy only had exp, fabs, log, log1p, log2, rsqrt, sigmoid, popcount (u32) and fma, all f32. These use host `std` (`.exp()`, `.ln()`, ...). | Host `std`/libm for every op. Transcendentals are therefore host-libm values, not CUDA libdevice values. | A sweep of `T.sin`, `T.cos`, `T.tanh`, `T.exp2`, `T.pow` and `T.atan2` over representative f32 inputs, compared ULP-exact with the host values. If they differ, either model libdevice bit-exactly or mark these ops `deterministic_representative`. |
| D4 | Open | `Cast.sat` | Legacy never set it. | Float destinations saturate to finite: ±inf becomes ±max finite and NaN stays NaN. Int destinations clamp to range. This is not PTX `.sat` (clamp to [0,1]). | None needed until lowering emits `sat`. Then settle which meaning lowering intends. |
| D5 | New | Casts to e5m2 / e2m3 / e3m2 / e2m1 with `Rounding::Default` | Legacy had no cast to these types and rejected them. | Round to nearest even, saturate to finite. This is believed to match the CUDA `__nv_fp8_e5m2` / `__nv_fp4_e2m1` constructors. | Cast `{448, 1e6, NaN, ±tie values}` through `T.Cast` to each type and store the bytes. |
| D6 | New | Explicit-rounding casts to fp8/fp6/fp4 | Rejected | Only `Rn` from sources that fit f32 exactly; e5m2 also requires `sat`. Everything else is `Unsupported`. | Already covered by the PTX `cvt` goldens. |
| D7 | Confirmed (W4-12, 2026-10-08) | f64 `BinOp::Min/Max` signed zeros | The scalar TIR `min`/`max` in legacy used Rust `f64::min/max`, where the result for `(-0, +0)` is unspecified (`+0` on x86). Legacy tile reductions used PTX `min/max.f64` instead. | `cuda_f64_min/max`: NaN-ignoring, with `-0 < +0` (device `min.f64`/`max.f64`). f32 already followed this rule. | Covered by `test_local_float64_reductions_match_b200_edge_bits`, whose B200 edge bits require `min(-0, +0) = -0`. |
| D8 | Confirmed (W4, 2026-10-08) | NaN payload of host-computed f32/f64 `fma` and of TIR `+ - * /` when the result is NaN | Legacy computed these with host `mul_add` and `+ - * /`. The payload was whatever the C library's `fmaf` returned, or which NaN operand the optimizer happened to put first, so optimized and unoptimized builds could disagree. | Pinned (`numsim_oplib::scalar::host_fma_f32/f64`, `pin_nan2_f32/f64`). FMA: the first NaN among (b, a, c), quieted; this equals glibc `fmaf` on x86 FMA hardware, and a test compares the two. `+ - * /`: the first NaN among (a, b), quieted. An invalid result (inf*0, inf-inf, 0/0) gives the x86 default NaN (`0xffc00000` / `0xfff8000000000000`). The hardware SIMD paths recompute NaN lanes with the same rule. | Whether to model the GPU instead: device f32 arithmetic returns the canonical NaN `0x7fffffff`. A kernel that stores `fma(NaN_a, NaN_b, c)` and `NaN_a + NaN_b` would show it. |

## CUDA helper and PTX value ops (`resolve_ptx`)

| ID | Status | Change | Legacy | New | What a GPU golden would settle |
| --- | --- | --- | --- | --- | --- |
| P1 | Open | `tirx.cuda.bfloat162float` NaN | The `pure.rs` emitter calls `bf16_bits_to_f32` with no explicit canonicalisation, but the registry binding listed `cuda_canonicalize_nan_f32`. | Canonicalises NaN to `0x7fffffff`, following the binding. | `__bfloat162float(0xffc1)` stored as u32. Payload-preserving widening gives `0xffc10000`; canonical gives `0x7fffffff`. |
| P2 | New | `tirx.cuda.hmin2` / `hmax2` on `float16x2` carriers | bf16x2 on `uint32` operands only | Unchanged for uint32. `float16x2` source carriers select the f16 kernels. | `__hmin2` / `__hmax2` on `__half2` with NaN and ±0 operands. |
| P3 | Confirmed | `tirx.cuda.sm100_2sm_leader_smem_addr` | Clears hardware CTA-rank bit 24 | `Unsupported`. Under the `arena::addr` cluster encoding (`(rank+1)<<24`) bit 24 has no faithful meaning. | Not a numeric question; it is an address-model decision for the contract worker. |
| P4 | New | `tirx.cuda.float22half2` / `float8tohalf8` / `half8tofloat8` | Modeled as pointer read-modify-writes | `Unsupported` through `resolve_ptx`. The operands are memory, not registers. | None. Lowering must route these to memory instructions. |
| P5 | Confirmed | `shfl.sync` whose resolved source lane is not an active participant | Error: "warp shuffle reads a non-participant lane" | The same error from `oplib::shfl_sync` and from `tirx.ptx.shfl_sync*`. The infallible compatibility `oplib::shfl` returns the source value with predicate false. | — |
| P6 | Open (W2) | `prefetch.L1::32B.valid_addr` | Validated that the address names one addressable global byte | No-op in OpLib; the check needs engine memory (CONTRACT_REQUESTS W4-9). | — |
| P7 | Open (W2) | `applypriority.async.bulk*` with `completion=bulk_group` | Joined the thread's bulk async group | No-op in OpLib unless the handler registers it with the group (CONTRACT_REQUESTS W4-9). Values never change; only group counting would differ. | — |

## Tile reductions (`tirx.tile.*` lowered through TVM dispatch, Decision 6)

v2 runs the code TVM's tile dispatch emits for the GPU (`ir_walk.dispatch_tile_primitives`). The legacy frontend (`frontend-rs/src/emit/tile.rs`) emitted its own sequential reductions seeded with an identity value. Legacy's order and seed were a simulator artefact (coordinator ruling, 2026-10-08), and the tests that pinned them are being ported to the new values (W9).

| ID | Status | Change | Legacy | New | What a GPU golden would settle |
| --- | --- | --- | --- | --- | --- |
| R1 | Confirmed (coordinator ruling, 2026-10-08) | Summation order of warp-collective reductions (`Tx.warp.sum/max/min` with `dispatch="local"`, and partial-warp widths) | Lexicographic in lane order: `acc = identity; acc = acc ⊕ v[lane]` for lane 0..width. | The dispatched `shfl.bfly` butterfly tree with offsets 16, 8, 4, 2, 1 (`Shfl` + `Binary` in the lowered program). Lane 0's sum of `[1e20, 1, -1e20, 1]` (rest 0) is `(1e20 + -1e20) + (1 + 1) = 2.0`. Legacy gave `1.0`. | The dispatched code is what runs on hardware, so no golden is needed. Affects `test_warp_collective_reduction_follows_physical_lane_ownership` and `test_local_collective_uses_lexicographic_order`. |
| R2 | Confirmed (coordinator ruling, 2026-10-08) | `Tx.max/min(..., dispatch="3input_maxmin")` on f32 | Legacy ignored the dispatch: a sequential `max.f32`/`min.f32` chain seeded with `∓FLT_MAX`, so all-NaN input gave `0xFF7FFFFF` (max) and `0x7F7FFFFF` (min). | The dispatched code: pairwise `Binary Max/Min` partials with no identity seed, then `tirx.ptx.max3`/`min3`. These ignore NaN, but an all-NaN input gives the canonical NaN `0x7FFFFFFF`. Mixed rows are unchanged, for example `max(NaN, -0, +0, ...) = +0` and `min = -inf`. | Same as R1. Affects `test_maxmin_uses_canonical_lexicographic_nan_and_signed_zero_order`. |

## Async / tensor-core numerics

| ID | Status | Change | Legacy | New | What a GPU golden would settle |
| --- | --- | --- | --- | --- | --- |
| T1 | Open | The canonical bf16 typed tile GEMM (`tile_gemm_bf16_f32_ss_cta1`) in plain NumSim runs | `numpy.matmul`, whose BLAS summation order is unspecified. Racecheck/synccheck runs used the increasing-K FMA chain. | One increasing-K binary32 FMA chain per element in every mode (`ReferenceBackend`). A `NumpyBackend` exists behind the `numpy` feature. | A bf16 GEMM with inexact data compared bit-for-bit against the GPU. The tensor-core accumulation order decides which of the two models is faithful; neither is guaranteed. |
| T2 | Confirmed | Every other MMA (`mma.sync`, `mma.sp`, `tcgen05.mma` of every kind) | One increasing-K binary32 FMA chain per output element, seeded with `D*scale` or +0. f16 D is rounded once at store. Integer forms accumulate exactly in i64, then saturate or wrap. | Same (`numsim-oplib/src/mma/mod.rs`). | Existing tcgen05 microtest goldens. |
| T3 | Open | Disabled output lanes of `tcgen05.mma` reading invalid TMEM | An invalid cell reads as NaN (float) or 0 (int). | Reads the stored bytes. This has no effect on results, because disabled lanes are never written back. | — |
| T4 | Confirmed (W1 triage 001a09f; W5 row) | `tcgen05.st` / `tcgen05.ld` (and buffer-form TMEM views) addressing TMEM lanes outside the issuing warp's 32-lane sub-partition (`raw_tcgen_mma_tf32_ts_predicated`: warp 0 stores lanes up to 111) | Accepted | `bad_address` execution error. A warp may access only TMEM lanes `32 * (warp_id % 4) .. +32`; the BufferDecl TMEM rule enforces the same restriction. | PTX ISA tcgen05 data-movement (`tcgen05.ld`/`tcgen05.st`): each warp of a warpgroup accesses only its own 32-lane sub-partition |

## Synccheck bounds and limits (v2 Python layer, W8)

| ID | Status | Change | Legacy | New | Why |
| --- | --- | --- | --- | --- | --- |
| S1 | Confirmed (W8, 2026-10-08) | `CoverageBounds` (`max_warp_preemptions`, `max_completion_schedule_deviations`) | Bounded the schedule search; the payload reported usage against the bounds. | Accepted and echoed as `coverage.requested_bounds`, with `coverage.bounded_exploration = false`. The explorer does not bound preemptions: it explores every interleaving of each projected protocol with sleep sets. A clean result is therefore at least as strong as any bounded one. | No bounded search exists to configure. |
| S2 | Confirmed (W8, 2026-10-08) | `ResourceLimits` | All seven fields bounded the search. | Enforced: `max_backtrack_nodes` → `SynccheckConfig.state_budget`, `max_loop_steps` → `transition_budget`, `max_wall_time_ms` (checked inside the search since 7f8f292). Passed through but only echoed in `coverage.resource_limits`: `max_schedules`, `max_events_per_run`, `max_total_events`, `max_diagnostic_bytes`. | The explorer has no schedule replays or per-run event streams to count; diagnostic size is bounded by deduplication. Ask W6 before adding enforcement. |
