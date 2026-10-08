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
| D1 | Confirmed (coordinator ruling, 2026-10-07) | Scalar f16/bf16 TIR expression chains | The frontend carried every f16/bf16 TIR value as an `f32` (`expr.rs` `cast_atom`). Arithmetic stayed in f32 across the chain. It rounded only at a cast to half (`f32_to_fp16_bits`) or a memory store. | Matches legacy. An inexact scalar half result keeps its rounded bits in 0..16, which is what stores and PTX ops read. It also keeps the unrounded f32 in bits 32..64 and sets flag bit 16 (`tir/alu.rs` `CARRY_FLAG`). TIR ops, compares and casts read the carried f32. `Mov`/`Select` copy it, and stores drop it. Exact results stay zero-extended. Vector halves (`float16x2` TIR arithmetic, which legacy never emitted) round per op. | Whether TVM's CUDA codegen evaluates `half` expression trees in `float` or with `__hadd`-style per-op rounding. Settle with a kernel that computes `(a+b)+c` with `a=1`, `b=c=2^-11` in f16 and stores the result. Legacy and the current model give `0x3c01`; per-op rounding gives `0x3c00`. |
| D2 | Open | `UnOp::Round` | Legacy never emitted `tirx.round`. | Ties away from zero (`f32::round`), on the assumption that TVM's CUDA backend emits `roundf`. | `round(2.5f)` and `round(-0.5f)` on the GPU through `T.round`. `roundf` gives 3.0 and -1.0; `nearbyint`/`rint` gives 2.0 and -0.0. |
| D3 | Open | Unary/binary math legacy never emitted: Sqrt, Exp2, Sin, Cos, Tanh, Floor, Ceil, Trunc, Pow, Atan2, Copysign, f64 math, and f16 math beyond what legacy carried | Legacy only had exp, fabs, log, log1p, log2, rsqrt, sigmoid, popcount (u32) and fma, all f32. These use host `std` (`.exp()`, `.ln()`, ...). | Host `std`/libm for every op. Transcendentals are therefore host-libm values, not CUDA libdevice values. | A sweep of `T.sin`, `T.cos`, `T.tanh`, `T.exp2`, `T.pow` and `T.atan2` over representative f32 inputs, compared ULP-exact with the host values. If they differ, either model libdevice bit-exactly or mark these ops `deterministic_representative`. |
| D4 | Open | `Cast.sat` | Legacy never set it. | Float destinations saturate to finite: ±inf becomes ±max finite and NaN stays NaN. Int destinations clamp to range. This is not PTX `.sat` (clamp to [0,1]). | None needed until lowering emits `sat`. Then settle which meaning lowering intends. |
| D5 | New | Casts to e5m2 / e2m3 / e3m2 / e2m1 with `Rounding::Default` | Legacy had no cast to these types and rejected them. | Round to nearest even, saturate to finite. This is believed to match the CUDA `__nv_fp8_e5m2` / `__nv_fp4_e2m1` constructors. | Cast `{448, 1e6, NaN, ±tie values}` through `T.Cast` to each type and store the bytes. |
| D6 | New | Explicit-rounding casts to fp8/fp6/fp4 | Rejected | Only `Rn` from sources that fit f32 exactly; e5m2 also requires `sat`. Everything else is `Unsupported`. | Already covered by the PTX `cvt` goldens. |

## CUDA helper and PTX value ops (`resolve_ptx`)

| ID | Status | Change | Legacy | New | What a GPU golden would settle |
| --- | --- | --- | --- | --- | --- |
| P1 | Open | `tirx.cuda.bfloat162float` NaN | The `pure.rs` emitter calls `bf16_bits_to_f32` with no explicit canonicalisation, but the registry binding listed `cuda_canonicalize_nan_f32`. | Canonicalises NaN to `0x7fffffff`, following the binding. | `__bfloat162float(0xffc1)` stored as u32. Payload-preserving widening gives `0xffc10000`; canonical gives `0x7fffffff`. |
| P2 | New | `tirx.cuda.hmin2` / `hmax2` on `float16x2` carriers | bf16x2 on `uint32` operands only | Unchanged for uint32. `float16x2` source carriers select the f16 kernels. | `__hmin2` / `__hmax2` on `__half2` with NaN and ±0 operands. |
| P3 | Confirmed | `tirx.cuda.sm100_2sm_leader_smem_addr` | Clears hardware CTA-rank bit 24 | `Unsupported`. Under the `arena::addr` cluster encoding (`(rank+1)<<24`) bit 24 has no faithful meaning. | Not a numeric question; it is an address-model decision for the contract worker. |
| P4 | New | `tirx.cuda.float22half2` / `float8tohalf8` / `half8tofloat8` | Modeled as pointer read-modify-writes | `Unsupported` through `resolve_ptx`. The operands are memory, not registers. | None. Lowering must route these to memory instructions. |
| P5 | Confirmed | `shfl.sync` whose resolved source lane is not an active participant | Error: "warp shuffle reads a non-participant lane" | The same error from `oplib::shfl_sync` and from `tirx.ptx.shfl_sync*`. The infallible compatibility `oplib::shfl` returns the source value with predicate false. | — |

## Async / tensor-core numerics

| ID | Status | Change | Legacy | New | What a GPU golden would settle |
| --- | --- | --- | --- | --- | --- |
| T1 | Open | The canonical bf16 typed tile GEMM (`tile_gemm_bf16_f32_ss_cta1`) in plain NumSim runs | `numpy.matmul`, whose BLAS summation order is unspecified. Racecheck/synccheck runs used the increasing-K FMA chain. | One increasing-K binary32 FMA chain per element in every mode (`ReferenceBackend`). A `NumpyBackend` exists behind the `numpy` feature. | A bf16 GEMM with inexact data compared bit-for-bit against the GPU. The tensor-core accumulation order decides which of the two models is faithful; neither is guaranteed. |
| T2 | Confirmed | Every other MMA (`mma.sync`, `mma.sp`, `tcgen05.mma` of every kind) | One increasing-K binary32 FMA chain per output element, seeded with `D*scale` or +0. f16 D is rounded once at store. Integer forms accumulate exactly in i64, then saturate or wrap. | Same (`numsim-oplib/src/mma/mod.rs`). | Existing tcgen05 microtest goldens. |
| T3 | Open | Disabled output lanes of `tcgen05.mma` reading invalid TMEM | An invalid cell reads as NaN (float) or 0 (int). | Reads the stored bytes. This has no effect on results, because disabled lanes are never written back. | — |
