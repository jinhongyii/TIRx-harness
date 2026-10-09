---
orphan: true
---

# NumSim: ISA answers for instruction forms legacy accepted

These rulings settle forms that the legacy transpiler accepted but the
hardware does not, so that v2 fails closed on them (coordinator ruling,
2026-10-08). They are checked against:

- **PTX ISA**, version 9.4, §9.7.16 (tcgen05) and its tcgen05.mma
  matrix-shape table;
- **CUDA Driver API**, `cuTensorMapEncodeTiled`.

TVM encodes the same rules (`tvm/backend/cuda/cpp/descriptors.py`
`_TCGEN05_MMA_SHAPE_RULES`, and the TMA dispatch descriptor checks). Behaviour
rows L4–L6 are in `numsim-behaviour-deltas.md`.

## tcgen05.mma shapes

For `.kind::f16`, `.kind::tf32` and `.kind::f8f6f4` (dense):

| cta_group | M | valid N |
| --- | --- | --- |
| 1 | 64 | N % 8 == 0, 8 ≤ N ≤ 256 |
| 1 | 128 | N % 16 == 0, 16 ≤ N ≤ 256 |
| 2 | 128, 256 | N % 32 == 0, 32 ≤ N ≤ 256 |

The rows for `.kind::i8` and the block-scaled kinds follow the same table,
reproduced in `_TCGEN05_MMA_SHAPE_RULES`.

Consequences:

- `M = 128, N = 8` at `cta_group::1` and `N = 16` at `cta_group::2` are not
  hardware instructions. Legacy's table (`emit/tcgen_descriptor.rs`
  `SHAPE_ENCODINGS`, which uses N%8 for M=128 and N%16 at cta_group::2) was
  wrong.
- v2 rejects these at transpile time: `tile.gemm_async` through TVM's
  dispatch, and raw `tcgen05.mma` through the oplib shape table, once
  `numsim-oplib/src/tcgen05/encode.rs` is corrected (W4).

## Block-scaled scale-factor K extent

For block-scaled `tcgen05.mma` (`kind::mxf8f6f4` / `mxf4` / `mxf4nvf4`), the
scale-factor operand's K extent per MMA instruction must be one of {1, 4, 16},
as TVM's dispatch requires. An SFA/SFB region whose K extent is 8 does not
describe a hardware scale-vector layout. v2 rejects it.

## TMA inner box size

`cuTensorMapEncodeTiled` requires, for non-interleaved tensor maps
(`CU_TENSOR_MAP_INTERLEAVE_NONE`), that `boxDim[0] * elementSizeInBytes` be a
multiple of 16 bytes.

- A 4-element 16-bit box (8 bytes) cannot be encoded, so no TMA (load, store
  or `cp.reduce.async.bulk.tensor`) exists for it.
- v2 rejects such `copy_async(dispatch="tma*")` forms at transpile time; TVM
  declines both `tma_auto` and `tma_explicit`.

## TMA into a padded shared layout

A tile `copy_async` whose shared destination is a padded (non-contiguous) slice
cannot be one TMA box, because TMA writes a dense box.

- Legacy compiled `tma_padded_narrow_rows` and then rejected it at run time
  ("TMA shared payload component 1 … must be 128-byte aligned").
- v2 rejects it at transpile time instead. It still fails closed; only the
  phase differs.

## Cluster size (open hardware limit)

CUDA limits a thread-block cluster to 8 CTAs (portable) or 16 CTAs with the
non-portable cluster-size opt-in on sm_90 and sm_100. A kernel declaring more
cannot launch.

- v2 keeps the legacy engine limit: more than 64 CTAs per cluster fails closed
  at transpile (`topology: N CTAs per cluster, maximum 64`). The 16-CTA hardware
  limit is **not** enforced.
- Reason: 14 kernels in the lowering sweep use 20-CTA clusters, and legacy
  accepted them. Enforcing the hardware limit needs a behaviour-delta row and a
  test migration; that decision is deferred to after the legacy deletion
  (coordinator ruling, 2026-10-08).
- Threads per CTA: more than 1024 (32 warps) fails closed at transpile, which is
  both the hardware limit and legacy's `warps_per_cta` maximum.
