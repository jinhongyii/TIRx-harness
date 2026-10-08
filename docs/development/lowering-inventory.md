---
orphan: true
---

# Lowering: TIRx inventory and `Program` design (W1)

Status: draft, 2026-10-07, branch `refactor/clean-core`. This is the W1 input to
[`numsim-redesign.md`](numsim-redesign.md), covering §2.1 (Lowering), §2.2
(`Program`), §4.1 (W1) and §6 (the tile-form/layout risk). The skeleton lives in
`tirx_harness/src/tirx_harness/numsim/v2/lowering/`. Its tests are in
`tirx_harness/tests/numsim/v2/test_lowering_vector_add.py`.

The contract crate `core-rs/numsim-core` is not final. Anything this document
asks of it is listed in [§B.13](#b13-what-the-contract-needs-that-the-sketch-lacks).
In the code, every such point carries a `# CONTRACT:` marker.

---

## Part A: Inventory

### A.0 Method

The input is whatever reaches `numsim.transpile`. A pytest plugin wrapped the
legacy entry point `transpiler.host_prelude.normalize_host_tensor_map_prelude`.
For every PrimFunc the test suite hands to NumSim, it saved the post-prelude IR
(`tvm.ir.save_json`) and then skipped the test. It was run over `tests/numsim`
and `tests/analysis_tools`.

- **Captured:** 3010 transpile calls, which give **2339 distinct PrimFuncs**.
- **Added by hand:** two corpus kernels that the current legacy host-prelude
  normalizer rejects. They were captured raw, before normalization:
  - `fp16_bf16_gemm`
  - `mla_dsv4_multishape`

  Both fail on `main` today with `unsupported host statement before
  tirx.device_entry: Evaluate`. Their host prelude contains
  `T.tensormap_encode_tiled(...)` as an `Evaluate` statement, and the legacy
  normalizer appears to accept only the `tvm_call_packed` encode form.
- **Walker:** a reflective IR walker over `__tvm_ffi_type_info__.fields`. It
  dispatches on `type_key`, so it misses no node kind.

Two populations are reported:

- **corpus (194 PrimFuncs).** Every kernel reached from:
  - `tests/numsim/corpus/*` (`canonical_cases.py` and the per-family corpus tests)
  - `tests/analysis_tools/racecheck/{corpus,wiki}`
  - `tests/analysis_tools/synccheck/corpus`
  - the few `tests/numsim/microtests/*` that transpile without a GPU

  The wiki kernels are the canonical `tirx_kernels` families with other
  configs. They arrive through `racecheck/wiki/_cases.py`, which builds them
  from the same `prepare_*` functions. The `microtests/cases/*` kernels are
  GPU-gated: the GPU check skips them before transpile. So only the CPU-runnable
  conversion microtests are in the capture. The `runtime/` tests cover the same
  ops.
- **all (2339 PrimFuncs).** The corpus plus every unit/runtime/integration
  kernel. This is the long tail that the legacy registry was built for.

Format below: `occurrences / kernels`.

**Headline facts.**

1. **The corpus is tirx-lite, not tile IR.**
   - Every corpus kernel is authored in `tirx_lite` (`@txl.kernel`).
   - `tirx_lite/low_level_ir.py` *forbids*:
     - `TilePrimitiveCall`
     - direct `BufferLoad`/`BufferStore` on `global`/`shared*`
     - `cuda.func_call`, unless explicitly allowed
   - In the corpus, `tirx.tile.*` appears **4 times in 2 kernels** (`tile.copy`).
     All 23 tile ops appear only in unit tests: 804 calls in 401 PrimFuncs.
   - Consequence: the frontend-rs `tile_forms/` analysis (about 6K lines) is
     *not* on the corpus critical path. See [§B.7](#b7-tile-ops).
2. **Everything is a `tirx.ptx.*` / `tirx.cuda.*` call.** Memory traffic to
   global and shared goes through `ptx.ld/st*`, TMA, `tcgen05`, and similar.
   Plain `TensorLoad`/`BufferStore` is almost entirely on `local` buffers.
3. **Locals are registers.** All 63.5K corpus local allocations have a static
   shape (92% are scalars). Of their 403K loads, 98.5% use constant indices:
   - 397K constant-index local loads
   - 2.9K (0.7%) indexed only by an unrolled-loop variable
   - 3.1K (0.8%) truly dynamic

### A.1 IR node kinds (corpus)

| node | occ / kernels | notes |
| --- | --- | --- |
| `ir.IntImm` | 606781 / 194 | 93% int32. Also bool, uint32, int64, uint64, uint16, uint8, int16, int8 |
| `ir.Var` | 562445 / 194 | |
| `ir.StringImm` | 531668 / 194 | PTX modifier slots and the marker (see A.9) |
| `ir.TensorLoad` | 423834 / 194 | local 430K (all). Many are destination operands or `address_of` targets, not reads |
| `ir.Call` | 195878 / 194 | 220 distinct ops (566 in all) |
| `tirx.Evaluate` | 126010 / 194 | |
| `tirx.AllocBuffer` | 63703 / 187 | local 63524, shared.dyn 166, shared 13 |
| `prim.Add` / `Mul` / `Cast` | 45716 / 43291 / 27552 | |
| `tirx.BufferStore` | 29632 / 187 | local 97%, then global and shared scalars |
| `ir.FloatImm` | 27041 / 157 | float32, float64, float16, bfloat16, float8_e4m3fn, float8_e8m0fnu |
| `prim.BitwiseAnd/Xor/Or/Not`, `LShift`, `RShift` | 11872 / 4513 / 3266 / 2813 / 6073 / 3237 | descriptor and phase-bit arithmetic |
| `tirx.IfThenElse` | 11699 / 186 | 10% have an else |
| `tirx.SeqStmt` | 9220 / 194 | |
| `prim.FloorDiv` / `FloorMod` | 8633 / 5638 | `Div` 129/15, `Mod` 22/4 |
| `prim.EQ` / `LT` / `NE` / `GE` / `GT` / `LE` | 7213 / 3565 / 3177 / … | |
| `prim.Select` | 4674 / 54 | |
| `tirx.For` | 3234 / 169 | `T.unroll` 2425 (static), serial static 365, serial dynamic 444 |
| `tirx.DeclBuffer` | 2888 / 163 | views over the `shared.dyn` pool (2155), local (471), global (228), tmem (276 in all) |
| `tirx.ScopeIdDefStmt` | 816 / 194 | see A.6 |
| `tirx.While` | 1176 / 92 | mostly uniform conditions |
| `tirx.Bind` | 55 / 16 | `x: T.let = …` (1668 in all) |
| `tirx.AttrStmt` | 491 / 194 | keys: `device_entry`, `launch_bounds_min_blocks_per_sm`, `dyn_smem_bytes`, `required_block_size`, `max_registers` |
| `tirx.Break` | 135 / 14 | `Continue` appears only as the `tirx.continue_loop` call, in unit tests |
| `tirx.TilePrimitiveCall` + `ExecScope` | 4 / 2 | 804 / 401 in all |
| `ir.TensorRegion` | 1751 (all) | tile operands only |
| rare (all only) | | `prim.Shuffle` 12, `prim.Ramp` 4, `prim.Let` 2, `tirx.Return` 4, `tirx.AssertStmt` 1, `tirx.TensorMapEncodeTiledAttr` 5, `tirx.IterVar` 4 |

There are no `LetStmt`, `Allocate`, `BufferRealize`, or block nodes. Scalars
are 1-element `local` buffers: `i = bx*128+tx` becomes an `AllocBuffer` of
`i[1]` plus a `BufferStore`.

### A.2 Builtins: top 30 by frequency (corpus)

| # | op | occ / kernels | family |
| --- | --- | --- | --- |
| 1 | `tirx.ptx.mov_pack_b32x2` | 26865 / 108 | register |
| 2 | `tirx.address_of` | 22978 / 186 | pointer formation |
| 3 | `tirx.ptx.mov_unpack_b32x2` | 13202 / 120 | register |
| 4 | `tirx.cuda.cvta_generic_to_shared` | 12611 / 168 | address |
| 5 | `tirx.ptx.mov` | 9208 / 117 | register |
| 6 | `tirx.cuda.make_float2` | 8832 / 48 | register (vector) |
| 7 | `tirx.reinterpret` | 8730 / 185 | pure |
| 8 | `tirx.ptx.mul` | 8619 / 95 | register |
| 9 | `tirx.ptx.ex2` | 7337 / 62 | register (approx) |
| 10 | `tirx.ptx.fma` | 5858 / 129 | register |
| 11 | `tirx.ptx.cvt_f16x2_f32` | 4373 / 27 | cvt |
| 12 | `tirx.ptx.ld` | 3855 / 178 | memory |
| 13 | `tirx.ptx.add` | 3711 / 83 | register |
| 14 | `tirx.ptx.cvt_bf16x2_f32` | 3168 / 43 | cvt |
| 15 | `tirx.ptx.st` | 2698 / 163 | memory |
| 16 | `tirx.buffer_data` | 2686 / 163 | pointer formation |
| 17 | `tirx.cuda.elect_sync` | 2602 / 116 | warp |
| 18 | `tirx.ptx.abs_f` | 2032 / 20 | register |
| 19 | `tirx.cuda.float2_x` / `float2_y` | 1901 / 31 each | register (vector) |
| 20 | `tirx.ptx.cvt` | 1830 / 98 | cvt |
| 21 | `tirx.ptx.ld_vec` | 1787 / 88 | memory |
| 22 | `tirx.cuda.mbarrier_wait` | 1703 / 114 | sync, **blocking** |
| 23 | `tirx.ptx.add_int` | 1670 / 96 | register |
| 24 | `tirx.ptx.st_vec` | 1613 / 149 | memory |
| 25 | `tirx.ptx.mbarrier_init` | 1401 / 141 | sync |
| 26 | `tirx.cuda.get_tmem_addr` | 1311 / 90 | tcgen |
| 27 | `tirx.ptx.shfl_sync` | 1188 / 55 | warp collective |
| 28 | `tirx.ptx.max` | 1112 / 31 | register |
| 29 | `tirx.ptx.tcgen05_ld` | 1108 / 135 | tcgen ld/st |
| 30 | `tirx.ptx.tcgen05_mma_ss` | 1086 / 44 | tcgen MMA |

Next in line: `cuda.thread_rank` 885, `ptx.mbarrier_arrive` 858,
`prim.if_then_else` 844, `ptx.sub` 842, `ptx.tcgen05_mma_ts` 796,
`ptr_byte_offset`/`type_annotation` 726, `ptx.max3` 699, `ptx.shl` 693,
`ptx.rcp` 683, `cuda.iket_range_*` 670-679, `ptx.tcgen05_st` 651,
`ptx.tcgen05_commit` 606, `ptx.tcgen05_wait` 596, `ptx.bar_sync_count` 594,
`ptx.prefetch` 593, `cp_async_bulk_tensor_g2s_cluster` 573, `ptx.fence_proxy`
541.

**By family (corpus, totals):**

| family | total | distinct | members (abridged) |
| --- | --- | --- | --- |
| register (PTX scalar/packed ALU, cvt) | 94384 | 62 | mov/pack/unpack, mul/add/sub/fma/ex2/rcp/rsqrt/lg2/tanh, cvt_* (f16x2, bf16x2, f8x2, f4x2, ue8m0x2, tf32, `cvt_rs_*`), max/max3/min, selp/setp, shl/shr/and/or/xor/popc/bfind/clz/prmt/fns |
| pure TIR | 36556 | 19 | address_of, reinterpret, buffer_data, ptr_byte_offset, type_annotation, tvm_warp_shuffle{,_xor,_up}, tvm_warp_activemask, fabs/exp/log/rsqrt/popcount, tvm_storage_sync, isnullptr, host-only tvm_stack_alloca / tensormap_encode_tiled |
| cuda helpers | 34754 | 35 | cvta_generic_to_shared, make_float2/float2_x/y, elect_sync, get_tmem_addr, thread_rank, float22{half2,bfloat162_rn}, hmin2/hmax2, fmul2_rn/fadd2_rn, uint_as_float, ffs_u32, clock64, mov_sreg, printf, trap_when_assert_failed, iket_* (instrumentation), runtime_instr_desc |
| memory ld/st | 10408 | 9 | ld, ld_vec, ld_vec256, st, st_vec, st_vec256, st_async(_vec), st_bulk |
| mbarrier | 5059 | 12 | init, arrive, arrive_expect_tx, arrive_nocount, arrive_count_state, try_wait(_parity)(_no_hint), expect_tx, complete_tx, cuda.mbarrier_wait(_acquire_cluster) |
| tcgen05 | 6719 | 23 | mma_{ss,ts,ws_ss,ws_ts,block_scale_ss,block_scale_block_ss}, ld/st/ld_red/ld_split, cp, commit(_multicast), wait, fence, alloc/dealloc(_exclusive), relinquish_alloc_permit, encode_{matrix,instr}_descriptor(_block_scaled) |
| barriers/fences | 1995 | 13 | bar_sync(_count), barrier_sync_count, bar_arrive, bar_warp_sync, barrier_cluster_{arrive,wait}, bar_red_popc_count, fence_proxy, fence_mbarrier_init, fence_proxy_tensormap_{release,acquire}, cta_sync, warp_sync, warpgroup_sync, cluster_sync |
| TMA / bulk / cp.async | 1599 | 18 | cp_async_bulk_tensor_{g2s_cluster,g2s_cta,s2g,prefetch}, cp_reduce_async_bulk_tensor, cp_async_bulk_{g2s_cta,g2s_cluster,s2g,s2c,prefetch,commit_group,wait_group}, cp_async_{cg,ca}_src_size, cp_async_{commit,wait}_group, cp_async_mbarrier_arrive |
| warp collectives | 1650 | 7 | shfl_sync, vote_sync(_ballot), redux_sync(_f32,_bitwise), ptx.elect_sync, `__shfl_sync` |
| atomics | 485 | 3 | ptx.atom, ptx.red, red_half |
| matrix (sync) | 314 | 4 | ldmatrix, stmatrix(_m16n8_b8), mma (3 kernels) |
| misc | — | — | setmaxnreg 258, mapa/mapa_u32/cvta 302, tensormap_replace_{dim,address,stride} 144, clusterlaunchcontrol_* 98, griddepcontrol 86, wait_until 160/10 |

**The long tail outside the corpus** (346 ops, all in unit tests):

- tile ops: copy_async, gemm_async, and the 21 elementwise/reduce ops
- `cuda.func_call`: 31 kernels, gdn/flashkda helper bodies matched by source text
- `cuda.atomic_{add,cas}`, `cuda.warp_reduce`, `cuda.cta_reduce`
- the `*_report` mbarrier waits
- `break_loop` / `continue_loop`
- im2col / override / multicast TMA variants
- sparse and ti16 / lut / collector MMA variants
- `mma_sp*`, `ptx_legacy.*`
- `spcompress` / `spdecompress`
- `ldu`, `ld_proxy_readonly`, `createpolicy_*`, `applypriority*`
- `getctarank`, `isspacep`

**Exotic items to flag:**

- `tirx.cuda.iket_*` instrumentation markers: 25 kernels, about 1.9K calls.
- `tirx.cuda.printf`: 8 kernels.
- `cuda.trap_when_assert_failed`: 40 kernels. This is the `T.cuda` assert form,
  not `AssertStmt`.
- `uint128` locals: `clc_response`, the clusterlaunchcontrol payload.
- `float4_e2m1fnx32` / `boolx128` scalar params: one kernel.
- `tirx.TensorMapEncodeTiledAttr` nodes.
- `numsim.testing.FutureStatement`: an unknown statement kind in one test
  kernel. This is deliberate and proves fail-closed behavior on new node kinds.
- `numsim.unknown_control` / `numsim.unknown_loop` attrs: tests of the same
  kind.

### A.3 Dtypes

- **Expression dtypes (corpus):**

  | dtype | count |
  | --- | --- |
  | float32 | 248K |
  | int32 | 143K |
  | uint64 | 126K |
  | `""` (handle and void) | 126K |
  | uint32 | 122K |
  | bool | 21K |
  | int64 | 12K |
  | uint16 | 5.8K |
  | bfloat16 | 3.1K |
  | uint8 | 2K |
  | float16 | 1.6K |
  | int8 | 448 |
  | uint128 | 274 |
  | float8_e4m3fn | 142 |
  | float4_e2m1fn | 42 |
  | float64 | 16 |
  | uint32x4 | 16 |
  | bfloat16x2 | 3 |

  The full suite adds:
  - float16x2/x8, bfloat16x4/x8, float32x2/x4, uint64x2, int16, int32x2
  - float8_e5m2, float8_e8m0fnu, float8_e3m4x2, float6, uint6
- **Buffer element dtypes:**
  - **local:** uint64, float32, uint32, int32, uint16, int64, uint128, bfloat16, bool, uint8, float16, int8
  - **shared.dyn:** uint64 (mbarriers), float32, uint8, uint32, bfloat16, float16, int32, int8, float8_e4m3fn, float4_e2m1fn, uint16, int64
  - **global:** float32, int32, bfloat16, uint32, uint8, int64, float16, uint64, int8, uint16, float64, float8_e4m3fn, uint32x4
- **Vector lanes:**
  - Vector dtypes such as `float16x2` and `uint32x4` exist as expression,
    buffer and param dtypes.
  - `Ramp` is used for vector buffer indices only 4 times, all in unit tests.
  - Packed values are produced and consumed through `mov_pack/unpack`,
    `make_float2`, and `cvt_*x2`.

### A.4 Buffers, params, layouts, swizzle, tensor maps

**Params (corpus):**

| kind | occ / kernels |
| --- | --- |
| global `T.Buffer` | 1209 / 181 |
| `PointerType` (`txl.gptr`) | 689 / 137 |
| `int64` scalar | 680 / 22 |
| `int32` scalar | 183 / 42 |
| `uint32` scalar | 82 / 64 |
| `float32` scalar | 31 / 23 |

- 898 corpus global buffers have **dynamic shapes**. Those shapes are
  expressions over scalar params; the legacy `shapes.rs` makes them
  host-evaluable.
- Tensor maps reach the kernel in two forms:
  - An explicit `TensorMap` param. Inside the kernel it is used as
    `T.reinterpret(handle, T.address_of(v))`.
  - The **host prelude**: `v: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)`
    followed by `T.tensormap_encode_tiled(v, a.data, dims…, strides…, box…, elem_strides…, descriptor_dtype=…, rank=…, interleave=…, swizzle=…, l2_promotion=…, oob_fill=…)`
    before `tirx.device_entry`. The legacy frontend promotes these to
    `numsim.implicit_tensor_maps` (23 corpus kernels).
- **Func attrs:**

  | attr | count |
  | --- | --- |
  | `global_symbol` | all |
  | `tirx.cuda_arch` | 185 |
  | `tirx.kernel_launch_params` | 97 |
  | `numsim.implicit_tensor_maps` | 23 |
  | `tirx.persistent_kernel` | 5 |

- **Kernel-owned memory:**
  - Shared memory is one `AllocBuffer((0,), "uint8", scope="shared.dyn")` pool
    per kernel (166 of 194).
  - Every logical shared buffer is a `DeclBuffer(data=pool.data, elem_offset=…, align=…)`
    view (2155 in the corpus). Static `shared` allocations are rare (13).
  - TMEM appears as `DeclBuffer(scope="tmem")` views (276 in all).
    `cuda.get_tmem_addr` forms the address.
- **Layouts:**

  | layout | occurrences | where |
  | --- | --- | --- |
  | Trivial `TileLayout` (single axis `m`, stride 1) | 63995 local, 1930 shared.dyn, 1437 global | |
  | `ComposeLayout(swizzle_len, per_element, atom_len, TileLayout(...))` | 391 | shared.dyn |

  The `ComposeLayout` parameters, by frequency:
  - `(3,1,3)`, `(3,2,3)`, `(3,3,3)`, `(4,1,3)`, `(4,3,3)`, `(2,1,3)`,
    `(4,2,2)`: the 32/64/128B swizzles at 8/16/32-bit element widths.
  - Multi-dim inner TileLayouts such as `S[(1,32,16,8):(256,1,256,32)]`.

  Register layouts with `laneid` / `tid_in_wg` axes appear only in tile-op unit
  tests (49 occurrences). Swizzles matter to the lowering only through
  descriptors and TMA. No corpus kernel does a plain load/store through a
  swizzled buffer: those go through `ptx.ld/st` on `address_of`, and the
  address arithmetic is already explicit in the IR.
- **Swizzle and tensor-map configs** (from `tensormap_encode_tiled` and the
  implicit maps):
  - rank 2 to 5
  - `swizzle` 0 to 3, plus the `128B_ATOM_*` variants in unit tests
  - `interleave` 0
  - `l2_promotion` 2
  - `oob_fill` 0
  - box rows up to 256
  - element strides 1

  The runtime `tensormap_replace_{dim,address,stride}` path appears in 8 corpus
  kernels (gdn/kda).

### A.5 Control flow shapes (corpus)

- **`If` conditions:** classified by taint from scope ids through Bind and
  stores, so the split is approximate.

  | class | count | meaning |
  | --- | --- | --- |
  | uniform | 4910 / 167 | kernel-constant or CTA/cluster-uniform |
  | elect-gated | 2539 / 116 | |
  | warp-uniform but warp-varying | 2385 / 152 | role dispatch on `warp_id` |
  | lane-divergent | 1865 / 167 | |

- **`For`:**
  - 2425 `T.unroll` loops, all with static extents
  - 365 serial loops with static extents
  - 444 serial loops with dynamic, uniform extents
  - 55 loops with lane-divergent extents (4 kernels)
  - maximum nesting depth 3
  - annotations: `disable_unroll` and `pragma_unroll` only, in unit tests
  - one explicit `step`, in a unit test
- **`While`:**
  - 1017 uniform conditions: the persistent scheduler loops,
    `while sched_done == 0`
  - 122 lane-divergent conditions
  - 37 warp-varying conditions
- **`Select`:** 4674 / 54. `prim.if_then_else`: 844 / 100.
- **Loop exits:** `Break` statements 135 in 14 corpus kernels. The
  `break_loop` and `continue_loop` calls (18 and 14) appear only in unit
  kernels.
- **Spin loops:** the corpus uses blocking ops, not spin loops:
  - `cuda.mbarrier_wait`, which is blocking
  - `ptx.mbarrier_try_wait_parity` *inside* `while` loops (20 kernels): a
    data-dependent spin that the engine must park
  - `cuda.wait_until` (10 kernels)

### A.6 Scope ids and topology (corpus)

| `ScopeIdDef` binding | count |
| --- | --- |
| warp>thread (`lane_id`) | 194 |
| cta>warp | 192 |
| kernel>cta | 189 |
| cta>thread | 184 |
| cluster>cta | 35 |
| cta>warpgroup | 14 |
| warpgroup>thread | 5 |
| warpgroup>warp | 2 |
| kernel>cluster | 1 |
| cluster>cta_pair | 2 (all) |

- **Extents:** 810 static and 6 dynamic, where the grid depends on a scalar
  param. In the full suite, 44 are deferred (`extents=None`, inferred from
  siblings).
- **Multi-coordinate ids:** for example `cta_id_in_cluster([2,1])`. The first
  coordinate varies fastest (legacy `emit/stmt.rs:1200`).

### A.7 Elect and lane predicates

- **Elect:**
  - `if T.cuda.elect_sync() != T.uint32(0):` or `if T.cuda.elect_sync():`
    (2602 calls)
  - `T.ptx.elect_sync` (382), which writes a pred and a lane id through
    destination operands
- **Lane predicates:** comparisons on `lane_id`, `thread_id` or
  `cuda.thread_rank()`, e.g. `if T.cuda.thread_rank() == 0`, `if v_5 == 0`,
  `if (v_4 & 3 == 0) & (v_5 == 0)`.
- **PTX instruction predicates:**
  - passed as operands, with the `pred` marker
  - `p<N>` markers make operand N a predicate register, e.g. the
    `enable_input_d` of `tcgen05_mma`
- **Warp roles:** nested `if v_4 == k` / `if v_4 >> 2 == 1` on `warp_id`, which
  is warp-uniform.

### A.8 Declarations

| TIRx form | IR | lowering view |
| --- | --- | --- |
| `x: T.int32` / `x = e` | `AllocBuffer(x, (1,), local)` + `BufferStore` | register |
| `buf = T.alloc_local((64,))` | `AllocBuffer` local, static | register array (64 regs) |
| `v: T.let = e` | `Bind(var, value)` | register |
| `T.decl_buffer(..., data=pool.data, elem_offset=k)` | `DeclBuffer` | view into the CTA shared pool, static byte offset |
| global `T.Buffer` param | `Var` with `BufferType(global)`; shape may reference scalar params | `BufferDecl` + `ParamSlot` |
| `txl.gptr(T)` param | `Var` with `PointerType` | `ParamSlot(pointer)` + 64-bit address register |
| `T.cta_id/warp_id/lane_id/thread_id/...` | `ScopeIdDefStmt` | `ReadSpecial` + decomposition |

### A.9 Call idioms the lowering must understand

1. **Table-driven PTX ABI.**
   - The wire form of `tirx.ptx.<entry>(operands…, [pred], "mod_slot_0", …, "marker")`
     is one `StringImm` per modifier slot, then a marker string.
   - The marker grammar: `pred`, `keep`, `pN` (lane N is a predicate register),
     `sN` (lane N is sunk).
   - Destination operands are passed in place: a `TensorLoad` of a local
     buffer, or `address_of(local)`. Examples:
     - `T.ptx.ld(clc_response, addr, …)`
     - `T.ptx.mapa(buffer_26, …)`
     - `encode_matrix_descriptor(T.address_of(desc), …)`
   - `tvm.backend.cuda.ptx.table.TABLE` (506 entries) defines operand roles:
     `reg` / `addr` / `ptr` / `imm`, read/write, lanes, carrier dtypes.
   - The legacy Python `ptx_dialect.decode_ptx_call` already decodes all of
     this into a `DecodedPtxCall`.
2. **Address formation:**
   - `address_of(buf[idx])` gives a generic 64-bit pointer.
   - `cuda.cvta_generic_to_shared(...)` gives a 32-bit shared::cta address.
   - `ptx.mapa` gives a shared::cluster address.
   - `ptr_byte_offset`, `buffer_data`, `reinterpret(handle, …)`, and
     `type_annotation` are pointer plumbing.
3. **Blocking helpers:**
   - `cuda.mbarrier_wait(address_of(mbar[i]), parity)`
   - `cuda.cta_sync()`, `warpgroup_sync`, `cluster_sync`
   - `ptx.bar_sync*`, `ptx.barrier_cluster_wait`
   - `ptx.tcgen05_wait`, `cp_async_bulk_wait_group(.read)`
4. **`wait_until`:** `T.cuda.wait_until(dst, T.address_of(word), pred(dst, captured…), scope, space, width, flags)`.
   - In the corpus:
     - scope `gpu`/`sys`
     - space `global`
     - width `b32`/`u32`/`u64`
     - the predicate is pure in `dst` and loop-invariant registers, e.g.
       `((dst ^ phase) & 0x80000000) != 0`, `dst >= target`,
       `dst == cast(expected)`
   - Unit tests also use predicates that **load memory**, e.g.
     `table[seen[0]] == 1`.
5. **`cuda.func_call`:** unit tests only. Recognized by name *and* by a
   whitespace-insensitive match of the CUDA source text against a fixed
   allowlist (`cuda_helper.rs:411`, 23 kinds).

### A.10 What the legacy frontend computes statically

Source: frontend-rs (58K lines). `analyze/` (7.2K lines) proves legality.
`emit/` (44K lines) prints Rust.

| fact | legacy home | verdict | shape in `Program` |
| --- | --- | --- | --- |
| Launch topology (clusters, ctas/cluster, warps/CTA, warps/warpgroup, min blocks/SM) from ScopeIdDef extents, with a constraint solver; re-evaluable under concrete scalars | `analyze/topology.rs` | **genuine** | `topology: Launch`. Extents are expressions over scalar slots (Q3) |
| Host ABI slots: buffers, then pointers, then scalars, then tensor_maps; canonical names `k{i}:{name}` for multi-kernel modules; implicit tensor map tied to its base slot; buffer shapes as host expressions | `host_abi.rs`, `analyze/frontend.rs`, `analyze/shapes.rs`, `emit/raw_tma.rs` | **genuine** | `host_abi: Vec<ParamSlot>` plus `TensorMapSpec` fields |
| Shared-memory allocation: one backing per AllocBuffer; DeclBuffer views (backing, byte_offset, byte_len); CTA arena packing at `max(16, align)`; `pool_max_bytes` / `dyn_smem_bytes` checks; TMEM footprint (lanes × columns) | `analyze/memory.rs`, `emit/module.rs:685` | **genuine**: the shared virtual addresses are observable through descriptors and `cvta` | `buffers: Vec<BufferDecl>` with `space, base, byte_len, align, view_of` |
| Per-buffer `LayoutInfo`: itemsize, `elem_offset`, strides, physical axes, fp4 nibble packing, TMEM lane/col span | `analyze/layout.rs` | **genuine** (facts) | `BufferDecl` + `TileLayout` |
| Layout signature strings | `layout.rs:169` | artifact | drop |
| Tile-op parse and legality: 23 ops; destination/operand regions; selected hardware form (tcgen05 ld/st shape/num, MMA shape, `GemmAsyncFacts` instr M/N/K, cta_group, trans) | `analyze/tile_forms/*` | **genuine** (selection + legality) | `layouts` (element maps) + Instr payload |
| Lazy closure vs materialized element map, `#[inline(never)]`, `NestedLoadSite` | `emit/tile_gemm_async.rs` | artifact | drop |
| Instruction-descriptor constants (`tcgen05_encode_instr_descriptor[_block_scaled]` folded to u32); arch variant sm_100/103/107 | `emit/tcgen_descriptor.rs` | **genuine** | `consts` + `Program.arch` |
| Matrix (smem) descriptor | runtime engine call | genuine *runtime* op | OpLib |
| Source map: `op_id` = post-order index over unique nodes; `{kind, text ≤1000 chars, span, op_name}` | `analyze/frontend.rs:339` | **genuine** | `sites: Vec<SiteInfo>`. `op_id` parity is not needed (Q7) |
| Uniform vs varying | `emit/mod.rs` | artifact (optimization) | optional `RegDecl.uniform` hint |
| ElectSync control provenance | `emit/mod.rs:90`, `stmt.rs:1375` | **genuine, small**: the engine treats elect-gated regions specially | `If.elect` flag (Q5) |
| Loop validation (SERIAL/UNROLLED only, no thread binding, limited annotations), loop enter/exit sites | `topology.rs:669`, `stmt.rs:1449` | **genuine** | `LoopBegin{site}` + fail-closed |
| 0/1-trip special cases, unroll pragmas, split thresholds | `stmt.rs`, `scaffold.rs` | artifact | drop |
| `shared_blocks.rs` dedup | Rust helper-body token dedup for rustc time, **not** shared memory | artifact | drop |
| `suspends` registry flag (140 rows), used for `.await` coloring | `registry.rs` | artifact; the fact "may return Blocked" is genuine | per-Instr property in the contract |
| Racecheck write seed (points-to over global roots) | `analyze/pointer_targets.rs` | optional hint, with a runtime fallback | drop for now |
| `semantic_requirements` (raw tensor-map registry, grid dependency), `uses_readonly_proxy`, `requires_implicit_tmem`, `uses_dynamic_tmem_lifecycle` | `analyze/frontend.rs` | **genuine** header flags | `Program.requirements` |
| wait_until recognition and predicate compile, synthesizing PTX `ld` through a Python callback | `emit/wait_until.rs` | **genuine** | `WaitUntil` + predicate sub-program (B.8) |
| cuda.func_call helper matching | `emit/cuda_helper.rs` | **genuine** op selection | builtin table entry keyed by name + source hash |
| `structural_hash`, span-free semantic JSON for cache keys | `lib.rs`, `semantic_ir.py` | cache artifact | `compile.py` owns the cache key |

**Fail-closed model.**

- Analysis collects every per-node problem as `op#N:<msg>` into
  `manifest.unsupported`. Emission runs only if that list is empty.
- The error kinds are `Unsupported`, `Unmodeled`, `NotCovered` and `Ffi`. They
  map to `UnsupportedTIRxError` and `UnmodeledTIRxFormError` in Python.
- The registry rejects:
  - WGMMA (21 rows)
  - fabric / multimem (24 rows)
  - PARALLEL / VECTORIZED / THREAD_BINDING loops
  - unknown attrs
  - non-TileLayout/ComposeLayout layouts
  - replicated register layouts
  - TMEM corner cases

### A.11 Python callbacks during legacy compile

Python applies **no TVM passes**. The only IR rewrite is the host-prelude
tensor-map promotion, and that is done in Rust (`raw_tma/host_prelude.rs`).
Live `Array<PrimFunc>` FFI objects cross into the cdylib. Rust calls back into
Python three times:

| callback | registered | input | supplies | new lowering |
| --- | --- | --- | --- | --- |
| `numsim.frontend.decode_ptx_call` | `ptx_dialect.py:396` | the `tirx.ptx.*` Call | `{op_name, modifiers, predicate, preserve_dst, result_type, operands[{name, kind, rw, lanes, literal, operand_type, dtypes, values}]}` from TVM `TABLE` helpers | call `ptx_dialect.decode_ptx_call` directly and keep the live PrimExpr operands |
| `numsim.frontend.build_ptx_call` | `ptx_dialect.py:419` | PTX spelling + operands | a fresh `tirx.ptx.ld*` Call (used to compile wait_until loads) | not needed: emit `Load`/`LoadAddr` directly |
| `numsim.frontend.tcgen05_cp_plan_error` | `native_frontend.py:55` | `tirx.tile.copy_async` node | TVM's `tcgen05_cp._build_plan` rejection text | call `_build_plan` directly and use the *plan* too |

The other TVM APIs the Rust side reaches through FFI are all reachable from
Python, so the new lowering needs **no native helper**:

- `ffi.StructuralHash`, `ToJSONGraph`
- `tirx.TileLayoutIsTrivial`
- `Layout.canonicalize/apply/apply_with_shape`
- `Analyzer.simplify/can_prove`

---

## Part B: Design

### B.1 Pipeline

```text
PrimFunc ──normalize_host_prelude──► PrimFunc'  (port host_prelude.rs to Python, accepting
                                                  the Evaluate(tensormap_encode_tiled) form)
        ──pre-pass──► topology, buffer/backing plan, local-array promotion set,
                      elect taint, unsupported list (collect, don't stop)
        ──walk──► Program (code, consts, sites, layouts, regs, buffers, host_abi, topology)
        ──strict?──► raise LoweringUnsupported(all reasons)  |  keep Unsupported instrs
```

The walk is one recursive descent over statements, plus expression lowering
into fresh registers (`ir_walk.Lowerer`). It dispatches on the FFI `type_key`
(`stmt_tirx_IfThenElse`, …), so a new node kind lands in the unsupported list
instead of being silently skipped.

### B.2 Program as the lowering emits it

The skeleton's provisional `Program` (`program_builder.py`) has the plan's six
fields plus five more:

| field | type |
| --- | --- |
| `code` | `Vec<Instr>` |
| `consts` | `Vec<Scalar{dtype, bits}>` |
| `sites` | `Vec<SiteInfo>` |
| `layouts` | `Vec<TileLayout>` |
| `topology` | `Launch` |
| `host_abi` | `Vec<ParamSlot>` |
| `regs` | `Vec<RegDecl{dtype, name, uniform}>` (added) |
| `buffers` | `Vec<BufferDecl>` (added) |
| `arch` | `Option<String>` (added) |
| `unsupported` | `Vec<String>` (added) |
| `requirements` | flags (to come) |

- **Operands:** `Reg(u32) | Const(u32)`. Constants are interned and broadcast,
  so no `Mov` from an immediate is needed.
- **Floats** are stored as bit patterns in `Scalar.bits`.
- **JSON** is serde's externally tagged form: `{"Load": {...}}`, and `"Else"`
  for unit variants. A Rust `serde_json::from_str::<Program>` works without
  adapters once the names match.

### B.3 Register model

- **A register is a `[T; 32]` with a static dtype** (`RegDecl.dtype`).
  - The lowering is SSA-like: each expression result gets a fresh register.
  - Named state gets one register per element. This covers locals, `Bind`
    vars, loop counters, and scope ids.
  - Writes to named registers are masked by the engine's active mask. A Mov
    under a divergent `If` writes only active lanes. This is where the legacy
    `WarpValue` + mask merging lives today.
- **Register-promoted local arrays.** A `local` `AllocBuffer` with a static
  shape and a trivial layout becomes `n` consecutive registers.
  - Constant-index access resolves to a register at lowering time (98.5% of
    corpus local loads).
  - Indices that depend only on unrolled-loop variables (0.7%) and truly
    dynamic indices (0.8%) use two indexed forms:

    ```
    LoadRegIndexed  { dst, base: Reg, len, idx: Operand, site }   // OOB => error finding
    StoreRegIndexed { base: Reg, len, idx, value, site }
    ```

  - Proposed: **do not unroll `T.unroll` loops in the lowering.** Indexed
    register access is exact, and unrolling 2425 loops multiplies `code` and
    sites by the trip counts. An optional later pass can unroll for the
    codegen backend.
- **Local arrays used through `address_of`.** If a local array's address
  escapes (`address_of(local[i])` passed as a `ptr` operand), the array is
  *not* promoted. It gets a per-lane `Local` backing instead
  (`BufferDecl(space=Local)`), and the escape becomes a 64-bit generic
  address. A `reg` (destination) operand is not an escape: it is lowered as a
  write to the register.
- **Uniformity** is a static hint (`RegDecl.uniform`). It is true when every
  write is provably warp-uniform. The sources are scope ids for
  cluster/CTA/warp/warpgroup, constants, and uniform loop bounds. The engine
  must be correct when it ignores the hint. The codegen backend may use it to
  store a scalar.
- **Wide and vector dtypes:**
  - `uint128`, `float16x2`, `uint32x4`, and so on are register dtypes whose
    lane payload is wider than 64 bits.
  - The contract needs `Dtype` to cover vector lanes and 128-bit values (Q4).
  - The alternative is to split them into register tuples at lowering time.
    PTX operands already arrive split per lane (`values` lists), so splitting
    is natural for PTX. `make_float2` / `float2_x` and `cvt_*x2` then become
    lane-tuple moves.

### B.4 Structured control flow

**`if c { A } else { B }`:**

```
      c' = <cond>
pc_if:   If   { cond: c', else_pc: pc_else, end_pc: pc_end, elect, site }
           A
pc_else: Else { end_pc: pc_end }           // omitted when no else; then else_pc = pc_end
           B
pc_end:  EndIf
```

- **Mask semantics:** `If` pushes `(saved = active)` and sets
  `active &= cond`. `Else` sets `active = saved & !cond`. `EndIf` pops.
- **Skips:** if `active` becomes empty at `If`, jump to `else_pc`; at `Else`,
  jump to `end_pc`.
- **Uniform conditions:** these are 42% of corpus `If`s. They take exactly one
  arm with no special opcode.
- **`elect` flag:** set when the condition is derived from `elect_sync`. The
  derivation comes from a taint pass that follows the value through
  `Bind`/locals, like the legacy `ControlProvenance::ElectSync`. Synccheck and
  racecheck use the flag to attribute the region to a single elected lane.

**Loops (`For` serial/unrolled, `While`):**

```
         <init: i = min; stop = min + extent>            (For only)
pc_lb:   LoopBegin { end_pc: pc_le, site }               // push frame: saved mask, budget, iter=0
pc_head: <cond>                                          // i < stop | while-condition
         LoopIf   { cond, end_pc: pc_le }                // active &= cond; if empty → pc_le+1 (pop)
           body
         <i = i + step>                                  (For only)
pc_le:   LoopEnd  { head_pc: pc_head }                   // active |= continued; iter++; budget/quantum; jump
```

- `LoopEnd` owns three jobs, as the plan requires:
  1. iteration budget: on overflow, the run is `incomplete` with this site;
  2. scheduler quantum: it may yield the warp;
  3. spin parking: if a whole iteration executed only non-progress
     instructions (failed `try_wait`), the warp parks until the watched
     resource changes.
- `LoopBegin.site` gives the loop-enter event that racecheck and synccheck
  need. The legacy emitter calls `for_enter` / `for_exit`.
- **`Break` / `Continue`** need two more instructions. `Break` statements
  occur in 14 corpus kernels; `tirx.break_loop` and `tirx.continue_loop`
  occur in unit tests.
  - `Break`: `break_mask |= active; active = 0` within the innermost frame.
  - `Continue`: `cont_mask |= active; active = 0`; `LoopEnd` restores it.

  `LoopIf`/`LoopEnd` then treat the frame's live mask as
  `saved & !break_mask`.
- **Divergent loop bounds** (55 occurrences) and divergent `While` (122) work
  without change: the lanes that fail `LoopIf` drop out, and the frame
  reconverges at `LoopEnd`'s exit.
- **`Return`** (4, unit tests): `Exit` with the active mask. The lanes retire
  and the remaining lanes continue. The engine needs per-lane exit in the mask
  stack (`exited` mask).
- **`AssertStmt` and `cuda.trap_when_assert_failed`** lower to
  `Assert { cond, site, message }`. A failing active lane produces an error
  finding with that site.

### B.5 Addresses and memory

There are two address forms.

1. **Buffer-relative** (`Load` / `Store { buf, offset }`). Used whenever the
   IR names a buffer and an index: `TensorLoad`, `BufferStore`, and PTX `addr`
   operands of the form `address_of(buf[idx])` whose space is known
   statically.
   - `offset` is an element offset computed from row-major strides of the
     trivial layout.
   - For a `ComposeLayout`, `TileLayout.apply` gives an affine-plus-XOR form.
     The lowering emits the swizzle arithmetic as plain
     `Binary{Xor, Shr, And}` ops, generated from the layout's
     `(swizzle_len, per_element, atom_len)`.
   - This form keeps `(alloc, range)` precise for racecheck and gives exact
     OOB per buffer.
2. **Raw address** (`LoadAddr` / `StoreAddr { addr: Reg, space, … }`). Used
   when only a pointer value exists: `txl.gptr` params, `ptr_byte_offset`
   chains, pointers loaded from memory, and shared addresses produced by
   `cvta`/`mapa` arithmetic.

**Address values:**

| expression | address register |
| --- | --- |
| `address_of(buf[i])` | u64 generic |
| `cvta_generic_to_shared(p)` | u32 shared::cta |
| `mapa(p, rank)` | u32 shared::cluster |
| `get_tmem_addr` | u32 TMEM `(lane << 16 \| col)` |

- These are ordinary register values. The engine's Arena resolves them with a
  **segment table**: each backing has a virtual base per space. The lowering
  assigns the shared bases from the pool layout, and the engine assigns the
  global bases at bind time.
- Generic → shared conversion (`cvta`) and back are pure arithmetic on these
  bases. They are OpLib ops, not lowering logic.
- **Static shared plan.** The lowering owns the `shared.dyn` pool layout:
  `BufferDecl{space: Shared, base, byte_len, align}` for the pool, and one
  view per `DeclBuffer` (`view_of: pool, byte_offset = elem_offset*itemsize`).
  The legacy arena packing rule (`max(16, align)`, sequential) is kept, because
  descriptor addresses depend on it.
- **Space selection.** For PTX memory ops the space comes from the modifier
  slot (`shared`, `shared::cta`, `shared::cluster`, `global`, `""` =
  generic). For generic accesses the engine resolves the space at run time
  (`Space::Generic`).
- **64-bit addresses.** Generic addresses are u64. Shared addresses are u32.
  The lowering never truncates silently: `Cast(u64→u32)` on an address is an
  explicit instruction, as it is in the IR.

### B.6 Builtin calls

`builtins.py` is the only acceptance table: op name → handler + family +
`may_block`.

- **PTX** (`tirx.ptx.*`, 506 table entries):
  - **Decode** with the existing `ptx_dialect.decode_ptx_call`. It is pure
    Python over `tvm.backend.cuda.ptx.table`, and the call goes there
    directly.
  - **Emit one generic instruction:**

    ```
    Ptx { op: OpId, mods: ModsId, dsts: [Reg…], srcs: [Operand…], pred: Option<Operand>,
          keep_dst: bool, space: Option<Space>, site }
    ```

    - `OpId` indexes a per-Program op table of `(table op_name, canonical modifier tuple)`.
    - The engine (OpLib) dispatches on that pair once at load time into a
      handler pointer. The bytecode never carries strings in the hot path.
  - **Promote families with their own engine semantics** to dedicated Instr
    variants, so the Observer can see them without decoding modifiers.
    Proposed:
    - `MbarInit/Arrive/ExpectTx/TryWait/Wait`
    - `TmaLoad/TmaStore/TmaReduce/TmaPrefetch`
    - `BulkCopy`, `AsyncGroupCommit/Wait`
    - `TcgenMma/TcgenLd/TcgenSt/TcgenCp/TcgenCommit/TcgenAlloc/TcgenDealloc/TcgenWait/TcgenFence`
    - `Fence`, `Barrier{kind, id, count}`, `Atomic`, `Shfl/Vote/Redux/Elect`

    The generic `Ptx` variant covers the pure register families
    (`mov/pack/cvt/fma/ex2/...`, 94K corpus calls).
- **`tirx.cuda.*` helpers.** These are mostly thin wrappers: `elect_sync`,
  `mbarrier_wait`, `cta_sync`, `thread_rank`, `make_float2`, and so on. Each
  lowers to the same Instr as its PTX equivalent, so `cuda.mbarrier_wait`
  becomes `MbarWait`. `cuda.tcgen05_encode_instr_descriptor` with all-literal
  args **folds to a const** at lowering time (port of
  `tcgen_descriptor.rs`). `cuda.tcgen05_encode_matrix_descriptor` stays a
  runtime OpLib op.
- **Pure TIR ops:**
  - `address_of`, `reinterpret`, `buffer_data`, `ptr_byte_offset`,
    `type_annotation` are address plumbing (B.5).
  - `if_then_else` becomes `Select`.
  - The math ops become `Unary`/`Math`.
  - `tvm_warp_shuffle*` becomes `Shfl`.
- **`cuda.func_call`.** Keep the legacy allowlist keyed by
  `(name, sha256(normalized source))`. Each entry maps to an Instr sequence.
  Unknown helpers are unsupported.
- **Blocking.** An instruction whose handler may return `Blocked(resource)` is
  a scheduling point. The lowering never duplicates or reorders it, and it
  must be the *only* side-effecting work in its instruction. A
  `try_wait`-style op returns a value and does not block; spin loops around it
  are parked by `LoopEnd` (B.4).
- **Instrumentation** (`iket_*`, `printf`, `clock64`): `iket_*` and `printf`
  lower to `Nop{site}` with a note. `clock64` lowers to a deterministic
  monotone counter. It must not be unsupported, or 25 corpus kernels drop.

### B.7 Tile ops

The corpus barely uses tile ops (B.0). They should land *after* the corpus
works on raw PTX, and the frontend-rs `tile_forms` logic should not be
re-derived. Use TVM's own production dispatch instead:

- Run `tirx.transform.TilePrimitiveDispatch` / `LowerTIRxOpaque` *only on the
  tile-op subtrees*. The tcgen05.cp plan already comes from TVM
  `_build_plan`. This yields raw PTX-level IR that the normal path lowers.
  Then the simulated semantics are, by construction, the semantics of the code
  that runs on the GPU. Today they are a parallel reimplementation (§6 risk).
- If dispatch output is too low-level (for example, it loses
  `copy_async`/`gemm_async` as one async op that racecheck must see as one
  footprint), fall back to an element-map instruction:

  ```
  TileOp { kind, dst: Region, srcs: [Region…], map: LayoutId, scalars, site }
  TileLayout { buffer, lanes, slots, entries[lane][slot] -> elem offset | -1 }
  ```

  - The table is computed at lowering time with
    `Layout.canonicalize().apply_with_shape(...)`, the same call the legacy
    `emit/layout.rs` makes, when the region mins are static.
  - When they are dynamic, it uses a base-offset register plus a static table
    of relative offsets.
  - Tables are deduplicated by content in `Program.layouts`.

Recommendation: try dispatch first (Q6). Measure on the 401 unit-test tile
kernels against the legacy goldens.

### B.8 `wait_until` predicates

`T.cuda.wait_until(dst, addr, pred, scope, space, width, flags)`:

```
WaitUntil { dst: Reg, addr: Addr, dtype, space, scope, sem,
            pred: PredId, captures: [Reg…], site }          // may return Blocked(addr word)
Program.preds: Vec<PredProgram { arg: Reg, code: Range<pc>, result: Reg, reads_memory: bool }>
```

- **Proposal: the predicate is a sub-program**, not a restricted AST. It is a
  straight-line code range stored out of line, after the main `Exit`. It
  allows only:
  - pure ops: `Binary`, `Compare`, `Cast`, `Select`, `Unary`
  - `Load` / `LoadAddr`
  - no stores, no control flow, no blocking ops

  The engine evaluates it against a candidate value written into `arg`.
  `captures` are the registers referenced by the predicate, other than `dst`.
  They are **snapshotted at issue**, which is how CUDA evaluates the loop
  condition each spin.
- **Why not an AST.** Unit tests already use `table[seen[0]] == 1`, a load
  indexed by the candidate value. An AST would need its own load, cast and
  dtype rules, which duplicates the instruction set. A sub-program reuses the
  same handlers, so it adds no second semantics. That is plan principle 1.
- **For racecheck's declared-word bitmap** (plan §2.5): the engine runs the
  predicate over the word's write history and records the bitmap. When
  `reads_memory` is true, the history verdicts depend on *current* memory, not
  on memory at each historical write. The checker must then treat the
  earliest-accepting-write HB edge as a `review` or fall back. That is a
  W5 decision, flagged here.
- **Lowering.** The predicate expression is lowered with the `dst`
  variable/local bound to the predicate's `arg` register and with fresh
  scratch registers. After the wait, `dst` holds the accepted value. The
  legacy `build_ptx_call` callback is unnecessary: the `ld` is the
  `WaitUntil` itself.

### B.9 Sites

- Each `SiteInfo` has:
  - `kind`: the node `type_key`
  - `spans`: from `node.span`, flattening a `SequentialSpan` into a list,
    because tirx-lite helpers chain call sites
  - `op_name`
  - `text`: short, children elided, ≤ 200 chars
  - `dtype`
  - `buffer`
- **Which instructions carry a site.** Every instruction that can produce a
  finding or an event:
  - memory ops, sync ops, async issues
  - `If` (for elect regions)
  - `LoopBegin`, `Unsupported`, `Assert`

  Pure ALU instructions do not carry a site. A reported value mismatch is
  attributed to the store site.
- **Lowering order.** Sites are allocated in lowering order. Their identity is
  their index, so there is no post-order `op_id` (Q7).
- **Interning.** Snapshots compare findings by `kind + span + byte overlap`,
  never by site index. Sites are interned by `(node handle, role)`, so the
  multiple accesses one node performs share a site.

### B.10 Host ABI, inputs and topology

- **`ParamSlot { name, kind: Buffer|Pointer|Scalar|TensorMap, param_index, dtype, shape: [DimExpr], tensor_map: Option<TensorMapSpec> }`**
  - Order is param order. Bindings are by name, as legacy `bindings.py` does.
    The legacy buffers → pointers → scalars → tensor_maps ordering is a
    manifest detail and is not needed.
  - Implicit tensor maps from the host prelude become `TensorMap` slots with
    `implicit_base: slot index` and fields that are `DimExpr` over scalar
    slots.
- **Scalars.** A scalar param is read with `ReadParam { dst, slot }`. It is
  emitted once at entry and marked uniform.
- **Topology.** `Launch { clusters: DimExpr, ctas_per_cluster, threads_per_cta, warps_per_warpgroup: 4, min_blocks_per_sm, dyn_smem_bytes }`.
  - **`DimExpr`** is a tiny expression tree over scalar slots and constants:
    add, mul, floordiv, ceildiv, min, max.
  - It is evaluated at bind time. Six corpus kernels have dynamic grid
    extents.
  - Deferred extents (`extents=None`) are inferred with the closure rule
    TVM's verifier uses (44 in all).
- **Pre-entry statements** (`Bind`, `AssertStmt`, `DeclBuffer`,
  `tvm_stack_alloca`, `tensormap_encode_tiled`, and now `Evaluate`) are
  consumed by the Python prelude normalizer into host-ABI facts. They never
  become device code.

### B.11 Unsupported and fail-closed

- **Collect then reject**, as the legacy frontend does. Every unmodeled node
  kind, op, dtype, layout, attr, or loop kind is appended to
  `Program.unsupported` as `site#N kind: reason`. An `Unsupported{reason, site}`
  instruction is also emitted in place.
- **`strict=True`** (the default for `transpile`) raises
  `LoweringUnsupported` with *all* reasons. This matches today's
  `UnsupportedTIRxError`.
- **`strict=False`** keeps the program. Reaching an `Unsupported` at run time
  makes the verdict `incomplete`, with that site. This lets partially covered
  kernels run their supported paths during migration. It never produces
  `clean`.
- **Always rejected:**
  - WGMMA, fabric, multimem
  - PARALLEL / VECTORIZED / thread-bound loops
  - non-trivial layouts on direct `TensorLoad`/`BufferStore` that are not
    `ComposeLayout`
  - replicated register layouts
  - unknown `AttrStmt` keys (the allowlist is in `ir_walk._IGNORED_ATTRS`)
  - unknown `cuda.func_call`
  - dtypes outside the contract
- **Lowering exceptions.** A Python exception inside a handler is a lowering
  bug, not an unsupported construct. It propagates as an error, never as
  `incomplete`.

### B.12 Worked examples

The listings use a pseudo-bytecode: `rN` is a register and `kN` a const.
`@sN` is a site.

**(1) Vector add.** This is what the skeleton emits today, verbatim modulo
spelling. The test is `test_vector_add_code`.

```text
consts: k0=i32 128, k1=i32 1024     topology: clusters=8 ctas/cluster=1 threads/cta=128
host_abi: [a: buffer f32[1024] #0, b: buffer f32[1024] #1, c: buffer f32[1024] #2]
buffers:  [b0=a(Global,slot0), b1=b(Global,slot1), b2=c(Global,slot2)]
00  ReadSpecial r0 <- CtaLinear              ; bx   (uniform)
01  ReadSpecial r1 <- ThreadInCta            ; tx
02  Binary Mul.i32  r3 <- r0, k0
03  Binary Add.i32  r4 <- r3, r1
04  Mov             r2 <- r4                 ; i (promoted local i[1])
05  Compare Lt.i32  r5 <- r2, k1
06  If r5 else=11 end=11              @s0
07  Load  f32 Global r6 <- b0[r2]     @s1    ; a[i]
08  Load  f32 Global r7 <- b1[r2]     @s2    ; b[i]
09  Binary Add.f32  r8 <- r6, r7
10  Store f32 Global b2[r2] <- r8     @s3    ; c[i] = …
11  EndIf
12  Exit
```

**(2) TMA + mbarrier producer/consumer** (`fp16_bf16_gemm` producer warp,
abridged from the corpus IR). Buffers are renamed (`empty`, `full`, `A_s`)
and the register and buffer numbers are illustrative. The source:

```text
if T.cuda.elect_sync() != 0:
    while ld_sched_done == 0:
        T.cuda.mbarrier_wait(T.address_of(empty[stage]), phase ^ 0)
        T.ptx.cp(cvta(address_of(A_s[stage,0,0,0])), reinterpret(address_of(tmapA)), k*64, row,
                 cvta(address_of(full[stage])), "async","bulk","tensor","2d","shared::cluster","global",
                 "", "mbarrier::complete_tx::bytes", "", "cta_group::2", "", "")
        if cta_in_pair == 0:
            T.ptx.mbarrier(cvta(address_of(full[stage])), 65536, "arrive","expect_tx", …, "shared::cluster","b64","")
        stage = stage + 1
        if stage == 5: stage = 0; phase = phase ^ 1
```

```text
buffers: b5=pool(Shared,0..),  b7=empty(view of pool @ +48B, u64[5]), b8=A_s(view @ +1024B, f16[5,1,128,64], Swizzle(3,3,3))
         b9=full(view, u64[5]), slot3=tmapA(TensorMap, implicit, base=a)
20  ReadSpecial r10 <- WarpInCta                       ; warp-uniform
21  Elect          r11 <- (mask=active)                @s20   ; cuda.elect_sync()
22  Compare Ne.u32 r12 <- r11, k0
23  If r12 elect=true else=60 end=60                   @s21
24  LoopBegin end=59                                   @s22
25    Compare Eq.i32 r13 <- r14(ld_sched_done), k0
26    LoopIf r13 end=59
27    AddrOf  r15 <- b7[r16(stage)]                    ; u64 generic, buffer-relative
28    Binary Xor.i32 r17 <- r18(phase), k0
29    MbarWait addr=r15 parity=r17                     @s23   ; may Block(mbar b7[stage])
30    Binary Mul.i32 r19 <- r16, k_8192                ; A_s[stage,0,0,0] element offset
31    AddrOf  r20 <- b8[r19]
32    CvtaToShared r21 <- r20                          ; u32 shared::cta
33    AddrOf  r22 <- b9[r16]; CvtaToShared r23 <- r22
34    TmaLoad  dst_smem=r21 tmap=slot3 coords=[r24,r25] mbar=r23
              space=SharedCluster rank=2 cta_group=2 complete_tx    @s24  ; async issue, read-side/write-side milestones
35    Compare Eq.i32 r26 <- r27(cta_in_pair), k0
36    If r26 else=39 end=39                            @s25
37      MbarArrive addr=r23 space=SharedCluster expect_tx=k_65536   @s26
38    (Else elided)
39    EndIf
40    Binary Add.i32 r16 <- r16, k1                    ; stage += 1
41    Compare Eq.i32 r28 <- r16, k5
42    If r28 else=45 end=45                            @s27
43      Mov r16 <- k0
44      Binary Xor.i32 r18 <- r18, k1
45    EndIf
    …  (CLC query, mapa, remote arrive)
58    …
59  LoopEnd head=25                                    ; budget, quantum, spin-park
60  EndIf
```

The consumer side mirrors this: `MbarWait(full[stage], phase)` is followed by
`TcgenMma …` and then
`TcgenCommit mbar=empty[stage] multicast=k_mask cta_group=2`. That commit
pairs with the producer's `MbarWait(empty[stage])`.

What synccheck needs from the bytecode is already explicit:

- the resources: `b7[stage]` and `b9[stage]` as mbarrier words
- the async op (`TmaLoad`) with its completion target
- the elect region

**(3) tcgen05 MMA** (same kernel, MMA warp, inner k-step):

```python
T.cuda.tcgen05.encode_instr_descriptor(T.address_of(idesc), "float32","float16","float16",
                                      256, 256, 16, False, False, 2, False, False, False, False)
T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(adesc), T.address_of(A_s[0,0,0,0]), 0, 64, 3)
...
T.cuda.mbarrier_wait(T.address_of(full[s]), ph ^ 0)
T.ptx.tcgen05(cast(u32, acc*256), adesc + cast(u64, (s + warp%4)*128*64//8), bdesc + cast(u64, s*128*64//8),
              idesc, 0,0,0,0,0,0,0,0, cast(bool, accumulate), "mma","cta_group::2","kind::f16","", "p12")
...
T.ptx.tcgen05(cvta(address_of(empty[s])), cast(u16, 3), "commit","cta_group::2","mbarrier::arrive::one",
              "shared::cluster","multicast::cluster","b64","")
```

```text
consts: k_idesc = u32 <folded encode_instr_descriptor(f32,f16,f16,M=256,N=256,K=16,…,cta_group=2)>
70  Mov  r40(idesc) <- k_idesc                          ; folded at lowering time, site kept on the const
71  AddrOf r41 <- b8[0]
72  TcgenMatrixDesc r42(adesc) <- r41, lbo=k0, sbo=k64, swizzle=k3   @s30   ; OpLib, runtime address
    …
80  AddrOf r43 <- b9[r44(s)]
81  MbarWait addr=r43 parity=r45                        @s31
82  Binary Mul.i32 r46 <- r47(acc), k256 ; Cast u32 r48 <- r46          ; TMEM column of D
83  … adesc + (s + warp%4)*1024 → r49 (u64);  bdesc + s*1024 → r50
84  Cast bool r51 <- r52(accumulate)
85  TcgenMma kind=f16 cta_group=2 form=SS d_tmem=r48 a_desc=r49 b_desc=r50 idesc=r40
            disable_lane=[k0×8] enable_input_d=r51          @s32  ; async issue; reads smem A/B (via descs), RMW TMEM
86  … ×3 more k-steps (descriptor +2/+4/+6, enable_input_d=k1)
90  AddrOf r53 <- b7[r44]; CvtaToShared r54 <- r53
91  TcgenCommit mbar=r54 cta_group=2 multicast=k3 space=SharedCluster   @s33  ; arrives when prior MMAs complete
```

The lowering decodes `idesc` statically when it is a constant. That lets it
validate the M/N/K/cta_group legality and record the selected form in the
Instr. An `idesc` from a register (`cuda.runtime_instr_desc`, 5 kernels) is
decoded by OpLib at issue time instead. The smem footprint of A/B for
racecheck is derived at run time from the descriptors; it is not a lowering
fact.

### B.13 What the contract needs that the sketch lacks

| item | why |
| --- | --- |
| `Program.regs: Vec<RegDecl{dtype, uniform}>` | `[T;32]` registers have a static dtype; the codegen backend needs it for monomorphic handlers |
| `Program.buffers: Vec<BufferDecl{name, space, dtype, shape, strides, param_slot, base, byte_len, align, view_of}>` | buffer-relative accesses, shared pool layout and views, exact OOB, racecheck `(alloc, range)` |
| `Program.arch` | tcgen descriptor variants (sm_100/103/107) and arch-gated ops |
| `Program.requirements` / flags | `implicit_tmem`, `dynamic_tmem_lifecycle`, `readonly_proxy`, `grid_dependency`, `raw_tensor_map_registry` |
| `Program.preds: Vec<PredProgram>` | `wait_until` sub-programs (B.8) |
| `Program.ops: Vec<OpKey{op_name, mods}>` | the generic `Ptx` variant's op table, interned |
| `Operand = Reg(u32) \| Const(u32)` | avoids `Mov`-from-immediate and keeps consts foldable by codegen |
| `Scalar{dtype, bits: u64}` plus 128-bit and vector-lane dtypes | `uint128`, `float16x2`, `uint32x4`, … (Q4) |
| `SpecialReg` enum + `ReadSpecial` | lane, warp, thread, CTA, cluster ids |
| `ReadParam { dst, slot }` | scalar params |
| `If { cond, else_pc, end_pc, elect, site }`, `Else { end_pc }` | jump targets for all-false arms; elect provenance |
| `LoopBegin { end_pc, site }`, `LoopIf { cond, end_pc }`, `LoopEnd { head_pc }`, `Break`, `Continue` | the sketch names only `LoopIf`/`LoopEnd`; the frame push and the enter site need `LoopBegin`; break/continue need masks |
| `Exit` with per-lane retirement; `Assert { cond, site, msg }` | `Return`, `trap_when_assert_failed` |
| `Load/Store { dtype, space, buf, offset, site }` **and** `LoadAddr/StoreAddr { addr, space, sem, scope, cache, site }` | buffer-relative vs raw pointer (B.5) |
| `LoadRegIndexed/StoreRegIndexed` | dynamic indexing of promoted local arrays |
| `AddrOf { dst, buf, offset }` | `address_of` |
| `Unsupported { reason, site }` | fail closed at run time (`strict=False`) |
| `SiteInfo { kind, spans: Vec<Span>, op_name, text, dtype, buffer }` | `SequentialSpan` chains from tirx-lite helpers |
| `Launch` fields as `DimExpr` over scalar slots; `min_blocks_per_sm`, `dyn_smem_bytes` | dynamic grids; shared pool sizing |
| `ParamSlot { …, tensor_map: Option<TensorMapSpec>, implicit_base }` | the host-prelude tensor maps |
| A per-variant "may block" property | the scheduler retry contract; replaces the legacy `suspends` colouring |

### B.14 Open questions for the coordinator

- **Q1. Instruction granularity.** Dedicated variants for sync, async and tcgen
  families plus one generic `Ptx{op, mods}` for the ALU tail (proposed)? Or
  everything generic, with the Observer decoding modifiers? The proposal keeps
  checker code free of PTX strings.
- **Q2. Unrolling.** Keep `T.unroll` loops as loops and use indexed register
  access (proposed)? Or unroll in the lowering for simpler register
  allocation, at the cost of program size?
- **Q3. Dynamic topology and shapes.** Is `DimExpr` over scalar slots in
  `Launch` and `ParamSlot.shape` acceptable? The alternative is lowering per
  binding, i.e. caching per scalar values. That is simpler, but it recompiles
  whenever a grid-size scalar changes.
- **Q4. Wide and vector dtypes.** Should the contract `Dtype` carry
  vector/128-bit values in one register? Or must the lowering split them into
  register tuples? Splitting is natural for PTX operands, but awkward for
  `uint128` loads and `float16x8` buffer stores.
- **Q5. `elect` flag on `If`.** Is it the right carrier? Or should `Elect`
  produce a mask the engine tracks by provenance, with no static flag?
- **Q6. Tile ops.** Reuse TVM `TilePrimitiveDispatch` output (proposed), or
  port the legacy element-map approach? This decides whether about 6K lines of
  `tile_forms` analysis are rewritten at all.
- **Q7. Site identity.** Do snapshots and finding anchors need legacy `op#N`
  post-order ids? The proposal is no: anchor on spans and drop `op_id`.
- **Q8. Host prelude.** Port `host_prelude.rs` to Python in W1? The
  `Evaluate(tensormap_encode_tiled)` form it rejects today already breaks two
  canonical kernels on `main`.
- **Q9. `wait_until` predicates that read memory.** Where should these sit in
  racecheck's declared-word rule (B.8)?
- **Q10. Where `may_block` and the "progress" classification live** (needed
  for spin-parking in `LoopEnd`). Is it a property of the contract Instr enum,
  or a table in OpLib?

---

## Part C: Phase 2 status (2026-10-08)

The lowering now emits the committed `numsim_core::program` contract
(`FORMAT_VERSION` 2, strict serde). Acceptance check:
`Module::from_json` + `Program::validate()` + postcard round trip, run by
`scripts/numsim-v2/validate.sh` through `core-rs/tools/validate-program`.

### C.1 Result

| population | kernels | lower with no `Unsupported` | Rust decode + validate |
| --- | --- | --- | --- |
| corpus (canonical + wiki + corpus tests) | 195 | **195** | **195 OK** |
| full captured suite (`strict=False`) | 2343 | 1804 | 2341 OK; the other 2 load only with test-only node types |

`tests/numsim/v2/test_lowering_corpus.py` (marker `slow`) re-lowers every
`CANONICAL_KERNEL_CASES` and `WIKI_RACECHECK_SPECS` kernel strictly and
validates each module in Rust: 131 cases.

### C.2 Modules

All modules live in `numsim/v2/lowering/`; each is under 900 lines.

| module | role |
| --- | --- |
| `program_builder.py` | contract mirror: `Instr(variant, **fields)` checked against a per-variant `SCHEMA`; the tables; `Module` JSON |
| `ir_walk.py` | statements, expressions, topology, control flow, `thread_extent`, tile dispatch |
| `memory.py` | register promotion and escape analysis; shared pool and views; layouts and swizzles; lvalues |
| `host_prelude.py` | parameters, implicit shape variables, host TensorMap prelude, `DimExpr` |
| `calls.py` | CUDA helpers, pointer plumbing, `wait_until` predicate programs |
| `ptx_lower.py` | `tirx.ptx.*` table ops mapped to dedicated variants; pure tail to `Ptx` |
| `ptx_decode.py` | pure-Python decoder for the TVM PTX table |
| `builtins.py` | helper operand roles |
| `dtypes.py` | dtype helpers |

### C.3 Conventions the contract does not spell out

These need a coordinator ack.

1. **Pack and unpack ops for vector lanes.**
   - `Ptx` ops `numsim.pack` / `numsim.unpack` with mods `ty=<Elem>x<N>` split a
     vector register into lanes and back. They are used for `ld.v*`/`st.v*`,
     vector atomics and `prim.Broadcast`/`Shuffle`.
   - W4 must implement both.
2. **Value-form `Ptx` op for converters that work through pointers.**
   - `cuda.float22half2`, `float8tohalf8` and `half8tofloat8` lower to
     `LoadAddr` + `Ptx <name>.value` + `StoreAddr`.
   - W4 implements `<name>.value` as the pure conversion.
3. **`ParamSlot` for implicit shape variables.**
   - These become `Scalar` slots named `<buffer>.shape<axis>`, with
     `local_name` set to the TVM variable.
   - The slot is referenced as `Param` in the buffer slot's `shape`.
   - The binder (W8) must take its value from the bound array, because
     `ParamKind` has no shape-variable kind.
4. **`TensorMapSpec` has no `force_cu_dtype`.**
   - `11` (TFLOAT32) maps to `dtype: TF32`.
   - `13` (16U4_ALIGN8B) maps to dense `E2M1`.
   - `14` (padded) fails closed.
   - This affects 7 corpus kernels that use `13`.
5. **`redux.sync.f32 .NaN` (2 corpus kernels).**
   - Composed as `Redux` + `Unary IsNan` + `Vote Any` + `Select(canonical NaN)`.
   - `.abs` is lowered as `Unary Abs` before the `Redux`.
6. **Warp-specialized MMA (`tcgen05.mma.ws`).**
   - `zero_col_mask` is carried as the single `disable_output_lane` operand.
   - `collector::bN` is carried in `ws_b_buffer`.
7. **`tirx.cuda.func_call`.**
   - TVM's own `tvm_builtin_pointer_offset` is lowered as pointer arithmetic.
   - Reviewed pure helpers become `Ptx` ops `tirx.cuda.func_call.<name>`, with
     mods `source_sha256=<16 hex>` (whitespace-normalized source).
   - Effectful helpers fail closed: the gdn/flashkda tensormap acquire,
     release and replace helpers (unit tests only).
8. **`OpKey.mods`** is always `slot=token`.
   - PTX table ops use their table slot names.
   - CUDA helpers use `argN=<string literal>`.
9. **Tile ops (decision 6).**
   - `tirx.transform.TilePrimitiveDispatch` runs under
     `Target({"kind": "cuda", "arch": <tirx.cuda_arch>})` only when the
     function contains `TilePrimitiveCall`.
   - Its output uses `launch_thread` (`thread_extent`), which the walker now
     supports.
   - When dispatch fails, the tile op is left in place and fails closed.
10. **Topology defaults** follow legacy `topology.rs`.
    - No CTA-level extent gives one warp.
    - Warpgroup ids only give one warpgroup.
    - The grid defaults to one cluster.

### C.4 Residuals in the full suite (`strict=False`)

All residuals are in unit and integration kernels; the corpus has none.
Ranked by kernel count:

| count | residual | owner / action |
| --- | --- | --- |
| 220 (+121 / 109 follow-on loads and stores) | TMEM `DeclBuffer` views and direct `BufferLoad`/`Store` on `tmem` | W1 next: needs a TMEM access lowering (tcgen05 ld/st or a contract `Tmem` buffer form) |
| 71 | `TilePrimitiveCall` where TVM dispatch rejects the call or the function | W1: inspect the dispatch errors |
| 17 | unbound variables (`cta_reduce` / `tvm_access_ptr` result vars) | W1 |
| 14 + 11 | `cuda.warp_reduce` / `cuda.cta_reduce` (multi-instruction helpers) | W1: expand to `Shfl`/`Barrier` sequences |
| 13 + ~100 | `mbarrier.*_wait.*report*` | contract: `MbarTestWait` has no report destinations |
| 15 | `st.async.release` / `red.async.release` without mbarrier | contract: `StAsyncArgs` requires an mbarrier |
| ~30 | bulk-copy `byte_mask` / `ignore_oob` / `report`; TMA overrides | contract: `BulkCopyArgs` lacks the fields; `TmaArgs.overrides` exists but W1 has not wired it yet |
| 19 | register layouts (`laneid` / `tid_in_wg` axes) on local buffers | tile unit tests; resolved by dispatch once those kernels dispatch |
| ~15 | `tcgen05.mma` `kind::ti16`, `lut_b`; commit `sync_restrict` / `multicast_width` | contract: not in `TcMmaKind` / `TcgenCommit` |
| 9 | `ptx_legacy.ldmatrix` / `ptx_legacy.mma` | out of scope (legacy surface) |
| 6 | `For` kind VECTORIZED | fails closed by design |
| 1 | `boolx128` parameter | coordinator ruling: lower as a `u8[128]` buffer (not done yet) |

---

## Part D: Phase 3 status (2026-10-08)

The corpus still lowers clean: 195/195 strict, and 131/131 cases in
`test_lowering_corpus.py`.

Full captured suite:

- 2123/2343 kernels lower with no `Unsupported`.
- All 2341 loadable kernels decode and `validate()` in Rust against the
  current contract (batch 2 + `lut_b_addr`).

### D.1 Rules added in phase 3

**TMEM views (contract item 17).** A `DeclBuffer(scope="tmem")` becomes a
`Space::Tmem` Buf:

- `shape = [lane_span, col_span]` of the physical rectangle its `TileLayout`
  covers;
- `base` = the static `allocated_addr`;
- `Load`/`Store` offset = `lane * col_span + col`, computed from
  `Layout.apply` (axes `TLane`/`TCol`).

Fail closed:

- direct access that is not 32-bit;
- replicated (`T.R`) layouts on direct access;
- runtime `allocated_addr`.

Views without a `tcgen05.alloc` in the kernel set
`requirements.implicit_tmem`.

**Half chains (ruling D1).** f16/bf16 expression trees compute in
`Ty{F32, lanes}` and round once where the value leaves the chain: store,
explicit `Cast`, call operand, `Bind`, or another consumer.
`ir_walk.is_half_chain` / `wide` / `half_chain` implement this.

**Reductions.** `cuda.warp_reduce` and `cuda.cta_reduce` are expanded exactly
as TVM's `cpp/builtins.py` helpers:

- `log2(width)` steps of `Shfl Bfly`;
- the CTA form adds a scratch round trip with two `Barrier Sync`;
- the leader warp's reduction of `num_warps` partials uses the op's identity
  (`0` / `-inf` / `+inf`, or the integer extremes).

**Effectful CUDA helpers whose body is one reviewed PTX statement.**
`gdn_*` / `flashkda_*` tensormap replace / acquire / release lower to
`TensorMapReplace` / `Fence{Tensormap*}`. The semantics are parsed from the
helper's own asm text, not from its name.

**Other additions.**

- `tvm_access_ptr` and `T.ptx.addr` are address arithmetic.
- `%clock_hi` / `%globaltimer_hi|lo` are derived from `Clock64` /
  `GlobalTimer`.
- An `Evaluate` of a pure expression evaluates its reads.
- Out-of-range constant indices into promoted locals become
  `LoadRegIndexed` / `StoreRegIndexed`, so the engine reports the OOB access.
- By-value parameters wider than 256 bits (`boolx128`) are `u8[N]`
  Param-space Buffer slots. Using them as a value fails closed.

**Contract batch 2.**

- `MbarTestWait.report` / `report_value` (conditional parity is now
  accepted).
- `StAsync.mbar = null` for the `.release` forms.
- `BulkCopy.byte_mask` / `ignore_oob` / `report`, `Tma.report`, and
  `Tma.overrides` (`GlobalAddress`, `GlobalDim` with `elem_bits`).
- `TcgenCommit.sync_restrict` / `multicast_width`.
- `TcgenMma.lut_b` + `lut_b_addr`.
- `TensorMapSpec.force_cu_dtype` raw.
- `ParamKind::ImplicitShape`.

### D.2 The 71 tile-dispatch rejections

Each was re-run through `TilePrimitiveDispatch` and classified by the test
that produced it:

- **18 are negative tests.** These are kernels a test expects to be
  rejected, e.g. `test_typed_tma_reduce_rejects_*` and
  `*_invalid_*`.
- **53 are positive tests of legacy-only tile forms.** These are forms that
  the installed TVM's own dispatch cannot compile for the GPU:

| count | TVM's rejection | example |
| --- | --- | --- |
| 14 | invalid tcgen05 MMA shape (e.g. `kind::f16`, M128 **N8**) | `test_gemm_async_artifact.py::*two_cluster*` |
| 10 | `permute_layout` warp-xor-swizzle not well-formed / no bank-free XOR | `test_permute_layout_artifact.py` |
| 11 | dispatch variant names not registered in this TVM (`reg`, `gmem_smem`, `tma`) | `test_tile_codegen.py::test_cta_copy_uses_canonical_semantics_independent_of_dispatch` |
| 6 | TMA inner-box-bytes constraint | `test_typed_tma_reduce_accepts_*` |
| 3 | block-scale SFA K extent | `test_block_scaled_gemm_artifact.py` |
| 9 | other (reduction / fill / elementwise variants, owner-transport mismatch) | `test_tile_reduction_variants.py`, `test_tile_unary_codegen.py` |

Decision 6 defines tile semantics as TVM's dispatch output. These kernels
therefore cannot run on hardware through this TVM, and fail closed is the
correct outcome. They should move to W8's legacy-test retirement list. The
alternative is to restore the element-map path (B.7) only for forms TVM
rejects, which I do not recommend.

### D.3 Remaining residuals (full suite, kernel-reason counts)

| count | residual | class |
| --- | --- | --- |
| 105 | tile ops TVM's dispatch rejects (D.2) | out of scope (TVM-uncompilable / negative tests) |
| 29 | `tcgen05.mma kind::ti16` | contract gap (`TcMmaKind` has no `Ti16`) |
| 15 | TMA `override_global_dim_stride_*` (lower/upper stride operands) | contract gap (`TmapOverride` cannot encode the split stride) |
| 13 + 7 | direct access to replicated / non-32-bit TMEM views | fails closed by ruling |
| 12 (+11 follow-on) | TMEM view with runtime `allocated_addr` | contract gap (`BufferDecl.base` static) |
| 19 | local buffers with register layouts (`laneid` / `tid_in_wg` axes) accessed directly | tile-only forms; out of scope with D.2 |
| 5 | `tcgen05.ld .spcompress` / `.abs` / `.NaN` | contract gap |
| 5 | `cp.async.bulk` `ignore_bytes_left/right` counts | contract gap (`BulkCopyArgs.ignore_oob` is a bool) |
| 9 + 2 | `ptx_legacy.*`, `mma_store` | out of scope (legacy surface) |
| 8 | `For` kind PARALLEL / VECTORIZED / THREAD_BINDING | fails closed by design |
| 2 | `%nwarpid` | contract gap (no `SpecialReg`) |
| ~10 | single-kernel negative tests (`wait_group -1`, unreviewed helper, `boolx128` used as a value, malformed `Shuffle` / `warp_reduce`, host `Div` extent, `address_of(handle)`) | fail closed by design |

---

## Part E: Contract batch 3 and item 27 (2026-10-08)

Emitted:

- `TcMmaKind::Ti16` for `kind::ti16`.
- Split-stride TMA overrides: per-`ord` `GlobalStride` for the lower strides,
  plus one `GlobalStrideUpper`.
- `BufferDecl.base_reg` for TMEM views whose `allocated_addr` or layout
  offset is runtime. The view's `(TLane, TCol)` layout offset is folded into
  the base and subtracted from each access.
- `TcgenLd.red_abs` / `red_nan`, and `.spcompress` forms (dsts = `mdata`
  lanes then `cdata` lanes, in TVM table order).
- `BulkCopy.ignore_oob = {ignore_bytes_left, ignore_bytes_right}`.
- `SpecialReg::NWarpId`.
- `lut_b_addr` as a TMEM taddr.
- `TcgenLd` / `TcgenSt` / `TcgenCp` `row` / `col` = `Const 0`. The table
  forms already carry the full taddr.

Also added:

- address operands of register ops (`createpolicy.range`) are passed as
  values;
- `prim.Div` is accepted in extents.

### E.1 Validation tooling

`scripts/numsim-v2/validate.sh` now builds `validate-program` against a
generated contract-only shim (`make_contract_shim.py`). The shim contains
`program.rs`, `site.rs`, `numsim-types`, and the `Space` / `Domain` enums,
so concurrent engine edits cannot break validation.

### E.2 Result

| scope | lowered without `Unsupported` | Rust decode + `validate()` |
| --- | --- | --- |
| corpus | 195/195 | yes |
| full suite | 2196/2343 | 2341/2341 loadable kernels |

### E.3 Remaining residuals

These are only out-of-scope or fail-closed-by-design rows:

- tile ops that TVM's dispatch rejects (D.2);
- direct access to replicated or non-32-bit TMEM views;
- register-layout locals (tile-only forms);
- `ptx_legacy.*` / `mma_*` legacy surface;
- PARALLEL / VECTORIZED / thread-bound loops;
- about 20 negative tests that expect rejection;
- two single-kernel items:
  - a runtime TensorMap box dim (`TensorMapSpec.box_dim` is static);
  - `smem_desc_make_lo_uniform`, a warp-collective CUDA helper. It is not
    reviewed and fails closed.


---

## Part F: Residuals (2026-10-08, after the tile forms)

Live tree (W12 sweep, `scripts/numsim-v2/lower_sweep.py` over the full
capture set, `strict=False`; `program_builder.FORMAT_VERSION` 3):

| scope | result |
| --- | --- |
| captured kernels | 2343 (2341 loadable; 2 need test-only node types, as in C.1) |
| lower with no `Unsupported` | **2250** (E.2: 2196; before the tile forms: 2217) |
| lowering exceptions | 0 |
| Rust decode + `validate()` (`validate.sh`) | 2341/2341 loadable kernels |

The 91 loadable kernels that still carry an `Unsupported` are listed below.
Each has an owner ruling, and none is a finding (a kernel legacy compiled
without a ruling). The "legacy" column comes from running legacy
`numsim.transpile` on the same capture.

- **Legacy compiled, v2 fails closed by ruling (43):**
  - L1 replicated TMEM view (13; 3 of them also hit L5 in their `gemm_async`);
  - L2 `ptx_legacy.*` (9);
  - L4 tcgen05.mma shape that is invalid on hardware (14; L4's text says 13, and the 14th is `dense_fp8_gemm_async_cta1`, `kind::f8f6f4`, same rule);
  - L6 TMA innermost box under 16 B (6);
  - L7 TMA into a padded shared slice (1).
- **Both reject (48):**
  - L3 negative tile-form tests (11: 8 `UnsupportedTIRxError` and 3 `UnmodeledTIRxFormError` in legacy);
  - other negative fail-closed tests that legacy rejects too (37: 34 `UnsupportedTIRxError` and 3 `UnmodeledTIRxFormError`). Three of these are tile ops TVM *would* dispatch: `float64_directed_rounding_is_unsupported`, `tile_unary_unknown_config` and `_warp_gemm_wrong_a_fragment_layout`. v2 keeps legacy's fail-closed rule for them (`tile_checks.py`).
- **Open V2C-TF1 (legacy compiled, tile form not yet ported): none.** Of
  W1's 28 `gemm_async`/`copy_async` kernels, the hint repairs lower 8. Every
  other one is an L4/L5/L6/L7 ruling or a negative test. W12's 30 kernels
  lower except `fp8_scale_permute_tmem_roundtrip` (L1).

"Ruling" values: `Lx` = row of `numsim-behaviour-deltas.md` "Lowering
fail-closed forms"; "legacy also rejects (E)" = a negative test whose legacy
`transpile` raises E.

| capture | kernel | test (first capturing node) | v2 reason (first) | class | ruling |
| --- | --- | --- | --- | --- | --- |
| `6c673522` | `_block_scaled_fp8_dynamic_shared_stage` | `test_fp8_snapshot_gather_honors_a_dynamic_shared_stage` | tmem_replicated_view: scale_a_tmem | direct access to a replicated TMEM view | L1 |
| `ce2d8cb9` | `_block_scaled_nvfp4_gemm_cta_group2_pair23` | `test_cta_group2_uses_the_issuing_ctas_pair_2_and_3` | tmem_replicated_view: scale_a_tmem | direct access to a replicated TMEM view | L1 |
| `3f0bffe8` | `_block_scaled_runtime_instruction_descriptor` | `test_block_scaled_desc_i_rejects_static_abi_mismatch_at_runtime` | tmem_replicated_view: scale_a_tmem | direct access to a replicated TMEM view | L1 |
| `b260fd72` | `_tcgen_cp_two_pairs_in_one_cluster` | `test_tcgen_cp_cta_group2_routes_each_pair_in_four_cta_cluster` | tmem_replicated_view: scale_tmem | direct access to a replicated TMEM view | L1 |
| `08e6b09d` | `block_scaled_mxfp4_gemm` | `test_mxfp4_uses_ue8m0_scales_over_32_element_vectors` | tmem_replicated_view: scale_a_tmem | direct access to a replicated TMEM view | L1 |
| `37a1e22b` | `block_scaled_nvfp4_gemm` | `test_nvfp4_block_scaled_gemm_decodes_nibbles_and_e4m3_scales` | tmem_replicated_view: scale_a_tmem | direct access to a replicated TMEM view | L1 |
| `4776fa77` | `block_scaled_nvfp4_gemm_cta_group2_scale_rows` | `test_cta_group2_batched_gemm_accumulates_both_target_ctas` | tmem_replicated_view: scale_a_tmem | direct access to a replicated TMEM view | L1 |
| `afa1d772` | `fp8_scale_permute_tmem_roundtrip` | `test_fp8_scale_permute_and_sf_reuse_tmem_roundtrip` | tmem_replicated_view: scale_tmem | direct access to a replicated TMEM view | L1 |
| `41a767bf` | `tcgen_scale_bitcast_cta_group2` | `test_tcgen_cp_cta_group2_reads_and_writes_each_cta_scale_backing` | tmem_replicated_view: scale_tmem | direct access to a replicated TMEM view | L1 |
| `22072b4b` | `tcgen_scale_bitcast_shared_to_tmem` | `test_tcgen_cp_bitcasts_uint8_scale_payload_into_float8_tmem` | tmem_replicated_view: scale_tmem | direct access to a replicated TMEM view | L1 |
| `7c87de29` | `_block_scaled_interleaved_physical_streams` | `test_interleaved_block_scaled_calls_are_independent_of_prior_calls` | tmem_replicated_view: scale_1_a_tmem | direct access to a replicated TMEM view; its gemm_async also has SFA K extent 8 | L1 + L5 |
| `6a205528` | `_block_scaled_scale_region_min` | `test_block_scale_region_min_selects_the_physical_scale_coordinates` | tmem_replicated_view: scale_a_tmem | direct access to a replicated TMEM view; its gemm_async also has SFA K extent 8 | L1 + L5 |
| `2bfa4404` | `block_scaled_fp8_gemm_packed_scales` | `test_block_scaled_gemm_normalizes_instruction_kind_and_shape` | tmem_replicated_view: scale_a_tmem | direct access to a replicated TMEM view; its gemm_async also has SFA K extent 8 | L1 + L5 |
| `8f0d3872` | `kernel` | `test_emitter_decodes_only_table_driven_ptx_calls` | builtin tirx.ptx_legacy.mma | legacy-only builtin `ptx_legacy.*` | L2 |
| `f53174b8` | `legacy_ldmatrix_i8_transpose_fallback` | `test_legacy_ldmatrix_8bit_transpose_matches_tirx_manual_gather` | builtin tirx.ptx_legacy.ldmatrix | legacy-only builtin `ptx_legacy.*` | L2 |
| `6c149258` | `legacy_ldmatrix_i8_transpose_two_warps` | `test_legacy_ldmatrix_8bit_transpose_uses_full_thread_index_across_warps` | builtin tirx.ptx_legacy.ldmatrix | legacy-only builtin `ptx_legacy.*` | L2 |
| `6cebb99a` | `legacy_ldmatrix_i8_x4` | `test_legacy_ldmatrix_8bit_nontranspose_keeps_b16_fragment_abi` | builtin tirx.ptx_legacy.ldmatrix | legacy-only builtin `ptx_legacy.*` | L2 |
| `ddf2b60a` | `legacy_ldmatrix_x1_domain` | `test_legacy_ldmatrix_x1_domain_matches_independent_fragment_mapping` | builtin tirx.ptx_legacy.ldmatrix | legacy-only builtin `ptx_legacy.*` | L2 |
| `89f28510` | `legacy_ldmatrix_x2_trans` | `test_legacy_ldmatrix_transpose_preserves_b16_fragment_abi` | builtin tirx.ptx_legacy.ldmatrix | legacy-only builtin `ptx_legacy.*` | L2 |
| `c0d006ee` | `ptx_mma_legacy_f16_m16n8k16` | `test_ptx_mma_legacy_executes_actual_pointer_offset_abi` | builtin tirx.ptx_legacy.mma | legacy-only builtin `ptx_legacy.*` | L2 |
| `c9ff71fc` | `ptx_mma_legacy_s8_u8_m16n8k32` | `test_legacy_m16n8k32_int8_reuses_dense_form_and_engine` | builtin tirx.ptx_legacy.mma | legacy-only builtin `ptx_legacy.*` | L2 |
| `8849573f` | `reused_legacy_mma_pointer_bindings` | `test_reused_legacy_mma_reuses_a_b_and_accumulator_bindings_across_calls` | builtin tirx.ptx_legacy.mma | legacy-only builtin `ptx_legacy.*` | L2 |
| `0a97c1c1` | `dense_gemm_async_tf32_is_rejected` | `test_tile_tf32_gemm_async_fails_closed` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | negative tile-form test; TVM dispatch and legacy both reject | L3 (legacy UnmodeledTIRxFormError) |
| `5feb0e62` | `dense_gemm_async_wrong_tmem_a_layout` | `test_dense_gemm_async_rejects_wrong_tmem_a_layout` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | negative tile-form test; TVM dispatch and legacy both reject | L3 (legacy UnmodeledTIRxFormError) |
| `3cc1c18e` | `typed_tma_dtype_roundtrip_bool` | `test_typed_tma_rejects_unmodeled_production_dtypes` | tile op tirx.tile.copy_async: v2 tile form 'copy_async' not ported yet (TVM rejects) | negative tile-form test; TVM dispatch and legacy both reject | L3 (legacy UnmodeledTIRxFormError) |
| `39c10257` | `_block_scaled_invalid_scale_layout` | `test_block_scale_layout_that_violates_instruction_row_stride_fails_closed` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | negative tile-form test; TVM dispatch and legacy both reject | L3 (legacy UnsupportedTIRxError) |
| `d1cd6f35` | `_tcgen_cp_wrong_declared_shape` | `test_tcgen_cp_rejects_declared_shape_that_disagrees_with_layout` | tile op tirx.tile.copy_async: v2 tile form 'copy_async' not ported yet (TVM rejects) | negative tile-form test; TVM dispatch and legacy both reject | L3 (legacy UnsupportedTIRxError) |
| `f3ad2430` | `_tcgen_cp_wrong_destination_lane_permutation` | `test_tcgen_cp_rejects_destination_lane_permutation` | tile op tirx.tile.copy_async: v2 tile form 'copy_async' not ported yet (TVM rejects) | negative tile-form test; TVM dispatch and legacy both reject | L3 (legacy UnsupportedTIRxError) |
| `bb2f011e` | `_tcgen_ldst_wrong_m64_tmem_layout` | `test_tcgen_ldst_rejects_tmem_layout_outside_fixed_instruction_abi` | tile op tirx.tile.copy_async: v2 tile form 'copy_async' not ported yet (TVM rejects) | negative tile-form test; TVM dispatch and legacy both reject | L3 (legacy UnsupportedTIRxError) |
| `f0800bfb` | `dense_gemm_async_declared_geometry_mismatch` | `test_dense_gemm_async_rejects_declared_geometry_that_disagrees_with_operands` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | negative tile-form test; TVM dispatch and legacy both reject | L3 (legacy UnsupportedTIRxError) |
| `083d8740` | `kernel` | `test_typed_tma_reduce_rejects_invalid_operation_dtype_pairs[unknown-uint32-unsup` | tile op tirx.tile.copy_async: v2 tile form 'copy_async' not ported yet (TVM rejects) | negative tile-form test; TVM dispatch and legacy both reject | L3 (legacy UnsupportedTIRxError) |
| `3a730827` | `tma_reduce_wrong_direction` | `test_typed_tma_reduce_rejects_non_store_direction` | tile op tirx.tile.copy_async: v2 tile form 'copy_async' not ported yet (TVM rejects) | negative tile-form test; TVM dispatch and legacy both reject | L3 (legacy UnsupportedTIRxError) |
| `a82a5aca` | `tma_unknown_cache_hint` | `test_typed_tma_rejects_unknown_cache_hint` | tile op tirx.tile.copy_async: v2 tile form 'copy_async' not ported yet (TVM rejects) | negative tile-form test; TVM dispatch and legacy both reject | L3 (legacy UnsupportedTIRxError) |
| `354eea5a` | `commit_forwards_only_issued_work` | `test_commit_republishes_causal_predecessors_of_local_work` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | tcgen05.mma shape invalid on hardware (TVM dispatch) | L4 |
| `016eda97` | `dense_fp8_gemm_async_cta1` | `test_dense_fp8_gemm_async_matches_numpy` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | tcgen05.mma shape invalid on hardware (TVM dispatch) | L4 |
| `e4d8a4a4` | `dense_gemm_async_cta1` | `test_dense_gemm_async_gathers_physical_operands_and_accumulates_tmem` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | tcgen05.mma shape invalid on hardware (TVM dispatch) | L4 |
| `ade818c4` | `dense_gemm_async_cta_group2` | `test_cta_group2_gathers_both_shared_shards_and_scatters_tmem` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | tcgen05.mma shape invalid on hardware (TVM dispatch) | L4 |
| `ae6de81e` | `dense_gemm_async_dynamic_right_index` | `test_dense_gemm_async_handles_repeated_dynamic_index_loads` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | tcgen05.mma shape invalid on hardware (TVM dispatch) | L4 |
| `f5034afe` | `dense_gemm_async_m64_cta2_layout_b` | `test_cta_group2_m64_tcgen_mma_uses_layout_b_independently_of_declared_layout` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | tcgen05.mma shape invalid on hardware (TVM dispatch) | L4 |
| `5501e406` | `dense_gemm_async_no_swizzle_shared` | `test_dense_gemm_async_no_swizzle_descriptor_matches_numpy` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | tcgen05.mma shape invalid on hardware (TVM dispatch) | L4 |
| `b0a8bc90` | `dense_gemm_async_tmem_a_cta_group2` | `test_cta_group2_gathers_both_tmem_a_shards` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | tcgen05.mma shape invalid on hardware (TVM dispatch) | L4 |
| `067520aa` | `dense_gemm_async_two_clusters` | `test_dense_gemm_async_runs_numpy_backend_on_two_cluster_workers` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | tcgen05.mma shape invalid on hardware (TVM dispatch) | L4 |
| `5aed5aa2` | `dense_gemm_async_two_cta_pairs_in_one_cluster` | `test_cta_group2_routes_each_pair_within_a_four_cta_cluster` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | tcgen05.mma shape invalid on hardware (TVM dispatch) | L4 |
| `8d2db317` | `dense_gemm_async_two_thread_issuers` | `test_thread_scope_gemm_async_requires_one_runtime_issuer` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | tcgen05.mma shape invalid on hardware (TVM dispatch) | L4 |
| `6e8be779` | `inactive_gemm_async_is_noop` | `test_all_inactive_gemm_async_is_a_noop` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | tcgen05.mma shape invalid on hardware (TVM dispatch) | L4 |
| `7e3d1cc8` | `repeated_tcgen_fence_handoff` | `test_repeated_thread_fence_handoff_keeps_full_frontier_ordered` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | tcgen05.mma shape invalid on hardware (TVM dispatch) | L4 |
| `d024eb22` | `tcgen_cp_to_mma_handoff` | `test_a_declared_wait_merges_its_tcgen_frontier_without_deadlocking` | tile op tirx.tile.gemm_async: v2 tile form 'gemm' not ported yet (TVM rejects) | tcgen05.mma shape invalid on hardware (TVM dispatch) | L4 |
| `0e9b7884` | `kernel` | `test_typed_tma_reduce_accepts_every_ptx_operation_dtype_pair[max-bfloat16]` | tile op tirx.tile.copy_async: v2 tile form 'copy_async' not ported yet (TVM rejects) | TMA innermost box under 16 B | L6 |
| `14f16144` | `kernel` | `test_typed_tma_reduce_accepts_every_ptx_operation_dtype_pair[max-float16]` | tile op tirx.tile.copy_async: v2 tile form 'copy_async' not ported yet (TVM rejects) | TMA innermost box under 16 B | L6 |
| `4d6f137d` | `kernel` | `test_typed_tma_reduce_accepts_every_ptx_operation_dtype_pair[min-float16]` | tile op tirx.tile.copy_async: v2 tile form 'copy_async' not ported yet (TVM rejects) | TMA innermost box under 16 B | L6 |
| `77e30a85` | `kernel` | `test_typed_tma_reduce_accepts_every_ptx_operation_dtype_pair[add-bfloat16]` | tile op tirx.tile.copy_async: v2 tile form 'copy_async' not ported yet (TVM rejects) | TMA innermost box under 16 B | L6 |
| `810f07d0` | `kernel` | `test_typed_tma_reduce_accepts_every_ptx_operation_dtype_pair[min-bfloat16]` | tile op tirx.tile.copy_async: v2 tile form 'copy_async' not ported yet (TVM rejects) | TMA innermost box under 16 B | L6 |
| `e43b7b8c` | `kernel` | `test_typed_tma_reduce_accepts_every_ptx_operation_dtype_pair[add-float16]` | tile op tirx.tile.copy_async: v2 tile form 'copy_async' not ported yet (TVM rejects) | TMA innermost box under 16 B | L6 |
| `8960cb4a` | `tma_padded_narrow_rows` | `test_checkers_reject_a_misaligned_tma_shared_component[racecheck]` | tile op tirx.tile.copy_async: v2 tile form 'copy_async' not ported yet (TVM rejects) | TMA into a padded shared slice | L7 |
| `c41fc9ba` | `float64_directed_rounding_is_unsupported` | `test_float64_directed_rounding_fails_closed` | tile op tirx.tile.add: TilePrimitiveCall(add): directed float64 rounding is not implemented | negative test; legacy fail-closed rule kept in v2 `tile_checks` (TVM would dispatch) | legacy also rejects (UnmodeledTIRxFormError) |
| `c24dc32f` | `fp8_identity_reinterpret_roundtrip` | `test_scalar_fp8_identity_reinterpret_rejects_256_payload_roundtrip[float8_e8m0fn` | tirx.reinterpret: raw payload reinterpret is not modeled for scalar low-precision/storage-only dtype | negative test of a fail-closed form | legacy also rejects (UnmodeledTIRxFormError) |
| `ec7bbd85` | `fp8_identity_reinterpret_roundtrip` | `test_scalar_fp8_identity_reinterpret_rejects_256_payload_roundtrip[float8_e4m3fn` | tirx.reinterpret: raw payload reinterpret is not modeled for scalar low-precision/storage-only dtype | negative test of a fail-closed form | legacy also rejects (UnmodeledTIRxFormError) |
| `1fa183ad` | `?` | `test_cuda_atomic_cas_classifier_rejects_non128_or_non_byte_addressable_vectors[b` | dtype 'boolx128' has no numsim_core::Ty | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `38b6aa34` | `?` | `test_ordinary_handle_address_is_not_classified_as_a_tensor_map` | address of variable descriptor_like_name | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `43d04f19` | `?` | `test_async_group_wait_count_rejects_negative_values[cp.async.wait_group]` | tirx.ptx.cp_async_wait_group: wait_group count -1 out of range | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `52434463` | `?` | `test_pure_call_form_strings_must_be_compile_time_static[form1]` | warp_reduce op None | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `602cd118` | `?` | `test_resolver_remains_closed_for_unknown_calls` | builtin tirx.tvm_stack_alloca | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `79c96872` | `?` | `test_resolver_rejects_opaque_cuda_helpers_at_the_single_boundary` | cuda.func_call of unreviewed or effectful helper 'arbitrary_helper' | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `98c392b4` | `?` | `test_async_group_wait_count_rejects_negative_values[cp.async.bulk.wait_group.rea` | tirx.ptx.cp_async_bulk_wait_group: wait_group count -1 out of range | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `e6baf144` | `?` | `test_pure_call_form_strings_must_be_compile_time_static[form2]` | mov_sreg register name must be a literal | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `e7a93836` | `?` | `test_async_group_wait_count_rejects_runtime_values` | tirx.ptx.cp_async_wait_group: group must be a constant | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `f7ee9326` | `?` | `test_pure_call_form_strings_must_be_compile_time_static[form0]` | cta_reduce op None | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `cd42c619` | `_warp_gemm_wrong_a_fragment_layout` | `test_warp_gemm_rejects_layout_that_disagrees_with_instruction_abi` | tile op tirx.tile.gemm: TilePrimitiveCall(gemm): no mma.sync.m16n8k{16,8} instruction matches M=16,  | negative test; legacy fail-closed rule kept in v2 `tile_checks` (TVM would dispatch) | legacy also rejects (UnsupportedTIRxError) |
| `47e2290c` | `flashkda_rsqrtf_wrong_dtype` | `test_flashkda_math_helper_dtype_mismatch_fails_closed` | tirx.cuda.func_call helper 'flashkda_rsqrtf' requires ['float32'] -> float32 | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `e07d5f51` | `host_encoded_unregistered_integer_tensor_map` | `test_host_tensor_map_integer_expressions_fail_closed_on_unregistered_nodes` | unsupported integer operation BitwiseAnd in a host extent expression (no DimExpr) | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `a94a1ef2` | `invalid` | `test_bulk_wait_group_rejects_runtime_count_during_numsim_transpilation` | tirx.ptx.cp_async_bulk_wait_group: group must be a constant | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `6adc14ce` | `invalid_vector_shuffle_extract` | `test_vector_shuffle_classifier_and_frontend_reject_out_of_range_extract` | Shuffle index outside the concatenated lanes | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `0f67083b` | `kernel` | `test_typed_tma_reduce_rejects_invalid_operation_dtype_pairs[min-float32-invalid ` | cp.reduce.async.bulk.tensor operation .min is invalid for TensorMap dtype F32 | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `8f859059` | `kernel` | `test_typed_tma_reduce_rejects_invalid_operation_dtype_pairs[inc-int32-invalid fo` | cp.reduce.async.bulk.tensor operation .inc is invalid for TensorMap dtype S32 | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `90d84e56` | `kernel` | `test_typed_tma_reduce_rejects_invalid_operation_dtype_pairs[add-float64-invalid ` | cp.reduce.async.bulk.tensor operation .add is invalid for TensorMap dtype F64 | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `aa0c2443` | `kernel` | `test_registry_rejects_wrong_arity_or_dtype[handle-tirx.reinterpret-arguments3]` | tirx.reinterpret: source and result must have identical bit widths | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `af4ab65a` | `kernel` | `test_registry_rejects_wrong_arity_or_dtype[uint64-tirx.reinterpret-arguments2]` | tirx.reinterpret: source and result must have identical bit widths | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `e303f0f8` | `kernel` | `test_typed_tma_reduce_rejects_invalid_operation_dtype_pairs[add-int64-invalid fo` | cp.reduce.async.bulk.tensor operation .add is invalid for TensorMap dtype S64 | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `ddb6a2f4` | `modified_combine_int_frac_ex2` | `test_modified_known_helper_remains_fail_closed` | tirx.cuda.func_call helper 'combine_int_frac_ex2' body does not match the validated bit-composition  | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `a163865f` | `modified_flashkda_fmaf_rn` | `test_flashkda_math_helper_semantic_mutations_fail_closed[modified_flashkda_fmaf_` | tirx.cuda.func_call helper 'flashkda_fmaf_rn' body does not match the validated fused round-to-neare | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `f372c489` | `modified_flashkda_rsqrtf` | `test_flashkda_math_helper_semantic_mutations_fail_closed[modified_flashkda_rsqrt` | tirx.cuda.func_call helper 'flashkda_rsqrtf' body does not match the validated float32 reciprocal-sq | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `1a4e7bcc` | `modified_flashkda_tanh_approx` | `test_flashkda_math_helper_semantic_mutations_fail_closed[modified_flashkda_tanh_` | tirx.cuda.func_call helper 'flashkda_tanh_approx' body does not match the validated tanh.approx.f32  | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `17f33096` | `modified_fma_scale_sub_f32x2` | `test_modified_packed_fma_helper_remains_fail_closed` | tirx.cuda.func_call helper 'tvm_builtin_fma_scale_sub_f32x2' body does not match the validated packe | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `e3591b3c` | `modified_gdn_lg2_approx_ftz` | `test_modified_gdn_lg2_helper_remains_fail_closed` | tirx.cuda.func_call helper 'gdn_lg2_approx_ftz' body does not match the validated lg2.approx.ftz.f32 | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `41bb86ad` | `opaque_statement_helper` | `test_opaque_or_spoofed_cuda_helpers_are_rejected[opaque_statement_helper]` | cuda.func_call of unreviewed or effectful helper 'tvm_builtin_tcgen05_mma_mxf4_block32_ss' | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `ffd3c5a0` | `opaque_value_helper` | `test_opaque_or_spoofed_cuda_helpers_are_rejected[opaque_value_helper]` | cuda.func_call of unreviewed or effectful helper 'opaque_fdividef' | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `369bfe5c` | `parallel_loop` | `test_non_serial_loop_semantics_are_not_silently_sequentialized[parallel_loop-PAR` | for-loop kind 1 | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `208ad05e` | `thread_bound_loop` | `test_non_serial_loop_semantics_are_not_silently_sequentialized[thread_bound_loop` | for-loop kind 4 | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `3bb8d714` | `tile_unary_unknown_config` | `test_unary_tile_ops_fail_closed_on_unknown_config` | tile op tirx.tile.exp: TilePrimitiveCall(exp): unsupported config keys ['undocumented_mode'] | negative test; legacy fail-closed rule kept in v2 `tile_checks` (TVM would dispatch) | legacy also rejects (UnsupportedTIRxError) |
| `e55f4a71` | `unknown_attr` | `test_unknown_attr_semantics_fail_closed` | attribute 'numsim.unknown_control' | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
| `fdb571e7` | `unsupported_local_layout` | `test_frontend_finding_points_at_the_offending_node[racecheck-rebuilt-root]` | host statements after tirx.device_entry | negative test of a fail-closed form | legacy also rejects (UnsupportedTIRxError) |
