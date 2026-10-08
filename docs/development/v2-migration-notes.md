---
orphan: true
---

# NumSim v2: migration notes for kernel authors and tool users

NumSim, Racecheck and Synccheck now run on the v2 engine. The public calls
are the same as before (`numsim.transpile`, `Engine().run`, `racecheck()`,
`synccheck()`, `report.verdict` / `.findings` / `.format()` /
`.require_clean()`), and so is the payload `schema_version` 5. These notes
list what you will see differently, why it changed, and what to do about it.
Each bullet names the behaviour-delta row that rules it:

- `N:` = [numsim-behaviour-deltas.md](numsim-behaviour-deltas.md)
- `R:` = [racecheck-behaviour-deltas.md](racecheck-behaviour-deltas.md)
- `S:` = [sync-behaviour-deltas.md](sync-behaviour-deltas.md)

Each bullet reads: change: before → now. Why.

## 1. What changed

### Running NumSim

- **Pointer bits** (N:H1): the address you read back with `reinterpret("uint64", buf.ptr_to(...))` → was the host numpy pointer → is now a synthetic device address with a 4 KiB-aligned base. Why: device addresses are not host addresses.
- **Input arrays** (N:H2): legacy copied results back into your numpy inputs → inputs are never mutated; read `result.outputs[name]`. Why: one input dict can feed several runs.
- **Spin-loop interleaving** (N:H3): a fixed quantum gave fixed spin counts → spin counts follow the seeded scheduler, and `poll_order` is gone. Why: scheduling order is not a kernel property.
- **Loop iteration budget** (N:H4): exceeding it was an `error` → it is now `incomplete` (reason `Budget: loop exceeded its iteration budget of N`). Why: a budget limits coverage; it proves nothing about the kernel.
- **Uninitialized reads** (N:H5, N:F1): reported where the value was later used → reported where the bytes are read from memory. `permute_layout` padding that was never written still reads as 0, but is now also reported as `uninitialized_read`/`review`. Why: shared memory and TMEM are not initialized at launch.
- **Tile ops** (N:F2, N:F4): v2 runs the code TVM's tile dispatch emits.
  - A `dispatch=` hint never changes semantics. A hint TVM does not register is dropped.
  - Forms TVM cannot lower run the legacy form: snapshot copies, `permute_layout`, and shared-memory reductions.
  - Register copies that TVM would lower with its single-thread fallback also run the legacy form.
  - Expect ≤ a few ulp differences where TVM's arithmetic differs, for example the pairwise f32x2 `sum` and rounding toward zero (`rz`) f32 `add` (N:R1, N:R9). Why: the dispatched code is what runs on the GPU.
- **Partial-warp tile ops** (N:F3): `warp`/`warpgroup` tile ops still require all 32 lanes; only the message changed (`warp_collective_divergence`). Why: PTX `.sync` membermask rule.
- **Numerics** (N:D8–D11):
  - NaN payloads of host-computed `fma` and `+ - * /` are pinned (D8);
  - `log1p`/`sigmoid` accept f16, bf16 and f64 (D9);
  - `erf`, `exp10`, `log10` and `nearbyint` are new (D10);
  - integer-vector `&`, `|`, `^`, `<<` and `>>` run lane-wise, while vector `~` is rejected (D11).
  - Why: these follow TVM's CUDA codegen, or they need a pinned reference until there is a GPU golden.

### Reading racecheck findings

- **Qualifier-less cluster arrive** (R:B7): `mbarrier.arrive.shared::cluster` without a scope, on a peer CTA's barrier → was clean → is now a `ScopeMismatch` error. Write `.release.cluster`. Why: the ISA default scope is `.cta`.
- **Strong generic load vs a strong store** (R:R3): → not a race. An undeclared protocol word gets an `UndeclaredProtocolWord` review instead. Why: both accesses are morally strong (§8.7.1).
- **Scope-only failures** (R:R4): an unordered strong pair that fails only on scope → is now `data_race` with `missing_release_acquire`. Why: PTX §8.10.3.
- **Raw read of a declared `wait_until` word** (R:T18): → `review` (`declared_word_raw_read`), not a race.
- **`elect.sync` is not a memory fence** (R:T19): lanes that only meet the elected lane at `elect.sync` are not ordered with it. Expect true-positive `data_race` findings there.
- **Cross-CTA async-proxy writes** ordered only by base causality (R:X4): were clean → now a `CrossCtaAsyncOrder` review. Why: the ISA is silent here.
- **Thread-scope tile copies to shared memory** (R:T21): all lanes writing the same bytes in one instruction is no longer a race (R:V11). A warpgroup copy from an unlayouted local can now race, because TVM's code has every thread store the whole tile.

### Reading synccheck findings

- **TMEM allocation** (S:T2, S:T8): an `.exclusive` `tcgen05.alloc` blocks until no other allocation is live. Any allocation while an exclusive one is live is an error (`tcgen_alloc_while_exclusive`). Why: §9.7.18.7.1.
- **`pending_count` on a token not from `.noComplete`** (S:M14): still an error; the message is now `pending_count: NotNoComplete`.
- **Divergent blocking waits and partial `__syncwarp`** (S:M15, S:B8): were errors → are now `incomplete` with reason `divergent_block`. Why: under independent thread scheduling the kernel can be valid, and the engine cannot prove otherwise.

### Binding inputs

- **Overlapping host arrays**: they are one device allocation, so writes through one name are seen through the other, and races between them are reported.
- **Raw pointer words**: to pass a buffer's address as data, take it from `Engine().address_of(module, inputs, name)`. Do not use `ndarray.ctypes.data` (N:H1).
- **Duplicate public names**: two parameters of one kernel with the same public name are rejected (`InputError: ambiguous host binding`).
- **Packed sub-byte dtypes** (fp4/fp6/int4) must be bound as packed `uint8`.
- **Wrong-shape scalars**: binding an array to a scalar parameter raises `InputError` ("scalar argument '<name>' has a buffer value").
- **`run_case`**: the reference function cannot change what the kernel sees. Inputs are frozen and restored around it, and its keys must name selected outputs.

### Selecting launches

- **Phase-indexed subsets** (N:H6): `subset={phase: ExecutionSubset}` on a multi-kernel module must select the same clusters for every launch. Otherwise it raises `InputError` ("per-phase subsets must be equal"). A bare subset on a multi-kernel `Engine.run` is rejected ("not broadcast"). Why: one engine run serves every launch.
- **`cta_ids`**: allowed, but must name whole clusters. Combined with `cluster_ids`, the run uses their intersection.
- **Partial runs**: a subset run reports `subset_execution`, and its checker verdict is at least `incomplete` for the unexecuted part.

## 2. Environment and engine switches

| Variable / argument | Meaning | Default |
| --- | --- | --- |
| `NUMSIM_CACHE_DIR` | Cache root: lowered modules in `v2-modules/`. Delete the directory to force re-lowering. | `~/.cache/tirx-harness/numsim` |
| `NUMSIM_V2_SEED` / `Engine(seed=...)` | Scheduler seed. Results are reproducible for a fixed module, inputs and seed. | `0` |
| `NUMSIM_V2_NO_CACHE` | `1` disables the module cache. | unset |
| `Engine(max_workers=...)` | Scheduler threads (`"auto"` = CPU count). Results do not depend on it. | `8` |
| `Engine(native_loop_iteration_budget=..., native_loop_reschedule_quantum=...)` | Loop budget and slice quantum: positive integers or `None`. Invalid values raise at construction. | engine default |

- **Rename:** the `NUMSIM_V2_` prefix becomes `NUMSIM_` when the legacy engine is deleted. Until then only the `NUMSIM_V2_*` spellings are read; no alias exists yet.
- **`NUMSIM_WORKER_AFFINITY`** is read only by the legacy engine, and v2 ignores it.
- **`NUMSIM_IMPL=v2`** is a test-suite switch (`tests/conftest.py`) that points the public names at v2. Library users call `tirx_harness.numsim` directly.

## 3. Reading the new report fields

`report.format()` and `ExecutionError` text share one renderer. Only these
fragments are stable; see
[architecture.md "Rendered report text"](architecture.md):

- `<checker> <VERDICT> - <n> finding(s)`, then one `[<STATUS>] <kind>: <message>` line per finding, with `at <file>:<line>`.
- `Prior:` / `Current:`: the earlier and the later side of a race. Each gives the warp, the source op with its TVMScript text and location, the lane, the access kind, the memory space, and the allocation byte range.
- `Stop:`: why the engine stopped. Either `warp W, lane(s) L, <op>, operands A, B`, or a budget (`loop budget N at iteration I`, `round budget N`).
- `Access pair:` / `Reason:` / `Cause:` / `Hint:`: how the finding was classified, and what to change.
- **Structured fields.** Prefer them over text: finding `kind`, `details` (`attrs`, evidence bytes), and the stop diagnostic's `error`, `protocol`, `lanes`, `faulting_lanes` and `operands`. `to_dict()` separates errors (`findings`), `review` items (`advisories`), `incomplete` reasons and `execution_error`.
- **`incomplete` reasons** are listed in [engine-review.md "Engine incomplete reasons"](engine-review.md). The common ones are `Budget: loop exceeded …`, `divergent_block: …`, `Unsupported: not modeled: …`, `round budget of N exhausted` and `subset_execution`. `incomplete` means coverage could not be established: it is never `clean` and never an `error`.

## 4. When a kernel is rejected at transpile

`UnsupportedTIRxError` lists every reason in `error.unsupported`. Each reason
reads `site#N <node>: <reason>`. The checkers return an `incomplete` report
(`native_frontend_unsupported`) instead of raising. Find your message
fragment below:

| Message fragment | Meaning | What to do |
| --- | --- | --- |
| `tile op tirx.tile.<op>: … (TVM: …)` | Neither TVM's dispatch nor a v2 tile form lowers this call (N:L3). | Read TVM's reason. Pick a dispatch TVM registers, or fix the operands (scopes, layouts, shapes). |
| `Invalid matrix shape for Tcgen05 MMA` | MMA shape is outside the PTX table (N:L4). | Use a legal M/N for the kind and `cta_group`. |
| `SFA K extent=8 must be in {16, 1, 4}` | Block-scaled scale layout (N:L5). | Use a legal scale-vector extent. |
| `innermost box is 8B` / TMA shared-chain stride | TMA box below 16 bytes (N:L6), or a padded shared slice (N:L7). | Widen the box; use a dense shared tile. |
| `StructuralEqual check failed … TileLayout` (gemm_async) | Accumulator layout contradicts the MMA output layout (N:L8). | Declare the instruction's TMEM layout. |
| `tmem_replicated_view: <buffer>` | Direct load or store on a replicated TMEM view (N:L1). | Access TMEM through `tcgen05.ld`/`st` or a non-replicated view. |
| `builtin tirx.ptx_legacy.<op>` | Deprecated spelling (N:L2). | Use the `T.ptx.*` table form (`mma.sync`, `ldmatrix`). |
| `kind::f8f6f4 cta_group::2 requires tirx.cuda_arch in [...]` | The descriptor semantics differ by architecture. | Set `tirx.cuda_arch` (for example `sm_100a`). |
| `cuda.ldg has no __ldg overload for '<dtype>'` / `pointer to … does not match the loaded dtype` | The CUDA `__ldg` overload set. | Load a supported dtype through a pointer of that dtype. |
| `bitwise_not on vector operand` | CUDA vector types have no `~` (N:D11). | Apply `~` per lane. |
| `for-loop kind VECTORIZED` / `PARALLEL` / `THREAD_BINDING`, `for-loop annotations […]` | Loop semantics that are not modeled. | Use serial or unrolled loops, or `launch_thread`. |
| `topology: …` | Contradictory launch extents. | Make the CTA, warp, warpgroup and cluster extents agree. |
| `cuda.func_call of unreviewed or effectful helper '<name>'` / `body does not match the validated …` | Only reviewed helper bodies are modeled. | Use the reviewed helper unchanged, or the equivalent `T.ptx` op. |
| `unsupported config keys [...]` | A tile-op config that legacy also rejected. | Remove the key. |
| `statement kind not lowered` / `builtin <name>` | Not modeled yet. | Report it with the kernel; this is a fail-closed gap, not a verdict on the kernel. |

A rejection is never a statement that your kernel is wrong. It means v2 will
not simulate a form it cannot model exactly.
