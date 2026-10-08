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

**One executor.** v2 had an optional `codegen` backend (`NUMSIM_V2_BACKEND`,
`Engine(backend=...)`) → it is deleted; the interpreter is the only executor,
and every instruction's semantics live in one handler. Why: codegen was never
faster (median 0.87–0.94x of interpreter core time, plus ~117 ms per call and
long cold builds), while the interpreter is 2.3–3.0x faster than legacy
([backend-comparison.md](backend-comparison.md) "Summary").

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
- **Shared-memory window** (N:F5): a window beyond the 18-bit (`sm_100`/`sm_103`) or 19-bit (`sm_107a`) tcgen05 descriptor range was rejected at transpile, even without a descriptor over it → only a descriptor that addresses past its range stops the run (`invalid_operand`), and on `sm_100*`/`sm_103*` a CTA needing more than 232448 bytes of shared memory is rejected at transpile ("above the 232448-byte per-CTA capacity"). A `decl_buffer` view past its pool no longer grows the window. Why: the bit limit belongs to the descriptor's address field, and a CTA above capacity cannot launch.
- **Numerics** (N:D8–D13):
  - NaN payloads of host-computed `fma` and `+ - * /` are pinned (D8);
  - `log1p`/`sigmoid` accept f16, bf16 and f64 (D9);
  - `erf`, `exp10`, `log10` and `nearbyint` are new (D10);
  - integer-vector `&`, `|`, `^`, `<<` and `>>` run lane-wise, while vector `~` is rejected (D11);
  - f64 `min`/`max` of two NaNs returned the second operand → returns the canonical NaN `0x7fffffffffffffff`, as f32 does (D12);
  - f32/f64 `atom.add`/`red.add` with both the old value and the operand NaN propagated the operand's NaN → propagates the old value's NaN (PTX order `old + b`, D8 rule) (D13).
  - Why: these follow TVM's CUDA codegen, or they need a pinned reference until there is a GPU golden.

### Reading racecheck findings

- **Qualifier-less cluster arrive** (R:B7): `mbarrier.arrive.shared::cluster` without a scope, on a peer CTA's barrier → was clean → is now a `ScopeMismatch` error. Write `.release.cluster`. Why: the ISA default scope is `.cta`.
- **Strong generic load vs a strong store** (R:R3): → not a race. An undeclared protocol word gets an `UndeclaredProtocolWord` review instead. Why: both accesses are morally strong (§8.7.1).
- **Scope-only failures** (R:R4): an unordered strong pair that fails only on scope → is now `data_race` with `missing_release_acquire`. Why: PTX §8.10.3.
- **Raw read of a declared `wait_until` word** (R:T18): → `review` (`declared_word_raw_read`), not a race.
- **`elect.sync` is not a memory fence** (R:T19): lanes that only meet the elected lane at `elect.sync` are not ordered with it. Expect true-positive `data_race` findings there.
- **Cross-CTA async-proxy writes** ordered only by base causality (R:X4): were clean → now a `CrossCtaAsyncOrder` review. Why: the ISA is silent here.
- **Thread-scope tile copies to shared memory** (R:T21): all lanes writing the same bytes in one instruction is no longer a race (R:V11). A warpgroup copy from an unlayouted local can now race, because TVM's code has every thread store the whole tile.
- **Evidence `epoch` of async-op accesses** (R:T22): was the internal stamp epoch → is the milestone, 1 = read side, 2 = write side. Why: the old value moved with internal slot reuse, and reports must depend only on module, inputs, config and seed.
- **`alias_stale_read` names** (R:P7, superseded by W5-15): the logical name came from the op's first pointer → each access is named by its own operand's buffer. A same-dtype view (including a strided alias or two `decl_buffer`s over one pool word) shares its root's name, so legacy's `review` there is now clean; only a dtype-changing view is a new name. Why: one name per operand, no borrowed names.
- **Racecheck wall time**: more `max_workers` speed up only the engine; the checker consumes events on one thread, so a racecheck run is engine time plus serial checker time ([racecheck-semantics.md](racecheck-semantics.md) "Merge design and the serial-checker limit").

### Reading synccheck findings

- **TMEM allocation** (S:T2, S:T8): an `.exclusive` `tcgen05.alloc` blocks until no other allocation is live. Any allocation while an exclusive one is live is an error (`tcgen_alloc_while_exclusive`). Why: §9.7.18.7.1.
- **`pending_count` on a token not from `.noComplete`** (S:M14): still an error; the message is now `pending_count: NotNoComplete`.
- **Divergent blocking waits and partial `__syncwarp`** (S:M15, S:B8): were errors → are now `incomplete` with reason `divergent_block`. Why: under independent thread scheduling the kernel can be valid, and the engine cannot prove otherwise.
- **Partial-lane waits nothing can complete** (S:M17): a TMA that lands fewer bytes than its `expect_tx`, or a `wait_until` on a word nobody writes, reached by only some lanes → was a `deadlock` error → is `incomplete` (`divergent_block`) in all three tools; it is never reported clean. Why: the same structured-SIMT limit as M15/B8; the other lanes run only after the branch.
- **TMEM access outside every live allocation** (S:T9): was a synccheck-only `synchronization_collective_publication` finding → is a `bad_address` error in every mode ("not in a live tcgen05 allocation"). Cross-warp dealloc itself stays legal. Why: §9.7.18.7.1, `taddr` must point into a live allocation.
- **Restricted `tcgen05.commit`** (`.sync_restrict::shared::read::mma::a`, S:T10): was treated as a full commit → orders only operand-A shared reads, after earlier restricted commits and before later unrestricted ones. Why: §9.7.18.12.1.
- **Multicast mask naming a rank outside the cluster** (S:M16): same error, now `bad_address` ("multicast CTA mask … names ranks outside the N-CTA cluster"), raised before any target is touched. Why: a mask bit with no CTA fails closed.

### Binding inputs

- **Overlapping host arrays**: they are one device allocation, so writes through one name are seen through the other, and races between them are reported.
- **Raw pointer words**: to pass a buffer's address as data, take it from `Engine().address_of(module, inputs, name)`. Do not use `ndarray.ctypes.data` (N:H1).
- **One parameter under several names** (N:H7): `k1:output` and its alias `output` in one dict was rejected ("provided through multiple aliases") → the same object, or arrays with identical bytes, is one binding; different values raise `InputError` ("bound more than once with different values"). Why: the binding is unambiguous, and one dict can serve several modules.
- **Integer scalar width and signedness** (N:H8): e.g. `np.uint32(7)` for an `int32` parameter was rejected ("requires dtype int32, got uint32") → any integer value that fits the parameter's dtype binds; an out-of-range value raises `InputError` ("outside <dtype> range"), and a float for an integer parameter raises "requires dtype …". Why: Python ints and `np.int64` arithmetic results must bind when the value is representable.
- **Duplicate public names**: two parameters of one kernel with the same public name are rejected (`InputError: ambiguous host binding`).
- **Packed sub-byte dtypes** (fp4/fp6/int4) must be bound as packed `uint8`.
- **Wrong-shape scalars**: binding an array to a scalar parameter raises `InputError` ("scalar argument '<name>' has a buffer value").
- **`run_case`**: the reference function cannot change what the kernel sees. Inputs are frozen and restored around it, and its keys must name selected outputs.

### Selecting launches

- **Phase-indexed subsets** (N:H6): `subset={phase: ExecutionSubset}` on a multi-kernel module must select the same clusters for every launch. Otherwise it raises `InputError` ("per-phase subsets must be equal"). A bare subset on a multi-kernel `Engine.run` is rejected ("not broadcast"). Why: one engine run serves every launch.
- **`ExecutionSubset(cta_ids=...)`**: the same selector as legacy, now `tirx_harness.numsim.v2.ExecutionSubset`: flattened CTA ids that must form whole clusters of a static grid (`InputError` otherwise); with `cluster_ids` too, the run uses their intersection ([dev-loop.md](dev-loop.md) "v2 binder rules"). Why: the engine schedules whole clusters.
- **Partial runs**: a subset run records the `incomplete` reason `subset_execution` (`analysis_scope` kind `subset`), so its verdict is never `clean` ([engine-review.md](engine-review.md) "Engine `incomplete` reasons"). Why: unexecuted clusters are unchecked.

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
| `non-writable physical pointer` / `cannot add write access` / `outside tvm_access_ptr range` | A load or store breaks its `tvm_access_ptr` mask or extent (N:L9). Legacy raised the same fragments at run time; v2 rejects at transpile, anchored at the access. | Fix the access mask or extent; the GPU would not check it, so the IR is malformed. |
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

## 5. For contributors: building the extension

- **Private builds only** ([dev-loop.md](dev-loop.md), `core-rs/numsim-py/build_dev.sh`): rebuilding the shared `v2/numsim_core_py.abi3.so` while other runs have it loaded crashed them (`Bus error`) → build into a private target and package, `CARGO_TARGET_DIR=<dir>/target bash core-rs/numsim-py/build_dev.sh --out <dir>/ext`, and run pytest with `-o "pythonpath=<dir>/ext/pkg ."`. Why: concurrent builds must share neither the cargo target dir nor the installed `.so`.
