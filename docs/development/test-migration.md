---
orphan: true
---

# Legacy test retirement (completed)

The legacy Python test suite that exercised the deleted NumSim engine was
retired in favour of the layers in `numsim-redesign.md` §3. The retirement
finished with the step-5 deletion commit **79f04eb** ("Step 5: delete the
legacy NumSim engine; v2 is the implementation and the conformance oracle").
The migration tooling and its ledger (`scripts/numsim-v2/coverage/`, the
classifier and the retire scripts) were deleted in that commit;
`scripts/numsim-v2/RETIREMENT.md` records what each tool was and what stayed.
The full plan as executed, with per-phase counts, the port maps and the
blocker history, is this file at the parent commit
(`git show 7c7d049:docs/development/test-migration.md`).

## Completed (79f04eb)

The commit changed 651 files (+2,029/−312,589).

| what | step 5 |
| --- | --- |
| legacy engine code | 240 tracked files deleted: `engine-rs/`, `frontend-rs/`, `numsim/transpiler/`, nine legacy `numsim/*.py` modules, the `thirdparty/tvm-rust-ext` submodule (and `.gitmodules`) |
| kept from the legacy layer | `numsim/{errors,cases,dtype_abi,report}.py` and `dtype_registry.json`; the output comparison moved to `numsim/v2/_compare.py` |
| legacy test files removed | 140 files (700 test functions), plus 4 support modules no surviving test imported |
| legacy test functions cut from surviving files | 467 functions in 94 files |
| test modules kept with every test cut | 3 (`numsim/runtime/test_mbarrier_multicast.py`, `test_ptx_float_register_ops.py`, `test_sm107_register_predicates.py`), because surviving tests import their helpers |
| unreachable legacy helpers pruned | 22 (including the five legacy views of `tests/numsim/support/manifest.py`) |
| conformance snapshots | 83 files in 30 cases changed: 17 `<mode>.delta.json` folded into their base snapshots (rows numsim H5, racecheck B1, B7, R3, R4, T18, T19, X4, sync S1) and deleted; the rest regenerated from v2 under the `relax_unanchored` projection change (anchors filled in, unanchored legacy findings split per anchor; no verdict, finding kind or byte coverage changed) |
| new oracles | 5 snapshots where legacy had none (it rejected the case with `UnsupportedTIRxError`) and v2 runs clean: `fp16_bf16_gemm/{numsim,synccheck}`, `mla_dsv4_multishape/{numsim,racecheck,synccheck}` |
| conformance after regeneration | 304 passed |

### Waves as executed

All six waves ran together in the one `--apply`. Counts are per wave in
isolation (test functions retired); waves overlap, so they sum to more than
the 1,167 functions (700 removed + 467 cut) retired in total.

| wave | what it retired | functions |
| --- | --- | ---: |
| 0 | E: kernels with a tile op TVM's own dispatch rejects | 55 |
| 1 | B rows covered or ported by Rust contract-event tests (`numsim-core/tests/{racecheck,synccheck}_*`) | 227 |
| 2 | B rows replaced by v2 kernel-level checker tests (`tests/numsim/v2/checkers/`) | 52 |
| 3 | A corpus/wiki verdict gates covered by `tests/conformance` snapshots | 44 |
| 4 | A tests with a passing v2 copy in `tests/numsim/v2/ports/` (315 public-API flips ran unchanged under v2 and survive in place) | 414 |
| 5b | C: tests that pinned the legacy implementation (generated Rust text, poll counts, payload shapes, legacy internals) | 401 |

Before deletion the dry run reported zero blockers: no A test blocked by v2,
no replacement held by a `v2_gap` mark, and no surviving module reaching a
deleted name (directly, through a removed test module, or through a support
helper's lazy import).

### What remains permanently

- **Two strict xfails (racecheck delta T18, documented limitation):**
  `tests/numsim/v2/ports/test_p7_racecheck_t18.py::test_declared_word_shape_read_once_no_retry`
  and `::test_reading_the_word_once_is_not_made_correct_by_declaring_it`. A raw
  read of a declared word before its publication is indistinguishable from a
  spin's first iteration without loop information, so v2 reports clean.
- **GPU-only tests:** the functions marked `numsim_gpu` (13 files under
  `tests/numsim/microtests/`, 4 under `tests/numsim/runtime/`, and the GPU
  halves of `tests/numsim/v2/ports/test_p8_tcgen_descriptor_dispatch.py` and
  `test_w9_tcgen_inactive_boundaries.py`) skip without a CUDA device; they run
  on a GPU host.
- **`NUMSIM_V2_*` environment aliases:** `numsim/v2/options.py` reads
  `NUMSIM_<name>` and still accepts `NUMSIM_V2_<name>` for one release.
- **The snapshot rule:** every later commit that changes a conformance
  snapshot cites a file-qualified delta row, or carries
  `Snapshot-Regen: schema <reason>` (CI: `scripts/numsim-v2/check_snapshot_deltas.py`).

## Where tests live

| layer | location | holds |
| --- | --- | --- |
| Rust scenarios | `tirx_harness/src/tirx_harness/numsim/core-rs/numsim-core/tests/` | hand-built `Program`s and contract events, including the Racecheck/Synccheck legacy ports |
| lowering | `tests/numsim/v2/test_lowering_*.py` | `Program` contents |
| v2 kernel checks | `tests/numsim/v2/checkers/` | Racecheck/Synccheck facts contract events cannot express |
| v2 copies | `tests/numsim/v2/ports/` | public-API legacy tests ported to the v2 API; each docstring cites its legacy test and delta row |
| tile forms | `tests/numsim/v2/tile_forms/` | tile-op kernels TVM's dispatch rejects |
| surviving public-API tests | `tests/numsim/{runtime,integration,microtests,corpus}/`, `tests/analysis_tools/` | legacy-era tests that ran unchanged on v2 |
| corpus | `tests/conformance/` | canonical cases in three modes against snapshots |
| performance | `tests/perf/` | relative baselines |

## Categories

| cat | meaning | fate |
| --- | --- | --- |
| A | Semantic kernel-level test: runs a kernel and checks outputs, verdict, error, or finding kinds. | `conformance`: corpus or wiki kernels, already snapshotted. `v2-kernel-case`: small kernel, rerun through the v2 API (see the "A" section). `v2-api`: `test_api.py`; v2 keeps the legacy signatures. |
| B | Checker-semantics unit test with a small kernel. | Rust scenario test from contract events (`numsim-core/tests/{racecheck,synccheck}_*.rs`). Status is `covered`, `ported`, `v2_kernel_test` (a pytest in `tests/numsim/v2/checkers`), or `needs_kernel`. |
| C | Pins the implementation: generated Rust text, `v2::` paths, poll or transition counts, payload field shapes, report text, legacy Python internals (transpiler, registry, ABI, compile cache, source map), Rust build workspace. | Delete together with the code it pins. |
| D | GPU-paired numerical op test (`tests/numsim/microtests`). | `D-recorded`: the cvt goldens. Already replayed bit-exactly by `numsim-oplib`; retire the Python harness. `D-live`: paired against a live GPU run. Record the outputs as goldens first. |
| E | The kernel contains a tile op that TVM's own `TilePrimitiveDispatch` rejects (lowering-inventory §D.2). | Delete. |
| F | Infrastructure, CLI, packaging, `dump_kernel`, TVM-script parser checks, corpus fixtures and their independent references. | Keep. |
| N | Already a new-layer test (`tests/conformance`, `tests/numsim/v2`, `tests/perf`). | Keep. |

The task brief said "private `_run_*` APIs → C". The classifier applies a
narrower reading. Many semantic tests reach a kernel through
`tirx_harness.numsim.checkers._run_racecheck` or `_run_synccheck`, or through
module-local `_run_*` helpers. Using such an API only as the way to run the
kernel does not make a test C; the test is classified by what it asserts. A
test is C only when it asserts about the private API itself.

## Counts (2026-10-08, phase 5, 2a5895a)

| directory | A | B | C | D | E | F | N | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (top) |  |  |  |  |  | 13 |  | 13 |
| analysis_tools |  |  | 9 |  |  |  |  | 9 |
| analysis_tools/racecheck | 6 | 171 | 32 |  |  |  |  | 209 |
| analysis_tools/shared | 11 | 3 | 11 |  |  |  |  | 25 |
| analysis_tools/synccheck | 8 | 69 | 45 |  |  |  |  | 122 |
| conformance |  |  |  |  |  |  | 2 | 2 |
| numsim/abi |  |  | 3 |  |  |  |  | 3 |
| numsim/corpus | 29 |  |  |  |  | 23 |  | 52 |
| numsim/integration | 322 | 3 | 140 |  | 42 | 3 |  | 510 |
| numsim/microtests |  |  | 1 | 55 |  |  |  | 56 |
| numsim/registry | 17 |  | 81 |  | 1 |  |  | 99 |
| numsim/runtime | 605 | 33 | 67 |  | 18 | 5 |  | 728 |
| numsim/v2 |  |  |  |  |  |  | 214 | 214 |
| **total** | **998** | **279** | **389** | **55** | **61** | **44** | **216** | **2042** |

| A target | public facade only | imports legacy internals |
| --- | ---: | ---: |
| conformance | 19 | 22 |
| v2-api | 0 | 23 |
| v2-kernel-case | 433 | 501 |

| category | target / status | tests |
| --- | --- | ---: |
| A | conformance | 41 |
| A | v2-api | 23 |
| A | v2-kernel-case | 934 |
| B | covered | 88 |
| B | ported | 139 |
| B | v2_kernel_test | 52 |
| C | delete | 389 |
| D | D-live | 48 |
| D | D-recorded | 7 |
| E | delete | 61 |
| F | keep | 44 |
| N | keep | 216 |

**E (tile forms TVM rejects).** The list was regenerated from scratch rather
than copied from W1:

1. A capture run of `tests/numsim` and `tests/analysis_tools`
   (`capture_plugin.py`, `-m 'not numsim_gpu'`) saved 2,475 kernels.
2. `tile_rejections.py` (deleted at step 5) lowered each one.

71 kernels contain a rejected tile op, which matches W1's §D.2 count. Seven
functions hit a rejected form in only some of their parametrizations: for
example `test_typed_tma_reduce_accepts_every_ptx_operation_dtype_pair` hits it
in 6 of 38 kernels. Those seven stay in their own category, with a
`[partial E: k/n]` note in the CSV; drop only the affected params by hand. They map
to 89 parametrized node ids in 69 test functions. W1's "105" in §D.3 counts
kernel × reason pairs, not tests.

Rerun at dea8c9d (lowering stable): all 2,376 loadable kernels lower without
an exception (2,231 clean, 145 with `Unsupported`). The rejected set is
unchanged: 71 kernels, the same 69 functions. The first run's 347 tcgen
kernels that raised mid-batch-3 contain no rejected tile op.

Seven E rows are in `analysis_tools/racecheck`, and the reviewed racecheck map
takes precedence for them. Their semantics, for example the tcgen
thread-fence hand-off, are covered by Rust scenarios built from contract
events, so the tile form is irrelevant to the contract.

**D (goldens).** All 133 GPU-recorded cvt forms
(`ptx_cvt_{scalar,fp8,narrow}_goldens.py`) have a generated test in
`numsim-oplib/src/cvt/goldens/tests.rs`. `--check-goldens` verifies this, and
the generator's `SKIPPED` table is empty. Every other microtest (`D-live`)
compares against a GPU run at test time and has no recorded data yet.

## Provenance of the replacement layers

- **Rust contract-event ports.** `numsim-core/tests/racecheck_legacy_ports.rs`
  (86 tests) and `synccheck_legacy_ports.rs` (60 tests); every test is
  doc-commented with the legacy `file.py::test[param]` it ports. Together with
  existing scenario tests they covered 227 hand-read B rows (88 already
  covered, 139 ported). Where a legacy expectation contradicted a delta row,
  the port asserts the new behaviour and cites the row.
- **v2 kernel-level checker tests.** `tests/numsim/v2/checkers/` holds one
  function per legacy B test that contract events cannot express (51 rows),
  using only the public `tirx_harness.numsim.v2` API and the
  `requires_v2_engine` probe (`_runnable.py`).
- **v2 copies.** `tests/numsim/v2/ports/` holds the A tests whose legacy form
  pinned an implementation detail or a legacy behaviour a delta row changed.

## Open: semantics in legacy tests that no new spec mentions

No test is xfailed on these items any more (the `no_spec` marks were all
resolved or ruled before step 5); the unresolved ones remain spec questions.
Each item below was confirmed by grep across `racecheck-semantics.md`,
`racecheck-isa-answers.md`, `sync-semantics.md`, `sync-isa-answers.md`,
`synccheck-explorer.md` and the three delta docs. Each needs a spec sentence
or a delta row, whichever way it is ruled.

1. *(Resolved: racecheck-behaviour-deltas P7 and T8 rule `alias_stale_read`; identity rule W5-9)*
2. *(Resolved: host-input aliasing is implemented in the v2 binder, CONTRACT_REQUESTS W8-6, dev-loop.md "v2 binder rules")*
3. *(Resolved: racecheck-behaviour-deltas X10)*
4. **How a bulk reduction (`cp.reduce.async.bulk`) is emitted.** It is atomic
   per PTX element. A matching-width peer `red` is clean; a `.u32` peer racing
   on f16 elements races. Nothing specifies that the producer emits an
   atomic `Rmw`, relaxed `.gpu`, `returns_value = false`, one span per
   element. A wrong emission gives a false race or a missed one.
5. *(Resolved: the core now reports a wide release write to a declared word as incomplete, as the spec says; the port is un-ignored.)*
6. *(Resolved: racecheck-behaviour-deltas W6: a declared word may live in any allocation)*
7. **A declared `wait_until` as a thread sync for the tcgen05 fence pair.**
   It counts as an execution-ordering thread sync for the tcgen05
   before/after fence pair. The spec states this only for relaxed mbarrier
   arrive/wait (semantics table row 10).
8. **Fully predicated-off accesses** emit no access. No active-lane rule
   exists anywhere.
9. **Footprint narrowing.** `.ignore_oob` read clipping, `.cp_mask` byte
   selection and gather4 rows are not covered; only multicast is.
10. *(Resolved: the `cta_sum` scratch and barrier accesses are emitted; `permute_layout` zero-fill is numsim-behaviour-deltas F1. `tests/numsim/v2/checkers/test_tile_hidden_accesses.py` passes)*
11. **The default `.cta` scope for a qualifier-less raw arrive on a remote
    mbarrier.** B1 states the default but not that it flips a real kernel's
    verdict.
12. *(Resolved by 1159d17: the allocation-result check is restored.)*
13. *(Resolved by 1159d17: host `Configure` is now the initial state.)*
14. **Kind names.** (Partly resolved: v2 emits `warp_collective_divergence`
    again, and racecheck-behaviour-deltas T6 lists the racecheck kinds.)
    Several kinds were renamed with no delta row:
    - `synchronization_contract_mismatch` (an execution_error) is now
      `named_barrier_contract_mismatch`;
    - `warp_collective_divergence` is now `divergence`;
    - `mbarrier_local_arrive_remote_address` is now `bad_address`;
    - the synccheck `oob` is now `out_of_bounds`.

    `setmaxnreg_pool_deadlock` is restored (1159d17).
15. **Liveness at a later named barrier** of lanes that broke out of a loop
    but have not exited. B1 and B6 define "non-exited", but not this
    divergence and reconvergence case.
16. **TMEM access after an unordered cross-warp dealloc** (legacy
    `synchronization_collective_publication`). It is in no sync doc; it may
    belong to racecheck.
17. **`warp_collective_divergence` for partial `shfl.sync` participation.**
    This includes width-16 `warp_sum` with a full warp, and boolean or
    data-guarded participation. numsim-behaviour-deltas P5 covers only a
    non-participant source lane.
18. **Docs disagree on an uncommitted bulk TMA store at exit.**
    sync-semantics says it is implicitly committed; racecheck P6 says
    `AsyncNeverCompleted`.
19. **Two synccheck assumptions documented only in lowering-inventory.md:**
    `griddepcontrol.wait` is assumed satisfied, and an exhausted poll budget is
    `incomplete resource_limit{polls}`, not a deadlock.
20. **The readonly-bytes overlap check** (a write overlapping bytes read by
    `ld.global.nc` / `proxy::readonly`) is assigned to no checker.
21. **The proxy of `st.async` / `red.async`.** Legacy treats them as generic;
    the interpreter emits the async proxy.
22. *(Resolved: racecheck-behaviour-deltas T2 and T13)*
23. *(Resolved: racecheck-behaviour-deltas T3, restored as `SignalProtocolError`)*
24. *(Resolved: racecheck-behaviour-deltas T1)*