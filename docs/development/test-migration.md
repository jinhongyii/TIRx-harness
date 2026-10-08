---
orphan: true
---

# Legacy test retirement plan

This plan covers how the legacy Python test suite under `tirx_harness/tests/` is
retired in favour of the layers in `numsim-redesign.md` §3. It is driven by
data: every test function has one row in
`scripts/numsim-v2/coverage/test_classification.csv`, which
`scripts/numsim-v2/classify_tests.py` writes. Regenerate the CSV and the
summary below with:

```bash
source scripts/dev-env.sh
$PY scripts/numsim-v2/classify_tests.py --check-goldens --summary /tmp/summary.md
```

The classifier is a rule engine. Each CSV row records the rule that decided it
(`rule` column), so a disputed row can be traced to its rule and overridden in
`OVERRIDES`. Rules apply in this order:

1. the reviewed coverage maps `coverage/{other_b,racecheck,synccheck}.tsv`,
   written by reading every B test by hand;
2. manual overrides;
3. the tile-dispatch rejection list `coverage/tile_dispatch_rejections.txt`;
4. path rules;
5. AST signals: the calls a test makes (resolved transitively through
   module helpers, fixtures and `tests/**/support`), its assert text, the
   implementation identifiers it touches, and its name.

Outside the reviewed maps the categories are heuristic.

**Spot checks.** Two random samples of 50 heuristic rows were each read by
hand.

- **Sample 1:** 5 rows were wrong.
  - Four were A rows whose real contract is legacy plumbing: native-call
    counting, classifier internals, a legacy Engine budget keyword, and
    Rust-text inspection.
  - One was C instead of A: a cache-hint no-op test caught by the
    `cache` name rule.
- **Sample 2:** after fixing the first sample's rules, 5 of a fresh sample
  were wrong.
  - Three were A instead of C: `ExecutionSubset` plumbing (2) and a
    backing-size helper.
  - One was A instead of C: a registry classifier.
  - One was C instead of A: a runtime ABI rejection caught by the `_abi_`
    name rule.

Rules were added for each. Expect about **10% error** on heuristic rows. The
dominant error is A versus C at the boundary where a test runs a kernel only
to inspect legacy internals.

Two further rows had the right category but missed a stats pin
(`poll_order`, `worker_count`). Their semantic part is still portable.

Review each row when its file is converted; the counts are good enough to
plan from.

## Categories

| cat | meaning | fate |
| --- | --- | --- |
| A | Semantic kernel-level test: runs a kernel and checks outputs, verdict, error, or finding kinds. | `conformance`: corpus or wiki kernels, already snapshotted. `v2-kernel-case`: small kernel, rerun through the v2 API (see the "A" section). `v2-api`: `test_api.py`; v2 keeps the legacy signatures. |
| B | Checker-semantics unit test with a small kernel. | Rust scenario test from contract events (`numsim-core/tests/{racecheck,synccheck}_*.rs`). Status is `covered`, `ported`, `v2_kernel_test` (a pytest in `tests/numsim/v2/checkers`), or `needs_kernel`. |
| C | Pins the implementation: generated Rust text, `v2::` paths, poll or transition counts, payload field shapes, report text, legacy Python internals (transpiler, registry, ABI, compile cache, source map), Rust build workspace. | Delete together with the code it pins. |
| D | GPU-paired numerical op test (`tests/numsim/microtests`). | `D-recorded`: the cvt goldens. Already replayed bit-exactly by `numsim-oplib`; retire the Python harness. `D-live`: paired against a live GPU run. Record the outputs as goldens first. |
| E | The kernel contains a tile op that TVM's own `TilePrimitiveDispatch` rejects (lowering-inventory §D.2). | Delete. |
| F | Infrastructure, CLI, packaging, `dump_kernel`, TVM-script parser checks, corpus fixtures and their independent references. | Keep. |
| N | Already a new-layer test (`tests/conformance`, `tests/numsim/v2`). | Keep. |

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
2. `tile_rejections.py` lowered each one.

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

## B: checker semantics vs. the Rust scenario tests

Every B row has been read and mapped by hand, in three maps under
`scripts/numsim-v2/coverage/`:

- `racecheck.tsv`: all 209 tests in `analysis_tools/racecheck`;
- `synccheck.tsv`: all 122 tests in `analysis_tools/synccheck`;
- `other_b.tsv`: the 50 checker-verdict tests elsewhere in the tree. Most are
  in `numsim/runtime`; a few are synccheck assertions in racecheck files and
  in `analysis_tools/shared`.

Each map has one row per legacy function: its status, the tests that cover
it, and a note. A parametrized test counts as covered only when every
semantically distinct parameter is covered; otherwise the note lists the
uncovered parameters. `other_b.tsv` takes precedence over the other two maps.

| B status | rows | meaning |
| --- | ---: | --- |
| covered | 88 | An existing Rust scenario test reproduces the kernel and asserts the same verdict or kind. |
| ported | 139 | A new test in `racecheck_legacy_ports.rs` or `synccheck_legacy_ports.rs`. |
| v2_kernel_test | 51 | Contract events cannot express it. Replaced by a v2 kernel-level pytest in `tests/numsim/v2/checkers/`: 43 from the first two maps, plus 8 from `other_b` (`test_needs_kernel_batch.py`, phase 3). |
| needs_kernel | 1 | `test_reported_layout_regressions.py::test_sync_qualified_warp_operations_reject_single_lane_participation`. It surfaced in phase 3 when partial-E functions left category E. |

The classifier refines the maps' "not B" rows:

- legacy-executor and frontend plumbing becomes C;
- op-support runs become A.

### Rust ports from contract events

Both files are in `numsim-core/tests/`. Every test is doc-commented with the
legacy `file.py::test[param]` it ports. With these files,
`cargo test -p numsim-core --test 'racecheck_*' --test 'synccheck_*'` is
fully green in the main tree. `synccheck_engine` also passes again.

- `racecheck_legacy_ports.rs`: 86 tests in sections g1 to g6, all passing.
  W5 ruled the five ignored ports in a3df3a2 and un-ignored them. The map
  notes now say "un-ignored".
- `synccheck_legacy_ports.rs`: 60 tests, all passing.

Results at HEAD (9484204): `racecheck_*` and `synccheck_*` are all green,
86 and 60 tests respectively. I ran them on a `git archive` export, because
the main tree's library briefly did not compile under another worker's
in-flight edit.

When a legacy expectation contradicts a delta row, the test asserts the new
behaviour and cites the row. Rows cited:

- **Racecheck:** R1, R2/R3, R4, P1, P5, P6, B1/V5, B2, X1, X2/S2, X3, X9, W2,
  I2, V3.
- **Sync:** B2, B3, A3, M2, M5, M6.

### v2 kernel-level tests for what contract events cannot express

`tests/numsim/v2/checkers/` holds one pytest function per legacy test (43),
grouped by theme:

| file | legacy source | n |
| --- | --- | ---: |
| `test_exact_oob.py` | `racecheck/test_native_exact_oob.py` (7), `test_native_racecheck_exact_control.py` (3), `…artifact.py::…ignores_fully_predicated_shared_pointer_load` | 11 |
| `test_global_write_seed.py` | `racecheck/test_native_global_write_seed.py` | 7 |
| `test_alias_advisory.py` | `racecheck/test_native_alias_advisory.py` | 7 |
| `test_tile_hidden_accesses.py` | `racecheck/test_native_racecheck_artifact.py` (`cta_sum`, `permute_layout`, tile-region index) | 6 |
| `test_warp_collectives.py` | `synccheck/test_native_kernel_contracts.py`, `test_native_synccheck_exact_control.py` | 4 |
| `test_remote_mbarrier_address.py` | `synccheck/test_native_synccheck_artifact.py` | 2 |
| `test_divergence_liveness.py` | `synccheck/test_native_break_continue.py`, `…cross_warp_tmem_quiescence_before_dealloc` | 2 |
| `test_engine_level.py` | subset run, `.ignore_oob` clipping, conditional TMEM lifecycles, deadlock with no publisher | 4 |

The tests use only the public `tirx_harness.numsim.v2` API.

- **Skip marker.** Each test carries
  `requires_v2_engine` (`_runnable.py`). That marker is a cached end-to-end
  probe: transpile a vector add, run it, and require clean racecheck and
  synccheck verdicts. `NUMSIM_V2_FORCE_CHECKERS=1` overrides the probe.
- **Status today.** The v2 engine already passes the probe, so the tests run
  for real. The result is 29 passed, 26 xfailed and 23 xpassed (78 items).
- **xfail markers.** Each is `strict=False`:
  - `no_spec(item)`: 24 functions whose semantics no spec covers. Each
    cites its item in the list at the end of this document.
  - `v2_gap`: 6 functions where the port is right and v2 is wrong today.

`v2_gap` failures as of 2026-10-08, for W8:

- Three TMEM view tests: NumSim reads back 0.
- `alias_stale_read` fires on a `view`/`rearrange` of one shared buffer. This
  is a regression from the site→buffer-name wiring.
- `warp-conditional-alloc`: a lane-order race caused by the missing
  `WarpSync` on `tcgen05.alloc`.
- A wait with no possible publisher is reported as incomplete
  `divergent_block`, not `deadlock`. In the synccheck case the payload
  verdict is "error" although the only entry is incomplete.
- TensorMap with distinct inputs:
  - initial base: `bad_address`;
  - replaced base: `missing_proxy_bridge` despite the fence pair.

Host aliasing raises `NotImplementedError` (W8-6), so all aliased parameters
are xfail. Kernels fed raw host addresses (`ctypes.data`) cannot be ported
until v2 exposes a binding's engine address.

### Remaining gaps

- **Undocumented divergences:** none left. The 4 first-batch divergences
  were fixed by 1159d17. The 5 later racecheck divergences were ruled in
  a3df3a2: completed tcgen05 work joins HB, async release heads,
  `SignalProtocolError`, verdict-edge precedence, and `WaitExitUnproven`.
- **`needs_kernel`:** 1 row is left (see the table above). The 8 from
  `other_b` were written in phase 3: 19 items pass and 17 are xfail. The
  xfails are `no_spec`: item 11 (default `.cta` scope on remote
  `arrive.expect_tx` → `scope_mismatch`), item 20 (readonly overlap),
  restricted `tcgen05.commit` treated as a full commit, and an
  out-of-cluster ctaMask bit dropped silently.
- **v2 xfails** (`tests/numsim/v2/checkers`: 54 passed, 35 xfailed, 25 xpassed): resolved by
  rulings on the `no_spec` items, or by W8 for the `v2_gap` ones.

**Retirement readiness of B.**

- 227 rows can be deleted once their Rust tests are accepted (wave 1).
- 51 rows have a v2 replacement that runs today (wave 2, once green without
  xfail).
- 1 row still needs a replacement.
- No Rust test is waiting on a ruling.

**Found in passing, outside the checkers.**

- **Missing `WarpSync` on `tcgen05.alloc`.** racecheck-semantics §3 row 3
  says `tcgen05.alloc.sync.aligned` emits `WarpSync{mask}`.
  `interp/handlers/tcgen.rs` emits only a lane-0 write plus a Protocol event.
  `g4_tcgen_alloc_result_visible_to_every_lane` pins both sides, and the v2
  `warp-conditional-alloc` test now shows the race end to end.
- **Remote cluster arrive flips verdict (B1/V5).** A qualifier-less raw
  `mbarrier.arrive.shared::cluster` on a peer CTA's barrier
  (`support/remote_mbarrier.py`), and the remote `arrive.expect_tx` in
  `red_async`, are clean in legacy. The new core, using the default `.cta`
  scope, reports `ScopeMismatch`. The result depends on which scopes
  lowering assigns.
- **Kind renames with no delta row:**
  - `warp_collective_divergence` → `divergence`;
  - `mbarrier_local_arrive_remote_address` → `bad_address` (sync-semantics
    still names `MbarrierLocalArriveRemoteAddress`);
  - synccheck `oob` → `out_of_bounds` (P5 covers racecheck only);
  - `synchronization_contract_mismatch` → `named_barrier_contract_mismatch`.
- **Proxy of `st.async`/`red.async`.** Legacy treats these as generic; the
  interpreter emits `Proxy::Async`. The `red_async` kernel has no
  `fence.proxy.async`, so it would race end to end.

## A: how the small-kernel tests move

A is the bulk of the suite. Nearly all of it is `numsim/runtime` and
`numsim/integration` tests that build a TIRx kernel inline, run it, and compare
outputs or the error.

1. **Public-surface files** (column `surface = public`): the module reaches
   NumSim only through `tirx_harness.numsim` (`transpile`, `Engine`,
   `compare`, `run_case`). `v2.api` mirrors these names, so these files need no
   rewrite. Run them under `NUMSIM_IMPL=v2` through the same facade switch
   that `tests/conformance` uses, and they become the v2 kernel tests at
   step 5. Gate: a per-file pass/fail diff between legacy and v2.
2. **Internal-surface files** (`surface = internal:<module>`) import the
   transpiler, bindings, ABI or checker internals, usually for one helper:
   `analyze` and `verify`, `emit_rust_module`, or a TensorMap or binding
   constructor. Port the helper to the public API, or replace the assertion
   with an assertion on the `Program` (redesign §3 layer 4). Delete only the
   asserts that pin the implementation.
3. **Three-mode op tests** (`rule = signals:three-mode`) also require clean
   racecheck and synccheck verdicts. Keep all three modes in the v2 version;
   the cheap clean-verdict check is where scheduler and observer regressions
   show up first.
4. **Corpus** (`target = conformance`): the conformance snapshots already
   cover the canonical cases' verdict, findings and output bits. The per-case
   test files keep their fixture and reference self-tests (F). The
   determinism, full-launch and GPU three-way checks move to conformance
   once `NUMSIM_IMPL=v2` is green.

## Public-API A tests under `NUMSIM_IMPL=v2` (phase 3 review)

I reran the 440 public-surface A functions (762 items) under
`NUMSIM_IMPL=v2` at 9484204 plus the working tree: 367 items passed and
395 failed. `coverage/v2_public_status.tsv` records each function's outcome
class, and the classifier copies it into the CSV's `v2_status` column.

The coordinator asked whether the two "test-port" failure classes are
category C. **By this plan's rules, almost none of them are.** A test is C
only when its contract *is* the pin. A test that runs a kernel and checks a
semantic outcome, and also pins text or internals, is A: port the semantic
part and drop the pin.

| v2 failure class | items / functions | ruling | action |
| --- | --- | --- | --- |
| Internals pin: `stats["task_count"/"poll_order"]`, `module.rust_source`, `KernelSpec.semantic_requirements`, resolved op-name sets, the `Hint:` text | 64 / 34 | 33 functions also assert outputs or a verdict, so they are A (mixed). 1 function, `test_non_tensor_bulk_forms.py::test_bulk_g2s_cluster_dynamic_predicate_transpiles`, asserts nothing but the op-name set, so it is C. | Strip the pin; the semantic asserts already pass or fail on their own. The single C row joins wave 5b (`OVERRIDES`). |
| Legacy error text: `pytest.raises(match=…)` regex mismatch, but v2 raises the same exception type for the same fault | 19 / 14 | A. The fail-closed contract holds; only the wording differs. Examples: `invalid_operand: … 29 / 0`, `trap`, `misaligned`, `out_of_bounds`, `sync_protocol_error: RegPool(MissingWarpgroupSync)`, `divergence`. | Replace `match=` with the v2 error kind. |
| Same as above, but v2 reports **incomplete** where legacy raised an error | 8 / 6 | A, and a semantic delta: a TMEM column not in a live allocation is `Unsupported: not modeled`; a blocking wait with mixed lane readiness or a divergent `__syncwarp` mask is `divergent_block`; a loop budget is `analysis_incomplete`. | Needs a delta row or a v2 fix (interp/sync). Not a test-port issue. |
| `DID NOT RAISE`: v2 accepts a kernel legacy rejected | 17 / 13 | A, and a real v2 fail-closed gap. These cover mutated known-CUDA helpers (`flashkda`, `gdn_lg2`, packed fma: 6 functions), fp8 identity reinterpret, `tensormap` update without release, a TMEM allocation live at exit, `ignore_oob` count range, `shfl` width, unknown tile config keys, warp-gemm fragment layout, and f64 directed rounding. | Owners: lowering (helper validation, tile config) and interp (runtime checks). |

**Result: of the "44 legacy error text" and "34 legacy internals" failures,
1 function is C.** It is added to the wave-5b list. The other 77 functions
stay A. I marked them in the CSV (`v2_status` = `pin-internals`,
`pin-message`, `pin-message(error->incomplete)`,
`v2-accepts-legacy-rejection`); they are the A-internal port queue. The 14
`v2-accepts-legacy-rejection` and `error->incomplete` functions should go to
the owners named above, not to test porting.

### Phase 4: v2 copies of the pinned A tests

`tests/numsim/v2/ports/` holds v2 copies that use the public `v2` API and
the `requires_v2_engine` marker. Each docstring cites its legacy test. They
are mapped in `coverage/v2_ports_messages.tsv` (21 rows) and
`coverage/v2_ports_internals.tsv` (33 rows).

| source | legacy functions | v2 copy | result |
| --- | ---: | --- | --- |
| `pin-message` | 14 | `test_error_kinds.py` asserts exception type + v2 error `kind` (+ structured detail such as `MissingWarpgroupSync`), never text | 13 pass. 1 `v2_gap`: v2's binder does not reject a one-byte-per-value float4 array, so this was not a wording difference after all. |
| `pin-internals` | 33 | `test_internals_<stem>.py`, pins stripped (`stats`, `rust_source`, `semantic_requirements`, op-name sets, `Hint:`); completion now asserted via `result.status` | 28 pass. 5 have xfail cases: predicated `.v2`/`.v4` f32 atomics update predicated-off lanes (6 cases, interp); `sm100_2sm_leader_smem_addr` unsupported; TMEM store outside the warp's sub-partition; zero-step loop is a budget incomplete rather than an error. No function turned out to be a pure pin. |
| partial E | 7 | `test_partial_tile_forms.py`, `test_single_lane_participation.py`, keeping only TVM-compilable params | 5 pass. 2 have `v2_gap` cases: `tcgen05.wait::ld/st` by one elected lane is clean, because `tcgen_wait` lacks the `full_warp` check; v2 accepts TMA reduce operation/dtype pairs that PTX does not define. |

`tests/numsim/v2/ports` + `tests/numsim/v2/checkers` together: 203 passed,
51 xfailed, 27 xpassed. Two raw-spelling `wait_until` racecheck halves assert
the delta-R3 behaviour (`review` `undeclared_protocol_word`), not the
legacy clean or data-race verdict.

**W4-12 reduction ruling.** `ports/test_tile_reduction_dispatch.py`
(mapped in `coverage/v2_ports_reductions.tsv`) ports three tests from
`test_tile_reduction_variants.py`. The legacy versions pinned the legacy
frontend's sequential, identity-seeded reductions. The ports assert what the
TVM-dispatched code computes instead, and all 3 pass:

- **Warp sum.** Checked against an independent `shfl.bfly` model (xor 16, 8,
  4, 2, 1). `[1e20, 1, -1e20, 1]` gives 2.0 where legacy gave 1.0.
- **Unseeded `3input_maxmin`.** An all-NaN input gives `0x7FFFFFFF` for both
  max and min (legacy: `0xFF7FFFFF` / `0x7F7FFFFF`).

The tests cite the numsim-behaviour-deltas rows W4 is adding.

**`test_atomic_f32_noftz`.** The port already dropped the `rust_source` pin
in phase 4. Its 6 width-1 cases pass numerically. The 6 `.v2`/`.v4` cases
stay `v2_gap`: predicated-off lanes are still updated.

Further legacy expectations v2 does not meet (not xfailed, because the
legacy assertion admitted the v2 behaviour):

- the divide-by-zero diagnostic names the whole warp mask, not lane 7;
- a lane-varying but aligned `ldu` address runs to completion, so the
  uniformity check is missing.

## Kill order (aligned with redesign §4 step 5)

Each wave lands as one change. Judge it with the clean-baseline diff procedure
in `tests/CLAUDE.md`: compare failure sets, not counts. A wave that removes a
check adds its replacement in the same change.

| wave | precondition | delete | add or keep |
| --- | --- | --- | --- |
| 0 | none (TVM cannot compile these forms) | E rows outside the reviewed maps | none |
| 1 | `cargo test -p numsim-core --test 'racecheck_*' --test 'synccheck_*'` green, with `*_legacy_ports.rs`; rulings on the 5 ignored ports | B rows with status `covered` or `ported` (227) | the Rust scenario files (already in place) |
| 2 | every v2 replacement of the row passes (`retire_tests.py --wave 2` runs them) | B rows with status `v2_kernel_test` | `tests/numsim/v2/checkers`, `tests/numsim/v2/ports/test_single_lane_participation.py` |
| 3 | conformance green under `NUMSIM_IMPL=v2` (steps 1 to 3) | A `conformance` rows (corpus verdict gates, wiki racecheck, synccheck corpus) | corpus fixtures (F); `tests/conformance` |
| 4 | every v2 copy in `coverage/v2_ports_*.tsv` passes (`retire_tests.py --wave 4` runs them) | public-API A functions with a passing v2 copy; the rest are `flip` (pass unchanged under v2) or `hold` | `tests/numsim/v2/ports`; flipped public files |
| 5a | D-live outputs recorded once as sha256 or bit goldens (conformance README "Not covered yet") | the `run_paired_primfunc` harness, D rows | recorded goldens + a v2 runner; the `numsim-oplib` cvt goldens stay |
| 5b | step 5 of the redesign: delete `engine-rs/`, `frontend-rs/` and the legacy Python layer | every C row (`retire_tests.py --wave 5b`), in the same change as the code it pins; A-internal leftovers that were never ported | F, N |

Until wave 5b, C tests stay green against the legacy code. Deleting them
earlier gains nothing and loses the legacy oracle while v2 is still catching
up.

## Deletion lists

`scripts/numsim-v2/retire_tests.py --wave {0,1,5b} [--dry-run|--markdown]`
applies a wave from the CSV. It `git rm`s a file when every test function in
it is retired, and otherwise cuts the selected functions out, decorators
included. Every cut was simulated on scratch copies for all three waves: the
files still parse, and exactly the selected functions disappear. **Do not run
it without `--dry-run` yet**: legacy remains the oracle until conformance is
complete.

Wave sizes (dry run at 2a5895a plus the working tree; waves 2 and 4 run the
replacements first):

- wave 0: 1 file, plus 60 functions in 14 files. This includes the six W1
  out-of-scope tests (001a09f): the three `tcgen_cp_*` replicated-TMEM-view
  tests, `mxfp4_uses_ue8m0`, `legacy_m16n8k32_int8` and `legacy_ldmatrix_x1`.
- wave 1: 26 files (92 tests), plus 135 functions in 26 files.
- wave 2: 1 file (2 tests), plus 36 functions in 14 files. 14 rows are held.
- wave 4: 9 files (11 tests), plus 71 functions in 31 files, i.e. 82 retired
  functions. 310 public functions flip. 48 are held.
- wave 5b: 43 files (192 tests), plus 197 functions in 51 files. Run
  `--wave 5b --markdown` for the list.

Port maps used by wave 4: `coverage/v2_ports_{messages,internals,reductions,stats,triage,deltas,racedeltas,w1triage}.tsv`.

### Wave 0 (E: tile forms TVM's dispatch rejects)

Wave 0: 1 whole files (1 tests), 60 functions cut from 14 files.

```bash
cd tirx_harness
git rm tests/numsim/registry/test_tile_dispatch_invariance.py
```

- `tests/numsim/integration/test_block_scaled_gemm_artifact.py`: `test_block_scaled_gemm_normalizes_instruction_kind_and_shape`, `test_fp8_scale_permute_and_sf_reuse_tmem_roundtrip`, `test_fp8_block_scaled_gemm_derives_scale_columns_from_each_invocation`, `test_repeated_block_scaled_callsites_inside_a_loop_do_not_share_history`, `test_interleaved_block_scaled_calls_are_independent_of_prior_calls`, `test_block_scale_layout_that_violates_instruction_row_stride_fails_closed`, `test_block_scale_region_min_selects_the_physical_scale_coordinates`
- `tests/numsim/integration/test_gemm_async_artifact.py`: `test_dense_gemm_async_gathers_physical_operands_and_accumulates_tmem`, `test_dense_fp8_gemm_async_matches_numpy`, `test_dense_gemm_async_no_swizzle_descriptor_matches_numpy`, `test_thread_scope_gemm_async_requires_one_runtime_issuer`, `test_dense_gemm_async_rejects_wrong_tmem_a_layout`, `test_dense_gemm_async_rejects_declared_geometry_that_disagrees_with_operands`, `test_tile_tf32_gemm_async_fails_closed`, `test_cta_group2_m64_tcgen_mma_uses_layout_b_independently_of_declared_layout`, `test_cta_group2_routes_each_pair_within_a_four_cta_cluster`, `test_dense_gemm_async_handles_repeated_dynamic_index_loads`, `test_dense_gemm_async_runs_numpy_backend_on_two_cluster_workers`, `test_cta_group2_gathers_both_shared_shards_and_scatters_tmem`, `test_cta_group2_gathers_both_tmem_a_shards`, `test_all_inactive_gemm_async_is_a_noop`
- `tests/numsim/integration/test_permute_layout_artifact.py`: `test_shared_permute_layout_copies_logical_values_between_physical_layouts`, `test_explicit_permute_layout_dispatch_uses_canonical_semantics`, `test_global_permute_layout_writes_destination_physical_order`, `test_shared_permute_layout_zero_fills_only_uninitialized_padding`, `test_shared_permute_layout_zero_fills_fp16_padding`, `test_shared_permute_layout_zero_fills_bf16_padding`, `test_shared_permute_layout_zero_fills_fp8_padding`
- `tests/numsim/integration/test_reported_layout_regressions.py`: `test_tma_reductions_are_atomic_but_plain_overlapping_stores_still_race`, `test_singleton_outer_gemm_operand_reaches_each_semantic_checker`, `test_warp_gemm_rejects_an_outer_single_lane_election`, `test_warpgroup_tile_op_allows_independent_full_warp_participation`, `test_warpgroup_tile_op_rejects_partial_warp_participation`, `test_numsim_rejects_a_misaligned_tma_shared_component`, `test_checkers_reject_a_misaligned_tma_shared_component`, `test_checkers_accept_an_aligned_tma_tensor_issue`
- `tests/numsim/integration/test_tcgen_transfer_artifact.py`: `test_tcgen_cp_expands_tlane_replicas`, `test_tcgen_cp_supports_rank3_multi_instruction_layout`, `test_tcgen_cp_cta_group2_supports_float16_payloads`, `test_tcgen_ldst_rejects_tmem_layout_outside_fixed_instruction_abi`, `test_tcgen_cp_rejects_destination_lane_permutation`, `test_tcgen_cp_rejects_declared_shape_that_disagrees_with_layout`
- `tests/numsim/runtime/test_dense_mma_forms.py`: `test_legacy_m16n8k32_int8_reuses_dense_form_and_engine`
- `tests/numsim/runtime/test_matrix_memory_domain_oracle.py`: `test_legacy_ldmatrix_x1_domain_matches_independent_fragment_mapping`
- `tests/numsim/runtime/test_runtime_form_domain_oracles.py`: `test_snapshot_copy_variable_min_runtime_domain_has_exact_oracle`
- `tests/numsim/runtime/test_tile_codegen.py`: `test_fp8_register_to_shared_owner_copy_runs_in_numsim`, `test_forced_elementwise_dispatch_does_not_change_local_semantics`, `test_typed_tma_reduce_rejects_non_store_direction`, `test_typed_tma_rejects_unknown_cache_hint`, `test_cta_copy_uses_canonical_semantics_independent_of_dispatch`
- `tests/numsim/runtime/test_tile_general_semantics.py`: `test_mxfp4_uses_ue8m0_scales_over_32_element_vectors`
- `tests/numsim/runtime/test_tile_owner_transport.py`: `test_pointwise_transports_unique_owners_across_warps`, `test_cast_unary_and_binary_share_owner_transport`
- `tests/numsim/runtime/test_tile_reduction_variants.py`: `test_shared_cta_reductions_support_float16_and_scope_completion`, `test_shared_cta_accum_reduction_writes_each_output_once`, `test_shared_reduction_uses_lexicographic_order`, `test_empty_reduction_axes_follow_identity_reduction_semantics`
- `tests/numsim/runtime/test_tile_unary_codegen.py`: `test_fill_accepts_untyped_python_literals`, `test_fill_accepts_an_integer_literal_wider_than_int32`
- `tests/numsim/runtime/test_tma_dtype_runtime_domain.py`: `test_typed_tma_rejects_unmodeled_production_dtypes`

### Wave 1 (B rows covered or ported by Rust scenario tests)

Wave 1: 26 whole files (92 tests), 135 functions cut from 26 files.

```bash
cd tirx_harness
git rm tests/analysis_tools/racecheck/test_bulk_reduction_widths.py
git rm tests/analysis_tools/racecheck/test_declared_word_shapes.py
git rm tests/analysis_tools/racecheck/test_native_async_lifetime_contracts.py
git rm tests/analysis_tools/racecheck/test_native_atomic_semantics.py
git rm tests/analysis_tools/racecheck/test_native_matrix_collective_sync.py
git rm tests/analysis_tools/racecheck/test_native_ordered_b128_load.py
git rm tests/analysis_tools/racecheck/test_native_proxy_async_fence.py
git rm tests/analysis_tools/racecheck/test_native_racecheck_arrive_snapshot.py
git rm tests/analysis_tools/racecheck/test_native_racecheck_lane_order.py
git rm tests/analysis_tools/racecheck/test_native_racecheck_per_warp.py
git rm tests/analysis_tools/racecheck/test_native_racecheck_release_rmw_handoff.py
git rm tests/analysis_tools/racecheck/test_native_racecheck_tmem_arrive_snapshot.py
git rm tests/analysis_tools/racecheck/test_native_same_warp_tmem_review.py
git rm tests/analysis_tools/racecheck/test_native_shared_publication.py
git rm tests/analysis_tools/racecheck/test_native_tcgen_thread_fence.py
git rm tests/analysis_tools/shared/test_native_dense_cta2_mma_ordering.py
git rm tests/analysis_tools/synccheck/test_native_drain_tail.py
git rm tests/analysis_tools/synccheck/test_native_shared_control_relay.py
git rm tests/analysis_tools/synccheck/test_native_split_site_named_barrier.py
git rm tests/analysis_tools/synccheck/test_register_allocation.py
git rm tests/numsim/integration/test_fp8_tmem_a_effects.py
git rm tests/numsim/runtime/test_bulk_copy_scopes.py
git rm tests/numsim/runtime/test_bulk_g2s_scopes.py
git rm tests/numsim/runtime/test_mixed_version_reads.py
git rm tests/numsim/runtime/test_store_sinks.py
git rm tests/numsim/runtime/test_tensormap_publication.py
```

- `tests/analysis_tools/racecheck/test_declared_word_regressions.py`: `test_a_relaxed_publication_leaves_the_payload_unordered`, `test_a_declared_wait_takes_the_edge_against_a_releasing_store`, `test_a_relaxed_store_composes_into_a_release_with_a_preceding_fence`, `test_raw_protocol_reports_only_the_proven_payload_race`, `test_a_phase_flip_predicate_takes_the_edge_of_the_arrival_that_flipped_it`, `test_two_waits_on_one_word_never_conflict_with_each_other`, `test_a_morally_strong_pair_is_exempt_unless_one_side_is_a_plain_read`, `test_a_publication_too_wide_to_poll_fails_closed`, `test_a_publication_a_wait_can_poll_is_clean`, `test_a_publication_whose_scope_reaches_the_waiter_is_clean`, `test_a_publication_whose_scope_stops_short_of_the_waiter_is_reported`, `test_a_wait_does_not_bridge_the_async_proxy`, `test_a_generic_payload_behind_the_same_wait_is_clean`, `test_a_first_lap_ring_credit_wait_is_explained`, `test_a_later_lap_ring_credit_wait_still_needs_its_publication`, `test_a_host_initialized_value_explains_the_wait`
- `tests/analysis_tools/racecheck/test_native_alias_advisory.py`: `test_public_native_non_stale_pool_alias_controls_are_clean`, `test_public_native_tcgen_alloc_result_is_visible_to_every_lane`
- `tests/analysis_tools/racecheck/test_native_global_scoped_hb.py`: `test_scoped_global_message_passing_and_fence_halves_are_clean`, `test_scoped_global_missing_edge_or_misplaced_fence_reports_data_race`, `test_plain_flag_and_data_conflicts_are_both_reported`, `test_ptx_cluster_fence_orders_same_cta_message_passing`
- `tests/analysis_tools/racecheck/test_native_global_scoped_hb_matrix.py`: `test_same_cluster_release_acquire_is_clean`, `test_host_initialized_version_is_before_all_kernel_reads`, `test_same_value_and_aba_observations_use_only_the_exact_writer`, `test_publication_and_acquisition_are_lane_precise`, `test_rmw_release_sequence_and_morally_strong_rules`, `test_torn_scoped_read_reports_the_unordered_mixed_size_writes`, `test_async_store_requires_completion_handoff_before_publication`, `test_tcgen_commit_mbarrier_forwards_the_issuing_thread_publication`
- `tests/analysis_tools/racecheck/test_native_kernel_contracts.py`: `test_local_scalar_indices_preserve_strided_race_and_nested_clean_case`
- `tests/analysis_tools/racecheck/test_native_racecheck_artifact.py`: `test_native_racecheck_reports_exact_cross_warp_shared_write_race`, `test_native_racecheck_oob_is_exact_error_or_clean_from_concrete_mask`, `test_native_racecheck_raw_bulk_uses_the_concrete_active_issuer_lane`, `test_native_racecheck_raw_bulk_rejects_a_true_async_write_race`, `test_native_racecheck_typed_tma_multicast_records_each_target_allocation`, `test_public_native_racecheck_typed_tma_overlapping_destinations_race`, `test_public_native_racecheck_typed_tma_uses_exact_global_subregion`, `test_native_racecheck_raw_tma_records_exact_global_and_shared_payload`, `test_public_native_racecheck_raw_tma_overlapping_destinations_race`, `test_native_racecheck_mbarrier_release_acquire_orders_shared_access`, `test_public_native_racecheck_accepts_cluster_arrive_through_remote_view`, `test_native_racecheck_direct_named_barrier_hb_is_clean`, `test_native_racecheck_reports_cross_cluster_global_waw_independent_of_workers`, `test_native_racecheck_reports_same_cluster_global_waw`, `test_native_racecheck_reports_cross_cluster_global_tile_copy_conflict`, `test_native_racecheck_cross_cluster_atomic_modification_order_is_clean`
- `tests/analysis_tools/racecheck/test_native_racecheck_exact_control.py`: `test_public_native_racecheck_resolves_data_guarded_cross_lane_hazard_exactly`, `test_public_native_racecheck_preserves_tile_address_local_scalars`
- `tests/analysis_tools/racecheck/test_native_raw_async_copy_footprints.py`: `test_cp_async_mbarrier_arrive_pending_count_completes_clean`, `test_cp_async_mbarrier_arrive_flags_an_unwaited_destination_read`, `test_cp_async_mbarrier_arrive_synccheck_is_unchanged`, `test_bulk_g2s_cta_completes_clean`, `test_bulk_g2s_cta_flags_an_unwaited_destination_read`, `test_bulk_g2s_cta_synccheck_reports_no_finding`, `test_bulk_g2s_cta_ignore_oob_short_source_synccheck_reports_no_finding`, `test_bulk_g2s_multicast_completes_clean`, `test_bulk_g2s_multicast_flags_an_unwaited_remote_target_read`, `test_bulk_g2s_multicast_ignores_a_read_on_an_unselected_cta`, `test_bulk_g2s_multicast_synccheck_reports_no_finding`, `test_bulk_s2s_cluster_completes_clean`, `test_bulk_s2s_cluster_flags_the_unwaited_consumer`, `test_bulk_s2s_cluster_synccheck_reports_no_finding`, `test_bulk_s2g_masked_completes_clean`, `test_bulk_s2g_masked_flags_a_write_to_a_selected_byte`, `test_bulk_s2g_masked_ignores_a_write_to_an_unselected_byte`, `test_bulk_s2g_masked_synccheck_reports_no_finding`, `test_tma_gather4_completes_clean`, `test_tma_gather4_flags_an_unwaited_destination_read`, `test_tma_gather4_synccheck_reports_no_finding`
- `tests/analysis_tools/racecheck/test_native_sub_word_accesses.py`: `test_racecheck_sees_a_sub_word_access_as_a_one_byte_footprint`
- `tests/analysis_tools/racecheck/test_native_vector_destination_loads.py`: `test_racecheck_traces_the_element_footprint_of_a_vector_destination_load`, `test_racecheck_reports_a_vector_destination_load_racing_an_unordered_write`, `test_acquire_vector_destination_load_keeps_its_ordering_specialization`
- `tests/analysis_tools/racecheck/test_signal_diagnostics.py`: `test_raw_spin_missing_hb_reports_race_with_conditional_wait_hint`, `test_hb_ordered_raw_spin_does_not_need_a_wait_declaration`
- `tests/analysis_tools/synccheck/runtime/test_device_protocol_ops.py`: `test_divergent_unaligned_named_barrier_runtime`, `test_cp_async_predicate_tracks_only_issuing_lanes`, `test_cp_async_zero_fill_keeps_all_lanes_in_their_groups`, `test_ldgsts_cp_async_uses_the_same_thread_local_group_protocol`, `test_uncommitted_cp_async_is_a_protocol_error`
- `tests/analysis_tools/synccheck/test_native_kernel_contracts.py`: `test_native_synccheck_models_the_trailing_ninth_warp_and_launch_attr`, `test_native_synccheck_trailing_warp_does_not_hide_real_under_arrival`, `test_native_synccheck_counts_exact_lane_and_elect_participants`, `test_native_synccheck_resolves_data_guard_instead_of_guessing_participation`, `test_native_synccheck_does_not_drop_setmaxnreg_inside_loop`, `test_native_synccheck_accepts_setmaxnreg_with_trailing_warps`, `test_native_synccheck_preserves_depth_two_multi_generation_pipeline`, `test_native_synccheck_reduces_tma_completion_waiter_interleavings`, `test_native_synccheck_keeps_conditional_tcgen_alloc_at_one_warp`, `test_native_synccheck_executes_conditional_tmem_pool_with_exact_topology`, `test_native_synccheck_accepts_native_setmaxnreg_budget_shapes`, `test_native_synccheck_rejects_native_setmaxnreg_oversubscription`, `test_native_synccheck_accepts_six_warpgroup_launch_allocation`, `test_native_synccheck_rejects_six_warpgroup_rounding_residual_as_pool`
- `tests/analysis_tools/synccheck/test_native_synccheck_artifact.py`: `test_fixed_sync_state_finds_bad_tcgen_order_after_clean_native_execution`, `test_public_native_synccheck_wait_before_init_is_exact_error`, `test_public_native_synccheck_plain_arrive_before_init_is_exact_error`, `test_public_native_synccheck_accepts_cta_sync_as_mbarrier_init_publication`, `test_public_native_synccheck_accepts_cluster_arrive_through_remote_view`, `test_public_native_synccheck_arrive_expect_tx_before_init_is_exact_error`, `test_public_native_synccheck_plain_arrival_overflow_is_typed_error`, `test_public_native_synccheck_executes_data_dependent_protocol_exactly`, `test_native_synccheck_deadlock_is_error_even_with_staged_waits`, `test_public_native_synccheck_handles_completion_issued_before_future_expectation`, `test_native_synccheck_tracks_completion_tokens_across_future_generation`, `test_native_synccheck_accepts_late_wait_ordered_before_next_completion`, `test_public_native_synccheck_rejects_unconsumed_generation_reuse`, `test_public_native_synccheck_transaction_under_delivery_is_exact_error`, `test_public_native_synccheck_transaction_over_delivery_is_exact_error`, `test_public_native_synccheck_executes_one_racy_control_path_without_replay`, `test_synccheck_reports_nonblocking_arrival_successor_error_without_replay`, `test_synccheck_reports_blocking_wait_handoff_successor_error_without_replay`, `test_synccheck_no_waiter_arrival_checks_its_successor_without_replay`, `test_native_synccheck_named_barrier_gateway_reports_execution_contract_error`
- `tests/numsim/integration/test_artifact_build.py`: `test_bulk_shared_to_cluster_has_exact_racecheck_payload_accesses`
- `tests/numsim/runtime/test_async_release.py`: `test_bulk_wait_does_not_acquire_async_release`, `test_async_release_publishes_only_pre_issue_work`
- `tests/numsim/runtime/test_bulk_reduce_s2g_f32.py`: `test_bulk_reduce_partial_overlap_is_elementwise_atomic`
- `tests/numsim/runtime/test_mbarrier_drop.py`: `test_mbarrier_drop_invalid_count`
- `tests/numsim/runtime/test_mbarrier_lane_semantics.py`: `test_checkers_group_pending_blocking_waits_on_distinct_barriers`, `test_synccheck_keeps_one_projection_for_a_multi_barrier_wait`, `test_checkers_allow_distinct_barrier_waits`, `test_relaxed_query_does_not_acquire_arriving_threads_memory`
- `tests/numsim/runtime/test_memory_ops.py`: `test_bulk_g2s_cta_has_exact_racecheck_payload_accesses`
- `tests/numsim/runtime/test_red_async.py`: `test_shared_async_reduction_completion`
- `tests/numsim/runtime/test_tcgen05_ld_red.py`: `test_tcgen_load_reduction_preserves_the_store_wait_contract`
- `tests/numsim/runtime/test_tcgen05_ld_spcompress.py`: `test_tcgen_compression_preserves_store_wait`
- `tests/numsim/runtime/test_tma_im2col_multiissuer.py`: `test_im2col_multiissuer_requires_completion_wait`
- `tests/numsim/runtime/test_tma_multiissuer.py`: `test_tma_multiissuer_requires_wait_and_disjoint_destinations`
- `tests/numsim/runtime/test_wait_until.py`: `test_wait_has_its_own_hb_event`, `test_a_wait_that_waits_for_both_arrivals_may_read_both`, `test_a_wait_that_waits_for_one_arrival_may_not_read_the_other`, `test_a_wait_woken_by_a_plain_write_reports_the_missing_edge`, `test_a_wait_on_a_word_that_carries_its_own_payload_is_clean`

## Step 5: final deletion plan (dry run, 2a5895a + working tree)

`scripts/numsim-v2/retire_legacy.py --dry-run [--list]` prints the whole
step. `--apply` performs it, and refuses while the blockers below remain
unless given `--force`. It has not been run.

### What goes

- **Legacy code (240 tracked files).** Removed with `git rm`:
  - `numsim/engine-rs/` (126 files)
  - `tirx_harness/frontend-rs/` (90 files)
  - `numsim/transpiler/` (14 files)
  - `api.py`, `bindings.py`, `checker_runner.py`, `checkers.py`,
    `checker_report.py`, `checker_render.py`, `abi.py`, `host_abi.py`,
    `value_analysis.py`
  - the `thirdparty/tvm-rust-ext` submodule and its `.gitmodules` entry
  - the untracked `_tvm_rust_ext.abi3.so`, `.identity` and
    `_thirdparty_licenses/`
- **Legacy tests.** The union of waves 0, 1, 2, 4 and 5b removes 92 test
  files (420 tests) and cuts 375 functions from 85 files. 4 support modules
  are left with no importer: `racecheck/_native_race_trace.py`,
  `support/paths.py`, `support/runtime_cases.py` and
  `support/tirx_device_surface.py`. Wave 3 (corpus gates) is added once
  conformance matches.
- **Migration tooling that needs a legacy capture:** `capture_plugin.py`,
  `tile_rejections.py` and `lower_sweep.py`.

### What stays or moves (v2 imports it)

- **Kept:**
  - `numsim/errors.py`;
  - `numsim/cases.py` (public `TensorMap` / `Im2col` / `NumSimCase` /
    `ComparisonSpec`; v2 `run.py` decodes `cases.TensorMap` images);
  - `dtype_abi.py` + `dtype_registry.json` (used by `cases`);
  - `report.py` (`Mismatch`, `NumSimReport`).
- **Moved:**
  - The `compare` closure of `api.py` (169 lines: `NumSimResult`,
    `_comparison_*`, `compare`) moves to `v2/compare.py`, extracted from the
    AST. `v2/report.py` imports it from there.
  - `engine-rs/SUPPORTED_OPS.md` moves to `numsim-oplib/`, because
    `gen_registry.py` reads it.

### What is rewritten

The script generates these files:

- `tirx_harness/__init__.py`: `racecheck` and `synccheck` call `numsim.v2`.
- `numsim/__init__.py`: re-exports v2 plus `cases` and `errors`.
- `setup.py`: builds `numsim_core_py` (`cargo build -p numsim-py --features
  extension-module`) instead of the tvm-rust-ext frontend; the sdist
  submodule copy is dropped.
- `MANIFEST.in`
- `gen_registry.py`

These are checklist items in the plan, edited by hand:

- `tests/conftest.py`: drop the `NUMSIM_IMPL` shim.
- `tests/conformance/{snapshot.py,test_conformance.py,README.md}`: v2 is
  the implementation, the snapshots are the oracle, and a regeneration only
  writes `<mode>.delta.json` naming a delta row.
- `tests.yml`: the submodule step becomes a `build_dev.sh` step, plus
  `cargo test --workspace`.
- `build_wheels.yml`
- `scripts/smoke_wheel.py`
- `numsim/{CLAUDE,AGENTS}.md`
- `support/paths.py`
- optionally, `NUMSIM_V2_*` becomes `NUMSIM_*` in `v2/options.py`.

### Docs

The submodule line is removed from `docs/optimization-runs.md`,
`docs/installation.md`, `tests/CLAUDE.md` and `tests.yml`.

There are 26 `(pending: ...)` markers:

- **13 resolve on deletion** and are removed:
  - README:47
  - api/checkers:29
  - api/index:19
  - api/inputs:102
  - api/numsim:10
  - components/tools:39
  - architecture:16 and 215
  - installation:22, 54 and 71
  - numsim/CLAUDE:5 and AGENTS:5
- **10 need a decision made as part of step 5**, and are listed by the
  script:
  - the snapshot policy: architecture:164 and 191, CLAUDE:43 and AGENTS:43;
  - the `NUMSIM_V2_` prefix (api/numsim:134);
  - the `NumSimBuildError` type (api/numsim:163);
  - the report exception class (api/checkers:67);
  - the operation-table move (tools:162);
  - single-launch entry points (tools:431);
  - `InputError` versus `incomplete` (tools:440).
- **3 stay:** the `"auto"` worker count and the backend decision (2).

Installation:22 needs a reread when it is removed. A Cargo toolchain is still
needed to build `numsim_core_py`, and also for the codegen backend if that
backend survives.

### Blockers (what must be green under v2 before deletion)

I ran fresh `NUMSIM_IMPL=v2` runs at 2a5895a:

- public-API A: 544 of 755 items pass;
- internal-surface A: 518 of 793 items pass.

The internal-surface tests run under the same shim, because their legacy
imports still resolve today. `coverage/step5_a_status.tsv` places every A
function:

| step-5 bucket | public | internal | what happens |
| --- | ---: | ---: | --- |
| flip (passes unchanged under v2) | 310 | 275 | Kept. Internal files only lose their module-level legacy import. |
| retired: a passing v2 copy exists (wave 4) | 75 | 5 | Deleted. |
| gpu-only (skipped without a GPU) | 4 | 0 | Decided with the microtests. |
| **needs-port** (fails only on pins) | 6 | 56 | W9: a v2 copy or a kind assertion. |
| **uses-legacy-internals** (the test calls `analyze` ×67, `verify` ×15, `_descriptor_storage` ×10, `prepare_bindings` ×8, `emit_rust_module` ×8, …) | 0 | 96 | Port to `v2.transpile` / `Program` assertions, or delete if the assertion is the pin. |
| **blocked-v2** (v2 behaviour differs, no ruling) | 38 | 92 | Owner work. |

The `blocked-v2` functions break down by class:

- **Public:** engine-stops 20, other-assertion 5, the 4 bugs filed under
  `CONTRACT_REQUESTS.md` "W9-public-API", synccheck-verdict 4,
  lowering-rejects 3, racecheck-verdict 2, v2 accepts a legacy rejection 1.
- **Internal:** other-assertion 40 (not triaged yet), engine-stops 24,
  lowering-rejects 16, v2 accepts a legacy rejection 12.

The other blockers:

- **Conformance itself.** `tests/numsim/corpus/kernels/state_update.py`
  imports the legacy `host_abi` and `transpiler.frontend` at module level.
  `attention.py` and `native_multishape.py` import `bindings` helpers. These
  modules are reached through `canonical_cases`, so deleting the legacy
  modules breaks `tests/conformance` at import. Port these three fixture
  modules first; this is the hard blocker.
- **Surviving modules that import deleted code.** 85 surviving test or
  support modules import a deleted legacy module (`--list` shows them).
  Besides the corpus fixtures, these include:
  - `microtests/harness.py` (`bindings.prepare_bindings`), which affects
    every D test: switch it to the v2 binder, then run the microtests once
    on a GPU under v2;
  - `support/three_way.py` (`bindings`, `host_abi`, `analyze`), which
    affects the corpus GPU three-way tests;
  - `support/manifest.py`;
  - the A-internal files above.
- **Held replacements.** 14 wave-2 rows and the wave-4 holds have a v2
  replacement that is still xfail (`v2_gap` or `no_spec`).
- **Wave 3 (corpus gates).** It retires once the remaining racecheck
  conformance cases match (90 of 98 with oracle at 43b3e2f).

In short, step 5 can run when:

1. the corpus fixture modules and the microtest harness are ported to v2;
2. the 62 `needs-port` and 96 `uses-legacy-internals` A functions are ported
   or deleted;
3. the 130 `blocked-v2` functions are fixed, ruled with a delta row, or
   accepted as failures to delete;
4. the held replacements pass.

The rest (585 flips, 80 retired copies, waves 0, 1 and 5b) is mechanical.


## Semantics in legacy tests that no new spec mentions

Each item below was confirmed by grep across `racecheck-semantics.md`,
`racecheck-isa-answers.md`, `sync-semantics.md`, `sync-isa-answers.md`,
`synccheck-explorer.md` and the three delta docs. Each needs a spec sentence
or a delta row, whichever way it is ruled.

1. **`alias_stale_read`.** This review advisory flags a stale logical buffer
   name over pooled smem or TMEM. It is dropped silently: there is no delta
   row, no `AdvisoryKind`, and no contract field. `v2/report.py` still lists
   it in `ADVISORY_KINDS`.
2. **Host-input aliasing.** Overlapping host arrays bound to two parameters
   must be checked as one allocation. This covers raw pointer offsets,
   TensorMap base replacement and `discard`.
3. **Same-rank `mapa`.** Legacy resolves a `mapa` to the CTA's own rank as a
   `shared::cta` window: a `.shared::cta` fence is clean and a
   `.shared::cluster` fence races. The new core treats every `mapa` address as
   `shared::cluster`, which gives the opposite verdicts. Delta X2's "Legacy:
   Race" column is inaccurate for this case.
4. **How a bulk reduction (`cp.reduce.async.bulk`) is emitted.** It is atomic
   per PTX element. A matching-width peer `red` is clean; a `.u32` peer racing
   on f16 elements races. Nothing specifies that the producer emits an
   atomic `Rmw`, relaxed `.gpu`, `returns_value = false`, one span per
   element. A wrong emission gives a false race or a missed one.
5. *(Resolved: the core now reports a wide release write to a declared word as incomplete, as the spec says; the port is un-ignored.)*
6. **Declared words are global-only.** A `shared::cluster` poll can never be
   declared, so it always draws `UndeclaredProtocolWord`. Legacy kept it as a
   deliberately clean control.
7. **A declared `wait_until` as a thread sync for the tcgen05 fence pair.**
   It counts as an execution-ordering thread sync for the tcgen05
   before/after fence pair. The spec states this only for relaxed mbarrier
   arrive/wait (semantics table row 10).
8. **Fully predicated-off accesses** emit no access. No active-lane rule
   exists anywhere.
9. **Footprint narrowing.** `.ignore_oob` read clipping, `.cp_mask` byte
   selection and gather4 rows are not covered; only multicast is.
10. **Hidden accesses of tile and helper primitives** (`cta_sum` scratch and
    barriers, `permute_layout` zero-fill).
11. **The default `.cta` scope for a qualifier-less raw arrive on a remote
    mbarrier.** B1 states the default but not that it flips a real kernel's
    verdict.
12. *(Resolved by 1159d17: the allocation-result check is restored.)*
13. *(Resolved by 1159d17: host `Configure` is now the initial state.)*
14. **Kind names.** Several kinds were renamed with no delta row:
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
22. **Global `st.async.release` / `red.async.release` with no mbarrier as a
    publication.** Async-actor writes carry no release head.
23. **The `signal_protocol_error` kind** (a declared-word bypass) has no
    equivalent.
24. **The tcgen05 fence requirement** after a waited `tcgen05.st` or commit
    across a plain `cta_sync`. The new core races where legacy is clean, and
    no delta row covers it.
