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

## Counts (2026-10-08, after phase 2)

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
| numsim/integration | 325 | 2 | 140 |  | 40 | 3 |  | 510 |
| numsim/microtests |  |  | 1 | 55 |  |  |  | 56 |
| numsim/registry | 14 |  | 81 |  | 4 |  |  | 99 |
| numsim/runtime | 607 | 33 | 65 |  | 18 | 5 |  | 728 |
| numsim/v2 |  |  |  |  |  |  | 83 | 83 |
| **total** | **1000** | **278** | **387** | **55** | **62** | **44** | **85** | **1911** |

| A target | public facade only | imports legacy internals |
| --- | ---: | ---: |
| conformance | 19 | 22 |
| v2-api | 0 | 23 |
| v2-kernel-case | 440 | 496 |

| category | target / status | tests |
| --- | --- | ---: |
| A | conformance | 41 |
| A | v2-api | 23 |
| A | v2-kernel-case | 936 |
| B | covered | 88 |
| B | needs_kernel | 8 |
| B | ported | 139 |
| B | v2_kernel_test | 43 |
| C | delete | 387 |
| D | D-live | 48 |
| D | D-recorded | 7 |
| E | delete | 62 |
| F | keep | 44 |
| N | keep | 85 |

cvt goldens: 133 recorded forms in Python, 133 generated numsim-oplib tests

**E (tile forms TVM rejects).** The list was regenerated from scratch rather
than copied from W1:

1. A capture run of `tests/numsim` and `tests/analysis_tools`
   (`capture_plugin.py`, `-m 'not numsim_gpu'`) saved 2,475 kernels.
2. `tile_rejections.py` lowered each one.

71 kernels contain a rejected tile op, which matches W1's §D.2 count. They map
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
| v2_kernel_test | 43 | Contract events cannot express it. Replaced by a v2 kernel-level pytest in `tests/numsim/v2/checkers/`. |
| needs_kernel | 8 | Same as `v2_kernel_test`, but the pytest is not written yet. These are from the `other_b` review. |

The classifier refines the maps' "not B" rows:

- legacy-executor and frontend plumbing becomes C;
- op-support runs become A.

### Rust ports from contract events

Both files are in `numsim-core/tests/`. Every test is doc-commented with the
legacy `file.py::test[param]` it ports. With these files,
`cargo test -p numsim-core --test 'racecheck_*' --test 'synccheck_*'` is
fully green in the main tree. `synccheck_engine` also passes again.

- `racecheck_legacy_ports.rs`: 86 tests in sections g1 to g6; 81 pass and 5 are `#[ignore]`d.
- `synccheck_legacy_ports.rs`: 60 tests, all passing.

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

- **Undocumented divergences** (5, all `#[ignore]`d in
  `racecheck_legacy_ports.rs`). The four from the first batch were fixed by
  1159d17 and later commits, and their tests are un-ignored. Each remaining
  one needs a ruling: fix the core, or add a delta row and flip the test.

  | test | legacy | new core |
  | --- | --- | --- |
  | `g6_fp8_tmem_a_legacy_publication_is_clean` | `wait::st`, then `fence::after_thread_sync`, then `cta_sync`: clean | `async_lifetime_not_drained` races. `cta_sync` followed by `tcgen05.ld` with no after-fence also races, which is a likely false positive on legacy-clean kernels. The legacy kernel looks wrong per PTX, but no delta row says so. |
  | `g6_async_release_publishes_pre_issue_work` | `st.async`/`red.async .release` global, no mbarrier: clean | async-actor writes get no release head (`own_rel` is set only for warp lanes), so `wait_until` gets no edge. racecheck-semantics §5 implies async publications are edge sources. |
  | `g6_wait_bypass_is_a_signal_protocol_error_not_a_race` | `{signal_protocol_error}` | `data_race`; the kind does not exist. The verdict half passes. |
  | `g6_wait_for_one_arrival_with_acquiring_poll_still_races` | race | clean. The interpreter emits the poll as an `.acquire` Access before `WaitVerdicts`, and read-from takes the latest write. That hides the race and contradicts W1 ("earliest accepted"). The `declared_word` flag is never consulted. |
  | `g6_wait_woken_by_a_plain_write_is_incomplete` | `analysis_incomplete`; delta W2 says `WaitExitUnproven` | a data race, no incomplete. The error half passes. |

- **`needs_kernel`** (8, from `other_b.tsv`): `griddepcontrol.wait`
  (interpreter no-op, no event), multicast mask outside the cluster,
  `cp_mask` readonly overlap, two `st.async` mapped-offset tests,
  `sparse_b16` and `lut_b` lifetimes, TMA override destination footprint.
  These become v2 kernel tests in the next batch.
- **v2 xfails** (30 functions in `tests/numsim/v2/checkers`): resolved by
  rulings on the `no_spec` items, or by W8 for the `v2_gap` ones.

**Retirement readiness of B.**

- 227 rows can be deleted once their Rust tests are accepted (wave 1).
- 43 rows have a v2 replacement that runs today (wave 2, once green without
  xfail).
- 8 rows still need a replacement.
- 5 Rust tests still need rulings.

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

## Kill order (aligned with redesign §4 step 5)

Each wave lands as one change. Judge it with the clean-baseline diff procedure
in `tests/CLAUDE.md`: compare failure sets, not counts. A wave that removes a
check adds its replacement in the same change.

| wave | precondition | delete | add or keep |
| --- | --- | --- | --- |
| 0 | none (TVM cannot compile these forms) | E rows outside the reviewed maps | none |
| 1 | `cargo test -p numsim-core --test 'racecheck_*' --test 'synccheck_*'` green, with `*_legacy_ports.rs`; rulings on the 5 ignored ports | B rows with status `covered` or `ported` (227) | the Rust scenario files (already in place) |
| 2 | `tests/numsim/v2/checkers` green without xfail for a row; `needs_kernel` rows given a v2 test | `v2_kernel_test` and `needs_kernel` rows, and the `other_b` rows marked covered or ported | `tests/numsim/v2/checkers` |
| 3 | conformance green under `NUMSIM_IMPL=v2` (steps 1 to 3) | A `conformance` rows (corpus verdict gates, wiki racecheck, synccheck corpus) | corpus fixtures (F); `tests/conformance` |
| 4 | v2 facade switch: public-surface A files pass under v2 | none; A files flip to v2 | A public files; internal files ported one at a time |
| 5a | D-live outputs recorded once as sha256 or bit goldens (conformance README "Not covered yet") | the `run_paired_primfunc` harness, D rows | recorded goldens + a v2 runner; the `numsim-oplib` cvt goldens stay |
| 5b | step 5 of the redesign: delete `engine-rs/`, `frontend-rs/` and the legacy Python layer | every C row, in the same change as the code it pins; A-internal leftovers that were never ported | F, N |

Until wave 5b, C tests stay green against the legacy code. Deleting them
earlier gains nothing and loses the legacy oracle while v2 is still catching
up.

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
