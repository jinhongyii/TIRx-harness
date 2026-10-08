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

1. the reviewed coverage maps `coverage/{racecheck,synccheck}.tsv`, which
   were written by reading every test in `analysis_tools/{racecheck,synccheck}`;
2. manual overrides;
3. the tile-dispatch rejection list `coverage/tile_dispatch_rejections.txt`;
4. path rules;
5. AST signals: the calls a test makes (resolved transitively through
   module helpers, fixtures and `tests/**/support`), its assert text, the
   implementation identifiers it touches, and its name.

Outside the two reviewed directories the categories are heuristic. Spot
checks put the error rate at roughly 1 row in 10. The usual mistake is
classifying an A row whose real contract is legacy plumbing. Review each row
when its file is converted; the counts are good enough to plan from.

## Categories

| cat | meaning | fate |
| --- | --- | --- |
| A | Semantic kernel-level test: runs a kernel and checks outputs, verdict, error, or finding kinds. | `conformance`: corpus or wiki kernels, already snapshotted. `v2-kernel-case`: small kernel, rerun through the v2 API (see the "A" section). `v2-api`: `test_api.py`; v2 keeps the legacy signatures. |
| B | Checker-semantics unit test with a small kernel. | Rust scenario test from contract events (`numsim-core/tests/{racecheck,synccheck}_*.rs`). Status is `covered`, `ported`, `gap_unportable`, or `unreviewed`. |
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

## Counts (2026-10-08)

| directory | A | B | C | D | E | F | N | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (top) |  |  |  |  |  | 13 |  | 13 |
| analysis_tools |  |  | 9 |  |  |  |  | 9 |
| analysis_tools/racecheck | 4 | 173 | 32 |  |  |  |  | 209 |
| analysis_tools/shared | 11 | 3 | 11 |  |  |  |  | 25 |
| analysis_tools/synccheck | 8 | 69 | 45 |  |  |  |  | 122 |
| conformance |  |  |  |  |  |  | 2 | 2 |
| numsim/abi |  |  | 3 |  |  |  |  | 3 |
| numsim/corpus | 29 |  |  |  |  | 23 |  | 52 |
| numsim/integration | 338 | 2 | 127 |  | 40 | 3 |  | 510 |
| numsim/microtests |  |  | 1 | 55 |  |  |  | 56 |
| numsim/registry | 26 |  | 69 |  | 4 |  |  | 99 |
| numsim/runtime | 628 | 35 | 42 |  | 18 | 5 |  | 728 |
| numsim/v2 |  |  |  |  |  |  | 39 | 39 |
| **total** | **1044** | **282** | **339** | **55** | **62** | **44** | **41** | **1867** |

| A target | public facade only | imports legacy internals |
| --- | ---: | ---: |
| conformance | 19 | 22 |
| v2-api | 0 | 23 |
| v2-kernel-case | 450 | 530 |

| category | target / status | tests |
| --- | --- | ---: |
| A | conformance | 41 |
| A | v2-api | 23 |
| A | v2-kernel-case | 980 |
| B | covered | 85 |
| B | gap_unportable | 43 |
| B | ported | 104 |
| B | unreviewed | 50 |
| C | delete | 339 |
| D | D-live | 48 |
| D | D-recorded | 7 |
| E | delete | 62 |
| F | keep | 44 |
| N | keep | 41 |

cvt goldens: 133 recorded forms in Python, 133 generated numsim-oplib tests

**E (tile forms TVM rejects).** The list was regenerated from scratch rather
than copied from W1:

1. A capture run of `tests/numsim` and `tests/analysis_tools`
   (`capture_plugin.py`, `-m 'not numsim_gpu'`) saved 2,475 kernels.
2. `tile_rejections.py` lowered each one.

71 kernels contain a rejected tile op, which matches W1's §D.2 count. They map
to 89 parametrized node ids in 69 test functions. W1's "105" in §D.3 counts
kernel × reason pairs, not tests.

Caveat: at capture time, 347 tcgen kernels raised
`TypeError: TcgenMma … ti16`, because contract batch 3 was in flight in the
lowering. Their tile ops were not inspected. Rerun `tile_rejections.py` once
the lowering is consistent again.

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

Every test in `analysis_tools/racecheck` (209) and `analysis_tools/synccheck`
(122) was read and mapped by hand. The maps are
`scripts/numsim-v2/coverage/racecheck.tsv` and `synccheck.tsv`. Each has one
row per legacy function, giving its status, the Rust tests that cover it, and
a note. Parametrized tests count as covered only when every semantically
distinct parameter is covered; otherwise the uncovered parameters are listed
in the note.

| map | covered (existing Rust) | ported (this batch) | gap_unportable | not B (A / C / F) |
| --- | ---: | ---: | ---: | ---: |
| racecheck (209) | 74 | 54 | 35 | 4 / 31 / 11 |
| synccheck (122) | 11 | 50 | 8 | 2 / 27 / 24 |

The classifier refines the maps' "not B / F" rows. Legacy-executor and
frontend plumbing becomes C. Op-support runs become A. The synccheck verdicts
asserted from racecheck files stay B `unreviewed`.

**First conversion batch.** Both new files are in
`numsim-core/tests/`, and both are green in the main tree. Every test is
doc-commented with the legacy `file.py::test[param]` it ports.

- `racecheck_legacy_ports.rs`: 56 tests for 54 legacy rows; 55 pass and
  1 is `#[ignore]`d.
- `synccheck_legacy_ports.rs`: 47 tests for 50 legacy rows; 44 pass and
  3 are `#[ignore]`d.

When a legacy expectation contradicts a delta row, the test asserts the new
behaviour and cites the row. Rows cited:

- **Racecheck:** R1, R2/R3, R4, P1, P5, P6, B1/V5, B2, X1, X2/S2, X3, X9, W2,
  I2, V3.
- **Sync:** B2, B3, A3, M2.

### The real gaps (no Rust equivalent yet)

**1. Undocumented divergences, pinned as `#[ignore]` tests (4).** In each
case the new core disagrees with the legacy test, and no delta row explains
the difference. Each needs a ruling. Then either fix the core or add a delta
row and flip the test.

| test | legacy | new | spec says |
| --- | --- | --- | --- |
| `racecheck_legacy_ports::g2_publication_too_wide_to_poll_fails_closed` (`test_declared_word_regressions.py::test_publication_too_wide_to_poll_fails_closed`) | 16-byte `st.release.v4` over a 4-byte declared word, then waited on: incomplete | clean when the verdict bit accepts the write | racecheck-semantics §0.2 and §5: wide writes are incomplete (agrees with legacy) |
| `synccheck_legacy_ports::launch_bounds_register_redistribution_is_clean` | clean | `setmaxnreg_invalid_direction`. The scheduler applies the launch-bounds `Configure{regs_per_thread}` straight to the live `SyncTable` (`sched/mod.rs`) without logging a `SyncEvent`, and `ResourceInit` has no field for it, so synccheck replays from 168 registers. | sync-semantics §7 defines `Configure` but not how synccheck learns it. This is a **false positive** on every launch-bounds kernel that uses setmaxnreg. |
| `synccheck_legacy_ports::order_dependent_tcgen_allocations_with_cta_syncs_are_an_error` | error: "allocation result changed" | clean. The explorer replays protocol steps only, and the swapped allocation order succeeds. | synccheck-explorer §1.1 and §4 still require fixed allocation results |
| `synccheck_legacy_ports::unconfigured_setmaxnreg_budget_is_setmaxnreg_pool_deadlock` | kind `setmaxnreg_pool_deadlock` | kind `deadlock` (the verdict half is ported and passes) | none |

**2. `gap_unportable`: B contracts that contract events cannot express
(43).** Each needs a v2 kernel-level test, through lowering, the interpreter,
or both, before its legacy test can go (wave 2).

| group | n | legacy tests | what is missing |
| --- | ---: | --- | --- |
| Exact OOB / guard evaluation | 13 | `racecheck/test_native_exact_oob.py` (7), `test_native_racecheck_exact_control.py` (3), `test_native_racecheck_artifact.py::…ignores_fully_predicated_shared_pointer_load` | The interpreter decides which lane goes out of bounds, and a fully predicated-off access must emit no access. Contract-level OOB is covered by `racecheck_async_copy::reuse_and_oob`. |
| Host-input aliasing and pointer provenance | 7 | `racecheck/test_native_global_write_seed.py` | Two parameters bound to overlapping host arrays must be one allocation. This covers raw pointer arithmetic, `selp` and loop-carried pointers, TensorMap base replacement, and `discard` as a write. |
| Logical-buffer identity (`alias_stale_read`) | 7 | `racecheck/test_native_alias_advisory.py` | The contract `Access` has no logical-buffer identity, and the new core has no such advisory. There is no delta row for the drop. |
| Hidden accesses of tile and helper primitives | 6 | `racecheck/test_native_racecheck_artifact.py` (`cta_sum` scratch and internal barriers, `permute_layout` zero-fill, index load hidden in a tile region) | Lowering must emit these accesses. Note: `permute_layout` is also on the E list. |
| Warp-collective participation (`shfl.sync`) | 4 | `synccheck/test_native_kernel_contracts.py` (2), `test_native_synccheck_exact_control.py` (2) | `warp_collective_divergence` is an interpreter `ExecError`, not a sync event. |
| Remote-address arrive / expect_tx | 2 | `synccheck/test_native_synccheck_artifact.py` | `mbarrier_local_arrive_remote_address` is decided at address resolution. |
| Other engine-level | 4 | `racecheck/…subset_is_typed_incomplete`, `…ignore_oob_does_not_bounds_check_the_ignored_bytes`, `test_native_kernel_contracts.py::…conditional_tmem_lifecycles`, `test_signal_diagnostics.py::…without_any_possible_publisher_is_a_sync_deadlock` | Subset runs, `.ignore_oob` read clipping, conditional TMEM lifecycle lowering, and scheduler deadlock. |
| Divergence and liveness | 2 | `synccheck/test_native_break_continue.py::…lane_divergent_break…`, `…cross_warp_tmem_quiescence_before_dealloc` (unordered half) | Whether a lane that broke out of a loop counts as live at a later barrier; TMEM access after a cross-warp dealloc. |

**3. `unreviewed` B (50).** These are checker-verdict tests outside the two
reviewed directories. Nobody has checked them against the Rust scenarios yet;
they are the next batch to review.

- 7 are synccheck-only assertions in `racecheck/test_native_raw_async_copy_footprints.py`.
- 4 are elsewhere in `analysis_tools/racecheck` and `shared`, including 3 in
  `shared/test_native_dense_cta2_mma_ordering.py`.
- 39 are in `numsim/runtime` and `numsim/integration`. The largest files are
  `test_wait_until.py` (5), `test_mbarrier_lane_semantics.py` (4), and
  `test_non_tensor_bulk_forms.py` (3).

**Gap count.** 4 + 43 = **47 known B gaps**, plus up to 50 in the unreviewed
rows. This batch ported 104 legacy rows (54 racecheck, 50 synccheck) into
103 new Rust tests.

**Found in passing, outside the checkers.**

- **Missing `WarpSync` on `tcgen05.alloc`.** racecheck-semantics §3 row 3
  says `tcgen05.alloc.sync.aligned` emits `WarpSync{mask}`.
  `interp/handlers/tcgen.rs` emits only a lane-0 write plus a Protocol event.
  An end-to-end run of `alias_advisory.py::…tcgen_alloc_result_is_visible_to_every_lane`
  would therefore report a lane-order race where legacy is clean.
  `g4_tcgen_alloc_result_visible_to_every_lane` pins both sides.
- **Remote-view cluster arrive flips verdict (B1/V5).**
  `support/remote_mbarrier.py::mapped_remote_mbarrier_cluster_view` is a raw
  qualifier-less `mbarrier.arrive.shared::cluster` on a peer CTA's barrier.
  Legacy reports it clean. The new core, using the default `.cta`
  arrive/wait scope, reports `ScopeMismatch`. The result depends on which
  scopes lowering assigns. `g1_remote_view_cluster_arrive` pins both.

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
| 1 | `cargo test -p numsim-core --test 'racecheck_*' --test 'synccheck_*'` green, with `*_legacy_ports.rs` | `analysis_tools/{racecheck,synccheck}` rows with status `covered` or `ported` | the Rust scenario files (already in place) |
| 2 | `gap_unportable` rows rewritten as v2 kernel cases (they need the interpreter, lowering or source anchors) | the rest of B in those directories; B `unreviewed` rows once reviewed like wave 1 | v2 kernel cases |
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
5. **Wide release writes to a declared word.** The spec says incomplete; the
   core accepts the write (the ignored racecheck test).
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
12. **Fixed tcgen05 allocation results across schedules** ("allocation result
    changed"). The explorer spec still describes this, but the core dropped it.
13. **Launch-bounds register budget (`regs_per_thread`) reaching synccheck.**
    Neither the log nor `ResourceInit` carries it.
14. **Kind names.** `setmaxnreg_pool_deadlock` is now `deadlock`;
    `synchronization_contract_mismatch` (an execution_error) is now
    `named_barrier_contract_mismatch`.
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
