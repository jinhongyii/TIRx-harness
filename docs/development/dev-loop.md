---
orphan: true
---

# NumSim development loop

Day-to-day commands for the NumSim / Racecheck / Synccheck redesign
(see `numsim-redesign.md`). Everything runs from `tirx_harness/` unless noted.

## Environment

Interactive shells on the development hosts export `PYTHONPATH`, `TVM_HOME`,
`TVM_LIBRARY_PATH` and `LD_LIBRARY_PATH` pointing at a local TVM 0.26 tree.
That TVM shadows the pinned `apache-tvm` wheel and the frontend panics with
`sym.Analyzer is not registered`. Always start with:

```bash
source scripts/dev-env.sh   # repo root; clears the TVM variables, sets $PY,
                            # NUMSIM_CACHE_DIR and NUMSIM_WORKER_AFFINITY=off
```

and use `$PY` (the project `.venv`, Python 3.12) for every Python command.

Create or refresh the environment with

```bash
git submodule update --init thirdparty/tvm-rust-ext
uv sync --locked --extra test --group benchmark --inexact
```

The `benchmark` group matters: canonical kernels from `tirx-kernels` import
`torch` at module scope, so without it the corpus suites (and the conformance
tests) fail at collection with `ModuleNotFoundError: torch`. `--inexact`
keeps packages other workers may have added.

Other pitfalls:

- Always pass `-n` to pytest, even for one test; set
  `NUMSIM_WORKER_AFFINITY=off` (dev-env.sh does) or engine workers pile onto
  the cores idle at launch time.
- Artifacts are cached under `$NUMSIM_CACHE_DIR`. dev-env.sh points it at a
  dedicated directory so refactor work does not reuse artifacts built by an
  older engine checkout; delete it to force rebuilds.
- Engine changes: `(cd src/tirx_harness/numsim/engine-rs && cargo test --all-features)`.
  New core: `(cd src/tirx_harness/numsim/core-rs && cargo test)`.

## NumSim v2 (core-rs + Python layer)

The new engine is `tirx_harness/src/tirx_harness/numsim/core-rs/` (Rust
workspace) driven from `tirx_harness.numsim.v2` through the pyo3 extension
`numsim_core_py` (crate `core-rs/numsim-py`).

```bash
# Rust: whole workspace, and the binding crate alone
(cd src/tirx_harness/numsim/core-rs && cargo test --workspace)
(cd src/tirx_harness/numsim/core-rs && cargo test -p numsim-py)

# Build the extension (release; --debug for a debug build). Rerun after any
# core-rs change you want Python to see. Concurrent work: private target and
# package, never the shared .so (rebuilding it crashes runs that loaded it).
CARGO_TARGET_DIR=<dir>/target bash src/tirx_harness/numsim/core-rs/numsim-py/build_dev.sh --out <dir>/ext
$PY -m pytest -q -n 8 -o "pythonpath=<dir>/ext/pkg ." tests/numsim/v2
# Single developer: install into the source tree, then run pytest as usual.
bash src/tirx_harness/numsim/core-rs/numsim-py/build_dev.sh
$PY -m pytest -q -n 8 tests/numsim/v2
```

The extension lands at `src/tirx_harness/numsim/v2/numsim_core_py.abi3.so`
(git-ignored). Without it the v2 tests skip and `NUMSIM_IMPL=v2` conformance
runs skip with "numsim_core_py is not built". The hand-built Module fixture
`tests/numsim/v2/fixtures/vector_add.module.json` is generated from
`ProgramBuilder`; refresh it after a `program.rs` change with
`UPDATE_FIXTURES=1 cargo test -p numsim-py`.

The release profile (`core-rs/Cargo.toml`) uses `lto = "thin"`. Criterion benches
override it back (`[profile.bench] lto = false`) so recorded bench baselines stay
comparable. Measured on 2026-10-08 (256-core host, load about 20-45, a HEAD snapshot, `numsim-py`
release build with a private `CARGO_TARGET_DIR`):

| profile | clean build | incremental (touch numsim-oplib) | incremental (touch numsim-core) |
| --- | --- | --- | --- |
| no LTO (before) | 38.2 s | 35.3 s | 33.1 s |
| `lto = "thin"` (adopted) | 38.5 s | 33.4 s | 31.9 s |
| `lto = "thin"`, `codegen-units = 1` (rejected) | 116.9 s | 100.6 s | 101.7 s |

End to end with a private `--out` extension (1 worker, the interpreter, min of 3 interleaved runs):

| kernel | no LTO | thin | thin + 1 CGU |
| --- | --- | --- | --- |
| `rmsnorm` | 1.01 ms | 0.88 ms | 0.88 ms |
| `fp16_bf16_gemm` | 62.1 ms | 59.5 ms | 60.1 ms |
| `deepgemm_sm100_fp8_gemm_1d1d` | 6.06 ms | 6.23 ms | 5.92 ms |

Thin LTO costs nothing to build. It gains about 13% on the ALU-bound kernel and is within noise
on the MMA-bound ones. One codegen unit triples the incremental build for no further
gain, so it is not used.

`numsim-core/tests/synccheck_equivalence.rs` explores only every 10th
generated case by default (5 s debug). Set `SYNCCHECK_EQUIV_FULL=1` to check
every case (45 s debug, 13 s release) after changing a synccheck reduction rule.

The `NUMSIM_V2_*` prefix is temporary: these variables become `NUMSIM_*`
when the legacy engine is deleted (redesign step 5).

v2 reads its environment in one place, `v2/options.py`:
`NUMSIM_CACHE_DIR` (modules are cached under `<root>/v2-modules/`, keyed by
TIRx source, format version and the lowering sources), `NUMSIM_V2_SEED`,
`NUMSIM_V2_NO_CACHE=1`. There is one executor, the interpreter; the codegen
backend was measured and deleted (`backend-comparison.md`).

Python surface (`from tirx_harness.numsim import v2`): `transpile`,
`Engine(max_workers=...)` with `.run`, `.run_racecheck_phase`,
`.run_synccheck_phase`, `compare`, `racecheck`, `synccheck`,
`CoverageBounds`, `ResourceLimits`, report types with
`.verdict/.findings/.to_dict/.print/.require_clean`, and
`v2.report.payload_json_schema()` (payload `schema_version` 5). A TIRx form
the lowering does not support raises `UnsupportedTIRxError` at transpile, and
an engine stop that cannot be modelled is an `incomplete` diagnostic; neither
counts as a pass.

### v2 binder rules (`v2/run.py::canonicalize_inputs`)

- Inputs bind by canonical name, local name, alias, or legacy kernel-qualified
  `k<i>:<name>`; the same object bound under several names is one binding.
- `ParamKind::ImplicitShape { buffer, axis }` slots are filled from the bound
  array's shape (never by name); static `Const` dims of a buffer slot must
  match the array (a byte-identical reinterpretation, e.g. fp8 data in a u8
  slot, is accepted).
- A `numsim.cases.TensorMap` image binds as `ArgValue::TensorMapOf` over its
  base array (bound as a buffer, or reusing the parameter bound to that very
  array); selecting a tensor map as an output returns its base array.
  Tensor-map slots with an engine spec and no value are encoded by the engine.
- **Host-input aliasing**: buffer arguments bound to overlapping host memory
  (one array bound twice, or overlapping views) share ONE engine allocation:
  each connected group of overlapping C-contiguous spans becomes a hidden
  region buffer `__host_region_<i>` (the union of the bytes) and every member
  an `ArgValue::View` of it at its byte offset (CONTRACT_REQUESTS W8-6), so
  writes through one name are visible through the other and Racecheck sees
  the aliasing. A non-contiguous view in an aliasing group raises
  `NotImplementedError`. Host tensor-map bases take part in the grouping.
- Outputs come back in the layout of the array the caller bound under the
  selector name (two kernels may bind one buffer name with different
  dtypes); an output selected through a host tensor map is returned as the
  map's logical tensor (dims outermost first, map strides, base dtype).
- Sub-byte buffers (E2M1, U4/S4, FP6) must be bound packed as a contiguous
  uint8 array; a one-value-per-byte `ml_dtypes` array (e.g.
  `float4_e2m1fn`) raises `InputError`, as legacy did.
- A tensor-map parameter with neither a host value nor an engine-encodable
  spec is bound to an all-zero image, which the descriptor decoder rejects:
  legacy kernels that never use it run, any use fails closed.
- `Engine(max_workers=8)` (legacy default; `"auto"` = CPU count) sets the
  scheduler's `RunConfig.workers`; results do not depend on it.
- Public `racecheck()`/`synccheck()` return an `incomplete` report
  (`missing_input_bindings` with the missing names, or
  `native_frontend_unsupported`) instead of raising, as legacy did;
  `Engine.run` and the `run_*_phase` methods still raise `InputError` /
  `UnsupportedTIRxError`. `require_clean()` raises `CheckFailed`.
- Execution subsets: `ExecutionSubset` (exported by `numsim.v2`) maps to
  `RunConfig::subset`. `cluster_ids` are linear cluster ids; `cta_ids` must
  cover whole clusters of a static grid (`InputError` otherwise); both given
  means their intersection. A `{phase: ExecutionSubset}` mapping must name
  every launch with the same subset (numsim H6), and `Engine.run` on a
  multi-kernel module needs that form. A subset run is at least `incomplete`
  (`analysis_scope` kind `subset`). Host `ExecutionAssumptions` are accepted
  and ignored (dropped in v2; no corpus case sets one).

### Timing

`NumSimResult.timing` and each checker phase payload's `timing` hold
wall-clock milliseconds: `lower` (transpile, or the module-cache load when
`CompiledModule.cache_hit`), `bind` (Python input canonicalization), `run`
(`sched::run_with_config`,
including arena binding), `check` (checker finish or offline exploration plus
serialization; 0 for NumSim) and `report` (Python result construction). One
engine run serves all phases of a module, so phase payloads repeat its
`run`/`check`.

### v2 report decisions (`v2/report.py`)

- Checker payloads come from the checkers' own `serialize` (one per launch);
  numsim-py adds `phase`, `analysis_scope`, source spans for every `site`,
  the launch's runtime diagnostics, and `stats`.
- Synccheck `review` items (exit lints `sync_exit_lint`, e.g. `DanglingAtExit`
  for a dangling `bar.arrive`, `UncommittedAtExit`; sync-behaviour-deltas
  B5/A3) are reported under `advisories`, whether the checker emits them in
  its `review` slot or as `review`-status findings; the verdict stays
  `review`. This keeps the legacy shape (`findings` holds errors only)
  without hiding the lint.
- When a launch did not run to completion (a run-status diagnostic,
  `source: "run_status"`), the engine's error/incomplete is the phase's
  `execution_error`/`incomplete`; the checker's verdict from the truncated
  event log (including its own `truncated_launch` incomplete) is void and kept
  only under `checker_on_truncated_log`. Other engine incompletes (subset
  execution, the W2-14 stream-cycle diagnostic) keep the checker's findings
  and make the verdict at least `incomplete`.
- Uninitialized reads use `ValidityPolicy::ZeroAndReport` in every mode and
  surface as `uninitialized_read`/`review`.

## Conformance snapshots

`tests/conformance/` replays every canonical corpus case
(`tests/numsim/corpus/canonical_cases.py`) in three modes and compares the
normalized result with `tests/conformance/snapshots/<case>/<mode>.json`. These
snapshots are the oracle for the migration; see
`tirx_harness/tests/conformance/README.md` for what they contain.

```bash
# legacy engine (default)
$PY -m pytest -q -n 32 --dist=worksteal tests/conformance
# new engine; cases skip while numsim-core bodies are unimplemented
NUMSIM_IMPL=v2 $PY -m pytest -q -n 32 --dist=worksteal tests/conformance
# one kernel, one mode
$PY -m pytest -q -n 2 tests/conformance -k "rmsnorm and racecheck"
```

A mismatch prints a unified diff of the normalized snapshot. For non-legacy
implementations, diagnostics legacy recorded without any source anchor are
compared without anchors (`snapshot.relax_unanchored`); kinds, statuses and
byte footprints are still exact. The per-case v2 status lives in
`docs/development/v2-conformance-status.md`.

### Updating snapshots

Only regenerate from the legacy engine, and only for an intentional change of
legacy behavior (or a corpus change such as a new canonical case):

```bash
NUMSIM_IMPL=legacy $PY -m pytest -q -n 32 --dist=worksteal tests/conformance --update-snapshots
git diff --stat tests/conformance/snapshots
```

Review the diff and explain each changed case in the commit message, citing
the delta row id that justifies it (CI: `check_snapshot_deltas.py`). Never
update snapshots from `NUMSIM_IMPL=v2` to make the new engine pass while the
legacy engine exists; a ruling that legacy was wrong goes in a
`<mode>.delta.json`. After the legacy engine is deleted, v2 is the oracle and
`--update-snapshots` regenerates from it (the deletion commit folds the delta
files with `scripts/numsim-v2/fold_snapshot_deltas.py`); see the conformance
README, "Snapshot policy after the legacy engine is deleted".

## CI

`.github/workflows/tests.yml` runs `cargo test --all-features` in engine-rs,
`cargo test` in core-rs (when present) and
`pytest -n auto -m "not numsim_gpu and not performance"`. The conformance tests
are part of that run, so snapshot drift fails CI.

## Performance

Performance tests carry the `performance` marker and are excluded from CI.
Before measuring, follow the preflight in `tirx_harness/tests/CLAUDE.md`:

```bash
nproc; uptime; vmstat 1 3     # require a near-idle host
$PY -m pytest -q -n 16 --dist=worksteal -m performance tests
```

New relative checks use `tests/perf/perf_baseline.py` with per-host-class
files in `tests/perf/baselines/` (see the README there). The two legacy
absolute-threshold tests stay unchanged until the legacy engine is deleted.

### v2 performance gate

`tests/perf/test_corpus_perf.py` (marker `performance`, interp backend) times the Mega
MoE workloads of the legacy gates and six canonical checker cases. Each canonical case
runs at 1 and 32 engine workers: synccheck records `run` and `check`, and racecheck
records `run`, because it checks inside the run. `scripts/numsim-v2/perf_gate.py`
runs the module with `NUMSIM_PERF_ENFORCE=0` and compares each metric with
`tests/perf/baselines/v2/<host-class>.json`. The limit is `max(1.5 x baseline,
baseline + 0.1 s)`. The gate fails on a regression only when the host is near idle
(1-minute load average at most 10% of the CPU count, sampled every 15 s through the
run). Under load it only warns, and it always prints the load. A correctness failure
always fails the gate.

```bash
$PY scripts/numsim-v2/perf_gate.py --run -n 1                 # check
$PY scripts/numsim-v2/perf_gate.py --run -n 1 --record        # (re)record, near idle only
$PY scripts/numsim-v2/perf_gate.py --run -n 1 -k checker_phase # one group
```

The first baselines were recorded on 2026-10-08 at 8b204f4 plus the working tree, on
the 256-CPU EPYC 7763 host. The host was never quiet that day: the load average ran
20.9-81.8 during the run, and the file says so in `preflight`. They were recorded with
`--record --force`; re-record them on a quiet host before tightening anything.

| metric (seconds) | w1 | w32 |
| --- | --- | --- |
| synccheck check, `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 0.20 | 0.21 |
| synccheck check, `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 0.25 | 0.26 |
| synccheck check, `flash_attention4` | 0.22 | 0.23 |
| synccheck run, the same three | 0.31 / 0.20 / 1.70 | 0.28 / 0.12 / 0.77 |
| racecheck run, `recurrent_kda_decode_one_warp` | 1.89 | 1.41 |
| racecheck run, `gdn_decode_bf16_wide_vec_mtp` | 1.19 | 1.02 |
| racecheck run, `selective_state_update_stp_simple` | 0.02 | 0.02 |

The Mega MoE workloads ran with 16 engine workers. For each one, the table gives the v2
time first, then the legacy baseline from the same host class.

| workload | v2 interp | legacy |
| --- | --- | --- |
| numsim max config (`t8192_m8192_h7168_i3072_e384_k6_g1`), `Engine.run` | 402.0 s | 216.4 s |
| racecheck `two_tokens` | 4.43 s | 1.94 s |
| racecheck `sixteen_tokens` | 4.91 s | 2.00 s |
| racecheck `twenty_four_experts` | 21.92 s | 2.75 s |
| racecheck `shared_expert` | 5.46 s | 2.04 s |
| racecheck `medium_moe`, `large_moe` | known slow: more than 42 min, serial checker | 22.3 s, 38.0 s |

The v2 Mega MoE racecheck verdict is `error`, not legacy's `review`: the findings are
racecheck B1/R4, B7 and T19, and the advisories X4 and P7. The gate asserts these
kinds. `medium_moe` and `large_moe` are skipped unless `NUMSIM_PERF_SLOW=1`, because
the racecheck checker replays serially on the calling thread (racecheck-semantics.md,
"Merge design and the serial-checker limit"; W5 is working on the actor space).
This gap is tracked: the fix is the parallel checker planned in
`docs/development/racecheck-parallel-design.md` (W5), and both workloads get baselines
once it lands.

Follow-up: re-record every v2 baseline on a quiet host
(`perf_gate.py --run -n 1 --record`, without `--force`) and then review each limit.

### Legacy vs v2 corpus comparison

`scripts/numsim-v2/bench_backends.py` times every canonical case in each mode where v2
matches legacy (`v2-conformance-status.md`). It runs legacy and v2 interp at
`max_workers` 1/8/32, 3 interleaved repetitions each, keeping the minimum. Every run's
normalized result is checked against the legacy conformance snapshot, and a row aborts on
any difference. Each case runs in its own child process. The 1-minute load average is
checked before each (case, mode) and the run waits while it exceeds `--max-load` (default
40); samples taken above 40 are flagged `(L)`. Results are written per case under
`$NUMSIM_CACHE_DIR/bench-backends/` and a rerun skips cases already present (`--force` redoes
them). On a near-idle host, from `tirx_harness/`:

```bash
nproc; uptime; vmstat 1 3                     # preflight (tests/CLAUDE.md)
$PY ../scripts/numsim-v2/bench_backends.py run --variants legacy,interp
$PY ../scripts/numsim-v2/bench_backends.py mega            # Mega-MoE perf workloads, 900 s cap per run
$PY ../scripts/numsim-v2/bench_backends.py render          # docs/development/backend-comparison.{md,json}
$PY ../scripts/numsim-v2/bench_backends.py regressions --engine "$(git rev-parse --short HEAD)"
                                              # scripts/numsim-v2/coverage/perf_regressions.tsv
```

A full corpus sweep takes about 4-6 h. To time a fixed engine while the tree keeps changing,
copy `numsim/v2/` (with `numsim/dtype_registry.json` beside it) and `core-rs/`, build the
extension from the copy, and pass `--v2-package <copy>/numsim_v2_snap`. Narrow a rerun with
`--cases REGEX`, `--modes` and `--workers`.
