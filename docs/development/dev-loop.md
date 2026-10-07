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

## Conformance snapshots

`tests/conformance/` replays every canonical corpus case
(`tests/numsim/corpus/canonical_cases.py`) in three modes and compares the
normalized result with `tests/conformance/snapshots/<case>/<mode>.json`. These
snapshots are the oracle for the migration; see
`tirx_harness/tests/conformance/README.md` for what they contain.

```bash
# legacy engine (default)
$PY -m pytest -q -n 32 --dist=worksteal tests/conformance
# new engine; skips cleanly while tirx_harness.numsim.v2 does not exist
NUMSIM_IMPL=v2 $PY -m pytest -q -n 32 --dist=worksteal tests/conformance
# one kernel, one mode
$PY -m pytest -q -n 2 tests/conformance -k "rmsnorm and racecheck"
```

A mismatch prints a unified diff of the normalized snapshot.

### Updating snapshots

Only regenerate from the legacy engine, and only for an intentional change of
legacy behavior (or a corpus change such as a new canonical case):

```bash
NUMSIM_IMPL=legacy $PY -m pytest -q -n 32 --dist=worksteal tests/conformance --update-snapshots
git diff --stat tests/conformance/snapshots
```

Review the diff and explain each changed case in the commit message. Never
update snapshots from `NUMSIM_IMPL=v2` to make the new engine pass.

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
