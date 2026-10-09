# NumSim Tests

These source-tree-only tests are not packaged in the `tirx-harness` wheel. The
engine is v2 (`tirx_harness.numsim`); the legacy engine and its tests were
deleted at step 5 (`docs/development/test-migration.md` has the history).

## Structure

- `v2/`: lowering tests that assert the contents of the lowered `Program`
  (`test_lowering_*.py`) and the Python layer over a hand-built module
  (`test_v2_layer.py`, `fixtures/`), plus
  - `checkers/`: kernel-level Racecheck/Synccheck facts that contract events
    cannot express;
  - `ports/`: v2 copies of legacy public-API tests, each docstring citing the
    legacy test and any behaviour-delta row;
  - `tile_forms/`: tile-op kernels that TVM's dispatch rejects, with their
    `captures/`.
- `runtime/`: focused public-API numerical, bit-level, layout, lane and
  synchronization-state oracles.
- `integration/`: public-API end-to-end behaviour (`transpile`, `Engine().run`,
  reports, mismatch diagnostics, multi-kernel and host-prelude handling).
- `microtests/`: the same small PrimFunc and inputs run on NumSim and a live
  GPU (`numsim_gpu` marker; kernels in `cases/`, runner in `harness.py`).
- `corpus/`: canonical complete kernels (`kernels/`, `canonical_cases.py`) and
  their tests; runnable cases compare GPU, NumSim and an independent reference
  pairwise.
- `support/`: reusable kernels, input domains and helpers. Production NumSim
  code must not import it.
- `conftest.py`: shared fixtures and the GPU opt-out flag.

Corpus behaviour per mode is checked separately by `tests/conformance/`
against the snapshots; Rust scenarios live in `core-rs/numsim-core/tests/`.

## Where a Test Goes

Put each test at the lowest layer that can express it: a hand-built `Program`
or contract-event scenario in Rust, then `v2/` lowering tests, then a public-API
kernel here, then the corpus. Do not pin internals: assert outputs, physical
bits, verdicts and findings, never generated text, scheduler counts, internal
payload fields or absolute times. Every intended behaviour change cites a row
of `docs/development/{numsim,sync,racecheck}-behaviour-deltas.md`.

When hardware behaviour is uncertain, add the smallest paired NumSim/GPU
microtest that varies one factor and asserts concrete values; keep it as
regression evidence. For runnable corpus cases assert GPU == NumSim,
GPU == reference and NumSim == reference.

## Running

```bash
python -m pytest -q -n 16 --dist=worksteal tests/numsim
```

Always pass `-n`. GPU microtests run by default; `--no-run-numsim-gpu` (only
when targeting this directory) or `-m "not numsim_gpu"` skips them. To see a
core-rs change while others may be running tests, build privately
(`CARGO_TARGET_DIR=<dir>/target bash src/tirx_harness/numsim/core-rs/numsim-py/build_dev.sh --out <dir>/ext`)
and run pytest with `-o "pythonpath=<dir>/ext/pkg ."`; never rebuild the shared
extension under concurrent runs.
