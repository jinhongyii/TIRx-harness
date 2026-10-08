# Migration tooling: what stays after the legacy engine is deleted

One line per file. **keep** = permanent dev tooling; **delete** = removed in
step 5 (redesign §4, commit 79f04eb) together with the legacy engine
(`retire_legacy.py --apply` section 6; the deleted files are in the parent
commit 7c7d049). Every former "decide"
line was ruled by the coordinator on 2026-10-08. Paths are relative to the
repository root.

## `scripts/numsim-v2/`

- `bench_backends.py`: deleted at step 5 (its legacy-vs-v2 and codegen modes were dead; it used `relax_unanchored` and the delta snapshots). `perf_gate.py` and `tests/perf/` keep the interp measurement.
- `capture_plugin.py`: keep (ruled 2026-10-08). It hooks `v2.transpile`. Until step 5 it also captures kernels reached only through legacy internals at the legacy normalization hook; that block disables itself once the legacy modules are gone. The file format is unchanged.
- `check_snapshot_deltas.py`: keep (CI: every snapshot change cites a delta row).
- `classify_tests.py`: delete.
- `fold_snapshot_deltas.py`: keep. It runs in the deletion commit, and afterwards whenever a ruled delta is folded into a base snapshot.
- `inventory.py`: keep (ruled). Generates the IR inventory in `lowering-inventory.md` from a capture directory.
- `lower_sweep.py`: keep (ruled). Corpus lowering regression sweep over a `capture_plugin.py` capture.
- `lowering_ops.py`: keep (input to the oplib `SUPPORTED_OPS.md` generator).
- `make_contract_shim.py`: delete at step 5 (ruled). The Rust `validate` API stays; this shim and its shell wrapper go.
- `perf_gate.py`: keep.
- `record_race_fixtures.py`: moved (W5) to `tirx_harness/src/tirx_harness/numsim/core-rs/numsim-core/examples/record_race_fixtures.py`, beside the racecheck tuning harness that calls it. Kept there; `numsim-race-core` is deleted.
- `retire_legacy.py`: delete.
- `retire_tests.py`: delete.
- `status.py`: keep (one-command conformance status). Its legacy-comparison columns go with the legacy engine.
- `step5_status.py`: delete.
- `tile_rejections.py`: delete (needs a legacy capture run).
- `v2_public_status.py`: delete.
- `validate.sh`: delete at step 5 (ruled; pairs with `make_contract_shim.py`).
- `walk.py`: keep (ruled; helper of `inventory.py`).

## `scripts/numsim-v2/coverage/` (the retirement ledger)

- `test_classification.csv`: delete.
- `category_overrides.tsv`: delete.
- `v2_ports_*.tsv` (all: `deltas`, `failclosed`, `internals`, `messages`, `p6b`, `p6c`, `p6d`, `p7`, `p8`, `p9`, `racedeltas`, `reductions`, `stats`, `triage`, `validshape`, `w1triage`, `w4`, `w6`, `w8`, `w9`, `w11`, `w12`): delete.
- `v2_public_status.tsv`: delete.
- `v2_xfail_inventory.tsv`: delete. The open gaps live as `v2_gap` marks and in `CONTRACT_REQUESTS.md`.
- `step5_a_status.tsv`: delete.
- `other_assertion_triage.tsv`: delete.
- `other_b.tsv`, `racecheck.tsv`, `synccheck.tsv`: delete (B-row coverage maps for waves 1/2).
- `racecheck_suite_triage.tsv`: delete.
- `tile_dispatch_rejections.txt`: delete (wave-0 list).

## `tirx_harness/src/tirx_harness/numsim/core-rs/tools/`

- `port_sync.py`: delete. It is a one-shot generator that copied the `numsim-sync-ref` types into `numsim-core/src/sync/`, and the copies are now maintained in place.
- `validate-program/`: delete at step 5 (ruled; the shim-based validator binary, not a workspace member).

## Elsewhere

- `scripts/dev-env.sh`: keep. Every path is an overridable default (`PY`, `NUMSIM_CACHE_DIR`, `NUMSIM_WORKER_AFFINITY`; `NUMSIM_LD_LIBRARY_PATH` overrides the library path). `LD_LIBRARY_PATH` is derived from the venv's `nvidia/*/lib`, `torch/lib` and `tvm/lib*`.
- `tirx_harness/tests/conformance/test_conformance.py`, `README.md`, `snapshots/`: keep.
- `tirx_harness/tests/conformance/snapshot.py`: keep the file; delete its legacy-comparison code at step 5 (ruled). That is `legacy_available`, `selected_impl_name`, `oracle_impl_name`, `IMPL_ENV`/`IMPLS`, and the legacy branch of `load_implementation`; `relax_unanchored`; the `<mode>.delta.json` lookup in `load_expected` (after the fold); and the legacy-spelling space and offset normalizations in `record_space`/`normalize_records` (the "legacy spelled differently" and "not comparable to legacy" branches).

## CI

`.github/workflows/tests.post-deletion.yml` references only `check_snapshot_deltas.py` (keep) from this list. It does not reference any file in the delete set.
