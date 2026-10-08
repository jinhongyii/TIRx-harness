# Migration tooling: what stays after the legacy engine is deleted

One line per file. **keep** = permanent dev tooling; **delete** = removed in
step 5 (redesign §4) together with the legacy engine
(`retire_legacy.py --apply` lists it in section 6); **decide** = needs a ruling
before step 5. Paths are relative to the repository root.

## `scripts/numsim-v2/`

- `bench_backends.py`: keep, renamed `bench_engine.py`. Drop the legacy and codegen variants and keep the interp measurement.
- `capture_plugin.py`: decide. It hooks the legacy transpiler entry points, and `retire_legacy.py` currently deletes it. It is the only producer of the PrimFunc captures that `lower_sweep.py` and `inventory.py` read. Keep it only if it is retargeted to `v2.transpile`.
- `check_snapshot_deltas.py`: keep (CI: every snapshot change cites a delta row).
- `classify_tests.py`: delete.
- `fold_snapshot_deltas.py`: keep. It runs in the deletion commit, and afterwards whenever a ruled delta is folded into a base snapshot.
- `inventory.py`: decide. It generates the IR inventory in `lowering-inventory.md` and needs a capture directory. Keep it with `capture_plugin.py` or drop both.
- `lower_sweep.py`: keep (corpus lowering regression sweep). Conflict: `retire_legacy.py` deletes it because it "consumes the legacy capture". It stays usable only if `capture_plugin.py` is retargeted.
- `lowering_ops.py`: keep (input to the oplib `SUPPORTED_OPS.md` generator).
- `make_contract_shim.py`: decide. It builds a contract-only crate so `validate.sh` works while other workers edit the engine concurrently, which is a migration-era need. After step 5, `cargo test -p numsim-py` with the fixture refresh covers module validation.
- `perf_gate.py`: keep.
- `record_race_fixtures.py`: keep (racecheck bench fixtures). It reads from `numsim-race-core/examples`, so it moves or goes with that crate (decide with `numsim-race-core`). The `record_observer_stream` named in the request does not exist in the tree; this is the closest file.
- `retire_legacy.py`: delete.
- `retire_tests.py`: delete.
- `status.py`: keep (one-command conformance status). Its legacy-comparison columns go with the legacy engine.
- `step5_status.py`: delete.
- `tile_rejections.py`: delete (needs a legacy capture run).
- `v2_public_status.py`: delete.
- `validate.sh`: decide (pairs with `make_contract_shim.py` and `core-rs/tools/validate-program`).
- `walk.py`: decide (helper of `inventory.py`, same fate).

## `scripts/numsim-v2/coverage/` (the retirement ledger)

- `test_classification.csv`: delete.
- `category_overrides.tsv`: delete.
- `v2_ports_*.tsv` (all: `deltas`, `failclosed`, `internals`, `messages`, `p6b`, `p6c`, `p6d`, `p7`, `p8`, `p9`, `racedeltas`, `reductions`, `stats`, `triage`, `validshape`, `w11`, `w1triage`, `w4`, `w6`): delete.
- `v2_public_status.tsv`: delete.
- `v2_xfail_inventory.tsv`: delete. The open gaps live as `v2_gap` marks and in `CONTRACT_REQUESTS.md`.
- `step5_a_status.tsv`: delete.
- `other_assertion_triage.tsv`: delete.
- `other_b.tsv`, `racecheck.tsv`, `synccheck.tsv`: delete (B-row coverage maps for waves 1/2).
- `racecheck_suite_triage.tsv`: delete.
- `tile_dispatch_rejections.txt`: delete (wave-0 list).

## `tirx_harness/src/tirx_harness/numsim/core-rs/tools/`

- `port_sync.py`: delete. It is a one-shot generator that copied the `numsim-sync-ref` types into `numsim-core/src/sync/`, and the copies are now maintained in place.
- `validate-program/`: decide (see `validate.sh`; not a workspace member).

## Elsewhere

- `scripts/dev-env.sh`: keep, with a fix needed. Unsetting the TVM variables is permanent, but the file hard-codes this host's paths (`LD_LIBRARY_PATH`, `NUMSIM_CACHE_DIR`, `PY`), so they should become overridable defaults.
- `tirx_harness/tests/conformance/test_conformance.py`, `README.md`, `snapshots/`: keep.
- `tirx_harness/tests/conformance/snapshot.py`: keep, but simplify in step 5. The `NUMSIM_IMPL` switch, the legacy-anchor mapping and `relax_unanchored` compare v2 against legacy-recorded snapshots and become dead once the snapshots are folded and regenerated from v2.

## CI

`.github/workflows/tests.post-deletion.yml` references only `check_snapshot_deltas.py` (keep) from this list. It does not reference any file in the delete set.
