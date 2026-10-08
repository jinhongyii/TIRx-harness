# Corpus conformance snapshots

The NumSim oracle. Every canonical corpus case
(`tests/numsim/corpus/canonical_cases.py::CANONICAL_KERNEL_CASES`) runs in
three modes, and the normalized result must match
`snapshots/<case>/{numsim,racecheck,synccheck}.json`.

## Running

From `tirx_harness/`, after `source ../scripts/dev-env.sh`:

```bash
$PY -m pytest -q -n 32 --dist=worksteal tests/conformance
$PY -m pytest -q -n 32 --dist=worksteal tests/conformance --update-snapshots
$PY -m pytest -q -n 1 tests/conformance -k "<case>-<mode>"        # one row
```

`--update-snapshots` is declared in `tests/conftest.py`, so any invocation
accepts it. `NUMSIM_SNAPSHOT_ROOT` points the suite at another snapshot tree.
`scripts/numsim-v2/status.py --run` prints the per-mode matrix.

## Snapshot policy

- The snapshots are regenerated only with `--update-snapshots`, and the diff is
  reviewed.
- Every commit that changes a snapshot cites the behaviour-delta row that
  justifies it, file-qualified (`racecheck B7`, `numsim H5`, `sync S1`; a bare
  id is accepted only when exactly one of
  `docs/development/{numsim,racecheck,sync}-behaviour-deltas.md` has it). A
  regeneration that changes no verdict or finding (a schema bump, a
  normalization change) carries `Snapshot-Regen: schema <reason>` instead.
  CI enforces both (`scripts/numsim-v2/check_snapshot_deltas.py`, job
  `snapshot-deltas`).
- A changed snapshot with no delta row is a regression.

History: until step 5 the snapshots were the legacy engine's output, and v2
was compared with them through `<mode>.delta.json` files (a legacy snapshot
corrected by a delta row) and a relaxation for diagnostics legacy recorded
without a source anchor. The deletion commit folded the 17 delta files
(`scripts/numsim-v2/fold_snapshot_deltas.py`) and regenerated the snapshots
from v2 (`Snapshot-Regen: schema relax_unanchored projection`); the per-row
justifications are in that commit and in `git log` of this directory.

## What a snapshot contains (`snapshot.py`)

- **numsim**: per output buffer `dtype`, `shape` and `sha256` of the raw bytes;
  `reference_ok` (the case's independent reference comparison, as in
  `run_case`); verdict; diagnostic groups.
- **racecheck / synccheck**: per launch phase, the verdict and diagnostic
  groups (Synccheck budgets as in `snapshot._synccheck_limits`).
- **diagnostic group**: `category` (`findings`, `advisories`, `incomplete`,
  `sync.*`, `execution_error`, or `diagnostics` for NumSim), `kind`, `status`,
  `space`, the stable classification fields `access_pair`, `ordering_domain`,
  `ordering_failure`, `reason`, `cause`, and the sorted set of **source
  anchors** (`file:line:col-end_line:end_col`, from embedded `source_span`s
  and from `source_op_id` site ids resolved through the module's sites). Byte
  overlaps of all members of a group are merged into
  `bytes: {"<region>": "a-b,c-d"}` (half-open). The region is
  `global#<allocation>` for global memory (host parameter order) and just the
  space for per-CTA windows (`shared`, `tmem`, `local`, `register`, `param`),
  whose allocation numbering is engine-internal. Space `reg` is spelled
  `register`. TMEM footprints are column bytes (`tmem-columns`, from the
  records' `tmem_columns` ranges, x4), because the lane quadrant of the
  witnessing warp depends on the schedule.
- **exception**: `{"error": "<ExceptionType>"}` when NumSim raises before
  producing a payload.

Dropped on purpose: messages, hints, timings, stats, poll/transition counts,
occurrence and finding counts, warp ids, per-warp sequence numbers, loop
iteration ordinals, internal op ids, anonymous buffer names.

Racecheck `findings` are witness-based (one prior/current pair per site pair).
A scheduler change may catch a different dynamic instance of the same race,
with different bytes; such a diff is reviewed and cited like any other (for
example `sm100_fp8_fp4_mega_moe`, racecheck T19/X4 after engine commit
f1d920d).

## Projection rule: register-space uninitialized reads (schema 3)

Ruling on V2C-20: register-space footprints are not externally meaningful
(per-thread byte numbering is an engine detail). An `uninitialized_read` in
space `register` is therefore projected to its (category, kind, status, space)
only: no byte offsets and no anchors. A difference in whether such reads exist
still fails. This is a projection rule, not a behaviour delta.

## Projection rule: `tmem_lifetime_review` (schema 4)

Which dynamic instance of a static (load site, store site) pair witnesses a
TMEM lifetime conflict is schedule-dependent, so these findings are compared
by kind, status and source anchors only; their byte/column footprint is
dropped (W5 note in CONTRACT_REQUESTS).

## Not covered yet

- **Microtests** (`tests/numsim/microtests/`) are not snapshotted. They are
  already paired against live GPU output (a stronger oracle), but their cases
  are spread across ~11 differently shaped `*_CASES` tuples plus many inline
  test bodies, all funneled through `harness.run_paired_primfunc`, which needs a
  GPU. Follow-up: give `run_paired_primfunc` an implementation switch (or record
  GPU outputs once as sha256 goldens) so v2 can be checked on CPU-only hosts.
- The task-steal subset variants and the Mega MoE performance configurations
  in the corpus test files are not part of `CANONICAL_KERNEL_CASES` and are not
  snapshotted.
