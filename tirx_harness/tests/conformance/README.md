# Corpus conformance snapshots

The oracle for the NumSim redesign (`docs/development/numsim-redesign.md`, §3
and §4 step 0). Every canonical corpus case
(`tests/numsim/corpus/canonical_cases.py::CANONICAL_KERNEL_CASES`, the same list
the NumSim, Racecheck and Synccheck corpus gates parametrize over) runs in three
modes, and the normalized result must match
`snapshots/<case>/{numsim,racecheck,synccheck}.json`.

## Running

From `tirx_harness/`, after `source ../scripts/dev-env.sh`:

```bash
$PY -m pytest -q -n 32 --dist=worksteal tests/conformance                    # legacy
NUMSIM_IMPL=v2 $PY -m pytest -q -n 32 --dist=worksteal tests/conformance     # new engine
NUMSIM_IMPL=legacy $PY -m pytest -q -n 32 --dist=worksteal tests/conformance --update-snapshots
```

`--update-snapshots` is declared in `tests/conftest.py`, so it is accepted by
any invocation. Only regenerate from `legacy`, and review the diff.

`NUMSIM_IMPL=v2` loads `tirx_harness.numsim.v2`; the tests skip with a reason
while that module is missing or does not yet expose `transpile`, `Engine`
(`run`, `run_racecheck_phase`, `run_synccheck_phase`), `compare`,
`CoverageBounds` and `ResourceLimits` with the legacy signatures. Cases whose
legacy snapshot is an exception are skipped under v2 (no oracle), and so is
any v2 run that raises `NotImplementedError` (unfinished numsim-core bodies).

## What a snapshot contains (`snapshot.py`)

- **numsim**: per output buffer `dtype`, `shape` and `sha256` of the raw bytes;
  `reference_ok` (the case's independent reference comparison, as in
  `run_case`); verdict; diagnostic groups.
- **racecheck / synccheck**: per launch phase (all phases, `advance_prefix=True`,
  same transpile options, workers and Synccheck budgets as the corpus gates),
  the verdict and diagnostic groups.
- **diagnostic group**: `category` (`findings`, `advisories`, `incomplete`,
  `sync.*`, `execution_error`, or `diagnostics` for NumSim), `kind`, `status`,
  `space`, the stable classification fields `access_pair`, `ordering_domain`,
  `ordering_failure`, `reason`, `cause`, and the sorted set of **source
  anchors** (`file:line:col-end_line:end_col`, resolved from legacy
  `(kernel_index, source_op_id)` through the module source map, or taken from an
  embedded serialized `source_span`). Byte overlaps of all members of a group
  are merged into `bytes: {"<region>": "a-b,c-d"}` (half-open). The region is
  `global#<allocation>` for global memory (host parameter order in both
  engines) and just the space for per-CTA windows (`shared`, `tmem`, `local`,
  `register`, `param`), whose allocation numbering is engine-internal
  (snapshot schema 2).
  TMEM ranges are projected to column bytes (`tmem-columns#N`) because the lane
  quadrant of the witnessing warp depends on the schedule.
- **exception**: `{"error": "<ExceptionType>"}` when the implementation raises
  before producing a payload.

Dropped on purpose: messages, hints, timings, stats, poll/transition counts,
occurrence and finding counts, warp ids, per-warp sequence numbers, loop
iteration ordinals, internal op ids, anonymous buffer names.

For non-legacy implementations, groups legacy recorded with no source anchor
at all are compared without anchors (`relax_unanchored`): legacy could not
name a source for some diagnostics, and v2 naming one is not a regression.

Caveat for v2: racecheck `findings` are witness-based (one prior/current pair
per site pair). The legacy witness is stable across runs, but a different
scheduler may pick a different witness pair with different anchors or bytes.
Treat such a diff as a review item, not automatically as a regression.

## Initial generation (legacy, 2026-10-07)

Host: 2x AMD EPYC 7763 (256 logical CPUs), 1 TB RAM; commit c3eac62 plus the
working tree; tirx-kernels 0.1.2.post1; `-n 32 --dist=worksteal`.

- 101 cases x 3 modes = 303 snapshots (3.2 MB, mostly
  `gdn_decode_fp32_mtp_warp`'s scattered uninitialized-read footprint).
- Cold run (empty `$NUMSIM_CACHE_DIR`, one process per case/mode, 32 at once):
  407 s. Warm `--update-snapshots`: 120 to 174 s. Warm verification:
  158 to 196 s.
- Schema 2 regeneration (2026-10-08, allocation ids dropped for window
  spaces): 184 s generate, 179-185 s verify. One verify run at load average
  51 had a single failure that did not reproduce in three later runs (two at
  `-n 64`); the failing case id was not captured. Treat a lone failure under
  heavy load as a rerun-first signal.
- Stability: after the final normalization, the snapshots were regenerated and
  then verified four times at `-n 16`, `-n 64`, `-n 96` and `-n 32`; all 303
  matched every time. (Before TMEM column projection, `gdn_cp_prefill_sm100`
  racecheck flipped one TMEM lane quadrant under `-n 64`.)

Outcome per mode:

| mode | clean | review | error verdict | exception |
| --- | --- | --- | --- | --- |
| numsim | 80 | 18 | 0 | 3 |
| racecheck (per case) | 64 | 28 + 5 mixed clean/review phases | 1 | 3 |
| synccheck (per case) | 80 | 16 + 2 mixed clean/review phases | 0 | 3 |

No case is `incomplete` in any mode. All 98 NumSim runs that produce outputs
match their independent reference (`reference_ok: true`). Every verdict agrees
with the expectations in `canonical_cases.py` except
`alphamoe_fp8_blockscale_qwen3next` racecheck, which reports a `data_race`
(`error`); the racecheck corpus gate xfails that case ("Racecheck does not yet
support AlphaMoE's wait over multiple tagged records").

Legacy exceptions (all three modes each; the existing corpus gates fail on them
too in this environment):

- `fp16_bf16_gemm`, `mla_dsv4_multishape`: `UnsupportedTIRxError: unsupported
  host statement before tirx.device_entry: Evaluate`.
- `kda_forward_portfolio_multishape`: `NumSimExecutionError: NumSim input
  binding 'cu' is unknown` (the case passes an argument the kernel no longer
  declares).

These look like a tirx-kernels / test-fixture version skew rather than engine
behavior; their snapshots record only the exception type.

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
