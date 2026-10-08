# NumSim and Native Analysis Engineering

NumSim lowers TIRx to `Program` bytecode in Python (`v2/lowering/`) and executes it on one Rust engine (`core-rs/numsim-core`); Racecheck and Synccheck observe that same execution. Current source, the specs under `docs/development/`, and public APIs define implemented interfaces—not intended hardware semantics or stale migration plans. Production code must not import tests or depend on paths outside packaged `tirx_harness`. `docs/development/architecture.md` maps every concept to its file; `docs/development/dev-loop.md` has the build, test, and environment commands.

The legacy engine (`engine-rs/`, `../../../frontend-rs/`, and the top-level `numsim/*.py` modules) stays untouched until the migration deletes it (pending: legacy deletion). Do not fix the new engine by changing legacy code or legacy snapshots.

## Priorities

1. Correctness: model intended GPU semantics; passing tests or simulator output are not their own oracle.
2. Completeness: state exactly what was checked; unsupported semantics or limits must fail closed.
3. Performance: optimize only after correctness is established, without weakening simulation or analysis.

## Problem-Solving Workflow

- Reproduce with the smallest focused case: a hand-built `Program` (`testutil::ProgramBuilder`) or contract-event scenario first, then a small kernel, then every affected corpus case, including wiki kernels when relevant.
- Before editing disputed semantics, settle the facts and a simple causal algorithm: actors, state, footprints, lifetimes, HB, terminal conditions, counterexamples, and complexity.
- Distinguish issue order, execution/completion order, HB, and register dependency; never infer one from another.
- For every kernel-invalid finding, attach every causal source site or witness its schema supports and explain why the evidence proves invalidity; classify infrastructure failures separately.
- `error` requires proof; `review` is one precise advisory or unresolved risk, never missing coverage or a substitute for a provable error, and must not stop later findings; `incomplete` means execution or coverage cannot support the claim and is never success.
- Triangulate PTX ISA wording, canonical implementations, and preferably a focused GPU microtest; record the ruling in `sync-isa-answers.md` or `racecheck-isa-answers.md`, and flag remaining hardware uncertainty instead of choosing the most convenient source.
- Numerical changes define dtype, rounding, overflow/saturation, special-value, mask/lane, and approximation behavior, then validate it against an independent reference or GPU.
- Diagnose measured bottlenecks. Separate lowering/module-cache time, input binding, engine run, and checker time (`timing` in every result).
- Prefer the smallest complete fix, but never replace requested semantics with an easier approximation or a test-specific exception.

## System Design

- **Contract ownership.** `program.rs`, `dtype.rs`, `value.rs`, `site.rs`, `observe.rs`, `report.rs`, `lib.rs`, the public `arena` API, the `sync` type shapes, and the `interp::handlers` signatures are coordinator-owned. Change them through `core-rs/numsim-core/CONTRACT_REQUESTS.md`, not in passing. Each worker edits only its own directory (`core-rs/README.md`).
- **One execution path.** Every mode runs the same numerical, control, and memory path; the mode is a run-time parameter. Observers must not project values, skip paths, or change program-visible behavior. The interpreter is the only executor, and each instruction's semantics live in exactly one `interp::handlers` function.
- **One step per protocol.** Each synchronization protocol has exactly one `step(state, cmd)` in `numsim-core/src/sync/`, transactional and deterministic. The engine, NumSim, Racecheck, and Synccheck all use it; never re-derive protocol state in a checker. `numsim-sync-ref` is the independent reference, compared by `tests/sync_differential.rs`; change both together.
- **Checkers consume event streams.** Racecheck and Synccheck see only the `Observer` callbacks: hot `Access` records and cold `SyncEvent`s. When a checker needs a fact, add it to the event contract; do not read engine state from a checker or add checker branches to handlers. Blocking returns `Blocked(resource)` and is retried; do not add wakers or waiter registries.
- **Model state transitions, not syntax.** Lowering owns source-known facts (sites, layouts, dtypes, declared sync words); the engine owns mutable launch state, scheduling, address resolution, and async lifecycle; checkers own HB and verdict policy. Do not match kernel syntax, loop shapes, or variable names.
- **Fail closed.** Unsupported forms raise `UnsupportedTIRxError` at lowering or stop the run as `incomplete`; exhausted budgets, unmerged shadows, lost qualifiers, and unexplained waits are `incomplete`. Declare a deadlock only when no warp, completion, landing, or inbox delivery can make progress; one spinning warp or a threshold is not proof.
- Put state at its narrowest real owner; add shared state, locks, or cross-actor effects only after identifying actual conflicting readers/writers and dependencies.
- Keep scheduling reproducible for fixed module, inputs, config, and seed; results and observer streams must not depend on the worker count. Synccheck explores only from the recorded log, with explicit coverage; exhausted coverage is `incomplete`.
- Keep pruning techniques (packed stamps, single witness per actor, exact-hit updates, memoized clock chunks, projections, certificates, fingerprints, strong diamonds) guarded by their criterion benchmarks; do not remove one without a measurement.
- Read the environment only in `v2/options.py`. Bump the module format version, or the lowering-source key, when a cached module would no longer mean the same thing.
- Delete superseded experiments, dead compatibility paths, and half-implemented algorithms once the replacement is proven.
- Keep public tool interfaces aligned (`transpile`, `Engine().run`, `racecheck`, `synccheck`, report `.verdict/.findings/.to_dict/.print/.require_clean`, payload `schema_version` 5); user-facing reports and docs expose actionable source evidence and scope, not internal plumbing.

## Behaviour Changes, Snapshots, and Tests

- **Every observable behaviour change needs a delta row** in `docs/development/{numsim,sync,racecheck}-behaviour-deltas.md`: legacy behaviour, new behaviour, and the PTX ISA basis or ruling. A snapshot diff with no matching row is a regression.
- **Snapshot policy.** `tests/conformance/snapshots/<case>/<mode>.json` is regenerated only from the legacy engine (`NUMSIM_IMPL=legacy ... --update-snapshots`), only for an intentional, documented change, with each changed case explained in the commit message. Never update snapshots from v2 to make it pass. A ruling that legacy was wrong goes in a hand-edited `<mode>.delta.json` naming its delta row. After legacy deletion v2 becomes the oracle: the deletion commit folds the delta files into the base snapshots (`scripts/numsim-v2/fold_snapshot_deltas.py`), `--update-snapshots` regenerates from v2, and every commit that changes a snapshot cites the justifying delta row id in its message (CI: `scripts/numsim-v2/check_snapshot_deltas.py`).
- **Do not pin internals.** Allowed goldens are GPU results for op numerics, corpus output bits, and corpus finding sets (kind, source anchor, byte overlap). Tests must not assert generated Rust text, scheduler poll/transition counts, internal payload fields, or absolute times.
- Put tests at the lowest layer that can express them: Rust scenarios from hand-built `Program`s or contract events in `core-rs/numsim-core/tests/`; lowering tests that assert `Program` contents under `tests/numsim/v2/`; corpus behavior in `tests/conformance/`; performance in criterion benches and `tests/perf/` (relative baselines, `performance` marker). Private Rust units stay beside implementation.
- Explicitly read `../../../tests/CLAUDE.md` before test or performance work; it is a sibling guide, not inherited here.
- Do not weaken existing tests to accommodate a change. Preserve or migrate every non-deprecated contract; intentional contract changes need proof and an explicit delta.
- Run focused tests while iterating, then `(cd core-rs && cargo test --workspace)`, the v2 Python tests, and `NUMSIM_IMPL=v2` conformance; rebuild the extension (`core-rs/numsim-py/build_dev.sh`) before any Python run. Deliver no unexpected `incomplete` and report exact blockers.
- Benchmark like-for-like per kernel and per mode (NumSim, Racecheck, Synccheck) and per backend; gate changes require clean measurements on a near-idle host, and temporary relaxation must be explicit and scoped.
- Keep each PR scoped to the requested component; justify shared/cross-component changes and preserve excluded or experimental work separately.
- Use the caller-assigned worktree; otherwise isolate implementation work. Update an owned feature branch against the requested base before handoff and never disturb unrelated changes.
