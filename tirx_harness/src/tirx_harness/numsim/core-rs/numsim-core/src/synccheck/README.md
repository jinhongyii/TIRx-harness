# `synccheck/`: offline protocol explorer

Synccheck proves that a recorded launch's synchronization is correct under every schedule,
not just the one that ran. Its input is the `SyncEvent` log (`RecordingObserver`), and all
semantics come from `sync::*::step`. The spec is `docs/development/synccheck-explorer.md`.

## Shape

1. **Phase A** (`mod.rs`): `Protocol` events that `Failed` or were `BlockedAtExit` in the
   concrete run are reported as-is. Phase B runs only when the run was clean.
2. **Program** (`program.rs`): committed commands per warp, with collectives unioned. Retries
   the explorer re-derives (named `Resume`, setmaxnreg `Poll`, failed polls) are dropped.
3. **Reference run** (`reference.rs`): one complete schedule per connected component. It
   gives per-command vector clocks and generations, landing commits in FIFO per
   `(warp, restricted)`.
4. **Projection** (`projection.rs`): one transition system per resource group with HB gates.
   Each is solved by a causal certificate (`certificate.rs`), fingerprint reuse
   (`fingerprint.rs`), or DFS (`explore.rs` over `ts.rs`) with state hashing, sleep sets,
   strong diamonds and persistent singletons.
5. **Report** (`payload.rs`): findings, plus `incomplete` for anything unproven.

## Singleton rules (`explore::Rules`, `ts.rs::singleton_persistent`)

| Switch | Fires on |
| --- | --- |
| `private_issue` | A warp-private async-group issue or commit |
| `ready_observer` | A ready mbarrier wait/test with only observers before it (`only_observers_before`) |
| `deferred_completion` | A deferred mbarrier completion proven `independent_of_future` (S8) |
| `regpool_sync` | A setmaxnreg `WarpgroupSync` credit when no `Set` of that warpgroup can run first |
| `sole_landing` | A deferred completion that is the only possible mutation of its barrier |
| `twin_landings` | Symmetry: only the lowest of interchangeable enabled pendings is explored |

`Rules::ALL` is the default. `Rules::NONE` plus the `Options` switches (`sleep_sets`,
`strong_diamonds`, `persistent`) are for the bench and the oracle.

## Budgets and incomplete

`SynccheckConfig` bounds the search with `state_budget` (1M) and `transition_budget` (10M)
per projection, and `limits.max_wall_time_ms` (reference run and between projections).
Each overrun is `BudgetExhausted` (incomplete), never Clean. Unmodelled input is also
incomplete:

- a log that does not build → `fixed_sync_program_build`;
- a generation mismatch with the reference → `generation_assignment_differs`.

## Adding a rule (reduction)

1. **Switch.** Add a field to `explore::Rules` (also in `ALL` and `NONE`), and implement it
   in `ts.rs`, guarded by `self.rules.<name>`. State the soundness argument in
   synccheck-explorer.md §5.10.
2. **Oracle.** `tests/synccheck_equivalence.rs` compares each variant with the all-failures
   exhaustive oracle across its generators. Add a generator shape that makes the rule fire.
   Then check that a deliberately unsound version of the rule fails the comparison.
3. **Scenario.** Add a `tests/synccheck_scenarios.rs` test with a state-count bound that
   fails when the switch is off, and a bug variant that is still found when it is on.
4. **Bench row.** Add one row to `benches/synccheck.rs` that turns the rule off. The bench
   asserts every "on" run is Clean; `SYNCCHECK_TABLE_ONLY=1` prints the table. Record the
   numbers in synccheck-explorer.md §5.9.
