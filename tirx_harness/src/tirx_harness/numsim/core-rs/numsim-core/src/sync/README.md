# `sync/`: synchronization protocols

This module holds one pure transition function per GPU synchronization concept. The engine,
racecheck's producer and the synccheck explorer all call the same `step`. The spec is
`docs/development/sync-semantics.md`, and the ISA rulings are in `sync-isa-answers.md`.

## Protocols

| Module | Concept | `State` shape | Notes |
| --- | --- | --- | --- |
| `mbarrier` | mbarrier | `expected`, `gen`, `arrived`, `tx_expected/completed`, `outstanding` (per gen), `buffered_next`, `armed` | `stuck` gives `TxUnderDelivered` once the launch is quiescent |
| `named` | `bar.sync/arrive/red` | `expected`, `gen`, `arrived`, `warps` by `(warp, Flavor)` | `Gather` per warp for non-aligned partial warps |
| `cluster` | `barrier.cluster` | `live` lanes per warp, `gen`, `arrived`, `last_arrival/waited` | `Gather` for non-aligned partial warps |
| `async_group` | cp.async / bulk groups | `open`, a FIFO of `groups`, `next_ordinal` | `exit_lint` |
| `tcgen` | TMEM alloc/dealloc/relinquish | `capacity`, `exclusive_max`, `ctas[2]` | `work_step`/`WorkCmd` for the commit FIFO, including `CommitSharedA` |
| `setmaxnreg` | register pool | `current`, `pending`, `needs_sync`, `available` per warpgroup | `WarpgroupSync` on an incomplete warpgroup is a no-op |
| `query` | `pending_count` tokens, layout | pure functions | — |

`SyncTable` (`mod.rs`) maps a `ResourceId` to a protocol state:

- `step` lifts a protocol `Blocked` to `Step::Blocked(id)`.
- `step_all` applies a multi-target instruction (multicast) all-or-nothing.
- `stuck` is the quiescence query (sync-semantics §1.5, §2.9).

## Rules every `step` obeys

- **Signature:** `fn step(&mut State, Cmd) -> Result<Outcome, Error>`.
- **Transactional:** an `Err` leaves the state unchanged.
- **Deterministic and total:** it never panics.
- **Blocking returns `Outcome::Blocked`:** the scheduler retries it. A blocked step changes
  nothing, with one exception: a blocked mbarrier `WaitParity` sets `armed`.
- **The caller does the pre-step work:** lane aggregation, address resolution, multicast
  expansion and collective rendezvous all happen before `step`.

## Reference model and differential contract

`numsim-sync-ref` is the executable spec; production copies its `State`/`Cmd`/`Outcome`/
`Error` shapes verbatim (plus serde). `tests/sync_differential.rs`
drives both `step`s with proptest command streams and requires:

- equal results and equal states after every step;
- unchanged state on `Err` or `Blocked`.

`coverage_reaches_every_variant` requires every reference `Cmd`/`Outcome`/`Error` variant
to be hit or listed in `UNREACHABLE` with a reason (`SYNC_COVERAGE_TABLE=1` prints hits).
Reference-only properties live in `numsim-sync-ref/tests/properties.rs`.

## Adding a command

1. Spec it in `sync-semantics.md` (rule plus ISA quote or ruling) and add a row to its §10
   rule map.
2. Add the variant and its semantics to `numsim-sync-ref/src/<proto>.rs`, with a property
   test.
3. Mirror it in `sync/<proto>.rs`: same shape, same semantics.
4. Map it in `tests/sync_differential.rs`: the op strategy plus the spec→prod command pair.
   The coverage test fails until every new variant is hit.
5. If the engine emits it, map it in `synccheck/program.rs`. If it changes ordering, also
   update `synccheck/{ts,reference}.rs`, and check the racecheck edge
   (`docs/development/checker-review.md` §2).
6. Errors: give the new variant a `finding_kind` and a name in `synccheck/kinds.rs`.
