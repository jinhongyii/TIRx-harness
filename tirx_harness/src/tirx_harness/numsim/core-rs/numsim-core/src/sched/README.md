# sched: launches, partitions, rounds

`run_with_config(module, inputs, observer, config)` binds the inputs (`allocate_host`) and runs each kernel of the module in order.

## Partitions and rounds (mod.rs, partition.rs)
- One `Partition` per resident cluster. A partition owns its CTAs, its `SyncTable`, its `LaunchAux` and an `EventBuffer`. Clusters are admitted whole, as long as `max_resident_ctas` allows (`turnover`).
- Each round has these phases:
  1. **Parallel phase.** Each partition runs `run_round` on its arena shard (copy-on-write 4 KiB stripes of global memory) on the worker pool. Per CTA, a round marks `round_boundary`, runs one quantum-sized slice per runnable warp, then lands async ops (`land`, seeded subset) and applies completions.
  2. **Merge.** Shards merge in replay order (`Arena::shard_replay_order`: readers of bytes another partition wrote go first; a read/write cycle is `incomplete`). Events are delivered in that order.
  3. **Serial phase**, on the main arena in partition order: parked serial points (global RMWs, CLC claims) and deferred global reductions.
  4. **Turnover**: retire finished clusters and admit pending ones.
- Deadlock and stuck detection run when a round makes no progress: `TxUnderDelivered`, a blocked-forever error, or `incomplete`.

## Declared-word history (`wait_until` verdicts)
- The launch-wide `WordTable` (interp/aux.rs) is authoritative. Each allocation's regions are sorted with bounded searches, and `declare`/`log_lane` mark regions dirty.
- `merge_words` appends each partition's dirty regions in delivery order (after the parallel phase, then per partition in the serial phase and in `drain_all`). It renumbers buffered `WaitVerdicts` and verdict caches, then refreshes only the touched regions in every partition.
- History exists only for history-consuming observers.

## CLC task queue
`interp::aux::ClcTasks` is launch-wide, behind an `Arc`. With `RunConfig::subset`, `try_cancel` hands each non-resident cluster's task to exactly one caller. Claims happen only on the main arena, so they are deterministic. With no subset nothing is claimable.

## Determinism invariants (tests)
- Outputs and observer streams are independent of the worker count:
  - `partitioned_wait_until_is_deterministic_across_workers` (tests/interp_scenarios.rs);
  - `every_scenario_is_observer_and_worker_independent` and `history_overflow_is_worker_independent` (tests/sched_partition_review.rs);
  - `clc_claims_non_resident_tasks_under_a_subset`.
- Nothing program-visible depends on the observer: `partitioned_words_do_not_depend_on_the_observer`. A fixed seed gives a fixed schedule: `deterministic_for_fixed_seed`.
- Verdict indices follow the delivery order: `wait_verdict_indices_follow_the_delivery_order`, `two_writers_and_a_poller_number_like_the_delivery`.

## Single-partition fallback
`cooperative` topology or any `grid.sync`; `max_resident_ctas == 0`; kernels that mix tcgen05 `cta_group::1` and `::2`; `RunConfig::single_partition`. See docs/development/engine-review.md, "Partitioning and the single-partition fallback".

## Rules
- There is one execution path: no behaviour may depend on whether anything observes.
- Async ops, ids and RNG draws are partition-scoped. Merges and replays follow partition order or replay order, never thread order.
