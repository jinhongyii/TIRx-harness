---
orphan: true
---

# Checker review: racecheck and synccheck

Scope: `numsim-core/src/{racecheck,synccheck}/`, `numsim-core/src/sync/` and
`tests/{racecheck,synccheck,sync}_*.rs` on `refactor/clean-core` (2026-10-08).

The first adversarial review (commit 1c1b9b6) is closed. Every item it raised is fixed and
has a regression test. Section 1 maps each item to its fix. Section 2 is the cross-checker
consistency audit.

## 1. Status of the first review

| Item | Checker | Status | Where recorded |
| --- | --- | --- | --- |
| S1(a), S1(b) certificate holes | synccheck | Fixed. The certificate declines and the search decides. | synccheck-explorer.md §5.5 |
| S5 DFS generation check | synccheck | Fixed. Mismatches report `generation_assignment_differs`. | synccheck-explorer.md §5.5 |
| S8 one-step strong diamonds | synccheck | Fixed. Requires the `independent_of_future` proof. | synccheck-explorer.md §5.5 |
| F4 commit coupling, TMA without `Issue`, duplicate cluster waits | synccheck | Fixed | synccheck-explorer.md §5.5 |
| `SyncEvent.kernel` (launches merged) | both | Fixed. Synccheck uses `check_launches`; racecheck reports `KernelMismatch`. | synccheck-explorer.md §5.5; racecheck-behaviour-deltas V9 |
| S2 eviction across window/proxy | racecheck | Fixed | racecheck-behaviour-deltas V1 |
| S3 `scope: None` | racecheck | Fixed. Reports `SyncQualifierUnknown`. | V2 |
| S4 declared-word numbering | racecheck | Fixed. Numbered per `(Access, lane)` for every overlapping word. | V3 |
| S6 tensormap acquire filter | racecheck | Fixed. Filters by the releaser. | V4 |
| S7, F2 silent scope failure | racecheck | Fixed. One `ScopeMismatch` per site pair. | V5 |
| F1 `CrossCtaAsyncOrder` | racecheck | Fixed | V6 |
| F3 `fence.sc` history | racecheck | Fixed. Keyed per `(thread, scope)`. | V7 |
| R1 out-of-range completion warp | racecheck | Fixed. Reports `CompletionWarpOutOfRange` (incomplete). | `checker.rs` |
| R2 unbounded incompletes | racecheck | Fixed. One entry with a count. | V10 |
| R5 GC across CTAs | racecheck | Fixed. The meet is per reach. | `checker.rs::gc` |
| R6 thread-local chunk ids | racecheck | Fixed. Uses a global `AtomicU64`. | `clock.rs` |
| R7 zero-length span | racecheck | Fixed. Skipped. | V8 |
| R3/R4 growth and hot paths | racecheck | Superseded by the `knowledge.rs` rewrite and `tuning.rs` | racecheck-semantics §6 |
| §4 `ALL_LANES` on split async ops | contract | Fixed. An async `LaneSpan` names the issuing lane (contract, 1c1b9b6). | observe.rs |
| §5 equivalence generators | synccheck | Fixed. 7 generators compared against an all-failures oracle. | synccheck-explorer.md §5.5 |

## 2. Cross-checker consistency (sync rulings → racecheck HB)

Both checkers read the same `SyncEvent` stream but encode protocol semantics twice:

- Synccheck encodes it as explorer transitions over `sync/`.
- Racecheck encodes it as HB edges (racecheck-semantics §3, map §3.1).

Racecheck ignores `SyncKind::Protocol`, which is synccheck-only (`observer.rs`). Racecheck
therefore adds an edge only where the engine emits `Arrive`, `Wait`, `AsyncIssue`,
`AsyncComplete`, `Fence`, `WaitVerdicts` or `WarpSync`.

The table below covers every ruling and delta row owned by the sync model. For each one it
states the edge that racecheck must (not) add, where racecheck implements it, and the test
that pins it on contract events. Tests named `racecheck_sync_consistency::*` are in
`numsim-core/tests/racecheck_sync_consistency.rs`.

**Result: 0 mismatches.** One row (C3/Q11) depends on an engine hook that W2 has queued; see
section 3.

| Ruling / row | Sync-model behaviour | Racecheck must | Racecheck (§3 row, code) | Pinned by | Verdict |
| --- | --- | --- | --- | --- | --- |
| Q1, T2 (`alloc` blocks while columns are taken / exclusive live) | `Blocked` until a dealloc frees the columns | Add **no** dealloc → alloc edge. TMEM reuse across lifetimes is separated by lifetime, not by HB. | row 33, `AllocBegin/AllocEnd` lifetime only | — (lifetime only) | consistent |
| Q2 (re-init without inval) | Error `ReinitWithoutInval` | No edge from `init`/`inval` | row 32 | — (no event reaches racecheck) | consistent |
| Q3 (named barrier, `bar.arrive`) | Per-warp arrival; `arrive` does not wait | Arrive releases; only `sync`/`red` acquire | row 7 | `q3_arrive_only_warp_acquires_nothing` | consistent |
| Q4 (cluster barrier, exited threads) | Exited threads are dropped from the expected set | No edge from an exit. Only arrivers release. | rows 7–8: phase join of actual `Arrive`s | `q4_exited_warp_at_cluster_barrier_publishes_nothing` | consistent |
| B7 scope rule (whole-warp exit leaves count-less named barriers) | Exit releases the barrier (`Protocol` event only) | No edge: the exit publishes nothing | `Protocol` ignored; `release_named_on_exit` emits no `SyncEvent` for racecheck | `b7_exit_release_publishes_no_memory` | consistent |
| Q5, Q3 Gather (non-aligned partial warps, named) | `named::Gather` collects the pieces into one per-warp arrival | One `Arrive` naming every gathered lane, so each lane's prior writes are released | row 7; engine `barrier_partial` emits one `Arrive` with all gathered lanes | `q5_gathered_partial_warp_releases_every_lane` | consistent |
| C3, Q11 (cluster barrier, non-aligned partial warps) | `cluster::Gather`; `.aligned` partial is `PartialWarp` | Same as Q5 at `.cluster`: one `Arrive` naming every gathered lane | row 8 | `c3_gathered_cluster_arrive_releases_every_lane` | consistent (engine hook pending, §3) |
| Q6, T8 (alloc size increase, alloc while exclusive) | Error (`AllocationSizeIncrease`, `AllocWhileExclusive`) | No edge | row 33 | — | consistent |
| Q7 (async groups per thread) | `wait_group` observes only the issuing thread's groups | `AsyncComplete{Warp{lanes}}` acquires per lane | row 16 | `q7_wait_group_is_per_lane` | consistent |
| Q7 `.read` | `.read` completes the source-read milestone only | `Milestone::Read`: the source may be reused, and the destination is not published | row 16 | `q7_wait_group_read_publishes_only_the_source_read` | consistent |
| Q8, T9 (dealloc by another warp; TMEM access outside every live allocation) | Error | No edge from `dealloc`. The engine error stops the run. | row 33 | `test_synccheck_requires_cross_warp_tmem_quiescence_before_dealloc` (W9 port) | consistent |
| Q9, M16 (multicast mask outside the cluster) | Error `bad_address` | None: the run stops before any completion | — | W9 delta ports | consistent |
| Q10, T10, CommitSharedA (restricted `tcgen05.commit`) | Arrival is ordered after earlier restricted commits and before later unrestricted ones. Tokens are reported but not drained. | Completion publishes only the shared-A reads (no MMA write, no a2g). A later unrestricted commit still tracks the full MMAs. | rows 18 and 20; delta T11 | `racecheck_tcgen` T11 tests; `restricted_commit_orders_only_after_restricted_commits` | consistent |
| Commit landing FIFO (per warp) | Commits land in issue order | Each commit's completion covers every tracked prior op (`preds` closure). No cross-commit edge is needed. | row 20; delta T7 | `racecheck_tcgen` | consistent |
| M14 (`pending_count` without `.noComplete`) | Error | No edge | row 32 | — | consistent |
| M15 (blocking wait, lanes on different barriers) | `incomplete` (`divergent_block`) | Nothing beyond each lane's own `Wait`; the run is incomplete for both | row 9 per lane; the engine stops | `test_p6c_mbarrier_lane_semantics.py` | consistent |
| M17 (partial-warp wait nobody can complete) | `incomplete` (`divergent_block`). Full-warp case: deadlock plus `stuck` → `TxUnderDelivered`. | No edge; never reported clean | Run status Error or incomplete | `interp_scenarios::tma_under_delivery_is_a_protocol_error` (W2, uncommitted) | consistent |
| S1 (generations proved schedule-independent) | The explorer proves each wait's generation, or reports | Keys by the run's `(obj, phase)`. The verdict is per execution, and synccheck owns schedule independence. | row 11 | corpus `msa_prefill_multishape` (V2C-28) | consistent |
| S7 drain (bulk op in flight at CTA exit) | No lint for an uncommitted or unwaited bulk issue at exit | No race at exit: the op drains before shared memory is reclaimed | row 33 (`AllocEnd` drain); racecheck delta S7 | `uncommitted_bulk_issue_at_exit_is_not_a_lint`; racecheck S7 tests | consistent |
| R1 setmaxnreg (pool, `WarpgroupSync`) | Register pool only. `WarpgroupSync` on an incomplete warpgroup is a no-op. | **No** memory edge | row 33 | `regpool_credits_stay_small` (synccheck); no racecheck event | consistent |
| Per-element sync words (W5-14) | One declared word per element of the `sync_words` dtype (`sync_word_spans`) | A `wait_until` on element i acquires only from writes to element i | row 30; delta W8 | `per_element_sync_word_acquires_only_its_element` | consistent |
| T2, T8, T9, T10, M14, M16 deltas | As above | As above | — | — | consistent |

## 3. Open dependencies

- **C3/Q11 engine hook (W2, queued).** The `barrier.cluster` handler must call
  `cluster::gather` and emit **one** `Arrive` naming every gathered lane, at the completing
  piece. This is what the named-barrier `barrier_partial` already does. With one `Arrive` per
  piece, racecheck would still be sound: each piece releases its own lanes, and the blocked
  lanes run nothing in between. The racecheck scenario above pins the intended
  one-`Arrive` shape, which matches what the sync model counts (one arrival per warp).
- **`WorkCmd::CommitSharedA` emission (W2, queued).** Synccheck already models it, and
  racecheck already follows T11. Until the engine emits it, a restricted commit is logged as a
  full `Commit`. That is conservative in synccheck: the arrival is ordered after every earlier
  commit, so the result may be a spurious deadlock but never a missed one.
