# Sync behaviour deltas vs. legacy

Use this list to review conformance-snapshot diffs. The new `SyncTable::step` behaviour is defined by `core-rs/numsim-sync-ref/`, with the full specification in `sync-semantics.md`. Each row says which legacy model changes: E is the engine checker path, N is the engine NumSim path, S is strict synccheck. A snapshot diff that matches no row here is a regression.

ISA cites use PTX 9.4 section numbers. Quotes are in `sync-isa-answers.md`.

## mbarrier

| ID | Change | Legacy | New | ISA basis |
| --- | --- | --- | --- | --- |
| M1 | Re-init without `inval` | N allowed it on an inactive slot. E and S rejected it. | Error in every mode. The most specific kind wins: `ReinitBeforeConsumption`, then `ReinitActive`, then `ReinitWithoutInval`. | §9.7.15.16.12, §9.7.15.16.4 |
| M2 | Arrive-on into the next phase before any successful wait on the completed phase. Covers `arrive`, `arrive.expect_tx`, the non-`.noinc` `cp.async.mbarrier.arrive` increment, and deferred arrive-ons from `cp.async.mbarrier.arrive` / `tcgen05.commit`. | N and E: silent roll-over. S: error, except for the pending increment (F1). | `ReuseBeforeConsumption` in every mode | §9.7.15.16.5.1 |
| M3 | Standalone `expect_tx` before consumption | N and E: roll. S: error. | Unchanged: roll in Numeric, error in Strict only (policy extension) | §9.7.15.16.5.1 names arrive-on only |
| M4 | tx-count range | E and N: each operand ≤ 2^20-1; the phase total is checked for `expect_tx` only. S: u64 overflow only. | The signed state `expected − completed` must stay in ±(2^20−1) after every op, including bytes buffered for the next phase (`TxCountOutOfRange`). Operands are not range-checked. | §9.7.15.16.3 Table 43, §9.7.15.16.14 |
| M5 | `arrive_drop` that brings the expected count to 0 | Allowed in all models | `DropUnderflow` | §9.7.15.16.17 |
| M6 | `.noComplete` arrive that would complete the phase | E and N: untyped `EngineError`. S: no check. | Typed `NoCompleteWouldComplete`. The rule is still "pending > count". | §9.7.15.16.16, §9.7.15.16.17 |
| M7 | Init count limit with `.layout::v1` | E and N: 511. S: 2^20−1. | 511 (v1), 2^20−1 (v0) | §9.7.15.16.12, Table 43 |
| M8 | Pending-count increment beyond the limit | E and N: error. S: u64 overflow only. | `PendingOverflow` | §9.7.15.16.18 |
| M9 | Deferred arrive-on bound to a phase that other arrivals already completed (D1) | E: lands silently on the next phase; only a `debug_assert` catches it. S: error. | `CompletionAfterComplete` | Over-delivery to a completed phase |
| M10 | Stale completion, bound to an older phase (D2) | E: stays queued and surfaces as non-quiescent at exit. S: error at landing. | `StaleCompletion` at landing | Fail closed earlier |
| M11 | `cp.async.mbarrier.arrive` increment on a completed, unobserved phase (F1) | S: raises the old phase, and the increment is then lost | Covered by M2 | §9.7.15.16.5.1, §9.7.15.16.18 |
| M12 | `arrive_drop` atomicity (D5) | E and N: the drop is committed even if the arrival fails | Transactional | — |
| M13 | Waiter registry (D7) | E and S: `DuplicateWaiter` | Impossible by construction. A parked waiter still consumes at completion. | — |

## Named barriers

| ID | Change | Legacy | New | ISA basis |
| --- | --- | --- | --- | --- |
| B1 | Barrier executed by a strict subset of the warp's non-exited lanes, including an elected single lane | E: error for arrive and aligned sync, but waived under `elect.sync`; unaligned sync was accumulated. S: error only if the mask differed from the elect entry mask. | `PartialWarp` for every form | §9.7.15.1, §9.7.15.15 |
| B2 | Lanes of one warp reaching different barrier instructions (unaligned recombination) | E: full-CTA counts recombine and partial paths pass without waiting. S: partitioned resume for any count. | `PartialWarp` (fail closed) | §9.7.15.1 is silent, so fail closed |
| B3 | Mixing aligned and unaligned syncs on one barrier, or different static sites | S: `AlignedSyncContractMismatch` | Allowed | §9.7.15.1 "Different warps may execute different forms" |
| B4 | `bar.red` mixed with `sync`/`arrive` in one generation | Allowed in all models | `RedMixed` | §9.7.15.1 "unpredictable" |
| B5 | Incomplete generation at exit (dangling `bar.arrive`) | E: `CompletionSourceNotQuiescent` error. S: only full-CTA all-aligned generations. | Review lint `DanglingAtExit` | §9.7.14.7 |
| B6 | Arrival counting | E and S: per active lane | 32 per warp arrival, once all non-exited lanes have executed the instruction | §9.7.15.1 "marks warps' arrival" |
| B7 | Exit of whole warps releasing a barrier | Not modeled | Still not modeled (G8). The scheduler must report such a hang as `incomplete`, not success. | §9.7.14.7 |

## Cluster barrier

| ID | Change | Legacy | New | ISA basis |
| --- | --- | --- | --- | --- |
| C1 | Membership | E: the selected warps, fixed at launch. S: the full cluster range. | The non-exited threads. `Exit` shrinks membership and may complete the generation, which releases waiters. | §9.7.15.3, §9.7.14.7 |
| C2 | Incomplete generation at exit | E: error, or a deadlock with exit evidence. S: incomplete. | No finding: exits complete it | §9.7.15.3 |
| C3 | Partial-warp arrive | E: unaligned lanes accumulate. S: always rejected. | `PartialWarp` unless the mask is exactly the warp's non-exited lanes | §9.7.15.3 "wait for all non-exited threads from its warp" |
| C4 | Partial-warp wait | E: `UnalignedWaitUnsupported`. S: `PartialWarpParticipation`. | `PartialWarp`, under the same rule as C3 | §9.7.15.3 |
| C5 | Rearrival without an intervening wait | E: silent. S: unmodeled (incomplete). | A flag in `Outcome::Arrived`; checkers report it as unmodeled | §9.7.15.3 is silent |

## Async groups

| ID | Change | Legacy | New | ISA basis |
| --- | --- | --- | --- | --- |
| A1 | Wait visibility | Racecheck: one merged warp clock acquired by the whole wait mask | Per lane: a wait acquires only the executing lane's groups | §9.7.10.28.1.1, §9.7.10.28.3.3 |
| A2 | `cp.async.bulk.wait_group.read` | Acquires the source-read milestone | `acquired: ReadsDone` only. Never publishes destination writes. A destination read after only `.read` is a race. | §9.7.10.28.6.2 |
| A3 | Uncommitted `cp.async` issues at exit | E and N: `CompletionSourceNotQuiescent` error | Review lint `UncommittedAtExit` | The ISA does not require a commit |

## tcgen05

| ID | Change | Legacy | New | ISA basis |
| --- | --- | --- | --- | --- |
| T1 | `alloc` with no free columns | E, N and verifier: `AllocationUnavailable` | `Blocked` and retried. A deadlock is reported only when nothing can progress. | §9.7.18.7.1 |
| T2 | `.exclusive` alloc while other allocations are live, or any alloc while an exclusive allocation is live | Placed first-fit | `Blocked` | §9.7.18.7.1 |
| T3 | Dealloc exclusivity mismatch | Not checked | `DeallocationMismatch` | §9.7.18.7.1 "deallocated with .exclusive if and only if" |
| T4 | Exclusive width limit | `min(cap, 512)` | `exclusive_max` parameter: 512 on sm_100f/103/110, 576 on sm_107f | Table 58 |
| T5 | `cta_group` uniformity | E: across lifecycle ops only; commit silently ignored work of the other group | Kernel-wide across all tcgen05 ops; any mismatch is `CtaGroupMismatch` | §9.7.18.7.1 |
| T6 | `cta_group::2` peer warp index | E: must equal `warp_id_in_cta` | Any warp of the peer CTA | §9.7.18.5 Table 55 |
| T7 | `AllocationSizeIncrease`, sticky across deallocs | Same | Unchanged; now ISA-cited | §9.7.18.7.1 |

## setmaxnreg

| ID | Change | Legacy | New | ISA basis |
| --- | --- | --- | --- | --- |
| R1 | NumSim pool semantics (D6) | N: no pool, no direction check, `inc` never blocks | Checker-mode pool in every mode. NumSim can now report `InvalidDirection` or a pool deadlock. | setmaxnreg: `inc` blocks until registers are available |
| R2 | Warpgroup-sync rule in the fixed verifier | Verifier: not modeled | `MissingWarpgroupSync` in every mode | setmaxnreg: "synchronize explicitly before a subsequent setmaxnreg" |
