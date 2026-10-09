# Racecheck: parallel checker (planned project)

Status: design only, not implemented. Owner: racecheck (W5). The current
serial checker and the measurements that motivate this work are in
`racecheck-semantics.md` § "Merge design and the serial-checker limit".

## 1. Problem

The engine runs scheduling partitions (clusters) in parallel within a round,
but every partition's events are replayed into the one `RaceObserver` after
the round (`sched::partition::EventBuffer::replay`), so checker time does not
scale with workers.

On mega_moe e24 (148 SMs, 16 workers) the checker takes about 17 s of a
19.8 s racecheck run:
- `sync()` takes about 12 s, about 70% of it clock joins on acquires over
  ~10–15K live async actors;
- `access()` takes about 5 s.

Legacy scales from 12.8 s to 2.5 s across workers. Four single-thread levers
on the actor space were measured and rejected (< 1.5x; see the semantics
section), because the witnesses that pin the actors are live. The checker
therefore has to run in parallel.

## 2. Principle: mirror the engine's partitioning

The scheduler already guarantees what a parallel checker needs (sched/mod.rs,
"Partitions and parallelism"):
- A partition (a cluster, or the whole launch for programs with launch-wide
  state) owns its CTAs, warps, `SyncTable` and private allocations. Shared
  memory, TMEM and mbarriers are cluster-private.
- Within a round, a partition sees other partitions' global writes only from
  the next round on. Global memory goes through a copy-on-write stripe
  overlay, and cross-partition effects are applied at the merge.
- The replay order (`Arena::shard_replay_order`) is a deterministic
  interleaving consistent with what each partition observed, and `seq` is
  assigned there.

So the checker can run one **checker partition** per scheduling partition in
parallel, with cross-partition interaction confined to a deterministic
**round merge**.

## 3. State split

| State | Owner | Notes |
| --- | --- | --- |
| Shadow of shared / TMEM allocations | checker partition (by CTA's cluster) | never touched by another partition |
| `Warp` (knowledge, lane rows, tcgen views, pending acquires, fence heads) | partition of the warp | mutated only by its own events |
| `AsyncActor` slots | partition of the issuer | slot ids drawn from a per-partition range (deterministic, worker-independent) |
| `Phase` payloads (mbarrier, named barriers) | partition owning the barrier | remote arrives from another cluster are impossible; cluster barriers are within one partition |
| Global-memory shadow | **stripe shards** (4 KiB stripes, as the engine's overlay) | read and written by every partition; see §5 |
| Declared-word histories, release heads on global words, `grid.sync` | global | launch-wide state forces a single engine partition today; see §6 |
| `JoinMemo` | per checker partition | memo identity is per partition; chunk ids stay process-global (R6) |
| Findings, advisories, incompletes | per partition, merged | §7 |

## 4. Per-round immutable HB snapshots

A partition's ordering tests need knowledge from other partitions only
through global memory: release/acquire on global words, and global
async-proxy completions observed via phases in its own cluster. That
knowledge was produced in earlier rounds.

At the end of each round the merge publishes an immutable snapshot of every
cross-partition HB carrier:
- the release heads attached to global write witnesses (`Entry.rel` / `base`);
- the knowledge of async ops whose completion another partition can observe.

`Knowledge` is a set of `Arc`-shared clocks (`Epochs` groups, `Lanes`
blocks), so a snapshot is pointer copies, never a deep clone.

During the next round, a partition reads these snapshots immutably; its own
mutations go to its private state. The per-access test (`ordered`) already
only reads HB state, which is what makes this split possible.

## 5. Global memory: stripe shards and the round merge

Within a round, each partition records its global accesses in its event
buffer, as today. At the round merge, global accesses are applied by
**stripe shard**:
1. **Fan-out.** Each stripe shard (an address range of a global allocation)
   receives, in the replay order `(round, partition, seq)`, the global access
   pieces that fall in it. Accesses spanning stripes are split, as
   `Checker::access` already splits per segment.
2. **Parallel apply.** Stripe shards are independent. Each applies its pieces
   in order against its shadow cells. Every ordering test reads the
   *issuing* partition's knowledge as of that access, which needs the
   attached HB handle described next.
3. **HB handles.** When a partition processes an access in its own pass, it
   records a handle to its current knowledge with the access: the warp's
   `(base, extra[lane], row, tcgen[lane])` Arcs, plus a version number that
   bumps only when the warp's knowledge changed, so consecutive accesses
   share one handle. For an async access, the handle is the op's `k`.
   - The global pass uses the handle, never the live warp, so it is
     independent of how far the partition has run.
4. **Strong global accesses.** A strong global read that reads-from a release
   head acquires knowledge; that mutates the reading warp. These, together
   with wait_until verdicts, run in a short **serial global-sync step** after
   the parallel apply, in replay order. They are a few percent of global
   accesses. Their effects become visible to the partition's subsequent
   events.
   - A partition's events after a strong global read in the same round
     depend on that read's acquire. The engine already treats such reads as
     serial points (atom/red re-run in the serial phase), so the scheduler's
     serial requests mark exactly the events that need this.
   - **Design risk.** This is the part that needs prototyping: either
     split the partition's processing at such points (resume after the
     global-sync step), or require acquire reads of global words to end the
     warp's slice. A contract change on the engine side, owned by W2.

## 6. Launch-wide state

Programs with declared sync words, `wait_until`, `grid.sync` or mixed
`cta_group`s run as one engine partition today, except where W2's
partitioned-words change already splits word histories. For them, checker
parallelism comes only from stripe-sharding the global shadow (§5) and from
running the HB pass for different warps concurrently.

The second source is not pursued in v1. Within one partition, sync events
are a total order and joins dominate. The single-threaded optimisations
measured so far do not change that.

Expected gain for such programs: the access part only (about 5 s of 17 s on
e24). **This is the main risk to the 2.7 s target.** mega_moe declares sync
words, and whether it runs partitioned depends on W2's partitioned-words
support. The first measurement in §10 settles it.

## 7. Deterministic merge of findings

Each partition and stripe shard produces findings keyed as today:
- `(kind, sites, alloc, bytes)`, with occurrence counts and a representative
  witness.

The merge works as follows:
- **Deduplicate by key.** Sum `occurrences`; union the byte ranges.
- **Representative.** Keep the earliest by the first `seq` of the current
  access, which is the canonical rule. Ties break by partition index, never
  by thread.
- **Witness actors** (warp ids, async op ids) are already global. Async op
  ids come from the engine (`AsyncId`). Slot ids differ by partition but are
  never reported (T22 made the reported epoch the milestone).
- **Advisories and incompletes** merge the same way.

Proof obligation: findings, advisories and the payload are bit-identical to
the serial checker for any worker count. CI runs the corpus at 1, 16 and 32
workers and compares hashes, as the engine's determinism tests do.

## 8. Single-witness rule and pruning across shards

- **Frontier eviction** (`EVICT_WINDOW`) is per cell. A cell lives in exactly
  one shard (private allocations in one partition; global cells in one
  stripe), so eviction semantics are unchanged.
- **View-aware GC** needs the meet of every live actor's knowledge. It runs
  at round merges:
  - each partition contributes the meet over its warps and in-flight async
    ops;
  - the global meet is the meet of those;
  - every shard then retires its own cells against it, in parallel.
- **Async-slot reclaim** stays per partition, with the same rule: done, and
  no witness in any shard. Global stripe shards report the actors they still
  name back to the issuing partition at the merge.

## 9. Observer contract changes (coordinator-owned)

- **Round end.** `Observer::round_end(round)` (new), or the next round's
  first `round_boundary`, so the checker knows when to run its merge. Today
  `round_boundary` marks only each CTA turn's start.
- **Partition replay.** `EventBuffer::replay` hands the observer one
  partition's events at a time, tagged with the partition index:
  `Observer::partition_events(p, &[Event])`. The observer can then process
  partitions on the scheduler's pool. Alternatively the scheduler calls
  per-partition observers it obtains through `Observer::fork(p)` and merges
  them with `Observer::join(p, child)`.
  - Preferred: fork/join. Partition processing then needs no event copies,
    and NoopObserver stays free.
- **Serial points (§5.4).** The global-sync step needs to know which events
  are acquire reads of global words. `Access` already carries `sem`/`scope`,
  so no new field is needed. The open question is whether the engine ends
  the slice there.
- Synccheck consumes the same stream. Fork/join must be optional: the
  default is serial replay, as today.

## 10. Measurements that gate the project

Before starting:
1. **Partition count.** On e24 and medium_moe, under W2's partitioned-words
   scheduling: how many engine partitions run per round, and the share of
   checker time in sync() for events of partitions that need no global-sync
   step.
   - If e24 runs as one partition, the projected gain is limited to the
     access part (§6). In that case the project should first answer whether
     declared words can be partitioned for racecheck too.
2. **Merge share.** The share of global accesses that are strong
   (acquire/release/atomic). This sizes the serial global-sync step.
3. **Memory.** The cost of HB handles: number of knowledge versions per
   round.

During:
- findings and payload hashes bit-identical to the serial checker over the
  conformance corpus and the recorded streams, at workers 1, 16 and 32;
- e24 and medium_moe racecheck wall time vs workers; target: scaling within
  2x of NumSim's engine scaling;
- the criterion guards in `benches/racecheck.rs` unchanged.

### 10.1 Gating measurements (taken; counters on a private build, reverted)

Counting observer (word history on, as for racecheck) over the recorded
streams; partitions counted in `Scheduler::parallel_phase`.

| case | workers | rounds | partitions / round | accesses | global accesses | strong global (atomic or non-weak) | acquire-capable global | warp HB handles (total; per round mean / max) | async ops with global accesses per round (mean / max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mega_moe e24 (t8_h1024_i512_e24_k2_g1) | 16 | 174 | 74 (all rounds) | 152,933 | 10,933 (7.1%) | 4,329 (39.6% of global) | 4,049 (37.0%) | 4,369; 25.1 / 424 | 27.5 / 420 |
| mega_moe medium (t64_h2048_i1536_e96_k4_g1) | 16 | 2,391 | 74 (all rounds) | 3,038,486 | 236,208 (7.8%) | 35,529 (15.0%) | 31,593 (13.4%) | 48,988; 20.5 / 1,266 | 70.7 / 487 |
| fp16_bf16_gemm | 16 | 38 | 8 | 3,240 | 96 | 0 | 0 | 0 | 2.5 / 15 |
| gdn_decode_bf16_wide_vec_mtp | 1 | 10 | 128 | 56,576 | 18,432 | 0 | 0 | 1,024; 102.4 / 512 | 0 |

Reading against the 2.7 s target (e24):
1. **Partitions.** e24 and medium already run as 74 partitions every round
   under the partitioned-words scheduler (one per 2-CTA cluster). The §6 risk
   does not materialise for them, and checker work is divisible across
   partitions.
2. **Serial share.** Global accesses are 7–8% of accesses. The strong ones
   that need the serial global-sync step number 4.3K on e24 (35K on medium).
   That is about 0.1 s of acquire joins at today's ~20 µs per join.
   Everything else (shared/TMEM accesses, mbarrier and tcgen sync, which
   carry the clock joins that cost about 12 s) is partition-local.
3. **Handles.** HB handles are cheap: 4.4K warp-knowledge versions over the
   whole e24 run (at most 424 in a round), plus at most 420 async ops per
   round. Each handle is a few Arc clones.
4. **Projection.** Checker about 17 s split across min(16 workers, 74
   partitions), so ≈ 1.1 s, plus the serial global step and merge
   (≈ 0.1–0.5 s), plus the engine and replay (≈ 0.5–1 s) ≈ **1.7–2.6 s**.

**Verdict: go**, with the 2.7 s cap within reach but with little margin; load
imbalance across partitions is the unmeasured term. The first prototype
milestone should be the partition-parallel `sync()`/`access()` pass alone
(serial global pass), measured on e24 before the stripe shards are built.

## 11. Review (W6, 2026-10-08)

The review checks the design against the scheduler's tested invariants (numsim-redesign.md §2.3; I1–I13 in `tests/sched_partition_review.rs`). Each item is **pass**, or **open** with the rule the implementation must follow.

### 11.1 Determinism of the merge (§7)

| # | Item | Status |
| --- | --- | --- |
| D1 | `seq` is assigned at replay in `shard_replay_order`; that order does not depend on the worker count (I7). The representative "earliest first `seq`" rule is therefore worker-independent. | pass |
| D2 | `AsyncId`s are partition-scoped by cluster, `(cluster + 1) << 40`, so they are deterministic. | pass |
| D3 | **Partition index is not stable.** `Scheduler::partitions` is a `Vec` that loses entries at turnover (`partitions.remove(k)`), so index p names different clusters in different rounds. Tie-breaks, per-partition slot ranges and `JoinMemo` identity must key on the partition's **first cluster id** (`Partition::clusters[0]`, the same value the RNG seed and `AsyncId` range use), never on the `Vec` index or a pool-thread id. | pass (milestone 1, §11.5) |
| D4 | **Join order.** `Observer::join(p, child)` must run after `par_for` returns, in partition order, as `merge_shard` does today. The merged finding list must be sorted by (key, representative `seq`), never in completion order. | pass (milestone 1, §11.5) |
| D5 | **`max_findings` cap.** The serial checker stops recording after the cap, in `seq` order ("later findings were not recorded"). A per-partition cap would keep a different set. Rule: each partition keeps its earliest `max_findings` by `seq`, the merge keeps the global earliest `max_findings`, and the `incomplete` note fires iff the global count exceeds the cap. The global earliest N is a subset of the union of each partition's earliest N, so the result equals serial. | pass (milestone 1, §11.5) |
| D6 | Per-cell processing order. A private cell sees its partition's events in `seq` order. A global cell (one stripe) receives pieces in replay order `(round, partition, seq)`, which is `seq` order. `EVICT_WINDOW` decisions are therefore identical to serial. | pass, if fan-out preserves `seq` order |
| D7 | GC at round merges (§8) uses a meet that is at most the serial checker's at the same `seq`. Retiring dominated witnesses changes no finding, but it can change which witness is reported if serial GC ran at a different point. Pin GC to round merges in the serial checker too, or prove reported witnesses never depend on GC timing. | pass (milestone 1, §11.5) |

### 11.2 HB snapshots versus the engine's merge/replay rules (§4, §5)

The engine guarantees that within a phase, a partition reads global memory as of the phase start, except for bytes it wrote itself. The replay order puts any partition that read bytes another wrote in that phase **before** the writer (I11/I12). The snapshot claim holds if the checker never shows a partition anything the engine's shard did not show it. States that a checker partition could observe **earlier** than the engine's partitions do:

| # | State | Why the design exposes it | Rule | Status |
| --- | --- | --- | --- | --- |
| H1 | Release heads of global words written by a later-replayed partition in the same round | §5.4 runs the serial global-sync step **after** the parallel stripe apply. An acquire read by B (replayed before A because B read A's bytes at the round start) would then see A's same-round write as the latest morally strong write and acquire its head. That is a spurious edge and a missed race. | Each strong read is resolved against the cell state at its own `seq`. Either interleave the stripe apply and the sync step in `seq` order per stripe, or keep per-cell writes versioned by `seq` and pick the latest write with a lower `seq`. | pass (milestone 1, §11.5) |
| H2 | `wait_until` `pred_reads` stability check | "Every retained write to those bytes is ordered before the wait" would also see same-round writes with a higher `seq` (applied in the parallel pass) and report `WaitPredicateReadsUnstable` where serial does not. | Consider only writes with a lower `seq`. | pass (milestone 1, §11.5) |
| H3 | The serial phase and `drain_all` | The engine has up to three replay batches per round: the parallel phase, the serial phase (global RMWs, partition order, main arena), and `drain_all`. A serial-phase RMW of A reads B's same-round parallel-phase write. | The checker's "round" is the engine **phase**. Serial-phase events of A are processed after the global apply of that round's parallel phase, in partition order, each partition's word history merged before the next one runs (as S-b). | pass (milestone 1, §11.5) |
| H4 | Events handed to children before the engine's merge | Under fork/join, events must reach `fork(p)` children only after `shard_replay_order` assigned `seq` and `merge_words` renumbered the buffered `WaitVerdicts` (W6-P1). Live delivery during the parallel phase would see pre-merge `observed`/`accepted` indices and no `seq`. | Children receive each partition's buffer at the same point `EventBuffer::replay` runs today. | pass (milestone 1, §11.5) |
| H5 | Snapshot contents | Release heads and async-op knowledge produced in earlier rounds/phases are exactly what the engine let other partitions read (one round later). A snapshot taken at the end of each phase cannot show more. | — | pass |
| H6 | `cross_cluster_same_round_cycle` | When no faithful order exists the engine reports `incomplete` and falls back to partition order. The parallel checker must take the same fallback order and add nothing. | — | pass, given H4 |

### 11.3 Declared words and `WaitVerdicts` numbering (I10)

| # | Item | Status |
| --- | --- | --- |
| W1 | §6 is stale. Declared words and `wait_until` no longer force one engine partition (934f2b3); the fallback list is cooperative/`grid.sync`, `max_resident_ctas == 0`, mixed `cta_group`, and `single_partition`. The §10.1 measurements already reflect this. | doc fix |
| W2 | A declared word (4 or 8 bytes, aligned) lies in one stripe, so its history is appended by one shard in `seq` order. That equals the engine's merged delivery order (I10, W6-P1), so the indices match. | pass |
| W3 | A `WaitVerdicts` is resolved after every write with a lower `seq` is in the history. Entries with a higher `seq` (same-round writes applied in the parallel pass) are never accepted, because the engine set no bit for them. The earliest-accepted rule is unaffected. Its `observed` index must still be checked against the count of entries with a lower `seq` only. | pass, with the `observed` check rule |
| W4 | Verdict edges come from the accepted entry's `Rel`, taken from the writer's HB handle as of that write (§5.3), not from the writer's live knowledge. | pass, if handles are used |

### 11.4 Scenarios W5 must add

Each runs the serial checker and the fork/join checker at 1, 8 and 32 workers, and requires identical payload hashes (findings, advisories, incompletes) plus the stated verdict.

1. **H1** (`racecheck_sync_consistency`, engine-level): `scenarios::cross_cluster_flag`, plus a variant where B's `ld.acquire` of the flag happens in the same round as A's `st.release`, before it in replay order. B's later data read must be a race, as in serial.
2. **H2:** a `wait_until` whose predicate reads a word that another cluster writes in the same round after the wait in replay order. Serial and parallel must agree on `WaitPredicateReadsUnstable`.
3. **H3:** a cluster's `atom` (serial phase) reading another cluster's same-round plain write. The reads-from and the race verdict must match serial.
4. **H4 / W6-P1:** `same_round_writers_each_waiting_on_their_own_value` under `RaceObserver`. The verdict acquires the waiter's own write, with no edge from the other cluster.
5. **D3:** a launch with `max_resident_ctas` small enough for several turnovers, so partition `Vec` indices are reused. Payload hashes must be equal across workers and versus serial.
6. **D5:** more than `max_findings` races spread over several clusters. The kept set and the `incomplete` note must equal serial.
7. **H6:** `scenarios::cross_cluster_sb` (store-buffering). The same incomplete and payload as serial.
8. **All:** extend `every_scenario_is_observer_and_worker_independent` (`sched_partition_review`) with a `RaceObserver` arm, fork/join on and off, over every scenario and the recorded corpus fixtures (e24, medium_moe, fp16_bf16_gemm, gdn).

### 11.5 Re-review of milestone 1 (d82cf38, W6)

Checked against `sched/mod.rs::replay_partitions`, `racecheck/observer.rs` (`RaceChild`, `fork`/`join`/`phase_end`) and `racecheck/checker/partition.rs`. Tests: `racecheck_parallel_review` 20/20 (release, both arms, every scenario and the 8 corpus fixtures at 1/8/32 workers); `sched_partition_review` 11/11 and `racecheck_sync_consistency` 9/9 (debug).

| Item | Evidence | Status |
| --- | --- | --- |
| D3 | `PartitionInfo.key` = `clusters.first()`; child slot pool = `key + 1` (`Checker::split`). | pass |
| D4 | Children are joined after `par_for`, in replay order (`order`), each with a pre-assigned `seq` range (`starts`); the merge sorts by tag. | pass |
| D5 | `d5_findings_cap_*`: 7 races, cap 3, truncation reported; fork/join equals serial. | pass |
| D7 | New `d7_phase_end_gc_does_not_change_the_payload`: periodic vs phase-end GC give identical findings, witnesses, verdicts and coverage over every review scenario and fixture. Only the collector counters (`gc_runs`, `witnesses_retired`) and the async-slot peak (`async_slots`, e.g. 423 vs 424 on fp16_bf16_gemm) differ, by construction. Those counters are in the payload's `coverage`, so the default switch changes them relative to older baselines. | pass |
| H1, H2 | A child suspends at its first strong/atomic global access or `WaitVerdicts`; the main checker continues in `seq` order. Exact by construction (milestone 2 must keep the round-start rule of §13). | pass |
| H3 | The serial phase and `drain_all` replay through the main observer (`absorb`), never forked; `phase_end` follows each phase. | pass |
| H4 | Parallel phase: `merge_words` (verdict renumbering) runs before `replay_partitions`. Single-partition path replays first, which is exact there (local history = merged history). | pass |
| I8 | New `race_observer_does_not_change_the_run`: status and outputs equal with `NoopObserver` and with `RaceObserver` (serial and fork/join, phase-end GC on), at 1 and 8 workers, over every review scenario and fixture. The three W13 observer-dependence items are fixed and un-ignored in `sched_partition_review`. | pass |
| W1 | §6 still describes declared words as forcing one partition. | open (doc fix) |

## 12. Effort estimate

| Part | Estimate |
| --- | --- |
| Observer fork/join contract and scheduler plumbing (W2 + coordinator) | 2–3 days |
| Checker state split into partition / stripe shards, per-partition slot ranges, JoinMemo per partition | 3–4 days |
| HB handles on accesses and the round-merge global pass | 3–4 days |
| Serial global-sync step and the engine serial-point decision (§5.4) | 2–4 days, depending on the prototype |
| Deterministic finding merge plus GC/reclaim across shards | 2 days |
| Validation (bit-identical corpus at 1/16/32 workers, perf) | 2–3 days |

Total: about 3 weeks of focused work, gated by the §10 measurements.

## 13. Milestone 1 status (fork/join, suspend at the first global-dependent event)

Implemented: `RaceObserver::fork`/`join`/`phase_end` (decision 17) on W2's
`replay_partitions` (98f344c). Checker partitions are in
`racecheck/checker/partition.rs`; phase-end GC is the default in both modes
(D7).

**Result: 1.1x on e24 at 16 workers (19.8–20.8 s serial vs 18.3–19.1 s
fork/join), below the 1.5x bar for a default-on second path.** The path
lands as infrastructure with `tuning::FORK_JOIN` default **off**; it stays
off until milestone 2 shows the gain. (Milestone 2 showed it; the default
is now on, §14.) `phase_gc` stays default on (D7-gated,
findings unchanged). The review tests run both arms explicitly. Profile at
16 workers: join/absorb 68% of samples, `wait_verdicts` 52%.

- A child processes its partition's events in seq order. It suspends at the
  first event that needs state outside its cluster: a strong or atomic global
  access, `WaitVerdicts`, `AllocBegin`/`AllocEnd`/`DeclareWord`, an SC or
  tensormap fence, an object outside the cluster, or an exhausted slot
  reserve. Everything from that event on is stashed for the main checker.
- Weak global accesses are deferred with an HB handle: a warp snapshot shared
  by consecutive deferred accesses, or an async-slot clone. `join` applies them
  in seq order (H1/H2/W3/W4 by construction).
- Findings, incompletes and stats merge by dedup key in tag order (D4/D5).
  Decode registries cover witnesses whose warp or slot a not-yet-joined
  partition holds.
- Slot pools are keyed by the first cluster id (D3), and the reservation is
  twice the pool's per-phase peak.

**Validation**
- All nine `racecheck_parallel_review.rs` fork/join arms pass at 1/8/32
  workers, including every scenario and the recorded corpus fixtures. The
  ignores are removed.
- Payload hashes on fp16_bf16_gemm, deepgemm_1d1d, gdn_decode, kda, stp,
  radix_topk and mega_moe e24 equal the serial checker's at 1 and 16 workers.
- Racecheck conformance: 101/101 with fork/join forced on.
- D7: findings are unchanged under phase-end GC on the same fixtures.

**e24 (t8_h1024_i512_e24_k2_g1), recorded stream, loaded host**

| workers | serial | fork/join |
| --- | --- | --- |
| 1 | 21.6–28.9 s | 24.3–26.2 s |
| 16 | 19.8–20.8 s | 18.3–19.1 s |

The perf gate (`tests/perf/test_corpus_perf.py -k racecheck`) passes; e24
takes 18.7 s there.

**Why milestone 1 gains little on e24.** At 16 workers, 68% of the samples
are in `join`, i.e. the main checker processing the stashed remainders, and
52% are in `wait_verdicts`. mega_moe's warps poll declared global words.
Each `WaitVerdicts` suspends its partition, and its acquire joins a large
release clock into the waiting warp, serially. The 70% of events that
precede the first suspension, measured in §10.1, are cheap events. The
costly joins come after it.

**Milestone 2 (next).** Resolve `WaitVerdicts` and strong global reads inside
the child:
- Word history and release heads come from the round-start state, which is
  immutable during the parallel replay. That is exact, because the engine lets
  a partition read only round-start values plus its own writes (I11/I12).
  A strong own write suspends earlier anyway.
- Child-side checks fall back to suspension for:
  - an accepted index at or beyond the round-start history length;
  - a `pred_reads` range this partition already wrote in the round (H2);
  - a `consumed` mark, which is applied at join in tag order.
- Measure first, on e24 and medium: the fraction of `WaitVerdicts` and strong
  global reads whose witnesses lie entirely in round-start state plus the
  partition's own history. Report it before building child-side resolution.

**Milestone-2 measurement (2026-10-08).** A probe observer (no checker)
runs the real scheduler at 16 workers. It does the following:

- It maps clusters to partitions per replay batch from the `fork` offers and
  delimits batches with `phase_end`.
- It records global touches per batch at 4-byte granularity.
- It rebuilds each declared word's history in delivery order, with batch,
  partition and async flag per entry plus each lane's own latest write.

For every `WaitVerdicts` it takes the entry the checker would acquire: the
earliest accepted entry at or after the waiter's own-write floor, or the
observed index for an async entry. It then classifies that entry:

- *round-start*: written in an earlier batch;
- *own*: same partition, with no other partition's entry before it in the
  batch;
- *foreign*: anything else.

A `WaitVerdicts` is also foreign when another partition wrote a `pred_reads`
granule in the same batch. A strong or atomic global access is foreign when
another partition wrote an overlapping granule in the same batch (for writes,
when it read one).

| e24 (t8_h1024_i512_e24_k2_g1) | total | round-start | own | foreign |
| --- | --- | --- | --- | --- |
| `WaitVerdicts` in partitions | 18068 | 18068 (1088 owe no edge) | 0 | 0 |
| strong global reads in partitions | 2750 | 2750 | 0 | 0 |
| strong global writes / RMW in partitions | 0 | – | – | – |
| strong global accesses in serial-phase batches | 1579 | | | |

| medium (t64_h2048_i1536_e96_k4_g1) | total | round-start | own | foreign |
| --- | --- | --- | --- | --- |
| `WaitVerdicts` in partitions | 87716 | 87716 (64688 owe no edge) | 0 | 0 |
| strong global reads in partitions | 25778 | 25778 | 0 | 0 |
| strong global writes / RMW in partitions | 0 | – | – | – |
| strong global accesses in serial-phase batches | 9751 | | | |

The share of partition events before the first suspension changes as
follows:

| Case | Milestone 1 | Milestone 2 | Partition events | Partition-batches |
| --- | --- | --- | --- | --- |
| e24 | 73.9% | 98.1% | 332045 | 6879 |
| medium | 88.2% | 99.2% | 5.32M | 144184 |

On e24 at 1 worker the counts are the same. Under milestone 2 the only first
suspension left is alloc/declare: 141 on e24, 411 on medium. Strong writes
and atomics run in serial-phase batches. A partition reads only round-start
values plus its own writes (I11/I12), so child-side resolution against the
round-start state is exact by construction. The join-time overlap check is a
safety net, not a policy.

## 14. Milestone 2 status (child-side strong reads and `WaitVerdicts`)

Implemented in `racecheck/checker/partition.rs` and `observer.rs`.

**Lent globals.** At the first fork of a phase, the main checker moves its
global allocations (shadow, wide table, retired summary) and their declared
words into an `Arc<Globals>`. Every child gets a read-only handle.

- During the parallel replay the main checker processes no event: every fork
  of a phase precedes its joins.
- `join` only queues the child. The queue is absorbed, in join order, at the
  next call that reaches the main checker: `phase_end`, the next fork batch,
  an event, `end_launch` or `finish`.
- Before absorbing, every queued child drops its handle and the main checker
  takes the globals back with `Arc::try_unwrap`. It panics if a handle
  survives.
- Barrier objects are indexed by cluster once per phase, so a fork no longer
  scans every phase record.

**Strong global reads in the child.** This covers lane reads with a scope,
not atomic, generic proxy, in bounds and at most 4 KiB.

- The child computes the read-from exactly as `access_core` does: the latest
  non-sibling write per segment, if it is morally strong, then
  `effective_heads`. It uses the lent shadow, then applies the read-from (or
  holds it as a poll) on its live warp.
- The shadow check and record are deferred to the join with the warp
  snapshot, like a weak access. The main checker skips the read-from there
  (`skip_read_from`).
- The child suspends instead when:
  - the range overlaps the partition's own deferred writes of the phase
    (`dwrites`);
  - the prior's performing warp cannot be decoded in the child (an async
    slot it does not hold);
  - the access is wide.

**`WaitVerdicts` in the child.**

- `wait_verdicts` and `pred_reads_stable` read the lent words and shadow
  through `words_ref`/`alloc_ref`.
- The child suspends instead when:
  - the entry it would acquire lies beyond the lent history (an entry of this
    phase), including the observed index of an async entry;
  - the word or a `pred_reads` range overlaps the partition's own writes of
    the phase;
  - an allocation is neither held nor lent.
- Alloc/declare events, SC/tensormap fences, strong writes and atomics still
  suspend.

**Safety net.** Debug builds assert at the join that no range a child resolved
against round-start state was written by a partition joined earlier in the
phase (I11/I12).

**Validation (FORK_JOIN on)**
- All 20 tests in `racecheck_parallel_review.rs` pass in release (185 s) and
  debug. That covers the 9 scenario pairs at 1/8/32 workers, the recorded
  corpus and the streams, plus D7.
- Payload hashes are bit-identical to serial at 1 and 16 workers on
  fp16_bf16_gemm, deepgemm_1d1d, gdn_decode, kda, stp, radix_topk and e24.
- Racecheck conformance passes 101/101, and `cargo test --workspace` (debug)
  passes.

**e24, recorded stream, three runs each (host load 3 to 17)**

| workers | serial | fork/join |
| --- | --- | --- |
| 1 | 20.6–21.0 s | 23.0–23.5 s |
| 16 | 19.8–21.1 s | 9.8–12.0 s (1.7–2.1x) |

**Perf gate (`tests/perf/test_corpus_perf.py -k racecheck`), back to back:
10 passed, 2 skipped in both arms.**

| case | serial | fork/join |
| --- | --- | --- |
| twenty_four_experts | 22.58 s | 13.32 s |
| shared_expert | 7.45 s | 6.73 s |
| sixteen_tokens | 7.08 s | 6.49 s |
| two_tokens | 7.13 s | 6.07 s |
| kda w1 / w32 | 2.07 / 1.53 s | 2.46 / 1.82 s |
| gdn w1 / w32 | 1.54 / 1.27 s | 1.69 / 1.09 s |
| stp w1 / w32 | 0.28 / 0.14 s | 0.33 / 0.16 s |

e24 clears the 1.5x bar, so `FORK_JOIN` now defaults on.

**Regressions.** Single-worker runs pay the fork overhead with no
parallelism: e24 +12%, radix_topk +15%, kda +19%. kda also loses at 32
workers (many small partitions).

**Next lever: contention, not serial work.** The main checker's share of e24
wall time at 16 workers is now 4.6–5.4 s; the rest is children. The children
spend 18.3 s of total CPU at 1 worker, but 39.5 s at 4 workers and 43–53 s
at 8–16. Self-time samples show atomic refcount add/sub at about 17%, plus
glibc malloc (`_int_malloc`, `unlink_chunk`, `mprotect`). Every child joins
the same round-start release clocks (Arc'd chunks, groups and lane vectors),
so refcounts bounce between cores. The cap of 2.7 s needs that cost cut, for
example a per-worker allocator, or borrowing instead of cloning in
`Clock::join` when the incoming side dominates.

## 15. The main checker's serial segment (W14, 2026-10-08)

Status: measured; parallel phase-end GC prototyped (scratch, bit-identical);
address sharding and offline analysis evaluated, not built. Owner for
implementation: W5. Numbers are from a scratch copy of 6cfabec with
wall-clock timers on the main thread, the recorded e24 stream
(`mega_moe_t8_h1024_i512_e24_k2_g1`), `FORK_JOIN` on, phase-end GC on.
Host: 256 CPUs, load average 7.5–15.8 throughout (never quiet), so walls
are given as min of 3 with the spread.

### 15.1 Where the serial segment goes

The main checker runs on the scheduler thread between the children's
`par_for`s. Per-phase counts are over 174 parallel phases (12,876 forks) and
350 `phase_end`s.

| Part (main thread) | count | 16 workers (s) | 1 worker (s) | Notes |
| --- | --- | --- | --- | --- |
| `fork`: `Checker::split` (incl. `lend_globals`, 174×) | 12,876 | 0.27–0.37 | 0.21 | HashMap moves of warps, pools, allocs, words, phases |
| `join` (queue only) | 12,876 | 0.01 | 0.01 | |
| `settle` total | 174 | 1.00–1.33 | 0.75 | |
| – release + `reclaim_globals` | 174 | 0.001 | 0.001 | `Arc::try_unwrap`, map extend |
| – `absorb`: move state back | 12,876 | 0.15–0.18 | 0.10 | |
| – `absorb`: build + sort items | 12,876 | 0.03 | 0.02 | |
| – `absorb`: merge findings / incompletes | 1,647 | 0.001 | 0.001 | |
| – `absorb`: `apply_deferred` (global shadow) | 268,550 (54,802 resolved strong reads) | 0.59–0.64 | 0.39 | ≈2.3 µs each |
| – `absorb`: stash replay | 139,365 (136,685 accesses, 2,676 syncs) | 0.21–0.22 | 0.17 | events after a child's first suspension |
| – `absorb`: drop the child shell | 12,876 | 0.09–0.10 | 0.05 | |
| `phase_end` → `gc` | 19 runs | 1.36–1.88 | 1.28 | |
| – meets | 20 | 0.12–0.15 | 0.09 | |
| – shadow walk | 20 | 1.47–1.71 | 1.22 | shared 85%, global 11%, TMEM 4% of the walk; 98,136 alloc visits, 14.6M cell visits; the largest single allocation is ≤ 5 ms per run |
| – declared-word history | 20 | 0.20–0.21 | 0.17 | ~250 live entries per run, ~40 µs each (`leq` of large release clocks) |
| – slot reclaim + site tables | 20 | 0.04–0.06 | 0.04 | |
| serial-phase / drain events into main (`access` / `sync`) | 1,583 / 7,849 | 0.49–0.70 | 0.42 | global atomics, H3 |
| `finish_launch` | 1 | 0.62–1.24 | 0.55 | final `gc` 0.23–0.27, `Checker::finish` + drop 0.27–0.44 |
| **Serial segment (sum)** | | **4.52–5.35** | **3.95** | matches §14's 4.6–5.4 s |
| children `par_for` wall (for scale) | 174 | 4.9–6.3 | 17.4 | children CPU 43.7–51.2 s at 16 vs 17.4 s at 1 (§14 contention) |
| engine parallel run | 174 | 0.59–0.82 | 2.68 | |

### 15.2 What is inherently serial, and what is not

| Part | s @16 | Class | Ruling |
| --- | --- | --- | --- |
| GC shadow walk | 1.5–1.7 | **(a) parallel** | Per allocation, reads only the meets. Merge is a sum (retired), a min per actor (site-table floor), a union (live actors). **Prototyped, §15.5.** |
| GC final `finish` drop | 0.3–0.4 | **(c) cheaper** | Free the checker on a detached thread. **Prototyped.** |
| GC word history | 0.2 | **(a) parallel** | Per entry, `leq` never touches the memo. **Prototyped.** It cannot be made incremental: the global meet is not monotone (turnover admits warps with fresh knowledge), so an entry that was useful may become useless and vice versa only forward. |
| GC meets | 0.12–0.15 | (a) parallel | Per cluster; not prototyped (≤ 0.13 s to gain). |
| `apply_deferred` | 0.6 | (a) only with address sharding | One mutable global shadow, applied in tag order. Tree-reducing child results pairwise does not reduce this work, it only pre-sorts the lists. Parallelises across stripes (§15.7); the hottest 4 KiB stripe holds 11% of e24's global accesses, so ≤ 9x. |
| `split` + `absorb` moves | 0.4–0.55 | (c) cheaper | Keep cluster-owned state (warps, pools, shared/TMEM allocs, words, phases) in one `Box<ClusterState>` per cluster key, moved by pointer, instead of per-entry HashMap remove/insert. Not prototyped. |
| stash replay | 0.2 | (b) fewer suspensions | 141 alloc/declare first suspensions (§13) carry 139K events into main. Let a child process `DeclareWord`/`AllocBegin` of its own cluster's shared allocation. W5 territory. |
| serial-phase events | 0.5 | **inherently serial** | Global RMWs in partition order (H3), the engine's own serial cut. |
| `fork`/`join`/release/reclaim bookkeeping | 0.03 | serial, negligible | |
| tree reduction of child results | — | **ruled out** | Child results are (findings: 1.6K, trivial) + (deferred global accesses: need the one shadow) + (state moves: pointer work). Pairwise pre-merging on workers removes at most the 0.03 s sort. |

### 15.3 Prototype: parallel phase-end GC

**Algorithm** (`Checker::gc`, scratch diff `w14_parallel_gc.diff`).
1. Compute the meets as today (serial; 0.12 s).
2. Build one job per allocation: `(&mut Alloc, its meet, its liveness)`.
   Shared/TMEM allocations take their cluster's meet, global ones the global
   meet, exactly as the serial loop selects them.
3. Run `gc_walk` per job on `gc_threads` threads, largest shadow first
   (atomic work index). `gc_walk` retains/retires witnesses and returns
   `(retired count, folded witnesses, per-actor min epoch)`.
4. Merge in the allocation map's iteration order (the order the serial
   loop used): sum `retired`, min-merge `min_epoch`, union live actors; fold
   each allocation's `folded` witnesses into its retired summary on the main
   thread (it decodes witnesses through `&self`, whose `JoinMemo` is not
   `Sync`).
5. Word histories: collect entries that still carry a payload; test each
   with `leq` on `gc_threads` threads (a fresh `JoinMemo` per call; `leq`
   does not read it).
6. Slot reclaim and site tables: unchanged, serial.
7. `Checker::finish` moves the report out and, when `gc_threads > 1`,
   drops the rest on a detached thread.

Below 16K live cells (32 history entries) the walk stays inline, so small
launches spawn nothing.

**Invariants.**
- I1. A job touches only its own `Alloc`. Shared inputs (meets, liveness)
  are immutable during the walk.
- I2. Every merged quantity is order-free: `retired` is a sum, `min_epoch`
  a per-key min, live actors a set; the retired-summary folds are per
  allocation and run in the serial order.
- I3. The decision per witness is the serial decision: same meet, same
  `seen` mask, same `future_proxies`, same pinning of the latest
  head-carrying write.
- I4. The observer never selects an execution path: `gc_threads` only
  changes which thread walks an allocation. `phase_end` timing (D7) is
  unchanged.
- I5. Detached drop: the report and stats are moved out first; nothing
  reads the dropped checker.

**Gates.**
- `racecheck_parallel_review` (release, corpus fixtures, 1/8/32 workers):
  20/20 with the parallel path forced on everywhere (`gc_threads` 16,
  thresholds 0).
- racecheck Rust suites (`racecheck_*`, `sched_partition_review`, lib):
  all pass with the same forcing.
- Racecheck conformance (private extension, forced): 101/101.
- e24 payload hash equal to HEAD (`a35995f721d7c1e1`) at 1 and 16 workers
  in every run (instrumented and clean builds).
- Keep: a criterion row for the walk on a recorded e24 phase (W5).

**Results.** Instrumented build, 16 workers, same binary with
`gc_threads` 1 vs 16, three interleaved runs each (load 7.5–13):

| | `gc_threads` 1 | `gc_threads` 16 | ratio |
| --- | --- | --- | --- |
| `phase_end` GC | 1.36–1.88 s | 0.39–0.42 s | 3.5–4.5x |
| `finish_launch` | 1.06–1.24 s | 0.12–0.14 s | 8x |
| **serial segment** | **4.52–5.35 s** | **2.56–2.79 s** | **1.77x (min/min)** |
| e24 wall | 11.35–12.79 s | 8.87–9.63 s | 1.28x (min/min) |

Clean builds (HEAD vs HEAD + prototype, no timers), interleaved, min of 3:

| workers | HEAD | prototype | ratio |
| --- | --- | --- | --- |
| 1 | 22.96 s (22.96–24.98) | 22.93 s (22.93–23.51) | 1.00 (path unchanged at 1 worker) |
| 16 | 8.72 s (8.72–12.57) | 7.45 s (7.45–10.03) | 1.17x |

Verdict: clears the 1.5x serial-segment bar (1.77x). It does not reliably
clear 1.3x on wall at 16 workers (1.17–1.28x): the children's `par_for`
(4.9–6.3 s, CPU-inflated 2.5–2.9x by contention, §14) now dominates.
Recommended as the first step: small, local, no semantic surface. Landing it
needs CONTRACT_REQUESTS "W14 1" (the pool at `phase_end`/`end_launch`) or,
interim, `RaceObserver::gc_threads` set from `RunConfig::workers` by
`numsim-py` (the prototype's `std::thread::scope` costs ≈ 19 × 16 spawns
per launch, negligible).

After it, the serial segment is: `settle` 1.0–1.3 (of which
`apply_deferred` 0.6), serial-phase events 0.5, `split` 0.3, `phase_end`
0.4 (meets 0.13, slot reclaim, the residual walk), `finish` 0.13.

### 15.4 Premise check: does an `Access` mutate anything but its location's shadow?

Coordinator's candidate (§15.7): shard the shadow by address; a sync pass
per partition produces each actor's clock trajectory; access shards apply
their buckets against it. Correct only if an access reads its actor's clock
and mutates only its location. Every place `access` / `access_core` /
`defer_access_as` reads or writes something else:

| # | Effect of an `Access` | Where | Class | Consequence for address sharding / offline two-pass |
| --- | --- | --- | --- | --- |
| R1 | **Read-from of a strong read / RMW**: the latest morally strong write's `effective_heads` are acquired into the warp (`acquire_rel`, `push_pending`, `tcgen_in`) | `access_core` → `apply_read_from` | **cross-location** (shadow → actor) | The actor's clock after the read depends on the location's write history. Needs the engine's `reads_from` (CONTRACT_REQUESTS W14 2) or stays a suspension point. Milestone 2 resolves it against round-start state today. e24: 64,770 strong reads (54,802 child-resolved). |
| R2 | **Declared-word poll stash**: a strong read of a declared word holds its heads in `poll_stash`; the next event of the warp flushes them (`flush_polls`), a following `WaitVerdicts` replaces them | `access_core`, `access`/`defer_access_as` prelude | cross-location (as R1) | Same as R1. |
| R3 | `WaitVerdicts` (a sync event) reads the word history (`rel`, `is_async`, `mixed_size`, `consumed`, the own-write floor `own`) and **`pred_reads_stable` reads the shadow** of the predicate inputs; unstable ⇒ no acquire | `sync` → `wait_verdicts`, `pred_reads_stable` | **cross-location** (shadow and word state → actor) | The writer of the accepted entry can be recorded by the engine (`accepted_writer`). Predicate stability is an HB judgement, not an engine fact: the actor pass must speculate "stable" and the word's shard must confirm it; a refuted speculation needs a serial re-run of the launch (it is `incomplete` anyway, but the later payload must still equal serial). e24: 18,068 verdicts, 0 unstable. |
| R4 | Word history append (`history.push`, `own.insert`, `last`) | `access_core` | per-location (the word's shard) | Stays with the word's stripe. Its readers are R3. |
| R5 | `tick` (warp epoch), site table push | `access` prelude | per-actor | Actor pass. |
| R6 | `asyncs[i].drained = true` (st.async/red.async `.release` generic) | `access_core` | per-actor (async slot) | Actor pass must see it; it is decidable from the access record alone. |
| R7 | TensorMap view: `asyncs[i].k.g2t` / `lane_g2t` from the actor's acquired ranges | `access` | per-actor, read-only of its own ranges | Actor pass computes the per-access view; travels with the HB handle (as `Deferred::lane_g2t`). |
| R8 | Prior decoding: `slot_of_w` (`warp`, `site`, `kind`, `gen_base`, `completed_ctas` with `as_of_seq`) of **another** actor | `ordered`, `morally_strong`, `cross_cta_async`, `info`, `classify`, TcgenLd read retain | per-actor **read-only, time-varying** | The shard needs every async actor's metadata as of the access: an `op_reg`-style registry keyed `(actor, epoch)` (exists for globals), and `completed_ctas` with seq stamps (exists). Generation reuse (T22 milestone epoch) is GC-dependent, so slot reclaim must be a barrier reduction (R12). |
| R9 | `register_global`: `site_reg`, `op_reg` inserts | `access_core` | per-actor evidence, insert-only | Commutative. |
| R10 | `Witness::pack` appends to the allocation's `wide` table; `seen` proxy bits; `check_retired_generic` reads the allocation's retired summary | `access_core` | **per-allocation, not per-address** | A shard boundary inside one allocation needs per-shard wide tables (fine: witnesses never leave their cell), an OR-reduction of `seen` before GC, and retired-summary hulls split at shard boundaries (a hull can span pages). |
| R11 | Alias tracker (`alias_writers`), `alias_dedup` | `alias_access` | per-location (shared/TMEM only) | Range map per allocation; dedup key global (merge by first seq). |
| R12 | Frontier eviction (`EVICT_WINDOW`), plain-write reader supersession | `cell.writes.record`, `cell.reads.retain` | per-location | Needs the current actor's `ordered` only. Clean. |
| R13 | Collective tcgen05.ld/st lane meet | `AsyncIssue` (a sync event) | per-actor | Actor pass. Not an access effect. |
| R14 | Findings: `dedup` maps, `push_finding` order, `max_findings`, occurrence counts | `report_race`, `report_advisory`, `report_scope_mismatch` | global, mergeable | Merge by (key, first seq) as D4/D5 already do. `report_scope_mismatch` comes from R1 and lands in the actor pass. |
| R15 | Counters (`stats.accesses`, `since_gc`, `last_seq`) | prelude | global, commutative | Sum/max. |
| R16 | GC: slot reclaim needs every shard's live actors; the meets need every actor | `gc` | **global reduction** | A barrier between phases: shards report live actors and `seen`; the actor side computes meets. Slot reuse feeds the next phase's `AsyncIssue` (R8), so it must complete before the next actor pass. |

**Verdict.** The premise holds for weak accesses (≈ 99% of e24's 7.27M core
accesses and of the shadow work). It fails for R1–R3: three effects feed
location state back into an actor's clock, and R3's predicate check is an HB
question no engine field can answer. R8/R10/R16 are satisfiable with
registries, per-shard tables and a phase barrier. So address sharding is
exact only with (i) the engine recording `reads_from`/`accepted_writer`
(W14 2), (ii) speculation on `pred_reads_stable` with a serial-rerun
fallback, or (iii) today's milestone-2 round-start resolution plus
suspension. (iii) exists and is exact (I11/I12); (i)+(ii) is new contract
and a fallback path. Per the brief, no address-shard prototype was built.

### 15.5 Cost model of address sharding (e24 measured, medium where noted)

| Quantity | e24 | medium | Source |
| --- | --- | --- | --- |
| contract `Access` records | 152,933 | 3,038,486 | EventBuffer counters |
| lane spans (core accesses before async coalescing) | 12.75M | 374.9M | EventBuffer counters |
| core accesses after async span merge | 7.27M (lane 0.62M, async 6.66M) | 244.8M (lane 7.2M, async 237.6M) | recorder at the checker input |
| global / shared / TMEM (core accesses) | 0.29M / 3.74M / 3.24M | 8.87M / 121.2M / 114.7M | |
| clock-table entries, lane actors: distinct (phase, warp, version) with an access, version bumped at every sync event naming the warp and every strong read (upper bound) | 73,939 total; per phase mean 211, max 10,736 | 921,766 total; per phase mean 193, max 40,761 | recorder |
| …of which used by global accesses | 65,743 | 843,346 | |
| async actors with accesses (one entry each; `k` fixed at issue except `g2t`) | 15,328 (4,792 global) | 484,200 (169,024 global) | |
| bucketing cost (32-byte record into 64 address shards) | 7.2 ns/access → 52 ms CPU total | 7.8 ns/access → 1.9 s CPU total | microbenchmark over the recorded accesses |
| HB handle cost (today's `Warp::snapshot`) | 16 Arc clones + one allocation (≈0.2–0.5 µs uncontended; the §14 contention applies) | | |

Address distribution (all lane spans, by `(allocation, 64 KiB range)`):

| Case | ranges (global / shared / TMEM) | top range share | 16 shards: hash / LPT max÷mean | 64 shards: hash / LPT | global only, 16 shards LPT |
| --- | --- | --- | --- | --- | --- |
| mega_moe e24 | 3,634 (233 / 572 / 2,829) | 0.6% | 1.14 / 1.00 | 1.42 / 1.00 | 3.91 |
| mega_moe medium | 11,173 (7,512 / 592 / 3,069) | 0.2% | 1.26 / 1.00 | 1.42 / 1.00 | 1.50 |
| gdn_decode | 665 (25 / 128 / 512) | 7.8% | 2.78 / 1.25 | 6.47 / 5.02 | 4.57 |
| recurrent_kda | 74 (74 / 0 / 0) | 19.3% | 3.47 / 3.08 | 12.7 / 12.3 | 3.08 |
| radix_topk | 13 (7 / 6 / 0) | 32.4% | 5.18 / 5.18 | 20.7 / 20.7 | 3.30 |
| fp16_bf16_gemm | 342 (21 / 64 / 257) | 0.7% | 1.09 / 1.00 | 1.76 / 1.01 | 1.03 |

At 4 KiB stripes, e24's global accesses (0.29M) touch 3,427 stripes; the
hottest holds 11.2% (LPT over 16 shards: 1.80, 64 shards: 7.2). Medium's
8.87M touch 119,624 stripes; the hottest holds 6.7% (LPT 16: 1.07,
64: 4.27). Shared/TMEM, 96% of e24's
accesses, are cluster-private, so the cluster is already the natural address
shard and the children already process it in parallel. Address sharding adds
parallelism only for the global 4%: `apply_deferred` (0.6 s) and the GC walk
(1.5–1.7 s, already parallel by §15.3).

### 15.6 Estimate: e24 wall at 16 workers

Starting from the measured parts (min runs): engine 0.6–0.7 s, children
`par_for` 4.9–5.6 s, serial 2.6 s after §15.3.

| Design | Serial left | Children | Estimate | vs HEAD 8.7 s |
| --- | --- | --- | --- | --- |
| HEAD | 4.5 | 5.0 | 8.7 (measured) | 1.0 |
| §15.3 parallel GC | 2.6 | 5.0 | 7.45 (measured) | 1.17 |
| + per-cluster state boxes, child-side alloc/declare (§15.2 (b)(c)) | ≈ 1.9 | 5.0 | ≈ 6.8 | ≈ 1.28 |
| tree reduction of child results (instead of the above) | ≈ 2.55 | 5.0 | ≈ 7.4 | ≈ 1.18 |
| address-sharded checker: actor pass per partition + global stripe shards, phase barrier, milestone-2 resolution for R1–R3 | ≈ 1.0 (serial-phase events 0.5, barrier/GC reduction ≈ 0.2, finish 0.13, sort) | ≈ 5.2 (+ handles for every global access; bucketing 0.05 CPU) | ≈ 6.5–6.9 | ≈ 1.3 |
| offline two-pass over a recorded trace (§15.8) | ≈ 0.7 + recording 0.2 | pass 1 + pass 2 ≈ 5.0–5.5 at 16 shards (LPT 1.00 by 64 KiB range) | ≈ 6.3–6.8 | ≈ 1.3–1.4 |

Every design is bounded by the children (43–51 s CPU for 17.4 s of work at
1 worker, §14). Without W5's contention fix the best serial-segment design
moves e24 from 7.45 s to ≈ 6.5 s; with a 2x cut in child CPU inflation,
the children fall to ≈ 2.5 s and the serial residual decides between
≈ 4.9 s (§15.3 alone) and ≈ 3.8 s (address sharding).

Recommendation, in order:
1. Land §15.3 (parallel GC + detached drop). 1.77x on the serial segment,
   bit-identical, ≈ 150 lines, one small contract addition.
2. W5's child contention work (the dominant term).
3. Cheap serial cuts (§15.2: per-cluster boxes 0.3–0.4 s, child-side
   alloc/declare 0.2 s).
4. Address sharding of the global shadow only when 1–3 are in and the
   serial residual (≈ 1.9 s) is again ≥ 40% of wall. Shared/TMEM stay with
   the cluster's child; global accesses go to stripe shards at the round
   merge (§5), R1–R3 stay on milestone-2 resolution.

### 15.7 Address-sharded checker (design sketch, not built)

**Pass 1 (per partition, parallel; today's child).** Process sync events and
weak cluster-local accesses as today. Global accesses are deferred with an
HB handle (today's `Deferred`), and each child buckets them by 4 KiB stripe
into `Vec<Deferred>` per stripe shard (≈ 7 ns per record). Strong reads and
`WaitVerdicts` resolve against the lent round-start state (milestone 2) or
suspend.
**Barrier.** No state merge: children return their buckets, findings and
live-actor sets.
**Pass 2 (per stripe shard, parallel).** Each shard owns its stripes'
shadow, wide table and retired summaries across rounds and applies its
buckets k-way merged by tag (= seq) order, using each record's handle
(`cur_override`, `as_of_seq`). Findings carry their first seq; the merge
sorts by (key, seq) (D4, D5).
**GC.** Per shard at `phase_end`, with the meets from the actor side, and
live actors / `seen` reduced at the barrier (R10, R16).
**Residual serial.** Serial-phase events (atomics, H3), the barrier, the
finding sort, slot reclaim.

Invariants to keep: D1–D7, H1–H6, W1–W4 as in §11; one cell is owned by one
shard; a shard applies records in seq order; a record's ordering test reads
only its handle and as-of registries; nothing in pass 2 changes an actor.

### 15.8 Offline analysis of a recorded trace

**Trace volume** (scratch counters in `EventBuffer::replay`, summed over
every replay call; "as buffered" = `size_of::<Event>()` 168 B per event +
16 B per `LaneSpan` + the heap of sync-event vectors; "encoded" = a
delta/varint, site-run, alloc-run estimate per access and span):

| Case | Access records | lane spans | SyncEvents | other | bytes as buffered | encoded estimate | peak buffered per parallel phase |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mega_moe e24 | 152,933 | 12.75M | 180,695 | 28,120 | 497 MiB | 121 MiB | 27.5 MiB |
| mega_moe medium | 3,038,486 | 374.9M | 2,289,047 | 356,236 | 14,141 MiB | 3,631 MiB | 32.1 MiB |
| gdn_decode | 56,576 | 1.67M | 1,536 | 1,792 | 47.9 MiB | 5.3 MiB | 14.2 MiB |
| recurrent_kda | 53,248 | 1.36M | 2,048 | 12,288 | 42.0 MiB | 4.4 MiB | 8.4 MiB |
| radix_topk | 20,903 | 0.67M | 21,628 | 330 | 23.8 MiB | 3.2 MiB | 1.6 MiB |
| fp16_bf16_gemm | 3,240 | 1.22M | 6,440 | 736 | 36.2 MiB | 7.5 MiB | 3.2 MiB |

e24 sync events by kind: protocol 67,438, wait 28,032, arrive 27,212,
`WaitVerdicts` 18,068, async issue 15,840, warp sync 9,206, async complete
8,664, fence 6,000, declare 235. Spans dominate the bytes (59% on e24, 61% on
medium; TMA/MMA fragments before the checker's span merge), then sync heap
(protocol command vectors, 30–33%).

**Recording overhead on the engine** (the EventBuffer path: an enabled
observer that wants word history and does nothing, vs `NoopObserver`; min
of 3; includes the serial replay into the observer):

| Case | 1 worker noop / recording | 16 workers noop / recording |
| --- | --- | --- |
| mega_moe e24 | 1.64 / 2.32 s (+0.68) | 0.34 / 0.53 s (+0.19) |
| mega_moe medium | — | 7.18 / 11.05 s (+3.87, min of 2) |
| gdn_decode | 0.21 / 0.27 s | 0.029 / 0.057 s |
| recurrent_kda | 0.58 / 0.96 s | 0.11 / 0.55 s |
| radix_topk | 0.047 / 0.18 s | 0.054 / 0.22 s |
| fp16_bf16_gemm | 0.038 / 0.055 s | 0.050 / 0.071 s |

(`examples/record_race_fixtures.py` records engine inputs, not events, so it
is not the recording path.)

**Correctness.** Offline analysis needs the §15.4 premise. R1–R3 have no
round-start lend to fall back on offline, so it needs CONTRACT_REQUESTS
W14 2 (`reads_from`, `accepted_writer`, `observed_writer`) and speculation on
`pred_reads_stable` with a serial-rerun fallback. With those, pass 1 is a
scan of the 0.18M (e24) / 2.3M (medium) sync events plus the per-access
handle table; pass 2 is 16–64 address shards that balance perfectly on e24,
medium and the GEMM (LPT 1.00) and poorly on kda/topk/gdn (3–5x at 16;
those are sub-second cases).

**Where the trace lives.**
- In-memory full trace: 0.5 GiB for e24 (0.12 GiB encoded), **14 GiB for
  medium (3.6 GiB encoded)**. Ruled out for medium-size launches.
- Per-round flush to shard threads: peak 27–32 MiB per parallel phase on
  both mega_moe sizes, independent of launch length. Pass 2 overlaps the
  next round's engine work. This is the fork/join design with the shards
  moved off the scheduler thread. **Recommended** if offline analysis is
  pursued.
- On disk: 3.6 GiB written and read back for medium (encoded), plus
  encoding. Only worth it for re-analysis without re-running the engine
  (e.g. trying checker versions on a fixed trace), not for speed.

### 15.9 Parallel collector: prototype, landing tied to the `FORK_JOIN` default

The §15.3 prototype is a complete, gated patch. It is rebased onto d719a9d
(W16's GC back-off) and kept outside the tree:
`w14-parallel-gc-rebased-d719a9d.patch`. It lands only together with turning
`FORK_JOIN` on. Under the default serial checker it gains nothing (table
below). If the re-measurement after W5's re-attribution keeps fork/join off,
it stays a recorded prototype.

**What it changes**
- Files: `racecheck/checker.rs` (`gc`, `gc_walk`, `gc_hist`,
  `Checker::finish`), `racecheck/observer.rs`, and `numsim-py`.
- The route is the interim one: `numsim-py` sets `RaceObserver::gc_threads`
  from `RunConfig::workers`. There is no Observer-contract change.
- The collector runs the per-allocation shadow walk and the declared-word
  scan with `std::thread::scope`.
- `Checker::finish` frees the launch state on a detached thread.
- Thresholds are run-time fields. `gc_par_min_cells` defaults to 16K live
  cells, and the word scan to 32 payload-carrying entries; below them the
  walk stays inline.
- The payload is the same at any `gc_threads` (§15.3 I1–I5). With the
  back-off, the walk's total cells feed both the thread threshold and the
  productivity rule.

**Gates** (on d719a9d)
- `cargo test --workspace` (debug): all binaries pass.
- `racecheck_parallel_review` 27/27, `racecheck_retirement` 11/11 (one
  ignore, pre-existing) and `racecheck_reattribution` 7/7, release, both at
  default settings and with 16 threads and threshold 0 forced.
- New `tests/racecheck_parallel_gc.rs` (2/2): `gc_threads` 2 and 16 against
  the inline collector, over every scenario at 1/8/32 workers (serial and
  fork/join) and the recorded corpus at 1 and 32 workers.
- e24 payload hash `c804e3c3622da394` is identical with and without the
  patch at 1 and 16 workers, serial and fork/join.
- Criterion guard `gc_parallel/{1,16}`: one `gc()` over 296 shared
  allocations, about 0.9M live cells (one e24 collection).
  - 32.0 ms vs 10.4 ms (3.1x) at load average 3.8–4.6.
  - 40.8 vs 11.0 ms (3.7x) before the rebase.
  - The bar is ≥ 1.5x.
- Not run yet: racecheck conformance and the v2 Python tests (pending the
  user's decision on the private extension build).

**e24 wall**: recorded stream, d719a9d vs d719a9d + patch, interleaved, min
of 3 (range), load average 3.9–9.0.

| mode | workers | d719a9d | patched | ratio |
| --- | --- | --- | --- | --- |
| fork/join | 16 | 8.77 s (8.77–10.26) | 7.33 s (7.33–8.85) | 1.20x |
| fork/join | 1 | 17.38 s (17.38–18.36) | 17.44 s (17.44–18.21) | 1.00 (same inline path) |
| serial (`FORK_JOIN` default off since 784e2df) | 16 | 16.86 s (16.86–18.41) | 16.65 s (16.65–19.12) | 1.01x |
| serial | 1 | 17.88 s (17.88–18.29) | 17.76 s (17.76–18.97) | 1.01x |

On a23dc03, before the back-off and at load average 9.8–15.3, fork/join at
16 workers went from 10.56 s to 7.47 s (1.41x). The back-off removes part of
the same collector time, so the two gains overlap.

`gc_threads` follows the configured worker count, not the scheduler's pool
capped at the resident partition count (e06f874). It stays within the
configured budget:
- the walk runs after `par_for` returns, while pool threads are parked;
- it is capped by the number of allocations.

## 16. Serial checker growth on mega_moe medium (W5, 2026-10-09)

Measured with the prof harness on the recorded
`mega_moe_t64_h2048_i1536_e96_k4_g1` fixture, serial checker, 16 workers,
private builds. Times are wall seconds from the per-phase progress line.

**JoinMemo prune cadence (landed, 54d0f94).** Up to round 1400, 31% of the
run was in `JoinMemo::join`'s prune (`map.retain(strong_count)`). The prune
ran every 1024 inserts but scanned the whole memo map, so once the live map
was large the run went quadratic. The prune now runs when
`inserts >= max(1024, map.len()/2)`, which is amortised O(1). The memo only
affects which chunk object a join returns, so findings cannot change; they
are byte-identical on 7 fixtures at 1/8/16/32 workers and conformance passes
101/101.

| Round | Before | After |
| --- | --- | --- |
| 999 | 160 s | 136 s |
| 1599 | 809 s | 374 s |
| end | no finish within 1500 s | 1433 s |

**Medium end to end** (after the memo fix): about 2300 rounds, wall
1432.6 s, of which the checker took 1416.6 s and the engine plus replay
16.0 s. Peak RSS was 11.2 GB. Verdict Error, 1508 findings:

| Kind | Findings | Occurrences |
| --- | --- | --- |
| data_race (Error) | 1184 | 7,330,048 |
| scope_mismatch (Error) | 16 | 37,530 |
| alias_stale_read (Review) | 3 | 17,920 |
| cross_cta_async_order (Review) | 305 | 28,648,724 |

Legacy's verdict on medium is Review (alias_stale_read), at 21.1 s on 16
workers. v2 reports Error from the B7 pattern (remote mbarrier.arrive
scope_mismatch, then data_race), as on e24, so the legacy time is not a
like-for-like target.

**What still grows.** After the memo fix the curve stays superlinear. The
growth is clock width.

- Async slots are never reclaimed mid-run: about 140K by the end. Per live
  warp, the base hb clock holds 31 chunks at GC run 10 and 2752 at run 110.
  Lane entries grow from 60 to 507 and the memo map from 2K to 5.9M entries.
- What pins the slots is completed copies' global-memory witnesses: weight
  TMA reads, and workspace TMA loads and stores. The global meet over every
  live warp never dominates them.
- Shared-memory witnesses stay bounded. A stage's next TMA write replaces
  the previous one through the empty/full chain (W6's
  `tma_stage_reuse_same_source_*` guards).
- A ceiling experiment (unsound, scratch only) dropped completed copies'
  global-only witnesses and reclaimed their slots. Slots stayed at about
  18K, per-warp chunks saturated near 556, and the window 1800→2000
  (access + sync) fell from about 232 s to 99 s, roughly 2.3x. That is the
  case for re-attribution (W6 review F1–F6).

**Negative: in-place group join (not landed).** When a warp's clock owns its
group slice, `Epochs::join` was changed to join touched groups slot by slot
in place instead of copying the slice and each 16-chunk group. The memo calls
were unchanged and findings byte-identical (28/28). The criterion bench
`clock_join_in_place` (16 groups, k changed chunks per group) came out at
13.1 µs owned against 17.0 µs before for k=1, and 19.7 against 23.7 µs for
k=4: 1.2–1.3x, below the 1.5x bar. Medium did not improve: the window
1800→2000 took 228 s against 202 s side by side, and round 2000 was reached
at 857 s against 818 s. Per join the cost is chunk dominance tests and memo
work, and what grows is the number of changed groups per join (clock width),
not the copy.

**Negative: per-warp index for unrestricted tcgen05.commit (not landed).**
The commit's walk over every async slot was replaced by a per-warp ordered
set of in-use pipelined slots, kept at issue and reclaim and moved in
split/absorb. Findings were byte-identical. Medium was unchanged (same run
as above). In the bench, the slot walk with n=16384 other-warp slots
dropped from 983 to 751 ms (1.3x). A commit legitimately tracks all of its
own warp's in-flight pipelined ops (T15), and that per-commit cost is
semantic, not the walk.

**Negative: re-attribution v1 (not landed).** Built on d719a9d to W6's
F1–F6 review.

- **Mechanism.**
  - Completed copies are watched from their first completion.
  - Every rise of a watched component in a warp's hb or a2g[Global] view is
    recorded, lane-precise. Rises already implied by an existing observer
    are skipped.
  - At GC, an eligible copy's witnesses are rewritten to a virtual actor
    whose record holds a decode snapshot and the observers. Eligible means:
    `Copy`, `done == 2`, every kept witness global, async-proxy and in the
    Global (or no) domain, and no retained phase record carrying the raw
    completion.
  - The slot is then reclaimed. Ordering against a virtual witness is
    answered from the observers' stamps: hb stamps in hb, a2g observers via
    cur's hb or a2g, hb observers only inside cur's a2g (bridged).
- **Correctness.** Exact.
  - Findings byte-identical on the 7 fixtures at 1/8/16/32 workers.
  - `racecheck_reattribution` 7/7 and `racecheck_retirement` 11/11.
  - The fresh-source slot guard improved from 128 to 26 against same-source
    21. That missed the original bar (+2), but the gap is constant (11/26/47
    against 7/21/46 at 32/128/256 iterations), so the bar is now +8
    (27df6cc), which v1 meets. The guard stays ignored while no
    re-attribution is in the tree.
  - W6's later guard `a3_read_from_release_carries_the_observation` (the
    observation travels through a strong release/acquire read-from, not a
    barrier payload) also passes against v1 (8/8).
- **It does not pay.**
  - On e24 it re-attributed nothing: every candidate still had shared-memory
    witnesses at each collection. Wall was 19.0 s against 16.6 s.
  - On medium, by round 1100: 82.9K of about 90K candidates were blocked
    because a retained phase record still held the raw completion. A record
    keeps 4 phases per object and per-expert barriers advance rarely.
    Another 7.3K still had non-global witnesses.
  - Only 5–39 copies were re-attributed per collection.
  - The observation hook (a pre/post clock diff around every warp sync event
    against about 90K watched actors) made medium about 5x slower: 931 s
    against 186 s at round 1100.
- **Files (scratch).**
  - `scratchpad/w5_reattr_v1.patch` (racecheck/{cell,checker,clock,tuning}.rs
    and checker/partition.rs);
  - `scratchpad/w5_reattr_v1_reattr.rs` (the new `checker/reattr.rs`).
  - Both are in session 6ce3d6ba's scratchpad,
    `~/.cache/claude-code-tmp/claude-2792/-localhome-local-hongyij-TIRx-harness/6ce3d6ba-272f-405d-861d-e0b86a77dccc/scratchpad/`.

**The only known path, not pursued (B).** Free carrier-pinned slots by
making each phase record that carries a re-attributed completion an observer
in its own right.

- A per-record token component is raised in the record's completion clock,
  so "knows the token" stands for "acquired the raw completion".
- Observation then happens only where a raw completion enters a warp: phase
  waits and warp-target completions. Everything after that is covered by
  stamps, which removes the per-event diff.
- Tokens are new actor ids, one per carrying phase record rather than one
  per op. They need their own soundness case, in particular waits on a phase
  after it has been pruned from the record list. W6 should review before any
  code.

**Acceptance list for (B) (W6).** Whoever resumes (B) must cover each case
below with a guard in `tests/racecheck_reattribution.rs` (contract events,
run with `gc_every = 1` and the default period, identical payloads) before
building. The token must never order something the raw completion would
not, and must order everything it would.

1. **Wait after pruning.** A wait on a phase whose record has left the
   retained list is `BarrierPayloadUnavailable` (incomplete) today. The token
   must not turn it into "ordered".
2. **Parity aliasing.** A wait at generation g+2 with the same parity as g.
   Tokens are per generation, never per parity.
3. **Relaxed wait, later acquire fence.** A relaxed `test_wait`/`try_wait`
   parks the completion in `pending_acq` until a later `fence.acquire`.
   `pending_acq` is a carrier, so the token must ride it and be acquired only
   at the fence.
4. **Multicast.** One multicast completion lands in several CTAs' records.
   Each record's token is separate, and acquiring one says nothing about the
   others.
5. **Remote and cluster-barrier arrivals.** Remote `mbarrier.arrive` into a
   record, and a `barrier.cluster` record shared by many CTAs: the token is
   acquired exactly where today's phase payload is.
6. **tcgen05 commit forwarding.** A commit completion forwards its preds'
   `hb`/`tcgen_rel` into a record. The token covers what the commit
   forwarded, and nothing about uncommitted work.
7. **Fork/join.** A record's token is owned by the record's partition
   (its cluster), and the observation log merges in tag order (§11 D4).
8. **Scope-filtered arrivals.** A `.cta` waiter does not acquire a remote
   `.release.cluster` arrival (R4, `ScopeMismatch`). It must not acquire the
   token either.
9. **Tensormap path.** `fence.proxy.tensormap::generic.release` snapshots
   `hb` into `tmap_rel`, and the acquire moves it into `g2t` ranges. That
   path carries the raw completion, so the token must travel with it.
   Re-attribution v1's a2g/hb stamp rules were checked on barrier payloads
   and release heads only (`c3_*`, `a3_*`), not on this path.

Existing guards that must keep passing:
- `racecheck_reattribution` (C1, F1, F2, C3, a3);
- `racecheck_retirement`, with the fresh-source slot guard un-ignored;
- `racecheck_parallel_review`;
- the 7-fixture byte-identity check at 1/8/16/32 workers.

**Skipped: lane-entry normalisation (iii)(b).** In the medium late-run profile
(memo fix in, MAX_ROUNDS=2000), all lane-entry work adds up to about 12% of
samples:

- 8.8% self: `Clock::join`'s per-entry "already known?" scan over the
  incoming lane blocks;
- about 3% inclusive: `Lanes::with_blocks` and drop glue;
- under 0.5%: `raise_lanes` and `normalize_actor`.

Even removing all of it caps the gain at about 1.13x on the medium window,
below the 1.5x case criterion, so it was not built.

**FORK_JOIN decision run (W5, 2026-10-09): default flipped on, together with
W14's parallel phase-end collector.**

Setup: HEAD e5d1582, prof harness on the recorded fixtures, interleaved
runs, min of 3 (e24 at 16 workers: min of 6), host load average 6–11. The
"+ parallel GC" arm is W14's patch with `gc_threads = workers`.

| Case | serial | fork/join | fork/join + parallel GC |
| --- | --- | --- | --- |
| e24, 16 workers | 16.60 s | 7.68 s (2.16x) | 6.50 s (2.55x) |
| e24, 1 worker | 17.68 s | 17.46 s | — |
| medium (2000 rounds), 16 workers | 907.97 s | 555.28 s (1.64x) | 555.88 s (1.63x) |

The serial checker with the parallel collector takes 16.38 s on e24 (1.01x).

1-worker cost on the other fixtures, serial vs fork/join: fp16_bf16_gemm
1.00, deepgemm_1d1d 1.00, gdn 1.03, kda 1.02, stp 1.00, radix_topk 0.98.
Since W5-17a no fork is offered without a replay pool, so the milestone-2
1-worker slowdowns are gone. Their `perf_regressions.tsv` rows are retired
with a note line.

The rule was: at least 1.5x over serial on e24 at 16 workers, and no 1-worker
regression beyond noise. It is met.
