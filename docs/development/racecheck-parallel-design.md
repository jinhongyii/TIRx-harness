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
| D3 | **Partition index is not stable.** `Scheduler::partitions` is a `Vec` that loses entries at turnover (`partitions.remove(k)`), so index p names different clusters in different rounds. Tie-breaks, per-partition slot ranges and `JoinMemo` identity must key on the partition's **first cluster id** (`Partition::clusters[0]`, the same value the RNG seed and `AsyncId` range use), never on the `Vec` index or a pool-thread id. | open |
| D4 | **Join order.** `Observer::join(p, child)` must run after `par_for` returns, in partition order, as `merge_shard` does today. The merged finding list must be sorted by (key, representative `seq`), never in completion order. | open |
| D5 | **`max_findings` cap.** The serial checker stops recording after the cap, in `seq` order ("later findings were not recorded"). A per-partition cap would keep a different set. Rule: each partition keeps its earliest `max_findings` by `seq`, the merge keeps the global earliest `max_findings`, and the `incomplete` note fires iff the global count exceeds the cap. The global earliest N is a subset of the union of each partition's earliest N, so the result equals serial. | open |
| D6 | Per-cell processing order. A private cell sees its partition's events in `seq` order. A global cell (one stripe) receives pieces in replay order `(round, partition, seq)`, which is `seq` order. `EVICT_WINDOW` decisions are therefore identical to serial. | pass, if fan-out preserves `seq` order |
| D7 | GC at round merges (§8) uses a meet that is at most the serial checker's at the same `seq`. Retiring dominated witnesses changes no finding, but it can change which witness is reported if serial GC ran at a different point. Pin GC to round merges in the serial checker too, or prove reported witnesses never depend on GC timing. | open |

### 11.2 HB snapshots versus the engine's merge/replay rules (§4, §5)

The engine guarantees that within a phase, a partition reads global memory as of the phase start, except for bytes it wrote itself. The replay order puts any partition that read bytes another wrote in that phase **before** the writer (I11/I12). The snapshot claim holds if the checker never shows a partition anything the engine's shard did not show it. States that a checker partition could observe **earlier** than the engine's partitions do:

| # | State | Why the design exposes it | Rule | Status |
| --- | --- | --- | --- | --- |
| H1 | Release heads of global words written by a later-replayed partition in the same round | §5.4 runs the serial global-sync step **after** the parallel stripe apply. An acquire read by B (replayed before A because B read A's bytes at the round start) would then see A's same-round write as the latest morally strong write and acquire its head. That is a spurious edge and a missed race. | Each strong read is resolved against the cell state at its own `seq`. Either interleave the stripe apply and the sync step in `seq` order per stripe, or keep per-cell writes versioned by `seq` and pick the latest write with a lower `seq`. | open |
| H2 | `wait_until` `pred_reads` stability check | "Every retained write to those bytes is ordered before the wait" would also see same-round writes with a higher `seq` (applied in the parallel pass) and report `WaitPredicateReadsUnstable` where serial does not. | Consider only writes with a lower `seq`. | open |
| H3 | The serial phase and `drain_all` | The engine has up to three replay batches per round: the parallel phase, the serial phase (global RMWs, partition order, main arena), and `drain_all`. A serial-phase RMW of A reads B's same-round parallel-phase write. | The checker's "round" is the engine **phase**. Serial-phase events of A are processed after the global apply of that round's parallel phase, in partition order, each partition's word history merged before the next one runs (as S-b). | open |
| H4 | Events handed to children before the engine's merge | Under fork/join, events must reach `fork(p)` children only after `shard_replay_order` assigned `seq` and `merge_words` renumbered the buffered `WaitVerdicts` (W6-P1). Live delivery during the parallel phase would see pre-merge `observed`/`accepted` indices and no `seq`. | Children receive each partition's buffer at the same point `EventBuffer::replay` runs today. | open |
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
off until milestone 2 shows the gain. `phase_gc` stays default on (D7-gated,
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
