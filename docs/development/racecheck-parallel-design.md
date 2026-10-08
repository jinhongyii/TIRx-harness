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

## 11. Effort estimate

| Part | Estimate |
| --- | --- |
| Observer fork/join contract and scheduler plumbing (W2 + coordinator) | 2–3 days |
| Checker state split into partition / stripe shards, per-partition slot ranges, JoinMemo per partition | 3–4 days |
| HB handles on accesses and the round-merge global pass | 3–4 days |
| Serial global-sync step and the engine serial-point decision (§5.4) | 2–4 days, depending on the prototype |
| Deterministic finding merge plus GC/reclaim across shards | 2 days |
| Validation (bit-identical corpus at 1/16/32 workers, perf) | 2–3 days |

Total: about 3 weeks of focused work, gated by the §10 measurements.
