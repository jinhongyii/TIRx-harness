---
orphan: true
---

# NumSim engine review: interpreter and scheduler (HEAD fff0479)

Scope: `numsim-core/src/{interp,sched,arena.rs}`, tests, and benches. Line numbers refer to HEAD. Programs use `testutil::ProgramBuilder` (`b.`).

Each finding is labelled as follows:
- **C**: confirmed. Either it was traced through the code end to end, or it was reproduced with a scratch probe built from HEAD; the repository was not touched.
- **L**: likely. It comes from reading the code but was not executed.

## Critical / high

**H1. A named barrier that is waiting only on exited warps is reported as `Deadlock` (C).** `sync.rs:78-81`, `control.rs:275-317`, `sched/mod.rs:1096-1123`
- A `bar.sync` with no count expects `warps_per_cta*32` threads.
- `exit` reports to the cluster barrier only, and the deadlock classifier exempts only divergent warps.
- sync-semantics §3 G8 requires this hang to be *incomplete*. PTX §9.7.14.7 releases the barrier.
- Trigger: 64 threads; `b.compare(Ge, U32, p, warp, 1); b.if_(p); b.exit(); b.end_if(); b.bar_sync(0); b.exit();`. The result is `RunStatus::Deadlock`, which is a false error.
- Fix: in the deadlock classifier, return `Incomplete("G8")` when a blocked `Named{cta}` CTA has an exited warp (or implement exit release).

**H2. A bounded loop with a failed poll is spin-parked and becomes a false `Deadlock` (C).** `control.rs:209-213`, `mod.rs:1096`
- Parking assumes that any iteration with a failed poll and no progress is a spin.
- A `for k<3: r = mbar_test_wait(bar, 0)` probe (bar initialised, never arrived) parks in round 2 with no progress. The run is declared Deadlock even though the next retry would exit the loop.
- sync-semantics §1.4 says "a single spinning warp is not proof".
- Fix: park only if the register file hash at this `LoopEnd` equals the hash at the previous parked `LoopEnd` of the same frame (a true fixed point). Otherwise treat the parked retry as possible progress.

**H3. `cp.async.bulk.wait_group.read N` publishes the read milestone for the N younger, uncovered groups (C, probe).** `async_copy.rs:270-282`
- `retired || read` covers every member group of the resource.
- Trigger: lane 0 issues two s2g `BulkCopy{completion: Group}`, each followed by `AsyncCommit{Bulk}`, then `AsyncWait{Bulk, n:1, read:true}`. Both ops get `AsyncComplete{Read}`, which creates a false HB edge and hides a WAR race on the TMA-store source.
- Fix: cover only the ordinals of `wait_prefix_len(state, n)`.

**H4. `mbar_arrive` gives every target's `Arrive` event the full active mask (C, probe).** `sync.rs:333` (the multicast path is the same)
- Trigger: `b.binary(Shr,U32,i,lane,4); b.smem_addr(m,bar,i); b.mbar_init(m,16); b.mbar_arrive(m,None); b.mbar_wait_parity(m,0)`. Both `Arrive` events show `lanes=0xffffffff`.
- Effect: lanes are credited with releasing on barriers they never touched, so racecheck misses races.
- Fix: pass each target's own lane mask to `sync_event`.

**H5. `atom` / `red` on `.b128` panics (C, probe).** `mem.rs:331-333`
- `rmw_bytes` copies `eb=16` bytes into `[u8;8]`.
- Trigger: `Instr::Atom{op: Exch, ty: B128, ..}`. The result is `Internal: panic` from a validated program.
- Fix: do a byte-wise exch/cas for `eb>8`, or fail closed as `Unsupported`.

**H6. The serial phase lands every ready async op every round, so `CompletionPolicy::Seeded` behaves like `Eager` (C).** `partition.rs:324-335` → `land(None, .., all=true)`
- `run_serial` runs for every partition every round, including the single-partition path.
- No async op survives past the end of the round that issued it, so seed-dependent missing-wait bugs no longer surface. This is a regression in fff0479.
- Trigger: `cp_async_copy` with the `commit/wait_all` removed and a few register ops before the `ld`; `quantum:1`, `ValidityPolicy::Error`, seeds 0..32. No seed fails.
- Fix: in `run_serial`, land only the deferred global reductions, and only in shard mode.

## Medium

**M1. An inner loop clears the poll record when it exits by `break`, so the enclosing spin never parks (C).** `control.rs:204-207`
- Trigger: `while(!ok){ for s<2 { p=test_wait(bar[s]); if(!p){break} } }` on barriers that never complete.
- The inner `LoopEnd` (all lanes broken) resets `poll`, so the outer iteration has no failed poll. The result is a livelock until `loop_budget` (2^24 outer iterations), reported as Budget, not Deadlock.
- Separately, an inner loop's per-iteration `take(poll)` parks *inside* a bounded `for` loop (one round per stage).
- Fix: on loop exit, merge the inner `PollState` into the parent iteration instead of clearing it.

**M2. Spinning on a plain or acquire `Load` never parks (C).** `program.rs:1654` (`Load` is non-progress but never calls `note_failed_poll`)
- `while(ld.acquire(flag)==0)` (stream-K semaphores, hand-rolled grid barriers) burns 64 iterations per round per warp.
- A flag that is never set ends as `Budget` after about 16M iterations, not as `Deadlock`.
- Fix: have `ld.acquire` / `ld.volatile` / `ld.relaxed.{gpu,sys}` in a loop note a failed poll on `Word{alloc, span}`.

**M3. A lane-varying `mbar_wait` / `test_wait` emits one `Protocol` per target (C, probe).** `sync.rs:409-412`, called at `:470` and `:501`
- W2-5 says one event per instruction. Each extra event bumps `sync_seq`, so synccheck sees two committed instructions.
- Fix: collect `ProtocolCmd`s with per-target `observed_parity`, then emit once via `protocol_cmds`.

**M4. A lane-varying `mbar_wait` does not latch completed targets (L).** `sync.rs:494-498`
- Target A is Ready and target B blocks, so the whole warp retries. If A advances two phases meanwhile, A's lanes re-block (and set `armed`), although on hardware they already left. This can give a false deadlock or strict-consumption error.
- Fix: latch per-lane completion in `resume`, as `wait_until` does.

**M5. `st.async` / `red.async` drop `sem` and `scope` (C).** `async_copy.rs:682-735`, `partition.rs:579-588`
- The landing `Access` is `Weak` with no release head, so a peer's `ld.acquire.cluster` gets no edge. The result is a false race or `WaitExitUnproven`.
- Fix: carry `sem`/`scope` in `AsyncMeta` and emit the landing as an atomic release.

**M6. The fast-path `load` skips `capture_reads` (C).** `mem.rs:92-128` vs `support.rs:490`
- A `wait_until` predicate that loads from a bound buffer never records its reads, so `WaitPredicateReadsUnstable` cannot fire. Fast and slow paths diverge.
- Fix: `fast_target` returns `None` when `aux.capture_reads.is_some()`.

**M7. Direct `tcgen05.ld` / `tcgen05.st` never check that the TMEM columns are allocated (L).** `tcgen.rs:395-501`
- The buffer path fails closed (`support.rs:416`), but a direct `st` to deallocated columns writes silently.
- Fix: call `tmem_live(col, ncols)` per plan piece.

**M8. A panic on the main thread with `workers>1` hangs the process (C).** `sched/mod.rs:891-897`, `pool.rs:55-60`
- Only a normal return reaches `pool.shutdown()`; `thread::scope` then joins workers that are blocked on `work`.
- Reachable panic sources:
  - observer panics during replay;
  - `partition.rs:451` `expect`;
  - arena asserts in `admit`.
- Fix: put `shutdown` in a drop guard.

**M9. Divergent-switch limits (C, by design but unreported).** `mod.rs:671-674`
- `if lane!=0 { wait(A) }; if lane==0 { arrive(A) }` (no `Else`), and loop-exit divergence (`for i<lane { wait }`), both end as `divergent_block` Incomplete.
- Swap-blocked warps report the *other* arm's resource in `Blocked(r)`, which misattributes deadlock evidence.
- Fix: when swapping, record `Blocked` with the resumed arm's resource. Document the no-`Else` gap in README.

**M10. The observer stream's reads-from order disagrees with the values actually read (C).** `partition.rs:79-107`
- Partition k+1's same-round `ld.acquire flag` read the round-start snapshot (0), but replay places it after P0's `st.release flag=1`. Racecheck credits an edge that was never observed.
- Fix: emit a per-round marker; checkers ignore same-round cross-partition writes for reads-from.

**M11. Declared-word history silently truncates at `MAX_WORD_HISTORY` (65 536) (C).** `aux.rs:173`
- After that many writes, later writes are not logged, so `wait_until` verdicts are computed on stale history. This can cause a false hang or a wrong verdict, e.g. for a tile-scheduler counter in a persistent kernel.
- Fix: on overflow set a flag and fail closed (`Unsupported`) at the next verdict.

## Low

- **L1.** `tcgen05.alloc` re-emits `WarpSync` on every blocked retry (`tcgen.rs:125-147`). Fix: emit only on the committing attempt.
- **L2.** `wait_until` predicate loads emit an `Access` on every failed poll (`sync.rs:729`). Fix: buffer them and emit only on latch.
- **L3.** `tcgen_ld` / `tcgen_st` emit `AsyncIssue` / `Access` before `step_all` (`tcgen.rs:414-454`); `mbar_tx` Complete (`sync.rs:~370`) and `cp_async_mbar_arrive` (`async_copy.rs:319-346`) step targets one at a time, so a later failure leaves earlier commits unlogged. Fix: step first, with `step_all`.
- **L4.** Event order is inconsistent: `bar.sync` emits `Arrive` before `Protocol`, while `mbar_arrive` / `cluster_arrive` emit it after. Fix: always emit after the `Protocol`.
- **L5.** `isspacep.shared::cta` is true for a peer CTA's DSMEM generic address (`mem.rs:502`). Fix: also require `rank == own`.
- **L6.** `own_shared` masks offsets ≥ 2^24 instead of erroring (`support.rs:261`), so `addr_of(smem, 1<<24 + k)` silently aliases offset `k`. Fix: use `shared_addr(..)` and return `BadAddress` on `None`.
- **L7.** The budget is checked before the all-broken exit (`control.rs:195`), so a loop that breaks out on iteration `budget+1` errors. Fix: test `next.is_empty()` first.
- **L8.** A serial-point `atom` runs `begin_instr` twice (instrs, epoch, clock) and its parked `Yield` counts as progress (`mem.rs:369-376`, `partition.rs:262`). Fix: do the shard check before `begin_instr`, and treat the yield as non-progress.
- **L9.** On a partition error, later partitions' private state and stats have already advanced (`sched/mod.rs:1007-1024`). Fix: exclude them from `finish()`.
- **L10.** `exit` contains a dead `let _ = SyncKind::WarpSync{..}` (`control.rs:311-313`), so no event is emitted. Fix: delete it, or emit as intended.
- **L11.** `%globaltimer` / `%clock` are per-warp `steps`, so the global timer is not monotonic across warps (`alu.rs:103-105`). Fix: derive the global timer from `counters.instrs`.
- **L12.** TMA resolves its smem destination with `len=1`, and `cp.async` does no alignment check (`async_copy.rs:~560`, `:172-187`). Fix: resolve the full box and check alignment.
- **L13.** Rank-tagged `shared::cta` addresses (W2-2) mean any kernel arithmetic on high address bits (an unmasked `addr>>4` placed into a UMMA descriptor, `addr < limit`) differs from hardware on rank > 0. Fix: lint lowering constants and descriptor builders for unmasked use.

## Robustness and performance

- **P1.** `async_wait` scans the launch-wide `groups.members` map once per lane, twice (`async_copy.rs:270`, `:307`). In non-observing mode, and for `.read` waits, `members` is never pruned, so the map grows without bound and each scan is O(n²). Fix: nest the map by resource and prune on every wait.
- **P2.** `emit_verdicts` clones the whole word history on every successful wait (`sync.rs:783`). Fix: borrow a tail slice.
- **P3.** Vecs are allocated on every instruction even when `!observing`:
  - `emit_accesses` `spans` / `raw` (`support.rs:612,624`);
  - mbarrier `pcmds` / `all_cmds.clone()` (`sync.rs:309`);
  - `read_words` / `write_words` per lane (`warp.rs:156,173`);
  - `step` / `protocol` `vec![]` arguments.

  Fix: use `SmallVec`, or gate on `observing`.
- **P4.** `uninit_seen`, `mbar_reports`, `bar_red`, `bar_aligned`, `wg_credited` and `verdicts` are keyed by generation or span and are never pruned. Fix: retire entries when their generation completes.
- **P5.** The single-thread round still pays `run_serial` and `land(all)` on every partition every round (see H6).
- **Codegen parity:** no issue found. `if_` / `loop_begin` read `ctx.warp.pc`, which `end_instr` keeps current in both backends; the verdict cache key `(warp, pc)` is backend-independent.
- **Fast paths in `mem.rs`:** OOB, uninit and shard bailout match the slow path; the only gap is M6.
- **Stale documentation:** `sched/mod.rs:42-73` still says "parallel: not implemented" and describes re-execution on stripe conflicts, which contradicts W2-11.

## Scheduler soundness answers

- **Same-stripe writes:** sound. `written` is a per-byte bitset; for the same bytes the later partition wins, per W2-11. Validity is merged per byte (`arena.rs:813-837`).
- **RMW routing:** `atom` / `red` / `cas` (one handler), bulk and tensor reductions (`is_global_reduce`, `partition.rs:412`) are all serial points. `red.async` targets DSMEM and stays private. `multimem` is rejected by lowering. No bypass found.
- **Replay determinism:** deterministic and independent of the worker count. It is *not* single-partition order; see M10.
- **Fallback:** a cluster is a whole partition, so cluster barriers, DSMEM, remote arrives and `cta_group::2` pairs stay inside one partition. Nothing else needs single mode. (The inbox/outbox routing and its silent drop of unroutable messages are deleted; cross-CTA effects apply at issue.)
- **Pool:** the rotation is seeded by (seed, round, CTA); there is no dependence on thread identity.

## Test gaps: the 10 most valuable scenarios

The existing `workers_do_not_change_results_or_streams` test compares 1 worker with 2/8/33, but shards are created whenever there is more than one partition, regardless of the worker count. Both sides therefore run the same shard model, and the test cannot catch shard-semantics bugs.

1. **G8 exit barrier.** Warps ≥ 1 exit, then `bar_sync(0)`. Expect Incomplete or Completed, never Deadlock (H1).
2. **Bounded probe loop.** `for k<3 { test_wait }` on an unarrived barrier, then `exit`. Expect Completed (H2).
3. **Nested spin with break.** M1's program with an unreachable barrier. Expect Deadlock in fewer than 10 rounds, not Budget.
4. **Load-flag spin.** CTA 0 sets a flag after a long loop; CTA 1 spins on `ld.acquire`. Run once with the flag never set (expect Deadlock) and once with a different cluster per CTA (expect cross-partition visibility one round late) (M2, M10).
5. **Seeded latency survives rounds.** Missing `wait_all` before an `ld` of cp.async data. Expect some seed in 0..32 to see uninit (H6).
6. **`wait_group.read 1` with two bulk groups.** Assert exactly one `AsyncComplete{Read}` (H3).
7. **Lane-split mbarrier.** `bar[lane>>4]`, arrive and wait. Assert per-target `Arrive` lanes and a single `Protocol` (H4, M3). Add a variant where target A advances two phases while B blocks (M4).
8. **Divergent nesting stress.**
   - `if lane<8 { wait A; arrive B } else { if lane<16 { arrive A; wait C } else { arrive C; wait B } }`, which needs a three-way swap.
   - The same body inside a loop with `continue` in one arm.
   - Assert completion, each arm running exactly once, and stores in order.
9. **Shard semantics vs single.** Run a race-free multi-cluster program with and without forced single mode. Assert identical outputs. Add an observer that panics with `workers:2` and assert it does not hang (M8).
10. **Robustness pack.**
    - `atom.exch.b128` (H5);
    - `tcgen05.st` after dealloc (M7);
    - more than 65 536 writes to a declared word followed by `wait_until` (M11);
    - `addr_of(smem, 1<<24)` (L6);
    - a loop that breaks out on iteration `budget+1` (L7).

## Engine `incomplete` reasons (inventory, 2026-10-08)

A run is `incomplete` when coverage cannot be established. This is never `clean`, and never an error, because nothing provable is wrong with the kernel. `sched::classify` turns every `ExecErrorKind::Budget`, `ExecErrorKind::Unsupported` and `Op(OpErrorKind::Unsupported)` into `RunStatus::Incomplete { reason: "<kind>: <message>", site }`. The scheduler also emits three reasons of its own.

| reason (prefix / text) | emitted by | when |
| --- | --- | --- |
| `round budget of N exhausted` | `sched::Scheduler::run` | `RunConfig::max_rounds` reached |
| `named_barrier_after_exit (G8): …` | deadlock classifier | no progress while a warp waits on a named barrier of a CTA with exited warps; explicit-count release by exit is not modelled |
| `divergent_block: …` | deadlock classifier | no progress while a warp is blocked with a divergent mask that structured SIMT cannot interleave |
| `cross_cluster_same_round_cycle` (finding attr `reason`) | `Scheduler::stream_cycle` | two clusters each read global bytes the other wrote in one round; the observer stream cannot be ordered faithfully, so checker verdicts are incomplete |
| `Budget: loop exceeded its iteration budget of N (raise loop_budget)` | `control::loop_end` | per-loop `RunConfig::loop_budget` exceeded |
| `Unsupported: <reason>` | `Instr::Unsupported` (`control::unsupported`) | the lowering emitted an explicit unsupported form; the reason string comes from the module |
| `Unsupported: not modeled: <what>` | `support::unsupported` | see the next table |
| `Unsupported: <op> <mods>: <msg>` | `alu::ptx` | an oplib generic op rejected at load or at run (`unresolved generic op`, unmodelled carrier/form) |
| `Op(Unsupported): <msg>` | every `support::op_err` | an oplib form that is not modelled (W4 domain): tcgen/TMA/MMA descriptors, tensor-map fields, cvt forms, … |
| `Unsupported: launch bounds cannot provide the minimum 24 registers per thread required by setmaxnreg` | `Scheduler::initial_regs_per_thread` | launch bounds too tight for the register pool model |
| `Unsupported: declared-word history exceeded N writes; …` | `sync::wait_until` | `MAX_WORD_HISTORY` overflow under a history-consuming observer |
| `subset_execution` | checkers (`RunOutcome::subset` echoed) | a `RunConfig::subset` run covers only part of the grid |

`support::unsupported` call sites (`Unsupported: not modeled: …`):

| what | site |
| --- | --- |
| `const state space` | `support::resolve_buf` |
| `<buf>[i]: a sub-word tmem vector spanning cells is not modelled` / `<bits>-bit tmem elements are not modelled` / `tmem access crosses a lane row` | `support::resolve_buf` (TMEM buffer views) |
| `address of a non-addressable buffer` / `… tensor-memory buffer` / `… register-space buffer` | `support::buf_generic_addr` (`AddrOf`) |
| `sub-byte atomics` | `mem::atom` |
| `cvta of const/tmem` / `cvta to const/tmem` | `mem::cvta` |
| `mapa in this state space` | `mem::mapa` |
| `copy report .per_16bytes without its pattern (lower to Per16BytesPattern, W2-8)` | `async_copy::check_report` |
| `tcgen05.ld .spcompress without its max/min op` | `tcgen::tcgen_ld` |
| `tcgen05.mma .lut_b address outside a live TMEM allocation` | `tcgen::tcgen_mma` |
| `wait_until on a word wider than 64 bits` | `sync::wait_until` |

## `Report.meta.timing` phases (milliseconds)

| key | measured by | covers |
| --- | --- | --- |
| `lower` | `v2.compile` (`CompiledModule.lower_ms`, cache miss) | TIR -> module lowering + validation |
| `module_cache` | `v2.compile` (`lower_ms`, `cache_hit`) | module load from the module cache (exactly one of `lower` / `module_cache` is non-zero) |
| `bind` | `v2.run` | input canonicalization and binding (host prelude, descriptor patching, address planning) |
| `build` | `numsim-py::execute` (`backend_for`) | backend selection, i.e. codegen build or cache load for `Backend::Codegen` (about 0 for the interpreter) |
| `run` | `numsim-py::execute` | engine execution. **Includes online racecheck**: `RaceObserver` consumes events during the run |
| `check` | `numsim-py::execute` | post-run checker work: synccheck exploration, racecheck `finish` |
| `report` | `v2.run` | Python report assembly |

## Partitioning and the single-partition fallback (2026-10-08)

Every launch runs as CTA-lockstep partitions, one per resident cluster. Each partition has its own copy-on-write overlay (stripes) of global memory, and the partitions are merged deterministically at the end of each round (redesign.md §2.3). A global word written by another cluster becomes visible one round later. That includes `wait_until` polls and `sync_words` words. Results and observer streams do not depend on the worker count: `partitioned_wait_until_is_deterministic_across_workers` checks this at 1, 8 and 32 workers.

**Declared-word history (`wait_until` verdicts).**
- The launch-wide `WordTable` is authoritative. A partition runs each phase on a copy and logs its own writes.
- `Scheduler::merge_words` appends each partition's new entries, rebased onto the launch table's last image, after:
  - the parallel phase, in event replay order;
  - the serial phase, in partition order;
  - `drain_all`, in partition order.
  It then refreshes the partitions' copies.
- Verdict indices equal the delivery order because the replay order puts a reader of bytes another partition wrote this round before that writer. A read/write cycle is reported as `incomplete` (`cross_cluster_same_round_cycle`).
- The history exists only for history-consuming observers. Nothing program-visible depends on it (`partitioned_words_do_not_depend_on_the_observer`).

**Single-partition fallback.** These forms run the whole launch as one partition. The merge/replay protocol cannot order them yet; they are documented, not `incomplete`.

| form | why one partition |
| --- | --- |
| `cooperative` topology / any `grid.sync` | the grid barrier is launch-wide state that all CTAs must reach in one schedule |
| `RunConfig::max_resident_ctas == 0` | every CTA resident at once (cooperative launches) |
| kernels mixing `tcgen05` `cta_group::1` and `::2` | the kernel-wide tcgen05 `cta_group` rule is checked across all CTAs |
| `RunConfig::single_partition` | explicit request (tests, debugging) |

**Decision: CTA-level (intra-cluster) partitioning is cancelled (2026-10-08, coordinator).** The design existed: peer shared memory as copy-on-write shared state, remote mbarrier ops applied in (sender rank, issue seq) order, and `shared::cluster` atomics as serial points. It is not implemented, because HEAD is already faster than legacy on the headline kernel. `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in`, NumSim `Engine.run`, interp backend; host load 67–131, so absolute times are noisy:

| workers | v2 (HEAD e6dafca + worktree) | legacy |
| --- | --- | --- |
| 32 | 4.79 s | 12.63 s |
| 8 | 9.31 s | 17.80 s |
| 1 | 53.4 s | 73.2 s |

W10's earlier 73 s at 32 workers came from an engine 84 commits older. Large single-cluster kernels therefore keep the cluster-level partition.

## Mega MoE max config: where the time goes, and the spin-poll skip decision (2026-10-08)

Case `t8192_m8192_h7168_i3072_e384_k6_g1`, `Engine(max_workers=16)`, NumSim mode. Measured with temporary counters on private builds; the numbers are CPU-seconds summed over threads.
- **Partitioning.** 74 partitions in every one of 74,478 rounds, so no single-partition fallback. The serial phase is 0.5 s, turnover 0.9 s, host allocation of inputs 9.2 s.
- **Balanced load.** Partition time totals 3,460–3,712 CPU-s over balanced partitions (about 53 s each for the busiest), against a 203 s critical path. The run is throughput-bound, not serialized.
- **Split of partition time.** Warp execution (`run_cta`) 1,326. `land` 2,074, of which:
  - MMA arithmetic (`run_mma` → `oplib::tc_mma_ctas`, block-scaled mxf8f6f4): 1,309–1,356 (W4);
  - TcgenCp landing: 324 → 131 after the direct-byte fast path;
  - Copy landing: 163 → 114 after gathering without a `Vec` per span;
  - landing scan: 86 → 75 with the lazy live-id set.
  `apply_completions` is 4.
- **Spin polls.**
  - 72% of partition-rounds start with every warp blocked, but only 11% make no progress at all (48 CPU-s).
  - 151 M slices re-poll a parked warp that stays blocked (288 CPU-s).

**Decision: no spin-poll skip (coordinator).**
- A partition-level skip would save at most 48 CPU-s.
- A per-warp resource-version skip (≤ 288 CPU-s, about 9%) would have to be disabled whenever an observer is attached, because parked retries emit poll events. That makes it an execution path that exists only without an observer, which the one-execution-path rule forbids.
- The lever is the MMA arithmetic (W4) and interpreter hot paths (W13).

### Mega MoE medium: per-worker CPU inflation (W2 split, W13 follow-up)

**W2's phase split (NumSim mode).**

| Workers | Outside partition work | Partition critical path |
| --- | --- | --- |
| 1 | ~0.3 s | 3.1 s |
| 16 | ~0.3 s | ~1.5x the 1-worker partition CPU |
| 32 | ~0.3 s | 5.7 s (~2.3x the 1-worker partition CPU) |

Everything outside partition work stays at ~0.3 s for every worker count. The scheduler is not the cost; the partitions' own CPU is.

**Hypothesis tested: register-file footprint (~910 KB per warp) causes the inflation. Not confirmed.**

Measurement setup:
- private build of HEAD 65a1be1, NoopObserver;
- "partition CPU" = thread CPU time summed inside `Partition::run_round`;
- host load 3–6; min of several runs.

Partition CPU at 16 and 32 workers, relative to 1 worker:

| Case | Register file per warp | 16 workers | 32 workers |
| --- | --- | --- | --- |
| synthetic loop, 128 CTAs × 4 warps, 16 registers live | 4 KB | 1.33x | 1.37x |
| synthetic, 256 registers live | 64 KB | 1.39x | 1.71x |
| synthetic, 4,096 registers live | 1 MB | 1.30x | 1.65x |
| recurrent_kda_decode_one_warp (2,048 one-warp CTAs) | 280 KB | 1.45x | 1.55x |
| gdn_decode_bf16_wide_vec_mtp (128 CTAs × 4 warps) | 370 KB | 1.45x | 1.54x |
| mega_moe e24 | 910 KB | 1.84x | 3.0x |

Kernels with 4–370 KB register files inflate about as much as 1 MB ones at 16 workers. Footprint is therefore not the main driver; it at most adds to e24 at 32 workers.

Ruled out:
- **Shared frequency or bandwidth limits:** 16 or 32 independent single-worker processes running concurrently show no inflation (1.00–1.05x).
- **Partition-to-thread migration:** static partition-to-thread assignment and/or pinned threads leave recurrent_kda at 1.4–1.6x.
- **Allocator trimming / munmap:** `GLIBC_TUNABLES` mmap and trim thresholds at 4 GiB change nothing.
- **System time:** it is small (e24: +0.27 s at 16 workers, against +1.4 s of inflation).

Partly explained:
- **Worker sleep between rounds.** Busy-waiting up to 2 ms for the next round instead of sleeping on the condvar cuts e24 at 32 workers from 5.0 to 3.2 CPU-s (wall 0.46 → 0.33 s), but leaves recurrent_kda unchanged.

Open: the remaining ~1.3–1.5x is an intra-run effect: it is absent across processes and independent of footprint. Hardware counters are unavailable on this host (`perf_event_paranoid=4`), so it is not attributed yet. Compact register storage stays a noted, unstarted lever (see "Interpreter hot paths"), not a fix for this inflation.

**Spin-before-park in the worker pool: measured, not landed (below the 1.5x bar).**

Prototype (private build of a6c5202, `sched/pool.rs` only; the tree is untouched):
- Before parking on the condvar, a worker busy-polls a lock-free copy of the round generation for up to a fixed budget. The caller does the same for the round's end.
- Variants: pure `spin_loop` for 500/1000/2000/5000 µs, or `yield_now` polling for 1000 µs.
- Results do not depend on it: it only changes when a thread notices the next round.
- A 1-worker run has no pool and never spins (e24 at 1 worker: 1.64 → 1.60 s).

Measurements:
- Wall and process CPU, min of 1–5 runs per variant, interleaved with the baseline.
- Host load 6–20 (not near-idle), NoopObserver.
- Medium = `mega_moe_t64_h2048_i1536_e96_k4_g1`.

| Case | Workers | Baseline wall / CPU | Best spin variant wall / CPU | Wall gain |
| --- | --- | --- | --- | --- |
| mega_moe medium | 32 | 8.15 s / 103 s | yield 1 ms: 5.83 s / 140 s; spin 500 µs: 5.92 s / 113 s | 1.40x |
| mega_moe medium | 16 | 7.12 s / 67 s | spin 2 ms: 7.24 s / 98 s | 1.0x |
| mega_moe e24 | 32 | 0.364 s / 4.05 s | yield 1 ms: 0.288 s / 7.05 s | 1.26x |
| mega_moe e24 | 16 | 0.364 s / 2.98 s | spin 500 µs: 0.359 s / 4.12 s | 1.0x |
| recurrent_kda_decode_one_warp | 16 / 32 | 0.128 / 0.084 s | spin: 0.129 / 0.091 s; yield: 0.124 / 0.088 s | 1.0x (pure spin up to 8% slower) |
| gdn_decode_bf16_wide_vec_mtp | 16 / 32 | 0.027 / 0.025 s | spin: 0.030 / 0.029 s; yield: 0.029 / 0.029 s | 0.9x |

Conclusions:
- Only 32-worker Mega MoE gains: up to 1.40x wall on medium.
- With spinning, medium at 32 workers (5.8 s) beats the 16-worker baseline (7.1 s). Without it, 32 workers is slower than 16.
- Every spin variant raises process CPU by 10–90%.
- Pure spinning makes the small kernels 5–15% slower at 16 and 32 workers, likely by stealing SMT-sibling cycles; polling with `yield_now` keeps them neutral.
- No criterion reaches 1.5x, so `pool.rs` stays as is.
- The loaded-host check (another 16-worker job running alongside) was not run, because the change does not land.
- **Lever, if 32-worker Mega MoE matters:** a per-round handoff that avoids the condvar park/wake. For example, workers that stay on a partition across rounds until the serial phase. That is a scheduler design question, not a spin budget.

### Partition lookahead (workers keep partitions across rounds): design and verdict (W13, 2026-10-08)

**Question.** Mega MoE at 32 workers is slower than at 16 (medium: 8.15 s against 7.12 s). Would a scheduler where each worker keeps its partitions and runs them through several rounds help? That means no park/wake and no barrier between rounds, until a partition needs merged state.

**Answer: no, not as an exact protocol.** In Mega MoE nearly every round carries cross-partition state through the serial phase. A lookahead that must fall back whenever a partition needs merged state would fall back almost every round. Measured below; not built.

#### 1. What the round boundary does today (`sched/mod.rs` `run_loop`)

One round, in order:
1. **`parallel_phase`:**
   - Each partition gets an arena shard: a copy-on-write overlay of the shared allocations, with reads tracked when observing. `Partition::run_round` runs each partition on the pool; `par_for` hands out partition indices dynamically.
   - **Barrier:** the caller waits until every partition is done.
   - The first failing partition discards the later ones.
   - When observing, `Arena::shard_replay_order` computes the replay order: a partition that read bytes another wrote this round goes first; a read/write cycle reports `incomplete`.
   - Readonly-proxy conflicts are checked, then shards are merged in partition order: last writer wins, byte granularity.
2. **`merge_words(order)`:** each partition's new declared-word history entries are appended in delivery order, buffered `WaitVerdicts` indices are renumbered to the merged history (W6-P1), and every partition gets the merged table back.
3. **`replay_partitions(order)`:**
   - `Access::seq` is assigned in replay order.
   - Partitions are offered to `Observer::fork`; children replay on the pool, are joined in replay order, and the rest replay serially.
   - Then one `phase_end`.
4. **`serial_phase`:**
   - Partition by partition, on the main arena, run the work parked at a serial point: global read-modify-writes inside a shard (atomics, reductions, CLC `try_cancel`) and deferred landings.
   - Each partition's events are replayed and its words merged before the next partition runs (W6 S-b), then `phase_end`.
5. **`turnover`:**
   - Retire finished clusters and their partitions; collect leftover sync state.
   - Admit pending clusters: new partitions, register files allocated or deferred, declared words, register-pool configuration.
   - CLC claims are drawn from the launch-wide `Arc<ClcTasks>` queue during step 4.
6. Bump `round`. If nothing made progress: `drain_all` (land everything ready); otherwise the deadlock/stuck diagnosis.

**What a partition's round r+1 depends on, other than its own state:**
- the shared allocations as merged after round r: other partitions' parallel-phase writes, plus every serial-phase write;
- the declared-word table: only for observer verdict bookkeeping, never program-visible;
- the CLC queue and admission (turnover).
- Event replay and `phase_end` do not feed back into execution.

#### 2. The design that was considered

- **Execution.** A worker owns a fixed set of partitions and runs them round after round. Each partition buffers its round-r events and shard overlay under the round index.
- **Commit.** Round r of partition p is committed (merged, words merged, replayed) by a sequencer thread in the same order as today: partition order, `shard_replay_order`, delivery-order history.
- **Lookahead condition.** p may start round r+1 before the others finish round r only if nothing it will read in round r+1 can be changed by others' round-r work:
  - no other partition wrote, in its round-r shard, a stripe p reads;
  - the serial phase of round r did not run;
  - turnover changed nothing p reads.
- **Validation.** The first condition can only be checked after the fact: p's round r+1 read set against the others' round-r write sets. A failed check therefore needs a rollback of p's round r+1, i.e. a snapshot of p's warps, register files, private allocations and sync table taken before every speculative round. That is ~15 MB of register files per CTA on Mega MoE, too much for ~177 k partition-rounds. Without rollback, the protocol must be conservative: stop and wait whenever round r had a serial phase or p polls any shared word.
- **Invariant (why results and streams would not depend on the worker count).**
  - Every committed round is the lockstep round: p's round r+1 inputs are, by the lookahead condition, byte-identical to what the lockstep scheduler would give it.
  - Commits, replay order, `seq` numbering, history delivery order and `phase_end` placement are produced by the sequencer exactly as today, from per-round buffers.
  - Only *when* a partition computes changes, never *what* it computes or the order in which it is published.
  - **Fallback:** when the condition fails, p waits for round r's commit, exactly today's barrier.

#### 3. Dependency census (scratch instrumentation, private build of a6c5202)

- **Memory dependencies:** every shard tracked its reads. For each partition-round, I counted whether it read a stripe byte that another partition wrote in the previous round's parallel phase.
- **Serial dependencies:** I counted rounds whose serial phase ran.

| Case | Rounds | Partition-rounds | Read another partition's round r-1 shard writes | Rounds with a serial phase |
| --- | --- | --- | --- | --- |
| mega_moe e24 (t8_h1024_i512_e24_k2_g1) | 174 | 12,876 | 0 | 126 (72%) |
| mega_moe medium (t64_h2048_i1536_e96_k4_g1) | 2,391 | 176,934 | 0 | 2,148 (90%) |
| gdn_decode_bf16_wide_vec_mtp | 10 | 1,280 | 0 | 0 |
| recurrent_kda_decode_one_warp | 10 | 10,240 | 0 | 0 |

- In Mega MoE, all cross-partition communication goes through global read-modify-writes: dispatch/combine counters and flags. Those run in the serial phase, which ran in 72–90% of rounds. The partitions that wait on those words poll them every round.
- The conservative protocol would therefore resynchronize in 72–90% of rounds.
- The kernels with no dependencies (gdn_decode, recurrent_kda) finish in 10 rounds and are not barrier-bound.

#### 4. Cost model

Phase timings (scratch counters; wall of `par_for`, CPU per partition-round, NoopObserver; host load 10–15):

| Case | Workers | Wall | `par_for` wall | Σ partition CPU | Σ over rounds of max partition | Serial + turnover |
| --- | --- | --- | --- | --- | --- | --- |
| mega_moe medium | 16 | 6.5 s | 5.33 s | 59.4 s | 3.25 s | 0.11 s |
| mega_moe medium | 32 | 7.3 s | 6.05 s | 89.5 s | 5.11 s | 0.20 s |
| mega_moe e24 | 1 | 1.84 s | 1.80 s | 1.78 s | 0.15 s | 0.02 s |
| mega_moe e24 | 16 | 0.36 s | 0.27 s | 2.85 s | 0.18 s | 0.02 s |
| mega_moe e24 | 32 | 0.40 s | 0.32 s | 4.25 s | 0.27 s | 0.02 s |

Upper bounds for the parallel phase:
- **Lockstep:** each round costs at least its slowest partition, so `par_for` ≥ Σ over rounds of the max partition: 3.25 s at 16 workers and 5.11 s at 32 on medium.
- **Barrier-free, no dependencies:** `par_for` ≥ Σ partition CPU / workers: 3.71 s at 16 and 2.80 s at 32.
- **Best possible gain:** about 1.4x at 16 workers (5.33 → 3.71) and about 2.2x at 32 (6.05 → 2.80), and only if no round needed a resync.

With resyncs in 90% (medium) or 72% (e24) of rounds:
- Only the dependency-free rounds can overlap.
- Expected gain is at most about 1.1x on medium (10% of rounds) and about 1.3x on e24 (28%), before the sequencer and snapshot costs.

**Gate.** Not built: no criterion reaches ≥1.5x. The criterion would be `mega_moe medium` wall at 32 workers, min of 3 runs on a near-idle host, with digests identical at 1/8/32 workers.

#### 5. What would help instead (measured or estimated)

- **Cheaper barrier.** Spin-before-park, measured above: medium at 32 workers 8.15 → 5.83 s (1.40x), +10–90% CPU, small kernels neutral with `yield_now` polling. Below the bar on its own.
- **Per-partition CPU inflation.** This is the bigger factor: Σ partition CPU is 59.4 s at 16 workers and 89.5 s at 32, against ~36–40 s at 1 worker. Its cause is unattributed: it is not register-file footprint, migration or the allocator (previous section). Hardware counters (`perf_event_paranoid` ≤ 2) would be the next step.
- **Parallel serial phase.** Serial RMWs on disjoint words could run partition-parallel. The serial phase plus turnover is only 0.11–0.20 s on medium, so this would not move wall time.
- **Choice of worker count.** For Mega MoE medium, 16 workers already beats 32 (7.12 s against 8.15 s). A default of at most about partitions/4 workers costs nothing in correctness.

## tcgen05.mma cost: arithmetic, not the callback boundary (2026-10-08)

**Decision (coordinator): borrowed-view MMA I/O is not landed.** Rule: no optimization without a measured gain.
- **Background.** oplib calls the engine about 2,200 times per block-scaled MMA on Mega MoE e24 (544 shared-memory reads, 1,397 TMEM reads, 256 TMEM writes; about 41 KB). Stubbing those callbacks once seemed to halve `run_mma` (250 → 105–165 µs). That experiment was flawed: the stubs zero-filled the operands, which let oplib take cheaper arithmetic paths.
- **Measured with a correct A/B.** Whole-allocation borrowed views (`TcViews`, W2/W4) were bit-identical to the callbacks (all scenarios × 3 validity policies × 1/16 workers, plus the e24 and radix streams) but gave:
  - e24: 207–271 µs per MMA without views, 244–262 µs with them;
  - max config, same-moment A/B: MMA landing 1,530 → 1,412 CPU-s, but untouched Copy and TcgenCp also fell about 7%, which is host drift.
- **Also neutral, measured:**
  - a validated-run validity cache (2× worse);
  - moving TMEM out of the arena so callbacks are slices;
  - fixed-size small copies.
- **Conclusion.** The MMA cost is oplib's arithmetic per MMA. Engine-side callback work is not the lever. Don't retry views or callback micro-optimizations without a new profile that attributes time below `tc_mma_ctas` correctly (py-spy's native unwinding truncates there).


## Interpreter hot paths (W13, 2026-10-08)

**Results.** Every change keeps outputs, `RunStatus`, stats and observer streams bit-identical. That was checked by a digest of outputs plus every observer callback, with and without word history, over all `testutil` scenarios and the corpus fixtures, at 1, 8 and 32 workers. Each change has a criterion row in `numsim-core/benches/interp_hot.rs`; corpus rows read `examples/record_race_fixtures.py` fixtures. Exactness tests are in `tests/interp_hot_equivalence.rs`, plus unit tests beside `coalesce` and `subtract_spans`.

Single worker, before → after the whole series:

| Kernel | Before | After | Speedup |
| --- | --- | --- | --- |
| rmsnorm | 0.97 ms | 0.64 ms | 1.4x |
| deepgemm_sm100_fp8_gemm_1d1d | 9.5 ms | 5.4 ms | 1.7x |
| fp16_bf16_gemm | 75 ms | 57 ms | 1.3x |

Mega MoE e24 (`mega_moe_t8_h1024_i512_e24_k2_g1`) at 16 workers:

| Mode | Before | After |
| --- | --- | --- |
| Engine only, no observer | 652 ms | 379 ms |
| Observed (counting observer) | 1.85 s | 0.61 s |

The cost of being observed on e24 fell from ~1.2 s to ~230 ms.

| Change | Where | Bench row | Before → after |
| --- | --- | --- | --- |
| Exact memo of `WarpState::spin_hash` (memcmp against the last hashed state) and an 8-chain multiply-rotate hash | interp/mod.rs | `spin_wait_regs/pad768_iters4096` | 7.35 → 2.19 ms |
| Register file from one zeroed allocation (`vec![[0u64;32]; n]` cloned slot by slot) | interp/mod.rs | `admit_regs/ctas64_pad256` | 1.56 → 1.48 ms; 765 slots per warp 100 → 65 µs |
| Lane-parallel `read_special` / `read_param` (`%tid` stepped, not divided; one masked write) | alu.rs, support.rs | `read_special/iters256` | 66.2 → 17.5 ms |
| `tcgen_ld` register-major writes from the run images, with run offsets cached per map; chosen by data shape only | tcgen.rs | `tcgen_ld/x64_iters64` | 16.8 → 4.15 ms |
| Warp-uniform index fast path for register arrays | alu.rs | `reg_indexed/iters2048` | 12.0 → 4.74 ms |
| Vector stores up to 32 bytes on the store fast path | mem.rs | `store_v4/iters2048` | 14.6 → 5.1 ms |
| Resident-CTA count kept, not re-summed per admission | sched/mod.rs `turnover` | `corpus_numsim/rmsnorm` | 6.5% of rmsnorm removed |
| `run_mma` span `coalesce`: unstable sort, plus a bitmap union when spans are dense | partition.rs | `corpus_numsim/deepgemm_sm100_fp8_gemm_1d1d` | `coalesce` 14.7% → 7.6% of 1d1d |
| `subtract_spans` without quadratic splitting (union walk, empty-span cuts kept) | partition.rs | `observed_overhead/*/counting` | e24 observed, 1 worker: 9.0 → 2.3 s |
| Process-wide MMA shared-A footprint cache (each miss ran two MMAs) | tcgen.rs | `observed_overhead/fp16_bf16_gemm/counting` | 91 → 73 ms |
| In-place tcgen access items; event-buffer lane-span pool; sorted-input sort skip | tcgen.rs, partition.rs, support.rs | `observed_overhead/*/counting` | fp16_bf16_gemm 73 → 62.5 ms; e24 at 16 workers 845 → 606 ms |
| Register files ≥ 128 KiB zeroed at the warp's first step, on the worker | interp/mod.rs, sched/mod.rs `admit` | `observed_overhead/mega_moe…/noop` | 652 → 379 ms at 16 workers; small files unchanged |
| Unstable key sort in `emit_accesses`; event-buffer access counter | support.rs, partition.rs | `observed_overhead/fp16_bf16_gemm/counting` | ~1.5% |
| Incremental `spin_hash`: sum of per-slot hashes, re-hashing only slots written since the last call; every write goes through `WarpState::reg_mut` / `reg_write_raw`, and debug builds recompute the hash from scratch on every call | interp/mod.rs, alu.rs, support.rs, tcgen.rs, mem.rs | `corpus_numsim/kda_backward_packed`, `spin_wait_regs/pad32768_iters1024` | kda_backward_packed (34,649 registers, 8.7 MB per warp) 1.02 s → 0.121 s at 1 worker (legacy: 1.007 s); spin row 27.3 → 16.3 ms; 1d1d 4.24 → 3.97 ms |

Observer-gated fast paths were made observer-independent, so each path is selected by data shape alone and emits the same events: `tcgen_ld` direct copy, `RunImages` under predicate capture, load/store fast paths under word history, `tcgen_st`. The audit table is in the W13 reports.

**Rejected or deferred, measured.**
- **Register-file pool across runs** (zero on reuse): 3–5% on rmsnorm for 64 MB held per process. Not landed.
- **Deferring every register file to the first step:** e24 at 16 workers improved 1.4x, but `corpus_numsim/rmsnorm` went 0.66 → 3.37 ms and `admit_regs` 1.40 → 7.8 ms. Zeroing at admission gets fresh heap, so untouched pages are never faulted. Zeroing at first step interleaves with other allocations and becomes a full memset plus brk churn. Landed only for files of at least 128 KiB.
- **Malloc churn:**
  - sharing one `Arc<[BufBinding]>` across CTAs (3% of rmsnorm allocations): no measurable gain;
  - `partitions.reserve` in `turnover`: no measurable gain.
  Neither landed. Malloc is now 2–7% of a run.

**What remains on the observed path** (e24, 16 workers):
- `SyncEvent` clones into `EventBuffer` and their drop at replay: ~4.4%. Contract-bound: the observer receives an owned `SyncEvent`.
- `merge_shard` and `shard_replay_order`: ~3%, on W5's replay/fork-join path.
- `emit_accesses` copying lane spans into each `Access`: ~5%.

Engine-side, mega_moe is bounded by MMA arithmetic (previous section). The per-poll register-file scan of `spin_hash` is gone: polls now cost O(written slots).

**Noted, not started: compact register storage.** Mega MoE medium loses per-partition CPU as workers increase: 1.5x at 16 workers and 2.3x at 32 (W2's phase split). The likely cause is cache pressure from ~910 KB per-warp register files: one 64-bit × 32-lane slot per SSA register, with no reuse. Compact register storage (paged or sparse `RegFile`) or slot reuse in lowering would address it. Either needs a `RegFile`/lowering contract change.
