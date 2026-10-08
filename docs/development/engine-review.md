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
- **Fallback:** a cluster is a whole partition, so cluster barriers, DSMEM, remote arrives, `inbox_drain` and `cta_group::2` pairs stay inside one partition. Nothing else needs single mode. `route_outbox` silently drops unroutable messages (`partition.rs:403`); this predates fff0479.
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
