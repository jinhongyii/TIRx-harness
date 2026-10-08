---
orphan: true
---

# Adversarial review: racecheck and synccheck (committed code)

Scope: `numsim-core/src/{racecheck,synccheck}/` and `tests/{racecheck,synccheck}_*.rs` at `HEAD`.
`HEAD` moved from 8a0afff to 1e637b4 while this review ran. In both checkers the only change in between
is the u64-epoch adapter (`observer.rs` +12 lines), so line numbers are for 1e637b4.

Notes:
- The committed tree does not build. `value::WarpMask` duplicates `numsim_types::WarpMask`, and
  `interp/mod.rs:371` passes a u32 epoch where a u64 is expected.
- Items marked **[confirmed]** were reproduced on a scratch `git archive` copy with those two lines
  patched. The scratch tests are not committed.
- The engine emits no observer events at `HEAD`. Every finding below is reachable only through
  contract-valid synthetic streams, which is exactly what the tests use.

## 1. Soundness (ranked)

**S1. The mbarrier certificate accepts schedule-dependent generation assignments. [confirmed]**
Location: `certificate.rs:273-277`, `322-334`.

There are two holes:

1. *(a) A wait whose reference generation is `None` is ignored.* This is a parity wait that passed
   before gen 0 completed. Sequence:
   - w0: `init(1)`, `bar.sync 0`, `wait(parity 1)`.
   - w1: `bar.sync 0`, `arrive`.

   With the waiter as warp 0, the default config gives **Clean**. Whole/plain search gives
   **deadlock** (arrive first, then the wait blocks forever). The verdict flips with warp numbering,
   because the round-robin reference decides it.
2. *(b) No rule says a wait of gen g (g≥1) must happen after gen g−1 completed.* Without it, a wait
   meant for gen 2 can pass on gen 0 (same parity). The reference clocks then gate other projections
   on edges that a legal schedule does not have. Sequence, with M, Y, E, Z all `init(1)` after a
   cta_sync:
   - P: `arrive Y; arrive M; wait E(0); arrive M; wait E(1); arrive M`.
   - C: `wait Y(0); wait M(0); arrive E; wait M(1); arrive E`.
   - W: `12×(arrive Z; wait Z)`, then `wait M(0)`, then `arrive Y`.

   The default config **certifies all 5 projections Clean**. Whole search finds `ReuseBeforeConsumption`
   on W's `arrive Y`.

The legacy "terminal generation" fix (line 322) is right but does not address either hole.

Fix: treat a gen-`None` wait as a wait of gen −1 in the overtaking rule, and require every wait of
gen g≥1 to be HB-after some wait of g−1 or some mutation of g. Otherwise return `None` and fall back
to DFS.

**S2. Frontier eviction across window domains loses X2 races. [confirmed]**
Location: `cell.rs:186-199`, `checker.rs:979`, `984`.

`subsumes_contract` ignores `domain` (and `proxy` for weak accesses). The bridge view is chosen by the
*prior's* domain, so the transitivity argument in §2.6 fails. Sequence (one lane):
1. `st [mapa smem]` through the shared::cluster window.
2. `st` to the same bytes through the shared::cta window.
3. `fence.proxy.async.shared::cta`.
4. A TMA that reads and writes those bytes.

Result: 1 race without step 2, **0 races with step 2**.

Fix: add `prior.proxy()==self.proxy() && prior.domain()==self.domain()` to `subsumes_contract`, and
apply the same condition to the plain-write read eviction at line 984.

**S3. `Arrive`/`Wait` with `scope: None` on an mbarrier or cluster barrier gives unconditional HB.
[confirmed]**
Location: `checker.rs:1157-1160` (`_ => true`).

The contract says a `None` scope means the qualifier was lost, which must be incomplete. The checker
instead treats it as a named barrier. Sequence: CTA1 `st g`; CTA1 mbarrier arrive with
`release:Some(true), scope:None`; CTA0 waits at `.cta`; CTA0 `ld g`. Result: clean, no incomplete.

Fix: when `obj` is not `Named` and either scope is `None`, report `SyncQualifierUnknown` and add no
edge.

**S4. Declared-word history is numbered per lane and only for the first overlapping word, but the
contract numbers it per write `Access` for every overlapping word. [confirmed]**
Location: `checker.rs:912-917`, `976-1002`; `observe.rs:25-28`.

Sequence:
1. A 2-lane `atom.relaxed` on the flag (one Access, engine index 1).
2. `st.release` on the flag (engine index 2).
3. The waiter's verdict is `0b100`.

The checker maps index 2 to lane 1's relaxed atom and reports a false race. With other bitsets the
same mismatch acquires the wrong release (a missed race). An 8-byte store over two declared 4-byte
words appends to the first word only.

Fix: append one entry per contract Access, to every overlapping word. Alternatively, change the
contract to per-lane numbering and make the engine match.

**S5. DFS projections never check that their generation assignment matches the reference run.**
Location: `ts.rs:699-701` drops `Effects`; gates are built at `ts.rs:244-266`.

Gates are sound only if every projection reproduces the reference generations. Certificates check
part of this (see S1). DFS checks none of it: a parity alias with inval/re-init, a wait batch, or
`TestState` is accepted when it ends in an equal terminal state.

Fix: in `Ts::step`, compare `fx.gens`/`issued_gens` with `reference.gens` and fail closed with
`generation_assignment_differs`.

**S6. TensorMap acquire filters the components it inherits by their actor's scope, not by the
releaser's.**
Location: `checker.rs:1538-1541`.

- Missed race: thread X in CTA0 writes the descriptor and signals W in CTA5. W does
  `release.cta` and signals C in CTA0. C does `acquire.cta`. The W/C pair is not morally strong, but
  X's component passes the filter.
- False positive: the reverse case, where the writer is remote and the releaser is local.

Fix: store `(releaser warp, scope, clock)` per release and filter on the releaser.

**S7. A failed mbarrier scope test is silent. [confirmed]**
Location: `checker.rs:1161-1165`.

A remote `.release.cluster` arrive observed by a `.cta` wait adds no edge. The result is reported as
`DataRace{MissingInterActorSync}` with no `ScopeMismatch` (`mbarrier_scope_assumed` defaults to
false).

Fix: push `ScopeMismatch` with the arrive and wait sites whenever `ok == false`.

**S8. The strong-diamond reduction checks only one-step enabledness.**
Location: `explore.rs:343-381`.

A transition that becomes enabled only after two of the pruned siblings, and that conflicts with the
canonical one, is never ordered before it. This is the textbook gap in persistent-set reasoning.
I found no protocol instance; `synccheck_equivalence` cannot exercise it (see §5).

Fix: also require that no un-issued command of the projection touches a resource of the canonical
transition, mirroring the persistent rule.

## 2. False positives on standard pipelines

The pipeline walked through: an SM100 GEMM with a TMA producer (elected lane, `arrive.expect_tx` +
TMA → full[s]), an MMA warp (try_wait full → `after_thread_sync` → mma → `commit` → empty[s] /
tmem_full), 4 epilogue warps (tmem_full → `after_thread_sync` → `tcgen05.ld` → `wait::ld` → st.shared
→ `fence.proxy.async` → `bar.sync 1` → elected TMA store → `wait_group.read`; `before_thread_sync` →
arrive tmem_empty), a 2-CTA cluster with B multicast, and named-barrier role dispatch.

The racecheck edges all close: completion→hb/a2g at 1414-1418; commit forwarding at 1397-1410;
tcgen_rel through arrive/wait at 1123-1154; bridge rows through bar.sync at 259-271. The racecheck
emits:

- **F1. `CrossCtaAsyncOrder` review on every multicast or 2-CTA operand consumer site. [confirmed]**
  (`checker.rs:712-719`, `930`). Every cluster kernel's verdict becomes Review.
  Fix: emit it only when the pair is ordered without passing through the prior op's own completion
  (Phase or Warp milestone).
- **F2. The SM90 multicast variant reports a data race on each stage refill.** The consumer release
  is a remote `.release.cluster` arrive and the producer waits at the default `.cta`, so R4 gives no
  edge, and the result is unlabelled (S7). This follows the R4 ruling, but it will flood CUTLASS SM90.
  Fix: report `ScopeMismatch` once per site pair, with a hint, instead of N races.
- **F3. `fence.sc` keeps only the latest fence per thread. [confirmed]** (`checker.rs:1496`).
  `fence.sc.gpu; fence.sc.cta` followed by a remote `fence.sc.gpu` gives a false race; the gpu fence
  alone is clean. Fix: keep the latest fence per `(thread, scope)`.
- If lowering loses qualifiers, `SyncQualifierUnknown` is pushed once per event (see R2).

The synccheck emits:

- **F4. tcgen05.commit couples all empty[s] barriers, tmem_full and the MMA warp's `TcgenWork` queue
  into one projection** (`projection.rs:58-62`). Certificates need a single resource
  (`certificate.rs:42`), so this projection always goes to DFS. Measured, release build, all Clean:

  | stages, MMA k-blocks, tiles | states | time |
  |---|---|---|
  | 4, 4, 2 | 3,019 | |
  | 6, 8, 8 | 105,335 | 3.8 s |
  | 6, 16, 16 | 320,223 | 12 s |

  Real persistent kernels exceed 1M states and become `BudgetExhausted`. Wall time is checked only
  between projections (`mod.rs:143`).
  Fix: do not union through `TcgenWork`/`TcgenKernel`; `work_step` is total and never blocks.
- Multicast TMA and lane-varying remote arrives also merge barriers across CTAs, so those projections
  are never certified. Fix: extend `items()` to multi-resource groups whose commands each touch one
  resource, plus issued targets.
- A TMA `Protocol` event that carries only `issued` and no explicit `Mbarrier(Issue)` cmd is never
  certified (`certificate.rs:48` looks at `program` cmds; `ts.rs` synthesizes the token later). The
  test builder always adds `Issue` (`build.rs:79-82`).

## 3. Robustness

- **R1. Panic.** `AsyncComplete{Warp{warp}}` with an out-of-range warp panics with "index out of
  bounds" at `checker.rs:1221` [confirmed]. Use `warps.get_mut` and report an incomplete.
- **R2. Unbounded `report.incomplete`.** It is pushed without deduplication at 556-574, 813, 1119,
  1141, 1174, 1242, 1334 and 1600-1613. 500 lost-qualifier arrives produce 500 entries [confirmed].
  `ScopeMismatch` is pushed per acquire with no evidence and `alloc = u32::MAX` (1058-1067); 200 loop
  iterations produce 200 findings [confirmed]. Fix: deduplicate through `note_incomplete` (backed by a
  set), and deduplicate `ScopeMismatch` by site pair with sites attached.
- **R3. Other unbounded growth:**
  - `WideSpans` gets a new entry on every pack of a wide span, including re-packing retired hulls on
    each non-generic access (1038), witnesses that are only checked (881) and OOB packs (835).
  - `phases` is never pruned (1125).
  - `pending_acq` is drained only by fences.
  - `g2t_ranges` is append-only and cloned into every TMA issue (1328, 1549), so the cost is
    O(#acquires) per TMA.
  - `reclaimed` grows by one entry per op (1751).
  - Word history is never pruned, and verdict bitsets are O(history/64) per wait.
- **R4. Quadratic or allocating hot paths:**
  - `report_advisory` scans all findings (747-754).
  - `words` is scanned on every access (912).
  - `check_retired_generic` collects all retired entries into a `Vec` on every non-generic access
    (1030).
  - `Clock::meet` and `filter` call `raise` per component, and `raise` copies the chunk vector
    (`clock.rs:199-213`, 423-455). That is O(n²/32) per meet, times views times warps per GC.
  - `Clock::join` clones `lanes` before `make_mut`, which forces a deep copy (`clock.rs:405-406`).
  - `Epochs::leq` allocates through `memo.join` (288).
- **R5. GC is ineffective across CTAs.** The meet covers every non-done warp of the launch (1657).
  CTA A's shared-memory witnesses never die while any unrelated CTA is live, and never at all with
  non-resident waves. Fix: compute the meet per space, over the warps that can reach the allocation
  plus in-flight ops.
- **R6. Thread-local chunk ids in `JoinMemo`** (`clock.rs:99-111`). If a checker migrates threads
  (pyo3 `Send`), ids collide and memo hits return the wrong chunks. Fix: use a global `AtomicU64`.
- **R7. A zero-length span** (for example cp.async zfill with `src-size 0`) is reported as an
  `OutOfBounds` error (833). Skip empty spans instead.

## 4. Contract misuse

- **`SyncEvent.kernel`**: synccheck takes the first event's kernel (`program.rs:104`) and merges
  `per_warp` across launches; restarted `seq` values then give a `program_build` incomplete. Racecheck
  ignores the field. Fix: partition by `kernel`, or reject mixed logs.
- **`ALL_LANES`**: `observe.rs:102` says async spans carry `ALL_LANES`, but `observer.rs:241` needs the
  span to name its lane once an op has been split per lane. A contract-following engine therefore
  gets `AsyncLaneUnknown` on every multi-lane cp.async, and the access is dropped.
  `observer.rs:226,231` maps `ALL_LANES & 31` to lane 31 for warp actors. The test
  `per_lane_async_ops_are_not_merged` codifies this conflict.
- **`returns_value`** is honoured (1006). However, `red` still feeds `tcgen_in` (1009). `Actor::Async`
  side is honoured.
- **`Arrive/Wait scope: None`**: see S3.

## 5. Test adequacy

- `synccheck_equivalence` generators never emit a first wait at parity 1, a wait that skips a
  generation, tx/TMA, conditional waits, inval, multi-target waits or tcgen. They compare only the
  verdict, so both S1 holes and S8 are out of reach. Add these shapes and compare finding kinds.
- No synccheck scenario covers a UMMA ring, a multicast or 2-CTA pipeline (F4), or the named
  `.aligned` same-site rule (§2.3; unimplemented). Duplicate cluster waits are silently overwritten
  (`certificate.rs:171`) and untested.
- Racecheck has no test for:
  - cross-domain or cross-proxy eviction (S2);
  - mbarrier or cluster `scope: None` (S3) — the helper's `default_scope` always returns `Some`;
  - multi-lane or multi-word declared-word numbering (S4);
  - several `fence.sc` in one thread (F3);
  - a tensormap releaser that is not the writer (S6);
  - GC across CTAs or waves;
  - retired hulls wider than 32 KiB;
  - `ScopeMismatch` deduplication;
  - `AsyncComplete` for a reclaimed op;
  - §3 row 6 (`grid.sync`; no event exists).
- Some tests assert on what the helper synthesizes, not on engine-observable behaviour:
  - `build::issue` injects `Mbarrier(Issue)`.
  - `racecheck_common` chooses scopes and phase numbers itself.
  - `aacc` uses `ALL_LANES` only for unsplit ops.

  The certification and scope tests therefore pass on a stream shape the contract does not
  guarantee.
