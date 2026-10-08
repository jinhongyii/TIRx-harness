---
orphan: true
---

# Racecheck semantics (legacy extraction + unified-core spec)

Status: draft, 2026-10-07, branch `refactor/clean-core`. Worker W5.
Revised after the PTX rulings in `racecheck-isa-answers.md` (R1–R9). Every
behaviour change vs legacy is listed in `racecheck-behaviour-deltas.md`.

This document has two jobs:

1. Pin down what the legacy racecheck **actually does**, with citations, so that
   the rewrite can be judged against it. Section 9 lists the places where the
   legacy behaviour looks unjustified or contradictory.
2. Specify the unified core in plan §2.5. The standalone prototype lives at
   `tirx_harness/src/tirx_harness/numsim/core-rs/numsim-race-core/`.

**Path abbreviations.** All paths are relative to
`tirx_harness/src/tirx_harness/numsim/engine-rs/src/`.

| Short | Path |
|---|---|
| **RS** | `native_analysis/racecheck/race_shadow.rs` (shared memory and TMEM shadow) |
| **G** | `native_analysis/racecheck/global_race.rs` (global shadow) |
| **RC** | `native_analysis/racecheck/race_check.rs` (mode glue, HB plumbing, TCGEN) |
| **TIM** | `native_analysis/racecheck/transactional_interval_map.rs` |
| **TF** | `native_analysis/racecheck/tcgen_fence.rs` |
| **RCP** | `native_analysis/racecheck/race_check_python.rs` |
| **PA** | `physical_access.rs` |
| **EF** | `effect.rs` |
| **KE** | `kernel_engine.rs` |
| **SY** | `runtime/sync.rs` |
| **CR** | `numsim/checker_report.py` |

Line numbers are for `c3eac62` plus the working tree at the time of writing.

---

## 0. Corrections to the redesign plan

Each correction was verified in the code.

1. **A shared-memory or TMEM race does abort the operation.** The plan's section 0
   says "`findings.push` 后继续 (`race_check.rs:3165`)". That is wrong for the RS
   path:
   - RC:3163-3167 pushes the finding and then runs
     `return Err(EngineError::message(..))`.
   - The engine propagates that error with `?` (KE:2149, KE:1257).
   - All eleven RS `findings.push` sites behave this way: RC:3165, 3357, 3542,
     3962, 4276, 4430, 4869, 4951, 6122, 6487, 6668.
   - Only **global** races continue. `GlobalRaceShared::insert_finding` (G:5149,
     G:9495) aggregates the finding and lets execution go on.

   Consequence: today an RS run reports at most one shared-memory or TMEM race.
   The RS write rule "a plain write clears every reader" (RS:5823-5825) is sound
   *only because* of this fail-fast behaviour. The new core never aborts, so it
   must not clear readers unconditionally (see §2.6).
2. **The `wait_until` rule is "earliest accepted write *after the already-observed
   prefix*", not a full-history scan.**
   - In `declared_word_candidates`, `skip` is the `take_while` prefix of history
     entries whose carrier epoch the waiter already covers (G:8718-8737).
   - The engine scans from `skip` (KE:3263-3269).
   - The fallback to the observed `current_version` (G:8784-8852) is gated only
     on `accepted == None && !satisfied_on_entry`. That gate covers async
     publications, but also truncated histories, wide (b128) strong writes, plain
     writes and launch-value exits.
   - The plan's statement is the *intended* rule. Section 5 adopts it, with the
     fallback restricted to async publications as the plan says.

---

## 1. Model overview

### Actors

| Actor | Legacy RS | Legacy G | Unified core |
|---|---|---|---|
| Warp | one `RaceVectorClock` component per warp (RS:7286-7289). Lanes are sub-actors carried by lane stamps and lane-order matrices (RS:4030-4224, RC:985-1347) | one component per lane, `warp*32+lane` (G:39-58) | warp component + exact lane vectors (`row` matrix, sparse lane entries) |
| Async op | slot in `AsyncClockRegistry`, with a 14-bit generation and 16-bit epoch (RS:94-276) | `AsyncClockLease` from a launch-wide registry (G:394-467) | a dense actor index ≥ `num_warps`, allocated at issue |
| Cross-CTA warp | absent. A shard is one cluster; foreign history arrives as `SharedClockFrontier` (RS:1817-1947) | ordinary components | ordinary components |
| Proxy | a witness tag plus per-lane bridge frontiers (RS:28-31, 1041-1600) | a witness tag plus two per-lane bridges (G:2311-2350) | a view dimension of `Knowledge` (`hb`, `g2a[d]`, `a2g[d]`, `tcgen`, `tcgen_rel`) |

### Spaces

- RS covers Shared and TMEM (`tracks_race_conflicts`, RS:87-92).
- G covers Global. It also tracks **strong** Shared accesses, but only for
  read-from and versions (G:3757-3765, 6447-6451, 9250-9255).
- Registers and local memory are not checked.

---

## 2. Conflict rule

### 2.1 Access classes

- `PhysicalAccessKind` is `Read | Write | AtomicReadModifyWrite` (PA:44-58). RMW
  reads and writes.
- `MemoryOrder` is `Weak | Relaxed | Acquire | Release | AcqRel | Sc`. Strong
  means anything except Weak (PA:76-97).
- `MemoryAccessClass` is `Plain | Atomic | Reduction | Async`:
  - `is_atomic_class` is `Atomic | Reduction` (PA:204-216);
  - `can_acquire` is `Atomic` only (PA:214-220).
- `MemoryScope` is ordered `Cta < Cluster < Gpu < Sys` (PA:115-120).
- Constructors (PA:240-340):
  - `plain`
  - `volatile`: `Relaxed/Sys/Generic/Atomic`
  - `async_proxy`: `Weak/Async/Async`
  - `async_generic`: classic `cp.async`, `Weak/Generic/Async`
  - `async_reduction`
  - `generic_async`: `st.async`
- RS keeps one derived bit, `strong_scope`. It is set only when the access is
  strong **and** atomic-class (RS:2869-2874).

### 2.2 Conflict

- Two accesses conflict when their bytes overlap and at least one of them writes.
- **RS** (`ShadowState::validate_access`, RS:5709-5804) checks:
  - the current read against retained writes (WriteRead, RS:5723-5748);
  - the current write against retained reads (ReadWrite, RS:5750-5776);
  - the current write against retained writes (WriteWrite, RS:5777-5800).
- **G** (`validate_pair_in`, G:9357-9450) uses the same three classes (G:9641-9644).
- The overlap is reported per segment (RS:6029-6041). G clips to the unit hull
  of a run (G:9425-9471).

### 2.3 Scopes and mutual coverage

- `required_between_warps` (PA:125-143) returns the narrowest scope that
  covers two warps:

  | Relation between the warps | Required scope |
  |---|---|
  | same warp | Cta |
  | no topology | **Gpu** |
  | same CTA | Cta |
  | same cluster | Cluster |
  | otherwise | Gpu |

- `scope_covers_warps(scope, a, b)` is `scope >= required(a, b)` (G:9748-9775).
  Lanes are ignored: "PTX scopes do not distinguish lanes within one warp"
  (G:9768). Sys is never required.
- Two scoped operations **mutually cover** when
  `covers(s_a, a, b) && covers(s_b, b, a)` (RS:6015-6026, G:9722-9760).

### 2.4 Morally-strong exemption

The two shadows apply different rules. This is a genuine semantic difference
(see §8).

**RS, `atomic_modification_order_pair` (RS:5995-6027).** A pair is exempt if
either holds:
- **both sides are RMW**. This is unconditional: no scope, proxy or span check
  (RS:6000-6004). The fast path repeats it at RS:5213-5218, 5253-5258,
  5288-5294 and 5308-5313.
- otherwise, all of:
  - both sides have a `strong_scope`;
  - the spans are **identical**;
  - the proxies are the same;
  - the scopes mutually cover.

  This does exempt a strong load against a strong store.

**G, `mutually_morally_strong` (G:9659-9720).** A pair is exempt only if all of
these hold:
- both sides are atomic-class;
- both sides are strong;
- the proxies are the same;
- the scopes mutually cover;
- both access the same element (equal width and unit alignment, G:9676-9704);
- the access is not a run (`unit_bytes == 0`);
- **neither side is a generic-proxy load** (G:9353-9355, 9384-9409). A strong
  `ld.relaxed`/`ld.acquire` falls through to the HB test.

The design note at G:9686-9705 justifies the load carve-out by schedule
independence. It records verdicts of 11/20 vs 0/20 across worker counts.

If an unordered strong pair fails only on scope, G emits a
`GlobalScopeMismatchDiagnostic` instead of a race (G:9473-9496, 9578-9602).

**Unified core (ruled: ISA R2, R9).** The core applies the PTX rule (§8.7) to
every space:
- both sides strong (`scope.is_some()`; atomicity is **not** required, so
  `st.relaxed` vs `ld.relaxed` qualifies);
- scopes mutually include the other thread;
- same proxy (cross-proxy pairs are never morally strong, R3);
- complete overlap (same exact span).

RMW pairs follow the same rule. So RS's unconditional RMW/RMW exemption is wrong,
and G's both-atomic requirement is too strict.

A data race is overlap ∧ at least one write ∧ not morally strong ∧ not in
causality order (§8.7.1).
- G's generic-load carve-out is **dropped**.
- Instead, a strong load that observes an unordered, morally strong write on a
  word that is not a declared `wait_until` word is reported as a `review`
  advisory, `UndeclaredProtocolWord`.

Tests: `rmw_cross_cluster_cta_scope_is_not_morally_strong`,
`r2_partial_overlap_atomics_race`, `r3_cross_proxy_atomics_race`,
`r9_strong_load_exempt_with_undeclared_word_advisory`.

### 2.5 Same-lane, same-operation and lane-order shortcuts

**Same lane is program order.**
- RS fast path: RS:5239-5244 (compact) and RS:5273-5279 (wide).
- RS general path: `component(l,l) + 1 >= e` (RS:4214-4221).
- G: the lane tick per access (G:6577).

**Sibling lanes of one instruction are simultaneous.** This is true even for
same-address atomics, where the pairs are exempt only by moral strength.
- Same operation with no lane verdict gives `false` (RS:4739-4742).
- The fast path gives the same result (RS:5314-5317).

**`allow_same_operation`** (RS:4733-4735): a pair from the same registered
operation in the same proxy counts as ordered. This is used only by
explicit-clock and async batches (RS:7271, 8947, 8701, 10723); warp batches pass
`false`.

**Ownership skip** (RS:2637-2674): validation is skipped when every witness in
the exact segment belongs to the current operation.

**Record dedupe:** `records_same_operation_kind` (RS:5649-5674) and
`represents_same_direct_event` (RS:4681-4700).

**Lane-order matrix** (`RaceLaneOrder`, RS:4030-4224, maintained by
`WarpLaneOrder`, RC:985-1110):
- State: `common[32]` and `observed[32][32]`. Lane `c`'s view of lane `p` is
  `max(common[p], observed[c][p])`.
- Updates:
  - a full-mask acquire raises `common`;
  - a partial-mask acquire raises rows;
  - `warp_sync` is release(mask) followed by acquire(mask) (RC:1332-1348).
- Cross-warp lane-stamped priors are judged **only** by lane release frontiers
  (`incoming_common` / `incoming_lanes`, RS:4199-4213). `incoming_common` is
  always `Some` in production (RC:1202-1213), so the vector-clock fallback at
  RS:5268 and RS:5304-5306 is dead code.

**Unified core** (`checker.rs`, `Checker::ordered`):
- Each warp keeps `row[c][p]`. `WarpSync(mask)` sets every row in the mask to
  the element-wise max over the mask.
- A prior event `(W, p, e)` is ordered before the current event `(W, c, e')` if
  any of these hold:
  - `p == c`;
  - `e != e'` and `row[c][p] >= e`;
  - the lane's knowledge contains it (the event came back through other actors).
- Releases publish an exact per-lane own vector:
  `v[p] = max_{c in L}(c == p ? e : row[c][p])`, stored as a sparse lane entry.
  This gives `lane_precise_publication` and `shared_frontier_relay[partial_warp]`.

### 2.6 Supersession (frontier eviction)

**RS** (`AccessFrontier`, RS:4852-5192):
- The frontier is an antichain of witnesses.
- A new witness evicts a prior only if it **subsumes its contract** and observes
  it (RS:5026-5137). Observation is checked by bridge, then lane order, then
  clock (RS:4895-4919).
- `subsumes_contract` (RS:2946-2967) requires the same kind. A strong current
  witness also requires the same scope, proxy and exact span, plus mutual cover.
- A **plain Write clears every read without an HB test** (RS:5823-5825,
  5847-5849, 5480-5482). RMW and strong writes do not clear reads.
- Non-exact geometry collapses the whole span to a fresh state holding only the
  new witness (RS:6726-6748, 6853-6871, 7176-7190), for any `kind.writes()`.
  This is a bug for RMW (§9).

**G** (`GlobalFrontiers`, G:3394-3804):
- There is one entry per frontier actor (lane or async lease).
- `insert` replaces the actor's previous entry **regardless of semantics**
  (G:3446-3456, 3682-3683). This is a bug (§9).
- A write removes only its own actor's reader entry (G:3786).
- Non-atomic supersession requires the same kind, the same semantics and HB
  (G:3793-3804).

**Unified core** (`cell.rs`):
- Each cell holds two frontiers, `writes` and `reads`.
- A new entry evicts a prior iff `new.subsumes_contract(prior)` and the prior is
  ordered before the new entry.
- A **plain** write also evicts the reads that are ordered before it.
- A **strong** write keeps the reads, because a later access that is morally
  strong with the write can still race a read.
- No eviction depends on fail-fast behaviour.
- **Per-proxy write slots are not needed.** The plan sketched
  `last_write per proxy`, but every bridge view is a snapshot of `hb`
  (bridge ⊑ hb), and joins are whole-clock. A prior that is observed through any
  view is therefore observed transitively by everything that observes the
  evictor. Legacy does not keep per-proxy slots either.

### 2.7 Proxies in the check

**Bridge storage.**
- RS: per lane, `[direction] × [prior domain] × [current domain]`
  (`PROXY_BRIDGE_SLOT_COUNT = 2·3·3`, RS:28-31, 1580-1600).
- G: per lane, two directions, global domain only (G:2311-2350, 7714-7719).

**Cross-proxy pairs.** A cross-proxy pair is judged **only** by the bridge,
never by plain HB: RS:4747-4761, G:9557-9564.

**Fences.**
- `fence.proxy.async[.d]` merges the lane's current frontier into **both**
  directions:
  - RS:1041-1077 iterates over domains and their aliases;
  - G:7700-7726 replaces the bridge and ticks, and handles only `All|Global`.
- Implicit async completion adds only async→generic (RS:1097-1115,
  G:2326-2333).

**Fast-path gating (RS).** Generic accesses check proxies on the fast path only
once an allocation has seen an async-proxy access (proxy-sensitive, RS:5219-5230,
7765-7770). After GC the check uses retired generic history (RS:8273-8324).

**Unified core** (`knowledge.rs`):

| Prior proxy | Current proxy | View consulted |
|---|---|---|
| generic | async | `g2a[prior.domain]` |
| async | generic | `a2g[prior.domain]` |
| tcgen | tcgen | `tcgen` |
| same proxy | same proxy | `hb` |

- The bridge slot is keyed by the **prior** access's window domain.
- `fence.proxy.async` is bidirectional, scopeless and thread-local, and it is
  limited to its state space (R3, §9.7.15.4). Slots it sets:

  | Fence | Slots set |
  |---|---|
  | `.global` | `global` |
  | `.shared::cta` | `shared::cta` |
  | `.shared::cluster` | `shared::cta` **and** `shared::cluster` |
  | unqualified | all |

  `.shared::cluster` covers `shared::cta` because the `shared::cta` window lies
  inside the `shared::cluster` window (§5.1.7). Whether a `.shared::cta` fence
  covers a `shared::cluster` (mapa) window prior is ISA-silent, so it fails
  closed and does not (`same_rank_mapa_shared_cluster`).
- Bridges are per lane: a lane overlay plus `bridge_rows` for own-warp lanes.
- Bulk-copy completion carries an implicit async→generic fence for **the copy's
  own results only** (§9.7.10.28.2). Generic→async always needs an explicit
  fence.
- Two async-proxy accesses whose ops were issued from **different CTAs**, ordered
  only by base causality, get a `review` advisory, `CrossCtaAsyncOrder`. PTX
  §8.9.5 preserves same-proxy order only "by the same thread block", and it is
  ISA-silent on which block an async op belongs to.

---

## 3. Happens-before edge table

Layers in legacy:
- **WC**: RS per-warp clock.
- **LO**: lane order / `SharedClockFrontier`.
- **GL**: G per-lane `SparseLaneClock`.
- **TG**: TCGEN frontiers (RC:1390-2061, TF).
- **PB**: proxy bridge.

The "Core" column names the unified-core event and rule.

| # | Source → sink | Legacy (clock joined; condition; code) | Core |
|---|---|---|---|
| 1 | Program order, same lane | LO lane epoch (RC:1025-1033, 1255-1261); GL tick (G:6577); WC tick (RS:7516) | `prior.lane == lane` in the same warp |
| 2 | Lanes of one warp | no implicit order (RS:4207-4222; test RC:10063) | `row` matrix only |
| 3 | `bar.warp.sync`, `__syncwarp`, `tcgen05.alloc.sync.aligned` | LO release+acquire(mask) (RC:1332-1347); WC collapses only PB lanes (RS:9292-9306); GL `release_mask` (G:8149-8174); TG mask (RC:8226-8291). Emitted at KE:1878-1891 | `WarpSync{mask}` |
| 4 | `ldmatrix` / `stmatrix .sync.aligned` | LO + PB collapse before and after the access; **not** GL/TG (RC:8293-8310; KE:3531-3592) | producer emits `WarpSync` before and after |
| 5 | `shfl`, `vote`, `match`, `redux`, `mma.sync` | **no edge** | none |
| 6 | `grid.sync` | **no edge** (KE:1892-1896) | open (§10) |
| 7 | `bar.sync` / `bar.red` / `bar.arrive` (named barrier, generation g) | arrive: WC release masked, LO release, TG publish into `named_barrier_*[(id, g)]`; resume: acquire (RC:5548-5699). GL: lane clock **replaced** by the payload (G:8263-8327, 8559-8580). No scope (implicitly CTA) | `Arrive{release: Some(true), scope: None}` for every participant; `Wait` only for `sync`/`red`. An `arrive`-only thread gets no acquire (R5, §8.9.4) |
| 8 | `barrier.cluster.arrive[.release/.relaxed]` → `wait[.acquire]` | TG always; memory payload only if `publishes_memory` (RC:5701-5787); a missing payload in A is treated as relaxed (RC:6933-6936); GL (G:8329-8399) | defaults release/acquire at `.cluster` (§9.7.15.3); a qualifier lost in lowering (`None`) → `SyncQualifierUnknown` incomplete, never assumed relaxed |
| 9 | `mbarrier.arrive` (release) → successful wait on the same generation | WC/LO/TG into `barrier_*_payloads[(bar, g)]`; waiter: TG always, copy payload always, WC+LO **only if `has_acquire`** (RC:5240-5395, 1919-1956); GL (G:8190-8261). **No scope field** (SY:387-394, 626-632) | `Phase.arrivals` keep the arriver and its scope (default `.release.cta`). The waiter (default `.acquire.cta`) acquires an arrival only if the scopes mutually include each other (R4, §8.9.4). A `.relaxed` wait parks arrivals **and** completions in `pending_acq` until a later `fence.acquire`/`acq_rel` (§8.8) |
| 10 | Relaxed `mbarrier.arrive` / relaxed `test_wait` | TG only (RC:1950-1955; PTX 9.7.18.6.4.4) | `tcgen_rel` travels through relaxed arrives and waits |
| 11 | Which generation a waiter acquires | `completed_generation()` when the parity matches (hardware_barriers.rs:2423-2428). Payload outside the window `RETAINED_BARRIER_GENERATIONS = 8` → `BarrierPayloadUnavailable` (RC:5361-5375, 6849; G:8226-8239) | the SyncTable resolves the phase; the checker keys by `(obj, phase)` |
| 12 | TMA / bulk mbarrier copy, issue | issue reads recorded at the **issuer's** warp clock and lane stamp (RS:9043-9058); token clock = issuer's lane-acquired clock, `tick_async_issue` (RC:4199-4314, 4907-4998). G records issue reads at the **token** clock (G:7328-7336) | `AsyncIssue` forks `Knowledge` from `publication(lanes)`; the op's accesses carry `Who::Async` |
| 13 | Copy completion → mbarrier `complete_tx` | `copy_completion_projection` = the token component + its async→generic bridges only (RS:1117-1154; G:7427-7450). **Acquired by relaxed waits too** (RC:797-799, 5341-5353; G:8247-8253) | `AsyncComplete{Phase}` → `Phase.completion` = `{A:m}` in `hb` and `a2g[*]`. complete-tx is release at `.cluster` (§9.7.10.28.4.1). An acquire wait of **any** scope receives the copy's own bytes (the async-op thread is ISA-silent). A relaxed wait receives them only through a later acquire fence |
| 14 | `cp.async` / bulk group issue | fork from the lane-acquired clock, no issuer tick (RC:5000-5090; RS:8438-8462) | `AsyncIssue` |
| 15 | Group milestones `SourceReadComplete` / `FullComplete` | `advance_async_actor`; accesses committed at the new token clock; implicit a2g (RC:6040-6160, 6591-6818) | read-side accesses stamped epoch 1, write-side epoch 2 |
| 16 | `cp.async.wait_group`, `cp.async.bulk.wait_group[.read]` | WC `barrier_acquire_masked` of the members' milestone clocks; `.read` → `source_read`. **The full token clock includes issuer history**; no LO acquire (RC:5092-5153). GL `acquire_async_token` (G:7511-7545) | `AsyncComplete{Warp{lanes}}`, **per lane** (ISA Q7); `.read` gives `Milestone::Read` only |
| 17 | `cp.async.mbarrier.arrive[.noinc]` | arrival payload = completed copy actors of that `(warp, lane)` only, flagged `copy_completion` (RC:5792-5844, 6760-6800; G:7481-7509) | `AsyncComplete{Phase}` per copy |
| 18 | `tcgen05.{mma,cp,shift,ld,st}` issue | token = lane-acquired WC with uncompleted TCGEN components **cleared** ⊔ pipeline predecessor (only if every active lane has one) (RC:4316-4482, 2009-2061; RS:8461-8501). Pipeline pairs: TF:46-65 | `AsyncIssue{kind}`; `k.tcgen` = the warp's `tcgen` view ⊔ the producer-resolved `preds` |
| 19 | `tcgen05.wait::ld/st` | `complete_tcgen_work_set`; WC acquire; `tcgen_wait_frontiers` (RC:5521-5547, 4484-4550) | `AsyncComplete{Warp}` on a tcgen op: the warp's `tcgen` and `tcgen_waited` gain `{T}` |
| 20 | `tcgen05.commit` → mbarrier | implicit before-fence; marks work complete **at commit issue**; payload = WC release ⊔ commit frontiers ⊔ TG (RC:4569-4593, 5412-5514, 6333-6400). Relaxed waiters get TG only | `AsyncIssue{TcgenCommit, preds}` (an implicit `TcgenBefore`). Completion = the issuer's generic knowledge at commit + `{T}` ⊔ `T.tcgen` for each tracked T. **No a2g** |
| 21 | `tcgen05.fence::before_thread_sync` | capture commit-bound pipeline frontiers ⊔ fenced ⊔ wait → published, fenced (RC:1669-1757) | `tcgen_rel ⊔= issued ⊔ waited ⊔ tcgen`; `tcgen ⊔= issued` |
| 22 | Any thread sync carries TG | `tcgen_release_mask` → `tcgen_acquire_mask` into incoming (RC:1806-1868) | `Knowledge.tcgen_rel` → the receiver's `tcgen_in` |
| 23 | `tcgen05.fence::after_thread_sync` | `fenced ⊔= incoming` (RC:1759-1804) | `tcgen ⊔= tcgen_in` |
| 24 | Strong release write → strong acquire read that **reads from** it | `GlobalVersion` + `ReleaseHead{actor, scope, proxy, clock, tcgen, shared_frontier}` (G:9163-9227); `try_acquire_head` with a mutual-scope check, else `ScopeMismatch` and no edge (G:9109-9161). The SMEM frontier is returned to LO (RC:2434-2493) | a cell write entry carries `Heads` (a list of `Rel{k, scope, warp}`). The reader takes the latest non-sibling morally strong write and checks each head's scope against itself (the first op of X and the last op of Y must be morally strong, §8.9.4). A failure → `ScopeMismatch` |
| 25 | RMW chain (PTX *observation order*, §8.9.2; there is no "release sequence") | an RMW inherits the predecessor's payload when mutually morally strong (G:9173-9199) | an atomic appends its own head to the heads of the write it reads from. **Sibling lanes of one same-address RMW instruction never inherit each other's heads**, because their coherence order is unconstrained (§8.9.1; the ISA is silent on intra-instruction order). Each entry keeps `base` = the heads from writes preceding the whole instruction; whoever reads from a sibling group gets only `base` (R1; legacy correct) |
| 26 | Release fence + relaxed store / relaxed load + acquire fence | `release_fence` head (latest only) carried by relaxed atomic writes; relaxed reads → `pending_acquire`, drained by an acquire fence (G:8046-8147, 9200-9206, 9100-9106) | `fence_rel[lane]` and `pending_acq` |
| 27 | `fence.sc` | launch-wide `sc_fences` keyed `(cta, scope, proxy)`. SC order = **runtime linearisation** under a mutex (G:5078, 8064-8133) | per thread, the latest `fence.sc` and its scope. A new `fence.sc` acquires every earlier one that is **morally strong** with it (each scope includes the other thread, §8.9.3), then publishes. `fence.sc` is also `acq_rel` (ISA-silent; R8) |
| 28 | `fence.proxy.async[.d]` | see §2.7. PB is transported through barrier payloads (RS:1623-1671) | per-lane bridge snapshot |
| 29 | Implicit a2g at async completion | RS:1097-1117; G:2326-2333 | in the copy completion projection |
| 30 | `wait_until` | §5 | `WaitVerdicts` |
| 31 | TensorMap release / acquire / consume | descriptor frontiers; consume without acquire → **Err** (G:7555-7698) | not in the prototype (§10) |
| 32 | `fence.mbarrier_init`, `expect_tx`, `init`/`inval` | no racecheck edge; synccheck owns them (RC:5163-5168, 5208-5239) | none (synccheck) |
| 33 | TMEM alloc/dealloc/relinquish, `setmaxnreg` | no-op in racecheck (RC:5788-5791) | `AllocBegin/AllocEnd` give lifetime only |

---

## 4. Async ops, lifetimes, memory reuse, OOB

**Milestones.**
- Legacy AsyncPayload: issue (reads), then completion (writes, no tick)
  (RC:5960-6575).
- Legacy groups: issue, `SourceReadComplete`, `FullComplete`.
- Legacy TCGEN: issue (all accesses), then wait or commit.

**Core milestones.** An async op `A` is one actor with epochs 1 (read side) and 2
(write side).
- A read-only completion publishes `A:1`, which orders exactly the reads.
- A full completion publishes `A:2`, which orders both.

**Memory reuse under an unfinished op** is an ordinary conflict.
- When ordinary HB says the op's *issue* is ordered but its completion is not,
  the failure is classified `AsyncLifetimeNotDrained`
  (RS:4383-4413, 4816-4821; G:9900-9906).
- An unwaited `tcgen05.ld` read conflicting with a later write is **review**, not
  error (`tmem_lifetime_review`; RS:5964-5993; RCP:465-467).
- At launch end, outstanding work becomes `EffectCommitUnobserved` incomplete
  (RC:3668-3740).

**Core lifetime checks.**
- `AllocEnd` while an op with a footprint in that allocation has reached no
  milestone → `AsyncLifetime` finding.
- An op that never completes by `finish` → `AsyncNeverCompleted` incomplete.
  An *uncommitted bulk* async op (`cp.async.bulk` / TMA store with no
  `commit_group`) is not such an op: at warp exit it is committed implicitly
  and lands before launch end (sync-semantics §5, async-group `Exit`), so it
  completes normally. Uncommitted `cp.async` issues at exit are only the
  Review lint `UncommittedAtExit` (sync-behaviour-deltas A3); their copies
  still land. `AsyncNeverCompleted` is reserved for ops the engine never
  landed (an aborted or budget-stopped launch).

**OOB.**
- Legacy: batch construction is all-or-nothing (PA:1096-1100). OOB is an
  engine `execution_error{kind: "oob"}` (executor.rs:1008), not a race finding.
  `RaceReport` hides it whenever another error finding exists (CR:285-286).
- Core: an `OutOfBounds` finding, and the access is skipped.

**TMEM lifecycle.**
- Racecheck: no edge.
- Synccheck: owns alloc/dealloc protocol.
- Core: TMEM is just an allocation with `Proxy::Tcgen` accesses.

---

## 5. Declared-word waits (`wait_until`)

### Legacy

**Engine** (KE:3060-3125, 3228-3312):
1. Poll until every lane accepts.
2. Re-read under a shared transaction; publishers take it exclusive for non-plain
   ops (KE:2987-2993).
3. For each lane, compute `skip` (the observed prefix, G:8718-8737) and
   `accepted` = the earliest index ≥ `skip` that the predicate accepts.
4. Otherwise compute `satisfied_on_entry` and `satisfied_by_launch_value`.
5. Send `DeclaredWordWaitPlan` (SY:648-717).

**History.**
- Post-images of **every** 4- or 8-byte global atomic-class write (KE:2998-3025,
  G:6682-6692, 6757-6786).
- Keyed `(allocation, byte_offset)`.
- Capped at 2^16 entries → `DeclaredWordHistoryTruncated`.
- Wrong width → `unrecorded_strong_writes` → `DeclaredWordWriteUnrecorded`
  (G:5327-5431).

**Checker** (`apply_declared_word_wait`, G:8757-8922):
1. Tick the wait.
2. Claim the protocol word: bypass detection against plain accesses
   (G:5380-5433).
3. If `accepted = Some(i)`: `apply_load_ordering` on `writes[i].version`, i.e. the
   same path as an acquire load.
4. Else if not satisfied on entry: fall back to the observed `current_version`.
   If that is empty and the launch value does not satisfy the predicate →
   `DeclaredWordWaitUnexplained`.

**Waits always acquire.** A relaxed *publisher* still gives no edge
(test_declared_word_regressions.py:28-117).

### Core rule (`Checker::wait_verdicts`)

- **History.** The checker numbers every write to a declared word in delivery
  order:
  - index 0 is the launch value;
  - index i is the i-th write.

  The engine evaluates the predicate over the same history and sends the
  **verdict bitset**. Because both sides number one stream, no values cross the
  boundary.
- **Earliest accepted.** Edge source = the lowest set bit.
  - Bit 0 → no edge owed.
  - No bit set → `WaitExitUnproven` incomplete.
  - The edge is the accepted entry's `Rel`: the release head, the fence-release
    head, or the RMW chain. It is scope-checked like an acquire load.
  - This is schedule independent. A full scan is affordable because the bitset is
    computed engine-side.
  - Test: `earliest_accepted_write_is_schedule_independent`.
- **Fallback.** Only when the accepted entry is an **async publication**
  (`st.async` / `red.async` / bulk write). Those land their bytes at completion,
  so the history cannot pair a value with a release; the run's `observed` index
  is used. Everything else that legacy routes through the fallback is
  `incomplete` instead:
  - truncated history;
  - wide writes;
  - plain writes, because they have no `Rel`, so no edge, and a later read races.
- **Predicates that read memory** (a lowered `PredProgram` with `reads_memory`).
  The bitset is meaningful only if the extra inputs were constant during the
  wait. Rule:
  - The producer lists `pred_reads`.
  - If every retained write to those bytes is ordered before the wait for every
    waiting lane, use the bitset.
  - Otherwise report `WaitPredicateReadsUnstable` (incomplete, fail closed).
  - The predicate's loads are also emitted as ordinary `Access` reads, so racing
    inputs are reported as well.
  - Test: `predicate_reading_memory`.

---

## 6. Clock representation tricks (and why each is needed)

### 6.1 Packed `(actor, epoch)` stamps and the one-component HB test

**Legacy.**
- `RaceEventTimestamp(u64)` (RS:3956-3962, 4225-4610) has three encodings:
  - plain compact: `epoch<<32 | actor`, where actor bit 31 means registered
    and bit 30 means async;
  - lane-stamped compact (bit 62): 4-bit local warp, 5-bit lane, 2-bit kind,
    24-bit event epoch, 24-bit lane epoch;
  - wide (bit 63): an index into a registry.
- The test (`observed_by`, RS:4529-4563) is `C[actor] >= epoch`.
- `has_single_actor` (RS:4514-4527) lets incoming payloads be tested one at a
  time, without materialising the join (RS:4117-4136).
- G uses the same idea, keyed on the frontier actor's epoch (G:9556-9572).
- Retained witnesses pack into 16 bytes (RS:2942-3369, 2969-2984).

**Operation count.**
- A check costs one indexed load and one compare.
- A full vector-clock ≤ costs O(actors), with actors = warps + live async slots
  (hundreds to thousands).
- Each access is checked against every retained witness of every overlapped
  segment, so this is the innermost loop.

**Core.** `clock::Stamp`, `Clock::observes`.
- Bench `packed_stamp_compare` (256 checks):

  | Actors | One-component test | Full VC ≤ |
  |---|---|---|
  | 64 | 0.24 µs | 4.2 µs |
  | 1024 | 0.24 µs | 51 µs |

  The one-component test is independent of the actor count.
- The core's witness is 48 bytes. The 16-byte packing is a later memory
  optimisation, not semantics.

### 6.2 Single witness per actor-contract (frontier antichain)

**Legacy.**
- `AccessFrontier` is `Empty | One | Two | Many` (RS:4852-5192), with
  `record_lane_group` as a multi-lane bulk path (RS:4921-5023).
- In G, the frontier is one entry per actor, grouped by warp, plus a flat async
  vector (G:3394-3437).
- `strong_gpu_len` lets an all-strong-gpu frontier be skipped (G:3426-3437).

**Operation count.**
- In race-free code each write supersedes, so `writes` is `One`.
- `reads` holds one entry per concurrently live reader.
- A check is therefore O(concurrent readers), not O(history length).
- Dropping an observed prior loses no race. By transitivity, any future access
  ordered after the evictor is ordered after the prior. Any future access not
  ordered after the evictor races the evictor, which is reported. The contract
  condition preserves moral-strength exemptions.

**Core.** `cell::Frontier`. The `Two` variant is dropped for brevity.

### 6.3 Exact-hit in-place segment update

**Legacy.**
- Patch-overlay exact hits (RS:6322-6432).
- In-place commit (RS:6492-6558).
- Direct path `segments.get_mut(start)` with an equal end (RS:7939-8043).
- `replace_segment_map_range` (RS:9585-9600).
- `try_for_each_exact_range_mut` (TIM:438-482, 794-849).

**Operation count.** Tile loops revisit identical ranges.
- Exact hit: one O(log n) lookup and zero allocations.
- Otherwise: two splits, k removals and reinserts, and merges.

**Core.** `IntervalShadow::update`.
- Bench `interval_shadow_update`, 1024 segments:

  | Path | Time |
  |---|---|
  | exact hit | 14 ns |
  | misaligned split and merge (3 updates) | 1.8 µs |

- The legacy `TransactionalIntervalMap` also has an `Indexed` per-4 KiB page
  cell table, promoted one-way to a `BTreeMap` (TIM:7-125, 484-534). The core
  keeps a `BTreeMap` only.

### 6.4 `Arc`-shared memoised clock chunks and the join memo

**Legacy.** `EpochChunks<CHUNK>` (RS:278-720):
- Chunks are immutable `Arc<EpochChunk{id, [u64; CHUNK]}>`; 64 slots for async,
  32 for warps.
- Join order (`join_epoch_chunks`, RS:465-490):
  1. pointer-equal;
  2. dominance (adopt the dominating chunk);
  3. memo `(id_a<<32 | id_b) → Weak` with 64 shards, pruned every 256 inserts,
     capped at 2^15 (RS:377-460; tuning note on MegaMoE ≈ 9 GB at RS:394-400).
- If only the incoming side changed, merge adopts `other.chunks` wholesale
  (RS:646-649).

G uses a two-level group→block `NodeId` slab with atomics and mark-and-sweep
(G:886-1520, 1700-1945). The motivation was "24 GB of dead clocks" and "13 % of
worker time" (G:990-995). G also has group-join, whole-list, acquire and supersede
memos (G:1341-1792, 2884-2952, 8961-8985).

**Operation count.**
- Async actors are never recycled, so a flat clock is O(all ops ever issued).
  Every copy (witness release payload, async fork, barrier payload) and every
  join would cost that much.
- With shared chunks:
  - a copy is a refcount bump;
  - a join costs O(differing chunks × CHUNK) after O(chunks) pointer compares;
  - fan-in of one payload into N warps converges every warp onto the payload's
    storage, so the next join is a single pointer compare.

**Core.** `clock::Epochs` + `JoinMemo` (32-slot `u32` chunks), plus a slice-level
fast path that adopts the incoming slice when it dominates chunk-wise.

Bench `clock_join`, 32 warps × one payload:

| Case | Actors | Chunked | Flat |
|---|---|---|---|
| shared history | 128 | 0.42 µs | 1.28 µs |
| shared history | 4096 | 2.97 µs | 29.7 µs |
| already synchronised (pointer-equal) | any | **6 ns** | — |
| **disjoint history** (every warp differs in every chunk) | 128 | 5.7 µs | 1.3 µs |
| **disjoint history** | 4096 | 85 µs | 32 µs |

The chunking pays **only through sharing**. Its worst case is 3–4× slower than a
flat join, so it must stay benchmarked.

The legacy `happens_before` (and the core's `Epochs::leq`) allocate a merged chunk
to answer a ≤ query on incomparable chunks (RS:583-589). That is fixable.

### 6.5 Dominated-frontier GC

**Legacy RS.**
- Runs every `AUTOMATIC_GC_SAFE_POINT_INTERVAL = 1024` safe points (RS:27,
  9491-9498).
- The frontier is the meet of all warp clocks and active async clocks, plus
  per-lane bridge meets (RS:9351-9360, 9501-9527).
- It drops witnesses the frontier observes, including on the bridge for
  proxy-sensitive allocations (RS:6086-6135).
- It **never** drops lane-stamped (direct warp) witnesses (RS:6092-6096).
- Generic witnesses of non-sensitive allocations fold into
  `RetiredGenericHistory` so that a later async access is still checked
  (RS:3587-3950, 9365-9450).
- Async slots are reclaimed when `fresh >= max(64, 4×occupied)`
  (RS:111-135, 248-275).

**Legacy G.**
- `GlobalFloor` is the meet over all shards, collected with `try_lock`
  (G:9948-10220; RC:6280-6328).
- It drops floor-dominated entries **by epoch only, proxy-blind**
  (G:10153-10166; see §9).
- Lanes created after a retirement are laggards: `RetiredRecordsUnobserved`
  (G:268-391, 2969-2985).

**Operation count.**
- Without GC, frontiers of long-lived ranges keep witnesses that no future
  access can race with.
- GC costs O(segments) every K safe points and bounds memory by live
  concurrency.

**Core.** Not implemented in the prototype. The rule to keep:
- a witness may be dropped iff, for **every** view an unseen future access could
  use against it, the meet of all live actors' views observes it;
- if it cannot be decided, keep the witness.

### 6.6 Sparse lane clocks (global)

**Legacy G.** `SparseLaneClock` (G:870-2546):
- a shared base of warp-group blocks (32 warps × 32 lane epochs);
- a lane-local update overlay (`component_updates`), so a warp sync does not
  rebuild the list once per lane (G:887-890);
- joins cost one pointer comparison per shared group (G:873-879).

**Core.** One scalar component per warp, plus *sparse* lane vectors
(`LaneEntries`) only where a release came from a lane subset. Entries dominated
by the scalar are dropped (`Clock::normalize_actor`). Converged code pays nothing
for lane precision.

### 6.7 Stripe sharding

**Legacy G.**
- 64 KiB stripes keyed `((space, allocation), stripe)`.
- One `RwLock` per cell, taken in sorted key order (G:4802-4865, 5536-5696).
- Deferred weak-read logs: `PENDING_READS_CAP = 64`, drained FIFO by the next
  writer (G:4829-4865, 5736-5815).
- Separately, 1024 engine transaction stripes (RC:733-765).

**Why.** Shards (one per cluster) share the global shadow. Without stripes,
tiles of one weight buffer serialise on one lock.

**Core.** Single-threaded per checker instance. Plan §2.4 replaces locks with
stripe ownership plus inbox drains (merge points). That is a W2/W5 integration
item and is not in the prototype.

---

## 7. Report taxonomy

### 7.1 Payload

Built by `build_native_race_check_phase_result` (RCP:184-319):

| Key | Content |
|---|---|
| `schema_version` | 3 |
| `execution_model` | `"direct_online_vc"` |
| `checked_memory_spaces` | `["global","shared","tmem"]` only with `global_memory_model_enabled` (RCP:205-213) |
| `phase` | |
| `analysis_scope` | |
| `verdict` | |
| `findings` | races, then scope diagnostics, then declared-word bypasses (RCP:325-337) |
| `advisories` | `alias_stale_read`; `uninitialized_read` is appended later (runtime/python.rs:1379-1404) |
| `sync` | the embedded Synccheck result |
| `incomplete` | |
| `access_count`, `accesses_complete` | |
| `accesses` | only with `inspect_accesses` (RCP:737-798) |
| `stats` | |
| `execution_error` | |
| `resource_limits` | |
| `timing` | |

Evidence is `operation` (`kernel_index`, `global_warp_id`, `per_warp_sequence`,
`source_op_id`, `loop_frames`; SCP:2152-2163) plus `lane`, `access_kind`,
`space` and `span`.
- **Async accesses are attributed to the issuing warp and lane**; there is no op
  id.
- No clocks are emitted.

### 7.2 Finding kinds

| Kind | Produced | Evidence | Status |
|---|---|---|---|
| `data_race` | RCP:459-523 | `access_pair` ∈ {`write_read`, `read_write`, `write_write`}; `ordering_domain` ∈ {execution, memory, completion}; `ordering_failure` ∈ {`missing_inter_actor_sync`, `missing_same_warp_lane_order`, `missing_release_acquire`, `async_lifetime_not_drained`, `missing_proxy_bridge`} (RS:2018-2045); `proxy_bridge{…}`; `prior`/`current` witnesses; `overlap`; `message`; optional `hint` | error |
| `tmem_lifetime_review` | same struct, prior is an unwaited `tcgen05.ld` (RS:3471-3473) | same | review |
| `scope_mismatch` | G:178-226, 9136-9156, 9578-9602; RCP:372-411 | release/acquire op, scopes, warp/lane, `actor_relation`; **no span** | error |
| `signal_protocol_error` | declared-word bypass (G:137-176; RCP:339-370) | wait/plain op, warp/lane, overlap | error |
| `alias_stale_read` | RCP:413-457 | buffers, ops, overlaps | review (advisory) |
| `uninitialized_read` | runtime/python.rs:1347-1404 | | review |
| sync findings | CR:254-260 | raw | **always error** |
| execution error (`oob`, `deadlock`, …) | executor.rs:1007-1045 | kind, message, op | error; shown only if no other error finding (CR:284-294) |

**Classification of `ordering_failure`.**
- RS (RS:4765-4822):
  - proxies differ → bridge;
  - same warp → lane order;
  - no ordinary order and either side RMW → release/acquire;
  - no ordinary order otherwise → inter-actor;
  - ordinary order exists but an async actor is involved → lifetime.
- G (G:9872-9909) is the same except "release/acquire" fires when either side is
  atomic-class **or** RMW.

**Deduplication.**
- RS: by site, with a span hull (RS:3420-3447).
- G: by `FindingSite`, with touching ranges merged and a cap of 2^20
  (G:4884-5040).

**Core.** `checker::FindingKind` / `OrderingFailure` mirror these kinds. Evidence
carries the issuer's warp and lane **and** the async op id. Dedupe is by
`(alloc, class, prior site, current site)` with a byte hull.

### 7.3 Verdict policy

**`combine_peer_verdicts`** (RC:245-270):
- **error** if any of: non-review finding, scope diagnostic, bypass, or
  Synccheck error;
- else **incomplete**;
- else **review**;
- else **clean**.

**Payload verdict** (RCP:236-244), checked in order:
1. status error;
2. execution error that is not incomplete-class;
3. subset execution → incomplete;
4. status incomplete;
5. incomplete-class execution;
6. review or clean.

### 7.4 Incomplete reasons

From `RaceCheckIncompleteReason` (RC:71-140) and RCP:592-735:

| Reason | Where |
|---|---|
| `global_write_seed_incomplete` | |
| `tcgen_access_unmodeled` / `cluster_barrier_unaligned_unmodeled` / `atomic_lane_serialization_unmodeled` | EF:388-399 |
| `effect_commit_unobserved` | |
| `barrier_generation_unavailable` | |
| `barrier_payload_unavailable` | |
| `wait_exit_unproven` | |
| `signal_history_truncated` | |
| `signal_write_not_recorded` | |
| `retired_records_unobserved` | |
| `findings_truncated` | |
| `shadow_rejected` | |
| `async_payload_access_unmodeled` | |
| `GlobalMemoryModelUnsupported{…}` | `mmio_device_behavior_unmodeled` G:6461; `multi_lane_async_publication` G:7300; `async_access_missing_proxy_metadata` G:7846; `mixed_or_partial_read_from` G:8593; `unaligned_read_from` G:8645; `cross_proxy_publication_unmodeled` G:8684, 9128; `cross_proxy_rmw_ancestry_unmodeled` G:9189 |
| `cluster_barrier_warp_exit_unmodeled` | |
| `cluster_barrier_rearrival_without_wait_unmodeled` | |
| `resource_limit` | |
| `subset_execution` | |

**Core reasons** (`checker::Incomplete`):
- `EpochRegression`
- `EpochOverflow`
- `UnknownAsyncOp`
- `UnknownAlloc`
- `WaitExitUnproven`
- `WaitPredicateReadsUnstable`
- `SyncQualifierUnknown`
- `AsyncNeverCompleted`

**Core review advisories** (`FindingKind::Advisory`):
- `CrossCtaAsyncOrder`
- `UndeclaredProtocolWord`

---

## 8. Shared/TMEM shadow vs global shadow: can they be unified?

**Verdict: yes.** Every difference is one of three things:
- an actor-set or topology difference;
- a fence or scope table difference;
- a representation or performance choice;
- or a legacy inconsistency to be *resolved*, not preserved.

No difference needs a second algorithm. The prototype runs every test case,
across shared, global and TMEM, through one `Checker` (74 tests).

| # | Difference | Class | Unified treatment |
|---|---|---|---|
| 1 | Warp clocks + lane overlay (RS) vs per-lane components (G) | representation | warp scalar + sparse lane vectors |
| 2 | Per-cluster actors (RS) vs launch-wide actors (G) | actor set | one dense index; shared memory simply never sees foreign CTAs' accesses |
| 3 | Async registry: 14-bit generation/16-bit epoch (RS) vs launch-wide lease (G); G rejects multi-lane issuers | actor set + representation | async actor index; the issuing lane set is part of the fork |
| 4 | Morally-strong rule: RS unconditional RMW/RMW and strong loads exempt; G excludes generic loads and adds a scope diagnostic | **legacy inconsistency** (both wrong per R2/R9) | one PTX rule (§2.4); the carve-out is replaced by a `review` advisory |
| 5 | Release/acquire machinery lives in G; RS receives `SharedClockFrontier`s (RC:2434-2493) | fence table (layering) | cell entries carry `Rel` in every space |
| 6 | Bridge table `2×3×3` with aliasing (RS) vs `2×1` (G) | fence table | `2 × 3` keyed by the prior's domain, no aliasing |
| 7 | Copy-completion bridge only for already-bridged domains (RS) vs always (G:7446) | legacy inconsistency (minor) | always a2g for the op's own milestone (§3 row 13) |
| 8 | Bridge propagation: per-mask collapse (RS) vs merge in every join (G) | representation | per-lane overlays; propagate on acquire |
| 9 | Tick discipline: RS ticks at barriers, G does not | representation | the producer supplies epochs per instruction |
| 10 | Frontier replacement: RS contract-aware, G same-actor overwrite | **legacy bug in G** | contract-aware (§2.6) |
| 11 | Race policy: RS aborts, G continues | **legacy inconsistency** | continue always |
| 12 | GC: RS keeps retired cross-proxy history, G is proxy-blind | legacy bug in G | view-aware GC (§6.5) |
| 13 | Failure classification (RMW vs atomic-class) | cosmetic | strong or RMW → release/acquire |
| 14 | Spatial structure: stripes + `RwLock` + deferred reads (G) vs per-allocation `Arc` map + patches + tile windows (RS) | representation / concurrency | `IntervalShadow` per allocation; stripes become an ownership partition |
| 15 | TMEM owned by RS; G transports the TCGEN frontier | actor set | TMEM accesses use the `tcgen` view |

---

## 9. Unjustified, contradictory or ad-hoc legacy behaviour

Roughly ordered by impact.

1. **RS aborts on its first race; G continues** (§0.1). The verdicts and the read
   clearing depend on it. With fail-fast, one RS race hides every later one,
   including the second item in a test such as
   `test_native_same_warp_tmem_review.py`, which relies on review findings not
   aborting.
2. **RMW/RMW is always exempt in RS** (RS:6000-6004). This ignores scope, proxy
   and span. CTA-scoped atomics from two CTAs of a cluster on DSMEM are treated
   as morally strong. G would report a scope mismatch.
3. **The non-exact write collapse drops prior reads and RMW writers even for RMW**
   (RS:6726-6748, 6853-6871, 7176-7190). The exact path keeps them, so the result
   depends on geometry. That can miss a race.
4. **G same-actor frontier replacement ignores semantics** (G:3446-3456,
   3682-3683). Plain `st X` then `atom.gpu X` by one lane, followed by a remote
   unordered `atom.gpu X`: the plain-store race is lost.
5. **G GC is proxy-blind** (G:10153-10166). A cross-proxy pair can be lost after
   retirement. RS has `RetiredGenericHistory` for exactly this case.
6. **The compact (direct) RS path never sets `strong_scope`** (RS:5410, 5554,
   6683, 7113, 7794, 8143 use `from_parts`). Scoped shared atomics on this path
   lose the morally-strong exemption (false positive) and a strong atomic write
   clears reads.
7. **Cross-warp lane-stamped priors never fall back to the vector clock** (§2.5).
   An HB path that updates WC but not LO gives false positives. G release paths
   feed LO only when `order.has_release()` (RC:3851-3857, 4623-4627).
8. **`observed_by_frontier` uses only `shared_lanes` when present**
   (RS:4571-4575). Merged lane-aware and lane-less bridges can reject a covered
   access.
9. **Retention uses the non-acquired clock while validation uses the acquired
   one** (RS:4917, 4974 vs 4744-4746). Classification also uses the raw clock
   (RS:5962-5968, 4811-4814). That can turn a TMEM *review* into an *error*.
10. **WC merges lane-masked acquisitions into the whole warp** (RS:927-938,
    9243-9258). Same-proxy async witnesses — classic `cp.async` and the TMEM
    domain — are judged by WC, so a lane that did not wait inherits another
    lane's wait. ISA Q7 (`sync-isa-answers.md`) confirms that completion is
    **per thread**. The core acquires per lane (`async_group_wait_is_per_lane`).
11. **The full async-group token clock includes issuer history** (RC:5092-5153),
    whereas the mbarrier copy projection strips it (RS:1119-1154). `.read` waits
    acquire `source_read`. The core publishes only the op's own milestone; `.read`
    → `Milestone::Read` (`bulk_wait_read_does_not_publish_destination`).
12. **RS records mbarrier bulk-copy source reads at the issuer warp's clock**
    (RS:9043-9058), but G records them at the token clock (G:7328-7336).
    Consequently, after the issuer's `bar.sync`, another warp's overwrite of the
    source counts as "ordered" while the copy may still be in flight.
13. **Mbarrier and named-barrier HB has no scope** (SY:387-394, 626-632). A
    `.cta` arrive observed cross-CTA through `shared::cluster` still yields full
    HB.
14. **A missing cluster-barrier payload is treated as "relaxed only"**
    (RC:6933-6936), unlike named barriers (Err) and G (`ShadowRejected`).
15. **`grid.sync` gives no HB edge** (KE:1892-1896).
16. **Declared word:**
    - the fallback is not limited to async publications (G:8784);
    - a launch-value exit still takes the observed edge (G:8831-8859),
      contradicting G:8743-8746;
    - "earliest" means earliest after the observed prefix (G:8730-8736);
    - multi-lane post-images are taken after the whole warp batch, so
      intermediate RMW values are missing (KE:2997-3024);
    - history is kept for **every** 4- or 8-byte global atomic write, not only
      declared words, and is never retired (G:5051, 4129-4132 vs KE:3018-3025).
17. *(Closed in legacy's favour, R1.)* `test_native_racecheck_release_rmw_handoff.py`
    asserts a race for `L0: st data; all lanes atom.release flag; L1: ld.acquire
    flag; ld data`. Legacy is right:
    - PTX has observation order, not release sequences;
    - the 32 lanes are independent threads, and the coherence order of their
      same-address RMWs is unconstrained.

    The core now withholds sibling-lane heads and reports the race
    (`release_rmw_handoff_sibling_lanes_race`).
18. **Only the latest release fence per lane is kept** (G:8136). A narrower later
    fence drops an earlier wider head.
19. **The TCGEN cross-thread predecessor is all-or-nothing across lanes**
    (RC:2033-2059). Commit retires work at commit *issue* (RC:5446).
    `TcgenFenceFrontier::covers` and `merge` disagree on a missing descriptor
    (TF:107-111 vs 152-155).
20. **`RetiredCrossProxyHistory` is a hard error built on an over-approximation**
    (64 ranges or 4096 allocations; RS:3745-3753, 3894-3902) and carries no
    witness.
21. **Magic thresholds and panics instead of `incomplete`:**
    - process-global `u32` chunk id with `assert!` (RS:318-323);
    - `AppendOnlyTable` 2^28 (RS:2246-2289);
    - async epoch 16 bits with a silent clamp on import (RS:877);
    - generation windows 8 and 64 justified by MegaMoE (RC:6840-6849);
    - tile windows 32 / 2^16 / 1024 (RS:10086-10095).
22. **Stale comments:** G:9363-9370 (bypass check moved), G:875-877 and
    G:2468-2478 (structure descriptions).

---

## 10. Unified core: decisions and open items

**Decided in the prototype:**
- One conflict rule (PTX moral strength).
- Contract-aware eviction.
- Continue after races.
- Per-lane async completion.
- Two milestones per async op.
- Bridges keyed by the prior's window domain.
- tcgen ordering through `tcgen_rel` and the fence pair.
- Commit forwards the issuer's generic knowledge plus the tracked ops and their
  causal predecessors.
- Earliest-accepted declared-word rule, with an async-only fallback.
- Fail-closed `reads_memory` predicates.
- After the R1–R9 rulings:
  - the PTX moral-strength rule, with no atomic requirement and no load
    carve-out;
  - the undeclared-word advisory;
  - scoped mbarrier edges;
  - relaxed waits that need a `fence.acquire`;
  - named-barrier arrive as a source only;
  - lost cluster-barrier qualifiers → incomplete;
  - pairwise Fence-SC;
  - `.shared::cluster` proxy fences cover `shared::cta`;
  - cross-CTA async writers → review advisory;
  - sibling-lane RMW heads withheld.

**Open:**

| # | Open item | Notes |
|---|---|---|
| a | *Closed (R9).* | PTX semantics plus the `UndeclaredProtocolWord` review advisory. |
| b | *Closed (R4, R5).* | mbarrier edges are scoped, and the SyncTable must carry `.sem`/`.scope` on arrive and wait. Named barriers stay scopeless and participants-only. |
| c | `grid.sync` | Edge, or `incomplete`. |
| d | TensorMap release/acquire/consume | Not ported. |
| e | View-aware GC and async-slot reclamation | Not ported. |
| f | Inbox-drain merge of the global shadow (plan §2.4) | Not ported. |
| g | *Closed in legacy's favour (R1).* | §9.17 |
| i | `red` never forms an acquire pattern (§8.8) | The input needs a `returns_value` bit (atom vs red). Today a relaxed RMW parks its read heads for a later `fence.acquire`, which is wrong for `red`. |
| h | tcgen state is per warp in the prototype | The elected issuing lane makes this exact for current kernels. Per-lane tcgen is a mechanical extension. |

**Phase 3 (integration) status.** The core now lives in
`numsim-core/src/racecheck/`. `RaceObserver: Observer` adapts contract events;
see §11. The open items d, e and h are implemented there: TensorMap, GC,
slot reclaim and per-lane tcgen. Item i is also done: `red` never acquires,
using `Access::returns_value`.

**Input shape needed from the contract.** It is defined in
`numsim-race-core/src/input.rs`, the only file to change when the contract lands.

`Access` (hot path, per lane or per async op):
- `who`: `Lane{warp, lane, epoch}` | `Async{op, side: Read|Write}`;
- `alloc`, `range`;
- `kind` (R/W/RMW), `order`, `scope: Option`, `atomic`;
- `proxy` (Generic/Async/Tcgen);
- `domain` (window: global / `shared::cta` / `shared::cluster`; `None` for TMEM);
- `site`.

Epochs are per warp instruction; sibling lanes share one epoch.

`SyncEvent` (cold path):

| Event | Fields |
|---|---|
| `AllocBegin` / `AllocEnd` | |
| `DeclareWord` | |
| `WarpSync` | `mask` |
| `Arrive` | `obj`, `phase`, `release: Option<bool>`, `scope: Option<Scope>` |
| `Wait` | `obj`, `phase`, `acquire: Option<bool>`, `scope: Option<Scope>` |
| `Fence` | `AcqRel(s)`, `Sc(s)`, `ProxyAsync(d?)`, `TcgenBefore`, `TcgenAfter` |
| `AsyncIssue` | `op`, `lanes`, `kind ∈ {Copy, TcgenPipelined, TcgenLd, TcgenSt, TcgenCommit}`, `proxy`, `preds` (pipeline predecessors or commit-tracked ops, resolved by the SyncTable), `footprint` |
| `AsyncComplete` | `op`, `milestone`, `target: Phase{obj, phase} \| Warp{warp, lanes}` |
| `WaitVerdicts` | `lanes`, `alloc`, `range`, `scope`, verdict bitset, `observed`, `pred_reads` |

The SyncTable resolves:
- phases and generations;
- which async groups a `wait_group N` completes;
- per-thread group membership (ISA Q7);
- tcgen pipeline pairs (TF:46-65);
- the ops a commit tracks.

The checker sees only resolved milestones.


---

## 11. Integration with the contract (phase 3)

### Layout

| Path | What it is |
| --- | --- |
| `numsim-core/src/racecheck/` | the core: `clock`, `knowledge`, `shadow`, `cell`, `checker`; an alias layer `input`; the adapter `observer`; and `payload` |
| `numsim-race-core/` | re-exports the core and keeps the criterion benches |
| `numsim-core/tests/racecheck_*.rs` | 80 scenario tests that drive `RaceObserver` through contract `Access`/`SyncEvent` values |

### Adapter (`observer.rs`)

- A contract `Access` batch is split per `LaneSpan`.
- `Sem` is normalised:
  - `Volatile` and `Mmio` → relaxed at `.sys`;
  - `Sc` → an SC fence followed by `AcqRel`.
- `Proxy::ReadOnly` is checked as generic.
- `Local`, `Param` and `Reg` accesses are skipped, because they are
  thread-private.
- `Protocol` events, `FenceEvent::MbarrierInit` and `FenceEvent::ProxyAlias` are
  ignored. Synccheck owns mbarrier init. The shadow is keyed by physical bytes,
  so virtual aliases need nothing.
- `begin_launch` registers every arena allocation in the global, shared and
  tmem spaces.
- `end_launch` runs a final GC and finalises the launch. Outstanding async
  work becomes `AsyncNeverCompleted`.
- Events outside a launch become `EventOutsideLaunch`.

### Merge design

The scheduler is single-threaded, so there is one shadow and it is always
merged; `inbox_drain` is only a GC safe point. When CTA parallelism lands:
- shared memory and TMEM stay CTA-private;
- the global shadow is partitioned by stripe;
- a round's global accesses are applied at the receiving CTA's next drain, in
  `(round, cta, seq)` order;
- `end_launch` drains every buffer before finalising.

### View-aware GC (`Checker::gc`, every `gc_every` events and at drains)

1. Compute the meet, per view, of what every live actor knows:
   - for each warp, its `base` knowledge;
   - for lane-local tcgen state, the meet over the warp's lanes;
   - each in-flight async op's `k`.
2. A witness is **fully dead** when, for every proxy a future access to its
   space can use, the view `select_view(witness proxy, that proxy, domain)`
   observes it. Fully dead witnesses are dropped. The proxies per space are:
   - TMEM: tcgen;
   - otherwise: generic, async and tensormap.
3. Generic witnesses get a narrower rule while the allocation is not yet proxy
   sensitive. If such a witness is dead in its own proxy but not bridged:
   - it is folded into a per-allocation summary, one entry per
     `(actor, lane, write, window)`, holding the latest stamp and the byte hull;
   - the first non-generic access to the allocation is checked against the
     summary and marks the allocation sensitive.

   This is RS's `RetiredGenericHistory`, made view-correct; legacy G's
   proxy-blind floor is fixed.
4. The latest write that carries release heads is never dropped.

### Async-slot reclaim

- Once an op has fully completed and no witness names it, its slot is reused
  with `gen_base += 2`.
- A view that carries a newer generation was snapshotted after every live
  actor's `hb` observed the old one. Comparing an old summary stamp against it
  is therefore sound.
- Reclaimed `AsyncId`s are remembered, so a late predecessor reference is not
  reported as unknown.

### Epoch bound

The contract's `Actor::Warp.epoch` is a `u64`. The core keeps 32-bit epochs
inside its packed `(actor, epoch)` stamps, so it checks at most 2^32
instructions per warp per launch (in practice 2^32 − 2, because the core
reserves the top values). The adapter converts the epoch with
`u32::try_from`. On overflow it records `Incomplete::EpochOverflow` and skips
the event; it never truncates or panics. Async-slot epochs share the same
32-bit field, at 2 per slot generation.

### Test-migration rulings (`test-migration.md`)

**Phase 2 rulings** (see `racecheck-behaviour-deltas.md` T1–T6):
- **tcgen05 work.** Completed work (`tcgen05.wait::ld/st`, or a commit
  observed through its mbarrier) joins the waiter's hb and is ordered through
  ordinary thread sync. A tcgen05 op inherits completed work from its issuer's
  hb. The `before_thread_sync`/`after_thread_sync` pair is required only for
  uncompleted (pipelined) work. This refines §3 rows 19–23.
- **`st.async`/`red.async` `.release`** is a generic-proxy strong release
  whose head holds the issuer's knowledge at issue.
- **`SignalProtocolError`.** A race on declared-word bytes where one side is
  weak is reported with this kind.
- **Poll read-froms.** The read-from of a strong pure read of a declared word
  is held back until the warp's next event. A `WaitVerdicts` for that lane and
  word discards it; anything else applies it.
- **Plain-write wake-ups.** A wait explained only by a plain write is
  `WaitExitUnproven`.

- **Wide or narrow writes to a declared word.** These are numbered (V3), but a
  wait that accepts one is `SignalWriteNotRecorded`, the legacy
  `signal_write_not_recorded` (delta W5). Mixed-size accesses are outside the
  morally strong relation (PTX §8.7.2).
- **`alias_stale_read` is ported** (delta P7). It is driven by
  `SiteInfo::buffer`.
- **Same-rank `mapa` is the CTA's own `shared::cta` window**, because
  `mapa(p, own rank) == p`. The checker normalises the window by comparing the
  accessor's CTA with the allocation's CTA (delta X10).
- **Declared words may live in shared memory** (delta W6).
- **A qualifier-less remote `mbarrier.arrive.shared::cluster` defaults to
  `.cta`**, which gives `ScopeMismatch` (delta B7). Lowering keeps the PTX
  default.

### Fixes from the adversarial review (`checker-review.md`)

**Soundness**
- **S2 — eviction respects proxy and window.** A frontier entry evicts a
  prior only if both have the same proxy and the same address window. A
  plain write evicts only same-class readers. A `shared::cta` store therefore
  no longer hides an unbridged `shared::cluster` store.
- **S3 — a missing scope on a non-named barrier is a lost qualifier.**
  Named barriers are recognised by `ResourceId::Named`, not by
  `scope: None`. On an mbarrier or cluster barrier, a `None` scope gives
  `SyncQualifierUnknown` and no edge.
- **S4 — declared-word history numbering follows README decision 14.** Bit i
  is the i-th `(Access, lane)` write, lanes ascending, counted for every
  declared word the write overlaps. This deliberately differs from
  `observe.rs:25`, which says "per Access".
- **S6 — tensormap heads keep their releaser.** A head is
  `(releasing warp, scope, hb)`, and the acquire filters on the releasing
  fence rather than on each component's actor.
- **S7 / F2 — failed mbarrier scope checks are reported.** They emit
  `ScopeMismatch`, carrying both sites and an occurrence count, deduplicated
  per (release site, acquire site, scopes).

**False positives**
- **F1 — `CrossCtaAsyncOrder` is narrower.** It is emitted only when the
  current issuer's CTA did not itself observe the prior op's completion
  (through its mbarrier or a warp wait). Multicast and 2-CTA consumers no
  longer trigger it.
- **F3 — `fence.sc` history.** The latest `fence.sc` is kept per
  `(thread, scope)` instead of per thread.

**Robustness**
- **R1.** A completion that targets a warp outside the launch is reported as
  incomplete.
- **R2.** Incompletes are deduplicated with counts (`incomplete_counts`, and
  `occurrences` in the payload). Advisories are deduplicated through an
  index.
- **R3 — bounded growth.**
  - The wide-span table is interned.
  - Phases are kept for the last 4 per object.
  - `pending_acq` is deduplicated and capped at 1024; dropping entries only
    loses edges.
  - Tensormap ranges are shared through an `Arc`, and identical ranges are
    joined.
  - The reclaimed-op set is gone: a non-live predecessor adds no ordering,
    which is conservative.
  - A declared-word history payload that every live actor already holds is
    dropped by GC; its index is kept.
- **R4 — faster paths.**
  - `Clock::meet` and `filter` are rebuilt chunk by chunk.
  - `Clock::join` no longer deep-copies lane entries.
  - `Epochs::leq` does not allocate.
  - Retired-generic summaries are indexed by page, and retired witnesses are
    packed once.
  - Words are looked up per allocation.
- **R5.** GC uses per-cluster meets for shared memory and TMEM, so they
  retire without waiting for other CTAs or later waves. Global memory uses
  the launch-wide meet.
- **R6.** Chunk identities come from a process-global `AtomicU64`.
- **R7.** Zero-length spans are skipped instead of being reported as out of
  bounds.

**Contract**
- `SyncEvent.kernel` must match the launch, otherwise `KernelMismatch`
  (incomplete).
- A warp access with `ALL_LANES` is warp-collective and is attributed to
  lane 0.
- `red` no longer feeds `tcgen_in`.

**Cost.** `checker_tile_loop` went from 1.46 ms to 1.70 ms (phase map and
per-arrival sites); `checker_readers` went from 2.77 ms to 2.93 ms.

### Contract update (e2551cf, b3c72f3)

- Arrive/Wait carry qualifiers and scope. There is no adapter-side
  assumption: `None` → `SyncQualifierUnknown`.
- `WaitVerdicts` is judged per lane group.
- `Finding.attrs` carries every structured fact: `access_pair`,
  `ordering_*`, `proxy_bridge`, `hint`, `occurrences`, `reason`, and
  `prior`/`current` witness objects.
- New `FindingKind` variants replace `Other(..)`.
- Tensormap fences carry a scope, and the acquire carries a byte range.
- Per-thread async ops issued by several lanes are split per lane by the
  adapter (contract review item 5).

### Further ports

- **TensorMap.**
  - `TensormapRelease` snapshots `hb` into `tmap_rel`, a propagating bridge
    with own-warp rows.
  - `TensormapAcquire` moves what reached the lane into its local `g2t`.
  - A TMA inherits `g2t` and reads the descriptor through `Proxy::TensorMap`.
  - A consume without acquire is a `missing_proxy_bridge` race. Legacy aborted
    with a hard error.
- **Per-lane tcgen.** `tcgen`, `tcgen_in`, `tcgen_issued`, `tcgen_waited` and
  `tcgen_pub` are kept per PTX thread.
- **16-byte witnesses** (`cell::Witness`):
  - the layout is a packed stamp plus one metadata word;
  - spans that are too wide go to a side table;
  - sites come from a per-warp `(epoch, site)` table, pruned by GC, and from
    the async slot.

### Benchmarks

`numsim-race-core/benches/core.rs`, same machine. "Before" is the phase-2
prototype with 48-byte witnesses; "after" is the phase-3 core.

| Benchmark | Before | After |
| --- | --- | --- |
| `checker_tile_loop` (8.7K events) | 1.62 ms | 1.46 ms |
| `checker_readers` (9.7K events, 512-way read frontiers) | 4.01 ms | 2.77 ms |

The "after" column also includes two other optimisations:
- barrier arrivals are grouped by `(CTA, scope)`, so a wait costs O(groups);
- empty tcgen frontiers are skipped.

The micro-benches are unchanged: packed stamps, exact hits and the join memo.
