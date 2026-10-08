# Racecheck behaviour deltas vs. legacy

Use this list to review finding-snapshot diffs. The new behaviour is defined by
`core-rs/numsim-race-core/`, and the full specification is
`racecheck-semantics.md`. A snapshot diff that matches no row here is a
regression.

**Legacy columns.**
- **RS** is the legacy shared-memory/TMEM shadow (`race_shadow.rs` +
  `race_check.rs`).
- **G** is the legacy global shadow (`global_race.rs`).

"Confirmed" rows record a ruling that keeps legacy behaviour. They are listed so
that a future change is caught.

**ISA cites.** They use PTX 9.4 section numbers. Quotes and reasoning are in
`racecheck-isa-answers.md` (R1–R9). Test names refer to
`numsim-race-core/tests/`.

## Conflict rule and moral strength

| ID | Change | Legacy | New | ISA basis |
| --- | --- | --- | --- | --- |
| R1 | RMW/RMW pairs | RS: always exempt, with no scope, proxy or span check. G: exempt only when mutually morally strong. | Exempt only when morally strong: both strong, mutual scope inclusion, same proxy, complete overlap. Cross-CTA `atom.cta` on DSMEM, mixed-size atomics and cross-proxy atomics now race. | §8.7, §8.10.3 Litmus 2 (R2) |
| R2 | Strong non-atomic pairs (`st.relaxed` vs `ld.relaxed`, `st.release` vs `ld.acquire`) | RS and G: exempt only if both are atomic-class. Plain strong loads and stores race. | Exempt when morally strong; atomicity is not required. | §8.4 Table 20, §8.7 (R2) |
| R3 | Strong *generic* load vs morally strong store | G: never exempt; it falls through to HB and reports a race. RS: exempt. | Exempt (not a race). When the word is not a declared `wait_until` word, the pair produces a `review` advisory, `UndeclaredProtocolWord`. | §8.7.1, §8.10.3 (R9) |
| R4 | Unordered strong pair that fails only on scope | G: `scope_mismatch` diagnostic *instead of* a race. RS: race, unless both are RMW. | Data race with `missing_release_acquire`. `ScopeMismatch` remains for a release/acquire whose heads fail mutual inclusion at acquire time. | §8.7.1, §8.10.3 Litmus 2 (R2) |
| R5 | Weak load vs strong store | Race | Race (confirmed) | §8.4, §8.7 (R9) |
| R6 | Compact shared fast path | RS: `strong_scope` is never set, so scoped shared atomics lose the exemption and clear readers. | One rule on every path | §8.7 |
| R7 | Cross-proxy pairs | Judged only by the bridge | Judged only by the bridge, never morally strong, and unfenced pairs race even when barrier-ordered (confirmed) | §8.6, §8.9.5 (R3) |
| R8 | Failure label `missing_release_acquire` | RS: if either side is RMW. G: if either side is atomic-class or RMW. | If either side is strong or RMW | cosmetic |

## Reporting and shadow bookkeeping

| ID | Change | Legacy | New | ISA basis |
| --- | --- | --- | --- | --- |
| P1 | First shared/TMEM race | RS: `findings.push` then `Err` aborts execution (race_check.rs:3163-3167), so later races are never reported. G: continues. | Always continue; every race is reported, deduplicated by `(alloc, class, prior site, current site)` with a byte hull. | — |
| P2 | Readers after a write | RS: a plain write clears all reads; a non-exact write collapses the range even for RMW. G: a write drops only its own actor's read. | A plain write evicts the reads ordered before it. A strong write keeps the reads, because a later access that is morally strong with it can still race them. The geometry does not change the result. | §8.7.1 |
| P3 | Same-actor frontier replacement | G: overwrites regardless of semantics, so a plain store hidden behind the same lane's atom can lose a race. | Eviction requires happens-before **and** a subsuming conflict contract. | §8.7 |
| P4 | Async witness evidence | Issuer warp/lane only | Issuer warp/lane plus the async op id | — |
| P5 | Out-of-bounds access | `execution_error{oob}`. The report shows it only if no other error finding exists. | `OutOfBounds` finding; the access is skipped. | — |
| P7 | `alias_stale_read` advisory | `review`: a read through one logical name of pooled smem/TMEM observes bytes last written through another name | Ported. The checker compares `SiteInfo::buffer` of the reading site with that of the last ordered write in the same cell (shared memory and TMEM only). It produces the same `review` advisory with the legacy keys (`reader_buffer`, `writer_buffer`, `space`, `allocation_id`, `overlaps`). This needs lowering to fill `SiteInfo::buffer` with the logical name; until `FindingKind` gains a variant it is `Other("alias_stale_read")` (CONTRACT_REQUESTS W5-7). | — |
| P6 | Allocation end with an in-flight async footprint | An ordinary conflict, or `effect_commit_unobserved` at exit | `AsyncLifetime` finding at `AllocEnd`; `AsyncNeverCompleted` incomplete at launch end, only for ops the engine never landed. An uncommitted bulk TMA store at exit is committed implicitly and lands (sync-semantics §5 `Exit`), so it is neither | §9.7.10.28.1.1 |

## Release/acquire, observation order, fences

| ID | Change | Legacy | New | ISA basis |
| --- | --- | --- | --- | --- |
| A1 | Sibling lanes of one same-address RMW instruction | G: withholds sibling serialisation edges, so the race is reported (`test_native_racecheck_release_rmw_handoff.py`). | Confirmed: siblings never inherit each other's release heads. A reader of a sibling group gets only the heads that precede the whole instruction. | §8.9.1, §8.9.2, §8.9.4 (R1) |
| A2 | RMW chain across different, ordered instructions | G: inherits the predecessor payload when mutually morally strong | Same. Heads are kept separate, and each head is scope-checked against the acquirer. | §8.9.2, §8.9.4 (R1) |
| A3 | Acquire in shared memory | RS: edges arrive only via G-exported `SharedClockFrontier`, and only when `order.has_release()` | The same cell rule in every space | §8.8 |
| A4 | `fence.sc` ordering | G: keeps the latest head per `(cta, scope, proxy)` (G:8128-8133) and acquires the stored heads with the same proxy and mutual scope coverage (G:8096-8106). Collapsing a key to its latest head is exact only by transitivity through earlier fences. | Each new `fence.sc` acquires every earlier `fence.sc` that is morally strong with it: each scope includes the other thread. `fence.sc.cta` + `fence.sc.gpu` in one CTA are related; two `.cta` fences in different CTAs are not. | §8.9.3, §8.10.2 (R8) |
| A5 | `fence.sc` as `acq_rel` | G: release and acquire halves | Same (confirmed; ISA-silent, see S6) | §9.7.15.4 (R8) |
| A6 | Release fence head | G: latest per lane | Same (confirmed) | §8.8 |

## Barriers and mbarrier

| ID | Change | Legacy | New | ISA basis |
| --- | --- | --- | --- | --- |
| B1 | mbarrier arrive→wait scope | No scope field. A `.cta` arrive observed across CTAs, or a `.cluster` arrive observed by a `.cta` wait, gives full HB. | An arrival is acquired only if the arrive scope (default `.cta`) and the wait scope (default `.cta`) mutually include the two threads. | §8.9.4, §9.7.15.16.16, §9.7.15.16.19 (R4) |
| B2 | `.relaxed` mbarrier wait | Copy completions are acquired; arrivals are not. | Synchronises nothing. Arrivals **and** completions are parked until a later `fence.acquire`/`acq_rel` in the same thread, which then forms the acquire pattern. Test: `raw_try_wait_acquire_variants`. | §8.8, §9.7.15.16.19 (R4) |
| B3 | complete-tx scope | Unscoped | Release at `.cluster`. An acquire wait of any scope receives the copy's own bytes, without issuer history. | §9.7.10.28.4.1, §8.9.1.1 (R4) |
| B4 | Named barrier `bar.arrive`-only thread | No resume, so no acquire | Same (confirmed): `arrive` is a source only, and `sync`/`red` are targets among participants, with no scope | §8.9.4, §9.7.15.1 (R5) |
| B5 | Cluster barrier with a missing memory payload | RS: treated as "relaxed only" (race_check.rs:6933-6936). G: `ShadowRejected`. | Omitted qualifiers default to arrive `.release` / wait `.acquire` at `.cluster`. A qualifier lost in lowering → `SyncQualifierUnknown` incomplete, never relaxed. | §9.7.15.3 (R5) |
| B7 | Qualifier-less `mbarrier.arrive.shared::cluster` on a **peer** CTA's barrier | Clean (no scope model) | `ScopeMismatch` error: the ISA default is `.release.cta` for the `shared::cluster` form too, and a `.cta` arrive does not include the peer waiter. Lowering must not invent `.cluster`; kernels must write `.release.cluster` (and wait at `.cluster`). | §9.7.15.16.16 ("If the .scope qualifier is not specified then it defaults to .cta"); R4 |
| B6 | `bar.warp.sync` / `__syncwarp` | RS: the WC clock merges lane-masked acquisitions into the whole warp | Ordering only among masked lanes; no ordering for in-flight async ops | §9.7.15.2, §8.5; CUDA PG (R7) |

## Proxies and async operations

| ID | Change | Legacy | New | ISA basis |
| --- | --- | --- | --- | --- |
| X1 | `fence.proxy.async.shared::cluster` on a `shared::cta` prior | Legacy test `shared_cta_proxy_fence[.shared::cluster]` expects a race | Covered (clean). The `shared::cta` window lies inside the `shared::cluster` window. | §5.1.7, §9.7.15.4 (R3) |
| X2 | `fence.proxy.async.shared::cta` on a prior made through a **remote-rank** `shared::cluster` (mapa) window | Race | Race (confirmed; ISA-silent, see S2). A *same-rank* `mapa` is the CTA's own `shared::cta` window (X10). Legacy also resolved it that way, so the old "Legacy: Race" entry did not apply to same-rank `mapa`. | §9.7.15.4 (R3) |
| X10 | Same-rank `mapa` (`mapa(p, own rank) == p`) | Resolved to `shared::cta`: a `.shared::cta` fence is clean, a `.shared::cluster` fence races | Normalised to the `shared::cta` window. A `.shared::cta` fence is clean (as legacy). A `.shared::cluster` fence is also clean, because it covers `shared::cta` (X1); this differs from legacy. | §9.7.9.20 (`mapa`), §5.1.7, §9.7.15.4 |
| X3 | Implicit async→generic bridge at copy completion | RS: only for domains already bridged for every lane. G: always. | Always, but only for the op's own milestone (its results), never issuer history | §9.7.10.28.2 (R3) |
| X4 | Async-proxy writes issued from different CTAs, ordered by base causality | Ordered, no finding | `review` advisory `CrossCtaAsyncOrder` | §8.9.5 (R3; ISA-silent, see S3) |
| X5 | Async-group completion visibility | RS: the WC clock merges a lane's group wait into the whole warp | Per thread: the waiting lane only | §9.7.10.28.1.1 (sync Q7, R6) |
| X6 | `cp.async.bulk.wait_group.read` | RS: acquires the `source_read` clock, which includes issuer history | Read-side milestone only: source reuse is safe, and destination writes stay unpublished | §9.7.10.28.6.2 (R6) |
| X7 | Full `wait_group` | RS: the full token clock includes issuer history | The op's write milestone only | §8.9.1.1 (R4, R6) |
| X8 | mbarrier bulk-copy source reads | RS: stamped at the issuer warp's clock. G: stamped at the token. | Stamped at the async op actor, in every space | §8.9.1.1 |
| X9 | `tcgen05.commit` completion | Retires work at commit *issue* | Retires at completion; it has an implicit `before_thread_sync` but no implicit async→generic bridge (confirmed by `tcgen_copy_completion_vs_generic_reuse`) | §9.7.18 (tcgen05) |

## Declared words (`wait_until`)

| ID | Change | Legacy | New | ISA basis |
| --- | --- | --- | --- | --- |
| W1 | Edge source | The earliest accepted write *after the already-observed prefix* (G:8718-8737) | The earliest accepted write over the full history (verdict bitset); schedule independent | §8.9.4 |
| W2 | Observed-version fallback | Any exit the history cannot name: truncated history, wide writes, plain writes, launch-value exits | Only async publications. A launch-value exit owes no edge. Other unexplained exits → `WaitExitUnproven` incomplete. | — (fail closed) |
| W3 | Predicate that reads memory (`PredProgram.reads_memory`) | Not modelled | Uses the bitset iff every write to the predicate's inputs happens before the wait; otherwise `WaitPredicateReadsUnstable` incomplete | — (fail closed) |
| W4 | History scope | Every 4- or 8-byte global atomic write, never retired | Declared words only | — |
| W5 | A wait whose accepted write does not exactly cover the declared word (wider, e.g. a `.v4.b32` release over a 4-byte word; narrower; misaligned) | `incomplete` (`signal_write_not_recorded`) | Same, implemented: the history entry is marked mixed-size, and a wait that accepts it is `SignalWriteNotRecorded`. Such a write is still numbered (V3). | §8.7.2: mixed-size accesses are not morally strong, so single-copy atomicity with the word's polls does not hold |
| W6 | Declared words outside global memory | G: global only (a `shared::cluster` poll could never be declared) | A declared word may live in any allocation, including shared memory reached through `shared::cta` or `shared::cluster`. Lowering's `sync_words` hint carries them and the engine emits `DeclareWord`/`WaitVerdicts` for them. A declared shared word gets no `UndeclaredProtocolWord` advisory. | — |

## ISA-silent points and their fail-closed choices

| ID | Silent point | Choice | Where |
| --- | --- | --- | --- |
| S1 | Order among same-address RMWs of one warp instruction (§9.7.15.5) | Unconstrained. Sibling lanes never inherit each other's release heads. | A1 (R1) |
| S2 | Whether `fence.proxy.async.shared::cta` covers an object reached through a remote-rank `mapa` `shared::cluster` address | It does not. A same-rank `mapa` is not ambiguous: it is the CTA's own window (X10). | X2 (R3) |
| S3 | Which thread block an async op belongs to, for same-proxy preservation (§8.9.5) | Cross-CTA async-proxy pairs ordered by base causality → `review` advisory, not error | X4 (R3) |
| S4 | Which thread an async op's complete-tx counts as executing, for the mutual-scope test | An acquire wait of any scope receives the copy's own bytes. Thread arrives keep the mutual-scope test. | B3 (R4) |
| S5 | `bar.warp.sync` and in-flight async ops | No ordering for in-flight async ops | B6 (R7) |
| S6 | Whether `fence.sc` is also an `acq_rel` fence | Treated as `acq_rel` at its scope | A5 (R8) |
| S7 | Async copy (bulk/TMA, e.g. a shared→global tensor store) still in flight when its CTA exits, with no final `cp.async.bulk.wait_group` (V2C-5: deepgemm, flashmla, cudnn bsa/gdn/dsa) | Clean | **Coordinator ruling, ISA-silent.** The ISA does not say whether CTA exit drains outstanding bulk reads of shared memory. Hardware keeps a CTA's shared memory until its outstanding bulk ops complete, and these production kernels run correctly. Implementation: a shared allocation ends only at CTA exit, so a `Copy`-class op still in flight at that `AllocEnd` is marked drained, and it raises neither `AsyncLifetime` (`async_lifetime_not_drained`) nor `AsyncNeverCompleted` (`effect_commit_unobserved`). The copy's accesses stay unordered with later work, so a race with a later access by another still-running CTA is still reported. Non-shared allocation ends keep `AsyncLifetime`. The contract's `AsyncClass::Copy` does not separate bulk from non-bulk `cp.async`, so both are drained. | ISA silent; ruling |

## Phase 3 (integration)

| ID | Change | Legacy | New | ISA basis |
| --- | --- | --- | --- | --- |
| I1 | TensorMap consume without `fence.proxy.tensormap::generic.acquire` | G: hard `Err` (aborts the run) | `missing_proxy_bridge` data race. A release without an acquire, or an acquire without a release, is reported as a race. | §9.7.15.4 |
| I2 | `red` never acquires | G: `can_acquire` = Atomic class only | Same (confirmed), using `Access::returns_value`. A relaxed `red` + `fence.acquire` gives no edge. | §8.8 |
| I3 | mbarrier / cluster arrive-wait qualifiers | No scope. A missing payload is treated as relaxed. | The contract now carries `release`/`acquire: Option<bool>` and `scope`. Edges require mutual scope inclusion. A lost qualifier (`None`) → `SyncQualifierUnknown` incomplete. | §9.7.15.16.16, §9.7.15.3 |
| I4 | GC of generic witnesses | RS: retired generic history with 64-range coalescing (spurious errors possible). G: proxy-blind floor. | Per-`(actor, lane, kind, window)` summary with a byte hull, checked on the first non-generic access. Fully-dead witnesses are dropped only when every view observes them. | §8.9.5 |
| I5 | tcgen ordering state | Per warp in RS lane-order layers; per thread in TG | Per lane (PTX thread) | §9.7.18 |
| I6 | Multi-lane per-thread async ops (cp.async by several lanes in one instruction) | RS: one token, completion merged into the warp clock | The adapter splits the op into one virtual actor per issuing lane. `AsyncComplete{Warp{lanes}}` completes only those lanes' copies. A span that does not name its lane → `AsyncLaneUnknown` incomplete. | §9.7.10.28.1.1 (contract review item 5) |
| I7 | `wait_until` verdicts | One verdict per wait (G: per lane, conjunctive plan) | Per lane group. Each group's earliest accepted write gives that group's edge. | §8.9.4 (contract review item 6) |
| I8 | Tensormap acquire | Descriptor-generation frontiers, scope checked | Release snapshots per scope. The acquire keeps only releasers whose scope mutually includes it, and only for the acquired byte range. A TMA's descriptor read is checked per acquired range. | §9.7.15.4 |

## Review fixes (`checker-review.md`)

| ID | Change | Before (committed core) | New | Basis |
| --- | --- | --- | --- | --- |
| V1 | Frontier eviction across window or proxy (S2) | A `shared::cta` store evicted a `shared::cluster` store, so an X2 race was lost | Eviction requires the same proxy and the same window | X2, §9.7.15.4 |
| V2 | `scope: None` on an mbarrier or cluster barrier (S3) | Unconditional HB | `SyncQualifierUnknown`, no edge | R4/R5 |
| V3 | Declared-word history numbering (S4) | Per lane, first overlapping word only | Per `(Access, lane)`, lanes ascending, for every overlapping word (README decision 14) | contract |
| V4 | Tensormap acquire filter (S6) | Filtered by each component's actor | Filtered by the releasing fence's warp and scope | §8.9.4 |
| V5 | Failed mbarrier scope check (S7, F2) | Silent, so it surfaced as an unlabelled race | `ScopeMismatch` with sites, one per site pair, with a count | R4 |
| V6 | `CrossCtaAsyncOrder` (F1) | Every multicast or 2-CTA consumer | Only when the consumer CTA did not observe the prior op's completion | §8.9.5, §9.7.15.16.19 |
| V7 | `fence.sc` history (F3) | Latest per thread | Latest per `(thread, scope)` | §8.9.3 |
| V8 | Zero-length span (R7) | `OutOfBounds` | Skipped | — |
| V9 | Events of another kernel | Merged | `KernelMismatch` (incomplete) | contract |
| V10 | Repeated incompletes (R2) | One entry each | One entry with an occurrence count | — |
| V11 | Sibling lanes of one warp instruction storing to the same bytes (V2C-16: every lane writes `s_g[t]`) | Clean | Clean. Before this fix: `missing_same_warp_lane_order` write/write. The CUDA Programming Guide defines this case: when a non-atomic warp instruction writes one location from several threads, the writes are serialised and one of them is the final value. So it is not reported as a race. Stores by *different* instructions still need a warp sync. | CUDA PG, "Shared Memory" / warp serialisation of same-address writes |
| V12 | `alias_stale_read` with an unnamed buffer | — | No advisory when either site has no logical buffer name. An unnamed buffer has no identity to compare, so it is never a false "stale name". | — |

## Test-migration phase 2 rulings

| ID | Change | Legacy | New (before → after this change) | Basis |
| --- | --- | --- | --- | --- |
| T1 | tcgen05 work across a plain thread sync (`cta_sync`, mbarrier, flag) | Waited `tcgen05.st` + `fence::after_thread_sync` + `cta_sync` publishes the stores to another warp's MMA (clean) | Before: only the fence pair (`before_thread_sync` … `after_thread_sync`) carried any tcgen05 work, so this was a race. After: **completed** work (waited by `tcgen05.wait::ld/st`, or committed and observed through the mbarrier) is ordered like any memory effect the thread observed, through ordinary hb. A tcgen05 op is issued with the completed tcgen05 work in its issuer's hb. The fence pair is required only for **uncompleted** (pipelined, unwaited) work. So the legacy kernel is clean; unwaited stores still race. | §9.7.18 (tcgen05.wait: "all prior tcgen05.st … have completed"); completion observed in program order before the sync |
| T2 | `st.async` / `red.async` `.release` (global, no mbarrier) | Publication: orders the issuer's pre-issue writes | Before: no release head, so a race. After: a strong release at `.scope` carrying the issuer's knowledge at issue. A waiter that reads from it (or `wait_until`s it) synchronises; post-issue work is not published. These accesses are in the **generic** proxy (CONTRACT_REQUESTS W5-8). | §9.7.10.12, §9.7.15.7 ("strong memory operation with .release semantics at the scope"; "performed in the generic proxy") |
| T3 | A plain access racing on a declared `wait_until` word | `signal_protocol_error` (declared-word bypass), not `data_race` | Restored as a distinct kind, `SignalProtocolError` (error, same evidence and hint). Rule: a race on declared-word bytes where at least one side is weak. | protocol rule (legacy G:137-176) |
| T4 | Acquiring `wait_until` poll | Edge from the earliest accepted write | Before: the poll `Access` gave a read-from edge to the latest write, overriding W1. After: the read-from of a strong pure read on a declared word is held back. A `WaitVerdicts` for that lane and word replaces it; any other event of the warp applies it as an ordinary read-from (raw polls keep their edge). | W1, §8.9.4 |
| T5 | `wait_until` exit explained only by a plain (weak) write | `analysis_incomplete` | `WaitExitUnproven` (incomplete; payload reason `wait_exit_unproven`), together with the `SignalProtocolError` for the plain write. Before: no edge and no incomplete. | W2 |
| T6 | Finding kind names | `oob` (execution_error), `data_race`, `signal_protocol_error`, `scope_mismatch`, `tmem_lifetime_review`, `alias_stale_read` | Contract `FindingKind`s serialize as `out_of_bounds`, `data_race` / `proxy_race` / `async_race`, `signal_protocol_error`, `scope_mismatch`, `tmem_lifetime_review`, `alias_stale_read`. The racecheck payload keeps the legacy `kind` strings for races (`data_race`, `tmem_lifetime_review`, `signal_protocol_error`, `scope_mismatch`, `alias_stale_read`). `divergence`, `bad_address` and `named_barrier_contract_mismatch` are runtime and synccheck kinds, not racecheck, and belong in those delta lists. | — |
| T7 | `tcgen05.commit` whose `preds` miss earlier in-flight pipelined ops | All prior tcgen05 work of the thread completes with the commit | The commit publishes the transitive pipeline closure of its tracked ops (preds of preds, stopping at ops already completed), and marks them done. This is a guard against incomplete `preds`; with correct events it adds nothing. It does **not** widen a restricted commit (T11): the closure follows `preds` only. | PTX `tcgen05.commit` ("all prior asynchronous tcgen05 operations initiated by the executing thread") |
| T8 | `alias_stale_read` (V2C-18) | Legacy `AliasTracker`: the last *named* warp-lane writer of each byte; a warp-lane read through another name reports exactly those bytes; one advisory per (reader name, writer name, reader warp and site, writer warp and site) | Before: the checker used the shadow cell's latest write, required it to be ordered before the read, grouped advisories by reader site, and reported the whole read span. After: a separate per-allocation map of last-named-writer segments (shared and TMEM), which unnamed writes and async copies do not touch. There is no ordering requirement: the advisory is about logical identity, as in legacy. `overlaps` lists the exact merged byte segments of every occurrence, and there is no `overlap` key. | review advisory; no PTX rule |
| T9 | TMEM evidence (V2C-34) | Column footprint of the TMEM conflict | `Evidence.bytes` stays the taddr-encoded hull (byte = (lane * 512 + column) * 4). Each TMEM finding also carries `attrs.tmem_lanes` and `attrs.tmem_columns`, each a list `[[lo, hi], ...]` in lane and column units (not bytes). The list holds the exact overlap of every occurrence, sorted, and merges ranges only where they overlap or are adjacent. A range that crosses lanes contributes one column segment per lane. A multi-lane hull does not project to columns. | PTX TMEM addressing (taddr = lane << 16 \| column) |
| T10 | `tcgen05.ld` / `tcgen05.st` never followed by `tcgen05.wait::ld/st` (V2C-5) | Clean | Before: `AsyncNeverCompleted`, which was `incomplete` (`effect_commit_unobserved`). After: not incomplete. Every access of these observer-only ops is delivered at issue, so nothing is unobserved. The accesses stay unordered with later work, so races and `TmemLifetimeReview` still fire. Using the destination registers before the wait is a `Space::Reg` matter (V2C-20). | PTX `tcgen05.wait::ld/st` (completion of prior ld/st; no implicit wait at exit) |
| T11 | `tcgen05.commit…sync_restrict::shared::read::mma::a` | Completion orders only the MMA's shared-A read | Expected events (CONTRACT_REQUESTS W5-10): the shared-A read is its own async op with no preds, and the restricted commit tracks only that op. The checker then publishes A and not B; the regression test is `restricted_commit_publishes_only_shared_a_read`. Today the engine tracks the whole MMA, so the B overwrite is a false negative until W2 lands. | PTX `.sync_restrict` commit (legacy `MmaSharedARead`) |
| T12 | `tmem_lifetime_review`, one finding per (load site, store site) pair (`msa_sparse_atten_fwd_nvfp4_kv_sm100`) | Legacy: 2 findings, columns 64–80 and 96–112 | v2: 4 findings, columns 64–80, 80–96, 96–112 and 112–128. **True positives.** The kernel loads S with an un-waited `tcgen05.ld` (kernel.py:2763: one site, an `unroll(4)` loop of 32-column loads; the source comments that it has no `tcgen05.wait::ld`). It then publishes P over the same columns with four 16-column `tcgen05.st` instructions (kernel.py:2960, `for k in range(4)`, `rep = 16`). Each store overwrites columns the load may still be reading, so all four (load, store k) pairs are the same hazard; none is a hull or a split. Legacy reports only stores 0 and 2. | PTX `tcgen05.wait::ld` (loads complete only at the wait) |

## Known gaps (not deltas yet)

- **CTA-parallel inbox-drain merge.** The scheduler is single-threaded today,
  so there is nothing to merge yet. The design is in `racecheck-semantics.md`
  §11.
- **mbarrier scope and lost `.sem` qualifiers.** These need a contract change
  (W5-1).
