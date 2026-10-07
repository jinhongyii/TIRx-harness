---
orphan: true
---

# Racecheck semantics: ISA answers to memory-model questions R1–R9

These answers resolve the open items in `racecheck-semantics.md` §8, §9 and §10
(in particular §9.2, §9.13, §9.14, §9.17 and §10 a, b, g). They are checked
against:

- **PTX.** *Parallel Thread Execution ISA*, version **9.4**
  (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), fetched
  2026-10-07. Section numbers refer to that version. Chapter 8 is the "Memory
  Consistency Model".
- **CUDA PG.** *CUDA Programming Guide*, "C++ Language Extensions" (only for
  `__syncwarp`).

Quotes are verbatim, except that footnote markers are dropped. Where the ISA is
silent, the answer says so and recommends a fail-closed behaviour.

**Verdict labels.** *Legacy* = the current checker (RS = shared/TMEM shadow,
G = global shadow). *Prototype* = `numsim-race-core`.

---

## Core definitions used throughout

**Morally strong** (§8.7, *Morally strong operations*):

> "Two operations are said to be morally strong relative to each other if they
> satisfy all of the following conditions:
> - The operations are related in program order (i.e, they are both executed by
>   the same thread), or each operation is strong and specifies a scope that
>   includes the thread executing the other operation.
> - Both operations are performed via the same proxy.
> - If both are memory operations, then they overlap completely."

**Data race** (§8.7.1, *Conflict and Data-races*):

> "Two overlapping memory operations are said to conflict when at least one of
> them is a write."
>
> "Two conflicting memory operations are said to be in a data-race if they are
> not related in causality order and they are not morally strong."

**Strong / weak** (§8.4, *Operation types*, Table 20):

> "strong operation: A memory fence operation, or a memory operation with a
> .relaxed, .acquire, .release, .acq_rel, .volatile, or .mmio qualifier."
>
> "weak operation: An ld or st instruction with a .weak qualifier."
>
> "atomic operation: atom or red instruction."

The `atom` default is `.relaxed`, so an `atom` is always strong. Its default scope
is `.gpu` (§9.7.15.5): "If the .sem qualifier is absent, .relaxed is assumed by
default." and "If the .scope qualifier is absent, .gpu scope is assumed by
default."

**Scope** (§8.5, *Scope*):

> "Each strong operation must specify a scope, which is the set of threads that
> may interact directly with that operation and establish any of the relations
> described in the memory consistency model."
>
> "Note that the warp is not a scope; the CTA is the smallest collection of
> threads that qualifies as a scope in the memory consistency model."

---

## R1. Release sequences through RMW chains; `test_native_racecheck_release_rmw_handoff.py`

**ANSWER.**

PTX has **no "release sequence" object**. The equivalent is *observation order*.
A write W precedes a read R in observation order if either:
- R reads from W and the two are morally strong; or
- there is a chain of **atomic operations** (`atom` *or* `red`) between them,
  where each link reads the value written by the previous one and each link is
  morally strong.

A release pattern X synchronises with an acquire pattern Y when both hold:
- a write in X precedes a read in Y in observation order;
- the *first* operation of X and the *last* operation of Y are morally strong
  with each other.

**So yes: an RMW chain extends the synchronisation.** An `ld.acquire` that reads
from an RMW Z, where Z read from L0's `atom.release` (directly or through further
atomics), synchronises with L0's release.

Differences from C++20:
1. The chain is defined by reads-from (observation) links, not by "subsequent in
   modification order". With atomicity (§8.10.3) the two coincide for morally
   strong RMWs.
2. Every link must be morally strong, and the head and the acquire must also be
   morally strong with *each other*. Scope therefore matters at both ends.
3. The release pattern also includes "release op; strong write in program order"
   and "fence.release; strong write". These are C++11-style same-thread
   continuations that C++20 removed.
4. `red` can act as a link, but it never forms an acquire pattern (§8.11.1).

**The legacy test is correct; the prototype's `clean` verdict is not.** The 32
lanes are 32 *threads*. Program order does not relate them (§8.9.1), and the warp
is not a scope, so the coherence order among the 32 same-address RMWs of one warp
instruction is unconstrained at runtime.

Synchronisation of L1 with L0 holds **only in executions where L1's `ld.acquire`
reads L0's RMW or a write after it in coherence order**:
- Per-location SC (§8.10.5) forces L1's load to read L1's own RMW or later.
- That guarantees the L0→L1 chain *only* when L0's RMW precedes L1's RMW in
  coherence order.

Take the legal execution where L1's RMW is coherence-ordered before L0's, and L1's
load reads from L1's own RMW, or from any RMW before L0's:
- no observation-order path exists from L0's release to L1's acquire;
- L1's own `atom.release` is not an acquire;
- so `st data` (L0) and `ld data` (L1) conflict, are weak, and are not causally
  ordered.

That is a data race (§8.7.1).

The prototype comment says L0 "heads a sequence that every later RMW of the chain
continues". That is true, but nothing makes L1's RMW *later*. The prototype
(`checker.rs` ~L572-595) takes `cell.writes.last()`, which is the simulator's
lane serialisation (lane 0 first), so it certifies one schedule only. The
companion test (`bar.warp.sync` between the RMW and the acquire) is correctly
`clean`; see R7.

**QUOTE** (§8.9.2, *Observation Order*):

> "Observation order relates a write W to a read R through an optional sequence
> of atomic read-modify-write operations.
> A write W precedes a read R in observation order if:
> - R and W are morally strong and R reads the value written by W, or
> - For some atomic operation Z, W precedes Z and Z precedes R in observation
>   order."

**QUOTE** (§8.9.4, *Memory synchronization*):

> "A release pattern X synchronizes with an acquire pattern Y, if a write
> operation in X precedes a read operation in Y in observation order, and the
> first operation in X and the last operation in Y are morally strong."

**QUOTE** (§8.8, *Release and Acquire Patterns*):

> "A release pattern on a location M consists of one of the following: A release
> operation on M … Or a release or acquire-release operation on M followed by a
> strong write on M in program order … Or a release or acquire-release memory
> fence followed by a strong write on M in program order …"
>
> "Note that while atomic reductions conceptually perform a strong read as part
> of its read-modify-write sequence, this strong read does not form an acquire
> pattern."

**QUOTE** (§8.9.1, *Program Order*):

> "It is a transitive relation that forms a total order over the operations
> performed by the thread, but does not relate operations from different
> threads."

**QUOTE** (§8.10.3, *Atomicity*, RMW atomicity):

> "When an atomic operation A and a write W overlap and are morally strong, then
> the following two communications cannot both exist in the same execution …:
> A reads any byte from a write W' that precedes W in coherence order. A follows
> W in coherence order."

The `atom` instruction description (§9.7.15.5) does not specify an order among
the threads of one warp instruction that target the same address. **The ISA is
silent**, so the order must be treated as unconstrained.

**SECTION.** §8.8, §8.9.1, §8.9.2, §8.9.4, §8.10.3, §8.10.5, §8.11.1.

**Verdict.**
- **Legacy correct.** It withholds sibling-lane RMW serialisation edges and
  reports the race.
- **Prototype wrong** for sibling lanes of one instruction.

Recommended rule (fail-closed): for a multi-lane, same-address RMW instruction, a
sibling lane j may inherit through the chain only:
- the release heads of writes that precede the *whole* instruction;
- lane j's own head.

It must not inherit the heads of other sibling lanes. RMW chains across
*different* instructions that are already ordered (by program order or HB) can
keep the observed-order chain. Close §9.17 and §10 g in favour of legacy, and
flip `release_rmw_handoff_follows_release_sequence` to expect a race.

---

## R2. Morally strong: exact definition; RMW/RMW pairs; non-covering scopes

**ANSWER.** The definition is the §8.7 quote above. For two operations in
**different threads**, all of these are required:
- **(a) Both strong.** Weak `ld`/`st` never qualify; `atom`/`red` always do.
- **(b) Mutual scope inclusion.** A's scope instance, relative to A's thread,
  contains B's thread, *and* vice versa.
- **(c) Same proxy.**
- **(d) Complete overlap.** The memory locations must be identical, not just
  intersecting.

Same-thread pairs are morally strong by program order. Atomicity does **not**
appear in the definition: `st.relaxed` vs `ld.relaxed` qualifies.

**There is no RMW/RMW exemption.** RMW pairs follow the same rule. Litmus Test 2 of
§8.10.3 is exactly the case of two relaxed atomics whose scopes do not cover each
other: `atom.cta` and `atom.gpu` in different CTAs are *not* morally strong.

Two such atomics conflict, since both write. Unless causality orders them, they
**are a data race**, and atomicity (a lost update) is not guaranteed.
- Mixed-size (partially overlapping) strong accesses are never morally strong,
  so they race when unordered.
- With a mixed-size data race, the axioms do not apply at all (§8.7.2).
- §8.7.2 keeps only a weaker guarantee: RMW atomicity between overlapping atomics
  with mutually including scopes.

Note §8.3: the scope is further limited by the state space. For example,
`.shared` + `.sys` behaves as `.cluster`. This never *widens* inclusion.

**QUOTE** (§8.10.3, *Atomicity*, Litmus Test 2):

> "T1: A1: atom.cta.inc.u32 %r0, [x]; T2 (In a different CTA): A2:
> atom.gpu.inc.u32 %r0, [x]; FINAL STATE: x == 1 OR x == 2. Atomicity is not
> guaranteed if the operations are not morally strong."

**QUOTE** (§8.7.2, *Limitations on Mixed-size Data-races*):

> "The axioms in the memory consistency model do not apply if a PTX program
> contains one or more mixed-size data-races."
>
> "… for every pair of overlapping atomic operations A1 and A2 such that each
> specifies a scope that includes the other: Either the read-modify-write
> operation specified by A1 is performed completely before A2 is initiated, or
> vice versa."

**QUOTE** (§8.3, *State spaces*):

> "the synchronizing effect of the PTX instruction ld.relaxed.shared.sys is
> identical to that of ld.relaxed.shared.cluster, since no thread outside the
> same cluster can execute an operation that accesses the same memory location."

**SECTION.** §8.4, §8.5, §8.7, §8.7.1, §8.7.2, §8.10.3.

**Verdict.**

| Rule | Status | Why |
|---|---|---|
| Legacy RS unconditional RMW/RMW exemption (§9.2) | **Wrong** | False negatives: cross-CTA `atom.cta` on DSMEM; mixed-size atomics; cross-proxy atomics such as `cp.reduce.async.bulk` vs generic `red`. |
| Legacy RS non-RMW branch | Matches PTX | |
| Legacy G "both atomic-class" | **Too strict** | PTX does not need atomics: `st.relaxed` vs `ld.relaxed` is morally strong. |
| Legacy G scope-mismatch diagnostic | Should be a race | Fine as extra context on the race. |
| Prototype | **Correct** | Strong + same proxy + exact span + mutual cover, applied to RMW too. |

---

## R3. Generic vs async proxy; `fence.proxy.async` directions

**ANSWER.** Yes. A generic-proxy access and an async-proxy access to the same
location are **never morally strong**, because they use different proxies.

They are also **not in causality order** merely because base causality orders
them. *Proxy-preserved* base causality order holds only when one of these is true:
- both accesses are generic, to the same address;
- both use the same proxy and the same thread block;
- they are aliases with an alias proxy fence on the path.

So an unfenced generic↔async conflicting pair is a data race, even when a barrier
or a release/acquire orders it.

The formal §8.9.5 clauses name only the *alias* proxy fence. The async case is
stated normatively in §8.6 and in the `fence.proxy` and *Async Proxy*
descriptions: "a proxy fence is required". The ISA does not spell out the exact
causality-path clause for `fence.proxy.async`. Model it by analogy to the alias
clause: a `fence.proxy.async` of the right state space must lie on the
base-causality path between the two accesses.

**`fence.proxy.async` directions.**
- It is **bi-directional**. It establishes both generic→async ordering (prior
  generic access, later async access) and async→generic ordering (prior async,
  later generic).
- It is **thread-local**: there is no scope, and it does not synchronise by
  itself. It composes with other synchronisation along a causality path.
- The state-space suffix limits it to "operations performed on objects in the
  state space specified". With no suffix it covers all state spaces.

`.global`, `.shared::cta` and `.shared::cluster` select the state space. They do
not select a direction. Uni-directional `.release` / `.acquire` proxy fences exist
only for `tensormap::generic`, `fabric` and `alias`, plus the
`async::generic.{acquire,release}.sync_restrict` cluster forms.

Completion of `cp{.reduce}.async.bulk` (and of `wgmma`) carries an **implicit
generic-async proxy fence**. Once completion is observed, the async writes are
visible to the generic proxy. That covers the async→generic direction for the
op's own results only. Generic→async (for example, smem written by threads and
then read by a TMA store) always needs an explicit `fence.proxy.async`.

**Window containment.** §5.1.7 says "The addresses in the .shared::cta window
also fall within the .shared::cluster window." A `fence.proxy.async.shared::cluster`
is therefore defensibly a superset that covers objects in the executing CTA's
smem. The ISA is **silent** on whether `fence.proxy.async.shared::cta` covers a
same-CTA object reached through a `mapa` `shared::cluster` address. The
prototype's no-aliasing rule is fail-closed for that direction. Consider
widening it so that `.shared::cluster` covers the `shared::cta` domain.

**Same proxy, different thread blocks.** The "same proxy, and by the same thread
block" clause means two **async-proxy** accesses from *different* CTAs, for
example multicast TMA writes into one CTA's smem issued by two CTAs, are not
proxy-preserved-ordered by base causality alone. The ISA does not say which
"thread block" an async op belongs to.
- Strict reading: such pairs race.
- Recommendation: emit an advisory, not a hard error, until a GPU or spec ruling.

**QUOTE** (§8.6, *Proxies*):

> "A proxy fence is required to synchronize memory operations across different
> proxies."
>
> "Operations using methods of access distinct from the generic method include:
> … async-proxy and tensormap-proxy accesses by .async.bulk operations …"

**QUOTE** (§8.9.5, *Causality Order*):

> "A memory operation X precedes a memory operation Y in proxy-preserved base
> causality order if X precedes Y in base causality order, and:
> - X and Y are performed to the same address, using the generic proxy, or
> - X and Y are performed to the same address, using the same proxy, and by the
>   same thread block, or
> - X and Y are aliases and there is an alias proxy fence along the base
>   causality path from X to Y."

**QUOTE** (§9.7.15.4, *membar / fence*):

> "A uni-directional proxy ordering from the from-proxykind to the to-proxykind
> establishes ordering between a prior memory access performed via the
> from-proxykind and a subsequent memory access performed via the to-proxykind.
> A bi-directional proxy ordering between two proxykinds establishes two
> uni-directional proxy orderings …"
>
> "Bi-directional proxy fences do not directly synchronize with other fences in
> the sense that fences with .release or .acquire semantics do. Instead,
> bi-directional proxy fences take effect within a single thread and therefore
> do not have a .scope qualifier. Their cross-proxy synchronizing effect composes
> with other forms of synchronization according to the rules of the Memory
> Consistency Model."
>
> "Value .async of the .proxykind qualifier specifies that the memory ordering is
> established between the async proxy and the generic proxy. The memory ordering
> is limited only to operations performed on objects in the state space
> specified. If no state space is specified, then the memory ordering applies on
> all state spaces."

**QUOTE** (§9.7.10.28.2, *Async Proxy*):

> "Accessing the same memory location across multiple proxies needs a
> cross-proxy fence. For the async proxy, fence.proxy.async should be used to
> synchronize memory between generic proxy and the async proxy. The completion
> of a cp{.reduce}.async.bulk operation is followed by an implicit generic-async
> proxy fence. So the result of the asynchronous operation is made visible to the
> generic proxy as soon as its completion is observed."

**SECTION.** §5.1.7, §8.6, §8.9.5, §9.7.10.28.2, §9.7.15.4.

**Verdict.**
- Both legacy and the prototype correctly judge cross-proxy pairs **only** by the
  bridge, never by plain HB.
- Legacy RS aliases `shared::cta` and `shared::cluster` slots. The ISA allows
  `.shared::cluster ⊇ .shared::cta` coverage, but not the reverse.
- The prototype is correct but possibly over-strict for a `.shared::cluster`
  fence on `shared::cta` priors.
- Legacy G handling only `All|Global` is correct for global memory.

---

## R4. mbarrier arrive/wait as release/acquire; bulk-copy completion visibility

**ANSWER.**

**`mbarrier.arrive`:**
- `.sem ∈ {.release, .relaxed}`, default `.release`;
- `.scope ∈ {.cta, .cluster}`, default `.cta`;
- `.release` makes it a release operation on the mbarrier (a release pattern);
- `.relaxed` gives "no memory ordering … and visibility guarantees".

**`mbarrier.test_wait` / `try_wait`:**
- `.sem ∈ {.acquire, .relaxed}`, default `.acquire`;
- `.scope ∈ {.cta, .cluster}`, default `.cta`;
- it forms an acquire pattern **only when it returns True** with `.acquire`.

**Scope matters.** Synchronisation needs the arrive (first op of the release
pattern) and the wait (last op of the acquire pattern) to be morally strong. That
means mutual scope inclusion. So:
- a remote-CTA `arrive.release.cluster` observed by `try_wait.acquire.cta` does
  **not** synchronise, because the waiter's `.cta` scope excludes the arriver;
- a `.cta` arrive performed on a remote CTA's barrier through `shared::cluster`
  likewise does not synchronise with the remote waiter.

**`.relaxed` wait.** It synchronises **nothing** by itself. Followed in program
order by `fence.acquire` (or `fence.acq_rel`) of suitable scope, it is "a strong
read on M followed by an acquire memory fence". That is an acquire pattern on
the mbarrier. The ISA example uses
`try_wait.relaxed.cluster` + `fence.acquire.sync_restrict::shared::cluster.cluster`.
The §8.8 example also shows this for observing completion of an async op with a
strong *read* of M.

**`cp.async.bulk` completion.**
- The implicit complete-tx has **`.release` semantics at `.cluster` scope**.
- An acquire `try_wait` returning True guarantees the bulk ops "using the same
  mbarrier object" are "performed and made visible to the executing thread".
- Because completion carries an implicit generic-async proxy fence, the
  destination bytes are visible **in the generic proxy** to the waiting thread.
- Other threads get them only transitively through later synchronisation from
  the waiter, or through their own acquire wait.
- The complete-tx "does not transitively establish ordering with respect to
  prior instructions from the issuing thread". Only the op's own writes are
  published, not the issuer's history (confirms §9.11 / prototype).

**Async-op thread for scope.** The ISA is **silent** on which thread an async op's
complete-tx counts as "executing" for the mutual-scope test. CUTLASS multicast TMA
waits with the default `.cta` scope, and the try_wait bullet promises visibility
for bulk ops on that barrier. Recommendation: accept an acquire wait of any scope
for the op's own destination bytes, without issuer history. Keep the mutual-scope
test for *thread* arrives (`mbarrier.arrive.release`).

**QUOTE** (§9.7.15.16.16, *mbarrier.arrive*):

> "The optional .sem qualifier specifies a memory synchronizing effect … If the
> .sem qualifier is absent, .release is assumed by default. The .relaxed
> qualifier does not provide any memory ordering semantics and visibility
> guarantees. The optional .scope qualifier indicates the set of threads that
> directly observe the memory synchronizing effect of this operation … If the
> .scope qualifier is not specified then it defaults to .cta."

**QUOTE** (§9.7.15.16.19, *mbarrier.test_wait / mbarrier.try_wait*):

> "When mbarrier.test_wait and mbarrier.try_wait operations with .acquire
> qualifier returns True, they form the acquire pattern … If the .sem qualifier
> is absent, .acquire is assumed by default. The .relaxed qualifier does not
> provide any memory ordering semantics and visibility guarantees. The optional
> .scope qualifier indicates the set of threads that the mbarrier.test_wait and
> mbarrier.try_wait instructions can directly synchronize. If the .scope
> qualifier is not specified then it defaults to .cta."
>
> "The following ordering of memory operations hold for the executing thread when
> mbarrier.test_wait or mbarrier.try_wait having acquire semantics returns True :
> - All memory accesses (except async operations) requested prior, in program
>   order, to mbarrier.arrive having release semantics during the completed phase
>   by the participating threads of the CTA are performed and are visible to the
>   executing thread. …
> - All cp.async.bulk asynchronous operations using the same mbarrier object
>   requested prior, in program order, to mbarrier.arrive having release
>   semantics during the completed phase by the participating threads of the CTA
>   are performed and made visible to the executing thread."

**QUOTE** (§9.7.10.28.4.1, *cp.async.bulk*):

> "The complete-tx operation on the mbarrier has .release semantics at the
> .cluster scope as described in the Memory Consistency Model."

**QUOTE** (§8.9.1.1, *Asynchronous Operations*):

> "The implicit mbarrier complete-tx operation that is part of all variants of
> cp.async.bulk and cp.reduce.async.bulk instructions is ordered only with
> respect to the memory operations performed by the same asynchronous
> instruction, and in particular it does not transitively establish ordering with
> respect to prior instructions from the issuing thread."

**QUOTE** (§8.8, acquire pattern example):

> "cp.async.bulk.mbarrier::complete_tx::bytes.relaxed.sys.b128 [dst], [M], size,
> [barrier]; // strong read on M / mbarrier.try_wait.relaxed p, [barrier]; //
> observes completion of async op that performs strong read on M / @p
> fence.acquire; // acquire fence in program order after observing completion"

**SECTION.** §8.8, §8.9.1.1, §8.9.4, §9.7.10.28.2, §9.7.10.28.4.1,
§9.7.15.16.16, §9.7.15.16.19.

**Verdict.**
- Legacy §9.13, "mbarrier HB has no scope", is **wrong** for thread arrives.
  Close §10 b: the SyncTable must carry arrive/wait `.sem` and `.scope`.
- The edge applies only when both are release/acquire and mutually covering. A
  relaxed arrive or relaxed wait gives no edge, unless a later `fence.acquire`
  makes it an acquire pattern; the fence's scope then replaces the wait's.
- The prototype's per-op milestone publication with no issuer history is
  **correct**.

---

## R5. Named barriers and `barrier.cluster`

**ANSWER.**

**`bar{.cta}.sync` / `.red` / `.arrive`.** These are synchronizing operations.
- `sync`, `red` and `arrive` each **synchronize with** a `sync` or `red` on the
  same barrier. `arrive` is only a source, never a target: it gets no acquire.
- There is **no scope qualifier**. The `.cta` suffix "doesn't change the
  semantics". The effect is limited to the threads **participating** in that
  barrier instance (operand `b`, or the whole CTA).
- Memory ordering is full both ways. Prior accesses are "performed relative to
  all threads participating". `sync` and `red` also block new accesses until
  completion.
- The edge does not need moral strength, and it covers weak accesses. It does not
  cover async ops (cp.async*, bulk), which are not in program order (§8.9.1.1).

**`barrier.cluster.arrive{.release|.relaxed}` / `wait{.acquire}`.**
- `arrive` synchronizes with `wait`.
- The defaults are `.release` for arrive and `.acquire` for wait.
- The scope is the whole **cluster**: all non-exited threads of the cluster.
- After `wait`, accesses requested before every thread's `arrive` are complete
  and visible to the waiter, **except asynchronous operations**.
- With `.relaxed` arrive there are no guarantees for that thread's prior
  accesses. The ISA pairs it with an explicit `fence.cluster.acq_rel`, or
  `fence.mbarrier_init.release.cluster`, before the arrive.
- Accesses between the thread's own arrive and wait get no guarantee.

**QUOTE** (§8.9.4, *Memory synchronization*):

> "A bar{.cta}.sync or bar{.cta}.red or bar{.cta}.arrive operation synchronizes
> with a bar{.cta}.sync or bar{.cta}.red operation executed on the same barrier.
> A barrier.cluster.arrive operation synchronizes with a barrier.cluster.wait
> operation."

**QUOTE** (§9.7.15.1, *bar, barrier*):

> "The barrier{.cta}.sync or barrier{.cta}.red or barrier{.cta}.arrive
> instruction guarantees that when the barrier completes, prior memory accesses
> requested by this thread are performed relative to all threads participating in
> the barrier. The barrier{.cta}.sync and barrier{.cta}.red instruction further
> guarantees that no new memory access is requested by this thread before the
> barrier completes."
>
> "The optional .cta qualifier simply indicates CTA-level applicability of the
> barrier and it doesn't change the semantics of the instruction."

**QUOTE** (§9.7.15.3, *barrier.cluster*):

> "The barrier.cluster.wait instruction guarantees that when it completes the
> execution, memory accesses (except asynchronous operations) requested, in
> program order, prior to the preceding barrier.cluster.arrive by all threads in
> the cluster are complete and visible to the executing thread. There is no
> memory ordering and visibility guarantee for memory accesses requested by the
> executing thread, in program order, after barrier.cluster.arrive and prior to
> barrier.cluster.wait. The optional .relaxed qualifier on barrier.cluster.arrive
> specifies that there are no memory ordering and visibility guarantees provided
> for the memory accesses performed prior to barrier.cluster.arrive. … If the
> optional .sem qualifier is absent for barrier.cluster.arrive, .release is
> assumed by default. If the optional .acquire qualifier is absent for
> barrier.cluster.wait, .acquire is assumed by default."

**SECTION.** §8.9.1.1, §8.9.4, §9.7.15.1, §9.7.15.3.

**Verdict.**
- Named barriers: model the edge as source = `{sync, red, arrive}` →
  target = `{sync, red}` among participants. **Do not** give an `arrive`-only
  thread an acquire. For barriers, "no scope" in legacy is correct (§9.13 applies
  only to mbarrier).
- Cluster barrier: an *absent* qualifier means `.release`, not relaxed. Legacy
  §9.14 ("missing payload ⇒ relaxed only") is wrong if "missing" means
  "qualifier omitted". If it means "lowering lost the information", fail closed
  with `incomplete`, as named barriers already do.
- Neither barrier orders in-flight async ops.

---

## R6. `cp.async.bulk.wait_group.read` vs `wait_group`

**ANSWER.**

**`wait_group N`** waits until each completed group's ops have done all of the
following:
- (optionally) read the tensormap;
- read the sources;
- written the destinations;
- made those **writes visible to the executing thread**.

**`.read`** waits only for the tensormap and **source reads**. It gives *no*
destination guarantee. It only makes it safe for the executing thread to
overwrite or reuse the source.

Groups are **per thread** ("committed by the executing threads"). Visibility is
to the executing thread only. Other threads need further synchronisation from
it. Completion carries the implicit generic-async proxy fence (R3), so the
non-`.read` form makes destinations visible in the generic proxy.

**QUOTE** (§9.7.10.28.6.2, *cp.async.bulk.wait_group*):

> "By default, cp.async.bulk.wait_group instruction will cause the executing
> thread to wait until completion of all the bulk async operations in the
> specified bulk async-group. A bulk async operation includes the following:
> Optionally, reading from the tensormap. Reading from the source locations.
> Writing to their respective destination locations. Writes being made visible to
> the executing thread. The optional .read modifier indicates that the waiting has
> to be done until all the bulk async operations in the specified bulk
> async-group have completed: reading from the tensormap the reading from their
> source locations."

**QUOTE** (§9.7.10.28.1.1, *Async-group mechanism*):

> "A commit operation creates a per-thread async-group containing all prior
> asynchronous operations tracked by async-group completion and initiated by the
> executing thread …"

**SECTION.** §9.7.10.28.1.1, §9.7.10.28.2, §9.7.10.28.6.2.

**Verdict.** The prototype is **correct**:
- `.read` → `Milestone::Read` (source-side only);
- the full wait → the write milestone, published to the executing lane only.

Legacy §9.11 (full token clock including issuer history) is wrong for `.read`.

---

## R7. `bar.warp.sync` / `__syncwarp`: per-lane ordering

**ANSWER.**
- `bar.warp.sync membermask` provides memory ordering **only among the threads
  that participate**, meaning the lanes in `membermask` that executed it with the
  same mask.
- It is not a scope; the warp is explicitly not a scope.
- It does not appear in the §8.9.4 list of synchronizing relations. Its ordering
  is stated only in the instruction prose. The CUDA PG states it as
  strongly-happens-before from the call to the unblocking of every named lane.
- Lanes outside the mask get nothing. There is no per-warp ordering.
- Correct model: all-to-all synchronizes-with among the mask's lanes for that
  instance, covering weak and strong accesses alike.
- The ISA is silent on async ops. Since they are not in program order, fail
  closed: no ordering for in-flight async ops.

**QUOTE** (§9.7.15.2, *bar.warp.sync*):

> "bar.warp.sync also guarantee memory ordering among threads participating in
> barrier. Thus, threads within warp that wish to communicate via memory can store
> to memory, execute bar.warp.sync, and then safely read values stored by other
> threads in warp."
>
> "The behavior of bar.warp.sync is undefined if the executing thread is not in
> the membermask."

**QUOTE** (CUDA PG, *Warp Synchronization*):

> "Calling __syncwarp(mask) provides memory ordering among the participating
> threads within a warp named in mask: the call to __syncwarp(mask) strongly
> happens before … any warp thread named in mask is unblocked from the wait or
> exits."

**QUOTE** (§8.5): "Note that the warp is not a scope …"

**SECTION.** §8.5, §9.7.15.2; CUDA PG C++ Language Extensions.

**Verdict.** Per-lane, mask-restricted ordering matches the ISA. The prototype
`relay("partial_warp")` race, and legacy §9.10 being a bug (WC merges masked
acquisitions into the whole warp), are both consistent with this.
`warp_sync()` in the R1 companion test correctly yields `clean`.

---

## R8. `fence.sc` total order and composition

**ANSWER.**
- Fence-SC order is a runtime, acyclic partial order that relates **every pair of
  morally strong** `fence.sc`. Morally strong here means each fence's scope
  includes the other's thread, or the two are in the same thread.
- If X precedes Y in Fence-SC order, then X synchronizes with Y. That edge joins
  base causality order, and it composes transitively with release/acquire
  patterns, barriers and program order.
- Fence-SC order may not contradict causality order.
- The prose says "a total order per scope". The formal rule is the
  pairwise-morally-strong one. A `fence.sc.cta` and a `fence.sc.gpu` in the same
  CTA **are** related, but two `fence.sc.cta` in different CTAs are not.

**Is `fence.sc` also an acq_rel fence?** The ISA is **silent** in the formal text:
release and acquire patterns name "release or acquire-release memory fence". It
does call `fence.sc` a stronger fence and makes `membar` a synonym.
Recommendation: treat `fence.sc` as also `acq_rel` at its scope. It is strictly
stronger in every NVIDIA model, and treating it as weaker would flag every
MP-with-`membar` kernel.

**Runtime linearisation model.** Linearising all `fence.sc` by simulated execution
order is **valid**: any such order is a legal Fence-SC order, since the simulator
respects causality. Add an edge from each earlier fence to each later fence
**that is morally strong with it**; per-scope-instance keying is not enough.

Caveat: like read-from-based acquire, this certifies only the simulated order.
- SB-style programs are race-free under every order, so they are unaffected.
- A program that is race-free only under one SC order is still a real hazard.

Report such cases as schedule-dependent if detectable; otherwise accept them.

**QUOTE** (§8.9.3, *Fence-SC Order*):

> "The Fence-SC order is an acyclic partial order, determined at runtime, that
> relates every pair of morally strong fence.sc operations."

**QUOTE** (§8.9.4): "A fence.sc operation X synchronizes with a fence.sc
operation Y if X precedes Y in the Fence-SC order."

**QUOTE** (§8.10.2, *Fence-SC*):

> "Fence-SC order cannot contradict causality order. For a pair of morally strong
> fence.sc operations F1 and F2, if F1 precedes F2 in causality order, then F1
> must precede F2 in Fence-SC order."

**QUOTE** (§9.7.15.4): "Instances of fence.sc with sufficient scope always
synchronize by forming a total order per scope, determined at runtime. This total
order can be constrained further by other synchronization in the program." and
"On sm_70 and higher membar is a synonym for fence.sc".

**SECTION.** §8.9.3, §8.9.4, §8.10.2, §8.10.6 (SB litmus), §9.7.15.4.

**Verdict.** "Per scope instance" is an acceptable approximation only if the
instance is computed pairwise. Mixed-scope `fence.sc` pairs in one CTA must be
related. Use the moral-strength test, not equality of scope.

---

## R9. Data races precisely; the generic-load carve-out

**ANSWER.** A pair (A, B) is a data race iff **all** of these hold:
1. **Overlap.** The virtual-address ranges intersect, partially or completely
   (§8.2.1).
2. **Conflict.** At least one is a write. `atom` that writes and `red` count as
   writes.
3. **Not morally strong** (§8.7). That is, *any* of:
   - different threads, and one is weak;
   - a scope does not include the other thread;
   - different proxies;
   - not a complete overlap.
4. **Not related in causality order** (§8.9.5). This means proxy-preserved base
   causality, optionally prefixed by observation order. Cross-proxy pairs need
   the proxy fence on the path.

So "at least one non-atomic" is **not** the criterion. Two `atom`s can race
(R2), and `st.relaxed`/`ld.relaxed` cannot, when the other conditions hold.

**Weak load vs strong store.** A weak (`.weak` / plain) load is never morally
strong with another thread's store. An unordered weak load vs strong store is
**always** a race. Exempting it would be wrong.

**Strong generic load vs strong store.** This is the actual legacy G carve-out:
G does not exempt `ld.relaxed` / `ld.acquire` generic loads. When the two are
morally strong, PTX says they are **not** a data race. Single-copy atomicity
applies, and the load may return either value.

PTX therefore does *not* license the carve-out. It is a stricter checker policy
for schedule independence, and it produces false positives against the ISA
definition.

**QUOTE.** §8.7.1, reproduced in "Core definitions" above, plus:

> "Two memory operations are said to overlap when the range of virtual addresses
> accessed by the two operations intersect." (§8.2.1)
>
> "Conflicting morally strong operations are performed with single-copy
> atomicity." (§8.10.3)
>
> "weak operation: An ld or st instruction with a .weak qualifier." (§8.4)

**SECTION.** §8.2.1, §8.4, §8.7, §8.7.1, §8.7.2, §8.9.5, §8.10.3.

**Verdict.**
- Never exempt a *weak* load. Legacy and the prototype both comply, since the
  prototype requires `scope.is_some()`.
- For *strong* generic loads, follow PTX: exempt them when morally strong.
- Move G's schedule-independence concern to the separate
  `UndeclaredProtocolWordDiagnostic` advisory. This confirms the recommendation
  in §10 a.

---

## Summary table

| # | Question | ISA answer | Legacy | Prototype |
|---|---|---|---|---|
| R1 | RMW extends release (release-rmw handoff) | Yes via observation order, but sibling-lane coherence order is unconstrained, so the test pattern **races** | **Correct** (race) | **Wrong** (clean); fix to withhold sibling-lane heads |
| R2 | Morally strong; RMW/RMW; non-covering scopes | Strong + mutual scope + same proxy + complete overlap; no RMW exemption; non-covering atomics race | RS wrong (unconditional RMW); G too strict (requires atomics) | Correct |
| R3 | Generic vs async without fence | Race; `fence.proxy.async.<ss>` is bi-directional, thread-local, limited to the state space; completion gives implicit async→generic for the op's results | Both judge by bridge; RS aliasing partly OK | Correct; optionally let `.shared::cluster` cover `shared::cta` |
| R4 | mbarrier scopes; relaxed wait; bulk visibility | Arrive `.release.cta` default; wait `.acquire.cta` default; mutual scope needed; relaxed = nothing unless followed by `fence.acquire`; bulk dest visible to waiter in generic proxy, no issuer history | No scope (wrong, §9.13) | Milestone-only correct; add scope (§10 b) |
| R5 | `bar.sync`, `barrier.cluster` | bar: participants, `arrive` → `sync`/`red`, no scope; cluster: default release/acquire at cluster, excludes async ops | Missing payload ⇒ relaxed is wrong if it means "default" | Model `arrive` as source only |
| R6 | `wait_group.read` vs `wait_group` | `.read` = source reads only; full = dest writes visible to the executing thread | Wrong for `.read` (§9.11) | Correct |
| R7 | `bar.warp.sync` | Ordering only among mask participants; warp is not a scope | WC whole-warp merge wrong (§9.10) | Correct (per lane) |
| R8 | `fence.sc` | Runtime order over morally strong pairs; synchronizes-with; treat as acq_rel too (ISA silent) | n/a | Linearise by sim order, pairwise moral strength |
| R9 | Data race definition; load carve-out | Overlap ∧ conflict ∧ ¬morally strong ∧ ¬causality; weak loads never exempt; strong loads exempt | G strong-load carve-out stricter than PTX | Correct; keep the advisory separate |
