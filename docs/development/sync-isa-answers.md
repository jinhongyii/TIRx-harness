# Sync semantics: ISA answers to open questions Q1–Q7

These answers resolve the open questions in `sync-semantics.md` §8, and the
disagreements in §2.7, §3.5, §4.4, §5.4 and §6.5. They are checked against the
following sources:

- **PTX.** *Parallel Thread Execution ISA*, version **9.4**
  (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), fetched
  2026-10-07. Section numbers refer to that version.
- **CUDA PG.** *CUDA Programming Guide*
  (https://docs.nvidia.com/cuda/cuda-programming-guide/), Appendix "C++
  Language Extensions" and Special Topic "Asynchronous Barriers".

Quotes are verbatim. The only change is that superscripts flattened in the HTML
(for example "220 - 1") are written as 2^20 - 1. Where the ISA is silent, the
answer says so and recommends a fail-closed behaviour.

---

## Q1. `tcgen05.alloc` when the requested columns are not free

**ANSWER.** It **blocks**: `alloc` waits until the columns become available. It is not
an error and not UB. Both the engine and strict are wrong to fail immediately.

**QUOTE** (PTX §9.7.18.7.1, *Tensorcore 5th Generation Instructions:
tcgen05.alloc, tcgen05.dealloc, tcgen05.relinquish_alloc_permit*):

> "tcgen05.alloc is a blocking instruction which dynamically allocates the
> specified number of columns in the Tensor Memory and writes the address of the
> allocated Tensor Memory into shared memory at the location specified by address
> operand dst. The tcgen05.alloc blocks if the requested amount of Tensor Memory is
> not available and unblocks as soon as the requested amount of Tensor Memory
> becomes available for allocation."

> "An exclusive allocation operation blocks until there is no other live
> allocation. Until the sole live allocation is deallocated, no other CTA may
> allocate, whether exclusively or nonexclusively."

> "tcgen05.dealloc is a potentially blocking instruction … If .cta_group::2 is
> specified, … tcgen05.dealloc may block to collectively performs the
> deallocation with the other peer CTA's warp."

**Relinquish** (same section):

> "Instruction tcgen05.relinquish_alloc_permit specifies that the CTA of the
> executing thread is relinquishing the right to allocate Tensor Memory. So, it
> is illegal for a CTA to perform tcgen05.alloc after any of its constituent
> threads execute tcgen05.relinquish_alloc_permit."

**`cta_group` ownership and issue rules** (same section):

> "If .cta_group::1 is specified, ownership of the sole allocation is held by a
> single CTA. If .cta_group::2 is specified, ownership is held jointly by both
> peer CTAs."

> "When .cta_group::1 is specified, one warp from the CTA must perform the
> allocation and de-allocation. When .cta_group::2 is specified, one warp from
> each of the peer CTAs must collectively perform the allocation and
> de-allocation. … When .cta_group::2 is specified, the issuing warp must make
> sure that peer CTA is launched and its warps eventually participate in
> collective operations."

> "All tcgen05 instructions within a kernel must specify the same value for the
> .cta_group qualifier."

> "The behavior of the instruction is undefined if all the threads in the warp do
> not use the same values of nCols, or if any thread in the warp has exited."

PTX §9.7.18.5, *Issue Granularity*, Table 55, row `.alloc, .dealloc,
.relinquish_alloc_permit`, case `::2`:

> "Issue from two warps, one in each of the current CTA and its Peer CTA, in
> order to collectively perform the operation, i.e., the first warp to perform
> the operation could block until the the second warp in the Peer CTA also
> performs the operation".

PTX §9.7.18.1.2, *Tensor Memory Allocation*:

> "All of the Tensor Memory that was allocated in a kernel, must be explicitly
> deallocated before the kernel exits."

**SECTIONS.** PTX §9.7.18.7.1, §9.7.18.5 (Table 55), §9.7.18.5.1 (*CTA Pair*),
§9.7.18.1.2.

**Verdict on the models (§6.5):**

- **§6.5.1.** Return `Blocked` and let the scheduler retry. A deadlock is reported
  only when no live allocation can ever be freed, that is, when every holder is
  blocked or has exited. Exiting while holding an allocation is already an error
  under the exit rule above. This is the ISA-faithful behaviour. If the scheduler
  cannot prove progress, report a deadlock (fail closed). Do not report an
  immediate `AllocationUnavailable`.
- **§6.5.2.** The one-`cta_group` rule applies to **all** tcgen05 instructions in
  the kernel, not only to lifecycle operations. A commit or mma that uses the
  other group must be rejected, not silently ignored.
- **§6.2, peer warp.** The requirement that the peer use the same
  `warp_id_in_cta` (TG:757-759) has **no ISA basis**. The ISA asks only for "one
  warp from each of the peer CTAs". Drop the check or turn it into a lint.
- **`AllocAfterRelinquish`.** ISA-backed ("illegal"). Keep it.
- **`LiveAllocationsAtExit`.** ISA-backed. Keep it.
- **Partial-warp alloc and dealloc.** ISA-backed (`.sync.aligned`, plus UB if any
  thread in the warp has exited). Keep the engine's check.
- **Column rule.** Non-exclusive allocations need a power of two in [32, 512].
  `.exclusive` allocations need a multiple of 32 in [32, 512] on sm_100f/103/110,
  or **[32, 576] on sm_107f** (Table 58). The `min(cap, 512)` bound in §6.2 is
  correct only for the non-107 targets.
- **`DeallocationMismatch` (exact `(base, cols)`).** The ISA says only "The operand
  taddr must point to a previous Tensor Memory allocation". For `.exclusive` it
  adds "deallocated by a matching tcgen05.dealloc.exclusive with the same number
  of columns". The ISA is silent on partial deallocation of a non-exclusive
  allocation. Keep the exact match (fail closed).

---

## Q2. `mbarrier.init` on a live mbarrier without `mbarrier.inval`

**ANSWER.** It is **undefined behaviour** whenever the location holds a valid
mbarrier, **whether or not it is "in use"**. The object stays valid from `init` to
`inval`, so the inactive and active cases are both UB.

**QUOTE** (PTX §9.7.15.16.12, *mbarrier.init*):

> "The behavior of performing an mbarrier.init operation on a memory location
> containing a valid mbarrier object is undefined; invalidate the mbarrier object
> using mbarrier.inval first, before repurposing the memory location for any
> other purpose, including another mbarrier object."

PTX §9.7.15.16.4, *Lifecycle of the mbarrier object*:

> "An mbarrier object must be invalidated to repurpose its memory for any
> purpose, including repurposing it for another mbarrier object."

PTX §9.7.15.16.13, *mbarrier.inval*:

> "An mbarrier object must be invalidated before using its memory location for
> any other purpose. Performing any mbarrier operation except mbarrier.init on a
> memory location that does not contain a valid mbarrier object, results in
> undefined behaviour."

The ISA does not use the term "in use" for `init`. The rule turns on "valid",
not on activity.

**SECTIONS.** PTX §9.7.15.16.4, §9.7.15.16.12, §9.7.15.16.13.

**Verdict:**

- **§2.7.2.** Strict and the engine analysis path are correct. The engine NumSim
  path, which permits re-init of an inactive slot without `inval`, is wrong.
  - `ReinitWithoutInval` should be an error under **every** policy, not
    Strict-only.
  - `ReinitActive` and `ReinitBeforeConsumption` become special cases with better
    diagnostics. Report the more specific kind when it applies.
- **§2.6 #7 and #8.** Remove the [VERIFY] tags. The PTX basis is the quote above.

---

## Q3. Named barriers: count granularity, aligned vs. unaligned, `bar.arrive`

**ANSWER.**

- The count `b` must be a multiple of 32.
- Arrival is marked **per warp**. Each executing thread first waits for all
  **non-exited** threads of its warp. This applies to every form, including
  `arrive`.
- `.aligned` is a convergence promise, not barrier state. The ISA explicitly
  allows different warps to use different forms of the barrier with the same id
  and count. It does not forbid mixing aligned and unaligned forms.
- Mixing `.red` with `sync`/`arrive` on one active barrier is "unpredictable".

**QUOTE** (PTX §9.7.15.1, *bar, barrier*):

> "Operand b specifies the number of threads participating in the barrier. If no
> thread count is specified, all threads in the CTA participate in the barrier.
> When specifying a thread count, the value must be a multiple of the warp size.
> Note that a non-zero thread count is required for barrier{.cta}.arrive."

> "Depending on operand b, either specified number of threads (in multiple of
> warp size) or all threads in the CTA participate in barrier{.cta} instruction."

> "barrier{.cta} instruction causes executing thread to wait for all non-exited
> threads from its warp and marks warps' arrival at barrier. In addition to
> signaling its arrival at the barrier, the barrier{.cta}.red and
> barrier{.cta}.sync instructions causes executing thread to wait for non-exited
> threads of all other warps participating in the barrier to arrive.
> barrier{.cta}.arrive does not cause executing thread to wait for threads of
> other participating warps."

> "Instruction barrier{.cta} has optional .aligned modifier. When specified, it
> indicates that all threads in CTA will execute the same barrier{.cta}
> instruction. In conditionally executed code, an aligned barrier{.cta}
> instruction should only be used if it is known that all threads in CTA evaluate
> the condition identically, otherwise behavior is undefined."

> "Different warps may execute different forms of the barrier{.cta} instruction
> using the same barrier name and thread count. … Care must be taken to keep a
> warp from executing more barrier{.cta} instructions than intended
> (barrier{.cta}.arrive followed by any other barrier{.cta} instruction to the
> same barrier) prior to the reset of the barrier. barrier{.cta}.red should not
> be intermixed with barrier{.cta}.sync or barrier{.cta}.arrive using the same
> active barrier. Execution in this case is unpredictable."

> "bar{.cta}.sync is equivalent to barrier{.cta}.sync.aligned. bar{.cta}.arrive
> is equivalent to barrier{.cta}.arrive.aligned."

> "Note: For .target sm_6x or below, barrier{.cta} instruction without .aligned
> modifier is equivalent to .aligned variant and has the same restrictions as of
> .aligned variant. All threads in warp (except for those have exited) must
> execute barrier{.cta} instruction in convergence."

PTX §9.7.14.7, *exit*:

> "Barriers exclusively waiting on arrivals from exited threads are always
> released."

PTX §13 (release notes for an early ISA version):

> "Semantics of bar instruction were updated to indicate that executing thread
> waits for other non-exited threads from it's warp."

CUDA PG, C++ Language Extensions, `__syncthreads`:

> "__syncthreads*() wait until all non-exited threads in the thread block
> simultaneously reach the same __syncthreads*() intrinsic call in the program
> or exit."

> "The __syncthreads*() intrinsics are permitted in conditional code, but only if
> the condition evaluates uniformly across the entire thread block."

**SECTIONS.** PTX §9.7.15.1, §9.7.14.7; CUDA PG "Synchronization Functions" in the
C++ Language Extensions appendix.

**Answers to the sub-questions:**

- **"Executed on a per-warp basis".** The ISA's wording is "marks warps' arrival at
  barrier". The thread count is in threads but must be a multiple of the warp
  size, so in effect each warp contributes 32. A partial warp is not counted
  until all of its non-exited threads have executed the barrier.
- **`bar.arrive` and the full warp.** Yes. The "wait for all non-exited threads
  from its warp" sentence applies to every `barrier{.cta}` form, `arrive`
  included.
  - For `.aligned` (and every `bar.*`), the lanes must do this convergently, in
    one instruction instance. Otherwise it is UB.
  - For unaligned `barrier.*` on sm_70+, the ISA implies that lanes may arrive
    divergently. It does not say how lanes at *different* barrier instructions
    (different PCs) recombine. **The ISA is silent here.**
- **Mixing aligned and unaligned.** There is no explicit prohibition. The ISA
  explicitly allows different forms across warps with the same name and count.
  `.aligned` constrains only the threads that execute that instruction. Read
  literally, its text says "all threads in CTA", but it is phrased as a
  convergence promise.
- **Same count.** "using the same barrier name and thread count" is the ISA basis
  for `ContractMismatch`.

**Verdict:**

- **§3.5.2.** The engine's full-warp requirement on unaligned `barrier.arrive` and
  `barrier.red` is **not** stricter than the ISA in what it requires: all
  non-exited lanes must arrive. It is stricter only in requiring them in a single
  convergent instance. The "stricter than the ISA" note in §3.5.2 should be
  corrected. Strict's lack of any check is wrong.
- **§3.5.3, `AlignedSyncContractMismatch`.** No ISA basis for rejecting a mix of
  aligned and unaligned forms across warps. Downgrade it to a lint, or drop it.
- **§3.5.4, recombination (Q3).**
  - The ISA supports accumulating a warp's non-exited lanes across divergent
    paths for the same `(id, count)`.
  - It is silent on lanes that reach different static barrier instructions.
  - Fail closed: accumulate lanes per `(warp, id, generation)` only when all
    partitions use the same id and count. Report anything else as an analysis
    gap or unsupported pattern, not as success.
  - Exited lanes are removed from the expected set (§9.7.14.7).
- **§3.5.8.** `.red` mixed with `sync`/`arrive` on one active barrier is
  "unpredictable". Reject it (gap G5).
- **§3.4 `Duplicate`.** Backed by "Care must be taken to keep a warp from executing
  more barrier instructions than intended … prior to the reset of the barrier".
- **§3.4 `IncompleteAtExit`.** Exiting with a dangling arrive is not UB; the ISA
  says the opposite, that waiters on exited threads are released. This should not
  be an error. At most it is a lint.

---

## Q4. Cluster barrier: do exited threads count as arrived?

**ANSWER.** **Yes, effectively.** Completion requires only "all non-exited threads
in the cluster". Exited threads are dropped from the expected set, and a wait that
depends only on exited threads is released.

**QUOTE** (PTX §9.7.15.3, *barrier.cluster*):

> "barrier.cluster.wait instruction causes the executing thread to wait for all
> non-exited threads of the cluster to perform barrier.cluster.arrive."

> "In addition, barrier.cluster instructions cause the executing thread to wait
> for all non-exited threads from its warp."

> "When all non-exited threads in the cluster have executed
> barrier.cluster.arrive, the barrier completes and is automatically
> reinitialized. After using barrier.cluster.wait to detect completion of the
> barrier, a thread may immediately arrive at the barrier once again."

> "Each thread must arrive at the barrier only once before the barrier
> completes."

> "The optional .aligned qualifier indicates that all threads in the warp must
> execute the same barrier.cluster instruction. In conditionally executed code,
> an aligned barrier.cluster instruction should only be used if it is known that
> all threads in the warp evaluate the condition identically, otherwise behavior
> is undefined."

PTX §9.7.14.7, *exit*:

> "Barriers exclusively waiting on arrivals from exited threads are always
> released."

**SECTIONS.** PTX §9.7.15.3, §9.7.14.7.

**Verdict (§4.4.5):**

- **Both models are wrong.** The engine raises an error and strict yields
  incomplete. The reference must make membership exit-aware:
  - When a thread exits, remove it from `participants`.
  - Re-evaluate completion.
  - A generation completed by the exit is a normal completion.
  - The exit rule is per thread, so a partially exited warp shrinks that warp's
    lane set.
- **`IncompleteAtExit`.** Should not be an error.
- **`EarlyArrival`.** ISA-backed ("must arrive … only once before the barrier
  completes"). Remove the [VERIFY].
- **`WaitBeforeArrival`.** The ISA has no explicit rule. A wait before one's own
  arrive waits on a non-exited thread (itself) that cannot arrive, so it is a
  guaranteed hang. Keep it as an error (fail closed). Change the PTX-basis column
  to "derived: self-deadlock".
- **Rearrival without wait.** The ISA allows an immediate rearrive *after* a wait.
  It is silent on rearriving after completion with no wait in between. Strict's
  "unmodeled" flag (fail closed) is the safer choice of the two.
- **§4.2, unaligned partial arrive.**
  - The "wait for all non-exited threads from its warp" rule matches the engine's
    lane accumulation.
  - Strict's outright rejection is stricter than the ISA, but it fails closed.
  - Prefer the engine's accumulation together with a quiescence check.

---

## Q5. `elect.sync` plus a named barrier executed by the elected lane only

**ANSWER.** **Not legal.**

- With `.aligned`, which includes every `bar.sync`/`bar.arrive`, single-lane
  execution is UB.
- With unaligned `barrier.sync`/`barrier.arrive`, the elected lane waits for all
  other non-exited lanes of its warp to execute the barrier. If they never do, it
  hangs; the warp's arrival is never marked.

`elect.sync` does not change barrier participation.

**QUOTE** (PTX §9.7.15.1, *bar, barrier*):

> "barrier{.cta} instruction causes executing thread to wait for all non-exited
> threads from its warp and marks warps' arrival at barrier."

> "When specified, it indicates that all threads in CTA will execute the same
> barrier{.cta} instruction. In conditionally executed code, an aligned
> barrier{.cta} instruction should only be used if it is known that all threads
> in CTA evaluate the condition identically, otherwise behavior is undefined."

PTX §9.7.15.15, *elect.sync*:

> "elect.sync elects one predicated active leader thread from among a set of
> threads specified by membermask. … The predicate destination p is set to True
> for the leader thread, and False for all other threads."

`.aligned` on the other warp-collective sync instructions (for example
`barrier.cluster`, §9.7.15.3, and `tcgen05.alloc`, §9.7.18.7.1):

> "The mandatory .aligned qualifier indicates that all threads in the warp must
> execute the same instruction. In conditionally executed code, the instruction
> should only be used if it is known that all threads in the warp evaluate the
> condition identically, otherwise behavior is undefined."

`@p` predicated on the `elect.sync` result is, by construction, a condition that
does not evaluate identically across the warp.

**SECTIONS.** PTX §9.7.15.1, §9.7.15.15, §9.7.15.3, §9.7.18.7.1.

**Verdict (§3.5.1):**

- **The engine's waiver is wrong.** It accepts a hang or UB as success.
- **Strict's `ElectSyncParticipation` is closer but is not the ISA rule.** It
  compares against the *elect entry mask*. The ISA requires all **non-exited
  lanes of the warp**, independent of `elect.sync`. If the entry mask is
  narrower, because the warp was already diverged, strict can wrongly accept.
- **Recommendation.**
  - Delete the elect special case.
  - Apply the general rule: the participation mask must equal the warp's
    non-exited mask.
  - For `.aligned`, require this in one convergent instance; otherwise report
    `PartialWarp`.
  - For unaligned forms, accumulate lanes as in Q3. A warp that never completes
    its lane set is a hang. Report it as a deadlock, or as `PartialWarp` at
    quiescence.

---

## Q6. `tcgen05.alloc` size increase and multiple allocations per CTA

**ANSWER.** There **is** an ISA basis for `AllocationSizeIncrease`. The ISA says the
allocated column count "should not increase" between any two allocations, in
execution order, within a CTA. Multiple non-exclusive allocations may be live at
the same time. `.exclusive` claims the permit and must be the only live
allocation. After `relinquish_alloc_permit`, any further alloc by the CTA is
illegal.

**QUOTE** (PTX §9.7.18.7.1):

> "The unsigned 32-bit operand nCols specify the number of columns to be allocated
> or de-allocated. The unit of allocation and de-allocation is 32 columns and all
> of lanes per column. The number of columns allocated should not increase between
> any two allocations in the execution order within the CTA."

> "When .exclusive is not specified, the allocation is non-exclusive. The
> instruction uses the allocation permit, but does not claim ownership of it.
> Multiple non-exclusive allocations may exist simultaneously."

> "When .exclusive is specified, the allocation is exclusive. The instruction
> claims ownership of the allocation permit. No other allocation may exist at the
> same time as an exclusive allocation."

> "Memory must be deallocated with .exclusive if and only if it is allocated with
> .exclusive."

> "… it is illegal for a CTA to perform tcgen05.alloc after any of its constituent
> threads execute tcgen05.relinquish_alloc_permit."

**SECTION.** PTX §9.7.18.7.1.

**Verdict:**

- **Keep `AllocationSizeIncrease`.** Remove the [VERIFY] and cite the sentence
  above. The ISA says "should not" and does not state the consequence, so treat
  a violation as an error (fail closed).
  - The rule covers *any two allocations in execution order within the CTA*,
    whether or not the earlier one is still live. The engine's
    `last_allocation_columns` (sticky, never reset by dealloc) is therefore the
    correct reading.
- **Add `.exclusive` rules.** Model the exclusive and non-exclusive pairing for
  alloc and dealloc. An `.exclusive` alloc blocks while any other allocation is
  live (Q1).
- **Re-alloc while holding an allocation.** Legal for non-exclusive allocations,
  subject only to capacity (block, see Q1) and the non-increase rule.

---

## Q7. Async-group and `cp.async.mbarrier.arrive` completion: per thread or per warp?

**ANSWER.** **Per thread.**

- Async-groups are created per thread.
- `wait_group` waits only for groups committed by the executing thread.
- Visibility is granted only "to the executing thread".
- `cp.async.mbarrier.arrive` tracks the cp.asyncs of the executing thread only.
  A successful mbarrier wait then publishes them to every waiting thread that
  participates in that mbarrier.
- `cp.async.bulk.wait_group.read` waits only until the *source reads* (and the
  tensormap read) complete. It does **not** make the destination writes visible.

**QUOTE** (PTX §9.7.10.28.1.1, *Async-group mechanism*):

> "A commit operation creates a per-thread async-group containing all prior
> asynchronous operations tracked by async-group completion and initiated by the
> executing thread but none of the asynchronous operations following the commit
> operation."

> "When an async-group completes, all the asynchronous operations belonging to
> that group are complete and the executing thread that initiated the
> asynchronous operations can read the result of the asynchronous operations."

PTX §9.7.10.28.3.2, *cp.async.commit_group*:

> "cp.async.commit_group instruction creates a new cp.async-group per thread and
> batches all prior cp.async instructions initiated by the executing thread but
> not committed to any cp.async-group into the new cp.async-group."

PTX §9.7.10.28.3.3, *cp.async.wait_group / cp.async.wait_all*:

> "cp.async.wait_group instruction will cause executing thread to wait till only
> N or fewer of the most recent cp.async-groups are pending and all the prior
> cp.async-groups committed by the executing threads are complete."

> "Writes performed by cp.async operations are made visible to the executing
> thread only after: The completion of cp.async.wait_all or The completion of
> cp.async.wait_group on the cp.async-group in which the cp.async belongs to or
> mbarrier.test_wait and mbarrier.try_wait returns True on an mbarrier object
> which is tracking the completion of the cp.async operation."

> "cp.async.wait_group and cp.async.wait_all does not provide any ordering and
> visibility guarantees for any other memory operation apart from cp.async."

PTX §9.7.10.28.6.2, *cp.async.bulk.wait_group*:

> "By default, cp.async.bulk.wait_group instruction will cause the executing
> thread to wait until completion of all the bulk async operations in the
> specified bulk async-group. A bulk async operation includes the following:
> Optionally, reading from the tensormap. Reading from the source locations.
> Writing to their respective destination locations. Writes being made visible to
> the executing thread."

> "The optional .read modifier indicates that the waiting has to be done until
> all the bulk async operations in the specified bulk async-group have completed:
> reading from the tensormap the reading from their source locations."

PTX §9.7.15.16.18, *cp.async.mbarrier.arrive*:

> "Makes the mbarrier object track all prior cp.async operations initiated by
> the executing thread." … "Causes an arrive-on operation to be triggered by the
> system on the mbarrier object upon the completion of all prior cp.async
> operations initiated by the executing thread."

PTX §9.7.15.16.19, *mbarrier.test_wait / mbarrier.try_wait*:

> "All cp.async operations requested prior, in program order, to
> cp.async.mbarrier.arrive during the completed phase by the participating
> threads of the CTA are performed and made visible to the executing thread."

**SECTIONS.** PTX §9.7.10.28.1.1, §9.7.10.28.3.2, §9.7.10.28.3.3,
§9.7.10.28.6.1, §9.7.10.28.6.2, §9.7.15.16.18, §9.7.15.16.19.

**Verdict (§5.4, gap G6):**

- **Racecheck's merged warp clock is too permissive.** It lets lane A acquire lane
  B's copies through `wait_group`. Under the ISA, a `cp.async.wait_group` or
  `cp.async.bulk.wait_group` performed by lane L acquires only the copies issued
  by L.
  - Cross-lane use needs an extra sync after the wait, for example
    `__syncwarp`/`bar.warp.sync` or a barrier.
  - Use per-lane acquire.
- **`cp.async.mbarrier.arrive`.** Tracking is per issuing thread. Visibility flows
  through the mbarrier's acquire to every thread that waits successfully. This
  matches the existing mbarrier model.
  - The fixed verifier lets the arrive land at any time after issue. That is a
    sound superset of "upon the completion of all prior cp.async operations
    initiated by the executing thread".
  - The engine's FIFO tie to full completion is a valid refinement.
- **`.read` (§5.3 `InvalidForm`).**
  - `.read` exists only on `cp.async.bulk.wait_group`. This is correct.
  - `wait_group.read` must release **only the source** for reuse (WAR on the
    source). It must **not** acquire the destination writes.
  - A destination read after only `wait_group.read` is a race. Racecheck must
    model the two milestones separately.
- **`UncommittedAtExit`.** The ISA does not require a commit before exit. Keep it
  as engine policy and label it a lint, not PTX. The ISA does make it UB to read
  the destination or modify the source before completion (§9.7.10.28.1: "modifying
  the source memory location … or reading from the destination memory location
  before the asynchronous operation completes, exhibits undefined behavior").

---

## Additional limits and forms

### Expected-arrival, pending and tx-count ranges

PTX §9.7.15.16.3, *Contents of the mbarrier object*, Table 43:

| Layout | Expected arrival count | Pending arrival count | tx-count |
| --- | --- | --- | --- |
| `.layout::v0` | 1 … 2^20 - 1 | 0 … 2^20 - 1 | -(2^20 - 1) … 2^20 - 1 |
| `.layout::v1` | 1 … 2^9 - 1 (= 511) | 0 … 2^9 - 1 | -(2^20 - 1) … 2^20 - 1 |

PTX §9.7.15.16.12, *mbarrier.init*:

> "[1, …, 2^20 - 1] for mbarrier with .layout::v0 [1, …, 2^9 - 1] for mbarrier
> with .layout::v1"

PTX §9.7.15.16.18, *cp.async.mbarrier.arrive*:

> "The pending count of the mbarrier object after the increment should not exceed
> the limit as mentioned in Contents of the mbarrier object. Otherwise, the
> behavior is undefined."

PTX §9.7.15.16.16, *mbarrier.arrive*:

> "The value of the operand count must be in the range as specified in Contents of
> the mbarrier object."

PTX §9.7.15.16.17, *mbarrier.arrive_drop*:

> "If the decrement causes the expected arrivals count to be zero, the behavior is
> undefined."

**Verdict (§2.7.3):**

- **The engine is correct.** It checks the 2^20 - 1 tx limit, the pending limit,
  and the layout-v1 count of 511.
- **Strict's fixed 2^20 - 1 init range is wrong for `.layout::v1`.**
- **D3 is consistent with the ISA.** The tx-count range is a per-phase state
  range, so it bounds the phase total. The ISA states no explicit consequence
  for exceeding it, apart from the pending-count UB quoted above. Erroring is
  the fail-closed choice.
- **The tx-count is signed.** Its range is -(2^20 - 1) … 2^20 - 1, so a
  `complete_tx` may transiently precede its `expect_tx` within one phase. A
  negative intermediate tx-count is legal if it stays in range.
  - `TxOverDelivery` is still correct as a phase-completion check. The phase
    completes when pending arrivals = 0 and tx-count = 0, so over-delivery that
    leaves tx-count ≠ 0 at that point hangs or mis-completes.
  - A model that rejects any negative intermediate value is stricter than the
    ISA. Verify that the reference allows it.

### `expect_tx` / `arrive.expect_tx` txCount width

PTX §9.7.15.16.14, *mbarrier.expect_tx*:

> "The 32-bit unsigned integer operand txCount specifies the expectCount argument
> to the expect-tx operation."

PTX §9.7.15.16.16, *mbarrier.arrive*:

> "The 32-bit unsigned integer operand txCount specifies the expectCount argument
> to the expect-tx operation. When both qualifiers .arrive and .expect_tx are
> specified, then the count argument of the arrive-on operation is assumed to be
> 1."

So the operand is a **u32**, but the mbarrier's tx-count state is limited to
±(2^20 - 1). Validate the resulting state, not the operand width.
`arrive.expect_tx` performs the expect-tx *before* the arrive-on, which is why
the arrive cannot complete the phase with the new tx outstanding.

### `try_wait` vs. `test_wait`, and parity

PTX §9.7.15.16.19:

> "mbarrier.test_wait is a non-blocking instruction which tests for the
> completion of the phase. mbarrier.try_wait is a potentially blocking
> instruction which tests for the completion of the phase. If the phase is not
> complete, the executing thread may be suspended. Suspended thread resumes
> execution when the specified phase completes OR before the phase completes
> following a system-dependent time limit."

> "The .parity variant of the instructions test for the completion of the phase
> indicated by the integer parity of the operand phaseParity, which denotes
> either the current phase or the immediately preceding phase of the mbarrier
> object. An even phase has integer parity 0 and an odd phase has integer parity
> of 1."

> "The test_wait and try_wait operations are valid only for: the current
> incomplete phase, for which waitComplete returns False. the immediately
> preceding phase, for which waitComplete returns True."

**Consequences for the model:**

- **Parity and phase are the same for both instructions.** The only difference
  is that `try_wait` may suspend.
- **`try_wait` can return False even though the phase will complete**, because of
  the time limit. Kernels must loop, and the model may treat a False result as
  "retry".
- **A wait whose parity names a phase two or more behind is outside the valid
  range.** It aliases.
- **`InvalidPhase`.** The parity operand is a u32 whose *integer parity* is used,
  so values other than 0 and 1 are not malformed per se; only the low bit
  matters. If a frontend passes a raw u32, rejecting values outside {0, 1} is
  stricter than the ISA. Keep it only if the ABI guarantees 0 or 1.

**S1 (consumption) has an ISA basis.** PTX §9.7.15.16.5.1, *Primary phase*:

> "For each primary phase of the mbarrier object, at least one test_wait or
> try_wait operation must be performed which returns True for waitComplete
> before an arrive-on operation in the subsequent primary phase."

This settles §2.7.1 in favour of strict for **arrive-on**:

- `ArriveBeforeConsumption` is ISA-backed.
- Policy S1 should cite this sentence, and arguably should apply under every
  policy.
- The rule names only arrive-on operations, so `ExpectTxBeforeConsumption` (§2.6
  #10) remains a policy extension. The wait may be by *any* thread ("at least
  one").
- Reading the same rule strictly, a `cp.async.mbarrier.arrive` (non-`.noinc`)
  increment in a phase whose predecessor was never observed is also covered.
  It is an arrive-on in the subsequent phase (see D4 and F1).

### `.noComplete`

PTX §9.7.15.16.16, *mbarrier.arrive*:

> "A mbarrier.arrive operation with .noComplete qualifier must not cause the
> mbarrier to complete its current phase, otherwise the behavior is undefined."

> "Note: for sm_8x, when the argument count is specified, the modifier
> .noComplete is required."

PTX §9.7.15.16.17, *mbarrier.arrive_drop*:

> "A mbarrier.arrive_drop with .noComplete qualifier must not complete the
> mbarrier, otherwise the behavior is undefined."

Syntax: `mbarrier.arrive.noComplete{.release.cta}{.shared{::cta}}.b64 state,
[addr], count;`. The form takes only `.release.cta`, `.shared::cta` and a
mandatory count.

**Verdict (§2.7.4).** The engine's noComplete check is correct and ISA-backed.
Strict must add it, as a typed error, for example `NoCompleteWouldComplete`. It
should not be an untyped `EngineError`.

---

## Summary table

| Q | ISA answer | Which model is right |
| --- | --- | --- |
| Q1 | `alloc` blocks until columns are free; relinquish makes a later alloc illegal; `cta_group` must be uniform across **all** tcgen05 ops | Neither (both error). Use `Blocked` plus deadlock detection. The peer same-warp-id check has no basis. |
| Q2 | init on a valid (not invalidated) mbarrier is UB, active or not | Strict and engine-analysis. NumSim's permissive path is wrong. Make `ReinitWithoutInval` all-policy. |
| Q3 | `b` is a multiple of 32; arrival is per warp after all non-exited lanes; mixed forms are allowed with the same id and count; `.red` mixing is unpredictable | Engine's full-warp lane requirement (relax convergence for unaligned). Strict's aligned-mix rejection has no basis. Recombination across different PCs: silent → fail closed. |
| Q4 | Completion needs only non-exited threads; exited-only waits are released | Neither. Make membership exit-aware. |
| Q5 | Single-lane barrier: UB (`.aligned`) or hang (unaligned) | Neither exactly. Drop the elect waiver; require the non-exited warp mask. |
| Q6 | "should not increase between any two allocations … within the CTA"; multiple non-exclusive allocations allowed | Keep `AllocationSizeIncrease` (sticky). |
| Q7 | Per-thread groups and visibility; `.read` = source read only | Racecheck must acquire per lane; `.read` must not publish the destination. |
| Limits | v0 1..2^20-1, v1 1..511; tx ±(2^20-1); txCount operand u32 | Engine (layout-aware). Strict's fixed range is wrong for v1. |
| S1 | At least one successful wait per phase before the next arrive-on | Strict (for arrive-on). |
| noComplete | Completing the phase is UB | Engine. Add it to strict. |
