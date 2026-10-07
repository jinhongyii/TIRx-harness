---
orphan: true
---

# NumSim synchronization semantics (SyncTable spec)

Status: specification for redesign worker W3 (`numsim-redesign.md` §2.1 SyncTable, §2.6, §4.1).

Executable form: `tirx_harness/src/tirx_harness/numsim/core-rs/numsim-sync-ref/`, run with `cargo test --release` in that directory.

ISA resolutions: `sync-isa-answers.md` (PTX 9.4) answers the open questions Q1–Q7 of §8. The reference crate and this document follow those answers. Every resulting change against the legacy engine and strict models is listed, with its ISA cite, in `sync-behaviour-deltas.md`. Sections 2–7 still describe the legacy models faithfully. The **Reference** column or paragraph in each section gives the resolved semantics.

This document records how the legacy engine implements each synchronization protocol today, and how the legacy strict synccheck models re-implement it. It then states the semantics the new `SyncTable::step` functions must have. Where the engine and strict models disagree, both readings are kept and a resolution is proposed. The disagreements are the "second reading of the ISA" that the redesign must not lose.

## 0. Conventions

### 0.1 Citations

Source paths are relative to `tirx_harness/src/tirx_harness/numsim/engine-rs/src/` unless noted otherwise.

| Abbreviation | File |
| --- | --- |
| HB | `hardware_barriers.rs` (PhysicalBarrierHub, NamedBarrierHub) |
| SM | `native_analysis/synccheck/strict_mbarrier.rs` |
| SNB | `strict_named_barrier.rs` |
| CB | `cluster_barriers.rs` |
| SCB | `strict_cluster_barrier.rs` |
| AG | `async_groups.rs` |
| DP | `deferred_payload.rs` |
| TG | `tcgen.rs` |
| SR | `setmaxnreg.rs` |
| SV | `native_analysis/synccheck/setmaxnreg_verifier.rs` |
| RS | `runtime/sync.rs` |
| IS | `runtime/instructions/sync.rs` |
| AC | `runtime/instructions/async_copy.rs` |
| KE | `kernel_engine.rs` |
| SC | `native_analysis/synccheck/sync_check.rs` |
| SFU | `native_analysis/synccheck/sync_fixed_unified.rs` |
| FE | `tirx_harness/frontend-rs/src/emit/` |

### 0.2 PTX claims

PTX ISA statements are paraphrased. Where `sync-isa-answers.md` settles a point, the cite is "ISA §x" (PTX 9.4 numbering). **[VERIFY]** marks wording that is still unchecked. Per the numsim guide, such claims need the ISA, a canonical implementation, or a GPU microtest before they justify an `error` verdict.

### 0.3 Policies

The ISA answers showed that almost every legacy strict-only rule is ISA-backed. Those rules now hold in **every** mode, including NumSim. Examples:

- mbarrier consumption before the next arrive-on (ISA §9.7.15.16.5.1).
- Re-init only after `inval` (ISA §9.7.15.16.12).
- Full non-exited warp participation for named and cluster barriers (ISA §9.7.15.1, §9.7.15.3).

Only the mbarrier module keeps a `Policy`:

- **`Numeric`**: what NumSim and the online checkers execute.
- **`Strict`**: Numeric plus the single policy extension left, `ExpectTxBeforeConsumption`. ISA §9.7.15.16.5.1 names arrive-on operations only; Strict extends the consumption rule to `expect_tx`.

Strict refines Numeric. For any command sequence, while Strict accepts every command, Numeric returns identical outcomes and reaches an identical state. This is property-tested.

In the redesign, NumSim and the online checkers run the `Numeric` step. The offline synccheck explorer (§2.6) uses the `Strict` rules as its protocol-state check. This replaces the deliberate counter duplication described at SM:610-642, where the strict tracker re-derives the hub counters so that `after_effect` can cross-check them. That positive control now lives in the differential test against the reference crate, which is the plan's "one independent reference state machine".

## 1. Proposed contract (`numsim-core::sync`)

The shapes below are the ones implemented by `numsim-sync-ref`. The production types may add `SiteId` or `DynamicOpId` witnesses to each `Cmd`. Witnesses are carried, never interpreted, so the differential test strips them and compares the rest mechanically.

```rust
pub enum Policy { Numeric, Strict }

/// One per protocol; the SyncTable maps ResourceId -> Resource(enum of States).
pub trait Protocol {
    type State: Clone + Eq;
    type Cmd;
    type Outcome: Eq;
    type Error: Eq;
    /// Transactional: Err leaves `state` unchanged. Deterministic. Never panics.
    fn step(state: &mut Self::State, cmd: Self::Cmd) -> Result<Self::Outcome, Self::Error>;
}
// Exit check per resource, run after the completion queue drains:
fn quiescent(&State) -> Result<(), Error>;      // mbarrier (via SyncTable), tcgen, setmaxnreg, async_group
fn exit_lint(&State) -> Option<Lint>;           // named (dangling arrive), async_group (uncommitted)
```

### 1.1 Resources and commands

Each `step` takes one resource state and one already-resolved command:

- Lane aggregation, address resolution and multicast target expansion happen before `step`. For example, an arrive whose lanes name several barriers becomes one `Cmd` per barrier.
- A warp-collective rendezvous is resolved before `step`, and so is lane-operand uniformity. One example is the four warps of a `setmaxnreg`. Another is the two CTAs of `tcgen05.alloc.cta_group::2`.
- A multi-target instruction applies `step` to each target on cloned states and commits all of them, or none.

| Protocol | ResourceId | State | Cmd |
| --- | --- | --- | --- |
| mbarrier | `(alloc, offset, cta)` | `mbarrier::State` | `Init{count,layout_v1}`, `Inval`, `Arrive{count,tx,drop,no_complete}`, `ExpectTx{bytes}`, `IncPending{count}`, `Issue`, `CompleteTx{gen,bytes}`, `DeferredArrive{gen,count}`, `TestParity{parity}`, `WaitParity{parity}`, `TestState{gen}` |
| named barrier | `(cta, id 0..15)` | `named::State` | `Arrive(Contribution)`, `Sync(Contribution)`, `Red(Contribution)`, `Resume{gen}` |
| cluster barrier | `cluster` | `cluster::State` | `Arrive{warp,mask,aligned}`, `Wait{warp,mask,aligned}`, `Exit{warp,lanes}` |
| async group | `(warp, lane, domain)` | `async_group::State` | `Issue`, `Commit`, `ArriveOn`, `Complete{ordinal,milestone}`, `Wait{n,read}`, `Exit` |
| tcgen kernel | `ResourceId::TcgenKernel` | `tcgen::KernelState` | `SyncCmd::TcgenGroup(g)` (`use_cta_group`). `SyncTable::step`/`step_all` add it implicitly before every lifecycle command, committed atomically with it. Handlers add it explicitly in the same `step_all` batch for mma/cp/shift/commit. |
| tcgen lifecycle | `cta pair` | `tcgen::State` (`exclusive_max`: 512 or 576) | `Alloc{who,columns,exclusive}` → `Allocated{base}` or `Blocked`, `Dealloc{who,taddr,columns,exclusive}`, `Relinquish{who}` |
| tcgen work | `(warp, lane)` | `tcgen::WorkState` | `Issue`, `Load`, `Store`, `Commit`, `WaitLd`, `WaitSt` |
| setmaxnreg | `cta` | `setmaxnreg::State` | `Configure{count}`, `Set{wg,inc,count}`, `WarpgroupSync{wg}`, `Grant{wg}`, `Poll{wg}` |

`Contribution` is `{warp, mask, live, count}`, where `live` is the warp's non-exited lane mask.

Every tcgen05 instruction (lifecycle, mma, cp, shift, commit) first steps the kernel-wide `tcgen::KernelState` with `use_cta_group(g)`.

### 1.2 Outcomes

Outcomes are small enums. Their key variants:

- **Completion facts:**
  - `Arrived{gen, pending_before, completed}`
  - `Landed{completed: Option<gen>}`
  - `Completed{arrivals}`
- **Asynchronous issue:** `Issued{gen}`
- **Wait results:**
  - `Ready{gen}`
  - `NotReady`, returned by non-blocking queries
  - `Blocked`, returned by blocking waits

The complete variant lists are in the crate.

### 1.3 Completion queue

The SyncTable owns a `VecDeque<Completion>`, where `Completion` is one of:

- `MbarTx{res, gen, bytes}`
- `MbarArrive{res, gen, count}`
- `GroupMilestone{res, ordinal, milestone}`
- `SetmaxGrant{res, wg}`

Each completion carries the generation captured by `Issue`. The scheduler pops enabled completions and applies them with `step`:

- **Enabled mbarrier completion:** the action's generation equals `gen`, or the phase is complete and the action's generation equals `gen + 1` (HB:2643-2653).
- **Enabled async-group milestone:** one of the FIFO rules in §5.
- **Enabled grant:** `required <= available`.

### 1.4 Blocking without wakers

A blocking command returns `Blocked`. The scheduler retries the warp on a later round. No waiter is registered anywhere.

The legacy engine used Rust Futures with wakers. The table below records what woke each wait, so the retry scheduler polls at least as often. In the retry model, "wake" means: the next retry after this event returns `Ready`.

| Wait | Legacy wake source | Retry becomes Ready after |
| --- | --- | --- |
| mbarrier parity wait (`WaitUntilParity`) | `complete_physical_if_ready` removes waiters whose `requested_phase == completed_parity` and wakes them (HB:2663-2691). The future re-checks parity on poll and re-registers if the parity flipped again (HB:2412-2462). Pre-registered waiters also consume the phase in strict (SM:1669-1689). | Any arrive, `CompleteTx` or `DeferredArrive` that completes the awaited phase |
| mbarrier `try_wait` / `test_wait` | Never blocks. `try_wait` is executed as the zero-suspension case of `test_wait` (IS:1072-1112). Only the frontend `WaitUntilParity` blocks (IS:205-211, 1239-1270). | — |
| named `bar.sync` | Push-based: the contribution that completes the generation wakes waiters after `publish_before_wake` (HB:2885-2899, 2928-2937). A late poll still completes via `completed_through` (HB:3213-3259). `pump()` is a no-op (HB:3105). | The arrive or sync that completes the generation |
| cluster wait | The arrive that completes the generation, from any CTA of the cluster, under the per-cluster mutex (CB:215-235). Clusters are pinned to one worker FIFO (`executor.rs:298-303`). | That arrive |
| async-group wait | After each applied milestone, the hub re-checks waiters of that warp only (AG:1547-1571, 1745-1757). | The milestone that satisfies the last lane |
| tcgen `cta_group::2` lifecycle | `CollectiveHub` publish once both warps contribute; the pump publishes (TG:617-619). | Collective layer, outside `step` |
| `tcgen05.alloc` with no free columns | Never blocked: `AllocationUnavailable` (TG:952-966) | A `Dealloc` in either CTA of the pair that frees enough columns (or the last live allocation, for `.exclusive`) |
| cluster wait on exited members | Never released: a deadlock with exit evidence (`executor.rs:822-878`) | The `Exit` that leaves every remaining member arrived |
| setmaxnreg `inc` | `SetmaxnregHub::pump` grants the lowest action id among enabled pendings, one per pump (SR:795-826, 1410-1420; `mode_completions.rs:311-349`). | `Grant` |

**Deadlock** is declared when no warp is runnable and a full completion-pump round makes no progress (`executor.rs:516-527`, 1464-1480; pump cadence `executor.rs:1301-1303`). In the redesign, a round with every live warp `Blocked` and no enabled completion is a deadlock. Per the numsim guide, a single spinning warp is not proof.

**Eager numeric completion.** In the legacy NumSim (non-observing) mode, TMA transaction bytes and `tcgen05.commit` arrivals land at issue: `complete_numeric` → `complete_transactions_immediately` / `arrive_many` (RS:1045-1060, 1100-1111; KE:4732, 5410-5418). The redesign keeps one path for every mode:

- **Issue** is `step(Issue)` and returns the bound generation.
- **Landing** is a queued `Completion`.

NumSim may pop the completion immediately, which is equivalent to the eager path. Racecheck and synccheck may pop it later.

## 2. mbarrier

### 2.1 PTX forms implemented (ABI)

All forms are declared at IS:14-75 and their variant markers at IS:77-209. They map onto the reference commands as follows.

- **`mbarrier.init[.layout::v1]`** → `Init`.
  - Variant `MbarrierInit<V1>` (IS:559-581).
  - Analysis modes require the count to be warp-uniform (KE:3711-3719).
  - NumSim requires the pointer to be warp-uniform or one-to-one, and equal counts for equal targets (RS:1777-1808).
- **`mbarrier.inval`** → `Inval` (IS:505-519).
- **`mbarrier.arrive[_drop][.release|.relaxed][.shared::cta|::cluster]`** → `Arrive`.
  - Operand variants: with or without count, predicated, local or remote: `ArriveLocal*`, `ArriveRemote*` (IS:133-151, 583-753).
  - The `.noComplete` state-returning form is `ArriveLocalCountState<NO_COMPLETE,DROP,RELEASE>`.
  - `.noComplete` requires release semantics (KE:3777-3781).
  - Remote forms resolve per lane. A local form naming a remote CTA is rejected with `MbarrierLocalArriveRemoteAddress` (RS:1586-1601).
- **`mbarrier.arrive.expect_tx`** → `Arrive{tx: Some}`. Variants `ArriveExpectTx{Local,LocalState,LocalPredicated,Remote,RemotePredicated}` (IS:153-162, 755-850).
- **`mbarrier.expect_tx`** → `ExpectTx` (IS:171-177, 852-906). It consumes no arrival.
- **`mbarrier.complete_tx`** → `Issue` + `CompleteTx`, with local, remote, predicated and multicast forms (IS:179-186, 908-1004).
  - Negative counts are rejected (IS:908-923).
- **`mbarrier.test_wait` / `try_wait`**, each in `.parity` and state-token forms, each also `.relaxed`, and each with a `CONDITIONAL` variant for layout v1 copy-report → `TestParity` / `TestState` (IS:188-201, 1006-1237).
  - Report variants return `(ready, report, value)` from one physical snapshot (IS:1150-1237).
  - The suspend-time hint is not an operand (IS:1080-1083).
- **`WaitUntilParity`** is the engine's blocking wait. It is not a PTX mnemonic (IS:205-211, 1239-1270) → `WaitParity`.
- **`mbarrier.pending_count`** decodes the private state token (HB:530-549, IS:1051-1068).
- **`mbarrier.check_layout`** (IS:521-547).
- **`fence.mbarrier_init`** (IS:1536-1550; coverage tracker HB:676-764).
- **Asynchronous producers bound to a barrier** (`Issue` followed by a landing):
  - TMA and bulk copies, `st.async`, `red.async` (§5 and the AC forms).
  - `cp.async.mbarrier.arrive[.noinc]` → `IncPending` (unless `.noinc`) + `Issue` + `DeferredArrive{count: 1}` (RS:918-988).
  - `tcgen05.commit[.multicast]` → `Issue` + `DeferredArrive{count: 1}` per target CTA (RS:1719-1771).

### 2.2 State

| Field | Engine (`PhysicalBarrierEntry`, HB:333-375) | Strict (`StrictSlot`, SM:668-696) | Reference (`mbarrier::State`) |
| --- | --- | --- | --- |
| liveness | entry present | slot present | `live` |
| layout | `layout_v1` | — (not modeled) | `layout_v1` |
| steady expected | `expected_arrivals` | `expected_arrivals` | `expected` |
| generation | `generation` | `generation` | `gen` |
| phase complete | `phase.complete` (lazy roll) | `lifecycle ∈ {Pending, CompletedUnconsumed, Consumed}` | `complete`, `consumed` |
| last completed | `last_completed_generation`, `completed_parity` | `last_completed_generation`, `last_completed_phase` | `last_completed` (parity = `gen & 1`, or 1 before any completion) |
| arrivals | `phase.arrival_count`, `arrived_warps` | same | `arrived` (warps are diagnostic only) |
| phase-only extra | `phase.pending_arrival_increments` | same | `extra` |
| tx | `phase.expected_transactions`, `completed_transactions` | same | `tx_expected`, `tx_completed` |
| future bytes | `buffered_transactions: gen→bytes` | same | `buffered_next` (only `gen+1` is ever enabled) |
| in-flight completions | queued actions in `transaction_completions` | `outstanding_tokens` | `outstanding: gen→count` |
| waiters | `waiters: warp→{phase, waker}` | `waiters: (gen,warp)→…` | `armed: bool` (a blocked wait exists) |
| conditional/report | `conditional_completed_*`, `phase.report`, `previous_report` | — | not modeled (§2.8 G1) |

`required = expected + extra` (HB:353-356, SM:1591-1594).

After `Init`:

- `gen = 0`, phase pending.
- The completed parity is **1**, so a wait on parity 1 is vacuously ready. Its generation is `None` and it does not consume generation 0 (HB:72-77, 956-957; SM:1596-1613; SM test 1820).

### 2.3 Transitions

**Lazy roll-over (`begin`).**

- Rule: if the phase is complete, then `gen += 1`, all counters reset, and `tx_completed = buffered[gen]` (HB:2628-2641, SM:1615-1644).
- It runs at the start of: `Arrive` (HB:2124), `ExpectTx` (HB:2222), `IncPending` (HB:2192), the drop half of an arrive (HB:1202), and a `DeferredArrive` landing (HB:2523).
- The reference requires `consumed` before every arrive-on (`Arrive`, `IncPending`, `DeferredArrive`) under both policies. Strict also requires it before `ExpectTx` (§2.5 S1).

**`Init{count, layout_v1}`** (HB:889-966; SM:846-871, 1517-1582).

- Precondition: `1 <= count <= limit(layout_v1)`, where the limit is 2^20-1, or 511 for v1 (HB:16-25, 900; RS:371-385).
- If the slot is live, the checks run in this order:
  1. The phase is `CompletedUnconsumed` → `ReinitBeforeConsumption` (SM:1543-1550). Strict-only in legacy.
  2. Waiters exist → engine `BarrierReinitializedWhileWaiting` (HB:916-921).
  3. Queued completions, an active phase, or buffered bytes → `ReinitActive` (HB:922-939; SM:1551-1574).
     - "Active" means: not complete, and any of arrivals, extra, `tx_expected`, `tx_completed` is nonzero (HB:2655-2661).
     - Strict's `active_pending` omits `extra` (SM:1556-1559). This is harmless, because step 4 rejects every live slot in strict anyway.
  4. Otherwise:
     - Engine analysis modes return `BarrierReinitializedWithoutInvalidation` (HB:940-944).
     - Engine NumSim mode permits the re-init (`init_numeric_layout`, HB:874-887; RS:1806).
     - Strict always rejects it (SM:1575-1581).
     - **Reference:** both policies reject it. ISA §9.7.15.16.12 says "performing an mbarrier.init operation on a memory location containing a valid mbarrier object is undefined", whether or not the object is in use. The reference reports the most specific kind: `ReinitBeforeConsumption`, then `ReinitActive`, then `ReinitWithoutInval`.
- Duplicate same-address lanes collapse. Distinct lane targets are validated transactionally (HB:849-856, SM:843-871).

**`Inval`** (HB:807-835; SM:798-831).

- Precondition: no waiters and no in-flight completion. A partially arrived phase may be abandoned (HB:817-818).
- Effect: the slot becomes uninitialized, and the init-fence record is dropped (KE:3645-3648).
- Every non-`Init` command on an uninitialized slot → `Uninitialized`.

**`Arrive{count, tx, drop, no_complete}`** (HB:1079-1152, 1157-1216, 2109-2175; SM:1015-1113; plan RS:1496-1562, 498-536).

- `count == 0` is a fully masked no-op. It returns the current generation and does not roll (HB:2117-2123, SM:1025-1027).
- **`no_complete`** (engine only):
  - Evaluated before roll and drop: `pending = complete ? expected : required - arrived`.
  - Error unless `count < pending` (HB:1157-1184). This holds even when outstanding tx would prevent completion.
  - Strict has no check.
- Then `begin`.
- **`drop`:**
  - `expected -= count` (`DropUnderflow` if it would go negative).
  - `extra += count`, so the current phase still needs `count`.
  - Future phases need fewer (HB:1189-1216, SM:1053-1065).
  - Engine: the drop is committed before the arrival in a separate call (RS:510-517, 595-597), so a failing arrival leaves the drop applied. This is non-transactional. The reference is transactional.
- **Arrival:**
  - `arrived + count <= required`, else `ArrivalOverflow` (HB:2128-2145, SM:1066-1084).
  - `tx_expected += tx`.
  - If `arrived == required` and `tx_completed > tx_expected` → `TxOverDelivery` (HB:2159-2168, SM:1095-1105).
- Completion check: `maybe_complete`.
- Outcome:
  - `gen` doubles as the state token.
  - `pending_before = required - arrived` before this arrival. `.noComplete` encodes it in the token (HB:500-527).
  - `completed`.

**`maybe_complete`** (HB:2663-2691, SM:1656-1695).

- Rule: if not complete, `arrived == required` and `tx_completed == tx_expected`, then:
  - `complete = true`,
  - `last_completed = gen`.
- Engine: waiters whose parity matches are woken.
- Strict and reference: if a waiter was pre-registered (`armed`), the phase is consumed immediately.

**`ExpectTx{bytes}`** (HB:1001-1034, 2208-2244; SM:945-989).

- `begin` (roll), then `tx_expected += bytes`. It never completes a phase.

**`IncPending{count}`** (HB:1043-1071, 2177-2206; SM:1115-1160).

- `begin`, then `extra += count`.
- Engine: `required + count <= limit`, else `BarrierArrivalOverflow`.

**`Issue`** captures a completion (strict `capture_completion` SM:1162-1215; engine `enqueue_*` HB:1375-1582).

- The bound generation is `gen + complete`, i.e. the next phase if the current one has completed (HB:1445-1453, 1543-1549; SM:1180-1193).

**`CompleteTx{gen, bytes}`** (engine apply HB:2477-2626; strict SM:1217-1335).

- The token must be outstanding (`UnknownToken`).
- `bytes == 0`: only the token is consumed. The engine never queues a zero-byte action (HB:1435-1437). Strict captures only nonzero transfers (SC:4197-4204).
- If the bound generation equals the current generation:
  - Current phase complete → `CompletionAfterComplete`.
  - Otherwise: `tx_completed += bytes`, the over-delivery check, then `maybe_complete`.
- If the bound generation is `gen + 1` and the current phase is complete: the bytes are buffered.
- A bound generation below the current one → `StaleCompletion`.
- Anything else → `FutureNotBufferable`.

**`DeferredArrive{gen, count}`** (engine HB:2515-2565; strict SC:2983-2990, i.e. `complete_tx(token, 0)` then `arrive`).

- `count == 0` → `InvalidCount` (HB:1531-1536).
- The bound generation is validated as for `CompleteTx`.
- Then `begin` and the arrival rules, without tx.

**`TestParity` / `WaitParity{parity}`.**

- `parity > 1` → `InvalidPhase` (HB:1601-1606, 1681-1690; SM:1347-1353).
- Ready iff `parity == completed_parity`. It returns `last_completed`: `None` means vacuous, otherwise the generation that was acquired (HB:1702-1715, 2423-2428).
- Strict and reference: a ready query whose generation is the current completed one consumes it (SM:1378-1388, 1646-1654).
- Not ready:
  - `TestParity` → `NotReady`.
  - `WaitParity` → `Blocked`, and sets `armed`.
- **Parity aliasing is real.** If two phases complete between a block and its retry, the parity matches the old value again and the warp blocks once more. The legacy Future does the same thing by re-registering (HB:2449-2459).

**`TestState{gen}`** (HB:1716-1729; RS:1687-1717).

- The token must name the current generation or the immediately preceding one, else `InvalidStateToken`.
- Ready iff the token names the preceding generation, or the current phase is complete.
- Every lane of one warp query must use the same barrier and token (RS:1697-1702).

### 2.4 tx-count and phase arithmetic: edge cases

| Case | Engine | Strict | Reference |
| --- | --- | --- | --- |
| `expect_tx` before `arrive` (same phase) | Accumulates on the pending phase (HB:2222-2242) | Same (SM:972-987) | Same |
| `expect_tx` or `arrive.expect_tx` after the phase completed, before any wait | Rolls to `gen+1` and applies there | `ExpectTxBeforeConsumption` / `ArriveBeforeConsumption` (SM:963-970, 1038-1045) | `arrive.expect_tx`: `ReuseBeforeConsumption{Arrive}`, both policies (ISA §9.7.15.16.5.1). Standalone `expect_tx`: Numeric rolls, Strict `ReuseBeforeConsumption{ExpectTx}`. |
| Bytes land before `expect_tx` (tx-count transiently negative) | Allowed while arrivals are incomplete (HB:2614-2624) | Same (SM:1277-1288) | Same (test `mbarrier_early_bytes_then_expectation`) |
| Bytes exceed the expectation once arrivals are complete | `TransactionOverflow` | `TransactionOverDelivery` | `TxOverDelivery`. Checked when arrivals become complete and on each landing afterwards. |
| Under-delivery (arrivals complete, bytes missing) | Phase stays pending. A waiter makes it deadlock. At exit, an incomplete phase with no waiters, no buffered bytes and all arrivals issued is tolerated as a "terminal reservation" (HB:2361-2373). | No exit rule | Pending. `quiescent` is left to the SyncTable (§2.8 G4). |
| tx-count range | `expect_tx`: checked against the phase total (HB:2214-2241). `arrive.expect_tx`: only the per-instruction aggregate is checked (RS:1522-1533). Each operand is checked ≤ 2^20-1 (RS:33-42). | Only u64 overflow (`CounterOverflow`) | The signed **state** `tx_expected - tx_completed` must stay in ±(2^20-1) after every op (`TxCountOutOfRange`, ISA §9.7.15.16.3 Table 43). Bytes buffered for the next phase count as its negative tx-count. The u32 operand is not range-checked. |
| `MAX_MBARRIER_EXPECTED_ARRIVALS` | 2^20-1, or 511 for `layout::v1` (HB:16-25) | 2^20-1 regardless of layout (SM:856, 1523) | Layout-dependent |
| `IncPending` beyond the limit | `BarrierArrivalOverflow` (HB:2194-2203) | Only u64 overflow | `PendingOverflow` |
| Arrival over `required` | `BarrierArrivalOverflow` | `ArrivalOverflow` | `ArrivalOverflow` |
| `arrive_drop` to `expected == 0` | Allowed. The next phase is dead: any arrive overflows and a wait hangs. | Allowed | `DropUnderflow` ("If the decrement causes the expected arrivals count to be zero, the behavior is undefined", ISA §9.7.15.16.17) |
| Parity vs phase | `completed_parity = gen & 1`; initial value 1 | `last_completed_phase`, initial 1 | `completed_parity()` |
| `try_wait` vs `test_wait` | Identical readiness (IS:1072-1112) | Both are `Wait`, consumed when ready | Both `TestParity` |
| Invalidate while pending | OK if there are no waiters and no queued completions | OK if there are no waiters and no outstanding tokens | Same (`InvalWithOutstanding`) |
| Re-init with an unconsumed completed phase | NumSim: allowed. Analysis: `WithoutInvalidation`. | `ReinitializeBeforeConsumption` | `ReinitBeforeConsumption`, both policies (ISA §9.7.15.16.12) |
| `.noComplete` arrive that would complete the phase | Untyped `EngineError` (HB:1157-1184) | Not checked | Typed `NoCompleteWouldComplete` ("must not cause the mbarrier to complete its current phase, otherwise the behavior is undefined", ISA §9.7.15.16.16). Keeps the engine's conservative "pending > count" rule. |
| Parity names a phase two or more behind | Aliases | Aliases | Aliases. Only the current or immediately preceding phase is valid (ISA §9.7.15.16.19), which the state-token form enforces with `InvalidStateToken`. A parity query cannot detect it. |
| Deferred arrival bound to `g`, but `g` completed through other arrivals before it lands | Action still enabled (`gen == current`). `begin_next_generation` rolls to `g+1` and the arrival lands **there**; only a `debug_assert` catches it (HB:2523-2524). | `CompletionAfterGenerationComplete` (SM:1254-1263) | **Both policies reject it** (`CompletionAfterComplete`). The engine behaviour is a bug. |
| Stale completion (bound generation < current) | Never enabled. It stays queued until `validate_quiescent` fails (HB:2348-2355). | `StaleCompletion`, immediately | `StaleCompletion`, immediately (fail closed earlier) |

### 2.5 Protocol rules (S) and fixes (F)

- **S1 Consumption before reuse.** A completed phase may not be advanced, by arrive, expect_tx, pending increment, deferred arrival or re-init, until a wait has observed it (SM:937-944, 1038-1049, 1543-1550).
  - This is ISA-backed for arrive-on operations: "For each primary phase of the mbarrier object, at least one test_wait or try_wait operation must be performed which returns True for waitComplete before an arrive-on operation in the subsequent primary phase" (ISA §9.7.15.16.5.1).
  - It therefore holds under both policies for `Arrive`, `IncPending` (the non-`.noinc` increment is part of an arrive-on) and `DeferredArrive`.
  - It stays Strict-only for `ExpectTx`.
  - Consumption is per slot, not per consumer: one observing wait suffices.
  - Buffering bytes for the next phase does not require consumption.
- **S2 Re-init requires `inval`** (SM:1575-1581; ISA §9.7.15.16.12). It now holds under both policies.
- **F1** (strict bug). `increase_pending_arrivals` on a `CompletedUnconsumed` slot raises the increment on the **completed** phase (SM:1143-1157), because there is no roll and no consumption check. The roll that follows drops it, so the deferred arrival then counts as a real arrival on `gen+1`. The engine rolls first (HB:2192). The reference rolls first. Under S1 it rejects the case in both policies, because the increment is part of an arrive-on.
- **F2** (engine bug). Deferred arrival into the next generation, as described in the §2.4 table.
- **F3** (engine). The drop half of an arrive is non-transactional. The reference is transactional.

### 2.6 Error taxonomy

**Strict kinds.** `StrictMbarrierError` (SM:261-382) has **20** kinds. The PTX rationale for each:

| # | Kind (SM line) | Condition | PTX justification | Reference |
| --- | --- | --- | --- | --- |
| 1 | `InvalidateWithOutstandingWork` (262) | `inval` with waiters or outstanding tokens | Using an invalidated mbarrier, including by an in-flight async completion, is UB | `InvalWithOutstanding` |
| 2 | `Uninitialized` (266) | Any operation except init on an uninitialized slot | "Performing any mbarrier operation except init on an uninitialized or invalidated object is UB" | `Uninitialized` |
| 3 | `InvalidPhase` (271) | Wait parity ∉ {0,1} | `phaseParity` is a 1-bit operand | `InvalidPhase` |
| 4 | `AcquireBeforeCompletion` (276) | The engine reported an acquired generation beyond the last completed one | None: it cross-checks engine against strict | dropped (no second machine) |
| 5 | `InvalidExpectedArrivals` (282) | Init count ∉ 1..=2^20-1 | Count range of `mbarrier.init` | `InvalidCount` (layout-aware) |
| 6 | `ReinitializeBeforeConsumption` (287) | Init over `CompletedUnconsumed` | Policy S1 (a phase completion would be lost unobserved) | `ReinitBeforeConsumption` (Strict) |
| 7 | `ReinitializeWhileActive` (293) | Init over an active phase, waiters, tokens or buffered bytes | UB: init on a valid object (ISA §9.7.15.16.12) | `ReinitActive` |
| 8 | `ReinitializeWithoutInvalidation` (302) | Init over any live slot | "The behavior of performing an mbarrier.init operation on a memory location containing a valid mbarrier object is undefined" (ISA §9.7.15.16.12) | `ReinitWithoutInval` (both policies) |
| 9 | `ArriveBeforeConsumption` (309) | Arrive on `CompletedUnconsumed` | ISA §9.7.15.16.5.1: at least one successful wait per phase before the next arrive-on | `ReuseBeforeConsumption{Arrive \| DeferredArrive}` |
| 10 | `ExpectTxBeforeConsumption` (315) | `expect_tx` on `CompletedUnconsumed` | S1 | `ReuseBeforeConsumption{ExpectTx}` |
| 11 | `ArrivalOverflow` (321) | Arrivals > required, or a drop larger than expected | Pending count underflow is UB. Dropping the expected count to 0 is UB (ISA §9.7.15.16.17). | `ArrivalOverflow`, `DropUnderflow` |
| 12 | `CounterOverflow` (328) | u64 overflow of any counter | None (implementation guard). Superseded by the state ranges of ISA Table 43. | `TxCountOutOfRange`, `PendingOverflow` |
| 13 | `TransactionOverDelivery` (334) | `completed_tx > expected_tx` once arrivals are complete | tx-count would go negative at phase completion: UB | `TxOverDelivery` |
| 14 | `DuplicateWaiter` (341) | Second wait registration by the same warp | None (implementation) | impossible by construction (no registry) |
| 15 | `CompletionAfterGenerationComplete` (347) | Completion bound to an already completed phase | The async op was counted in a phase that already completed: the program over-delivered | `CompletionAfterComplete` |
| 16 | `StaleCompletion` (354) | Bound generation < current | Same, older | `StaleCompletion` |
| 17 | `FutureCompletionNotBufferable` (361) | Bound generation > current+1, or current+1 while current is pending | Unreachable for well-formed tokens. Guards hardware's one-phase lookahead. | `FutureNotBufferable` |
| 18 | `UnknownCompletionToken` (369) | Landing without issue | None (implementation) | `UnknownToken` |
| 19 | `GenerationOverflow` (372) | Generation counter u64 overflow | None | unreachable (u64) |
| 20 | `CompletionTokenSpaceExhausted` (378) | Token id overflow | None | n/a |

**Engine kinds.** These are `SynchronizationError` variants raised by HB and RS:

- `BarrierUninitialized`
- `InvalidBarrierArrivalCount`
- `InvalidBarrierPhase`
- `BarrierReinitializedWhileWaiting`
- `BarrierReinitializedWhileActive`
- `BarrierReinitializedWithoutInvalidation`
- `BarrierArrivalOverflow`
- `TransactionOverflow`, which covers both the tx limit and over-delivery
- `DuplicateWaiter`
- `DuplicateMbarrierArrivalTarget`
- `InvalidMbarrierStateToken`
- `MbarrierLocalArriveRemoteAddress`
- `CompletionSourceOperationFailed`, for a duplicate expect_tx or pending target, and generation overflow
- `CompletionSourceNotQuiescent`

The following untyped `EngineError` messages are also raised:

- the noComplete check
- the drop underflow
- negative counts and phases (RS:22-42)
- "pending count requires a noComplete state"

**Reference.** `mbarrier::Error` has 19 kinds. Each one maps to a row above.

`TxCountOutOfRange` replaces the legacy operand-level tx check. `NoCompleteWouldComplete` is the typed form of the engine's untyped noComplete error.

### 2.7 Engine vs strict: disagreements

1. **Consumption (S1).** The engine never tracks whether a wait observed a completion. Strict does, and it errors on reuse.
2. **Re-init.**
   - The engine NumSim path permits re-init of an inactive slot without `inval`. The engine analysis path and strict reject it.
   - Strict checks consumption first.
3. **Limits.**
   - The engine checks 2^20-1 tx and pending limits, and a layout-v1 count of 511.
   - Strict checks only u64 overflow and a fixed 2^20-1 init range. Strict therefore accepts some transitions the engine rejects. Today this surfaces as an engine error after the strict preview.
4. **noComplete.** Engine only.
5. **Stale completion.** The engine reports it late, as non-quiescence. Strict reports it at landing.
6. **Late deferred arrival.** F2: the engine silently retargets it. Strict errors.
7. **Pending increment on an unconsumed completion.** F1: strict mis-attributes it. The engine is correct.
8. **Waiter release granularity.**
   - Engine: waiters keyed by warp, released by parity.
   - Strict: waiters keyed by `(generation, warp)`, released only for the exact generation (SM:1669-1676).
   - The two are equivalent, because a non-ready wait always targets the next completing generation.
9. **Acquire generation provenance.** Strict validates the engine-reported generation (`AcquireBeforeCompletion`, SM:1420-1456). The reference returns the generation itself.
10. **Copy-report / conditional parity (layout v1).** Only the engine models it (`report_on` HB:767-791, `ConditionalParity` HB:1702-1709).
11. **Init-fence coverage.**
    - Engine: `MbarrierInitFenceTracker` (HB:676-764).
    - Strict: `mark_init_fenced_many` (SM:873-900).
    - Both ignore not-yet-initialized targets. Ordering is checked by the causality checker (SM test 1762), not by either state machine.

### 2.8 Gaps left for the production step

- **G1.** Copy-report / conditional parity (layout v1) is not in the reference. Add it with a differential test when the corpus needs it.
- **G2.** State-token encoding (HB:530-549) belongs to the ABI layer. `Outcome::Arrived{gen, pending_before}` carries all of its content.
- **G3.** `fence.mbarrier_init` coverage is a causality fact (racecheck and synccheck), not barrier state.
- **G4.** Exit policy for under-delivered phases: keep the engine's "terminal reservation" tolerance (HB:2361-2373). It belongs in `quiescent`.

## 3. Named barriers

### 3.1 PTX forms (ABI)

**Frontend decode** (FE `sync.rs:645-710`):

- `bar.sync` and `barrier.sync.aligned` → `bar_sync`, aligned.
- `barrier.sync` → `barrier_sync`, unaligned.
- Every arrive → `bar_arrive` (IS:337-354), aligned in practice.
- The `bar.red`/`barrier.red` reductions `popc.u32`, `and.pred` and `or.pred` → `bar_reduce` (`runtime/instructions/collective.rs:637-720`).

**Operands.**

- The count `b` is optional for sync and reduction; it defaults to `warps_per_cta*32` (FE:2121-2125). It is required for arrive (FE:689-691).
- The id and count must be warp-uniform.
- The plan requires `id < 16`, and `b > 0 && b % 32 == 0` (RS:1413-1438).
- No check that `b <= CTA threads`: an oversized count simply never completes.

### 3.2 State

**Engine** `NamedBarrierEntry` (HB:2801-2814), keyed `(cta, id, namespace)`:

- `expected_arrivals`, set from the first contribution of a generation
- `generation`, `completed_through`
- the phase's `arrival_count` (lanes), `arrived_warps`, `complete`
- `contributors: (warp, is_sync) → mask`
- waiters

**Strict** `NamedBarrierSlot` (SNB:371-381) holds the same fields plus:

- per-contribution `{operation, sync_aligned, witness}`
- exact `completed_waiters[gen]`

**Reference** `named::State`: `expected`, `gen`, `complete`, `arrived`, `lanes[(warp, flavor)]`, and the strict `sync_origins`.

### 3.3 Transitions

The engine's `register_contribution` (HB:2951-3070) and strict's `contribute` (SNB:589-714) agree on these steps:

1. `b == 0` → `InvalidCount`.
2. An empty mask → `EmptyMask`.
3. **Lazy roll.** If the generation is complete: `gen += 1`, the new `b` is taken from this contribution, and the bookkeeping is cleared (HB:2983-2997, SNB:397-422).
4. `b` differs from the generation's `b` → `ContractMismatch` (HB:2998-3002, SNB:625-635).
5. The same `(warp, flavor)` contributes overlapping lanes → `Duplicate` (HB:3006-3018, SNB:636-648).
   - `bar.arrive` followed by `bar.sync` on the same lanes is allowed. This is the producer/consumer idiom (HB:2807-2811).
   - Disjoint lane subsets accumulate.
6. `arrived + popcount(mask) > b` → `ArrivalOverflow`.
7. `arrived == b` → complete, and the waiters are released.
   - `Sync` returns `Ready` if it completed the generation, else `Registered{gen}`.
   - A later `Resume{gen}` is `Ready` iff `gen < current`, or `gen` is current and complete.

**Engine-only collective checks** (KE:4109-4228, 7902-7915):

- Every arrive, and every aligned sync, must have a full 32-lane mask, unless the operation has `elect.sync` provenance.
- Unaligned full-CTA syncs recombine divergent paths: partial paths pass without waiting, and the full-mask path waits once (KE:4158-4176, RS:1309-1328).
- An aligned `bar.sync` credits the setmaxnreg warpgroup sync (KE:4139-4151).

### 3.4 Errors and PTX basis

| Reference kind | Engine | Strict | PTX basis (ISA §9.7.15.1 unless noted) |
| --- | --- | --- | --- |
| `InvalidCount` | plan `"positive multiple of 32"`; hub `InvalidBarrierArrivalCount` | `InvalidExpectedArrivals` | "the value must be a multiple of the warp size" |
| `PartialWarp{mask, live}` | `warp_collective_divergence`, waived under `elect.sync`; hub `InvalidBarrierArrivalCount{0}` for an empty mask | `ElectSyncParticipation` (elect entry mask); `InvalidArrivalCount` | "causes executing thread to wait for all non-exited threads from its warp and marks warps' arrival". A strict subset of the non-exited lanes is UB (`.aligned`) or a hang (unaligned). Lanes at different barrier instructions fail closed (sync-isa-answers Q3/Q5). |
| `ContractMismatch` | `ContractMismatch` | `ContractMismatch` | "using the same barrier name and thread count" |
| `Duplicate` | `DuplicateArrival` (lane overlap) | `DuplicateContribution` | "keep a warp from executing more barrier instructions than intended … prior to the reset of the barrier" |
| `RedMixed` | — | — | "barrier{.cta}.red should not be intermixed with barrier{.cta}.sync or barrier{.cta}.arrive using the same active barrier. Execution in this case is unpredictable." |
| `ArrivalOverflow` | `BarrierArrivalOverflow` | `ArrivalOverflow` / `CounterOverflow` | More warp arrivals than `b` |
| *(dropped)* | — | `AlignedSyncContractMismatch` (SNB:719-773) | "Different warps may execute different forms of the barrier{.cta} instruction using the same barrier name and thread count". Mixing aligned and unaligned forms is allowed. |
| `ResumeFuture` | `UndefinedOccurrence` | `ResumeWithoutRegistration` / `ResumeBeforeCompletion` | Internal |
| Lint `DanglingAtExit` | `validate_quiescent` error (HB:3132-3154) | only `FullCtaAlignedMissingParticipants` (SNB:779-824) | "Barriers exclusively waiting on arrivals from exited threads are always released" (§9.7.14.7). A dangling arrive at exit is a Review lint, not an error. |

The strict kinds are 11 (SNB:136-206):

- `ElectSyncParticipation`
- `InvalidExpectedArrivals`
- `InvalidArrivalCount`
- `ContractMismatch`
- `DuplicateContribution`
- `ArrivalOverflow`
- `CounterOverflow`
- `GenerationOverflow`
- `ResumeWithoutRegistration`
- `AlignedSyncContractMismatch`
- `FullCtaAlignedMissingParticipants`

### 3.5 Disagreements

1. **elect.sync is opposite in spirit.**
   - The engine waives the full-warp requirement under `elect.sync`.
   - Strict demands that the participation mask equals the elect entry mask, and applies this to arrive and aligned sync only.
   - Resolved (Q5): both are wrong. The reference requires the participation mask to equal the warp's non-exited lanes, independent of `elect.sync` (§3.6).
   - Reference: Numeric waives the requirement; Strict also checks `ElectSyncParticipation`.
2. **Full-warp requirement.**
   - The engine applies it to every arrive, including unaligned `barrier.arrive`, which the frontend collapses to `bar_arrive`. It also applies it to `barrier.red`. Per the ISA answers this is correct in *what* it requires (all non-exited lanes). It is stricter only in requiring one convergent instance, which the reference adopts as the fail-closed rule.
   - Strict has no such check.
3. **Aligned-origin contract.** Strict only. It rejects the completing contribution.
4. **Unaligned recombination.**
   - Engine: only for full-CTA counts, where partial paths do not block.
   - Strict: allows a partitioned resume for every unaligned count.
   - Reference: no recombination. Lanes at different instructions fail closed (§3.6).
5. **Exit.** The engine flags any incomplete generation. Strict flags only full-CTA all-aligned generations, and only when no other finding exists.
6. **Error granularity.** Strict splits overflow into `CounterOverflow` and `ArrivalOverflow`, and has `GenerationOverflow`.
7. **Waiter retention.** The engine keeps only `completed_through`. Strict keeps exact per-generation waiter sets. The reference needs neither, because readiness is a function of `gen`.
8. **`bar.red`.**
   - The engine reduces over `arrived_warps`, which includes arrive-only warps.
   - Neither model rejects mixing `bar.red` with sync or arrive on one generation. The ISA calls that unpredictable; the reference rejects it (`RedMixed`).
9. **Possible race (unverified).** `accumulated_arrival_mask` is called after `register` without holding the hub lock (KE:4133-4134; HB:3198-3201).

### 3.6 Reference (ISA-resolved)

- **Unit.** A contribution must be executed by exactly the warp's non-exited lanes (`mask == live`). The warp's arrival then counts 32 threads toward `b`. There is no lane accumulation across contributions, so lanes reaching different barrier instructions fail closed as `PartialWarp`. The engine's unaligned full-CTA recombination (KE:4158-4176) is therefore not reproduced.
- **Flavors.** `Arrive`, `Sync` and `Red`. One warp may arrive and then sync in one generation; the CUTLASS producer/consumer idiom depends on this. The ISA warns against it ("care must be taken") without forbidding it. Mixing `Red` with the other two flavors is `RedMixed`.
- **`.aligned`** carries no barrier state. It does not appear in `Contribution`.
- **Not modeled: whole-warp exit.** A barrier whose missing warps have all exited should be released (§9.7.14.7). With an explicit `b` this needs the scheduler to know which warps were expected. That is gap G8; the SyncTable's deadlock check must treat such a hang as `incomplete`, not as success.

## 4. Cluster barrier

### 4.1 PTX forms (ABI)

- `barrier.cluster.arrive[.release|.relaxed][.aligned]` → `ClusterArrive<Sem, ALIGNED>` (IS:114-117, 437-472).
  - `DefaultRelease` and `Release` publish memory; `Relaxed` does not.
- `barrier.cluster.wait[.acquire][.aligned]` → `ClusterWait<ALIGNED>` (IS:119-123, 487-501). The wait is always acquire.
- Barrier state never reads the semantics. Only racecheck and synccheck causality do (`race_check.rs:5701-5784`, SC:3400-3440).
  - A relaxed arrive still carries a pending `fence.mbarrier_init` (SC:3409-3418).

### 4.2 State and transitions

**State.**

- Engine `ClusterBarrierState` (CB:79-94): `current_generation`, per-phase `lane_arrivals`, `arrivals`, `waiters`, and per-warp `last_arrival_generation` and `last_waited_generation`.
- Strict (SCB:217-231): the same, at whole-warp granularity, with participants frozen at the first arrive.
- Reference `cluster::State`: `participants`, `gen`, `lanes`, `last_arrival`, `last_waited`.

**Arrive** (CB:162-241; SCB:340-423):

1. Partial-mask checks:
   - `.aligned` with a partial mask → `PartialWarp`.
   - The engine accumulates lanes for unaligned partial arrivals.
   - Strict rejects any partial arrival.
2. Overlap with lanes this warp already arrived in this generation → `EarlyArrival`.
3. A full warp counts. When every participant has counted, the generation completes and `gen += 1`, eagerly.
4. A rearrival after the previous generation completed without a wait is legal.
   - The engine's `ClusterBarrierRearrivalWithoutWait` is never constructed (`completion.rs:986-990`).
   - Strict flags it in the outcome; synccheck treats it as incomplete (SC:2801-2816, SFU:4724-4752).
   - The reference returns `rearrival_without_wait` in `Outcome::Arrived`.

**Wait** (CB:266-321, 444-510; SCB:425-529):

- A partial mask → error in both models.
  - The engine reports `UnalignedWaitUnsupported` for an unaligned partial wait, since the warp-unit model cannot represent it.
- No full arrival recorded → `WaitBeforeArrival`.
- `last_waited >= last_arrival` → `DuplicateWait`.
- Otherwise the wait targets the warp's **latest** full arrival. It is ready once that generation has completed.
  - The CB doc comment at 71-72 and 243-244 says "oldest". That is a doc/code mismatch.

### 4.3 Errors and PTX basis

| Reference kind | Engine | Strict | PTX basis |
| --- | --- | --- | --- |
| `UnexpectedParticipant` | `ClusterBarrierContextMismatch` | `UnexpectedParticipant` / `ContractMismatch` | Internal |
| `PartialWarp` | `PartialWarpSynchronization` | `PartialWarpParticipation` | `.aligned`: all threads of the warp execute the same instruction |
| `EarlyArrival` | `DuplicateArrival` | `EarlyArrival` | "Each thread must arrive at the barrier only once before the barrier completes" (ISA §9.7.15.3) |
| `WaitBeforeArrival` | `ClusterBarrierWaitBeforeArrival` | `WaitBeforeArrival` | Derived: a wait before one's own arrive is a self-deadlock |
| `DuplicateWait` | `DuplicateWaiter` | `DuplicateWait` | Model rule: one arrive, one wait |
| `IncompleteAtExit` | `CompletionSourceNotQuiescent`, or a deadlock with exit evidence (`executor.rs:822-878`) | `incomplete_generations()` → incomplete verdict | Not an error: exited threads leave the membership (ISA §9.7.15.3, §9.7.14.7). The reference has no such kind. |

The strict kinds are 9: the five in the table, plus `ResumeWithoutRegistration`, `ResumeBeforeCompletion`, `GenerationOverflow` and `ContractMismatch`.

### 4.4 Disagreements

1. **Participant set.**
   - Engine: the *selected* warps of the cluster (CB:110-124).
   - Strict: the full `cluster_warp_range` (RS:113-119).
   - A partial-cluster launch selection therefore makes the cross-check fail with an infrastructure error.
   - The reference takes `participants` as an input. The SyncTable must pass the selected set.
2. **Unaligned partial arrive.** The engine accumulates lanes; strict rejects. In checker modes the engine's analysis-gap gate masks the difference (KE:4237-4249).
3. **Rearrival without wait.** The engine accepts silently; strict marks it as unmodeled.
4. **Duplicate wait.** The engine detects it at poll; strict detects it at register.
5. **Exit.** The engine raises an error; strict yields incomplete. Both are wrong: membership is exit-aware (§4.5).

### 4.5 Reference (ISA-resolved)

- **Exit-aware membership.** `State.live[w]` holds each participant warp's non-exited lanes. `Cmd::Exit{warp, lanes}` removes lanes. A warp with no live lane leaves the membership. If every remaining member has arrived, the exit completes the generation (`Outcome::Exited{completed: true}`). "wait for all non-exited threads of the cluster to perform barrier.cluster.arrive" (ISA §9.7.15.3); "Barriers exclusively waiting on arrivals from exited threads are always released" (§9.7.14.7).
- **Participation.** Arrive and wait must be executed by exactly the warp's non-exited lanes ("wait for all non-exited threads from its warp"). A strict subset is `PartialWarp`. This replaces both the engine's unaligned lane accumulation and strict's blanket full-mask rule.
- **Error bases.**
  - `EarlyArrival`: "Each thread must arrive at the barrier only once before the barrier completes".
  - `WaitBeforeArrival`: derived. The waiter waits on itself, a guaranteed hang.
- **Rearrival without a wait** is reported in `Outcome::Arrived` and treated as unmodeled by checkers; the ISA is silent.
- **Exit check.** There is no exit error: at kernel exit every thread has exited, so every generation completes.

## 5. Async groups

### 5.1 PTX forms (ABI)

Sources: AC:108-254, `abi/v2/async_copy.rs`.

- **`cp.async`:** `CpAsync<4|8|16, NoFill|ZeroFill|SourceSize>` (AC:416-570).
- **Commit:** `cp_async_commit_group`, `cp_async_bulk_commit_group`.
- **Waits:** `CpAsyncWaitGroup(n)`, `BulkWaitGroup(n)`, `BulkWaitGroupRead(n)`. There is no `.read` for `cp.async`.
- **`cp.async.wait_all`** has no engine form. The frontend lowers it to commit followed by `wait_group 0` under the same mask.
- **Arrive-on:** `CpAsyncMbarrierArrive`, `CpAsyncMbarrierArriveNoInc`.
- **Bulk-group domain:** `BulkS2g*`, `TensorS2g*`, `*ApplyPriority`.
- **Not async groups (mbarrier `complete_tx` producers instead, §2):** G2S, S2S, S2C, TMA G2S/gather4/im2col, `st.async`, `red.async`.
- **Release domain:** `StRelease`/`RedRelease`. This domain is internal and has no PTX wait.

### 5.2 State and transitions

State is per (warp, lane, domain): `DomainState {next_group_ordinal, open_issues, committed_groups}` (AG:603-629). Each group has:

- `completion ∈ Pending → ReadsComplete → FullyComplete`
- `closes_group`
- `full_physical_actions`, the deferred mbarrier arrive-ons

**Issue** (AG:1107-1238) appends to the open issues.

- `cp.async` writes its bytes to shared memory **at issue** (KE:5491, 5568). A stale read is therefore visible only to racecheck.
- Bulk shared-to-global copies capture the source at issue and publish the global writes at full completion (AG:819-889).

**Commit** (AG:701-741, 1282-1405) moves the open issues into a new group.

- An empty commit creates an empty group that is already complete. It still counts toward wait depth.

**ArriveOn** (`cp.async.mbarrier.arrive`; AG:1313-1405, RS:918-988). Exactly one of:

1. If there are open issues, it cuts a **non-closing** batch that carries the arrival.
2. Else, if an unfinished group exists, it attaches to the newest one.
3. Else, the arrival is released immediately.

The mbarrier half is separate: `IncPending` (unless `.noinc`) and `Issue`.

**Milestones** (AG:794-889):

- `ReadsDone` is enabled on the oldest `Pending` group.
- `FullyDone` is enabled on the oldest non-full group, and only once its reads are done.
- `FullyDone` releases that group's arrivals (MC:291-298).

**`Wait{n, read}`** (AG:897-983, 1456-1613):

- **Prefix:** groups up to and including the (n+1)-th newest *closing* group. Uncommitted issues are never awaited.
- **Ready:** every group in the prefix has reached the needed milestone in every lane of the mask.
- **Full wait:** retires the prefix.
- **`.read` wait:** retires only empty, complete groups in the prefix. Non-empty groups stay for a later full wait.

**Exit** (AG:1417-1454):

- Bulk open issues are committed implicitly. `cp.async` open issues are not.
- `quiescent` then rejects uncommitted issues and groups that are not fully complete (AG:1800-1830).

### 5.3 Errors

| Reference | Engine | PTX basis |
| --- | --- | --- |
| `InvalidForm` | `.read` on cp.async; Release-domain wait (AG:1078-1105) | `.read` exists only on `cp.async.bulk.wait_group` |
| `NotEnabled` | "blocked by FIFO milestone order", unknown action, already completed (AG:831-877) | Internal |
| Lint `UncommittedAtExit` | `CompletionSourceNotQuiescent` "uncommitted issue(s)" | The ISA does not require committing before exit, so this is a lint (§5.5). |
| `PendingAtExit` | "pending full completion" | — |

Mask, issuer and footprint validation errors (AG:213-313, 1131-1280) belong to the issue layer, not to `step`.

### 5.4 Model comparison

- **Synccheck has no async-group model at all.** The effects are discarded (SC:3140-3151, 3750, 3767; SFU:3524-3527). Only the mbarrier half of `cp.async.mbarrier.arrive` is modeled.
  - The fixed verifier lets that arrival land at any time after issue; the engine ties it to FIFO full completion. The verifier is therefore a sound superset.
  - The reference `async_group` module is the first independent model of these rules.
- **Racecheck** orders `cp.async` bytes at milestones, while the engine moves them at issue.
  - One merged warp clock is acquired by the whole wait mask (`race_shadow.rs:9243-9257`). This lets lane A acquire lane B's copies, which is more permissive than PTX per-thread semantics.
  - Gap G6 is resolved: acquire per lane (§5.5).

### 5.5 Reference (ISA-resolved)

- **Granularity.** Async-groups and their visibility are per thread (ISA §9.7.10.28.1.1, §9.7.10.28.3.3). State is per (warp, lane, domain), with separate read and write milestones.
- **`.read` waits.** `Wait{read: true}` reports `acquired: ReadsDone`. It releases only the source and tensormap reads; it never makes destination writes visible (ISA §9.7.10.28.6.2). Racecheck must acquire per lane, and must not publish destinations on `.read`. That closes gap G6.
- **Exit.** Uncommitted issues at exit are a lint (`exit_lint`, `UncommittedAtExit`), not an error: the ISA does not require a commit before exit. `quiescent` keeps only the infrastructure check that every committed group completed.

## 6. tcgen05

### 6.1 Forms

Sources: `runtime/instructions/tcgen05.rs`.

| Instruction | Location | Notes |
| --- | --- | --- |
| `tcgen05.alloc<EXCLUSIVE>(dst, ncols, cta_group)` | :364-408 | |
| `tcgen05.dealloc<EXCLUSIVE>(taddr, ncols, cta_group)` | :424-454 | |
| `tcgen05.relinquish_alloc_permit(cta_group)` | :456-481 | |
| `tcgen05.commit` / `commit.multicast` / `CommitSharedA[Multicast]` | :483-550 | |
| `wait::ld` / `wait::st` | :553-590 | |
| `fence::before_thread_sync` / `after_thread_sync` | :592-617 | No-ops in the engine; their meaning lives in racecheck (`race_check.rs:5188-5194`; `tcgen_fence.rs:46-64`) |

### 6.2 Lifecycle state and transitions

**State.**

- The engine keeps one `TcgenCtaSnapshot` per CTA (TG:187-193): sorted `allocations`, `relinquished`, sticky `cta_group`, `last_allocation_columns`.
- The reference `tcgen::State` holds a pair of these plus the capacity (default 512, from `execution_policy.tmem_columns`).

**`apply_contributions`** (TG:798-950) checks, in order:

1. The `cta_group` is 1 or 2.
2. The sticky `cta_group` matches → `CtaGroupMismatch`. This is checked only across lifecycle operations.
3. Column rule (TG:703-710): a power of two in [32, min(cap, 512)]; with `.exclusive`, any multiple of 32 in that range.
4. Alloc after relinquish → `AllocAfterRelinquish`.
5. Columns larger than the last allocation → `AllocationSizeIncrease` (ISA §9.7.18.7.1: "should not increase between any two allocations").
6. First-fit 32-aligned base, common to both CTAs for a pair → `AllocationUnavailable` (the reference returns `Blocked`, §6.6).
7. Dealloc must match an exact live `(base, cols)` → `DeallocationMismatch`.

After the checks:

- **Commit:** all participating CTAs are updated atomically.
- **Exit:** live allocations → `LiveAllocationsAtExit` (TG:528-551).

**Blocking.**

- `cta_group::1` is immediate (TG:572-588).
- `cta_group::2` is a two-warp collective keyed by static op and loop path (TG:735-766).
  - The peer must use the same `warp_id_in_cta` (TG:757-759). This has no ISA basis and is dropped (§6.6).

### 6.3 Work tokens and commit

These are the reference's `WorkState` and `work_step`.

- `mma`, `cp` and `shift` enqueue per lane and per `cta_group` (ordering.rs:67-99, 780-813). Exactly one issuing lane is allowed.
- `ld`/`st` require a full warp.
- `commit.cta_group::N` drains only group N's work (ordering.rs:815-849). Work issued under the other group stays untracked, and no error is raised.
- **The commit arrival:**
  - One issuing lane.
  - Multicast to the `ctaMask` CTAs at the same shared-memory offset (RS:1719-1771).
  - `arrive::one`, with zero tx.
  - Its generation is bound at issue.
  - NumSim lands it immediately; checker modes defer it (§1.4).
- `wait::ld`/`wait::st` never block. They drain the per-lane queue (ordering.rs:851-879, 910-921).

### 6.4 Engine vs fixed verifier

The fixed verifier is SFU:3803-3955 and 4832-4986.

- **Shared rules:** column rule, stickiness, alloc after relinquish, size increase, exact dealloc, exit check.
- **Verifier only:**
  - It requires the allocation base to match the canonical base under every explored order.
  - It delays a dealloc while a matching alloc has not yet run (SFU:4192-4227).
  - It treats argument mismatches as build-time `InvalidTcgenCollective`.
- **Engine only:** partial-warp and `cta_group ∉ {1,2}` checks at plan time.

### 6.5 Disagreements with PTX

1. **Allocation blocking.** PTX `alloc` blocks until columns are free; both models error immediately.
   - **Resolved (Q1):** return `Blocked` and let the scheduler retry (§6.6).
2. **One `cta_group` per kernel.** PTX wants it across all tcgen05 ops. The engine checks it across lifecycle ops only, and commit silently ignores work of the other group.
3. **Single-lane commit.** PTX commit is per-thread; the engine allows one issuing lane only.
4. **Pending work at exit.** Uncommitted or unwaited tcgen work at exit is not diagnosed (ordering.rs:1036+). This is gap G7.
5. **TMEM memo.** The `Immediate` `cta_group::1` path does not bump the hub generation (TG:651-669 vs 242-246). The cached bounds check uses the constant 512 rather than the capacity (TG:397 vs 475). These are bugs to avoid in the port.

### 6.6 Reference (ISA-resolved)

- **Blocking alloc.** `Alloc` returns `Blocked` when no common free interval exists. An `.exclusive` alloc also blocks while any allocation is live, and every alloc blocks while an exclusive allocation is live. "The tcgen05.alloc blocks if the requested amount of Tensor Memory is not available"; "An exclusive allocation operation blocks until there is no other live allocation" (ISA §9.7.18.7.1). The scheduler reports a deadlock only when nothing can progress. `AllocationUnavailable` is gone.
- **Exclusivity.** Allocations record `exclusive`. A dealloc must match base, width **and** exclusivity: "Memory must be deallocated with .exclusive if and only if it is allocated with .exclusive".
- **Width rules.**
  - Non-exclusive: a power of two in [32, 512].
  - Exclusive: a multiple of 32 in [32, `exclusive_max`], where `exclusive_max` is 512 on sm_100f/103/110 and 576 on sm_107f (PTX Table 58). It is a `State` parameter.
- **`AllocationSizeIncrease`** is kept. "The number of columns allocated should not increase between any two allocations in the execution order within the CTA." `last_alloc_columns` is sticky across deallocs.
- **`AllocAfterRelinquish`** is kept ("illegal").
- **`cta_group` uniformity is kernel-wide.** It covers lifecycle, mma, cp, shift and commit: "All tcgen05 instructions within a kernel must specify the same value for the .cta_group qualifier". It lives in `tcgen::KernelState`. A commit or mma with the other group is `CtaGroupMismatch`, not ignored. `WorkState` therefore has a single uncommitted queue.
- **Peer warp index.** The same-`warp_id_in_cta` requirement for `cta_group::2` (TG:757-759) is dropped. The ISA asks only for "one warp from each of the peer CTAs".

## 7. setmaxnreg

### 7.1 Forms and constants

**Instruction.** `setmaxnreg.{inc,dec}.sync.aligned.u32 count`.

**ABI pre-check** (`runtime/instructions/control.rs:36-70`):

- `count ∈ [24, 256]` and a multiple of 8. This is untyped, so the hub's `InvalidCount` is unreachable from the ABI.
- A full warp is required (KE:4512-4591).

**Constants** (SR:16-20, 973-977):

- Pool: 512 per-thread register units per CTA.
- Default count per warpgroup: `floor(512 / ceil(warps/4) / 8) * 8`.

### 7.2 Three legacy models

1. **`SetmaxnregHub`** (checker modes; SR):
   - **State:** a real pool with `available` starting at 0, so only `dec` releases.
   - **Collective:** a four-warp collective per dynamic ordinal.
   - **Checks:**
     - `Divergence`
     - `SequenceMismatch`
     - `MissingWarpgroupSync`: a second setmaxnreg without an intervening warpgroup-wide `.aligned bar.sync`. The sync is credited at SR:740-791 and KE:4134-4146.
     - `InvalidDirection`: `inc` below current, or `dec` above current. Equal is allowed.
   - **dec:** releases registers into the pool.
   - **inc:** either immediate, or `Pending` until a pump grants it. Grants go to the lowest hashed action id, one per pump.
   - **Exit:** checks for pending increases and for oversubscription (SR:1460-1525).
   - **Errors:** 10 kinds (SR:87-116).
2. **NumSim occurrence protocol** (ordering.rs:923-1033; KE:4522-4550):
   - A warpgroup rendezvous plus the missing-sync rule (untyped).
   - No pool, no direction check. **`inc` never blocks.**
3. **Fixed verifier** (SV; SFU:3581-3801, 4797-4823, 5049-5074):
   - The same pool rules.
   - Grants are explored in every enabled order.
   - It has `WarpgroupPending`.
   - It does **not** model the warpgroup-sync rule.
   - Its initial-count validation differs: `> 0`, a multiple of 8, and `Σ <= 512`.

### 7.3 Reference (`setmaxnreg`)

The reference follows the hub. Its commands are `Configure`, `Set`, `WarpgroupSync`, `Grant` and `Poll`.

**Errors:**

- `InvalidCount`
- `ConfigureConflict`
- `IncompleteWarpgroup`
- `MissingWarpgroupSync`
- `InvalidDirection`
- `WarpgroupPending`
- `GrantNotEnabled`
- `PendingAtExit`

**Invariant:** `available + Σ current` is conserved and `<= 512`.

### 7.4 PTX basis

- All warps of the warpgroup execute the same setmaxnreg. This underlies `Divergence`, the full-warp checks and `IncompleteWarpgroup`.
- The count is in [24, 256] and a multiple of 8.
- `inc` requests more registers and blocks until they are available. `dec` releases them.
- A warpgroup must synchronize explicitly before a subsequent setmaxnreg.
- **[VERIFY]**:
  - whether an equal count is legal for both directions,
  - whether `inc` to a lower count is UB or an error.

### 7.5 Disagreements

1. **Warpgroup sync.** The hub and NumSim enforce it; the verifier does not.
2. **Grant schedule.** The hub uses one deterministic hashed order; the verifier explores every order; NumSim grants nothing and never blocks.
3. **Pool, direction and blocking.** Only the checker modes have them. A kernel that deadlocks on the pool in racecheck or synccheck **completes in NumSim**.
4. **Validation.** Initial-count validation differs between the hub and the verifier. Only the engine records provenance lots.

### 7.6 Decision

The redesign runs one numeric path in every mode (numsim CLAUDE.md, "System Design"). The reference therefore adopts the hub semantics as the only semantics, with no `Policy`. This is an intentional NumSim behaviour change: NumSim can now report `InvalidDirection` or a pool deadlock. It needs an explicit test delta.

## 8. Consolidated decisions

The open questions Q1–Q7 are answered in `sync-isa-answers.md` (PTX 9.4). The complete list of behaviour changes relative to legacy, each with its ISA cite, is `sync-behaviour-deltas.md`. In summary:

| | Decision | Basis |
| --- | --- | --- |
| D1 | mbarrier late deferred arrival → `CompletionAfterComplete` (fixes F2) | Over-delivery to a completed phase |
| D2 | Stale completion rejected at landing | Fail closed earlier |
| D3 | Signed tx-count **state** range ±(2^20-1), checked after every op; no operand check | ISA §9.7.15.16.3 Table 43, §9.7.15.16.14 |
| D4 | `IncPending` rolls first (fixes F1) and needs consumption (S1) | ISA §9.7.15.16.5.1, §9.7.15.16.18 |
| D5 | Transactional drop | — |
| D6 | setmaxnreg pool semantics in every mode | §7.6 |
| D7 | No waiter registries; `armed` models a pre-registered strict waiter | §1.4 |
| Q1 | `tcgen05.alloc` blocks (`Blocked`); `.exclusive` waits for no live allocation; relinquish → later alloc is an error | ISA §9.7.18.7.1 |
| Q2 | mbarrier re-init without `inval` is an error under every policy | ISA §9.7.15.16.12 |
| Q3 | Named: `b` is a multiple of 32; every form needs all non-exited lanes; aligned/unaligned mixing is allowed; `.red` mixing is an error; lanes at different instructions fail closed; dangling arrive at exit is a lint | ISA §9.7.15.1, §9.7.14.7 |
| Q4 | Cluster membership drops exited threads | ISA §9.7.15.3, §9.7.14.7 |
| Q5 | A barrier executed by a strict subset of the warp's non-exited lanes is an error; the elect waiver and the entry-mask rule are both dropped | ISA §9.7.15.1, §9.7.15.15 |
| Q6 | `AllocationSizeIncrease` is kept, sticky | ISA §9.7.18.7.1 |
| Q7 | Async-group completion and visibility are per lane; `.read` releases sources only | ISA §9.7.10.28.1.1, §9.7.10.28.6.2 |
| L1 | Arrival limits 1..2^20-1 (v0) and 1..511 (v1); pending overflow is an error; `drop` to 0 is an error; typed `NoCompleteWouldComplete` | ISA §9.7.15.16.3, .12, .16, .17, .18 |
| L2 | S1 holds under both policies for arrive-on ops; `ExpectTxBeforeConsumption` is Strict-only | ISA §9.7.15.16.5.1 |
| L3 | `cta_group` is uniform across all tcgen05 ops; the same-warp-id peer check is dropped; `exclusive_max` is a parameter (512 or 576) | ISA §9.7.18.7.1, Table 58 |

**Still open.**

- **G8.** Named-barrier release when the missing warps have all exited.
- **setmaxnreg** equal-count and direction semantics (§7.4).
- **Copy-report / conditional parity (G1).**

## 9. Reference crate and tests

`core-rs/numsim-sync-ref/` is a standalone crate with an empty `[workspace]`. Its only dev-dependency is `proptest =1.6.0`.

| Module | Lines | Covers |
| --- | --- | --- |
| `mbarrier.rs` | ~600 | §2, both policies |
| `named.rs` | ~230 | §3 |
| `cluster.rs` | ~230 | §4, exit-aware |
| `async_group.rs` | ~305 | §5 |
| `tcgen.rs` | ~340 | §6, kernel `cta_group`, lifecycle and work queues |
| `setmaxnreg.rs` | ~245 | §7 |

`tests/properties.rs` runs 2048 random sequences per protocol. Each sequence is up to 60 commands, and token and ordinal operands are resolved against the live state.

**Generic properties, checked for every protocol:**

- determinism
- `Err` leaves the state unchanged
- every reachable state satisfies `check_invariants`

**Protocol-specific properties:**

- **mbarrier:**
  - The phase is monotone and advances by at most one per step.
  - `arrived <= required`.
  - Completion implies `tx_completed == tx_expected` and `arrived == required`.
  - No completion happens without a pending phase.
  - No phase completes unreported.
  - Ready iff the parity matches.
  - The bound generation of `Issue` is `gen + complete`.
  - Bytes landed per generation equal the tx-count that generation completes with.
  - Strict refines Numeric.
- **named / cluster:**
  - The generation is monotone.
  - Ready only for completed generations.
  - No accepted contribution has a partial warp.
  - Named: `.red` never mixes. Cluster: exits release waiters.
- **async groups:**
  - FIFO milestones.
  - A ready wait has its prefix satisfied, and `.read` acquires only `ReadsDone`.
  - Deferred arrive-ons are conserved (attached = released + held).
- **tcgen:**
  - Allocations stay inside TMEM and do not overlap.
  - An exclusive allocation is the only live one.
  - A blocked alloc leaves the state unchanged.
  - No alloc after relinquish, and allocation widths never increase.
  - Relinquish is sticky; `cta_group` is kernel-wide.
- **setmaxnreg:**
  - Pool conservation and no oversubscription.
  - A successful grant clears the pending increase.

**Mutation checks.** Each of these mutations of `mbarrier.rs` is caught by a test:

- dropping the tx condition from completion
- dropping strict consumption-at-completion
- dropping next-generation buffering
- dropping buffered bytes on roll

**Differential use (later).** Production `numsim-core` step functions must:

1. map their `Cmd`/`Outcome`/`Error` 1:1 onto these enums, after stripping witnesses;
2. run the same strategies;
3. assert equal results and an equal projected state after every step.

Per redesign §2.6, this runs under `cfg(test)` and nightly.

### 9.1 Production implementation (`numsim-core::sync`)

The production bodies in `numsim-core/src/sync/*.rs` implement the reference semantics. They validate against `&State` and then commit, with no per-step state clone.

`tests/sync_differential.rs` feeds random sequences to both implementations: 4096 cases per property, up to 80 commands each. After every command it requires identical results, states and exit checks. Coverage:

- every protocol, with mbarrier under both policies;
- the tcgen kernel group and work queues;
- the `SyncTable` dispatch (transactional on error, `Blocked` lifting);
- `SyncTable::enabled`. An mbarrier completion is enabled unless landing it now would be premature. A completion that can never apply is enabled, so it errors at landing. A milestone or grant is enabled exactly when applying it succeeds.

Eight seeded mutations of the production code were each caught.

Side queries are pinned in `sync/query.rs`: the state-token layout, `pending_count`, and `check_layout`.

Error mapping: `SyncError::finding_kind` maps each error to a report kind. `RuntimeError` marks infrastructure faults. Exit lints go through `SyncTable::exit_lints` and are reported as Review.
