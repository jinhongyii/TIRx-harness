---
orphan: true
---

# Synccheck explorer: spec of today's algorithm and the plan for the rewrite

Status: draft, 2026-10-07, branch `refactor/clean-core`, worker W6.
Contract: [`numsim-redesign.md`](numsim-redesign.md) §2.1 (Synccheck row), §2.6, §3, §6.
Prototype: `tirx_harness/src/tirx_harness/numsim/core-rs/numsim-sync-explore/`.

This document pins down how today's Synccheck works so the rewrite keeps the
semantics and the pruning, and says what the new explorer needs from the
`SyncEvent` contract. Unless stated otherwise, paths are relative to
`tirx_harness/src/tirx_harness/numsim/engine-rs/src/`. Short names used below:

| short | file |
| --- | --- |
| `check` | `native_analysis/synccheck/sync_check.rs` |
| `unified` | `native_analysis/synccheck/sync_fixed_unified.rs` |
| `verifier` | `native_analysis/synccheck/sync_fixed_verifier.rs` |
| `po` | `native_analysis/synccheck/sync_partial_order.rs` |
| `py` | `native_analysis/synccheck/sync_check_python.rs` |
| `causal` | `native_analysis/sync_causality.rs` |
| `log` | `resolved_transition.rs` |
| `search` | `analysis_search.rs` |
| `setmax` | `native_analysis/synccheck/setmaxnreg_verifier.rs` |
| `strict` | `native_analysis/synccheck/strict_mbarrier.rs` |

---

## 1. Two phases

Synccheck runs the kernel once on the CPU (Phase A, online) and then checks the
synchronization program that run produced for every other interleaving
(Phase B, offline). Driver: `run_native_sync_check_phase` (`py:103-196`).

### 1.1 What is "fixed"

For one invocation (inputs, launch shape), Phase B fixes everything the
concrete run decided (`docs/components/tools.md` "Guarantee", and the
`FixedSyncLogSnapshot` fields at `log:1456-1468`):

* **Each warp's executed path.** Sync operations per warp are ordered by
  `DynamicOpId::per_warp_sequence` (`unified:2196-2210`). Data-dependent
  branches, loop trip counts, lane masks, `elect_sync` winners, and computed
  barrier addresses are already resolved, so a command carries concrete
  resource ids and counts.
* **Values.** Arrival counts, transaction bytes, requested parities, named
  barrier expected counts, setmaxnreg targets, and TCGEN columns come from the
  run.
* **Causal annotations.** Each operation has an `initial_clock`
  (at issue/registration) and a committed `clock` (after acquire)
  (`log:1448-1453`, `log:1482-1503`). Each operation has a
  `canonical_generation` when it touches exactly one generation
  (`unified:3143-3202`). Conditional (`try_wait` returned true) completions
  are fixed per barrier and issuer (`log:1462-1463`, used at `unified:788-839`).
  Setmax participants/resolutions and TCGEN allocation results are fixed too
  (`log:1464-1467`).

Free in Phase B: the interleaving of warps, the delivery time of async
completions (`FixedSyncTransition::Complete`), and setmaxnreg pool grants
(`SetmaxGrant`) (`unified:59-64`).

Out of scope ("Limitations" in `tools.md`): other inputs, ordinary-memory
values, atomic return orders, and control flow that depends on a protocol
return value. The last one is enforced statically: the frontend marks a kernel
`fixed_trace_statically_eligible` unless its sync slice has an unknown tile call
(`tirx_harness/frontend-rs/src/emit/module.rs:435-442`,
`frontend-rs/src/analyze/tile_forms/mod.rs:442-451`). An ineligible kernel whose
Phase A is clean is reported as incomplete with
`fixed_sync_state_ineligible` (`py:706-716`).

### 1.2 Phase A (online) findings

`SyncCheckMode` is an engine mode (`check:3444-3786`). Every sync effect is
staged in `before_effect` against a clone of the strict protocol
(`check:1364-1454`), then committed in `after_effect` (`check:1456`), with
causal checks along the way: `apply_mbarrier_commit_causality`
(`check:3155`), `apply_mbarrier_wait_causality` (`check:3285`),
`apply_named_barrier_causality` (`check:3352`), and
`apply_cluster_barrier_causality` (`check:3400`). On a rejection, the engine
records a `SyncCheckFinding` (`check:491-511`) and returns an `EngineError`, so
the offending warp operation fails. Racecheck continues after a finding;
Phase A does not.

Phase A finds:

* Strict protocol errors, with `kind` from `mbarrier_error_kind`
  (`py:2080-2117`), `named_barrier_error_kind` (`py:2119-2145`), and
  `cluster_barrier_error_kind` (`py:2054-2078`). Examples:
  `mbarrier_use_before_init`, `mbarrier_arrival_overflow`,
  `mbarrier_arrive_before_consumption`, `mbarrier_transaction_over_delivery`,
  `named_barrier_contract_mismatch`, and `full_cta_aligned_control`
  (the terminal check at `check:1231-1247`).
* Happens-before errors in the observed run, with `kind` from
  `causality_error_kind` (`py:2000-2052`). The main ones are
  `mbarrier_init_not_happens_before_use` (`causal:1143-1161`) and
  `mbarrier_prior_generation_consumption_not_happens_before`
  (`causal:1163-1235`). The tracker records only causal evidence. CPU order
  alone is never accepted (`causal:1-6`, test `causal:1386-1406`).
* Execution errors from the engine: the executor `deadlock` (with
  `blocked_operations` and `stalled_operations`, `py:1730-1843`),
  `synchronization_contract_mismatch`, `warp_collective_divergence`,
  `setmaxnreg_missing_warpgroup_sync`, `setmaxnreg_pool_deadlock`, and similar.
  `engine_error_kind` names them.
* Incomplete reasons, listed in §2.8.

Phase B runs only when Phase A is clean and the launch succeeded
(`py:151-166`). `fixed_verification_required` (`py:257-258`) states the same
condition.

### 1.3 Phase B (offline) findings

`verify_fixed_sync_programs` (`verifier:240-377`) produces:

* `FixedSyncVerificationError::Protocol`: a strict step failed in some
  schedule, or a certificate refuted the program (`verifier:72-79`). Payload
  kind: `fixed_sync_protocol_error`.
* `Deadlock`: no transition is enabled and the program is not complete
  (`verifier:80-85`). Payload kind: `deadlock`, with `verification: "fixed_sync"`.
* `NonConfluent`: more than one distinct terminal state (`verifier:86-91`).
  Payload kind: `fixed_sync_nonconfluent`.
* `FixedSyncVerificationIncomplete`, listed in §2.8.

---

## 2. Phase B algorithm

### 2.1 Log to per-warp command sequences

1. **Snapshot.** `with_fixed_sync_snapshot` (`log:1886-2108`) locks the log
   and exposes `operations_with_clocks()` (`log:1482-1503`). Each operation
   yields `(DynamicOpId, ResolvedTransitionSummary, initial_clock, clock)`. It
   also exposes completion generations by action id and the side tables.
2. **Collectives first.** `build_setmax_commands` (`unified:3579`) and
   `build_tcgen_commands` (`unified:3818`) merge the per-warp records of one
   collective (one setmaxnreg request, one TCGEN lifecycle op) into a single
   `FixedSyncCommand` with several `participants`
   (`unified:67-74`, `unified:697-712`). The command's clocks are the join of
   the participants' clocks (`merge_command_clock`, `unified:3204-3218`).
3. **Everything else.** `stage_noncollective_operations` (`unified:2723-2758`)
   turns each summary into one `FixedSyncCommandKind` through
   `command_from_summary` (`unified:3327`). The kinds are in `unified:118-179`:
   init, init fence, inval, arrive, expect_tx, wait, wait batch, completion
   issue, named arrive/sync, cluster arrive/wait, setmax, and TCGEN. This step
   runs on up to 8 threads (`unified:29`, `unified:730-779`).
4. **Per-warp programs.** In each projection, `build_projection` sorts every
   participant's commands by `per_warp_sequence`. The result is
   `warp_programs[warp] = [CommandId…]` and the reverse map
   `warp_command_positions` (`unified:2196-2296`, `unified:2367-2383`).
   `validate_collective_placement` (`unified:2304-2364`) checks that every
   collective appears exactly once in each participant's program.

### 2.2 Per-resource projection and the "unified" state

**Keys.** `FixedSyncProjectionKey` (`unified:554-566`):

| key | resources in one projection |
| --- | --- |
| `Mbarrier(anchor)` | one mbarrier, or every barrier joined by one multi-barrier wait (`mbarrier_component_anchors`, `unified:2638-2687`) |
| `NamedBarrier(id)` | one named barrier of one CTA |
| `ClusterBarrier(id)` | one cluster barrier |
| `SetmaxnregPool{kernel, cta}` | one CTA register pool |
| `TcgenCtaComponent{kernel, anchor}` | CTAs connected by 2-CTA TCGEN ops (`tcgen_component_anchors`, `unified:2595-2636`) |

`stage_fixed_sync_command` (`unified:2769-2838`) splits a command that touches
several anchors (for example a multi-barrier `mbarrier.init`) into one projected
command per anchor (`for_each_projected_command_kind`, `unified:2880-3131`). A
wait over several barriers is one atomic blocking command, so those barriers
stay in one projection.

**Happens-before gating.** Projection hides cross-resource ordering. Today's
code puts it back as `causal_predecessors`. Command `d` gates command `c` when
both are in the projection and `d.clock` (committed) strictly happens-before
`c.initial_clock` (`build_causal_predecessors`, `unified:2696-2721`).
`command_ready` requires every gating command to be completed: each
participant's cursor must be past it (`unified:4019-4052`). Readiness uses
the clock at issue, not the acquired clock, because a blocking wait's
committed clock already includes the operation that woke it
(`unified:2701-2704`).

**The "unified" state.** `FixedSyncState` (`unified:239-335`) holds every
protocol at once: mbarriers, named, cluster, setmax pools, pending
completions, TCGEN CTAs, blocked warps, and cluster waits. The same
`FixedSyncProgram` transition system can therefore explore one projection or
a whole multi-protocol program. Tests build whole programs through
`direct_program` (`unified:5499-5558`).
`cross_protocol_cycle_is_a_deadlock_not_a_false_clean` (`unified:5883-5964`)
pins that a named/cluster cycle is reported as a deadlock.

**How cross-resource cycles are still caught when production only explores
projections:**

1. A cycle that exists in every schedule makes the concrete Phase A run
   deadlock, so it is reported as an engine `deadlock` before Phase B.
2. A cycle that depends on the schedule requires some resource whose
   generation assignment depends on the schedule. That resource's own
   projection reports it as a certificate refutation, a protocol error, or
   non-confluence.
3. The argument is assume-guarantee. Each projection assumes the other
   resources behave as committed, which only adds HB gates, and each
   projection proves its own resource behaves as committed. Phase B stops at
   the first failing projection (`verifier:270-285`, `verifier:347-351`).
4. Commands that change several resources atomically are never split across
   projections: wait batches, multi-CTA TCGEN, and collectives.

The prototype tests this argument directly. On 3000 random and structured
logs, the gated per-resource verdict equals the ungated whole-program verdict
(`numsim-sync-explore/tests/equivalence.rs`).

**Projections that can only be certified.** For named and cluster projections,
and for plain mbarrier projections (no transactions, completions, expect_tx,
drop, wait batch, or inval), `build_projection` builds a skeleton with empty
`warp_programs` (`uses_linear_causal_certificate`, `unified:2131-2187`). These
projections must be decided by a certificate. A named-barrier projection is
never state-searched (`unified:5145-5148`).

### 2.3 Causal certificates

`verify_program_causally` (`verifier:449-498`) tries the mbarrier certificate,
then the named certificate, then the cluster certificate. `None` means not
applicable. In that case the projection goes to fingerprinting and state
search. When a certificate applies, its stats are fixed:
`visited_states = 1`, `explored_transitions = command_count`
(`verifier:454-460`). On success, the search is skipped. On failure, the
certificate's error is reported directly with `transition = ValidateExit` and
an empty witness. Incomplete details become `ProgramModel` incompletes.

Every certificate uses the same counting and vector-clock argument:

* **Counting.** Within one generation, contributions only add to counters, so
  they commute. If every generation's totals are exact, every schedule with the
  same generation assignment goes through the same protocol states.
* **Assignment.** A schedule can only move a contribution to another
  generation by overtaking. HB conditions between consecutive generations rule
  that out for every schedule that respects HB.

**Named barrier** (`verify_named_barrier_causally`, `unified:1043-1296`)
applies to every `NamedBarrier` projection:

* Every command has a `canonical_generation` and an `initial_causal_clock`.
  If either is missing, the result is incomplete.
* Generations are contiguous from 0. A gap is incomplete.
* Within a generation, `expected_arrivals` is constant (else
  `contract mismatch`). Lane masks are disjoint per
  `(warp, Arrive|Sync)`, and the arrival total equals `expected_arrivals`
  exactly.
* If a generation has an aligned blocking sync, then every blocking sync in it
  is aligned. Within one warp, all of them share one static instruction
  (`same_static_instruction`). Different warps may use different inlined sites.
* The release of generation `g` is the join of the initial clocks of all its
  contributions. It must happen-before every contribution to `g + 1`
  (`unified:1259-1279`).

**Cluster barrier** (`verify_cluster_barrier_causally`, `unified:1298-1554`)
applies to every `ClusterBarrier` projection:

* Arrivals are full-warp (partial is incomplete), and the participant
  contract is constant without duplicates.
* Each generation has exactly one arrival from every participant. A missing
  participant is incomplete
  ("exit-aware membership is not modeled", `unified:1462-1480`).
* A warp's wait in `g` is program-ordered after its arrival in `g`. Its
  arrival in `g + 1` is program-ordered after its wait in `g`. Violations are
  `CausalProtocol`. An arrival with no prior wait is incomplete.
* Clocks are not needed here. A wait blocks until all arrivals exist, so
  program order is enough (doc comment `unified:1289-1296`).

**Mbarrier** (`verify_mbarrier_causally`, `unified:1556-2069`) returns `None`
for `WaitBatch`, `Invalidate`, or an arrive with `drop`. Otherwise:

* Exactly one `init`. Zero or several inits is incomplete.
* Every init fence is HB-after the init.
* Every other command is HB-after the init (`strict_happens_before`:
  `≤` and not equal, `unified:2558-2560`).
* Per generation, it accumulates arrivals, pending-arrival raises, expected
  and completed transaction bytes, mutations, and waits. Non-conditional waits
  must request parity `g & 1`.
* Generations are contiguous. If a generation is waited on or followed by
  another, it must be exact: `arrivals == expected + pending raises` and
  `completed == expected tx`. Over-arrival is always an error.
* A generation with a successor must have at least one consuming wait.
* Every mutation of `g > 0` that requires consumption (all except
  transaction-only completion issues, `unified:639-651`) must be HB-after some
  wait of `g - 1`. This is the "arrive/expect_tx before consumption" check.
* Every wait of `g` must be HB-before some mutation of `g + 1` (the "phase
  lap" check: the wait cannot be overtaken by the next completion,
  `unified:1997-2066`). For conditional waits, the successor generation comes
  from `conditional_wait_successor` (`unified:4100-4132`).

> **Found while prototyping:** the overtaking check also fires when `g + 1` is
> a terminal generation that never completes. In that case nothing can
> overtake the wait, and the exhaustive search accepts the program. The
> prototype only applies the check when `g + 1` completes
> (`certificate.rs`, found by `tests/equivalence.rs`).

### 2.4 Fingerprint dedup

The planning loop is `verifier:287-315`. It runs on projections that no
certificate decided:

* `mbarrier_state_search_fingerprint` (`unified:883-991`) is defined only for
  `Mbarrier` projections with no setmax/TCGEN state. It hashes the command
  count and, for each command, the witness warp, the participants, and the
  kind with counts, parities, and conditional lifetimes. Warps are renamed to
  their projection-local index. It also hashes `warp_programs` and
  `causal_predecessors`.
* A fingerprint match is confirmed by `has_equivalent_mbarrier_state_search`
  (`unified:993-1041`, per command at `unified:2385-2556`), which guards
  against hash collisions.
* Only the first projection of each class is searched. The others report
  `programs: 1, reused_clean_programs: 1` and no states (`verifier:331-345`).
  Reuse is only valid after a clean primary, and the code asserts this
  (`verifier:332-335`). Since search stops at the first failure, a failing
  primary is never followed by a reuse.

### 2.5 Explicit-state DFS

**Trait** `SyncTransitionSystem` (`po:12-92`): `State: Clone + Eq + Hash`,
`initial_state`, `enabled_transitions`, `step` (returns `Result`),
`is_complete`, `describe_deadlock`, and three reduction hooks:
`persistent_transition`, `strong_diamond`, `commutes`/`successors_commute`.

**Concrete model** (`impl SyncTransitionSystem for FixedSyncProgram`,
`unified:5133-5461`):

* Transitions are `Issue(cmd)`, `Complete(completion_id)`, `SetmaxGrant`, and
  `ValidateExit` (`unified:59-64`).
* `Issue(cmd)` is enabled when the command is the head of every participant,
  no participant is blocked, its causal predecessors are completed, and the
  kind-specific readiness holds (`issue_ready`, `unified:4054-4089`):
  * a conditional wait needs its committed completion to be visible;
  * a setmax request needs no pending increase in its warpgroup;
  * a TCGEN dealloc needs quiescence.
* Every pending completion is enabled. Setmax grants come from
  `pool.enabled_grants()`, gated by `command_ready`. `ValidateExit` is enabled
  only when nothing else is, all warps are finished, there are no pending
  completions, and the pools are quiescent (`unified:5157-5201`).
* A blocking wait registers and blocks the warp
  (`StrictMbarrierWaitOutcome::Registered`, `unified:4536-4569`). A later
  arrive or completion wakes it (`wake_*_waiters`, `unified:4341-4460`).
* `complete_mbarrier` applies a deferred transaction or arrival to the
  generation captured at issue (`unified:4989-5044`).
* `validate_exit` (`unified:5075-5131`):
  * a cluster generation that is still incomplete is `Incomplete`;
  * live TCGEN allocations are a `Protocol` error;
  * otherwise it sets `exit_validated`.
* `is_complete` returns `exit_validated` (`unified:5216-5218`).

**Deadlock** means `enabled_transitions` is empty in a state that is not
complete (`po:30-32`). The description lists the unfinished warps, the blocked
warps, pending setmax increases, the projection domain and its semantic
protocol state, and for each unfinished head its unmet causal predecessors
(`unified:5220-5302`).

**State hashing.** Equality and hashing (`unified:300-335`) cover cursors,
blocked warps, cluster waits, setmax pools, and TCGEN CTAs. They also cover
the semantic keys of pending completions and of each protocol
(`StrictMbarrierSemanticState`, `strict:711-732`: counts, phase, the arrived
warp set, waiters, and buffered transactions). Witnesses, provenance, and
completion tokens are excluded, so equal protocol situations reached through
different histories merge. A complete state is "the" terminal protocol state.

**Search** (`explore_sync_states`, `po:234-364`, and the sleep-set variant
`po:442-640`):

* DFS with an explicit stack and a node arena with parent pointers for
  witnesses (`po:197-229`).
* Visited set, plain mode: `HashMap<Arc<State>, id>`.
* Visited set, sleep-set mode: `HashMap<Arc<State>, Vec<sleep set>>`, an
  antichain of subset-minimal sleep sets (`register_sleep_context`,
  `po:390-418`). A state is re-entered only with a sleep set that is not a
  superset of a recorded one. A stale node whose context was superseded is
  skipped (`po:468-471`).
* Enabled transitions are sorted and deduplicated. All one-step successors
  are computed once per state (`po:534-540`).
* **Persistent set** (`po:505-514`) is tried only when the inherited sleep
  set is empty. If the model returns a persistent transition, it is the only
  one explored. The model rule (`unified:5304-5380`) is that a pending
  *transaction* completion is persistent when:
  * its barrier has no conditional lifetimes;
  * no other pending completion targets a different generation or kind; and
  * no un-issued command can mutate the barrier or waits on another phase or
    generation.
* **Strong diamonds** (`po:515-533`): when every pair of enabled transitions
  is a strong diamond, only the first non-sleeping transition is explored, and
  `strong_diamond_pruned_transitions += active - 1`. The pair `(l, r)` is a
  strong diamond when both steps succeed, each first step leaves exactly the
  other enabled transitions enabled, and `l;r == r;l` (`unified:5382-5423`).
* **Sleep sets** (`po:542-625`): after exploring `t`, it is added to the
  branch sleep set. A child inherits the sleeping transitions that are still
  enabled after `t` and commute with `t`. Commutation means both orders
  succeed and reach equal states (`successors_commute`, `unified:5444-5460`).
* Limits: if visited states would exceed `max_states`, the search stops with
  `StateLimit`. If explored transitions would exceed `max_transitions`, it
  stops with `TransitionLimit` (`po:554-593`).
* A failure records `SyncStateFailure::{Error, Deadlock}` with its witness.
  With `stop_on_first_failure` (always on in production), the termination
  is `FirstFailure`.
* Complete states go into a `HashSet`, keeping up to two witnesses.
  **Non-confluence** means the search was exhausted with
  `complete_states != 1` (`po:171-178`, `verifier:573-586`).

Production options: `stop_on_first_failure`, `reduce_all_strong_diamonds`, and
`reduce_sleep_sets`, all `true` (`verifier:504-512`).

The verifier then maps the first failure (`verifier:523-605`):

* an `Error` with `is_incomplete()` becomes a `ProgramModel` incomplete;
* any other `Error` becomes `Protocol`, with a replayable witness;
* a `Deadlock` becomes `Deadlock`.

If there is no failure, termination maps to `NonConfluent`, `StateLimit`,
`TransitionLimit`, `FirstFailureStop` (failure not retained), or clean.

### 2.6 Budgets and coverage

* `SyncStateSearchLimits { max_states, max_transitions }` defaults to
  1,000,000 / 10,000,000 (`po:95-107`). Production sets
  `max_states = ResourceLimits.max_backtrack_nodes` and
  `max_transitions = ResourceLimits.max_loop_steps` (`py:71-95`).
* `ResourceLimits` (`search:55-127`) has `max_schedules`,
  `max_backtrack_nodes`, `max_events_per_run`, `max_total_events`,
  `max_loop_steps`, `max_wall_time`, and `max_diagnostic_bytes`. Every limit
  must be positive (`py:58-70`).
* Usage is charged as follows (`py:198-221`):
  * `schedules = 1`;
  * `backtrack_nodes` = visited states;
  * `events` = recorded effects plus fixed transitions;
  * `loop_steps` = fixed transitions;
  * `wall_time` counts only Phase B (Phase A is a completed prerequisite);
  * `diagnostic_bytes` comes from the payload plan.
* `first_exceeded_resource_limit` (`py:526-547`) checks the limits in the
  order schedules, backtrack, events/run, total events, loop steps, wall time,
  bytes. A state limit maps to `BacktrackNodes` and a transition limit maps to
  `LoopSteps` (`py:304-319`). Any hit makes the result incomplete and sets
  `termination = ResourceLimit`.
* `CoverageBounds { max_warp_preemptions, max_completion_schedule_deviations }`
  (`search:3-23`) is accepted and echoed. The fixed search does not use it:
  `maximum_observed_usage` is always the default (`py:363`). It is left
  over from the removed replay explorer, and so are `run_count = 1`,
  `backtrack_count = 0`, `sleep_pruned_branch_count = 0`, and `runs[0]`
  (`py:1061-1111`).
* Verdict (`py:398-407`):
  * `error` if Phase A has an error, an execution error is not incomplete, or
    Phase B has an error;
  * otherwise `incomplete` if only a subset of warps ran or anything is
    incomplete;
  * otherwise `clean`.

### 2.7 setmaxnreg verifier

`SetmaxnregVerifierCore` (`setmax:15-30`) is a pure pool per
`(kernel, cta)`. It has a capacity, an available count, a current count per
warpgroup, and pending increases. It has no waiters and no side effects.

* `apply_request` (`setmax:320`): a decrease releases registers. An increase
  is applied immediately if the pool has room. Otherwise it stays pending with
  its `required_count`.
* `enabled_grants` (`setmax:285`) lists pending increases that now fit.
  `apply_grant` (`setmax:373`) applies one.
* A pool is quiescent when nothing is pending (`setmax:308`).

In the explorer:

* the pool is in the state (`unified:246`);
* `Setmax` commands are collectives over the warpgroup;
* issue is blocked while the warpgroup has a pending increase
  (`unified:4076-4080`);
* grants are separate transitions (`grant_setmax`, `unified:5046-5073`);
* exit requires quiescent pools.

A stalled pool shows up as a deadlock that lists `pending_setmaxnreg`. Phase A
also reports `setmaxnreg_pool_deadlock`. That deadlock is downgraded to
incomplete when only part of the scheduling domain ran (`py:1953-1977`).

### 2.8 `incomplete` reasons to preserve

Every reason is serialized with `kind = "analysis_incomplete"` and a `reason`
string.

| source | reason string | payload extras | cite |
| --- | --- | --- | --- |
| `SyncCheckIncompleteReason::AnalysisGap` (TCGEN mma/shift) | `tcgen_protocol_unmodeled` | `operation`, `domain`, `effect` | `py:1384-1399` |
| `AnalysisGap` (cluster) / `ClusterBarrierUnalignedUnmodeled` | `cluster_barrier_unaligned_unmodeled` | `operation`, `effect` | `py:1392`, `py:1436-1440` |
| `AnalysisGap` (atomic) | `atomic_lane_serialization_unmodeled` | `operation`, `domain`, `effect` | `py:1393` |
| `CompletionActionUnobserved` | `completion_action_unobserved` | `action_id`, `barrier`, `generation` | `py:1401-1410` |
| `CompletionTransitionUnobserved` | `completion_transition_unobserved` | `operation`, `barrier`, `generation` | `py:1411-1420` |
| `EffectCommitUnobserved` | `effect_commit_unobserved` | `operation`, `effect` | `py:1421-1425` |
| `ClusterBarrierParticipantExitUnmodeled` | `cluster_barrier_warp_exit_unmodeled` | `kernel_index`, `cluster_id`, `generation`, `missing_warps` | `py:1426-1435` |
| `ClusterBarrierRearrivalWithoutWaitUnmodeled` | `cluster_barrier_rearrival_without_wait_unmodeled` | `operation`, `kernel_index`, `cluster_id`, `generation`, `warp_id`, `verification: "[VERIFY]"` | `py:1441-1453` |
| engine poll limit | `resource_limit`, `resource: "polls"` | `limit`, `pending_warps` | `py:1902-1910` |
| engine native loop limit | `resource_limit`, `resource: "native_loop_iterations"` | `loop`, `limit` | `py:1894-1900` |
| engine `AnalysisIncomplete{kind}` | `<kind>` | `effect`, `message`, `operation` | `py:1911-1920` |
| subset launch | `subset_execution` | `selected_warp_count`, `total_warp_count` | `py:1356-1363` |
| static ineligibility | `fixed_sync_state_ineligible` | `message` | `py:706-716` |
| verification missing | `fixed_sync_state_verification_missing` | `message` | `py:717-727` |
| `FixedSyncVerificationIncomplete::ProgramBuild` | `fixed_sync_program_build` | `source`, `message` | `py:861-864` |
| `ProgramModel` | `fixed_sync_program_model_incomplete` | `operation`, `transition`, `source`, `witness` | `py:865-884` |
| `StateLimit` | `resource_limit`, `resource: "fixed_sync_states"` | `operation`, `limit` | `py:885-893` |
| `TransitionLimit` | `resource_limit`, `resource: "fixed_sync_transitions"` | `operation`, `limit` | `py:894-902` |
| `FirstFailureStop` | `fixed_sync_first_failure_unretained` | `operation` | `py:903-909` |
| generic coverage limit | `resource_limit`, `resource ∈ {schedules, backtrack_nodes, events_per_run, total_events, loop_steps, wall_time, diagnostic_bytes}` | `limit`/`usage` as `{kind: count \| milliseconds, value}` | `py:695-704`, `py:1232-1260` |

After a terminal error, the three `*_unobserved` reasons are suppressed
(`py:1367-1374`). `ProgramModel` sources include the model's own `Incomplete`
errors, for example the cluster exit-membership incomplete at
`unified:5093-5110` and the certificate incompletes in §2.3.

---

## 3. Operation-count arguments for each pruning

The numbers come from the prototype's bench, `cargo bench --bench pipeline`.
The benchmark is a ring of 1 producer + 15 consumers, 4 stages, 32 iterations:
1,020 events, budget 100k states. The TMA variant has 1,076 events.

| pruning | what it saves | when it applies | count argument | measured (16x4x32) |
| --- | --- | --- | --- | --- |
| **Per-resource projection + HB gates** | The cross product of independent resources' states, and the interleavings of commands on other resources | Always. Commands that change several resources atomically are joined into one projection | Whole program: about the product of per-warp cursor offsets inside the K-stage window, exponential in the number of consumers. With projection, each resource keeps only its own commands. HB gates collapse rounds, so generation `g + 1` cannot start before `g` is consumed | whole + every reduction: >100k states (budget hit). Per-resource + diamonds: 1,062 states, clean |
| **Warp/resource components** (prototype, plan §2.6) | Interleavings between disconnected subsystems | Only when warps and resources really partition | Sound without clocks, but a pipeline is one component, so this saves nothing here | Same as whole: >100k |
| **Strong diamonds** | `n` independent ready transitions (consumer waits on a completed phase, arrivals before the last one) go from `2^n` states to `n + 1` | Every pair of enabled transitions commutes, and no first step exposes or disables another transition | A generation with C ready consumers takes C + 1 states instead of 2^C | per-resource: plain >100k, diamond-only 1,062 |
| **Sleep sets** | Re-exploring commuting transitions: `n·2^(n-1)` transitions become `2^n − 1`. **States are not reduced** | Whenever two enabled transitions commute (the state-local check) | Each state is still visited (unit test: 1,024 states, 5,120 → 1,023 transitions) | per-resource sleep-only: >100k states. "16 warps may break sleep sets" (plan §6) holds |
| **Persistent transition** | Orders of a terminal TMA completion against waiters and unrelated work | A pending transaction completion whose barrier has no conflicting future command (§2.5) | W waiters × completion: the completion moves to the front, so waiters see a ready phase and become a diamond chain | TMA-many-waiters test (16 warps): ≤ 32 + 18 states (`tests/scenarios.rs`) |
| **Fingerprint dedup** | Searching K isomorphic stage projections | Single-resource projections with equal structure under warp renaming (today: mbarrier only) | full[s] and empty[s] are isomorphic for s ∈ 0..K. Only 2 of the 2K projections are searched, which saves (K − 1)/K | 1,062 → 279 states, 6 of 9 projections reused |
| **Causal certificates** | All state search for a resource | named and cluster: always. mbarrier: one init, no wait batch, inval, or drop | One pass over the commands plus HB checks between adjacent generations: O(n·W) for named, O(Σ_g waits_g·mutations_{g+1}·W) for mbarrier. `visited_states = 1` per projection by definition | 9 projections → 9 states, 1.4 ms |

Wall time on the build host for the same configurations, in the same order:
>700 ms (budget), 13 ms, 4.4 ms, 1.4 ms. The TMA variant: 1,126 → 295 → 9
states, with 1.5 ms using certificates.

At 4 warps × 2 stages × 8 iterations (70 events) everything fits:

| config | states | transitions |
| --- | --- | --- |
| whole plain | 1,113 | 2,943 |
| whole sleep | 1,113 | 1,112 |
| whole all | 1,102 | 1,101 |
| per-resource plain | 155 | 247 |
| per-resource diamond | 80 | 75 |
| per-resource + fingerprint | 43 | 40 |
| per-resource + certificates | 5 | 70 (= commands) |

**Plan §6 risk.** A 16-warp K-stage pipeline does break sleep sets alone.
Today's production code survives it only because strong diamonds, the
fingerprint, and above all certificates decide those projections first. The
rewrite must ship strong diamonds and certificates together with the DFS, not
"later when the corpus needs it". Certificates are what make the realistic
kernels O(n).

---

## 4. Report payload to preserve

`build_native_sync_check_phase_dict` (`py:441-524`) and
`build_native_sync_check_phase_result` (`py:223-439`) write these keys:

* **Top level:**
  * `schema_version: 3` (pinned by `test_native_synccheck_artifact.py:424`);
  * `execution_model: "direct_fixed_sync_state"`;
  * `phase: {index, name, topology: {clusters, ctas_per_cluster, warps_per_cta, warp_count}}`;
  * `analysis_scope: {kind: "full_launch" | "subset", selected_warp_count, total_warp_count}`;
  * `findings`, `incomplete`;
  * `effects` (full journal) or `effect_summary` (bounded);
  * `stats` (executor), `execution_error`, `resource_limits`
    (`max_polls`, `max_transitions`, `native_loop_iteration_budget`,
    `native_loop_reschedule_quantum`);
  * `timing: {execution_wall_time_us, fixed_verification_wall_time_us, total_wall_time_us}`;
  * `verdict: "clean" | "incomplete" | "error"`;
  * `coverage`, `search`, `counterexample: None`, `replay_resource_limits`.
* **`search`** (`py:1061-1111`):
  * `algorithm: "fixed_sync_state"` (pinned in 11 places);
  * `run_count: 1`, `backtrack_count: 0`, `sleep_pruned_branch_count: 0`;
  * `program_count`, `reused_clean_program_count`, `visited_state_count`,
    `explored_transition_count`, `strong_diamond_pruned_transition_count`;
  * `incomplete_reason`;
  * `runs: [{prefix: [], warp_preemption_bound: 0, trace_digest: None, trace_digest_hex: None, coverage_usage, status: "finding" | "complete" | "incomplete"}]`.
* **`coverage`** (`py:1113-1151`):
  * `status: "complete_within_bounds" | "finding" | "incomplete"`;
  * `eligible_for_clean`;
  * `bounds: {max_warp_preemptions, max_completion_schedule_deviations}`;
  * `maximum_observed_usage: {warp_preemptions, completion_schedule_deviations}`;
  * `resource_limits: {max_schedules, max_backtrack_nodes, max_events_per_run, max_total_events, max_loop_steps, max_wall_time_ms, max_diagnostic_bytes}`;
  * `resource_usage: {schedules, backtrack_nodes, events_in_current_run, total_events, loop_steps, wall_time_ms, diagnostic_bytes}`;
  * `pending_work_items`, `pending_backtracks`;
  * `termination: {kind: "worklist_exhausted" | "finding" | "resource_limit" | "unsupported" | "cancelled", resource_limit: {resource, limit, usage} | None}`.
* **Phase B findings** (`py:739-832`):
  * `fixed_sync_protocol_error` with `message`, `operation`,
    `protocol ∈ {"Mbarrier", "NamedBarrier", "ClusterBarrier", "Setmaxnreg", "TcgenLifecycle", "Internal"}`
    (the Debug names of `FixedSyncProtocolKind`, `unified:337-344`; tests pin
    `"TcgenLifecycle"`), `transition` (Debug), `source` (Display; tests match
    on `"allocation result changed"`), `related_operations`, `witness`
    (Debug strings), and `witness_evidence: [{transition, description, operation}]`;
  * `deadlock` with `verification: "fixed_sync"`, `deadlock` (Debug of
    `FixedSyncDeadlock`), `witness`, and `witness_evidence`;
  * `fixed_sync_nonconfluent` with `complete_states`, `witness` (last),
    `witnesses`, and `witnesses_evidence`.
* **Phase A findings:** `{kind, effect, operation, message, related_operations?}`.
  `kind` is one of the strict/causality kind strings (`py:1979-2145`). `effect`
  is one of the `SyncCheckEffectKind::name()` strings (`check:109-135`:
  `mbarrier.init`, `mbarrier.arrive`, `mbarrier.wait`, `mbarrier.complete_tx`,
  `bar.sync.register`, …).
* **Operation:** `{kernel_index, global_warp_id, per_warp_sequence, source_op_id, loop_frames: [{loop_site_id, iteration_ordinal}]}` (`py:2147-2165`).
  **Barrier:** `{allocation_id, byte_offset, target_global_cta_id}`.

Python tests also pin implementation counts. Plan §3 says these should go:

* `program_count == 1|2|0`, `visited_state_count == 3`, and
  `explored_transition_count == 6` (`test_native_synccheck_artifact.py:499-532`, `1085-1135`);
* `visited_state_count <= 32` and `explored_transition_count <= 64`
  (`test_native_kernel_contracts.py:554-555`).

Keep the key names; drop the exact values from the tests.

---

## 5. The rewrite (W6)

### 5.1 Shape

```
Vec<SyncEvent> ──► Program (per-warp sequences, collectives joined)
               ──► reference run (one complete schedule; per-command clocks + generations)
               ──► projections (per resource + atomic joins; HB gates from reference clocks)
               ──► certificate? ── yes ──► visited 1
               ──► fingerprint seen clean? ── yes ──► reused
               ──► DFS (state hash, sleep sets, strong diamonds, persistent hook)
               ──► findings | incomplete (state/transition limit, model) | clean
```

What changes compared with today:

* **No clocks or generations in the log.** The prototype recomputes them from
  one complete schedule of the whole program (`program::reference_run`). Any
  complete schedule is acceptable, because the certificates and the per-resource
  searches prove the annotations do not depend on the schedule. Phase A's
  online causal tracker (`sync_causality.rs`, about 1.6K lines, plus the clock
  plumbing in `resolved_transition.rs`, about 4K lines) leaves the engine.
* **The reference run is Phase A's replacement.** If it errors or deadlocks,
  that is a finding with a witness schedule. Errors that today come from
  online HB checks (for example `mbarrier_init_not_happens_before_use`) come
  from the certificate instead.
* **Waits block by being disabled.** A blocking wait is enabled only when it
  is ready. There is no waiter registry. This matches the contract's
  `Step::Blocked` retry model (`numsim-core/src/sync/mod.rs`). Named syncs
  still need a blocked flag, because their contribution and their release are
  separate.
* **The explorer owns the strong-diamond and commutation checks.** They only
  need `step` and `enabled`, so the model implements only the state machine
  and the optional `persistent_transition` hook.
* **Fingerprint for every single-resource projection**, not only mbarrier. The
  full encoding is the map key, so no second equivalence pass is needed.

### 5.2 Prototype status (`core-rs/numsim-sync-explore/`)

* `src/event.rs` defines the local `SyncEvent`. `src/protocol.rs` has
  stand-in mbarrier/named/cluster machines. It is a PLACEHOLDER, to be
  swapped for the W3 `step` functions (`numsim-sync-ref` did not exist when
  this was written).
* `src/ts.rs` is the transition system. `src/explore.rs` is the DFS, a port of
  `po`. `src/projection.rs`, `src/certificate.rs`, `src/fingerprint.rs`, and
  `src/check.rs` are the driver and the reasons.
* `tests/scenarios.rs` has 19 scenarios translated from the Python tests:
  * use-before-init, both in program order and across warps without
    publication;
  * `cta_sync` publication;
  * under-arrival deadlock (trailing ninth warp) and arrival overflow;
  * producer lap;
  * the depth-two pipeline and a K-stage ring with and without TMA, both
    clean;
  * wrong parity;
  * the cross-protocol cycle;
  * repeated `cta_sync`;
  * named arrive reuse;
  * cluster exit, which is incomplete;
  * transaction over-delivery;
  * TMA with many waiters;
  * budget exhaustion, which is incomplete;
  * fingerprint reuse;
  * a malformed log.

  Each scenario runs under four configurations, which must agree on the
  verdict.
* `tests/equivalence.rs` runs 3000 random logs. Every reduced or projected
  configuration gives the same verdict as the unreduced whole-program search.
* `benches/pipeline.rs` prints the tables in §3 and runs the criterion
  timings.

Not yet ported:

* setmaxnreg pools, TCGEN lifecycle, async groups;
* wait batches, inval/re-init, `.noinc` pending raises;
* conditional waits (`try_wait` returning true);
* named-barrier lane masks and aligned-site checks;
* the payload serializer.

The prototype's `IncompleteReason` and `Finding` variants already map 1:1 to
the reason and kind strings in §2.8 and §4.

### 5.3 What the explorer needs from the contract `SyncEvent`

Compared with today's draft in `numsim-core/src/observe.rs` (`SyncEvent {actor,
seq, site, resource, lanes, cmd, outcome}`):

1. **Per-actor `seq` over committed commands only.** Synccheck uses `seq` as
   the cursor. A `Blocked` attempt that is retried must not create a second
   program position. One option is to log only `Done`/`Failed` attempts.
   Another is to guarantee that exactly one `Done` follows a run of `Blocked`
   attempts with the same `seq`.
2. **Counts in `cmd`, not derived from `lanes`.** Arrival count, `expect_tx`
   bytes, named `expected`/`count` in threads, and cluster participants. The
   explorer must never re-derive participation from a mask.
3. **A resource identity that is the physical object.** Remote (cluster-mapped)
   mbarrier arrivals must carry the *target* CTA's barrier id. Today this is
   `PhysicalBarrierId.target_global_cta_id`. The draft's
   `ResourceId::Mbarrier{alloc, offset}` is enough only if `alloc` is
   per-CTA.
4. **Async completions bound to an issuing command.** The issuer's
   `(warp, seq)`, the target resource or resources, the bytes per target, and
   the `arrive` flag (`cp.async.mbarrier.arrive`, `tcgen05.commit`). The
   explorer turns each target into one pending `Complete` transition whose
   generation is captured at issue. `AsyncIssue.targets` in the draft has
   this. It also needs the issuer's `seq`.
5. **Multi-resource atomic commands**, either as one event with several
   resources, or with an id that groups them. Cases: lane-varying waits on
   several barriers (today's `MbarrierWaitBatch`), multi-barrier `init`/`inval`,
   2-CTA TCGEN ops.
6. **Collective identity** for setmaxnreg and TCGEN alloc/dealloc/relinquish:
   one id shared by the participating warps' records, plus the participant set.
   The explorer joins the records into one command with several participants.
7. **Static site and loop frames** (`site`) for reporting and for the
   aligned named-barrier "same static instruction" rule.
8. **Conditional successes**: a `Test`/`try_wait` that returned true must
   record which phase it observed. Today these are `conditional_mbarrier_completions`.
   Without them, conditional control is not fixed by the run.
9. **Not needed:** vector clocks, observed generations, outcomes of successful
   steps. The explorer recomputes them. Observed generations are still useful
   as a debug cross-check.

### 5.4 Integration status (phase 2)

The explorer now lives in `numsim-core/src/synccheck/` and runs on the
production `crate::sync::*::step` functions (`backend.rs` is the only file
that names them). `numsim-sync-explore/` only hosts the pipeline bench.

* Entry points: `synccheck::check(&RecordingObserver, &SynccheckConfig) -> report::Report`
  and `synccheck::serialize(&Report) -> serde_json::Value` (today's payload
  keys, §4).
* Phase A: `Protocol` events with `status: Failed` or `BlockedAtExit` are
  reported as-is, with today's strict kinds and effect names (`kinds.rs`).
  Phase B runs only after a clean Phase A.
* Ported protocols: mbarrier (including inval/re-init, `.noinc`,
  `IncPending`, deferred arrivals, multi-target waits, conditional
  `try_wait` successes), named (lane-mask rule from the `step`), cluster,
  setmaxnreg (collective `Set`, grants as transitions), TMEM lifecycle,
  tcgen05 work/commit, and async groups (milestones as completions).
* Arming change: a blocked parity wait that sets mbarrier `armed` is one
  `Arm(resource)` transition, not one transition per waiting warp. Per-warp
  arming broke strong diamonds and made the 16-warp ring exponential in
  the number of consumers.
* Contract gaps: `numsim-core/CONTRACT_REQUESTS.md` W6-1 (non-confluence
  kind, structured payload, `.aligned`, kernel index, `TestState` success
  flag).
* Tests: `numsim-core/tests/synccheck_scenarios.rs` (40),
  `synccheck_equivalence.rs` (1,500 random logs), `synccheck_payload.rs` (3),
  plus the explorer unit tests in `synccheck/explore.rs` (3).

Bench (`cargo bench -p numsim-sync-explore --bench pipeline`), 16 warps × 4
stages × 32 iterations, 1,044 contract events, 100k-state budget:

| config | states | verdict |
| --- | --- | --- |
| whole program, any reductions | >100k | incomplete |
| per-resource, plain or sleep sets only | >100k | incomplete |
| per-resource + strong diamonds | 4,209 | clean |
| + fingerprint | 1,180 (6 of 9 reused) | clean |
| + certificates | 9 (all certified, 10 ms) | clean |

### 5.5 Review fixes (2026-10-08, `checker-review.md`)

Each item has a regression test built from contract events.

| Item | Fix | Regression |
| --- | --- | --- |
| S1(a): a vacuous parity-1 wait was ignored by the mbarrier certificate | The certificate declines (falls back to the search) unless the wait is HB-before a prerequisite of generation 0's completion | `s1a_vacuous_parity_one_wait_is_not_certified` (deadlock found) |
| S1(b): a wait of generation g could pass on g−2 | The certificate declines unless every wait of g ≥ 1 is HB-after a consuming wait of g−1 or a mutation of g. "Overtaken" waits also fall back instead of being reported, because the search decides whether they deadlock or pass later | `s1b_wait_that_can_pass_on_an_older_generation_is_not_certified` (never Clean; gated configurations fail closed) |
| S5: gated DFS projections never checked the reference generations | `Ts` compares every issued command's generations (and captured issue generations) with the reference run. A mismatch is `incomplete`: `fixed_sync_program_model_incomplete`, with source `generation_assignment_differs` | `s5_generation_assignment_is_checked_against_the_reference` (clean, confluent program; the gated search fails closed) |
| S8: strong diamonds checked one step only | New proof obligation `TransitionSystem::independent_of_future`; the default declines. `Ts` grants it only when, on the transition's single resource, every command that can still run first (un-issued commands not HB-gated behind it, pending completions, retries) is an observer or a contributor, and their arrivals cannot complete the open phase. Contributors that may complete it also require that no mbarrier observer of the last completed parity can still run | explorer unit test `one_step_diamonds_need_the_independence_proof` (discriminating) and `s8_strong_diamond_does_not_hide_a_two_step_lap` (end to end) |
| F4: `tcgen05.commit` coupled empty[s], tmem_full and the MMA queue | `TcgenWork` commands are dropped from the explored program: they are total, never block, and carry no state the search needs. Each commit is then a single-resource deferred arrival, and its barrier is certified | `f4_umma_ring_is_certified_per_barrier`: 6 stages × 16 k-blocks × 16 tiles, 15 projections all certified, 15 states (was 320k). With certificates off: 3,994 states |
| TMA events without `Mbarrier(Issue)` were never certified | The certificate reads the projection's commands, including the synthesized `Issue`. `build::issue` no longer injects `Issue` | `tma_event_without_issue_command_is_certified` |
| `SyncEvent.kernel` (launches merged) | `check` on a mixed log is `incomplete` (`fixed_sync_program_build`). `check_launches` / `split_launches` return one `Report` per launch | `launches_are_checked_separately` |
| Duplicate cluster waits were overwritten in the certificate | The certificate declines, and the state machine reports `DuplicateWait` | (covered by the state machine) |
| W2 engine smoke: 32 per-lane cp.async groups were a product (20K states took 16 s; 1M did not finish) | Async-group milestones fire eagerly, in FIFO order, inside the transition that creates the group. Only the issuing thread's `wait_group` observes them. Deferred mbarrier arrivals stay separately schedulable `Complete` transitions gated on the group. Delaying a milestone is indistinguishable from not scheduling the warp | `synccheck_engine.rs`: cp.async scenario Clean in ≤16 states and under 2 s; every interpreter scenario stays within a 20K budget and 5 s |

**Equivalence testing.** The tests now compare against an all-failures exhaustive oracle (`stop_on_first_failure: false`). Each search-based variant must reach the oracle's verdict and report a finding kind the oracle found. The only tolerated difference is a fail-closed `generation_assignment_differs`. There are five generators: random, structured, rich, rich-structured, and lap. The rich generators emit TMA issues without `Issue`, conditional waits, inval/re-init, multi-target waits, tcgen05 alloc/dealloc and commit, cluster barriers, parity-1 first waits and generation-skipping waits.

Mutation checks:
- Re-opening the S1 holes makes the rich generator fail (Clean against Error).
- Disabling the S5 check makes the S5 scenario fail.
- Reverting S8 makes the S8 unit test fail.

**Bench after the fixes.** The sound independence rule keeps the earlier numbers: 16-warp ring 4,209 states with diamonds only, 1,180 with fingerprints, 9 with certificates. The UMMA row has been added (15 states certified).
