---
orphan: true
---

# numsim-core contract review (adversarial, 2026-10-07)

Paths are relative to `tirx_harness/src/tirx_harness/numsim/core-rs/numsim-core/src/`.
"HEAD" means committed lines. "WT" means W2's in-progress handlers, which show
how a worker is already reading the contract. Findings are ranked by the cost of
finding them late. Measured on a HEAD copy: `size_of::<Instr>() = 72`,
`SyncEvent = 248` (plus up to 4 heap `Vec`s), `SyncKind = 192`, `Access = 64`,
`AsyncOp = 152`, `Const = 32`.

## 1. The `shared::cluster` address encoding is not the hardware's (critical)

**Evidence.**
- `arena.rs:505-509, 561-567` encode `(rank+1)<<24 | off` and make a
  `shared::cta` address the bare offset.
- Legacy probes on GB200 found `rank<<24 | off`
  (`engine-rs/src/instruction_codec.rs:17-30`).
- The corpus uses the CUTLASS pair-base idiom
  `cvta_generic_to_shared(p) & 0xFEFF_FFFF`
  (`frontend-rs/src/emit/memory_support.rs:547`, Sm100MmaPeerBitMask). That idiom
  only works if `cvta` yields rank-tagged addresses.

**Failure.** In a 2-CTA tcgen05 kernel, the odd CTA computes `cvta(&full[s]) & ~(1<<24)`.
- Under the contract, the result stays "self", so the odd CTA's TMA completes on
  its own barrier, not the leader's. The leader waits forever, giving a false
  `Deadlock` with status Error.
- A mapa'd address of rank 1, `2<<24`, keeps rank 1 after masking.
- `mapa(p, my_rank) == p` is true on hardware and false here. Self-skip tests in
  all-gather loops take the wrong branch.

**Change.** Adopt the hardware encoding everywhere before W1 or W2 bake it in:
```rust
// shared::cta and shared::cluster share one encoding: rank<<24 | off (off < 2^24).
pub const fn shared_addr(rank: u32, off: u32) -> Option<u32>;   // None if off >= 2^24 or rank > 255
pub const fn decode_shared(a: u32) -> (u32 /*rank*/, u32 /*off*/);
// cvta.to.shared and AddrOf→Cvta tag the executing CTA's rank; .shared::cta ops
// require rank == own (else BadAddress), .shared::cluster ops route by rank.
```
Also add a generic distributed-shared aperture, so that `mapa.u64` and generic
ld/st to a peer's shared memory resolve:
`GENERIC_SHARED_BASE + (rank<<24 | off)`. Today `classify_generic` maps every
shared address to the executing CTA (`arena.rs:540-550`).

## 2. The outbox/inbox breaks commit, atomicity and program order (critical)

**Evidence.**
- `sched/mod.rs:14-16, 164-171` defers cross-CTA effects to the next drain.
- WT `interp/handlers/sync.rs:301-322` commits local targets, pushes remote ones
  to the outbox, and emits **one** `Protocol{status: Committed}` covering both.
  `Arrive` is emitted only for local targets. W6 point 5 and
  `observe.rs:21-23` ("delivered after the transition committed") are both
  violated.
- WT `mem.rs:121-130` defers remote stores, while `load_addr` (`mem.rs:88-100`)
  and `atom` (`mem.rs:234+`) touch the peer's bytes immediately.

**Failures.**
- `st.shared::cluster [peer+x], 1; ld.shared::cluster r, [peer+x]` in one thread
  reads the old value, which is a same-thread coherence violation.
- `st [peer+x]; atom.add [peer+x]`: the store lands after the atom and erases it.
- A multicast arrive whose remote leg overflows at drain was already logged as
  Committed. The error has no owner, and `seq` order across warps is now
  schedule-dependent.
- Racecheck gets no remote `Arrive`, so cross-CTA release edges are missing and
  races are reported falsely. Remote store `Access`es arrive out of per-actor
  order.

**Change.** There is one `Arena` and one `SyncTable` (README decision 10), so
apply remote effects **synchronously at issue** through `step_all` across CTAs,
and delete `InboxMsg::{Write, Sync}` until CTA parallelism exists. If delayed
visibility is wanted as a racecheck feature, model it as an observer-side
publication delay rather than an engine-side reorder.

## 3. `Arrive`/`Wait` cannot carry scope or an unknown qualifier (high)

**Evidence.**
- `observe.rs:265-268` has `release: bool` and `acquire: bool`, with no scope.
- Racecheck R4/R5/R8 are closed: mbarrier edges are scoped, and a lost cluster
  qualifier is `incomplete` (racecheck-semantics.md §10, row 8).
- W5 already hard-codes `.cta` and fails closed (`racecheck/observer.rs:86-89`).
- WT `mbar_tx` discards `scope` (`let _ = scope`).

**Failure.** `cuda.mbarrier_wait_acquire_cluster` (corpus) followed by a peer's
`arrive.release.cluster` is judged as a `.cta`/`.cluster` mismatch, so every
cluster pipeline falls back to `incomplete`.

**Change.**
```rust
Arrive { obj: ResourceId, phase: u64, sem: Option<Sem>, scope: Option<Scope> },
Wait   { obj: ResourceId, phase: u64, sem: Option<Sem>, scope: Option<Scope> },
```
`None` means the qualifier was lost in lowering.

## 4. The `Protocol` event shape contradicts README decision 9 and W6 requirement 1 (high)

**Contradictions.**
- **Generations in the log.** `cmds: Vec<(ResourceId, SyncCmd)>`
  (`observe.rs:250`) ships table-level commands that carry
  schedule-dependent generations: `CompleteTx{gen}`, `DeferredArrive{gen}`,
  `Resume{gen}` and `TestState{gen}` (`sync/mbarrier.rs:147-156`,
  `named.rs:75`). README decision 9 says "no observed generations". W6 §5.1
  says "no generations in the log".
- **Is a blocked step stateless?** `sync/mod.rs:17` and `:70` say a blocked
  step leaves the state unchanged. `:255` says it "keeps bookkeeping (`armed`)".
  `step_all` drops the staged state on Blocked (`:295`). A wait through
  `step_all` never arms, so lane-varying wait batches differ from single
  waits. An armed but undelivered attempt also changes state that the explorer
  never sees.
- **What `seq` numbers.** A named `bar.sync` is `Sync` → `Registered`
  (state changes) and later `Resume`. That is two commits, or one? The handler
  docs (`interp/mod.rs:18-21`) and README decision 9 do not say.
- **Per-event scalars on multi-resource events.** `observed_parity: Option<u8>`
  and `Counts` are per event, but `cmds` is multi-resource. A lane-varying
  `try_wait` on two barriers with different parities, or a multicast arrive
  with different per-CTA counts, cannot be represented.

**Change.** One event per **instruction completion**, with `seq` assigned at
that moment and symbolic commands:
```rust
Protocol { targets: Vec<ProtoTarget>, collective: Option<Collective>, issued: Vec<AsyncTarget>, status }
struct ProtoTarget { res: ResourceId, op: ProtoOp /* Arrive{count,tx,drop,..}|ExpectTx|Init|Inval|WaitParity{parity}|TestParity{parity, ready: bool}|NamedSync{expected,contributed}|... no gens */ }
```
Make "Blocked = no state change" normative. Fix `armed` in W3, or deliver it as
an explicit `ProtoOp::Arm`.

## 5. Per-thread async ops are modelled per warp (high)

**Evidence.** Async accesses use `lane = ALL_LANES` (`observe.rs:102-109`). One
`AsyncId` covers a warp instruction. Async groups are per `(warp, lane)`
(`ResourceId::AsyncGroup`), and per-lane completion is decided (redesign §2.5,
racecheck §10).

**Failure.** Lane 3 runs `cp.async.wait_group 0` while lane 5 has not committed
its group.
- `AsyncComplete{op, Warp{lanes: {3}}}` publishes `A:2`, which covers lane 5's
  destination bytes as well.
- Lane 3 then reads lane 5's destination. In PTX this is a race; racecheck
  reports it as ordered. That is a false negative.

**Change.** Use one `AsyncId` per `(instruction, lane)` for per-thread ops
(cp.async, st.async, cp.async.mbarrier.arrive), or put the lane on async
spans (`LaneSpan.lane` = the issuing lane) and give
`AsyncComplete{lanes}` the meaning "only those lanes' footprint".

## 6. Whole-warp blocking under structured SIMT reports false errors (high)

**Evidence.** `Flow::Blocked` re-runs the same pc for the whole warp
(`interp/mod.rs:15-21`). The else-arm of a divergent `If` cannot run while the
then-arm is blocked. WT `wait_until` requires every active lane to accept in
the **same** poll (`sync.rs:690-695`).

**Failures.**
- `if lane==0: mbarrier_wait(b) else: mbarrier_arrive(b)` passes on hardware
  (Volta+ ITS) but is reported here as `Deadlock` with status Error. "Error
  requires proof" is violated.
- Consider per-lane flags or a toggling phase word where lane A's acceptance is
  later revoked. On hardware, lane A leaves the loop when its condition holds.
  Here the instruction never completes.
- `WaitVerdicts` is a lane-wise conjunction (`observe.rs:282-283`, WT
  `sync.rs:730`). If lane A first accepts write 1 and lane B write 2, both get
  the edge from write 2. That edge is too strong, so races are missed.

**Change.**
- `RunStatus::Deadlock` becomes `Incomplete{reason:"divergent_block"}` when any
  blocked warp is divergent (the frame stack shows it).
- `WaitUntil` latches lanes as they accept (per-lane `dst` write, active mask
  shrinks).
- Verdicts are per lane:
  `WaitVerdicts { word: (AllocId, ByteSpan), lanes: Vec<(u8 /*lane*/, Vec<u64> /*bits*/)>, observed: Vec<(u8,u32)>, .. }`.

## 7. `Program::validate` and serde let corrupt programs run (medium-high)

**Gaps.** `validate` (`program.rs:1517-1654`) checks operands only for
`Mov/Load/Store/If/LoopIf`. It does **not** check:
- that `ty.slots() <= regs[dst].ty.slots()`.
  `write_slot` (`interp/mod.rs:358-362`) writes `slot(r)+i` without a
  per-register bound. A `Binary{ty: F32x8, dst: r5}` with `r5: U32` silently
  overwrites `r6..r8`. Codegen and the interpreter agree, so the differential
  test cannot catch it.
- whether `Else.end_pc`, `LoopIf.end_pc` and `Break` belong to **their own**
  frame. Only the target's variant is checked (`:1585-1614`).
- that predicate ranges come after the main body, and that main code cannot
  fall through into them. `step_warp` falls through (`interp/mod.rs:405`). The
  `in_pred` skip also hides main code that a bad range covers.
- `Ty` invariants. `lanes: 0` deserializes successfully.
- `Const.bits` fitting its `Ty`.
- indices nested in `TmaArgs/BulkCopyArgs/TileArgs`, `StrId`, `LayoutId` and
  `Buf` in `AddrOf`.

**Serde.** There is no `deny_unknown_fields`, and serde treats a missing
`Option` field as `None`. A misspelled `"multicst"` in the Python emitter
therefore silently drops TMA multicast. Negative signed constants fail as
`u128` unless the emitter masks them.

**Change.** Add `#[serde(deny_unknown_fields)]` on every program type, and a
full per-variant `validate`: a reg/const/buf/str/layout visitor, a frame-matched
target check, a `ty` vs register slot check, preds placed after an `Exit` or
`Unsupported` terminator, and `Ty`/`Const` invariants. This is cheap now and
expensive after the corpus JSON exists.

## 8. `report.rs` cannot carry the snapshot keys (medium)

`snapshot.py:263-273` keys groups by category, `kind`, `status`, `space`, the
`_SEMANTIC_FIELDS` (`access_pair`, `ordering_domain`, `ordering_failure`,
`reason`, `cause`; `:249`), and anchors.

**Gaps.**
- `Finding` has no typed home for these fields. Both checkers stuff JSON into
  `Evidence.detail` and use `FindingKind::Other(..)` three times
  (`racecheck/payload.rs:171,207,219`; `synccheck/payload.rs:371`), which the
  enum's doc forbids at merge.
- Incomplete `reason`s collapse into `Unsupported` or `BudgetExhausted`.
- `Space` serializes as `"Tmem"` while `snapshot.py:277` tests
  `startswith("tmem")`. Without a renderer mapping, the column-only TMEM
  normalization silently stops applying and snapshots become
  schedule-dependent.
- `Window` (cta vs cluster) is lost.

**Change.**
```rust
struct Finding { kind: String /*stable snake id*/, status, message, sites,
                 evidence, fields: BTreeMap<String, String> /*access_pair, ordering_*, reason, cause*/ }
#[serde(rename_all = "snake_case")] enum Space { .. }   // "tmem", "shared"
struct Evidence { .., window: Option<Window> }
```

## 9. Performance traps on the hot path (medium)

- **`WaitVerdicts` is quadratic.** For every successful wait, WT
  `sync.rs:726-735` re-runs the predicate over the **whole** history. A
  flag word written and waited once per iteration costs O(H²) predicate
  runs, for example about 10⁸ at 10K iterations. Send
  `earliest_accepted: Option<u32>` and cache verdicts per
  `(PredId, capture values, word)` incrementally.
- **Blocked retries.** Every blocked warp re-runs its handler every round
  (`sched/mod.rs:9-10`). That means operand reads, address resolution, a
  `HashMap` lookup and, for batches, `step_all` clones. `PollState.failed_on`
  holds **one** resource (`interp/mod.rs:75-82`): a loop polling `A || B` parks
  on B only. Once W2 "skips warps whose ResourceId did not change", that loop
  deadlocks falsely. Add `SyncTable::version(ResourceId) -> u64`, park on a
  `SmallVec<[(ResourceId, u64); 2]>`, and give `Word` resources an
  arena-side write version.
- **`RegFile` layout.** Each 64-bit slot is `[u64; 32]` (256 B).
  - A `WarpValue<f32>` read touches 256 B and converts lane by lane.
  - A `float32x8` value is four slots 256 B apart.
  - `write_as`/`write_raw` iterate `mask.lanes()` (`value.rs:158-171`), which
    does not vectorize.

  At minimum, make the masked write a branchless 32-lane blend. If 32-bit slots
  are wanted, decide before W4 and W7 hard-code `u64`.
- **`Operand::Const`** is resolved through `ctx.program.consts` at run time
  (`interp/mod.rs:321-334`). Codegen prints the `ConstId` (`codegen/emit.rs:59`)
  and cannot fold it. Let the printer emit `K{k}_CONSTS[i]` as a literal
  through a `Src::Imm(u128)` handler argument.
- **Span lists.**
  - TMA payloads and footprints are `Vec<(AllocId, ByteSpan)>`
    (`sync/completion.rs:136`, `observe.rs:277`), about 1K entries per swizzled
    box, and they are duplicated in `AsyncIssue` and `Access`.
  - `ByteSpan::coalesce` allocates on every call (`arena.rs:91`).
  - Warp loads build 32 `LaneSpan`s (24 B each) plus a byte buffer, then
    transpose them (`arena.rs:385`).

  Add an affine footprint `{base, lane_stride, len, mask}` / rows form and
  `Arena::read_lanes(view, base, stride, n, mask, &mut [u64; 32])`.
- **Event cost.** `SyncEvent` is 248 B plus `frames`, `cmds`, `issued` and
  `footprint` `Vec`s. `RecordingObserver` clones each one
  (`observe.rs:386`). This is acceptable on the cold path. `Instr` (72 B) is
  fine.

## 10. Smaller items (low, but cheap now)

- **Sub-byte buffers.** `Load/Store` offsets are in elements of the buffer
  dtype, and `mem_bytes` rounds fp4 up to 1 byte (`dtype.rs:73-82`). Shared
  `float4_e2m1fn` buffers (inventory A.3) therefore get byte = element. Define
  offsets in bits for sub-byte dtypes, or reject them.
- **Missing dtypes.** `Dtype` lacks `float8_e3m4` and `uint6` (inventory A.3).
  `boolx128` and `e2m1x32` parameters exceed `MAX_VALUE_BITS = 256`.
- **Pair identity.** `TcgenLifecycle{pair}` uses "the global id of the even CTA"
  (`sync/completion.rs:42`). For cluster dims `[1,2,1]` the peer is
  `ctarank ^ 1`, not `global ^ 1`. Define it by cluster rank.
- **`may_block` misses rendezvous instructions.** `may_block` is false
  (`program.rs:1093-1109`) for instructions that must rendezvous:
  `TcgenDealloc`/`Relinquish` with `cta_group::2`, and `SetMaxNReg{inc:false}`.
  W2 already invents a private rendezvous for them (`interp/aux.rs:207-226`).
  Make the rendezvous contract-level (`Collective.id` = `(site, resource,
  instance)`).
- **`epoch` wraps silently.** `epoch: u32` uses `wrapping_add`
  (`interp/mod.rs:410`), which breaks racecheck's monotone stamps. Use `u64` or
  saturate and fail closed.
- **Silent offset masking.** `shared_cluster` masks offsets of 2^24 or more
  silently (`arena.rs:566`), so overflowing address arithmetic aliases another
  CTA's window. Return `None`.
- **Tensormap acquire.** `FenceEvent::TensormapAcquire` drops the address
  (`observe.rs:234`), so racecheck open item (d) will need a contract change.
  Carry `(AllocId, ByteSpan)` now. TMA's own tensor-map read should be an
  `Access{proxy: TensorMap}`.
- **Declared-word history numbering.** "The i-th delivered write `Access`"
  (`observe.rs:25-28`) is ambiguous when one `Access` has 32 lanes writing the
  word, or when a write partially overlaps it. Number per `(Access, lane)` in
  lane order, using byte-merged post-images.
  consumer that iterates it is nondeterministic. Today's `exit_lints` and
  `quiescent` sort by `Debug` string, which gives lexicographic `CtaId(10) <
  CtaId(2)`. Use `BTreeMap` or a typed sort key.
