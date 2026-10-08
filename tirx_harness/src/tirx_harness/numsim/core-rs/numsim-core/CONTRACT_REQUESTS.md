# Contract requests

Requests against the `numsim-core` contract modules (coordinator-owned).
Append-only; one section per request, prefixed by the requesting worker.

## W8-1: load built codegen step functions

`codegen::build` returns a cdylib path, but nothing turns it into
`sched::Backend::Codegen(Vec<WarpStepFn>)`. numsim-py should not dlopen
engine artifacts itself (ABI/version checks belong next to the printer).
Request: `codegen::load(path: &Path, module: &Module) -> Result<Backend, BuildError>`
(owning the library handle for the backend's lifetime, checking the
`STEP_SYMBOL_PREFIX{i}` symbols for every kernel), or a
`codegen::backend_for(module, opts, cache_dir) -> Result<Backend, BuildError>`
that prints, builds (cached by source hash) and loads. Until then
`numsim_core_py.run(..., backend="codegen")` raises `NotImplementedError`.

## W8-2: kernel index (and memory space) on findings and runtime errors

`report::Finding`/`Evidence` and `interp::ExecError` carry `SiteId`s, but
`SiteId` indexes `Program::sites` of one kernel and a `Module` has several.
`RunStatus` reports "the first kernel that did not complete" without saying
which. The Python layer currently attributes findings to phases by running
every kernel prefix and diffing (O(phases^2) runs) and guesses the kernel
for runtime errors. Request:
- `Evidence.kernel: u32` (or `Finding.kernel`), `ExecError.kernel: u32`,
  and `RunOutcome.failed_kernel: Option<u32>`;
- `Evidence.space: Option<Space>` and `Evidence.alloc: Option<AllocId>`
  next to `bytes` (the conformance snapshots key findings by memory space
  and allocation-relative byte overlap; legacy payloads always carry
  `space`);
- optionally `Report` per launch (`Vec<Report>` or `launch: u32` on
  `Report`) so racecheck/synccheck phases map 1:1.

## W8-3: tensor-map arguments bound against engine addresses

`ArgValue::TensorMap(Vec<u8>)` takes an encoded 128-byte image, but the
global address inside it is the engine's synthetic VA of another argument,
which the host cannot know. Corpus cases pass host-side descriptors
(base buffer + dims/strides/box/swizzle...). Request either
`ArgValue::TensorMapOf { base: String, offset: u64, desc: oplib::TensorMapDesc }`
resolved at bind time, or a documented rule that the scheduler encodes
every `ParamSlot` with `tensor_map: Some(spec)` and no bound value from its
`implicit_base` buffer.

## W8-4: execution subsets and assumptions

Legacy `Engine.run(..., subset=ExecutionSubset(cluster_ids|cta_ids),
assumptions=ExecutionAssumptions)` is used by corpus cases (task-steal
variants, multi-CTA kernels). `RunConfig` has no equivalent. Request a
`RunConfig::subset: Option<Vec<u32>>` (resident cluster ids; others
non-resident, analysis verdict `Incomplete { reason: "subset_execution" }`)
and a place for host assumptions (or a ruling that v2 drops them).

## W8-5: NumSim uninitialized reads as review diagnostics

Legacy NumSim materializes uninitialized bytes as zero and reports one
`uninitialized_read` diagnostic with status `review` per read range (98
corpus cases depend on outputs computed this way; 18 are `review`).
`ValidityPolicy::Error` aborts with `ExecErrorKind::Uninit`; `Allow` is
silent. Request a third policy (`ZeroAndReport`) whose reports surface in
`RunOutcome` (e.g. `RunOutcome.diagnostics: Vec<Finding>` with
`UninitRead`/`Review`, carrying space, alloc, byte span and site), or a
ruling on the intended NumSim behavior.

## W3-1 (2026-10-07): reference shape change, ported

Added to `numsim-sync-ref` and re-ported with `tools/port_sync.py`:
`mbarrier::Error::IncompleteAtExit { gen, outstanding, buffered, tx_count }`,
`mbarrier::quiescent`, `named::quiescent`, `cluster::quiescent`. The last two
always return `Ok`: dangling named generations are a lint (`exit_lint`), and
exit-aware cluster membership completes every generation. Spec:
`sync-semantics.md` §2.8 G4.

**Warning:** `port_sync.py` replaces every `pub fn` with an `unimplemented!`
stub, so re-running it now deletes the W3 bodies. Either teach it to keep
existing bodies of functions whose signature is unchanged, or apply future
type changes to `numsim-core/src/sync/*.rs` by hand. Production-only helpers
live in `sync/query.rs` and in the `//@tail` sections after
`check_invariants`.

## W3-2 (2026-10-07): numsim-core dev-dependencies

`numsim-core/Cargo.toml` `[dev-dependencies]` gains `numsim-sync-ref` (path)
and `proptest = "=1.6.0"` for `tests/sync_differential.rs`. The dev-dependency
on `numsim-sync-ref` stays after W6 drops its temporary normal dependency.

## W3-3 (2026-10-07): kernel-wide tcgen05 `cta_group` resource

PTX 9.4 §9.7.18.7.1 requires one `.cta_group` value for every tcgen05
instruction of a kernel. The rule is `tcgen::KernelState` /
`tcgen::use_cta_group`, but `ResourceId`/`Resource`/`SyncCmd` have no home for
it. Request: `ResourceId::TcgenKernel`, `Resource::TcgenKernel(tcgen::KernelState)`
and `SyncCmd::TcgenGroup(u8)`, with outcome `Outcome::TcgenGroup` or reuse of
`Outcome::Tcgen(Done)`. Every tcgen05 handler (alloc, dealloc, relinquish,
mma, cp, shift, commit) steps it first, in the same `step_all` batch. Until
then the rule is unenforced.

## W3-4 (2026-10-07): `MbarQueryOp::CheckLayout` needs the requested layout

`mbarrier.check_layout` is a predicate: "is this barrier initialized with
layout X". See legacy `mbarrier_check_layout<LAYOUT>`,
`runtime/instructions/sync.rs:521-547`. `MbarQueryOp::CheckLayout { mbar, space }`
has no layout operand. Request: `CheckLayout { mbar, space, layout_v1: bool }`.
Pinned semantics, implemented in `sync::query`:
- `check_layout(state, layout_v1) -> Result<bool, mbarrier::Error>`;
  an uninitialized barrier is `Error::Uninitialized`.
- `pending_count(token)`: only for `.noComplete` state tokens, else error.
  Lane k reads the pending count before its own arrival (ascending-lane
  order).
- `encode` / `generation` cover the state-token layout used by `TestState`.

## W3-5 (2026-10-07): synccheck backend swap is unblocked

All `sync::*::step` / `quiescent` / `work_step` bodies and
`SyncTable::enabled` are implemented and differentially tested against
`numsim-sync-ref` (`tests/sync_differential.rs`). W6 can switch
`synccheck/backend.rs` to `use crate::sync as p;`.

No new `report::FindingKind` variants are needed. Each module has
`finding_kind(&Error)`, and `SyncError::finding_kind` /
`SyncError::is_infrastructure` aggregate them. Exit lints map through
`lint_kind` and are reported with `Status::Review`; see
`SyncTable::exit_lints`. `RuntimeError` marks errors that can only come from an
engine or scheduler fault.

## 2026-10-07 — W7 (codegen)

1. **Panics across the cdylib boundary (sched, W2).** The generated cdylib
   statically links its own std; a Rust panic unwinding from it into the
   host aborts the process ("Rust cannot catch foreign exceptions"), and
   vice versa. Generated step functions therefore catch panics and return
   `StepResult::Error(ExecErrorKind::Internal, "panic: <msg>")`
   (`codegen::rt::guard` / `rt::panic_error`). Request: the scheduler wraps
   the *interpreter* slice the same way (call `codegen::rt::guard(ctx, q,
   interp::step_warp)` or an equivalent in `interp`), so a handler bug
   yields the identical `RunStatus` on both backends. Until then the
   differential test treats "interp panicked with M" == "codegen Internal
   error `panic: M`".
2. **Observers and resolved `PtxFn`s must not panic** when the codegen
   backend is active (a host panic unwinding through generated frames
   aborts). Please state this on `observe::Observer` and `oplib::PtxFn`.
3. **Host allocator.** Heap blocks cross the boundary (Vec growth in
   handlers, error strings), so the host must use the system allocator (no
   `#[global_allocator]` in numsim-py).
4. **Handlers must not write `ctx.warp.pc`** (already in the handlers
   contract); generated code keeps the pc in a local updated only through
   `interp::end_instr`. Thanks for `begin_instr`/`end_instr`/`fall_off_end`
   — the printer calls exactly those.
5. **Shared test programs (W2).** Please expose the programs your interp
   tests build (e.g. `testutil::scenarios() -> Vec<(name, Module, Inputs)>`)
   so `tests/codegen_equivalence.rs` can run all of them on both backends
   without editing your tests. W7's own scenarios live in
   `tests/codegen_scenarios/mod.rs`.
6. **Cargo.toml (coordinator).** W7 added to `numsim-core/Cargo.toml`:
   `libloading = "=0.8.6"`, `sha2 = "=0.10.8"`, dev-dep
   `criterion = "=0.5.1"` (default-features off) and `[[bench]] codegen`
   (harness = false).

## W3-3 done (2026-10-07)

Added `ResourceId::TcgenKernel`, `Resource::TcgenKernel(tcgen::KernelState)`
and `SyncCmd::TcgenGroup(u8)`. A `TcgenGroup` step returns
`Outcome::Tcgen(tcgen::Outcome::Done)` or `SyncError::Tcgen(CtaGroupMismatch | InvalidCtaGroup)`.

`SyncTable::step` and `step_all` stage an implicit `TcgenGroup(who.group())`
before every `SyncCmd::Tcgen` lifecycle command. If the lifecycle command
errors or blocks, the group is not committed. W2's handlers must add
`(ResourceId::TcgenKernel, SyncCmd::TcgenGroup(g))` to the same `step_all`
batch for tcgen05 mma / cp / shift / commit, because `WorkCmd` carries no group.

To keep the build green I added one match arm to each of two W6 files:
- `synccheck/backend.rs`: `ResourceId::TcgenKernel => return None`, so the
  explorer does not model the group yet.
- `synccheck/kinds.rs`: `SyncCmd::TcgenGroup(_) => "tcgen05.cta_group"`.

The differential test `table_tcgen_kernel_group` compares the `SyncTable`
against the reference composition `use_cta_group`, then `tcgen::step`,
committed only on a non-blocked success.

## W6-1 (2026-10-07): synccheck integration — contract gaps and local workarounds

`synccheck::check(&RecordingObserver, &SynccheckConfig) -> Report` and
`synccheck::serialize(&Report) -> serde_json::Value` are implemented on the
production `crate::sync` step functions (no `numsim-sync-ref` dependency).
Gaps found while integrating, each with the workaround in place:

1. **`FindingKind` has no non-confluence kind.** Distinct terminal protocol
   states (legacy payload kind `fixed_sync_nonconfluent`) are reported as
   `FindingKind::Other("fixed_sync_nonconfluent")`. Request: `NonConfluent`.
2. **No structured per-finding payload.** The legacy keys Python pins
   (`kind`, `reason`, `resource`, `protocol`, `source`, `witness`,
   `witness_evidence`, `related_operations`, ...) travel as one
   `Evidence { role: "payload", detail: Some(<json>) }` per finding, which
   `synccheck::serialize` unpacks. Request: `Finding.details:
   Option<serde_json::Value>` (or a string map) so the payload is not
   smuggled through `Evidence.detail`.
3. **`Report.coverage` is `Vec<(String, u64)>`.** Search stats and echoed
   limits fit; the termination kind does not and is re-derived in
   `serialize` from the findings. Fine unless W8 needs it verbatim.
4. **`named::Cmd` has no `.aligned` flag.** The legacy aligned-site rule (all
   blocking syncs of a generation aligned, one static site per warp;
   `sync_fixed_unified.rs:1180-1236`) cannot be checked. Request:
   `aligned: bool` on `named::Contribution` (or on the `Protocol` event).
   Until then the rule is not enforced offline.
5. **`SyncEvent` carries no kernel index.** `SynccheckConfig.launch` sets
   `Report.launch` / `Evidence.kernel`; payload operations carry
   `kernel_index: null` for W8 to fill.
6. **Two-phase blocking commands.** The explorer drops logged named
   `Resume` and setmaxnreg `Poll` commands and re-derives them from the
   `Registered` / `Pending` outcome, so either logging convention works.
   Please document which one the engine uses.
7. **Successful `TestState`.** `observed_parity` exists only for parity
   tests; every logged `TestState` is treated as a recorded success
   (conditional: only schedules where it succeeds are explored). Request:
   log only successful `TestState`, or add `observed: bool`. Failed parity
   polls (`observed_parity: None`) are dropped as no-ops; the engine need not
   log them.
8. **mbarrier `armed` on `Blocked`.** `WaitParity` returns `Blocked` *and*
   sets `armed` (single-target `SyncTable::step` commits it, `step_all`
   discards it). The explorer models arming as one transition per resource
   (`Transition::Arm`), not per warp; per-warp arming made the K-stage
   pipeline exponential in the number of consumer warps. W3: please confirm
   the step/step_all asymmetry is intended.
9. **Async-group milestones are not in `issued`.** The explorer derives the
   `ReadsDone` / `FullyDone` completions of every non-empty committed group
   from the `async_group` state after `Commit` / `ArriveOn` / `Exit`, and
   gates a `cp.async.mbarrier.arrive` deferred arrival on its group's full
   completion (`ArriveOn { group: Some(o) }`). No contract change needed if
   the engine follows the same rule.
10. **Kernel-wide tcgen05 `cta_group`** (W3-3) is not explored either; it is
    deterministic per command, so Phase A covers it once W3-3 lands.

## W5-1: mbarrier arrive/wait scope and lost qualifiers

PTX ISA §9.7.15.16.16 and §9.7.15.16.19: `mbarrier.arrive` defaults to
`.release.cta` and `try_wait`/`test_wait` default to `.acquire.cta`. The edge
exists only when the arrive scope and the wait scope mutually include each
other's thread (§8.9.4; `racecheck-isa-answers.md` R4). `SyncKind::Arrive`
and `SyncKind::Wait` carry `release`/`acquire: bool` but no scope. They also
cannot say that lowering lost the `.sem` qualifier (R5: never assume relaxed).

Request: `Arrive { obj, phase, release: Option<bool>, scope: Option<Scope> }`
and `Wait { obj, phase, acquire: Option<bool>, scope: Option<Scope> }`:
- `scope = None` for named barriers (participants, no scope);
- `Some(Cluster)` for `barrier.cluster`;
- the explicit or default scope for mbarrier;
- `release`/`acquire = None` when the qualifier is unknown.

Workaround (`racecheck::observer::barrier_scope`): mbarrier is assumed `.cta`
and a named barrier is scopeless. A cross-CTA mbarrier arrival that fails the
assumed check reports `Incomplete::MbarrierScopeUnknown`, never a race or a
silent pass. A lost qualifier cannot be detected.

## W5-2: typed racecheck facts on `report::Finding`

Racecheck findings carry facts the Python layer pins:
- `access_pair`
- `ordering_domain`
- `ordering_failure`
- `proxy_bridge{prior,current}_{proxy,domain}`
- `hint`
- `occurrences`
- per-witness lane / access kind / proxy / scope

`Finding`/`Evidence` have no typed slot for them. Request either
`Finding.data: serde_json::Value` (checker-owned and schema-versioned) or
typed fields.

Workaround: one `Evidence { role: "race", detail: <JSON object> }` per finding,
plus `prior`/`current`/`overlap` evidence whose `detail` is JSON.
`racecheck::serialize` parses these back.

## W5-3: finding kinds

Request these `FindingKind` variants. Today they use `Other(..)`:

| Kind | Status | Meaning |
| --- | --- | --- |
| `TmemLifetimeReview` | review | Unwaited `tcgen05.ld` vs reuse |
| `ScopeMismatch` | error | A release/acquire pair whose scopes do not mutually include the other thread |
| `UndeclaredProtocolWord` | review | A strong poll on a word not declared for `wait_until` (R9) |
| `CrossCtaAsyncOrder` | review | Async-proxy writers from different CTAs ordered only by base causality (R3, ISA-silent) |

The async-lifetime finding (an allocation ended under an in-flight async
footprint) currently maps to `AsyncRace`. An `AsyncLifetime` variant would be
clearer.

## W5-4: tensormap proxy fences

`FenceEvent::TensormapRelease` / `TensormapAcquire` carry neither scope nor
the acquired address range (`fence.proxy.tensormap::generic.acquire.<scope>
[addr], size`). The checker therefore models them as thread-local
bridges over all tensormap bytes, without a scope check. Request
`TensormapRelease(Scope)` and `TensormapAcquire { scope, alloc, span }`.

## W5-5: per-launch reports for numsim-py

`RaceObserver` keeps one result per launch (`launches`), and
`racecheck::payload::reports()` returns one `Report` per launch. Today
`Racecheck::finish()` (used by numsim-py) returns them merged, with `launch` set
to the last launch. Request that numsim-py switch to
`racecheck::payload::reports(&obs)` when it maps phases (W8-2).

## W4 (2026-10-07): oplib wiring, numsim-types, resolve_ptx

### W4-1: new leaf crate `numsim-types` (done, please review)
`numsim-oplib` must share the contract `Dtype`/`Ty`/`WarpValue`/`WarpMask`, and
`numsim-core` now depends on `numsim-oplib`, so the types moved to a new
workspace member `numsim-types` (`src/{dtype.rs,value.rs}`; serde only).
`numsim-core/src/dtype.rs` is now `pub use numsim_types::dtype::*;`;
`numsim-core/src/value.rs` re-exports `numsim_types::value::*` and keeps
`RegFile` (it needs `oplib::Scalar`). No API change. Workspace `Cargo.toml`
gained the `numsim-types` member; `numsim-core` gained deps `numsim-types`,
`numsim-oplib` and dev-dep `numsim-oplib[goldens]`.

### W4-2: `OpKey.mods` format
W1's `ptx_lower.py` emits bare tokens (`d.mod_tokens`: `["rn","ftz","f32"]`),
`calls.py` emits `(slot, token)` pairs for helpers. `resolve_ptx` accepts both
`"slot=token"` and bare tokens; bare tokens are assigned to TVM-table slots in
table order (unambiguous for every table op today), unknown/duplicate/missing
fail closed. Request: lowering emits `"slot=token"` everywhere so the key is
self-describing and does not depend on slot order.

### W4-3: `PtxFn` cannot carry data
`PtxFn = fn(&mut PtxIo)` has no environment, so every modifier combination
needs its own fn item. Adapted locally: parameterized forms resolve once into a
boxed closure, interned process-wide by (key, tys), and returned as one of 2048
pre-instantiated trampolines (`ptx/tramp_table.rs`; exhaustion fails closed).
Hot forms (mov/pack/unpack, f32 add/mul/fma, ex2, ...) are direct fn items.
Request: `PtxFn = Arc<dyn Fn(&mut PtxIo) -> OpResult + Send + Sync>` (or
`(fn, &'static OpData)`), which removes the global table.

### W4-4: `oplib::OPS` static removed
The empty `pub static OPS: &[OpEntry]` is replaced by `registry()` (built once
from `numsim_oplib::registry::OPS`, 600 rows). `render_supported_ops_md`
reproduces the legacy `engine-rs/SUPPORTED_OPS.md` byte-for-byte (ABI v38
header, CUDA/PTX table then tile table) — tested. `OpEntry.instr` is derived
from W1's `builtins.py` family table; `""` for rejected ops.

### W4-5: smaller signature issues
- `shfl(...) -> (WarpValue, WarpMask)` cannot report legacy's
  "warp shuffle reads a non-participant lane"; a non-member source returns the
  source's value with predicate false. Request `OpResult<..>`.
  (`tirx.ptx.shfl_sync` through `resolve_ptx` does return the error.)
- `redux(op, ty, ..)`: `.NaN`/`.abs` f32 forms are not expressible through
  `ReduxOp`; they resolve via `resolve_ptx` (`redux_sync_f32`, `.abs` fails
  closed as legacy had no variant).
- `FloatScalar::from_f64(x, Rounding::Rs, _)` cannot fail; falls back to Rn.
- `unary` with `IsNan/IsInf/IsFinite` writes `Ty{Pred, lanes}`; handlers must
  size `out` by the destination type (same for `cast`). `Cast.sat` is
  implemented as saturate-to-finite for float destinations and range clamp for
  int destinations (not PTX `.sat` = clamp to [0,1]).
- Lowering emits `Unary op="BitNot"` (`ir_walk.py`); `UnOp` only has `Not`.
- `tma_plan` has no direction (load plan returned; `Im2colNoOffs` = store
  plan); `TmaPlan` cannot express NaN OOB fill or the TF32 load rounding (both
  fail closed); `TensorMapDesc` lacks im2col corners/`wide` and a separate
  swizzle atomicity; `TensorMapDesc::encode` cannot fail (unencodable -> zeros,
  rejected by `decode`). Request `dir`, fill pattern, im2col fields,
  `OpResult<[u8;128]>`.
- `tc_mma` closures address one CTA (no cta_group::2) and oplib cannot resolve
  `args.variant: StrId` (lut_b/ti16) or the target arch; `decode_instr_desc`
  has no `cta_group`. Request a CTA index in the closures and the arch/variant
  string in the payload.

## W5 status (after e2551cf, b3c72f3)

W5-1 through W5-4 are resolved and adopted in `racecheck/`. W5-5 is done:
numsim-py uses `racecheck::payload::reports`.

Remaining note on item 5 of `contract-review.md`: the racecheck adapter splits
a multi-lane per-thread async op into one virtual actor per issuing lane. This
requires either one `AsyncId` per (instruction, lane), or `LaneSpan.lane` set
to the issuing lane on async spans. `ALL_LANES` spans on a multi-lane op are
`incomplete` (`async_lane_unknown`), never merged.

## W6-1 resolved (2026-10-07)

Adopted in `synccheck/`: per-target `ProtocolCmd` (counts and observed
parity per target; a recorded `observed_parity` or a `TestState` marks the
instruction conditional), `SyncEvent.kernel` -> `Report.launch` /
`Evidence.kernel` / payload `operation.kernel_index`, `Finding.attrs` holds the
legacy payload entry (the `Evidence{role:"payload"}` workaround is gone),
`FindingKind::NonConfluent`, and `Report.meta` with `algorithm`,
`execution_model` and `termination`. Kernel-wide `.cta_group` (item 10) is
checked once over the program with `tcgen::use_cta_group`, because the rule
does not depend on order. Explicit `TcgenGroup` commands are stripped, and
`ResourceId::TcgenKernel` is never explored. Named `Resume` and setmaxnreg
`Poll` commands are still dropped defensively, as the ruling expects.

## Contract changes for W1 (coordinator, review items 7 and 10)

All of these are JSON-visible. `FORMAT_VERSION` is now **2**.

1. **Strict serde.** Every program type in `program.rs`, `site.rs`
   (`SiteInfo`, `Span`) and `Ty` carries `#[serde(deny_unknown_fields)]`:
   a misspelled field is a decode error.
2. **Every `Option` field must be present.** Use JSON `null` for `None`.
   A missing `Option` field is now an error (`deserialize_with =
   "required"`). This applies to every `Option` in
   `Instr` variants, the arg structs (`TmaArgs`, `BulkCopyArgs`,
   `MbarArriveArgs`, `TcgenMmaArgs`, `TcgenLdArgs`, `TileArgs`,
   `TmapOverride`), `MemMods.policy`, `BufferDecl`
   (`param_slot`, `byte_len`, `view_of`), `ParamSlot` (`dtype`,
   `tensor_map`, `implicit_base`, `buf`), `Launch.min_blocks_per_sm`,
   `Program.arch`, `SiteInfo.dtype/buffer` and `Span.file`. Examples:
   `"multicast": null`, `"msg": null`, `"pred": null`.
3. **`UnOp::BitNot` added; `UnOp::Not` is now logical-only.** `prim.Not`
   becomes `"Not"` and must have a `Pred` operand. `prim.BitwiseNot` must
   emit `"BitNot"`. `ir_walk.py:503` currently emits `Not` for both.
   Remaining TIR coverage: `_BINARY`/`_COMPARE` in `ir_walk.py`, the unary
   ops in `builtins.UNARY_OPS`, `prim.Select`, `prim.Cast`,
   `prim.Broadcast`/`Shuffle` (through `Ptx` pack) and `prim.Let` are all
   representable. `tirx.sigmoid/exp10/log10/erf/nearbyint` already go
   through `Ptx`.
4. **Constants.** `Const.bits` must fit `ty.bits()`, so signed values are
   masked to their width (`-1i32` = `4294967295`). `validate` rejects
   wider bits and any `Const` type wider than 128 bits.
5. **New `Dtype` variants (JSON names):**

   | Variant | TVM dtype |
   | --- | --- |
   | `U6` | `uint6` |
   | `E3M4` | `float8_e3m4` |
   | `E4M3Ieee` | `float8_e4m3` |
   | `E4M3B11Fnuz` | `float8_e4m3b11fnuz` |
   | `E4M3Fnuz` | `float8_e4m3fnuz` |
   | `E5M2Fnuz` | `float8_e5m2fnuz` |

   `Dtype::from_tvm` now covers all 24 `dtype_registry.json` types.
6. **Sub-byte buffers.** `Load`/`Store.offset` stays in elements of the
   buffer's `dtype.elem`. The access starts at *bit* `offset * elem.bits()`
   (`BufferDecl::bit_offset` / `byte_offset`). For fp4/fp6/u4/u6 buffers
   that bit offset must be byte-aligned and `ty.bits()` a multiple of 8;
   otherwise the access is `Misaligned`. Byte lengths of packed arrays use
   `Dtype::array_bytes(n)`; `index * mem_bytes()` is wrong for these
   buffers. Emit `AddrOf` element offsets the same way.
7. **`validate` is now exhaustive.** It checks:
   - every nested index (registers, register ranges, consts, buffers,
     strings, layouts, ops, preds, params, `DimExpr` params);
   - `Ty` invariants (`lanes >= 1`, at most 256 bits);
   - that each destination's written type fits the register's declared
     bits: `Compare` writes `Pred x lanes`, `AddrOf` writes `U64`, `Cast`
     writes `to`;
   - frame-matched control-flow targets. `Else.end_pc` must equal its
     `If.end_pc`. A no-else `If` must have `else_pc == end_pc ==` its own
     `EndIf`. `LoopIf` must sit directly in its loop, exactly once, with
     `end_pc` = its `LoopEnd`. `LoopEnd.head_pc` must lie strictly inside
     its loop;
   - **`PredProgram` placement.** Predicate ranges tile `code[main_end..]`
     exactly, disjoint and in any order. The main body must end in `Exit`
     or `Unsupported` when preds exist.
8. **`may_block()`** is now also true for `SetMaxNReg { inc: false }` and
   for `TcgenDealloc`/`TcgenRelinquish` with `cta_group == 2`. Lowering
   must treat them as scheduling points.
9. **`OpKey.mods`:** please emit `"slot=token"` (W4-2). Bare tokens still
   resolve.
10. **`ParamSlot.tensor_map`:** a slot with `tensor_map: Some(spec)` gets no
    host value. The engine encodes the map from `implicit_base` at bind
    time (W8-3; implementation W2).

### Batch 2 (2026-10-08, lowering residuals, Part C.4)

`FORMAT_VERSION` stays 2. All new `Option` fields are required, so emit
`null` when absent.

11. **mbarrier wait reports.** `MbarTestWait` gains `report: Option<Reg>`
    (`_report`) and `report_value: Option<Reg>` (`_report_value`). Both come
    from the same snapshot as `dst` (sync-semantics.md §2.4, copy-report /
    conditional parity). Lower `mbarrier_{test,try}_wait*_report*` to these
    fields. `MbarWait` has no report forms.
12. **`StAsyncArgs.mbar` is `Option<Operand>`.** Use `null` for
    `st.async.release` / `red.async.release` without an mbarrier.
13. **`BulkCopyArgs` gains three fields:**
    - `byte_mask: Option<Operand>` (`.cp_mask`, 16-bit per 16-byte chunk)
    - `ignore_oob: bool`
    - `report: Option<ReportMode>`, where `ReportMode` is `"PerElementFf"`
      or `"Per16Bytes"`

    **Deviation from the request:** `report` is a mode, not a register.
    The `_report` copy forms have no register destination: they OR a
    validity predicate into the completion mbarrier's report bit, which the
    kernel reads back through `MbarTestWait.report`. `TmaArgs` gains the
    same `report: Option<ReportMode>` for the `cp_async_bulk_tensor_*_report`
    forms.

    **TMA overrides:** `TmaArgs.overrides: Vec<TmapOverride{field, ord,
    value, elem_bits}>` covers every override spelling in SUPPORTED_OPS:
    | spelling | override |
    | --- | --- |
    | `override_address[_im2col]` | `GlobalAddress`, `ord: null` |
    | `override_global_dim_b8/_b16` | one `GlobalDim` per `ord`, `elem_bits` 8/16 |
    | `override_global_dim_stride_b8/_b16` | `GlobalDim` + `GlobalStride` per `ord` |
    | `applypriority_*_override_*` | same overrides, `dir: Prefetch` (ordering-only) |
14. **tcgen05:**
    - `TcgenMmaArgs.variant: Option<StrId>` is removed. Use
      `ti16: bool` and `lut_b: bool` instead; both are orthogonal to
      `kind`, since block-scaled `lut_b` forms exist.
    - `TcgenCommit` gains `sync_restrict: bool` and
      `multicast_width: Option<u8>`.
15. **`TensorMapSpec.force_cu_dtype: Option<u8>`** carries the raw
    `CUtensorMapDataType` when it differs from what `dtype` implies (11, 13,
    14, ...). Stop mapping 11 to TF32 or 13 to E2M1, and stop failing 14
    closed. Emit the raw value and the engine decides.
16. **`ParamKind::ImplicitShape { buffer: ParamId, axis: u8 }`** replaces
    the `Scalar` slots named `<buf>.shape<axis>`. JSON:
    `{"ImplicitShape":{"buffer":0,"axis":1}}`. `buffer` must name a `Buffer`
    slot (`validate` checks this). The binder (W8/W2) reads the value from
    the bound array.
17. **TMEM buffers (C.4 row 1).** The ruling is in the `BufferDecl` doc. A
    `Space::Tmem` buffer may be used with `Load`/`Store`. These execute as
    `tcgen05.ld/st 32x32b` with dense addressing:
    - lane = `(offset / cols) % 128`;
    - column = `base_col + offset % cols`;
    - 32-bit elements;
    - each active lane addresses its own warp sub-partition.

    Anything else fails closed. `AddrOf` / `LoadAddr` / `StoreAddr` on TMEM
    are not allowed.

18. **`TcgenMmaArgs.lut_b_addr: Option<Operand>`** (W2-7). This is the LUT
    table address for `lut_b` forms. It must be non-null exactly when
    `lut_b` is true; `validate` enforces this.
19. **`sync::AsyncKind::TcgenCommit`** (W2-4). This is engine-internal and
    not in the JSON. A `tcgen05.commit` is an `AsyncOp` with
    `Payload::None`, `after` = the tracked ops, and `signals` = the deferred
    arrive(s). It matches `observe::AsyncClass::TcgenCommit`.

### Batch 3 (2026-10-08, lowering §D.3)

20. **`TcMmaKind::Ti16`** (`"Ti16"`) is now a real kind for
    `.kind::i16`. `TcgenMmaArgs.ti16: bool` is **removed**. `lut_b: bool`
    stays and remains orthogonal to the kind.
21. **Split-stride TMA overrides** (`override_global_dim_stride_*`). Emit:
    - one `GlobalStride` override per dimension `ord`, holding that
      dimension's *lower* stride operand;
    - one `{"field": "GlobalStrideUpper", "ord": null, ...}` override,
      holding the shared upper operand.

    Oplib combines them exactly as legacy `override_tensor_map` did.
22. **`BufferDecl.base_reg: Option<Reg>`** (required field; `null` when
    unused). It covers TMEM views whose `allocated_addr` is only known at
    run time. When set:
    - the buffer starts at that register's value (a TMEM `taddr`), read at
      each access;
    - the register must be warp-uniform;
    - `base` must be 0 (`validate`).
23. **`TcgenLdArgs` gains `red_abs: bool` and `red_nan: bool`** for
    `.red.abs` / `.red.NaN`. `.spcompress` was already the `spcompress`
    bool.
24. **`BulkCopyArgs.ignore_oob` is now `Option<IgnoreOob>`**, replacing
    the bool. The struct is
    `IgnoreOob { ignore_bytes_left: Option<Operand>, ignore_bytes_right: Option<Operand> }`.
    `null` means no `.ignore_oob`; a `null` count means 0.
25. **`SpecialReg::NWarpId`** (`"NWarpId"`) for `%nwarpid`. The engine
    returns a deterministic representative value.
26. **`TcgenMmaArgs.lut_b_addr`** is documented as a *TMEM* address
    (`addr@tmem`, `lane<<16 | column`). `validate` still requires it to be
    set iff `lut_b`. The engine fails closed if the address is outside a
    live allocation.

27. **tcgen05 row/column offsets** (W4-7). `TcgenLdArgs`, `TcgenStArgs`
    and `TcgenCpArgs` each gain two required operands, `row: Operand` and
    `col: Operand`. Emit const 0 when the source has none.
    - Effective lane = `((taddr >> 16) + row)` mod 2^16; effective column =
      `((taddr & 0xffff) + col)` mod 2^16. This is legacy
      `raw_tcgen05_address`.
    - `ld`/`st` require both operands to be warp-uniform. `cp` reads them
      in the issuing lane.

28. **Sub-word TMEM cells.** `Load`/`Store` on a `Space::Tmem` buffer
    with an 8- or 16-bit element type is allowed. `per_cell = 32 / bits`
    elements share one 32-bit cell:
    - the cell index is `offset / per_cell`, addressed by the unchanged
      lane/column rule;
    - the element sits at bit `(offset % per_cell) * bits` within the cell;
    - a sub-word `Store` is a read-modify-write of its cell.

    `validate` accepts these dtypes. Other widths fail closed at run time.
    The ruling is documented on `BufferDecl`.
29. **Replicated TMEM views stay fail-closed.** Emit
    `Unsupported { reason: "tmem_replicated_view: <buffer>" }`. No corpus
    kernel uses them.
30. **`TensorMapSpec.box_dim` and `element_stride` are now
    `Vec<DimExpr>`** (were `Vec<u32>`). Emit `{"Const": n}` for static
    values. `validate` checks their params. W2's bind-time encode must
    evaluate them like `global_dim`.

**C.3 acks.**
- **Accepted:** 1 (`numsim.pack`/`unpack`, W4), 2 (`<name>.value`, W4),
  5, 6, 7, 8, 9 and 10.
- **3 is superseded** by item 16.
- **4 is superseded** by item 15.

**Breakage in other owners' files** caused by batch 2. `program.rs` is
clean, and the lib tests pass with minimal stand-in patches, since
reverted:
- **W2 / W7:** the patterns at `interp/handlers.rs:528,539`,
  `codegen/emit.rs:130,150` and `interp/handlers/async_copy.rs:557` need
  updating (`mbar` is now `Option`).
- **W2:** add the new `BulkCopy` / `MbarTestWait` fields to
  `testutil/scenarios.rs:297,443`, and handle `ImplicitShape` in
  `sched/mod.rs:486,1551`.

## Review item 10 residue: requests for non-coordinator-file owners

These are outside `program.rs`, `dtype.rs`, `value.rs`, `site.rs` and
`lib.rs`, so they were not applied.

- **W3, `sync/completion.rs`:** `ResourceId::TcgenLifecycle { pair }`
  should identify the pair by cluster rank,
  `{ cluster: u32, pair_rank: u8 }` with `pair_rank = ctarank >> 1`, the
  peer being `ctarank ^ 1`. It should not be "the global id of the even
  CTA".
- **Coordinator, `observe.rs` + W2, `interp`:** widen
  `Actor::Warp.epoch` and `WarpState::epoch` to `u64` (no
  `wrapping_add`).
- **Coordinator, `observe.rs`:** reword the declared-word history doc to
  say: "bit 0 = the word's value at launch start (or at declaration); bit
  i >= 1 = the i-th (Access, lane) write overlapping the word, in delivery
  order, lanes ascending within an Access, value = byte-merged
  post-image". `LaneVerdict` should cite the same rule.
- **W2, `arena.rs`:** `addr::shared_cluster(rank, offset)` must return
  `Option<u32>`, giving `None` when `offset >= 1 << 24` or the rank does
  not fit, never masking. The decode side should likewise reject
  out-of-window offsets.
- **Not applied (needs a decision):** `boolx128` exceeds
  `MAX_VALUE_BITS = 256` (`Pred` counts 8 bits per lane). Either lower
  `boolx128` params as `B128`-packed bit vectors, or reject them. Today it
  is rejected by `Ty::from_tvm` and `validate`. `float4_e2m1fnx32` is 128
  bits and fits.

## Sweep of W2/W5/W6/W7/W8 requests against program/dtype/value/site

- **W4-5 `BitNot`:** applied (item 3 above).
- **W4-2 `OpKey.mods`:** documented on `OpKey` (item 9).
- **W8-3:** documented on `ParamSlot.tensor_map` (item 10).
- **W5-4:** applies to `observe::FenceEvent`, already resolved. The
  instruction-level `FenceKind::TensormapAcquire { addr, space }` keeps its
  implicit 128-byte size, and `Fence.scope` carries the scope. No change.
- **W8-2:** `SiteId` stays per kernel. Kernel attribution is
  `SyncEvent.kernel` / `Evidence.kernel` (done elsewhere). No change to
  `site.rs`.
- **W7-1..6, W6-1, W5-1..3, W5-5, W3-*:** these target observe, report,
  sync, sched, oplib or Cargo. None touches the files above; no action.

## W3 (2026-10-07): tcgen pair identity done

`ResourceId::TcgenLifecycle` is now `{ cluster: u32, pair_rank: u8 }`, with
`pair_rank = ctarank >> 1`. The W2 handler `interp/handlers/tcgen.rs` builds
it this way; its rendezvous key is a new local `pair_cta(ctx)`, unchanged.
W6's `synccheck/build.rs::tmem(pair)` maps to `{cluster: pair, pair_rank: 0}`.
`numsim-sync-ref` does not name the pair, so it needs no change.

## W2 (2026-10-08): interpreter + scheduler

Changes inside W2-owned modules that other workers see, plus requests.

### W2-1: additions to `interp` / `sched` types (done, W2-owned)
- `ExecCtx` gained `aux: &mut interp::LaunchAux` (async-group membership,
  declared-word histories and verdict caches, collective rendezvous,
  tcgen pipeline order, uninit diagnostics). Generated code only passes
  `ctx` through, so W7's printer is unaffected (ABI fingerprint changes).
- `Loaded` gained `op_errors`, `progress`, `param_offsets`, `param_bytes`,
  `local_per_lane`, `uses_*`; build it with `Loaded::new(&program)`.
- `MaskFrame.origin` (pc of the `If`/`LoopBegin`), `WarpState.suspended`
  (`Suspension`: divergent-arm scheduling rule, review item 6),
  `PollState.also/overflow` (every polled resource, review item 9),
  `CtaCtx.cluster_tmem`, `LaunchCounters.progress`.
- Shared step pieces for both backends: `interp::{begin_instr, end_instr,
  fall_off_end, divergent_switch}`. The scheduler wraps every slice in
  `codegen::rt::guard` (W7 request 1). Handlers never write `ctx.warp.pc`;
  only `end_instr`/`divergent_switch` (dispatcher level) do.
- `RunConfig` gained `workers` (parallel path designed, not implemented:
  `> 1` runs the single-threaded path, results identical),
  `max_resident_ctas` (default 1024; cooperative / `grid.sync` launches are
  fully resident) and `subset` (W8-4). Default validity is
  `ValidityPolicy::ZeroAndReport` (W8-5). `RunOutcome` gained
  `failed_kernel`, `diagnostics`, `subset`; `ArgValue::TensorMapOf` (W8-3).
- `testutil::scenarios` (W7 request 5): `Scenario { name, module, inputs,
  config }` and `scenarios::all()`; expectations are in
  `tests/interp_scenarios.rs`.

### W2-2: shared-address encoding (review item 1, done in `arena::addr`)
`shared_addr(rank, off) -> Option<u32>` = `rank << 24 | off`,
`decode_shared(a) -> (rank, off)`; the old `shared_cluster` /
`decode_shared_cluster` are removed. `.shared::cta` accesses require
`rank == own` (else `BadAddress`); `.shared::cluster` and generic shared
addresses route by rank (`GENERIC_SHARED_BASE + shared_addr` is the
distributed-shared generic aperture). `cvta.to.shared` of an own-window
generic pointer therefore yields `own_rank << 24 | off`, `mapa(p, own)
== p`, and `p & 0xFEFF_FFFF` names the pair leader. Also added
`addr::GENERIC_PARAM_BASE` / `Generic::Param` (generic addresses of
`__grid_constant__` tensor maps).
**W1:** shared addresses are no longer bare window offsets in clusters with
`rank > 0`; offsets must be computed as differences of addresses (or with
`decode_shared`), never by assuming `cvta(...)` < window size. Constants
built by lowering for shared addresses must be rank-tagged (`rank 0` is
correct only for the leader CTA).

### W2-3: `may_block` for rendezvous instructions (applied by the
coordinator in 8a0afff)
`setmaxnreg.dec` and `cta_group::2` `tcgen05.dealloc/relinquish` block until
the whole warpgroup / the peer CTA's warp has arrived (the handlers now
return `Blocked`).

### W2-4: `AsyncKind` has no `TcgenCommit`
`tcgen05.commit` is queued as an `AsyncOp` with `Payload::None`,
`after = tracked ops` and the deferred `MbarArrive` signals, so its arrival
lands only after the committed mma/cp ops. Its `AsyncKind` is
`TcgenMma` for lack of a variant (the observer sees
`AsyncClass::TcgenCommit`). Request `AsyncKind::TcgenCommit`.

### W2-5: protocol logging convention (W6-1 item 6)
One `Protocol` event per *completed* instruction: a named `bar.sync`/`red`
that registers and blocks logs its `Sync`/`Red` contribution command when
it completes (immediately or at the successful `Resume`); `Resume` is never
logged. `setmaxnreg` logs `Set` per warp at completion (after the grant
poll for `inc`), with `Collective{id, participants}`. Failed polls and
blocked attempts are not logged. Cross-CTA targets (remote / multicast
arrive, expect_tx, complete_tx) are stepped synchronously in the same
`step_all` (review item 2); the inbox carries nothing today but each drain
still emits `Observer::inbox_drain`.

### W2-6: oplib performance (for W4)
`oplib::binary(Add, U32)` costs ~650 ns and `compare(Lt, U32)` ~300 ns per
32-lane call (release build), which is ~90% of the interpreter's time on a
scalar loop (`benches/interp.rs`: ~420 ns per warp instruction; the
interpreter's own dispatch is ~30 ns per instruction). A per-type
monomorphized fast path for 32-bit int/float ops would make the
interpreter ~10x faster on ALU-bound kernels.

### W2-7: open items
- `TcgenLd/TcgenSt` other than `.32x32b` (and `.pack/.unpack/.red`),
  `tcgen05.cp`, and `Tile` fail closed (`Unsupported`); `tcgen05.cp` needs
  a smem-descriptor -> spans plan from oplib.
- `tc_mma` closures address one CTA (W4-5); the scheduler maps rank-tagged
  smem addresses and TMEM lanes >= 128 to the pair peer.
- `wait_until` captures: registers cannot change while the warp waits, so
  the register file is the snapshot; verdict caches reset per lane when
  the capture values change.

## W4-6 (2026-10-08): phase 3 signature changes — W2 call sites

All approved by the coordinator on 2026-10-07. Old forms are kept as thin
wrappers where noted, so existing call sites keep compiling; please migrate.

- **`PtxFn` is a data-carrying `Copy` struct** (no more trampoline table):
  `oplib::PtxFn { .. }` with `PtxFn::new(fn(&mut PtxIo) -> OpResult)`,
  `PtxFn::from_static(&'static PtxOp)`, `f.call(&mut io) -> OpResult`,
  `is_direct()`, `same(&other)`. Parameterized forms are resolved once into a
  closure (interned per distinct (key, tys), leaked so `PtxFn` stays `Copy`).
  W4 applied the two mechanical call-site edits in W2's files to keep the tree
  building: `interp/mod.rs` `ops.push(PtxFn::new(support::unresolved_ptx))`
  and `interp/handlers/alu.rs` `f.call(&mut io)`.
- **`oplib::shfl_sync(mode, src, lane, clamp, membermask: &WarpValue<u64>,
  active) -> OpResult<(WarpValue<u64>, WarpMask)>`** with legacy validation
  (non-participant source / lane missing from its membermask = `Invalid`).
  `oplib::shfl` kept (infallible compatibility form). Migrate
  `interp/handlers/warp.rs`.
- **TMA**: `tma_plan_dir(map, TmaPlanDir::{Load,Store}, mode, coords,
  im2col_offsets, smem_offset)`; `TmaPlan` gained `fill: TmaFill`,
  `fill_pattern: Vec<u8>` (repeat over each `smem_oob_fill` span; empty =
  zeros; NaN fill = `[0xf7, 0x7f]`) and `tf32_round: bool` (round the copied
  4-byte elements with `oplib::tma_tf32_round` on landing). Reductions plan as
  `Store`. `tma_plan` kept (Load, except store-only `Im2colNoOffs` /
  `TileScatter4`). Named `TmaPlanDir` because `program::TmaDir` exists.
  `TensorMapDesc` gained `swizzle_atomicity: u8` and `im2col:
  Option<Im2colBox{lower, upper, wide}>` (Default-compatible);
  `try_encode() -> OpResult<[u8;128]>` (`encode()` kept, zeros on failure).
- **tcgen05**: `tc_mma_ctas(payload, &TcMmaOptions, smem: Fn(cta, addr, buf),
  tmem_read: Fn(cta, lane, col, buf), tmem_write: FnMut(cta, lane, col,
  bytes))` — `cta` is the index within the group (0 = issuer for
  cta_group::1; 0/1 = even/odd CTA of the pair). `TcMmaOptions{arch: TcArch,
  ti16, lut_b: Option<u32>, zero_col_mask: Option<u64>, fixed_vectors}` with
  `TcMmaOptions::parse_variant(&str)` for the resolved `args.variant` string.
  `.ws` zero-column masks above bit 31 need `zero_col_mask` or both
  `disable_output_lane` words; `.lut_b` needs the table taddr in `lut_b`
  (not in `TcgenMmaArgs`). `tc_collector_transition(state, a, b, b_buffer)`
  is the legacy collector state machine (engine keeps the per-lane state).
  `decode_instr_desc_for(idesc, kind, cta_group)` added. `tc_mma` kept
  (cta 0, default options; cta_group::2 -> Unsupported "needs tc_mma_ctas").
  Migrate `sched::run_mma`.
- **Register-encoding exception (behaviour delta D1, ruled "match legacy")**:
  a scalar F16/BF16 value produced by TIR `Unary/Binary/Ternary` that is not
  exactly representable keeps its rounded bits in 0..16 plus the unrounded f32
  in bits 32..64 and flag bit 16 (legacy f32 carrier). Stores must write only
  `ty.mem_bytes()` low bytes (drops the carrier = legacy round-at-store); `Mov`
  / `Select` / `LoadRegIndexed` copy slots unchanged (keep it). Anything that
  compares whole slot values of f16/bf16 registers must mask to `ty.bits()`.
  See docs/development/numsim-behaviour-deltas.md (D1). Alternative if the
  contract worker prefers a clean encoding: W1 lowers half expression trees in
  f32 (Cast at leaves, Cast back at stores), and this carrier can be removed.
- **W1 §C.3 ops implemented**: `numsim.pack` / `numsim.unpack` (mod
  `ty=<Dtype debug name>x<N>`; pieces' bit widths must tile the vector, pure
  bit moves incl. sub-byte) and `tirx.cuda.{float22half2,float8tohalf8,
  half8tofloat8}.value` (legacy `cuda_f32_to_fp16_bits` /
  `cuda_fp16_bits_to_f32` element conversions) in `oplib/ptx/vector.rs`.
- `TcgenMmaArgs::{ti16, lut_b}` (contract) are honoured by `tc_mma_ctas`;
  `lut_b` still needs the table taddr in `TcMmaOptions::lut_b` (fails closed
  without it) because the args carry only the flag.

## W5-6: declared-word history numbering is stated two ways

There are two conflicting statements:
- `README.md` decision 14: bit i = the i-th **(Access, lane)** write, lanes
  ascending.
- `observe.rs:25-28`: index i = the i-th delivered write **Access**.

Racecheck follows decision 14: one entry per (Access, lane), for every
overlapping declared word, lanes ascending (`checker-review.md` S4). Please
reconcile the `observe.rs` comment, and make the engine's history match,
before W2 emits `WaitVerdicts`.

## W1: half chains lowered in f32 — done (2026-10-08)

Ruling D1 is implemented in `numsim/v2/lowering/ir_walk.py` (`is_half_chain` /
`wide` / `half_chain`). An f16/bf16 TIR expression tree (`Add Sub Mul Div Min
Max`, `Select`, the `UNARY_OPS` math calls, `tirx.fma`) computes in
`Ty{F32, lanes}` registers. Leaves are widened with one `Cast` each; half
`FloatImm`s become f32 constants of the half-rounded value. Exactly one `Cast`
to the half type is emitted where the value leaves the chain: a store, an
explicit `Cast`, a call operand, a `Bind`, or any other non-chain consumer.
Test: `tests/numsim/v2/test_lowering_vector_add.py::test_half_chain_stays_f32_until_the_store`.
W4 can drop the bit-16 hack.

## W1 (2026-10-08): phase 3 status and asks

- **W2-2 (rank-tagged shared addresses).** The emitter builds no
  shared-address constants and never assumes `cvta(...)` is below the window
  size.
  - Shared addresses come from `Cvta` / `Mapa` instructions or from the
    kernel's own integer arithmetic on them.
  - `BufferDecl.base` stays a window offset (buffer metadata).
  - The only literal shared operand is `TmaArgs.smem = 0` for
    `dir: Prefetch`, which has no shared side.
- **TMEM.** The emitter follows the `BufferDecl` ruling:
  - `shape = [lane_span, col_span]` of the view's physical rectangle;
  - `base` = the static `allocated_addr`;
  - offset = `lane * col_span + col`;
  - 32-bit elements only;
  - replicated layouts fail closed on direct access.

  Remaining gap: 12 test kernels have a runtime `allocated_addr` (read from
  shared memory) and cannot be represented, because `BufferDecl.base` is
  static. Request: `BufferDecl.base_reg`, or a `Tmem` buffer whose base is a
  register.
- **Still unrepresentable (unit tests only):**
  - `kind::ti16` MMA (25 kernels): `TcMmaKind` has no `Ti16`, and the `ti16`
    flag alone cannot state the kind the PTX names.
  - ~~`lut_b` MMA (10)~~: done with item 18. `lut_b_addr` carries
    `b_decompress_metadata`. Note that TVM's PTX table types that operand
    `addr@tmem`; the `lut_b_addr` doc says "shared-memory address value".
  - `cp.async.bulk(.tensor)` `ignore_bytes_left/right` counts (5) and the
    `override_global_dim_stride_*` lower/upper stride operands (15).
  - `tcgen05.ld` `.spcompress` / `.abs` / `.NaN` (5).
  - `%nwarpid` (2): there is no `SpecialReg`.

## W4-7 (2026-10-08): tcgen ld/st/cp and ldmatrix APIs for W2

New pure functions in `oplib/mem.rs` (+ `oplib/mem/{tcgen,matrix,tests}.rs`),
re-exported from `oplib`. They delegate to `numsim_oplib::{tcgen05::{layouts,
ld, smem_desc}, layout::matrix}` (the legacy layout code) and keep closure
errors' kind. Engine checks (full warp, TMEM allocation/lifecycle, byte
validity, memory resolution, footprints) stay with W2.

**tcgen05.ld / tcgen05.st** (every shape `.32x32b/.16x64b/.16x128b/.16x256b/
.16x32bx2`, `.num` x1..x128, `.pack::16b`/`.unpack::16b`):
- `tcgen_ldst_registers(shape: TcShape, num: u16) -> OpResult<usize>` — data
  registers (`registers_per_num * num`), validating the shape's `.num`.
- `tcgen_ldst_map(shape: TcShape, num: u16, pack16: bool, warp_in_cta: u32,
  taddr: u32) -> OpResult<TcgenLdstMap>` — legacy `raw_tcgen05_ldst_location`
  for all 32 lanes. `TcShape::S16x32bx2 { split_off }` carries
  immHalfSplitoff. A taddr lane < 32 is warp-relative, otherwise it must be in
  the warp's subpartition `32 * (warp_in_cta % 4)` (both legacy conventions).
  `map.pieces(register, lane) -> &[TcgenLdstPiece { tmem_lane, column,
  cell_byte, reg_byte, len }]`: 1 piece (4 bytes) or, packed, 2 pieces (the low
  2 bytes of columns c and c+1 <-> register bytes 0..2 / 2..4). `map.all()`
  for footprints (TMEM byte offset = `addr::tmem_byte_offset(lane, col) +
  cell_byte`).
  - ld: for each register r (`args.dsts` in order, excluding the `.red`
    register) and lane, read the pieces into the register's bytes.
  - st: write the register's bytes into the pieces (`.unpack::16b` writes
    only bytes 0..2 of each cell).
- `tcgen_ld_dst_count(shape, num, pack16, red: bool, spcompress: Option<(max,
  abs)>) -> OpResult<usize>` — destination count incl. the trailing `.red`
  register; validates `.red` (unpacked 32x32b/16x32bx2, >= x2) and
  `.spcompress` (unpacked 32x32b, >= x4).
- `TcgenLdRed::new(op: ReduxOp /*Min|Max*/, ty: Dtype /*F32|U32|S32*/, abs,
  nan) -> OpResult<TcgenLdRed>`; `tcgen_ld_reduce(red, values: &[u32]) ->
  OpResult<u32>`: one lane's left fold over the loaded words in register
  order (legacy reduces the values just loaded; the result validity is the AND
  of the inputs'). `TcgenLdArgs.red` gives `op` and the reduction register's
  type; `.abs`/`.NaN` are not in `TcgenLdArgs` (lowering rejects them), so
  pass `false`.
- `tcgen_ld_spcompress(values: &[u32], valid: &[bool], max, abs) ->
  OpResult<(Vec<u32>, Vec<bool>)>` — one lane: `num.div_ceil(32)` metadata
  words then `num / 2` kept values, with per-output validity (lowering does
  not emit `.spcompress` yet; `TcgenLdArgs.spcompress` would also need
  max/abs).

**tcgen05.cp**:
- `tcgen_cp_plan(rows: u16, bits: u16, multicast: u8, decompress_bits: u8,
  sdesc: u64, taddr: u32, cta_group: u8, arch: TcArch) -> OpResult<TcgenCpPlan>`
  — `TcgenCpArgs` fields as lowered (multicast 0/1=`warpx2::02_13`/
  2=`warpx2::01_23`/3=`warpx4`; decompress 0/4/6). Legacy `raw_tcgen05_cp`:
  `plan.words: Vec<TcgenCpWord { src: ByteSpan /*shared-window address, 4 or
  2 (b4) / 3 (b6) bytes, swizzled per the descriptor*/, lanes, lane_count,
  column }>` in legacy row/word order; `word.lanes()` are the destination
  TMEM lanes (multicast replication); `plan.lane_end/column_end` bound the
  destination rectangle. The plan is CTA-independent: for `cta_group::2`
  apply it in this CTA and its peer (`rank ^ 1`), each reading its *own*
  shared window and writing its own TMEM (legacy `target_views`).
- `plan.pairs() -> (Vec<ByteSpan>, Vec<(lane, col)>)` — pairwise form for
  `Payload::TcgenCp { src, dst, decompress_bits }`: `src[k]` (2/3/4 bytes,
  offset relative to the shared window) lands decoded as the 4-byte cell
  `dst[k]` (`addr::tmem_byte_offset(lane, col)`, len 4). This is pairwise,
  NOT concatenated like `Payload::Copy`.
- `tcgen_cp_decode(src: &[u8], decompress_bits: u8) -> OpResult<[u8; 4]>` —
  decodes one source word at landing (b4: nibbles `<< 2`; b6: four 6-bit
  codes).

**ldmatrix / stmatrix** (all legacy forms: `m8n8.b16` x1/2/4 [.trans],
`m16n16.b8.trans`, `m8n16.s8.s4`, `m8n16|m16n16 .b8x16.b6x16_p32|.b4x16_p64`,
`stmatrix m8n8.b16` [.trans], `stmatrix m16n8.b8.trans`):
- `ldmatrix_plan(shape: MatrixShape, num: u8, trans: bool, fmt: MatrixFmt) ->
  OpResult<LdMatrixPlan { registers, transpose, source_bits, signed,
  providers, row_bytes }>` — `registers` = destination count (`2 * num` for
  m16n16); lanes `0..providers` supply row addresses; each provider row
  contributes `row_bytes` (16/12/8).
- `ldmatrix_fragments(&plan, row_address: Fn(provider) -> OpResult<u64>,
  read: FnMut(provider, byte_delta, len) -> OpResult<Vec<u8>>) ->
  OpResult<Vec<WarpValue<u32>>>` — `result[r][lane]` = destination register
  r. `read` returns `len` bytes at `byte_delta` past the provider's row
  pointer (resolve the provider lane's `addr` operand). b8 formats require
  16-byte-aligned rows (via `row_address`). s4 sign-extends each nibble.
  `plan.accesses(lane, row_address) -> OpResult<Vec<MatrixAccess>>` for
  footprints.
- `stmatrix_plan(shape, num, trans) -> OpResult<StMatrixPlan { registers,
  providers, .. }>`; `stmatrix_writes(&plan, sources: &[WarpValue<u32>]
  /*register-major*/, row_address) -> OpResult<Vec<(provider, byte_delta,
  bytes)>>` — legacy `raw_stmatrix` order; rows must be 16-byte aligned.
- These replace W2's hand-written `m8n8.b16` path in `handlers/warp.rs`
  (same results for that form).

## W4-8 (2026-10-08): phase 4 — ALU fast paths, contract batch 3, W2 call sites

- **ALU performance (W2-6).** `tir::{unary,binary,ternary,compare,cast}`
  now dispatch on `(op, Dtype)` once per call and run a branch-free
  `[u64; 32]` loop on the native type + masked blend (`oplib/tir/fast.rs`);
  everything else falls back to the generic path. Hardware FMA / F16C are
  used with runtime detection where bit-identical (NaN lanes recomputed by
  the scalar definition). Bit-exactness: `tir::tests::fast_paths_match_the_
  generic_path` (random + special operands, partial masks, every fast
  (op, dtype)), `simd` tests (FMA incl. NaNs, exhaustive f16 decode, f16
  encode per exponent class), `ptx/cvt/hot.rs` tests (hot cvt forms vs the
  spelling dispatch). Bench: `cargo bench -p numsim-core --bench oplib`
  (numbers in the W4 report). No signature change.
- **`TcMmaKind::Ti16`** is handled as the kind::i8 driver with the s1z4m11
  operand spelling (`tc_mma_ctas` sets `options.ti16` from the kind).
  `TcMmaOptions::ti16` stays for callers that pass it explicitly. W2:
  `sched/mod.rs:1384` still reads the removed `args.ti16` — drop that field
  from the options literal.
- **`.lut_b`**: `TcMmaOptions::lut_b` is the TMEM taddr (`lane<<16|col`) of
  the table, i.e. the value of `TcgenMmaArgs::lut_b_addr` (`addr@tmem`); the
  table is read through `tmem_read` (never smem). W2: evaluate
  `args.lut_b_addr` and pass it as `options.lut_b`, then remove the
  `tcgen05.mma .lut_b` fail-closed in `handlers/tcgen.rs`.
- **Split-stride overrides**: new `TensorMapDesc::apply_overrides(&[(field,
  ord, value)])` combines the per-ord `GlobalStride` lower operands with the
  shared `GlobalStrideUpper` exactly like legacy `override_tensor_map`
  (`stride = (lower | nibble(ord) << 32) << 4`, full `GlobalDim`/stride sets,
  dims in 1..=255). `replace(GlobalStrideUpper, ..)` alone is `Invalid`. W2:
  replace the per-override `desc.replace` loop in
  `handlers/async_copy.rs` (~547) with one `apply_overrides` call.
- **`tcgen05.ld.red` modifiers**: pass `TcgenLdArgs::{red_abs, red_nan}` to
  `TcgenLdRed::new(op, ty, abs, nan)` (W4-7).

## W2 phase 2 (2026-10-08): parallel scheduler, access rules, open items

### W2-8: `ReportMode::Per16Bytes` needs its pattern (request)
`mbarrier::report::validity::per_16bytes::{8,80,8000,80000000}` inspect the
lowest-addressed element of each 16-byte source chunk against that pattern
(legacy `copy_report_matches_runs`), but `ReportMode::Per16Bytes` drops the
pattern and lowering (`ptx_lower.py::_report`) does too. Request
`Per16Bytes(u32 /*pattern*/)`. Until then the engine fails closed on
`Per16Bytes`; `PerElementFf` is implemented (report bit per (mbarrier,
generation) in `LaunchAux::mbar_reports`, read by report `test/try_wait`;
`report_value` is always 0, as legacy). W3 may move the bit into
`mbarrier::State` (synccheck does not need it).

### W2-9: access-emission rules (documented, implemented)
- **Reductions** (`cp.reduce.async.bulk`, tensor reduce, `red.async`): one
  `Access{kind: Rmw, atomic: true, returns_value: false, sem: Relaxed,
  scope: Gpu}` per landing, actor `Async{op, Write}`, with element-granular
  spans (one `LaneSpan` per element of the reduction dtype).
- **Predicated-off** instructions (empty active mask, guard false, every
  lane skipped) emit no `Access` and no `SyncEvent`.
- **Footprints are the transferred bytes**: `.cp_mask` byte selection,
  `.ignore_oob` left/right clipping, TMA plans (incl. gather4/scatter4 rows
  and OOB fill, which is a separate write of the filled spans) narrow both
  the `AsyncIssue.footprint` and the landing `Access` spans; nothing is
  emitted for skipped bytes.
- Async accesses name the issuing lane; warp-collective ones (ldmatrix,
  stmatrix, tcgen05.ld/st) name each thread's own lane.

### W2-10: synccheck needs the launch's `ResourceInit` (for W6)
`tests/synccheck_engine.rs` checks every `testutil::scenarios::all()` log
with `SynccheckConfig::default()`, whose `init.warps_per_cta` is 0, so any
`setmaxnreg` `Set{wg}` replays as `IncompleteWarpgroup`
(`setmaxnreg_launch_bounds` scenario). The engine run completes and W3's
`step` accepts the same commands with the launch's init. Either the test
passes `ResourceInit{warps_per_cta: shape.warps_per_cta(), cluster_warps,
..}` or synccheck takes it from the log (a `LaunchInfo` record in
`RecordingObserver`). The launch-bounds `Configure` is now logged as a
`Protocol` event with `actor: Host` before the CTA's warps run.
Also fixed engine-side (was W2's bug): `tcgen05.commit` and
`clusterlaunchcontrol.try_cancel` now list their deferred mbarrier
arrivals / complete_tx in `Protocol.issued` (synccheck reported a false
deadlock on `tcgen_cp_ld`).

### W2-11: parallel scheduler semantics (implemented)
- **Partition** = unit of ownership: one cluster, or one partition for the
  whole launch when it has launch-wide state (cooperative / `grid.sync`,
  `sync_words` / `wait_until`, mixed `cta_group`). Each partition owns its
  CTAs, a `SyncTable` and a `LaunchAux` (every resource is cluster-local),
  and an event buffer. Decided from the program only (never from the
  observer or the worker count).
- **Round**: every partition runs its CTAs against an `Arena` shard
  (private allocations in place; global/param through a copy-on-write
  4 KiB-stripe overlay over the round-start snapshot) on up to
  `RunConfig::workers` threads (launch-lifetime pool). Shards merge in
  partition order (later partition wins per byte); buffered events replay
  in partition order (`Access::seq` assigned at replay). With one resident
  partition the arena is used directly.
- **Cross-partition visibility**: another partition's global writes become
  visible at the round boundary (plan 2.4); within a partition, effects are
  synchronous (ruling 2). Global RMWs (`atom`, `red`, bulk/tensor
  reductions into global) inside a shard are **serial points**: the warp
  stops before the instruction, which executes after the merge in partition
  order (no lost updates, no rollback).
- Async op ids are partition-scoped (`(cluster + 1) << 40 | n`).
- Invariant (tested for workers 1/2/8/33, with and without word history):
  bit-identical outputs, statuses, stats and observer streams.

### W2-12: `tcgen05.ld .spcompress` (for W4 / coordinator)
Lowering emits `dsts` = mdata lanes then cdata lanes (dea8c9d), and W4's
`tcgen_ld_spcompress(values, valid, max, abs)` returns `(mdata ++ cdata)` in
the same order, but `TcgenLdArgs` carries neither `max` nor `abs`; the
engine fails closed on `.spcompress` until they are in the contract.

## W5-7: `FindingKind::AliasStaleRead`

Racecheck now ports the legacy `alias_stale_read` review advisory (deltas
P7). It reports through `FindingKind::Other("alias_stale_read")`. Please add
an `AliasStaleRead` variant.

Lowering (W1) must fill `SiteInfo::buffer` with the **logical** buffer name
of each access site, for example `A_shared` versus `B_shared` over one pooled
allocation. An explicit view of one buffer must keep that buffer's name.

Also for W1: a qualifier-less `mbarrier.arrive` keeps the PTX default
`.release.cta`, including on a peer CTA's barrier (deltas B7). Do not
default it to `.cluster`.

## W6-2 (2026-10-08): test-migration follow-ups

- **W2 `tcgen_cp_ld` synccheck error:** I could not reproduce it at or after
  0c67a9e. The scenario's stream conforms to the contract: `Alloc`, `Init`,
  `bar.sync`, `TcgenGroup` + `TcgenWork(Issue)`, then a commit with
  `TcgenWork(Commit)` + `Mbarrier(Issue)` + `issued{arrivals: 1}`, waits,
  per-lane `Load`/`WaitLd`, `Dealloc`. It is now Clean. The fault was in the
  explorer: before 0c67a9e, `TcgenWork` commands joined every commit's barrier
  into one projection, which was then searched. Nothing is needed from W2.
- **Launch-bounds `Configure`:** consumed from host-side (`Actor::Host`)
  `Protocol` events in `RecordingObserver::other`, and hoisted from any
  per-warp copy.
- **Request:** `RecordingObserver` should keep the `LaunchShape` from
  `begin_launch` (or a ready `ResourceInit`). Synccheck needs `warps_per_cta`
  (setmaxnreg pools) and the number of cluster participants, and today the
  caller has to pass them in `SynccheckConfig.init`. Workaround:
  `synccheck::resource_init(&LaunchShape)`.
- **tcgen05.alloc result:** the restored check compares each schedule's base
  with the reference run's. That is sound without engine data, because any
  differing schedule proves the base depends on the order. Recording the
  engine's observed base on the `Alloc` `ProtocolCmd` would let the finding
  name the run's actual value; this is optional.

## v2 conformance (W8, 2026-10-08)

Filed from the sweep in `docs/development/v2-conformance-status.md` (all canonical cases x
{numsim, racecheck, synccheck} under `NUMSIM_IMPL=v2`). Reproduce one row from `tirx_harness/`
after `source scripts/dev-env.sh` and `core-rs/numsim-py/build_dev.sh` with
`NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "<case>-<mode>"`. Owners are
W8's best guess; reassign freely.

### V2C-1 [oplib]: `tirx.ptx.prefetch` has no oplib implementation (tensormap / global L2 forms)

- Cases (37): `alphamoe_fp8_blockscale_qwen3next` (numsim/racecheck/synccheck), `blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin` (numsim/racecheck/synccheck), `bmm_fp8_rubin` (numsim/racecheck/synccheck), `cudnn_sm100_bsa_backward_blk128` (numsim/racecheck/synccheck), `cudnn_sm100_bsa_backward_blk64` (numsim/racecheck/synccheck), `cudnn_sm100_bsa_forward_blk64` (numsim/racecheck/synccheck), `cudnn_sm100_dense_blockscaled_gemm_persistent_amax` (numsim/racecheck/synccheck), `cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant` (numsim/racecheck/synccheck), `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` (numsim/racecheck/synccheck), `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` (numsim/racecheck/synccheck), `cudnn_sm100_dense_gemm_persistent_swiglu` (numsim/racecheck/synccheck), `cudnn_sm100_dsa_sparse_attention_backward` (numsim/racecheck/synccheck), `cudnn_sm100_flex_attention_backward` (numsim/racecheck/synccheck), `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` (numsim/racecheck/synccheck), `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` (numsim/racecheck/synccheck), `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` (numsim/racecheck/synccheck), `cudnn_sm100_moe_grouped_gemm_dglu_dbias` (numsim/racecheck/synccheck), `deepgemm_sm100_fp4_mqa_logits` (numsim/racecheck/synccheck), `deepgemm_sm100_fp4_paged_mqa_logits` (numsim/racecheck/synccheck), `deepgemm_sm100_fp8_bmm` (numsim/racecheck/synccheck), `deepgemm_sm100_fp8_gemm_1d1d` (numsim/racecheck/synccheck), `deepgemm_sm100_fp8_mqa_logits` (numsim/racecheck/synccheck), `deepgemm_sm100_fp8_paged_mqa_logits` (numsim/racecheck/synccheck), `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` (numsim/racecheck/synccheck), `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` (numsim/racecheck/synccheck), `deepgemm_sm100_m_grouped_fp8_gemm_masked` (numsim/racecheck/synccheck), `deepgemm_sm100_tf32_hc_prenorm_gemm` (numsim/racecheck/synccheck), `dense_blockscaled_gemm_sm107` (numsim/racecheck/synccheck), `flash_attention4_fp4` (numsim/racecheck/synccheck), `flash_attention_backward_sm100` (numsim/racecheck/synccheck), `flash_mla_sparse_fwd` (numsim/racecheck/synccheck), `grouped_gemm_masked_rubin` (numsim/racecheck/synccheck), `kda_backward_packed` (numsim/racecheck/synccheck), `msa_decode_multishape` (numsim/racecheck/synccheck), `sparse_flashmla_prefill_head128_small_topk_phase1` (numsim/racecheck/synccheck), `sparse_flashmla_prefill_head64_phase1` (numsim/racecheck/synccheck), `vsa_multishape` (numsim/racecheck/synccheck)
- Minimal reproduction: `bmm_fp8_rubin` / numsim: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "bmm_fp8_rubin-numsim"`
- Observed: analysis_incomplete/incomplete: Unsupported: tirx.ptx.prefetch ["tensormap=tensormap"]: no oplib implementation for tirx.ptx.prefetch ["tensormap=tensormap"] (site 14)

### V2C-6 [lowering]: implicit tensor-map slots share the canonical name of a buffer parameter (`v`)

- Cases (5): `cudnn_sm100_bsa_forward_blk128` (numsim/racecheck/synccheck), `cudnn_sm100_flex_attention_forward_hd256` (numsim/racecheck/synccheck), `cudnn_sm103_flex_attention_forward` (numsim/racecheck/synccheck), `msa_sparse_atten_fwd_nvfp4_kv_sm100` (numsim/racecheck/synccheck), `msa_sparse_atten_fwd_sm100` (numsim/racecheck/synccheck)
- Minimal reproduction: `msa_sparse_atten_fwd_sm100` / numsim: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "msa_sparse_atten_fwd_sm100-numsim"`
- Observed: ValueError: bad argument "v": argument does not match a TensorMap parameter

### V2C-7 [lowering + contract]: two kernels of one module declare different parameters with the same canonical name

- Cases (1): `gdn_cp_prefill_sm100` (numsim/racecheck/synccheck)
- Minimal reproduction: `gdn_cp_prefill_sm100` / numsim: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "gdn_cp_prefill_sm100-numsim"`
- Observed: InputError: NumSim input 'k_map' is bound more than once with different values

### V2C-8 [oplib]: TensorMapDesc cannot represent FP4 align16-padded element type from a host descriptor

- Cases (1): `sm100_fp8_fp4_mega_moe` (numsim/racecheck/synccheck)
- Minimal reproduction: `sm100_fp8_fp4_mega_moe` / numsim: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "sm100_fp8_fp4_mega_moe-numsim"`
- Observed: ValueError: tensor_map_l1_weights: undecodable tensor map: OpError { kind: Unsupported, message: "TensorMap element type float4_e2m1fn (shared layout Some(Align16Padded)) is not representable as a Dtype" }

### V2C-9 [interp (arena::addr) / lowering]: load/store of global address 0x7d00_0000_00xx (unmapped aperture)

- Cases (3): `fastcu_nvfp4_gemm_gb300` (numsim/racecheck/synccheck), `flash_attention4` (numsim/racecheck/synccheck), `gdn_prefill_sm100` (numsim/racecheck/synccheck)
- Minimal reproduction: `flash_attention4` / numsim: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "flash_attention4-numsim"`
- Observed: bad_address/error: Global address 0x7d0000000000 is not mapped (site 65)

### V2C-10 [lowering / interp]: tcgen05 matrix descriptor encoder receives address 0x0

- Cases (2): `nvfp4_gemm` (numsim/racecheck/synccheck), `sparse_flashmla_prefill_head128_phase1` (numsim/racecheck/synccheck)
- Minimal reproduction: `nvfp4_gemm` / numsim: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "nvfp4_gemm-numsim"`
- Observed: invalid_operand/error: tirx.cuda.tcgen05_encode_matrix_descriptor: address 0x0 is not a generic shared-memory address (site 32)

### V2C-11 [lowering / interp]: 16-byte vector access at offset 3076 of `state`

- Cases (1): `gdn_decode_bf16_wide_vec_t1` (numsim/racecheck/synccheck)
- Minimal reproduction: `gdn_decode_bf16_wide_vec_t1` / numsim: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "gdn_decode_bf16_wide_vec_t1-numsim"`
- Observed: misaligned/error: 16-byte access at offset 3076 of state is not 16-byte aligned (site 170)

### V2C-14 [sync / synccheck]: synccheck rejects setmaxnreg `IncompleteWarpgroup` (cf. sync-behaviour-deltas R1/R2, not the same rule)

- Cases (1): `cudnn_sm100_kda_bprop_f16` (synccheck)
- Minimal reproduction: `cudnn_sm100_kda_bprop_f16` / synccheck: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "cudnn_sm100_kda_bprop_f16-synccheck"`
- Observed: fixed synchronization program rejected Some(Issue(395)): RegPool(IncompleteWarpgroup { wg: 3 })

### V2C-15 [sync / synccheck]: synccheck `Cluster(UnexpectedParticipant)` (cf. sync-behaviour-deltas C1 membership)

- Cases (1): `fast_topk_clusters` (synccheck)
- Minimal reproduction: `fast_topk_clusters` / synccheck: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "fast_topk_clusters-synccheck"`
- Observed: fixed synchronization program rejected Some(Issue(1152)): Cluster(UnexpectedParticipant { warp: 28 })

### V2C-16 [racecheck]: `data_race` `missing_same_warp_lane_order` write/write within one warp

- Cases (2): `gdn_decode_bf16_wide_vec_mtp` (racecheck), `gdn_decode_fp32_mtp_warp` (racecheck)
- Minimal reproduction: `gdn_decode_fp32_mtp_warp` / racecheck: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "gdn_decode_fp32_mtp_warp-racecheck"`
- Observed: write_write conflict on bytes [3200..3204) of allocation 14: missing_same_warp_lane_order; write_write conflict on bytes [3216..3220) of allocation 14: missing_same_warp_lane_order

### V2C-17 [racecheck]: `data_race` `async_lifetime_not_drained` over a large shared range (cf. deltas X9/P6)

- Cases (1): `cudnn_sm100_kda_bprop_f16` (racecheck)
- Minimal reproduction: `cudnn_sm100_kda_bprop_f16` / racecheck: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "cudnn_sm100_kda_bprop_f16-racecheck"`
- Observed: write_read conflict on bytes [1728..229088) of allocation 69: async_lifetime_not_drained; write_read conflict on bytes [34496..261856) of allocation 69: async_lifetime_not_drained

### V2C-18 [racecheck]: racecheck no longer emits `alias_stale_read` (W5 deciding whether to port it)

- Cases (1): `stable_sort_topk_by_value` (racecheck)
- Minimal reproduction: `stable_sort_topk_by_value` / racecheck: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "stable_sort_topk_by_value-racecheck"`
- Observed: expected diagnostics [{"diagnostics": [{"anchors": ["tirx_kernels/ported/flashinfer/utils/topk_radix.py:29:1-29:55", "tirx_kernels/ported/flashinfer/utils/topk_radix.py:44:1-44:57"], "bytes": {"shared": "128-130,512-514,898-900,902-904,906-908,910-912,914-916,918-920,922 / actual [{"verdict": "clean", "diagnostics": []}]

### V2C-19 [interp]: an uninitialized read legacy reports is not reported under `ZeroAndReport`

- Cases (2): `flashinfer_rmsnorm_quant` (numsim/racecheck/synccheck), `gdn_decode_fp32_mtp_warp` (numsim/synccheck)
- Minimal reproduction: `flashinfer_rmsnorm_quant` / numsim: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "flashinfer_rmsnorm_quant-numsim"`
- Observed: expected diagnostics [{"anchors": ["<TensorLoad> buffer"], "bytes": {"register": "0-128"}, "category": "diagnostics", "kind": "uninitialized_read", "space": "register", "status": "review"}, {"anchors": ["tirx_kernels/ported/flashinfer/norm/rmsnorm_quant.py:195:1-195:67"] / actual null

### V2C-20 [interp]: legacy reports register-space uninitialized reads; v2 reports a different shared footprint

- Cases (1): `flashinfer_qk_rmsnorm` (numsim/racecheck/synccheck)
- Minimal reproduction: `flashinfer_qk_rmsnorm` / numsim: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "flashinfer_qk_rmsnorm-numsim"`
- Observed: expected diagnostics [{"anchors": ["<TensorLoad> buffer[0]"], "bytes": {"register": "4-6,8-10,12-14,16-18,20-22,24-26,28-30,36-38,40-42,44-46,48-50,52-54,56-58,60-62,68-70,72-74,76-78,80-82,84-86,88-90,92-94,100-102,104-106,108-110,112-114,116-118,120-122,124-126"}, "cat / actual [{"category": "diagnostics", "kind": "uninitialized_read", "status": "review", "space": "shared", "anchors": [], "bytes": {"shared": "32-512"}}]

### V2C-22 [interp / oplib]: outputs differ from legacy AND fail the independent reference

- Cases (1): `cudnn_sm100_kda_bprop_f16` (numsim)
- Minimal reproduction: `cudnn_sm100_kda_bprop_f16` / numsim: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "cudnn_sm100_kda_bprop_f16-numsim"`
- Observed: outputs differ: ['dgate']; reference_ok=False

### W8-6 [contract (sched)]: buffer parameters aliasing one host array

19 canonical cases bind the *same* host array (or overlapping views) to two
buffer parameters (`selective_state_update_*`: `x`/`z`,
`dst_indices`/`rand_seed`; cudnn gdn/kda: `scheduler`/`work_item_staging`;
...; full list under W8-6 in `docs/development/v2-conformance-status.md`).
Legacy mapped them to one engine allocation with per-parameter offsets, so
racecheck saw the aliasing and writes through one name were visible through
the other. `sched::run_with_config` allocates one allocation per
`ArgValue::Buffer` name, so v2 would silently give each parameter its own
copy; the binder therefore fails closed (`NotImplementedError`; rule in
`dev-loop.md`). Request: `ArgValue::View { target: String, offset: u64, len: u64 }`
accepted for `ParamKind::Buffer` slots (and as a `Pointer` target), bound
to the target's allocation at `offset`. The binder will then emit one
`Buffer` per distinct host memory region (the union of overlapping spans) and
a `View` per parameter, and read each parameter's output as a slice of the
region.

### V2C-23 [synccheck]: verdict on a truncated event log

When a launch stops early (runtime error or fail-closed `Incomplete`, e.g.
V2C-1/V2C-9), `synccheck::check` still explores the partial log and reports
`execution_error: executor deadlock; N blocked warps` (e.g.
`selective_state_update_stp_vertical`/synccheck, `bmm_fp8_rubin`/synccheck).
The explorer cannot tell a truncated log from a hang. numsim-py now gives the
engine's own error/incomplete precedence and keeps the checker's verdict as
`checker_on_truncated_log`, but the checker should fail closed itself:
request a `SynccheckConfig`/input flag (or a `RecordingObserver` marker from
`warp_done`) saying which warps never ended, so the explorer reports
`Incomplete { reason: "truncated_log" }` instead of a deadlock.

## W5-8: st.async / red.async proxy and release; `SignalProtocolError`

- **Proxy.** PTX §9.7.10.12 says "st.async is performed in the generic proxy",
  and §9.7.15.7 says the same for red.async. W2 must emit their `Access`es with
  `Proxy::Generic` (not `Async`) on `Actor::Async{op, side: Write}`, with
  `LaneSpan.lane` = the issuing lane.
- **Release.** The `.release.<scope>` global form (no mbarrier) is a strong
  release at that scope. Emit it with `sem: Release` and `scope`. Racecheck
  gives it a release head holding the issuer's knowledge at issue (deltas T2).
- **Kind.** Please add `FindingKind::SignalProtocolError`. The rule is
  confirmed: a race on bytes of a declared `wait_until` word where at least
  one side is a weak (plain) access. It is reported as `status=error`, with
  the race evidence and the bypass hint (deltas T3). Until the variant exists
  it uses `Other("signal_protocol_error")`.

### W8-6 update (2026-10-08): identical-span aliasing implemented in Python

Parameters bound to the *identical* host byte span (one array bound twice,
the common corpus case) now share one allocation without a core change: the
binder marks the later parameters `alias` of the first and specializes the
module so their `ParamSlot.aliases` name the first parameter, which
`run_with_config` already resolves to the same allocation
(`CompiledModule.with_buffer_aliases`). Only partially overlapping views
still fail closed and still need `ArgValue::View`.

### W8-7 [sched]: engine addresses of bindings before the run

Raw-pointer kernels (legacy `test_native_global_write_seed.py`: pointer
words such as `pointer_bits = [source.ctypes.data]` passed as data) need
the engine address of a binding to put into another input before the run.
Synthetic global VAs are deterministic, but only the arena knows the policy
(alignment, guard gaps, allocation order). Request a pure
`sched::plan_global_addresses(module, inputs) -> Result<BTreeMap<String, u64>, RunError>`
(same allocation order as `run_with_config`), which numsim-py will expose as
`Engine.address_of(module, inputs, name)`.

### V2C-24 [interp]: `Requirements.implicit_tmem` is ignored

`tests/numsim/v2/checkers/test_alias_advisory.py::test_tmem_view_keeps_one_logical_identity`
(and the partitioned / full-extent variants): a TMEM `decl_buffer(...,
allocated_addr=0)` store/load without `tcgen05.alloc`. Lowering sets
`requirements.implicit_tmem = true`; the engine stops with `Incomplete
"tmem[0]: tmem column 0 is not in a live allocation"` and the output reads
0. Legacy treated such programs as owning the whole TMEM.

### V2C-25 [sched]: `ArgValue::TensorMapOf` base is "not mapped"

`tests/numsim/v2/checkers/test_global_write_seed.py::test_tensor_map_initial_and_replaced_bases_keep_compact_alias_races[distinct-initial-base]`:
a host `numsim.TensorMap` over an array that is not otherwise a kernel
argument binds as `TensorMapOf { base: "target_map.__base__", offset: 0, .. }`
with that array as a `Buffer` argument; the TMA store faults with
`bad_address: Global address 0x7d0000000040 is not mapped`. Probably the
same root cause as V2C-9 (corpus `flash_attention4`, `fastcu_nvfp4_gemm_gb300`,
`gdn_prefill_sm100`, all host-descriptor TMA kernels). The `replaced-base`
variant reports `missing_proxy_bridge` on the descriptor bytes after
`tensormap.replace` + `fence.proxy.tensormap::generic` (racecheck).

## W4-9 (2026-10-08): v2 conformance V2C-1 / V2C-8 (oplib)

- **V2C-1 `tirx.ptx.prefetch*` / `applypriority*`** now resolve
  (`oplib/ptx/hints.rs`): every `prefetch`, `prefetch_valid_addr`,
  `prefetchu`, `applypriority`, `applypriority_async_bulk*` form W1 lowers
  through `lower_generic_ordering` (`Instr::Ptx`, no dsts). Modifiers are
  validated against the TVM table (bare tokens or `slot=token`; unknown /
  missing / duplicate fail closed); the op is a no-op with no `Access`
  (legacy `ptx_cache_hint`, `ordering_only`; `prefetch.tensormap` had no
  effect in legacy either). **W2**, two legacy checks need engine state and
  belong in the `Ptx` handler (or a dedicated variant) if they are wanted:
  `tirx.ptx.prefetch_valid_addr` validated that its address names at least
  one addressable global byte (legacy
  `validate_global_cache_hint_address(.., 1, 1, "prefetch.L1::32B.valid_addr")`),
  and `applypriority.async.bulk*` (`completion=bulk_group`) joined the
  thread's bulk async group (a later `cp.async.bulk.commit_group` /
  `wait_group` counts it). Without them both are plain no-ops (recorded as
  delta P6/P7).
- **V2C-8**: `TensorMapDesc` gained `fp4_padded: bool` (Default false): with
  `elem: Some(E2M1)` it selects the 16-byte-aligned padded FP4 shared layout
  (`CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B`) instead of the packed
  `16U4_ALIGN8B`. `decode` sets it, `encode` honours it, `tma_plan` plans it
  with the legacy padded unit (8 data bytes per 16-byte shared slot). Also
  `Dtype::U6` now maps to/from the `16U6_ALIGN16B` element type. W2/W8:
  host-side tensor maps built from a `CUtensorMap` data type 14 must set
  `fp4_padded = true` (the binder previously had no way to say so).

## W5-9 (for W1): logical buffer identity for `alias_stale_read` (V2C-18)

`stable_sort_topk_by_value` aliases one static shared array as `counters32`
(u32) and `counters16 = counters32.view("uint16")`. Legacy reports
`alias_stale_read` because the u32 reads observe bytes last written through
the u16 view.

v2 lowering defeats this in two ways:
- the shared buffers are lowered with **empty names**;
- the site identity is the **root of the view chain**, so `counters16` and
  `counters32` are one identity.

Requested rule: the logical identity is the root of the view chain, **but a
dtype-changing view (`.view("uint16")`) starts a new identity**, as a union
member does. Same-dtype reshapes and rearranges keep the root, which is what
the legacy "explicit view keeps one logical identity" tests expect. Every
buffer also needs a non-empty name. Racecheck now ignores unnamed sites (V12),
so with names missing this kernel stays clean rather than producing a false
advisory. No racecheck change is needed once lowering carries the names.

## W1 (2026-10-08): v2 conformance items tagged lowering, and W5-9

- **V2C-6, fixed (lowering).** Implicit tensor-map slots, the `TensorMap` slots
  whose `tensor_map` spec is set and which come from the host prelude, are now
  named `<prelude var>.tmap`.
  - If that name is taken, the slot gets `<var>.tmap<N>`.
  - `local_name` equals the canonical name, so the prelude var (`v`) is never
    a binding key and cannot shadow the buffer parameter `v`.
  - Binder rule for W8 (unchanged): slots with `tensor_map` set need no binding.
    The engine encodes them from `implicit_base`.
- **V2C-7, fixed (lowering); no contract change.** `ParamSlot::name` stays
  "unique within a Module" and keeps its meaning of one binding.
  - In a module with more than one kernel, `lower_module` renames a slot to
    `k<i>:<name>` (and keeps `local_name = <name>`) when either:
    - it holds a per-launch value (`Scalar`, `TensorMap`, or a Param-space
      byte blob), or
    - its declaration (kind, dtype, space) differs between kernels.
  - Buffer and pointer slots with matching declarations keep the shared name,
    because launches run in order on the same memory.
  - Implicit shape slots follow their buffer.
  - The existing binder already accepts `k<i>:<name>`. A bare name is
    ambiguous only when more than one kernel declares it.
- **V2C-11, fixed (lowering).** `Load`/`Store`/`AddrOf` offsets into a
  vector-dtype buffer (for example a `uint32x4` view over bf16 `state`) are
  now counted in `dtype.elem` units: each index is multiplied by the lane
  count. The engine was right to report misalignment.
- **V2C-9, partly lowering (fixed), partly engine-side (for W2).**
  - Fixed: `Tma.tmap_space` is now `Generic` when the operand is a u64
    address (PTX: the map's generic address in .param, .const or .global).
    The TVM table tags the operand `.global`, which turned `AddrOf` of a
    `__grid_constant__` map, an address in the param aperture, into a Global
    access. This resolves `flash_attention4` and `msa_sparse_atten_fwd*`.
  - Engine-side: `gdn_prefill_sm100` (`copy_desc`, line 444) and similar code
    execute a literal `ld.global.v4.b64` from `address_of(<tensor map
    param>)`. Lowering emits what the PTX says: `LoadAddr{space: Global}` on
    an `AddrOf(Param buf)` address.
  - W2 needs to decide whether a Global-space load whose address falls in the
    param aperture reads the Param buffer read-only, as hardware tolerates for
    grid-constant params, or reports it as a source bug. Lowering does not
    rewrite the space.
  - `fastcu_nvfp4_gemm_gb300` was not re-checked separately and is probably
    the same pattern.
- **V2C-10, engine-side (oplib/W2).** The address operand is correct. The
  kernels themselves encode a template descriptor from a null address, for
  example:
  - `nvfp4_gemm.py:474`: `sf_desc.init(reinterpret("handle", uint64(0)), ...)`;
  - `sparse_prefill_head128_phase1.py:491`: the same pattern.

  The kernels then mask the address field (`& ~0x3FFF`) and overwrite it. The
  hardware encoding is `(addr & 0x3FFFF) >> 4`, so 0 is legal. The
  `tcgen05_encode_matrix_descriptor` oplib entry should accept address 0, or
  more generally encode the bits without requiring a shared aperture.
- **W5-9, done.**
  - Every `BufferDecl` has a non-empty name.
    - Unnamed TIR buffers become `<space><index>`.
    - Unnamed views become `<parent>+<byte base>.<dtype>`.
    - Example: `stable_sort_topk_by_value` lowers to `shared2` (u32) and
      `shared2+0.u16`. The TIR names are empty in the source, so these are the
      names you will see.
  - `SiteInfo::buffer` walks the `view_of` chain only while the dtype is
    unchanged, and stops at the dyn-smem pool. A dtype-changing view is its
    own identity.

## W4-10 (2026-10-08): CLC query decode, V2C-10

- `tirx.ptx.clusterlaunchcontrol_query_cancel_{is_canceled,get_first_ctaid,
  get_first_ctaid_v4}` resolve (`oplib/ptx/clc.rs`): bytes 0..4 of the
  response are the linear base CTA id (x; y = z = 0, legacy linear launch
  domain); `is_canceled = low32 != 0 && low32 != 0xFFFF_FFFF`, so both the v2
  all-zero never-cancel representative and legacy's `0xFFFF_FFFF` decode as
  not cancelled (a real cancel never yields base CTA 0). W2: keep the
  all-zero response, or write `0xFFFF_FFFF` — both are "not cancelled".
- V2C-10: `tcgen05_encode_matrix_descriptor` no longer validates its address:
  a generic shared pointer contributes its window offset, anything else (0
  included) its own bits, encoded as `(addr & 0x3FFFF) >> 4`.

## W6-3 (2026-10-08): v2 conformance V2C-14 / V2C-15 / V2C-23

- **V2C-14 and V2C-15 have one root cause: the launch shape never reaches
  synccheck in numsim-py (W8).** `PerLaunchRecorder` (numsim-py
  `src/lib.rs`) forwards `access`/`sync`/`warp_done`/`inbox_drain` to its
  per-launch `RecordingObserver` but not `begin_launch`. As a result
  `RecordingObserver::launches` is empty, and synccheck had built
  setmaxnreg pools with 0 warps (`IncompleteWarpgroup`, V2C-14) and the
  cluster barrier with 0 participants (`UnexpectedParticipant`, V2C-15).
  Since W2-10, synccheck fails closed (`incomplete`, "launch shape unknown").
  **Fix (W8):** call `self.current.begin_launch(info)` in
  `PerLaunchRecorder::begin_launch` after resetting `current`.
  I verified this by replaying both kernels' captured v2 modules and inputs
  through `sched` with a plain `RecordingObserver`, which records the shape:
  - `fast_topk_clusters`: Clean in 0.9 s (8 CTAs × 32 warps, cluster 8,
    27k events).
  - `cudnn_sm100_kda_bprop_f16`: phase 0 and phase 1 both Clean (four
    setmaxnreg warpgroups, the `inc 144/168/144` and `dec 56` collectives).
  The cluster replay also showed that the certificate rejected every
  `Cluster::Exit`. Exits that are a warp's last cluster command are now
  accepted (regression `cluster_exit_after_last_wait_is_certified`), and the
  reference run picks transitions round-robin instead of recomputing every
  enabled transition. Before these two changes, checking this launch did not
  finish in 20 minutes.
- **V2C-23 (truncated log). Request:** `RecordingObserver` should record
  `warp_done` (`warp_ends: Vec<(WarpId, WarpEnd)>`) and whether `end_launch`
  was reached. A launch that stopped early (`WarpEnd::Budget`/`Error`/`Trapped`,
  or a warp with no end) would then be `incomplete` (`truncated_launch`)
  instead of Phase A's "executor deadlock" from `BlockedAtExit` or a Phase B
  deadlock on missing events. Only a `Deadlocked` end, or `BlockedAtExit`
  after a completed launch, stays a deadlock finding. Synccheck will use the
  field as soon as it exists. Without it, the log alone cannot tell a budget
  stop from a real hang.

## W2 phase 3 (2026-10-08): engine review + conformance sweep

### W2-13: exit-released named barriers (sync model / synccheck)

Count-less named barriers are exit-aware (ruling H1, PTX §9.7.14.7). When a
warp's last lanes exit, each open count-less generation it has not arrived
at receives an exit-time `Named` command through `SyncTable::step`, logged as
one committed `Protocol` event of that warp: `Arrive` (or `Red` when the
generation is a `.red` one), with `mask = live = the exiting lanes` and the
generation's `count`. Later count-less contributions use `b = 32 * (warps
in CTA - exited warps)`. No type changes; synccheck replays the event like
any other arrival. Explicit-count barriers are not released by exit: a hang
there ends as `Incomplete` ("named_barrier_after_exit (G8)"), never
Deadlock. A blocked `bar.sync`/`bar.red` logs the count it registered with.

### W2-14: replay order and the stream-cycle diagnostic (W5, W6, W8)

Sharded rounds replay partitions in an order consistent with what they
observed (`Arena::shard_replay_order`, ruling M10): a partition that read
global bytes another wrote in the same round replays first. When no such
order exists (store buffering: each read what the other wrote), partition
order is used and `RunOutcome::diagnostics` carries one
`Finding { kind: Unsupported, status: Incomplete, attrs.reason:
"cross_cluster_same_round_cycle" }` (only when observing; outputs are
unchanged). **Request:** checker drivers (numsim-py `execute`, W5/W6
reports) must treat that diagnostic as making the checker verdict
`incomplete`. Read tracking costs nothing when no observer is attached.

### W2-15: done in phase 3 (no contract change)

- W5-8 / review M5: `st.async` / `red.async` landings are `Proxy::Generic`,
  `sem: Release`, `atomic: true`, `scope` = the instruction's, on
  `Actor::Async{op, Write}` with the issuing lane (with or without an
  mbarrier; the mbarrier `complete_tx` stays the async completion).
  `AsyncIssue.proxy` is `Generic` too.
- W8-6: `ArgValue::View { target, offset, len }` (sched): one allocation per
  `Buffer` target, views bind `Buffer` slots at `offset` (length capped at
  `len`), work as `Pointer`/`TensorMapOf` targets, and come back in
  `Outputs` as their slice. **W8:** add the `("view", target, offset, len)`
  tuple to numsim-py `arg_value` (your file) and drop the fail-closed
  binder path.
- W4-9: `prefetch.L1::32B.valid_addr` checks one addressable global byte per
  executing lane (BadAddress otherwise); `applypriority.async.bulk*` issues a
  `Payload::None` bulk op into the thread's bulk group (commit / wait_group
  count it). Host tensor maps from a spec honour `force_cu_dtype`
  (11 TF32, 13 E2M1 packed, 14 E2M1 `fp4_padded`, 15 U6; any other value
  that is not the dtype's canonical code fails closed).
- V2C-9 / W1 ruling: `ld.global` (or a generic load) of a parameter-aperture
  address reads the param block; any store into param space is a
  `BadAddress` finding. `tests/numsim/v2/checkers/test_global_write_seed.py`
  `tensor_map_initial_and_replaced_bases...` now XPASS (V2C-25): **W8**, drop
  the xfail.
- V2C-14 ruling: without `Launch::regs_per_thread`, the initial setmaxnreg
  budget is the legacy caller base (min of 512/warpgroups, the largest
  `setmaxnreg.inc` target or 256, and 512/(warpgroups*min_blocks_per_sm)),
  configured per CTA and logged as the host `Configure` event.
- Found by gdn_prefill_sm100 once Seeded latency was restored (H6):
  `tcgen05.commit` now tracks EVERY prior in-flight tcgen05 op of the thread,
  not only those since its previous commit (a second commit with nothing new
  issued completed immediately). gdn_prefill_sm100 / gdn_cp_prefill_sm100
  numsim conformance pass.
- TMA with `.cta_group::2`: the mbarrier operand is a shared::cluster address;
  each destination's signal goes to CTA `(dst & !1) | (mbar_rank & 1)` (the
  pair CTA the address's bit 24 names). fastcu_nvfp4_gemm_gb300 now runs to
  completion (outputs identical across seeds/partitioning).

### W2-16: rulings needed (coordinator)

- **Remote mbarrier arrives through a shared::cta operand.** nvfp4_gemm:841,
  deepgemm fp8_fp4_gemm_1d1d:1434, sm100_fp8_fp4_mega_moe:1999,
  flash_attention_backward:1277 pass a `mapa.shared::cluster` result to
  `mbarrier.arrive{.expect_tx}.b64 _, [addr]` with NO state space. PTX says
  generic addressing then, but W1 lowers it `AddrSpace::Shared`, and the
  engine's shared::cta rank check (earlier ruling) rejects the remote rank
  (`bad_address: shared::cta address 0x380c0 names CTA rank 0, not the
  executing CTA (rank 1)`). Legacy accepted it. Either W1 lowers the
  sink-form (`_`) arrive without space as `SharedCluster` (the only space
  the sink form has besides generic), or the engine treats a rank-tagged
  address in a shared-space mbarrier operand as cluster-addressed.
- **V2C-19/20 (register-space uninitialized reads).** W1 scalarizes
  register buffers into registers, which carry no validity, so nothing is
  reported. Proposal: W1 lowers buffers legacy reported as `register` space
  as `Space::Reg` `BufferDecl`s accessed by `Load`/`Store`; the engine binds
  them like `Local` (per-lane, `Init::Uninit`, findings with `space:
  register`). Needs W1 + a small sched change; not done.

### W2-17: conformance failures that are not engine bugs (for owners)

- W4: `Op(Unsupported): unary BitNot on u64/u32` (deepgemm fp8_bmm,
  fp4_paged_mqa_logits, flash_mla_sparse_fwd, sparse_flashmla_prefill_*,
  msa_prefill_multishape); `tirx.ptx.cp_async_bulk_prefetch` has no oplib
  entry (alphamoe_fp8_blockscale_qwen3next); deepgemm mqa_logits_fp4:661 TMA
  writes past the shared window (`[376832, ..)` of 213844 bytes), likely the
  `sf_q` map's plan.
- W3: `tcgen05.alloc.exclusive` of 576 columns is rejected
  (`Tcgen(InvalidColumns{576})`; dense_blockscaled_gemm_sm107:1331,
  grouped_gemm_masked_rubin:1749).
- W8: cudnn_sm100_kda_bprop_f16 (V2C-22) outputs are byte-identical to legacy;
  only `dgate`'s dtype/shape view differs (uint8[512] vs float32[128]).
  flash_attention4 returns `O` in the base array's shape (1,256,32,128) while
  legacy returns the tensor-map view (64,256,64); the reference compares the
  latter. sparse_flashmla_decode_head64: missing binding `q_tail_tensormap`.

## W1 (2026-10-08, cont.): W2-16 rulings applied, W4 mqa_logits_fp4

- **Unqualified mbarrier operands (W2-16 ruling): done, with one refinement.**
  - Scope: `mbarrier.*`, `cp.async.mbarrier.arrive` and the `tcgen05.commit`
    mbar operand, when written with no state-space qualifier.
  - A 64-bit operand lowers as `Generic` and a 32-bit operand as
    `SharedCluster`. Lowering never forces `Shared`.
  - Refinement: a 64-bit register whose only writers are
    `Mapa{space: SharedCluster}` lowers as `SharedCluster`.
    - Reason: `mapa.shared::cluster.u64` returns a shared::cluster window
      address held in a u64, not a generic pointer.
    - Without this, nvfp4_gemm:512 failed as Global address 0x37858; with it,
      nvfp4_gemm runs to completion.
- **V2C-19/20 (uninitialized register reads): lowering done; one engine
  binding is needed before the space can become `Reg`.**
  - New pre-pass in `lowering/uninit.py`. It is a forward "definitely
    written" dataflow over the TIR, covering:
    - If-meet: the state after an If keeps only what both branches wrote.
    - Loops: first-iteration semantics; small constant loops (64 iterations or
      fewer, no break) are unrolled.
    - PTX `w` destinations, helper out-parameters and `wait_until`.
  - A local that may be read before it is written is kept as a memory buffer
    accessed by `Load`/`Store`. Every other local is still promoted to
    registers.
  - The memory space is set by `uninit.TRACKED_SPACE`, currently `Local`. The
    scheduler binds `Space::Reg` BufferDecls as `Unbound` (sched/mod.rs:567).
    - Once W2 binds `Reg` like `Local` (per lane, `Init::Uninit`), set
      `TRACKED_SPACE = "Reg"` and the findings will read `space: register`.
  - Results:
    - flashinfer_rmsnorm_quant and gdn_decode_fp32_mtp_warp now report
      `uninitialized_read` in `local` space.
    - flashinfer_qk_rmsnorm still reports only the shared footprint, so the
      register part of V2C-20 is still open.
    - Corpus: 433 of 2343 kernels have a tracked local (204 before). The new
      ones are mostly PTX microtests whose source operands really are
      uninitialized.
- **W4 mqa_logits_fp4: fixed (lowering).**
  - Sub-byte sizes were already correct (E2M1: numel × 4 bits).
  - The real bug: a `DeclBuffer` over another view's data (`data=view.data`)
    took `elem_offset` as relative to that view. It is relative to the shared
    data pointer, so the offset was counted twice: `smem_sf_q_2d` landed at
    188416 + 188416 = 376832.
  - `BufferDecl.base` now subtracts the parent's chain offset, and falls back
    to the chain root if the result would be negative.
  - Also: `finish_shared` now keeps every BufferDecl field when it rewrites
    root shared buffers. It used to drop `sync_words`.

## v2 conformance, sweep 2 (W8, 2026-10-08, at 62c4226)

Status of the earlier rows: V2C-1 (prefetch), V2C-3 (BitNot), V2C-6, V2C-8,
V2C-9/V2C-25 (TMA through host tensor maps), V2C-10, V2C-11, V2C-13 and the
W8-6 aliasing rows no longer reproduce; V2C-23 is fixed in synccheck
(b6c4254). numsim-py now forwards every observer callback to the per-launch
recorder (launch shapes, warp ends), binds overlapping host arrays as
`ArgValue::View`s of one region, and voids a checker verdict computed from a
truncated log. Current table: `docs/development/v2-conformance-status.md`
(numsim 81 match / racecheck 34 / synccheck 63 of 101). New rows:

### V2C-4 [synccheck]: synccheck could not build the fixed sync program

- Cases (6): `bmm_fp8_rubin` (synccheck), `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` (synccheck), `deepgemm_sm100_fp8_gemm_1d1d` (synccheck), `fastcu_nvfp4_gemm_gb300` (synccheck), `nvfp4_gemm` (synccheck), `sparse_flashmla_prefill_head128_phase1` (synccheck)
- Minimal reproduction: `nvfp4_gemm` / synccheck: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "nvfp4_gemm-synccheck"`
- Observed: expected diagnostics [{"diagnostics": [], "verdict": "clean"}] / actual [{"verdict": "incomplete", "diagnostics": [{"category": "incomplete", "kind": "analysis_incomplete", "status": "incomplete", "reason": "fixed_sync_program_build", "anchors": []}]}]

### V2C-5 [racecheck]: racecheck-behaviour-deltas P6 (`AsyncNeverCompleted` incomplete at launch end) -- verify

- Cases (35): `bmm_fp8_rubin` (racecheck), `cudnn_sm100_bsa_backward_blk128` (racecheck), `cudnn_sm100_bsa_backward_blk64` (racecheck), `cudnn_sm100_bsa_forward_blk128` (racecheck), `cudnn_sm100_bsa_forward_blk64` (racecheck), `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` (racecheck), `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` (racecheck), `cudnn_sm100_dense_gemm_persistent_swiglu` (racecheck), `cudnn_sm100_dsa_sparse_attention_backward` (racecheck), `cudnn_sm100_gdn2_recompute_f16` (racecheck), `cudnn_sm100_gdn_bprop_f16` (racecheck), `cudnn_sm100_gdn_prefill_f16` (racecheck), `cudnn_sm100_gdn_recompute_f16` (racecheck), `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` (racecheck), `cudnn_sm100_moe_grouped_gemm_dglu_dbias` (racecheck), `cudnn_sm103_flex_attention_forward` (racecheck), `deepgemm_sm100_fp4_mqa_logits` (racecheck), `deepgemm_sm100_fp8_bmm` (racecheck), `deepgemm_sm100_fp8_gemm_1d1d` (racecheck), `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` (racecheck), `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` (racecheck), `deepgemm_sm100_m_grouped_fp8_gemm_masked` (racecheck), `deepgemm_sm100_tf32_hc_prenorm_gemm` (racecheck), `dense_blockscaled_gemm_sm107` (racecheck), `fastcu_nvfp4_gemm_gb300` (racecheck), `flash_attention4` (racecheck), `flash_attention4_fp4` (racecheck), `flash_mla_sparse_fwd` (racecheck), `gdn_cp_prefill_sm100` (racecheck), `gdn_prefill_sm100` (racecheck), `grouped_gemm_masked_rubin` (racecheck), `msa_sparse_atten_fwd_sm100` (racecheck), `nvfp4_gemm` (racecheck), `sparse_flashmla_prefill_head128_phase1` (racecheck), `sparse_flashmla_prefill_head64_phase1` (racecheck)
- Minimal reproduction: `nvfp4_gemm` / racecheck: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "nvfp4_gemm-racecheck"`
- Observed: release .cta (warp 196) and acquire .cluster (warp 204) do not mutually cover each other's thread; allocation 7 ended while async op 1099511627784 still had it in its footprint

### V2C-28 [synccheck]: synccheck: fixed sync program model incomplete

- Cases (3): `deepgemm_sm100_fp4_mqa_logits` (synccheck), `deepgemm_sm100_fp8_mqa_logits` (synccheck), `msa_prefill_multishape` (synccheck)
- Minimal reproduction: `msa_prefill_multishape` / synccheck: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "msa_prefill_multishape-synccheck"`
- Observed: expected diagnostics [{"diagnostics": [], "verdict": "clean"}] / actual [{"verdict": "incomplete", "diagnostics": [{"category": "incomplete", "kind": "analysis_incomplete", "status": "incomplete", "reason": "fixed_sync_program_model_incomplete", "anchors": ["<unmapped op 4310>"]}]}]

### V2C-30 [interp (arena::addr) / lowering]: shared::cta address names another CTA rank

- Cases (2): `alphamoe_fp8_blockscale_qwen3next` (numsim/racecheck/synccheck), `flash_attention_backward_sm100` (numsim/racecheck/synccheck)
- Minimal reproduction: `flash_attention_backward_sm100` / numsim: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "flash_attention_backward_sm100-numsim"`
- Observed: ExecutionError: NumSim execution error: bad_address: shared::cta address 0x1000058 names CTA rank 1, not the executing CTA (rank 0) at /localhome/local-hongyij/TIRx-harness/.venv/lib/python3.12/site-packages/tirx_kernels/ported/flashattention/flash_attention_backward.py:1277

### V2C-31 [synccheck]: no result within the sweep timeout (explorer ignores the wall-time limit inside one projection)

- Cases (10): `cudnn_sm100_bsa_backward_blk64` (synccheck), `cudnn_sm100_dsa_sparse_attention_backward` (synccheck), `cudnn_sm100_gdn_bprop_f16` (synccheck), `cudnn_sm100_gdn_prefill_f16` (synccheck), `cudnn_sm100_gdn_recompute_f16` (synccheck), `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` (racecheck/synccheck), `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` (racecheck/synccheck), `fp16_bf16_gemm` (numsim/racecheck/synccheck), `gdn_prefill_sm100` (synccheck), `sparse_flashmla_prefill_head128_small_topk_phase1` (numsim/racecheck/synccheck)
- Minimal reproduction: `fp16_bf16_gemm` / numsim: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "fp16_bf16_gemm-numsim"`
- Observed: no result within the 900 s sweep timeout

### V2C-32 [oplib]: sub-byte (FP4/U4) TMA store fragments not representable in TmaPlan

- Cases (1): `blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin` (numsim/racecheck/synccheck)
- Minimal reproduction: `blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin` / numsim: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin-numsim"`
- Observed: ExecutionError: NumSim execution incomplete: analysis_incomplete: Op(Unsupported): sub-byte (FP4/U6) TMA store fragments are not representable in TmaPlan at /localhome/local-hongyij/TIRx-harness/.venv/lib/python3.12/site-packages/tirx_kernels/ported/flashinfer/fused_moe/blockscaled_contiguous_gather

### V2C-33 [lowering]: lowering rejects `wait_until` whose destination is not a promoted local

- Cases (2): `radix_topk_multi_cta` (numsim/racecheck), `sm100_fp8_fp4_mega_moe` (numsim/racecheck/synccheck)
- Minimal reproduction: `radix_topk_multi_cta` / numsim: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "radix_topk_multi_cta-numsim"`
- Observed: UnsupportedTIRxError: radix_topk_multi_cta: unsupported TIRx: site#16 ir.Call: wait_until destination must be a promoted local scalar; site#44 ir.Call: wait_until destination must be a promoted local scalar; site#103 ir.Call: wait_until destination must be a promoted local scalar

### V2C-34 [racecheck (contract: TMEM span convention)]: TMEM byte spans use a different addressing than legacy (lane * 2048 + col * 4); the snapshot's column projection cannot compare them

- Cases (1): `msa_sparse_atten_fwd_nvfp4_kv_sm100` (racecheck)
- Minimal reproduction: `msa_sparse_atten_fwd_nvfp4_kv_sm100` / racecheck: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "msa_sparse_atten_fwd_nvfp4_kv_sm100-racecheck"`
- Observed: TMEM lifetime conflict requires review: the earlier tcgen05.ld may not have completed before the conflicting reuse (read_write conflict on bytes [256..260416) of allocation 17: async_lifetime_not_drai; TMEM lifetime conflict requires review: the earlier tcgen05.ld may not have completed before the conflicting reuse (read_write conflict on bytes [320..260480) of allocation 17: async_lifetime_not_drai


## W4-11 (2026-10-08): V2C-32 sub-byte TMA stores

`TmaPlan` gained `global_bits: Vec<TmaBitFragment{global, smem, source_shift,
target_shift, mask}>` (Default empty): masked partial-byte global writes of a
sub-byte store (packed FP4 E2M1, U6), from legacy `plan_tiled_s2g` /
`apply_s2g_copy`. **W2**: for `Store` plans apply the byte spans first, then
each fragment `g = (g & !(mask << target_shift)) | (((s >> source_shift) &
mask) << target_shift)` with `s` = shared byte at `smem`, `g` = global byte at
`global` (each is a 1-byte global write/read-modify-write and a 1-byte shared
read for the observer). Packed FP4 stores have *only* fragments (every
element is a nibble). The 16-byte-aligned padded FP4 layout has no
shared-to-global copy in PTX; legacy rejected it and so does `tma_plan_dir`
(`Invalid`, "align16 padded FP4 TensorMap does not support shared-to-global
Tensor Copy"). Scatter4 of sub-byte types stays `Unsupported` (no legacy).

### Public-API triage (W8, 2026-10-08)

The 762-item public-API legacy set under `NUMSIM_IMPL=v2` (367 pass) is
triaged in `docs/development/v2-conformance-status.md` ("Triage of the
'other assertion' public-API failures"). New bug groups with owners there:
`%smid`/fetch registers report the CTA index (interp); f32 atomics and a
1-ulp rounding leak (oplib); tile reduction order / NaN / signed-zero
(oplib or tile lowering); U6 TMA layout and `.ignore_oob` fill (oplib);
multicast out-of-cluster targets `incomplete` instead of `error` (interp);
a racecheck **false negative** in `test_tcgen05_restricted_commit` (racecheck);
19 Synccheck `incomplete` verdicts on legacy-clean payload kernels
(synccheck); an explicit tensor map named `tensor_map.tmap` (lowering).
Ruling needed (coordinator, `arena::addr`): legacy kept the host pointer's
low 8 bits in global VAs and exposed device-validated aperture bits through
`mapa`/`cvta`; two tests assert them.

## W4-12 (2026-10-08): W8 triage, oplib items (atomics, reductions, U6, ignore_oob)

Changes to W2-owned files (call-site patches, review please):
- `interp/handlers/mem.rs::atom`: `.add.f32` without `.noftz` flushes
  subnormals only when the target is **not** shared memory (legacy
  `atomic_f32(.., space)`: global/generic-to-global flush, shared keeps
  denormals). Clears the width-1 numeric cases of
  `test_atomic_f32_noftz`.
- `.ignore_oob` dead bytes (`interp/aux.rs` `AsyncMeta.dead`,
  `interp/handlers/async_copy.rs::bulk_copy`, `sched/partition.rs`): the
  ignored left/right edges (inside the `.cp_mask`, if any) are written as
  zero and then invalidated, as legacy `raw_bulk_copy_g2s_cta_ignore_oob`
  does (bytes `0`, validity `false`). They are part of the async write
  footprint. A later read reports `uninitialized_read` (verdict review).
  `testutil::scenarios::bulk_masked_expected` and
  `interp_scenarios::bulk_copy_mask_and_ignore_oob_narrow_bytes_and_footprint`
  are updated to match.

oplib: TIR `Min/Max` on f64 is now `cuda_f64_min/max` (NaN-ignoring,
`-0 < +0`, the same as device `min.f64` and the legacy tile reductions).
Before, it was Rust `f64::min/max`, which returned `+0` for `min(-0, +0)`.

Not oplib (Decision 6, for the coordinator): v2 lowers `tirx.tile.*` through
TVM dispatch. The emitted code reduces warp collectives with a `shfl.bfly`
tree, and the `3input_maxmin` dispatch has no identity seed, so the result
for all-NaN input is the canonical NaN. Legacy emitted its own sequential,
identity-seeded reductions instead. `test_warp_collective_reduction_follows_physical_lane_ownership`,
`test_local_collective_uses_lexicographic_order` and
`test_maxmin_uses_canonical_lexicographic_nan_and_signed_zero_order` pin the
legacy-frontend semantics, and the v2 results are what the dispatched code
computes on hardware. These are expectation deltas, not oplib bugs.

W4-12 landed in 65bec68, including the edits to W2's files. The coordinator
ruled on the reduction tests: v2's behaviour stands. See rows R1/R2 (and D7)
in `docs/development/numsim-behaviour-deltas.md`.

## W4-13 (2026-10-08): `cp.reduce.async.bulk.tensor` op/dtype validation

- oplib: `tma_reduce_valid(AtomOp, Dtype)` implements the PTX table, the same
  as legacy `RawTmaReductionOp::resolve`. A unit test cross-checks it
  against that `resolve`. **W2**: `async_copy.rs` (`TmaDir::Reduce`) calls it
  and returns `Invalid` at runtime for undefined pairs.
- lowering (W1's `ptx_lower.py`, `lower_tma`): when the `tmap` operand names
  a host-encoded TensorMap and no raw `force_cu_dtype` is set, the same table
  is checked at transpile time and fails with `UnsupportedTIRxError`, as
  legacy did. Maps that are not static fall back to the runtime check.
  `test_partial_tile_forms::test_typed_tma_reduce_rejects_invalid_operation_dtype_pairs`
  is no longer marked xfail.

## W4-14 (2026-10-08): reviewed `cuda.func_call` helpers (W1 round 2)

`oplib/ptx/func_call.rs` resolves `tirx.cuda.func_call.<name>` for the
reviewed helpers, using the legacy `emit/cuda_helper.rs` bodies:
`combine_int_frac_ex2`, `flashkda_fmaf_rn`, `flashkda_rsqrtf`,
`flashkda_tanh_approx`, `gdn_lg2_approx_ftz`, `shl_u32_clamp`,
`tvm_builtin_fma_scale_sub_f32x2` and `tvm_builtin_smem_desc_add_16B_offset`.

- A helper resolves only when its `source_sha256` modifier equals the digest
  in `REVIEWED_HELPERS`, and the carrier widths are exact. Anything else is
  `Unsupported`, so oplib fails closed as lowering does.
- `numsim-core/src/oplib/ptx/func_call_tests.rs` has bit-exact tests built
  from the legacy test vectors.

Open, for W1:
- `tvm_builtin_cast_{float32x2_float16x2, float16x2_float32x2, ...}` are in
  `PURE_FUNC_CALLS`, but TVM's `cast_vec2` emits them as
  `void f(void* dst, void* src)`: they load a packed pair, convert it, and
  store it. A value op cannot model this; oplib answers `Unsupported`
  ("pointer-based helper"). Lower them like `float22half2`, with a load, a
  `Cast` (rn) and a store.
- `smem_desc_make_lo_uniform` is effectful: a `__shfl_sync` of `lo` from
  lane 0 through a pointer. Lowering rejects it today.

## W4-15 (2026-10-08): reserved descriptor bits are errors; `.ws` 64-bit zero-column mask

1. oplib: any lifted `numsim-oplib` error whose text mentions reserved bits
   is now `Invalid`, checked before the "unsupported" keyword. The case that
   prompted this is `raw tcgen05.cp descriptor uses unsupported
   reserved/base/LBO-mode bits`, which was classified `Unsupported`. Legacy
   reported it as an error. Fixes
   `test_shared_descriptor_choices_validate_the_consumed_bits[False-*]`.
2. ti16 `.ws` (W2's `interp/handlers/tcgen.rs`, a call-site patch to review).
   - Cause: W1 lowers the `.ws` zero-column mask as one 64-bit operand in
     `disable_output_lane`, but the handler cast it to `u32`. That dropped
     bits 32..63: the zero-mask form (bit 39), the per-bank spans and the
     B shift.
   - Effect: masked B columns were decoded, and the `0x7800` S1Z4M11 filler
     failed as "nonzero reserved bits".
   - Fix: a 64-bit operand of a `.ws` MMA is passed as `[low, high]` words,
     which oplib already accepts. The ti16 decoder itself is unchanged and
     matches legacy.
   - Fixes `test_tcgen05_ti16.py::test_ti16_ws_banks_and_column_mask[*]`
     (6 params).

## W4-16 (2026-10-08): W2-21 perf hot spots (oplib side) and W2-20 prefetch

- **Contract widening (W2: `sched/partition.rs::run_mma` already
  conforms).** The `tc_mma_ctas` `tmem_read` / `tmem_write` buffers may now
  cover several consecutive cells of one lane: cell `col + i` is bytes
  `4i..4i+4`, and a run never crosses a lane. oplib now reads and writes the
  D window, and packed TMEM A, as one call per lane run instead of one per
  cell. K-major B16 smem operands are read 16 bytes (8 elements) at a time
  where the layout keeps them contiguous. Both read exactly the same bytes
  as before, so footprints are unchanged. Values are bit-exact; a test
  compares the chunked and per-element gathers for every swizzle mode.
- **TMA plans:** an interior tile-mode box (no OOB element, byte-multiple
  element type, no interleave) reuses a cached plan of the same map at
  coordinates 0, shifted by `sum c_i * stride_i`. The cache is
  thread-local, keyed by (map, direction, smem offset). A test compares the
  cached and direct planners on interior and OOB boxes.
- **`tcgen_ldst_map`:** maps are cached per (shape, num, pack, warp,
  taddr) and shared through `Rc`. New `TcgenLdstMap::cell_runs()` lists the
  touched cells as per-lane runs of consecutive columns. W2: read and write
  each run with one TMEM access instead of one per piece. That is the
  remaining 167 us/op (#4).
- **Generic Ptx (#5):** `PtxFn`s are already resolved once per op at load
  (`interp/mod.rs`, `Loaded::ops`); nothing re-resolves.
  - The slow oplib ops in `fp16_bf16_gemm` were `numsim.pack` (bit-by-bit
    insert) and `tcgen05_encode_instr_descriptor` (string dtype parsing per
    lane). `pack` now uses word shifts. The encoder now reuses its result
    for repeated operand tuples, within a call and across calls on the same
    thread.
  - The rest of the 2.6 us/op is handler-side (scratch marshalling,
    timers).
- **W2-20 `cp_async_bulk_prefetch`:** both table spellings already resolve
  (`ptx/hints.rs`), and lowering emits `cp.async.bulk.prefetch` as an
  ordering-only instruction, not a `Ptx` op. `alphamoe_fp8_blockscale_qwen3next`
  now stops on `bad_address` at the remote `BULK_S2C` (kernel line 1453),
  which is not oplib.

## W4-17 (2026-10-08): W6 triage oplib gaps (ftz maps, spcompress, interleave prefetch, wide no-offs im2col)

- **`TensorMapDesc.elem_ftz`** (new pub field, default false): `elem` F32/TF32
  with the FTZ data type (`tensormap.replace .elemtype` 8 / 12,
  `CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ` / `TFLOAT32_FTZ`). It round-trips
  through encode, decode and replace, and plans as the non-FTZ type. Struct
  literals need `..Default::default()` or the new field.
- **`tcgen05.ld .spcompress`**, modelled end to end:
  - Lowering (`ptx_lower.py`): the spcompress-only form carries its
    `rowop` as `red = (op, [])`; `.abs` is in `red_abs`. No schema change:
    `red` with no registers means "selection op only".
  - Handler (W2's `interp/handlers/tcgen.rs`): writes
    `oplib::tcgen_ld_spcompress` (metadata words, then kept values) into
    `dsts`; the `.red` value is still the fold over the loaded words.
- **Tensor prefetch** (W2's `interp/handlers/async_copy.rs`): no longer
  plans. The new `oplib::tma_prefetch_check` only checks the instruction
  rank against the descriptor rank, as legacy `execute_tma_cache_hint`
  did. A swizzled 16B-interleave map can therefore be prefetched; an
  issued transfer still fails with `tma_swizzled_16b_interleave_unmodeled`.
- **`im2col_no_offs::w`** (lowering): now `TmaMode::Im2colW`, the wide
  layout. It was `Im2colNoOffs`, which planned as spatial and hit "layout
  mismatch (wide)".
- Not oplib:
  - The synccheck/racecheck `analysis_incomplete` finding for
    `tma_swizzled_16b_interleave_unmodeled` has an empty message, so
    `report.format()` lacks the reason. The reason is in
    `details["reason"]`.
  - `test_im2col_store_and_reduce[*-*-False]` (and `test_im2col_interleaved`)
    read `outputs["tmap"]`: the public API's output naming for a
    TensorMap-bound array.

## W4-18 (2026-10-08): `tirx.log1p` / `tirx.sigmoid`; tensormap.replace ordinals

- `resolve_ptx` now implements `tirx.log1p` and `tirx.sigmoid` (no
  modifiers; one scalar float in, the same type out: f32, f64, f16, bf16).
  The kernels are in `numsim_oplib::scalar::math` (see delta D9).
  - W1: no lowering change. `builtins.py` already lowers both as pure
    `Ptx "tirx.log1p"` / `"tirx.sigmoid"` with the value argument.
  - Neither op is in the legacy `SUPPORTED_OPS.md` tables, so there is
    nothing to regenerate.
- `TensorMapDesc::replace`: a per-dimension ordinal outside the
  descriptor's slots (5 dimensions, 4 stored strides) is `Invalid`.
  - Ordinals between the current rank and the last slot stay legal, as in
    legacy. The GDN descriptor kernels and
    `test_raw_descriptor_copy_replace_release_and_acquire_drive_raw_tma`
    rewrite all five `global_dim` slots on a rank-2 map. A rank-based
    bound would reject them, so the bound is the slot count.
  - Test: `replace_rejects_ordinals_outside_the_descriptor_slots`, which
    also checks that no ordinal panics.

## W4-engine-stops (2026-10-08): sweep-5 `engine-stops` class (22 items, 19 functions), oplib share

Run against the live tree:
- 13 functions stop with `bad_address: ... tmem lane N is outside warp 0's
  sub-partition`. With `test_launch_resource_facts::test_exclusive_tmem_uses_cta_local_lifecycle_without_placement`
  (`Tcgen(AllocWhileExclusive)`, sync delta T8) these are W2's 14 TMEM-slice
  functions, already ruled as deltas; W9 is porting them.
- 3 functions stop with `RegPool(InvalidDirection)`: setmaxnreg.dec, ruled
  as a delta.
- `test_gate_intrinsics::test_gate_intrinsics_match_float32_semantics`: the
  only oplib item (`tirx.log1p` / `tirx.sigmoid` were unimplemented). Fixed
  by W4-18; its numerics pass. It now fails only on the
  `module.rust_source` pin (pin-internals, W9).
- `test_tile_owner_transport::test_copy_transports_unique_owners_across_warps`:
  passes in the live tree. It was the lowering's register-owner `Assert`
  trap, fixed on the W1 side.

No engine (interp/sched/arena) item remains for W2 beyond the ruled deltas.

## W5-10 (for W2, 2026-10-08): restricted commit and tcgen smem operand proxy

Found with `test_tcgen05_restricted_commit` (a racecheck false negative) and
V2C-5.

1. **`tcgen05.commit…sync_restrict::shared::read::mma::a`.** This commit
   completes when the shared-memory **A** operand reads of the tracked MMAs
   are done. It does not wait for the B reads or the D writes. Today
   `tcgen_commit` ignores `sync_restrict`: `preds` and the arrival cover the
   whole MMA, and the MMA is dropped from `tcgen_uncommitted`. As a result:
   - the B overwrite in the test is not reported;
   - the next unrestricted commit tracks nothing. In the trace, the commit at
     barrier offset 64 has `preds: []`.

   Requested shape (legacy `TcgenPipelineOperation::MmaSharedARead`):
   - Issue the MMA's shared-A read as its own async op `ma`, with
     `class: TcgenPipelined`, `preds: []` (never the MMA itself), and only the
     A spans as its `Read` accesses. The MMA op keeps the B reads and the D
     read/write.
   - A restricted commit's `preds` = the uncommitted `ma` ops. Keep the MMA
     ops in `tcgen_uncommitted` for the next unrestricted commit, which tracks
     the MMA and `ma`.

   The checker test `racecheck_tcgen::restricted_commit_publishes_only_shared_a_read`
   pins this shape. No racecheck change is needed once the events have this
   shape.
2. **smem operand reads of `tcgen05.mma` / `tcgen05.cp` use `Proxy::Async`.**
   They are emitted as `Proxy::Tcgen` today, for example op 780 in the
   restricted-commit trace, which reads `alloc7 [512, 5120)`. The tensor core
   reads shared memory through the async proxy (PTX ISA, tcgen05 memory consistency model). With the
   wrong proxy, a generic overwrite after the wait loses the
   `fence.proxy.async` requirement. Legacy reports that case as
   `missing_proxy_bridge`. TMEM accesses stay `Proxy::Tcgen`.
3. **Done, no action:** `tcgen_commit` now tracks all in-flight tcgen ops, so
   the commit `preds` gap found on `nvfp4_gemm` is fixed. The checker's
   transitive closure (delta T7) stays as a guard.

## W6-4 (2026-10-08): round-2 conformance (V2C-4 / V2C-28 / V2C-31)

1. **`Collective.participants` (W2).** For cta_group::2 tcgen05 operations (`bmm_fp8_rubin`), each record of a collective lists only its recording warp. The contract (`observe.rs`) says participants is the full member set. Local workaround: synccheck takes the union, over records with the same id, of the declared participants and the recording warps. A record that is genuinely missing is still `fixed_sync_program_build`, and the message names the missing warps. Request: list every member in each record, or document "self only" as allowed.
2. **Public-API payload rows are not synccheck gaps.** Of the 19 rows (status doc, "bug | synccheck | 19"), none comes from the explorer:
   - `test_payload_runtime[*]`: the `incomplete` is `launch_not_executed`. The shim runs the whole module up to `phase_index`, and launch 26 (`scalar_warp_intrinsics`) stops on `Unsupported: tirx.cuda.sm100_2sm_leader_smem_addr` (oplib/lowering). Every later phase then inherits that `incomplete`. Owner: Python shim (run only the requested phase, or keep going after an `incomplete` launch) plus oplib for the intrinsic. Separately, 26 more params fail with `KeyError: 'task_count'`: the shim's `stats` lacks `task_count` (it has `completed_task_count`). Owner: Python shim.
   - `test_shared_descriptor_choices[False-*]`: `Op(Unsupported): raw tcgen05.cp descriptor uses unsupported reserved/base/LBO-mode bits`. Legacy reported an error ("reserved"). The oplib should report reserved descriptor bits as an operation error, not `Unsupported`. Owner: W4.
   - `test_reported_instruction_support[red_vec_packed_bf16]`: `Unsupported: numsim.pack ["ty=BF16x4"]: pieces [BF16, BF16] do not tile bf16x4`. Owner: lowering (W1).

## W1 round 2 (2026-10-08): V2C-33, V2C-7 remainder, lowering rejects, fail-closed gaps, W4-12

- **V2C-33: fixed.** `wait_until` destinations always stay registers.
  - `uninit.py` pins them, and the destination counts as written before the
    predicate reads it.
  - `radix_topk_multi_cta` and `sm100_fp8_fp4_mega_moe` lower again; the
    corpus test passes.
- **V2C-7 remainder: fixed, no contract change.**
  - The cause: one Module bundles unrelated kernels whose same-named buffer
    params (`high`, `out`, ...) are different arrays.
  - Rule: in a multi-kernel Module, every slot name that more than one kernel
    declares becomes `k<i>:<name>` (`local_name` stays the signature name);
    names declared by a single kernel stay bare.
  - Kernels that really share memory are bound to the same host array. The
    binder's identical-span aliasing (W8-6) maps them onto one allocation.
  - Result: all 42 listed items now bind. The remaining failures are not
    lowering:
    - 26 assert legacy `stats.task_count`;
    - 15 are synccheck phase runs reporting "an earlier launch stopped the
      module" (W6/W8; a plain numsim run of the whole module completes);
    - the tensor-map override item passes now that the implicit map keeps
      its prelude var as an alias.
- **Implicit tensor maps:** `<var>.tmap` slots also list the prelude var in
  `aliases` when no other slot uses that name, so a caller override such as
  `tensor_map` still binds. Explicit TensorMap params keep their declared
  name; a test covers this.
- **"lowering rejects the kernel" (29 items): 12 cleared, 17 out of scope or
  needing a contract change.**
  - Cleared:
    - VECTORIZED loops lower as serial loops (4);
    - `tirx.timer_finalize_cuda` is a Nop (1);
    - register-fragment layouts are supported, with storage offset = `m` and
      a runtime `Assert` that the owning laneid / tid_in_wg is the executing
      thread (6);
    - the host-extent fail-closed message now names the "unsupported integer
      operation" (1).
  - Contract-needed:
    - sub-word TMEM direct access (f16/u8/u16 cells, 7): the Tmem
      `BufferDecl` ruling only allows 32-bit cells;
    - replicated TMEM views (4);
    - runtime TensorMap box/element strides (2).
  - Out of scope:
    - `ptx_legacy` (2);
    - TVM dispatch semantics differ from the legacy expectation (1);
    - TVM's copy fallback does cross-thread register access, which our owner
      Assert rejects (1).
- **Fail-closed gaps (W9, `v2-accepts-legacy-rejection`): 17 of 17 now
  raise.**
  - Reviewed `cuda.func_call` helpers: `builtins.REVIEWED_HELPERS` holds
    sha256 digests taken from the legacy `*_SOURCE` constants, plus argument
    and result dtypes, and lowering checks both.
    - The mutated-body tests now fail with "body does not match the validated
      ..." and the wrong-dtype test with "requires ...".
    - For W4: the reviewed helpers themselves (`flashkda_*`, `shl_u32_clamp`,
      `combine_int_frac_ex2`, `gdn_lg2_approx_ftz`, `fma_scale_sub_f32x2`,
      `tvm_builtin_cast_*`) have no oplib implementation, so the positive
      tests stop "incomplete".
  - Legacy reinterpret validation (dtype_registry classes). Scalar fp8
    identity reinterpret is rejected again.
  - `lowering/tile_checks.py`, run before TVM dispatch:
    - unknown tile config keys (union of the legacy per-op key lists);
    - directed float64 rounding (TVM silently emits round-to-nearest);
    - the warp `tile.gemm` mma.sync m16n8k{16,8} fragment ABI (the legacy
      check, ported).
  - `__shfl*_sync` width is checked: a constant must be a power of two in
    [1, 32], otherwise lowering rejects it; a runtime width gets an `Assert`.
  - The TMEM-at-exit, tensormap-release and ignore_oob items pass from engine
    changes.
- **W4-12 (1) tile reductions: TVM-side, not our invocation.**
  - TVM's own `LowerTIRx` pipeline fails the same way on
    `shared_cta_accum_sum`, `shared_cta_f16_reductions` and
    `shared_empty_axis_reductions`.
  - The errors are "undefined variable tid_in_scope" / "undefined variable
    threadIdx.x", raised in
    `tvm/backend/cuda/tile_primitive/reduction/shared.py` (cta scope).
  - These are TVM dispatch bugs (category E), not a scope or layout we pass
    wrongly.
- **W4-12 (2) vector atomics: fixed.** `lower_atom` emitted `Atom` without
  the instruction guard. A guarded atom/red now runs inside `If(pred)`, and
  so do its destination write-backs. The `test_atomic_f32_noftz` numerics
  pass; the remaining failures assert the legacy `rust_source`.
3. **Implicit bulk commit at exit is not logged (W2).** `interp/handlers/control.rs` (exit) steps `AsyncGroup(Exit)` on every open Bulk group but emits no `Protocol` event for it. The explorer then sees open bulk groups at exit and raised `UncommittedAtExit` on `flash_mla_sparse_fwd`, `bsa_backward_blk128` and `sparse_flashmla_prefill_head{64,128}_phase1`. Local workaround: `synccheck::backend::exit_lint` applies `Cmd::Exit` to bulk groups itself. Request: log the exit command like the cluster `Exit`, so the recording is complete.

## W5-11 (for W2, 2026-10-08): `cp.async.mbarrier.arrive[.noinc]` must complete the tracked cp.async ops

Cause of every `missing_inter_actor_sync` race on `cudnn_sm100_gdn_{prefill,recompute,bprop}_f16`,
`cudnn_sm100_bsa_backward_blk64`, `cudnn_sm100_dsa_sparse_attention_backward`,
`gdn_prefill_sm100` (V2C-5 rows). These kernels are exactly the corpus users of
`cp.async.mbarrier.arrive`. Trace (`gdn_prefill_f16`, kernel 1): warp 8
issues 32 per-lane `cp.async.ca.shared.global` ops (`AsyncIssue … class: Copy,
targets: []`), then `cp.async.mbarrier.arrive.noinc` (site 4203: only
`Mbarrier(Issue)`). The arrival later fires and warps 0–3 `Wait` phase 0 of
that mbarrier and read the bytes, but **no `AsyncComplete` is ever delivered
for the cp.async ops**. The checker therefore sees their writes as never
published, and reports a race on the reads. `cp_async_mbar_arrive` pushes only
the `MbarArrive` completion.

Request: when the deferred arrive fires, deliver `AsyncComplete { op,
milestone: Write, target: Phase { obj: <mbar>, phase } }` for every cp.async op
of that lane that the arrive tracks (PTX: all prior `cp.async` operations
initiated by the executing thread). Deliver them before (or with) the
arrival that completes the phase, as `cp.async.bulk` complete_tx already does.
The checker path is covered by `racecheck_async_copy::cp_async_mbarrier_arrive_pending_count`.
No racecheck change is needed.

## W1 (2026-10-08): contract batch 4 (items 28–30) and W4-14

- **28, sub-word TMEM: done in lowering.**
  - An 8- or 16-bit TMEM view now gets `shape = [lanes, cells]`, where
    `cells = ceil(element columns / per_cell)`. TIR counts the TCol of these
    views in element units.
  - Offset = `lane * cells * per_cell + element column`.
  - A static column origin must be cell-aligned; a runtime origin is divided
    by `per_cell`.
  - Engine results:
    - `test_tmem_subword_views_alias_one_physical_cell` passes.
    - Four tests stop with `bad_address: <buf>[idx]: tmem lane L is outside
      warp W's sub-partition`, where L is twice the expected lane:
      `test_tcgen_cp_cta_group2_supports_float16_payloads`,
      `test_tcgen_cp_supports_rank3_multi_instruction_layout`, and two
      left_tmem gemm/raw-cta2 tests.
    - Example: `tcgen_float16_cta_group2`, `physical` is F16 with
      `shape [128, 4]` (cells). `physical[256]` is row 32, col 0, i.e. TMEM
      lane 32, but the engine reports 64.
    - The lowered offsets follow item 28 exactly, so I am handing this to W2
      to check the sub-word lane computation in the loaded binary.
- **29: done.** Direct access to a replicated TMEM view emits
  `Unsupported { reason: "tmem_replicated_view: <buffer>" }`. It applies to
  4 tests (`scale_tmem` in 3 tcgen_cp tests and mxfp4).
- **30: done.** `TensorMapSpec.box_dim` and `element_stride` are DimExprs.
  - Example: `host_encoded_dynamic_integer_tensor_map` gives
    `box_dim = [Add(FloorDiv(Param 2, 2), 20)]`.
  - Both runtime-box tests now lower. What remains is the test port: one
    calls the legacy `module.load()`.
- **W4-14 (1): done.** `tvm_builtin_cast_<s>x2_<d>x2(dst, src)` no longer
  goes through func_call.
  - The body must equal TVM's own `cast_vec2._intrinsic_source` text.
  - It lowers to two element `Load`/`LoadAddr`, two `Cast{rnd: Rn}` and two
    `Store`/`StoreAddr`.
  - `test_right_aligned_buffer_broadcast...` now runs. Its remaining mismatch
    (64/1024 elements, 1 ulp) is in the division part, i.e. TVM-dispatch
    division semantics, not the cast.
- **W4-14 (2): done.** `smem_desc_make_lo_uniform(uint64_t*)` is checked
  against the legacy digest (`344b73c0cc918023`) and signature
  (`['handle'] -> void`). It lowers to: load the u64, then
  `Shfl{Idx, lane 0, clamp 0x1f, full mask}` of the low word, then store
  `(hi & 0xffffffff00000000) | lo`. Verified end to end on a 32-lane run.

## W1 (2026-10-08, W5/W6 round-2 items)

- **`red_vec_packed_bf16`: fixed.** `lower_atom` packed vector sources as
  scalars of the element type. The pieces now tile the access type exactly:
  `red.v2.bf16x2` is two `bf16x2` pieces packed into `bf16x4`, and `.v4.f32`
  is four `f32` pieces. A vector result unpacks into one piece per
  destination. All `test_reported_tool_regressions` cases pass.
- **Tracked locals are `Space::Reg`.** `uninit.TRACKED_SPACE` is now `"Reg"`.
  W2's `BufBinding::Reg` is in HEAD: per-lane, `Init::Uninit`.
  - `flashinfer_rmsnorm_quant`, `gdn_decode_fp32_mtp_warp` and
    `selective_state_update_mtp_vertical` now report `uninitialized_read`
    with `space: "reg"` on `regs[w*]`.
  - For W8: the conformance projection must map `reg` to legacy `register`.
  - `flashinfer_qk_rmsnorm` (V2C-20) still reports only its shared finding.
    The bf16 register bytes legacy flags (`buffer[0]`, bytes 4-6, 8-10, ...)
    are not read uninitialized by the lowered program; that row is open.
- **Runtime tensor-map box (item 30), status of the two tests:**
  - `tests/numsim/integration/test_host_prelude.py::test_dynamic_tensor_map_expressions_run_in_the_loaded_artifact_prologue`
    needs a v2 port (W9): it calls the legacy `CompiledModule.load()`.
  - `tests/numsim/integration/test_host_prelude.py::test_dynamic_tensor_map_prologue_is_shared_by_native_checkers`
    lowers, but synccheck reports `invalid_operand: NumSim TensorMap image
    has invalid magic`. That is W2's bind-time encode of a map whose
    `box_dim` is a runtime `DimExpr`.

## W2 phase 4 (2026-10-08)

### W2-18: arch-dependent `.exclusive` TMEM limit (W6)

The engine creates each CTA pair's tcgen05 lifecycle state with
`tcgen::State::new(exclusive_max)`: 576 columns when `Program.arch` starts with
`sm_107`, 512 otherwise (PTX Table 58). Synccheck builds `State::default()`
(512), so it rejects a legal sm_107f 576-column `.exclusive` alloc. **W6:** take
the limit from the arch, or the coordinator adds it to `ResourceInit`. The
`tcgen_exclusive_576_sm107` scenario is in `scenarios::special()` until then.

### W2-19: buffer-form TMEM accesses under racecheck (W5)

`implicit_tmem` / `tmem_subword` (scenarios) access TMEM through `Space::Tmem`
buffers: warp actor, `Proxy::Tcgen`, synchronous in the engine. Racecheck
reports a same-lane write/read conflict (`missing_same_warp_lane_order`) even
across `tcgen05.wait::st`. **W5:** rule how buffer-form TMEM Load/Store (legacy
`TensorLoad` on a TMEM view) orders against itself.

### W2-20: reassigned conformance / public-API items (evidence)

| item | owner | evidence |
| --- | --- | --- |
| `sparse_flashmla_prefill_head128_small_topk_phase1` never ends | **W1** | Guarded `clusterlaunchcontrol.query_cancel.get_first_ctaid` (`pred=canceled`) is lowered as `Ptx{pred, keep_dst: false}` (pc 1087 of the dumped module), so a not-cancelled response writes 0 instead of keeping the 0xFFFF_FFFF sentinel; `jobs.valid` never clears. With `keep_dst: true` on every guarded Ptx the kernel completes in 82 rounds / 0.6 s (legacy 0.9 s). The engine also now writes the legacy "no cluster" response (first word 0xFFFF_FFFF). |
| `flash_attention_backward_sm100:1277` | **W1** | Source says `mbarrier.arrive.expect_tx.shared__cluster.b64(remote_mbar, ..)`; the module has `AddrSpace::Shared`, so the explicit-shared::cta rule (sync-semantics §2.1) rejects the remote rank. |
| tcgen05 ops with a PTX `pred=` (10 `test_tcgen_inactive_boundaries` params) | **W1** | `Instr::TcgenMma` has no guard; a guarded `T.ptx[mma](.., pred=...)` must be lowered inside an `If` (as W1 did for atom/red, W4-12). The handler then never validates an off instruction. |
| zero-step `For` | **W1** | The engine sees a structured loop with no step; emit an `Assert(step > 0)` for runtime steps. |
| `cuda.__shfl_sync` width validation | **W1** | PTX `shfl` accepts any segment mask; the CUDA width rule belongs to the intrinsic's lowering (emit an `Assert`). |
| `mxf8_cta2` uninitialized `smem[0..4)` | **W1** | The TMEM view's `allocated_addr=address[0]` (base_reg) is loaded at the `decl_buffer` (line 40), before `tcgen05.alloc` writes it. Load the base register at first use, or after the alloc. |
| `host_prelude` box_dim | **W1** + W2 (fixed) | The engine now sign-extends scalar params by their declared type before evaluating DimExprs (an int32 -7 arrived as 0xFFFF_FFF9). Separately, the prelude's `T.truncdiv` is lowered as `FloorDiv` (31 instead of 32 for delta = -7). |
| `sm100_fp8_fp4_mega_moe:70` | **W1/W8** | `st.global.u8` lowered as a `Store` into `symm_buffer` at element `-791038`: the symmetric-buffer pointer arithmetic produces an address outside that view (raw-pointer data; see W8-7 `plan_global_addresses`). |
| `cudnn_*_amax`, `*_dsrelu_quant` | **W3** | `RegPool(IncompleteWarpgroup{wg:1})`: the CTA has 6 warps and warpgroup 1 runs setmaxnreg; the sync model rejects a partial warpgroup, legacy accepted it (V2C-14). |
| `alphamoe_fp8_blockscale_qwen3next` | **W4** | `tirx.ptx.cp_async_bulk_prefetch` has no oplib entry. |
| `st/red.async.release` racecheck `incomplete` | **W5** | The mbarrier-less release form has no completion event, so racecheck reports the op as never completed. Rule what publishes it (ISA: release semantics only). |
| `tma_atomicity` (`Global address 0x2bfb81e0 is not mapped`) | **W8** | A raw host pointer inside a host tensor-map image (`ArgValue::TensorMap`); bind it as `TensorMapOf`. |
| `readonly_proxy` | **W5** | The engine emits `.nc` loads as `Proxy::ReadOnly`; the rule "no write overlapping read-only-path bytes in the same kernel" is an observation over both orders, which racecheck already sees. Proposed: racecheck reports it. |
| missing_proxy_bridge after W5-10 #2 | **W5** | The sampled finding (gdn_prefill_sm100) is TMA (line 1251) vs `tensormap.replace` (line 419) on the descriptor bytes, not an MMA operand. The kernel has `fence.proxy.tensormap::generic.release/acquire` (lines 1294/1245). Per R3 the MMA smem operand read IS async proxy, so the engine change stands. |
| register-space uninit reports (V2C-19/20) | done (W2) / **W1** | `Space::Reg` buffers are now memory-backed per lane and report `space: register`; W1 flips `uninit.TRACKED_SPACE` to "Reg". Legacy reports at the register use; v2 reports a TMEM/shared read into registers at the memory read (tf32_hc_prenorm, ssu_mtp_vertical): a documented reporting-point delta. |
| divergent blocking wait / divergent `__syncwarp` | delta | Both stay `divergent_block` incomplete. Neither is provable from engine state: lanes outside the arm can only run after the `If` (no Else), and on hardware they might still reach a matching wait or syncwarp. This is the documented divergent-switch limitation. |

### W2-21: perf hot spots (Mega-MoE-sized corpus kernels, interp)

`fp16_bf16_gemm` (16 CTAs x 256 threads, cluster 2, 200 rounds, 1.39M instrs).
No oracle: legacy cannot lower it. Release build, instrumented timers.

NoopObserver, 8.8 s -> 5.2 s after this phase's engine fixes:

| # | hot spot | cost | owner |
| --- | --- | --- | --- |
| 1 | tcgen05.mma numerics `oplib::tc_mma_ctas` | 2.8 ms/MMA, 26%. Per-4-byte-cell closure calls (~100k per MMA) | W4: bulk operand/tile read API |
| 2 | TMA handler (`tma_plan_dir` + per-span `resolve_global`) | 0.89 ms/op, 24% | W4 plan cost, W2 resolve |
| 3 | TMA landing `copy_spans` | 0.66 ms/op (~1.7 us per 16-byte swizzled span: check_oob + overlay lookup per span) | W2 |
| 4 | `tcgen05.ld` (`tcgen_ldst_map` + per-piece `mem_read`) | 167 us/op, 12% | W4 map, W2 per-piece reads |
| 5 | Generic `Ptx` ops | 2.6 us/op over 151k ops | W4 PtxFn dispatch |

Fixed in this phase (W2):
- `apply_completions` (39%): empty async groups queued milestone completions that could never be enabled, about 10k per CTA, and the queue was rescanned from the head after every application.
- MMA memory glue: direct cell access, and spans are merged as they are recorded instead of sorting 50M entries.
- An all-valid bulk copy is now one store.

RaceObserver: 147 s for the same 200 rounds (engine ~5 s). More than 95% is in RaceObserver callbacks, which are W5 internals and were not profiled per the rule. Engine-side under observation:
- `tcgen05.mma` handler 1.37 ms: the W5-10 shared-A footprint probe runs oplib twice and is cached per descriptor; misses dominate (W2: amortize).
- MMA landing 2.0 ms.

#### W2-21 phase 5 (engine perf, W2)

Uninstrumented release build of `fp16_bf16_gemm` at 200 rounds. 27b485c and the current tree were run back to back on the same host, at load average about 10.

| mode | 27b485c | now | instrs |
| --- | --- | --- | --- |
| NoopObserver | 5.07 s | 1.55 s (3.3x) | 1,390,361 -> 1,350,095 |
| RaceObserver | 161.5 s | 154.9 s | 1,350,095 both; 207 findings both |

The instruction count is now the same under every observer. Before, the W5-10 shared-A read op existed only when observing. It now always exists, and only its footprint is computed when observing.

Changes:

1. **Shared-A footprint cache.** Keyed by (descriptor bits without the start address, start mod 1024, idesc, cta_group). The cached footprint is stored relative to A's start, so a K-loop that advances the start reuses it. The footprint is computed only when observing.
2. **TMA landing.**
   - The root cause of hot spot #3, and most of hot spot #2, was copy-on-write stripe creation. It copied a 4096-bit validity mask one bit at a time, about 9 us per stripe. Every round's fresh shard overlay paid this for each global stripe a TMA store or reduce touched.
   - `BitSet::slice` / `copy_bits` are now word-level, used in stripe creation and in `merge_shard`.
   - `copy_spans` merges contiguous runs and stores each with one memcpy and one validity range.
   - Per-span observer events are unchanged.
   - Result: landing 706 -> 58 us per op, handler about 1 ms -> off the top list.
3. **TMA handler.** The global window is resolved once per op, and later spans are offsets within it.
4. **`tcgen05.ld`.**
   - One register buffer for all lanes.
   - Each lane's pieces are read as contiguous TMEM runs, with one liveness check and one read per run. A run that is not live, or has invalid bytes, falls back to per-piece reads, so errors and uninit findings are unchanged.
   - Spans are recorded only when observing.
   - Result: 188 -> 49 us per op.

Remaining NoopObserver profile (instrumented, 1.68 s):

| item | cost | share |
| --- | --- | --- |
| MMA landing (`oplib::tc_mma_ctas` numerics, W4) | 0.6 ms/op | 34% |
| generic `Ptx` | 2.6 us/op | 22% |
| `tcgen05.ld` | 49 us/op | 11% |
| TMA landing | 58 us/op | 5% |

RaceObserver time is still more than 95% inside W5 callbacks.

## W6-5 (2026-10-08): W2-18 / V2C-14

1. **W2-18 (done in synccheck).** `SynccheckConfig::tcgen_exclusive_max: Option<u32>`. Request to the numsim-py owner: set it from `Program.arch` via `sched::exclusive_tmem_columns`. Without it, synccheck uses the largest `.exclusive` width the run committed (at least 512), which is sound because the engine already validated each width against the arch. The `tcgen_exclusive_576_sm107` special scenario can now assert synccheck Clean (`tests/interp_checkers_smoke.rs`, W2).
2. **V2C-14 (rule clarified, reference and sync changed).** `setmaxnreg` by the trailing warps of a CTA (warp count not a multiple of 4) stays `IncompleteWarpgroup`. `WarpgroupSync` for such a tail is a no-op in `numsim-sync-ref::setmaxnreg` and `numsim-core::sync::setmaxnreg` (sync-semantics §7.4). Optional for W2: `interp/handlers/sync.rs` could skip crediting a group with fewer than 4 warps (`n < 4`). The model now tolerates it either way.

## W1 (2026-10-08): W2-20 lowering items

1. **Guarded `Ptx` keeps its destinations: fixed.** A `@p` op now emits
   `keep_dst: true`, and memory write-backs run under `If(p)`.
   `sparse_flashmla_prefill_head128_small_topk_phase1` completes, because the
   CLC sentinel survives.
2. **`flash_attention_backward_sm100:1277`: fixed.**
   - The op is a `cp.async.bulk.shared::cluster.shared::cta` whose mbarrier
     operand the TVM table tags `.shared`. PTX puts it in `.shared::cluster`,
     the destination CTA's barrier.
   - A bulk copy with a shared::cluster destination now gets a
     `SharedCluster` completion barrier (cvta'd if it is generic).
   - The case runs to completion.
3. **Guarded tcgen05 ops: already inside `If(pred)`** (`c.emit`). The inactive
   runs pass numsim and synccheck.
   - The 10 `test_tcgen_inactive_boundaries` params now fail only on
     `finding.details["operation"]`, a legacy report field (W2/report).
4. **Zero-step `For` and shuffle width: fixed.**
   - A runtime step gets `Assert(step > 0, "For step must be positive")`.
   - A constant step of 0 or less is rejected at lowering.
   - `__shfl*_sync` width: `Assert`, as in round 2.
   - Also fixed: `continue` inside a `for` skipped the increment (infinite
     loop). With a `continue` in the body, the increment now sits at the loop
     head and the variable starts one step early. The
     `test_loop_bounds` numerics pass; the remaining failures pin
     `rust_source`.
5. **`mxf8_cta2`: fixed.** A TMEM view with a runtime `allocated_addr`
   recomputes `base_reg` immediately before every access, not at the
   `decl_buffer`. `test_mxf8_cta2` passes 4/4.
6. **Host-prelude truncdiv/truncmod: fixed without a contract change.**
   - For a constant divisor `b > 0`: `trunc(a/b) = FloorDiv(Max(a,0), b) +
     CeilDiv(Min(a,0), b)`. `b < 0` negates the result, and
     `truncmod = a - b*trunc`.
   - A runtime divisor fails closed. That would need `DimExpr::TruncDiv`,
     which no corpus kernel requires.
7. **`mega_moe` negative element index: fixed.** A same-dtype, constant-offset
   view of a global buffer is now addressed in its root buffer
   (`elem_base` + index), like a C pointer: indices may reach before or past
   the view. Its logical identity was already the root.
   `sm100_fp8_fp4_mega_moe` runs to completion.

### W8-8 [sched]: allocate unreferenced buffer arguments

A host `numsim.TensorMap` image passed to a plain `T.Buffer((128,), "uint8")`
parameter (`tests/numsim/runtime/test_tma_atomicity.py`, non-`T.TensorMap()`
variants) addresses a host array that is not itself a kernel argument. The
binder passes that array as an extra `ArgValue::Buffer` and rewrites the
image's pointer from `plan_global_addresses`, but `allocate_host` only
allocates arguments some slot (or `View`/`Pointer`/`TensorMapOf`) references,
so the base gets no address. Request: allocate every `ArgValue::Buffer` in
`Inputs` (in a deterministic order after the referenced ones) and report it
in `plan_global_addresses` and `Outputs`. Until then numsim-py raises
`NotImplementedError` (W8-8) for this binding. The `T.TensorMap()` variant
binds through `TensorMapOf` and is unaffected.

## W5 (2026-10-08): responses to W2-19 and W2-20

- **W2-19 (buffer-form TMEM), done in racecheck.**
  - A warp-lane `Proxy::Tcgen` witness is a synchronous TMEM access, so hb orders it: program order, plus thread synchronisation across threads. The tcgen pipeline view no longer judges it.
  - Ordering against asynchronous tcgen05 ops is unchanged.
  - Delta T14. Test: `racecheck_buffer_tmem` (`implicit_tmem`, `tmem_subword` are now race-free). W2 can drop the caveat in `interp_checkers_smoke`.
- **W2-20 (1), st.async / red.async without an mbarrier: done.**
  - The strong generic release write makes the op complete at issue: no `AsyncNeverCompleted`, no `AsyncLifetime`. No `AsyncComplete` is expected.
  - Delta T13. Test: `g6_async_release_without_completion_event_is_complete`.
- **W2-20 (2), `readonly_proxy`: declined for racecheck. It belongs to the engine.**
  - Legacy rejects the kernel in NumSim itself. `tests/numsim/runtime/test_readonly_proxy.py` runs `assert_rejected`, which requires `NumSimExecutionError("write overlaps readonly bytes")` from `Engine.run` (no observer attached), plus the same reason in both checkers.
  - A racecheck finding cannot satisfy the NumSim half. Once the engine raises the execution error, every tool reports it anyway.
  - PTX (`ld.global.nc`): the bytes must be read-only for the whole kernel, so this is a property of program execution in either order, not an ordering question. Request: the engine keeps per-launch read-only-proxy byte ranges and rejects any overlapping write in either order with that message. No `ReadonlyProxyViolation` kind is needed.
- **W2-20 (3), `missing_proxy_bridge` on tensormap bytes: these were false positives, now fixed in racecheck.**
  - The engine emits the descriptor read as a warp-lane `Proxy::TensorMap` access at TMA issue. Racecheck only applied the acquired ranges to async readers.
  - Write-then-read: the reading lane's own acquired ranges now judge it.
  - Read-then-write: this is now ordered by hb, since the ISA defines the tensormap fence only for generic→tensormap.
  - On gdn_prefill_sm100 the kernel's release/acquire pattern is correct, so both directions are now clean.
  - Delta I9. Tests: `racecheck_tmap_lane`.

## v2 conformance, sweep 3 (W8, 2026-10-08, at 27b485c + W8 fixes)

Current table: `docs/development/v2-conformance-status.md` (numsim 88 / racecheck 59 / synccheck 83 of 101 match;
public-API set 484 of 762 pass). Resolved since sweep 2: V2C-1, -3, -4, -5, -14, -15, -18, -19, -20, -29,
-30, -32, -33, -34 rows no longer appear (rows of the 12 V2C-35 cases may be masked by that crash). V2C-35 is a regression from the `arena::addr` low-bits ruling and
needs a decision before more rows can be judged. New rows:

### V2C-35 [coordinator ruling + sched]: vector access at buffer offset 0 is misaligned: the engine now keeps the host pointer's low 8 bits (`arena::addr` ruling, `Inputs.host_addrs`) and numpy arrays are only 16-byte aligned; legacy ran these

- Cases (12): `cudnn_sm100_gdn2_bprop_f16` (numsim/racecheck/synccheck), `cudnn_sm100_gdn2_prefill_f16` (numsim/racecheck/synccheck), `cudnn_sm100_gdn_bprop_f16` (numsim/racecheck/synccheck), `cudnn_sm100_gdn_prefill_f16` (racecheck/synccheck), `cudnn_sm100_gdn_recompute_f16` (numsim/racecheck/synccheck), `cudnn_sm100_kda_bprop_f16` (numsim/racecheck/synccheck), `flash_mla_sparse_fwd` (synccheck), `flashinfer_fused_dit_layernorm` (numsim/racecheck/synccheck), `kda_backward_packed` (numsim/racecheck/synccheck), `sparse_flashmla_decode_head64` (numsim), `sparse_flashmla_prefill_head128_phase1` (racecheck/synccheck), `sparse_flashmla_prefill_head128_small_topk_phase1` (numsim/racecheck)
- Minimal reproduction: `kda_backward_packed` / numsim: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "kda_backward_packed-numsim"`
- Observed: ExecutionError: NumSim execution error: misaligned: 32-byte access at offset 0 of h0 is not 32-byte aligned at /localhome/local-hongyij/TIRx-harness/.venv/lib/python3.12/site-packages/tirx_kernels/kda/kda_backward_packed.py:4061

### V2C-36 [racecheck]: new `scope_mismatch` (+ `data_race`) findings where legacy was clean (cf. racecheck-behaviour-deltas R4/B1 scope rules) -- verify

- Cases (8): `bmm_fp8_rubin` (racecheck), `cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant` (racecheck), `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` (racecheck), `deepgemm_sm100_fp8_gemm_1d1d` (racecheck), `fastcu_nvfp4_gemm_gb300` (racecheck), `flash_attention_backward_sm100` (racecheck), `nvfp4_gemm` (racecheck), `sm100_fp8_fp4_mega_moe` (racecheck)
- Minimal reproduction: `nvfp4_gemm` / racecheck: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "nvfp4_gemm-racecheck"`
- Observed: release .cta (warp 196) and acquire .cluster (warp 204) do not mutually cover each other's thread

### V2C-37 [racecheck]: new `data_race` where legacy was clean or review

- Cases (6): `cudnn_sm100_dsa_sparse_attention_backward` (racecheck), `deepgemm_sm100_fp4_mqa_logits` (racecheck), `flash_attention4` (racecheck), `gdn_prefill_sm100` (racecheck), `msa_prefill_multishape` (racecheck), `vsa_multishape` (racecheck)
- Minimal reproduction: `vsa_multishape` / racecheck: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "vsa_multishape-racecheck"`
- Observed: read_write conflict on bytes [0..16384) of allocation 6: async_lifetime_not_drained

### V2C-38 [racecheck]: same finding kinds, different source anchors (witness site pair) and/or footprint

- Cases (9): `cudnn_sm100_bsa_forward_blk128` (racecheck), `cudnn_sm100_bsa_forward_blk64` (racecheck), `cudnn_sm100_gdn2_recompute_f16` (racecheck), `cudnn_sm103_flex_attention_forward` (racecheck), `flash_mla_sparse_fwd` (racecheck), `gdn_cp_prefill_sm100` (racecheck), `msa_sparse_atten_fwd_nvfp4_kv_sm100` (racecheck), `sparse_flashmla_prefill_head64_phase1` (racecheck), `stable_sort_topk_by_value` (racecheck)
- Minimal reproduction: `flash_mla_sparse_fwd` / racecheck: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "flash_mla_sparse_fwd-racecheck"`
- Observed: expected diagnostics [{"diagnostics": [{"anchors": ["tirx_kernels/ported/flashmla/sparse_prefill_head64_phase1.py:1383:1-1383:42", "tirx_kernels/ported/flashmla/sparse_prefill_head64_phase1.py:714:1-714:39"], "bytes": {"shared": "229376-229380"}, "category": "advisories" / actual [{"verdict": "review", "diagnostics": [{"category": "advisories", "kind": "alias_stale_read", "status": "review", "space": "shared", "anchors": ["<unmapped op 164>", "tirx_kernels/ported/flashmla/sparse_prefill_head64_phase1.py:1383:1-1383:42", "tirx


## W1 (2026-10-08): sweep 3, `SiteInfo.text` — done

- Every site's `text` is now the statement's source text: the lines of its
  innermost span with a readable file, whitespace-collapsed, at most 200
  characters (legacy source_map). Without a readable span, it is the op name,
  or the node kind for non-call statements.
- A view whose name differs from its logical buffer appends
  ` [view <name>]` (W9).
- Corpus sweep: 217,222 sites, 0 with empty text. The `tma_atomicity` `T.ptx.cp`
  sites carry their source line.

## W5-12 (for W2, 2026-10-08): `tcgen05.alloc` writes the TMEM address with one lane

On `sparse_flashmla_prefill_head128_phase1` (kernel.py:587/591), `tcgen05.alloc.cta_group::2.sync.aligned`
writes `tmem_start_addr` in shared memory, and every lane then reads it with `ld.shared.u32`.
The engine emits the write as a single-lane generic write. With no `__syncwarp`, every other lane's
read is a `missing_same_warp_lane_order` race. The instruction is `.sync.aligned`, a warp-collective
convergence point whose result all participating lanes may use. Request: emit the write with the
active lane mask as writers (each lane writes the same word), or emit `SyncKind::WarpSync { mask: active }`
right after the instruction. Racecheck needs no change.

## W5 (2026-10-08): V2C-38 TMEM review footprints (for W8)

`tmem_lifetime_review` keeps one representative per static (load site, store site) pair: the first
one delivered (legacy `record_review_findings`; deltas T16). Which instance is delivered first
depends on the schedule, so its column footprint can differ from legacy's. Examples:
`cudnn_sm100_bsa_forward_blk{64,128}`, `cudnn_sm103_flex_attention_forward`, `flash_attention4`
(v2 `256-512` vs legacy `256-320`). Request: compare `tmem_lifetime_review` by kind and anchors only,
not by `tmem-columns`.

## W2 (2026-10-08): V2C-35, W8-8, W4-16, W5 (c06149c), AsyncIssue.restricted

1. **V2C-35 (reversed ruling).** A top-level buffer's synthetic base is a fresh 4096-aligned allocation, so it is at least 256-aligned like `cudaMalloc`. The host pointer's bits are ignored. `Inputs.host_addrs` is still accepted, but placement no longer uses it. A `View` keeps its byte offset in its region.
   - `alloc_global_low_bits` was removed.
   - Scenario: `global_bases_are_aligned_and_views_keep_offsets`.
   - Corpus: 28 of the 36 rows of the 12 cases now pass. The misalignment crash is gone everywhere.
   - The 8 remaining rows were unmasked by the crash:
     - racecheck findings diffs (W5): `cudnn_sm100_gdn_bprop_f16`, `flash_mla_sparse_fwd`, `kda_backward_packed`, `sparse_flashmla_decode_head64`, `sparse_flashmla_prefill_head128_phase1`, `sparse_flashmla_prefill_head128_small_topk_phase1`;
     - `sparse_flashmla_decode_head64` numsim and synccheck: `named_barrier_contract_mismatch` `PartialWarp { mask: 0xfffffffe }` at a non-`.aligned` `barrier.sync` (kernel line 1743). This is the sync-isa-answers Q3/Q5 model, and needs a ruling.
   - Legacy `test_pointer_bits_preserve_binding_address_and_subview_offset` cannot pass under this rule.
     - It binds `storage[offset:offset+32]` of an *unbound* `storage` and expects the absolute host pointer's low 8 bits.
     - Only the first iteration (host-aligned offset) passes.
     - Proposal: rewrite the test to bind `storage` too, or to assert offsets relative to the bound region. Coordinator/W8 to decide.
   - `test_mapa_and_cvta_expose_the_device_validated_integer_bits` is unrelated: shared-window aperture bits, with no host pointers.
2. **W8-8.** `allocate_host` places every `Buffer` / `View` argument after the referenced ones, in name order. Planned addresses of referenced buffers do not move. Unreferenced ones are returned unchanged in `Outputs`.
   - Scenario: `unreferenced_buffer_arguments_are_allocated`.
   - The `NotImplementedError` guard in `v2/run.py::_patch_descriptor_pointers` can now go (Python owner).
3. **W4-16.**
   - `tcgen05.ld/st` use `TcgenLdstMap::cell_runs()`. Each live, fully valid run is read once into an image. `st` patches the image and writes it back once; validity is unchanged.
   - Every other piece takes the per-piece path, with the same errors and findings.
   - The MMA TMEM closures accept multi-cell buffers and reject one that runs past the lane's last column.
   - The two oplib NaN tests are **not** an FP-environment problem. They fail only in `--release`; the debug build passes 125/125. Each test runs on a fresh thread with the default MXCSR.
   - In release, the scalar reference `mul_add` is inlined to a hardware FMA form whose operand order the compiler picks. With two NaN inputs, AMD EPYC 7763 then returns a different payload. Back to W4.
   - `alphamoe_fp8_blockscale_qwen3next`: no `bad_address` at the current tree. numsim and synccheck pass; racecheck has a findings-only diff. Please send a reproduction command if it still happens.
4. **W5 (c06149c) TMEM footprint.** Not a 39b678e regression.
   - Rust builds of the core at 27b485c, aed5b1b and 39b678e were run under the current Python.
   - `gdn_prefill_sm100` racecheck shows the same `tmem_lifetime_review` (`async_lifetime_not_drained`, columns 0-512) at aed5b1b and at 39b678e, with identical diffs. 27b485c has them as well. The current tree has fewer.
   - Scenario `tcgen_ldst_spans_are_exact_on_both_paths` (`tcgen_ld_wide`) asserts that `tcgen05.ld/st` TMEM spans equal exactly each lane's cells, identically on the run path and the per-piece path.
5. **Readonly proxy.** The engine rejects any global write overlapping bytes read through `ld.global.nc`, in either order, with "write overlaps readonly bytes".
   - It covers sync stores, atomics, `st.bulk`, and async landings (TMA, bulk, reductions, `st.async`, dead bytes and bit fragments).
   - A conflict between partitions in the same round is caught at merge.
   - Tracking is per kernel launch, when the program has `nc` loads or `requirements.readonly_proxy` is set.
   - Scenarios: `readonly_proxy(clean|disjoint|after|before|cross_cta)`, with 1 and 2 workers.
   - `test_readonly_proxy.py` and the `test_needs_kernel_batch` readonly items pass; 2 of those were `xpassed`, so their xfail marks can go.
6. **`SyncKind::AsyncIssue::restricted`.** Set from `Issue::restricted`: `true` only for the `tcgen05.commit .sync_restrict` issue, `false` everywhere else.

## v2 conformance, sweep 4 (W8, 2026-10-08, at 5895aa2)

Matches out of 101: numsim 96, racecheck 87, synccheck 95 (B7 delta snapshots for six racecheck rows,
`tmem_lifetime_review` compared by kind + anchors, schema 4). Public-API set: 489 of 762 pass. Open rows:
V2C-36 (4 racecheck: `flash_attention_backward_sm100`, `sparse_flashmla_prefill_head128{,_small_topk}_phase1`,
`sm100_fp8_fp4_mega_moe` -- the last pending its R4/B1 delta), V2C-38 (`gdn_prefill_sm100`,
`gdn_cp_prefill_sm100` anchors), V2C-22 (`deepgemm_sm100_tf32_hc_prenorm_gemm`), V2C-28
(`msa_prefill_multishape` synccheck), and V2C-39 below (pending). V2C-35 is gone with the address
reversal; W8-8 is resolved by d5b0f09.

### V2C-39 [interp + sync (in progress)]: PENDING: W2/W6 are changing the non-aligned `barrier.sync` partial-warp rule this case depends on; it currently stops with `bad_address` on unmapped global address 0x1000000a9800 (re-check after that change)

- Cases (1): `sparse_flashmla_decode_head64` (numsim/racecheck/synccheck)
- Minimal reproduction: `sparse_flashmla_decode_head64` / numsim: `NUMSIM_IMPL=v2 $PY -m pytest -q -n 1 tests/conformance -k "sparse_flashmla_decode_head64-numsim"`
- Observed: ExecutionError: NumSim execution error: bad_address: Global address 0x1000000a9800 is not mapped at /localhome/local-hongyij/TIRx-harness/.venv/lib/python3.12/site-packages/tirx_kernels/ported/flashmla/sparse_decode_head64.py:2191


## W2 (2026-10-08): partial-warp non-aligned named barriers (Q3 ruling)

- A non-`.aligned` `barrier.{sync,arrive,red}` executed by a strict subset of a warp's non-exited lanes no longer fails at once (`interp/handlers/sync.rs::barrier_partial`, `LaunchAux::named_partial`).
- **Blocking and arrival.**
  - The executing lanes block, so the divergent-switch rule runs the complementary arm.
  - Lanes accumulate per warp at the same barrier id and flavor, from any site, with an equal `b`.
  - The group that completes the set makes ONE warp arrival with the full `live` mask. Its Protocol event and the Arrive event (over all gathered lanes) are logged there.
  - Each group then continues when the generation completes; for `arrive`, at once.
  - `bar.red` predicates are accumulated across the groups.
- **Errors.**
  - Missing lanes that reach a different barrier id or flavor, or exit, raise `PartialWarp`, the fail-closed Q3/Q5 case.
  - A different `b` raises `ContractMismatch`.
  - `.aligned` forms keep the immediate full-warp check.
- **Scenarios.** `divergent_named_barrier(same|other_id|exit)` covers both outcomes; it also runs in the checker smoke test.
- **Overlap with W6.** W6's in-progress `named::Gather` has the same semantics. The engine can call `named::gather` for these decisions once it lands. The sync_differential and synccheck_legacy_ports failures in the tree right now are W6's in-flight edits.
- **`sparse_flashmla_decode_head64`.**
  - Now passes the barrier at line 1743.
  - It next stops at line 2191 with `bad_address`: global 0x1000000a9800, which is 2048 bytes past the end of `out` (planned at 0x100000099000, 65536 bytes).
  - The op is `st.global.u64` through `o_ptr.view("uint64")` of a bf16 global `decl_buffer(data=out.data, elem_offset=out_offset)`.
  - Likely view / element-offset scaling in lowering (W1, aed5b1b "C-style global view offsets"). Please triage.

V2C-39 resolved (2026-10-08, at 72c7908): with the non-aligned partial-warp
`barrier.sync` gather (e75fbb0, 5241a22), `sparse_flashmla_decode_head64`
matches legacy in numsim, racecheck and synccheck. There was no delta
snapshot for it to remove; sync-behaviour-deltas B1/B2 are updated.

## W9-public-API (2026-10-08): other-assertion triage

Bugs found while triaging the 17 public-API legacy functions that fail under
`NUMSIM_IMPL=v2` with an "other assertion" (full table:
`scripts/numsim-v2/coverage/other_assertion_triage.tsv`). Not ported; each legacy
test stays until the owner fixes v2 (or rules it a delta).

- **[W2] interp, `discard` alignment.** `tests/numsim/runtime/test_discard.py::test_discard_indeterminate_read_and_alignment`. `T.ptx.discard.global_.L2(data.ptr_to([1]))` (address = buffer base + 4, base is 4096-aligned since V2C-35) is accepted: synccheck and racecheck give verdict `review` (only `uninitialized_read` of `data` bytes [128, 132), i.e. v2 discarded [4, 132)), and NumSim runs. Expected: verdict `error` with "discard requires a 128-byte aligned address" (legacy `engine-rs/src/runtime/instructions/mem.rs:2654`; `discard.L2 [a], 128` operates on a 128-byte line). `interp/handlers/mem.rs::discard` resolves the span but never checks `addr % 128`. The `restore=False` half of the test (review on reading discarded bytes, W8-5) already passes.
- **[W2] interp, `st.bulk` size validation.** `tests/numsim/runtime/test_non_tensor_bulk_forms.py::test_st_bulk_size_is_evaluated_per_issuing_lane`. With `size=1` (lane 0 stores 1 byte, lane 1 stores 2 bytes) synccheck and racecheck are `clean` and NumSim runs, for all four size dtypes. Expected: `error` / `ExecutionError` "st.bulk byte count 1 must be a multiple of 8 with maximum 16777216 on lane 0" (legacy `engine-rs/src/runtime/memory_ops.rs:505-509`; PTX `st.bulk` size is a multiple of 8). The other invalid sizes (-8, 16777224, 4294967304) are rejected, but only as `out_of_bounds` against the 32-byte buffer, not by a size check: `interp/handlers/mem.rs::st_bulk` has no size (multiple of 8, <= 16 MiB, non-negative) or 8-byte address-alignment check.
- **[W2] interp, `isspacep.shared::cta` ignores the CTA rank.** `tests/numsim/runtime/test_memory_sync_coverage.py::test_memory_sync_extensions[address_queries-inputs5-expected5]`. Column 5 of `address_queries`: `isspacep.shared::cta` of `mapa.u64(generic_shared_ptr, 1 - cta)` (a peer CTA's window in a 2-CTA cluster) returns 1 on both CTAs; expected 0 (legacy; the address is in the cluster window, not the executing CTA's `shared::cta` window). Column 6 (`mapa` to the own rank) correctly returns 1. `interp/handlers/mem.rs::isspacep` maps `AddrSpace::Shared` and `AddrSpace::SharedCluster` to the same `Generic::Shared(_)` test; `.shared::cta` must also require `decode_shared(..).rank == own rank`.
- **[W2] arena::addr (coordinator ruling still open), generic shared aperture bits.** `tests/numsim/runtime/test_scalar_control.py::test_mapa_and_cvta_expose_the_device_validated_integer_bits`. `addresses64[..., 2]` (generic `mapa.u64` of the peer's shared address): v2 0x00007F00_01000000 | off (139637993504768 + off for CTA 0), expected the device-validated 0x0000FFFE_01000000 | off (281466403553280 + off). The 32-bit columns and the other 64-bit columns match. Cause: `GENERIC_SHARED_BASE` (W2-2) is a synthetic 0x7F00_0000_0000, not the hardware window base 0xFFFE_0000_0000. The "Public-API triage (W8)" entry above asked the coordinator for an `arena::addr` ruling on synthetic address bits; there is still none, so this stays a bug until the base is changed or a delta row is written.

## W5-13 (for W2, 2026-10-08): two event-shape bugs from the public-API racecheck-verdict triage

1. **Multicast `.sync_restrict` commit is not restricted.** In `tcgen_commit`,
   `restricted = sync_restrict && multicast.is_none()`. A multicast restricted commit
   (`...sync_restrict::shared::read::mma::a.shared::cluster.multicast::cluster`) therefore
   tracks the whole MMA and is sent with `restricted: false`. This causes the false
   negative in `test_restricted_commit_preserves_full_mma_completion[1-True-True]` and
   `[2-False-True]`. Request: track only the shared-A reads and set
   `AsyncIssue.restricted = true` for both the multicast and non-multicast forms.
2. **`tensormap.cp_fenceproxy...sync.aligned` is emitted as 32 sibling-lane writes of
   the whole 128-byte descriptor.** Each lane's write is followed by one all-lane
   `TensormapRelease`. Lane 0's later `fence.proxy.tensormap::generic.acquire` cannot
   cover lanes 1..31's writes (no warp sync between them), so the TMA by lane 0 is a
   false `missing_proxy_bridge` (`test_tensor_map_predicate_effects`, shared::cta
   carriers). The instruction is one warp-collective copy plus release. Request: emit the
   copy as a single write by the first active lane (each lane writing a slice would hit
   the same problem), followed by the release.

## W5-14 (for W1, 2026-10-08): declare `sync_words` per element

`T.cuda.wait_until` on `state.ptr_to([lane])` declares the whole 128-byte buffer as one
word (`DeclareWord { span: 0..128 }`), but the verdicts name 4-byte elements. Racecheck now
accepts a launch-value exit inside a declared region (delta W8). Any other exit on such a
span is still `WaitExitUnproven`, because the history numbering is per declared word.
Request: one `DeclareWord` per element, matching the verdict spans.

## W1 (2026-10-08): decode_head64 view offset, public-API triage, W6 guard rule, W5-14

- **`sparse_flashmla_decode_head64`: fixed.**
  - A view over a runtime-offset global view (`o_ptr.view("uint64")`) no
    longer adds the parent's `elem_offset` a second time. TIR `elem_offset`
    counts from the shared data pointer.
  - Distinct unnamed DeclBuffers now keep distinct synthetic names
    (`…#<index>` on a collision), so racecheck still sees the legacy
    alias_stale_read pairs.
  - numsim, racecheck and synccheck all pass. No conformance regressions
    against HEAD.
- **Public-API triage, `lowering-rejects` (6): out of scope (W9 E/delete
  list).**
  - 4 use direct access to replicated TMEM views, which fail closed by
    contract item 29:
    - `test_tcgen_cp_bitcasts_uint8_scale_payload_into_float8_tmem`
    - `test_tcgen_cp_cta_group2_reads_and_writes_each_cta_scale_backing`
    - `test_tcgen_cp_cta_group2_routes_each_pair_in_four_cta_cluster`
    - `test_mxfp4_uses_ue8m0_scales_over_32_element_vectors`
  - 2 use `tirx.ptx_legacy`:
    - `test_legacy_m16n8k32_int8_reuses_dense_form_and_engine`
    - `test_legacy_ldmatrix_x1_domain_matches_independent_fragment_mapping`
- **Public-API triage, `v2-accepts-legacy-rejection` (2): rejection
  restored.**
  - Lowering marked a kernel `implicit_tmem` whenever it had TMEM views and
    no `tcgen05.alloc`. Now a view whose `allocated_addr` is a run-time
    value (an alloc result) never makes TMEM implicit, so it needs a live
    lease.
  - Static-address views, including ones with a run-time layout offset, are
    unchanged.
  - `test_tmem_runtime_address_without_a_dynamic_lease_is_rejected` and
    `dynamic_tmem_use_before_alloc` now raise.
  - For W9: all 4 tmem-lease items fail only on message wording. Legacy says
    "not covered by any live allocation"; the engine says "is not in a live
    tcgen05 allocation". This is a pin-message port.
- **W6 synccheck-verdict (23).**
  - (1) A guarded PTX op is now lowered entirely inside `If(guard)`,
    operands included (`lower_ptx`). A predicated-off lane evaluates no
    memory operand.
    - `register_extensions`, `logic_carriers`, `cvt_integer_sat` and
      `scalar_f64_rounding` pass.
    - `layout_lowering_contract::physical_buffers…` exposed a separate
      sizing bug: the byte length of strided views now covers
      `sum((e-1)*s)+1` elements, and ComposeLayout / `storage()`-offset
      layouts are sized by their physical span. Synccheck is clean.
    - That test's remaining racecheck expectation (an `alias_stale_read`
      between a same-dtype strided alias and its root) conflicts with the
      W5-9 identity rule: same-dtype views share the root. For W5.
    - The layout fix also clears `test_compose_layout…` and
      `test_local_view_exposes_raw_span…`.
    - `raw_tcgen_mma_tf32_ts_predicated` stops on direct TMEM stores to
      lanes outside the warp's sub-partition (`physical_lane` up to 111 from
      warp 0). That is the TMEM ruling; legacy allowed it. Contract/delta,
      not the guard.
  - (2) `im2col_cache_hints` (4): these kernels contain no `st.async`. The
    out-of-bounds read comes from `cp.async.bulk.prefetch.tensor`, lowered
    as `Tma{dir: Prefetch}` with `smem = 0, smem_space = Shared`. The engine
    resolves that 1-byte smem operand before the prefetch check, against a
    kernel with 0 bytes of shared memory.
    - W4-17 (in progress in `async_copy.rs`) skips the smem resolve for
      prefetch. These 4 pass once that change is built. Engine-side.
- **W5-14 (`sync_words` per element): contract needed.**
  - `BufferDecl.sync_words` is a `bool`, and the engine emits one
    `DeclareWord` spanning the whole buffer (sched/mod.rs 850/1013).
    Lowering cannot express a word size.
  - Request (pick one):
    - (a) Contract: `BufferDecl.sync_word_bytes: u32` (0 = none). Lowering
      fills it from the `WaitUntil` access width, and the engine emits one
      `DeclareWord` per word.
    - (b) No contract change: the engine splits the declared buffer into
      `dtype.bits()/8`-byte words. Lowering already marks the polled buffer
      itself (the view with the poll's element type), so its dtype is the
      poll width.
  - Lowering will add the field and a test as soon as (a) lands; (b) needs
    nothing from W1.

## W2 (2026-10-08): engine-stops triage (26 functions / 29 items), W5-12, W5-13, W5-14, W6 items, W2-8

### engine-stops triage (public-API set, `NUMSIM_IMPL=v2`, after this pass)

| outcome | functions (items) | which |
| --- | --- | --- |
| pass now | 6 (6) | `test_dynamic_shared_tile_view_uses_its_static_parent_bounds`; `test_compose_layout_combines_tile_and_swizzle_into_physical_alias_bytes` and `test_local_view_exposes_raw_span_including_layout_gaps_and_offset` (fixed by later commits); `test_explicit_uint8_backing_supplies_two_float4_values_per_byte`, `test_odd_float4_logical_count_uses_a_ceiling_byte_span` (**engine fix: sub-byte buffer element load/store**); `test_global_alias_view_uses_layout_physical_span` |
| delta, fail-closed by design | 17 (20) | see "Deltas" below |
| lowering, W1 | 2 (2) | see "Hand-offs" below |
| oplib, W4 | 1 (1) | `test_gate_intrinsics_match_float32_semantics`: `tirx.log1p` has no oplib implementation |

**Engine fix.** Sub-byte buffer element access (`float4` / `int4`): `interp/handlers/mem.rs`, `load_sub_byte` and `store_sub_byte`.
- Element `i` sits at bit `i * bits` of the packed bytes, low bits first.
- A load extracts the element's bits.
- A store is a read-modify-write of the holding byte, in lane order.

**Deltas** (fail-closed by design):
- **TMEM buffer access outside the warp's sub-partition (14 functions, 17 items).**
  - Functions: `test_fp8_cta1_extended_shared_addresses` x2, `test_bf16_m64_tcgen_mma_uses_layout_f`, `test_cta_group2_banked_a_selects_matching_b_shard`, `test_dense_gemm_async_reads_tmem_a_and_transposed_b_storage`, `test_dense_gemm_async_starts_the_fma_chain_from_input_d`, `test_large_bf16_gemm_async_uses_one_engine_gemm`, `test_m64_tcgen_mma_infers_weight_stationary_from_packed_layout_e`, `test_m64_tcgen_mma_uses_layout_f_independently_of_declared_tmem_layout`, `test_raw_cta2_mma_matches_the_typed_gemm_async_exactly` x3, `test_raw_cta2_ts_m128_selects_the_matching_a_lane_bank`, `test_tmem_layout_f_maps_rows_to_half_slabs`, `test_tmem_tlane_tcol_coordinates_are_observable_through_a_physical_alias`, `test_tf32_layout_f_unwritten_holes_are_zero_filled_and_require_review`.
  - All are single-warp kernels that read or write TMEM lanes 32..127 through a TMEM buffer.
  - Contract item 17 (C.4 row 1) executes buffer `Load`/`Store` on TMEM as `tcgen05.ld/st 32x32b`, which can address only the warp's own 32-lane sub-partition. Legacy modelled TMEM buffers abstractly.
  - Error: `bad_address: <buf>[i]: tmem lane L is outside warp W's sub-partition`.
- **`setmaxnreg` direction (3 functions).**
  - Functions: `test_setmaxnreg_is_an_ordering_call_not_a_tcgen_lifecycle_call`, `test_deleting_setmaxnreg_keeps_the_numerical_result` (it also runs the original kernel), `test_setmaxnreg_static_expressions_follow_public_parser_and_runtime`.
  - The kernels issue `setmaxnreg.dec` to a count above the current one (256 after 24; 64 after 32). PTX leaves that undefined, and the model rejects it: `RegPool(InvalidDirection)`. Legacy treated `setmaxnreg` as ordering-only.
- **Ordinary `tcgen05.alloc` while an `.exclusive` allocation is live (1).**
  - Function: `test_exclusive_tmem_uses_cta_local_lifecycle_without_placement`, the 96-column case.
  - The sync model (`sync/tcgen.rs::free_base`, copied from the reference) blocks any allocation while an exclusive one is live, so the run deadlocks. Legacy permitted it.
  - **Needs a ruling (coordinator/W6).** If legacy is right, the model rule changes in numsim-sync-ref first.

**Hand-offs**
- **W1, `test_pointer_conversions_and_runtime_descriptor_patch`.** `v2/lowering/calls.py::convert_through_memory` unpacks `dst_ptr, src_ptr = args`, but `T.cuda.float8tohalf8(src, dst)` and `half8tofloat8(src, dst)` take the source first. The test passes `float8tohalf8(source, half)`. As a result the engine loads F32x8 (32 bytes) from `half` at row offset 16, which is misaligned. `float22half2(dst, src)` is right as is.
- **W1, `test_copy_transports_unique_owners_across_warps`.** A cross-warp register-layout transport is not lowered: `v2/lowering/memory.py:517` asserts "register-layout element owned by another thread".
- **W4, `test_gate_intrinsics_match_float32_semantics`.** `tirx.log1p` has no oplib implementation (incomplete).

### W2-8, `.per_16bytes` copy reports (engine done; W1 lowering needed)

- **Contract (additive).** New `ReportMode::Per16BytesPattern { pattern: u32, bits: u8 }`. The old unit `Per16Bytes` stays and still fails closed.
- **Semantics.** For each 16-byte chunk of the source address space the copy reads, the lowest-addressed copied element (`bits` = 4, 8, 16 or 32; a 4-bit element is a low nibble) is compared with `pattern`. Any match sets the completion barrier's report bit.
  - Implemented in `sched/partition.rs::report_16`.
  - Scenario: `copy_report_16(sampled)`.
- **W1 request.** `ptx_lower.py::_report` should emit `{"Per16BytesPattern": {"pattern": int(hex, 16), "bits": 4 * len(hex)}}` from `per_16bytes::<hex>`.
- **Effect.** That clears the 4 `test_report_queries_and_phase_reset[per_16bytes::*]` items and the 2 `test_gather4_report_samples_only_selected_source_rows[*-per_16bytes::80000000]` items. They are incomplete until then.

### W6 items

1. **W2-8.** Engine side done, as above.
2. **`test_tensor_map_predicates_retain_errors`: passes.**
   - `tensormap.cp_fenceproxy...sync.aligned` requires the full warp (`warp_collective_divergence`, "requires all 32 lanes").
   - In-kernel tensor-map publications are tracked per descriptor (cp_fenceproxy, or a release fence over the warp's dirty descriptors). A TMA through a published descriptor needs `fence.proxy.tensormap::generic.acquire` of that generation by the same CTA ("latest published generation is not acquired within this CTA"). This tracking is partition-local.
   - A `tensormap.replace` rank outside 1..5 is an invalid operand, an error rather than incomplete. W4: `TensorMapDesc::replace` returns Unsupported for it.
3. **`test_pointer_array_initialization_and_bounds`.** `[valid]` passes. `[uninitialized]` / `[oob]` keep the documented kind and message delta (`tests/numsim/v2/ports/test_deltas_pointer_slot_arrays.py`).

### W5-12, W5-13, W5-14

- **W5-12.** `tcgen05.alloc` emits a `WarpSync` after its result store, on both the plain and `cta_group::2` paths. Scenario `tcgen_alloc_lanes_read` (cta_group::2): racecheck reports the race without the fix and is clean with it. W8 can take the two B7 delta snapshots.
- **W5-13.**
  1. A multicast `.sync_restrict` commit is `restricted` too.
  2. `tensormap.cp_fenceproxy` is ONE warp-collective copy (`ALL_LANES` access) with warp-uniform operands, followed by the release.
  3. New `ExecErrorKind::WarpCollectiveDivergence` (`warp_collective_divergence`) for `.aligned` collectives with divergent lanes (`full_warp` sites, `ldmatrix`), membermask mismatches, `__syncwarp`, `grid.sync`, `setmaxnreg` and cp_fenceproxy. `divergence` stays for non-uniform operands.
  - Passing: `test_tcgen05_restricted_commit.py` (all), `test_ldmatrix_b8.py`, `v2/checkers/test_warp_collectives.py`, `test_divergence_liveness.py`, `v2/ports/test_single_lane_participation.py`. 18 tests marked xfail now pass, so W8 can drop those marks.
  - `test_tensor_map_predicate_effects`: the data race is gone. It still fails on a racecheck REVIEW `alias_stale_read`: the `tensormap.replace` read through `image.ptr_to(...)` is treated as a different logical buffer (W5 naming).
- **W5-14.** A `sync_words` buffer declares one word per element of its dtype (`bits / 8` bytes, at least 1), both for shared windows and global views (`sched/mod.rs::sync_word_spans`). Scenario `polled_flag_words`: 128 bytes of u32 polled per element gives 32 four-byte words. W5: landed.

## v2 conformance, sweep 5 (W8, 2026-10-08, at 8fba7e1 + a3c4df9)

Matches out of 101 (3 per mode have no legacy oracle): numsim 98, racecheck 96, synccheck 98.
No open V2C row. The two racecheck rows that still differ from the legacy snapshot are explained
by delta rows and have no delta snapshot yet: `alphamoe_fp8_blockscale_qwen3next` (R3,
`undeclared_protocol_word` instead of `data_race`) and `kda_backward_packed` (X4,
`cross_cta_async_order` advisory). Public-API set: 551 of 762 pass; per-function status in
`scripts/numsim-v2/coverage/v2_public_status.tsv`, triage by owner in the status doc.

## W5-15 (for W1 + W2, 2026-10-08): per-operand logical buffer for `alias_stale_read`

`alias_stale_read` names an access by `SiteInfo::buffer`, which is ONE name per
site: the first pointer operand. A multi-operand op that reads one buffer and
writes another gets the wrong name for one of its accesses. Example:
`tensormap.cp_fenceproxy(destination, image)` reads shared `image` under the
name `destination`, which gave a false advisory in `test_tensor_map_predicate_effects`.
The pointer identity is not elsewhere in the stream either: `Evidence` carries
kernel, space and alloc, and `TensormapAcquire` carries alloc and span; neither
has the logical name.

- **Interim (racecheck, delta P7):** a site's name is used only when the
  access's allocation space equals the named buffer's declared space. This
  fixes the example. It cannot tell apart two same-space operands, e.g. a
  shared→shared copy.
- **Request:** `SiteInfo.buffers: Vec<String>`, one per pointer operand in
  operand order (W1, lowering), plus `Access.operand: Option<u8>`, the operand
  index the access came from (W2, engine emit). Racecheck then uses
  `buffers[operand]`.

## W1 (2026-10-08): report pattern, half8 pointer order, owner transport (no contract change)

- **`Per16BytesPattern`: done.** `_report` maps
  `.mbarrier::report::per_16bytes::<hex>` to
  `{"Per16BytesPattern": {pattern, bits = 4 x hex digits}}`. The bare form
  stays `Per16Bytes` (fails closed) and `per_element::ff` stays
  `PerElementFf`.
- **`float8tohalf8` / `half8tofloat8`: fixed.** Their signature is
  `(void* src_addr, void* dst_addr)`, while `float22half2` is
  `(void* dst, void* src)`. This matches TVM `backend/cuda/cpp/builtins.py`
  and legacy `cuda_helper.rs` `DPS_HELPERS`, where the destination index is
  1 vs 0. Lowering now reads the first pointer of the two eight-element
  helpers.
- **Cross-warp register transport: done with existing ops**
  (`lowering/owner_transport.py`).
  - Scope: a function whose tile ops are all element-wise and that has at
    least one op between register fragments of different thread layouts.
    Ops covered: copy, cast, add/sub/mul/div/max/min, sqrt/exp/log/abs and
    similar.
  - Such a function skips TVM dispatch. TVM's copy fallback writes other
    threads' registers, and it rejects mul and cast outright.
  - Instead, each source fragment owned differently from the executor is
    staged through a per-op shared scratch (`<buf>.transport`): every owner
    stores its elements, then the scope barrier runs (warpgroup:
    `bar.sync 8, 128`, as TVM's `warpgroup_sync(8)`).
  - Each destination owner then computes and writes its own elements, and a
    second barrier releases the scratch.
  - All 3 `test_tile_owner_transport` tests pass; synccheck and racecheck are
    clean.

## W9-public-API phase 6 (2026-10-08): internal other-assertion triage

The 40 internal-surface `other-assertion` functions (`test-migration.md` "Blockers", internal row) were run under `NUMSIM_IMPL=v2` and classified: 10 port, 25 delta, 5 bug (rows in `scripts/numsim-v2/coverage/other_assertion_triage.tsv`; v2 copies under `tests/numsim/v2/ports/test_p6d_*.py`, listed in `coverage/v2_ports_p6d.tsv`). Each bug below has a faithful copy marked `v2_gap` (xfail) that records the observed values.

- **[W8]** `tests/numsim/integration/test_api.py::test_engine_native_loop_policy_is_explicit_and_validated`. `v2.Engine` stores `native_loop_iteration_budget` / `native_loop_reschedule_quantum` unvalidated (`loop_budget` / `quantum`, default `None`). Observed: `budget=0` and `budget=True` construct and run to completion; `quantum=0` constructs and `Engine.run` hangs; `quantum=1.5` fails only at run time (`TypeError: 'float' object cannot be interpreted as an integer`); `budget=-1` fails only at run time (`OverflowError`). Expected (legacy): `ValueError("... must be positive")` for 0, `TypeError("... must be a positive integer")` for `True` / `1.5`, raised by the constructor. Copy: `tests/numsim/v2/ports/test_p6d_api.py::test_engine_native_loop_policy_is_explicit_and_validated`.
- **[W1]** `tests/numsim/runtime/test_ptx_bitops.py::test_ptx_bitop_forms_and_predicates_match_independent_oracle`, `.pred` forms only (`and/or/xor/not.pred`). TVM tags `.pred`-class operand positions with the `p<i>` marker (bridge through a pred register: `setp.ne.b32` on input, 0/1 on output); `v2/lowering/ptx_decode.py::decode` parses the flags and discards them (`_preds`), so the op runs on the raw u32 carrier and keeps bit 0. Observed: `and.pred(2, 2) = 0` (expected 1); `not.pred(2) = 1` (expected 0); with `preserve_dst` / `pred=False` a kept destination of 85 reads back 85 (expected the bridged truth value 1). Fix: convert `p<i>` sources with `!= 0` and write `p<i>` destinations back as 0/1, including the kept value. All non-pred forms pass. Copy: `tests/numsim/v2/ports/test_p6d_ptx_bitops.py::test_ptx_bitop_forms_and_predicates_match_independent_oracle_pred_forms`.
- **[W2]** `tests/numsim/runtime/test_memory_ops.py::test_cuda_uint64x2_atomic_cas_compares_and_replaces_one_128bit_value`. `T.cuda.atomic_cas` on `uint64x2` lowers to one `Atom{Cas, ty u64x2}`; `interp/handlers/mem.rs::rmw_bytes` splits it into 8-byte elements, so each component is compared and replaced on its own and the 128-bit value tears. Observed: old values `[[7,9],[11,13],[17,13]]`, final `[17,29]`; expected `[[7,9],[11,13],[11,13]]`, final `[23,29]`. Fix: a vector-typed Cas compares and replaces all 16 bytes at once (or W1 emits a b128 type). Copy: `tests/numsim/v2/ports/test_p6d_memory_ops.py::test_cuda_uint64x2_atomic_cas_compares_and_replaces_one_128bit_value`.
- **[W2]** `tests/numsim/runtime/test_memory_ops.py::test_cuda_128bit_atomic_cas_compares_float_vectors_by_bits`. Same component-wise split on `float32x4`: lane 0's compare differs only in component 0 (`0x00000000` vs `0x80000000`) yet components 1..3 are replaced. Observed lane 1 old value `[0x80000000, 2, 3, 4]`; expected the initial `[0x80000000, 0x7FC12345, 0x3F800000, 0x40000000]`. Copy: `tests/numsim/v2/ports/test_p6d_memory_ops.py::test_cuda_128bit_atomic_cas_compares_float_vectors_by_bits`.
- **[W2]** `tests/numsim/runtime/test_raw_tcgen_codegen.py::test_raw_tcgen_mxf4_bulk_smem_and_tmem_reads_require_review`, mode 2 (a `tcgen05.mma ... kind::mxf4.block_scale` with `enable_input_d` accumulates into a never-written TMEM D, lanes 0..127, cols 0..127). Observed: verdict `clean`, no diagnostics (the accumulation itself happens: D preset to 1.0 reads back 1.0). Expected: verdict `review` with `uninitialized_read` diagnostics in space `tmem`, as legacy reported; a direct buffer read of unwritten TMEM is reported by v2. Likely cause: `sched/partition.rs` calls `report_async_uninit` on the op's read spans after `run_mma` has written D and marked it valid. Fix: check validity before the write, or have `run_mma` report invalid reads as it goes. Mode 1 (shared) is reported correctly. Copy: `tests/numsim/v2/ports/test_p6d_raw_tcgen_codegen.py::test_raw_tcgen_mxf4_bulk_smem_and_tmem_reads_require_review[tmem]`.

Behaviour changes seen in this triage with no delta row yet (asserted as v2 behaviour in the copies; W8 may want rows): un-waited `tcgen05.cp` reads see zeros (8 `test_raw_tcgen_codegen` functions; W4-7 async cp, racecheck P6); `Tx.warp.copy` local->local runs TVM's `copy/fallback` in lane 0 only (Decision 6; 2 `test_copy_dispatch_contract` functions); default `Engine.run` output keys of a multi-kernel module are canonical buffer names, not `k<i>:` qualified; an output selected through an fp4 tensor map comes back with the 128-byte image shape; `dense_gemm_async_m64_wrong_f_group_order` is now rejected at transpile by TVM's `gemm_async` dispatch; the fp8 cta1 `sm_100a` 1<<18 descriptor check moved from transpile time to a run-time `invalid_operand` whose message names `tcgen05.cp` for a `tcgen05.mma` descriptor (W2 wording).

## W2 (2026-10-08): W9-public-API engine bugs fixed; engine-stops re-check

1. **`discard`.** The address must be 128-byte aligned: "discard requires a 128-byte aligned address", `Misaligned`.
2. **`st.bulk`.**
   - The byte count is read as signed per lane and must be a non-negative multiple of 8, at most 16777216: "st.bulk byte count N must be a multiple of 8 with maximum 16777216 on lane L".
   - The address must be 8-byte aligned.
3. **`isspacep.shared::cta`.** True only for the executing CTA's own rank. `.shared::cluster` stays true for any shared window.
4. **`arena::addr::GENERIC_SHARED_BASE`.** Now the hardware window base 0x0000_FFFE_0000_0000 (device-validated generic `mapa`/`cvta` bits).
- **Tests and scenarios.** `tests/w9_public_api.rs` (3 tests) uses the scenarios `discard_at`, `st_bulk_size` and `mapa_isspacep`; the non-error variants also run through the checker smoke test.
- **Legacy tests now passing:** `test_discard_indeterminate_read_and_alignment`, `test_st_bulk_size_is_evaluated_per_issuing_lane`, `test_memory_sync_extensions[address_queries]` and `test_mapa_and_cvta_expose_the_device_validated_integer_bits`.
- **Engine-stops (19 functions / 22 items).** No interp/sched/arena item remains beyond the ruled deltas, matching W4-engine-stops.
  - 13 TMEM sub-partition functions (16 items).
  - 3 `setmaxnreg` direction items.
  - The exclusive-TMEM item now stops with `Tcgen(AllocWhileExclusive)` as a protocol error, not a block or deadlock (T8).
  - `test_copy_transports_unique_owners_across_warps` passes.
  - `test_gate_intrinsics_match_float32_semantics` fails only on the `rust_source` pin (W9).
- **Unchanged and not mine:** `test_memory_sync_extensions[scalar_async_store]` remains a racecheck `scope_mismatch` (W5).
- **Conformance.** `NUMSIM_IMPL=v2` gives 295 passed, 9 skipped (numsim / racecheck / synccheck 98 / 98 / 98 run, 0 failed). `cargo test --workspace` passes 1024 / 1024, and codegen equivalence 4 / 4.

## W1 (2026-10-08): lowering-rejects, sweep residual table, mma_fill/mma_store

- **`lowering-rejects` (6): confirmed fail-closed.** Delta rows L1
  (replicated TMEM view, 4 tests) and L2 (`tirx.ptx_legacy`, 2 tests) are in
  `numsim-behaviour-deltas.md`. W9 ported all six as expected
  `UnsupportedTIRxError` (`tests/numsim/v2/ports/test_failclosed_lowering.py`).
- **Sweep residuals:** 2217 of 2343 kernels lower clean (up from 2215).
  Every unclean module falls in one documented class:

  | kernels | reason | row |
  | --- | --- | --- |
  | 69 | TVM `TilePrimitiveDispatch` rejects the tile op (gemm_async, copy_async, permute_layout, unregistered copy variants) | L3 (new) |
  | 37 | Legacy fail-closed rules kept: mutated or unreviewed helpers, invalid reinterprets, invalid TMA-reduce op or dtype, bad wait_group counts, unknown tile config keys, directed f64 rounding, off-ABI warp-gemm fragments, fuzz/negative kernels (`op None`, non-literal `mov_sreg`, `boolx128`, parallel or thread-bound loops, unknown attributes, shuffle out of range) | legacy rejections |
  | 9 | `tirx.ptx_legacy` | L2 |
  | 9 | Direct access to a replicated TMEM view | L1 |

- **New lowering: `tirx.mma_fill` / `mma_store` (+ `_legacy`).** This is
  the one real gap the sweep showed.
  - `mma_fill` zeroes `local_size` accumulator elements.
  - `mma_store` writes the 16x16 fragment by the lane/register ABI of legacy
    `emit/matrix.rs`: element `id` of lane `l` goes to
    `row = 8*((id%4)/2) + l/4`, `col = 8*(id/4) + 2*(l%4) + id%2`.
  - Both use generic `LoadAddr`/`StoreAddr` and need no contract change.
  - `test_mma_fragment_fill_and_store_observe_lane_register_layout` and
    `test_reused_mma_store_pointers_write_lane_register_layout` pass.
- **`test_mov_pointer_identity_masks_and_nulls`: engine-side, not
  allocation sizing.**
  - The positive global and shared cases pass numsim, synccheck and
    racecheck, and the shared storage is sized correctly (256 static bytes).
  - The failure is the `dereference_null=True` variant. Its `ld.b32` through
    a generic null pointer (0) is resolved into the CTA's shared window of
    0 bytes (`smem[cta0]`) and reported as out_of_bounds.
  - Legacy reports a null dereference. For W2: a generic address 0 should
    fail as `null`, not fall into the shared aperture.

## W11-other-assertion (2026-10-08): public-API other-assertion triage

The 25 public-API functions with class `other-assertion` in `scripts/numsim-v2/coverage/v2_public_status.tsv`, plus the one `v2-accepts-legacy-rejection` function, were rerun under `NUMSIM_IMPL=v2`. The full mapping is in `scripts/numsim-v2/coverage/v2_ports_w11.tsv`. Three v2 bugs remain. Each has a runnable reproducer in `tirx_harness/tests/numsim/v2/ports/test_w11_reproducers.py`; the reproducers assert the correct behaviour and are marked `v2_gap`, so an XPASS means the bug is fixed. The four W9-public-API bugs (discard alignment, `st.bulk` size, `isspacep.shared::cta` rank, generic shared window bits) were fixed by W2 during this pass ("W2 (2026-10-08): W9-public-API engine bugs fixed"), and their legacy tests now pass unchanged.

### W11-1 [W1 lowering]: a guarded op with a `.pred` destination keeps the raw carrier value

- **Legacy tests:**
  - `tests/numsim/runtime/test_compare_predicates.py::test_compare_instruction_predicates[True]` and `[False]` (the `setp.*` and `testp.*` rows).
  - `tests/numsim/runtime/test_mbarrier_maintenance.py::test_maintenance_predicates_and_carriers`, cases `(preserve=True, layout=1 or 0)` of `mbarrier.check_layout`.
- **Reproducer:** `test_w11_1_guarded_pred_destination_is_a_boolean_carrier`. Every lane is predicated off, and `d[i] = 91` beforehand:
  `T.ptx["setp.lt.s32"](d[0], 1, 2, pred=lane < 0, preserve_dst=True)`
  `T.ptx["setp.lt.s32"](d[1], 1, 2, pred=lane < 0)`
  `T.ptx["testp.normal.f32"](d[2], 1.0, pred=lane < 0, preserve_dst=True)`
  `T.ptx["set.lt.u32.s32"](d[3], 1, 2, pred=lane < 0, preserve_dst=True)`
- **Expected:** `[1, 0 or 1, 1, 91]`. Legacy gives `[1, 0, 1, 91]`. TVM's helpers bridge a `.pred` operand through a predicate register:
  - `_pred_keep`: `setp.ne.b32 pd, %0, 0; @p setp... pd; selp.b32 %0, 1, 0, pd`, so a kept value becomes `old != 0`.
  - `_pred_undef`: `selp.b32 %0, 1, 0, pd`, so the value is always 0 or 1.
- **Actual:** `[91, 91, 91, 91]`. In the maintenance test, `-7` (0xFFFFFFF9) is kept raw where 1 is expected.
- **Cause:** the same as the W9-public-API phase 6 [W1] `.pred` bridge item (`test_ptx_bitops`): `ptx_decode.decode` drops the `p<i>` flags. The fix requested there covers this: write `p<i>` destinations back as 0/1, *including the kept value* of a guarded or `preserve_dst` op. Destinations that are plain `uint32` keep their old value (numsim-behaviour-deltas P8).

### W11-2 [W2 interp]: each `Ptx` op is resolved with the operand types of its first use only

- **Legacy test:** `tests/numsim/runtime/test_cvt_carriers.py::test_cvt_carriers_truncate_extend_and_gate_reads[s]`. The modes 0 and 3 rows of every signed type with a wider destination carrier fail; `[u]` and `[f]` fail only on the P8 rows.
- **Reproducer:** `test_w11_2_ptx_op_resolves_per_use_carrier_types`. `src = int8(-1)`, then:
  `T.ptx["cvt.s8.s8"](h, src)` with an `int16` carrier
  `T.ptx["cvt.s8.s8"](w, src)` with an `int32` carrier
  `T.ptx["cvt.s8.s8"](q, src)` with an `int64` carrier
- **Expected:** `h = w = q = -1`. This is sign extension to the destination register width, as legacy computes.
- **Actual:** `h = -1`, `w = q = 0xFFFF`. Each call alone is correct.
- **Cause:** `interp/mod.rs::Loaded::new` keeps "Operand types of the first use of each op" and calls `resolve_ptx(key, first_dst_tys, first_src_tys)` once per `OpId`. The `OpKey` is `(name, mods)` without operand types, so every later use with different carrier types runs the handler resolved for the first use's types. In the legacy test the first `cvt.s8.s8` form has an 8-bit destination, so the `int16`/`int32`/`int64` forms are zero-extended from 8 bits.
- **Fix:** resolve per `(OpId, dst_tys, src_tys)`, for example a per-pc handler table. Alternatively, W1 can intern ops with the carrier types in `mods`, but that changes the `OpKey` contract (coordinator).

### W11-3 [W1 lowering]: a `shared.dyn` pool is sized from its views, not from the committed `tirx.dyn_smem_bytes`

- **Legacy test:** `tests/numsim/integration/test_reported_layout_regressions.py::test_explicit_shared_strides_still_reject_an_executed_oob_address` (class `v2-accepts-legacy-rejection`).
- **Reproducer:** `test_w11_3_pool_access_beyond_committed_dyn_smem_is_out_of_bounds`. The kernel is verbatim:
  ```
  pool = T.SMEMPool()
  scratch = pool.alloc((2, 4, 64), "float32", strides=(8192, 64, 1), align=16)
  pool.commit()
  if lane == 0: scratch[1, 0, 0] = 1.0
  ```
- **Expected:** `ExecutionError` `out_of_bounds`, as legacy raises. `SMEMPool.alloc` bumps by `prod(shape) * 4 = 2048` bytes, and `commit()` emits `tirx.dyn_smem_bytes = 2048`. On the GPU the CTA's dynamic shared memory is 2048 bytes, and `scratch[1, 0, 0]` is byte 32768.
- **Actual:** the run completes, and racecheck and synccheck are clean. The lowered `pool_buf` has `byte_len = 33792`.
- **Cause:** `lowering/memory.py::finish_shared` sizes a dynamic pool as `max(view extents, dyn_smem_bytes)`, so a view whose strided extent overruns the committed pool silently enlarges it.
- **Fix:** when `tirx.dyn_smem_bytes` is present, use it as the pool size and let the view's accesses be bounds-checked against it. A view that is larger than the pool could also be flagged at lowering time.

## W2 (2026-10-08): hygiene audit, incomplete/timing inventory, W1 null deref, W11-2, W8 generic tensormap.replace

**Dead code removed (interp/, sched/, arena.rs).** The audit used `RUSTFLAGS="-W dead_code -W unused"` (no warnings), a workspace-wide caller scan of every `pub`/`pub(crate)` fn, and a field scan.
- Removed functions, which had no callers anywhere:
  - `sched::Scheduler::end_reason`;
  - `sched::Scheduler::fire_completion` (documented "tests", unused);
  - `LaunchAux::next_collective_id` and its field `next_collective`;
  - `ExecCtx::note_progress`;
  - `ExecCtx::read_uniform`;
  - `Arena::subview`.
- Removed field: `WarpState::clock`, dead since the "physical special registers read 0" ruling.
- `sched::cluster_of` was `#[allow(dead_code)]`; it is now `#[cfg(test)]` (its only user is the module's test).
- Nothing else found:
  - no other `allow(dead_code/unused)`;
  - no `TODO`/`SCRATCH`/`legacy-shim` markers (`PTX_SCRATCH` is a scratch buffer);
  - no Cargo features or `cfg(feature)` in these modules.
- `codegen/` was not touched (backend decision pending).

**Inventories.** The tables were added to `docs/development/engine-review.md`: "Engine `incomplete` reasons" (12 reason families plus the 10 `support::unsupported` sites) and "`Report.meta.timing` phases".
- Timing fix: `lower` no longer mixes in module-cache loads. The new key `module_cache` holds `lower_ms` on a cache hit (`v2/run.py::_timing`; `test_v2_layer` key sets updated).
- Racecheck runs online, so its cost is part of `run`. Synccheck exploration and racecheck `finish` are in `check`. This is documented in the timing table.

**W1 addendum, null dereference.** A data access through a null generic or global pointer is now `bad_address` "null pointer dereference: Generic address 0x0 is in no aperture".
- It applies to `ld`/`st`/`atom`/`red`/`st.bulk`/`discard` via `support::resolve_data`.
- Synchronization operands keep the W1 rule that a zero-extended shared::cluster address 0 names CTA 0's byte 0 (`mapa_value_as_cluster_and_generic_mbarrier_operand`).
- Scenario: `generic_load(null)`.
- `test_mov_pointer_identity_masks_and_nulls[dereference_null]` passes. Legacy checks only the "null" message substring, so no kind port is needed (W9).

**W11-2.**
- `Loaded::new` resolves each PTX op once per distinct operand-type signature (`Loaded::op_variants`). Single-signature ops keep the old fast path.
- Scenario: `ptx_op_per_signature` (`cvt.s8.s8` into s16/s32/s64 gives -1 in each).
- `test_w11_2_ptx_op_resolves_per_use_carrier_types` now XPASSes (W11 to flip).

**W8, generic `tensormap.replace`.**
- Lowering (`v2/lowering/ptx_lower.py::lower_tensormap_replace`, a one-line change; W1 please review) defaulted an unqualified address to `Global`. PTX: no state space means generic. The engine already resolves generic through the shared window.
- Scenario: `tmap_replace_generic_shared`, a shared descriptor image replaced through its generic address.
- **Found while testing:** the replace `rank` field holds rank - 1. My earlier guard (accepting 1..=5) was off by one; it is now `<= 4`.

`cargo test --workspace`: 1032 passed, 1 failed. The failure is `synccheck_scenarios::four_producers_per_lane_arrivals_on_two_barriers_stay_small`, a pure-synccheck log test, against W6's uncommitted explorer edits. Codegen equivalence passes 4/4.
