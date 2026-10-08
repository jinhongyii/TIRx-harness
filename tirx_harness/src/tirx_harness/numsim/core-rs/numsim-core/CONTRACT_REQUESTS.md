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
