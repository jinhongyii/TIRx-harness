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
