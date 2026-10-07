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
