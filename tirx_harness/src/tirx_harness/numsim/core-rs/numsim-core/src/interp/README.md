# interp: the warp interpreter

One execution path. `step_warp` (mod.rs) runs one warp for a quantum, using the handlers in `handlers.rs` and `handlers/{alu,async_copy,control,mem,sync,tcgen,warp}.rs`. Observers watch; they never change what runs.

## ExecCtx (mod.rs)
Everything a handler may touch for one warp slice:
- `program` / `loaded`: the IR and its load-time resolution (op table, register slots).
- `launch`: the launch shape.
- `config`: the `RunConfig` (budget, quantum, policies).
- `warp`: the `WarpState` (registers, masks, frames, `resume`, `poll`).
- `cta`: the `CtaCtx` (ids, cluster peers, shared/TMEM allocations).
- `buffers`: the `BufBinding`s.
- `arena`: the main arena or a partition shard.
- `sync`: the partition's `SyncTable`.
- `observer`: the partition's event buffer; `observing` = the observer wants events.
- `counters`: access sequence and progress counters.
- `aux`: the `LaunchAux` (async metadata, word history, CLC queue, tcgen state; see aux.rs).

## Handler contract
- Signature `fn(ctx, ...) -> HResult = Result<Flow, ExecError>`. Start with `active_or_next!(ctx)` (no active lanes → `Next`).
- `Flow`: `Next`, `Jump(pc)`, `Yield(pc)` (back to the scheduler), `Blocked(res)` (re-run this pc later), `Exit`.
- Errors come from `support::err`, `support::op_err` and `support::unsupported`. A provable kernel bug is an error kind (`OutOfBounds`, `Divergence`, `Protocol`, `Op(Invalid)`, ...). An unmodelled case is `Unsupported`, which the run reports as `incomplete` with a reason.
- Memory goes through `support::{resolve, resolve_data, mem_write}` and arena views, never raw indices. Accesses are reported through `support::emit_accesses` with the pointer-operand index (`Access.operand`, README decision 15; values 240 and up are reserved).
- Sync state changes only through `support::step` / `step_all` on the `SyncTable`, followed by `support::protocol`, so checkers see every command.
- A global read-modify-write inside an arena shard is a serial point: set `ctx.aux.serial_request` and return `Yield(pc)`. The scheduler re-runs it on the main arena (atomics, claimable CLC `try_cancel`).

## Async ops and tokens
- `async_copy::issue_async(ctx, lanes, Issue { kind, class, payload, signals, after, preds, ... })` queues an `AsyncOp`.
  - `after` holds the ops it must land after.
  - `preds` holds the pipeline predecessors reported to checkers (defaults to `after`).
  - The payload is applied when the op lands (`sched/partition.rs::fire_op`).
- tcgen05 MMA:
  - The shared-A read is its own op (`shared_a_read`), which a `.sync_restrict` commit tracks.
  - TMEM operand reads belong to the MMA op itself.
  - Per-thread pipeline order uses `aux.tcgen_last_thread`.
- Completions (mbarrier tx/arrive, group milestones) are `Completion`s, applied in FIFO order by the scheduler.

## Incomplete reasons
The full inventory lives in docs/development/engine-review.md, section "Engine `incomplete` reasons". Add a row there when you add an `Unsupported` site.

## Adding an instruction
1. In `program.rs`, add an `Instr` variant (or a `Ptx` op handled by oplib), its `name()`, and its `validate()` rules. A contract change goes through CONTRACT_REQUESTS.md.
2. Write the handler (in the matching file under `handlers/`) and add its dispatch arm in `handlers.rs`.
3. Add the lowering in `v2/lowering` (W1).
4. Add a `ProgramBuilder` scenario in `testutil/scenarios.rs`, list it in `all()` so both checkers run it (`tests/interp_checkers_smoke.rs`), and add a behaviour test in `tests/interp_scenarios.rs`.
