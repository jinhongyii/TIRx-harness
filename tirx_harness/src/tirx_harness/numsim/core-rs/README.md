# core-rs: the NumSim / Racecheck / Synccheck core

Plan: `docs/development/numsim-redesign.md`. Worker inputs:
`docs/development/lowering-inventory.md` (W1),
`docs/development/sync-semantics.md` (W3),
`docs/development/synccheck-explorer.md` (W6),
`docs/development/racecheck-semantics.md` (W5).

Workspace members: `numsim-core` (everything below) and `numsim-py`
(pyo3 bindings, feature `python`). The standalone crates
`numsim-sync-ref`, `numsim-sync-explore`, `numsim-race-core`, `numsim-oplib`
have their own `[workspace]` tables; the coordinator merges them.

```
cargo build && cargo test && cargo doc --no-deps
PYO3_PYTHON=python3 cargo check -p numsim-py --features python
```

## The contract

| Module | What it fixes | Status |
| --- | --- | --- |
| `program` | `Module`, `Program`, `Instr`, operands, tables, `Launch`, host ABI, serde (JSON externally tagged + postcard), `validate()`, `Display` | complete |
| `dtype`, `value` | `Dtype`, `Ty{elem, lanes}`, register model, `WarpValue<T> = [T; 32]`, `WarpMask`, `RegFile` | complete |
| `site` | `SiteId`, `SiteInfo{kind, spans, op_name, text, dtype, buffer}` | complete |
| `arena` | `Arena`, `Allocation`, `AllocId`, `View`, `ByteSpan`, `Space`, validity, address encodings (`arena::addr`) | implemented |
| `observe` | `Observer`, `Access` (hot), `SyncEvent{actor, seq, site, frames, lanes, kind: SyncKind}` (cold), `NoopObserver`, `RecordingObserver` | complete |
| `sync` | `SyncTable`, `ResourceId`, `Completion`, `AsyncOp`, `Payload`, per-protocol `State/Cmd/Outcome/Error` copied from `numsim-sync-ref` | types complete, `step` bodies W3 |
| `interp` | `WarpState`, `MaskFrame`, `ExecCtx`, `Flow`, `StepResult`, `ExecError`, `step_warp`, **`interp::handlers`** (one fn per family + `dispatch`) | signatures; bodies W2 |
| `sched` | `Scheduler`, `CtaState`, `Inbox`, `RunConfig`, `Inputs`/`Outputs`, `Backend`, `run`, `resolve_launch` | signatures; bodies W2 |
| `oplib` | `Scalar`/`FloatScalar`, TIR ALU entry points, `PtxIo`/`PtxFn`/`resolve_ptx`, TMA/descriptor/MMA signatures, op registry -> SUPPORTED_OPS.md | signatures; bodies W4 |
| `report` | `Finding`, `FindingKind`, `Status`, `Verdict`, `Evidence`, `Report` | complete |
| `racecheck`, `synccheck`, `codegen` | entry-point skeletons | W5, W6, W7 |
| `testutil` | `ProgramBuilder` for handwritten tests | implemented |

## Ownership

Workers edit **only their own directory**. Changes to `program.rs`,
`dtype.rs`, `value.rs`, `site.rs`, `observe.rs`, the public `arena` API,
the `sync` type shapes, the `interp::handlers` signatures, `report.rs` and
`lib.rs` go through the coordinator.

| Worker | Owns |
| --- | --- |
| W1 lowering (Python) | `numsim/v2/lowering/` (emits `Module` JSON/postcard) |
| W2 interp + sched | `numsim-core/src/interp/` (bodies), `numsim-core/src/sched/`, arena internals |
| W3 sync | `numsim-core/src/sync/` (step bodies; types mirror `numsim-sync-ref`) |
| W4 oplib | `numsim-core/src/oplib/` |
| W5 racecheck | `numsim-core/src/racecheck/` |
| W6 synccheck | `numsim-core/src/synccheck/` |
| W7 codegen | `numsim-core/src/codegen/` |
| W8 python + test infra | `numsim-py/`, `numsim/v2/*.py`, snapshot infra, CI |

## Contract decisions

1. **Instr granularity** (W1 Q1): dedicated variants for all families with
   engine-visible semantics; one generic `Instr::Ptx{op: OpId, dsts, srcs,
   pred, keep_dst}` for the pure-register PTX tail, backed by the interned
   `Program::ops` table and resolved once at load by `oplib::resolve_ptx`.
   `mma.sync`, `movmatrix`, `match`, `createpolicy`, `prefetch`, descriptor
   encoders, cvt/pack/unpack all go through `Ptx`.
2. **Registers**: one `Reg` = one TIR value with a static `Ty`; storage is
   64-bit slots, a value uses `ty.slots()` (1..=4) consecutive slots
   (`Program::reg_slot_offsets`). Vectors/wide types (`float16x2`,
   `uint32x4`, `uint128`, `float32x8`) are ONE register (W1 Q4).
   Predicates are 0/1; addresses are u64 (generic/global) or u32
   (shared, tmem) values.
3. **Operands** are `Reg | Const(ConstId)` (no immediates); constants are
   interned `Const{ty, bits: u128}`.
4. **Sites** live in the parallel array `Program::code_sites` instead of a
   `site` field per variant (same information, one rule for handlers and
   codegen: `ctx.site()`).
5. **Control flow**: `If{elect}/Else/EndIf`, `LoopBegin/LoopIf/LoopEnd`,
   `Break/Continue`, per-lane `Exit`. Loops carry no id: the iteration
   counter lives in the mask frame; the `LoopBegin` site identifies the
   loop in `SyncEvent::frames`. No guard predicate except on `Ptx`.
6. **Addresses**: `Load/Store` are buffer-relative (element offset in the
   *buffer's* dtype); every other memory op takes an address value +
   `AddrSpace`. Encodings (`arena::addr`): synthetic global VAs with guard
   gaps; shared::cta = window offset; shared::cluster = `(rank+1)<<24 |
   offset` with 0 meaning "own CTA", so every shared::cta address is a valid
   shared::cluster address; generic shared/local apertures; taddr =
   `lane<<16 | col`.
7. **Blocking**: `Instr::may_block()` derived from the variant; handlers
   return `Flow::Blocked(ResourceId)`, the scheduler re-runs the same pc.
   `Instr::is_progress()` feeds spin parking in `LoopEnd`.
8. **Sync**: protocol shapes were seeded as verbatim copies of
   `numsim-sync-ref` by `tools/port_sync.py`. That script was ONE-SHOT:
   production bodies now live in `numsim-core/src/sync/*.rs` and re-running
   it would delete them. Type changes are applied by hand in both crates and
   guarded by `tests/sync_differential.rs`. `SyncTable::step` lifts `Outcome::Blocked` to
   `Step::Blocked(ResourceId)`; `step_all` is all-or-nothing for
   multi-target instructions. Two queues: sync `Completion`s
   (`MbarTx/MbarArrive/GroupMilestone/SetmaxGrant`, generation captured at
   issue) and data `AsyncOp`s whose payload lands then queues completions.
9. **Observer**: one hot `access` callback and one cold `sync` callback
   (plus launch/warp/inbox lifecycle). Fences, async issue/complete,
   arrive/wait phases, wait verdicts, alloc lifetime and declared words are
   `SyncKind` variants (W5 shape); committed protocol commands with explicit
   counts, collectives, issued targets and observed parity are
   `SyncKind::Protocol` (W6 shape). `seq` counts committed protocol events
   only. No clocks or observed generations except W5's `Arrive/Wait.phase`.
10. **Single-threaded scheduler, one Arena** for now; CTA parallelism is a
    later internal change.
11. **Tile ops** remain as a provisional `Instr::Tile` with W1's element-map
    `TileLayout`; W1 prefers lowering through TVM's dispatch to PTX-level IR.
12. **Strict serde, `FORMAT_VERSION` 2** (contract review item 7): every
    program type rejects unknown fields and every `Option` field must be
    present (`null`), so a misspelled or dropped field is a decode error, not
    a silent default. `Program::validate` is exhaustive (indices, type widths,
    destination fit, frame-matched targets, predicate sub-program layout).
13. **Register width limit is 256 bits**; buffer/param dtypes may be wider
    (`boolx128` lowers to a `u8[128]` buffer, never a register value).
14. **Epochs are `u64`**; declared-word history bit 0 = launch value, bit i =
    the i-th (Access, lane) write in delivery order, lanes ascending, value =
    byte-merged post-image.
