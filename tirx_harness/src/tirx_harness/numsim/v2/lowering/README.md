# NumSim v2 lowering

Turns a TIRx `PrimFunc` into a `Module` of `Program`s (the Rust contract in
`core-rs/numsim-core/src/program.rs`). Entry points: `lower(func, strict=)` and
`lower_module(funcs)` in `ir_walk.py`. The op-family map (family → file:function)
is `docs/development/lowering-inventory.md` B.0; tile forms are Part G.

## Pipeline

1. **Source walk** (`ir_walk.lower`): records loops written `T.vectorized`.
2. **Tile dispatch** (`ir_walk._dispatch_tile_primitives`): legacy rules first
   (`tile_checks`); cross-owner element-wise functions go to
   `owner_transport.py`; everything else runs TVM's `TilePrimitiveDispatch`.
   An op TVM rejects becomes a `numsim_v2_tile_form` placeholder that
   `tile_forms/` lowers later.
3. **Pre-pass** (`Lowerer.lower`): host prelude split (`host_prelude`), escaped
   and promotable locals (`memory`), maybe-uninitialized locals (`uninit`),
   launch topology (`collect_topology`).
4. **Walk**: `stmt_*` per statement kind, `expr` per expression;
   `calls.call` for `ir.Call`. `tirx.ptx.*` table ops go through
   `ptx_decode.decode` → `ptx_lower.lower_ptx` → `handler_for`.
5. **Build** (`program_builder`): every `Instr` is checked against `SCHEMA` as
   it is emitted. `finish` sizes the shared pools, checks the per-CTA shared
   capacity, sets `requirements`, and appends `wait_until` predicate programs.
6. **Validate**: `strict=True` raises `LoweringUnsupported` with every reason;
   the Rust side runs `Program::validate` on load (`scripts/numsim-v2/validate.sh`).

## Invariants

- **Fail closed.** A form the lowering cannot represent records a reason
  (`raise _Unsupported(node, reason)` inside a handler) and emits an
  `Unsupported` instruction; it never approximates. Unknown statement kinds,
  builtins, PTX forms and tile ops all land there.
- **Guarded ops run inside their guard.** A predicated PTX op is wrapped whole
  in `If(pred)` (`lower_ptx`): operands, addresses and write-backs are not
  evaluated by predicated-off lanes. `if_then_else` with impure arms lowers
  to `If`/`Else`, never `Select`.
- **Owner rules.** A register-fragment element belongs to the thread its layout
  names; an access to another thread's element is a run-time `Assert`
  anchored at the tile call (`memory.flat_offset`). Moving values between
  owners needs an explicit transport (`owner_transport`, `tile_forms/copy`).
- **One logical buffer per pointer operand.** `site(..., operands=)` fills
  `SiteInfo.buffers` in PTX operand order with the root buffer of each view
  chain; the view's own name goes to the site text.
- **Offsets count `dtype.elem`.** Load/Store offsets are in scalar elements of
  the buffer's element type, not vector elements.
- **Pools are their allocation.** A view never grows its pool; a zero-extent
  pool is sized by its views.

Fail-closed rules worth knowing (messages are matched by tests): launch
topology (`topology: …`), user-vectorized loops and unknown loop annotations,
per-CTA shared capacity on sm_100/103, replicated TMEM views
(`tmem_replicated_view`), spdecompress register overlap, reviewed
`cuda.func_call` helper bodies, `cuda.ldg` overloads, rejected PTX families
(`ptx_lower.REJECTED`). The deltas against legacy are
`docs/development/numsim-behaviour-deltas.md`.

## Adding an op

1. Find its family in inventory B.0. A CUDA builtin `tirx.cuda.foo` lowers in
   `calls.py:call_tirx_cuda_foo`; a pure unary or helper is a table entry in
   `builtins.py`; a PTX table op is a handler in `ptx_lower.py`
   (`_EXACT`/`_PREFIX`); a tile op TVM rejects is a form in `tile_forms/`.
2. Emit existing `Instr` variants. A new engine-visible effect needs a contract
   variant (request it in `core-rs/numsim-core/CONTRACT_REQUESTS.md`), then the
   matching `SCHEMA` entry in `program_builder.py`.
3. Add a Program-content test under `tests/numsim/v2/test_lowering_*.py`
   (assert the emitted instructions, not only that lowering succeeds).
4. Run `ruff check`, `ruff format --check` and `mypy -p
   tirx_harness.numsim.v2.lowering` (from `tirx_harness/src`), `tests/numsim/v2`,
   and the corpus sweep (`scripts/numsim-v2/lower_sweep.py`).
