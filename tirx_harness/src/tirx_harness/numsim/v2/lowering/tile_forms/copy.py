"""v2 tile form: ``tirx.tile.copy`` / ``cast`` / ``add`` where TVM registers no variant.

Port of the legacy frontend's canonical tile semantics (frontend-rs
``analyze/tile_forms/{copy,numeric,parse}.rs`` + ``emit/tile.rs``; engine-rs
``runtime/instructions/tile.rs::execute_typed_copy``). A ``dispatch=`` hint
never selects semantics; the canonical form is lowered with ordinary Program
ops by building TIR (loops, guards, loads, stores) and walking it.

Shared machinery (also used by ``fill``, ``permute`` and ``reduce``):

* **Regions.** A ``TensorRegion`` operand is a buffer, its region minima and
  static extents. Minima that are not literals are evaluated once, before the
  op (legacy ``tile_snapshot_copy_region_mins``).
* **Element owners.** A logical element is executed by one thread of the
  op's scope:

  - a *fragment* operand (local buffer whose layout maps to thread axes) is
    executed by the thread that owns the element in that layout;
  - otherwise, when the destination is a thread-private local, by every
    active thread (each owns its copy);
  - otherwise (memory operands) by thread ``linear % threads(scope)`` of the
    scope (thread scope: every active thread) — legacy ``scope_lanes``.

* **Participation.** ``warp`` / ``warpgroup`` tile ops require every lane of
  each executing warp (legacy ``collective::participate``, #594): a
  full-mask ``vote.sync.all`` raises ``warp_collective_divergence`` when
  lanes diverge. A ``warpgroup`` op adds no four-warp rendezvous.
* **Scope sync** (legacy ``copy_scope_sync``): warp ``bar.warp.sync``,
  warpgroup ``bar.sync 8, 128``, CTA ``bar.sync 0``.
* **Snapshot copy** (both operands memory, or both local): every owner reads
  its source elements into registers, the scope synchronizes, the owners
  write, the scope synchronizes again. A copy between a register fragment
  and memory has no snapshot and no sync (legacy ``NoSnapshotSync``).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable

import tvm
from tvm import tirx

from .. import program_builder as pb
from ..dtypes import type_key
from ..memory import RegArray, Shape, _Unsupported, handle

if TYPE_CHECKING:
    from ..ir_walk import Lowerer

THREAD_AXES = ("laneid", "tid_in_wg", "tid_in_cta", "wid_in_wg", "warpid")
SCOPES = ("thread", "warp", "warpgroup", "cta")
# Most registers one thread may hold for a snapshot or an accumulator table.
MAX_SNAPSHOT = 4096
_COPY_CONFIG = frozenset({"cache", "l1_evict", "l2_evict", "prefetch_size", "vec_len"})


def _i32(value: int) -> Any:
    return tirx.IntImm("int32", value)


@dataclasses.dataclass
class Region:
    var: Any                 # buffer Var
    mins: list[Any]          # PrimExpr per axis (snapshotted)
    extents: list[int]
    dtype: str
    scope: str               # "local" / "shared" / "global" / ...
    layout: Any

    @property
    def numel(self) -> int:
        count = 1
        for extent in self.extents:
            count *= extent
        return count

    @property
    def logical_shape(self) -> list[int]:
        return [e for e in self.extents if e != 1] or [1]

    @property
    def thread_axes(self) -> tuple[str, ...]:
        return layout_thread_axes(self.layout)

    @property
    def kind(self) -> str:
        if self.thread_axes:
            return "fragment"
        return "private" if self.scope == "local" else "memory"

    def indices(self, coords: list[Any]) -> list[Any]:
        out = []
        for minimum, coord in zip(self.mins, coords):
            dtype = str(minimum.ty.dtype) if hasattr(minimum, "ty") else "int32"
            coord = coord if str(coord.ty.dtype) == dtype else tirx.Cast(dtype, coord)
            out.append(tirx.Add(minimum, coord) if not _is_zero(minimum) else coord)
        return out

    def load(self, coords: list[Any]) -> Any:
        return self.var[tuple(self.indices(coords))]

    def store(self, coords: list[Any], value: Any, span: Any) -> Any:
        return tirx.BufferStore(self.var, value, self.indices(coords), span=span)

    def buffer_flat(self, coords: list[Any]) -> Any:
        """Row-major flat index into the whole buffer (the layout's input)."""
        shape = [int(s) for s in self.var.ty.shape]
        flat = None
        for extent, index in zip(shape, self.indices(coords)):
            index = index if str(index.ty.dtype) == "int32" else tirx.Cast("int32", index)
            flat = index if flat is None else tirx.Add(tirx.Mul(flat, _i32(extent)), index)
        return flat if flat is not None else _i32(0)


def _is_zero(expr: Any) -> bool:
    return type_key(expr) == "ir.IntImm" and int(expr.value) == 0


def layout_thread_axes(layout: Any) -> tuple[str, ...]:
    if layout is None or type_key(layout) != "tirx.TileLayout":
        return ()
    try:
        mapped = layout.apply(_i32(0))
    except Exception:  # noqa: BLE001
        return ()
    return tuple(sorted(str(k) for k in mapped.keys() if str(k) in THREAD_AXES))


def scope_of(call: Any) -> str:
    scope = str(getattr(call.scope, "name", call.scope))
    if scope not in SCOPES:
        raise _Unsupported(call, f"tile op {call.op.name}: exec scope {scope!r} is not implemented")
    return scope


def op_name(call: Any) -> str:
    return str(call.op.name).rpartition(".")[2]


def check_common(call: Any, allowed_config: frozenset[str] = frozenset()) -> str:
    scope = scope_of(call)
    if len(call.workspace) != 0:
        raise _Unsupported(call, f"tile op {call.op.name}: non-empty workspace is not implemented")
    unknown = sorted(str(k) for k in dict(call.config) if str(k) not in allowed_config)
    if unknown:
        raise _Unsupported(call, f"tile op {call.op.name}: unsupported config keys {unknown}")
    return scope


def region(ctx: "Lowerer", call: Any, arg: Any, role: str) -> Region:
    """A ``TensorRegion`` operand with its minima evaluated once (legacy snapshot)."""
    if not hasattr(arg, "region"):
        raise _Unsupported(call, f"tile op {call.op.name}: {role} must be a buffer region")
    var = arg.source
    extents = []
    for axis, item in enumerate(arg.region):
        if type_key(item.extent) != "ir.IntImm":
            raise _Unsupported(call, f"tile op {call.op.name}: {role} extent {axis} is not static")
        extents.append(int(item.extent.value))
    if any(e <= 0 for e in extents):
        raise _Unsupported(call, f"tile op {call.op.name}: {role} region extents must be positive, got {extents}")
    mins = []
    for axis, item in enumerate(arg.region):
        minimum = item.min
        if type_key(minimum) != "ir.IntImm":
            dtype = str(minimum.ty.dtype)
            if not dtype.startswith(("int", "uint")):
                raise _Unsupported(call, f"tile op {call.op.name}: {role} minimum {axis} must be integer")
            snap = tirx.Var(f"{role}_min{axis}", dtype)
            ctx.vars[handle(snap)] = ctx.expr(minimum)
            minimum = snap
        mins.append(minimum)
    scope = str(var.ty.storage_scope)
    if scope.startswith("shared"):
        scope = "shared"
    if ctx.ref_of(var) is None:
        raise _Unsupported(call, f"tile op {call.op.name}: {role} buffer {var.name} is not declared")
    return Region(var=var, mins=mins, extents=extents, dtype=str(var.ty.dtype.dtype), scope=scope,
                  layout=getattr(var.ty, "layout", None))


def scalar(call: Any, arg: Any, role: str) -> tuple[Any, str]:
    """A scalar operand as a TIR expression and its dtype (legacy ``parse::operand``)."""
    if isinstance(arg, bool):
        return tirx.IntImm("bool", int(arg)), "bool"
    if isinstance(arg, int):
        dtype = "int32" if -(1 << 31) <= arg < (1 << 31) else "int64"
        return tirx.IntImm(dtype, arg), dtype
    if isinstance(arg, float):
        return tirx.FloatImm("float64", arg), "float64"
    ty = getattr(arg, "ty", None)
    dtype = str(getattr(ty, "dtype", "")) if ty is not None else ""
    if not dtype or hasattr(arg, "region"):
        raise _Unsupported(call, f"tile op {call.op.name}: {role} must be a typed scalar")
    return arg, dtype


def cast(value: Any, dtype: str) -> Any:
    return value if str(value.ty.dtype) == dtype else tirx.Cast(dtype, value)


def unflatten(linear: Any, extents: list[int]) -> list[Any]:
    out, rest = [], linear
    for axis in reversed(range(len(extents))):
        extent = _i32(extents[axis])
        out.append(tirx.FloorMod(rest, extent) if axis else rest)
        rest = tirx.FloorDiv(rest, extent)
    return list(reversed(out))


def broadcast_coords(dst_extents: list[int], src_extents: list[int], coords: list[Any]) -> list[Any]:
    """Right-aligned broadcast of destination coordinates onto a source region."""
    pad = len(dst_extents) - len(src_extents)
    return [_i32(0) if extent == 1 else coords[pad + axis] for axis, extent in enumerate(src_extents)]


def thread_var(ctx: "Lowerer", axis: str) -> Any:
    var = tirx.Var(f"tile_{axis}", "int32")
    ctx.vars[handle(var)] = ctx.thread_coordinate(axis)
    return var


def scope_threads(ctx: "Lowerer", scope: str) -> int:
    if scope == "warp":
        return 32
    if scope == "warpgroup":
        return 128
    return int(ctx.builder.program.topology.warps_per_cta) * 32


def scope_thread(ctx: "Lowerer", scope: str) -> Any:
    return thread_var(ctx, {"warp": "laneid", "warpgroup": "tid_in_wg", "cta": "tid_in_cta"}[scope])


def fragment_guard(ctx: "Lowerer", reg: Region, coords: list[Any]) -> Any:
    """``this thread owns reg[coords]`` under the region's register layout."""
    mapped = {str(k): v for k, v in reg.layout.apply(reg.buffer_flat(coords)).items()}
    cond = None
    for axis in reg.thread_axes:
        term = tirx.EQ(cast(mapped[axis], "int32"), thread_var(ctx, axis))
        cond = term if cond is None else tirx.And(cond, term)
    return cond


def fragment_owner(regions: list[Region], call: Any) -> Region | None:
    """The fragment operand whose owners execute the op, or None."""
    fragments = [r for r in regions if r.kind == "fragment"]
    if not fragments:
        return None
    first = fragments[0]
    for other in fragments[1:]:
        if other.extents != first.extents or not tvm.ir.structural_equal(
                other.layout.canonicalize(), first.layout.canonicalize()) or [
                str(m) for m in other.mins] != [str(m) for m in first.mins]:
            # Values move between threads: owner transport (owner_transport.py)
            # handles whole-buffer element-wise forms; this one is not modeled.
            raise _Unsupported(call, f"tile op {call.op.name}: fragment operands with different thread owners")
    return first


def element_loop(ctx: "Lowerer", call: Any, scope: str, numel: int, owner: Region | None, private: bool,
                 body: Callable[[Any], Any], extents: list[int] | None = None) -> Any:
    """``for linear in elements this thread executes: body(linear)`` as TIR.

    ``owner``: the fragment whose layout owns each element (its extents index
    the elements); ``private``: every active thread executes every element;
    otherwise the scope's semantic owner ``linear % threads(scope)``.
    """
    span = call.span
    if owner is not None:
        linear = tirx.Var("tile_linear", "int32")
        guard = fragment_guard(ctx, owner, unflatten(linear, extents or owner.extents))
        return tirx.For(linear, _i32(0), _i32(numel), tirx.ForKind.SERIAL,
                        tirx.IfThenElse(guard, body(linear), None, span=span), span=span)
    if private or scope == "thread":
        linear = tirx.Var("tile_linear", "int32")
        return tirx.For(linear, _i32(0), _i32(numel), tirx.ForKind.SERIAL, body(linear), span=span)
    threads = scope_threads(ctx, scope)
    me = scope_thread(ctx, scope)
    slot = tirx.Var("tile_slot", "int32")
    linear = tirx.Add(tirx.Mul(slot, _i32(threads)), me)
    inner = body(linear)
    if numel % threads:
        inner = tirx.IfThenElse(tirx.LT(linear, _i32(numel)), inner, None, span=span)
    slots = (numel + threads - 1) // threads
    return tirx.For(slot, _i32(0), _i32(slots), tirx.ForKind.SERIAL, inner, span=span)


def participate(ctx: "Lowerer", call: Any, scope: str) -> None:
    """Warp / warpgroup tile ops execute with every lane of the warp (#594)."""
    if scope not in ("warp", "warpgroup"):
        return
    dst = ctx.builder.reg(pb.Ty("Pred"))
    ctx.builder.emit("Vote", site=ctx.site(call), mode="All", dst=dst, pred=ctx.const(pb.Ty("Pred"), 1),
                     membermask=ctx.full_mask())


def scope_sync(ctx: "Lowerer", call: Any, scope: str) -> None:
    if scope == "thread":
        return
    if scope == "warp":
        ctx.builder.emit("WarpSync", site=ctx.site(call), membermask=ctx.full_mask())
    elif scope == "warpgroup":
        ctx.barrier(call, "Sync", ctx.const("uint32", 8), ctx.const("uint32", 128))
    else:
        ctx.barrier(call, "Sync", ctx.const("uint32", 0), None)


def scratch(ctx: "Lowerer", call: Any, count: int, dtype: str, name: str) -> Any:
    """A per-thread register table of ``count`` ``dtype`` values, as a TIR buffer."""
    if count > MAX_SNAPSHOT:
        raise _Unsupported(call, f"tile op {call.op.name}: {count} snapshot registers per thread (limit {MAX_SNAPSHOT})")
    var = tirx.decl_buffer((count,), dtype, name=name, scope="local")
    ty = ctx.ty(dtype, call)
    ctx.refs[handle(var)] = RegArray(regs=[ctx.builder.reg(ty, name=name) for _ in range(count)],
                                    info=Shape(dtype=dtype, shape=(_i32(count),), strides=(), layout=None))
    return var


def lower_snapshot_copy(ctx: "Lowerer", call: Any, scope: str, dst: Region, src: Region) -> None:
    """Legacy ``tile::copy`` (``execute_typed_copy``) and the owner-driven register copy."""
    if dst.logical_shape != src.logical_shape:
        raise _Unsupported(call, f"tile op {call.op.name}: logical shape mismatch {dst.logical_shape} != {src.logical_shape}")
    if dst.dtype != src.dtype:
        raise _Unsupported(call, f"tile op {call.op.name}: dtype mismatch {dst.dtype} != {src.dtype}")
    for reg in (dst, src):
        if reg.scope not in ("global", "local", "shared"):
            raise _Unsupported(call, f"tile op {call.op.name}: unsupported memory pair {src.scope}->{dst.scope}")
    numel = dst.numel
    owner = fragment_owner([dst, src], call)
    # Both regions index the same logical element by their own extents.
    dst_coords = lambda linear: unflatten(linear, dst.extents)  # noqa: E731
    src_coords = lambda linear: unflatten(linear, src.extents)  # noqa: E731
    private = dst.kind == "private" and owner is None
    span = call.span
    participate(ctx, call, scope)
    if owner is not None and (dst.scope == "local") != (src.scope == "local"):
        # Register fragment <-> memory: each owner moves its own elements, no
        # snapshot and no scope sync (legacy NoSnapshotSync).
        ctx.stmt(element_loop(ctx, call, scope, numel, owner, False,
                              lambda linear: dst.store(dst_coords(linear), src.load(src_coords(linear)), span),
                              extents=owner.extents))
        return
    snap = scratch(ctx, call, numel, src.dtype, f"{src.var.name}.snapshot")
    ctx.stmt(element_loop(ctx, call, scope, numel, owner, private,
                          lambda linear: tirx.BufferStore(snap, src.load(src_coords(linear)), [linear], span=span),
                          extents=owner.extents if owner is not None else None))
    scope_sync(ctx, call, scope)
    ctx.stmt(element_loop(ctx, call, scope, numel, owner, private,
                          lambda linear: dst.store(dst_coords(linear), snap[linear], span),
                          extents=owner.extents if owner is not None else None))
    scope_sync(ctx, call, scope)


# ---------------------------------------------------------------- element-wise
_BINARY = {"add": tirx.Add, "sub": tirx.Sub, "mul": tirx.Mul, "fdiv": tirx.Div}
ELEMENTWISE = frozenset({"cast", "fill", *_BINARY})


def lower_elementwise(ctx: "Lowerer", call: Any, op: str, dst: Region, operands: list[Any]) -> None:
    """Legacy ``tile_emit_elementwise`` / cast: one value per destination element.

    ``operands`` are ``Region``s (broadcast right-aligned onto the destination)
    or scalar TIR expressions. Storage must be all local or all shared
    (legacy ``validate_elementwise_storage``); a shared op completes with a
    scope sync.
    """
    scope = scope_of(call)
    regions = [dst, *[o for o in operands if isinstance(o, Region)]]
    scopes = sorted({r.scope for r in regions})
    if scopes not in (["local"], ["shared"]):
        raise _Unsupported(call, f"tile op {call.op.name}: element-wise operands must all reside in local or all "
                                 f"in shared memory, got {scopes}")
    for reg in regions[1:]:
        if len(reg.extents) > len(dst.extents) or any(
                e != 1 and e != dst.extents[len(dst.extents) - len(reg.extents) + i] for i, e in enumerate(reg.extents)):
            raise _Unsupported(call, f"tile op {call.op.name}: source shape {reg.extents} cannot broadcast to {dst.extents}")
    owner = fragment_owner(regions, call)
    if owner is not None and owner is not dst and owner.extents != dst.extents:
        raise _Unsupported(call, f"tile op {call.op.name}: broadcast fragment owner")
    private = owner is None and scopes == ["local"]
    span = call.span

    def body(linear: Any) -> Any:
        coords = unflatten(linear, dst.extents)
        values = []
        for operand in operands:
            if isinstance(operand, Region):
                values.append(operand.load(broadcast_coords(dst.extents, operand.extents, coords)))
            else:
                values.append(operand)
        return dst.store(coords, elementwise_value(call, op, values, dst.dtype), span)

    participate(ctx, call, scope)
    ctx.stmt(element_loop(ctx, call, scope, dst.numel, owner, private, body, extents=dst.extents))
    if scopes == ["shared"]:
        scope_sync(ctx, call, scope)


def elementwise_value(call: Any, op: str, values: list[Any], dtype: str) -> Any:
    if op in ("cast", "fill"):
        return cast(values[0], dtype)
    if op in _BINARY and len(values) == 2:
        return _BINARY[op](cast(values[0], dtype), cast(values[1], dtype))
    raise _Unsupported(call, f"tile op {call.op.name}: element-wise op {op} with {len(values)} operands")


# ----------------------------------------------------------------------- entry
def repair(call: Any) -> Any | None:
    """The call without its ``dispatch=`` hint, or None.

    A hint never selects tile semantics (legacy contract,
    ``test_copy_dispatch_contract.py``): a hint naming a variant this TVM does
    not register (``reg``, ``gmem_smem``) or one that rejects the operands
    (``smem`` on locals) is dropped so TVM's dispatch picks its own variant.
    """
    if call.dispatch is None:
        return None
    return tirx.TilePrimitiveCall(*call.args, op=call.op, workspace=dict(call.workspace),
                                  config=dict(call.config), dispatch=None, scope=call.scope)


def lower(call: Any, ctx: "Lowerer") -> None:
    op = op_name(call)
    if op == "copy":
        scope = check_common(call, _COPY_CONFIG)
        if len(call.args) != 2:
            raise _Unsupported(call, f"tile op {call.op.name}: expected 2 args, got {len(call.args)}")
        # Cache hints (``cache``/``l1_evict``/...) have no numerical or ordering effect.
        dst = region(ctx, call, call.args[0], "dst")
        src = region(ctx, call, call.args[1], "src")
        lower_snapshot_copy(ctx, call, scope, dst, src)
        return
    if op == "cast":
        check_common(call)
        if len(call.args) != 2:
            raise _Unsupported(call, f"tile op {call.op.name}: expected 2 args, got {len(call.args)}")
        dst = region(ctx, call, call.args[0], "dst")
        src = region(ctx, call, call.args[1], "src")
        lower_elementwise(ctx, call, "cast", dst, [src])
        return
    if op in _BINARY:
        check_common(call)
        if len(call.args) != 3:
            raise _Unsupported(call, f"tile op {call.op.name}: expected 3 args, got {len(call.args)}")
        dst = region(ctx, call, call.args[0], "dst")
        operands = [region(ctx, call, a, f"src{i}") if hasattr(a, "region") else scalar(call, a, f"src{i}")[0]
                    for i, a in enumerate(call.args[1:])]
        lower_elementwise(ctx, call, op, dst, operands)
        return
    raise _Unsupported(call, f"tile op {call.op.name}: no v2 copy-family form")
