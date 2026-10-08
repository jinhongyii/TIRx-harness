"""v2 tile form: ``tirx.tile.sum`` / ``max`` / ``min`` where TVM registers no variant.

TVM's dispatch reduces ``local`` buffers only; legacy also accepted
reductions over ``shared`` regions (``Tx.cta.sum(shared_out, shared_in, ...)``).
Port of legacy ``resolve_reduction`` + ``tile_emit_reduction`` (frontend-rs
``analyze/tile_forms/numeric.rs``, ``emit/tile.rs``), non-collective form:

* each output element is computed by its owner (shared: thread
  ``linear % threads(scope)`` of the scope; local: every thread, or the
  fragment owner);
* ``acc = dst`` when ``accum`` else the identity (sum ``0``; max ``-MAX`` /
  integer minimum; min ``+MAX`` / integer maximum, ``MAX`` the dtype's largest
  finite value), then ``acc = acc (op) src[...]`` over the reduced
  coordinates in lexicographic order, rounded to the dtype at every step
  (f16/bf16: one rounding per step, exact through f32);
* the result is stored to the destination; a shared reduction completes with
  a scope sync. Warp / warpgroup ops require the full warp (``copy.participate``).

A ``dispatch=`` hint never selects semantics: ``repair`` drops it so TVM's
dispatch is tried first. Local warp-collective forms (laneid shard ->
replica) are TVM's (``Shfl`` butterfly, numsim-behaviour-deltas R1) and fail
closed here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tvm import tirx

from ..memory import _Unsupported
from .copy import (_i32, check_common, element_loop, fragment_owner, op_name, participate, region,
                   scope_sync, scratch, unflatten)

if TYPE_CHECKING:
    from ..ir_walk import Lowerer

_OPS = {"sum": tirx.Add, "max": tirx.Max, "min": tirx.Min}
_FLOAT_MAX = {"float16": 65504.0, "bfloat16": 3.3895313892515355e38, "float32": 3.4028234663852886e38,
              "float64": 1.7976931348623157e308}
_INTS = {"int8": 8, "int16": 16, "int32": 32, "int64": 64, "uint8": 8, "uint16": 16, "uint32": 32, "uint64": 64}


def identity(op: str, dtype: str) -> Any:
    if op == "sum":
        return tirx.const(0, dtype)
    if dtype in _INTS:
        bits = _INTS[dtype]
        lo, hi = (0, (1 << bits) - 1) if dtype.startswith("u") else (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        return tirx.const(lo if op == "max" else hi, dtype)
    biggest = _FLOAT_MAX[dtype]
    return tirx.const(-biggest if op == "max" else biggest, dtype)


def repair(call: Any) -> Any | None:
    if call.dispatch is None:
        return None
    return tirx.TilePrimitiveCall(*call.args, op=call.op, workspace=dict(call.workspace),
                                  config=dict(call.config), dispatch=None, scope=call.scope)


def _static_ints(call: Any, value: Any, role: str) -> list[int]:
    out = []
    for item in value:
        if isinstance(item, int):
            out.append(item)
        elif hasattr(item, "value") and isinstance(item.value, int):
            out.append(int(item.value))
        else:
            raise _Unsupported(call, f"tile op {call.op.name}: {role} must be static integers")
    return out


def lower(call: Any, ctx: "Lowerer") -> None:
    op = op_name(call)
    scope = check_common(call, frozenset({"thread_reduce"}))
    if dict(call.config):
        raise _Unsupported(call, f"tile op {call.op.name}: thread_reduce is a TVM warp-collective form")
    if len(call.args) != 4:
        raise _Unsupported(call, f"tile op {call.op.name}: expected 4 args, got {len(call.args)}")
    dst = region(ctx, call, call.args[0], "dst")
    src = region(ctx, call, call.args[1], "src")
    if dst.dtype != src.dtype or (src.dtype not in _INTS and src.dtype not in _FLOAT_MAX):
        raise _Unsupported(call, f"tile op {call.op.name}: source/destination must have the same supported "
                                 f"numeric dtype, got {src.dtype}/{dst.dtype}")
    rank = len(src.extents)
    axes: list[int] = []
    for raw in _static_ints(call, call.args[2], "axes"):
        axis = raw + rank if raw < 0 else raw
        if not 0 <= axis < rank:
            raise _Unsupported(call, f"tile op {call.op.name}: axis {raw} is outside rank {rank}")
        if axis not in axes:
            axes.append(axis)
    accum = call.args[3]
    accum = bool(accum.value) if hasattr(accum, "value") else bool(accum)
    spatial = [e for a, e in enumerate(src.extents) if a not in axes]
    spatial_logical = [e for e in spatial if e != 1] or [1]
    expected = 1
    for extent in spatial:
        expected *= extent
    dshape = dst.logical_shape
    exact = dshape == spatial_logical
    replicated = len(dshape) >= len(spatial_logical) and dshape[:len(spatial_logical)] == spatial_logical \
        and dst.numel % expected == 0
    if not exact and not replicated:
        raise _Unsupported(call, f"tile op {call.op.name}: destination logical shape {dshape} does not match "
                                 f"reduced shape {spatial_logical}")
    replication = dst.numel // expected
    if dst.scope != src.scope or dst.scope not in ("local", "shared"):
        raise _Unsupported(call, f"tile op {call.op.name}: reduction scopes must match and be local or shared, "
                                 f"got {src.scope}/{dst.scope}")
    if dst.scope == "local" and scope == "cta":
        raise _Unsupported(call, f"tile op {call.op.name}: local reduction does not support CTA scope")
    if dst.scope == "local" and scope == "warp" and any(a not in dst.thread_axes for a in src.thread_axes):
        raise _Unsupported(call, f"tile op {call.op.name}: local warp-collective reduction is TVM's form")

    owner = fragment_owner([dst, src], call) if dst.scope == "local" else None
    if owner is not None and owner is not dst:
        raise _Unsupported(call, f"tile op {call.op.name}: reduction owned by a source fragment")
    private = dst.scope == "local" and owner is None
    reduce_extents = [src.extents[a] for a in axes]
    count = 1
    for extent in reduce_extents:
        count *= extent
    acc = scratch(ctx, call, 1, dst.dtype, f"{dst.var.name}.{op}_acc")
    combine = _OPS[op]
    span = call.span

    def body(linear: Any) -> Any:
        dst_coords = unflatten(linear, dst.extents)
        spatial_linear = tirx.FloorDiv(linear, _i32(replication)) if replication != 1 else linear
        spatial_coords = iter(unflatten(spatial_linear, spatial) if spatial else [])
        r = tirx.Var("tile_reduce", "int32")
        reduce_coords = iter(unflatten(r, reduce_extents) if reduce_extents else [])
        coords = [next(reduce_coords) if a in axes else next(spatial_coords) for a in range(rank)]
        init = dst.load(dst_coords) if accum else identity(op, dst.dtype)
        step = tirx.BufferStore(acc, combine(acc[0], src.load(coords)), [_i32(0)], span=span)
        return tirx.SeqStmt([
            tirx.BufferStore(acc, init, [_i32(0)], span=span),
            tirx.For(r, _i32(0), _i32(count), tirx.ForKind.SERIAL, step, span=span),
            dst.store(dst_coords, acc[0], span),
        ])

    participate(ctx, call, scope)
    ctx.stmt(element_loop(ctx, call, scope, dst.numel, owner, private, body, extents=dst.extents))
    if dst.scope == "shared":
        scope_sync(ctx, call, scope)


__all__ = ["identity", "lower", "repair"]
