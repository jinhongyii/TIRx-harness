"""v2 tile form: ``tirx.tile.fill`` where TVM registers no variant.

Port of legacy ``resolve_unary_elementwise(Fill)`` + ``tile_emit_elementwise``:
the scalar (a typed expression or an untyped Python literal: ``int`` ->
``int32``/``int64``, ``float`` -> ``float64``, ``bool``) is cast to the
destination dtype and stored to every destination element by its owner
(``copy.lower_elementwise``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tvm import tirx

from ..memory import _Unsupported
from .copy import check_common, lower_elementwise, region, scalar

if TYPE_CHECKING:
    from ..ir_walk import Lowerer

FILL_DTYPES = frozenset(
    {
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bool",
        "float8_e4m3fn",
    }
)


def repair(call: Any) -> Any | None:
    """The call as TVM's ``fill`` accepts it, or None.

    The value of an untyped Python literal is the literal converted to the
    destination dtype (legacy: ``int`` -> ``int32``/``int64``, ``float`` ->
    ``float64``, then ``Cast(dst)``), so it is folded to a typed constant; a
    ``dispatch=`` hint never selects semantics and is dropped.
    """
    args = list(call.args)
    changed = call.dispatch is not None
    if len(args) == 2 and isinstance(args[1], (bool, int, float)) and hasattr(args[0], "source"):
        dtype = str(args[0].source.ty.dtype.dtype)
        if dtype not in FILL_DTYPES or dtype == "bool":
            return None
        value = args[1]
        if dtype.startswith(("int", "uint")) and isinstance(value, float):
            return None  # float -> integer conversion stays with the v2 form
        args[1] = tirx.const(value, dtype)
        changed = True
    if not changed:
        return None
    return tirx.TilePrimitiveCall(
        *args,
        op=call.op,
        workspace=dict(call.workspace),
        config=dict(call.config),
        dispatch=None,
        scope=call.scope,
    )


def lower(call: Any, ctx: Lowerer) -> None:
    check_common(call)
    if len(call.args) != 2:
        raise _Unsupported(call, f"tile op {call.op.name}: expected 2 args, got {len(call.args)}")
    dst = region(ctx, call, call.args[0], "dst")
    value, dtype = scalar(call, call.args[1], "fill.value")
    for kind in (dst.dtype, dtype):
        if kind not in FILL_DTYPES:
            raise _Unsupported(call, f"tile op {call.op.name}: dtype {kind} is not implemented")
    lower_elementwise(ctx, call, "fill", dst, [value])
