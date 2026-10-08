"""v2 tile form: ``tirx.tile.fill`` where TVM registers no variant.

Port of legacy ``resolve_unary_elementwise(Fill)`` + ``tile_emit_elementwise``:
the scalar (a typed expression or an untyped Python literal: ``int`` ->
``int32``/``int64``, ``float`` -> ``float64``, ``bool``) is cast to the
destination dtype and stored to every destination element by its owner
(``copy.lower_elementwise``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..memory import _Unsupported
from .copy import check_common, lower_elementwise, region, scalar

if TYPE_CHECKING:
    from ..ir_walk import Lowerer

FILL_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64",
                         "uint8", "uint16", "uint32", "uint64", "bool", "float8_e4m3fn"})


def lower(call: Any, ctx: "Lowerer") -> None:
    check_common(call)
    if len(call.args) != 2:
        raise _Unsupported(call, f"tile op {call.op.name}: expected 2 args, got {len(call.args)}")
    dst = region(ctx, call, call.args[0], "dst")
    value, dtype = scalar(call, call.args[1], "fill.value")
    for kind in (dst.dtype, dtype):
        if kind not in FILL_DTYPES:
            raise _Unsupported(call, f"tile op {call.op.name}: dtype {kind} is not implemented")
    lower_elementwise(ctx, call, "fill", dst, [value])
