"""v2 ports of legacy tile forms (amended Decision 6).

TVM's ``TilePrimitiveDispatch`` lowers ``tirx.tile.*`` first. A tile call whose
form TVM does not register (or rejects) is lowered here instead, porting the
legacy frontend's ``analyze/tile_forms`` + emit path onto existing Program ops
(no new contract unless unavoidable; file CONTRACT_REQUESTS).

Hook (``ir_walk.dispatch_tile_primitives``): when TVM rejects a function, every
call of the op TVM names (``op=tirx.tile.<op>``) is replaced by a placeholder
``Evaluate(call_extern("int32", PLACEHOLDER, <id>))`` and dispatch is retried, so
TVM still lowers everything else. The walker then calls :func:`lower` on each
placeholder's original call; a family that cannot lower it raises
``Unsupported`` with the reason (TVM's dispatch error is kept for the report).

One module per family: ``gemm`` (gemm_async), ``copy`` (copy with forced or
unregistered variants), ``copy_async`` (TMA), ``permute`` (permute_layout),
``fill``, ``reduce`` (sum/max/min over shared/local).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from ..memory import _Unsupported

if TYPE_CHECKING:
    from ..ir_walk import Lowerer

PLACEHOLDER = "numsim_v2_tile_form"

# Calls swapped out of the TIR before TVM dispatch: id -> original TilePrimitiveCall.
PENDING: dict[int, Any] = {}


def _family(op: str) -> Callable[[Any, "Lowerer"], None] | None:
    from . import copy, copy_async, fill, gemm, permute, reduce

    return {
        "gemm_async": gemm.lower,
        "copy": copy.lower,
        "add": copy.lower,    # forced-variant element-wise forms (W12, copy.py)
        "cast": copy.lower,
        "copy_async": copy_async.lower,
        "permute_layout": permute.lower,
        "fill": fill.lower,
        "sum": reduce.lower,
        "max": reduce.lower,
        "min": reduce.lower,
    }.get(op)


def handles(op_name: str) -> bool:
    """Is there a v2 tile-form family for ``tirx.tile.<op>``?"""
    return _family(op_name.rpartition(".")[2]) is not None


def repair(call: Any) -> Any | None:
    """A rewrite of ``call`` that TVM's dispatch may accept (legacy spellings of a
    form TVM registers under another name), or None. Tried before swapping out."""
    op = str(call.op.name).rpartition(".")[2]
    if op == "copy_async":
        from . import copy_async

        return copy_async.repair(call)
    if op in ("copy", "cast", "add"):
        from . import copy

        return copy.repair(call)
    if op == "fill":
        from . import fill

        return fill.repair(call)
    if op in ("sum", "max", "min"):
        from . import reduce

        return reduce.repair(call)
    return None


def lower(call: Any, ctx: "Lowerer") -> None:
    """Lower one TVM-rejected tile call to Program ops, or raise ``Unsupported``."""
    op = str(call.op.name).rpartition(".")[2]
    family = _family(op)
    if family is None:
        raise _Unsupported(call, f"tile op tirx.tile.{op}: no v2 tile form")
    family(call, ctx)


def unported(call: Any, family: str) -> None:
    """Stub body: the family has no port for this form yet."""
    raise _Unsupported(call, f"tile op {call.op.name}: v2 tile form '{family}' not ported yet")


__all__ = ["PENDING", "PLACEHOLDER", "handles", "lower", "repair", "unported"]
