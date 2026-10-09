"""v2 tile form: tirx.tile.copy_async (port of legacy analyze/tile_forms/copy.rs
``resolve_copy_async`` + emit/tile_async_copy.rs).

Legacy spells the TMA dispatch ``"tma"``; TVM registers the same descriptor
inference as ``"tma_auto"`` (``"tma_explicit"`` for an explicit layout). The
:func:`repair` step renames the variant so TVM's own TMA lowering applies
(Decision 6: TVM first). Forms TVM still declines are lowered by :func:`lower`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tvm import tirx

from . import unported

if TYPE_CHECKING:
    from ..ir_walk import Lowerer

# Legacy copy_async dispatch names -> TVM's registered variant.
_LEGACY_VARIANTS = {"tma": "tma_auto"}


def repair(call: Any) -> Any | None:
    """The same call spelled with TVM's variant name, or None."""
    dispatch = str(call.dispatch) if call.dispatch is not None else None
    target = _LEGACY_VARIANTS.get(dispatch or "")
    if target is None:
        return None
    return tirx.TilePrimitiveCall(
        *call.args,
        op=call.op,
        workspace=dict(call.workspace),
        config=dict(call.config),
        dispatch=target,
        scope=call.scope,
    )


def lower(call: Any, ctx: Lowerer) -> None:
    unported(call, "copy_async")
