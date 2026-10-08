"""v2 tile form: tirx.tile.sum/max/min (port of legacy analyze/tile_forms + emit)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import unported

if TYPE_CHECKING:
    from ..ir_walk import Lowerer


def lower(call: Any, ctx: "Lowerer") -> None:
    unported(call, "reduce")
