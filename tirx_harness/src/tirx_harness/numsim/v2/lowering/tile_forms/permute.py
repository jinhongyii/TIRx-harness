"""v2 tile form: ``tirx.tile.permute_layout`` where TVM rejects the layouts.

Legacy semantics (frontend-rs ``analyze/tile_forms/copy.rs::resolve_permute_layout``,
lowered by the same ``tile_emit_copy_or_cast`` as ``copy``): a warp-scope copy
of the *logical* values of the source region into the destination region's
physical layout. Layouts are applied by the ordinary buffer accesses, so the
copy is ``copy.lower_snapshot_copy`` (snapshot, warp sync, write, warp sync).

Legacy also marked a shared-memory source ``zero_fill_invalid_source``: an
element never written reads as zero. v2 reads every uninitialized byte as
zero already (``ValidityPolicy::ZeroAndReport``) and additionally reports the
read as ``uninitialized_read`` (numsim-behaviour-deltas, tile-form row).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..memory import _Unsupported
from .copy import check_common, lower_snapshot_copy, region

if TYPE_CHECKING:
    from ..ir_walk import Lowerer

SNAPSHOT_DTYPES = frozenset(
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
        "float8_e4m3fn",
        "float8_e8m0fnu",
    }
)


def lower(call: Any, ctx: Lowerer) -> None:
    scope = check_common(call)
    if scope != "warp":
        raise _Unsupported(call, f"tile op {call.op.name}: requires warp scope, got {scope}")
    if len(call.args) != 2:
        raise _Unsupported(call, f"tile op {call.op.name}: expected 2 args, got {len(call.args)}")
    dst = region(ctx, call, call.args[0], "dst")
    src = region(ctx, call, call.args[1], "src")
    if src.dtype not in SNAPSHOT_DTYPES:
        raise _Unsupported(
            call, f"tile op {call.op.name}: snapshot copy of {src.dtype} is not implemented"
        )
    lower_snapshot_copy(ctx, call, scope, dst, src)
