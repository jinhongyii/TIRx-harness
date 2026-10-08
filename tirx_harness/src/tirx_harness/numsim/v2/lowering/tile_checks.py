"""Pre-dispatch checks for tile ops that TVM's dispatch would accept silently.

TVM's ``TilePrimitiveDispatch`` ignores config keys it does not read and does
not verify that ``tile.gemm`` fragment layouts match the fixed ``mma.sync``
register ABI; legacy NumSim rejected both. These checks keep that fail-closed
behaviour (reasons as legacy frontend-rs ``analyze/tile_forms``).
"""

from __future__ import annotations

import itertools
from typing import Any

from .dtypes import type_key

# Tile config keys any legacy tile form accepted (frontend-rs analyze/tile_forms
# `unknown_keys` lists, union over ops).
TILE_CONFIG_KEYS = frozenset(
    {
        "cache",
        "cache_hint",
        "cta_group",
        "cta_mask",
        "descI",
        "gather4",
        "is_AB_tf32",
        "l1_evict",
        "l2_evict",
        "l2_promotion",
        "mbar",
        "mbarrier_addr",
        "mma_m",
        "mma_n",
        "multicast",
        "oob",
        "pred",
        "prefetch_size",
        "prefetch_tensormap",
        "remote_cta_id",
        "rounding_mode",
        "shape",
        "smem_desc",
        "tensormap_l2_promotion",
        "thread_reduce",
        "tma_dtype",
        "use_tma_reduce",
        "vec_len",
        "weight_stationary",
    }
)


def tile_rejection(node: Any) -> str | None:
    """Legacy fail-closed reason for tile call ``node``, or None."""
    op = str(node.op.name).rpartition(".")[2]
    keys = sorted(str(k) for k in node.config.keys() if str(k) not in TILE_CONFIG_KEYS)
    if keys:
        return f"TilePrimitiveCall({op}): unsupported config keys {keys}"
    mode = node.config.get("rounding_mode") if "rounding_mode" in node.config else None
    if mode is not None and str(mode) not in ("rn", "") and _touches_float64(node):
        # TVM's dispatch emits a plain round-to-nearest f64 op for these.
        return f"TilePrimitiveCall({op}): directed float64 rounding is not implemented"
    if op == "gemm" and "warp" in str(node.scope) and "warpgroup" not in str(node.scope):
        return _gemm_fragment_rejection(node)
    return None


def _touches_float64(node: Any) -> bool:
    for arg in node.args:
        source = getattr(arg, "source", None)
        ty = getattr(source, "ty", None) if source is not None else getattr(arg, "ty", None)
        dtype = getattr(ty, "dtype", None)
        if dtype is not None and str(dtype) == "float64":
            return True
    return False


def _abi(role: str, row: int, col: int, mma_k: int) -> tuple[int, int, tuple[int, int]]:
    if role in ("D", "C"):
        return (
            4 * (row % 8) + (col % 8) // 2,
            2 * ((row % 16) // 8) + col % 2,
            (row // 16, col // 8),
        )
    if role == "A":
        return (
            4 * (row % 8) + (col % 8) // 2,
            4 * ((col % mma_k) // 8) + 2 * ((row % 16) // 8) + col % 2,
            (row // 16, col // mma_k),
        )
    return (
        4 * (col % 8) + (row % 8) // 2,
        2 * ((row % mma_k) // 8) + row % 2,
        (row // mma_k, col // 8),
    )


def _region(arg: Any) -> tuple[Any, list[int]] | None:
    """(layout, storage shape) of a whole-buffer local region, else None (not checked)."""
    if type_key(arg) != "tirx.TensorRegion" and not hasattr(arg, "region"):
        return None
    ty = getattr(arg.source, "ty", None)
    layout = getattr(ty, "layout", None) if ty is not None else None
    if ty is None or layout is None or type_key(layout) != "tirx.TileLayout":
        return None
    try:
        shape = [int(s) for s in ty.shape]
        starts = [int(r.min) for r in arg.region]
        extents = [int(r.extent) for r in arg.region]
    except (TypeError, ValueError, AttributeError):
        return None
    if len(shape) != 2 or starts != [0, 0] or extents != shape:
        return None
    return layout, shape


def _check_fragment(
    layout: Any, shape: list[int], role: str, rows: int, cols: int, mma_k: int, transpose: bool
) -> str | None:
    from tvm import tirx

    if len(layout.replica) != 0:
        return (
            f"TilePrimitiveCall(gemm): {role} fragment layout has replica axes, which "
            f"mma.sync.m16n8k{mma_k} does not support"
        )
    bases: dict[tuple[int, int], int] = {}
    for row in range(rows):
        for col in range(cols):
            srow, scol = (col, row) if transpose else (row, col)
            mapped = {
                str(k): v
                for k, v in layout.apply(tirx.IntImm("int32", srow * shape[1] + scol)).items()
            }
            if set(mapped) != {"laneid", "m"} or any(
                type_key(v) != "ir.IntImm" for v in mapped.values()
            ):
                return (
                    f"TilePrimitiveCall(gemm): {role} fragment layout must map exactly to laneid and m "
                    f"for mma.sync.m16n8k{mma_k}, got {sorted(mapped)}"
                )
            lane, slot = int(mapped["laneid"].value), int(mapped["m"].value)
            expected_lane, expected_slot, tile = _abi(role, row, col, mma_k)
            if lane != expected_lane:
                return (
                    f"TilePrimitiveCall(gemm): {role} fragment layout does not match fixed "
                    f"mma.sync.m16n8k{mma_k} ABI at logical ({row}, {col}): expected laneid={expected_lane}, "
                    f"got {lane}"
                )
            base = bases.setdefault(tile, slot - expected_slot)
            if slot - expected_slot != base:
                return (
                    f"TilePrimitiveCall(gemm): {role} fragment layout does not match fixed "
                    f"mma.sync.m16n8k{mma_k} ABI at logical ({row}, {col}): register slot {slot} is not "
                    f"ABI slot {expected_slot} relative to one fragment base"
                )
    per_tile = {"D": 4, "C": 4, "A": mma_k // 2, "B": mma_k // 4}[role]
    spans = sorted(bases.items(), key=lambda item: item[1])
    for (tile, base), (other, other_base) in itertools.pairwise(spans):
        if base + per_tile > other_base:
            return (
                f"TilePrimitiveCall(gemm): {role} instruction tiles {tile} and {other} may alias the same "
                f"physical registers for mma.sync.m16n8k{mma_k}"
            )
    return None


def _gemm_fragment_rejection(node: Any) -> str | None:
    args = list(node.args)
    if len(args) < 6:
        return None
    found = [_region(a) for a in args[:4]]
    regions = [r for r in found if r is not None]
    if len(regions) != len(found):
        return None  # not a whole-buffer register-fragment gemm; TVM decides
    try:
        trans_a, trans_b = bool(args[4]), bool(args[5])
    except (TypeError, ValueError):
        return None
    (_, d_shape), (_, a_shape), (_, b_shape), (_, c_shape) = regions
    m, k = (a_shape[1], a_shape[0]) if trans_a else (a_shape[0], a_shape[1])
    b_k, n = (b_shape[1], b_shape[0]) if trans_b else (b_shape[0], b_shape[1])
    if b_k != k or d_shape != [m, n] or c_shape != [m, n] or m % 16 or n % 8:
        return None  # shape errors are TVM's to report
    failures = []
    for mma_k in (16, 8):
        if k % mma_k:
            continue
        checks = (
            ("D", regions[0], m, n, False),
            ("A", regions[1], m, k, trans_a),
            ("B", regions[2], k, n, trans_b),
            ("C", regions[3], m, n, False),
        )
        failure = None
        for role, (layout, shape), rows, cols, transpose in checks:
            failure = _check_fragment(layout, shape, role, rows, cols, mma_k, transpose)
            if failure is not None:
                break
        if failure is None:
            return None
        failures.append(failure)
    detail = f"; first rejected candidate: {failures[0]}" if failures else ""
    return f"TilePrimitiveCall(gemm): no mma.sync.m16n8k{{16,8}} instruction matches M={m}, N={n}, K={k}{detail}"


__all__ = ["TILE_CONFIG_KEYS", "tile_rejection"]
