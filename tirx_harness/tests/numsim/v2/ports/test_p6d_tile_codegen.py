"""v2 delta copies of five ``tests/numsim/runtime/test_tile_codegen.py`` tests.

All five are "other assertion" failures under ``NUMSIM_IMPL=v2`` and all five
are explained by Decision 6, "tile semantics are TVM's dispatch output"
(``docs/development/lowering-inventory.md`` D.2; coordinator ruling quoted in
``docs/development/numsim-behaviour-deltas.md``, section "Tile reductions ...
Decision 6": "v2 runs the code TVM's tile dispatch emits for the GPU"; same
ruling as the earlier ``test_triage_tile_general_semantics`` row of
``scripts/numsim-v2/coverage/other_assertion_triage.tsv``). The legacy frontend
emitted its own round-to-nearest, non-FTZ element loops; TVM's dispatch
(``TilePrimitiveDispatch`` at ``sm_100a``) emits:

- contiguous f32 pairs ``Tx.add/sub/mul/fma``: ``add/sub/mul/fma.rz.ftz.f32x2``
  (``binary_f32x2.py`` default ``rounding_mode="rz"``), so results round toward
  zero and subnormal inputs/outputs flush to zero;
- single-element ``Tx.add(..., rounding_mode=...)``: the scalar fallback emits a
  plain TIR ``+`` and drops ``rounding_mode`` (all four results are RN);
- ``Tx.sum`` over 8 lane-local f32: an ``add.rn.ftz.f32x2`` pair tree, then one
  plain ``+`` of the two halves (subnormal inputs flush);
- ``Tx.wg.add(tile[:, 1:17], tile[:, 0:16], 0)``: an in-place loop over f32x2
  pairs, read pair ``2f, 2f+1`` then write ``2f+1, 2f+2``, so a later pair reads
  an element an earlier pair already overwrote (legacy snapshotted the source).

Each copy keeps the legacy kernel and inputs and asserts the value the
dispatched code computes (host model below). Dropped pins: none besides the
legacy expected values (no ``rust_source`` assertion in these five).
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import wg_local_layout

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

_TINY = np.float32(np.finfo(np.float32).tiny)


def _f32(bits: int) -> np.float32:
    return np.asarray([bits], dtype=np.uint32).view(np.float32)[0]


def _ftz(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.where(np.abs(values) < _TINY, np.copysign(np.float32(0), values), values).astype(np.float32)


def _round_rz(exact) -> np.ndarray:
    """binary64 ``exact`` (exact for these operands) rounded toward zero to binary32."""
    exact = np.asarray(exact, dtype=np.float64)
    nearest = exact.astype(np.float32)
    overshoot = np.abs(nearest.astype(np.float64)) > np.abs(exact)
    return np.where(overshoot, np.nextafter(nearest, np.float32(0)), nearest).astype(np.float32)


def _rz_ftz(exact) -> np.ndarray:
    return _ftz(_round_rz(exact))


def _add_rn_ftz(a, b) -> np.ndarray:
    return _ftz((_ftz(a).astype(np.float64) + _ftz(b).astype(np.float64)).astype(np.float32))


def _run(kernel, inputs):
    return v2.Engine().run(v2.transpile(kernel), inputs)


# -- kernels (verbatim from tests/numsim/support/kernels.py and the legacy test) --


@T.prim_func
def tile_directed_rounding(output: T.Buffer((32, 4), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    lhs: T.f32[1]
    rhs: T.f32[1]
    nearest: T.f32[1]
    down: T.f32[1]
    up: T.f32[1]
    zero: T.f32[1]
    lhs[0] = T.float32(1)
    rhs[0] = T.float32(5.960464477539063e-08)
    Tx.add(nearest, lhs, rhs, rounding_mode="rn")
    Tx.add(down, lhs, rhs, rounding_mode="rm")
    Tx.add(up, lhs, rhs, rounding_mode="rp")
    Tx.add(zero, lhs, rhs, rounding_mode="rz")
    output[lane, 0] = nearest[0]
    output[lane, 1] = down[0]
    output[lane, 2] = up[0]
    output[lane, 3] = zero[0]


@T.prim_func
def packed_f32_default_contract(
    source: T.Buffer((32, 18), "float32"), output: T.Buffer((32, 8), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    lhs: T.f32[2]
    rhs: T.f32[2]
    addend: T.f32[2]
    result: T.f32[2]

    for index in T.serial(2):
        lhs[index] = source[lane, index]
        rhs[index] = source[lane, 2 + index]
    Tx.add(result, lhs, rhs)
    output[lane, 0] = result[0]
    output[lane, 1] = result[1]

    for index in T.serial(2):
        lhs[index] = source[lane, 4 + index]
        rhs[index] = source[lane, 6 + index]
    Tx.sub(result, lhs, rhs)
    output[lane, 2] = result[0]
    output[lane, 3] = result[1]

    for index in T.serial(2):
        lhs[index] = source[lane, 8 + index]
        rhs[index] = source[lane, 10 + index]
    Tx.mul(result, lhs, rhs)
    output[lane, 4] = result[0]
    output[lane, 5] = result[1]

    for index in T.serial(2):
        lhs[index] = source[lane, 12 + index]
        rhs[index] = source[lane, 14 + index]
        addend[index] = source[lane, 16 + index]
    Tx.fma(result, lhs, rhs, addend)
    output[lane, 6] = result[0]
    output[lane, 7] = result[1]


@T.prim_func
def owner_driven_scalar_load_fallback(
    source: T.Buffer((128, 16), "float32"),
    scale: T.Buffer((128,), "float32"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    tid = T.thread_id_in_wg([128])
    tile = T.alloc_buffer((128, 16), "float32", scope="local", layout=wg_local_layout(16))
    for col in T.serial(16):
        tile[tid, col] = source[tid, col]
    Tx.wg.mul(tile[:, :], tile[:, :], scale[tid])
    for col in T.serial(16):
        output[tid, col] = tile[tid, col]


@T.prim_func
def owner_driven_shifted_alias_fallback(output: T.Buffer((128, 17), "float32")):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    tid = T.thread_id_in_wg([128])
    tile = T.alloc_buffer((128, 17), "float32", scope="local", layout=wg_local_layout(17))
    for col in T.serial(17):
        tile[tid, col] = T.cast(tid * 100 + col, "float32")
    Tx.wg.add(tile[:, 1:17], tile[:, 0:16], T.float32(0))
    for col in T.serial(17):
        output[tid, col] = tile[tid, col]


@T.prim_func
def tile_reductions(source: T.Buffer((32, 8), "float32"), output: T.Buffer((32, 4), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    row: T.f32[8]
    sum_value: T.f32[1]
    max_value: T.f32[1]
    min_value: T.f32[1]
    Tx.copy(row[:], source[lane, 0:8])
    Tx.sum(sum_value, row)
    Tx.max(max_value, row, dispatch="local")
    Tx.min(min_value, row, dispatch="local")
    output[lane, 0] = sum_value[0]
    output[lane, 1] = max_value[0]
    output[lane, 2] = min_value[0]
    sum_value[0] = T.float32(10)
    Tx.sum(sum_value, row, accum=True)
    output[lane, 3] = sum_value[0]


# -- tests -------------------------------------------------------------------


def test_explicit_elementwise_rounding_is_layout_independent_and_not_ftz():
    """Delta copy of ``tests/numsim/runtime/test_tile_codegen.py::test_explicit_elementwise_rounding_is_layout_independent_and_not_ftz``.

    Decision 6: the single-element ``Tx.add`` dispatches to the scalar fallback,
    a plain TIR ``+`` with ``rounding_mode`` dropped, so ``1 + 2**-24`` rounds
    to nearest (1.0) for all four requested modes. Legacy honoured ``rp``
    (``0x3F800001`` in column 2). (Possibly a TVM dispatch issue; v2 runs what
    the dispatch emits.)"""

    result = _run(tile_directed_rounding, {"output": np.zeros((32, 4), dtype=np.float32)})
    np.testing.assert_array_equal(result.outputs["output"], np.ones((32, 4), dtype=np.float32))


def test_contiguous_f32_uses_canonical_rn_without_ftz():
    """Delta copy of ``tests/numsim/runtime/test_tile_codegen.py::test_contiguous_f32_uses_canonical_rn_without_ftz``.

    Decision 6: contiguous f32 pairs dispatch to ``{add,sub,mul,fma}.rz.ftz.f32x2``.
    Legacy expected RN without FTZ. Kept: legacy inputs; the expectation is
    round-toward-zero with subnormal inputs and results flushed."""

    smallest_subnormal = _f32(1)
    smallest_normal = _f32(0x0080_0000)
    one_up = _f32(0x3F80_0001)
    mul_lhs = _f32(0x3FD0_0281)
    mul_rhs = _f32(0x3FE6_CB52)
    three_quarter_ulp = np.float32(3 * 2**-25)
    quarter_ulp = np.float32(2**-25)
    row = np.asarray(
        [
            1.0, smallest_normal, three_quarter_ulp, smallest_subnormal,
            one_up, smallest_normal, quarter_ulp, -smallest_subnormal,
            mul_lhs, smallest_normal, mul_rhs, 0.5,
            1.0, smallest_subnormal, 1.0, np.ldexp(np.float32(1), 126),
            three_quarter_ulp, 0.0,
        ],
        dtype=np.float32,
    )
    source = np.tile(row, (32, 1))
    result = _run(packed_f32_default_contract, {"source": source, "output": np.zeros((32, 8), dtype=np.float32)})

    r = _ftz(row).astype(np.float64)
    expected = _rz_ftz(
        [
            r[0] + r[2], r[1] + r[3],
            r[4] - r[6], r[5] - r[7],
            r[8] * r[10], r[9] * r[11],
            r[12] * r[14] + r[16], r[13] * r[15] + r[17],
        ]
    )
    # Concrete values: 1+0.75ulp -> 1.0 (RZ), normal+subnormal -> normal (input FTZ),
    # normal*0.5 -> 0 (output FTZ), subnormal*2**126 -> 0 (input FTZ).
    assert expected[0] == np.float32(1.0) and expected[5] == 0 and expected[7] == 0
    np.testing.assert_array_equal(result.outputs["output"], np.tile(expected, (32, 1)))


def test_thread_owned_scalar_buffer_load_uses_canonical_rn():
    """Delta copy of ``tests/numsim/runtime/test_tile_codegen.py::test_thread_owned_scalar_buffer_load_uses_canonical_rn``.

    Decision 6: ``Tx.wg.mul(tile, tile, scale[tid])`` dispatches to
    ``mul.rz.ftz.f32x2`` (scale broadcast into both halves). 996 of 2048
    products are one ulp below legacy's RN. Kept: inputs, physical owners."""

    source = np.arange(128 * 16, dtype=np.float32).reshape(128, 16) / np.float32(32)
    scale = np.linspace(0.5, 1.5, 128, dtype=np.float32)
    result = _run(
        owner_driven_scalar_load_fallback,
        {"source": source, "scale": scale, "output": np.zeros_like(source)},
    )
    expected = _rz_ftz(source.astype(np.float64) * scale[:, None].astype(np.float64))
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_owner_driven_shifted_alias_falls_back():
    """Delta copy of ``tests/numsim/runtime/test_tile_codegen.py::test_owner_driven_shifted_alias_falls_back``.

    Decision 6: the dispatched ``add.rz.ftz.f32x2`` loop runs in place over
    pairs: iteration ``f`` reads columns ``2f, 2f+1`` and writes ``2f+1, 2f+2``,
    so iteration ``f+1`` reads column ``2f+2`` after iteration ``f`` wrote it.
    Legacy snapshotted the source (``out[:, c] = c - 1``). The copy models the
    dispatched loop."""

    result = _run(owner_driven_shifted_alias_fallback, {"output": np.zeros((128, 17), dtype=np.float32)})

    tile = np.arange(128, dtype=np.float32)[:, None] * np.float32(100) + np.arange(17, dtype=np.float32)
    for f in range(8):
        a, b = tile[:, 2 * f].copy(), tile[:, 2 * f + 1].copy()
        tile[:, 2 * f + 1] = a + np.float32(0)
        tile[:, 2 * f + 2] = b + np.float32(0)
    # e.g. row 0 is 0, 0, 1, 1, 3, 3, ..., 13, 13, 15 (legacy: 0, 0, 1, 2, ..., 15).
    np.testing.assert_array_equal(tile[0, :5], np.float32([0, 0, 1, 1, 3]))
    np.testing.assert_array_equal(result.outputs["output"], tile)


def test_reduction_preserves_subnormal_inputs_in_lexicographic_order():
    """Delta copy of ``tests/numsim/runtime/test_tile_codegen.py::test_reduction_preserves_subnormal_inputs_in_lexicographic_order``.

    Decision 6 (``numsim-behaviour-deltas.md`` R1 family): ``Tx.sum`` of 8
    lane-local f32 dispatches to an ``add.rn.ftz.f32x2`` pair tree plus a final
    plain ``+``. The subnormal ``0x807FFFFF`` flushes to -0, so the sum is
    ``FLT_MIN`` (``0x00800000``); legacy summed sequentially without FTZ and
    got ``0x00000001``."""

    source = np.zeros((32, 8), dtype=np.float32)
    source[:, 0] = _TINY
    source[:, 4] = _f32(0x807F_FFFF)
    result = _run(tile_reductions, {"source": source, "output": np.zeros((32, 4), dtype=np.float32)})

    s = source
    lo = _add_rn_ftz(s[:, 0:2], s[:, 2:4])  # (l0+l2, l1+l3)
    hi = _add_rn_ftz(s[:, 4:6], s[:, 6:8])  # (l4+l6, l5+l7)
    pair = _add_rn_ftz(lo, hi)
    expected = (pair[:, 0] + pair[:, 1]).astype(np.float32)
    assert np.all(expected.view(np.uint32) == np.uint32(0x0080_0000))
    np.testing.assert_array_equal(result.outputs["output"][:, 0], expected)
