"""v2 copies of three tile-reduction tests whose legacy expectations pinned the
legacy frontend's own sequential, identity-seeded reductions.

Ruling (coordinator, from ``numsim-core/CONTRACT_REQUESTS.md`` W4-12; the
matching rows are being added to ``docs/development/numsim-behaviour-deltas.md``
by W4): v2 lowers ``tirx.tile.*`` through TVM's own dispatch (Decision 6), so
it computes what the dispatched code computes on hardware:

* warp collectives reduce with a ``shfl.sync.bfly`` tree (xor 16, 8, 4, 2, 1),
  not a lane-0-first sequential fold;
* the ``3input_maxmin`` dispatch has no ``-FLT_MAX`` / ``+FLT_MAX`` seed, so an
  all-NaN input gives the canonical NaN ``0x7FFFFFFF``.

Expectations are computed by an independent butterfly model rather than
copied from a run.
"""

from __future__ import annotations

import numpy as np

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import R, S, TileLayout, laneid

pytestmark = requires_v2_engine


# Copied verbatim from tests/numsim/runtime/test_tile_reduction_variants.py.
@T.prim_func
def warp_collective_reductions(
    source: T.Buffer((32, 2), "float32"), output: T.Buffer((3, 32, 2), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source_local: T.f32[2]
    result_local: T.f32[2]
    for column in T.serial(2):
        source_local[column] = source[lane, column]
    source_view = source_local.view(32, 2, layout=TileLayout(S[(32, 2) : (1 @ laneid, 1)]))
    result_view = result_local.view(2, layout=TileLayout(S[2:1] + R[32 : 1 @ laneid]))
    Tx.warp.sum(result_view, source_view, axes=[0], dispatch="local")
    for column in T.serial(2):
        output[0, lane, column] = result_local[column]
    Tx.warp.max(result_view, source_view, axes=[0], dispatch="local")
    for column in T.serial(2):
        output[1, lane, column] = result_local[column]
    Tx.warp.min(result_view, source_view, axes=[0], dispatch="local")
    for column in T.serial(2):
        output[2, lane, column] = result_local[column]


@T.prim_func
def three_input_maxmin_order(
    source: T.Buffer((2, 8), "float32"), output: T.Buffer((4, 32), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    values: T.f32[8]
    result: T.f32[1]
    for row in T.serial(2):
        for index in T.serial(8):
            values[index] = source[row, index]
        Tx.max(result, values, dispatch="3input_maxmin")
        output[row * 2, lane] = result[0]
        Tx.min(result, values, dispatch="3input_maxmin")
        output[row * 2 + 1, lane] = result[0]


def _butterfly_sum(column: np.ndarray) -> np.ndarray:
    """``v += shfl.bfly(v, m)`` for m = 16, 8, 4, 2, 1, in binary32."""

    values = column.astype(np.float32).copy()
    lanes = np.arange(32)
    for mask in (16, 8, 4, 2, 1):
        values = (values + values[lanes ^ mask]).astype(np.float32)
    return values


def _run_warp_collectives(source: np.ndarray) -> np.ndarray:
    module = v2.transpile(warp_collective_reductions)
    output = np.zeros((3, 32, 2), dtype=np.float32)
    return v2.Engine().run(module, {"source": source, "output": output}).outputs["output"]


def test_warp_collective_reduction_follows_physical_lane_ownership():
    """Port of ``tests/numsim/runtime/test_tile_reduction_variants.py::test_warp_collective_reduction_follows_physical_lane_ownership``.

    Delta (W4-12 ruling): the sum is the butterfly tree of the dispatched
    code, not the legacy lane-ordered fold. Max and min are order-free and
    keep the legacy expectation."""

    source = np.linspace(-4, 3, 64, dtype=np.float32).reshape(32, 2)
    result = _run_warp_collectives(source)

    expected_sum = np.stack([_butterfly_sum(source[:, column]) for column in range(2)], axis=1)
    np.testing.assert_array_equal(result[0].view(np.uint32), expected_sum.view(np.uint32))
    np.testing.assert_array_equal(result[1], np.broadcast_to(source.max(axis=0), (32, 2)))
    np.testing.assert_array_equal(result[2], np.broadcast_to(source.min(axis=0), (32, 2)))


def test_local_collective_uses_lexicographic_order():
    """Port of ``tests/numsim/runtime/test_tile_reduction_variants.py::test_local_collective_uses_lexicographic_order``.

    Delta (W4-12 ruling): legacy folded lanes 0..31 in order, giving
    ``((1e20 + 1) - 1e20) + 1 = 1.0``. The dispatched butterfly pairs lanes 0
    and 2 first (xor 16, 8, 4 add zeros, then xor 2), so ``1e20`` and
    ``-1e20`` cancel exactly and the result is ``2.0`` in every lane."""

    source = np.zeros((32, 2), dtype=np.float32)
    source[:4, 0] = np.asarray([1.0e20, 1.0, -1.0e20, 1.0], dtype=np.float32)
    result = _run_warp_collectives(source)

    assert np.array_equal(_butterfly_sum(source[:, 0]), np.full(32, 2.0, dtype=np.float32))
    np.testing.assert_array_equal(result[0, :, 0], np.full(32, 2.0, dtype=np.float32))


def test_maxmin_uses_canonical_lexicographic_nan_and_signed_zero_order():
    """Port of ``tests/numsim/runtime/test_tile_reduction_variants.py::test_maxmin_uses_canonical_lexicographic_nan_and_signed_zero_order``.

    Rows with at least one number are unchanged (NaN-ignoring ``max``/``min``,
    ``-0 < +0``). Delta (W4-12 ruling): the ``3input_maxmin`` dispatch has no
    ``-FLT_MAX``/``+FLT_MAX`` seed, so the all-NaN row gives the canonical NaN
    ``0x7FFFFFFF`` for both max and min (legacy: ``0xFF7FFFFF`` /
    ``0x7F7FFFFF``)."""

    nan = np.asarray([0x7FC0_1234], dtype=np.uint32).view(np.float32)[0]
    source = np.asarray(
        [
            [nan, -0.0, 0.0, nan, -np.inf, -np.inf, nan, nan],
            [nan, nan, nan, nan, nan, nan, nan, nan],
        ],
        dtype=np.float32,
    )
    output = np.zeros((4, 32), dtype=np.float32)
    result = v2.Engine().run(v2.transpile(three_input_maxmin_order), {"source": source, "output": output})

    bits = result.outputs["output"].view(np.uint32)
    assert np.all(bits[0] == np.uint32(0x0000_0000))
    assert np.all(bits[1] == np.uint32(0xFF80_0000))
    assert np.all(bits[2] == np.uint32(0x7FFF_FFFF))
    assert np.all(bits[3] == np.uint32(0x7FFF_FFFF))
