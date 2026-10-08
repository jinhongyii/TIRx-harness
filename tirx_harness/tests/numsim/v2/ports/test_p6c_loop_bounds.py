"""v2 ports of the remaining legacy loop-bounds tests; the
``module.rust_source`` text pins (native loop scaffolding names) are dropped
and the observable outputs are kept. Kernels are copied verbatim."""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def varying_loop_control(extents: T.Buffer((32,), "int32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = 0
    for iteration in T.serial(extents[lane]):
        if iteration == 1:
            if lane % 2 == 0:
                continue
        if iteration == 2:
            if lane % 3 == 0:
                break
        output[lane] = output[lane] + 1


@T.prim_func
def static_single_trip_for(inside: T.Buffer((32,), "int32"), after: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    inside[lane] = 0
    for iteration in T.serial(1):
        if lane < 8:
            continue
        inside[lane] = iteration + 1
        if lane < 16:
            break
        inside[lane] = inside[lane] + 2
    after[lane] = 9


def test_lane_varying_loop_supports_break_and_continue():
    """Port of ``tests/numsim/runtime/test_loop_bounds.py::test_lane_varying_loop_supports_break_and_continue``.

    Dropped pins: ``"live_mask_loop"`` and
    ``"ctx.set_active_mask(WarpMask::EMPTY)"`` in ``module.rust_source``.
    """

    extents = np.arange(32, dtype=np.int32) % 6
    expected = np.zeros(32, dtype=np.int32)
    for lane, extent in enumerate(extents):
        for iteration in range(int(extent)):
            if iteration == 1 and lane % 2 == 0:
                continue
            if iteration == 2 and lane % 3 == 0:
                break
            expected[lane] += 1

    module = v2.transpile(varying_loop_control)
    result = v2.Engine().run(module, {"extents": extents, "output": np.zeros(32, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_static_single_trip_for_preserves_masks_and_loop_path():
    """Port of ``tests/numsim/runtime/test_loop_bounds.py::test_static_single_trip_for_preserves_masks_and_loop_path``.

    Dropped pins: ``"live_mask_loop"`` in, ``"loop_offset_loop"`` and
    ``"For loop offset overflow"`` not in ``module.rust_source``.
    """

    args = {"inside": np.zeros(32, dtype=np.int32), "after": np.zeros(32, dtype=np.int32)}
    module = v2.transpile(static_single_trip_for)
    result = v2.Engine().run(module, args)

    expected = np.full(32, 3, dtype=np.int32)
    expected[:8] = 0
    expected[8:16] = 1
    np.testing.assert_array_equal(result.outputs["inside"], expected)
    np.testing.assert_array_equal(result.outputs["after"], np.full(32, 9, dtype=np.int32))
