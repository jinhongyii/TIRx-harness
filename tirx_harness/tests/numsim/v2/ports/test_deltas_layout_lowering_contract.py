"""v2 copy of ``tests/numsim/runtime/test_layout_lowering_contract.py::test_physical_buffers_preserve_coordinates_aliases_and_lane_private_storage``.

Delta (W5-9, racecheck-behaviour-deltas P7): views of one root with the same
dtype share the root's logical identity, so reading ``storage`` after writing
its same-dtype strided ``alias`` is no longer an ``alias_stale_read`` review;
racecheck is clean (legacy: one ``review`` ``alias_stale_read``). Synccheck
clean and the NumSim outputs are unchanged. Kernel copied verbatim.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def physical_buffers_with_strided_alias(output: T.Buffer((32, 3), "int32", layout=None)):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    compact = T.alloc_buffer((2, 4), "int32", scope="shared", layout=None)
    storage = T.alloc_buffer((16,), "int32", scope="shared", layout=None)
    alias = T.decl_buffer(
        (2, 4), "int32", data=storage.data, strides=(6, 1), elem_offset=1,
        scope="shared", layout=None,
    )
    private = T.alloc_buffer((2,), "int32", scope="local", layout=None)
    private[0] = lane * 10
    private[1] = private[0] + 3
    if lane < 16:
        storage[lane] = -1
    if lane < 8:
        compact[lane // 4, lane % 4] = lane + 100
    T.cuda.warp_sync()
    if lane < 8:
        alias[lane // 4, lane % 4] = lane + 200
    T.cuda.warp_sync()
    output[lane, 0] = compact[((31 - lane) % 8) // 4, (31 - lane) % 4]
    output[lane, 1] = storage[lane % 16]
    output[lane, 2] = private[1]


def test_physical_buffers_preserve_coordinates_aliases_and_lane_private_storage():
    bindings = {"output": np.zeros((32, 3), dtype=np.int32)}
    v2.synccheck(physical_buffers_with_strided_alias, bindings).require_clean()
    report = v2.racecheck(physical_buffers_with_strided_alias, bindings)
    assert report.verdict == "clean", report.format()
    assert [(f.status, f.kind) for f in report.findings] == [], report.format()

    result = v2.Engine().run(v2.transpile(physical_buffers_with_strided_alias), bindings)
    storage = np.full(16, -1, dtype=np.int32)
    storage[1:5] = np.arange(200, 204)
    storage[7:11] = np.arange(204, 208)
    lanes = np.arange(32, dtype=np.int32)
    expected = np.column_stack((100 + (31 - lanes) % 8, storage[lanes % 16], lanes * 10 + 3))
    np.testing.assert_array_equal(result.outputs["output"], expected)
