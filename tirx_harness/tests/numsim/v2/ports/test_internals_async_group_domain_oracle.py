"""v2 port of the legacy async-group wait-count execution test; the
``task_count``/``completed_task_count`` stats pins are dropped."""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def async_group_wait_counts(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(8)
    T.ptx.cp.async_.wait_group(0)
    T.ptx.cp.async_.bulk.commit_group()
    T.ptx.cp.async_.bulk.wait_group(64)
    T.ptx.cp.async_.bulk.wait_group.read(255)
    T.ptx.cp.async_.bulk.wait_group.read(2147483647)
    T.ptx.cp.async_.bulk.wait_group(0)
    if lane == 0:
        output[0] = 1


def test_async_group_wait_counts_complete_on_an_empty_queue():
    """Port of ``tests/numsim/runtime/test_async_group_domain_oracle.py::
    test_async_group_wait_counts_complete_on_an_empty_queue``.

    Dropped pins: ``result.stats["task_count"] == 1`` and
    ``result.stats["completed_task_count"] == 1``. Completion is still
    asserted through the public ``result.status``.
    """

    module = v2.transpile(async_group_wait_counts)
    result = v2.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.array([1], dtype=np.int32))
    assert result.status.get("kind") == "completed"
