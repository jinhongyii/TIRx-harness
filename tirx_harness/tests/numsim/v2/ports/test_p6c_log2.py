"""v2 port of ``tests/numsim/runtime/test_log2.py::test_log2_matches_independent_reference``.

The legacy test built a legacy ``NumSimCase``/``ComparisonSpec`` and called
``numsim.run_case``; the port runs ``v2.Engine`` and compares with the same
tolerance (``rtol=atol=1e-6``) against ``np.log2``.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def log2_kernel(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.log2(T.cast(lane + 1, "float32"))


def test_log2_matches_independent_reference():
    """Port of ``tests/numsim/runtime/test_log2.py::test_log2_matches_independent_reference``."""

    result = v2.Engine().run(
        v2.transpile(log2_kernel), {"output": np.zeros(32, np.float32)}, outputs=("output",)
    )
    np.testing.assert_allclose(
        result.outputs["output"], np.log2(np.arange(1, 33, dtype=np.float32)), rtol=1e-6, atol=1e-6
    )
