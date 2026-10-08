"""v2 copies of two lazy-evaluation tests from ``tests/numsim/integration/test_artifact_build.py``.

Dropped pins: ``ctx.set_active_mask(and_rhs_mask_/or_rhs_mask_/select_then_mask/
select_else_mask`` substrings of the legacy generated Rust. The outputs (the
observable contract: an impure right-hand side or unselected branch load runs
only on the lanes that need it) are kept. v2 lowers ``&&``, ``||`` and
``T.Select`` with impure operands lazily under If/Else (W1). Kernels copied
verbatim / imported from the pure-TVM ``tests/numsim/support/kernels.py``.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.support.kernels import guarded_select_load
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def lazy_logical_buffer_load(source: T.Buffer((1,), "int32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    and_value: T.let = (lane == 0) and (source[lane] == 5)
    or_value: T.let = (lane != 0) or (source[lane] == 5)
    output[lane] = T.cast(and_value, "int32") + T.cast(or_value, "int32") * 2


def test_logical_and_or_only_evaluate_rhs_for_required_active_lanes():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_logical_and_or_only_evaluate_rhs_for_required_active_lanes``."""

    result = v2.Engine().run(
        v2.transpile(lazy_logical_buffer_load),
        {"source": np.array([5], dtype=np.int32), "output": np.zeros(32, dtype=np.int32)},
    )
    expected = np.full(32, 2, dtype=np.int32)
    expected[0] = 3
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_select_does_not_evaluate_an_unselected_nested_buffer_load():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_select_does_not_evaluate_an_unselected_nested_buffer_load``."""

    source = np.array([17, 23], dtype=np.uint64)
    expected = np.arange(100, 132, dtype=np.uint32)
    expected[:2] = source.astype(np.uint32)
    result = v2.Engine().run(
        v2.transpile(guarded_select_load), {"source": source, "output": np.zeros(32, dtype=np.uint32)}
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)
