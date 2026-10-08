"""v2 delta copies of two ``tests/numsim/registry/test_copy_dispatch_contract.py``
functions (W9 phase 6, internal other-assertion triage).

**Delta**, Decision 6 (``lowering-inventory.md`` D.2; the ruling already used
for the ``test_tile_general_semantics`` / ``test_tile_unary_codegen`` triage
rows): v2 runs TVM's dispatched code for tile primitives. For a
``Tx.warp.copy`` between two thread-local buffers with no layout every faster
variant is rejected and TVM picks ``copy/fallback`` ("scalar single-thread",
``tvm/backend/cuda/tile_primitive/copy/fallback.py``), which guards the body
with ``if tid == first_tid``: only lane 0 copies its own local buffer. Legacy
gave the copy its own semantics (every active lane copies). No
``numsim-behaviour-deltas`` row names this case (W8 may want one).

- ``test_auto_local_to_local_copy_runs_for_every_active_lane``: lane 0 reads
  ``(0.5, 1.5)``; lanes 1..31 keep their ``-1`` initial values.
- ``test_single_element_local_copy_runs_for_every_active_lane``: lane 0 reads
  ``0``; lanes 1..31 read their never-written ``destination`` (zero-filled).
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def _auto_local_to_local_copy(output: T.Buffer((32, 2), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source: T.f32[2]
    destination: T.f32[2]
    source[0] = T.cast(lane, "float32") + T.float32(0.5)
    source[1] = T.cast(lane, "float32") + T.float32(1.5)
    destination[0] = T.float32(-1)
    destination[1] = T.float32(-1)
    Tx.warp.copy(destination, source)
    output[lane, 0] = destination[0]
    output[lane, 1] = destination[1]


@T.prim_func
def _invalid_degenerate_local_copy(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source: T.f32[1]
    destination: T.f32[1]
    source[0] = T.cast(lane, "float32")
    Tx.warp.copy(destination, source)
    output[lane] = destination[0]


def test_auto_local_to_local_copy_runs_for_every_active_lane():
    """Delta copy of ``tests/numsim/registry/test_copy_dispatch_contract.py::test_auto_local_to_local_copy_runs_for_every_active_lane``."""

    result = v2.Engine().run(
        v2.transpile(_auto_local_to_local_copy), {"output": np.zeros((32, 2), dtype=np.float32)}
    )

    expected = np.full((32, 2), -1, dtype=np.float32)
    expected[0] = (0.5, 1.5)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_single_element_local_copy_runs_for_every_active_lane():
    """Delta copy of ``tests/numsim/registry/test_copy_dispatch_contract.py::test_single_element_local_copy_runs_for_every_active_lane``."""

    result = v2.Engine().run(
        v2.transpile(_invalid_degenerate_local_copy), {"output": np.zeros((32,), dtype=np.float32)}
    )

    np.testing.assert_array_equal(result.outputs["output"], np.zeros(32, dtype=np.float32))
