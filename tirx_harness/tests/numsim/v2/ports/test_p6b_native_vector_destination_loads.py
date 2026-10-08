"""v2 port of the legacy Synccheck vector destination load test.

Legacy: tests/analysis_tools/racecheck/test_native_vector_destination_loads.py::test_synccheck_accepts_the_ordered_vector_destination_load_kernels
ran the private ``tirx_harness.numsim.checkers._run_synccheck`` on the
barrier-ordered `ld.shared.v2` and `ld.acquire.cta.shared.v2` kernels and
required clean reports. The v2 copy runs ``v2.synccheck`` on the same kernels
and inputs (one parametrization per kernel of the legacy loop). Dropped: the
``cache_dir`` argument (legacy build cache).
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import assert_clean, assert_no_incomplete, requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


# --- copied from tests/analysis_tools/racecheck/test_native_vector_destination_loads.py ---

@T.prim_func
def barrier_ordered_v2_destination_load(output: T.Buffer((64,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "int32", scope="shared")
    pair = T.alloc_local((2,), "int32")
    if warp == 0:
        for index in T.unroll(2):
            shared[lane * 2 + index] = lane * 2 + index
    T.cuda.cta_sync()
    if warp == 1:
        T.ptx.ld.shared.v2.s32(pair[0], pair[1], shared.ptr_to([lane * 2]))
        for index in T.unroll(2):
            output[lane * 2 + index] = pair[index]


@T.prim_func
def barrier_ordered_acquire_v2_destination_load(output: T.Buffer((64,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "int32", scope="shared")
    pair = T.alloc_local((2,), "int32")
    if warp == 0:
        for index in T.unroll(2):
            shared[lane * 2 + index] = lane * 2 + index
    T.cuda.cta_sync()
    if warp == 1:
        T.ptx.ld.acquire.cta.shared.v2.s32(pair[0], pair[1], shared.ptr_to([lane * 2]))
        for index in T.unroll(2):
            output[lane * 2 + index] = pair[index]


@pytest.mark.parametrize(
    "kernel",
    [barrier_ordered_v2_destination_load, barrier_ordered_acquire_v2_destination_load],
    ids=["v2_destination_load", "acquire_v2_destination_load"],
)
def test_synccheck_accepts_the_ordered_vector_destination_load_kernels(kernel):
    """Port of tests/analysis_tools/racecheck/test_native_vector_destination_loads.py::test_synccheck_accepts_the_ordered_vector_destination_load_kernels.

    Dropped: ``cache_dir`` only.
    """

    report = v2.synccheck(kernel, {"output": np.zeros(64, dtype=np.int32)})
    assert_clean(report)
    assert_no_incomplete(report)
