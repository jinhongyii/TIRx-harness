"""v2 port of the legacy mixed-readiness blocking-wait test; the legacy
message text ("cannot suspend lanes with different readiness or
barrier/phase") is not pinned. Kernel copied verbatim."""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def lane_varying_blocking_wait(output: T.Buffer((2,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")

    if lane < 2:
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([lane]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if lane < 2:
        T.cuda.mbarrier_wait(barriers.ptr_to([lane]), 1 - lane)
        if lane == 0:
            T.ptx.mbarrier.arrive.shared.b64(barriers.ptr_to([1]))
        output[lane] = lane + 7


def _stops(error: v2.ExecutionError) -> list[dict]:
    return [d for d in error.diagnostics if d.get("status") in ("error", "incomplete")]


def test_blocking_wait_fails_closed_for_mixed_lane_readiness():
    """Port of ``tests/numsim/runtime/test_mbarrier_lane_semantics.py::test_blocking_wait_fails_closed_for_mixed_lane_readiness``.

    Fail-closed half: the run raises ``v2.ExecutionError`` with a stopping
    diagnostic. Dropped pin: the legacy message text. Whether the stop is an
    ``error`` is :func:`test_blocking_wait_mixed_lane_readiness_is_an_error`.
    """

    module = v2.transpile(lane_varying_blocking_wait)
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(module, {"output": np.zeros(2, dtype=np.int32)})
    assert _stops(caught.value), caught.value.diagnostics


def test_blocking_wait_mixed_lane_readiness_stops_incomplete():
    """Second half of the legacy test (W11 delta copy): legacy raised an error; v2 stops
    as ``incomplete`` with reason ``divergent_block`` (sync-behaviour-deltas M15: the engine cannot run
    the lanes of one warp independently here, so the fault is not provable)."""

    module = v2.transpile(lane_varying_blocking_wait)
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(module, {"output": np.zeros(2, dtype=np.int32)})
    stop = _stops(caught.value)[0]
    assert (stop["status"], stop["kind"]) == ("incomplete", "analysis_incomplete"), stop
    assert stop["reason"].startswith("divergent_block"), stop