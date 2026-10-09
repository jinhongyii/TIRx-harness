"""v2 delta copy of ``tests/numsim/runtime/test_memory_coverage_next.py::test_no_complete_rejects_exhausted_arrivals``.

Delta: ``docs/development/sync-behaviour-deltas.md`` row M6 (``.noComplete`` arrive
that would complete the phase): legacy raised an untyped error whose text contained
``noComplete ... pending arrival count``; v2 reports the typed
``NoCompleteWouldComplete`` (synccheck kind ``mbarrier_no_complete_violated``,
racecheck kind ``sync_protocol_error``, NumSim ``ExecutionError``). The rule is
unchanged ("pending > count"), so the same inputs are rejected in every mode.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def no_complete(count: T.uint32, out: T.Buffer((2,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    state = T.alloc_local((1,), "uint64")
    ready = T.alloc_local((1,), "uint32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 3)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        for phase in T.serial(2):
            T.ptx.mbarrier.arrive.noComplete.shared.b64(
                barrier.ptr_to([0]), T.uint32(99), pred=False
            )
            T.ptx.mbarrier.arrive.noComplete.shared.b64(barrier.ptr_to([0]), count)
            T.ptx.mbarrier.arrive.noComplete.shared.b64(state[0], barrier.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.arrive.shared.b64(barrier.ptr_to([0]))
            T.ptx.mbarrier.test_wait.shared.b64(ready[0], barrier.ptr_to([0]), state[0])
            out[phase] = ready[0]


@pytest.mark.parametrize("count", [2, 3])
def test_no_complete_rejects_exhausted_arrivals(count):
    """Delta copy (sync-behaviour-deltas M6): typed ``NoCompleteWouldComplete``."""
    inputs = {"count": count, "out": np.zeros(2, np.uint32)}
    expected_kind = {"synccheck": "mbarrier_no_complete_violated", "racecheck": "sync_protocol_error"}
    for name, checker in (("synccheck", v2.synccheck), ("racecheck", v2.racecheck)):
        report = checker(no_complete, inputs)
        assert report.verdict == "error", report.format()
        assert any(
            f.status == "error"
            and f.kind == expected_kind[name]
            # racecheck: the engine stop's structured ``error`` field;
            # synccheck: the explorer finding names the transition error.
            and (f.details.get("error") == "no_complete_would_complete" or "NoCompleteWouldComplete" in f.message)
            for f in report.findings
        ), report.format()
    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(v2.transpile(no_complete), inputs)
    assert any(d.get("error") == "no_complete_would_complete" for d in excinfo.value.diagnostics), (
        excinfo.value.diagnostics
    )
