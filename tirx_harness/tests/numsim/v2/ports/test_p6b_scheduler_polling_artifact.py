"""v2 ports of the ``tests/numsim/integration/test_scheduler_polling_artifact.py``
loop-budget tests. Kernels are copied verbatim.

v2 stops an exhausted native loop budget with an ``ExecutionError`` whose
stopping diagnostic is ``incomplete`` ``analysis_incomplete`` (reason
``Budget: ...``), where legacy raised an error: numsim-behaviour-deltas H4
(W11). Each test is split: the fail-closed half (the run raises; a large
enough configured budget completes) and the ``..._stops_incomplete`` half
that asserts the v2 status. The budget is set
through ``v2.Engine(native_loop_iteration_budget=...,
native_loop_reschedule_quantum=...)``, the same keywords as legacy.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

def _first_stop(error: v2.ExecutionError) -> dict:
    """The diagnostic ``Engine.run`` raised for (same selection as run.py)."""

    stops = [
        d
        for d in error.diagnostics
        if d.get("status") in ("error", "incomplete") and d.get("reason") != "subset_execution"
    ]
    assert stops, error.diagnostics
    return stops[0]


@T.prim_func
def time_sliced_finite_for(signal: T.Buffer((1,), "int32"), output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    current = T.local_scalar("int32")
    atomic_old = T.local_scalar("int32")
    if warp == 0:
        if lane == 0:
            for _step in T.serial(128):
                T.ptx.ld.acquire.gpu.global_.s32(current, signal.ptr_to([0]))
                if current == T.int32(0):
                    output[0] = output[0] + T.int32(1)
    elif lane == 0:
        T.ptx.atom.release.gpu.global_.add.s32(atomic_old, signal.ptr_to([0]), T.int32(1))


@T.prim_func
def long_native_while(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    count = T.local_scalar("int32")
    if lane == 0:
        count = T.int32(0)
        while count < T.int32(1_000_001):
            count = count + T.int32(1)
        output[0] = count


def _finite_for_args():
    return {"signal": np.zeros(1, dtype=np.int32), "output": np.zeros(1, dtype=np.int32)}


def _run_finite_for_with_budget_5():
    module = v2.transpile(time_sliced_finite_for)
    return v2.Engine(native_loop_iteration_budget=5, native_loop_reschedule_quantum=64).run(
        module, _finite_for_args()
    )


def test_finite_for_uses_native_loop_iteration_budget():
    """Port of ``tests/numsim/integration/test_scheduler_polling_artifact.py::test_finite_for_uses_native_loop_iteration_budget``
    (fail-closed half).

    Dropped: ``match="configured native loop iteration budget 5"`` (legacy
    text). Asserted: the 128-trip ``for`` with budget 5 raises
    ``v2.ExecutionError``. Error-status half: ``..._reports_error_status``.
    """

    with pytest.raises(v2.ExecutionError):
        _run_finite_for_with_budget_5()


def test_finite_for_uses_native_loop_iteration_budget_stops_incomplete():
    """Second half of ``tests/numsim/integration/test_scheduler_polling_artifact.py::test_finite_for_uses_native_loop_iteration_budget``
    (W11 delta copy). Legacy raised an error naming "configured native loop iteration
    budget 5"; v2 stops the run as ``incomplete`` (numsim-behaviour-deltas H4: an
    exhausted budget is a coverage limit, not a proven kernel fault)."""

    with pytest.raises(v2.ExecutionError) as caught:
        _run_finite_for_with_budget_5()
    stop = _first_stop(caught.value)
    assert (stop["status"], stop["kind"]) == ("incomplete", "analysis_incomplete"), stop
    assert stop["reason"].startswith("Budget"), stop


def _run_long_while_with_budget_5():
    module = v2.transpile(long_native_while)
    return v2.Engine(native_loop_iteration_budget=5, native_loop_reschedule_quantum=2_000_000).run(
        module, {"output": np.zeros(1, dtype=np.int32)}
    )


def test_finite_loop_can_exceed_default_budget_when_configured():
    """Port of ``tests/numsim/integration/test_scheduler_polling_artifact.py::test_finite_loop_can_exceed_default_budget_when_configured``
    (all but the error-status assertion).

    Dropped: ``match="configured native loop iteration budget 5"`` and
    ``stats["poll_order"] == [0]``. Asserted: budget 5 raises
    ``v2.ExecutionError``; budget 1_000_001 completes with the full count.
    (The v2 default budget is already above 1_000_001, so "exceed the
    default" is legacy-specific; the configured budgets are kept as is.)
    """

    with pytest.raises(v2.ExecutionError):
        _run_long_while_with_budget_5()

    module = v2.transpile(long_native_while)
    result = v2.Engine(
        native_loop_iteration_budget=1_000_001, native_loop_reschedule_quantum=2_000_000
    ).run(module, {"output": np.zeros(1, dtype=np.int32)})

    assert result.status.get("kind") == "completed", result.status
    np.testing.assert_array_equal(result.outputs["output"], np.asarray([1_000_001], dtype=np.int32))


def test_finite_loop_can_exceed_default_budget_when_configured_stops_incomplete():
    """Second half of ``tests/numsim/integration/test_scheduler_polling_artifact.py::test_finite_loop_can_exceed_default_budget_when_configured``
    (W11 delta copy). Legacy raised an error naming "configured native loop iteration
    budget 5"; v2 stops the run as ``incomplete`` (numsim-behaviour-deltas H4: an
    exhausted budget is a coverage limit, not a proven kernel fault)."""

    with pytest.raises(v2.ExecutionError) as caught:
        _run_long_while_with_budget_5()
    stop = _first_stop(caught.value)
    assert (stop["status"], stop["kind"]) == ("incomplete", "analysis_incomplete"), stop
    assert stop["reason"].startswith("Budget"), stop