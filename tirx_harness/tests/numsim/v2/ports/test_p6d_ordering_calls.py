"""v2 expected-error copies of the three ``setmaxnreg`` functions of
``tests/numsim/runtime/test_ordering_calls.py``:
``test_setmaxnreg_is_an_ordering_call_not_a_tcgen_lifecycle_call``,
``test_deleting_setmaxnreg_keeps_the_numerical_result`` and
``test_setmaxnreg_static_expressions_follow_public_parser_and_runtime``.

Ruling: ``sync-behaviour-deltas.md`` setmaxnreg section, row **R1** (NumSim runs the
checker-mode register pool, so it can report ``InvalidDirection``); W2 engine-stops
triage in CONTRACT_REQUESTS.md ("``setmaxnreg`` direction (3 functions)"). The
kernels issue ``setmaxnreg.dec`` to a count above the current one (256 after
``inc 24``; 64 after ``inc 32``). PTX leaves that undefined; v2 stops with a
``sync_protocol_error`` execution error carrying
``RegPool(InvalidDirection { inc: false, current, count })``. Legacy treated
``setmaxnreg`` as ordering-only and returned the output.

Dropped pin: ``emitted_calls(...) == ["v2::control::setmaxnreg"] * 2`` (legacy Rust
emission). Kept: the ``setmaxnreg``-free twin still produces ``7``; the static
expressions are still folded by the public parser (the error names
``current: 32, count: 64``). Kernels copied verbatim.
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def setmaxnreg_ordering_only(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    T.ptx.setmaxnreg.inc.sync.aligned.u32(24)
    T.cuda.warpgroup_sync(7)
    T.ptx.setmaxnreg.dec.sync.aligned.u32(256)
    if (warp == 0) and (lane == 0):
        output[0] = 7


@T.prim_func
def setmaxnreg_deleted(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    T.cuda.warpgroup_sync(7)
    if (warp == 0) and (lane == 0):
        output[0] = 7


def _setmaxnreg_static_expression_kernel():
    return tvm.script.from_source(
        """
@T.prim_func
def setmaxnreg_static_expressions(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    T.ptx.setmaxnreg.inc.sync.aligned.u32(T.int32(24) + T.int32(8))
    T.cuda.warpgroup_sync(7)
    T.ptx.setmaxnreg.dec.sync.aligned.u32(T.int32(72) - T.int32(8))
    if (warp == 0) and (lane == 0):
        output[0] = 1
""",
        extra_vars={"T": T},
    )


def _first_stop(error: v2.ExecutionError) -> dict:
    stops = [
        d
        for d in error.diagnostics
        if d.get("status") in ("error", "incomplete") and d.get("reason") != "subset_execution"
    ]
    assert stops, error.diagnostics
    return stops[0]


def _assert_invalid_direction(kernel, current: int, count: int) -> None:
    """Row R1: ``sync_protocol_error`` with ``RegPool(InvalidDirection ..)``."""

    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(v2.transpile(kernel), {"output": np.zeros(1, dtype=np.int32)})
    stop = _first_stop(excinfo.value)
    assert stop["status"] == "error", stop
    assert stop["kind"] == "sync_protocol_error", stop
    assert (stop.get("protocol"), stop.get("error")) == ("reg_pool", "invalid_direction"), stop
    assert (stop.get("inc"), stop.get("current"), stop.get("count")) == (False, current, count), stop


def test_setmaxnreg_is_an_ordering_call_not_a_tcgen_lifecycle_call():
    """Replaces ``tests/numsim/runtime/test_ordering_calls.py::test_setmaxnreg_is_an_ordering_call_not_a_tcgen_lifecycle_call``.
    Delta row R1: ``dec 256`` after ``inc 24`` is ``InvalidDirection``."""

    _assert_invalid_direction(setmaxnreg_ordering_only, 24, 256)


def test_setmaxnreg_static_expressions_follow_public_parser_and_runtime():
    """Replaces ``tests/numsim/runtime/test_ordering_calls.py::test_setmaxnreg_static_expressions_follow_public_parser_and_runtime``.
    The parser folds ``24 + 8`` / ``72 - 8``; delta row R1: ``dec 64`` after
    ``inc 32`` is ``InvalidDirection``."""

    _assert_invalid_direction(_setmaxnreg_static_expression_kernel(), 32, 64)


def test_deleting_setmaxnreg_keeps_the_numerical_result():
    """Replaces ``tests/numsim/runtime/test_ordering_calls.py::test_deleting_setmaxnreg_keeps_the_numerical_result``.
    The original kernel now fails (delta row R1); the ``setmaxnreg``-free twin
    still yields the legacy value ``7``."""

    _assert_invalid_direction(setmaxnreg_ordering_only, 24, 256)
    result = v2.Engine().run(v2.transpile(setmaxnreg_deleted), {"output": np.zeros(1, dtype=np.int32)})
    np.testing.assert_array_equal(result.outputs["output"], np.array([7], dtype=np.int32))
