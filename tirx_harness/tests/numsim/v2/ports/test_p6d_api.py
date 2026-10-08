"""v2 copy of ``tests/numsim/integration/test_api.py::test_engine_native_loop_policy_is_explicit_and_validated``.

Triage (W9 phase 6, internal other-assertion): **bug** [W8], filed in
``CONTRACT_REQUESTS.md`` "W9-public-API phase 6 (2026-10-08): internal
other-assertion triage".

v2 ``Engine`` accepts both knobs (``native_loop_iteration_budget``,
``native_loop_reschedule_quantum``) but stores them unvalidated as
``loop_budget`` / ``quantum`` (default ``None`` = the core default). Observed:
``budget=0`` and ``budget=True`` run to completion, ``quantum=0`` makes
``Engine.run`` hang, ``quantum=1.5`` raises ``TypeError`` only at run time,
``budget=-1`` raises ``OverflowError`` at run time.

Dropped pin: the legacy attribute names/defaults
(``native_loop_iteration_budget == 1_000_000``,
``native_loop_reschedule_quantum == 64``); v2 keeps the knobs as constructor
arguments only. Kept: positive values configure a run that completes
(``test_engine_native_loop_knobs_accept_positive_values``) and the
constructor-time validation (``v2_gap`` until W8 adds it; the copy never runs a
zero-quantum engine).
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def _lane_increment(values: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    values[lane] = values[lane] + T.float32(1)


def test_engine_native_loop_knobs_accept_positive_values():
    """Part of the legacy test that holds: explicit positive knobs configure a
    completed run."""

    module = v2.transpile(_lane_increment)
    for engine in (
        v2.Engine(),
        v2.Engine(native_loop_iteration_budget=2_000_000, native_loop_reschedule_quantum=17),
    ):
        result = engine.run(module, {"values": np.zeros(32, dtype=np.float32)})
        assert result.status["kind"] == "completed"
        np.testing.assert_array_equal(result.outputs["values"], np.ones(32, dtype=np.float32))


@v2_gap(
    "[W8] Engine does not validate native_loop_iteration_budget / native_loop_reschedule_quantum: "
    "0 and True are accepted (budget=0 completes, quantum=0 hangs Engine.run), 1.5 fails only at run "
    "time with TypeError"
)
def test_engine_native_loop_policy_is_explicit_and_validated():
    """Replaces the legacy validation half: invalid knobs are rejected at
    construction."""

    with pytest.raises(ValueError, match="native_loop_iteration_budget"):
        v2.Engine(native_loop_iteration_budget=0)
    with pytest.raises(TypeError, match="native_loop_iteration_budget"):
        v2.Engine(native_loop_iteration_budget=True)
    with pytest.raises(ValueError, match="native_loop_reschedule_quantum"):
        v2.Engine(native_loop_reschedule_quantum=0)
    with pytest.raises(TypeError, match="native_loop_reschedule_quantum"):
        v2.Engine(native_loop_reschedule_quantum=1.5)
