"""v2 port of the legacy divergent default-mask ``__syncwarp`` test; the
legacy message text ("participant mask names an inactive lane") is not
pinned. Kernel copied verbatim."""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def divergent_warp_sync(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 16:
        T.cuda.warp_sync()
    output[lane] = lane


def _stops(error: v2.ExecutionError) -> list[dict]:
    return [d for d in error.diagnostics if d.get("status") in ("error", "incomplete")]


def test_default_full_mask_warp_sync_rejects_divergent_execution():
    """Port of ``tests/numsim/runtime/test_ordering_calls.py::test_default_full_mask_warp_sync_rejects_divergent_execution``.

    Fail-closed half: the run raises ``v2.ExecutionError`` with a stopping
    diagnostic. Dropped pin: the legacy message text. The error-status half
    is :func:`test_default_full_mask_warp_sync_divergence_is_an_error`.
    """

    module = v2.transpile(divergent_warp_sync)
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(module, {"output": np.zeros(32, dtype=np.int32)})
    assert _stops(caught.value), caught.value.diagnostics


def test_default_full_mask_warp_sync_divergence_stops_incomplete():
    """Second half of the legacy test (W11 delta copy): legacy raised an error; v2 stops
    as ``incomplete`` with reason ``divergent_block`` (sync-behaviour-deltas B8: the engine cannot run
    the lanes of one warp independently here, so the fault is not provable)."""

    module = v2.transpile(divergent_warp_sync)
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(module, {"output": np.zeros(32, dtype=np.int32)})
    stop = _stops(caught.value)[0]
    assert (stop["status"], stop["kind"]) == ("incomplete", "analysis_incomplete"), stop
    assert stop["reason"].startswith("divergent_block"), stop