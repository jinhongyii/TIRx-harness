"""v2 copy of the single-lane ``.sync.aligned`` participation check.

Replaces ``tests/numsim/integration/test_reported_layout_regressions.py::test_sync_qualified_warp_operations_reject_single_lane_participation``
for the TVM-compilable params. The ``warp-scope-register-op`` param is
dropped: TVM's own TilePrimitiveDispatch rejects its kernel (wave 0, E).
Kernels are copied verbatim.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tirx_harness.numsim import v2
from tests.numsim.v2.checkers._runnable import WARP_COLLECTIVE_DIVERGENCE, requires_v2_engine

pytestmark = requires_v2_engine

LANES = 32

@T.prim_func
def nested_elect_only(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([LANES])
    if T.cuda.elect_sync():
        if T.cuda.elect_sync():
            output[0] = 1


@T.prim_func
def elected_tcgen_wait_ld():
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([LANES])
    if T.cuda.elect_sync():
        T.ptx.tcgen05.wait__ld.sync.aligned()


@T.prim_func
def elected_tcgen_wait_st():
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([LANES])
    if T.cuda.elect_sync():
        T.ptx.tcgen05.wait__st.sync.aligned()


@pytest.mark.parametrize(
    "checker",
    [
        pytest.param(v2.synccheck, id="synccheck"),
        pytest.param(v2.racecheck, id="racecheck"),
    ],
)
@pytest.mark.parametrize(
    ("kernel", "inputs"),
    [
        pytest.param(
            nested_elect_only,
            {"output": np.zeros((1,), dtype=np.int32)},
            id="nested-elect-sync",
        ),
        pytest.param(
            elected_tcgen_wait_ld,
            {},
            id="tcgen-wait-ld-sync",
        ),
        pytest.param(
            elected_tcgen_wait_st,
            {},
            id="tcgen-wait-st-sync",
        ),
    ],
)
def test_sync_qualified_warp_operations_reject_single_lane_participation(checker, kernel, inputs):
    """Replaces ``tests/numsim/integration/test_reported_layout_regressions.py::test_sync_qualified_warp_operations_reject_single_lane_participation``.

    Legacy: verdict ``error`` and the finding kinds are exactly
    ``{"warp_collective_divergence"}``; v2 calls that kind ``divergence``,
    so each finding's kind must be one of the two spellings."""

    report = checker(kernel, inputs)

    assert report.verdict == "error", report.format()
    kinds = {finding.kind for finding in report.findings}
    assert kinds, report.format()
    assert kinds <= WARP_COLLECTIVE_DIVERGENCE, report.format()
