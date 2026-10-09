"""v2 expected-error copy of
``tests/numsim/integration/test_tmem_artifact.py::test_tmem_layout_f_maps_rows_to_half_slabs``.

Ruling: ``numsim-behaviour-deltas.md`` row **T4** (CONTRACT_REQUESTS.md "Contract
changes for W1" Batch 2 item 17; W2 engine-stops triage "TMEM buffer access outside
the warp's sub-partition"). The kernel (copied from ``tests/numsim/support/kernels.py``)
runs 2 warps; row ``r = warp*32 + lane`` of the layout-F view sits at TMEM lane
``(r // 16) * 32 + r % 16``, so warp 0 lane 16 writes TMEM lane 32. A buffer-form
TMEM store executes as ``tcgen05.st .32x32b`` and may address only the warp's own
sub-partition, so v2 stops with ``bad_address``. Legacy modelled TMEM buffers
abstractly and returned ``2000 + row``.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.tirx.layout import tmem_datapath_layout

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def tmem_f_lane_mapping(output: T.Buffer((64,), "uint32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    f_view = T.decl_buffer(
        (64, 4), "uint32", scope="tmem", layout=tmem_datapath_layout("F", 64, 4), allocated_addr=0
    )
    d_alias = T.decl_buffer(
        (128, 4), "uint32", scope="tmem", layout=tmem_datapath_layout("D", 128, 4), allocated_addr=0
    )
    physical_lane = T.meta_var((row // 16) * 32 + row % 16)
    f_view[row, 1] = T.cast(2000 + row, "uint32")
    output[row] = d_alias[physical_lane, 1]


def _first_stop(error: v2.ExecutionError) -> dict:
    stops = [
        d
        for d in error.diagnostics
        if d.get("status") in ("error", "incomplete") and d.get("reason") != "subset_execution"
    ]
    assert stops, error.diagnostics
    return stops[0]


def _assert_tmem_subpartition_bad_address(excinfo, buffer: str, warp: int) -> None:
    """Row T4: the stopping diagnostic is an ``error`` of kind ``bad_address`` for a
    TMEM lane outside the issuing warp's 32-lane sub-partition."""

    stop = _first_stop(excinfo.value)
    assert stop["status"] == "error", stop
    assert stop["kind"] == "bad_address", stop
    message = str(stop.get("message", ""))
    assert f"{buffer}[" in message, stop
    assert f"outside warp {warp}'s sub-partition" in message, stop


def test_tmem_layout_f_maps_rows_to_half_slabs():
    """Replaces ``tests/numsim/integration/test_tmem_artifact.py::test_tmem_layout_f_maps_rows_to_half_slabs``.
    Delta row T4: ``bad_address`` on the layout-F store."""

    module = v2.transpile(tmem_f_lane_mapping)
    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(module, {"output": np.zeros(64, dtype=np.uint32)})
    stop = _first_stop(excinfo.value)
    warp = 0 if "warp 0's" in str(stop.get("message", "")) else 1
    _assert_tmem_subpartition_bad_address(excinfo, "f_view", warp)
