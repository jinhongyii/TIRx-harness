"""v2 expected-error copy of
``tests/numsim/runtime/test_layout_lowering_contract.py::test_tmem_tlane_tcol_coordinates_are_observable_through_a_physical_alias``.

Ruling: ``numsim-behaviour-deltas.md`` row **T4** (CONTRACT_REQUESTS.md "Contract
changes for W1" Batch 2 item 17; W2 engine-stops triage "TMEM buffer access outside
the warp's sub-partition"). One warp; lane 0 writes ``logical[1, *]``, which the
``(2, 2) : (64@TLane, 3@TCol)`` layout places at TMEM lane 64, outside warp 0's
sub-partition (lanes 0..31). A buffer-form TMEM store executes as
``tcgen05.st .32x32b``, so v2 stops with ``bad_address``. Legacy modelled TMEM
abstractly and read back ``[11, 12, 21, 22]`` through the physical alias.
Kernel copied verbatim.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.tirx.layout import S, TCol, TileLayout, TLane

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

_TMEM_LANE_COLUMN = TileLayout(S[(2, 2) : (64 @ TLane, 3 @ TCol)])
_TMEM_PHYSICAL = TileLayout(S[(128, 4) : (1 @ TLane, 1 @ TCol)])


@T.prim_func
def tmem_lane_column_physical_alias(output: T.Buffer((4,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    logical = T.decl_buffer(
        (2, 2),
        "uint32",
        scope="tmem",
        layout=_TMEM_LANE_COLUMN,
        allocated_addr=7,
    )
    physical = T.decl_buffer(
        (128, 4),
        "uint32",
        scope="tmem",
        layout=_TMEM_PHYSICAL,
        allocated_addr=7,
    )
    if lane == 0:
        logical[0, 0] = T.uint32(11)
        logical[0, 1] = T.uint32(12)
        logical[1, 0] = T.uint32(21)
        logical[1, 1] = T.uint32(22)
        output[0] = physical[0, 0]
        output[1] = physical[0, 3]
        output[2] = physical[64, 0]
        output[3] = physical[64, 3]



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


def test_tmem_tlane_tcol_coordinates_are_observable_through_a_physical_alias():
    """Replaces ``tests/numsim/runtime/test_layout_lowering_contract.py::test_tmem_tlane_tcol_coordinates_are_observable_through_a_physical_alias``.
    Delta row T4: ``bad_address`` on ``logical[1, 0]`` (TMEM lane 64)."""

    module = v2.transpile(tmem_lane_column_physical_alias)
    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(module, {"output": np.zeros(4, dtype=np.uint32)})
    _assert_tmem_subpartition_bad_address(excinfo, "logical", 0)
    assert "tmem lane 64" in str(_first_stop(excinfo.value).get("message", ""))
