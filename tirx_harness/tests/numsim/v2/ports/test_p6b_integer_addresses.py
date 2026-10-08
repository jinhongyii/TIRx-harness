"""v2 port of ``tests/numsim/integration/test_integer_addresses.py`` tests that
pinned legacy error text. The kernel is copied verbatim; the exception type and
the v2 stopping diagnostic's status and kind are asserted, never the wording.
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
def integer_address_mixed_spaces_active_typed_load(
    source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint32", scope="shared")
    slot = T.alloc_buffer((1,), "uint64", scope="local")
    if lane % 2 == 0:
        slot[0] = T.reinterpret("uint64", source.ptr_to([lane]))
    else:
        slot[0] = T.reinterpret("uint64", shared.ptr_to([lane]))
    T.ptx.ld.global_.u32(output[lane], T.reinterpret("handle", slot[0]))


def test_typed_pointer_load_rejects_active_lane_from_another_space():
    """Port of ``tests/numsim/integration/test_integer_addresses.py::test_typed_pointer_load_rejects_active_lane_from_another_space``.

    Dropped: ``match="resolved address.*does not match PTX state space
    global"`` (legacy text). Asserted: ``ld.global`` through a lane-varying
    integer address whose odd lanes hold a shared address fails closed with
    ``v2.ExecutionError`` whose stopping diagnostic is an ``error`` of kind
    ``bad_address`` (the shared address is not a mapped global address).
    """

    module = v2.transpile(integer_address_mixed_spaces_active_typed_load)
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(
            module,
            {
                "source": np.arange(32, dtype=np.uint32),
                "output": np.zeros(32, dtype=np.uint32),
            },
        )
    stop = _first_stop(caught.value)
    assert stop["status"] == "error", stop
    assert stop["kind"] == "bad_address", stop
