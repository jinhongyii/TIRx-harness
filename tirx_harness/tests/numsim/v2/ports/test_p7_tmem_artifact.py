"""v2 port of ``tests/numsim/integration/test_tmem_artifact.py::test_tmem_live_allocation_at_kernel_exit_is_rejected``.

The kernel is copied verbatim. Legacy matched the message "kernel exited with
live TMEM allocations"; the copy asserts the exception type and the stopping
diagnostic's kind (``sync_protocol_error``, tcgen protocol error
``live_allocations_at_exit``), not the wording.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import NumSimExecutionError

pytestmark = requires_v2_engine


@T.prim_func
def dynamic_tmem_leak(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    if lane == 0:
        output[0] = address[0]


def test_tmem_live_allocation_at_kernel_exit_is_rejected():
    """Port of ``tests/numsim/integration/test_tmem_artifact.py::test_tmem_live_allocation_at_kernel_exit_is_rejected``."""

    module = v2.transpile(dynamic_tmem_leak)
    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(module, {"output": np.zeros(1, dtype=np.uint32)})
    error = excinfo.value
    assert isinstance(error, NumSimExecutionError)
    stops = [d for d in error.diagnostics if d.get("status") in ("error", "incomplete")]
    assert stops and stops[0]["status"] == "error", error.diagnostics
    assert stops[0]["kind"] == "sync_protocol_error", stops[0]
    assert stops[0].get("protocol") == "tcgen", stops[0]
    assert stops[0].get("error") == "live_allocations_at_exit", stops[0]
