"""v2 copies of the TMEM-lease rejection tests in
``tests/numsim/integration/test_tmem_artifact.py`` (W1 public-API triage, 001a09f).

v2 rejects the same accesses but words them "tmem column N is not in a live
tcgen05 allocation" (legacy: "not covered by any live allocation"). The copies
assert the exception type and the stopping diagnostic's kind (``bad_address``,
status ``error``), not the text. Kernels copied verbatim (the first from
``tests/numsim/support/kernels.py``, pure TVM).
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.tirx.layout import S, TCol, TileLayout, TLane

from tests.numsim.support.kernels import tmem_dynamic_allocated_addr
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

_DYNAMIC_TMEM_LAYOUT = TileLayout(S[(128, 64) : (1 @ TLane, 1 @ TCol)])

@T.prim_func
def dynamic_tmem_use_before_alloc(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    tmem = T.decl_buffer(
        (128, 64), "uint32", scope="tmem", layout=_DYNAMIC_TMEM_LAYOUT, allocated_addr=address[0]
    )
    if lane == 0:
        address[0] = T.uint32(0)
    T.cuda.warp_sync()
    tmem[lane, 0] = T.cast(lane, "uint32")
    output[lane] = tmem[lane, 0]


@T.prim_func
def dynamic_tmem_outside_live_lease(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    tmem = T.decl_buffer(
        (128, 64), "uint32", scope="tmem", layout=_DYNAMIC_TMEM_LAYOUT, allocated_addr=address[0]
    )
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    T.cuda.warp_sync()
    tmem[lane, 32] = T.cast(lane, "uint32")
    output[lane] = T.uint32(1)


@T.prim_func
def dynamic_tmem_use_after_dealloc(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    tmem = T.decl_buffer(
        (128, 64), "uint32", scope="tmem", layout=_DYNAMIC_TMEM_LAYOUT, allocated_addr=address[0]
    )
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    T.cuda.warp_sync()
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
    tmem[lane, 0] = T.cast(lane, "uint32")
    output[lane] = T.uint32(1)




def _assert_bad_address(excinfo) -> None:
    error = excinfo.value
    assert isinstance(error, v2.ExecutionError), type(error)
    stops = [d for d in error.diagnostics if d.get("status") in ("error", "incomplete")]
    assert stops and stops[0]["status"] == "error", error.diagnostics
    assert stops[0]["kind"] == "bad_address", stops[0]
    # W11 (pin-message): the legacy text named the TMEM access; the diagnostic is
    # anchored at the statement that touches the out-of-lease TMEM view.
    span = stops[0]["source_span"]
    with open(span["source_name"]) as handle:
        line = handle.read().splitlines()[span["line"] - 1]
    assert "tmem[" in line, (line, stops[0])


def test_tmem_runtime_address_without_a_dynamic_lease_is_rejected():
    """Port of ``tests/numsim/integration/test_tmem_artifact.py::test_tmem_runtime_address_without_a_dynamic_lease_is_rejected``."""

    module = v2.transpile(tmem_dynamic_allocated_addr)
    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(module, {"output": np.zeros(32, dtype=np.uint32)})
    _assert_bad_address(excinfo)


@pytest.mark.parametrize(
    "kernel",
    [dynamic_tmem_use_before_alloc, dynamic_tmem_outside_live_lease, dynamic_tmem_use_after_dealloc],
    ids=["use_before_alloc", "outside_live_lease", "use_after_dealloc"],
)
def test_tmem_dynamic_lease_rejects_invalid_lifetimes_and_ranges(kernel):
    """Port of ``tests/numsim/integration/test_tmem_artifact.py::test_tmem_dynamic_lease_rejects_invalid_lifetimes_and_ranges``."""

    module = v2.transpile(kernel)
    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(module, {"output": np.zeros(32, dtype=np.uint32)})
    _assert_bad_address(excinfo)
