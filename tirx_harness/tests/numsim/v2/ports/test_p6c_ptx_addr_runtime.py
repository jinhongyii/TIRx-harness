"""v2 port of the legacy ``T.ptx.addr`` runtime test; legacy
``analyze(kernel).unsupported == ()`` becomes "``v2.transpile`` lowers the
kernel". Kernel copied verbatim from
``tests/numsim/runtime/test_ptx_addr_runtime.py``."""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def ptx_addr_global_load(source: T.Buffer((33,), "int32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.evaluate(T.ptx.ld.global_.s32(output[lane], T.ptx.addr(source.ptr_to([lane]), 4)))


def test_ptx_addr_applies_a_signed_byte_offset():
    """Port of ``tests/numsim/runtime/test_ptx_addr_runtime.py::test_ptx_addr_applies_a_signed_byte_offset``."""

    source = np.arange(33, dtype=np.int32) * 7 - 11
    output = np.zeros(32, dtype=np.int32)
    module = v2.transpile(ptx_addr_global_load)
    result = v2.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source[1:])
