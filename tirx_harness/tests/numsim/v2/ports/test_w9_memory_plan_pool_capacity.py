"""v2 copies of the two fail-closed ``tirx.pool_max_bytes`` tests in
``tests/numsim/integration/test_memory_plan.py``:
``test_conflicting_pool_capacity_attrs_fail_closed`` and
``test_malformed_pool_capacity_attr_fails_closed``.

Legacy rejected both kernels in its transpiler (``emitted_module``). The
kernels are copied verbatim. v2 lowering rejects both with the legacy
messages (``tirx.pool_max_bytes`` lowering, W1).
"""

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def conflicting_pool_capacities(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    storage = T.alloc_buffer((0,), "uint8", scope="shared")
    T.attr(storage.data, "tirx.pool_max_bytes", 16)
    T.attr(storage.data, "tirx.pool_max_bytes", 32)
    alias = T.decl_buffer((4,), "uint32", data=storage.data, scope="shared")
    output[0] = alias[0]


@T.prim_func
def negative_pool_capacity(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    storage = T.alloc_buffer((0,), "uint8", scope="shared")
    T.attr(storage.data, "tirx.pool_max_bytes", -1)
    output[0] = 0


def _run(kernel):
    return v2.Engine().run(v2.transpile(kernel), {"output": np.zeros(1, np.uint32)})




def test_conflicting_pool_capacity_attrs_fail_closed():
    with pytest.raises(Exception, match="conflicting capacities 16 and 32"):
        _run(conflicting_pool_capacities)


def test_malformed_pool_capacity_attr_fails_closed():
    with pytest.raises(Exception, match="cannot be negative"):
        _run(negative_pool_capacity)
