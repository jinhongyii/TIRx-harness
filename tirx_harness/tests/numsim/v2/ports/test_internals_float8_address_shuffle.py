"""v2 ports of the legacy float8-address and four-argument ``__shfl_sync``
tests; the ``module.rust_source`` text pins are dropped."""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def float8_address(source: T.Buffer((32,), "float8_e4m3fn")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.evaluate(T.address_of(source[lane]))


@T.prim_func
def cuda_shfl_sync_u32(source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.cuda.__shfl_sync(
        T.uint32(0xFFFFFFFF), source[lane], T.cast(31 - lane, "uint32"), 32
    )


def test_float8_address_uses_one_byte_physical_pointer():
    """Port of ``tests/numsim/runtime/test_float8_address_shuffle.py::
    test_float8_address_uses_one_byte_physical_pointer``.

    Dropped pins: ``"PhysicalPtr::new" in module.rust_source`` and
    ``"tirx.address_of" not in module.rust_source``.
    """

    source = np.zeros(32, dtype=np.uint8)

    module = v2.transpile(float8_address)
    result = v2.Engine().run(module, {"source": source})

    assert set(result.outputs) == {"source"}
    np.testing.assert_array_equal(result.outputs["source"], 0)


def test_cuda_four_argument_shfl_sync_uses_implicit_warp_size():
    """Port of ``tests/numsim/runtime/test_float8_address_shuffle.py::
    test_cuda_four_argument_shfl_sync_uses_implicit_warp_size``.

    Dropped pins: ``"fn warp_shuffle(" not in module.rust_source`` and
    ``"tirx.cuda.__shfl_sync" not in module.rust_source``.
    """

    source = np.arange(32, dtype=np.uint32) * np.uint32(7) + np.uint32(3)
    output = np.zeros(32, dtype=np.uint32)

    module = v2.transpile(cuda_shfl_sync_u32)
    result = v2.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source[::-1])
