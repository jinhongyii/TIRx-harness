"""v2 port of the legacy large register-buffer artifact test; the
``module.rust_source`` text pins are dropped."""

from __future__ import annotations

import numpy as np
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


_REGISTER_BUFFER_COUNT = 5_600


def _many_register_buffer_kernel(count: int):
    lines = [
        "@T.prim_func",
        'def main(output: T.Buffer((1,), "int32")):',
        "    T.device_entry()",
        "    _warp = T.warp_id([1])",
        "    lane = T.lane_id([32])",
    ]
    lines.extend(f'    value_{index} = T.alloc_local((1,), "int32")' for index in range(count))
    # Unused register declarations remain part of the physical memory plan.
    lines.extend(
        (
            "    value_0[0] = lane + 17",
            "    if lane == 0:",
            "        output[0] = value_0[0]",
        )
    )
    return tvm.script.from_source("\n".join(lines), {"T": T})


def test_large_register_buffer_table_is_heap_backed_and_executes():
    """Port of ``tests/numsim/integration/test_large_buffer_artifact.py::
    test_large_register_buffer_table_is_heap_backed_and_executes``.

    Dropped pins: the four ``module.rust_source`` substring assertions (legacy
    generated-Rust buffer table layout). Legacy ``Engine(max_workers=1)`` has
    no v2 equivalent; the default engine is used.
    """

    module = v2.transpile(_many_register_buffer_kernel(_REGISTER_BUFFER_COUNT))

    result = v2.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.array([17], dtype=np.int32))
