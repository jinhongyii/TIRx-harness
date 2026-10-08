"""v2 port of ``tests/numsim/runtime/test_bitwise_nodes.py::test_bitwise_nodes_match_integer_oracle``.

Dropped pin: the legacy frontend ``source_map`` node-kind set
(``BitwiseAnd`` ... ``RShift``), a legacy IR walk. The integer oracle and the
clean Racecheck/Synccheck runs (legacy ``run_checked``) are kept.
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

DTYPES = [f"{sign}int{bits}" for sign in ("", "u") for bits in (8, 16, 32, 64)]


def bitwise_case(dtype):
    bits = np.dtype(dtype).itemsize * 8
    kernel = tvm.script.from_source(
        f'''@T.prim_func
def bitwise_nodes(values: T.Buffer((32,), "{dtype}"), output: T.Buffer((32, 6), "{dtype}")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    x: T.let = values[lane]
    mask: T.let = T.cast(5, "{dtype}")
    count: T.let = T.cast(lane * {bits - 1} // 31, "{dtype}")
    output[lane, 0] = x & mask
    output[lane, 1] = x | mask
    output[lane, 2] = x ^ mask
    output[lane, 3] = ~x
    output[lane, 4] = x << count
    output[lane, 5] = x >> count
''',
        extra_vars={"T": T},
    )
    info = np.iinfo(dtype)
    values = np.resize(np.array([info.min, info.max, 0, 1, 5, 13], dtype=dtype), 32)
    expected = np.empty((32, 6), dtype=dtype)
    for lane, x in enumerate(values):
        x = int(x)
        count = lane * (bits - 1) // 31
        for column, value in enumerate((x & 5, x | 5, x ^ 5, ~x, x << count, x >> count)):
            value &= (1 << bits) - 1
            if dtype.startswith("int") and value >= 1 << (bits - 1):
                value -= 1 << bits
            expected[lane, column] = value
    return kernel, {"values": values, "output": np.zeros_like(expected)}, expected


@pytest.mark.parametrize("dtype", DTYPES)
def test_bitwise_nodes_match_integer_oracle(dtype):
    """Port of ``tests/numsim/runtime/test_bitwise_nodes.py::test_bitwise_nodes_match_integer_oracle``."""

    kernel, inputs, expected = bitwise_case(dtype)
    v2.synccheck(kernel, {k: v.copy() for k, v in inputs.items()}).require_clean()
    v2.racecheck(kernel, {k: v.copy() for k, v in inputs.items()}).require_clean()
    result = v2.Engine().run(v2.transpile(kernel), inputs, outputs=("output",))
    np.testing.assert_array_equal(result.outputs["output"], expected)
