"""v2 port of ``tests/numsim/registry/test_bitwise_nodes.py``.

The legacy test walked the legacy frontend (``analyze(kernel).unsupported``)
for the message "requires scalar integer or boolean operands, got int32x2".
Delta numsim-behaviour-deltas D11: TVM's CUDA codegen prints vector bitwise
and shift intrinsics lane by lane, so v2 lowers ``bitwise_and/or/xor`` and
``shift_left/right`` on ``int32x2`` lane-wise (asserted on the run's values);
``bitwise_not`` prints ``(~x)``, which CUDA vector types do not support, so it
stays rejected with ``UnsupportedTIRxError``. Message text is not pinned.
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine


_LANE_WISE = {
    "bitwise_and": np.bitwise_and,
    "bitwise_or": np.bitwise_or,
    "bitwise_xor": np.bitwise_xor,
    "shift_left": np.left_shift,
    "shift_right": np.right_shift,
}


def _kernel(operation: str, arguments: str, store: bool):
    body = f"    out[lane] = T.{operation}({arguments})\n" if store else f"    T.evaluate(T.{operation}({arguments}))\n"
    return tvm.script.from_source(
        "@T.prim_func\n"
        'def kernel(values: T.Buffer((32,), "int32x2"), amounts: T.Buffer((32,), "int32x2"), '
        'out: T.Buffer((32,), "int32x2")):\n'
        "    T.device_entry()\n"
        "    lane = T.lane_id([32])\n"
        "    x: T.let = values[lane]\n"
        "    y: T.let = amounts[lane]\n" + body,
        extra_vars={"T": T},
    )


@pytest.mark.parametrize("operation", sorted(_LANE_WISE))
def test_vector_bitwise_nodes_are_lane_wise(operation):
    """Delta D11 of ``tests/numsim/registry/test_bitwise_nodes.py::test_vector_bitwise_nodes_are_rejected``."""

    rng = np.random.default_rng(3)
    values = rng.integers(-(2**31), 2**31, (32, 2), dtype=np.int64).astype(np.int32)
    amounts = rng.integers(0, 32, (32, 2)).astype(np.int32)
    result = v2.Engine().run(
        v2.transpile(_kernel(operation, "x, y", store=True)),
        {"values": values.view(np.int64), "amounts": amounts.view(np.int64), "out": np.zeros(32, np.int64)},
    )
    actual = np.asarray(result.outputs["out"]).view(np.int32).reshape(32, 2)
    np.testing.assert_array_equal(actual, _LANE_WISE[operation](values, amounts))


def test_vector_bitwise_not_is_rejected():
    """Port of ``tests/numsim/registry/test_bitwise_nodes.py::test_vector_bitwise_nodes_are_rejected[bitwise_not]``."""

    with pytest.raises(UnsupportedTIRxError):
        v2.transpile(_kernel("bitwise_not", "x", store=False))
