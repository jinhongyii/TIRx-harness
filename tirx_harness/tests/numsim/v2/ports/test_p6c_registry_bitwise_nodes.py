"""v2 port of ``tests/numsim/registry/test_bitwise_nodes.py``.

The legacy test walked the legacy frontend (``analyze(kernel).unsupported``)
for the message "requires scalar integer or boolean operands, got int32x2".
The port keeps the fail-closed contract through the public surface: the
vector-operand bitwise node is rejected by ``v2.transpile`` with
``UnsupportedTIRxError``. The message text is not pinned.
"""

from __future__ import annotations

import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine


@v2_gap(
    "v2 lowers int32x2 bitwise/shift nodes lane-wise and the run completes "
    "(a stored result is the per-lane value); legacy rejected vector operands "
    "as outside the scalar dtype contract. No delta row rules this yet"
)
@pytest.mark.parametrize(
    "operation",
    ["bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not", "shift_left", "shift_right"],
)
def test_vector_bitwise_nodes_are_rejected(operation):
    """Port of ``tests/numsim/registry/test_bitwise_nodes.py::test_vector_bitwise_nodes_are_rejected``.

    Dropped: the legacy ``analyze(...).unsupported`` message pin.
    """

    arguments = "x" if operation == "bitwise_not" else "x, x"
    kernel = tvm.script.from_source(
        "@T.prim_func\n"
        'def kernel(values: T.Buffer((32,), "int32x2")):\n'
        "    T.device_entry()\n"
        "    lane = T.lane_id([32])\n"
        "    x: T.let = values[lane]\n"
        f"    T.evaluate(T.{operation}({arguments}))\n",
        extra_vars={"T": T},
    )
    with pytest.raises(UnsupportedTIRxError):
        v2.transpile(kernel)
