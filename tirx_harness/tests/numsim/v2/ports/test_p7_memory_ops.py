"""v2 port of ``tests/numsim/runtime/test_memory_ops.py::test_scalar_atomic_add_has_no_s64_form_while_min_and_max_do``.

PTX ISA 9.7.14.5 Table 35 lists ``.s64`` under ``.min, .max`` only, so
``atom.add.s64`` / ``red.add.s64`` are not PTX forms (TVM's wrapper rejects
them) while the ``min`` form is the positive control that the s64 carrier
resolves. Legacy pinned the emitted ``v2::mem::variant::Minimum`` /
``v2::reg::variant::I64`` generics (``emitted_calls``); the copy runs the
``min.s64`` kernel instead and checks a signed 64-bit minimum lands in
``counter``. Kernel copied verbatim.
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm import tirx
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@pytest.mark.parametrize("op_name", ["atom", "red"])
def test_scalar_atomic_add_has_no_s64_form_while_min_and_max_do(op_name):
    """Port of ``tests/numsim/runtime/test_memory_ops.py::test_scalar_atomic_add_has_no_s64_form_while_min_and_max_do``.

    Dropped: the emitted-generics pins; added: the ``min.s64`` result for a
    positive and a negative initial value (signed comparison)."""

    pointer = tirx.Var("pointer", "handle")
    wrapper = getattr(T.ptx, op_name)
    destination = tirx.decl_buffer((1,), "int64", name="destination")

    def call(operation):
        operands = (
            (destination[0], pointer, T.int64(1))
            if op_name == "atom"
            else (
                pointer,
                T.int64(1),
            )
        )
        return operation(*operands)

    with pytest.raises(ValueError, match=r"\.add requires"):
        call(wrapper.global_.add.s64)

    destination_operand = "destination, " if op_name == "atom" else ""
    kernel = tvm.script.from_source(
        f"""
@T.prim_func
def kernel(counter: T.Buffer((1,), "int64")):
    T.device_entry()
    destination = T.local_scalar("int64")
    T.ptx.{op_name}.global_.min.s64({destination_operand}T.address_of(counter[0]), T.int64(1))
""",
        {"T": T},
    )
    module = v2.transpile(kernel)
    for initial, expected in ((5, 1), (-3, -3)):
        result = v2.Engine().run(
            module, {"counter": np.array([initial], dtype=np.int64)}, outputs=("counter",)
        )
        assert result.status["kind"] == "completed", result.status
        np.testing.assert_array_equal(result.outputs["counter"], np.array([expected], dtype=np.int64))
