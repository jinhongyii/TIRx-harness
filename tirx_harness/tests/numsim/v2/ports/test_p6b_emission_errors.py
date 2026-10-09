"""v2 ports of the ``analyze``/``verify`` functions in
``tests/numsim/integration/test_emission_errors.py``.

Legacy rejected f64 ``exp``/``log`` and the ``Let`` expression node and
asserted how analysis collected those failures (``op#N:Call(...)`` entries
with source spans). v2 implements f64 math (delta D3,
numsim-behaviour-deltas.md) and lowers ``Let``, so the rejected kernels now
transpile and compute; the failure-collection bookkeeping has nothing to
collect and is dropped. Kernels are copied verbatim.
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


def _kernel(dtype):
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel(output: T.Buffer((1,), "float32")):
    T.device_entry()
    value: T.let = T.exp(T.{dtype}(1))
    output[0] = T.cast(value, "float32")
    T.evaluate(T.log(T.{dtype}(1)))
""",
        {"T": T},
    )


def _device_kernel(body, params=()):
    """Copy of ``tests/numsim/support/manifest.py::device_kernel``."""

    entry = tirx.AttrStmt(0, "tirx.device_entry", tirx.IntImm("bool", 1), body)
    return tirx.PrimFunc(list(params), entry)


def _run(kernel, output):
    module = v2.transpile(kernel)
    assert module.document["kernels"][0]["unsupported"] == []
    result = v2.Engine().run(module, {"output": output})
    assert result.status["kind"] == "completed", result.status
    return result.outputs["output"]


def test_analysis_collects_independent_failures_after_a_failed_binding():
    """Port of ``tests/numsim/integration/test_emission_errors.py::test_analysis_collects_independent_failures_after_a_failed_binding``.

    Delta D3: f64 ``exp``/``log`` are implemented (host libm), so the float64
    kernel is accepted and computes ``exp(1)``. Dropped: the two collected
    ``op#N:Call(tirx.exp|log(...))`` entries, their source spans and the
    ``verify`` rejection (nothing is unsupported any more)."""

    output = _run(_kernel("float64"), np.zeros(1, dtype=np.float32))
    np.testing.assert_allclose(output, np.float32(np.exp(np.float64(1))), rtol=1e-6)


def test_valid_signature_control_emits_and_executes():
    """Port of ``tests/numsim/integration/test_emission_errors.py::test_valid_signature_control_emits_and_executes``."""

    output = _run(_kernel("float32"), np.zeros(1, dtype=np.float32))
    np.testing.assert_allclose(output, np.exp(np.float32(1)), rtol=1e-6)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_error_recovery_respects_expression_let_scope(dtype):
    """Port of ``tests/numsim/integration/test_emission_errors.py::test_error_recovery_respects_expression_let_scope``.

    Legacy rejected the ``Let`` expression node (and, for float64, f64
    ``exp``/``log``). v2 lowers ``Let`` with its binding scope and implements
    f64 math (delta D3), so both dtypes compute ``exp(exp(1) + log(1))``. No
    delta row covers ``Let`` acceptance; the legacy rejection was a frontend
    capability limit ("outside the public supported-node set"), not a
    semantic requirement. The float32 supported-expression control is kept."""

    output = tvm.tirx.decl_buffer((1,), "float32", name="output")
    variable = tvm.tirx.Var("scoped", "float32")
    bound = tvm.tirx.Let(variable, T.float32(1), T.exp(variable))
    expression = T.exp(T.cast(bound, dtype) + T.log(tvm.tirx.FloatImm(dtype, 1)))
    kernel = _device_kernel(
        tvm.tirx.BufferStore(output, T.cast(expression, "float32"), [0]), (output,)
    )
    expected = np.exp(np.exp(np.float64(1)))
    np.testing.assert_allclose(_run(kernel, np.zeros(1, dtype=np.float32)), expected, rtol=1e-6)
    if dtype == "float32":
        # The equivalent supported expression is the successful control.
        expression = T.exp(T.exp(T.float32(1)) + T.log(T.float32(1)))
        kernel = _device_kernel(tvm.tirx.BufferStore(output, expression, [0]), (output,))
        np.testing.assert_allclose(
            _run(kernel, np.zeros(1, dtype=np.float32)), np.exp(np.exp(np.float32(1))), rtol=1e-6
        )
