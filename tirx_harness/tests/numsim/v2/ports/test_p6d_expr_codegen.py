"""v2 delta copy of ``tests/numsim/runtime/test_expr_codegen.py::test_low_precision_arithmetic_rounds_at_every_expression_node``.

Delta: ``docs/development/numsim-behaviour-deltas.md`` row D1 (Confirmed,
coordinator ruling 2026-10-07): a scalar f16/bf16 TIR expression tree is
evaluated in f32 and rounded once, where the value leaves the chain (here the
``T.cast(..., "float32")`` of the outer sum). D1 names exactly this kernel's
case: ``(a+b)+c`` with ``a=1``, ``b=c=2**-11`` in f16 gives ``0x3c01``
(1.0009765625); per-node rounding (what this legacy test pinned) gives
``0x3c00``. bf16 with ``b=c=2**-8`` likewise gives ``1 + 2**-7`` (1.0078125).

Kept: kernel and inputs; dropped: nothing else (the legacy test had no pins).
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def low_precision_expression_chain(
    fp16_a: T.Buffer((32,), "float16"),
    fp16_b: T.Buffer((32,), "float16"),
    fp16_c: T.Buffer((32,), "float16"),
    bf16_a: T.Buffer((32,), "bfloat16"),
    bf16_b: T.Buffer((32,), "bfloat16"),
    bf16_c: T.Buffer((32,), "bfloat16"),
    fp16_output: T.Buffer((32,), "float32"),
    bf16_output: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    fp16_output[lane] = T.cast((fp16_a[lane] + fp16_b[lane]) + fp16_c[lane], "float32")
    bf16_output[lane] = T.cast((bf16_a[lane] + bf16_b[lane]) + bf16_c[lane], "float32")


def test_low_precision_arithmetic_rounds_at_every_expression_node():
    """Delta copy (row D1); see the module docstring."""

    bf16 = lambda bits: np.full(32, np.uint16(bits), dtype=np.uint16).view(ml_dtypes.bfloat16)  # noqa: E731
    result = v2.Engine().run(
        v2.transpile(low_precision_expression_chain),
        {
            "fp16_a": np.ones(32, dtype=np.float16),
            "fp16_b": np.full(32, 2**-11, dtype=np.float16),
            "fp16_c": np.full(32, 2**-11, dtype=np.float16),
            "bf16_a": bf16(0x3F80),
            "bf16_b": bf16(0x3B80),
            "bf16_c": bf16(0x3B80),
            "fp16_output": np.zeros(32, dtype=np.float32),
            "bf16_output": np.zeros(32, dtype=np.float32),
        },
    )
    np.testing.assert_array_equal(result.outputs["fp16_output"], np.full(32, 1 + 2**-10, dtype=np.float32))
    np.testing.assert_array_equal(result.outputs["bf16_output"], np.full(32, 1 + 2**-7, dtype=np.float32))
