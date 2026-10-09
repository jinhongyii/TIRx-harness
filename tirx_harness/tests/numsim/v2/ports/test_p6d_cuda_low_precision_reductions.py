"""v2 delta copy of ``tests/numsim/runtime/test_cuda_low_precision_reductions.py::test_cuda_low_precision_max_min_select_the_rhs_for_equal_zeros`` (both params, fp16 and bf16).

``T.cuda.warp_max/warp_min`` expand to TVM's butterfly helper
(``val = max(val, shuffled)``); v2 lowers each step to TIR ``Binary Max/Min``
(``v2/lowering/calls.py`` ``butterfly``), whose float rule orders ``-0 < +0``
(``docs/development/numsim-behaviour-deltas.md`` row D7, Confirmed W4-12:
``cuda_*_min/max`` are NaN-ignoring with ``-0 < +0``; "f32 already followed
this rule", the half types use the same kernels). Legacy selected the
right-hand operand for equal zeros, so lane 0 (``+0`` vs ``-0``) got
``max = -0`` and lane 1 ``max = +0``. v2 gives ``max = +0`` and ``min = -0``
on every lane. D7 has no explicit f16/bf16 sentence (W8 may add one; GPU
golden: ``__hmax(+0, -0)``).
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def cuda_float16_zero_reductions(output: T.Buffer((32, 2), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    raw: T.float32 = T.if_then_else(lane % 2 == 0, T.float32(0.0), T.float32(-0.0))
    value: T.float16 = T.cast(raw, "float16")
    output[lane, 0] = T.cast(T.cuda.warp_max(value, width=2), "float32")
    output[lane, 1] = T.cast(T.cuda.warp_min(value, width=2), "float32")


@T.prim_func
def cuda_bfloat16_zero_reductions(output: T.Buffer((32, 2), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    raw: T.float32 = T.if_then_else(lane % 2 == 0, T.float32(0.0), T.float32(-0.0))
    value: T.bfloat16 = T.cast(raw, "bfloat16")
    output[lane, 0] = T.cast(T.cuda.warp_max(value, width=2), "float32")
    output[lane, 1] = T.cast(T.cuda.warp_min(value, width=2), "float32")


@pytest.mark.parametrize(
    "kernel",
    [cuda_float16_zero_reductions, cuda_bfloat16_zero_reductions],
    ids=["cuda_float16_zero_reductions-fp16", "cuda_bfloat16_zero_reductions-bf16"],
)
def test_cuda_low_precision_max_min_select_the_rhs_for_equal_zeros(kernel):
    """Delta copy (row D7); see the module docstring."""

    result = v2.Engine().run(v2.transpile(kernel), {"output": np.zeros((32, 2), dtype=np.float32)})
    bits = result.outputs["output"].view(np.uint32)
    np.testing.assert_array_equal(
        bits, np.tile(np.array([[0x00000000, 0x80000000]], dtype=np.uint32), (32, 1))
    )
