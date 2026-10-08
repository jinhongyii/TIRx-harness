"""v2 ports of ``tests/numsim/runtime/test_cuda_generic_scalar_forms.py``.

``analyze(kernel).unsupported == ()`` becomes ``v2.transpile`` accepting the
kernel (it raises ``UnsupportedTIRxError`` on any unsupported site). Kernels
and oracles are copied verbatim.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def warp_float64_reductions(output: T.Buffer((32, 3), "float64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value: T.let = T.cast(lane, "float64") * T.float64(0.5) - T.float64(8)
    output[lane, 0] = T.cuda.warp_sum(value, width=8)
    output[lane, 1] = T.cuda.warp_max(value, width=8)
    output[lane, 2] = T.cuda.warp_min(value, width=8)


@T.prim_func
def cta_int32_reductions(output: T.Buffer((3, 3), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    sum_scratch = T.alloc_buffer((2,), "int32", scope="shared")
    max_scratch = T.alloc_buffer((2,), "int32", scope="shared")
    min_scratch = T.alloc_buffer((2,), "int32", scope="shared")
    value: T.let = warp * 100 + lane - 50
    sum_result: T.let = T.cuda.cta_sum(value, 2, sum_scratch.ptr_to([0]))
    max_result: T.let = T.cuda.cta_max(value, 2, max_scratch.ptr_to([0]))
    min_result: T.let = T.cuda.cta_min(value, 2, min_scratch.ptr_to([0]))
    if (warp == 0) and (lane == 0):
        output[0, 0] = sum_result
        output[0, 1] = sum_scratch[0]
        output[0, 2] = sum_scratch[1]
        output[1, 0] = max_result
        output[1, 1] = max_scratch[0]
        output[1, 2] = max_scratch[1]
        output[2, 0] = min_result
        output[2, 1] = min_scratch[0]
        output[2, 2] = min_scratch[1]


@T.prim_func
def cuda_ldg_scalar_types(
    source_f64: T.Buffer((32,), "float64"),
    source_i64: T.Buffer((32,), "int64"),
    source_f16: T.Buffer((32,), "float16"),
    source_bf16: T.Buffer((32,), "bfloat16"),
    output_f64: T.Buffer((32,), "float64"),
    output_i64: T.Buffer((32,), "int64"),
    output_f16: T.Buffer((32,), "float16"),
    output_bf16: T.Buffer((32,), "bfloat16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    reverse: T.let = 31 - lane
    output_f64[lane] = T.cuda.ldg(source_f64.ptr_to([reverse]), "float64")
    output_i64[lane] = T.cuda.ldg(source_i64.ptr_to([reverse]), "int64")
    output_f16[lane] = T.cuda.ldg(source_f16.ptr_to([reverse]), "float16")
    output_bf16[lane] = T.cuda.ldg(source_bf16.ptr_to([reverse]), "bfloat16")


@T.prim_func
def cuda_ldg_misaligned_float64(
    source: T.Buffer((33,), "float64"), output: T.Buffer((32,), "float64")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pointer = T.ptr_byte_offset(source.ptr_to([lane]), T.uint32(4), "float64")
    output[lane] = T.cuda.ldg(pointer, "float64")



def test_warp_float64_reductions_use_generic_typed_runtime():
    """Port of ``tests/numsim/runtime/test_cuda_generic_scalar_forms.py::test_warp_float64_reductions_use_generic_typed_runtime``."""

    module = v2.transpile(warp_float64_reductions)
    result = v2.Engine().run(module, {"output": np.zeros((32, 3), dtype=np.float64)})

    values = np.arange(32, dtype=np.float64) * 0.5 - 8.0
    expected = np.empty((32, 3), dtype=np.float64)
    for group_start in range(0, 32, 8):
        group = values[group_start : group_start + 8]
        expected[group_start : group_start + 8, 0] = group.sum()
        expected[group_start : group_start + 8, 1] = group.max()
        expected[group_start : group_start + 8, 2] = group.min()
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_cta_int32_reductions_preserve_typed_scratch():
    """Port of ``tests/numsim/runtime/test_cuda_generic_scalar_forms.py::test_cta_int32_reductions_preserve_typed_scratch``."""

    module = v2.transpile(cta_int32_reductions)
    result = v2.Engine().run(module, {"output": np.zeros((3, 3), dtype=np.int32)})

    np.testing.assert_array_equal(
        result.outputs["output"],
        np.array([[992, 992, 2096], [81, 81, 81], [-50, -50, 50]], dtype=np.int32),
    )


def test_cuda_ldg_supports_cuda_scalar_overloads():
    """Port of ``tests/numsim/runtime/test_cuda_generic_scalar_forms.py::test_cuda_ldg_supports_cuda_scalar_overloads``."""

    source_f64 = np.linspace(-4.0, 7.0, 32, dtype=np.float64)
    source_i64 = np.arange(32, dtype=np.int64) * 1_000_000_007 - 11
    source_f16 = np.linspace(-2.0, 3.0, 32, dtype=np.float16)
    source_bf16 = (
        np.linspace(-1.0, 2.0, 32, dtype=np.float32).view(np.uint32) >> np.uint32(16)
    ).astype(np.uint16)

    module = v2.transpile(cuda_ldg_scalar_types)
    result = v2.Engine().run(
        module,
        {
            "source_f64": source_f64,
            "source_i64": source_i64,
            "source_f16": source_f16,
            "source_bf16": source_bf16,
            "output_f64": np.zeros(32, dtype=np.float64),
            "output_i64": np.zeros(32, dtype=np.int64),
            "output_f16": np.zeros(32, dtype=np.float16),
            "output_bf16": np.zeros(32, dtype=np.uint16),
        },
    )

    np.testing.assert_array_equal(result.outputs["output_f64"], source_f64[::-1])
    np.testing.assert_array_equal(result.outputs["output_i64"], source_i64[::-1])
    np.testing.assert_array_equal(result.outputs["output_f16"], source_f16[::-1])
    np.testing.assert_array_equal(
        np.asarray(result.outputs["output_bf16"]).view(np.uint16), source_bf16[::-1]
    )


def test_cuda_ldg_keeps_natural_alignment_checks():
    """Port of ``tests/numsim/runtime/test_cuda_generic_scalar_forms.py::test_cuda_ldg_keeps_natural_alignment_checks``.

    Dropped pin: the legacy message ``"8-byte alignment"``; the port asserts
    the v2 error kind ``misaligned``.
    """

    module = v2.transpile(cuda_ldg_misaligned_float64)
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(
            module,
            {"source": np.arange(33, dtype=np.float64), "output": np.zeros(32, dtype=np.float64)},
        )
    stops = [d for d in caught.value.diagnostics if d.get("status") in ("error", "incomplete")]
    assert stops and (stops[0]["status"], stops[0]["kind"]) == ("error", "misaligned"), stops
