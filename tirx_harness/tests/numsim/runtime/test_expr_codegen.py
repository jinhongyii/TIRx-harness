from __future__ import annotations

import re
import subprocess

import numpy as np
import pytest
from tvm import tirx

from tirx_harness import numsim
from tests.numsim.support.kernels import fixed_width_integer_expression_mix
from tests.numsim.support.manifest import device_kernel, parse_kernel
from tvm.script import tirx as T


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


@T.prim_func
def truncating_integer_expression_mix(output: T.Buffer((32, 3), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    dividend: T.let = lane - T.int32(7)
    output[lane, 0] = T.truncdiv(dividend, T.int32(2))
    output[lane, 1] = T.truncmod(dividend, T.int32(2))
    output[lane, 2] = T.floordiv(dividend, T.int32(2))


@T.prim_func
def captured_expression_inputs(
    source: T.Buffer((32,), "int32"),
    scratch: T.Buffer((32,), "int32"),
    output: T.Buffer((32, 6), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane, 0] = source[lane] * T.int32(2) + T.int32(2)
    output[lane, 1] = source[lane] * T.int32(2) + T.int32(3)
    scratch[lane] = source[lane] * T.int32(3) + T.int32(2)
    output[lane, 2] = scratch[lane] * T.int32(2) + T.int32(3)
    scratch[lane] = source[lane] * T.int32(-3) + T.int32(4)
    output[lane, 3] = scratch[lane] * T.int32(2) + T.int32(3)
    output[lane, 4] = T.cast((source[lane] & T.int32(31)) < T.int32(17), "int32")
    output[lane, 5] = T.if_then_else(
        lane < 16,
        source[lane] * T.int32(2) + T.int32(7),
        source[lane] * T.int32(-2) + T.int32(5),
    )


def test_expression_captures_preserve_constants_load_order_and_wrapping(tmp_path):
    source = np.resize(
        np.array([0, 1, -1, 2**30, -(2**30), 2**31 - 1, -(2**31), 17], dtype=np.int32),
        32,
    )
    wide = source.astype(np.int64)
    expected = np.stack(
        [
            wide * 2 + 2,
            wide * 2 + 3,
            (wide * 3 + 2) * 2 + 3,
            (wide * -3 + 4) * 2 + 3,
            (wide & 31) < 17,
            np.where(np.arange(32) < 16, wide * 2 + 7, wide * -2 + 5),
        ],
        axis=1,
    ).astype(np.int32)
    module = numsim.transpile(
        captured_expression_inputs,
        cache_dir=tmp_path,
    )
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "scratch": np.zeros(32, dtype=np.int32),
            "output": np.zeros((32, 6), dtype=np.int32),
        },
    )
    assert result.verdict == "clean", result.diagnostics
    np.testing.assert_array_equal(result.outputs["output"], expected)


def _lines_with(body: str, fragment: str) -> list[str]:
    return [line.strip() for line in body.splitlines() if fragment in line]


def test_fixed_width_integer_operations_preserve_casts_and_wrap_observably(tmp_path):
    output = np.zeros((32, 7), dtype=np.uint64)

    module = numsim.transpile(fixed_width_integer_expression_mix, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    expected = []
    for lane in range(32):
        unsigned = (lane - 1) & ((1 << 32) - 1)
        expected.append(
            [
                unsigned + 1 if unsigned else (1 << 64) - 1,
                (lane - 1) & ((1 << 64) - 1),
                unsigned & 0xFFFF,
                int(unsigned != 0),
                unsigned // 5,
                unsigned % 5,
                int(unsigned + 3 < 10),
            ]
        )
    expected = np.array(expected, dtype=np.uint64)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_truncating_integer_division_and_remainder_match_tir_semantics(tmp_path):
    output = np.zeros((32, 3), dtype=np.int32)

    module = numsim.transpile(truncating_integer_expression_mix, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    dividend = np.arange(32, dtype=np.int32) - 7
    quotient_magnitude = np.abs(dividend) // 2
    quotient = np.where(dividend < 0, -quotient_magnitude, quotient_magnitude)
    floor = np.floor_divide(dividend, np.int32(2))
    expected = np.stack((quotient, dividend - quotient * 2, floor), axis=1).astype(np.int32)
    assert tuple(expected[0, (0, 2)]) == (-3, -4)
    np.testing.assert_array_equal(result.outputs["output"], expected)


