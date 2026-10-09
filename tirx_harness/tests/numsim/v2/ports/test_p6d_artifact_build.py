"""v2 copies of three ``tests/numsim/integration/test_artifact_build.py``
functions triaged in W9 phase 6 (internal other-assertion). Kernels copied
verbatim (``lane_add`` and ``overlapping_alias_write`` from
``tests/numsim/support/kernels.py``).

- ``test_physical_numpy_aliases_survive_the_artifact_boundary``: **port**.
  Dropped the in-place mutation pins on the caller's ``backing`` array and the
  ``expect_harness_surface`` fixture (v2 ``Engine.run`` never mutates its
  inputs; same API change as the earlier ``test_global_alias_artifact`` port).
  Kept: the overlapping views read the pre-launch source and the destination
  output is ``original[:32] + 1``; the caller's array is unchanged.
- ``test_run_case_uses_declared_output_bindings``: **port**. Dropped the
  in-place check of the caller's ``output`` array; kept ``report.ok`` and
  asserted the result values through ``Engine.run`` outputs.
- ``test_cuda_float_reductions_match_nan_and_signed_zero_bits``: **delta**,
  ``numsim-behaviour-deltas.md`` row D8 (Confirmed: NaN result of TIR ``+``
  is the first NaN operand, quieted). ``T.cuda.cta_sum`` of one
  ``0x7FC12345`` NaN and zeros gives ``0x7FC12345`` (payload kept); legacy
  pinned the device canonical NaN ``0x7FFFFFFF`` (the open GPU-golden question
  of D8). The six signed-zero max/min words are unchanged. Dropped the
  ``rust_source`` pins.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.cases import NumSimCase

pytestmark = requires_v2_engine


@T.prim_func
def lane_add(
    left: T.Buffer((100,), "float32"),
    right: T.Buffer((100,), "float32"),
    output: T.Buffer((100,), "float32"),
):
    T.device_entry()
    _cta = T.cta_id([1])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    index = T.meta_var(warp * 32 + lane)
    if index < 100:
        output[index] = left[index] + right[index]


@T.prim_func
def overlapping_alias_write(
    source: T.Buffer((32,), "float32"), destination: T.Buffer((32,), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    destination[lane] = source[lane] + T.float32(1)


@T.prim_func
def cuda_float_reduction_edge_bits(
    operands: T.Buffer((2,), "uint32"), output: T.Buffer((7,), "uint32")
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    sum_scratch = T.alloc_buffer((2,), "float32", scope="shared")
    max_scratch = T.alloc_buffer((2,), "float32", scope="shared")
    min_scratch = T.alloc_buffer((2,), "float32", scope="shared")
    is_special = (warp == 0) and (lane == 1)
    nan_value = T.if_then_else(
        is_special, T.reinterpret("float32", T.uint32(0x7FC12345)), T.float32(0.0)
    )
    zero_value = T.if_then_else(is_special, T.float32(0.0), T.float32(-0.0))
    sum_result = T.cuda.cta_sum(nan_value, 2, sum_scratch.ptr_to([0]))
    max_result = T.cuda.cta_max(zero_value, 2, max_scratch.ptr_to([0]))
    min_result = T.cuda.cta_min(zero_value, 2, min_scratch.ptr_to([0]))
    if (warp == 0) and (lane == 0):
        lhs = T.reinterpret("float32", operands[0])
        rhs = T.reinterpret("float32", operands[1])
        output[0] = T.reinterpret("uint32", sum_result)
        output[1] = T.reinterpret("uint32", max_result)
        output[2] = T.reinterpret("uint32", min_result)
        output[3] = T.reinterpret("uint32", T.max(lhs, rhs))
        output[4] = T.reinterpret("uint32", T.max(rhs, lhs))
        output[5] = T.reinterpret("uint32", T.min(lhs, rhs))
        output[6] = T.reinterpret("uint32", T.min(rhs, lhs))


def test_physical_numpy_aliases_survive_the_artifact_boundary():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_physical_numpy_aliases_survive_the_artifact_boundary``
    (no input mutation; see the module docstring)."""

    backing = np.arange(48, dtype=np.float32)
    original = backing.copy()
    source = backing[:32]
    destination = backing[16:]

    result = v2.Engine().run(
        v2.transpile(overlapping_alias_write),
        {"source": source, "destination": destination},
        outputs=("destination",),
    )

    assert set(result.outputs) == {"destination"}
    np.testing.assert_array_equal(result.outputs["destination"], original[:32] + 1)
    np.testing.assert_array_equal(backing, original)


def test_run_case_uses_declared_output_bindings():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_run_case_uses_declared_output_bindings``
    (no input mutation; see the module docstring)."""

    left = np.arange(100, dtype=np.float32)
    right = np.linspace(0, 1, 100, dtype=np.float32)
    output = np.zeros(100, dtype=np.float32)
    expected = left + right
    case = NumSimCase(
        kernel=lane_add,
        args={"left": left, "right": right, "output": output},
        outputs=("output",),
        reference=lambda: {"output": expected.copy()},
    )

    report = v2.run_case(case)

    assert report.ok
    result = v2.Engine().run(v2.transpile(lane_add), case.args, outputs=case.outputs)
    np.testing.assert_array_equal(result.outputs["output"], expected)
    np.testing.assert_array_equal(output, np.zeros(100, dtype=np.float32))


def test_cuda_float_reductions_match_nan_and_signed_zero_bits():
    """Delta copy of ``tests/numsim/integration/test_artifact_build.py::test_cuda_float_reductions_match_nan_and_signed_zero_bits``
    (row D8; see the module docstring)."""

    operands = np.array([0x80000000, 0x00000000], dtype=np.uint32)
    result = v2.Engine().run(
        v2.transpile(cuda_float_reduction_edge_bits),
        {"operands": operands, "output": np.zeros(7, dtype=np.uint32)},
    )

    np.testing.assert_array_equal(
        result.outputs["output"],
        np.array(
            [0x7FC12345, 0x00000000, 0x80000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000],
            dtype=np.uint32,
        ),
    )
