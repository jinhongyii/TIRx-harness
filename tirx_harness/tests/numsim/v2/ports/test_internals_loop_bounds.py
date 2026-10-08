"""v2 ports of the legacy loop-bounds tests; the ``module.rust_source`` text
pins (native loop scaffolding names, Rust types) are dropped."""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.v2.lowering import lower
from tirx_harness.numsim.v2.lowering import program_builder as pb

pytestmark = requires_v2_engine

BT = 16


@T.prim_func
def loop_var_dtype_select(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for iteration in T.serial(3):
        output[lane] = T.Select(lane < 16, T.int32(7), iteration)


@T.prim_func
def runtime_for_range(
    minimum: T.Buffer((32,), "int32"),
    extents: T.Buffer((32,), "int32"),
    steps: T.Buffer((32,), "int32"),
    output: T.Buffer((32,), "int32"),
    selected: T.Buffer((32,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for iteration in T.serial(minimum[lane], minimum[lane] + extents[lane], step=steps[lane]):
        output[lane] = iteration
        selected[lane] = T.Select(lane < 16, T.int32(7), iteration)


@T.prim_func
def nested_triangular_for(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    j = T.lane_id([32])
    output[j] = 0
    if j < BT:
        for outer in T.serial(j + 1, BT):
            output[j] = output[j] + 1
            for _inner in T.serial(j + 1, outer):
                output[j] = output[j] + 1


@T.prim_func
def static_zero_trip_for(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = 7
    for iteration in T.serial(0):
        output[lane] = iteration + 100



def test_uniform_loop_var_preserves_tir_dtype_in_select():
    """Port of ``tests/numsim/runtime/test_loop_bounds.py::test_uniform_loop_var_preserves_tir_dtype_in_select``.

    Dropped pin: ``": i32 =" in module.rust_source``.
    """

    module = v2.transpile(loop_var_dtype_select)
    result = v2.Engine().run(module, {"output": np.zeros(32, dtype=np.int32)})

    expected = np.where(np.arange(32) < 16, 7, 2).astype(np.int32)
    np.testing.assert_array_equal(result.outputs["output"], expected)
    # W11: the pinned fact on the lowered Program: the Select over the loop variable
    # keeps the TIR int32 dtype (legacy: ``: i32 =`` in the generated Rust).
    (select,) = [i for i in lower(loop_var_dtype_select).code if i.variant == "Select"]
    assert select.ty == pb.Ty("S32"), select


def test_lane_varying_min_extent_and_step_use_masked_native_loop():
    """Port of ``tests/numsim/runtime/test_loop_bounds.py::test_lane_varying_min_extent_and_step_use_masked_native_loop``.

    Dropped pins: ``"iteration_mask_loop"``, ``"WarpValue<i32>"`` in and
    ``"differs between active lanes"`` not in ``module.rust_source``. The
    zero-step rejection half lives in
    :func:`test_lane_varying_for_rejects_zero_step` (v2 gap).
    """

    minimum = np.arange(32, dtype=np.int32) - 16
    extents = np.arange(32, dtype=np.int32) % 9 + 1
    steps = np.arange(32, dtype=np.int32) % 4 + 1
    initial = np.full(32, -999, dtype=np.int32)

    module = v2.transpile(runtime_for_range)
    result = v2.Engine().run(
        module,
        {
            "minimum": minimum,
            "extents": extents,
            "steps": steps,
            "output": initial.copy(),
            "selected": initial.copy(),
        },
    )

    last = minimum + ((extents - 1) // steps) * steps
    selected = np.where(np.arange(32) < 16, 7, last).astype(np.int32)
    np.testing.assert_array_equal(result.outputs["output"], last)
    np.testing.assert_array_equal(result.outputs["selected"], selected)


def test_lane_varying_for_rejects_zero_step():
    """Second half of ``tests/numsim/runtime/test_loop_bounds.py::test_lane_varying_min_extent_and_step_use_masked_native_loop``
    (split out so the passing output half is not hidden by this gap).

    Keeps the fail-closed error contract: the run raises
    ``v2.ExecutionError`` and reports an ``error`` (not ``incomplete``)
    diagnostic. The legacy ``match="For step must be positive"`` text is
    not pinned.
    """

    minimum = np.arange(32, dtype=np.int32) - 16
    extents = np.arange(32, dtype=np.int32) % 9 + 1
    steps = np.arange(32, dtype=np.int32) % 4 + 1
    initial = np.full(32, -999, dtype=np.int32)
    module = v2.transpile(runtime_for_range)

    invalid_steps = steps.copy()
    invalid_steps[5] = 0
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(
            module,
            {
                "minimum": minimum,
                "extents": extents,
                "steps": invalid_steps,
                "output": initial.copy(),
                "selected": initial.copy(),
            },
        )
    assert any(d.get("status") == "error" for d in caught.value.diagnostics), caught.value.diagnostics


def test_v0_style_nested_triangular_bounds():
    """Port of ``tests/numsim/runtime/test_loop_bounds.py::test_v0_style_nested_triangular_bounds``.

    Dropped pin: ``module.rust_source.count("iteration_mask_loop") >= 2``.
    """

    module = v2.transpile(nested_triangular_for)
    result = v2.Engine().run(module, {"output": np.zeros(32, dtype=np.int32)})

    expected = np.zeros(32, dtype=np.int32)
    outer_iterations = BT - 1 - np.arange(BT, dtype=np.int32)
    expected[:BT] = outer_iterations * (outer_iterations + 1) // 2
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_static_zero_trip_for_omits_native_loop_scaffolding():
    """Port of ``tests/numsim/runtime/test_loop_bounds.py::test_static_zero_trip_for_omits_native_loop_scaffolding``.

    Dropped pins: ``"loop_offset_loop"`` and ``"live_mask_loop"`` not in
    ``module.rust_source``.
    """

    module = v2.transpile(static_zero_trip_for)
    result = v2.Engine().run(module, {"output": np.zeros(32, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.full(32, 7, dtype=np.int32))
