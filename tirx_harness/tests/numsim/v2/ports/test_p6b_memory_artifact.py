"""v2 ports of ``tests/numsim/integration/test_memory_artifact.py`` tests that
failed under ``NUMSIM_IMPL=v2`` only on implementation pins (legacy scheduler
``stats``, legacy error text) or on the legacy ``suspend_scaffold`` monkeypatch.
Kernels are copied verbatim; observable outputs, completion status and the
stopping diagnostic's status/kind are asserted instead of the pins.

Loop-budget tests: v2 stops an exhausted native loop budget with an
``ExecutionError`` whose stopping diagnostic is ``incomplete``
``analysis_incomplete`` (reason ``Budget: ...``), where legacy raised an error.
No delta row covers this (test-migration.md, "v2 reports incomplete where
legacy raised an error": needs a delta row or a v2 fix), so each such test is
split: the fail-closed half (the run raises) passes, and the legacy
error-status expectation is kept under ``v2_gap``.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import OOB, requires_v2_engine, v2_gap
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

_BUDGET_GAP = (
    "an exhausted native loop iteration budget stops the run with v2.ExecutionError whose "
    "stopping diagnostic is status 'incomplete' kind 'analysis_incomplete' (reason 'Budget: loop "
    "exceeded its iteration budget of N'); legacy raised an error (NumSimExecutionError "
    "'configured native loop iteration budget N'); no delta row"
)


def _first_stop(error: v2.ExecutionError) -> dict:
    """The diagnostic ``Engine.run`` raised for (same selection as run.py)."""

    stops = [
        d
        for d in error.diagnostics
        if d.get("status") in ("error", "incomplete") and d.get("reason") != "subset_execution"
    ]
    assert stops, error.diagnostics
    return stops[0]


def _run(kernel, inputs, **engine_kwargs):
    return v2.Engine(**engine_kwargs).run(v2.transpile(kernel), inputs)


def _assert_completed(result) -> None:
    assert result.status.get("kind") == "completed", result.status


@T.prim_func
def integer_address_same_backing(
    source: T.Buffer((33,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address_bits: T.uint64 = T.reinterpret("uint64", source.ptr_to([0]))
    if lane < 16:
        address_bits = T.reinterpret("uint64", source.ptr_to([lane]))
    else:
        address_bits = T.reinterpret("uint64", source.ptr_to([31 - lane]))
    T.ptx.ld.global_.u32(output[lane], T.reinterpret("handle", address_bits))


@T.prim_func
def accessed_out_of_bounds_shared_view(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    storage = T.alloc_buffer((4,), "uint8", scope="shared")
    alias_data: T.let[
        T.Var(
            name="accessed_out_of_bounds_shared_data",
            ty=PointerType(PrimType("uint32"), "shared"),
        )
    ] = T.reinterpret(PointerType(PrimType("uint32"), "shared"), storage.ptr_to([8]))
    alias = T.decl_buffer((1,), "uint32", data=alias_data, scope="shared")
    if lane == 0:
        output[0] = alias[0]


@T.prim_func
def global_acquire_poll_woken_by_atomic(
    signal: T.Buffer((1,), "uint32"), output: T.Buffer((1,), "uint32")
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    current = T.local_scalar("uint32")
    if warp == 0:
        if lane == 0:
            T.ptx.ld.acquire.gpu.global_.b32(current, signal.ptr_to([0]))
            while current != T.uint32(1):
                T.ptx.ld.acquire.gpu.global_.b32(current, signal.ptr_to([0]))
            output[0] = current
    elif lane == 0:
        T.ptx.atom.release.gpu.global_.add.u32(current, signal.ptr_to([0]), T.uint32(1))


@T.prim_func
def global_acquire_poll_woken_by_cross_cluster_atomic(
    signal: T.Buffer((1,), "uint32"), output: T.Buffer((1,), "uint32")
):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    current = T.local_scalar("uint32")
    if cta == 0:
        if lane == 0:
            T.ptx.ld.acquire.gpu.global_.b32(current, signal.ptr_to([0]))
            while current != T.uint32(1):
                T.ptx.ld.acquire.gpu.global_.b32(current, signal.ptr_to([0]))
            output[0] = current
    elif lane == 0:
        T.ptx.atom.release.gpu.global_.add.u32(current, signal.ptr_to([0]), T.uint32(1))


@T.prim_func
def direct_global_acquire_poll_woken_by_same_cluster_atomic(
    done_counter: T.Buffer((1,), "int32"),
    work_total: T.Buffer((1,), "int32"),
    output: T.Buffer((1,), "int32"),
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    old = T.local_scalar("int32")
    current = T.local_scalar("int32")
    if warp == 0:
        if lane == 0:
            T.ptx.ld.acquire.gpu.global_.s32(current, done_counter.ptr_to([0]))
            while current < work_total[0]:
                T.ptx.ld.acquire.gpu.global_.s32(current, done_counter.ptr_to([0]))
            output[0] = done_counter[0]
    elif lane == 0:
        T.ptx.atom.release.gpu.global_.add.s32(old, done_counter.ptr_to([0]), T.int32(1))


@T.prim_func
def global_volatile_poll_zero_init_first_reload(
    signal: T.Buffer((1,), "uint64"), output: T.Buffer((1,), "uint64")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    current = T.local_scalar("uint64")
    if lane == 0:
        current = T.uint64(0)
        while current != T.uint64(1):
            T.ptx.ld.volatile.global_.u64(current, signal.ptr_to([0]))
        output[0] = current


@T.prim_func
def global_acquire_poll_two_lane_ranges(
    signal: T.Buffer((2,), "uint32"), output: T.Buffer((2,), "uint32")
):
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    current = T.local_scalar("uint32")
    if warp == 0:
        if lane < 2:
            T.ptx.ld.acquire.gpu.global_.b32(current, signal.ptr_to([lane]))
            while current != T.uint32(1):
                T.ptx.ld.acquire.gpu.global_.b32(current, signal.ptr_to([lane]))
            output[lane] = current
    elif warp == 1:
        if lane == 0:
            T.ptx.atom.release.gpu.global_.add.u32(current, signal.ptr_to([0]), T.uint32(1))
        T.ptx.bar.sync(T.uint32(0), T.uint32(64))
        if lane == 0:
            T.ptx.atom.release.gpu.global_.add.u32(current, signal.ptr_to([1]), T.uint32(1))
    else:
        T.ptx.bar.sync(T.uint32(0), T.uint32(64))


@T.prim_func
def non_polling_loop_with_extra_body_effect(
    signal: T.Buffer((1,), "uint32"), output: T.Buffer((1,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    current = T.local_scalar("uint32")
    mirror = T.local_scalar("uint32")
    if lane == 0:
        T.ptx.ld.acquire.gpu.global_.b32(current, signal.ptr_to([0]))
        while current != T.uint32(1):
            T.ptx.ld.acquire.gpu.global_.b32(current, signal.ptr_to([0]))
            mirror = current
        output[0] = mirror


@T.prim_func
def direct_global_poll_with_body_effect(
    signal: T.Buffer((1,), "int32"), output: T.Buffer((1,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    current = T.local_scalar("int32")
    if lane == 0:
        T.ptx.ld.acquire.gpu.global_.s32(current, signal.ptr_to([0]))
        while current < T.int32(1):
            output[0] = output[0] + 1
            T.ptx.ld.acquire.gpu.global_.s32(current, signal.ptr_to([0]))


@T.prim_func
def direct_global_poll_with_two_watched_loads(
    left: T.Buffer((1,), "int32"), right: T.Buffer((1,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_current = T.local_scalar("int32")
    right_current = T.local_scalar("int32")
    if lane == 0:
        T.ptx.ld.acquire.gpu.global_.s32(left_current, left.ptr_to([0]))
        T.ptx.ld.acquire.gpu.global_.s32(right_current, right.ptr_to([0]))
        while left_current < right_current:
            T.ptx.ld.acquire.gpu.global_.s32(left_current, left.ptr_to([0]))
            T.ptx.ld.acquire.gpu.global_.s32(right_current, right.ptr_to([0]))


@T.prim_func
def plain_global_buffer_poll_two_lane_ranges(
    signal: T.Buffer((2,), "int32"), output: T.Buffer((2,), "int32")
):
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    current = T.local_scalar("int32")
    old = T.local_scalar("int32")
    if warp == 0:
        if lane < 2:
            T.ptx.ld.acquire.gpu.global_.s32(current, signal.ptr_to([lane]))
            while current < T.int32(1):
                T.ptx.ld.acquire.gpu.global_.s32(current, signal.ptr_to([lane]))
            T.ptx.ld.acquire.gpu.global_.s32(output[lane], signal.ptr_to([lane]))
    elif warp == 1:
        if lane == 0:
            T.ptx.atom.release.gpu.global_.add.s32(old, signal.ptr_to([0]), T.int32(1))
        T.ptx.bar.sync(T.uint32(0), T.uint32(64))
        if lane == 0:
            T.ptx.atom.release.gpu.global_.add.s32(old, signal.ptr_to([1]), T.int32(1))
    else:
        T.ptx.bar.sync(T.uint32(0), T.uint32(64))


@T.prim_func
def plain_global_buffer_poll_with_body_effect(
    signal: T.Buffer((1,), "int32"), output: T.Buffer((1,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        while signal[0] < T.int32(1):
            output[0] = output[0] + T.int32(1)


@T.prim_func
def plain_global_buffer_poll_with_two_loads(
    left: T.Buffer((1,), "int32"), right: T.Buffer((1,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        while left[0] < right[0]:
            T.evaluate(0)


def test_partitioned_integer_address_flow_preserves_lane_values():
    """Port of ``tests/numsim/integration/test_memory_artifact.py::test_partitioned_integer_address_flow_preserves_lane_values``.

    Dropped: the ``suspend_scaffold._ROOT_SYNC_SPLIT_MIN_LINES`` monkeypatch
    (legacy generated-Rust partitioning knob; v2 has no suspend scaffold).
    The lane values are asserted unchanged.
    """

    source = np.arange(33, dtype=np.uint32) * np.uint32(7) + np.uint32(3)
    result = _run(
        integer_address_same_backing,
        {"source": source, "output": np.zeros(32, dtype=np.uint32)},
    )

    indices = np.concatenate((np.arange(16), np.arange(15, -1, -1)))
    np.testing.assert_array_equal(result.outputs["output"], source[indices])


def test_accessed_out_of_bounds_shared_view_is_rejected():
    """Port of ``tests/numsim/integration/test_memory_artifact.py::test_accessed_out_of_bounds_shared_view_is_rejected``.

    Dropped: ``match="exceeds allocation"`` (legacy text). Asserted: the run
    raises ``v2.ExecutionError`` whose stopping diagnostic is an ``error`` of
    kind ``out_of_bounds``.
    """

    with pytest.raises(v2.ExecutionError) as caught:
        _run(accessed_out_of_bounds_shared_view, {"output": np.zeros(1, dtype=np.uint32)})
    stop = _first_stop(caught.value)
    assert stop["status"] == "error", stop
    assert stop["kind"] in OOB, stop


def test_global_acquire_poll_reschedules_and_atomic_makes_progress():
    """Port of ``tests/numsim/integration/test_memory_artifact.py::test_global_acquire_poll_reschedules_and_atomic_makes_progress``.

    Dropped pins: ``stats["poll_order"] == [0, 1, 0]`` and the
    ``expect_harness_surface`` check of ``completed_task_count == task_count``
    (legacy scheduler stats); completion is asserted via ``result.status``.
    """

    result = _run(
        global_acquire_poll_woken_by_atomic,
        {"signal": np.zeros(1, dtype=np.uint32), "output": np.zeros(1, dtype=np.uint32)},
    )

    _assert_completed(result)
    np.testing.assert_array_equal(result.outputs["signal"], np.array([1], dtype=np.uint32))
    np.testing.assert_array_equal(result.outputs["output"], np.array([1], dtype=np.uint32))


@pytest.mark.parametrize("max_workers", [1, 2])
def test_cross_cluster_atomic_progresses_with_shared_or_parallel_workers(max_workers):
    """Port of ``tests/numsim/integration/test_memory_artifact.py::test_cross_cluster_atomic_progresses_with_shared_or_parallel_workers``
    (the legacy loop over ``max_workers in (1, 2)`` is a parametrization).

    Dropped pins: ``stats["worker_count"]`` and
    ``stats["scheduling_domain_count"]`` (legacy scheduler stats).
    """

    module = v2.transpile(global_acquire_poll_woken_by_cross_cluster_atomic)
    result = v2.Engine(max_workers=max_workers).run(
        module, {"signal": np.zeros(1, dtype=np.uint32), "output": np.zeros(1, dtype=np.uint32)}
    )

    _assert_completed(result)
    np.testing.assert_array_equal(result.outputs["signal"], np.array([1], dtype=np.uint32))
    np.testing.assert_array_equal(result.outputs["output"], np.array([1], dtype=np.uint32))


def test_direct_global_acquire_poll_reschedules_same_cluster_peer():
    """Port of ``tests/numsim/integration/test_memory_artifact.py::test_direct_global_acquire_poll_reschedules_same_cluster_peer``.

    Dropped pin: ``stats["poll_order"]``.
    """

    result = _run(
        direct_global_acquire_poll_woken_by_same_cluster_atomic,
        {
            "done_counter": np.zeros(1, dtype=np.int32),
            "work_total": np.ones(1, dtype=np.int32),
            "output": np.zeros(1, dtype=np.int32),
        },
    )

    np.testing.assert_array_equal(result.outputs["done_counter"], np.ones(1, dtype=np.int32))
    np.testing.assert_array_equal(result.outputs["output"], np.ones(1, dtype=np.int32))


def test_global_volatile_poll_performs_first_reload_without_short_loop_suspend():
    """Port of ``tests/numsim/integration/test_memory_artifact.py::test_global_volatile_poll_performs_first_reload_without_short_loop_suspend``.

    Dropped pin: ``stats["poll_order"] == [0]``.
    """

    result = _run(
        global_volatile_poll_zero_init_first_reload,
        {"signal": np.ones(1, dtype=np.uint64), "output": np.zeros(1, dtype=np.uint64)},
    )

    np.testing.assert_array_equal(result.outputs["output"], np.array([1], dtype=np.uint64))


def test_global_poll_reschedules_with_lane_varying_active_mask():
    """Port of ``tests/numsim/integration/test_memory_artifact.py::test_global_poll_reschedules_with_lane_varying_active_mask``.

    Dropped pin: ``stats["poll_order"]``.
    """

    result = _run(
        global_acquire_poll_two_lane_ranges,
        {"signal": np.zeros(2, dtype=np.uint32), "output": np.zeros(2, dtype=np.uint32)},
    )

    np.testing.assert_array_equal(result.outputs["signal"], np.ones(2, dtype=np.uint32))
    np.testing.assert_array_equal(result.outputs["output"], np.ones(2, dtype=np.uint32))


def test_plain_global_buffer_poll_uses_general_time_slice():
    """Port of ``tests/numsim/integration/test_memory_artifact.py::test_plain_global_buffer_poll_uses_general_time_slice``.

    Dropped pin: ``stats["poll_order"]``.
    """

    result = _run(
        plain_global_buffer_poll_two_lane_ranges,
        {"signal": np.zeros(2, dtype=np.int32), "output": np.zeros(2, dtype=np.int32)},
    )

    np.testing.assert_array_equal(result.outputs["signal"], np.ones(2, dtype=np.int32))
    np.testing.assert_array_equal(result.outputs["output"], np.ones(2, dtype=np.int32))


def _copy(inputs):
    return {name: value.copy() for name, value in inputs.items()}


_EXTRA_BODY_INPUTS = {"signal": np.zeros(1, dtype=np.uint32), "output": np.zeros(1, dtype=np.uint32)}


def test_native_loop_with_extra_body_effect_uses_engine_budget():
    """Port of ``tests/numsim/integration/test_memory_artifact.py::test_native_loop_with_extra_body_effect_uses_engine_budget``
    (fail-closed half).

    Dropped: ``match="configured native loop iteration budget 1"`` (legacy
    text). Asserted: with ``native_loop_iteration_budget=1`` the run raises
    ``v2.ExecutionError`` (a ``NumSimExecutionError``). The legacy
    error-status half is ``..._reports_error_status`` (v2 gap).
    """

    with pytest.raises(v2.ExecutionError):
        _run(non_polling_loop_with_extra_body_effect, _copy(_EXTRA_BODY_INPUTS), native_loop_iteration_budget=1)


@v2_gap(_BUDGET_GAP)
def test_native_loop_with_extra_body_effect_uses_engine_budget_reports_error_status():
    """Second half of ``tests/numsim/integration/test_memory_artifact.py::test_native_loop_with_extra_body_effect_uses_engine_budget``.

    Keeps the legacy contract that budget exhaustion is an execution
    error: the stopping diagnostic has status ``error``.
    """

    with pytest.raises(v2.ExecutionError) as caught:
        _run(non_polling_loop_with_extra_body_effect, _copy(_EXTRA_BODY_INPUTS), native_loop_iteration_budget=1)
    assert _first_stop(caught.value)["status"] == "error", caught.value.diagnostics


_WHILE_SHAPES = pytest.mark.parametrize(
    ("kernel", "inputs"),
    [
        (
            direct_global_poll_with_body_effect,
            {"signal": np.zeros(1, dtype=np.int32), "output": np.zeros(1, dtype=np.int32)},
        ),
        (
            direct_global_poll_with_two_watched_loads,
            {"left": np.zeros(1, dtype=np.int32), "right": np.ones(1, dtype=np.int32)},
        ),
        (
            plain_global_buffer_poll_with_body_effect,
            {"signal": np.zeros(1, dtype=np.int32), "output": np.zeros(1, dtype=np.int32)},
        ),
        (
            plain_global_buffer_poll_with_two_loads,
            {"left": np.zeros(1, dtype=np.int32), "right": np.ones(1, dtype=np.int32)},
        ),
    ],
)


@_WHILE_SHAPES
def test_all_native_while_shapes_use_engine_loop_budget(kernel, inputs):
    """Port of ``tests/numsim/integration/test_memory_artifact.py::test_all_native_while_shapes_use_engine_loop_budget``
    (fail-closed half, same four parametrizations).

    Dropped: ``match="configured native loop iteration budget 1"``.
    Asserted: with ``native_loop_iteration_budget=1`` the run raises
    ``v2.ExecutionError``. The legacy error-status half is
    ``..._reports_error_status`` (v2 gap).
    """

    with pytest.raises(v2.ExecutionError):
        _run(kernel, _copy(inputs), native_loop_iteration_budget=1)


@v2_gap(_BUDGET_GAP)
@_WHILE_SHAPES
def test_all_native_while_shapes_use_engine_loop_budget_reports_error_status(kernel, inputs):
    """Second half of ``tests/numsim/integration/test_memory_artifact.py::test_all_native_while_shapes_use_engine_loop_budget``:
    the stopping diagnostic has status ``error`` (legacy contract)."""

    with pytest.raises(v2.ExecutionError) as caught:
        _run(kernel, _copy(inputs), native_loop_iteration_budget=1)
    assert _first_stop(caught.value)["status"] == "error", caught.value.diagnostics
