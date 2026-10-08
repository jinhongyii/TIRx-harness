"""Exact out-of-bounds and guard evaluation: which lane goes OOB is decided by
the interpreter (guards, lane predicates, data-dependent bounds, loops).

Replaces the ``gap_unportable`` rows of
``tests/analysis_tools/racecheck/test_native_exact_oob.py`` (7),
``tests/analysis_tools/racecheck/test_native_racecheck_exact_control.py`` (3)
and ``test_native_racecheck_artifact.py::test_native_racecheck_ignores_fully_predicated_shared_pointer_load``.
Contract-level OOB is covered by ``racecheck_async_copy.rs::reuse_and_oob``.

Delta P5 (racecheck-behaviour-deltas.md): the legacy ``execution_error{oob}``
(kind ``oob``) is now an ``OutOfBounds`` finding (public kind
``out_of_bounds``) and the access is skipped. Each test asserts the clean
half and the error half with the new kind; message text (lane numbers),
artifact keys, input digests, task counts and ``loop_frames`` payloads are
not asserted (they are legacy payload shape, status C).
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tirx_harness.numsim import v2

from ._runnable import (
    OOB,
    assert_clean,
    assert_error_kind,
    assert_no_incomplete,
    coverage_bounds,
    no_spec,
    requires_v2_engine,
    resource_limits,
)

pytestmark = requires_v2_engine


# -- kernels (copied from the legacy tests) -----------------------------------


@T.prim_func
def native_buffer_selected_lane_oob(
    selected_lane: T.Buffer((32,), "int32"),
    source: T.Buffer((1,), "int32"),
    output: T.Buffer((1,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == selected_lane[lane]:
        output[0] = source[lane]


@T.prim_func
def native_concrete_lane_oob_variants(
    mode: T.int32, bound: T.int32, data_bound: T.Buffer((1,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    direct = T.alloc_buffer((8,), "int32", scope="shared")
    then_buffer = T.alloc_buffer((32,), "int32", scope="shared")
    else_buffer = T.alloc_buffer((8,), "int32", scope="shared")

    if mode == 0:
        if lane < bound:
            direct[lane + 4] = 1
    elif mode == 1:
        if lane < bound:
            direct[7 - lane] = 2
    elif mode == 2:
        nonlinear_bound: T.let = data_bound[0] * data_bound[0]
        if lane < nonlinear_bound:
            direct[lane] = 3
    elif mode == 3:
        if lane < bound:
            then_buffer[lane] = 4
        else:
            else_buffer[lane] = 5
    else:
        if lane < bound:
            direct[lane] = 6


@T.prim_func
def native_warp_tiled_oob(warp_limit: T.int32):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((40,), "int32", scope="shared")
    if (warp < warp_limit) and (lane < 8):
        shared[warp * 16 + lane] = 7


@T.prim_func
def native_boolean_guard_oob(mode: T.int32, bound: T.int32, flag: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "int32", scope="shared")

    if mode == 0:
        if (flag[0] != 0) and (lane < bound):
            shared[lane] = 1
    elif mode == 1:
        if (flag[0] != 0) & (lane < bound):
            shared[lane] = 2
    elif mode == 2:
        if (flag[0] != 0) | (lane < bound):
            shared[lane] = 3
    else:
        if not ((flag[0] == 0) | (lane >= bound)):
            shared[lane] = 4


@T.prim_func
def native_data_guarded_oob(mode: T.int32, enabled: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "int32", scope="shared")

    if mode == 0:
        if enabled[0] != 0:
            shared[lane] = 1
    else:
        if lane < enabled[0]:
            shared[50] = 2


@T.prim_func
def native_numeric_loop_relay_oob(
    limit: T.int32,
    base: T.Buffer((1,), "int32"),
    output: T.Buffer((2,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    index = T.local_scalar("int32")

    for iteration in T.serial(limit):
        if lane == 0:
            index = base[0] + iteration
            output[index] = iteration


@T.prim_func
def native_racecheck_exact_data_dependent_control(
    mode: T.int32,
    if_limit: T.int32,
    for_extents: T.Buffer((32,), "int32"),
    while_extents: T.Buffer((32,), "int32"),
    source: T.Buffer((4,), "int32"),
    output: T.Buffer((32,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    iteration = T.local_scalar("int32")
    iteration = T.int32(0)

    if mode == 0:
        if lane < if_limit:
            output[lane] = source[lane]
    elif mode == 1:
        for step in T.serial(for_extents[lane]):
            if lane < 4:
                output[lane] = source[lane + step]
    else:
        while iteration < while_extents[lane]:
            if lane < 4:
                output[lane] = source[lane + iteration]
            iteration = iteration + 1


@T.prim_func
def native_racecheck_fully_predicated_shared_pointer_load(
    output: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "float32", scope="shared")
    if lane < 0:
        T.ptx.ld.shared.f32(output[lane], shared.ptr_to([lane]))


# -- helpers --------------------------------------------------------------------


def _selected_lane_inputs(selected_lane: int):
    lane_selectors = np.full(32, -1, dtype=np.int32)
    lane_selectors[selected_lane] = selected_lane
    return {
        "selected_lane": lane_selectors,
        "source": np.array([17], dtype=np.int32),
        "output": np.zeros(1, dtype=np.int32),
    }


def _sync(kernel, inputs):
    return v2.synccheck(
        kernel,
        inputs,
        coverage_bounds=coverage_bounds(),
        resource_limits=resource_limits(
            max_schedules=16, max_backtrack_nodes=100, max_events_per_run=1_000,
            max_total_events=10_000, max_loop_steps=1_000,
        ),
    )


def _assert_oob(report) -> None:
    assert_error_kind(report, OOB)
    assert_no_incomplete(report)


def _control_inputs(*, mode: int, if_limit: int = 0, for_extents=None, while_extents=None):
    return {
        "mode": np.int32(mode),
        "if_limit": np.int32(if_limit),
        "for_extents": np.zeros(32, dtype=np.int32) if for_extents is None else for_extents,
        "while_extents": np.zeros(32, dtype=np.int32) if while_extents is None else while_extents,
        "source": np.arange(4, dtype=np.int32),
        "output": np.zeros(32, dtype=np.int32),
    }


def _lane3_extents():
    clean = np.zeros(32, dtype=np.int32)
    clean[:4] = 1
    error = clean.copy()
    error[3] = 2
    return clean, error


# -- test_native_exact_oob.py ---------------------------------------------------


def test_racecheck_resolves_buffer_selected_oob_exactly():
    """Replaces ``tests/analysis_tools/racecheck/test_native_exact_oob.py::test_public_native_racecheck_resolves_buffer_selected_oob_exactly``.

    Lane 1 selected -> ``source[1]`` is OOB (error); lane 0 selected -> clean.
    Delta P5: kind ``oob`` -> ``out_of_bounds``.
    """

    _assert_oob(v2.racecheck(native_buffer_selected_lane_oob, _selected_lane_inputs(1)))
    assert_clean(v2.racecheck(native_buffer_selected_lane_oob, _selected_lane_inputs(0)))


def test_racecheck_and_synccheck_scope_typed_lane_predicate_oob():
    """Replaces ``tests/analysis_tools/racecheck/test_native_exact_oob.py::test_native_racecheck_and_synccheck_scope_typed_lane_predicate_oob``.

    Synccheck runs the interpreter too: selected lane 0 is clean, lane 1 is an
    OOB error. Legacy kind ``oob`` (execution_error); v2 reports the runtime
    diagnostic ``out_of_bounds`` (P5 naming; no sync delta row).
    """

    assert_clean(_sync(native_buffer_selected_lane_oob, _selected_lane_inputs(0)))
    _assert_oob(_sync(native_buffer_selected_lane_oob, _selected_lane_inputs(1)))


def test_synccheck_executes_numeric_loop_address_relay():
    """Replaces ``tests/analysis_tools/racecheck/test_native_exact_oob.py::test_public_native_synccheck_executes_numeric_loop_address_relay``.

    A loop-carried index ``base[0] + iteration`` reaches ``output[2]`` only when
    ``base == 1``: clean for base 0, OOB error for base 1.
    """

    def run(base: int):
        return _sync(
            native_numeric_loop_relay_oob,
            {"limit": np.int32(2), "base": np.array([base], dtype=np.int32), "output": np.zeros(2, dtype=np.int32)},
        )

    assert_clean(run(0))
    _assert_oob(run(1))


@pytest.mark.parametrize(
    ("mode", "clean_bound", "error_bound", "clean_data_bound", "error_data_bound"),
    [
        pytest.param(0, 4, 5, 0, 0, id="offset"),
        pytest.param(1, 8, 9, 0, 0, id="decreasing"),
        pytest.param(2, 0, 0, 2, 3, id="nonlinear-data-bound"),
        pytest.param(3, 32, 8, 0, 0, id="then-else-complement"),
        pytest.param(4, 8, 9, 0, 0, id="single-warp-lane-guard"),
    ],
)
def test_racecheck_resolves_concrete_lane_oob_variants_exactly(
    mode, clean_bound, error_bound, clean_data_bound, error_data_bound
):
    """Replaces ``tests/analysis_tools/racecheck/test_native_exact_oob.py::test_public_native_racecheck_resolves_concrete_lane_oob_variants_exactly`` (all params).

    Delta P5: kind ``oob`` -> ``out_of_bounds``.
    """

    def run(bound, data_bound):
        return v2.racecheck(
            native_concrete_lane_oob_variants,
            {"mode": np.int32(mode), "bound": np.int32(bound), "data_bound": np.array([data_bound], dtype=np.int32)},
        )

    assert_clean(run(clean_bound, clean_data_bound))
    _assert_oob(run(error_bound, error_data_bound))


def test_racecheck_resolves_warp_tiled_oob_exactly():
    """Replaces ``tests/analysis_tools/racecheck/test_native_exact_oob.py::test_public_native_racecheck_resolves_warp_tiled_oob_exactly``.

    ``warp_limit`` 3 is clean; 4 lets warp 3 write ``shared[48..56)`` of a
    40-element buffer. Delta P5: ``out_of_bounds``; the legacy task counts
    (abort after warp 3) are not asserted -- under P5 the access is skipped.
    """

    assert_clean(v2.racecheck(native_warp_tiled_oob, {"warp_limit": np.int32(3)}))
    _assert_oob(v2.racecheck(native_warp_tiled_oob, {"warp_limit": np.int32(4)}))


@pytest.mark.parametrize(
    ("mode", "clean_flag", "clean_bound", "error_flag", "error_bound"),
    [
        pytest.param(0, 1, 8, 1, 9, id="logical-and"),
        pytest.param(1, 1, 8, 1, 9, id="boolean-bitwise-and"),
        pytest.param(2, 0, 8, 1, 0, id="boolean-bitwise-or"),
        pytest.param(3, 1, 8, 1, 9, id="not-of-or"),
    ],
)
def test_racecheck_resolves_boolean_guard_spellings_exactly(mode, clean_flag, clean_bound, error_flag, error_bound):
    """Replaces ``tests/analysis_tools/racecheck/test_native_exact_oob.py::test_public_native_racecheck_resolves_boolean_guard_spellings_exactly`` (all params).

    Delta P5: kind ``oob`` -> ``out_of_bounds``.
    """

    def run(flag, bound):
        return v2.racecheck(
            native_boolean_guard_oob,
            {"mode": np.int32(mode), "bound": np.int32(bound), "flag": np.array([flag], dtype=np.int32)},
        )

    assert_clean(run(clean_flag, clean_bound))
    _assert_oob(run(error_flag, error_bound))


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(0, id="pure-data-guard"),
        pytest.param(1, id="constant-index-under-data-derived-lane-guard"),
    ],
)
def test_racecheck_resolves_data_guarded_oob_exactly(mode):
    """Replaces ``tests/analysis_tools/racecheck/test_native_exact_oob.py::test_public_native_racecheck_resolves_data_guarded_oob_exactly`` (all params).

    ``enabled`` 0 is clean, 1 is OOB. Delta P5: ``out_of_bounds``.
    """

    def run(enabled):
        return v2.racecheck(
            native_data_guarded_oob, {"mode": np.int32(mode), "enabled": np.array([enabled], dtype=np.int32)}
        )

    assert_clean(run(0))
    _assert_oob(run(1))


# -- test_native_racecheck_exact_control.py ------------------------------------


def test_racecheck_resolves_scalar_lane_guard_exactly():
    """Replaces ``tests/analysis_tools/racecheck/test_native_racecheck_exact_control.py::test_public_native_racecheck_resolves_scalar_lane_guard_exactly``.

    ``if_limit`` 4 is clean; 5 reads ``source[4]`` (OOB). Delta P5.
    """

    kernel = native_racecheck_exact_data_dependent_control
    assert_clean(v2.racecheck(kernel, _control_inputs(mode=0, if_limit=4)))
    _assert_oob(v2.racecheck(kernel, _control_inputs(mode=0, if_limit=5)))


def test_racecheck_executes_lane_varying_dynamic_for_exactly():
    """Replaces ``tests/analysis_tools/racecheck/test_native_racecheck_exact_control.py::test_public_native_racecheck_executes_lane_varying_dynamic_for_exactly``.

    A per-lane ``for`` extent: lane 3 running two iterations reads
    ``source[4]`` (OOB). Delta P5. The ``loop_frames`` payload is C.
    """

    clean, error = _lane3_extents()
    kernel = native_racecheck_exact_data_dependent_control
    assert_clean(v2.racecheck(kernel, _control_inputs(mode=1, for_extents=clean)))
    _assert_oob(v2.racecheck(kernel, _control_inputs(mode=1, for_extents=error)))


def test_racecheck_executes_lane_varying_dynamic_while_exactly():
    """Replaces ``tests/analysis_tools/racecheck/test_native_racecheck_exact_control.py::test_public_native_racecheck_executes_lane_varying_dynamic_while_exactly``.

    Same as the ``for`` case with a per-lane ``while`` trip count. Delta P5.
    """

    clean, error = _lane3_extents()
    kernel = native_racecheck_exact_data_dependent_control
    assert_clean(v2.racecheck(kernel, _control_inputs(mode=2, while_extents=clean)))
    _assert_oob(v2.racecheck(kernel, _control_inputs(mode=2, while_extents=error)))


# -- test_native_racecheck_artifact.py -----------------------------------------


@no_spec(8, "fully predicated-off accesses emit no access")
def test_racecheck_ignores_fully_predicated_shared_pointer_load():
    """Replaces ``tests/analysis_tools/racecheck/test_native_racecheck_artifact.py::test_native_racecheck_ignores_fully_predicated_shared_pointer_load``.

    ``if lane < 0`` predicates every lane off: the ``ld.shared`` emits no
    access, so no uninitialized-read advisory, no OOB (``output[lane]`` with a
    negative lane is never evaluated) and a clean verdict. The legacy
    ``access_count == 0`` is not observable through the public report.
    """

    assert_clean(
        v2.racecheck(native_racecheck_fully_predicated_shared_pointer_load, {"output": np.zeros(32, dtype=np.float32)})
    )
