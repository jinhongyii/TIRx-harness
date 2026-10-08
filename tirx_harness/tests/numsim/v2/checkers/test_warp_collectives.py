"""Warp-collective (``shfl.sync`` via ``T.cuda.warp_sum``) participation.

Replaces four ``gap_unportable`` synccheck rows:
``tests/analysis_tools/synccheck/test_native_kernel_contracts.py`` (2) and
``tests/analysis_tools/synccheck/test_native_synccheck_exact_control.py`` (2).
``warp_collective_divergence`` is an interpreter ``ExecError`` (membermask not
equal to the participating lanes), not a sync event, so it needs a kernel run.

Kind: legacy ``warp_collective_divergence``; v2 surfaces the interpreter's
``ExecErrorKind::Divergence`` as the runtime diagnostic ``divergence``. No
delta row records that rename, so both names are accepted
(``WARP_COLLECTIVE_DIVERGENCE``). Membermask text and source rendering are not
asserted.

No spec covers partial ``shfl.sync`` participation (test-migration.md no-spec
item 17); the module passes since W5-13 (8fba7e1), so it is no longer xfail.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tirx_harness.numsim import v2

from ._runnable import (
    WARP_COLLECTIVE_DIVERGENCE,
    assert_clean,
    assert_error_kind,
    assert_no_incomplete,
    coverage_bounds,
    requires_v2_engine,
    resource_limits,
)

pytestmark = [requires_v2_engine]


@T.prim_func
def native_collective_in_while(mode: T.int32, bound: T.int32):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    iteration = T.alloc_local((1,), "int32")
    iteration[0] = 0
    while iteration[0] < 1:
        if mode == 0:
            _uniform = T.cuda.warp_sum(T.float32(1.0))
        elif mode == 1:
            if lane == 0:
                _lane_zero = T.cuda.warp_sum(T.float32(1.0))
        elif mode == 2:
            if warp == 0:
                _whole_warp = T.cuda.warp_sum(T.float32(1.0))
        elif mode == 3:
            if T.cuda.elect_sync():
                _elected = T.cuda.warp_sum(T.float32(1.0))
        else:
            if lane < bound:
                _width_sixteen = T.cuda.warp_sum(T.float32(1.0), width=16)
        iteration[0] = iteration[0] + 1


@T.prim_func
def native_boolean_guard_warp_collective(
    mode: T.int32, bound: T.int32, flag: T.Buffer((1,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])

    if mode == 0:
        if (flag[0] != 0) and (lane < bound):
            _sum0 = T.cuda.warp_sum(T.float32(1.0))
    elif mode == 1:
        if (flag[0] != 0) & (lane < bound):
            _sum1 = T.cuda.warp_sum(T.float32(1.0))
    elif mode == 2:
        if (flag[0] != 0) | (lane < bound):
            _sum2 = T.cuda.warp_sum(T.float32(1.0))
    else:
        if not ((flag[0] == 0) | (lane >= bound)):
            _sum3 = T.cuda.warp_sum(T.float32(1.0))


def _sync(kernel, inputs, **limits):
    return v2.synccheck(kernel, inputs, coverage_bounds=coverage_bounds(), resource_limits=resource_limits(**limits))


def _run_while(mode: int, bound: int):
    return _sync(native_collective_in_while, {"mode": np.int32(mode), "bound": np.int32(bound)},
                 max_loop_steps=1_000)


def _run_guard(mode: int, flag: int, bound: int):
    return _sync(
        native_boolean_guard_warp_collective,
        {"mode": np.int32(mode), "bound": np.int32(bound), "flag": np.array([flag], dtype=np.int32)},
        max_schedules=16, max_backtrack_nodes=100, max_events_per_run=1_000, max_total_events=10_000,
        max_loop_steps=1_000,
    )


def _assert_divergence(report) -> None:
    assert_error_kind(report, WARP_COLLECTIVE_DIVERGENCE)
    assert_no_incomplete(report)


@pytest.mark.parametrize(
    ("mode", "bound"),
    [
        pytest.param(0, 32, id="uniform"),
        pytest.param(2, 32, id="warp-uniform-guard"),
        pytest.param(4, 32, id="width-sixteen-full-warp-participation"),
    ],
)
def test_synccheck_accepts_full_warp_collectives_inside_while(mode, bound):
    """Replaces ``tests/analysis_tools/synccheck/test_native_kernel_contracts.py::test_native_synccheck_accepts_full_warp_collectives_inside_while`` (all params)."""

    assert_clean(_run_while(mode, bound))


@pytest.mark.parametrize(
    ("mode", "bound"),
    [
        pytest.param(1, 32, id="lane-zero"),
        pytest.param(3, 32, id="elect-sync"),
        pytest.param(4, 16, id="width-sixteen-half-warp-branch"),
    ],
)
def test_synccheck_reports_divergent_collective_inside_while_at_source(mode, bound):
    """Replaces ``tests/analysis_tools/synccheck/test_native_kernel_contracts.py::test_native_synccheck_reports_divergent_collective_inside_while_at_source`` (all params).

    A ``shfl.sync`` reached by a strict subset of the warp is an exact
    ``warp_collective_divergence`` error (v2: ``divergence``). The legacy
    mask/source-text assertions are payload shape (C).
    """

    _assert_divergence(_run_while(mode, bound))


@pytest.mark.parametrize(
    ("mode", "full_flag", "full_bound", "subset_flag", "subset_bound"),
    [
        pytest.param(0, 1, 32, 1, 16, id="logical-and"),
        pytest.param(1, 1, 32, 1, 16, id="boolean-bitwise-and"),
        pytest.param(2, 1, 0, 0, 16, id="boolean-bitwise-or"),
        pytest.param(3, 1, 32, 1, 16, id="not-of-or"),
    ],
)
def test_synccheck_resolves_boolean_collective_participation_exactly(
    mode, full_flag, full_bound, subset_flag, subset_bound
):
    """Replaces ``tests/analysis_tools/synccheck/test_native_synccheck_exact_control.py::test_public_native_synccheck_resolves_boolean_collective_participation_exactly`` (all params).

    The full-warp guard is clean; the half-warp guard (lanes < 16) is a
    ``warp_collective_divergence`` error (v2: ``divergence``).
    """

    assert_clean(_run_guard(mode, full_flag, full_bound))
    _assert_divergence(_run_guard(mode, subset_flag, subset_bound))


@pytest.mark.parametrize("mode", [0, 1, 3])
def test_synccheck_skips_uniformly_false_data_guard(mode):
    """Replaces ``tests/analysis_tools/synccheck/test_native_synccheck_exact_control.py::test_public_native_synccheck_skips_uniformly_false_data_guard`` (all params).

    ``flag == 0`` makes the guard false for every lane: no collective runs, clean.
    """

    assert_clean(_run_guard(mode, 0, 32))
