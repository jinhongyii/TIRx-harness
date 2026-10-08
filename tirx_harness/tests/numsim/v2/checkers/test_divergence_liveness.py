"""Lane liveness after divergence, and TMEM access after a cross-warp dealloc.

Replaces two ``gap_unportable`` synccheck rows:
``tests/analysis_tools/synccheck/test_native_break_continue.py::test_public_native_lane_divergent_break_is_exact_collective_error``
and the unordered half of
``tests/analysis_tools/synccheck/test_native_kernel_contracts.py::test_native_synccheck_requires_cross_warp_tmem_quiescence_before_dealloc``
(the ordered half is also ported as
``synccheck_legacy_ports::cross_warp_tmem_dealloc_after_gate_is_clean``).
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tirx_harness.numsim import v2

from ._runnable import (
    WARP_COLLECTIVE_DIVERGENCE,
    assert_clean,
    assert_error_kind,
    assert_no_incomplete,
    coverage_bounds,
    no_spec,
    requires_v2_engine,
    resource_limits,
)

pytestmark = requires_v2_engine


@T.prim_func
def native_lane_divergent_break():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    iteration = T.local_scalar("int32")
    iteration = 0

    while iteration < 6:
        T.ptx.bar.sync(T.uint32(13), T.uint32(32))
        iteration = iteration + 1
        if (lane == 0) and (iteration >= 3):
            break


@T.prim_func
def native_cross_warp_tmem_dealloc(ordered: T.int32):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    gate = T.alloc_buffer((1,), "uint64", scope="shared")
    value = T.alloc_local((1,), "uint32")

    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(gate[0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    T.cuda.cta_sync()

    if ordered != 0:
        if warp == 1:
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](value[0], address[0])
            T.ptx.tcgen05.wait__ld.sync.aligned()
            if lane == 0:
                T.ptx.mbarrier.arrive.shared.b64(T.address_of(gate[0]))
        elif warp == 0:
            if lane == 0:
                T.cuda.mbarrier_wait(T.address_of(gate[0]), 0)
            T.cuda.warp_sync()
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
    else:
        if warp == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
            if lane == 0:
                T.ptx.mbarrier.arrive.shared.b64(T.address_of(gate[0]))
        elif warp == 1:
            if lane == 0:
                T.cuda.mbarrier_wait(T.address_of(gate[0]), 0)
            T.cuda.warp_sync()
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](value[0], address[0])
            T.ptx.tcgen05.wait__ld.sync.aligned()


def _sync(kernel, inputs, **limits):
    return v2.synccheck(kernel, inputs, coverage_bounds=coverage_bounds(), resource_limits=resource_limits(**limits))


# Delta B1 (sync-behaviour-deltas.md): a barrier executed by a strict subset of
# the warp's non-exited lanes is ``PartialWarp`` for every form; for a named
# barrier its kind is ``named_barrier_invalid_arrival_count``.
_PARTIAL_WARP = WARP_COLLECTIVE_DIVERGENCE | {"named_barrier_invalid_arrival_count", "barrier_mismatch"}


@no_spec(15, "liveness at a later named barrier of lanes that broke out of a loop but have not exited")
def test_lane_divergent_break_is_exact_collective_error():
    """Replaces ``tests/analysis_tools/synccheck/test_native_break_continue.py::test_public_native_lane_divergent_break_is_exact_collective_error``.

    Lane 0 breaks out after the third ``bar.sync 13, 32`` while lanes 1..31
    run a fourth: legacy ``warp_collective_divergence`` (mask 0xfffffffe).
    Delta B1 makes a strict subset of non-exited lanes ``PartialWarp``
    (``named_barrier_invalid_arrival_count``) -- the expectation is an error
    whose kind is either spelling. Whether lane 0 still counts as live is
    exactly the unspecified point.
    """

    report = _sync(native_lane_divergent_break, {})
    assert_error_kind(report, _PARTIAL_WARP)
    assert_no_incomplete(report)


@no_spec(16, "TMEM access after an unordered cross-warp dealloc (legacy synchronization_collective_publication)")
def test_synccheck_requires_cross_warp_tmem_quiescence_before_dealloc():
    """Replaces ``tests/analysis_tools/synccheck/test_native_kernel_contracts.py::test_native_synccheck_requires_cross_warp_tmem_quiescence_before_dealloc`` (unordered half; the ordered half is re-checked here as the control).

    Ordered (warp 1's ``tcgen05.ld`` drained before warp 0 deallocates): clean.
    Unordered (warp 1 loads after the dealloc): error, legacy kind
    ``synchronization_collective_publication`` ("not covered by any live
    allocation"). No new doc names this check, so any error kind is accepted
    alongside the legacy one. (Observed in v2: the ordered half draws a
    ``review`` ``uninitialized_read`` advisory for the fresh TMEM load.)
    """

    assert_clean(_sync(native_cross_warp_tmem_dealloc, {"ordered": np.int32(1)}, max_loop_steps=1_000))
    unordered = _sync(native_cross_warp_tmem_dealloc, {"ordered": np.int32(0)}, max_loop_steps=1_000)
    assert unordered.verdict == "error", unordered.format()
