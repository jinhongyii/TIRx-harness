"""v2 valid-shape copies of the row-L4 tests in ``tests/analysis_tools/racecheck/test_native_tcgen_thread_fence.py``.

Row L4 (``docs/development/numsim-behaviour-deltas.md``; ``numsim-isa-answers.md``
"tcgen05.mma shapes"): the legacy kernels ``tcgen_cp_to_mma_handoff``,
``repeated_tcgen_fence_handoff`` and ``commit_forwards_only_issued_work`` issue
an M=128 N=8 ``tcgen05.mma`` at cta_group::1, which is not a hardware
instruction (M=128 needs N % 16 == 0), so v2 rejects them at transpile time.
These copies are verbatim except for N: the B operand (``right``/``shared_b``)
is 16 x 16 and the accumulator is 128 x 16 (16 TMEM columns, still inside the
32-column allocation; in ``commit_forwards_only_issued_work`` it occupies
columns 16..31 as before). The checker verdicts asserted are the legacy ones.
Public v2 API: ``v2.racecheck(kernel, inputs)`` (no ``cache_dir`` /
``max_workers``).
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import ComposeLayout, R, S, TCol, TileLayout, TLane, tmem_datapath_layout

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

_MMA_F16_32B = ComposeLayout(3, 1, 3, TileLayout(S[(128,)]))
_REPLICATED_CP_LAYOUT = TileLayout(S[(32, 4) : (1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane])


@T.prim_func
def tcgen_cp_to_mma_handoff_n16(
    cp_source: T.Buffer((32, 4), "float32"),
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((16, 16), "float16"),
    flag: T.Buffer((1,), "int32"),
    with_before: T.int32,
    with_after: T.int32,
    handoff: T.int32,
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    observed = T.local_scalar("int32")
    query_ready = T.local_scalar("uint32")
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barriers = T.alloc_buffer((3,), "uint64", scope="shared")
    shared_cp = T.alloc_buffer((32, 4), "float32", scope="shared")
    shared_a = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    shared_b = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    copied = T.decl_buffer(
        (32, 4),
        "float32",
        scope="tmem",
        layout=_REPLICATED_CP_LAYOUT,
        allocated_addr=address[0],
    )
    accumulator = T.decl_buffer(
        (128, 16),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 16),
        allocated_addr=address[0],
    )

    if (warp == 0) and (lane == 0):
        Tx.copy(shared_cp[:, :], cp_source[:, :])
        Tx.copy(shared_a[:, :], left[:, :])
        Tx.copy(shared_b[:, :], right[:, :])
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[2]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if (warp == 0) and (lane == 0):
        Tx.copy_async(
            copied[:, :],
            shared_cp[:, :],
            dispatch="smem->tmem",
            shape="32x128b",
            multicast="warpx4",
        )
        T.ptx.tcgen05.fence__before_thread_sync(pred=with_before != 0)
        if handoff == 1:
            T.ptx.st.relaxed.cta.global_.s32(flag.ptr_to([0]), T.int32(1))
        elif handoff == 4:
            T.ptx.mbarrier.arrive.relaxed.cluster.shared.b64(T.address_of(barriers[2]))
        elif handoff >= 2:
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[2]))
    if handoff == 0:
        T.cuda.cta_sync()

    if (warp == 1) and (lane == 0):
        if handoff == 1:
            observed = T.int32(0)
            T.cuda.wait_until(
                observed, flag.ptr_to([0]), observed != T.int32(0), "cta", "global"
            )
        elif handoff == 2:
            T.cuda.mbarrier_wait(T.address_of(barriers[2]), 0)
        elif handoff >= 3:
            query_ready = T.uint32(0)
            while query_ready == T.uint32(0):
                T.ptx.mbarrier.test_wait.parity.relaxed.cta.shared.b64(
                    query_ready, T.address_of(barriers[2]), T.uint32(0)
                )
        T.ptx.tcgen05.fence__after_thread_sync(pred=with_after != 0)
        Tx.gemm_async(
            accumulator[:, :],
            shared_a[:, :],
            shared_b[:, :],
            accum=False,
        )
    T.cuda.cta_sync()

    if warp == 0:
        if lane == 0:
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barriers[0])
            )
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    if warp == 1:
        if lane == 0:
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barriers[1])
            )
        T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def repeated_tcgen_fence_handoff_n16(
    cp_source: T.Buffer((32, 4), "float32"),
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((16, 16), "float16"),
    rounds: T.int32,
):
    """Repeated full-frontier publication must stay bounded and ordered."""

    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    shared_cp = T.alloc_buffer((32, 4), "float32", scope="shared")
    shared_a = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    shared_b = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    copied = T.decl_buffer(
        (32, 4),
        "float32",
        scope="tmem",
        layout=_REPLICATED_CP_LAYOUT,
        allocated_addr=address[0],
    )
    accumulator = T.decl_buffer(
        (128, 16),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 16),
        allocated_addr=address[0],
    )

    if (warp == 0) and (lane == 0):
        Tx.copy(shared_cp[:, :], cp_source[:, :])
        Tx.copy(shared_a[:, :], left[:, :])
        Tx.copy(shared_b[:, :], right[:, :])
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if (warp == 0) and (lane == 0):
        Tx.copy_async(
            copied[:, :],
            shared_cp[:, :],
            dispatch="smem->tmem",
            shape="32x128b",
            multicast="warpx4",
        )
    for _ in T.serial(rounds):
        if (warp == 0) and (lane == 0):
            T.ptx.tcgen05.fence__before_thread_sync()
        T.cuda.cta_sync()
        if (warp == 1) and (lane == 0):
            T.ptx.tcgen05.fence__after_thread_sync()
        T.cuda.cta_sync()

    if (warp == 1) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            shared_a[:, :],
            shared_b[:, :],
            accum=False,
        )
    T.cuda.cta_sync()

    if warp == 0:
        if lane == 0:
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barriers[0])
            )
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    if warp == 1:
        if lane == 0:
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barriers[1])
            )
        T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def commit_forwards_only_issued_work_n16(
    first_source: T.Buffer((32, 4), "float32"),
    second_source: T.Buffer((32, 4), "float32"),
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((16, 16), "float16"),
    with_local_work: T.int32,
):
    """Commit publishes local work and its causes, but not bare imported work."""

    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barriers = T.alloc_buffer((4,), "uint64", scope="shared")
    first_shared = T.alloc_buffer((32, 4), "float32", scope="shared")
    second_shared = T.alloc_buffer((32, 4), "float32", scope="shared")
    shared_a = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    shared_b = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    copied = T.decl_buffer(
        (32, 4),
        "float32",
        scope="tmem",
        layout=_REPLICATED_CP_LAYOUT,
        allocated_addr=address[0],
    )
    accumulator = T.decl_buffer(
        (128, 16),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 16),
        allocated_addr=address[0] + T.uint32(16),
    )

    if (warp == 0) and (lane == 0):
        Tx.copy(first_shared[:, :], first_source[:, :])
        Tx.copy(second_shared[:, :], second_source[:, :])
        Tx.copy(shared_a[:, :], left[:, :])
        Tx.copy(shared_b[:, :], right[:, :])
        for i in T.unroll(4):
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[i]), 1)
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if (warp == 0) and (lane == 0):
        Tx.copy_async(
            copied[:, :],
            first_shared[:, :],
            dispatch="smem->tmem",
            shape="32x128b",
            multicast="warpx4",
        )
        T.ptx.tcgen05.fence__before_thread_sync()
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[2])
        )

    if warp == 1:
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        if lane == 0:
            T.ptx.tcgen05.fence__after_thread_sync()
            if with_local_work != 0:
                Tx.gemm_async(
                    accumulator[:, :],
                    shared_a[:, :],
                    shared_b[:, :],
                    accum=False,
                )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barriers[1])
            )

    if warp == 2:
        T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
        if lane == 0:
            T.ptx.tcgen05.fence__after_thread_sync()
            Tx.copy_async(
                copied[:, :],
                second_shared[:, :],
                dispatch="smem->tmem",
                shape="32x128b",
                multicast="warpx4",
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barriers[3])
            )

    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(barriers[2]), 0)
    if warp == 2:
        T.cuda.mbarrier_wait(T.address_of(barriers[3]), 0)
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


def _run(*, with_before: bool, with_after: bool, handoff: int):
    return v2.racecheck(
        tcgen_cp_to_mma_handoff_n16,
        {
            "cp_source": np.zeros((32, 4), dtype=np.float32),
            "left": np.zeros((128, 16), dtype=np.float16),
            "right": np.zeros((16, 16), dtype=np.float16),
            "flag": np.zeros((1,), dtype=np.int32),
            "with_before": np.int32(with_before),
            "with_after": np.int32(with_after),
            "handoff": np.int32(handoff),
        },
    )


@pytest.mark.parametrize(
    "handoff",
    [0, 1, 2, 3, 4],
    ids=["cta_barrier", "relaxed_flag", "mbarrier", "relaxed_wait", "relaxed_arrive_wait"],
)
def test_cross_thread_cp_to_mma_requires_both_thread_fences(handoff: int) -> None:
    """Replaces ``tests/analysis_tools/racecheck/test_native_tcgen_thread_fence.py::test_cross_thread_cp_to_mma_requires_both_thread_fences`` (all five params).

    Row L4: ``tcgen_cp_to_mma_handoff`` issued an M=128 N=8 MMA; the copy
    uses N=16. Both fences: clean; either one missing: not clean."""

    _run(with_before=True, with_after=True, handoff=handoff).require_clean()
    for missing in ("after", "before"):
        report = _run(
            with_before=missing != "before",
            with_after=missing != "after",
            handoff=handoff,
        )
        assert report.verdict != "clean", (missing, report.format())


def _commit_inputs(with_local_work: int):
    return {
        "first_source": np.zeros((32, 4), dtype=np.float32),
        "second_source": np.ones((32, 4), dtype=np.float32),
        "left": np.zeros((128, 16), dtype=np.float16),
        "right": np.zeros((16, 16), dtype=np.float16),
        "with_local_work": np.int32(with_local_work),
    }


def test_empty_commit_does_not_republish_tcgen_imported_from_another_thread() -> None:
    """Replaces ``tests/analysis_tools/racecheck/test_native_tcgen_thread_fence.py::test_empty_commit_does_not_republish_tcgen_imported_from_another_thread``.

    Row L4: ``commit_forwards_only_issued_work`` declared an M=128 N=8 MMA;
    the copy uses N=16 (the MMA is not issued here, ``with_local_work=0``)."""

    report = v2.racecheck(commit_forwards_only_issued_work_n16, _commit_inputs(0))
    assert report.verdict != "clean", report.format()


def test_commit_republishes_causal_predecessors_of_local_work() -> None:
    """Replaces ``tests/analysis_tools/racecheck/test_native_tcgen_thread_fence.py::test_commit_republishes_causal_predecessors_of_local_work``.

    Row L4: ``commit_forwards_only_issued_work`` issued an M=128 N=8 MMA; the
    copy issues M=128 N=16."""

    v2.racecheck(commit_forwards_only_issued_work_n16, _commit_inputs(1)).require_clean()


def test_repeated_thread_fence_handoff_keeps_full_frontier_ordered() -> None:
    """Replaces ``tests/analysis_tools/racecheck/test_native_tcgen_thread_fence.py::test_repeated_thread_fence_handoff_keeps_full_frontier_ordered``.

    Row L4: ``repeated_tcgen_fence_handoff`` issued an M=128 N=8 MMA; the copy
    uses N=16. 128 fence rounds, then the MMA: clean."""

    report = v2.racecheck(
        repeated_tcgen_fence_handoff_n16,
        {
            "cp_source": np.zeros((32, 4), dtype=np.float32),
            "left": np.zeros((128, 16), dtype=np.float16),
            "right": np.zeros((16, 16), dtype=np.float16),
            "rounds": np.int32(128),
        },
    )
    report.require_clean()
