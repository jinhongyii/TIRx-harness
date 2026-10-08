"""Engine-level checker behaviour: subset runs, ``.ignore_oob`` read clipping,
conditional TMEM lifecycles, and a scheduler deadlock.

Replaces four ``gap_unportable`` rows:
``racecheck/test_native_racecheck_artifact.py::test_native_racecheck_subset_is_typed_incomplete``,
``racecheck/test_native_raw_async_copy_footprints.py::test_bulk_g2s_cta_ignore_oob_does_not_bounds_check_the_ignored_bytes``,
``racecheck/test_native_kernel_contracts.py::test_racecheck_accepts_conditional_tmem_lifecycles`` and
``racecheck/test_signal_diagnostics.py::test_wait_without_any_possible_publisher_is_a_sync_deadlock``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness.numsim import v2

from ._runnable import DEADLOCK, assert_clean, assert_error_kind, requires_v2_engine, v2_gap

pytestmark = requires_v2_engine

_BULK_G2S_CTA_IGNORE_OOB = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.ignore_oob"


@T.prim_func
def native_racecheck_two_ctas():
    T.device_entry()
    _cta = T.cta_id([2])
    _warp = T.warp_id([1])


@T.prim_func
def bulk_g2s_cta_ignore_oob_short_source(
    source: T.Buffer((13,), "uint8"), output: T.Buffer((13,), "uint8")
):
    """`.ignore_oob` whose window reaches past the end of its source binding."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 16)
        T.ptx[_BULK_G2S_CTA_IGNORE_OOB](
            shared.ptr_to([0]),
            source.ptr_to([0]),
            16,
            T.uint32(0),
            T.uint32(3),
            barrier.ptr_to([0]),
        )
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        for element in T.serial(13):
            output[element] = shared[element]


@T.prim_func
def native_conditional_tcgen_alloc():
    T.device_entry()
    warp = T.warp_id([9])
    _lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")

    if warp == 8:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def native_rank_conditional_tmem_pool():
    T.device_entry()
    T.cta_id([2])
    cluster_rank = T.cta_id_in_cluster([2])
    T.warpgroup_id([1])

    pool = T.SMEMPool()
    tmem_address = pool.alloc((1,), "uint32", align=4)
    pool.commit()

    rank0_pool = T.TMEMPool(
        pool, total_cols=64, cta_group=1, alloc_warp=0, dealloc_warp=0, tmem_addr=tmem_address,
    )
    rank0_pool.alloc_tcgen05_mma_D((64, 64), "float32", M=64, cta_group=1)
    rank1_pool = T.TMEMPool(
        pool, total_cols=512, cta_group=1, alloc_warp=0, dealloc_warp=0, tmem_addr=tmem_address,
    )
    rank1_pool.alloc_tcgen05_mma_D((128, 128), "float32", M=128, cta_group=1)

    if cluster_rank == 0:
        rank0_pool.commit()
    else:
        rank1_pool.commit()
    T.cuda.cluster_sync()
    if cluster_rank == 0:
        rank0_pool.dealloc()
    else:
        rank1_pool.dealloc()


@T.prim_func
def native_thread_topology_conditional_tmem_pool():
    T.device_entry()
    T.cta_id([2])
    cluster_rank = T.cta_id_in_cluster([2])
    T.thread_id([64])

    pool = T.SMEMPool()
    tmem_address = pool.alloc((1,), "uint32", align=4)
    pool.commit()

    rank0_pool = T.TMEMPool(
        pool, total_cols=64, cta_group=1, alloc_warp=0, dealloc_warp=0, tmem_addr=tmem_address,
    )
    rank0_pool.alloc_tcgen05_mma_D((64, 64), "float32", M=64, cta_group=1)
    rank1_pool = T.TMEMPool(
        pool, total_cols=128, cta_group=1, alloc_warp=0, dealloc_warp=0, tmem_addr=tmem_address,
    )
    rank1_pool.alloc_tcgen05_mma_D((128, 128), "float32", M=128, cta_group=1)

    if cluster_rank == 0:
        rank0_pool.commit()
    else:
        rank1_pool.commit()
    T.cuda.cluster_sync()
    if cluster_rank == 0:
        rank0_pool.dealloc()
    else:
        rank1_pool.dealloc()


_BLOCKED_WAIT = """
@T.prim_func
def blocked_wait(flag: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    if lane == 0:
        T.cuda.wait_until(seen[0], flag.ptr_to([0]), seen[0] == 7, "gpu", "global")
"""


def test_racecheck_subset_is_typed_incomplete():
    """Replaces ``tests/analysis_tools/racecheck/test_native_racecheck_artifact.py::test_native_racecheck_subset_is_typed_incomplete``.

    Running a strict subset of the launch can never support ``clean``: the
    verdict is ``incomplete`` with reason ``subset_execution``
    (racecheck-semantics.md). v2 selects subsets by cluster only
    (``cta_ids`` raises ``NotImplementedError``); with the default 1-CTA
    cluster, cluster 0 is the legacy ``cta_ids=[0]``. v2 has no own
    ``ExecutionSubset`` type, so a duck-typed one is passed. The legacy
    ``analysis_scope`` warp counts are payload shape (C).
    """

    module = v2.transpile(native_racecheck_two_ctas)
    subset = SimpleNamespace(cluster_ids=[0], cta_ids=None)
    result = v2.Engine().run_racecheck_phase(module, {}, subset=subset)
    report = v2.RaceReport([result])
    assert report.verdict == "incomplete", report.format()
    incomplete = [f for f in report.findings if f.status == "incomplete"]
    assert any(
        f.details.get("reason") == "subset_execution" or f.kind == "subset_execution" for f in incomplete
    ), report.format()


def test_bulk_g2s_cta_ignore_oob_does_not_bounds_check_the_ignored_bytes():
    """Replaces ``tests/analysis_tools/racecheck/test_native_raw_async_copy_footprints.py::test_bulk_g2s_cta_ignore_oob_does_not_bounds_check_the_ignored_bytes``.

    A 16-byte ``.ignore_oob`` window over a 13-byte source (3 ignored bytes):
    the read footprint must be the in-bounds slice, so Racecheck is clean (no
    ``out_of_bounds``, no execution error).
    """

    report = v2.racecheck(
        bulk_g2s_cta_ignore_oob_short_source,
        {"source": np.arange(13, dtype=np.uint8), "output": np.zeros(13, dtype=np.uint8)},
    )
    assert_clean(report)


@pytest.mark.parametrize(
    "kernel",
    [
        pytest.param(
            native_conditional_tcgen_alloc,
            id="warp-conditional-alloc",
        ),
        pytest.param(native_rank_conditional_tmem_pool, id="rank-conditional-pool"),
        pytest.param(native_thread_topology_conditional_tmem_pool, id="thread-topology-conditional-pool"),
    ],
)
def test_racecheck_accepts_conditional_tmem_lifecycles(kernel):
    """Replaces ``tests/analysis_tools/racecheck/test_native_kernel_contracts.py::test_racecheck_accepts_conditional_tmem_lifecycles`` (its three kernels as params).

    ``tcgen05.alloc/dealloc/relinquish`` and ``TMEMPool`` under rank-, warp- and
    thread-conditional control flow: racecheck sees only ``AllocBegin``/
    ``AllocEnd`` (racecheck-semantics.md table row 33) and is clean.
    """

    assert_clean(v2.racecheck(kernel, {}))


@v2_gap(
    "a single-lane wait_until that can never succeed is reported as incomplete "
    "'divergent_block: no progress while warp 0 is blocked with a divergent mask', not deadlock "
    "(and the synccheck payload says verdict=error with only that incomplete entry)"
)
@pytest.mark.parametrize("checker", ["synccheck", "racecheck"])
def test_wait_without_any_possible_publisher_is_a_sync_deadlock(checker):
    """Replaces ``tests/analysis_tools/racecheck/test_signal_diagnostics.py::test_wait_without_any_possible_publisher_is_a_sync_deadlock`` (both params).

    ``wait_until(flag == 7)`` with nobody ever writing ``flag``: error
    ``deadlock`` from either checker, and every finding is an error.
    """

    kernel = tvm.script.from_source(_BLOCKED_WAIT, {"T": T})
    report = getattr(v2, checker)(kernel, {"flag": np.zeros(1, dtype=np.int32)})
    assert_error_kind(report, DEADLOCK)
    assert all(f.status == "error" for f in report.findings), report.format()
