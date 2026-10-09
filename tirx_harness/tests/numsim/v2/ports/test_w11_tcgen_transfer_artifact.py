"""v2 copies of three ``tests/numsim/integration/test_tcgen_transfer_artifact.py`` tests
(public-API ``other-assertion`` triage, W11):

- ``test_tcgen_cp_expands_tlane_replicas``
- ``test_tcgen_cp_supports_rank3_multi_instruction_layout``
- ``test_tcgen_cp_cta_group2_supports_float16_payloads``

**Delta (numsim-behaviour-deltas T20): the legacy kernels read TMEM before the
``tcgen05.cp`` completes.** Each legacy kernel issues ``Tx.copy_async(tmem, smem)``
(TVM dispatches it to ``tcgen05.cp``; the replicated TMEM view ``R[4 : 32 @ TLane]``
becomes ``.warpx4`` multicast) and then reads the destination through a physical
TMEM view after a plain ``cta_sync`` / ``cluster_sync``, with no
``tcgen05.commit`` + ``mbarrier`` wait and no ``fence.proxy.async`` between the
generic shared-memory writes and the copy's shared-memory read. PTX ISA
(tcgen05 memory consistency model): ``tcgen05.cp`` is asynchronous and its
completion is observed only through ``tcgen05.commit`` and a wait on the
committed mbarrier. Legacy executed the copy synchronously at issue; v2 models
it as an async op (``Payload::TcgenCp``) landed only when ordered, so the reads
see unwritten TMEM (zeros plus ``uninitialized_read`` reviews), and racecheck
reports ``data_race`` (racecheck-behaviour-deltas P6 for the never-landed op).
The W1 note "out of scope (replicated TMEM views, contract item 29)" no longer
applies: v2 lowers these kernels (only direct ``BufferLoad``/``BufferStore`` on a
replicated view is rejected, numsim-behaviour-deltas L1).

Each copy (1) runs the legacy kernel verbatim and asserts the racecheck race and
the NumSim ``uninitialized_read`` review on TMEM, and (2) runs a corrected kernel
(``fence.proxy.async`` + ``tcgen05.fence`` around the sync, ``tcgen05.commit`` to
an mbarrier right after the copy, every thread waits, then
``tcgen05.fence::after_thread_sync``) and asserts the legacy expected values
exactly and a clean racecheck. The data-path facts the legacy tests pinned
(``warpx4`` lane replication, rank-3 multi-instruction layouts, ``cta_group::2``
f16 payloads into both CTAs) are all kept.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import R, S, TCol, TileLayout, TLane, tmem_datapath_layout

from tests.numsim.support.kernels import (
    tcgen_float16_cta_group2,
    tcgen_shared_to_tmem_rank3,
    tcgen_shared_to_tmem_replica,
)
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


def _assert_legacy_kernel_reads_before_completion(kernel, inputs):
    """The verbatim legacy kernel: racecheck error with a data race, and NumSim
    reads unwritten TMEM (``uninitialized_read`` review on ``tmem``)."""

    report = v2.racecheck(kernel, dict(inputs))
    assert report.verdict == "error", report.format()
    assert ("error", "data_race") in {(f.status, f.kind) for f in report.findings}, report.format()
    result = v2.Engine().run(v2.transpile(kernel), dict(inputs))
    assert any(
        d["kind"] == "uninitialized_read" and d.get("space") == "tmem" for d in result.diagnostics
    )


def _run_clean(kernel, inputs):
    report = v2.racecheck(kernel, dict(inputs))
    assert report.verdict == "clean", report.format()
    return v2.Engine().run(v2.transpile(kernel), dict(inputs))


# -- corrected kernels: the legacy data path plus commit/wait and fences -----


@T.prim_func
def synced_shared_to_tmem_replica(
    source: T.Buffer((32, 4), "float32"), output: T.Buffer((128, 4), "float32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32, 4), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    replicated = T.decl_buffer(
        (32, 4),
        "float32",
        scope="tmem",
        layout=TileLayout(S[(32, 4) : (1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]),
        allocated_addr=0,
    )
    physical = T.decl_buffer(
        (128, 4),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 4),
        allocated_addr=0,
    )
    if warp == 0:
        for col in T.serial(4):
            shared[lane, col] = source[lane, col]
    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()
    if warp == 0 and lane == 0:
        Tx.copy_async(
            replicated[:, :],
            shared[:, :],
            dispatch="smem->tmem",
            shape="32x128b",
            multicast="warpx4",
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(4):
        output[row, col] = physical[row, col]


@T.prim_func
def synced_shared_to_tmem_rank3(
    source: T.Buffer((4, 32, 16), "uint8"),
    output: T.Buffer((128, 4, 16), "uint8"),
):
    T.device_entry()
    _cta = T.cta_id([1])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer(
        (4, 32, 16),
        "uint8",
        scope="shared",
        layout=TileLayout(S[(4, 32, 16) : (512, 16, 1)]),
    )
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    replicated = T.decl_buffer(
        (4, 32, 16),
        "uint8",
        scope="tmem",
        layout=TileLayout(S[(4, 32, 16) : (16 @ TCol, 1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]),
        allocated_addr=0,
    )
    physical = T.decl_buffer(
        (128, 64),
        "uint8",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 64),
        allocated_addr=0,
    )
    for col in T.serial(16):
        shared[warp, lane, col] = source[warp, lane, col]
    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()
    if (warp == 0) and (lane == 0):
        Tx.copy_async(replicated[:, :, :], shared[:, :, :], dispatch="smem->tmem")
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    physical_row = T.meta_var(warp * 32 + lane)
    for outer in T.serial(4):
        for col in T.serial(16):
            output[physical_row, outer, col] = physical[physical_row, outer * 16 + col]


@T.prim_func
def synced_float16_cta_group2(
    source: T.Buffer((2, 32, 8), "float16"),
    output: T.Buffer((2, 128, 8), "float16"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32, 8), "float16", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    replicated = T.decl_buffer(
        (32, 8),
        "float16",
        scope="tmem",
        layout=TileLayout(S[(32, 8) : (1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]),
        allocated_addr=0,
    )
    physical = T.decl_buffer(
        (128, 8),
        "float16",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    if warp == 0:
        for col in T.serial(8):
            shared[lane, col] = source[cta, lane, col]
    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cluster_sync()
    T.ptx.tcgen05.fence__after_thread_sync()
    if (cta == 0) and (warp == 0) and (lane == 0):
        Tx.copy_async(
            replicated[:, :],
            shared[:, :],
            dispatch="smem->tmem",
            cta_group=2,
        )
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(barrier[0]), T.uint16(3)
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(8):
        output[cta, row, col] = physical[row, col]
    T.cuda.cluster_sync()


# -- tests -------------------------------------------------------------------


def test_tcgen_cp_expands_tlane_replicas():
    """Delta copy; see the module docstring (un-waited async ``tcgen05.cp``)."""
    source = np.arange(32 * 4, dtype=np.float32).reshape(32, 4)
    inputs = {"source": source, "output": np.zeros((128, 4), dtype=np.float32)}
    _assert_legacy_kernel_reads_before_completion(tcgen_shared_to_tmem_replica, inputs)
    result = _run_clean(synced_shared_to_tmem_replica, inputs)
    np.testing.assert_array_equal(result.outputs["output"], np.tile(source, (4, 1)))


def test_tcgen_cp_supports_rank3_multi_instruction_layout():
    """Delta copy; see the module docstring (un-waited async ``tcgen05.cp``)."""
    source = np.arange(4 * 32 * 16, dtype=np.uint8).reshape(4, 32, 16)
    inputs = {"source": source, "output": np.zeros((128, 4, 16), dtype=np.uint8)}
    _assert_legacy_kernel_reads_before_completion(tcgen_shared_to_tmem_rank3, inputs)
    result = _run_clean(synced_shared_to_tmem_rank3, inputs)
    expected = np.tile(source.transpose(1, 0, 2), (4, 1, 1))
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_tcgen_cp_cta_group2_supports_float16_payloads():
    """Delta copy; see the module docstring (un-waited async ``tcgen05.cp``)."""
    source = np.linspace(-5, 7, 2 * 32 * 8, dtype=np.float16).reshape(2, 32, 8)
    inputs = {"source": source, "output": np.zeros((2, 128, 8), dtype=np.float16)}
    _assert_legacy_kernel_reads_before_completion(tcgen_float16_cta_group2, inputs)
    result = _run_clean(synced_float16_cta_group2, inputs)
    np.testing.assert_array_equal(result.outputs["output"], np.tile(source, (1, 4, 1)))
