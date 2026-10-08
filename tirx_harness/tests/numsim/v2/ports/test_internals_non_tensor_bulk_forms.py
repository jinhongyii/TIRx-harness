"""v2 ports of the legacy non-tensor bulk-form tests that also pinned the
resolved PTX op-name set (``call_op_names(module.spec.kernels[0])``); the
op-name pins are dropped."""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def bulk_g2s_cluster_static_true_predicate(
    source: T.Buffer((16,), "uint8"), output: T.Buffer((16,), "uint8")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
        T.ptx["cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"](
            shared.ptr_to([0]),
            source.ptr_to([0]),
            T.uint32(16),
            T.cuda.cvta_generic_to_shared(barrier.ptr_to([0])),
            pred=T.bool(True),
        )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.warp_sync()
    if lane < 16:
        output[lane] = shared[lane]


@T.prim_func
def raw_bulk_prefetch(
    source: T.Buffer((64,), "uint8"), num_bytes: T.uint32, output: T.Buffer((32,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx["cp.async.bulk.prefetch.L2.global"](source.ptr_to([16]), num_bytes)
    output[lane] = source[lane + 16]



def test_bulk_g2s_cluster_accepts_static_true_predicate_as_unconditional():
    """Port of ``tests/numsim/runtime/test_non_tensor_bulk_forms.py::test_bulk_g2s_cluster_accepts_static_true_predicate_as_unconditional``.

    Dropped pin: ``"tirx.ptx.cp_async_bulk_g2s_cluster" in
    call_op_names(module.spec.kernels[0])`` (legacy resolver op name).
    """

    source = np.arange(16, dtype=np.uint8) ^ np.uint8(0xA5)
    module = v2.transpile(bulk_g2s_cluster_static_true_predicate)
    result = v2.Engine().run(module, {"source": source, "output": np.zeros(16, dtype=np.uint8)})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_raw_bulk_prefetch_preserves_global_memory():
    """Port of ``tests/numsim/runtime/test_non_tensor_bulk_forms.py::test_raw_bulk_prefetch_preserves_global_memory``.

    Dropped pin: ``"tirx.ptx.cp_async_bulk_prefetch" in
    call_op_names(module.spec.kernels[0])`` (legacy resolver op name).
    """

    source = np.arange(64, dtype=np.uint8) ^ np.uint8(0xA5)
    module = v2.transpile(raw_bulk_prefetch)
    result = v2.Engine().run(
        module,
        {"source": source, "num_bytes": np.uint32(32), "output": np.zeros(32, dtype=np.uint8)},
    )

    np.testing.assert_array_equal(result.outputs["output"], source[16:48])
    assert result.verdict == "clean"
    assert result.diagnostics == []
