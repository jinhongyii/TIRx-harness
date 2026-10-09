"""v2 copy of ``tests/numsim/runtime/test_non_tensor_bulk_forms.py::
test_raw_bulk_s2c_preserves_mapped_remote_cta_ownership`` with racecheck delta
**B7**.

CTA 0 announces 16 bytes with a qualifier-less
``mbarrier.arrive.expect_tx.shared::cluster`` on CTA 1's barrier (``mapa``-ed
u32 address) and bulk-copies its ``source`` into CTA 1's ``destination``
(the ``pred=False`` copy issues nothing). Legacy required both checkers
clean. Under B7 the arrive defaults to ``.release.cta`` and does not include
CTA 1's ``mbarrier_wait``: Racecheck is exactly one ``scope_mismatch``
(release warp 0 -> acquire warp 1). No follow-on race: the copied bytes reach
CTA 1 through the copy's own complete-tx (delta B3) and ``cluster_sync``
orders the final reads. Synccheck clean and the output are unchanged. The
kernel is copied verbatim from the legacy file.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

from ._racedeltas import assert_b7_scope_mismatch

pytestmark = requires_v2_engine


@T.prim_func
def raw_bulk_s2c_uses_mapped_u32_addresses(output: T.Buffer((16,), "uint8")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source = T.alloc_buffer((16,), "uint8", scope="shared", align=128)
    destination = T.alloc_buffer((16,), "uint8", scope="shared", align=128)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 1) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    if (cta == 0) and (lane < 16):
        source[lane] = T.cast(lane + 19, "uint8")
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        remote_barrier = T.alloc_local((1,), "uint32")
        T.ptx.mapa.shared__cluster.u32(
            remote_barrier[0],
            T.cuda.cvta_generic_to_shared(barrier.ptr_to([0])),
            T.uint32(1),
        )
        remote_destination = T.alloc_local((1,), "uint32")
        T.ptx.mapa.shared__cluster.u32(
            remote_destination[0],
            T.cuda.cvta_generic_to_shared(destination.ptr_to([0])),
            T.uint32(1),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(
            remote_barrier[0], T.uint32(16), pred=True
        )
        T.ptx["cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"](
            remote_destination[0],
            source.ptr_to([0]),
            T.uint32(16),
            remote_barrier[0],
            pred=False,
        )
        T.ptx["cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"](
            remote_destination[0],
            source.ptr_to([0]),
            T.uint32(16),
            remote_barrier[0],
            pred=lane == 0,
        )
    if (cta == 1) and (lane == 0):
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.cluster_sync()
    if (cta == 1) and (lane < 16):
        output[lane] = destination[lane]


def test_raw_bulk_s2c_preserves_mapped_remote_cta_ownership(tmp_path):
    """Replaces ``tests/numsim/runtime/test_non_tensor_bulk_forms.py::test_raw_bulk_s2c_preserves_mapped_remote_cta_ownership``.

    Synccheck clean; Racecheck racecheck delta B7 (one ``scope_mismatch``);
    NumSim reads back ``19..34`` in CTA 1.
    """

    def args():
        return {"output": np.zeros(16, np.uint8)}

    v2.synccheck(raw_bulk_s2c_uses_mapped_u32_addresses, args()).require_clean()
    assert_b7_scope_mismatch(v2.racecheck(raw_bulk_s2c_uses_mapped_u32_addresses, args()), acquire_warps={1}, count=1)
    module = v2.transpile(raw_bulk_s2c_uses_mapped_u32_addresses, cache_dir=tmp_path)
    result = v2.Engine().run(module, args())
    np.testing.assert_array_equal(result.outputs["output"], np.arange(19, 35, dtype=np.uint8))
