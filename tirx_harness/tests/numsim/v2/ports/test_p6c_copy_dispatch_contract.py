"""v2 port of ``tests/numsim/registry/test_copy_dispatch_contract.py::test_cross_warp_copy_owner_remap_is_accepted_statically``.

Legacy ``verify(analyze(kernel))`` (static acceptance) becomes
``v2.transpile(kernel)`` succeeding. The kernel is copied verbatim.
"""

from __future__ import annotations

from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import wg_local_layout

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def _cross_warp_owner_remap():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    source_storage: T.f32[2]
    destination_storage: T.f32[2]
    source_view = source_storage.view(128, 2, layout=wg_local_layout(2))
    destination_view = destination_storage.view(128, 2, layout=wg_local_layout(2))
    Tx.wg.copy(destination_view[64:128, :], source_view[0:64, :])


def test_cross_warp_copy_owner_remap_is_accepted_statically():
    """Port of ``tests/numsim/registry/test_copy_dispatch_contract.py::test_cross_warp_copy_owner_remap_is_accepted_statically``."""

    module = v2.transpile(_cross_warp_owner_remap)
    assert len(module.spec.kernels) == 1
