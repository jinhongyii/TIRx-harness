"""v2 copy of ``tests/numsim/integration/test_topology_artifact.py::test_artifact_launch_rejects_topologies_outside_engine_representation``.

Phase change (W1): v2 rejects a topology the engine cannot represent at
``transpile`` with ``UnsupportedTIRxError`` ("1056 threads (33 warps) per CTA,
maximum 1024 (32 warps)", "65 CTAs per cluster, maximum 64"); legacy raised
``NumSimBuildError`` when the artifact was launched. Kernels copied verbatim.
"""

from __future__ import annotations

import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine


@T.prim_func
def too_many_warps(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([33])
    lane = T.lane_id([32])
    if (warp == 0) and (lane == 0):
        output[0] = 1


@T.prim_func
def too_many_cluster_ctas(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([65])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if (cta == 0) and (lane == 0):
        output[0] = 1


def test_artifact_launch_rejects_topologies_outside_engine_representation():
    with pytest.raises(UnsupportedTIRxError, match=r"33 warps.*maximum 1024|maximum 1024.*32 warps"):
        v2.transpile(too_many_warps)
    with pytest.raises(UnsupportedTIRxError, match=r"65 CTAs per cluster, maximum 64"):
        v2.transpile(too_many_cluster_ctas)
