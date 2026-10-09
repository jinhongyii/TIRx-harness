"""v2 port of ``tests/numsim/integration/test_topology_artifact.py::test_cta_pair_uses_two_cta_residue_without_reducing_cluster_topology``."""

from __future__ import annotations

import math

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def cluster_cta_and_pair_coordinates(output: T.Buffer((8, 2), "int32")):
    T.device_entry()
    cta = T.cta_id_in_cluster([8])
    pair = T.cta_id_in_pair()
    lane = T.lane_id([32])
    if lane == 0:
        output[cta, 0] = cta
        output[cta, 1] = pair


def test_cta_pair_uses_two_cta_residue_without_reducing_cluster_topology():
    """Port of ``tests/numsim/integration/test_topology_artifact.py::test_cta_pair_uses_two_cta_residue_without_reducing_cluster_topology``.

    ``analyze(...).topology`` -> the v2 document topology (grid / cluster
    CTA counts). Dropped: the two ``module.rust_source`` text pins."""

    module = v2.transpile(cluster_cta_and_pair_coordinates)
    topology = module.document["kernels"][0]["topology"]
    ctas = math.prod(dim["Const"] for dim in topology["grid"])
    ctas_per_cluster = math.prod(topology["cluster"])
    assert ctas // ctas_per_cluster == 1
    assert ctas_per_cluster == 8

    expected = np.stack([np.arange(8, dtype=np.int32), np.arange(8, dtype=np.int32) % 2], axis=1)
    result = v2.Engine().run(module, {"output": np.zeros((8, 2), dtype=np.int32)})
    assert result.status["kind"] == "completed", result.status
    np.testing.assert_array_equal(result.outputs["output"], expected)
