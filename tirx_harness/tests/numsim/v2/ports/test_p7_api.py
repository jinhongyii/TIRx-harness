"""v2 ports of the multi-kernel ``ExecutionSubset`` tests in
``tests/numsim/integration/test_api.py``.

Legacy asserted the private ``numsim.api._execution_subset_payload`` helper:
a ``{phase_index: ExecutionSubset}`` mapping selects clusters per launch, a
bare subset is not broadcast to a multi-kernel module, and bad phase indices
or duplicate ids fail closed. v2 has no public ``ExecutionSubset`` type and
duck-types ``subset.cluster_ids`` (``v2.Engine._subset_extra``), so these
copies use a duck-typed :class:`_Subset` and observe the contract through
``v2.Engine().run`` on a three-launch module (three kernels, four
single-CTA clusters each, every CTA writing its launch's tag into its slot).
Dropped: the payload dictionaries and the legacy messages; the legacy
``cta_ids`` subset is spelled as the equivalent ``cluster_ids`` subset (v2
selects clusters only and raises ``NotImplementedError`` for ``cta_ids``;
with one CTA per cluster the two select the same CTA).
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


class _Subset:
    """Duck-typed stand-in for the legacy ``ExecutionSubset``."""

    def __init__(self, cluster_ids):
        self.cluster_ids = cluster_ids
        self.cta_ids = None


def _tagging_kernel(tag: int):
    @T.prim_func
    def kernel(output: T.Buffer((4,), "int32")):
        T.device_entry()
        cta = T.cta_id([4])
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        if lane == 0:
            output[cta] = T.int32(tag)

    return kernel


def _module():
    return v2.transpile([_tagging_kernel(1), _tagging_kernel(2), _tagging_kernel(3)])


def _inputs():
    return {f"k{phase}:output": np.zeros(4, dtype=np.int32) for phase in range(3)}


def test_multi_kernel_subsets_are_phase_indexed_and_never_broadcast():
    """Delta H6 of ``tests/numsim/integration/test_api.py::test_multi_kernel_subsets_are_phase_indexed_and_never_broadcast``.

    Legacy ran each launch with its own clusters: phase 0 on {0, 2}, phase 1
    in full, phase 2 on cluster 3. v2 runs every launch of a module in one
    engine run, whose subset is one parameter, so a mapping that selects
    different clusters per launch (or leaves a launch out) fails closed with
    ``InputError``. A mapping that selects the same clusters for every launch
    runs. A bare subset on a three-launch module is still rejected, never
    broadcast (``ValueError``)."""

    module = _module()
    with pytest.raises(v2.InputError, match="must be equal"):
        v2.Engine().run(module, _inputs(), subset={0: _Subset([2, 0]), 2: _Subset([3])})

    same = {phase: _Subset([2, 0]) for phase in range(3)}
    result = v2.Engine().run(module, _inputs(), subset=same)
    for phase in range(3):
        np.testing.assert_array_equal(result.outputs[f"k{phase}:output"],
                                      [phase + 1, 0, phase + 1, 0])

    with pytest.raises(ValueError, match="not broadcast"):
        v2.Engine().run(module, _inputs(), subset=_Subset([0]))


@pytest.mark.parametrize(
    ("subset", "error"),
    [
        pytest.param({3: _Subset([0])}, ValueError, id="phase_index_outside"),
        pytest.param({True: _Subset([0])}, TypeError, id="bool_phase_index"),
        pytest.param({0: _Subset([1, 1])}, ValueError, id="duplicate_cluster_id"),
    ],
)
def test_phase_subset_indices_and_ids_fail_closed(subset, error):
    """Port of ``tests/numsim/integration/test_api.py::test_phase_subset_indices_and_ids_fail_closed``.

    Legacy used a two-kernel payload for each case; here the module has three
    launches, so the out-of-range index is 3 (legacy: 2 of 2). Dropped: the
    legacy messages ("outside", "phase indices", "duplicate")."""

    with pytest.raises(error):
        v2.Engine().run(_module(), _inputs(), subset=subset)
