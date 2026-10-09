"""v2 ports of the legacy register-layout artifact tests; the
``result.stats["task_count"]`` pin is dropped."""

from __future__ import annotations

import numpy as np

from tests.numsim.support.kernels import tcgen_atom_layout_roundtrip, wg_local_layout_roundtrip
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


def test_wg_local_layout_uses_tid_in_wg_as_owner_and_m_as_register_index():
    """Port of ``tests/numsim/integration/test_register_layout_artifact.py::
    test_wg_local_layout_uses_tid_in_wg_as_owner_and_m_as_register_index``.

    Dropped pin: ``result.stats["task_count"] == 4``.
    """

    source = np.arange(128 * 4, dtype=np.float32).reshape(128, 4) - np.float32(17)
    output = np.zeros_like(source)

    module = v2.transpile(wg_local_layout_roundtrip)
    result = v2.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source * np.float32(2))


def test_tcgen_atom_layout_uses_warp_and_lane_owners():
    """Port of ``tests/numsim/integration/test_register_layout_artifact.py::
    test_tcgen_atom_layout_uses_warp_and_lane_owners``.

    Dropped pin: ``result.stats["task_count"] == 4``.
    """

    source = np.arange(128 * 8, dtype=np.float32).reshape(128, 8) / np.float32(8)
    output = np.zeros_like(source)

    module = v2.transpile(tcgen_atom_layout_roundtrip)
    result = v2.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)
