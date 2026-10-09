"""v2 copy of ``tests/numsim/integration/test_written_buffer_inference.py::test_selected_outputs_preserve_all_host_writes``
(both ``frozen`` params).

Triage (W9 phase 6, internal other-assertion): **port**. Legacy read the
results from the caller's ``old_value`` / ``counter`` arrays (in-place write
back of every written buffer, even unselected ones). v2 ``Engine.run`` never
mutates its inputs (same API change as the ``test_global_alias_artifact``
port). The copy keeps the selected-output set and the ``run_case`` path and
asserts both written buffers through ``result.outputs`` (``counter`` is
reported when all outputs are requested); the caller's arrays stay unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.cases import NumSimCase

pytestmark = requires_v2_engine


@T.prim_func
def atomic_through_pointer_offset(
    counter: T.Buffer((2,), "uint32"), old_value: T.Buffer((1,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        target = T.ptr_byte_offset(counter.ptr_to([0]), T.uint32(4), "uint32")
        T.ptx.atom.release.gpu.global_.add.u32(
            old_value[0],
            target,
            T.uint32(3),
        )


@pytest.mark.parametrize("frozen", [False, True])
def test_selected_outputs_preserve_all_host_writes(frozen):
    """Port; see the module docstring."""

    counter = np.array([5, 7], dtype=np.uint32)
    old_value = np.zeros(1, dtype=np.uint32)
    inputs = {"counter": counter, "old_value": old_value}
    module = v2.transpile(atomic_through_pointer_offset)
    if frozen:
        case = NumSimCase(
            kernel=atomic_through_pointer_offset,
            args=inputs,
            outputs=("old_value",),
            reference=lambda: {"old_value": np.array([7], dtype=np.uint32)},
        )
        v2.run_case(case).require_ok()
    else:
        result = v2.Engine().run(module, inputs, outputs=("old_value",))
        assert set(result.outputs) == {"old_value"}
        np.testing.assert_array_equal(result.outputs["old_value"], [7])

    every = v2.Engine().run(module, inputs)
    np.testing.assert_array_equal(every.outputs["old_value"], [7])
    np.testing.assert_array_equal(every.outputs["counter"], [5, 10])
    np.testing.assert_array_equal(old_value, [0])
    np.testing.assert_array_equal(counter, [5, 7])
