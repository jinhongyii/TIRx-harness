"""v2 ports of the legacy ordering-call tests that also pinned
``KernelSpec.semantic_requirements`` and ``module.rust_source`` text; those
pins are dropped."""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.support.kernels import ordering_only_control_calls
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.v2.lowering import lower

pytestmark = requires_v2_engine


@T.prim_func
def griddep_producer(intermediate: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    intermediate[lane] = lane + 1
    T.ptx.griddepcontrol.launch_dependents()


@T.prim_func
def griddep_consumer(intermediate: T.Buffer((32,), "int32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.griddepcontrol.wait()
    output[lane] = intermediate[lane] * 2



def test_ordering_only_calls_preserve_native_source_order():
    """Port of ``tests/numsim/runtime/test_ordering_calls.py::test_ordering_only_calls_preserve_native_source_order``.

    Dropped pins: ``module.spec.kernels[0].semantic_requirements ==
    ("external_grid_dependency_satisfied",)`` and the three
    ``module.rust_source`` substring assertions.
    """

    output = np.zeros(32, dtype=np.int32)

    module = v2.transpile(ordering_only_control_calls)
    result = v2.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.arange(1, 33, dtype=np.int32))

    # W11: the pinned facts, on the lowered Program. `griddepcontrol.wait` records
    # the external grid-dependency assumption (legacy semantic_requirements), and
    # the ordering-only calls keep their source order.
    program = lower(ordering_only_control_calls)
    assert program.requirements.grid_dependency
    order = [i.variant for i in program.code if i.variant in ("MbarInit", "Fence", "GridDepControl", "Barrier")]
    assert order == ["MbarInit", "Fence", "Fence", "GridDepControl", "Barrier"], order


def test_griddep_token_crosses_sequential_kernel_phases():
    """Port of ``tests/numsim/runtime/test_ordering_calls.py::test_griddep_token_crosses_sequential_kernel_phases``.

    Dropped pin: ``"ordering.griddep_" not in module.rust_source``.
    """

    intermediate = np.zeros(32, dtype=np.int32)
    output = np.zeros(32, dtype=np.int32)
    module = v2.transpile((griddep_producer, griddep_consumer))

    result = v2.Engine().run(
        module,
        {
            "k0:intermediate": intermediate,
            "k1:intermediate": intermediate,
            "k1:output": output,
        },
        outputs=("k1:output",),
    )

    np.testing.assert_array_equal(result.outputs["k1:output"], 2 * np.arange(1, 33, dtype=np.int32))
