"""v2 ports of ``tests/numsim/integration/test_scalar_parameter_artifact.py``
tests that pinned ``module.rust_source`` / legacy error text or drove the
legacy ``prepare_bindings`` payload. Kernels are copied verbatim. Binding
range/shape checks are asserted as ``v2.InputError`` raised by
``v2.canonicalize_inputs`` (before any execution) and by ``Engine.run``.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def scalar_parameter_add(offset: T.int32, output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = offset + lane


@T.prim_func
def scalar_bound_shape(rows: T.int32, input_ptr: T.handle, output_ptr: T.handle):
    input_buffer = T.match_buffer(input_ptr, (rows, 32), "float32")
    output_buffer = T.match_buffer(output_ptr, (rows, 32), "float32")
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output_buffer[0, lane] = input_buffer[0, lane] + T.float32(1)


def test_scalar_parameter_range_is_checked_before_execution():
    """Port of ``tests/numsim/integration/test_scalar_parameter_artifact.py::test_scalar_parameter_range_is_checked_before_execution``.

    Dropped: ``match="outside int32 range"`` (legacy text) and the second
    half, which built a legacy ``prepare_bindings(...).to_payload()``,
    patched the scalar and fed it to ``module.load().run`` (legacy artifact
    payload; v2 has no such entry point). Asserted: ``offset = 1 << 31``
    is rejected with ``v2.InputError`` by ``v2.canonicalize_inputs``
    (binding, before execution) and by ``Engine.run``; ``2**31 - 1`` binds.
    """

    output = np.zeros(32, dtype=np.int32)
    module = v2.transpile(scalar_parameter_add)

    with pytest.raises(v2.InputError):
        v2.canonicalize_inputs(module, {"offset": 1 << 31, "output": output})
    with pytest.raises(v2.InputError):
        v2.Engine().run(module, {"offset": 1 << 31, "output": output})

    v2.canonicalize_inputs(module, {"offset": (1 << 31) - 1, "output": output})


def test_scalar_shape_consistency_is_checked_by_engine_api():
    """Port of ``tests/numsim/integration/test_scalar_parameter_artifact.py::test_scalar_shape_consistency_is_checked_by_engine_api``.

    Dropped: the ``module.rust_source`` pins (``validate_shape_scalar(``,
    the absent legacy messages) and ``match="disagrees with bound buffer
    extent"``. Asserted: ``rows=5`` runs; ``rows=4`` against 5-row buffers
    raises ``v2.InputError`` (also from ``v2.canonicalize_inputs``, i.e.
    before execution).
    """

    source = np.arange(5 * 32, dtype=np.float32).reshape(5, 32)
    output = np.zeros_like(source)
    module = v2.transpile(scalar_bound_shape)

    result = v2.Engine().run(
        module,
        {"rows": 5, "input_buffer": source, "output_buffer": output},
        outputs=("output_buffer",),
    )
    np.testing.assert_array_equal(result.outputs["output_buffer"][0], source[0] + 1)

    bad = {"rows": 4, "input_buffer": source, "output_buffer": output}
    with pytest.raises(v2.InputError):
        v2.canonicalize_inputs(module, bad)
    with pytest.raises(v2.InputError):
        v2.Engine().run(module, bad)
