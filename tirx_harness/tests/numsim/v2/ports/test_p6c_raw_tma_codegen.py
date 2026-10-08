"""v2 ports of ``tests/numsim/runtime/test_raw_tma_codegen.py`` functions that
called the legacy ``transpiler.frontend.analyze`` or pinned legacy error
wording. ``analyze(kernel).unsupported == ()`` becomes a successful
``v2.transpile``; ``match=`` wording becomes the v2 error kind. Kernels are
the shared ``tests.numsim.support.kernels`` fixtures (or copied verbatim).
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.support.kernels import (
    raw_tma_dynamic_wait_group_loop,
    raw_tma_sm100_barrier_address,
    raw_tma_transaction_mismatch,
)
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness import numsim
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine


def _tensor_map(array, *, global_shape, global_strides, box_shape):
    return numsim.TensorMap(
        base=array,
        global_shape=global_shape,
        global_strides=global_strides,
        box_shape=box_shape,
        element_strides=(1,) * len(global_shape),
    ).numpy()


def test_static_bulk_wait_group_immediates_execute_in_source_order():
    """Port of ``tests/numsim/runtime/test_raw_tma_codegen.py::test_static_bulk_wait_group_immediates_execute_in_source_order``.

    Dropped: legacy ``analyze(...).unsupported == ()``; ``v2.transpile``
    succeeding is the acceptance check."""

    module = v2.transpile(raw_tma_dynamic_wait_group_loop)
    result = v2.Engine().run(module, {"output": np.zeros(2, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.array([10, 11], dtype=np.int32))


def test_bulk_wait_group_rejects_runtime_count_during_numsim_transpilation():
    """Port of ``tests/numsim/runtime/test_raw_tma_codegen.py::test_bulk_wait_group_rejects_runtime_count_during_numsim_transpilation``.

    Kind: transpile-time ``UnsupportedTIRxError`` naming the
    ``cp_async_bulk_wait_group`` call. Dropped: the legacy wording
    ``"pending_group_count must be static"``."""

    source = """
@T.prim_func
def invalid(pending: T.Buffer((1,), "int32")):
    T.device_entry()
    T.ptx.cp.async_.bulk.wait_group.read(pending[0])
"""

    kernel = tvm.script.from_source(source, {"T": T})
    with pytest.raises(UnsupportedTIRxError) as excinfo:
        v2.transpile(kernel)
    assert any("wait_group" in str(entry) for entry in excinfo.value.unsupported), excinfo.value.unsupported


def test_sm100_two_cta_barrier_address_targets_pair_base():
    """Port of ``tests/numsim/runtime/test_raw_tma_codegen.py::test_sm100_two_cta_barrier_address_targets_pair_base``.

    Dropped: legacy ``analyze(...).unsupported == ()``; ``v2.transpile``
    succeeding is the acceptance check."""

    even = np.arange(4, dtype=np.float32).reshape(1, 4) + np.float32(10)
    odd = np.arange(4, dtype=np.float32).reshape(1, 4) + np.float32(20)
    even_map = _tensor_map(even, global_shape=(4, 1), global_strides=(16,), box_shape=(4, 1))
    odd_map = _tensor_map(odd, global_shape=(4, 1), global_strides=(16,), box_shape=(4, 1))

    module = v2.transpile(raw_tma_sm100_barrier_address)
    result = v2.Engine().run(
        module,
        {
            "input_map_even": even_map,
            "input_map_odd": odd_map,
            "output": np.zeros((2, 4), dtype=np.float32),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], np.concatenate([even, odd], axis=0))


def test_raw_tensor_map_under_delivery_reports_exact_bytes():
    """Port of ``tests/numsim/runtime/test_raw_tma_codegen.py::test_raw_tensor_map_under_delivery_reports_exact_bytes``.

    Delta sync-behaviour-deltas M17: the run fails closed as ``incomplete``
    (``divergent_block``) instead of legacy's ``transactions=48/52`` error;
    only the issuing lane waits, so the shortfall is not provable from engine
    state."""

    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    input_map = _tensor_map(source, global_shape=(4, 3), global_strides=(16,), box_shape=(4, 3))
    module = v2.transpile(raw_tma_transaction_mismatch)

    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(module, {"input_map": input_map})
    stops = [d for d in excinfo.value.diagnostics if d.get("status") in ("error", "incomplete")]
    assert stops and stops[0]["status"] == "incomplete", excinfo.value.diagnostics
    assert "divergent_block" in stops[0].get("reason", ""), stops[0]
