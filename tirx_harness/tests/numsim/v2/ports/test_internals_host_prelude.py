"""v2 copy of ``tests/numsim/integration/test_host_prelude.py::test_dynamic_tensor_map_expressions_run_in_the_loaded_artifact_prologue``.

Dropped: the legacy ``CompiledModule.load()`` call (Rust extension load and
validation, no v2 equivalent). The kernel runs through the public v2 API and
the output assertion (descriptor oracles global_shape=32, box_shape=16) is
kept. Kernel copied verbatim.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def host_encoded_dynamic_integer_tensor_map(
    source: T.Buffer((32,), "uint8"),
    output: T.Buffer((16,), "uint8"),
    delta: T.int32,
):
    tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        tensor_map,
        "uint8",
        1,
        source.data,
        T.truncdiv(delta, T.int32(2)) + T.int32(35),
        T.floordiv(delta, T.int32(2)) + T.int32(20),
        1,
        0,
        0,
        0,
        0,
    )
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](
                T.address_of(shared[0]),
                T.address_of(tensor_map),
                16,
                T.address_of(barrier[0]),
            )
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.warp_sync()
    if lane < 16:
        output[lane] = shared[lane]


def test_dynamic_tensor_map_expressions_run_in_the_loaded_artifact_prologue():
    """``-7/2`` truncates to -3 while floor division gives -4; reading at
    coordinate 16 makes the distinction observable (an incorrect
    global_shape=31 would zero-fill the final element)."""

    module = v2.transpile(host_encoded_dynamic_integer_tensor_map)
    source = np.arange(1, 33, dtype=np.uint8)
    inputs = {"source": source, "output": np.zeros(16, dtype=np.uint8), "delta": np.int32(-7)}
    result = v2.Engine().run(module, inputs)

    assert -(abs(-7) // abs(2)) == -3
    assert int(np.floor_divide(np.int64(-7), np.int64(2))) == -4
    np.testing.assert_array_equal(result.outputs["output"], source[16:])
