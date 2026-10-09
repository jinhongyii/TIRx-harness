"""v2 ports of ``tests/numsim/integration/test_direct_memory_artifact.py`` tests
that used the legacy ``analyze`` front end or pinned ``module.rust_source``.
``analyze(kernel).unsupported == ()`` becomes "``v2.transpile`` accepts the
kernel" (it raises ``UnsupportedTIRxError`` otherwise). Kernels are copied
verbatim from ``tests/numsim/support/kernels.py``.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def direct_cuda_ldg(
    source_f32: T.Buffer((32,), "float32"),
    source_i32: T.Buffer((32,), "int32"),
    output_f32: T.Buffer((32,), "float32"),
    output_i32: T.Buffer((32,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output_f32[lane] = T.cuda.ldg(source_f32.ptr_to([31 - lane]), "float32")
    output_i32[lane] = T.cuda.ldg(source_i32.ptr_to([31 - lane]), "int32")


@T.prim_func
def direct_tvm_access_ptr_shared(output: T.Buffer((128,), "uint32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    thread: T.let = warp * 32 + lane
    shared = T.alloc_buffer((128,), "uint32", scope="shared")
    shared[thread] = T.cast(thread * 3 + 1, "uint32")
    T.cuda.cta_sync()
    T.ptx.ld.shared.u32(output[thread], shared.access_ptr("r", offset=thread))


def test_cuda_ldg_reads_typed_global_memory():
    """Port of ``tests/numsim/integration/test_direct_memory_artifact.py::test_cuda_ldg_reads_typed_global_memory``.

    Dropped: legacy ``analyze(direct_cuda_ldg).unsupported == ()``; replaced
    by ``v2.transpile`` accepting the kernel.
    """

    source_f32 = np.linspace(-2.0, 3.0, 32, dtype=np.float32)
    source_i32 = np.arange(32, dtype=np.int32) * 7 - 11
    output_f32 = np.zeros(32, dtype=np.float32)
    output_i32 = np.zeros(32, dtype=np.int32)

    module = v2.transpile(direct_cuda_ldg)
    result = v2.Engine().run(
        module,
        {
            "source_f32": source_f32,
            "source_i32": source_i32,
            "output_f32": output_f32,
            "output_i32": output_i32,
        },
    )

    np.testing.assert_array_equal(result.outputs["output_f32"], source_f32[::-1])
    np.testing.assert_array_equal(result.outputs["output_i32"], source_i32[::-1])


def test_tvm_access_ptr_applies_element_offset_to_shared_pointer():
    """Port of ``tests/numsim/integration/test_direct_memory_artifact.py::test_tvm_access_ptr_applies_element_offset_to_shared_pointer``.

    Dropped: legacy ``analyze(...).unsupported == ()`` (replaced by
    ``v2.transpile`` accepting the kernel) and the ``module.rust_source``
    pins (``physical_ptr_access_view(``, ``access_ptr_byte_offsets``,
    ``.with_element_offset_extent(``).
    """

    output = np.zeros(128, dtype=np.uint32)

    module = v2.transpile(direct_tvm_access_ptr_shared)
    result = v2.Engine().run(module, {"output": output})

    expected = np.arange(128, dtype=np.uint32) * 3 + 1
    np.testing.assert_array_equal(result.outputs["output"], expected)
