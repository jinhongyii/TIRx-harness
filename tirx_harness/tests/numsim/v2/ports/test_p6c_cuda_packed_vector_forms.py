"""v2 ports of ``tests/numsim/runtime/test_cuda_packed_vector_forms.py``
(``test_cuda_ldg_supports_float32x2_as_one_packed_64bit_load``,
``test_cuda_shfl_sync_preserves_float16x2_payload_bits``).

``analyze(kernel).unsupported == ()`` becomes ``v2.transpile`` accepting the
kernel. Kernels and oracles are copied verbatim.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def cuda_ldg_float32x2(
    source: T.Buffer((32,), "float32x2"), output_bits: T.Buffer((32,), "uint64")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    loaded: T.let = T.cuda.ldg(source.ptr_to([lane]), "float32x2")
    output_bits[lane] = T.reinterpret("uint64", loaded)


@T.prim_func
def cuda_shfl_sync_float16x2(
    source_bits: T.Buffer((32,), "uint32"), output_bits: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    packed: T.let = T.reinterpret("float16x2", source_bits[lane])
    shuffled: T.let = T.cuda.__shfl_sync(
        T.uint32(0xFFFFFFFF), packed, T.cast(31 - lane, "uint32"), 32
    )
    output_bits[lane] = T.reinterpret("uint32", shuffled)



@v2_gap(
    "T.cuda.ldg(source.ptr_to([lane]), 'float32x2') over a float32x2[32] buffer "
    "stops with ExecutionError bad_address: the engine's AddrOf scales the "
    "scalar-element offset by the whole vector dtype (CONTRACT_REQUESTS "
    "W12-tile-forms 1)"
)
def test_cuda_ldg_supports_float32x2_as_one_packed_64bit_load():
    """Port of ``tests/numsim/runtime/test_cuda_packed_vector_forms.py::test_cuda_ldg_supports_float32x2_as_one_packed_64bit_load``."""

    source_bits = (np.arange(64, dtype=np.uint32) * np.uint32(0x01020305)) ^ np.uint32(0xA55AA55A)
    source = source_bits.view(np.uint64)

    module = v2.transpile(cuda_ldg_float32x2)
    result = v2.Engine().run(
        module,
        {"source": source, "output_bits": np.zeros(32, dtype=np.uint64)},
    )

    np.testing.assert_array_equal(result.outputs["output_bits"], source_bits.view(np.uint64))


def test_cuda_shfl_sync_preserves_float16x2_payload_bits():
    """Port of ``tests/numsim/runtime/test_cuda_packed_vector_forms.py::test_cuda_shfl_sync_preserves_float16x2_payload_bits``."""

    low = np.arange(32, dtype=np.uint32) * np.uint32(0x0211) + np.uint32(0x7C01)
    high = np.arange(32, dtype=np.uint32) * np.uint32(0x0103) + np.uint32(0x8000)
    source_bits = (high << np.uint32(16)) | (low & np.uint32(0xFFFF))

    module = v2.transpile(cuda_shfl_sync_float16x2)
    result = v2.Engine().run(
        module,
        {"source_bits": source_bits, "output_bits": np.zeros(32, dtype=np.uint32)},
    )

    np.testing.assert_array_equal(result.outputs["output_bits"], source_bits[::-1])
