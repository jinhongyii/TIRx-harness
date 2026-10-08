from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm.script import tirx as T


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


@T.prim_func
def cuda_directional_shuffles_float16x2(
    source_bits: T.Buffer((32,), "uint32"), output_bits: T.Buffer((32, 3), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    packed: T.let = T.reinterpret("float16x2", source_bits[lane])
    full: T.let = T.uint32(0xFFFFFFFF)
    output_bits[lane, 0] = T.reinterpret("uint32", T.cuda.__shfl_up_sync(full, packed, 1, 32))
    output_bits[lane, 1] = T.reinterpret("uint32", T.cuda.__shfl_down_sync(full, packed, 1, 32))
    output_bits[lane, 2] = T.reinterpret("uint32", T.cuda.__shfl_xor_sync(full, packed, 1, 32))


@T.prim_func
def cuda_shuffles_bfloat16x2(
    source_bits: T.Buffer((32,), "uint32"), output_bits: T.Buffer((32, 4), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    packed: T.let = T.reinterpret("bfloat16x2", source_bits[lane])
    full: T.let = T.uint32(0xFFFFFFFF)
    output_bits[lane, 0] = T.reinterpret(
        "uint32", T.cuda.__shfl_sync(full, packed, T.cast(31 - lane, "uint32"), 32)
    )
    output_bits[lane, 1] = T.reinterpret("uint32", T.cuda.__shfl_up_sync(full, packed, 1, 32))
    output_bits[lane, 2] = T.reinterpret("uint32", T.cuda.__shfl_down_sync(full, packed, 1, 32))
    output_bits[lane, 3] = T.reinterpret("uint32", T.cuda.__shfl_xor_sync(full, packed, 1, 32))


def test_cuda_directional_shuffles_preserve_float16x2_payload_bits(tmp_path):
    source_bits = (np.arange(32, dtype=np.uint32) * np.uint32(0x04110203)) ^ np.uint32(0x7C018000)
    module = numsim.transpile(cuda_directional_shuffles_float16x2, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source_bits": source_bits, "output_bits": np.zeros((32, 3), dtype=np.uint32)},
    )

    expected = np.empty((32, 3), dtype=np.uint32)
    expected[:, 0] = source_bits[np.maximum(np.arange(32) - 1, 0)]
    expected[:, 1] = source_bits[np.minimum(np.arange(32) + 1, 31)]
    expected[:, 2] = source_bits[np.arange(32) ^ 1]
    np.testing.assert_array_equal(result.outputs["output_bits"], expected)


def test_cuda_shuffles_preserve_bfloat16x2_payload_bits(tmp_path):
    source_bits = (np.arange(32, dtype=np.uint32) * np.uint32(0x01020411)) ^ np.uint32(0x7FC18000)
    module = numsim.transpile(cuda_shuffles_bfloat16x2, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source_bits": source_bits, "output_bits": np.zeros((32, 4), dtype=np.uint32)},
    )

    expected = np.empty((32, 4), dtype=np.uint32)
    expected[:, 0] = source_bits[::-1]
    expected[:, 1] = source_bits[np.maximum(np.arange(32) - 1, 0)]
    expected[:, 2] = source_bits[np.minimum(np.arange(32) + 1, 31)]
    expected[:, 3] = source_bits[np.arange(32) ^ 1]
    np.testing.assert_array_equal(result.outputs["output_bits"], expected)
