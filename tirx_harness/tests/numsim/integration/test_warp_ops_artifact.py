from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def warp_engine_ops(
    source: T.Buffer((32,), "uint32"),
    output: T.Buffer((32, 6), "uint32"),
    unpacked: T.Buffer((32, 2), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value: T.let = source[lane]
    full: T.let = T.uint32(0xFFFFFFFF)
    output[lane, 0] = T.tvm_warp_shuffle_xor(full, value, 1, 32, 32)
    output[lane, 1] = T.cuda.ballot_sync(full, lane < 16)
    output[lane, 2] = T.cuda.reduce_min_sync_u32(full, value)
    output[lane, 3] = T.cuda.reduce_add_sync_u32(full, value)
    output[lane, 4] = T.cast(T.cuda.any_sync(full, lane == 31), "uint32")
    output[lane, 5] = T.cuda.warp_sum(value, width=8)
    lane_f32: T.let = T.cast(lane, "float32")
    packed_bf16: T.let = T.cuda.float22bfloat162_rn(lane_f32, lane_f32 + T.float32(1))
    unpacked_f32: T.let = T.cuda.bfloat1622float2(packed_bf16)
    unpacked[lane, 0] = T.cuda.float2_x(unpacked_f32)
    unpacked[lane, 1] = T.cuda.float2_y(unpacked_f32)


@T.prim_func
def warp_participant_contract(
    participant_masks: T.Buffer((32,), "uint32"),
    active_count: T.Buffer((1,), "int32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < active_count[0]:
        output[lane] = T.cuda.ballot_sync(participant_masks[lane], lane < 16)


def test_warp_intrinsics_use_typed_engine_calls(tmp_path):
    source = np.arange(32, dtype=np.uint32)
    output = np.zeros((32, 6), dtype=np.uint32)
    unpacked = np.zeros((32, 2), dtype=np.float32)

    module = numsim.transpile(warp_engine_ops, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output, "unpacked": unpacked})

    expected = np.zeros_like(output)
    expected[:, 0] = source[np.arange(32) ^ 1]
    expected[:, 1] = np.uint32(0xFFFF)
    expected[:, 2] = np.uint32(0)
    expected[:, 3] = np.uint32(496)
    expected[:, 4] = np.uint32(1)
    expected[:, 5] = np.repeat(np.array([28, 92, 156, 220], dtype=np.uint32), 8)
    np.testing.assert_array_equal(result.outputs["output"], expected)
    np.testing.assert_array_equal(
        result.outputs["unpacked"],
        np.stack((np.arange(32, dtype=np.float32), np.arange(1, 33, dtype=np.float32)), axis=1),
    )


