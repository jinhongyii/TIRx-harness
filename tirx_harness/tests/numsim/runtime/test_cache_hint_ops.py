from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tvm.script import tirx as T


CACHE_HINT_OPS = frozenset(
    (
        "tirx.ptx.applypriority",
        "tirx.ptx.applypriority_async_bulk",
        "tirx.ptx.applypriority_async_bulk_tensor",
        "tirx.ptx.cp_async_bulk_prefetch_evict_last",
        "tirx.ptx.cp_async_bulk_tensor_prefetch_evict_last",
        "tirx.ptx.prefetch_valid_addr",
        "tirx.ptx.prefetchu",
    )
)


@T.prim_func
def raw_cache_hint_family(
    source: T.Buffer((256,), "uint8"),
    input_map: T.TensorMap(),
    output: T.Buffer((32,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    issue = lane < 16

    T.ptx["prefetch.L1::32B.valid_addr"](source.ptr_to([0]), pred=issue)
    T.ptx["prefetchu.L1"](source.ptr_to([0]), pred=issue)
    T.ptx["applypriority.L2::evict_normal"](source.ptr_to([0]), pred=issue)
    T.ptx["cp.async.bulk.prefetch.L2.global.L2::evict_last"](
        source.ptr_to([16]), T.uint32(32), pred=issue
    )
    T.ptx["applypriority.async.bulk.bulk_group.L2::evict_normal"](
        source.ptr_to([0]), T.uint32(32), pred=issue
    )
    T.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.L2::evict_last"](
        T.address_of(input_map), T.int32(0), T.int32(0), pred=issue
    )
    T.ptx["applypriority.async.bulk.tensor.2d.bulk_group.L2::evict_normal"](
        T.address_of(input_map), T.int32(0), T.int32(0), pred=issue
    )
    T.ptx.cp.async_.bulk.commit_group()
    T.ptx.cp.async_.bulk.wait_group.read(0)
    output[lane] = source[lane + 32]


@T.prim_func
def predicated_valid_address(source: T.Buffer((256,), "uint8"), enabled: T.int32):
    T.device_entry()
    _warp = T.warp_id([1])
    T.ptx["prefetch.L1::32B.valid_addr"](source.ptr_to([256]), pred=enabled)


@T.prim_func
def invalid_scalar_cache_hint_contracts(
    source: T.Buffer((256,), "uint8"),
    apply_offset: T.int32,
    prefetch_offset: T.int32,
    prefetch_size: T.uint32,
    bulk_apply_offset: T.int32,
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    issue = lane == 0
    T.ptx["applypriority.L2::evict_normal"](source.ptr_to([apply_offset]), pred=issue)
    T.ptx["cp.async.bulk.prefetch.L2.global.L2::evict_last"](
        source.ptr_to([prefetch_offset]), prefetch_size, pred=issue
    )
    T.ptx["applypriority.async.bulk.bulk_group.L2::evict_normal"](
        source.ptr_to([bulk_apply_offset]), T.uint32(16), pred=issue
    )


@T.prim_func
def raw_gather4_cache_hint(descriptor: T.Buffer((128,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["applypriority.async.bulk.tensor.2d.global.bulk_group.tile::gather4.L2::evict_normal"](
        descriptor.ptr_to([0]),
        T.int32(0),
        T.int32(0),
        T.int32(0),
        T.int32(0),
        T.int32(0),
        pred=lane == 0,
    )
    T.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::evict_last"](
        descriptor.ptr_to([0]),
        T.int32(0),
        T.int32(0),
        T.int32(0),
        T.int32(0),
        T.int32(0),
        pred=lane == 0,
    )


@T.prim_func
def varying_typed_tensor_map_hint(
    first_map: T.TensorMap(), second_map: T.TensorMap(), single_lane: T.int32
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    issue = T.Select(single_lane != 0, lane == 0, T.bool(True))
    T.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.L2::evict_last"](
        T.Select(lane == 0, T.address_of(first_map), T.address_of(second_map)),
        T.int32(0),
        T.int32(0),
        pred=issue,
    )


def _tensor_map(array: np.ndarray) -> np.ndarray:
    return numsim.TensorMap(
        base=array,
        global_shape=(4, 4),
        global_strides=(16,),
        box_shape=(4, 1),
        element_strides=(1, 1),
    ).numpy()


def _cache_hint_inputs() -> dict[str, np.ndarray]:
    return {
        "source": np.arange(256, dtype=np.uint8) ^ np.uint8(0xA5),
        "input_map": _tensor_map(np.arange(16, dtype=np.float32).reshape(4, 4)),
        "output": np.zeros(32, dtype=np.uint8),
    }


def test_tensor_map_selector_accepts_independent_lane_hint_descriptors(tmp_path):
    module = numsim.transpile(varying_typed_tensor_map_hint, cache_dir=tmp_path)
    inputs = {
        "first_map": _tensor_map(np.zeros((4, 4), dtype=np.float32)),
        "second_map": _tensor_map(np.ones((4, 4), dtype=np.float32)),
        "single_lane": 0,
    }

    # Descriptor selection is now lowered per issuing lane; independent cache
    # hints do not require all lanes to select the same descriptor.
    for single_lane in (0, 1):
        inputs["single_lane"] = single_lane
        result = numsim.Engine().run(module, inputs)
        assert result.verdict == "clean"
