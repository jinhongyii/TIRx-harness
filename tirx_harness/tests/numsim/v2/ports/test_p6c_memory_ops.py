"""v2 ports of the legacy memory-family tests that called legacy internals.

* ``analyze(kernel).unsupported == ()`` becomes "``v2.transpile`` lowers the
  kernel" (one parametrized case per kernel).
* ``prepare_bindings(...).tensor_map_outputs[name].dtype`` becomes the dtype
  of the tensor-map output ``v2.Engine().run`` returns under the map
  parameter name (2a5895a).

Kernels copied verbatim from ``tests/numsim/runtime/test_memory_ops.py``.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import TensorMap, v2

pytestmark = requires_v2_engine


@T.prim_func
def cp_async_plain_4(source: T.Buffer((128,), "uint8"), output: T.Buffer((128,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "uint8", scope="shared")
    T.ptx["cp.async.ca.shared.global.L2::64B"](
        T.address_of(shared[lane * 4]), T.address_of(source[lane * 4]), 4
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(4):
        output[lane * 4 + element] = shared[lane * 4 + element]


@T.prim_func
def cp_async_cache_hint_16(source: T.Buffer((512,), "uint8"), output: T.Buffer((512,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    T.ptx["cp.async.cg.shared.global.L2::cache_hint.L2::128B"](
        T.address_of(shared[lane * 16]),
        T.address_of(source[lane * 16]),
        16,
        T.uint64(0x12F0000000000000),
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(16):
        output[lane * 16 + element] = shared[lane * 16 + element]


@T.prim_func
def cp_async_ca_ignore_src_zero_fill_8(
    source: T.Buffer((128,), "uint8"), output: T.Buffer((256,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((256,), "uint8", scope="shared")
    for element in T.serial(8):
        shared[lane * 8 + element] = T.uint8(0xA5)
    T.cuda.warp_sync()
    T.ptx["cp.async.ca.shared.global"](
        T.address_of(shared[lane * 8]),
        T.address_of(source[lane * 8]),
        8,
        T.ptx.pred(lane >= 16),
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(8):
        output[lane * 8 + element] = shared[lane * 8 + element]


@T.prim_func
def cp_async_cg_ignore_src_zero_fill_16(
    source: T.Buffer((256,), "uint8"), output: T.Buffer((512,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    for element in T.serial(16):
        shared[lane * 16 + element] = T.uint8(0x5A)
    T.cuda.warp_sync()
    T.ptx["cp.async.cg.shared.global"](
        T.address_of(shared[lane * 16]),
        T.address_of(source[lane * 16]),
        16,
        T.ptx.pred(lane >= 16),
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(16):
        output[lane * 16 + element] = shared[lane * 16 + element]


@T.prim_func
def cuda_atomic_add_float32(
    counter: T.Buffer((1,), "float32"),
    old_values: T.Buffer((4,), "float32"),
    final_value: T.Buffer((1,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 4:
        old_values[lane] = T.cuda.atomic_add(counter.data, T.float32(0.5))
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = counter[0]


@T.prim_func
def cuda_atomic_add_float32x2(
    counter: T.Buffer((1,), "float32x2"),
    increments: T.Buffer((4,), "float32x2"),
    old_values: T.Buffer((4,), "float32x2"),
    final_value: T.Buffer((1,), "float32x2"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 4:
        old_values[lane] = T.cuda.atomic_add(counter.data, increments[lane])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = counter[0]


@T.prim_func
def cuda_atomic_add_float16x2(
    counter: T.Buffer((1,), "float16x2"),
    increments: T.Buffer((2,), "float16x2"),
    old_values: T.Buffer((2,), "float16x2"),
    final_value: T.Buffer((1,), "float16x2"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 2:
        old_values[lane] = T.cuda.atomic_add(counter.data, increments[lane])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = counter[0]


@T.prim_func
def cuda_atomic_add_float16x2_shared(
    initial: T.Buffer((1,), "float16x2"),
    increments: T.Buffer((2,), "float16x2"),
    old_values: T.Buffer((2,), "float16x2"),
    final_value: T.Buffer((1,), "float16x2"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    counter = T.alloc_buffer((1,), "float16x2", scope="shared")
    if lane == 0:
        counter[0] = initial[0]
    T.cuda.warp_sync()
    if lane < 2:
        old_values[lane] = T.cuda.atomic_add(counter.data, increments[lane])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = counter[0]


@T.prim_func
def cuda_atomic_add_bfloat16x2(
    counter: T.Buffer((1,), "bfloat16x2"),
    increments: T.Buffer((2,), "bfloat16x2"),
    old_values: T.Buffer((2,), "bfloat16x2"),
    final_value: T.Buffer((1,), "bfloat16x2"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 2:
        old_values[lane] = T.cuda.atomic_add(counter.data, increments[lane])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = counter[0]


@T.prim_func
def cuda_atomic_add_float32x4(
    counter: T.Buffer((1,), "float32x4"),
    increments: T.Buffer((2,), "float32x4"),
    old_values: T.Buffer((2,), "float32x4"),
    final_value: T.Buffer((1,), "float32x4"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 2:
        old_values[lane] = T.cuda.atomic_add(counter.data, increments[lane])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = counter[0]


@T.prim_func
def ptx_atomic_add_f32_vectors(
    counter: T.Buffer((6,), "float32"),
    increments: T.Buffer((6,), "float32"),
    old_values: T.Buffer((6,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["atom.global.add.v4.f32"](
        old_values[0],
        old_values[1],
        old_values[2],
        old_values[3],
        counter.ptr_to([0]),
        increments[0],
        increments[1],
        increments[2],
        increments[3],
        pred=lane == 0,
    )
    T.ptx["atom.global.add.v2.f32"](
        old_values[4],
        old_values[5],
        counter.ptr_to([4]),
        increments[4],
        increments[5],
        pred=lane == 0,
    )


@T.prim_func
def cuda_atomic_cas_uint64x2(
    cell: T.Buffer((1,), "uint64x2"),
    compares: T.Buffer((3,), "uint64x2"),
    replacements: T.Buffer((3,), "uint64x2"),
    old_values: T.Buffer((3,), "uint64x2"),
    final_value: T.Buffer((1,), "uint64x2"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 3:
        old_values[lane] = T.cuda.atomic_cas(cell.data, compares[lane], replacements[lane])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = cell[0]


@T.prim_func
def cuda_atomic_cas_float32x4(
    cell: T.Buffer((1,), "float32x4"),
    compares: T.Buffer((2,), "float32x4"),
    replacements: T.Buffer((2,), "float32x4"),
    old_values: T.Buffer((2,), "float32x4"),
    final_value: T.Buffer((1,), "float32x4"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 2:
        old_values[lane] = T.cuda.atomic_cas(cell.data, compares[lane], replacements[lane])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = cell[0]


@T.prim_func
def cuda_atomic_cas_uint32x4(
    cell: T.Buffer((1,), "uint32x4"),
    compare: T.Buffer((1,), "uint32x4"),
    replacement: T.Buffer((1,), "uint32x4"),
    old_value: T.Buffer((1,), "uint32x4"),
    final_value: T.Buffer((1,), "uint32x4"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        old_value[0] = T.cuda.atomic_cas(cell.data, compare[0], replacement[0])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = cell[0]


@T.prim_func
def st_bulk_default_weak(output: T.Buffer((4,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "uint32", scope="shared")
    if lane < 4:
        shared[lane] = T.cast(lane + 1, "uint32")
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.st_bulk.shared__cta(shared.ptr_to([0]), T.uint64(8))
    T.cuda.warp_sync()
    if lane < 4:
        output[lane] = shared[lane]


@T.prim_func
def legacy_global_acquire(source: T.Buffer((32,), "uint64"), output: T.Buffer((32,), "uint64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    local = T.alloc_buffer((1,), "uint64", scope="local")
    T.ptx.ld.acquire.gpu.global_.u64(local[0], T.address_of(source[lane]))
    output[lane] = local[0]


@T.prim_func
def guarded_ldg32(source: T.Buffer((32,), "float32"), output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    local = T.alloc_buffer((32,), "float32", scope="local")
    local[lane] = T.float32(-123.5)
    T.evaluate(T.s_tir.ldg32(local.data, lane < 16, source[lane], lane))
    output[lane] = local[lane]


@T.prim_func
def legacy_ldmatrix_x2_trans(output: T.Buffer((128,), "uint16")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((256,), "uint16", scope="shared")
    local = T.alloc_buffer((4,), "uint16", scope="local")
    for element in T.serial(8):
        shared[lane * 8 + element] = T.Cast("uint16", lane * 8 + element)
    T.cuda.warp_sync()
    T.evaluate(
        T.ptx_legacy.ldmatrix(True, 2, ".b16", local.data, 0, shared.data, lane * 8, dtype="uint16")
    )
    for element in T.serial(4):
        output[lane * 4 + element] = local[element]


@T.prim_func
def bulk_g2s_cta(source: T.Buffer((16,), "uint8"), output: T.Buffer((16,), "uint8")):
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
        T.ptx["cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"](
            T.address_of(shared[0]),
            T.address_of(source[0]),
            T.cast(16, "uint32"),
            T.address_of(barrier[0]),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.warp_sync()
    if lane < 16:
        output[lane] = shared[lane]


@T.prim_func
def raw_tma_gather4_bar_address(input_map: T.TensorMap(), output: T.Buffer((4, 4), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4, 4), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx[
            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1"
        ](
            T.address_of(shared[0, 0]),
            T.address_of(input_map),
            0,
            0,
            1,
            2,
            3,
            T.cuda.cvta_generic_to_shared(T.address_of(barrier[0])),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 64)
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.warp_sync()
    if lane < 16:
        output[lane // 4, lane % 4] = shared[lane // 4, lane % 4]


@T.prim_func
def raw_tma_reduce_add(source: T.Buffer((4,), "float32"), output_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    if lane < 4:
        shared[lane] = source[lane]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.ptx["cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.bulk_group"](
            T.address_of(output_map), 0, T.address_of(shared[0])
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def raw_tma_reduce_add_bfloat16(source: T.Buffer((8,), "bfloat16"), output_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "bfloat16", scope="shared")
    if lane < 8:
        shared[lane] = source[lane]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.ptx["cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.bulk_group"](
            T.address_of(output_map), 0, T.address_of(shared[0])
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def raw_tma_prefetch(input_map: T.TensorMap(), output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.tile"](T.address_of(input_map), 0, 0)
        output[0] = 1


@T.prim_func
def raw_bulk_prefetch(source: T.Buffer((32,), "int32"), output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx["cp.async.bulk.prefetch.L2.global"](T.address_of(source[0]), T.uint32(128))
        output[0] = source[0]


@T.prim_func
def raw_prefetchu(source: T.Buffer((32,), "int32"), output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx["prefetchu.L1"](T.address_of(source[0]))
        output[0] = source[0]


def _tensor_map(
    array: np.ndarray,
    *,
    global_shape: tuple[int, ...],
    global_strides: tuple[int, ...],
    box_shape: tuple[int, ...],
) -> np.ndarray:
    return TensorMap(
        base=array,
        global_shape=global_shape,
        global_strides=global_strides,
        box_shape=box_shape,
        element_strides=(1,) * len(global_shape),
    ).numpy()


def test_raw_tma_reduce_uses_retained_logical_dtype():
    """Port of ``tests/numsim/runtime/test_memory_ops.py::test_raw_tma_reduce_uses_retained_logical_dtype``.

    Legacy ``prepare_bindings(...).tensor_map_outputs["output_map"].dtype ==
    "float32"`` -> the returned ``output_map`` output is a float32 array.
    """

    source = np.array([1.5, -2.0, 4.25, 3.0], dtype=np.float32)
    destination = np.array([10.0, 20.0, -1.0, 8.0], dtype=np.float32)
    initial_destination = destination.copy()
    output_map = _tensor_map(destination, global_shape=(4,), global_strides=(), box_shape=(4,))

    module = v2.transpile(raw_tma_reduce_add)
    result = v2.Engine().run(module, {"source": source, "output_map": output_map})
    assert result.outputs["output_map"].dtype == np.float32
    np.testing.assert_array_equal(result.outputs["output_map"], initial_destination + source)


def test_raw_tma_reduce_bfloat16_uses_plain_logical_dtype_array():
    """Port of ``tests/numsim/runtime/test_memory_ops.py::test_raw_tma_reduce_bfloat16_uses_plain_logical_dtype_array``.

    Legacy ``prepare_bindings(...).tensor_map_outputs["output_map"].dtype ==
    "bfloat16"`` -> the returned ``output_map`` output is a bfloat16 array.
    Legacy compared the output against ``expected.view(np.uint16)`` (its
    payload returned the raw 16-bit words); v2 returns the plain logical
    bfloat16 array, which is what the legacy test name and binding assertion
    describe, so the copy compares physical bits of the bfloat16 output.
    """

    source = np.asarray([1.5, -2.0, 4.25, 3.0, -0.5, 0.25, 16.0, -8.0], dtype=ml_dtypes.bfloat16)
    destination = np.asarray([10.0, 20.0, -1.0, 8.0, 2.0, -4.0, 0.5, 32.0], dtype=ml_dtypes.bfloat16)
    initial_destination = destination.copy()
    output_map = _tensor_map(destination, global_shape=(8,), global_strides=(), box_shape=(8,))

    module = v2.transpile(raw_tma_reduce_add_bfloat16)
    result = v2.Engine().run(module, {"source": source, "output_map": output_map})
    expected = np.asarray(
        initial_destination.astype(np.float32) + source.astype(np.float32), dtype=ml_dtypes.bfloat16
    )
    output = result.outputs["output_map"]
    assert output.dtype == ml_dtypes.bfloat16
    np.testing.assert_array_equal(output.view(np.uint16), expected.view(np.uint16))


_CLOSED_WORLD_KERNELS = {
    "cp_async_plain_4": cp_async_plain_4,
    "cp_async_cache_hint_16": cp_async_cache_hint_16,
    "cp_async_ca_ignore_src_zero_fill_8": cp_async_ca_ignore_src_zero_fill_8,
    "cp_async_cg_ignore_src_zero_fill_16": cp_async_cg_ignore_src_zero_fill_16,
    "cuda_atomic_add_float32": cuda_atomic_add_float32,
    "cuda_atomic_add_float32x2": cuda_atomic_add_float32x2,
    "cuda_atomic_add_float16x2": cuda_atomic_add_float16x2,
    "cuda_atomic_add_float16x2_shared": cuda_atomic_add_float16x2_shared,
    "cuda_atomic_add_bfloat16x2": cuda_atomic_add_bfloat16x2,
    "cuda_atomic_add_float32x4": cuda_atomic_add_float32x4,
    "ptx_atomic_add_f32_vectors": ptx_atomic_add_f32_vectors,
    "cuda_atomic_cas_uint64x2": cuda_atomic_cas_uint64x2,
    "cuda_atomic_cas_float32x4": cuda_atomic_cas_float32x4,
    "cuda_atomic_cas_uint32x4": cuda_atomic_cas_uint32x4,
    "st_bulk_default_weak": st_bulk_default_weak,
    "legacy_global_acquire": legacy_global_acquire,
    "guarded_ldg32": guarded_ldg32,
    "legacy_ldmatrix_x2_trans": legacy_ldmatrix_x2_trans,
    "bulk_g2s_cta": bulk_g2s_cta,
    "raw_tma_gather4_bar_address": raw_tma_gather4_bar_address,
    "raw_tma_reduce_add": raw_tma_reduce_add,
    "raw_tma_reduce_add_bfloat16": raw_tma_reduce_add_bfloat16,
    "raw_tma_prefetch": raw_tma_prefetch,
    "raw_bulk_prefetch": raw_bulk_prefetch,
    "raw_prefetchu": raw_prefetchu,
}


_CLOSED_WORLD_GAPS = {
    "guarded_ldg32": "v2.transpile rejects builtin tirx.s_tir.ldg32 (UnsupportedTIRxError); legacy analyze accepted it",
    "legacy_ldmatrix_x2_trans": (
        "v2.transpile rejects builtin tirx.ptx_legacy.ldmatrix (UnsupportedTIRxError; "
        "lowering-inventory.md lists ptx_legacy.* as out of scope, no behaviour-delta row); "
        "legacy analyze accepted it"
    ),
}


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(name, marks=v2_gap(_CLOSED_WORLD_GAPS[name])) if name in _CLOSED_WORLD_GAPS else name
        for name in _CLOSED_WORLD_KERNELS
    ],
)
def test_memory_family_public_ops_have_closed_world_registration(name):
    """Port of ``tests/numsim/runtime/test_memory_ops.py::test_memory_family_public_ops_have_closed_world_registration``.

    Legacy ``analyze(kernel).unsupported == ()`` -> ``v2.transpile`` lowers
    the kernel without ``UnsupportedTIRxError``.
    """

    module = v2.transpile(_CLOSED_WORLD_KERNELS[name])
    assert len(module.spec.kernels) == 1
