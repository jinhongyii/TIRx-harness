"""v2 ports of the ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py``
functions that built raw descriptor storage with the legacy-only
``numsim.cases._descriptor_storage`` helper.

The helper only byte-copied a ``TensorMap`` image into slot 0 of a
128-byte storage array. The v2 copies bind the same image directly as that
storage array (``image.view(storage_dtype)``), which keeps the host
descriptor's base so ``v2.canonicalize_inputs`` rewrites its global address
to the engine address of the base array. Legacy ``match=`` wording pins are
replaced by the v2 error kind. Kernels are copied verbatim.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness import numsim
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


_FLASHKDA_ACQUIRE_SOURCE = r"""__device__ __forceinline__ void flashkda_tensormap_acquire(const void *tmap_ptr) {
    asm volatile(
        "fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\n"
        :: "l"(tmap_ptr) : "memory");
}
"""

_GDN_REPLACE_ADDRESS_SOURCE = r"""__device__ __forceinline__ void gdn_tensormap_replace_global_address(void *desc, const void *addr) {
    asm volatile("tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;"
                 :: "l"(desc), "l"(addr) : "memory");
}
"""


def _gdn_replace_dim_source(index: int) -> str:
    return f"""__device__ __forceinline__ void gdn_tensormap_replace_global_dim_{index}(void *desc, unsigned int value) {{
    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], {index}, %1;"
                 :: "l"(desc), "r"(value) : "memory");
}}
"""


def _gdn_replace_stride_source(index: int) -> str:
    return f"""__device__ __forceinline__ void gdn_tensormap_replace_global_stride_{index}(void *desc, unsigned long long value) {{
    asm volatile("tensormap.replace.tile.global_stride.global.b1024.b64 [%0], {index}, %1;"
                 :: "l"(desc), "l"(value) : "memory");
}}
"""


_GDN_RELEASE_SOURCE = r"""__device__ __forceinline__ void gdn_tensormap_release() {
    asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
}
"""

_GDN_ACQUIRE_SOURCE = r"""__device__ __forceinline__ void gdn_tensormap_acquire(const void *desc) {
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
                 :: "l"(desc) : "memory");
}
"""


@T.inline
def _descriptor_tma_load_body(descriptor, shared, barriers, output):
    T.evaluate(
        T.ptx[
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
        ](T.address_of(shared[0, 0]), descriptor, 0, 0, T.address_of(barriers[0]))
    )
    T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 48)
    T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    for row in T.serial(3):
        for column in T.serial(4):
            output[row, column] = shared[row, column]


def _replace_global_dim(descriptor, index: int, value):
    return T.cuda.func_call(
        f"gdn_tensormap_replace_global_dim_{index}",
        descriptor,
        value,
        source_code=_gdn_replace_dim_source(index),
        return_type="void",
    )


def _replace_global_stride(descriptor, index: int, value):
    return T.cuda.func_call(
        f"gdn_tensormap_replace_global_stride_{index}",
        descriptor,
        value,
        source_code=_gdn_replace_stride_source(index),
        return_type="void",
    )


@T.inline
def _copy_descriptor_payload(source_map, descriptor):
    payload = T.decl_buffer(
        (8,),
        "uint64",
        data=T.reinterpret("handle", T.address_of(source_map)),
        scope="param",
        align=16,
    )
    T.ptx.st.global_.v4.b64(descriptor, payload[0], payload[1], payload[2], payload[3])
    T.ptx.st.global_.v4.b64(
        T.reinterpret("handle", T.reinterpret("uint64", descriptor) + T.uint64(32)),
        payload[4],
        payload[5],
        payload[6],
        payload[7],
    )


@T.inline
def _copy_raw_descriptor_payload_with_scalar_stores(source_descriptor_storage, descriptor):
    payload = T.alloc_buffer((4,), "uint64", scope="local")
    source = T.reinterpret("uint64", source_descriptor_storage.ptr_to([0]))
    target = T.reinterpret("uint64", descriptor)
    for group in range(2):
        offset = T.uint64(group * 32)
        T.ptx.ld.global_.v4.b64(
            payload[0],
            payload[1],
            payload[2],
            payload[3],
            T.reinterpret("handle", source + offset),
        )
        for word in range(4):
            T.ptx.st.global_.b64(
                T.reinterpret("handle", target + T.uint64((group * 4 + word) * 8)),
                payload[word],
            )


@T.prim_func
def host_descriptor_tma_load(
    source: T.Buffer((3, 4), "float32"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_storage: T.Buffer((16,), "int64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    descriptor = descriptor_storage.ptr_to([0])
    if lane == 0:
        T.cuda.func_call(
            "flashkda_tensormap_acquire",
            descriptor,
            source_code=_FLASHKDA_ACQUIRE_SOURCE,
            return_type="void",
        )
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        _descriptor_tma_load_body(descriptor, shared, barriers, output)


@T.prim_func
def host_descriptor_tma_load_without_acquire(
    source: T.Buffer((3, 4), "float32"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_storage: T.Buffer((16,), "int64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    descriptor = descriptor_storage.ptr_to([0])
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        _descriptor_tma_load_body(descriptor, shared, barriers, output)


@T.prim_func
def host_descriptor_tma_load_to_raw_shared_offset(
    source: T.Buffer((3, 4), "float32"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_storage: T.Buffer((16,), "int64"),
    destination_row: T.int32,
    barrier_index: T.int32,
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((11, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    descriptor = descriptor_storage.ptr_to([0])
    shared_base: T.uint32 = T.cuda.cvta_generic_to_shared(shared.ptr_to([0, 0]))
    destination = shared_base + T.cast(destination_row * 16, "uint32")
    barrier_base: T.uint32 = T.cuda.cvta_generic_to_shared(barriers.ptr_to([0]))
    transaction_barrier = barrier_base + T.cast(barrier_index * 8, "uint32")
    if lane == 0:
        T.cuda.func_call(
            "flashkda_tensormap_acquire",
            descriptor,
            source_code=_FLASHKDA_ACQUIRE_SOURCE,
            return_type="void",
        )
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[barrier_index]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](destination, descriptor, 0, 0, transaction_barrier)
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(
            barrier_base + T.cast(barrier_index * 8, "uint32"), 48
        )
        T.cuda.mbarrier_wait(T.address_of(barriers[barrier_index]), 0)
        for row in T.serial(3):
            for column in T.serial(4):
                output[row, column] = shared[row + destination_row, column]


@T.prim_func
def raw_copied_and_replaced_descriptor_tma_load(
    source_descriptor_storage: T.Buffer((16,), "int64"),
    source: T.Buffer((3, 4), "float32"),
    replacement: T.Buffer((3, 4), "float32"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_workspace: T.Buffer((16,), "int64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    payload = T.alloc_buffer((4,), "uint64", scope="local")
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    source_descriptor = T.reinterpret("uint64", source_descriptor_storage.ptr_to([0]))
    descriptor = descriptor_workspace.ptr_to([0])
    destination_descriptor = T.reinterpret("uint64", descriptor)
    if lane == 0:
        for chunk in T.serial(2):
            byte_offset = T.cast(chunk * 32, "uint64")
            T.ptx.ld.global_.v4.b64(
                payload[0],
                payload[1],
                payload[2],
                payload[3],
                T.reinterpret("handle", source_descriptor + byte_offset),
            )
            T.ptx.st.global_.v4.b64(
                T.reinterpret("handle", destination_descriptor + byte_offset),
                payload[0],
                payload[1],
                payload[2],
                payload[3],
            )
        T.cuda.func_call(
            "gdn_tensormap_replace_global_address",
            descriptor,
            replacement.data,
            source_code=_GDN_REPLACE_ADDRESS_SOURCE,
            return_type="void",
        )
        _replace_global_dim(descriptor, 0, T.uint32(4))
        _replace_global_dim(descriptor, 1, T.uint32(3))
        _replace_global_dim(descriptor, 2, T.uint32(1))
        _replace_global_dim(descriptor, 3, T.uint32(1))
        _replace_global_dim(descriptor, 4, T.uint32(1))
        _replace_global_stride(descriptor, 0, T.uint64(16))
        _replace_global_stride(descriptor, 1, T.uint64(0))
        _replace_global_stride(descriptor, 2, T.uint64(0))
        _replace_global_stride(descriptor, 3, T.uint64(0))
        T.cuda.func_call(
            "gdn_tensormap_release",
            source_code=_GDN_RELEASE_SOURCE,
            return_type="void",
        )
        T.cuda.func_call(
            "gdn_tensormap_acquire",
            descriptor,
            source_code=_GDN_ACQUIRE_SOURCE,
            return_type="void",
        )
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        _descriptor_tma_load_body(descriptor, shared, barriers, output)


@T.prim_func
def raw_descriptor_scalar_store_copy_tma_load(
    source_descriptor_storage: T.Buffer((16,), "int64"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_workspace: T.Buffer((16,), "int64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    descriptor = descriptor_workspace.ptr_to([0])
    if lane == 0:
        _copy_raw_descriptor_payload_with_scalar_stores(source_descriptor_storage, descriptor)
        T.cuda.func_call(
            "gdn_tensormap_release",
            source_code=_GDN_RELEASE_SOURCE,
            return_type="void",
        )
        T.cuda.func_call(
            "gdn_tensormap_acquire",
            descriptor,
            source_code=_GDN_ACQUIRE_SOURCE,
            return_type="void",
        )
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        _descriptor_tma_load_body(descriptor, shared, barriers, output)


@T.prim_func
def incomplete_raw_descriptor_copy_tma_load(
    source_descriptor_storage: T.Buffer((16,), "int64"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_workspace: T.Buffer((16,), "int64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    payload = T.alloc_buffer((4,), "uint64", scope="local")
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    source_descriptor = T.reinterpret("uint64", source_descriptor_storage.ptr_to([0]))
    descriptor = descriptor_workspace.ptr_to([0])
    destination_descriptor = T.reinterpret("uint64", descriptor)
    if lane == 0:
        for chunk in T.serial(1):
            byte_offset = T.cast(chunk * 32, "uint64")
            T.ptx.ld.global_.v4.b64(
                payload[0],
                payload[1],
                payload[2],
                payload[3],
                T.reinterpret("handle", source_descriptor + byte_offset),
            )
            T.ptx.st.global_.v4.b64(
                T.reinterpret("handle", destination_descriptor + byte_offset),
                payload[0],
                payload[1],
                payload[2],
                payload[3],
            )
        T.cuda.func_call(
            "gdn_tensormap_release",
            source_code=_GDN_RELEASE_SOURCE,
            return_type="void",
        )
        T.cuda.func_call(
            "gdn_tensormap_acquire",
            descriptor,
            source_code=_GDN_ACQUIRE_SOURCE,
            return_type="void",
        )
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        _descriptor_tma_load_body(descriptor, shared, barriers, output)


@T.prim_func
def raw_descriptor_96b_copy_tma_load(
    source_descriptor_storage: T.Buffer((16,), "int64"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_workspace: T.Buffer((16,), "int64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    payload = T.alloc_buffer((4,), "uint64", scope="local")
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    source_descriptor = T.reinterpret("uint64", source_descriptor_storage.ptr_to([0]))
    descriptor = descriptor_workspace.ptr_to([0])
    destination_descriptor = T.reinterpret("uint64", descriptor)
    if lane == 0:
        for chunk in T.serial(3):
            byte_offset = T.cast(chunk * 32, "uint64")
            T.ptx.ld.global_.v4.b64(
                payload[0],
                payload[1],
                payload[2],
                payload[3],
                T.reinterpret("handle", source_descriptor + byte_offset),
            )
            T.ptx.st.global_.v4.b64(
                T.reinterpret("handle", destination_descriptor + byte_offset),
                payload[0],
                payload[1],
                payload[2],
                payload[3],
            )
        T.cuda.func_call(
            "gdn_tensormap_release",
            source_code=_GDN_RELEASE_SOURCE,
            return_type="void",
        )
        T.cuda.func_call(
            "gdn_tensormap_acquire",
            descriptor,
            source_code=_GDN_ACQUIRE_SOURCE,
            return_type="void",
        )
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        _descriptor_tma_load_body(descriptor, shared, barriers, output)


@T.prim_func
def host_descriptor_tma_store(
    source: T.Buffer((3, 4), "float32"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_storage: T.Buffer((16,), "int64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    descriptor = descriptor_storage.ptr_to([0])
    if lane < 12:
        shared[lane // 4, lane % 4] = source[lane // 4, lane % 4]
    T.cuda.warp_sync()
    if lane == 0:
        T.cuda.func_call(
            "gdn_tensormap_acquire",
            descriptor,
            source_code=_GDN_ACQUIRE_SOURCE,
            return_type="void",
        )
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
                descriptor, 0, 0, T.address_of(shared[0, 0])
            )
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def conflicting_host_descriptor_slots(
    output: T.Buffer((3, 4), "float32"),
    descriptor_storage_a: T.Buffer((128,), "uint8"),
    descriptor_storage_b: T.Buffer((128,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    descriptor = descriptor_storage_a.ptr_to([0])
    if lane == 0:
        T.cuda.func_call(
            "flashkda_tensormap_acquire",
            descriptor,
            source_code=_FLASHKDA_ACQUIRE_SOURCE,
            return_type="void",
        )
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        _descriptor_tma_load_body(descriptor, shared, barriers, output)


def _tensor_map(
    base: np.ndarray,
) -> np.ndarray:
    return numsim.TensorMap(
        base=base,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 3),
        element_strides=(1, 1),
    ).numpy()




def _descriptor_storage(image: np.ndarray, dtype) -> np.ndarray:
    """Slot 0 of a 128-byte descriptor storage array holding ``image``: the
    legacy ``_descriptor_storage(storage=np.zeros(...), slots={0: image})``.
    A view keeps the host descriptor's base for the v2 binder."""

    assert image.nbytes == 128
    return image.view(dtype)


def _host_descriptor_bindings(source: np.ndarray, output: np.ndarray):
    return {
        "source": source,
        "output": output,
        "descriptor_storage": _descriptor_storage(_tensor_map(source), np.int64),
    }


def _first_stop(error: v2.ExecutionError) -> dict:
    stops = [
        d
        for d in error.diagnostics
        if d.get("status") in ("error", "incomplete") and d.get("reason") != "subset_execution"
    ]
    assert stops, error.diagnostics
    return stops[0]


def _assert_error_kind(excinfo, kinds) -> None:
    stop = _first_stop(excinfo.value)
    assert stop["status"] == "error", stop
    assert stop["kind"] in set(kinds), stop


def test_host_bound_descriptor_acquire_enables_raw_tma():
    """Port of ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py::test_host_bound_descriptor_acquire_enables_raw_tma``.

    Dropped: legacy ``_descriptor_storage`` (the image is bound directly as
    the int64[16] storage)."""

    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    module = v2.transpile(host_descriptor_tma_load)

    result = v2.Engine().run(module, _host_descriptor_bindings(source, output), outputs=("output",))

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_host_bound_descriptor_without_acquire_fails_closed():
    """Port of ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py::test_host_bound_descriptor_without_acquire_fails_closed``.

    Delta-asserted, ``docs/development/racecheck-behaviour-deltas.md`` row
    I1: a TensorMap consumed without ``fence.proxy.tensormap::generic.acquire``
    is no longer a hard NumSim ``Err`` that aborts the run (legacy
    ``NumSimExecutionError`` "not acquired"); NumSim runs it to completion
    and the ordering question belongs to Racecheck. Dropped: legacy
    ``_descriptor_storage``. Not asserted: the Racecheck verdict (v2 reports
    this host-written descriptor clean; I1 names only device release/acquire
    pairs).
    """

    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    module = v2.transpile(host_descriptor_tma_load_without_acquire)

    result = v2.Engine().run(module, _host_descriptor_bindings(source, output), outputs=("output",))

    assert result.status.get("kind") == "completed", result.status
    np.testing.assert_array_equal(result.outputs["output"], source)


def test_raw_shared_address_binding_and_offset_preserve_tma_destination():
    """Port of ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py::test_raw_shared_address_binding_and_offset_preserve_tma_destination``.

    Dropped: legacy ``_descriptor_storage``."""

    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    module = v2.transpile(host_descriptor_tma_load_to_raw_shared_offset)

    result = v2.Engine().run(
        module,
        {**_host_descriptor_bindings(source, output), "destination_row": 8, "barrier_index": 1},
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_raw_shared_address_outside_declared_views_fails_closed():
    """Port of ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py::test_raw_shared_address_outside_declared_views_fails_closed``.

    Dropped: legacy ``_descriptor_storage`` and the ``match="does not name a
    declared shared-memory view"`` wording; v2 kind ``out_of_bounds``."""

    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    module = v2.transpile(host_descriptor_tma_load_to_raw_shared_offset)

    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(
            module,
            {**_host_descriptor_bindings(source, output), "destination_row": 100, "barrier_index": 1},
            outputs=("output",),
        )
    _assert_error_kind(excinfo, {"out_of_bounds"})


def test_raw_descriptor_copy_replace_release_and_acquire_drive_raw_tma():
    """Port of ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py::test_raw_descriptor_copy_replace_release_and_acquire_drive_raw_tma``.

    Dropped: legacy ``_descriptor_storage``."""

    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    replacement = (100 + np.arange(12, dtype=np.float32)).reshape(3, 4)
    output = np.zeros_like(source)
    module = v2.transpile(raw_copied_and_replaced_descriptor_tma_load)

    result = v2.Engine().run(
        module,
        {
            "source_descriptor_storage": _descriptor_storage(_tensor_map(source), np.int64),
            "source": source,
            "replacement": replacement,
            "output": output,
            "descriptor_workspace": np.full(16, -1, dtype=np.int64),
        },
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], replacement)


def test_raw_descriptor_v4_load_scalar_stores_drive_raw_tma():
    """Port of ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py::test_raw_descriptor_v4_load_scalar_stores_drive_raw_tma``.

    Dropped: legacy ``_descriptor_storage``."""

    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    module = v2.transpile(raw_descriptor_scalar_store_copy_tma_load)

    result = v2.Engine().run(
        module,
        {
            "source_descriptor_storage": _descriptor_storage(_tensor_map(source), np.int64),
            "output": output,
            "descriptor_workspace": np.zeros(16, dtype=np.int64),
        },
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_96b_descriptor_copy_ignores_tail_and_drives_raw_tma():
    """Port of ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py::test_96b_descriptor_copy_ignores_tail_and_drives_raw_tma``.

    Dropped: legacy ``_descriptor_storage``."""

    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    module = v2.transpile(raw_descriptor_96b_copy_tma_load)

    result = v2.Engine().run(
        module,
        {
            "source_descriptor_storage": _descriptor_storage(_tensor_map(source), np.int64),
            "output": output,
            "descriptor_workspace": np.full(16, -1, dtype=np.int64),
        },
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_32b_raw_descriptor_copy_fails_when_tma_decodes_missing_metadata():
    """Port of ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py::test_32b_raw_descriptor_copy_fails_when_tma_decodes_missing_metadata``.

    Dropped: legacy ``_descriptor_storage`` and the ``match="image has
    invalid magic"`` wording; the fail-closed error stop is kept."""

    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    module = v2.transpile(incomplete_raw_descriptor_copy_tma_load)

    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(
            module,
            {
                "source_descriptor_storage": _descriptor_storage(_tensor_map(source), np.int64),
                "output": output,
                "descriptor_workspace": np.zeros(16, dtype=np.int64),
            },
            outputs=("output",),
        )
    _assert_error_kind(excinfo, {"invalid_operand"})


def test_host_descriptor_store_updates_declared_base():
    """Port of ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py::test_host_descriptor_store_updates_declared_base``.

    Dropped: legacy ``_descriptor_storage``."""

    source = np.arange(12, dtype=np.float32).reshape(3, 4) + np.float32(0.25)
    output = np.zeros_like(source)
    module = v2.transpile(host_descriptor_tma_store)

    result = v2.Engine().run(
        module,
        {
            "source": source,
            "output": output,
            "descriptor_storage": _descriptor_storage(_tensor_map(output), np.int64),
        },
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_conflicting_host_descriptor_slots_fail_closed():
    """Port of ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py::test_conflicting_host_descriptor_slots_fail_closed``.

    Legacy bound one host storage array (descriptor image in slot 0) to two
    parameters and expected ``NumSimExecutionError`` "already bound".
    v2 fails closed with an ``ExecutionError`` (kind ``bad_address``: the
    aliased storage is not resolved to a mapped descriptor). Dropped: legacy
    ``_descriptor_storage`` and the wording."""

    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    storage = _descriptor_storage(_tensor_map(source), np.uint8)
    module = v2.transpile(conflicting_host_descriptor_slots)

    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(
            module,
            {"output": output, "descriptor_storage_a": storage, "descriptor_storage_b": storage},
            outputs=("output",),
        )
    _assert_error_kind(excinfo, {"bad_address"})
