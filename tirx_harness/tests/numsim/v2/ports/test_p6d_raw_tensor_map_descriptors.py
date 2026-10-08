"""v2 copies of three ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py`` functions
that fail under ``NUMSIM_IMPL=v2`` with an "other assertion".

- ``test_raw_descriptor_store_updates_replaced_backing`` -- **port**. v2 stores
  the replaced-backing TMA result correctly (``result.outputs["output"] ==
  source``); legacy additionally asserted the caller's ``output`` array was
  written in place. v2 ``Engine.run`` never mutates its inputs (same ruling as
  the ``test_global_alias_artifact`` port), so the copy asserts the caller's
  arrays are unchanged instead.
- ``test_typed_param_descriptor_scalar_copy_preserves_payload`` -- **delta**.
  The kernel copies the 16 words of a typed ``T.TensorMap()`` parameter's
  image to ``output``. A host ``TensorMap`` image binds as
  ``ArgValue::TensorMapOf`` and the engine re-encodes its global address
  against the base array's *engine* address (``CONTRACT_REQUESTS.md`` W8-3;
  ``docs/development/dev-loop.md`` "v2 binder rules"). Word 0 is therefore the
  engine address, not the host pointer legacy echoed; words 1..15 are the
  image bytes.
- ``test_ordinary_u64x4_copy_does_not_require_descriptor_metadata`` --
  **delta**. A plain ``int64[16]`` copy is verbatim (kept). Legacy fed it a
  ``.copy().view(int64)`` of a ``TensorMap`` image, which keeps the numpy
  subclass and its ``_tensor_map_base`` metadata, and ``np.zeros_like`` of it
  for ``output``. The v2 binder treats any 128-byte buffer argument that
  carries ``TensorMap`` metadata as a host descriptor image and rebinds its
  pointer word against the base array's engine address (same rule,
  ``v2/run.py::canonicalize_inputs``): the metadata-carrying ``source`` is
  copied with word 0 rewritten, and the metadata-carrying all-zero ``output``
  (pointer 0 does not address the base) is rejected with ``InputError``.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import TensorMap, v2

pytestmark = requires_v2_engine


_GDN_REPLACE_ADDRESS_SOURCE = r"""__device__ __forceinline__ void gdn_tensormap_replace_global_address(void *desc, const void *addr) {
    asm volatile("tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;"
                 :: "l"(desc), "l"(addr) : "memory");
}
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


@T.prim_func
def typed_param_descriptor_scalar_copy(
    source_map: T.TensorMap(), output: T.Buffer((16,), "uint64")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    payload = T.decl_buffer(
        (16,),
        "uint64",
        data=T.reinterpret(PointerType(PrimType("uint64"), "param"), T.address_of(source_map)),
        scope="param",
    )
    if lane == 0:
        for word in T.serial(16):
            T.ptx.st.global_.b64(output.ptr_to([word]), payload[word])


@T.prim_func
def ordinary_u64x4_copy(source: T.Buffer((16,), "int64"), output: T.Buffer((16,), "int64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    payload = T.alloc_buffer((4,), "uint64", scope="local")
    source_address = T.reinterpret("uint64", source.ptr_to([0]))
    output_address = T.reinterpret("uint64", output.ptr_to([0]))
    if lane == 0:
        for chunk in T.serial(4):
            byte_offset = T.cast(chunk * 32, "uint64")
            T.ptx.ld.global_.v4.b64(
                payload[0],
                payload[1],
                payload[2],
                payload[3],
                T.reinterpret("handle", source_address + byte_offset),
            )
            T.ptx.st.global_.v4.b64(
                T.reinterpret("handle", output_address + byte_offset),
                payload[0],
                payload[1],
                payload[2],
                payload[3],
            )


@T.prim_func
def copied_and_replaced_descriptor_tma_store(
    output_map: T.TensorMap(),
    source: T.Buffer((3, 4), "float32"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_storage: T.Buffer((128,), "uint8"),
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
        _copy_descriptor_payload(output_map, descriptor)
        T.cuda.func_call(
            "gdn_tensormap_replace_global_address",
            descriptor,
            output.data,
            source_code=_GDN_REPLACE_ADDRESS_SOURCE,
            return_type="void",
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


def _tensor_map(base: np.ndarray) -> np.ndarray:
    return TensorMap(
        base=base,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 3),
        element_strides=(1, 1),
    ).numpy()


def test_typed_param_descriptor_scalar_copy_preserves_payload():
    """Delta copy of ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py::test_typed_param_descriptor_scalar_copy_preserves_payload``
    (CONTRACT_REQUESTS W8-3: the bound image's address word is the engine address)."""
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    descriptor = _tensor_map(source)
    result = v2.Engine().run(
        v2.transpile(typed_param_descriptor_scalar_copy),
        {"source_map": descriptor, "output": np.zeros(16, dtype=np.uint64)},
        outputs=("output",),
    )
    words = np.asarray(descriptor).view(np.uint64)
    output = result.outputs["output"]
    np.testing.assert_array_equal(output[1:], words[1:])
    assert int(output[0]) != int(words[0]) == source.ctypes.data
    assert int(output[0]) != 0 and int(output[0]) % 16 == 0
    assert result.verdict == "clean"


def test_ordinary_u64x4_copy_does_not_require_descriptor_metadata():
    """Delta copy of ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py::test_ordinary_u64x4_copy_does_not_require_descriptor_metadata``
    (v2 binder: 128-byte buffers carrying TensorMap metadata are host descriptor images)."""
    tensor_map_base = np.arange(12, dtype=np.float32).reshape(3, 4)
    image = _tensor_map(tensor_map_base).copy()
    image[64] = 1
    module = v2.transpile(ordinary_u64x4_copy)

    # Plain bytes (no descriptor metadata): copied verbatim, as legacy asserted.
    plain = np.array(np.asarray(image).view(np.int64))
    result = v2.Engine().run(
        module, {"source": plain, "output": np.zeros(16, np.int64)}, outputs=("output",)
    )
    np.testing.assert_array_equal(result.outputs["output"], plain)

    # Metadata-carrying source: the pointer word is rebound to the engine address.
    source = image.view(np.int64)
    assert getattr(source, "_tensor_map_base", None) is not None
    result = v2.Engine().run(
        module, {"source": source, "output": np.zeros(16, np.int64)}, outputs=("output",)
    )
    np.testing.assert_array_equal(result.outputs["output"][1:], np.asarray(source)[1:])
    assert int(result.outputs["output"][0]) != int(np.asarray(source)[0])

    # Legacy's ``np.zeros_like(source)`` keeps the metadata with a null pointer.
    with pytest.raises(v2.InputError, match="does not address its base array"):
        v2.Engine().run(module, {"source": source, "output": np.zeros_like(source)})


def test_raw_descriptor_store_updates_replaced_backing():
    """Port of ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py::test_raw_descriptor_store_updates_replaced_backing``
    (dropped: in-place write of the caller's ``output`` array)."""
    source = np.arange(12, dtype=np.float32).reshape(3, 4) + np.float32(0.5)
    output = np.zeros_like(source)
    output_map = _tensor_map(output)
    output_map_before = output_map.copy()

    result = v2.Engine().run(
        v2.transpile(copied_and_replaced_descriptor_tma_store),
        {
            "output_map": output_map,
            "source": source,
            "output": output,
            "descriptor_storage": np.zeros(128, dtype=np.uint8),
        },
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], source)
    np.testing.assert_array_equal(output, np.zeros_like(source))  # v2 never mutates inputs
    np.testing.assert_array_equal(output_map, output_map_before)
    assert result.verdict == "clean"
