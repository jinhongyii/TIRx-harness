"""v2 copy of ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py::test_ordinary_copy_rejects_noncanonical_tensor_map_candidates``.

The legacy test asserted that the legacy binder's descriptor discovery
(``numsim.bindings.prepare_bindings(...).descriptor_allocations``) found no
descriptor in descriptor-like scratch bytes with a noncanonical inactive axis
or a null host base, then that an ordinary ``ld/st.global.v4.b64`` copy moved
the bytes verbatim with clean checkers.

v2 has no byte-pattern descriptor discovery: a buffer argument is a host
descriptor image only when it carries ``TensorMap`` metadata. The legacy
binding-payload pin is replaced by the public binder: ``v2.canonicalize_inputs``
binds ``source`` as a plain buffer (no descriptor base, no pointer patch). The
clean synccheck/racecheck verdicts and the verbatim-copy oracle are kept. The
kernel is copied verbatim; ``cache_dir`` is dropped.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import assert_clean, requires_v2_engine
from tirx_harness.numsim import TensorMap, v2

pytestmark = requires_v2_engine


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


def _tensor_map(base: np.ndarray) -> np.ndarray:
    return TensorMap(
        base=base,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 3),
        element_strides=(1, 1),
    ).numpy()


@pytest.mark.parametrize(
    ("host_address", "inactive_dimension"),
    [(False, 2), (True, 2), (True, 1)],
    ids=["runtime-inactive-dimension", "host-inactive-dimension", "null-host-address"],
)
def test_ordinary_copy_rejects_noncanonical_tensor_map_candidates(host_address, inactive_dimension):
    """Port of ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py::test_ordinary_copy_rejects_noncanonical_tensor_map_candidates``
    (dropped: legacy ``prepare_bindings().descriptor_allocations``; replaced by
    the v2 binder's plain-buffer binding of ``source``)."""

    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    data = np.asarray(_tensor_map(base)).copy()
    # Discovery rejects both noncanonical inactive axes and a null host base.
    # Otherwise descriptor-like scratch bytes must remain ordinary data.
    data[24:28] = np.frombuffer(inactive_dimension.to_bytes(4, "little"), dtype=np.uint8)
    address = 1 << 63 if inactive_dimension != 1 else 0
    data[:8] = np.frombuffer(address.to_bytes(8, "little"), dtype=np.uint8)
    if not host_address:
        data[60] &= np.uint8(0xDF)
    source = data.view(np.int64)

    def inputs():
        return {"source": source.copy(), "output": np.zeros_like(source)}

    module = v2.transpile(ordinary_u64x4_copy)
    bound = v2.canonicalize_inputs(module, inputs())
    assert bound["source"].kind == "buffer"
    assert bound["source"].base is None and bound["source"].patch_offset is None

    for checker in (v2.synccheck, v2.racecheck):
        assert_clean(checker(ordinary_u64x4_copy, inputs()))
    result = v2.Engine().run(module, inputs(), outputs=("output",))
    assert result.verdict == "clean"
    np.testing.assert_array_equal(result.outputs["output"], source)
