"""v2 copy of ``tests/numsim/runtime/test_raw_tensor_map_descriptors.py::test_replace_rejects_invalid_dimension_index``.

Phase change (W4): ``tensormap.replace.tile.global_dim`` with dimension ordinal
5 (valid: 0..4) is rejected when it executes, as ``invalid_operand``
(checkers: verdict ``error``), not at transpile as legacy did. Kernel and
helpers copied verbatim.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.cases import TensorMap

pytestmark = requires_v2_engine


def _gdn_replace_dim_source(index: int) -> str:
    return f"""__device__ __forceinline__ void gdn_tensormap_replace_global_dim_{index}(void *desc, unsigned int value) {{
    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], {index}, %1;"
                 :: "l"(desc), "r"(value) : "memory");
}}
"""


def _replace_global_dim(descriptor, index: int, value):
    return T.cuda.func_call(
        f"gdn_tensormap_replace_global_dim_{index}",
        descriptor,
        value,
        source_code=_gdn_replace_dim_source(index),
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


@T.prim_func
def invalid_dimension_index_descriptor_update(
    source_map: T.TensorMap(), descriptor_storage: T.Buffer((128,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    descriptor = descriptor_storage.ptr_to([0])
    if lane == 0:
        _copy_descriptor_payload(source_map, descriptor)
        _replace_global_dim(descriptor, 5, T.uint32(1))


def _inputs():
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    tensor_map = TensorMap(
        base=source, global_shape=(4, 3), global_strides=(16,), box_shape=(4, 3), element_strides=(1, 1)
    ).numpy()
    return {"source_map": tensor_map, "descriptor_storage": np.zeros(128, dtype=np.uint8)}


def test_replace_rejects_invalid_dimension_index():
    module = v2.transpile(invalid_dimension_index_descriptor_update)
    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(module, _inputs())
    stops = [d for d in excinfo.value.diagnostics if d.get("status") in ("error", "incomplete")]
    assert stops and stops[0]["status"] == "error" and stops[0]["kind"] == "invalid_operand", stops
    for check in (v2.racecheck, v2.synccheck):
        report = check(invalid_dimension_index_descriptor_update, _inputs())
        assert report.verdict == "error", report.format()
