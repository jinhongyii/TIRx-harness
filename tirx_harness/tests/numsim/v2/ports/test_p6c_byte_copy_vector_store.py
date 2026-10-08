"""v2 ports of ``tests/numsim/runtime/test_byte_copy_vector_store.py``
(``test_vector_store_registry_is_exact``, ``test_vector_store_checks_total_access_width_alignment``).

Kernels are copied verbatim.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def generic_vector_store_forms(
    source_u32: T.Buffer((256,), "uint32"),
    destination_u32: T.Buffer((256,), "uint32"),
    source_f64: T.Buffer((64,), "float64"),
    destination_f64: T.Buffer((64,), "float64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    base8 = lane * 8
    T.ptx.st.global_.v8.u32(
        T.address_of(destination_u32[base8]),
        source_u32[base8],
        source_u32[base8 + 1],
        source_u32[base8 + 2],
        source_u32[base8 + 3],
        source_u32[base8 + 4],
        source_u32[base8 + 5],
        source_u32[base8 + 6],
        source_u32[base8 + 7],
    )
    base2 = lane * 2
    T.ptx.st.global_.v2.f64(
        T.address_of(destination_f64[base2]),
        source_f64[base2],
        source_f64[base2 + 1],
    )


@T.prim_func
def misaligned_vector_store(destination: T.Buffer((64,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx.st.global_.v4.b32(
            T.address_of(destination[1]),
            T.uint32(1),
            T.uint32(2),
            T.uint32(3),
            T.uint32(4),
        )


def test_vector_store_registry_is_exact():
    """Port of ``tests/numsim/runtime/test_byte_copy_vector_store.py::test_vector_store_registry_is_exact``.

    Legacy ``analyze(kernel).unsupported == ()`` becomes ``v2.transpile``
    accepting the kernel (it raises ``UnsupportedTIRxError`` on any
    unsupported site) with an empty ``unsupported`` list in the module.
    """

    module = v2.transpile(generic_vector_store_forms)
    assert not module.document["kernels"][0]["unsupported"]


def test_vector_store_checks_total_access_width_alignment():
    """Port of ``tests/numsim/runtime/test_byte_copy_vector_store.py::test_vector_store_checks_total_access_width_alignment``.

    Dropped pin: the legacy message ``"16-byte alignment"``; the port asserts
    the v2 error kind ``misaligned``.
    """

    module = v2.transpile(misaligned_vector_store)
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(module, {"destination": np.zeros(64, dtype=np.uint8)})
    assert _stop(caught.value) == ("error", "misaligned")


def _stop(error: v2.ExecutionError) -> tuple[str, str]:
    stops = [d for d in error.diagnostics if d.get("status") in ("error", "incomplete")]
    assert stops, error.diagnostics
    return stops[0]["status"], stops[0]["kind"]
