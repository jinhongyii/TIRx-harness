"""v2 port of ``tests/numsim/integration/test_opaque_helper_artifact.py::test_opaque_or_spoofed_cuda_helpers_are_rejected``."""

from __future__ import annotations

import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine


@T.prim_func
def opaque_value_helper(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.cuda.func_call(
        "opaque_fdividef",
        T.float32(1),
        T.cast(lane + 1, "float32"),
        source_code="float opaque_fdividef(float, float);",
        return_type="float32",
    )


@T.prim_func
def opaque_statement_helper():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.evaluate(
            T.cuda.func_call(
                "tvm_builtin_tcgen05_mma_mxf4_block32_ss",
                source_code="void tvm_builtin_tcgen05_mma_mxf4_block32_ss();",
                return_type="void",
            )
        )


@pytest.mark.parametrize(
    "kernel",
    [opaque_value_helper, opaque_statement_helper],
    ids=["opaque_value_helper", "opaque_statement_helper"],
)
def test_opaque_or_spoofed_cuda_helpers_are_rejected(kernel):
    """Port of ``tests/numsim/integration/test_opaque_helper_artifact.py::test_opaque_or_spoofed_cuda_helpers_are_rejected``.

    ``analyze``/``verify``/legacy ``transpile`` rejections collapse into one
    ``v2.transpile`` ``UnsupportedTIRxError`` with exactly one ``func_call``
    entry. Dropped: the legacy ``tirx.cuda.func_call`` entry spelling."""

    with pytest.raises(UnsupportedTIRxError) as excinfo:
        v2.transpile(kernel)
    unsupported = [entry for entry in excinfo.value.unsupported if "func_call" in entry]
    assert len(unsupported) == 1, excinfo.value.unsupported
