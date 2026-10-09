"""v2 copy of ``tests/numsim/runtime/test_scalar_helpers.py::test_mega_scalar_helpers_have_exact_typed_registry_entries``.

Legacy asserted ``transpiler.frontend.analyze(mega_scalar_helpers).unsupported
== ()``: the legacy frontend registry accepted every call of the kernel. The
observable part is that the kernel is accepted for simulation; the copy
asserts ``v2.transpile`` accepts it (no ``UnsupportedTIRxError``) and emits a
site for each of the five calls. Dropped: the legacy registry-entry pin. The
kernel is copied verbatim.
"""

from __future__ import annotations

from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def mega_scalar_helpers(
    packed_lhs: T.Buffer((32,), "uint32"),
    packed_rhs: T.Buffer((32,), "uint32"),
    fp8_values: T.Buffer((32, 4), "float32"),
    masks: T.Buffer((32,), "uint32"),
    bases: T.Buffer((32,), "uint32"),
    offsets: T.Buffer((32,), "int32"),
    accum: T.Buffer((32,), "float32"),
    bf16_values: T.Buffer((32,), "uint16"),
    packed_output: T.Buffer((32, 4), "uint32"),
    float_output: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    packed_output[lane, 0] = T.cuda.hmin2(packed_lhs[lane], packed_rhs[lane])
    packed_output[lane, 1] = T.cuda.hmax2(packed_lhs[lane], packed_rhs[lane])
    packed_output[lane, 2] = T.cuda.fp8x4_e4m3_from_float4(
        fp8_values[lane, 0], fp8_values[lane, 1], fp8_values[lane, 2], fp8_values[lane, 3]
    )
    T.ptx.fns.b32(packed_output[lane, 3], masks[lane], bases[lane], offsets[lane])
    T.ptx.add.rn.f32.bf16(float_output[lane], bf16_values[lane], accum[lane])


def test_mega_scalar_helpers_have_exact_typed_registry_entries():
    """Port of ``tests/numsim/runtime/test_scalar_helpers.py::test_mega_scalar_helpers_have_exact_typed_registry_entries``
    (legacy ``analyze().unsupported == ()`` -> ``v2.transpile`` accepts the kernel)."""

    module = v2.transpile(mega_scalar_helpers)
    ops = [str(site.get("op_name", "")) for site in module.spec.kernels[0].sites]
    assert [op for op in ops if op] == [
        "tirx.cuda.hmin2",
        "tirx.cuda.hmax2",
        "tirx.cuda.fp8x4_e4m3_from_float4",
        "tirx.ptx.fns",
        "tirx.ptx.add",
    ]
