"""v2 ports of the ``analyze``/``verify`` functions in
``tests/numsim/integration/test_stmatrix_artifact.py``. Kernels are copied
verbatim; legacy ``analyze(...).unsupported == ()`` is a successful
``v2.transpile`` with an empty ``unsupported`` list."""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm import tirx
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def raw_stmatrix_layout(
    output_b8: T.Buffer((128,), "uint8"), output_b16: T.Buffer((256,), "uint16")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_b8 = T.alloc_buffer((8, 16), "uint8", scope="shared")
    shared_b16 = T.alloc_buffer((4, 8, 8), "uint16", scope="shared")
    source_b8 = T.alloc_buffer((1,), "uint32", scope="local")
    source_b16 = T.alloc_buffer((4,), "uint32", scope="local")
    source_b8[0] = (
        T.cast(lane * 4, "uint32")
        | T.shift_left(T.cast(lane * 4 + 1, "uint32"), T.uint32(8))
        | T.shift_left(T.cast(lane * 4 + 2, "uint32"), T.uint32(16))
        | T.shift_left(T.cast(lane * 4 + 3, "uint32"), T.uint32(24))
    )
    for matrix in T.unroll(0, 4):
        low = T.cast(matrix * 1000 + lane * 2, "uint32")
        high = T.cast(matrix * 1000 + lane * 2 + 1, "uint32")
        source_b16[matrix] = low | T.shift_left(high, T.uint32(16))
    T.ptx.stmatrix.sync.aligned.m16n8.x1.trans.shared.b8(
        T.address_of(shared_b8[lane % 8, 0]),
        source_b8[0],
    )
    T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
        T.address_of(shared_b16[lane // 8, lane % 8, 0]),
        source_b16[0],
        source_b16[1],
        source_b16[2],
        source_b16[3],
    )
    T.cuda.warp_sync()
    for index in T.unroll(0, 4):
        linear = lane * 4 + index
        output_b8[linear] = shared_b8[linear // 16, linear % 16]
    for index in T.unroll(0, 8):
        linear = lane * 8 + index
        output_b16[linear] = shared_b16[linear // 64, linear % 64 // 8, linear % 8]


@T.prim_func
def raw_stmatrix_x2_forms(
    output_b8: T.Buffer((256,), "uint8"), output_b16: T.Buffer((128,), "uint16")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_b8 = T.alloc_buffer((2, 8, 16), "uint8", scope="shared")
    shared_b16 = T.alloc_buffer((2, 8, 8), "uint16", scope="shared")
    source_b8 = T.alloc_buffer((2,), "uint32", scope="local")
    source_b16 = T.alloc_buffer((2,), "uint32", scope="local")
    for matrix in T.unroll(0, 2):
        source_b8[matrix] = (
            T.cast(matrix * 128 + lane * 4, "uint32")
            | T.shift_left(T.cast(matrix * 128 + lane * 4 + 1, "uint32"), T.uint32(8))
            | T.shift_left(T.cast(matrix * 128 + lane * 4 + 2, "uint32"), T.uint32(16))
            | T.shift_left(T.cast(matrix * 128 + lane * 4 + 3, "uint32"), T.uint32(24))
        )
        low = T.cast(matrix * 1000 + lane * 2, "uint32")
        high = T.cast(matrix * 1000 + lane * 2 + 1, "uint32")
        source_b16[matrix] = low | T.shift_left(high, T.uint32(16))
    T.ptx.stmatrix.sync.aligned.m16n8.x2.trans.shared__cta.b8(
        T.address_of(shared_b8[lane // 8, lane % 8, 0]),
        source_b8[0],
        source_b8[1],
    )
    T.ptx.stmatrix.sync.aligned.m8n8.x2.shared__cta.b16(
        T.address_of(shared_b16[lane // 8, lane % 8, 0]),
        source_b16[0],
        source_b16[1],
    )
    T.cuda.warp_sync()
    for index in T.unroll(0, 8):
        linear = lane * 8 + index
        output_b8[linear] = shared_b8[linear // 128, linear % 128 // 16, linear % 16]
    for index in T.unroll(0, 4):
        linear = lane * 4 + index
        output_b16[linear] = shared_b16[linear // 64, linear % 64 // 8, linear % 8]


def _stmatrix_chain(count, space, shape, dtype, transpose):
    chain = f"stmatrix.sync.aligned.{shape}.x{count}"
    if transpose:
        chain += ".trans"
    if space:
        chain += f".{space}"
    chain += dtype
    return chain


def _transpile_stmatrix_form(count, space, shape, dtype, transpose):
    chain = _stmatrix_chain(count, space, shape, dtype, transpose)
    parameters = ", ".join(f"value_{index}: T.uint32" for index in range(count))
    operands = ", ".join(f"value_{index}" for index in range(count))
    source = f'''
@T.prim_func
def store_matrix({parameters}):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32, 16), "uint8", scope="shared")
    T.ptx["{chain}"](T.address_of(shared[lane, 0]), {operands})
'''
    return v2.transpile(tvm.script.from_source(source, extra_vars={"T": T}))


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("space", ["shared", "shared::cta"])
@pytest.mark.parametrize(
    ("shape", "dtype", "transpose"),
    [("m8n8", ".b16", False), ("m8n8", ".b16", True), ("m16n8", ".b8", True)],
)
def test_stmatrix_registry_accepts_reviewed_shared_forms(count, space, shape, dtype, transpose):
    """Port of ``tests/numsim/integration/test_stmatrix_artifact.py::test_stmatrix_registry_accepts_reviewed_shared_forms`` (same parametrization)."""

    module = _transpile_stmatrix_form(count, space, shape, dtype, transpose)
    assert module.document["kernels"][0]["unsupported"] == []


def test_stmatrix_analyze_rejects_a_pointer_without_backing():
    """Port of ``tests/numsim/integration/test_stmatrix_artifact.py::test_stmatrix_analyze_rejects_a_pointer_without_backing``.

    Legacy rejected the unbacked ``handle`` operand at analysis ("Unbound
    TIRx variable"). v2 binds a ``handle`` parameter as a ``Pointer`` host
    input, so the kernel transpiles; it still fails closed when run: with no
    binding the run is an ``InputError`` (missing binding), and an integer
    binding that names no allocation is an ``out_of_bounds`` (or
    ``bad_address``) execution error.
    No delta row records the move from analysis to launch."""

    pointer = tirx.Var("pointer", "handle")
    value = tirx.Var("value", "uint32")
    call = T.ptx.stmatrix.sync.aligned.m8n8.x1.shared.b16(pointer, value)
    module = v2.transpile(tirx.PrimFunc([pointer, value], tirx.Evaluate(call)))

    with pytest.raises(v2.InputError):
        v2.Engine().run(module, {"value": 1})
    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(module, {"pointer": 0, "value": 1})
    stops = [d for d in excinfo.value.diagnostics if d.get("status") == "error"]
    assert stops and stops[0]["kind"] in {"out_of_bounds", "bad_address"}, excinfo.value.diagnostics


def test_stmatrix_registry_accepts_repository_and_x2_forms():
    """Port of ``tests/numsim/integration/test_stmatrix_artifact.py::test_stmatrix_registry_accepts_repository_and_x2_forms``."""

    for kernel in (raw_stmatrix_layout, raw_stmatrix_x2_forms):
        assert v2.transpile(kernel).document["kernels"][0]["unsupported"] == []
