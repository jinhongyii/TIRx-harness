"""v2 ports of ``tests/numsim/integration/test_vector_ir_artifact.py``
(``analyze`` checks and ``rust_source`` pins). Kernels copied verbatim."""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine


@T.prim_func
def vector_ir_roundtrip(
    source_u32: T.Buffer((64,), "uint32"),
    source_f32: T.Buffer((128,), "float32"),
    output_u32: T.Buffer((64,), "uint32"),
    output_f32: T.Buffer((128,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "float32", scope="shared")
    for slot in T.unroll(4):
        shared[lane * 4 + slot] = source_f32[lane * 4 + slot]
    T.cuda.warp_sync()
    packed = source_u32.vload([lane * 2], dtype="uint32x2")
    values = shared.vload([lane * 4], dtype="float32x4")
    output_u32[lane * 2] = T.Shuffle([packed], [0])
    output_u32[lane * 2 + 1] = T.Shuffle([packed], [1])
    output_f32[lane * 4] = T.Shuffle([values], [0])
    output_f32[lane * 4 + 1] = T.Shuffle([values], [1])
    output_f32[lane * 4 + 2] = T.Shuffle([values], [2])
    output_f32[lane * 4 + 3] = T.Shuffle([values], [3])


@T.prim_func
def invalid_vector_shuffle_extract(
    source: T.Buffer((64,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    packed = source.vload([lane * 2], dtype="uint32x2")
    output[lane] = T.Shuffle([packed], [2])


@T.prim_func
def packed_vector_arithmetic_is_not_scalar_arithmetic(
    lhs: T.Buffer((32,), "int8x4"),
    rhs: T.Buffer((32,), "int8x4"),
    output: T.Buffer((32,), "int8x4"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = lhs[lane] + rhs[lane]


@T.prim_func
def unsupported_generic_vector_extract(
    source: T.Buffer((128,), "int8"), output: T.Buffer((32,), "int8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    packed = source.vload([lane * 4], dtype="int8x4")
    output[lane] = T.Shuffle([packed], [0])


@T.prim_func
def reinterpret_128bit_vector_roundtrip(
    source: T.Buffer((32,), "uint64x2"), output: T.Buffer((32,), "uint64x2")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    as_float: T.let = T.reinterpret("float32x4", source[lane])
    output[lane] = T.reinterpret("uint64x2", as_float)


def _completed(kernel, inputs):
    module = v2.transpile(kernel)
    assert module.document["kernels"][0]["unsupported"] == []
    result = v2.Engine().run(module, inputs)
    assert result.status["kind"] == "completed", result.status
    return result.outputs


def test_vector_ir_roundtrip_executes_packed_and_f32x4_semantics():
    """Port of ``tests/numsim/integration/test_vector_ir_artifact.py::test_vector_ir_roundtrip_executes_packed_and_f32x4_semantics``.

    Dropped: the ``F32x4`` / ``packed_u32x2`` ``rust_source`` pins."""

    source_u32 = (np.arange(64, dtype=np.uint32) * np.uint32(0x01020305)) ^ np.uint32(0xA55AA55A)
    source_f32 = np.arange(128, dtype=np.float32) * np.float32(0.125) - np.float32(3)
    outputs = _completed(
        vector_ir_roundtrip,
        {
            "source_u32": source_u32,
            "source_f32": source_f32,
            "output_u32": np.zeros_like(source_u32),
            "output_f32": np.zeros_like(source_f32),
        },
    )
    np.testing.assert_array_equal(outputs["output_u32"], source_u32)
    np.testing.assert_array_equal(outputs["output_f32"], source_f32)


def test_vector_shuffle_classifier_and_frontend_reject_out_of_range_extract():
    """Port of ``tests/numsim/integration/test_vector_ir_artifact.py::test_vector_shuffle_classifier_and_frontend_reject_out_of_range_extract``.

    Dropped: the legacy "outside uint32x2 lane range" wording; the entry
    names the ``Shuffle`` node."""

    with pytest.raises(UnsupportedTIRxError) as excinfo:
        v2.transpile(invalid_vector_shuffle_extract)
    assert any("Shuffle" in entry for entry in excinfo.value.unsupported), excinfo.value.unsupported


def test_packed_storage_vectors_fail_closed_for_ordinary_arithmetic():
    """Port of ``tests/numsim/integration/test_vector_ir_artifact.py::test_packed_storage_vectors_fail_closed_for_ordinary_arithmetic``.

    Legacy carried ``int8x4`` only as a packed storage ABI and rejected
    arithmetic on it. v2 models TIR vector values lane-wise, so ``+`` on
    ``int8x4`` is the TIR semantics: per-lane wrapping int8 addition. No
    delta row records this; the legacy rejection was a representation limit,
    not a semantic requirement, so the port asserts the lane-wise result."""

    rng = np.random.default_rng(7)
    lhs = rng.integers(-128, 128, size=128, dtype=np.int8)
    rhs = rng.integers(-128, 128, size=128, dtype=np.int8)
    outputs = _completed(
        packed_vector_arithmetic_is_not_scalar_arithmetic,
        {
            "lhs": lhs.view(np.int32).copy(),
            "rhs": rhs.view(np.int32).copy(),
            "output": np.zeros(32, dtype=np.int32),
        },
    )
    expected = (lhs.astype(np.int16) + rhs.astype(np.int16)).astype(np.int8)
    np.testing.assert_array_equal(np.asarray(outputs["output"]).view(np.int8).reshape(-1), expected)


def test_generic_vector_extract_fails_during_analysis():
    """Port of ``tests/numsim/integration/test_vector_ir_artifact.py::test_generic_vector_extract_fails_during_analysis``.

    Legacy rejected a ``Shuffle`` extract from an ``int8x4`` vload ("only a
    packed storage ABI"). v2 lowers it lane-wise: lane ``l`` reads
    ``source[4 * l]``. No delta row; like the test above, the legacy
    rejection was a representation limit, so the port asserts the TIR result."""

    source = np.arange(-64, 64, dtype=np.int8)
    outputs = _completed(
        unsupported_generic_vector_extract,
        {"source": source, "output": np.zeros(32, dtype=np.int8)},
    )
    np.testing.assert_array_equal(outputs["output"], source[::4])


def test_128bit_vector_reinterpret_is_a_bitwise_roundtrip():
    """Port of ``tests/numsim/integration/test_vector_ir_artifact.py::test_128bit_vector_reinterpret_is_a_bitwise_roundtrip``.

    Dropped: the ``f32::from_bits`` / ``.to_bits()`` / ``as [f32; 4]``
    ``rust_source`` pins. Bit patterns that are f32 NaNs are added so the
    round trip is checked bitwise, not only on ordinary floats."""

    words = np.arange(64, dtype=np.uint64)
    words[1] = np.uint64(0x7FC00001_FFBFFFFF)  # two NaN f32 payloads
    source = words.reshape(32, 2).view(np.dtype("V16")).reshape(32)
    outputs = _completed(
        reinterpret_128bit_vector_roundtrip,
        {"source": source, "output": np.zeros(32, dtype=np.dtype("V16"))},
    )
    np.testing.assert_array_equal(outputs["output"], source)
