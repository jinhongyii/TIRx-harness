"""v2 ports of ``tests/numsim/runtime/test_cuda_ldg_overloads.py``
(``test_cuda_ldg_public_dtype_contract_is_accepted_or_rejected_exactly``,
``test_cuda_ldg_packed_vectors_preserve_physical_bits``).

Legacy ``analyze(kernel).unsupported`` (empty / naming the dtype) becomes
``v2.transpile`` accepting the kernel / raising ``UnsupportedTIRxError``
(``UnmodeledTIRxFormError`` is a subclass). The legacy
``dtype_abi.vector_dtype_abi(dtype).itemsize`` becomes a local
``lanes * element bytes``. Message texts and ``target_id`` are not pinned.
The kernel builders and dtype sets are copied verbatim.
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.ir.type import PointerType
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine

_FLOAT8_BASE_DTYPES = frozenset(
    {
        "float8_e3m4",
        "float8_e4m3",
        "float8_e4m3b11fnuz",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    }
)
_EXPECTED_CUDA_LDG_DTYPES = frozenset(
    {
        "bfloat16",
        "bfloat16x2",
        "bfloat16x8",
        "float16",
        "float16x2",
        "float16x8",
        "float32",
        "float32x2",
        "float32x4",
        "float64",
        "float64x2",
        "int8",
        "int8x2",
        "int8x4",
        "int8x8",
        "int8x16",
        "int16",
        "int16x2",
        "int16x4",
        "int16x8",
        "int32",
        "int32x2",
        "int32x4",
        "int64",
        "int64x2",
        "uint8",
        "uint8x2",
        "uint8x4",
        "uint8x8",
        "uint8x16",
        "uint16",
        "uint16x2",
        "uint16x4",
        "uint16x8",
        "uint32",
        "uint32x2",
        "uint32x4",
        "uint64",
        "uint64x2",
    }
) | frozenset(f"{dtype}x{lanes}" for dtype in _FLOAT8_BASE_DTYPES for lanes in (8, 16))
_EXPECTED_REJECTED_CUDA_LDG_DTYPES = frozenset(
    {
        "bool",
        "boolx2",
        "boolx4",
        "bfloat16x4",
        "float16x4",
    }
) | frozenset(
    dtype if lanes == 1 else f"{dtype}x{lanes}"
    for dtype in _FLOAT8_BASE_DTYPES
    for lanes in (1, 2, 4)
)
_VALID_UNMODELED_CUDA_LDG_DTYPES = frozenset({"boolx2", "boolx4"})


def _cuda_ldg_kernel(dtype: str, *, source_dtype: str | None = None):
    source_dtype = dtype if source_dtype is None else source_dtype
    function_name = f"cuda_ldg_{source_dtype}_to_{dtype}".replace("x", "_x")
    source = f'''
@T.prim_func
def {function_name}(source: T.Buffer((32,), "{source_dtype}"), output: T.Buffer((32,), "{dtype}")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.cuda.ldg(source.ptr_to([31 - lane]), "{dtype}")
'''
    return tvm.script.from_source(source, extra_vars={"T": T})


def _packed_array(itemsize: int) -> np.ndarray:
    raw = (np.arange(32 * itemsize, dtype=np.uint16) * np.uint16(37) + np.uint16(11)).astype(
        np.uint8
    )
    if itemsize == 2:
        return raw.view(np.uint16)
    if itemsize == 4:
        return raw.view(np.uint32)
    if itemsize == 8:
        return raw.view(np.uint64)
    return raw.view(np.dtype("V16"))


def _cuda_ldg_raw_pointer_kernel(dtype: str):
    pointer = tvm.tirx.Var("pointer", PointerType(tvm.ir.PrimType("uint8"), "global"))
    return tvm.tirx.PrimFunc([pointer], tvm.tirx.Evaluate(T.cuda.ldg(pointer, dtype)))



def _itemsize(dtype: str) -> int:
    base, _, lanes = dtype.rpartition("x") if dtype.rpartition("x")[2].isdigit() else (dtype, "", "1")
    element = 1 if base.startswith("float8") else np.dtype(base).itemsize
    return element * int(lanes)


def test_cuda_ldg_public_dtype_contract_is_accepted_or_rejected_exactly():
    """Port of ``tests/numsim/runtime/test_cuda_ldg_overloads.py::test_cuda_ldg_public_dtype_contract_is_accepted_or_rejected_exactly``
    (accepted half). The rejected half is
    :func:`test_cuda_ldg_public_dtype_contract_rejects_exactly` (v2 gap),
    split out so this passing half is not hidden.
    """

    for dtype in sorted(_EXPECTED_CUDA_LDG_DTYPES):
        v2.transpile(_cuda_ldg_kernel(dtype))


def test_cuda_ldg_public_dtype_contract_rejects_exactly():
    """Rejected half of ``tests/numsim/runtime/test_cuda_ldg_overloads.py::test_cuda_ldg_public_dtype_contract_is_accepted_or_rejected_exactly``."""

    for dtype in sorted(_VALID_UNMODELED_CUDA_LDG_DTYPES):
        with pytest.raises(UnsupportedTIRxError):
            v2.transpile(_cuda_ldg_raw_pointer_kernel(dtype))
    for dtype in sorted(_EXPECTED_REJECTED_CUDA_LDG_DTYPES - _VALID_UNMODELED_CUDA_LDG_DTYPES):
        with pytest.raises(UnsupportedTIRxError):
            v2.transpile(_cuda_ldg_kernel(dtype))
    with pytest.raises(UnsupportedTIRxError):
        v2.transpile(_cuda_ldg_kernel("float32x2", source_dtype="float32"))


@pytest.mark.parametrize(
    "dtype",
    [
        "int8x2",
        "bfloat16x2",
        "int16x4",
        "float8_e4m3fnx8",
        "float32x2",
        "float16x8",
        "float32x4",
        "uint64x2",
    ],
)
def test_cuda_ldg_packed_vectors_preserve_physical_bits(dtype):
    """Port of ``tests/numsim/runtime/test_cuda_ldg_overloads.py::test_cuda_ldg_packed_vectors_preserve_physical_bits``."""

    source = _packed_array(_itemsize(dtype))
    output = np.zeros_like(source)
    module = v2.transpile(_cuda_ldg_kernel(dtype))
    result = v2.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(
        np.asarray(result.outputs["output"]).view(source.dtype).reshape(source.shape), source[::-1]
    )
