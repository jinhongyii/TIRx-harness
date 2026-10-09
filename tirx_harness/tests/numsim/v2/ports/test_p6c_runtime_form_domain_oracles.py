"""v2 port of the legacy complete ``T.cuda.ldg`` finite-domain test. The
legacy-only ``numsim.dtype_abi.vector_dtype_abi`` (used only to size the
packed storage of vector dtypes) becomes the itemsize from
``tvm.DataType`` (bits * lanes / 8 for lanes > 1; identical for all 55
dtypes). ``numsim.Engine(max_workers=1)`` becomes ``v2.Engine()``. Kernel
builder and oracles copied verbatim from
``tests/numsim/runtime/test_runtime_form_domain_oracles.py``."""

from __future__ import annotations

import numpy as np
import tvm
from tvm.script import tirx as T

from tests.numsim.support.runtime_domains import CUDA_LDG_DTYPES
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


_NUMPY_SCALAR_DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
}


def _make_complete_cuda_ldg_runtime_kernel():
    parameters = []
    statements = []
    for dtype in CUDA_LDG_DTYPES:
        name = dtype.replace("_", "")
        parameters.extend(
            (
                f'    source_{name}: T.Buffer((1,), "{dtype}"),',
                f'    output_{name}: T.Buffer((1,), "{dtype}"),',
            )
        )
        statements.append(
            f'        output_{name}[0] = T.cuda.ldg(source_{name}.ptr_to([0]), "{dtype}")'
        )
    return tvm.script.from_source(
        "\n".join(
            (
                "@T.prim_func",
                "def cuda_ldg_complete_runtime_domain(",
                *parameters,
                "):",
                "    T.device_entry()",
                "    _warp = T.warp_id([1])",
                "    lane = T.lane_id([32])",
                "    if lane == 0:",
                *statements,
            )
        ),
        extra_vars={"T": T},
    )


CUDA_LDG_COMPLETE_RUNTIME_DOMAIN = _make_complete_cuda_ldg_runtime_kernel()


def _packed_storage(itemsize: int, *, initialized: bool) -> np.ndarray:
    if initialized:
        raw = (np.arange(itemsize, dtype=np.uint16) * np.uint16(37) + np.uint16(11)).astype(
            np.uint8
        )
    else:
        raw = np.zeros(itemsize, dtype=np.uint8)
    if itemsize == 2:
        return raw.view(np.uint16)
    if itemsize == 4:
        return raw.view(np.uint32)
    if itemsize == 8:
        return raw.view(np.uint64)
    if itemsize == 16:
        return raw.view(np.dtype("V16"))
    raise AssertionError(f"unexpected accepted CUDA ldg itemsize {itemsize}")


def _storage(dtype: str, *, initialized: bool) -> tuple[np.ndarray, bool]:
    datatype = tvm.DataType(dtype)
    if datatype.lanes > 1:
        return _packed_storage(datatype.bits * datatype.lanes // 8, initialized=initialized), True
    if dtype == "bfloat16":
        bits = 0x3FC0 if initialized else 0
        return np.array([bits], dtype=np.uint16), True
    numpy_dtype = _NUMPY_SCALAR_DTYPES[dtype]
    value = 1.5 if dtype.startswith("float") else 7
    return np.array([value if initialized else 0], dtype=numpy_dtype), False


def _assert_same_physical_value(actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.dtype.kind == "V":
        np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
    else:
        np.testing.assert_array_equal(actual, expected)


def test_cuda_ldg_complete_finite_domain_executes_with_exact_physical_oracle():
    """Port of ``tests/numsim/runtime/test_runtime_form_domain_oracles.py::test_cuda_ldg_complete_finite_domain_executes_with_exact_physical_oracle``."""

    module = v2.transpile(CUDA_LDG_COMPLETE_RUNTIME_DOMAIN)
    arguments: dict[str, object] = {}
    expected: dict[str, np.ndarray] = {}
    for dtype in CUDA_LDG_DTYPES:
        name = dtype.replace("_", "")
        source, _ = _storage(dtype, initialized=True)
        output, _ = _storage(dtype, initialized=False)
        arguments[f"source_{name}"] = source
        arguments[f"output_{name}"] = output
        expected[f"output_{name}"] = source.copy()

    result = v2.Engine().run(module, arguments)

    assert len(expected) == 55
    assert set(result.outputs) == set(arguments)
    for name, value in expected.items():
        _assert_same_physical_value(result.outputs[name], value)
