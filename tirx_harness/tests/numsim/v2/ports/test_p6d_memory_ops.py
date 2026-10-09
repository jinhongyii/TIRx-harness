"""v2 copies of the two 128-bit ``T.cuda.atomic_cas`` tests of
``tests/numsim/runtime/test_memory_ops.py``:

- ``test_cuda_uint64x2_atomic_cas_compares_and_replaces_one_128bit_value``
- ``test_cuda_128bit_atomic_cas_compares_float_vectors_by_bits``

Both are faithful copies of the legacy assertions (dropped: nothing; the legacy
functions had no pins) and were ``v2_gap`` until W2 fixed 128-bit CAS (now all-or-nothing by bits): v2 lowers the call to one
``Atom{Cas}`` of type ``u64x2`` / ``f32x4``, and ``interp/handlers/mem.rs``
``rmw_bytes`` applies CAS per vector element (8 / 4 bytes) instead of comparing
and replacing the whole 16-byte value. A compare that matches only some
components therefore replaces just those components (a torn CAS).
Filed in ``CONTRACT_REQUESTS.md`` "W9-public-API phase 6 ... internal
other-assertion triage".
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def cuda_atomic_cas_uint64x2(
    cell: T.Buffer((1,), "uint64x2"),
    compares: T.Buffer((3,), "uint64x2"),
    replacements: T.Buffer((3,), "uint64x2"),
    old_values: T.Buffer((3,), "uint64x2"),
    final_value: T.Buffer((1,), "uint64x2"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 3:
        old_values[lane] = T.cuda.atomic_cas(cell.data, compares[lane], replacements[lane])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = cell[0]


@T.prim_func
def cuda_atomic_cas_float32x4(
    cell: T.Buffer((1,), "float32x4"),
    compares: T.Buffer((2,), "float32x4"),
    replacements: T.Buffer((2,), "float32x4"),
    old_values: T.Buffer((2,), "float32x4"),
    final_value: T.Buffer((1,), "float32x4"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 2:
        old_values[lane] = T.cuda.atomic_cas(cell.data, compares[lane], replacements[lane])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = cell[0]


def _v16(values, dtype) -> np.ndarray:
    words = 16 // np.dtype(dtype).itemsize
    return np.asarray(values, dtype=dtype).reshape(-1, words).copy().view(np.dtype("V16")).reshape(-1)


def _unpack(array, dtype) -> np.ndarray:
    return np.asarray(array).view(dtype).reshape(-1, 16 // np.dtype(dtype).itemsize)


def test_cuda_uint64x2_atomic_cas_compares_and_replaces_one_128bit_value():
    """Replaces ``tests/numsim/runtime/test_memory_ops.py::test_cuda_uint64x2_atomic_cas_compares_and_replaces_one_128bit_value`` (bug, see module docstring)."""

    result = v2.Engine().run(
        v2.transpile(cuda_atomic_cas_uint64x2),
        {
            "cell": _v16([[7, 9]], np.uint64),
            "compares": _v16([[7, 9], [11, 999], [11, 13]], np.uint64),
            "replacements": _v16([[11, 13], [17, 19], [23, 29]], np.uint64),
            "old_values": _v16(np.zeros((3, 2)), np.uint64),
            "final_value": _v16(np.zeros((1, 2)), np.uint64),
        },
    )
    np.testing.assert_array_equal(
        _unpack(result.outputs["old_values"], np.uint64),
        np.asarray([[7, 9], [11, 13], [11, 13]], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        _unpack(result.outputs["final_value"], np.uint64), np.asarray([[23, 29]], dtype=np.uint64)
    )


def test_cuda_128bit_atomic_cas_compares_float_vectors_by_bits():
    """Replaces ``tests/numsim/runtime/test_memory_ops.py::test_cuda_128bit_atomic_cas_compares_float_vectors_by_bits`` (bug, see module docstring)."""

    initial = np.asarray([[0x80000000, 0x7FC12345, 0x3F800000, 0x40000000]], dtype=np.uint32)
    signed_zero_mismatch = initial.copy()
    signed_zero_mismatch[0, 0] = 0
    replacement = np.asarray([[1, 2, 3, 4], [11, 13, 17, 19]], dtype=np.uint32)
    result = v2.Engine().run(
        v2.transpile(cuda_atomic_cas_float32x4),
        {
            "cell": _v16(initial, np.uint32),
            "compares": _v16(np.concatenate([signed_zero_mismatch, initial]), np.uint32),
            "replacements": _v16(replacement, np.uint32),
            "old_values": _v16(np.zeros((2, 4)), np.uint32),
            "final_value": _v16(np.zeros((1, 4)), np.uint32),
        },
    )
    np.testing.assert_array_equal(
        _unpack(result.outputs["old_values"], np.uint32), np.concatenate([initial, initial])
    )
    np.testing.assert_array_equal(_unpack(result.outputs["final_value"], np.uint32), replacement[1:2])
