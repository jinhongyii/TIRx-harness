"""v2 copies of legacy tile-form tests, TVM-compilable params only.

The legacy functions mix params TVM's own TilePrimitiveDispatch rejects
(retired, wave 0 / E) with params it compiles. These copies keep only the
compilable params; kernels are copied verbatim from the legacy files. The
legacy "dispatch hint does not change semantics" idea survives as: every
remaining variant transpiles, and runnable variants give the same outputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx

from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tests.numsim.v2.checkers._runnable import requires_v2_engine

pytestmark = requires_v2_engine


# -- tests/numsim/registry/test_copy_dispatch_contract.py --------------------


@T.prim_func
def _valid_forced_ldstmatrix():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    shared = T.alloc_buffer((8, 128), "bfloat16", scope="shared")
    fp32_fragment = T.alloc_tcgen05_ldst_frag("16x256b", (128, 8), "float32")
    bfloat16_fragment = T.alloc_cast_frag(fp32_fragment, "bfloat16")
    Tx.wg.copy(shared[:, :], bfloat16_fragment.permute(1, 0), dispatch="ldstmatrix")


@T.prim_func
def _valid_bound_forced_ldstmatrix():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    shared = T.alloc_buffer((64, 64), "bfloat16", scope="shared", align=16)
    fp32_fragment = T.alloc_tcgen05_ldst_frag("16x256b", (64, 8), "float32")
    bfloat16_fragment = T.alloc_cast_frag(fp32_fragment, "bfloat16")
    for vb in T.unroll(8):
        column: T.let = vb * 8
        Tx.wg.copy(shared[:, column : column + 8], bfloat16_fragment[:, :], dispatch="ldstmatrix")


@T.prim_func
def _invalid_bound_forced_ldstmatrix_alignment():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    shared = T.alloc_buffer((64, 64), "bfloat16", scope="shared", align=16)
    fp32_fragment = T.alloc_tcgen05_ldst_frag("16x256b", (64, 8), "float32")
    bfloat16_fragment = T.alloc_cast_frag(fp32_fragment, "bfloat16")
    for vb in T.unroll(7):
        column: T.let = vb * 8 + 1
        Tx.wg.copy(shared[:, column : column + 8], bfloat16_fragment[:, :], dispatch="ldstmatrix")


@T.prim_func
def _invalid_forced_ldstmatrix_backing_alignment():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    shared = T.alloc_buffer((8, 128), "bfloat16", scope="shared", align=8)
    fp32_fragment = T.alloc_tcgen05_ldst_frag("16x256b", (128, 8), "float32")
    bfloat16_fragment = T.alloc_cast_frag(fp32_fragment, "bfloat16")
    Tx.wg.copy(shared[:, :], bfloat16_fragment.permute(1, 0), dispatch="ldstmatrix")


@T.prim_func
def _valid_forced_fallback(source: T.Buffer((32,), "float16"), output: T.Buffer((32,), "float16")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    Tx.copy(output[:], source[:], dispatch="fallback")


@T.prim_func
def _warpgroup_shared_overlap_forced_fallback(output: T.Buffer((17,), "float32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    thread = T.thread_id_in_wg([128])
    shared = T.alloc_buffer((17,), "float32", scope="shared")
    if thread < 17:
        shared[thread] = T.cast(thread, "float32")
    T.cuda.cta_sync()
    Tx.wg.copy(shared[0:16], shared[1:17], dispatch="fallback")
    T.cuda.cta_sync()
    if thread < 17:
        output[thread] = shared[thread]


@T.prim_func
def _warpgroup_shared_overlap_auto_fallback(output: T.Buffer((17,), "float32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    thread = T.thread_id_in_wg([128])
    shared = T.alloc_buffer((17,), "float32", scope="shared")
    if thread < 17:
        shared[thread] = T.cast(thread, "float32")
    T.cuda.cta_sync()
    Tx.wg.copy(shared[0:16], shared[1:17])
    T.cuda.cta_sync()
    if thread < 17:
        output[thread] = shared[thread]


@pytest.mark.parametrize(
    "kernel",
    [
        _valid_forced_ldstmatrix,
        _valid_bound_forced_ldstmatrix,
        _invalid_bound_forced_ldstmatrix_alignment,
        _invalid_forced_ldstmatrix_backing_alignment,
    ],
)
def test_ldstmatrix_dispatch_hint_does_not_select_numsim_semantics(kernel):
    """Replaces ``tests/numsim/registry/test_copy_dispatch_contract.py::test_ldstmatrix_dispatch_hint_does_not_select_numsim_semantics``.

    Dropped TVM-rejected param ``_invalid_forced_ldstmatrix_layout``. Legacy
    only asserts that the hinted copy transpiles."""

    v2.transpile(kernel)


@pytest.mark.parametrize("kernel", [_valid_forced_fallback])
def test_ordinary_copy_dispatch_does_not_select_semantics(kernel):
    """Replaces ``tests/numsim/registry/test_copy_dispatch_contract.py::test_ordinary_copy_dispatch_does_not_select_semantics``.

    Dropped TVM-rejected params ``_valid_forced_reg``, ``_valid_forced_gmem_smem``,
    ``_invalid_forced_reg_pair``, ``_invalid_forced_gmem_smem_pair``. Legacy
    ``verify(analyze(kernel))`` (static acceptance) becomes ``v2.transpile``."""

    v2.transpile(kernel)


_OVERLAP_EXPECTED = np.asarray([*range(1, 17), 16], dtype=np.float32)


@pytest.mark.parametrize(
    "kernel",
    [
        _warpgroup_shared_overlap_forced_fallback,
        _warpgroup_shared_overlap_auto_fallback,
    ],
)
def test_warpgroup_shared_copy_snapshots_overlap_independent_of_dispatch(kernel):
    """Replaces ``tests/numsim/registry/test_copy_dispatch_contract.py::test_warpgroup_shared_copy_snapshots_overlap_independent_of_dispatch``.

    Dropped TVM-rejected params ``_warpgroup_shared_overlap_reg_hint``,
    ``_warpgroup_shared_overlap_gmem_smem_hint``. The forced-fallback and
    auto variants must both give the same snapshot output."""

    output = np.zeros((17,), dtype=np.float32)

    module = v2.transpile(kernel)
    result = v2.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], _OVERLAP_EXPECTED)


# -- tests/numsim/runtime/test_tile_codegen.py -------------------------------


def _typed_tma_reduce_kernel(dtype: str, reduction: str, extent: int = 4):
    """Legacy kernel; ``extent`` is 4 as in legacy, or 8 for the row-L6
    valid-shape copies (a 4-element 16-bit box is 8 bytes, not a multiple of
    16, so no tensor map encodes it; 8 x 16-bit = 16 bytes)."""

    @T.prim_func
    def kernel(source: T.Buffer((extent,), dtype), output: T.Buffer((extent,), dtype)):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        shared = T.alloc_buffer((extent,), dtype, scope="shared")
        if lane < extent:
            shared[lane] = source[lane]
        T.cuda.warp_sync()
        T.ptx.fence.proxy.async_.shared__cta()
        if lane == 0:
            Tx.copy_async(output[:], shared[:], dispatch="tma_auto", use_tma_reduce=reduction)
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group.read(0)

    return kernel


@pytest.mark.parametrize(
    ("reduction", "dtype", "source", "initial", "expected"),
    [
        (
            "add",
            "float32",
            np.asarray([1.5, -2.0, 4.25, 3.0], dtype=np.float32),
            np.asarray([10.0, 20.0, -1.0, 8.0], dtype=np.float32),
            np.asarray([11.5, 18.0, 3.25, 11.0], dtype=np.float32),
        ),
        (
            "max",
            "int64",
            np.asarray([-3, 10, 20, 7], dtype=np.int64),
            np.asarray([5, -5, 9, 8], dtype=np.int64),
            np.asarray([5, 10, 20, 8], dtype=np.int64),
        ),
        (
            "inc",
            "uint32",
            np.asarray([4, 2, 4, 9], dtype=np.uint32),
            np.asarray([0, 2, 5, 8], dtype=np.uint32),
            np.asarray([1, 0, 0, 9], dtype=np.uint32),
        ),
        (
            "dec",
            "uint32",
            np.asarray([4, 2, 4, 9], dtype=np.uint32),
            np.asarray([0, 2, 5, 8], dtype=np.uint32),
            np.asarray([4, 1, 4, 7], dtype=np.uint32),
        ),
        (
            "and",
            "uint32",
            np.asarray([0x0F0F, 0xFF00, 0xAAAA, 0x1234], dtype=np.uint32),
            np.asarray([0xFFFF, 0x0FF0, 0x5555, 0xFFFF], dtype=np.uint32),
            np.asarray([0x0F0F, 0x0F00, 0x0000, 0x1234], dtype=np.uint32),
        ),
        (
            "or",
            "uint64",
            np.asarray([0x0F, 0xF0, 0xAA, 0x1234], dtype=np.uint64),
            np.asarray([0xF0, 0x0F, 0x55, 0xAB00], dtype=np.uint64),
            np.asarray([0xFF, 0xFF, 0xFF, 0xBB34], dtype=np.uint64),
        ),
        (
            "xor",
            "uint32",
            np.asarray([0x0F, 0xF0, 0xAA, 0x1234], dtype=np.uint32),
            np.asarray([0xF0, 0x0F, 0x55, 0xAB00], dtype=np.uint32),
            np.asarray([0xFF, 0xFF, 0xFF, 0xB934], dtype=np.uint32),
        ),
    ],
)
def test_typed_tma_reduce_reuses_raw_tensor_map_reduction_abi(
    reduction, dtype, source, initial, expected
):
    """Replaces ``tests/numsim/runtime/test_tile_codegen.py::test_typed_tma_reduce_reuses_raw_tensor_map_reduction_abi``.

    Dropped TVM-rejected param ``min-float16``."""

    module = v2.transpile(_typed_tma_reduce_kernel(dtype, reduction))
    result = v2.Engine().run(module, {"source": source, "output": initial.copy()})

    np.testing.assert_array_equal(result.outputs["output"], expected)


@pytest.mark.parametrize(
    ("reduction", "dtype"),
    [
        ("add", "int64"),
        ("add", "float64"),
        ("min", "float32"),
        ("inc", "int32"),
    ],
)
def test_typed_tma_reduce_rejects_invalid_operation_dtype_pairs(reduction, dtype):
    """Replaces ``tests/numsim/runtime/test_tile_codegen.py::test_typed_tma_reduce_rejects_invalid_operation_dtype_pairs``.

    Dropped TVM-rejected param ``unknown-uint32``. Kind: transpile-time
    ``UnsupportedTIRxError`` (legacy text "invalid for dtype" not pinned)."""

    with pytest.raises(UnsupportedTIRxError):
        v2.transpile(_typed_tma_reduce_kernel(dtype, reduction))


# Legacy ``_VALID_TMA_REDUCTION_DTYPES`` minus the pairs TVM's dispatch
# rejects (add/min/max x bfloat16/float16).
_VALID_TMA_REDUCTION_DTYPES = {
    "add": ("float32", "int32", "uint32", "uint64"),
    "min": ("int32", "int64", "uint32", "uint64"),
    "max": ("int32", "int64", "uint32", "uint64"),
    "inc": ("uint32",),
    "dec": ("uint32",),
    "and": ("float32", "float64", "int32", "int64", "uint32", "uint64"),
    "or": ("float32", "float64", "int32", "int64", "uint32", "uint64"),
    "xor": ("float32", "float64", "int32", "int64", "uint32", "uint64"),
}


@pytest.mark.parametrize(
    ("reduction", "dtype"),
    [
        (reduction, dtype)
        for reduction, dtypes in _VALID_TMA_REDUCTION_DTYPES.items()
        for dtype in dtypes
    ],
)
def test_typed_tma_reduce_accepts_every_ptx_operation_dtype_pair(reduction, dtype):
    """Replaces ``tests/numsim/runtime/test_tile_codegen.py::test_typed_tma_reduce_accepts_every_ptx_operation_dtype_pair``.

    Dropped TVM-rejected params ``add-bfloat16``, ``add-float16``,
    ``max-bfloat16``, ``max-float16``, ``min-bfloat16``, ``min-float16``.
    Legacy ``analyze``/``verify``/``emit_rust_module`` becomes ``v2.transpile``."""

    v2.transpile(_typed_tma_reduce_kernel(dtype, reduction))


# -- row L6 valid-shape copies (numsim-behaviour-deltas.md L6) ---------------
# The legacy 16-bit params used a 4-element (8-byte) TMA box, which no tensor
# map can encode (cuTensorMapEncodeTiled: boxDim[0] * elemsize must be a
# multiple of 16). These copies give the kernel 8 16-bit elements (16 bytes).


@pytest.mark.parametrize(
    ("reduction", "dtype", "source", "initial", "expected"),
    [
        (
            "min",
            "float16",
            np.asarray([3.0, -10.0, 20.0, 7.0, 0.5, -0.25, 1024.0, -3.0], dtype=np.float16),
            np.asarray([5.0, -5.0, 9.0, 7.0, -0.5, 0.25, 2048.0, -4.0], dtype=np.float16),
            # Element-wise min, written out by hand (first four are legacy's).
            np.asarray([3.0, -10.0, 9.0, 7.0, -0.5, -0.25, 1024.0, -4.0], dtype=np.float16),
        ),
    ],
    ids=["min-float16"],
)
def test_validshape_typed_tma_reduce_reuses_raw_tensor_map_reduction_abi(
    reduction, dtype, source, initial, expected
):
    """Replaces the ``min-float16`` param of ``tests/numsim/runtime/test_tile_codegen.py::test_typed_tma_reduce_reuses_raw_tensor_map_reduction_abi``.

    Row L6: the legacy box was 4 x float16 = 8 bytes; this copy uses 8 x
    float16 = 16 bytes. Same reduction (min) and the same first four
    elements; four more elements extend the hand-computed expectation."""

    module = v2.transpile(_typed_tma_reduce_kernel(dtype, reduction, extent=8))
    result = v2.Engine().run(module, {"source": source, "output": initial.copy()})

    np.testing.assert_array_equal(result.outputs["output"], expected)


@pytest.mark.parametrize(
    ("reduction", "dtype"),
    [
        (reduction, dtype)
        for reduction in ("add", "min", "max")
        for dtype in ("bfloat16", "float16")
    ],
)
def test_validshape_typed_tma_reduce_accepts_every_ptx_operation_dtype_pair(reduction, dtype):
    """Replaces the ``add|min|max`` x ``bfloat16|float16`` params of ``tests/numsim/runtime/test_tile_codegen.py::test_typed_tma_reduce_accepts_every_ptx_operation_dtype_pair``.

    Row L6: legacy box 4 x 16-bit = 8 bytes; this copy uses 8 elements (16
    bytes). Legacy ``analyze``/``verify``/``emit_rust_module`` becomes
    ``v2.transpile``."""

    v2.transpile(_typed_tma_reduce_kernel(dtype, reduction, extent=8))
