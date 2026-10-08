"""v2 valid-shape copies of the row-L5 tests in ``tests/numsim/integration/test_block_scaled_gemm_artifact.py``.

Row L5 (``docs/development/numsim-behaviour-deltas.md``; ``numsim-isa-answers.md``
"Block-scaled scale-factor K extent"): the legacy fp8 kernels give the
block-scaled ``gemm_async`` an SFA/SFB region whose K extent is 8
(``sf_tmem_layout(128, SF_K=2, sf_per_mma=1, sf_reuse=4)``, or a ``[:, 4:12]``
slice). For a K=128 fp8 GEMM (four K=32 MMAs) the hardware scale-vector layout
needs a K extent in {1, 4, 16}; v2 rejects 8. The kernels also issue M=128 N=8
MMAs (row L4) and store the scales straight into the replicated SF TMEM view
(row L1, ``tmem_replicated_view``), so each copy changes three things:

- SF K extent 4: ``sf_tmem_layout(128, SF_K=2, sf_per_mma=1, sf_reuse=2)``
  (shape ``(128, 4)``). MMA ``ki`` still reads scale ``ki // 2``, i.e. scale 0
  for K 0..63 and scale 1 for K 64..127, exactly the legacy oracle
  (``_expected_fp8_extent8_invocation``). The region-min test slices
  ``[:, 2:6]`` of a ``(128, 8)`` view instead of ``[:, 4:12]`` of
  ``(128, 16)``: same scales (1, 2).
- N = 16 (B is 16 x 128 fp8; SFB rows 0..15 are used).
- Scales reach TMEM the hardware way: ``Tx.copy`` into an
  ``sf_smem_layout(128, SF_K=4, sf_per_mma=1)`` shared buffer, then
  ``tcgen05.cp`` (``Tx.copy_async``) into the physical
  ``sf_tmem_layout(128, SF_K=4, sf_per_mma=1)`` TMEM buffer, which the K-extent-4
  reuse view aliases (unique scale ``k`` lives in byte ``k`` of both). The
  scale inputs are therefore padded to ``(128, 4)``: the legacy ``(rows, 2)``
  scales in columns 0..1 (rows 0..15 for SFB), 127 (= 1.0) elsewhere. The
  oracle uses only the legacy scales.

As in ``test_validshape_gemm_async_artifact.py``: four warps read back their own
TMEM sub-partition (row T4), and the MMA is committed to an mbarrier and waited
on before the read (``tcgen05.mma``/``tcgen05.cp`` are asynchronous).
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm import tirx
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import tmem_datapath_layout
from tvm.backend.cuda.tile_primitive.gemm_async.tcgen05 import sf_smem_layout, sf_tmem_layout
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
from tvm_ffi import structural_map

from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

_FP8_MMA_128X128 = mma_shared_layout("float8_e4m3fn", SwizzleMode.SWIZZLE_128B_ATOM, (128, 128))
_FP8_MMA_16X128 = mma_shared_layout("float8_e4m3fn", SwizzleMode.SWIZZLE_128B_ATOM, (16, 128))
_PACKED_FP4_MMA_128X32 = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (128, 32))
_PACKED_FP4_MMA_16X32 = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (16, 32))
_MXF8_SF_SMEM = sf_smem_layout(128, SF_K=4, sf_per_mma=1)
_MXF8_SF_TMEM = sf_tmem_layout(128, SF_K=4, sf_per_mma=1)
# K extent 4 view of two unique scales: logical column c reads scale c // 2.
_MXF8_SF_TMEM_K4 = sf_tmem_layout(128, SF_K=2, sf_per_mma=1, sf_reuse=2)
# (128, 8) view of four unique scales: logical column c reads scale c // 2.
_MXF8_SF_TMEM_K8_VIEW = sf_tmem_layout(128, SF_K=4, sf_per_mma=1, sf_reuse=2)
_NVFP4_SF_SMEM = sf_smem_layout(128, SF_K=4, sf_per_mma=4)
_NVFP4_SF_TMEM = sf_tmem_layout(128, SF_K=4, sf_per_mma=4)

_SWIZZLE_GAP = (
    "v2 gemm_async over SWIZZLE_32B/128B shared operands filled by Tx.copy reads K "
    "elements from the wrong 16-byte chunk (see test_validshape_gemm_async_artifact.py; "
    "the SWIZZLE_NONE dense copy matches numpy). Observed: 25-91% of outputs differ "
    "from the scaled-matmul oracle; the interleaved test, whose operands are uniform "
    "(swizzle-invariant), passes, so the K-extent-4 SF path itself matches"
)


# -- oracle helpers (copied from the legacy module) ---------------------------


def _decode_e4m3(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    exponent = ((bits >> np.uint8(3)) & np.uint8(0xF)).astype(np.int16)
    mantissa = (bits & np.uint8(0x7)).astype(np.float32)
    normal = np.ldexp(np.float32(1) + mantissa / np.float32(8), exponent - 7)
    subnormal = np.ldexp(mantissa / np.float32(8), -6)
    magnitude = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    result = np.where((bits & np.uint8(0x80)) != 0, -magnitude, magnitude)
    return np.where((exponent == 15) & ((bits & 7) == 7), np.nan, result).astype(np.float32)


def _decode_e8m0(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    return np.exp2(bits.astype(np.int16) - 127).astype(np.float32)


def _expected_fp8_two_scale_invocation(
    left: np.ndarray,
    right: np.ndarray,
    scale_a: np.ndarray,
    scale_b: np.ndarray,
    *,
    scale_indices: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Legacy ``_expected_fp8_extent8_invocation``: K 0..63 uses scale
    ``scale_indices[0]``, K 64..127 uses ``scale_indices[1]``."""

    a = _decode_e4m3(left)
    b = _decode_e4m3(right)
    a_scales = _decode_e8m0(scale_a)
    b_scales = _decode_e8m0(scale_b)
    result = np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)
    for k_slice, scale_index in zip((slice(0, 64), slice(64, 128)), scale_indices, strict=True):
        result += (a[:, k_slice] * a_scales[:, scale_index, None]) @ (
            b[:, k_slice] * b_scales[:, scale_index, None]
        ).T
    return result


def _padded_scales(scales: np.ndarray) -> np.ndarray:
    """Legacy scales in the top-left corner of a ``(128, 4)`` e8m0 buffer; 127
    (1.0) elsewhere (never read by the K-extent-4 view or rows >= N)."""

    padded = np.full((128, 4), 127, dtype=np.uint8)
    padded[: scales.shape[0], : scales.shape[1]] = scales
    return padded


# -- kernels ------------------------------------------------------------------


@T.prim_func
def block_scaled_fp8_gemm_packed_scales_k4(
    left: T.Buffer((128, 128), "float8_e4m3fn"),
    right: T.Buffer((16, 128), "float8_e4m3fn"),
    scale_a: T.Buffer((128, 4), "float8_e8m0fnu"),
    scale_b: T.Buffer((128, 4), "float8_e8m0fnu"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    left_shared = T.alloc_buffer((128, 128), "float8_e4m3fn", scope="shared", layout=_FP8_MMA_128X128)
    right_shared = T.alloc_buffer((16, 128), "float8_e4m3fn", scope="shared", layout=_FP8_MMA_16X128)
    scale_a_shared = T.alloc_buffer((128, 4), "float8_e8m0fnu", scope="shared", layout=_MXF8_SF_SMEM)
    scale_b_shared = T.alloc_buffer((128, 4), "float8_e8m0fnu", scope="shared", layout=_MXF8_SF_SMEM)
    accumulator = T.decl_buffer(
        (128, 16), "float32", scope="tmem", layout=tmem_datapath_layout("D", 128, 16), allocated_addr=0
    )
    scale_a_cells = T.decl_buffer(
        (128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM, allocated_addr=16
    )
    scale_b_cells = T.decl_buffer(
        (128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM, allocated_addr=32
    )
    scale_a_tmem = T.decl_buffer(
        (128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM_K4, allocated_addr=16
    )
    scale_b_tmem = T.decl_buffer(
        (128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM_K4, allocated_addr=32
    )
    if warp == 0 and lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        Tx.copy(scale_a_shared[:, :], scale_a[:, :])
        Tx.copy(scale_b_shared[:, :], scale_b[:, :])
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        Tx.copy_async(scale_a_cells[:, :], scale_a_shared[:, :], cta_group=1)
        Tx.copy_async(scale_b_cells[:, :], scale_b_shared[:, :], cta_group=1)
        for packed_call in T.serial(2):
            Tx.gemm_async(
                accumulator[:, :],
                left_shared[:, :],
                right_shared[:, :],
                SFA=scale_a_tmem[:, :],
                SFB=scale_b_tmem[:, :],
                accum=packed_call != 0,
                dispatch="tcgen05",
                cta_group=1,
            )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(16):
        output[row, col] = accumulator[row, col]


@T.prim_func
def block_scaled_nvfp4_gemm_n16(
    left_packed: T.Buffer((128, 32), "uint8"),
    right_packed: T.Buffer((16, 32), "uint8"),
    scale_a: T.Buffer((128, 4), "float8_e4m3fn"),
    scale_b: T.Buffer((128, 4), "float8_e4m3fn"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    left_shared_packed = T.alloc_buffer((128, 32), "uint8", scope="shared", layout=_PACKED_FP4_MMA_128X32)
    right_shared_packed = T.alloc_buffer((16, 32), "uint8", scope="shared", layout=_PACKED_FP4_MMA_16X32)
    left_shared = left_shared_packed.view("float4_e2m1fn")
    right_shared = right_shared_packed.view("float4_e2m1fn")
    scale_a_shared = T.alloc_buffer((128, 4), "float8_e4m3fn", scope="shared", layout=_NVFP4_SF_SMEM)
    scale_b_shared = T.alloc_buffer((128, 4), "float8_e4m3fn", scope="shared", layout=_NVFP4_SF_SMEM)
    accumulator = T.decl_buffer(
        (128, 16), "float32", scope="tmem", layout=tmem_datapath_layout("D", 128, 16), allocated_addr=0
    )
    scale_a_tmem = T.decl_buffer(
        (128, 4), "float8_e4m3fn", scope="tmem", layout=_NVFP4_SF_TMEM, allocated_addr=16
    )
    scale_b_tmem = T.decl_buffer(
        (128, 4), "float8_e4m3fn", scope="tmem", layout=_NVFP4_SF_TMEM, allocated_addr=32
    )
    if warp == 0 and lane == 0:
        Tx.copy(left_shared_packed[:, :], left_packed[:, :])
        Tx.copy(right_shared_packed[:, :], right_packed[:, :])
        Tx.copy(scale_a_shared[:, :], scale_a[:, :])
        Tx.copy(scale_b_shared[:, :], scale_b[:, :])
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        Tx.copy_async(scale_a_tmem[:, :], scale_a_shared[:, :], cta_group=1)
        Tx.copy_async(scale_b_tmem[:, :], scale_b_shared[:, :], cta_group=1)
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=1,
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(16):
        output[row, col] = accumulator[row, col]


@T.prim_func
def block_scaled_interleaved_physical_streams_k4(
    left: T.Buffer((128, 128), "float8_e4m3fn"),
    right: T.Buffer((16, 128), "float8_e4m3fn"),
    scale_1_a: T.Buffer((128, 4), "float8_e8m0fnu"),
    scale_1_b: T.Buffer((128, 4), "float8_e8m0fnu"),
    scale_2_a: T.Buffer((128, 4), "float8_e8m0fnu"),
    scale_2_b: T.Buffer((128, 4), "float8_e8m0fnu"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    left_shared = T.alloc_buffer((128, 128), "float8_e4m3fn", scope="shared", layout=_FP8_MMA_128X128)
    right_shared = T.alloc_buffer((16, 128), "float8_e4m3fn", scope="shared", layout=_FP8_MMA_16X128)
    s1a_shared = T.alloc_buffer((128, 4), "float8_e8m0fnu", scope="shared", layout=_MXF8_SF_SMEM)
    s1b_shared = T.alloc_buffer((128, 4), "float8_e8m0fnu", scope="shared", layout=_MXF8_SF_SMEM)
    s2a_shared = T.alloc_buffer((128, 4), "float8_e8m0fnu", scope="shared", layout=_MXF8_SF_SMEM)
    s2b_shared = T.alloc_buffer((128, 4), "float8_e8m0fnu", scope="shared", layout=_MXF8_SF_SMEM)
    accumulator = T.decl_buffer(
        (128, 16), "float32", scope="tmem", layout=tmem_datapath_layout("D", 128, 16), allocated_addr=0
    )
    s1a_cells = T.decl_buffer((128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM, allocated_addr=16)
    s1b_cells = T.decl_buffer((128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM, allocated_addr=32)
    s2a_cells = T.decl_buffer((128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM, allocated_addr=48)
    s2b_cells = T.decl_buffer((128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM, allocated_addr=64)
    scale_1_a_tmem = T.decl_buffer(
        (128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM_K4, allocated_addr=16
    )
    scale_1_b_tmem = T.decl_buffer(
        (128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM_K4, allocated_addr=32
    )
    scale_2_a_tmem = T.decl_buffer(
        (128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM_K4, allocated_addr=48
    )
    scale_2_b_tmem = T.decl_buffer(
        (128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM_K4, allocated_addr=64
    )
    if warp == 0 and lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        Tx.copy(s1a_shared[:, :], scale_1_a[:, :])
        Tx.copy(s1b_shared[:, :], scale_1_b[:, :])
        Tx.copy(s2a_shared[:, :], scale_2_a[:, :])
        Tx.copy(s2b_shared[:, :], scale_2_b[:, :])
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        Tx.copy_async(s1a_cells[:, :], s1a_shared[:, :], cta_group=1)
        Tx.copy_async(s1b_cells[:, :], s1b_shared[:, :], cta_group=1)
        Tx.copy_async(s2a_cells[:, :], s2a_shared[:, :], cta_group=1)
        Tx.copy_async(s2b_cells[:, :], s2b_shared[:, :], cta_group=1)
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_1_a_tmem[:, :],
            SFB=scale_1_b_tmem[:, :],
            accum=False,
        )
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_2_a_tmem[:, :],
            SFB=scale_2_b_tmem[:, :],
            accum=False,
        )
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_1_a_tmem[:, :],
            SFB=scale_1_b_tmem[:, :],
            accum=True,
        )
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_2_a_tmem[:, :],
            SFB=scale_2_b_tmem[:, :],
            accum=True,
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(16):
        output[row, col] = accumulator[row, col]


@T.prim_func
def block_scaled_scale_region_min_k4(
    left: T.Buffer((128, 128), "float8_e4m3fn"),
    right: T.Buffer((16, 128), "float8_e4m3fn"),
    scale_a: T.Buffer((128, 4), "float8_e8m0fnu"),
    scale_b: T.Buffer((128, 4), "float8_e8m0fnu"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    left_shared = T.alloc_buffer((128, 128), "float8_e4m3fn", scope="shared", layout=_FP8_MMA_128X128)
    right_shared = T.alloc_buffer((16, 128), "float8_e4m3fn", scope="shared", layout=_FP8_MMA_16X128)
    scale_a_shared = T.alloc_buffer((128, 4), "float8_e8m0fnu", scope="shared", layout=_MXF8_SF_SMEM)
    scale_b_shared = T.alloc_buffer((128, 4), "float8_e8m0fnu", scope="shared", layout=_MXF8_SF_SMEM)
    accumulator = T.decl_buffer(
        (128, 16), "float32", scope="tmem", layout=tmem_datapath_layout("D", 128, 16), allocated_addr=0
    )
    scale_a_cells = T.decl_buffer(
        (128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM, allocated_addr=16
    )
    scale_b_cells = T.decl_buffer(
        (128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM, allocated_addr=32
    )
    scale_a_tmem = T.decl_buffer(
        (128, 8), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM_K8_VIEW, allocated_addr=16
    )
    scale_b_tmem = T.decl_buffer(
        (128, 8), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM_K8_VIEW, allocated_addr=32
    )
    if warp == 0 and lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        Tx.copy(scale_a_shared[:, :], scale_a[:, :])
        Tx.copy(scale_b_shared[:, :], scale_b[:, :])
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        Tx.copy_async(scale_a_cells[:, :], scale_a_shared[:, :], cta_group=1)
        Tx.copy_async(scale_b_cells[:, :], scale_b_shared[:, :], cta_group=1)
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_a_tmem[:, 2:6],
            SFB=scale_b_tmem[:, 2:6],
            accum=False,
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(16):
        output[row, col] = accumulator[row, col]


def _duplicate_block_scaled_gemm_node(func):
    """Legacy helper without the emitted-Rust variant lookup: the kernel's
    only ``gemm_async`` is block-scaled (it carries SFA/SFB); follow it with
    an identical copy."""

    seen = []

    def rewrite(node):
        if type(node).__name__ != "TilePrimitiveCall":
            return node
        if str(node.op.name) != "tirx.tile.gemm_async":
            return node
        seen.append(node)
        return tirx.SeqStmt([node, node.replace()])

    body = structural_map(func.body, (tirx.TilePrimitiveCall, rewrite))
    assert len(seen) == 1
    return func.with_body(body)


def _block_scaled_instructions(module) -> set[tuple]:
    """``(D, A, B, SFA, SFB, M, N)`` of every block-scaled instruction
    descriptor the lowered program encodes. The program calls
    ``tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled``: its op record
    carries the five dtypes as ``argN=`` modifiers and its operands are
    ``(sfa_tmem_addr, sfb_tmem_addr, M, N, K, ...)``."""

    (kernel,) = module.document["kernels"]
    ops = kernel["ops"]
    consts = kernel["consts"]
    found = set()
    for record in kernel["code"]:
        call = record.get("Ptx") if isinstance(record, dict) else None
        if call is None:
            continue
        op = ops[call["op"]]
        if op["name"] != "tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled":
            continue
        dtypes = tuple(mod.split("=", 1)[1] for mod in op["mods"] if mod.startswith("arg"))
        m_operand, n_operand = call["srcs"][2], call["srcs"][3]
        assert "Const" in m_operand and "Const" in n_operand, call
        found.add((*dtypes, consts[m_operand["Const"]]["bits"], consts[n_operand["Const"]]["bits"]))
    return found


# -- tests --------------------------------------------------------------------


def test_block_scaled_gemm_normalizes_instruction_kind_and_shape() -> None:
    """Replaces ``tests/numsim/integration/test_block_scaled_gemm_artifact.py::test_block_scaled_gemm_normalizes_instruction_kind_and_shape``.

    Row L5 (fp8 SF K extent 8 -> 4) and row L4 (both kernels M=128 N=8 ->
    N=16). Legacy read the ``Gemm<BlockScaled<E8m0>, E4m3, E4m3, ..., 128, 8>``
    variant from its emitted Rust. v2 has no Rust; the copy reads the
    block-scaled instruction descriptors of the lowered program (public
    ``CompiledModule.document``): kind = (D, A, B, SFA, SFB) dtypes
    (fp8: E4m3 operands with E8m0 scales; nvfp4: E2m1 operands with E4m3
    scales) and instruction shape 128 x 16 (legacy 128 x 8)."""

    assert _block_scaled_instructions(v2.transpile(block_scaled_fp8_gemm_packed_scales_k4)) == {
        ("float32", "float8_e4m3fn", "float8_e4m3fn", "float8_e8m0fnu", "float8_e8m0fnu", 128, 16)
    }
    assert _block_scaled_instructions(v2.transpile(block_scaled_nvfp4_gemm_n16)) == {
        ("float32", "float4_e2m1fn", "float4_e2m1fn", "float8_e4m3fn", "float8_e4m3fn", 128, 16)
    }


@v2_gap(_SWIZZLE_GAP)
def test_fp8_block_scaled_gemm_derives_scale_columns_from_each_invocation():
    """Replaces ``tests/numsim/integration/test_block_scaled_gemm_artifact.py::test_fp8_block_scaled_gemm_derives_scale_columns_from_each_invocation``.

    Row L5: SF K extent 8 (SF_K=2, reuse 4) -> 4 (SF_K=2, reuse 2); row L4:
    N=8 -> 16; scales loaded via SMEM + ``tcgen05.cp`` (row L1). Same scale
    values; two packed calls give ``2 * once``."""

    left_codes = np.array([0x30, 0x38, 0x3C, 0x40, 0xB8, 0xBC], dtype=np.uint8)
    right_codes = np.array([0x28, 0x38, 0x40, 0xB0, 0xB8], dtype=np.uint8)
    left = np.resize(left_codes, (128, 128))
    right = np.resize(right_codes, (16, 128))
    scale_a = np.empty((128, 2), dtype=np.uint8)
    scale_a[:, 0] = np.where(np.arange(128) % 2 == 0, 127, 128)
    scale_a[:, 1] = np.where(np.arange(128) % 3 == 0, 126, 127)
    scale_b = np.resize(
        np.array([[127, 128], [128, 127], [126, 129], [129, 126]], dtype=np.uint8), (16, 2)
    )

    result = v2.Engine().run(
        v2.transpile(block_scaled_fp8_gemm_packed_scales_k4),
        {
            "left": left,
            "right": right,
            "scale_a": _padded_scales(scale_a),
            "scale_b": _padded_scales(scale_b),
            "output": np.zeros((128, 16), dtype=np.float32),
        },
    )

    expected_once = _expected_fp8_two_scale_invocation(left, right, scale_a, scale_b)
    np.testing.assert_array_equal(result.outputs["output"], 2 * expected_once)


@v2_gap(_SWIZZLE_GAP)
def test_repeated_block_scaled_callsites_inside_a_loop_do_not_share_history():
    """Replaces ``tests/numsim/integration/test_block_scaled_gemm_artifact.py::test_repeated_block_scaled_callsites_inside_a_loop_do_not_share_history``.

    Row L5/L4/L1 as above. The gemm inside the two-iteration loop is
    duplicated (overwrite, overwrite, accumulate, accumulate): ``3 * once``."""

    left = np.resize(np.array([0x30, 0x38, 0x3C, 0x40], dtype=np.uint8), (128, 128))
    right = np.resize(np.array([0x28, 0x38, 0x40], dtype=np.uint8), (16, 128))
    scale_a = np.tile(np.array([127, 129], dtype=np.uint8), (128, 1))
    scale_b = np.tile(np.array([128, 126], dtype=np.uint8), (16, 1))

    result = v2.Engine().run(
        v2.transpile(_duplicate_block_scaled_gemm_node(block_scaled_fp8_gemm_packed_scales_k4)),
        {
            "left": left,
            "right": right,
            "scale_a": _padded_scales(scale_a),
            "scale_b": _padded_scales(scale_b),
            "output": np.zeros((128, 16), dtype=np.float32),
        },
    )

    expected_once = _expected_fp8_two_scale_invocation(left, right, scale_a, scale_b)
    np.testing.assert_array_equal(result.outputs["output"], 3 * expected_once)


def test_interleaved_block_scaled_calls_are_independent_of_prior_calls():
    """Replaces ``tests/numsim/integration/test_block_scaled_gemm_artifact.py::test_interleaved_block_scaled_calls_are_independent_of_prior_calls``.

    Row L5: four SF regions of K extent 8 -> 4; row L4: N=8 -> 16; row L1:
    the legacy kernel stored the scales straight into the replicated SF TMEM
    views, which v2 rejects (``tmem_replicated_view``), so the copy loads each
    scale set via SMEM + ``tcgen05.cp`` and tests the valid part: calls with
    scale sets 1, 2, 1 (accumulate), 2 (accumulate) give ``first + 2 *
    second``."""

    left = np.full((128, 128), 0x38, dtype=np.uint8)
    right = np.full((16, 128), 0x38, dtype=np.uint8)
    scale_1_a = np.tile(np.array([127, 128], dtype=np.uint8), (128, 1))
    scale_1_b = np.tile(np.array([127, 128], dtype=np.uint8), (16, 1))
    scale_2_a = np.tile(np.array([129, 130], dtype=np.uint8), (128, 1))
    scale_2_b = np.tile(np.array([129, 130], dtype=np.uint8), (16, 1))

    result = v2.Engine().run(
        v2.transpile(block_scaled_interleaved_physical_streams_k4),
        {
            "left": left,
            "right": right,
            "scale_1_a": _padded_scales(scale_1_a),
            "scale_1_b": _padded_scales(scale_1_b),
            "scale_2_a": _padded_scales(scale_2_a),
            "scale_2_b": _padded_scales(scale_2_b),
            "output": np.zeros((128, 16), dtype=np.float32),
        },
    )

    first = _expected_fp8_two_scale_invocation(left, right, scale_1_a, scale_1_b)
    second = _expected_fp8_two_scale_invocation(left, right, scale_2_a, scale_2_b)
    np.testing.assert_array_equal(result.outputs["output"], first + 2 * second)


@v2_gap(_SWIZZLE_GAP)
def test_block_scale_region_min_selects_the_physical_scale_coordinates():
    """Replaces ``tests/numsim/integration/test_block_scaled_gemm_artifact.py::test_block_scale_region_min_selects_the_physical_scale_coordinates``.

    Row L5: legacy sliced ``[:, 4:12]`` (K extent 8) of a ``(128, 16)``
    SF_K=4 reuse-4 view; the copy slices ``[:, 2:6]`` (K extent 4) of a
    ``(128, 8)`` SF_K=4 reuse-2 view. Both start at unique scale 1, so K 0..63
    uses scale 1 and K 64..127 scale 2, as legacy. Row L4: N=8 -> 16; row L1:
    scales via SMEM + ``tcgen05.cp``."""

    left = np.resize(np.array([0x30, 0x38, 0x3C, 0x40], dtype=np.uint8), (128, 128))
    right = np.resize(np.array([0x28, 0x38, 0x40], dtype=np.uint8), (16, 128))
    scale_a = np.tile(np.array([126, 127, 129, 130], dtype=np.uint8), (128, 1))
    scale_b = np.tile(np.array([130, 128, 126, 127], dtype=np.uint8), (16, 1))

    result = v2.Engine().run(
        v2.transpile(block_scaled_scale_region_min_k4),
        {
            "left": left,
            "right": right,
            "scale_a": _padded_scales(scale_a),
            "scale_b": _padded_scales(scale_b),
            "output": np.zeros((128, 16), dtype=np.float32),
        },
    )

    expected = _expected_fp8_two_scale_invocation(
        left, right, scale_a, scale_b, scale_indices=(1, 2)
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)
