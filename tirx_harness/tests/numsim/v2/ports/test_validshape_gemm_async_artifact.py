"""v2 valid-shape copies of the row-L4 tests in ``tests/numsim/integration/test_gemm_async_artifact.py``.

Row L4 (``docs/development/numsim-behaviour-deltas.md``; ``numsim-isa-answers.md``
"tcgen05.mma shapes"): the legacy kernels issue ``tcgen05.mma`` shapes that are
not hardware instructions (``M = 128, N = 8`` at ``cta_group::1``; ``N = 16`` at
``cta_group::2``). v2 rejects them at transpile time. These copies keep each
legacy kernel's structure and the legacy test's oracle (numpy matmul of the
same operand values) but use a valid N: ``N = 16`` at ``cta_group::1`` (M=128)
and ``N = 32`` at ``cta_group::2``. The per-CTA right-operand shard grows to
match (``N / cta_group`` rows).

Two further edits make the kernels valid programs; neither changes what the
legacy test checks:

- Row T4: a warp may only access its own 32-lane TMEM sub-partition, so the
  legacy one-lane TMEM read-back (lane 0 reading 128 lanes) is done by four
  warps, each reading the rows its sub-partition holds under the declared
  TMEM layout. Direct TMEM stores (A operand, ``-777`` fill) are split the
  same way and followed by ``tcgen05.wait::st`` before the MMA.
- ``tcgen05.mma`` is asynchronous: the issuer commits to an mbarrier and every
  reader waits on it (plus ``tcgen05.fence::after_thread_sync``) before the
  TMEM read. Legacy completed the MMA at issue; without the wait v2 reads the
  pre-MMA TMEM contents (compare row T20 for ``tcgen05.cp``).

Dropped pins: ``stats["worker_count"]`` / ``stats["scheduling_domain_count"]``
(legacy scheduler stats, test-migration.md), ``cache_dir``.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import ComposeLayout, S, TCol, TileLayout, TLane, tmem_datapath_layout
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout

from tirx_harness.numsim.errors import UnsupportedTIRxError
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

_MMA_F16_32B = ComposeLayout(3, 1, 3, TileLayout(S[(128,)]))
_MMA_FP8_128X128 = mma_shared_layout("float8_e4m3fn", SwizzleMode.SWIZZLE_128B_ATOM, (128, 128))
_MMA_FP8_16X128 = mma_shared_layout("float8_e4m3fn", SwizzleMode.SWIZZLE_128B_ATOM, (16, 128))
_MMA_F16_NONE_128X16 = mma_shared_layout("float16", SwizzleMode.SWIZZLE_NONE, (128, 16))
_MMA_F16_NONE_16X16 = mma_shared_layout("float16", SwizzleMode.SWIZZLE_NONE, (16, 16))
# Legacy _TMEM_WRONG_B_ROW_COLUMN_GROUP for (64, 16), widened to (64, 32):
# lane = 64 * (r // 32) + r % 32 + 32 * (c // 16), column = c % 16.
_TMEM_WRONG_B_ROW_COLUMN_GROUP_N32 = TileLayout(
    S[(2, 32, 2, 16) : (64 @ TLane, 1 @ TLane, 32 @ TLane, 1 @ TCol)]
)



def _f16(rows: int, cols: int, mod: int, shift: int) -> np.ndarray:
    return (np.arange(rows * cols, dtype=np.float32).reshape(rows, cols) % mod - shift).astype(
        np.float16
    )


def _matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.matmul(left.astype(np.float32), right.astype(np.float32).T)


# -- cta_group::1, M = 128, N = 16 -------------------------------------------


@T.prim_func
def dense_gemm_async_cta1_n16(
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((16, 16), "float16"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    left_shared = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (128, 16), "float32", scope="tmem", layout=tmem_datapath_layout("D", 128, 16), allocated_addr=0
    )
    if warp == 0 and lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=1,
            mma_m=128,
            mma_n=16,
        )
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=True,
            dispatch="tcgen05",
            cta_group=1,
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(16):
        output[row, col] = accumulator[row, col]


@T.prim_func
def dense_fp8_gemm_async_cta1_n16(
    left: T.Buffer((128, 128), "float8_e4m3fn"),
    right: T.Buffer((16, 128), "float8_e4m3fn"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    left_shared = T.alloc_buffer((128, 128), "float8_e4m3fn", scope="shared", layout=_MMA_FP8_128X128)
    right_shared = T.alloc_buffer((16, 128), "float8_e4m3fn", scope="shared", layout=_MMA_FP8_16X128)
    accumulator = T.decl_buffer(
        (128, 16), "float32", scope="tmem", layout=tmem_datapath_layout("D", 128, 16), allocated_addr=0
    )
    if warp == 0 and lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=1,
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(16):
        output[row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_dynamic_right_index_n16(
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    right_start = T.alloc_buffer((1,), "int32", scope="local")
    left_shared = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((32, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (128, 16), "float32", scope="tmem", layout=tmem_datapath_layout("D", 128, 16), allocated_addr=0
    )
    right_start[0] = 1
    if warp == 0 and lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[right_start[0] : right_start[0] + 16, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=1,
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(16):
        output[row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_two_clusters_n16(
    left: T.Buffer((256, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((256, 16), "float32"),
):
    T.device_entry()
    cta = T.cta_id([2])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    left_shared = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (128, 16), "float32", scope="tmem", layout=tmem_datapath_layout("D", 128, 16), allocated_addr=0
    )
    if warp == 0 and lane == 0:
        Tx.copy(left_shared[:, :], left[cta * 128 : (cta + 1) * 128, :])
        Tx.copy(right_shared[:, :], right[cta * 16 : (cta + 1) * 16, :])
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=1,
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(16):
        output[cta * 128 + row, col] = accumulator[row, col]


@T.prim_func
def inactive_gemm_async_is_noop_n16(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (128, 16), "float32", scope="tmem", layout=tmem_datapath_layout("D", 128, 16), allocated_addr=0
    )
    for _step in T.serial(1):
        if lane >= 0:
            continue
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=1,
        )
    if lane == 0:
        output[0] = 7


@T.prim_func
def dense_gemm_async_no_swizzle_shared_n16(
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((16, 16), "float16"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    left_shared = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_NONE_128X16)
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_NONE_16X16)
    accumulator = T.decl_buffer(
        (128, 16), "float32", scope="tmem", layout=tmem_datapath_layout("D", 128, 16), allocated_addr=0
    )
    if warp == 0 and lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=False)
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(16):
        output[row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_two_thread_issuers_n16(
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((16, 16), "float16"),
    output: T.Buffer((1,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (128, 16), "float32", scope="tmem", layout=tmem_datapath_layout("D", 128, 16), allocated_addr=0
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
    T.cuda.warp_sync()
    if lane < 2:
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=False)
    if lane == 0:
        output[0] = 1


# -- cta_group::2, N = 32 ----------------------------------------------------


@T.prim_func
def dense_gemm_async_cta_group2_n32(
    left: T.Buffer((256, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((256, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    left_shared = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (128, 32), "float32", scope="tmem", layout=tmem_datapath_layout("D", 128, 32), allocated_addr=0
    )
    if warp == 0 and lane == 0:
        Tx.copy(left_shared[:, :], left[cta * 128 : (cta + 1) * 128, :])
        Tx.copy(right_shared[:, :], right[cta * 16 : (cta + 1) * 16, :])
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (warp == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=2,
        )
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(barrier[0]), T.uint16(3)
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(32):
        output[cta * 128 + row, col] = accumulator[row, col]
    T.cuda.cluster_sync()


@T.prim_func
def dense_gemm_async_tmem_a_cta_group2_n32(
    left: T.Buffer((256, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((256, 32), "float32"),
):
    """CTA-pair MMA whose A operand is each CTA's own TMEM shard."""

    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    left_tmem = T.decl_buffer(
        (128, 16), "float16", scope="tmem", layout=tmem_datapath_layout("D", 128, 16), allocated_addr=0
    )
    accumulator = T.decl_buffer(
        (128, 32), "float32", scope="tmem", layout=tmem_datapath_layout("D", 128, 32), allocated_addr=8
    )
    row = T.meta_var(warp * 32 + lane)
    if warp == 0 and lane == 0:
        Tx.copy(right_shared[:, :], right[cta * 16 : (cta + 1) * 16, :])
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    for col in T.serial(16):
        left_tmem[row, col] = left[cta * 128 + row, col]
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cluster_sync()
    T.ptx.tcgen05.fence__after_thread_sync()
    if (cta == 0) and (warp == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_tmem[:, :],
            right_shared[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=2,
        )
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(barrier[0]), T.uint16(3)
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    for col in T.serial(32):
        output[cta * 128 + row, col] = accumulator[row, col]
    T.cuda.cluster_sync()


def _layout_b_kernel(tmem_layout, *, wrong_grouping: bool, accumulate_twice: bool):
    """Legacy ``dense_gemm_async_m64_cta2_layout_b`` / ``..._wrong_b_grouping``
    at N = 32 (16 B rows per CTA). Warp ``w`` lane ``l`` owns TMEM lane
    ``L = 32 w + l``; under Layout B that lane holds logical row ``L % 64`` and
    the column half ``L // 64``; under the wrong grouping it holds row
    ``32 (L // 64) + L % 32`` and column half ``(L // 32) % 2``."""

    @T.prim_func
    def kernel(
        left: T.Buffer((128, 16), "float16"),
        right: T.Buffer((32, 16), "float16"),
        output: T.Buffer((128, 32), "float32"),
    ):
        T.device_entry()
        _cluster = T.cluster_id([1])
        cta = T.cta_id_in_cluster([2])
        warp = T.warp_id([4])
        lane = T.lane_id([32])
        barrier = T.alloc_buffer((1,), "uint64", scope="shared")
        left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
        right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
        accumulator = T.decl_buffer(
            (64, 32), "float32", scope="tmem", layout=tmem_layout, allocated_addr=0
        )
        tmem_lane = T.meta_var(warp * 32 + lane)
        if wrong_grouping:
            row = T.meta_var((tmem_lane // 64) * 32 + tmem_lane % 32)
            half = T.meta_var((tmem_lane // 32) % 2)
        else:
            row = T.meta_var(tmem_lane % 64)
            half = T.meta_var(tmem_lane // 64)
        if warp == 0 and lane == 0:
            Tx.copy(left_shared[:, :], left[cta * 64 : (cta + 1) * 64, :])
            Tx.copy(right_shared[:, :], right[cta * 16 : (cta + 1) * 16, :])
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
        for col in T.serial(16):
            accumulator[row, half * 16 + col] = T.float32(-777)
        T.ptx.tcgen05.wait__st.sync.aligned()
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx.fence.mbarrier_init.release.cluster()
        T.ptx.tcgen05.fence__before_thread_sync()
        T.cuda.cluster_sync()
        T.ptx.tcgen05.fence__after_thread_sync()
        if (cta == 0) and (warp == 0) and (lane == 0):
            Tx.gemm_async(
                accumulator[:, :],
                left_shared[:, :],
                right_shared[:, :],
                accum=False,
                cta_group=2,
            )
            if accumulate_twice:
                Tx.gemm_async(
                    accumulator[:, :],
                    left_shared[:, :],
                    right_shared[:, :],
                    accum=True,
                    cta_group=2,
                )
            T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
                T.address_of(barrier[0]), T.uint16(3)
            )
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        for col in T.serial(16):
            output[cta * 64 + row, half * 16 + col] = accumulator[row, half * 16 + col]
        T.cuda.cluster_sync()

    return kernel


dense_gemm_async_m64_cta2_layout_b_n32 = _layout_b_kernel(
    tmem_datapath_layout("B", 64, 32), wrong_grouping=False, accumulate_twice=True
)
dense_gemm_async_m64_cta2_wrong_b_grouping_n32 = _layout_b_kernel(
    _TMEM_WRONG_B_ROW_COLUMN_GROUP_N32, wrong_grouping=True, accumulate_twice=False
)


@T.prim_func
def dense_gemm_async_two_cta_pairs_in_one_cluster_n32(
    left: T.Buffer((256, 16), "float16"),
    right: T.Buffer((64, 16), "float16"),
    output: T.Buffer((256, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([4])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 32), "float32", scope="tmem", layout=tmem_datapath_layout("B", 64, 32), allocated_addr=0
    )
    tmem_lane = T.meta_var(warp * 32 + lane)
    row = T.meta_var(tmem_lane % 64)
    half = T.meta_var(tmem_lane // 64)
    if warp == 0 and lane == 0:
        Tx.copy(left_shared[:, :], left[cta * 64 : (cta + 1) * 64, :])
        right_start: T.let = (cta // 2) * 32 + (cta % 2) * 16
        Tx.copy(right_shared[:, :], right[right_start : right_start + 16, :])
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if ((cta % 2) == 0) and (warp == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=False,
            cta_group=2,
        )
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(barrier[0]), T.Cast("uint16", T.shift_left(3, cta))
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    for col in T.serial(16):
        output[cta * 64 + row, half * 16 + col] = accumulator[row, half * 16 + col]
    T.cuda.cluster_sync()


# -- tests --------------------------------------------------------------------


def test_dense_gemm_async_gathers_physical_operands_and_accumulates_tmem():
    """Replaces ``tests/numsim/integration/test_gemm_async_artifact.py::test_dense_gemm_async_gathers_physical_operands_and_accumulates_tmem``.

    Row L4: ``dense_gemm_async_cta1`` was M=128 N=8 (``mma_n=8``) at
    cta_group::1; this copy is N=16 (``mma_n=16``). Oracle unchanged: two MMAs
    (overwrite, then accumulate) give ``2 * left @ right.T``."""

    left = _f16(128, 16, 17, 8)
    right = _f16(16, 16, 11, 5)
    result = v2.Engine().run(
        v2.transpile(dense_gemm_async_cta1_n16),
        {"left": left, "right": right, "output": np.zeros((128, 16), dtype=np.float32)},
    )
    np.testing.assert_array_equal(result.outputs["output"], _matmul(left, right) * np.float32(2))


def test_dense_fp8_gemm_async_matches_numpy():
    """Replaces ``tests/numsim/integration/test_gemm_async_artifact.py::test_dense_fp8_gemm_async_matches_numpy``.

    Row L4: ``dense_fp8_gemm_async_cta1`` was M=128 N=8; this copy is N=16
    (B in a 16x128 SWIZZLE_128B layout)."""

    fp8 = pytest.importorskip("ml_dtypes").float8_e4m3fn
    left = (np.arange(128 * 128, dtype=np.float32).reshape(128, 128) % 5 - 2).astype(fp8)
    right = (np.arange(16 * 128, dtype=np.float32).reshape(16, 128) % 5 - 2).astype(fp8)
    result = v2.Engine().run(
        v2.transpile(dense_fp8_gemm_async_cta1_n16),
        {"left": left, "right": right, "output": np.zeros((128, 16), dtype=np.float32)},
    )
    np.testing.assert_array_equal(result.outputs["output"], _matmul(left, right))


def test_cta_group2_gathers_both_shared_shards_and_scatters_tmem():
    """Replaces ``tests/numsim/integration/test_gemm_async_artifact.py::test_cta_group2_gathers_both_shared_shards_and_scatters_tmem``.

    Row L4: ``dense_gemm_async_cta_group2`` was M=256 N=16 at cta_group::2
    (8 B rows per CTA); this copy is N=32 (16 B rows per CTA). Dropped the
    legacy worker/scheduling-domain stats pins."""

    left = _f16(256, 16, 19, 9)
    right = _f16(32, 16, 13, 6)
    result = v2.Engine(max_workers=2).run(
        v2.transpile(dense_gemm_async_cta_group2_n32),
        {"left": left, "right": right, "output": np.zeros((256, 32), dtype=np.float32)},
    )
    np.testing.assert_array_equal(result.outputs["output"], _matmul(left, right))


def test_dense_gemm_async_handles_repeated_dynamic_index_loads():
    """Replaces ``tests/numsim/integration/test_gemm_async_artifact.py::test_dense_gemm_async_handles_repeated_dynamic_index_loads``.

    Row L4: ``dense_gemm_async_dynamic_right_index`` was M=128 N=8 (B =
    ``right_shared[1:9]`` of 16 rows); this copy is N=16 (B =
    ``right_shared[1:17]`` of 32 rows), same runtime start index 1."""

    left = _f16(128, 16, 17, 8)
    right = _f16(32, 16, 11, 5)
    result = v2.Engine().run(
        v2.transpile(dense_gemm_async_dynamic_right_index_n16),
        {"left": left, "right": right, "output": np.zeros((128, 16), dtype=np.float32)},
    )
    np.testing.assert_array_equal(result.outputs["output"], _matmul(left, right[1:17]))


def test_cta_group2_gathers_both_tmem_a_shards():
    """Replaces ``tests/numsim/integration/test_gemm_async_artifact.py::test_cta_group2_gathers_both_tmem_a_shards``.

    Row L4: ``dense_gemm_async_tmem_a_cta_group2`` was M=256 N=16 at
    cta_group::2; this copy is N=32. Each CTA still contributes its own TMEM A
    rows (written per warp sub-partition, row T4)."""

    left = _f16(256, 16, 19, 9)
    right = _f16(32, 16, 13, 6)
    result = v2.Engine(max_workers=2).run(
        v2.transpile(dense_gemm_async_tmem_a_cta_group2_n32),
        {"left": left, "right": right, "output": np.zeros((256, 32), dtype=np.float32)},
    )
    np.testing.assert_array_equal(result.outputs["output"], _matmul(left, right))


def test_dense_gemm_async_runs_numpy_backend_on_two_cluster_workers():
    """Replaces ``tests/numsim/integration/test_gemm_async_artifact.py::test_dense_gemm_async_runs_numpy_backend_on_two_cluster_workers``.

    Row L4: ``dense_gemm_async_two_clusters`` was M=128 N=8 per CTA; this copy
    is N=16 (CTA ``c`` uses ``right[16c:16c+16]``). Dropped the
    ``worker_count``/``scheduling_domain_count`` stats pins."""

    left = _f16(256, 16, 17, 8)
    right = _f16(32, 16, 11, 5)
    result = v2.Engine(max_workers=2).run(
        v2.transpile(dense_gemm_async_two_clusters_n16),
        {"left": left, "right": right, "output": np.zeros((256, 16), dtype=np.float32)},
    )
    expected = np.concatenate(
        [_matmul(left[c * 128 : (c + 1) * 128], right[c * 16 : (c + 1) * 16]) for c in range(2)],
        axis=0,
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_all_inactive_gemm_async_is_a_noop():
    """Replaces ``tests/numsim/integration/test_gemm_async_artifact.py::test_all_inactive_gemm_async_is_a_noop``.

    Row L4: ``inactive_gemm_async_is_noop`` declared an M=128 N=8 MMA; this
    copy declares N=16. The MMA is never issued; the kernel writes 7."""

    result = v2.Engine().run(
        v2.transpile(inactive_gemm_async_is_noop_n16), {"output": np.zeros(1, dtype=np.int32)}
    )
    np.testing.assert_array_equal(result.outputs["output"], np.array([7], dtype=np.int32))


def test_cta_group2_m64_tcgen_mma_uses_layout_b_independently_of_declared_layout():
    """Replaces the Layout-B half of ``tests/numsim/integration/test_gemm_async_artifact.py::test_cta_group2_m64_tcgen_mma_uses_layout_b_independently_of_declared_layout``.

    Row L4: ``dense_gemm_async_m64_cta2_layout_b`` was M=128 N=16 at
    cta_group::2; this copy is N=32. The MMA writes Layout B and the kernel
    reads it back through Layout B: numpy-exact."""

    left = _f16(128, 16, 19, 9)
    right = _f16(32, 16, 13, 6)
    product = _matmul(left, right)

    correct = v2.Engine().run(
        v2.transpile(dense_gemm_async_m64_cta2_layout_b_n32),
        {"left": left, "right": right, "output": np.zeros((128, 32), dtype=np.float32)},
    )
    np.testing.assert_array_equal(correct.outputs["output"], product * np.float32(2))


def test_cta_group2_m64_tcgen_mma_wrong_declared_grouping_fails_closed():
    """Replaces the wrong-grouping half of ``tests/numsim/integration/test_gemm_async_artifact.py::test_cta_group2_m64_tcgen_mma_uses_layout_b_independently_of_declared_layout``.

    Row L8 (numsim-behaviour-deltas): TVM's tcgen05 dispatch requires the TMEM
    accumulator's declared layout to equal the MMA output layout, so a kernel
    declaring the accumulator without Layout B's column grouping is rejected
    at transpile (legacy ran it and swapped the off-diagonal quadrants)."""

    with pytest.raises(UnsupportedTIRxError, match="StructuralEqual|tile form"):
        v2.transpile(dense_gemm_async_m64_cta2_wrong_b_grouping_n32)


def test_dense_gemm_async_no_swizzle_descriptor_matches_numpy():
    """Replaces ``tests/numsim/integration/test_gemm_async_artifact.py::test_dense_gemm_async_no_swizzle_descriptor_matches_numpy``.

    Row L4: ``dense_gemm_async_no_swizzle_shared`` was M=128 N=8; this copy is
    N=16 (B in a 16x16 SWIZZLE_NONE layout)."""

    left = _f16(128, 16, 13, 6)
    right = _f16(16, 16, 7, 3)
    result = v2.Engine().run(
        v2.transpile(dense_gemm_async_no_swizzle_shared_n16),
        {"left": left, "right": right, "output": np.zeros((128, 16), dtype=np.float32)},
    )
    np.testing.assert_array_equal(result.outputs["output"], _matmul(left, right))


def test_cta_group2_routes_each_pair_within_a_four_cta_cluster():
    """Replaces ``tests/numsim/integration/test_gemm_async_artifact.py::test_cta_group2_routes_each_pair_within_a_four_cta_cluster``.

    Row L4: ``dense_gemm_async_two_cta_pairs_in_one_cluster`` was M=128 N=16 at
    cta_group::2 per pair; this copy is N=32 (pair ``p`` uses
    ``right[32p:32p+32]``, 16 rows per CTA). Each even CTA commits to its own
    pair (multicast mask ``3 << cta``)."""

    left = _f16(256, 16, 19, 9)
    right = _f16(64, 16, 13, 6)
    expected = np.concatenate(
        [
            _matmul(left[c * 64 : (c + 1) * 64], right[(c // 2) * 32 : (c // 2 + 1) * 32])
            for c in range(4)
        ],
        axis=0,
    )
    result = v2.Engine().run(
        v2.transpile(dense_gemm_async_two_cta_pairs_in_one_cluster_n32),
        {"left": left, "right": right, "output": np.zeros((256, 32), dtype=np.float32)},
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_thread_scope_gemm_async_requires_one_runtime_issuer():
    """Replaces ``tests/numsim/integration/test_gemm_async_artifact.py::test_thread_scope_gemm_async_requires_one_runtime_issuer``.

    Row L4: ``dense_gemm_async_two_thread_issuers`` was M=128 N=8; this copy is
    N=16. Two lanes reach a thread-scope ``gemm_async``: the run stops with an
    execution error (legacy text "exactly one active issuing lane" not pinned)."""

    with pytest.raises(v2.ExecutionError):
        v2.Engine().run(
            v2.transpile(dense_gemm_async_two_thread_issuers_n16),
            {
                "left": np.zeros((128, 16), dtype=np.float16),
                "right": np.zeros((16, 16), dtype=np.float16),
                "output": np.zeros((1,), dtype=np.int32),
            },
        )
