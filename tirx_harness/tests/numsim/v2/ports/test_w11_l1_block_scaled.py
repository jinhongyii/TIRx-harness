"""v2 copies of six ``tests/numsim/integration/test_block_scaled_gemm_artifact.py``
tests blocked by numsim-behaviour-deltas L1 (W11, blocked-v2 batch).

Each legacy kernel stores its scale factors straight into the replicated
scale-factor TMEM views (``scale_a_tmem`` / ``scale_b_tmem`` with an
``sf_tmem_layout``), which v2 rejects at transpile (``tmem_replicated_view``,
contract item 29). Every legacy function gets:

1. an expected-error test on the verbatim legacy kernel (``UnsupportedTIRxError``
   matching ``tmem_replicated_view``, row L1); and
2. a corrected kernel asserting the legacy observable outcome, written like
   ``test_validshape_block_scaled_gemm_artifact.py``:
   - scales reach TMEM the hardware way: ``Tx.copy`` into an ``sf_smem_layout``
     shared buffer, then ``Tx.copy_async`` (``tcgen05.cp``) into the physical
     ``sf_tmem_layout`` TMEM buffer;
   - the MMA is committed to an mbarrier and waited on (``tcgen05.mma`` and
     ``tcgen05.cp`` are asynchronous);
   - four warps read back their own TMEM sub-partition (row T4).

   The cta_group::1 kernels issued M=128 N=8 MMAs, which row L4 rejects, so their
   copies use N=16; B gets eight extra rows and SFB rows are padded to 128. The
   legacy columns 0..7 keep the legacy values, and the oracle covers all 16
   columns. cta_group::2 kernels keep M=256 N=256; each CTA ``tcgen05.cp``-s its
   own scales, commits and waits before the cluster sync that precedes the MMA.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import ComposeLayout, S, TileLayout, tmem_datapath_layout
from tvm.backend.cuda.tile_primitive.gemm_async.tcgen05 import sf_smem_layout, sf_tmem_layout
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout

from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine

_FP8_MMA_128X128 = mma_shared_layout("float8_e4m3fn", SwizzleMode.SWIZZLE_128B_ATOM, (128, 128))
_FP8_MMA_8X128 = mma_shared_layout("float8_e4m3fn", SwizzleMode.SWIZZLE_128B_ATOM, (8, 128))
_FP8_MMA_16X128 = mma_shared_layout("float8_e4m3fn", SwizzleMode.SWIZZLE_128B_ATOM, (16, 128))
_PACKED_FP4_MMA_128X32 = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (128, 32))
_PACKED_FP4_MMA_8X32 = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (8, 32))
_PACKED_FP4_MMA_16X32 = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (16, 32))
_MXF8_SF_SMEM = sf_smem_layout(128, SF_K=4, sf_per_mma=1)
_MXF8_SF_TMEM = sf_tmem_layout(128, SF_K=4, sf_per_mma=1)
_NVFP4_SF_SMEM = sf_smem_layout(128, SF_K=4, sf_per_mma=4)
_NVFP4_SF_TMEM = sf_tmem_layout(128, SF_K=4, sf_per_mma=4)
_NVFP4_SF_SMEM_256 = sf_smem_layout(256, SF_K=4, sf_per_mma=4)
_NVFP4_SF_TMEM_256 = sf_tmem_layout(256, SF_K=4, sf_per_mma=4)


# -- legacy kernels (verbatim) -----------------------------------------------


@T.prim_func
def _block_scaled_fp8_dynamic_shared_stage(
    left: T.Buffer((2, 128, 128), "float8_e4m3fn"),
    right: T.Buffer((2, 8, 128), "float8_e4m3fn"),
    scale_a: T.Buffer((128, 4), "float8_e8m0fnu"),
    scale_b: T.Buffer((8, 4), "float8_e8m0fnu"),
    stage: T.int32,
    output: T.Buffer((128, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer(
        (2, 128, 128),
        "float8_e4m3fn",
        scope="shared",
        layout=ComposeLayout(4, 3, 3, TileLayout(S[(1024,)])),
    )
    right_shared = T.alloc_buffer(
        (2, 8, 128),
        "float8_e4m3fn",
        scope="shared",
        layout=ComposeLayout(4, 3, 3, TileLayout(S[(1024,)])),
    )
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    scale_a_tmem = T.decl_buffer(
        (128, 4),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=1),
        allocated_addr=16,
    )
    scale_b_tmem = T.decl_buffer(
        (128, 4),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=1),
        allocated_addr=20,
    )
    if lane == 0:
        Tx.copy(left_shared[stage, :, :], left[stage, :, :])
        Tx.copy(right_shared[stage, :, :], right[stage, :, :])
        for row in T.serial(128):
            for scale_index in T.serial(4):
                scale_a_tmem[row, scale_index] = scale_a[row, scale_index]
        for row in T.serial(8):
            for scale_index in T.serial(4):
                scale_b_tmem[row, scale_index] = scale_b[row, scale_index]
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[stage, :, :],
            right_shared[stage, :, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
        )
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def _block_scaled_runtime_instruction_descriptor(
    left: T.Buffer((128, 128), "float8_e4m3fn"),
    right: T.Buffer((8, 128), "float8_e4m3fn"),
    scale_a: T.Buffer((128, 4), "float8_e8m0fnu"),
    scale_b: T.Buffer((8, 4), "float8_e8m0fnu"),
    selector: T.int32,
    corrupt_descriptor: T.int32,
    output: T.Buffer((128, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer(
        (128, 128), "float8_e4m3fn", scope="shared", layout=_FP8_MMA_128X128
    )
    right_shared = T.alloc_buffer((8, 128), "float8_e4m3fn", scope="shared", layout=_FP8_MMA_8X128)
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    scale_a_tmem = T.decl_buffer(
        (128, 4),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=1),
        allocated_addr=16,
    )
    scale_b_tmem = T.decl_buffer(
        (128, 4),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=1),
        allocated_addr=20,
    )
    descriptor: T.uint32
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        for row in T.serial(128):
            for scale_index in T.serial(4):
                scale_a_tmem[row, scale_index] = scale_a[row, scale_index]
        for row in T.serial(8):
            for scale_index in T.serial(4):
                scale_b_tmem[row, scale_index] = scale_b[row, scale_index]
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(descriptor),
            d_dtype="float32",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e4m3fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=16,
            sfb_tmem_addr=20,
            M=128,
            N=8,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        if corrupt_descriptor != 0:
            descriptor = descriptor ^ T.uint32(1 << 17)
        T.cuda.runtime_instr_desc(T.address_of(descriptor), T.Cast("uint32", selector))
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
            descI=descriptor,
        )
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def _block_scaled_nvfp4_gemm_cta_group2_pair23(
    left_packed: T.Buffer((4, 128, 32), "uint8"),
    right_packed: T.Buffer((4, 128, 32), "uint8"),
    scale_a: T.Buffer((4, 128, 4), "float8_e4m3fn"),
    scale_b: T.Buffer((4, 256, 4), "float8_e4m3fn"),
    output: T.Buffer((4, 128, 256), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([4])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared_packed = T.alloc_buffer(
        (128, 32), "uint8", scope="shared", layout=_PACKED_FP4_MMA_128X32
    )
    right_shared_packed = T.alloc_buffer(
        (128, 32), "uint8", scope="shared", layout=_PACKED_FP4_MMA_128X32
    )
    left_shared = left_shared_packed.view("float4_e2m1fn")
    right_shared = right_shared_packed.view("float4_e2m1fn")
    accumulator = T.decl_buffer(
        (128, 256),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 256),
        allocated_addr=0,
    )
    scale_a_tmem = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=4),
        allocated_addr=256,
    )
    scale_b_tmem = T.decl_buffer(
        (256, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(256, SF_K=4, sf_per_mma=4),
        allocated_addr=264,
    )
    if lane == 0:
        Tx.copy(left_shared_packed[:, :], left_packed[cta, :, :])
        Tx.copy(right_shared_packed[:, :], right_packed[cta, :, :])
        for row in T.serial(128):
            for scale_index in T.serial(4):
                scale_a_tmem[row, scale_index] = scale_a[cta, row, scale_index]
        for row in T.serial(256):
            for scale_index in T.serial(4):
                scale_b_tmem[row, scale_index] = scale_b[cta, row, scale_index]
    T.cuda.cluster_sync()
    if (cta == 2) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=2,
        )
    T.cuda.cluster_sync()
    if (cta >= 2) and (lane == 0):
        for row in T.serial(128):
            for col in T.serial(256):
                output[cta, row, col] = accumulator[row, col]


@T.prim_func
def block_scaled_nvfp4_gemm(
    left_packed: T.Buffer((128, 32), "uint8"),
    right_packed: T.Buffer((8, 32), "uint8"),
    scale_a: T.Buffer((128, 4), "float8_e4m3fn"),
    scale_b: T.Buffer((8, 4), "float8_e4m3fn"),
    output: T.Buffer((128, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared_packed = T.alloc_buffer(
        (128, 32), "uint8", scope="shared", layout=_PACKED_FP4_MMA_128X32
    )
    right_shared_packed = T.alloc_buffer(
        (8, 32), "uint8", scope="shared", layout=_PACKED_FP4_MMA_8X32
    )
    left_shared = left_shared_packed.view("float4_e2m1fn")
    right_shared = right_shared_packed.view("float4_e2m1fn")
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    scale_a_tmem = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=4),
        allocated_addr=16,
    )
    scale_b_tmem = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=4),
        allocated_addr=24,
    )
    if lane == 0:
        Tx.copy(left_shared_packed[:, :], left_packed[:, :])
        Tx.copy(right_shared_packed[:, :], right_packed[:, :])
        for row in T.serial(128):
            for scale_index in T.serial(4):
                scale_a_tmem[row, scale_index] = scale_a[row, scale_index]
        for row in T.serial(8):
            for scale_index in T.serial(4):
                scale_b_tmem[row, scale_index] = scale_b[row, scale_index]
    T.cuda.warp_sync()
    if lane == 0:
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
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def block_scaled_nvfp4_gemm_cta_group2_scale_rows(
    left_packed: T.Buffer((2, 128, 32), "uint8"),
    right_packed: T.Buffer((2, 128, 32), "uint8"),
    scale_a: T.Buffer((2, 128, 4), "float8_e4m3fn"),
    scale_b: T.Buffer((2, 256, 4), "float8_e4m3fn"),
    output: T.Buffer((2, 128, 256), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared_packed = T.alloc_buffer(
        (128, 32), "uint8", scope="shared", layout=_PACKED_FP4_MMA_128X32
    )
    right_shared_packed = T.alloc_buffer(
        (128, 32), "uint8", scope="shared", layout=_PACKED_FP4_MMA_128X32
    )
    left_shared = left_shared_packed.view("float4_e2m1fn")
    right_shared = right_shared_packed.view("float4_e2m1fn")
    accumulator = T.decl_buffer(
        (128, 256),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 256),
        allocated_addr=0,
    )
    scale_a_tmem = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=4),
        allocated_addr=256,
    )
    scale_b_tmem = T.decl_buffer(
        (256, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(256, SF_K=4, sf_per_mma=4),
        allocated_addr=264,
    )
    if lane == 0:
        Tx.copy(left_shared_packed[:, :], left_packed[cta, :, :])
        Tx.copy(right_shared_packed[:, :], right_packed[cta, :, :])
        for row in T.serial(128):
            for scale_index in T.serial(4):
                scale_a_tmem[row, scale_index] = scale_a[cta, row, scale_index]
        for row in T.serial(256):
            for scale_index in T.serial(4):
                scale_b_tmem[row, scale_index] = scale_b[cta, row, scale_index]
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=2,
        )
    T.cuda.cluster_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(256):
                output[cta, row, col] = accumulator[row, col]


# -- corrected kernels ---------------------------------------------------------


@T.prim_func
def fixed_fp8_dynamic_shared_stage(
    left: T.Buffer((2, 128, 128), "float8_e4m3fn"),
    right: T.Buffer((2, 16, 128), "float8_e4m3fn"),
    scale_a: T.Buffer((128, 4), "float8_e8m0fnu"),
    scale_b: T.Buffer((128, 4), "float8_e8m0fnu"),
    stage: T.int32,
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    left_shared = T.alloc_buffer(
        (2, 128, 128),
        "float8_e4m3fn",
        scope="shared",
        layout=ComposeLayout(4, 3, 3, TileLayout(S[(1024,)])),
    )
    right_shared = T.alloc_buffer(
        (2, 16, 128),
        "float8_e4m3fn",
        scope="shared",
        layout=ComposeLayout(4, 3, 3, TileLayout(S[(1024,)])),
    )
    scale_a_shared = T.alloc_buffer((128, 4), "float8_e8m0fnu", scope="shared", layout=_MXF8_SF_SMEM)
    scale_b_shared = T.alloc_buffer((128, 4), "float8_e8m0fnu", scope="shared", layout=_MXF8_SF_SMEM)
    accumulator = T.decl_buffer(
        (128, 16), "float32", scope="tmem", layout=tmem_datapath_layout("D", 128, 16), allocated_addr=0
    )
    scale_a_tmem = T.decl_buffer(
        (128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM, allocated_addr=16
    )
    scale_b_tmem = T.decl_buffer(
        (128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM, allocated_addr=20
    )
    if warp == 0 and lane == 0:
        Tx.copy(left_shared[stage, :, :], left[stage, :, :])
        Tx.copy(right_shared[stage, :, :], right[stage, :, :])
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
            left_shared[stage, :, :],
            right_shared[stage, :, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
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
def fixed_runtime_instruction_descriptor(
    left: T.Buffer((128, 128), "float8_e4m3fn"),
    right: T.Buffer((16, 128), "float8_e4m3fn"),
    scale_a: T.Buffer((128, 4), "float8_e8m0fnu"),
    scale_b: T.Buffer((128, 4), "float8_e8m0fnu"),
    selector: T.int32,
    corrupt_descriptor: T.int32,
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
    scale_a_tmem = T.decl_buffer(
        (128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM, allocated_addr=16
    )
    scale_b_tmem = T.decl_buffer(
        (128, 4), "float8_e8m0fnu", scope="tmem", layout=_MXF8_SF_TMEM, allocated_addr=20
    )
    descriptor: T.uint32
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
        Tx.copy_async(scale_a_tmem[:, :], scale_a_shared[:, :], cta_group=1)
        Tx.copy_async(scale_b_tmem[:, :], scale_b_shared[:, :], cta_group=1)
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(descriptor),
            d_dtype="float32",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e4m3fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=16,
            sfb_tmem_addr=20,
            M=128,
            N=16,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        if corrupt_descriptor != 0:
            descriptor = descriptor ^ T.uint32(1 << 17)
        T.cuda.runtime_instr_desc(T.address_of(descriptor), T.Cast("uint32", selector))
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
            descI=descriptor,
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
def fixed_nvfp4_gemm_n16(
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
def fixed_nvfp4_gemm_cta_group2_scale_rows(
    left_packed: T.Buffer((2, 128, 32), "uint8"),
    right_packed: T.Buffer((2, 128, 32), "uint8"),
    scale_a: T.Buffer((2, 128, 4), "float8_e4m3fn"),
    scale_b: T.Buffer((2, 256, 4), "float8_e4m3fn"),
    output: T.Buffer((2, 128, 256), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    left_shared_packed = T.alloc_buffer((128, 32), "uint8", scope="shared", layout=_PACKED_FP4_MMA_128X32)
    right_shared_packed = T.alloc_buffer((128, 32), "uint8", scope="shared", layout=_PACKED_FP4_MMA_128X32)
    left_shared = left_shared_packed.view("float4_e2m1fn")
    right_shared = right_shared_packed.view("float4_e2m1fn")
    scale_a_shared = T.alloc_buffer((128, 4), "float8_e4m3fn", scope="shared", layout=_NVFP4_SF_SMEM)
    scale_b_shared = T.alloc_buffer((256, 4), "float8_e4m3fn", scope="shared", layout=_NVFP4_SF_SMEM_256)
    accumulator = T.decl_buffer(
        (128, 256), "float32", scope="tmem", layout=tmem_datapath_layout("D", 128, 256), allocated_addr=0
    )
    scale_a_tmem = T.decl_buffer(
        (128, 4), "float8_e4m3fn", scope="tmem", layout=_NVFP4_SF_TMEM, allocated_addr=256
    )
    scale_b_tmem = T.decl_buffer(
        (256, 4), "float8_e4m3fn", scope="tmem", layout=_NVFP4_SF_TMEM_256, allocated_addr=264
    )
    if warp == 0 and lane == 0:
        Tx.copy(left_shared_packed[:, :], left_packed[cta, :, :])
        Tx.copy(right_shared_packed[:, :], right_packed[cta, :, :])
        Tx.copy(scale_a_shared[:, :], scale_a[cta, :, :])
        Tx.copy(scale_b_shared[:, :], scale_b[cta, :, :])
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (warp == 0) and (lane == 0):
        # cta_group::2 tcgen05.cp: each CTA of the pair copies its own shared
        # scales into its own TMEM; the commit arrives on both CTAs' barriers.
        Tx.copy_async(scale_a_tmem[:, :], scale_a_shared[:, :], cta_group=2)
        Tx.copy_async(scale_b_tmem[:, :], scale_b_shared[:, :], cta_group=2)
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(barriers[0]), T.uint16(3)
        )
    if cta >= 0:
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cluster_sync()
    T.ptx.tcgen05.fence__after_thread_sync()
    if (cta == 0) and (warp == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=2,
        )
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(barriers[1]), T.uint16(3)
        )
    if cta >= 0:
        T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        row = T.meta_var(warp * 32 + lane)
        for col in T.serial(256):
            output[cta, row, col] = accumulator[row, col]
    T.cuda.cluster_sync()


@T.prim_func
def fixed_nvfp4_gemm_cta_group2_pair23(
    left_packed: T.Buffer((4, 128, 32), "uint8"),
    right_packed: T.Buffer((4, 128, 32), "uint8"),
    scale_a: T.Buffer((4, 128, 4), "float8_e4m3fn"),
    scale_b: T.Buffer((4, 256, 4), "float8_e4m3fn"),
    output: T.Buffer((4, 128, 256), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([4])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    left_shared_packed = T.alloc_buffer((128, 32), "uint8", scope="shared", layout=_PACKED_FP4_MMA_128X32)
    right_shared_packed = T.alloc_buffer((128, 32), "uint8", scope="shared", layout=_PACKED_FP4_MMA_128X32)
    left_shared = left_shared_packed.view("float4_e2m1fn")
    right_shared = right_shared_packed.view("float4_e2m1fn")
    scale_a_shared = T.alloc_buffer((128, 4), "float8_e4m3fn", scope="shared", layout=_NVFP4_SF_SMEM)
    scale_b_shared = T.alloc_buffer((256, 4), "float8_e4m3fn", scope="shared", layout=_NVFP4_SF_SMEM_256)
    accumulator = T.decl_buffer(
        (128, 256), "float32", scope="tmem", layout=tmem_datapath_layout("D", 128, 256), allocated_addr=0
    )
    scale_a_tmem = T.decl_buffer(
        (128, 4), "float8_e4m3fn", scope="tmem", layout=_NVFP4_SF_TMEM, allocated_addr=256
    )
    scale_b_tmem = T.decl_buffer(
        (256, 4), "float8_e4m3fn", scope="tmem", layout=_NVFP4_SF_TMEM_256, allocated_addr=264
    )
    if warp == 0 and lane == 0:
        Tx.copy(left_shared_packed[:, :], left_packed[cta, :, :])
        Tx.copy(right_shared_packed[:, :], right_packed[cta, :, :])
        Tx.copy(scale_a_shared[:, :], scale_a[cta, :, :])
        Tx.copy(scale_b_shared[:, :], scale_b[cta, :, :])
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 2) and (warp == 0) and (lane == 0):
        # cta_group::2 tcgen05.cp: each CTA of the pair copies its own shared
        # scales into its own TMEM; the commit arrives on both CTAs' barriers.
        Tx.copy_async(scale_a_tmem[:, :], scale_a_shared[:, :], cta_group=2)
        Tx.copy_async(scale_b_tmem[:, :], scale_b_shared[:, :], cta_group=2)
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(barriers[0]), T.uint16(12)
        )
    if cta >= 2:
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cluster_sync()
    T.ptx.tcgen05.fence__after_thread_sync()
    if (cta == 2) and (warp == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=2,
        )
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(barriers[1]), T.uint16(12)
        )
    if cta >= 2:
        T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        row = T.meta_var(warp * 32 + lane)
        for col in T.serial(256):
            output[cta, row, col] = accumulator[row, col]
    T.cuda.cluster_sync()


# -- oracle helpers (copied from the legacy module) -----------------------------


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


def _decode_e2m1(bits: np.ndarray) -> np.ndarray:
    values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    bits = np.asarray(bits, dtype=np.uint8) & np.uint8(0xF)
    magnitude = values[(bits & np.uint8(0x7)).astype(np.intp)]
    return np.where((bits & np.uint8(0x8)) != 0, -magnitude, magnitude).astype(np.float32)


def _unpack_e2m1(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    result = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.uint8)
    result[..., 0::2] = packed & np.uint8(0xF)
    result[..., 1::2] = packed >> np.uint8(4)
    return result


def _expected_fp8_per_ki_scales(
    left: np.ndarray, right: np.ndarray, scale_a: np.ndarray, scale_b: np.ndarray
) -> np.ndarray:
    a = _decode_e4m3(left)
    b = _decode_e4m3(right)
    a_scales = _decode_e8m0(scale_a)
    b_scales = _decode_e8m0(scale_b)
    result = np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)
    for scale_index, start in enumerate(range(0, left.shape[1], 32)):
        k_slice = slice(start, start + 32)
        result += (a[:, k_slice] * a_scales[:, scale_index, None]) @ (
            b[:, k_slice] * b_scales[:, scale_index, None]
        ).T
    return result


def _assert_l1_rejected(kernel) -> None:
    with pytest.raises(UnsupportedTIRxError, match="tmem_replicated_view"):
        v2.transpile(kernel)


def _pad_rows(array: np.ndarray, rows: int, fill: int) -> np.ndarray:
    padded = np.full((rows, *array.shape[1:]), fill, dtype=array.dtype)
    padded[: array.shape[0]] = array
    return padded


# -- test_block_scaled_desc_i_rotates_scale_bytes_for_each_ki -----------------


def test_block_scaled_desc_i_rotates_scale_bytes_for_each_ki_rejects_replicated_view():
    """Legacy kernel ``_block_scaled_runtime_instruction_descriptor``: rejected (row L1)."""
    _assert_l1_rejected(_block_scaled_runtime_instruction_descriptor)


def test_block_scaled_desc_i_rotates_scale_bytes_for_each_ki():
    """Corrected kernel (module docstring; N=16 per row L4). For every runtime
    selector the MMA ``ki`` reads scale byte ``ki``: the legacy per-ki oracle,
    exact, on all 16 columns (columns 0..7 are the legacy inputs)."""
    left = np.resize(np.array([0x30, 0x38, 0x40, 0xB8], dtype=np.uint8), (128, 128))
    right = np.resize(np.array([0x28, 0x38, 0x40], dtype=np.uint8), (16, 128))
    scale_a = np.tile(np.array([125, 126, 127, 128], dtype=np.uint8), (128, 1))
    scale_b = np.tile(np.array([128, 127, 126, 125], dtype=np.uint8), (128, 1))
    module = v2.transpile(fixed_runtime_instruction_descriptor)
    expected = _expected_fp8_per_ki_scales(left, right, scale_a, scale_b[:16])
    for selector in range(4):
        result = v2.Engine().run(
            module,
            {
                "left": left,
                "right": right,
                "scale_a": scale_a,
                "scale_b": scale_b,
                "selector": selector,
                "corrupt_descriptor": 0,
                "output": np.zeros((128, 16), dtype=np.float32),
            },
        )
        np.testing.assert_array_equal(result.outputs["output"], expected)


# -- test_block_scaled_desc_i_rejects_static_abi_mismatch_at_runtime ----------


def test_block_scaled_desc_i_rejects_static_abi_mismatch_at_runtime():
    """Expected-error copy only (row L1): the legacy kernel is rejected at transpile.

    The corrected kernel cannot carry the legacy assertion yet: with
    ``corrupt_descriptor=1`` (bit 17, the N field, flipped in the runtime
    ``descI``) v2 runs to completion with different values instead of rejecting
    a runtime descriptor that disagrees with the typed ``gemm_async`` ABI (open
    v2 gap, reported to W11)."""
    _assert_l1_rejected(_block_scaled_runtime_instruction_descriptor)



@v2_gap("W11-7: a runtime descI that disagrees with the typed gemm_async ABI (N bit flipped) runs to completion instead of being rejected")
def test_block_scaled_desc_i_rejects_static_abi_mismatch_at_runtime_corrected_kernel():
    """Corrected kernel, legacy assertion: flipping the runtime ``descI`` N bit
    (``corrupt_descriptor=1``) must stop the run (legacy: "does not match the
    typed TCGEN ABI"). CONTRACT_REQUESTS W11-7."""
    left = np.resize(np.array([0x30, 0x38, 0x40, 0xB8], dtype=np.uint8), (128, 128))
    right = np.resize(np.array([0x28, 0x38, 0x40], dtype=np.uint8), (16, 128))
    scale = np.full((128, 4), 127, dtype=np.uint8)
    module = v2.transpile(fixed_runtime_instruction_descriptor)
    with pytest.raises(v2.ExecutionError):
        v2.Engine().run(
            module,
            {
                "left": left,
                "right": right,
                "scale_a": scale,
                "scale_b": scale,
                "selector": 0,
                "corrupt_descriptor": 1,
                "output": np.zeros((128, 16), dtype=np.float32),
            },
        )

# -- test_fp8_snapshot_gather_honors_a_dynamic_shared_stage --------------------


def test_fp8_snapshot_gather_honors_a_dynamic_shared_stage_rejects_replicated_view():
    """Legacy kernel ``_block_scaled_fp8_dynamic_shared_stage``: rejected (row L1)."""
    _assert_l1_rejected(_block_scaled_fp8_dynamic_shared_stage)


def test_fp8_snapshot_gather_honors_a_dynamic_shared_stage():
    """Corrected kernel (N=16, row L4): the MMA reads the runtime ``stage`` 1
    of the shared operands: 128 x (2.0 x 1.0) = 256 everywhere, as legacy."""
    left = np.full((2, 128, 128), 0x38, dtype=np.uint8)
    left[1] = np.uint8(0x40)
    right = np.full((2, 16, 128), 0x38, dtype=np.uint8)
    result = v2.Engine().run(
        v2.transpile(fixed_fp8_dynamic_shared_stage),
        {
            "left": left,
            "right": right,
            "scale_a": np.full((128, 4), 127, dtype=np.uint8),
            "scale_b": np.full((128, 4), 127, dtype=np.uint8),
            "stage": 1,
            "output": np.zeros((128, 16), dtype=np.float32),
        },
    )
    np.testing.assert_array_equal(result.outputs["output"], np.full((128, 16), 256.0, dtype=np.float32))


# -- test_nvfp4_block_scaled_gemm_decodes_nibbles_and_e4m3_scales -------------


def test_nvfp4_block_scaled_gemm_decodes_nibbles_and_e4m3_scales_rejects_replicated_view():
    """Legacy kernel ``block_scaled_nvfp4_gemm``: rejected (row L1)."""
    _assert_l1_rejected(block_scaled_nvfp4_gemm)


def test_nvfp4_block_scaled_gemm_decodes_nibbles_and_e4m3_scales():
    """Corrected kernel (N=16, row L4): e2m1 nibbles and e4m3 scales decode to
    the legacy independent oracle; rows 8..15 of B repeat the legacy pattern."""
    left_codes = np.resize(np.array([0x0, 0x1, 0x2, 0x3, 0x7, 0x9, 0xA, 0xF], dtype=np.uint8), (128, 64))
    right_codes = np.resize(np.array([0x1, 0x2, 0x4, 0x7, 0x9, 0xB], dtype=np.uint8), (16, 64))
    left_packed = (left_codes[:, 0::2] | (left_codes[:, 1::2] << np.uint8(4))).astype(np.uint8)
    right_packed = (right_codes[:, 0::2] | (right_codes[:, 1::2] << np.uint8(4))).astype(np.uint8)
    scale_a = np.resize(np.array([0x30, 0x38, 0x3C, 0x40], dtype=np.uint8), (128, 4))
    legacy_scale_b = np.resize(
        np.array(
            [
                [0x38, 0x40, 0x30, 0x3C],
                [0x40, 0x38, 0x3C, 0x30],
                [0x30, 0x3C, 0x38, 0x40],
                [0x3C, 0x30, 0x40, 0x38],
            ],
            dtype=np.uint8,
        ),
        (16, 4),
    )
    scale_b = _pad_rows(legacy_scale_b, 128, 0x38)
    result = v2.Engine().run(
        v2.transpile(fixed_nvfp4_gemm_n16),
        {
            "left_packed": left_packed,
            "right_packed": right_packed,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "output": np.zeros((128, 16), dtype=np.float32),
        },
    )
    a = (
        _decode_e2m1(_unpack_e2m1(left_packed)).reshape(128, 4, 16) * _decode_e4m3(scale_a)[:, :, None]
    ).reshape(128, 64)
    b = (
        _decode_e2m1(_unpack_e2m1(right_packed)).reshape(16, 4, 16) * _decode_e4m3(legacy_scale_b)[:, :, None]
    ).reshape(16, 64)
    np.testing.assert_array_equal(result.outputs["output"], a @ b.T)


# -- test_cta_group2_right_scales_use_the_combined_n_row ----------------------


def test_cta_group2_right_scales_use_the_combined_n_row_rejects_replicated_view():
    """Legacy kernel ``block_scaled_nvfp4_gemm_cta_group2_scale_rows``: rejected (row L1)."""
    _assert_l1_rejected(block_scaled_nvfp4_gemm_cta_group2_scale_rows)


def test_cta_group2_right_scales_use_the_combined_n_row():
    """Corrected kernel (cta_group::2 scale copy, commit/wait, T4 read-back):
    SFB rows 128..255 (scale 2.0) apply to output columns 128..255 of both CTAs."""
    left_packed = np.full((2, 128, 32), 0x22, dtype=np.uint8)
    right_packed = np.full((2, 128, 32), 0x22, dtype=np.uint8)
    scale_a = np.full((2, 128, 4), 0x38, dtype=np.uint8)
    scale_b = np.full((2, 256, 4), 0x38, dtype=np.uint8)
    scale_b[:, 128:, :] = np.uint8(0x40)
    result = v2.Engine().run(
        v2.transpile(fixed_nvfp4_gemm_cta_group2_scale_rows),
        {
            "left_packed": left_packed,
            "right_packed": right_packed,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "output": np.zeros((2, 128, 256), dtype=np.float32),
        },
    )
    expected = np.full((2, 128, 256), 64.0, dtype=np.float32)
    expected[:, :, 128:] = np.float32(128.0)
    np.testing.assert_array_equal(result.outputs["output"], expected)


# -- test_cta_group2_uses_the_issuing_ctas_pair_2_and_3 -----------------------


def test_cta_group2_uses_the_issuing_ctas_pair_2_and_3_rejects_replicated_view():
    """Legacy kernel ``_block_scaled_nvfp4_gemm_cta_group2_pair23``: rejected (row L1)."""
    _assert_l1_rejected(_block_scaled_nvfp4_gemm_cta_group2_pair23)


def test_cta_group2_uses_the_issuing_ctas_pair_2_and_3():
    """Corrected kernel: the MMA issued by CTA 2 uses the operands and scales of
    pair (2, 3) only, never pair (0, 1); CTAs 0 and 1 stay zero."""
    left_packed = np.full((4, 128, 32), 0x77, dtype=np.uint8)
    right_packed = np.full((4, 128, 32), 0x77, dtype=np.uint8)
    left_packed[2:] = np.uint8(0x22)
    right_packed[2:] = np.uint8(0x22)
    scale_a = np.full((4, 128, 4), 0x40, dtype=np.uint8)
    scale_b = np.full((4, 256, 4), 0x40, dtype=np.uint8)
    scale_a[2:] = np.uint8(0x38)
    scale_b[2:] = np.uint8(0x38)
    scale_b[3, 128:, :] = np.uint8(0x40)
    result = v2.Engine().run(
        v2.transpile(fixed_nvfp4_gemm_cta_group2_pair23),
        {
            "left_packed": left_packed,
            "right_packed": right_packed,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "output": np.zeros((4, 128, 256), dtype=np.float32),
        },
    )
    expected = np.zeros((4, 128, 256), dtype=np.float32)
    expected[2:, :, :128] = np.float32(64.0)
    expected[2:, :, 128:] = np.float32(128.0)
    np.testing.assert_array_equal(result.outputs["output"], expected)
