"""v2 copies of the 17 ``tests/numsim/runtime/test_raw_tcgen_codegen.py`` tests that
fail under ``NUMSIM_IMPL=v2`` only because a single warp touches TMEM lanes
outside its 32-lane sub-partition (numsim-behaviour-deltas T4: a warp may
access only TMEM lanes ``32 * (warp_id % 4) .. +32``; ``bad_address``, legacy
accepted it). W11 blocked-v2 batch.

For each legacy test:

1. ``test_<legacy>_original_kernel_violates_tmem_sub_partition``: the legacy
   kernel, verbatim, stops with ``bad_address`` at its first out-of-partition
   TMEM access (warp, faulting lane and source line asserted).
2. ``test_<legacy>``: a corrected kernel with the same MMA shapes,
   descriptors, operands and data paths. It launches four warps (one per TMEM
   sub-partition); setup and the MMA issue stay on warp 0; every TMEM access
   names only the issuing warp's lanes; and the MMA is committed to an mbarrier
   that every thread waits on before reading TMEM (the legacy kernels read TMEM
   after a plain ``cta_sync``; v2 lands an async ``tcgen05.mma`` only when
   ordered, the same model as ``tcgen05.cp`` in delta T20). The legacy test's
   exact expected outputs are asserted.

Kernels and helpers are copied verbatim from the legacy module.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.backend.cuda.tile_primitive.gemm_async.tcgen05 import sf_tmem_layout
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TCol, TileLayout, TLane, tmem_datapath_layout

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine

_TMEM_D_8 = TileLayout(S[(128, 8) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_16 = TileLayout(S[(128, 16) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_32 = TileLayout(S[(128, 32) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_104 = TileLayout(S[(128, 104) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_136 = TileLayout(S[(128, 136) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_512 = TileLayout(S[(128, 512) : (1 @ TLane, 1 @ TCol)])
_TMEM_NVF4_24 = TileLayout(S[(128, 24) : (1 @ TLane, 1 @ TCol)])
_PACKED_FP4_SMEM_128X64 = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (128, 64))
_PACKED_FP4_SMEM_8X64 = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (8, 64))


# -- legacy kernels and helpers (verbatim) -------------------------------------


@T.prim_func
def raw_tcgen_mma_block_scaled_mxf4_mqa(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((8192,), "uint8"),
    scale_a_cells: T.Buffer((4, 32), "uint32"),
    scale_b_cells: T.Buffer((4, 32), "uint32"),
    issue_second: T.int32,
    output: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
        shared_b[offset] = b_physical[offset]
    for row_group in T.unroll(4):
        tmem[lane, 128 + row_group] = scale_a_cells[row_group, lane]
        tmem[lane, 132 + row_group] = scale_b_cells[row_group, lane]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float4_e2m1fn",
            b_dtype="float4_e2m1fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=128,
            N=128,
            K=64,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(128),
            T.uint32(132),
            T.ptx.pred(T.uint32(0)),
            pred=lane == 0,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[32]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[32]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X"](
            T.uint32(0),
            desc_a,
            desc_b,
            T.bitwise_or(desc_i, T.uint32((2 << 29) | (2 << 4))),
            T.uint32(128),
            T.uint32(132),
            T.ptx.pred(T.uint32(1)),
            pred=T.And(lane == 0, issue_second != 0),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(128):
            output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def raw_tcgen_mma_block_scaled_mxf4nvf4_two_k_tiles(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((8192,), "uint8"),
    scale_a_cells: T.Buffer((2, 4, 32), "uint32"),
    scale_b_cells: T.Buffer((2, 32), "uint32"),
    output: T.Buffer((128, 8), "float32"),
    use_ue8m0_scales: T.int32,
):
    """`.kind::mxf4nvf4.block_scale.scale_vec::4X` over two K=64 tiles.

    `.scale_vec::4X` fixes SFA/SFB ID at 0 and spends all four bytes of one
    TMEM word on one K tile, so the second tile advances the scale *address*
    (four TMEM columns for M=128, one for N=8) instead of the descriptor's
    scale-factor ID the way `.kind::mxf4`'s `.scale_vec::2X` does. Descriptor
    bit 23 selects whether those bytes decode as UE8M0 or UE4M3.
    """

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 24), "uint32", scope="tmem", layout=_TMEM_NVF4_24, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
        shared_b[offset] = b_physical[offset]
    for k_tile in T.unroll(2):
        for row_group in T.unroll(4):
            tmem[lane, 8 + k_tile * 4 + row_group] = scale_a_cells[k_tile, row_group, lane]
        tmem[lane, 16 + k_tile] = scale_b_cells[k_tile, lane]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float4_e2m1fn",
            b_dtype="float4_e2m1fn",
            sfa_dtype="float8_e4m3fn",
            sfb_dtype="float8_e4m3fn",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=128,
            N=8,
            K=64,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        if use_ue8m0_scales != 0:
            desc_i = T.bitwise_or(desc_i, T.uint32(1 << 23))
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(8),
            T.uint32(16),
            T.ptx.pred(T.uint32(0)),
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[32]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[32]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(12),
            T.uint32(17),
            T.ptx.pred(T.uint32(1)),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(8):
            output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def typed_nvfp4_gemm_two_k_tiles(
    left_packed: T.Buffer((128, 64), "uint8"),
    right_packed: T.Buffer((8, 64), "uint8"),
    scale_a: T.Buffer((128, 8), "float8_e4m3fn"),
    scale_b: T.Buffer((8, 8), "float8_e4m3fn"),
    output: T.Buffer((128, 8), "float32"),
):
    """The typed `Tx.gemm_async` oracle for the raw mxf4nvf4 kernel above."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared_packed = T.alloc_buffer(
        (128, 64), "uint8", scope="shared", layout=_PACKED_FP4_SMEM_128X64
    )
    right_shared_packed = T.alloc_buffer(
        (8, 64), "uint8", scope="shared", layout=_PACKED_FP4_SMEM_8X64
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
        (128, 8),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=8, sf_per_mma=4),
        allocated_addr=16,
    )
    scale_b_tmem = T.decl_buffer(
        (8, 8),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=8, sf_per_mma=4),
        allocated_addr=32,
    )
    if lane == 0:
        Tx.copy(left_shared_packed[:, :], left_packed[:, :])
        Tx.copy(right_shared_packed[:, :], right_packed[:, :])
        for row in T.serial(128):
            for scale_index in T.serial(8):
                scale_a_tmem[row, scale_index] = scale_a[row, scale_index]
        for row in T.serial(8):
            for scale_index in T.serial(8):
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
def raw_tcgen_mma_block_scaled_mxf4_expression_input_d(
    mode: T.int32,
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((8192,), "uint8"),
    scale_a_cells: T.Buffer((4, 32), "uint32"),
    scale_b_cells: T.Buffer((4, 32), "uint32"),
    output: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
        shared_b[offset] = b_physical[offset]
    for row_group in T.unroll(4):
        tmem[lane, 128 + row_group] = scale_a_cells[row_group, lane]
        tmem[lane, 132 + row_group] = scale_b_cells[row_group, lane]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float4_e2m1fn",
            b_dtype="float4_e2m1fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=128,
            N=128,
            K=64,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(128),
            T.uint32(132),
            T.ptx.pred(T.cast(mode == 2, "uint32")),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(128):
            output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def raw_tcgen_mma_e4m3_mqa(
    a_physical: T.Buffer((16384,), "uint8"),
    b_physical: T.Buffer((16384,), "uint8"),
    output: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((16384,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((16384,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(512):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
        shared_b[offset] = b_physical[offset]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e4m3fn",
            M=128,
            N=128,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        for k_block in T.unroll(4):
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_a), T.address_of(shared_a[k_block * 32]), ldo=0, sdo=64, swizzle=3
            )
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_b), T.address_of(shared_b[k_block * 32]), ldo=0, sdo=64, swizzle=3
            )
            T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
                T.uint32(0),
                desc_a,
                desc_b,
                desc_i,
                T.uint32(0),
                T.uint32(0),
                T.uint32(0),
                T.uint32(0),
                T.ptx.pred(T.cast(k_block, "uint32")),
            )
    T.cuda.cta_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(128):
            output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def raw_tcgen_mma_tf32_ss(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((1024,), "uint8"),
    output: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
    for copy_i in T.serial(32):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[offset]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="tf32",
            b_dtype="tf32",
            M=64,
            N=8,
            K=8,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::tf32"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for col in T.unroll(8):
            output[row, col] = T.reinterpret("float32", tmem[physical_lane, col])


@T.prim_func
def raw_tcgen_mma_tf32_ts_predicated(
    a: T.Buffer((64, 8), "float32"),
    b_physical: T.Buffer((4096,), "uint8"),
    output: T.Buffer((64, 32), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_b = T.alloc_buffer((4096,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_b: T.uint64
    desc_b_low: T.uint32
    desc_b_replaced: T.uint64

    for copy_i in T.serial(128):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[offset]
    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for k in T.unroll(8):
            tmem[physical_lane, k] = T.reinterpret("uint32", a[row, k])
        for col in T.serial(32):
            tmem[physical_lane, 16 + col] = T.reinterpret("uint32", T.float32(4))
    T.cuda.cta_sync()

    T.cuda.tcgen05.encode_instr_descriptor(
        T.address_of(desc_i),
        d_dtype="float32",
        a_dtype="tf32",
        b_dtype="tf32",
        M=64,
        N=32,
        K=8,
        trans_a=False,
        trans_b=False,
        n_cta_groups=1,
    )
    T.cuda.tcgen05.encode_matrix_descriptor(
        T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
    )
    desc_b_low = T.cast(desc_b, "uint32")
    desc_b_replaced = T.bitwise_or(
        T.bitwise_and(desc_b, T.bitwise_not(T.uint64(0xFFFFFFFF))), T.cast(desc_b_low, "uint64")
    )
    T.ptx["tcgen05.mma.cta_group::1.kind::tf32"](
        T.uint32(16),
        T.uint32(0),
        desc_b_replaced,
        desc_i,
        T.uint32(1 << 3),
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        T.ptx.pred(T.uint32(1)),
        pred=T.cast(lane == 7, "uint32"),
    )
    T.cuda.cta_sync()

    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for col in T.serial(32):
            output[row, col] = T.reinterpret("float32", tmem[physical_lane, 16 + col])


@T.prim_func
def raw_tcgen_mma_e4m3_m64_n8(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((1024,), "uint8"),
    output: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
    for copy_i in T.serial(32):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[offset]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e4m3fn",
            M=64,
            N=8,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for col in T.unroll(8):
            output[row, col] = T.reinterpret("float32", tmem[physical_lane, col])


@T.prim_func
def raw_tcgen_mma_e5m2_e4m3_m64_n8(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((1024,), "uint8"),
    output: T.Buffer((64, 8), "float32"),
):
    """`kind::f8f6f4` with an E5M2 A and an E4M3 B: independent operand dtypes."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
    for copy_i in T.serial(32):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[offset]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float8_e5m2",
            b_dtype="float8_e4m3fn",
            M=64,
            N=8,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for col in T.unroll(8):
            output[row, col] = T.reinterpret("float32", tmem[physical_lane, col])


@T.prim_func
def raw_tcgen_mma_f8f6f4_f16_destination_m64_n8(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((1024,), "uint8"),
    seed: T.Buffer((128, 8), "uint32"),
    output: T.Buffer((64, 8), "uint32"),
):
    """`kind::f8f6f4` accumulating into a float16 destination, read as raw words."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
    for copy_i in T.serial(32):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[offset]
    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.unroll(8):
            tmem[row, col] = seed[row, col]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float16",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e4m3fn",
            M=64,
            N=8,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(1)),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for col in T.unroll(8):
            output[row, col] = tmem[physical_lane, col]


@T.prim_func
def raw_tcgen_mma_tf32_m128_n16(
    a: T.Buffer((128, 8), "float32"),
    b_physical: T.Buffer((2048,), "uint8"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_b = T.alloc_buffer((2048,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 32), "uint32", scope="tmem", layout=_TMEM_D_32, allocated_addr=0)
    desc_i: T.uint32
    desc_b: T.uint64

    for copy_i in T.serial(64):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[offset]
    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for k in T.unroll(8):
            tmem[row, k] = T.reinterpret("uint32", a[row, k])
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="tf32",
            b_dtype="tf32",
            M=128,
            N=16,
            K=8,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::tf32"](
            T.uint32(16),
            T.uint32(0),
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.unroll(16):
            output[row, col] = T.reinterpret("float32", tmem[row, 16 + col])


@T.prim_func
def raw_tcgen_mma_bf16_ss(
    a_physical: T.Buffer((4096,), "uint8"),
    b_physical: T.Buffer((1024,), "uint8"),
    output: T.Buffer((64, 8), "float32"),
    output_ws: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    prefix = T.alloc_buffer((128,), "uint8", scope="shared")
    shared_a = T.alloc_buffer((4096,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    descriptor_table = T.alloc_buffer((1,), "uint64", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_a_loaded: T.uint64
    desc_b: T.uint64
    prefix[lane] = T.cast(lane, "uint8")
    for copy_i in T.serial(128):
        shared_a[lane + copy_i * 32] = a_physical[lane + copy_i * 32]
    for copy_i in T.serial(32):
        shared_b[lane + copy_i * 32] = b_physical[lane + copy_i * 32]
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="bfloat16",
            b_dtype="bfloat16",
            M=64,
            N=8,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx.st.shared.u64(descriptor_table.ptr_to([0]), desc_a)
        T.ptx.ld.shared.u64(desc_a_loaded, descriptor_table.ptr_to([0]))
        T.ptx["tcgen05.mma.cta_group::1.kind::f16.collector::a::discard"](
            T.uint32(0),
            desc_a_loaded,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cta_sync()
    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for col in T.unroll(8):
            output[row, col] = T.reinterpret("float32", _tmem[physical_lane, col])
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"](
            T.uint32(0),
            desc_a_loaded,
            desc_b,
            desc_i,
            T.ptx.pred(T.uint32(0)),
            T.uint64(0),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[1])
        )
    T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
    T.cuda.cta_sync()
    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        for col in T.unroll(8):
            physical_lane = row + (col // 4) * 64
            physical_col = col % 4
            output_ws[row, col] = T.reinterpret("float32", _tmem[physical_lane, physical_col])


@T.prim_func
def raw_tcgen_mma_ws_bf16_ss_layout_e_tail(
    a_physical: T.Buffer((4096,), "uint8"),
    b_physical: T.Buffer((8192,), "uint8"),
    output: T.Buffer((64, 128), "float32"),
):
    """M64 `.ws` writes N halves into the two Layout-E lane banks."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((4096,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 512), "uint32", scope="tmem", layout=_TMEM_D_512, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64
    for copy_i in T.serial(128):
        shared_a[lane + copy_i * 32] = a_physical[lane + copy_i * 32]
    for copy_i in T.serial(256):
        shared_b[lane + copy_i * 32] = b_physical[lane + copy_i * 32]
    T.cuda.cta_sync()
    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="bfloat16",
            b_dtype="bfloat16",
            M=64,
            N=128,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"](
            T.uint32(400),
            desc_a,
            desc_b,
            desc_i,
            T.ptx.pred(T.uint32(0)),
            T.uint64(0),
        )
    T.cuda.cta_sync()
    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        for col in T.serial(128):
            physical_lane = row + (col // 64) * 64
            physical_col = 400 + col % 64
            output[row, col] = T.reinterpret("float32", tmem[physical_lane, physical_col])


@T.prim_func
def raw_tcgen_mma_f16_ts_mn_major_n128(
    a_packed: T.Buffer((128, 8), "uint32"),
    b_physical: T.Buffer((16384,), "uint8"),
    output: T.Buffer((128, 128), "float32"),
    output_ws: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_b = T.alloc_buffer((16384,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_b: T.uint64

    for copy_i in T.serial(512):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[offset]
    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for packed_k in T.unroll(8):
            tmem[row, packed_k] = a_packed[row, packed_k]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=128,
            N=128,
            K=16,
            trans_a=False,
            trans_b=True,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b),
            T.address_of(shared_b[0]),
            ldo=512,
            sdo=64,
            swizzle=3,
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::f16"](
            T.uint32(8),
            T.uint32(0),
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(128):
            output[row, col] = T.reinterpret("float32", tmem[row, 8 + col])
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"](
            T.uint32(8),
            T.uint32(0),
            desc_b,
            desc_i,
            T.ptx.pred(T.uint32(0)),
            T.uint64(0),
        )
    T.cuda.cta_sync()
    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(128):
            output_ws[row, col] = T.reinterpret("float32", tmem[row, 8 + col])


@T.prim_func
def raw_tcgen_mma_f16_cta2_datapaths(
    a_ss256_physical: T.Buffer((2, 16384), "uint8"),
    b_ss256_physical: T.Buffer((2, 1024), "uint8"),
    a_ts256_packed: T.Buffer((2, 128, 8), "uint32"),
    b_ts256_physical: T.Buffer((2, 8192), "uint8"),
    a_ss128_physical: T.Buffer((2, 8192), "uint8"),
    b_ss128_physical: T.Buffer((2, 8192), "uint8"),
    output_ss256: T.Buffer((256, 16), "float32"),
    output_ts256: T.Buffer((256, 16), "float32"),
    output_ss128: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a_ss256 = T.alloc_buffer((16384,), "uint8", scope="shared")
    shared_b_ss256 = T.alloc_buffer((1024,), "uint8", scope="shared")
    shared_b_ts256 = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_a_ss128 = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b_ss128 = T.alloc_buffer((8192,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 104), "uint32", scope="tmem", layout=_TMEM_D_104, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(512):
        offset = lane + copy_i * 32
        shared_a_ss256[offset] = a_ss256_physical[cta, offset]
    for copy_i in T.serial(32):
        offset = lane + copy_i * 32
        shared_b_ss256[offset] = b_ss256_physical[cta, offset]
    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_b_ts256[offset] = b_ts256_physical[cta, offset]
        shared_a_ss128[offset] = a_ss128_physical[cta, offset]
        shared_b_ss128[offset] = b_ss128_physical[cta, offset]
    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for packed_k in T.unroll(8):
            tmem[row, packed_k] = a_ts256_packed[cta, row, packed_k]
    T.cuda.cluster_sync()

    if cta == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=256,
            N=16,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a_ss256[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b_ss256[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
            T.uint32(8),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )

        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=256,
            N=16,
            K=16,
            trans_a=False,
            trans_b=True,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b_ts256[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
            T.uint32(24),
            T.uint32(0),
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )

        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=128,
            N=128,
            K=16,
            trans_a=True,
            trans_b=True,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a_ss128[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b_ss128[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
            T.uint32(40),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cluster_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(16):
            output_ss256[cta * 128 + row, col] = T.reinterpret("float32", tmem[row, 8 + col])
            output_ts256[cta * 128 + row, col] = T.reinterpret("float32", tmem[row, 24 + col])
    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        for col in T.serial(128):
            physical_lane = row + 64 * (col // 64)
            physical_col = col % 64
            output_ss128[cta * 64 + row, col] = T.reinterpret(
                "float32", tmem[physical_lane, 40 + physical_col]
            )


@T.prim_func
def raw_tcgen_mma_f8f6f4_cta2_rubin_k64_discard(
    a_physical: T.Buffer((2, 16384), "uint8"),
    b_physical: T.Buffer((2, 8192), "uint8"),
    seed: T.Buffer((2, 128, 128), "float32"),
    output: T.Buffer((2, 128, 128), "float32"),
):
    """Rubin's M256/N128/K64 form with a B descriptor above the SM100 address ceiling."""

    T.func_attr({"tirx.cuda_arch": "sm_107a"})
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_arena = T.alloc_buffer((327936,), "uint8", scope="shared", align=1024)
    shared_a = T.decl_buffer((16384,), "uint8", data=shared_arena.data, scope="shared")
    shared_b = T.decl_buffer(
        (8192,), "uint8", data=shared_arena.data, elem_offset=262144, scope="shared"
    )
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(512):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[cta, offset]
    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[cta, offset]
    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(128):
            tmem[row, col] = T.reinterpret("uint32", seed[cta, row, col])
    T.cuda.cluster_sync()

    if cta == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        # CUTLASS include/cute/arch/mma_sm100_desc.hpp makes this a 15-bit
        # field for SM107A/F. The source kernel likewise patches/adds the low
        # descriptor half rather than calling the SM100-oriented TVM encoder.
        desc_b = T.bitwise_or(
            T.bitwise_and(desc_b, T.bitwise_not(T.uint64(0x7FFF))),
            T.cast(
                T.shift_right(T.cuda.cvta_generic_to_shared(shared_b.ptr_to([0])), T.uint32(4)),
                "uint64",
            ),
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard"](
            T.uint32(0),
            desc_a,
            desc_b,
            T.uint32(0x30210490),
            T.uint32(1 << 5),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(1 << 7),
            T.uint32(0),
            T.ptx.pred(T.uint32(1)),
        )
    T.cuda.cluster_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(128):
            output[cta, row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def raw_tcgen_mma_mxf8_mn_major(
    a_physical: T.Buffer((4096,), "uint8"),
    b_physical: T.Buffer((4096,), "uint8"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((4096,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((4096,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 512), "uint32", scope="tmem", layout=_TMEM_D_512, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(128):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
        shared_b[offset] = b_physical[offset]
    for replica in T.unroll(4):
        for row_group in T.unroll(4):
            tmem[replica * 32 + lane, 256 + row_group] = T.uint32(0x7F7F7F7F)
        tmem[replica * 32 + lane, 260] = T.uint32(0x7F7F7F7F)
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e4m3fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=1 << 30,
            sfb_tmem_addr=1 << 30,
            M=128,
            N=16,
            K=32,
            trans_a=True,
            trans_b=True,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32((1 << 30) | 256),
            T.uint32((1 << 30) | 260),
            False,
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(16):
            output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def raw_tcgen_mma_mxf8_k_major_cta1(
    a_physical: T.Buffer((16384,), "uint8"),
    b_physical: T.Buffer((16384,), "uint8"),
    output: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((16384,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((16384,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 512), "uint32", scope="tmem", layout=_TMEM_D_512, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0:
        for copy_i in T.serial(512):
            offset = lane + copy_i * 32
            shared_a[offset] = a_physical[offset]
            shared_b[offset] = b_physical[offset]
        for replica in T.unroll(4):
            for row_group in T.unroll(4):
                tmem[replica * 32 + lane, 256 + row_group] = T.uint32(0x7F7F7F7F)
                tmem[replica * 32 + lane, 260 + row_group] = T.uint32(0x7F7F7F7F)
    T.cuda.cta_sync()

    if (warp == 0) and (lane == 0):
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e4m3fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=128,
            N=128,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=1, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=1, sdo=64, swizzle=3
        )
        for kblock in T.unroll(4):
            T.cuda.runtime_instr_desc(T.address_of(desc_i), T.cast(kblock, "uint32"))
            T.ptx["tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32"](
                T.uint32(0),
                desc_a + T.cast(kblock * 2, "uint64"),
                desc_b + T.cast(kblock * 2, "uint64"),
                desc_i,
                T.uint32(256 + kblock * (1 << 30)),
                T.uint32(260 + kblock * (1 << 30)),
                kblock != 0,
            )
    T.cuda.cta_sync()

    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(128):
        output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def raw_tcgen_mma_mixed_fp4_fp8_cta_group2(
    a_physical: T.Buffer((2, 16384), "uint8"),
    b_physical: T.Buffer((2, 2048), "uint8"),
    scale_a_cells: T.Buffer((2, 4, 32), "uint32"),
    scale_b_cells: T.Buffer((2, 1, 32), "uint32"),
    output: T.Buffer((2, 128, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((16384,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((2048,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(512):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[cta, offset]
    for copy_i in T.serial(64):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[cta, offset]
    for replica in T.unroll(4):
        for row_group in T.unroll(4):
            tmem[replica * 32 + lane, 32 + row_group] = scale_a_cells[cta, row_group, lane]
        tmem[replica * 32 + lane, 36] = scale_b_cells[cta, 0, lane]
    T.cuda.cluster_sync()

    if cta == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float4_e2m1fn",
            b_dtype="float8_e4m3fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=256,
            N=32,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(32),
            T.uint32(36),
            False,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[32]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[32]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"](
            T.uint32(0),
            desc_a,
            desc_b,
            T.bitwise_or(desc_i, T.uint32((1 << 29) | (1 << 4))),
            T.uint32(32),
            T.uint32(36),
            True,
        )
    T.cuda.cluster_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(32):
            output[cta, row, col] = T.reinterpret("float32", tmem[row, col])


def _bfloat16_bits(values: np.ndarray) -> np.ndarray:
    return (
        np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint32) >> np.uint32(16)
    ).astype(np.uint16)


def _mxf4_kmajor_physical(logical_nibbles: np.ndarray) -> np.ndarray:
    rows, k = logical_nibbles.shape
    assert rows == 128 and k == 128
    packed = logical_nibbles[:, 0::2].astype(np.uint8) | (
        logical_nibbles[:, 1::2].astype(np.uint8) << np.uint8(4)
    )
    return _kmajor_swizzle_physical(packed, swizzle_len=2, sdo=512)


def _kmajor_swizzle_physical(
    logical_bytes: np.ndarray, *, swizzle_len: int, sdo: int
) -> np.ndarray:
    rows, row_bytes = logical_bytes.shape
    physical = np.zeros(((rows + 7) // 8) * sdo, dtype=np.uint8)
    row_stride = 16 << swizzle_len
    swizzle_mask = (1 << swizzle_len) - 1
    for row in range(rows):
        for byte_in_row in range(row_bytes):
            atom = byte_in_row // 16
            byte_in_atom = byte_in_row % 16
            unswizzled = (row % 8) * row_stride + (row // 8) * sdo + atom * 16 + byte_in_atom
            atom_index = unswizzled >> 4
            swizzled_atom = atom_index ^ ((atom_index & (swizzle_mask << 3)) >> 3)
            physical[(swizzled_atom << 4) | byte_in_atom] = logical_bytes[row, byte_in_row]
    return physical


def _kmajor_swizzle_physical_at_start(
    logical_bytes: np.ndarray,
    *,
    start: int,
    swizzle_len: int,
    sdo: int,
    byte_len: int,
) -> np.ndarray:
    physical = np.zeros(byte_len, dtype=np.uint8)
    row_stride = 16 << swizzle_len
    swizzle_mask = (1 << swizzle_len) - 1
    for row in range(logical_bytes.shape[0]):
        for byte_in_row in range(logical_bytes.shape[1]):
            atom = byte_in_row // 16
            byte_in_atom = byte_in_row % 16
            unswizzled = (
                start + (row % 8) * row_stride + (row // 8) * sdo + atom * 16 + byte_in_atom
            )
            atom_index = unswizzled >> 4
            swizzled_atom = atom_index ^ ((atom_index & (swizzle_mask << 3)) >> 3)
            physical_offset = ((swizzled_atom << 4) | byte_in_atom) - start
            if not 0 <= physical_offset < byte_len:
                raise ValueError("swizzled address leaves the selected physical buffer")
            physical[physical_offset] = logical_bytes[row, byte_in_row]
    return physical


def _mnmajor_f16_swizzle_physical(logical_bits: np.ndarray) -> np.ndarray:
    """Encode the PTX canonical MN-major 128B-swizzled f16 layout."""
    assert logical_bits.shape == (128, 16)
    physical = np.zeros(16384, dtype=np.uint8)
    for n in range(128):
        for k in range(16):
            unswizzled = (n % 64) * 2 + (n // 64) * 8192 + (k % 8) * 128 + (k // 8) * 1024
            byte_in_atom = unswizzled & 15
            atom = unswizzled >> 4
            swizzled_atom = atom ^ ((atom & (0x7 << 3)) >> 3)
            offset = (swizzled_atom << 4) | byte_in_atom
            physical[offset : offset + 2] = np.frombuffer(
                np.uint16(logical_bits[n, k]).tobytes(), dtype=np.uint8
            )
    return physical


def _mnmajor_f16_swizzle_physical_cta2(logical_bits: np.ndarray) -> np.ndarray:
    """Encode one CTA's at-most-64-row MN-major f16 operand."""
    assert logical_bits.ndim == 2 and logical_bits.shape[0] <= 64 and logical_bits.shape[1] == 16
    physical = np.zeros(8192, dtype=np.uint8)
    for row in range(logical_bits.shape[0]):
        for k in range(16):
            unswizzled = row * 2 + (k % 8) * 128 + (k // 8) * 1024
            byte_in_atom = unswizzled & 15
            atom = unswizzled >> 4
            swizzled_atom = atom ^ ((atom & (0x7 << 3)) >> 3)
            offset = (swizzled_atom << 4) | byte_in_atom
            physical[offset : offset + 2] = np.frombuffer(
                np.uint16(logical_bits[row, k]).tobytes(), dtype=np.uint8
            )
    return physical


def _mnmajor_f8_swizzle_physical(logical_bits: np.ndarray) -> np.ndarray:
    """Encode the PTX canonical MN-major 128B/16B-atomic 8-bit layout."""
    logical_bits = np.asarray(logical_bits, dtype=np.uint8)
    assert logical_bits.ndim == 2
    assert logical_bits.shape[0] <= 128 and logical_bits.shape[1] in (32, 64)
    physical = np.zeros((logical_bits.shape[1] // 8) * 1024, dtype=np.uint8)
    for row in range(logical_bits.shape[0]):
        for k in range(logical_bits.shape[1]):
            unswizzled = row + (k % 8) * 128 + (k // 8) * 1024
            byte_in_atom = unswizzled & 15
            atom = unswizzled >> 4
            swizzled_atom = atom ^ ((atom & (0x7 << 3)) >> 3)
            physical[(swizzled_atom << 4) | byte_in_atom] = logical_bits[row, k]
    return physical


def _e4m3fn_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    exponent = ((bits >> np.uint8(3)) & np.uint8(0xF)).astype(np.int16)
    mantissa = (bits & np.uint8(0x7)).astype(np.float32)
    normal = np.ldexp(np.float32(1) + mantissa / np.float32(8), exponent - 7)
    subnormal = np.ldexp(mantissa / np.float32(8), -6)
    magnitude = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    return np.where(bits & np.uint8(0x80), -magnitude, magnitude).astype(np.float32)


def _e5m2_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    exponent = ((bits >> np.uint8(2)) & np.uint8(0x1F)).astype(np.int16)
    mantissa = (bits & np.uint8(0x3)).astype(np.float32)
    normal = np.ldexp(np.float32(1) + mantissa / np.float32(4), exponent - 15)
    subnormal = np.ldexp(mantissa / np.float32(4), -14)
    magnitude = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    assert not np.any(exponent == 0x1F), "E5M2 reference inputs must stay finite"
    return np.where(bits & np.uint8(0x80), -magnitude, magnitude).astype(np.float32)


def _decode_tcgen_tf32_payload(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    return (bits & np.uint32(0xFFFFE000)).view(np.float32)


def _tcgen_fma_matmul(
    a: np.ndarray, b_nk: np.ndarray, *, initial: np.float32 = np.float32(0)
) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b_nk = np.asarray(b_nk, dtype=np.float32)
    assert a.ndim == b_nk.ndim == 2 and a.shape[1] == b_nk.shape[1]
    accumulator = np.full((a.shape[0], b_nk.shape[0]), initial, dtype=np.float32)
    for k in range(a.shape[1]):
        accumulator = (
            np.float64(a[:, k, None]) * np.float64(b_nk[None, :, k]) + np.float64(accumulator)
        ).astype(np.float32)
    return accumulator


def _pack_nvf4_scale_cells(scales: np.ndarray, row_groups: int) -> np.ndarray:
    """Pack `.scale_vec::4X` scale bytes into their `(k_tile, row_group, lane)` TMEM words.

    One K=64 tile spends all four bytes of one 32-bit TMEM word on one row, so
    consecutive K tiles land in different TMEM columns rather than in different
    byte pairs of the same word.
    """

    rows, sf_k = scales.shape
    assert sf_k % 4 == 0 and rows <= row_groups * 32
    cells = np.zeros((sf_k // 4, row_groups, 32), dtype=np.uint32)
    for row in range(rows):
        for k_tile in range(sf_k // 4):
            value = sum(int(scales[row, k_tile * 4 + byte]) << (8 * byte) for byte in range(4))
            cells[k_tile, row // 32, row % 32] = np.uint32(value)
    return cells


def _decode_e2m1_nibbles(bits: np.ndarray) -> np.ndarray:
    magnitudes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    signs = np.where(bits & np.uint8(8), np.float32(-1.0), np.float32(1.0))
    return (magnitudes[bits & np.uint8(7)] * signs).astype(np.float32)


def _pack_e8m0_cells(scales: np.ndarray) -> np.ndarray:
    assert scales.ndim == 2 and scales.shape[1] == 4 and scales.shape[0] % 32 == 0
    cells = np.zeros((scales.shape[0] // 32, 32), dtype=np.uint32)
    for row in range(scales.shape[0]):
        value = sum(int(scales[row, index]) << (8 * index) for index in range(4))
        cells[row // 32, row % 32] = np.uint32(value)
    return cells


# -- corrected kernels ---------------------------------------------------------
#
# Same MMA shapes, descriptors and data paths as the legacy kernels above.
# Changes only: four warps (one per TMEM 32-lane sub-partition); the setup and
# the MMA issue stay on warp 0; every TMEM access by a warp names only lanes
# 32*warp .. 32*warp+31 (each access is guarded by its TMEM lane's
# sub-partition); and the issuing thread commits the MMA to an mbarrier that
# every thread waits on before reading TMEM (PTX tcgen05 completion rule; v2
# lands the async MMA only when ordered, as for tcgen05.cp in delta T20).


@T.prim_func
def sub_mxf4_mqa(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((8192,), "uint8"),
    scale_a_cells: T.Buffer((4, 32), "uint32"),
    scale_b_cells: T.Buffer((4, 32), "uint32"),
    issue_second: T.int32,
    output: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0:
        for copy_i in T.serial(256):
            offset = lane + copy_i * 32
            shared_a[offset] = a_physical[offset]
            shared_b[offset] = b_physical[offset]
        for row_group in T.unroll(4):
            tmem[lane, 128 + row_group] = scale_a_cells[row_group, lane]
            tmem[lane, 132 + row_group] = scale_b_cells[row_group, lane]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float4_e2m1fn",
            b_dtype="float4_e2m1fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=128,
            N=128,
            K=64,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(128),
            T.uint32(132),
            T.ptx.pred(T.uint32(0)),
            pred=lane == 0,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[32]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[32]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X"](
            T.uint32(0),
            desc_a,
            desc_b,
            T.bitwise_or(desc_i, T.uint32((2 << 29) | (2 << 4))),
            T.uint32(128),
            T.uint32(132),
            T.ptx.pred(T.uint32(1)),
            pred=T.And(lane == 0, issue_second != 0),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(128):
        output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def sub_mxf4_expression_input_d(
    mode: T.int32,
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((8192,), "uint8"),
    scale_a_cells: T.Buffer((4, 32), "uint32"),
    scale_b_cells: T.Buffer((4, 32), "uint32"),
    output: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0:
        for copy_i in T.serial(256):
            offset = lane + copy_i * 32
            shared_a[offset] = a_physical[offset]
            shared_b[offset] = b_physical[offset]
        for row_group in T.unroll(4):
            tmem[lane, 128 + row_group] = scale_a_cells[row_group, lane]
            tmem[lane, 132 + row_group] = scale_b_cells[row_group, lane]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float4_e2m1fn",
            b_dtype="float4_e2m1fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=128,
            N=128,
            K=64,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(128),
            T.uint32(132),
            T.ptx.pred(T.cast(mode == 2, "uint32")),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(128):
        output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def sub_mxf4nvf4_two_k_tiles(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((8192,), "uint8"),
    scale_a_cells: T.Buffer((2, 4, 32), "uint32"),
    scale_b_cells: T.Buffer((2, 32), "uint32"),
    output: T.Buffer((128, 8), "float32"),
    use_ue8m0_scales: T.int32,
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 24), "uint32", scope="tmem", layout=_TMEM_NVF4_24, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0:
        for copy_i in T.serial(256):
            offset = lane + copy_i * 32
            shared_a[offset] = a_physical[offset]
            shared_b[offset] = b_physical[offset]
        for k_tile in T.unroll(2):
            for row_group in T.unroll(4):
                tmem[lane, 8 + k_tile * 4 + row_group] = scale_a_cells[k_tile, row_group, lane]
            tmem[lane, 16 + k_tile] = scale_b_cells[k_tile, lane]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float4_e2m1fn",
            b_dtype="float4_e2m1fn",
            sfa_dtype="float8_e4m3fn",
            sfb_dtype="float8_e4m3fn",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=128,
            N=8,
            K=64,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        if use_ue8m0_scales != 0:
            desc_i = T.bitwise_or(desc_i, T.uint32(1 << 23))
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(8),
            T.uint32(16),
            T.ptx.pred(T.uint32(0)),
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[32]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[32]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(12),
            T.uint32(17),
            T.ptx.pred(T.uint32(1)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(8):
        output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def sub_e4m3_mqa(
    a_physical: T.Buffer((16384,), "uint8"),
    b_physical: T.Buffer((16384,), "uint8"),
    output: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((16384,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((16384,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0:
        for copy_i in T.serial(512):
            offset = lane + copy_i * 32
            shared_a[offset] = a_physical[offset]
            shared_b[offset] = b_physical[offset]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e4m3fn",
            M=128,
            N=128,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        for k_block in T.unroll(4):
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_a), T.address_of(shared_a[k_block * 32]), ldo=0, sdo=64, swizzle=3
            )
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_b), T.address_of(shared_b[k_block * 32]), ldo=0, sdo=64, swizzle=3
            )
            T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
                T.uint32(0),
                desc_a,
                desc_b,
                desc_i,
                T.uint32(0),
                T.uint32(0),
                T.uint32(0),
                T.uint32(0),
                T.ptx.pred(T.cast(k_block, "uint32")),
            )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(128):
        output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def sub_tf32_ss(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((1024,), "uint8"),
    output: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0:
        for copy_i in T.serial(256):
            offset = lane + copy_i * 32
            shared_a[offset] = a_physical[offset]
        for copy_i in T.serial(32):
            offset = lane + copy_i * 32
            shared_b[offset] = b_physical[offset]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="tf32",
            b_dtype="tf32",
            M=64,
            N=8,
            K=8,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::tf32"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        if physical_lane // 32 == warp:
            for col in T.unroll(8):
                output[row, col] = T.reinterpret("float32", tmem[physical_lane, col])


@T.prim_func
def sub_tf32_ts_predicated(
    a: T.Buffer((64, 8), "float32"),
    b_physical: T.Buffer((4096,), "uint8"),
    output: T.Buffer((64, 32), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_b = T.alloc_buffer((4096,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_b: T.uint64
    desc_b_low: T.uint32
    desc_b_replaced: T.uint64

    if warp == 0:
        for copy_i in T.serial(128):
            offset = lane + copy_i * 32
            shared_b[offset] = b_physical[offset]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        if physical_lane // 32 == warp:
            for k in T.unroll(8):
                tmem[physical_lane, k] = T.reinterpret("uint32", a[row, k])
            for col in T.serial(32):
                tmem[physical_lane, 16 + col] = T.reinterpret("uint32", T.float32(4))
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if warp == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="tf32",
            b_dtype="tf32",
            M=64,
            N=32,
            K=8,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        desc_b_low = T.cast(desc_b, "uint32")
        desc_b_replaced = T.bitwise_or(
            T.bitwise_and(desc_b, T.bitwise_not(T.uint64(0xFFFFFFFF))), T.cast(desc_b_low, "uint64")
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::tf32"](
            T.uint32(16),
            T.uint32(0),
            desc_b_replaced,
            desc_i,
            T.uint32(1 << 3),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(1)),
            pred=T.cast(lane == 7, "uint32"),
        )
        if lane == 7:
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barrier[0])
            )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        if physical_lane // 32 == warp:
            for col in T.serial(32):
                output[row, col] = T.reinterpret("float32", tmem[physical_lane, 16 + col])


@T.prim_func
def sub_e4m3_m64_n8(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((1024,), "uint8"),
    output: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0:
        for copy_i in T.serial(256):
            offset = lane + copy_i * 32
            shared_a[offset] = a_physical[offset]
        for copy_i in T.serial(32):
            offset = lane + copy_i * 32
            shared_b[offset] = b_physical[offset]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e4m3fn",
            M=64,
            N=8,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        if physical_lane // 32 == warp:
            for col in T.unroll(8):
                output[row, col] = T.reinterpret("float32", tmem[physical_lane, col])


@T.prim_func
def sub_e5m2_e4m3_m64_n8(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((1024,), "uint8"),
    output: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0:
        for copy_i in T.serial(256):
            offset = lane + copy_i * 32
            shared_a[offset] = a_physical[offset]
        for copy_i in T.serial(32):
            offset = lane + copy_i * 32
            shared_b[offset] = b_physical[offset]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float8_e5m2",
            b_dtype="float8_e4m3fn",
            M=64,
            N=8,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        if physical_lane // 32 == warp:
            for col in T.unroll(8):
                output[row, col] = T.reinterpret("float32", tmem[physical_lane, col])


@T.prim_func
def sub_f8f6f4_f16_destination_m64_n8(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((1024,), "uint8"),
    seed: T.Buffer((128, 8), "uint32"),
    output: T.Buffer((64, 8), "uint32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0:
        for copy_i in T.serial(256):
            offset = lane + copy_i * 32
            shared_a[offset] = a_physical[offset]
        for copy_i in T.serial(32):
            offset = lane + copy_i * 32
            shared_b[offset] = b_physical[offset]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    seed_row = T.meta_var(warp * 32 + lane)
    for col in T.unroll(8):
        tmem[seed_row, col] = seed[seed_row, col]
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float16",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e4m3fn",
            M=64,
            N=8,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(1)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        if physical_lane // 32 == warp:
            for col in T.unroll(8):
                output[row, col] = tmem[physical_lane, col]


@T.prim_func
def sub_tf32_m128_n16(
    a: T.Buffer((128, 8), "float32"),
    b_physical: T.Buffer((2048,), "uint8"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_b = T.alloc_buffer((2048,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 32), "uint32", scope="tmem", layout=_TMEM_D_32, allocated_addr=0)
    desc_i: T.uint32
    desc_b: T.uint64

    if warp == 0:
        for copy_i in T.serial(64):
            offset = lane + copy_i * 32
            shared_b[offset] = b_physical[offset]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    row = T.meta_var(warp * 32 + lane)
    for k in T.unroll(8):
        tmem[row, k] = T.reinterpret("uint32", a[row, k])
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="tf32",
            b_dtype="tf32",
            M=128,
            N=16,
            K=8,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::tf32"](
            T.uint32(16),
            T.uint32(0),
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    for col in T.unroll(16):
        output[row, col] = T.reinterpret("float32", tmem[row, 16 + col])


@T.prim_func
def sub_bf16_ss(
    a_physical: T.Buffer((4096,), "uint8"),
    b_physical: T.Buffer((1024,), "uint8"),
    output: T.Buffer((64, 8), "float32"),
    output_ws: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    prefix = T.alloc_buffer((128,), "uint8", scope="shared")
    shared_a = T.alloc_buffer((4096,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    descriptor_table = T.alloc_buffer((1,), "uint64", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_a_loaded: T.uint64
    desc_b: T.uint64
    if warp == 0:
        prefix[lane] = T.cast(lane, "uint8")
        for copy_i in T.serial(128):
            shared_a[lane + copy_i * 32] = a_physical[lane + copy_i * 32]
        for copy_i in T.serial(32):
            shared_b[lane + copy_i * 32] = b_physical[lane + copy_i * 32]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="bfloat16",
            b_dtype="bfloat16",
            M=64,
            N=8,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx.st.shared.u64(descriptor_table.ptr_to([0]), desc_a)
        T.ptx.ld.shared.u64(desc_a_loaded, descriptor_table.ptr_to([0]))
        T.ptx["tcgen05.mma.cta_group::1.kind::f16.collector::a::discard"](
            T.uint32(0),
            desc_a_loaded,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        if physical_lane // 32 == warp:
            for col in T.unroll(8):
                output[row, col] = T.reinterpret("float32", _tmem[physical_lane, col])
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()
    if warp == 0 and lane == 0:
        T.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"](
            T.uint32(0),
            desc_a_loaded,
            desc_b,
            desc_i,
            T.ptx.pred(T.uint32(0)),
            T.uint64(0),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[1])
        )
    T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        for col in T.unroll(8):
            physical_lane = row + (col // 4) * 64
            physical_col = col % 4
            if physical_lane // 32 == warp:
                output_ws[row, col] = T.reinterpret("float32", _tmem[physical_lane, physical_col])


@T.prim_func
def sub_ws_bf16_ss_layout_e_tail(
    a_physical: T.Buffer((4096,), "uint8"),
    b_physical: T.Buffer((8192,), "uint8"),
    output: T.Buffer((64, 128), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((4096,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 512), "uint32", scope="tmem", layout=_TMEM_D_512, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64
    if warp == 0:
        for copy_i in T.serial(128):
            shared_a[lane + copy_i * 32] = a_physical[lane + copy_i * 32]
        for copy_i in T.serial(256):
            shared_b[lane + copy_i * 32] = b_physical[lane + copy_i * 32]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="bfloat16",
            b_dtype="bfloat16",
            M=64,
            N=128,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"](
            T.uint32(400),
            desc_a,
            desc_b,
            desc_i,
            T.ptx.pred(T.uint32(0)),
            T.uint64(0),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        for col in T.serial(128):
            physical_lane = row + (col // 64) * 64
            physical_col = 400 + col % 64
            if physical_lane // 32 == warp:
                output[row, col] = T.reinterpret("float32", tmem[physical_lane, physical_col])


@T.prim_func
def sub_f16_ts_mn_major_n128(
    a_packed: T.Buffer((128, 8), "uint32"),
    b_physical: T.Buffer((16384,), "uint8"),
    output: T.Buffer((128, 128), "float32"),
    output_ws: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_b = T.alloc_buffer((16384,), "uint8", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_b: T.uint64

    if warp == 0:
        for copy_i in T.serial(512):
            offset = lane + copy_i * 32
            shared_b[offset] = b_physical[offset]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
    row = T.meta_var(warp * 32 + lane)
    for packed_k in T.unroll(8):
        tmem[row, packed_k] = a_packed[row, packed_k]
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=128,
            N=128,
            K=16,
            trans_a=False,
            trans_b=True,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b),
            T.address_of(shared_b[0]),
            ldo=512,
            sdo=64,
            swizzle=3,
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::f16"](
            T.uint32(8),
            T.uint32(0),
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    for col in T.serial(128):
        output[row, col] = T.reinterpret("float32", tmem[row, 8 + col])
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()
    if warp == 0 and lane == 0:
        T.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"](
            T.uint32(8),
            T.uint32(0),
            desc_b,
            desc_i,
            T.ptx.pred(T.uint32(0)),
            T.uint64(0),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[1])
        )
    T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    for col in T.serial(128):
        output_ws[row, col] = T.reinterpret("float32", tmem[row, 8 + col])


@T.prim_func
def sub_f16_cta2_datapaths(
    a_ss256_physical: T.Buffer((2, 16384), "uint8"),
    b_ss256_physical: T.Buffer((2, 1024), "uint8"),
    a_ts256_packed: T.Buffer((2, 128, 8), "uint32"),
    b_ts256_physical: T.Buffer((2, 8192), "uint8"),
    a_ss128_physical: T.Buffer((2, 8192), "uint8"),
    b_ss128_physical: T.Buffer((2, 8192), "uint8"),
    output_ss256: T.Buffer((256, 16), "float32"),
    output_ts256: T.Buffer((256, 16), "float32"),
    output_ss128: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_a_ss256 = T.alloc_buffer((16384,), "uint8", scope="shared")
    shared_b_ss256 = T.alloc_buffer((1024,), "uint8", scope="shared")
    shared_b_ts256 = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_a_ss128 = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b_ss128 = T.alloc_buffer((8192,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 104), "uint32", scope="tmem", layout=_TMEM_D_104, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0:
        for copy_i in T.serial(512):
            offset = lane + copy_i * 32
            shared_a_ss256[offset] = a_ss256_physical[cta, offset]
        for copy_i in T.serial(32):
            offset = lane + copy_i * 32
            shared_b_ss256[offset] = b_ss256_physical[cta, offset]
        for copy_i in T.serial(256):
            offset = lane + copy_i * 32
            shared_b_ts256[offset] = b_ts256_physical[cta, offset]
            shared_a_ss128[offset] = a_ss128_physical[cta, offset]
            shared_b_ss128[offset] = b_ss128_physical[cta, offset]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    row = T.meta_var(warp * 32 + lane)
    for packed_k in T.unroll(8):
        tmem[row, packed_k] = a_ts256_packed[cta, row, packed_k]
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cluster_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if cta == 0 and warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=256,
            N=16,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a_ss256[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b_ss256[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
            T.uint32(8),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )

        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=256,
            N=16,
            K=16,
            trans_a=False,
            trans_b=True,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b_ts256[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
            T.uint32(24),
            T.uint32(0),
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )

        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=128,
            N=128,
            K=16,
            trans_a=True,
            trans_b=True,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a_ss128[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b_ss128[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
            T.uint32(40),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(barrier[0]), T.uint16(3)
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    for col in T.serial(16):
        output_ss256[cta * 128 + row, col] = T.reinterpret("float32", tmem[row, 8 + col])
        output_ts256[cta * 128 + row, col] = T.reinterpret("float32", tmem[row, 24 + col])
    for row_group in T.unroll(2):
        half_row = row_group * 32 + lane
        for col in T.serial(128):
            physical_lane = half_row + 64 * (col // 64)
            physical_col = col % 64
            if physical_lane // 32 == warp:
                output_ss128[cta * 64 + half_row, col] = T.reinterpret(
                    "float32", tmem[physical_lane, 40 + physical_col]
                )
    T.cuda.cluster_sync()


@T.prim_func
def sub_f8f6f4_cta2_rubin_k64_discard(
    a_physical: T.Buffer((2, 16384), "uint8"),
    b_physical: T.Buffer((2, 8192), "uint8"),
    seed: T.Buffer((2, 128, 128), "float32"),
    output: T.Buffer((2, 128, 128), "float32"),
):
    T.func_attr({"tirx.cuda_arch": "sm_107a"})
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_arena = T.alloc_buffer((327936,), "uint8", scope="shared", align=1024)
    shared_a = T.decl_buffer((16384,), "uint8", data=shared_arena.data, scope="shared")
    shared_b = T.decl_buffer(
        (8192,), "uint8", data=shared_arena.data, elem_offset=262144, scope="shared"
    )
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0:
        for copy_i in T.serial(512):
            offset = lane + copy_i * 32
            shared_a[offset] = a_physical[cta, offset]
        for copy_i in T.serial(256):
            offset = lane + copy_i * 32
            shared_b[offset] = b_physical[cta, offset]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(128):
        tmem[row, col] = T.reinterpret("uint32", seed[cta, row, col])
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cluster_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if cta == 0 and warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        desc_b = T.bitwise_or(
            T.bitwise_and(desc_b, T.bitwise_not(T.uint64(0x7FFF))),
            T.cast(
                T.shift_right(T.cuda.cvta_generic_to_shared(shared_b.ptr_to([0])), T.uint32(4)),
                "uint64",
            ),
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard"](
            T.uint32(0),
            desc_a,
            desc_b,
            T.uint32(0x30210490),
            T.uint32(1 << 5),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(1 << 7),
            T.uint32(0),
            T.ptx.pred(T.uint32(1)),
        )
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(barrier[0]), T.uint16(3)
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    for col in T.serial(128):
        output[cta, row, col] = T.reinterpret("float32", tmem[row, col])
    T.cuda.cluster_sync()


@T.prim_func
def sub_mxf8_mn_major(
    a_physical: T.Buffer((4096,), "uint8"),
    b_physical: T.Buffer((4096,), "uint8"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((4096,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((4096,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 512), "uint32", scope="tmem", layout=_TMEM_D_512, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0:
        for copy_i in T.serial(128):
            offset = lane + copy_i * 32
            shared_a[offset] = a_physical[offset]
            shared_b[offset] = b_physical[offset]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    # Each warp writes its own sub-partition's replica of the scale columns.
    row = T.meta_var(warp * 32 + lane)
    for row_group in T.unroll(4):
        tmem[row, 256 + row_group] = T.uint32(0x7F7F7F7F)
    tmem[row, 260] = T.uint32(0x7F7F7F7F)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e4m3fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=1 << 30,
            sfb_tmem_addr=1 << 30,
            M=128,
            N=16,
            K=32,
            trans_a=True,
            trans_b=True,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32((1 << 30) | 256),
            T.uint32((1 << 30) | 260),
            False,
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    for col in T.serial(16):
        output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def sub_mxf8_k_major_cta1(
    a_physical: T.Buffer((16384,), "uint8"),
    b_physical: T.Buffer((16384,), "uint8"),
    output: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((16384,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((16384,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 512), "uint32", scope="tmem", layout=_TMEM_D_512, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0:
        for copy_i in T.serial(512):
            offset = lane + copy_i * 32
            shared_a[offset] = a_physical[offset]
            shared_b[offset] = b_physical[offset]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    # Each warp writes its own sub-partition's replica of the scale columns.
    row = T.meta_var(warp * 32 + lane)
    for row_group in T.unroll(4):
        tmem[row, 256 + row_group] = T.uint32(0x7F7F7F7F)
        tmem[row, 260 + row_group] = T.uint32(0x7F7F7F7F)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if (warp == 0) and (lane == 0):
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e4m3fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=128,
            N=128,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=1, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=1, sdo=64, swizzle=3
        )
        for kblock in T.unroll(4):
            T.cuda.runtime_instr_desc(T.address_of(desc_i), T.cast(kblock, "uint32"))
            T.ptx["tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32"](
                T.uint32(0),
                desc_a + T.cast(kblock * 2, "uint64"),
                desc_b + T.cast(kblock * 2, "uint64"),
                desc_i,
                T.uint32(256 + kblock * (1 << 30)),
                T.uint32(260 + kblock * (1 << 30)),
                kblock != 0,
            )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    for col in T.serial(128):
        output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def sub_mixed_fp4_fp8_cta_group2(
    a_physical: T.Buffer((2, 16384), "uint8"),
    b_physical: T.Buffer((2, 2048), "uint8"),
    scale_a_cells: T.Buffer((2, 4, 32), "uint32"),
    scale_b_cells: T.Buffer((2, 1, 32), "uint32"),
    output: T.Buffer((2, 128, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((16384,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((2048,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0:
        for copy_i in T.serial(512):
            offset = lane + copy_i * 32
            shared_a[offset] = a_physical[cta, offset]
        for copy_i in T.serial(64):
            offset = lane + copy_i * 32
            shared_b[offset] = b_physical[cta, offset]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    # Each warp writes its own sub-partition's replica of the scale columns.
    row = T.meta_var(warp * 32 + lane)
    for row_group in T.unroll(4):
        tmem[row, 32 + row_group] = scale_a_cells[cta, row_group, lane]
    tmem[row, 36] = scale_b_cells[cta, 0, lane]
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cluster_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if cta == 0 and warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float4_e2m1fn",
            b_dtype="float8_e4m3fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=256,
            N=32,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(32),
            T.uint32(36),
            False,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[32]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[32]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"](
            T.uint32(0),
            desc_a,
            desc_b,
            T.bitwise_or(desc_i, T.uint32((1 << 29) | (1 << 4))),
            T.uint32(32),
            T.uint32(36),
            True,
        )
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(barrier[0]), T.uint16(3)
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    for col in T.serial(32):
        output[cta, row, col] = T.reinterpret("float32", tmem[row, col])
    T.cuda.cluster_sync()


# -- cases: legacy inputs and expected arrays (copied from the legacy tests) ----


def _case_mxf4_mqa():
    row = np.arange(128, dtype=np.uint16)[:, None]
    k = np.arange(128, dtype=np.uint16)[None, :]
    k_atom = k // np.uint16(32)
    a_bits = ((row * 3 + k * 5 + k_atom + 1) % 16).astype(np.uint8)
    b_bits = ((row * 7 + k * 3 + k_atom * 3 + 2) % 16).astype(np.uint8)
    scale_pattern_a = np.array([126, 127, 128, 127], dtype=np.uint8)
    scale_pattern_b = np.array([128, 127, 126, 127], dtype=np.uint8)
    scale_indices = np.arange(4, dtype=np.int64)[None, :]
    row_indices = np.arange(128, dtype=np.int64)[:, None]
    scale_a_bits = scale_pattern_a[(scale_indices + row_indices) % 4]
    scale_b_bits = scale_pattern_b[(scale_indices + row_indices * 3) % 4]

    def inputs(**extra):
        return {
            "a_physical": _mxf4_kmajor_physical(a_bits),
            "b_physical": _mxf4_kmajor_physical(b_bits),
            "scale_a_cells": _pack_e8m0_cells(scale_a_bits),
            "scale_b_cells": _pack_e8m0_cells(scale_b_bits),
            "output": np.zeros((128, 128), dtype=np.float32),
            **extra,
        }

    values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    signs_a = np.where(a_bits & 8, np.float32(-1.0), np.float32(1.0))
    signs_b = np.where(b_bits & 8, np.float32(-1.0), np.float32(1.0))
    a = values[a_bits & 7] * signs_a
    b = values[b_bits & 7] * signs_b
    scale_a = np.exp2(scale_a_bits.astype(np.int16) - 127).astype(np.float32)
    scale_b = np.exp2(scale_b_bits.astype(np.int16) - 127).astype(np.float32)
    a *= np.repeat(scale_a, 32, axis=1)
    b *= np.repeat(scale_b, 32, axis=1)
    return inputs, a, b


def _case_mxf4nvf4():
    row = np.arange(128, dtype=np.uint16)[:, None]
    k = np.arange(128, dtype=np.uint16)[None, :]
    block = k // np.uint16(16)
    a_bits = ((row * 3 + k * 5 + block + 1) % 16).astype(np.uint8)
    b_bits = ((row * 7 + k * 3 + block * 3 + 2) % 16).astype(np.uint8)
    b_bits[8:, :] = np.uint8(0)
    finite_scales = np.array([0x30, 0x34, 0x38, 0x3C, 0x40, 0x44], dtype=np.uint8)
    scale_index = np.arange(8, dtype=np.int64)[None, :]
    row_index = np.arange(128, dtype=np.int64)[:, None]
    scale_a_bits = finite_scales[(scale_index + row_index) % len(finite_scales)]
    scale_b_bits = finite_scales[(scale_index * 2 + row_index[:8] * 3) % len(finite_scales)]
    return a_bits, b_bits, scale_a_bits, scale_b_bits, scale_index, row_index


def _nvf4_inputs(a_bits, b_bits, scale_a_bits, scale_b_bits, use_ue8m0_scales):
    return {
        "a_physical": _mxf4_kmajor_physical(a_bits),
        "b_physical": _mxf4_kmajor_physical(b_bits),
        "scale_a_cells": _pack_nvf4_scale_cells(scale_a_bits, 4),
        "scale_b_cells": _pack_nvf4_scale_cells(scale_b_bits, 1)[:, 0, :],
        "output": np.zeros((128, 8), dtype=np.float32),
        "use_ue8m0_scales": use_ue8m0_scales,
    }


def _case_e4m3_mqa():
    finite_codes = np.array([0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], dtype=np.uint8)
    row = np.arange(128, dtype=np.int64)[:, None]
    k = np.arange(128, dtype=np.int64)[None, :]
    a_bits = finite_codes[(row * 3 + k * 5 + 1) % len(finite_codes)]
    b_bits = finite_codes[(row * 7 + k * 2 + 3) % len(finite_codes)]
    inputs = {
        "a_physical": _kmajor_swizzle_physical(a_bits, swizzle_len=3, sdo=1024),
        "b_physical": _kmajor_swizzle_physical(b_bits, swizzle_len=3, sdo=1024),
        "output": np.zeros((128, 128), dtype=np.float32),
    }
    return inputs, {"output": _e4m3fn_bits_to_f32(a_bits) @ _e4m3fn_bits_to_f32(b_bits).T}


def _case_bf16_ss():
    row = np.arange(64, dtype=np.float32)[:, None]
    col = np.arange(8, dtype=np.float32)[:, None]
    inner = np.arange(16, dtype=np.float32)[None, :]
    a = ((row % np.float32(5)) - np.float32(2)) + inner * np.float32(0.5)
    b = ((col % np.float32(3)) - np.float32(1)) - inner * np.float32(0.25)
    a_bits = _bfloat16_bits(a)
    b_bits = _bfloat16_bits(b)
    a_physical = _kmajor_swizzle_physical_at_start(
        np.ascontiguousarray(a_bits).view(np.uint8).reshape(64, 32),
        start=128,
        swizzle_len=2,
        sdo=512,
        byte_len=4096,
    )
    b_physical = _kmajor_swizzle_physical_at_start(
        np.ascontiguousarray(b_bits).view(np.uint8).reshape(8, 32),
        start=4224,
        swizzle_len=2,
        sdo=512,
        byte_len=1024,
    )
    inputs = {
        "a_physical": np.pad(a_physical, (0, 4096 - a_physical.size)),
        "b_physical": np.pad(b_physical, (0, 1024 - b_physical.size)),
        "output": np.zeros((64, 8), dtype=np.float32),
        "output_ws": np.zeros((64, 8), dtype=np.float32),
    }
    a_bf16 = (a_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    b_bf16 = (b_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    expected = np.empty((64, 8), dtype=np.float32)
    for i in range(64):
        for j in range(8):
            accumulator = np.float32(0)
            for k in range(16):
                accumulator = np.float32(
                    np.float64(a_bf16[i, k]) * np.float64(b_bf16[j, k]) + np.float64(accumulator)
                )
            expected[i, j] = accumulator
    return inputs, {"output": expected, "output_ws": expected}


def _case_ws_layout_e_tail():
    row = np.arange(64, dtype=np.float32)[:, None]
    col = np.arange(128, dtype=np.float32)[:, None]
    inner = np.arange(16, dtype=np.float32)[None, :]
    a = ((row % np.float32(7)) - np.float32(3)) * np.float32(0.25) + inner * np.float32(0.125)
    b = ((col % np.float32(11)) - np.float32(5)) * np.float32(0.125) - inner * np.float32(0.0625)
    a_bits = _bfloat16_bits(a)
    b_bits = _bfloat16_bits(b)
    inputs = {
        "a_physical": _kmajor_swizzle_physical(
            np.ascontiguousarray(a_bits).view(np.uint8).reshape(64, 32),
            swizzle_len=2,
            sdo=512,
        ),
        "b_physical": _kmajor_swizzle_physical(
            np.ascontiguousarray(b_bits).view(np.uint8).reshape(128, 32),
            swizzle_len=2,
            sdo=512,
        ),
        "output": np.zeros((64, 128), dtype=np.float32),
    }
    a_bf16 = (a_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    b_bf16 = (b_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return inputs, {"output": _tcgen_fma_matmul(a_bf16, b_bf16)}


def _case_f16_ts_mn_major():
    row = np.arange(128, dtype=np.float32)[:, None]
    n = np.arange(128, dtype=np.float32)[:, None]
    k = np.arange(16, dtype=np.float32)[None, :]
    a = ((row % np.float32(7)) - np.float32(3)) * np.float32(0.125) + k * np.float32(0.0625)
    b = ((n % np.float32(11)) - np.float32(5)) * np.float32(0.25) - k * np.float32(0.03125)
    b[64:] += np.float32(3)
    a_f16 = np.asarray(a, dtype=np.float16)
    b_f16 = np.asarray(b, dtype=np.float16)
    a_bits = np.ascontiguousarray(a_f16).view(np.uint16)
    a_packed = a_bits[:, 0::2].astype(np.uint32) | (
        a_bits[:, 1::2].astype(np.uint32) << np.uint32(16)
    )
    inputs = {
        "a_packed": a_packed,
        "b_physical": _mnmajor_f16_swizzle_physical(np.ascontiguousarray(b_f16).view(np.uint16)),
        "output": np.zeros((128, 128), dtype=np.float32),
        "output_ws": np.zeros((128, 128), dtype=np.float32),
    }
    expected = _tcgen_fma_matmul(a_f16.astype(np.float32), b_f16.astype(np.float32))
    return inputs, {"output": expected, "output_ws": expected}


def _case_f16_cta2_datapaths():
    inner = np.arange(16, dtype=np.float32)[None, :]

    row256 = np.arange(256, dtype=np.float32)[:, None]
    col16 = np.arange(16, dtype=np.float32)[:, None]
    a_ss256 = np.asarray(
        ((row256 % 13) - 6) * np.float32(0.0625) + inner * np.float32(0.03125),
        dtype=np.float16,
    )
    b_ss256 = np.asarray(
        ((col16 % 7) - 3) * np.float32(0.125) - inner * np.float32(0.015625),
        dtype=np.float16,
    )
    a_ts256 = np.asarray(
        ((row256 % 11) - 5) * np.float32(0.03125) - inner * np.float32(0.0625),
        dtype=np.float16,
    )
    b_ts256 = np.asarray(
        ((col16 % 5) - 2) * np.float32(0.25) + inner * np.float32(0.015625),
        dtype=np.float16,
    )

    row128 = np.arange(128, dtype=np.float32)[:, None]
    col128 = np.arange(128, dtype=np.float32)[:, None]
    a_ss128 = np.asarray(
        ((row128 % 9) - 4) * np.float32(0.0625) + inner * np.float32(0.03125),
        dtype=np.float16,
    )
    b_ss128 = np.asarray(
        ((col128 % 15) - 7) * np.float32(0.03125) - inner * np.float32(0.015625),
        dtype=np.float16,
    )

    def kmajor_pair(values: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                _kmajor_swizzle_physical(
                    np.ascontiguousarray(part).view(np.uint8).reshape(part.shape[0], 32),
                    swizzle_len=3,
                    sdo=1024,
                )
                for part in np.split(values, 2)
            ]
        )

    def mnmajor_pair(values: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                _mnmajor_f16_swizzle_physical_cta2(np.ascontiguousarray(part).view(np.uint16))
                for part in np.split(values, 2)
            ]
        )

    a_ts_bits = np.ascontiguousarray(a_ts256).view(np.uint16)
    a_ts_packed = a_ts_bits[:, 0::2].astype(np.uint32) | (
        a_ts_bits[:, 1::2].astype(np.uint32) << np.uint32(16)
    )
    inputs = {
        "a_ss256_physical": kmajor_pair(a_ss256),
        "b_ss256_physical": kmajor_pair(b_ss256),
        "a_ts256_packed": np.stack(np.split(a_ts_packed, 2)),
        "b_ts256_physical": np.stack(
            [
                _mnmajor_f16_swizzle_physical_cta2(np.ascontiguousarray(part).view(np.uint16))
                for part in np.split(b_ts256, 2)
            ]
        ),
        "a_ss128_physical": mnmajor_pair(a_ss128),
        "b_ss128_physical": mnmajor_pair(b_ss128),
        "output_ss256": np.zeros((256, 16), dtype=np.float32),
        "output_ts256": np.zeros((256, 16), dtype=np.float32),
        "output_ss128": np.zeros((128, 128), dtype=np.float32),
    }
    expected = {
        "output_ss256": _tcgen_fma_matmul(a_ss256.astype(np.float32), b_ss256.astype(np.float32)),
        "output_ts256": _tcgen_fma_matmul(a_ts256.astype(np.float32), b_ts256.astype(np.float32)),
        "output_ss128": _tcgen_fma_matmul(a_ss128.astype(np.float32), b_ss128.astype(np.float32)),
    }
    return inputs, expected


def _case_rubin_k64():
    e5m2_codes = np.array(
        [0x00, 0x01, 0x02, 0x03, 0x04, 0x38, 0x3C, 0x3D, 0x40, 0xBC, 0xC0],
        dtype=np.uint8,
    )
    cta = np.arange(2, dtype=np.int64)[:, None, None]
    a_row = np.arange(128, dtype=np.int64)[None, :, None]
    b_row = np.arange(64, dtype=np.int64)[None, :, None]
    k = np.arange(64, dtype=np.int64)[None, None, :]
    a_bits = e5m2_codes[(cta * 5 + a_row * 3 + k * 7 + 1) % len(e5m2_codes)]
    b_bits = e5m2_codes[(cta * 2 + b_row * 5 + k * 3 + 2) % len(e5m2_codes)]

    seed_cta = np.arange(2, dtype=np.float32)[:, None, None]
    seed_row = np.arange(128, dtype=np.float32)[None, :, None]
    seed_col = np.arange(128, dtype=np.float32)[None, None, :]
    seed = (
        seed_cta * np.float32(2)
        + (seed_row % np.float32(7)) * np.float32(0.125)
        - (seed_col % np.float32(5)) * np.float32(0.0625)
    ).astype(np.float32)
    inputs = {
        "a_physical": np.stack(
            [_kmajor_swizzle_physical(part, swizzle_len=3, sdo=1024) for part in a_bits]
        ),
        "b_physical": np.stack([_mnmajor_f8_swizzle_physical(part) for part in b_bits]),
        "seed": seed,
        "output": np.zeros_like(seed),
    }
    joint_a = _e5m2_bits_to_f32(a_bits.reshape(256, 64))
    joint_b = _e5m2_bits_to_f32(b_bits.reshape(128, 64))
    expected = seed.reshape(256, 128).copy()
    for inner in range(64):
        expected = (
            np.float64(joint_a[:, inner, None]) * np.float64(joint_b[None, :, inner])
            + np.float64(expected)
        ).astype(np.float32)
    # Word 0 bit 5 masks CTA 0 lane/row 5; word 6 bit 7 masks CTA 1 row 71.
    expected[5, :] = seed[0, 5, :]
    expected[128 + 71, :] = seed[1, 71, :]
    return inputs, {"output": expected.reshape(2, 128, 128)}


def _case_tf32_ss():
    row = np.arange(64, dtype=np.float32)[:, None]
    col = np.arange(8, dtype=np.float32)[:, None]
    k = np.arange(8, dtype=np.float32)[None, :]
    a = (row % np.float32(5) - np.float32(2)) * np.float32(0.5) + k * np.float32(0.25)
    b = (col % np.float32(7) - np.float32(3)) * np.float32(0.25) - k * np.float32(0.5)
    a += ((row + k) % np.float32(3) - np.float32(1)) * np.float32(2**-12)
    b += ((col + np.float32(2) * k) % np.float32(3) - np.float32(1)) * np.float32(2**-12)
    a_bytes = np.ascontiguousarray(a, dtype=np.float32).view(np.uint8).reshape(64, 32)
    b_bytes = np.ascontiguousarray(b, dtype=np.float32).view(np.uint8).reshape(8, 32)
    inputs = {
        "a_physical": _kmajor_swizzle_physical(a_bytes, swizzle_len=3, sdo=1024),
        "b_physical": _kmajor_swizzle_physical(b_bytes, swizzle_len=3, sdo=1024),
        "output": np.zeros((64, 8), dtype=np.float32),
    }
    expected = _tcgen_fma_matmul(_decode_tcgen_tf32_payload(a), _decode_tcgen_tf32_payload(b))
    assert not np.array_equal(expected, a @ b.T)
    return inputs, {"output": expected}


def _case_tf32_low32():
    row = np.arange(64, dtype=np.float32)[:, None]
    k = np.arange(8, dtype=np.float32)[None, :]
    col = np.arange(32, dtype=np.float32)[:, None]
    a = ((row % np.float32(5)) - np.float32(2)) * np.float32(0.5) + k * np.float32(0.25)
    b = ((col % np.float32(7)) - np.float32(3)) * np.float32(0.25) - k * np.float32(0.5)
    a += ((row + k) % np.float32(3) - np.float32(1)) * np.float32(2**-12)
    b += ((col + np.float32(2) * k) % np.float32(3) - np.float32(1)) * np.float32(2**-12)
    b_bytes = np.ascontiguousarray(b.astype(np.float32)).view(np.uint8).reshape(32, 32)
    inputs = {
        "a": np.ascontiguousarray(a, dtype=np.float32),
        "b_physical": _kmajor_swizzle_physical(b_bytes, swizzle_len=3, sdo=1024),
        "output": np.zeros((64, 32), dtype=np.float32),
    }
    expected = _tcgen_fma_matmul(
        _decode_tcgen_tf32_payload(a),
        _decode_tcgen_tf32_payload(b),
        initial=np.float32(4),
    )
    expected[3, :] = np.float32(4)
    full_f32 = a @ b.T + np.float32(4)
    full_f32[3, :] = np.float32(4)
    assert not np.array_equal(expected, full_f32)
    return inputs, {"output": expected}


def _case_e4m3_m64_n8():
    finite_codes = np.array([0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], dtype=np.uint8)
    row_a = np.arange(64, dtype=np.int64)[:, None]
    row_b = np.arange(8, dtype=np.int64)[:, None]
    k = np.arange(32, dtype=np.int64)[None, :]
    a_bits = finite_codes[(row_a * 3 + k * 5 + 1) % len(finite_codes)]
    b_bits = finite_codes[(row_b * 7 + k * 2 + 3) % len(finite_codes)]
    inputs = {
        "a_physical": _kmajor_swizzle_physical(a_bits, swizzle_len=3, sdo=1024),
        "b_physical": _kmajor_swizzle_physical(b_bits, swizzle_len=3, sdo=1024),
        "output": np.zeros((64, 8), dtype=np.float32),
    }
    return inputs, {"output": _e4m3fn_bits_to_f32(a_bits) @ _e4m3fn_bits_to_f32(b_bits).T}


def _case_e5m2_e4m3():
    e5m2_codes = np.array(
        [0x00, 0x01, 0x02, 0x03, 0x04, 0x38, 0x3C, 0x3D, 0x40, 0xBC, 0xC0], dtype=np.uint8
    )
    e4m3_codes = np.array([0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], dtype=np.uint8)
    row_a = np.arange(64, dtype=np.int64)[:, None]
    row_b = np.arange(8, dtype=np.int64)[:, None]
    k = np.arange(32, dtype=np.int64)[None, :]
    a_bits = e5m2_codes[(row_a * 3 + k * 5 + 1) % len(e5m2_codes)]
    b_bits = e4m3_codes[(row_b * 7 + k * 2 + 3) % len(e4m3_codes)]
    inputs = {
        "a_physical": _kmajor_swizzle_physical(a_bits, swizzle_len=3, sdo=1024),
        "b_physical": _kmajor_swizzle_physical(b_bits, swizzle_len=3, sdo=1024),
        "output": np.zeros((64, 8), dtype=np.float32),
    }
    expected = _e5m2_bits_to_f32(a_bits) @ _e4m3fn_bits_to_f32(b_bits).T
    wrong_decoder = _e4m3fn_bits_to_f32(a_bits) @ _e4m3fn_bits_to_f32(b_bits).T
    assert not np.array_equal(wrong_decoder, expected)
    return inputs, {"output": expected}


def _case_f16_destination():
    a_bits = np.full((64, 32), 0x38, dtype=np.uint8)  # E4M3 1.0
    b_bits = np.full((8, 32), 0x28, dtype=np.uint8)  # E4M3 0.25
    b_bits[:, 0] = 0x00  # leave 31 contributing products, so the sum is not exact
    seed_low = int(np.array(1024.0, dtype=np.float16).view(np.uint16))
    seed = np.full((128, 8), (0xBEEF << 16) | seed_low, dtype=np.uint32)
    inputs = {
        "a_physical": _kmajor_swizzle_physical(a_bits, swizzle_len=3, sdo=1024),
        "b_physical": _kmajor_swizzle_physical(b_bits, swizzle_len=3, sdo=1024),
        "seed": seed,
        "output": np.zeros((64, 8), dtype=np.uint32),
    }
    accumulated = _e4m3fn_bits_to_f32(a_bits) @ _e4m3fn_bits_to_f32(b_bits).T + np.float32(1024.0)
    assert np.unique(accumulated).tolist() == [1031.75]
    assert np.float32(np.float16(np.float32(1031.75))) != np.float32(1031.75)
    expected = accumulated.astype(np.float16).view(np.uint16).astype(np.uint32)
    assert np.unique(expected).tolist() == [0x6408]  # binary16 1032.0, zero upper half
    return inputs, {"output": expected}


def _case_tf32_m128_n16():
    row = np.arange(128, dtype=np.float32)[:, None]
    col = np.arange(16, dtype=np.float32)[:, None]
    k = np.arange(8, dtype=np.float32)[None, :]
    a = (row % np.float32(7) - np.float32(3)) * np.float32(0.25) + k * np.float32(0.125)
    b = (col % np.float32(5) - np.float32(2)) * np.float32(0.5) - k * np.float32(0.25)
    a += ((row + k) % np.float32(3) - np.float32(1)) * np.float32(2**-12)
    b += ((col + np.float32(2) * k) % np.float32(3) - np.float32(1)) * np.float32(2**-12)
    b_bytes = np.ascontiguousarray(b, dtype=np.float32).view(np.uint8).reshape(16, 32)
    inputs = {
        "a": np.ascontiguousarray(a, dtype=np.float32),
        "b_physical": _kmajor_swizzle_physical(b_bytes, swizzle_len=3, sdo=1024),
        "output": np.zeros((128, 16), dtype=np.float32),
    }
    expected = _tcgen_fma_matmul(_decode_tcgen_tf32_payload(a), _decode_tcgen_tf32_payload(b))
    return inputs, {"output": expected}


def _case_mxf8_mn_major():
    finite = np.array([0x00, 0x30, 0x38, 0x3C, 0x40, 0xB0, 0xB8, 0xBC], dtype=np.uint8)
    row_a = np.arange(128, dtype=np.int64)[:, None]
    row_b = np.arange(16, dtype=np.int64)[:, None]
    k = np.arange(32, dtype=np.int64)[None, :]
    a_bits = finite[(row_a * 3 + k * 5 + 1) % len(finite)]
    b_bits = finite[(row_b * 7 + k * 3 + 2) % len(finite)]
    inputs = {
        "a_physical": _mnmajor_f8_swizzle_physical(a_bits),
        "b_physical": _mnmajor_f8_swizzle_physical(b_bits),
        "output": np.zeros((128, 16), dtype=np.float32),
    }
    expected = _tcgen_fma_matmul(_e4m3fn_bits_to_f32(a_bits), _e4m3fn_bits_to_f32(b_bits))
    return inputs, {"output": expected}


def _case_mxf8_k_major():
    finite = np.array([0x00, 0x30, 0x38, 0x3C, 0x40, 0xB0, 0xB8, 0xBC], dtype=np.uint8)
    row_a = np.arange(128, dtype=np.int64)[:, None]
    row_b = np.arange(128, dtype=np.int64)[:, None]
    k = np.arange(128, dtype=np.int64)[None, :]
    a_bits = finite[(row_a * 3 + k * 5 + 1) % len(finite)]
    b_bits = finite[(row_b * 7 + k * 3 + 2) % len(finite)]
    inputs = {
        "a_physical": _kmajor_swizzle_physical(a_bits, swizzle_len=3, sdo=1024),
        "b_physical": _kmajor_swizzle_physical(b_bits, swizzle_len=3, sdo=1024),
        "output": np.zeros((128, 128), dtype=np.float32),
    }
    expected = _tcgen_fma_matmul(_e4m3fn_bits_to_f32(a_bits), _e4m3fn_bits_to_f32(b_bits))
    return inputs, {"output": expected}


def _case_mixed_cta2():
    finite_fp8 = np.array([0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], dtype=np.uint8)
    cta = np.arange(2, dtype=np.int64)[:, None, None]
    row_a = np.arange(128, dtype=np.int64)[None, :, None]
    row_b = np.arange(16, dtype=np.int64)[None, :, None]
    k = np.arange(64, dtype=np.int64)[None, None, :]
    a_codes = ((cta * 5 + row_a * 3 + k * 7 + 1) % 16).astype(np.uint8)
    a_padded = np.full(a_codes.shape, np.uint8(0xA5), dtype=np.uint8)
    for k_block in range(2):
        for atom in range(2):
            code_start = k_block * 32 + atom * 16
            storage_start = k_block * 32 + atom * 16
            codes = a_codes[:, :, code_start : code_start + 16]
            a_padded[:, :, storage_start : storage_start + 8] = codes[:, :, 0::2] | (
                codes[:, :, 1::2] << np.uint8(4)
            )
    b_bits = finite_fp8[(cta * 2 + row_b * 5 + k * 3 + 2) % len(finite_fp8)]

    scale_a_bits = np.full((2, 128, 4), np.uint8(127), dtype=np.uint8)
    scale_b_bits = np.full((32, 4), np.uint8(127), dtype=np.uint8)
    scale_a_bits[0, :, 0] = np.uint8(126)
    scale_a_bits[1, :, 0] = np.uint8(128)
    scale_a_bits[0, :, 1] = np.uint8(128)
    scale_a_bits[1, :, 1] = np.uint8(126)
    scale_b_bits[:16, 0] = np.uint8(128)
    scale_b_bits[16:, 0] = np.uint8(126)
    scale_b_bits[:16, 1] = np.uint8(126)
    scale_b_bits[16:, 1] = np.uint8(128)
    scale_a_cells = np.stack([_pack_e8m0_cells(scale_a_bits[index]) for index in range(2)])
    scale_b_cells = np.broadcast_to(_pack_e8m0_cells(scale_b_bits), (2, 1, 32)).copy()
    inputs = {
        "a_physical": np.stack(
            [_kmajor_swizzle_physical(a_padded[index], swizzle_len=3, sdo=1024) for index in range(2)]
        ),
        "b_physical": np.stack(
            [_kmajor_swizzle_physical(b_bits[index], swizzle_len=3, sdo=1024) for index in range(2)]
        ),
        "scale_a_cells": scale_a_cells,
        "scale_b_cells": scale_b_cells,
        "output": np.zeros((2, 128, 32), dtype=np.float32),
    }
    values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    a = values[a_codes & 7] * np.where(a_codes & 8, np.float32(-1), np.float32(1))
    b = _e4m3fn_bits_to_f32(b_bits)
    a *= np.repeat(
        np.exp2(scale_a_bits[:, :, (0, 1)].astype(np.int16) - 127).astype(np.float32),
        32,
        axis=2,
    )
    b *= np.repeat(
        np.exp2(scale_b_bits.reshape(2, 16, 4)[:, :, (0, 1)].astype(np.int16) - 127).astype(
            np.float32
        ),
        32,
        axis=2,
    )
    joint_a = a.reshape(256, 64)
    joint_b = b.reshape(32, 64)
    return inputs, {"output": (joint_a @ joint_b.T).reshape(2, 128, 32)}


# -- harness ---------------------------------------------------------------------


def _assert_sub_partition_stop(kernel, inputs, *, warp, lane, anchor):
    """The legacy kernel stops with ``bad_address`` because warp ``warp`` touches
    a TMEM lane outside its 32-lane sub-partition (numsim-behaviour-deltas T4);
    the stop names the first faulting lane and is anchored at the TMEM access."""

    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(v2.transpile(kernel), inputs)
    stops = [d for d in caught.value.diagnostics if d.get("status") == "error"]
    assert stops, caught.value.diagnostics
    stop = stops[0]
    assert stop["kind"] == "bad_address", stop
    assert "sub-partition" in stop["message"], stop
    assert (stop["warp"], stop["lanes"]) == (warp, f"WarpMask(0x{1 << lane:08x})"), stop
    span = stop["source_span"]
    with open(span["source_name"]) as handle:
        line = handle.read().splitlines()[span["line"] - 1]
    assert anchor in line, (anchor, line)


def _assert_outputs(kernel, inputs, expected):
    result = v2.Engine().run(v2.transpile(kernel), inputs)
    assert result.status.get("kind") == "completed", result.status
    for name, value in expected.items():
        np.testing.assert_array_equal(result.outputs[name], value, err_msg=name)
    return result


# -- tests -------------------------------------------------------------------------


def test_raw_tcgen_mxf4_mqa_mma_uses_physical_smem_and_tmem_scales_original_kernel_violates_tmem_sub_partition():
    """Legacy kernel: warp 0 reads TMEM rows 0..127 (delta T4)."""
    inputs, _, _ = _case_mxf4_mqa()
    _assert_sub_partition_stop(
        raw_tcgen_mma_block_scaled_mxf4_mqa, inputs(issue_second=1), warp=0, lane=0,
        anchor="tmem[row, col]",
    )


def test_raw_tcgen_mxf4_mqa_mma_uses_physical_smem_and_tmem_scales():
    """Corrected kernels (see the module docstring); all three legacy runs kept."""
    inputs, a, b = _case_mxf4_mqa()
    _assert_outputs(sub_mxf4_mqa, inputs(issue_second=1), {"output": a @ b.T})
    _assert_outputs(sub_mxf4_mqa, inputs(issue_second=0), {"output": a[:, :64] @ b[:, :64].T})
    _assert_outputs(sub_mxf4_expression_input_d, inputs(mode=0), {"output": a[:, :64] @ b[:, :64].T})


def test_raw_tcgen_mxf4nvf4_scale_vec_4x_matches_the_typed_block_scaled_gemm_original_kernel_violates_tmem_sub_partition():
    """Legacy raw kernel: warp 0 reads TMEM rows 0..127 (delta T4)."""
    a_bits, b_bits, scale_a_bits, scale_b_bits, _, _ = _case_mxf4nvf4()
    _assert_sub_partition_stop(
        raw_tcgen_mma_block_scaled_mxf4nvf4_two_k_tiles,
        _nvf4_inputs(a_bits, b_bits, scale_a_bits, scale_b_bits, 0),
        warp=0,
        lane=0,
        anchor="tmem[row, col]",
    )


def test_raw_tcgen_mxf4nvf4_scale_vec_4x_matches_the_typed_block_scaled_gemm():
    """Corrected raw kernel against the independent reference, the UE8M0 run and
    the padding-MSB positive control, as in the legacy test.

    The legacy test also compared the raw result with the typed ``Tx.gemm_async``
    kernel ``typed_nvfp4_gemm_two_k_tiles``. v2 rejects that oracle at transpile
    (direct stores to its replicated scale-factor TMEM views,
    numsim-behaviour-deltas L1), which this copy asserts; the raw result is
    still compared element-exactly with the same independent reference the
    legacy test used for both kernels.
    """

    a_bits, b_bits, scale_a_bits, scale_b_bits, scale_index, row_index = _case_mxf4nvf4()
    a = _decode_e2m1_nibbles(a_bits) * np.repeat(_e4m3fn_bits_to_f32(scale_a_bits), 16, axis=1)
    b = _decode_e2m1_nibbles(b_bits[:8]) * np.repeat(_e4m3fn_bits_to_f32(scale_b_bits), 16, axis=1)
    _assert_outputs(
        sub_mxf4nvf4_two_k_tiles,
        _nvf4_inputs(a_bits, b_bits, scale_a_bits, scale_b_bits, 0),
        {"output": _tcgen_fma_matmul(a, b)},
    )

    with pytest.raises(UnsupportedTIRxError, match="tmem_replicated_view"):
        v2.transpile(typed_nvfp4_gemm_two_k_tiles)

    ue8m0_codes = np.array([125, 126, 127, 128, 129], dtype=np.uint8)
    ue8m0_scale_a_bits = ue8m0_codes[(scale_index + row_index) % len(ue8m0_codes)]
    ue8m0_scale_b_bits = ue8m0_codes[(scale_index * 2 + row_index[:8] * 3) % len(ue8m0_codes)]
    ue8m0_scale_a = np.exp2(ue8m0_scale_a_bits.astype(np.int16) - 127).astype(np.float32)
    ue8m0_scale_b = np.exp2(ue8m0_scale_b_bits.astype(np.int16) - 127).astype(np.float32)
    ue8m0_a = _decode_e2m1_nibbles(a_bits) * np.repeat(ue8m0_scale_a, 16, axis=1)
    ue8m0_b = _decode_e2m1_nibbles(b_bits[:8]) * np.repeat(ue8m0_scale_b, 16, axis=1)
    _assert_outputs(
        sub_mxf4nvf4_two_k_tiles,
        _nvf4_inputs(a_bits, b_bits, ue8m0_scale_a_bits, ue8m0_scale_b_bits, 1),
        {"output": _tcgen_fma_matmul(ue8m0_a, ue8m0_b)},
    )

    # Positive control (PTX ISA 5.2.3): the ue4m3 MSB is padding and must stay clear.
    poisoned = _nvf4_inputs(a_bits, b_bits, scale_a_bits, scale_b_bits, 0)
    poisoned["scale_a_cells"][0, 0, 0] |= np.uint32(0x80)
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(v2.transpile(sub_mxf4nvf4_two_k_tiles), poisoned)
    stop = [d for d in caught.value.diagnostics if d.get("status") == "error"][0]
    assert stop["kind"] == "invalid_operand", stop
    assert "padding MSB" in stop["message"], stop


_SIMPLE = [
    # legacy test, legacy kernel, corrected kernel, case, (warp, lane, anchor)
    (
        "test_raw_tcgen_e4m3_mqa_mma_accumulates_four_k_tiles",
        "raw_tcgen_mma_e4m3_mqa", "sub_e4m3_mqa", "_case_e4m3_mqa",
        (0, 0, "tmem[row, col]"),
    ),
    (
        "test_raw_tcgen_bf16_shared_shared_mma_matches_independent_matrix_product",
        "raw_tcgen_mma_bf16_ss", "sub_bf16_ss", "_case_bf16_ss",
        (0, 16, "_tmem[physical_lane, col]"),
    ),
    (
        "test_raw_tcgen_m64_weight_stationary_uses_layout_e_at_tmem_tail",
        "raw_tcgen_mma_ws_bf16_ss_layout_e_tail", "sub_ws_bf16_ss_layout_e_tail",
        "_case_ws_layout_e_tail",
        (0, 0, "tmem[physical_lane, physical_col]"),
    ),
    (
        "test_raw_tcgen_f16_tensor_shared_mma_uses_mn_major_ldo_for_second_n_tile",
        "raw_tcgen_mma_f16_ts_mn_major_n128", "sub_f16_ts_mn_major_n128", "_case_f16_ts_mn_major",
        (0, 0, "tmem[row, packed_k]"),
    ),
    (
        "test_raw_tcgen_f8f6f4_cta2_rubin_k64_gathers_masks_and_accumulates_both_ctas",
        "raw_tcgen_mma_f8f6f4_cta2_rubin_k64_discard", "sub_f8f6f4_cta2_rubin_k64_discard",
        "_case_rubin_k64",
        (0, 0, "tmem[row, col] = T.reinterpret"),
    ),
    (
        "test_raw_tcgen_tf32_shared_shared_matches_independent_matrix_product",
        "raw_tcgen_mma_tf32_ss", "sub_tf32_ss", "_case_tf32_ss",
        (0, 16, "tmem[physical_lane, col]"),
    ),
    (
        "test_raw_tcgen_tf32_low32_replacement_honors_predicate_and_accumulator",
        "raw_tcgen_mma_tf32_ts_predicated", "sub_tf32_ts_predicated", "_case_tf32_low32",
        (0, 16, "tmem[physical_lane, k]"),
    ),
    (
        "test_raw_tcgen_e4m3_m64_n8_uses_layout_f",
        "raw_tcgen_mma_e4m3_m64_n8", "sub_e4m3_m64_n8", "_case_e4m3_m64_n8",
        (0, 16, "tmem[physical_lane, col]"),
    ),
    (
        "test_raw_tcgen_f8f6f4_decodes_a_and_b_with_their_own_dtypes",
        "raw_tcgen_mma_e5m2_e4m3_m64_n8", "sub_e5m2_e4m3_m64_n8", "_case_e5m2_e4m3",
        (0, 16, "tmem[physical_lane, col]"),
    ),
    (
        "test_raw_tcgen_f8f6f4_float16_destination_rounds_once_on_store",
        "raw_tcgen_mma_f8f6f4_f16_destination_m64_n8", "sub_f8f6f4_f16_destination_m64_n8",
        "_case_f16_destination",
        (0, 0, "tmem[row, col] = seed[row, col]"),
    ),
    (
        "test_raw_tcgen_tf32_m128_n16_uses_regular_layout",
        "raw_tcgen_mma_tf32_m128_n16", "sub_tf32_m128_n16", "_case_tf32_m128_n16",
        (0, 0, "tmem[row, k]"),
    ),
    (
        "test_raw_tcgen_mxf8_mn_major_matches_independent_matrix_product",
        "raw_tcgen_mma_mxf8_mn_major", "sub_mxf8_mn_major", "_case_mxf8_mn_major",
        (0, 0, "tmem[replica * 32 + lane, 256 + row_group]"),
    ),
    (
        "test_raw_tcgen_mxf8_k_major_cta1_matches_independent_matrix_product",
        "raw_tcgen_mma_mxf8_k_major_cta1", "sub_mxf8_k_major_cta1", "_case_mxf8_k_major",
        (0, 0, "tmem[replica * 32 + lane, 256 + row_group]"),
    ),
    (
        "test_raw_tcgen_mixed_block_scale_cta_group2_gathers_and_scatters_both_ctas",
        "raw_tcgen_mma_mixed_fp4_fp8_cta_group2", "sub_mixed_fp4_fp8_cta_group2", "_case_mixed_cta2",
        (0, 0, "tmem[replica * 32 + lane, 32 + row_group]"),
    ),
]

_IDS = [legacy.removeprefix("test_") for legacy, *_ in _SIMPLE]


@pytest.mark.parametrize(("legacy", "kernel", "corrected", "case", "stop"), _SIMPLE, ids=_IDS)
def test_original_kernel_violates_tmem_sub_partition(legacy, kernel, corrected, case, stop):
    """The legacy kernel, verbatim: a warp touches TMEM lanes outside its 32-lane
    sub-partition, so the run stops with ``bad_address`` (delta T4)."""
    warp, lane, anchor = stop
    inputs, _ = globals()[case]()
    _assert_sub_partition_stop(globals()[kernel], inputs, warp=warp, lane=lane, anchor=anchor)


@pytest.mark.parametrize(("legacy", "kernel", "corrected", "case", "stop"), _SIMPLE, ids=_IDS)
def test_corrected_kernel_matches_the_legacy_expectation(legacy, kernel, corrected, case, stop):
    """The sub-partition-correct kernel (module docstring) gives the legacy
    test's exact expected outputs."""
    inputs, expected = globals()[case]()
    _assert_outputs(globals()[corrected], inputs, expected)


def test_raw_tcgen_f16_cta2_datapaths_match_independent_matrix_products_original_kernel_violates_tmem_sub_partition():
    """Legacy kernel: warp 0 stores the TS A operand into TMEM rows 0..127 (delta T4)."""
    inputs, _ = _case_f16_cta2_datapaths()
    _assert_sub_partition_stop(
        raw_tcgen_mma_f16_cta2_datapaths, inputs, warp=0, lane=0, anchor="tmem[row, packed_k]"
    )


def test_raw_tcgen_f16_cta2_datapaths_corrected_kernel_hits_the_tcgen05_shape_rule():
    """Expected-error only: this kernel cannot be corrected without changing what
    it tests. Its ``cta_group::2`` M=256 MMAs use N=16, which the PTX shape table
    rejects (``cta_group::2`` needs N a multiple of 32; numsim-behaviour-deltas
    L4). With the TMEM accesses made sub-partition-correct, v2 stops at the
    instruction descriptor with ``invalid_operand``; legacy produced numbers for
    an instruction the hardware does not define."""
    inputs, _ = _case_f16_cta2_datapaths()
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(v2.transpile(sub_f16_cta2_datapaths), inputs)
    stop = [d for d in caught.value.diagnostics if d.get("status") == "error"][0]
    assert stop["kind"] == "invalid_operand", stop
    assert "M=256, N=16" in stop["message"], stop
