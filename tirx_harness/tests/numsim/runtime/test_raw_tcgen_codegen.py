from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from tvm.backend.cuda.tile_primitive.gemm_async.tcgen05 import sf_tmem_layout
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TCol, TileLayout, TLane, tmem_datapath_layout

from tirx_harness import numsim
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tests.numsim.support.tcgen_descriptor import (
    INSTR_DESC,
    INSTR_DESC_BLOCK,
    MATRIX_DESC,
    encode_block_scaled_instr_descriptor_fields,
    encode_dense_instr_descriptor_fields,
    validate_tcgen05_instruction_shape,
)

TCGEN_DESCRIPTOR_LAYOUT = "<artifact-tcgen-descriptor-layout>"

_TMEM_D_4 = TileLayout(S[(128, 4) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_8 = TileLayout(S[(128, 8) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_16 = TileLayout(S[(128, 16) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_32 = TileLayout(S[(128, 32) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_104 = TileLayout(S[(128, 104) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_136 = TileLayout(S[(128, 136) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_512 = TileLayout(S[(128, 512) : (1 @ TLane, 1 @ TCol)])
_BF16_SMEM_64B = mma_shared_layout("bfloat16", SwizzleMode.SWIZZLE_64B_ATOM, (64, 64))
_TMEM_NVF4_24 = TileLayout(S[(128, 24) : (1 @ TLane, 1 @ TCol)])
_PACKED_FP4_SMEM_128X64 = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (128, 64))
_PACKED_FP4_SMEM_8X64 = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (8, 64))


@T.prim_func
def raw_tcgen_descriptor_encode(output: T.Buffer((3,), "uint64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "uint32", scope="shared")
    matrix_desc: T.uint64
    dense_desc: T.uint32
    block_desc: T.uint32
    if lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(matrix_desc), T.address_of(shared[4]), 1, 8, 3
        )
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(dense_desc),
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
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(block_desc),
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
        output[0] = matrix_desc
        output[1] = T.cast(dense_desc, "uint64")
        output[2] = T.cast(block_desc, "uint64")


@T.prim_func
def raw_tcgen_descriptor_encode_to_global(
    matrix_output: T.Buffer((1,), "uint64"), instr_output: T.Buffer((1,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "uint32", scope="shared")
    if lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(matrix_output[0]), T.address_of(shared[4]), 1, 8, 3
        )
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(instr_output[0]),
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


@T.prim_func
def raw_tcgen_ldst_32x32b(
    source: T.Buffer((128, 4), "uint32"), output: T.Buffer((128, 4), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    registers = T.alloc_local((4,), "uint32")
    runtime_address = T.alloc_local((1,), "uint32")
    runtime_address[0] = T.uint32(0)
    for col in T.unroll(4):
        tmem[row, col] = source[row, col]
    T.cuda.cta_sync()
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x4.b32"](
        registers[0], registers[1], registers[2], registers[3], runtime_address[0]
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    T.ptx["tcgen05.st.sync.aligned.32x32b.x4.b32"](
        T.cuda.get_tmem_addr(runtime_address[0], 0, 4),
        registers[0],
        registers[1],
        registers[2],
        registers[3],
    )
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.cuda.cta_sync()
    for col in T.unroll(4):
        output[row, col] = tmem[row, 4 + col]


@T.prim_func
def raw_tcgen_ld_16x256b_mapping(
    source: T.Buffer((128, 8), "uint32"), output: T.Buffer((4, 32, 4), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    physical_row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    registers = T.alloc_local((4,), "uint32")
    for col in T.unroll(8):
        tmem[physical_row, col] = source[physical_row, col]
    T.cuda.cta_sync()
    T.ptx["tcgen05.ld.sync.aligned.16x256b.x1.b32"](
        registers[0], registers[1], registers[2], registers[3], T.uint32(0)
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for register in T.unroll(4):
        output[warp, lane, register] = registers[register]


@T.prim_func
def raw_tcgen_ld_missing_shape_mappings(
    source: T.Buffer((128, 16), "uint32"), output: T.Buffer((3, 4, 32, 2), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    physical_row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    registers = T.alloc_local((2,), "uint32")
    for col in T.unroll(16):
        tmem[physical_row, col] = source[physical_row, col]
    T.cuda.cta_sync()
    T.ptx["tcgen05.ld.sync.aligned.16x32bx2.x2.b32"](
        registers[0], registers[1], T.uint32(0), 2 * (2) if False else 2
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[0, warp, lane, 0] = registers[0]
    output[0, warp, lane, 1] = registers[1]
    T.ptx["tcgen05.ld.sync.aligned.16x64b.x2.b32"](registers[0], registers[1], T.uint32(0))
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[1, warp, lane, 0] = registers[0]
    output[1, warp, lane, 1] = registers[1]
    T.ptx["tcgen05.ld.sync.aligned.16x128b.x1.b32"](registers[0], registers[1], T.uint32(0))
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[2, warp, lane, 0] = registers[0]
    output[2, warp, lane, 1] = registers[1]


@T.prim_func
def raw_tcgen_ld_pack_32x32b(
    source: T.Buffer((128, 4), "uint32"), output: T.Buffer((4, 32, 2), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    physical_row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    registers = T.alloc_local((2,), "uint32")
    for col in T.unroll(4):
        tmem[physical_row, col] = source[physical_row, col]
    T.cuda.cta_sync()
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x2.pack::16b.b32"](
        registers[0], registers[1], T.uint32(0)
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[warp, lane, 0] = registers[0]
    output[warp, lane, 1] = registers[1]


@T.prim_func
def raw_tcgen_st_unpack_32x32b(
    source: T.Buffer((4, 32, 2), "uint32"), output: T.Buffer((128, 4), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    physical_row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    registers = T.alloc_local((2,), "uint32")
    for col in T.unroll(4):
        tmem[physical_row, col] = T.uint32(0)
    registers[0] = source[warp, lane, 0]
    registers[1] = source[warp, lane, 1]
    T.ptx["tcgen05.st.sync.aligned.32x32b.x2.unpack::16b.b32"](
        T.uint32(0), registers[0], registers[1]
    )
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.cuda.cta_sync()
    for col in T.unroll(4):
        output[physical_row, col] = tmem[physical_row, col]


@T.prim_func
def raw_tcgen_cp_warpx4(
    source: T.Buffer((32, 4), "uint32"),
    issue: T.int32,
    output: T.Buffer((128, 4), "uint32"),
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32, 4), "uint32", scope="shared")
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    descriptor: T.uint64
    if warp == 0:
        for col in T.unroll(4):
            shared[lane, col] = source[lane, col]
    row = T.meta_var(warp * 32 + lane)
    for col in T.unroll(4):
        tmem[row, col] = T.uint32(0)
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0, 0]), 0, 8, 0
        )
        T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](
            T.uint32(0), descriptor, pred=T.And(lane == 0, issue != 0)
        )
    T.cuda.cta_sync()
    for col in T.unroll(4):
        output[row, col] = tmem[row, col]


@T.prim_func
def raw_tcgen_cp_4x256b(source: T.Buffer((512,), "uint8"), output: T.Buffer((4, 8), "uint32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    descriptor: T.uint64
    for copy_i in T.serial(16):
        shared[lane + copy_i * 32] = source[lane + copy_i * 32]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0]), ldo=8, sdo=0, swizzle=0
        )
        T.ptx["tcgen05.cp.cta_group::1.4x256b"](T.uint32(0), descriptor)
    T.cuda.cta_sync()
    if lane == 0:
        for col in T.unroll(8):
            output[warp, col] = tmem[warp * 32, col]


@T.prim_func
def raw_tcgen_cp_warpx2_01_23(
    source: T.Buffer((64, 4), "uint32"), output: T.Buffer((128, 4), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64, 4), "uint32", scope="shared")
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    descriptor: T.uint64
    if warp < 2:
        row = warp * 32 + lane
        for col in T.unroll(4):
            shared[row, col] = source[row, col]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0, 0]), ldo=0, sdo=8, swizzle=0
        )
        T.ptx["tcgen05.cp.cta_group::1.64x128b.warpx2::01_23"](T.uint32(0), descriptor)
    T.cuda.cta_sync()
    row = warp * 32 + lane
    for col in T.unroll(4):
        output[row, col] = tmem[row, col]


@T.prim_func
def raw_tcgen_cp_decompress_b4(
    source: T.Buffer((512,), "uint8"), output: T.Buffer((4, 8), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    descriptor: T.uint64
    for copy_i in T.serial(16):
        shared[lane + copy_i * 32] = source[lane + copy_i * 32]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0]), ldo=8, sdo=0, swizzle=0
        )
        T.ptx["tcgen05.cp.cta_group::1.4x256b.b8x16.b4x16_p64"](T.uint32(0), descriptor)
    T.cuda.cta_sync()
    if lane == 0:
        for col in T.unroll(8):
            output[warp, col] = tmem[warp * 32, col]


@T.prim_func
def raw_tcgen_cp_decompress_b6(
    source: T.Buffer((512,), "uint8"), output: T.Buffer((4, 8), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    descriptor: T.uint64
    for copy_i in T.serial(16):
        shared[lane + copy_i * 32] = source[lane + copy_i * 32]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0]), ldo=8, sdo=0, swizzle=0
        )
        T.ptx["tcgen05.cp.cta_group::1.4x256b.b8x16.b6x16_p32"](T.uint32(0), descriptor)
    T.cuda.cta_sync()
    if lane == 0:
        for col in T.unroll(8):
            output[warp, col] = tmem[warp * 32, col]


@T.prim_func
def raw_tcgen_cp_128b_base32b(
    source: T.Buffer((512,), "uint8"), output: T.Buffer((4, 8), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared", align=32)
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    descriptor: T.uint64
    for copy_i in T.serial(16):
        shared[lane + copy_i * 32] = source[lane + copy_i * 32]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0]), ldo=0, sdo=0, swizzle=4
        )
        T.ptx["tcgen05.cp.cta_group::1.4x256b"](T.uint32(0), descriptor)
    T.cuda.cta_sync()
    if lane == 0:
        for col in T.unroll(8):
            output[warp, col] = tmem[warp * 32, col]


@T.prim_func
def raw_tcgen_cp_128x256b_swizzle(
    source: T.Buffer((128, 32), "uint32"), output: T.Buffer((128, 8), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    shared = T.alloc_buffer((128, 32), "uint32", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    descriptor: T.uint64
    for col in T.unroll(32):
        shared[row, col] = source[row, col]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0, 0]), 1, 64, 3
        )
        T.ptx["tcgen05.cp.cta_group::1.128x256b"](T.uint32(0), descriptor)
    T.cuda.cta_sync()
    for col in T.unroll(8):
        output[row, col] = tmem[row, col]


@T.prim_func
def raw_tcgen_cp_64x128b_cta_group2(
    source: T.Buffer((2, 64, 32), "uint32"), output: T.Buffer((2, 128, 4), "uint32")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    shared = T.alloc_buffer((64, 32), "uint32", scope="shared")
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    descriptor: T.uint64
    if row < 64:
        for col in T.unroll(32):
            shared[row, col] = source[cta, row, col]
    T.cuda.cluster_sync()
    if cta == 0 and warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0, 0]), 1, 64, 3
        )
        T.ptx["tcgen05.cp.cta_group::2.64x128b.warpx2::02_13"](T.uint32(0), descriptor)
    T.cuda.cluster_sync()
    for col in T.unroll(4):
        output[cta, row, col] = tmem[row, col]


@T.prim_func
def raw_tcgen_cp_128x256b_swizzle64_logical(
    source: T.Buffer((64, 64), "bfloat16"), output: T.Buffer((2, 128, 8), "uint32")
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    physical_row = T.meta_var(warp * 32 + lane)
    shared = T.alloc_buffer((64, 64), "bfloat16", scope="shared", layout=_BF16_SMEM_64B)
    tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    descriptor: T.uint64

    if warp < 2:
        source_row = warp * 32 + lane
        for col in T.serial(64):
            shared[source_row, col] = source[source_row, col]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0, 0]), ldo=1, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.cp.cta_group::1.128x256b"](T.uint32(0), descriptor)
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0, 16]), ldo=1, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.cp.cta_group::1.128x256b"](T.uint32(8), descriptor)
    T.cuda.cta_sync()

    for word in T.unroll(8):
        output[0, physical_row, word] = tmem[physical_row, word]
        output[1, physical_row, word] = tmem[physical_row, 8 + word]


@T.prim_func
def raw_tcgen_cp_descriptor_crosses_short_pool_alias(
    source: T.Buffer((32, 16), "uint8"), output: T.Buffer((128, 4), "uint32")
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    physical_row = T.meta_var(warp * 32 + lane)
    backing = T.alloc_buffer((640,), "uint8", scope="shared")
    pool = T.meta_var(T.SMEMPool(backing.data))
    pool.move_base_to(64)
    descriptor_root = pool.alloc((16,), "uint8")
    pool.commit()
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    descriptor: T.uint64

    if warp == 0:
        for byte in T.unroll(16):
            backing[64 + lane * 16 + byte] = source[lane, byte]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(descriptor_root[0]), ldo=1, sdo=8, swizzle=0
        )
        T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](T.uint32(0), descriptor)
    T.cuda.cta_sync()

    for word in T.unroll(4):
        output[physical_row, word] = tmem[physical_row, word]


@T.prim_func
def raw_tcgen_cp_descriptor_cannot_cross_shared_backings():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    first = T.alloc_buffer((512,), "uint8", scope="shared")
    second = T.alloc_buffer((512,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    descriptor: T.uint64

    for byte in T.unroll(16):
        first[lane * 16 + byte] = T.uint8(lane)
        second[lane * 16 + byte] = T.uint8(lane + 1)
    T.cuda.cta_sync()
    if lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(first[0]), ldo=1, sdo=8, swizzle=0
        )
        descriptor = descriptor + T.uint64(512 // 16)
        T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](
            T.uint32(0),
            descriptor,
        )


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
def raw_tcgen_mxf4_bulk_reads_fail_closed(
    mode: T.int32,
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((8192,), "uint8"),
    scale_a_cells: T.Buffer((4, 32), "uint32"),
    scale_b_cells: T.Buffer((4, 32), "uint32"),
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
        if mode == 1:
            if offset != 16:
                shared_a[offset] = a_physical[offset]
        else:
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
def raw_tcgen_mma_f8f6f4_e5m2_f16_destination_predicated():
    T.device_entry()
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64
    T.cuda.tcgen05.encode_instr_descriptor(
        T.address_of(desc_i),
        d_dtype="float16",
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
        pred=T.uint32(1),
    )


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
def raw_tcgen_mma_tf32_rejects_malformed_runtime_descriptor():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_b = T.alloc_buffer((4096,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_b: T.uint64
    forged_desc: T.uint64
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
    forged_desc = T.bitwise_or(desc_b, T.uint64(1 << 14))
    T.ptx["tcgen05.mma.cta_group::1.kind::tf32"](
        T.uint32(16),
        T.uint32(0),
        forged_desc,
        desc_i,
        T.uint32(1 << 3),
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        T.ptx.pred(T.uint32(0)),
        pred=T.cast(lane == 0, "uint32"),
    )


@T.prim_func
def raw_tcgen_cp_rejects_absolute_shared_descriptor():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    for byte in T.unroll(4):
        shared[lane * 4 + byte] = T.uint8(lane)
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](
            T.uint32(0),
            T.uint64(1 << 46),
        )


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
def raw_tcgen_mma_f8f6f4_cta2_k32_without_arch():
    """The architecture-neutral K-major B, N=16 CTA2 form shared by SM100 and SM107."""

    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((16384,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_a: T.uint64
    desc_b: T.uint64

    if cta == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard"](
            T.uint32(0),
            desc_a,
            desc_b,
            T.uint32(0x10040490),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(1)),
        )


@T.prim_func
def raw_tcgen_mma_f8f6f4_cta2_k32_extended_span_without_arch():
    """K=32 does not imply the SM100 address width; architecture does."""

    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_arena = T.alloc_buffer((270336,), "uint8", scope="shared", align=1024)
    shared_a = T.decl_buffer((16384,), "uint8", data=shared_arena.data, scope="shared")
    shared_b = T.decl_buffer(
        (8192,), "uint8", data=shared_arena.data, elem_offset=262144, scope="shared"
    )
    _tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_a: T.uint64
    desc_b: T.uint64

    if cta == 0 and lane == 0:
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
            T.uint32(0x10210490),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(1)),
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
def raw_tcgen_mma_f8f6f4_cta2_rubin_rejects_descriptor_bit15():
    """Rubin widens the start field through bit 14 only; bit 15 stays reserved."""

    T.func_attr({"tirx.cuda_arch": "sm_107a"})
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16384,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_a: T.uint64
    desc_b: T.uint64

    if cta == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared[0]), ldo=0, sdo=64, swizzle=3
        )
        desc_b = T.bitwise_or(desc_b, T.uint64(1 << 15))
        T.ptx["tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard"](
            T.uint32(0),
            desc_a,
            desc_b,
            T.uint32(0x30210490),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(1)),
        )


@T.prim_func
def shared_span_above_18_bits_without_sm107_descriptor():
    """A large ordinary/SM100 shared arena must not inherit Rubin's wider descriptor."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((262160,), "uint8", scope="shared")
    if lane == 0:
        shared[262159] = T.uint8(1)


@T.prim_func
def raw_tcgen_mma_f8f6f4_reserved_operand():
    """Operand format 2 belongs to TF32, not the narrow-float family."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    if lane == 0:
        T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
            T.uint32(0),
            T.uint64(0),
            T.uint64(0),
            T.uint32((2 << 7) | (2 << 10)),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )


@T.prim_func
def raw_tcgen_mma_f8f6f4_cta_group2_reserved_destination():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    if lane == 0:
        T.ptx["tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard"](
            T.uint32(0),
            T.uint64(0),
            T.uint64(0),
            T.uint32(0x302104B0),
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


@T.prim_func
def raw_tcgen_mma_f8f6f4_cta_group2_invalid_n():
    """MN-major B keeps the CTA2 N=32 granularity, so N=16 is invalid."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    if lane == 0:
        T.ptx["tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard"](
            T.uint32(0),
            T.uint64(0),
            T.uint64(0),
            T.uint32(0x10050490),
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


@T.prim_func
def raw_tcgen_mxf4_ss_cta2_analysis_probe():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _shared = T.alloc_buffer((1,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    if lane == 0:
        T.ptx["tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X"](
            T.uint32(0),
            T.uint64(0),
            T.uint64(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            False,
        )


@T.prim_func
def raw_tcgen_mxf4_ts_cta1_analysis_probe():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _shared = T.alloc_buffer((1,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    if lane == 0:
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X"](
            T.uint32(0),
            T.uint32(0),
            T.uint64(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            False,
        )


@T.prim_func
def raw_tcgen_ws_ss_mask_analysis_probe():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _shared = T.alloc_buffer((1,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    if lane == 0:
        T.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"](
            T.uint32(0),
            T.uint64(0),
            T.uint64(0),
            T.uint32(1 << 4),
            T.ptx.pred(T.uint32(0)),
            T.uint64(1 << 39),
        )


@T.prim_func
def raw_tcgen_ws_ts_mask_analysis_probe():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _shared = T.alloc_buffer((1,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    if lane == 0:
        T.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"](
            T.uint32(0),
            T.uint32(0),
            T.uint64(0),
            T.uint32(1 << 4),
            T.ptx.pred(T.uint32(0)),
            T.uint64(1 << 39),
        )


@dataclass(frozen=True)
class _AcceptedCall:
    op_name: str


_TCGEN_CONTROL_CALLS = frozenset(
    {
        "tirx.ptx.tcgen05_alloc",
        "tirx.ptx.tcgen05_alloc_exclusive",
        "tirx.ptx.tcgen05_commit",
        "tirx.ptx.tcgen05_commit_multicast",
        "tirx.ptx.tcgen05_commit_multicast_width",
        "tirx.ptx.tcgen05_dealloc",
        "tirx.ptx.tcgen05_dealloc_exclusive",
        "tirx.ptx.tcgen05_fence",
        "tirx.ptx.tcgen05_relinquish_alloc_permit",
        "tirx.ptx.tcgen05_wait",
    }
)


def _matrix_descriptor(*, start_16b: int, ldo: int, sdo: int, layout_type: int) -> np.uint64:
    return np.uint64(start_16b | (ldo << 16) | (sdo << 32) | (1 << 46) | (layout_type << 61))


def _dense_descriptor() -> np.uint64:
    value = 0
    value |= 1 << 4  # D=f32
    value |= 1 << 7  # A=bf16
    value |= 1 << 10  # B=bf16
    value |= (128 >> 3) << 17
    value |= (64 >> 4) << 24
    return np.uint64(value)


def _block_descriptor() -> np.uint64:
    value = 0
    value |= 1 << 7  # mxf4 A format
    value |= 1 << 10  # mxf4 B format
    value |= (128 >> 3) << 17
    value |= 1 << 23  # E8M0 scale format
    value |= (128 >> 4) << 24
    return np.uint64(value)


def _swizzle_128_physical(logical: np.ndarray, row_words: int = 32) -> np.ndarray:
    physical = np.zeros((logical.shape[0], row_words), dtype=np.uint32)
    flat = physical.reshape(-1)
    for row in range(logical.shape[0]):
        for word in range(logical.shape[1]):
            unswizzled_byte = row * 128 + word * 4
            atom = unswizzled_byte // 16
            byte_in_atom = unswizzled_byte % 16
            swizzled_atom = atom ^ ((atom & (0x7 << 3)) >> 3)
            flat[(swizzled_atom * 16 + byte_in_atom) // 4] = logical[row, word]
    return physical


def _tcgen_cp_two_atom_physical(atom_bytes: np.ndarray) -> np.ndarray:
    assert atom_bytes.ndim == 3 and atom_bytes.shape[1:] == (2, 16)
    physical = np.zeros(512, dtype=np.uint8)
    for row in range(atom_bytes.shape[0]):
        for atom in range(2):
            offset = row * 16 + atom * 128
            physical[offset : offset + 16] = atom_bytes[row, atom]
    return physical


def _swizzle_128_base32_physical(logical: np.ndarray) -> np.ndarray:
    assert logical.ndim == 2 and logical.shape[1] == 32
    physical = np.zeros(512, dtype=np.uint8)
    for row in range(logical.shape[0]):
        unswizzled_atom = row * 4
        swizzled_atom = unswizzled_atom ^ ((unswizzled_atom & (0x3 << 2)) >> 2)
        offset = swizzled_atom * 32
        physical[offset : offset + 32] = logical[row]
    return physical


def _pack_b4_cp_source(codes: np.ndarray) -> np.ndarray:
    assert codes.ndim == 2 and codes.shape[1] == 32
    atoms = np.zeros((codes.shape[0], 2, 16), dtype=np.uint8)
    for row in range(codes.shape[0]):
        for atom in range(2):
            values = codes[row, atom * 16 : (atom + 1) * 16]
            atoms[row, atom, :8] = values[0::2] | (values[1::2] << np.uint8(4))
    return _tcgen_cp_two_atom_physical(atoms)


def _pack_b6_cp_source(codes: np.ndarray) -> np.ndarray:
    assert codes.ndim == 2 and codes.shape[1] == 32
    atoms = np.zeros((codes.shape[0], 2, 16), dtype=np.uint8)
    for row in range(codes.shape[0]):
        for atom in range(2):
            values = codes[row, atom * 16 : (atom + 1) * 16]
            packed = sum(int(value) << (6 * index) for index, value in enumerate(values))
            atoms[row, atom, :12] = np.frombuffer(packed.to_bytes(12, "little"), dtype=np.uint8)
    return _tcgen_cp_two_atom_physical(atoms)


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


def test_raw_tcgen_descriptor_encoders_write_exact_bits(tmp_path):
    output = np.zeros(3, dtype=np.uint64)
    module = numsim.transpile(raw_tcgen_descriptor_encode, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    expected = np.array(
        [
            _matrix_descriptor(start_16b=1, ldo=1, sdo=8, layout_type=2),
            _dense_descriptor(),
            _block_descriptor(),
        ],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tcgen_descriptor_encoders_resolve_global_integer_destinations(tmp_path):
    module = numsim.transpile(raw_tcgen_descriptor_encode_to_global, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "matrix_output": np.zeros(1, dtype=np.uint64),
            "instr_output": np.zeros(1, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(
        result.outputs["matrix_output"],
        np.array([_matrix_descriptor(start_16b=1, ldo=1, sdo=8, layout_type=2)], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        result.outputs["instr_output"], np.array([_dense_descriptor()], dtype=np.uint32)
    )


def test_raw_tcgen_32x32b_ld_st_roundtrip(tmp_path):
    source = np.arange(128 * 4, dtype=np.uint32).reshape(128, 4) * np.uint32(17) + 3
    output = np.zeros_like(source)
    module = numsim.transpile(raw_tcgen_ldst_32x32b, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_raw_tcgen_16x256b_ld_observes_exact_lane_register_mapping(tmp_path):
    source = np.arange(128 * 8, dtype=np.uint32).reshape(128, 8) + np.uint32(1000)
    output = np.zeros((4, 32, 4), dtype=np.uint32)
    module = numsim.transpile(raw_tcgen_ld_16x256b_mapping, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = np.empty_like(output)
    for warp in range(4):
        for lane in range(32):
            for register in range(4):
                row = warp * 32 + (lane >> 2) + 8 * ((register >> 1) & 1)
                col = (register & 1) + 2 * (lane & 3)
                expected[warp, lane, register] = source[row, col]
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tcgen_missing_ld_shapes_observe_exact_lane_register_mappings(tmp_path):
    source = np.arange(128 * 16, dtype=np.uint32).reshape(128, 16) + np.uint32(5000)
    output = np.zeros((3, 4, 32, 2), dtype=np.uint32)
    module = numsim.transpile(raw_tcgen_ld_missing_shape_mappings, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = np.empty_like(output)
    for warp in range(4):
        for lane in range(32):
            for register in range(2):
                expected[0, warp, lane, register] = source[
                    warp * 32 + lane % 16, register + (lane // 16) * 2
                ]
                expected[1, warp, lane, register] = source[
                    warp * 32 + (lane >> 2) + 8 * (lane & 1), ((lane >> 1) & 1) + 2 * register
                ]
                expected[2, warp, lane, register] = source[
                    warp * 32 + (lane >> 2) + 8 * (register & 1), (lane & 3) + 4 * (register >> 1)
                ]
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tcgen_ld_pack_and_st_unpack_use_adjacent_tmem_columns(tmp_path):
    source = np.arange(128 * 4, dtype=np.uint32).reshape(128, 4) * np.uint32(17) + 3
    packed_output = np.zeros((4, 32, 2), dtype=np.uint32)
    packed_module = numsim.transpile(raw_tcgen_ld_pack_32x32b, cache_dir=tmp_path / "ld")
    packed = (
        numsim.Engine()
        .run(packed_module, {"source": source, "output": packed_output})
        .outputs["output"]
    )
    expected_packed = np.empty_like(packed)
    for warp in range(4):
        for lane in range(32):
            row = warp * 32 + lane
            for register in range(2):
                expected_packed[warp, lane, register] = np.uint32(
                    source[row, register * 2] & np.uint32(0xFFFF)
                ) | np.uint32((source[row, register * 2 + 1] & np.uint32(0xFFFF)) << np.uint32(16))
    np.testing.assert_array_equal(packed, expected_packed)

    unpacked_output = np.zeros((128, 4), dtype=np.uint32)
    unpacked_module = numsim.transpile(raw_tcgen_st_unpack_32x32b, cache_dir=tmp_path / "st")
    unpacked = (
        numsim.Engine()
        .run(unpacked_module, {"source": expected_packed, "output": unpacked_output})
        .outputs["output"]
    )
    np.testing.assert_array_equal(unpacked, source & np.uint32(0xFFFF))


def test_raw_tcgen_descriptor_bits_can_cross_into_another_valid_backing(tmp_path):
    module = numsim.transpile(
        raw_tcgen_cp_descriptor_cannot_cross_shared_backings, cache_dir=tmp_path
    )
    result = numsim.Engine().run(module, {})
    assert result.verdict == "clean"


def test_raw_tcgen_f8f6f4_cta2_rubin_keeps_descriptor_bit15_reserved(tmp_path):
    module = numsim.transpile(
        raw_tcgen_mma_f8f6f4_cta2_rubin_rejects_descriptor_bit15,
        cache_dir=tmp_path,
    )

    with pytest.raises(
        numsim.NumSimExecutionError,
        match="descriptor uses unsupported reserved/base/LBO-mode bits",
    ):
        numsim.Engine().run(module, {})


def test_raw_tcgen_descriptor_bits_are_validated_by_the_engine(tmp_path):
    module = numsim.transpile(
        raw_tcgen_mma_tf32_rejects_malformed_runtime_descriptor, cache_dir=tmp_path
    )

    with pytest.raises(
        numsim.NumSimExecutionError, match="descriptor uses unsupported reserved/base/LBO-mode bits"
    ):
        numsim.Engine().run(module, {})


def test_raw_tcgen_absolute_shared_descriptor_resolves_from_its_bits(tmp_path):
    module = numsim.transpile(raw_tcgen_cp_rejects_absolute_shared_descriptor, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {})
    assert result.verdict == "clean"


