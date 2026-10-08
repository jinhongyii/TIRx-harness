"""v2 copies of nine ``tests/numsim/runtime/test_raw_tcgen_codegen.py`` tests
(internal-surface ``other-assertion`` triage, W9 phase 6).

**Eight ``tcgen05.cp`` tests: delta (the legacy kernels race).** Each legacy
kernel issues ``tcgen05.cp`` and then reads the destination TMEM after a plain
``cta_sync``/``cluster_sync``, with no ``tcgen05.commit`` + ``mbarrier`` wait
(and no ``fence.proxy.async`` between the generic shared-memory writes and the
copy's shared-memory read). PTX ISA (tcgen05 memory consistency model):
``tcgen05.cp`` is asynchronous; its completion is observed only through
``tcgen05.commit`` and a wait on the committed mbarrier. Legacy NumSim executed
the copy synchronously at issue, so the read saw the copied data; v2 models it
as an async op (``Payload::TcgenCp``, ``class: TcgenPipelined``;
``CONTRACT_REQUESTS.md`` "W4-7: tcgen ld/st/cp ... APIs for W2" and the
``tcgen_cp`` handler) that the engine lands only when ordered, so the un-waited
reads see the old TMEM bytes (zeros). racecheck reports these kernels as
``data_race`` (``missing_inter_actor_sync`` on the TMEM read and
``missing_proxy_bridge`` on the shared read) plus ``effect_commit_unobserved``
(``racecheck-behaviour-deltas.md`` row P6: ``AsyncNeverCompleted`` for ops the
engine never landed). No ``numsim-behaviour-deltas.md`` row records the NumSim
value change yet (W8 may want one).

Each copy therefore (1) runs the legacy kernel verbatim through
``v2.racecheck`` and asserts the race verdict, and (2) runs a corrected copy of
the kernel (``fence.proxy.async`` + ``tcgen05.fence`` around the CTA sync,
``tcgen05.commit`` to an mbarrier right after the copy, every thread waits on
it, then ``tcgen05.fence::after_thread_sync``) and asserts the legacy expected
values exactly, plus a clean racecheck. The data-path facts the legacy tests
pinned (warpx4/warpx2 replication, 4x256b atoms, 128B/32B-atom swizzle
decoding, b4/b6 decompression, cta_group::2 both-CTA writes, crossing a short
pool alias inside its backing) are all kept.

**``test_raw_tcgen_mxf4_bulk_smem_and_tmem_reads_require_review``: bug.** The
``mode=1`` (shared) half passes unchanged. The ``mode=2`` half (the MMA
accumulates into D = TMEM lanes 0..127, columns 0..127, never written) is
``clean`` in v2; legacy reports ``uninitialized_read`` (review, space
``tmem``). A direct buffer read of unwritten TMEM is reported in v2, so the gap
is the async MMA path: ``sched/partition.rs`` runs ``report_async_uninit`` on
the op's read spans *after* ``run_mma`` has written D (and so marked those
bytes valid), so the read-modify-write D read is never reported (owner W2).
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
from tvm.script import tirx as T
from tvm.tirx.layout import S, TCol, TileLayout, TLane

from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

_TMEM_D_4 = TileLayout(S[(128, 4) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_8 = TileLayout(S[(128, 8) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_16 = TileLayout(S[(128, 16) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_136 = TileLayout(S[(128, 136) : (1 @ TLane, 1 @ TCol)])
_BF16_SMEM_64B = mma_shared_layout("bfloat16", SwizzleMode.SWIZZLE_64B_ATOM, (64, 64))

_LEGACY = "tests/numsim/runtime/test_raw_tcgen_codegen.py"


# -- helpers (copied from the legacy module) ---------------------------------


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


def _pack_e8m0_cells(scales: np.ndarray) -> np.ndarray:
    assert scales.ndim == 2 and scales.shape[1] == 4 and scales.shape[0] % 32 == 0
    cells = np.zeros((scales.shape[0] // 32, 32), dtype=np.uint32)
    for row in range(scales.shape[0]):
        value = sum(int(scales[row, index]) << (8 * index) for index in range(4))
        cells[row // 32, row % 32] = np.uint32(value)
    return cells


def _run(kernel, inputs):
    return v2.Engine().run(v2.transpile(kernel), inputs)


def _assert_legacy_kernel_races(kernel, inputs):
    """The verbatim legacy kernel reads the ``tcgen05.cp`` destination without
    a commit/wait: racecheck reports a data race (module docstring)."""

    report = v2.racecheck(kernel, inputs)
    assert report.verdict == "error", report.format()
    kinds = {(f.status, f.kind) for f in report.findings}
    assert ("error", "data_race") in kinds, report.format()


def _assert_clean(kernel, inputs):
    report = v2.racecheck(kernel, inputs)
    assert report.verdict == "clean", report.format()


# -- legacy kernels (verbatim) -----------------------------------------------


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


# -- corrected kernels: the legacy data path plus commit/wait and fences -----


@T.prim_func
def synced_cp_warpx4(
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
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    descriptor: T.uint64
    if warp == 0:
        for col in T.unroll(4):
            shared[lane, col] = source[lane, col]
    row = T.meta_var(warp * 32 + lane)
    for col in T.unroll(4):
        tmem[row, col] = T.uint32(0)
    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0, 0]), 0, 8, 0
        )
        T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](
            T.uint32(0), descriptor, pred=T.And(lane == 0, issue != 0)
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    for col in T.unroll(4):
        output[row, col] = tmem[row, col]


@T.prim_func
def synced_cp_4x256b(source: T.Buffer((512,), "uint8"), output: T.Buffer((4, 8), "uint32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    descriptor: T.uint64
    if warp == 0:
        for copy_i in T.serial(16):
            shared[lane + copy_i * 32] = source[lane + copy_i * 32]
    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0]), ldo=8, sdo=0, swizzle=0
        )
        T.ptx["tcgen05.cp.cta_group::1.4x256b"](T.uint32(0), descriptor)
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    if lane == 0:
        for col in T.unroll(8):
            output[warp, col] = tmem[warp * 32, col]


@T.prim_func
def synced_cp_warpx2_01_23(
    source: T.Buffer((64, 4), "uint32"), output: T.Buffer((128, 4), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64, 4), "uint32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    descriptor: T.uint64
    if warp < 2:
        row = warp * 32 + lane
        for col in T.unroll(4):
            shared[row, col] = source[row, col]
    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0, 0]), ldo=0, sdo=8, swizzle=0
        )
        T.ptx["tcgen05.cp.cta_group::1.64x128b.warpx2::01_23"](T.uint32(0), descriptor)
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    row = warp * 32 + lane
    for col in T.unroll(4):
        output[row, col] = tmem[row, col]


@T.prim_func
def synced_cp_decompress_b4(
    source: T.Buffer((512,), "uint8"), output: T.Buffer((4, 8), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    descriptor: T.uint64
    if warp == 0:
        for copy_i in T.serial(16):
            shared[lane + copy_i * 32] = source[lane + copy_i * 32]
    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0]), ldo=8, sdo=0, swizzle=0
        )
        T.ptx["tcgen05.cp.cta_group::1.4x256b.b8x16.b4x16_p64"](T.uint32(0), descriptor)
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    if lane == 0:
        for col in T.unroll(8):
            output[warp, col] = tmem[warp * 32, col]


@T.prim_func
def synced_cp_decompress_b6(
    source: T.Buffer((512,), "uint8"), output: T.Buffer((4, 8), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    descriptor: T.uint64
    if warp == 0:
        for copy_i in T.serial(16):
            shared[lane + copy_i * 32] = source[lane + copy_i * 32]
    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0]), ldo=8, sdo=0, swizzle=0
        )
        T.ptx["tcgen05.cp.cta_group::1.4x256b.b8x16.b6x16_p32"](T.uint32(0), descriptor)
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    if lane == 0:
        for col in T.unroll(8):
            output[warp, col] = tmem[warp * 32, col]


@T.prim_func
def synced_cp_128b_base32b(
    source: T.Buffer((512,), "uint8"), output: T.Buffer((4, 8), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared", align=32)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    descriptor: T.uint64
    if warp == 0:
        for copy_i in T.serial(16):
            shared[lane + copy_i * 32] = source[lane + copy_i * 32]
    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0]), ldo=0, sdo=0, swizzle=4
        )
        T.ptx["tcgen05.cp.cta_group::1.4x256b"](T.uint32(0), descriptor)
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    if lane == 0:
        for col in T.unroll(8):
            output[warp, col] = tmem[warp * 32, col]


@T.prim_func
def synced_cp_128x256b_swizzle(
    source: T.Buffer((128, 32), "uint32"), output: T.Buffer((128, 8), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    shared = T.alloc_buffer((128, 32), "uint32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    descriptor: T.uint64
    for col in T.unroll(32):
        shared[row, col] = source[row, col]
    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0, 0]), 1, 64, 3
        )
        T.ptx["tcgen05.cp.cta_group::1.128x256b"](T.uint32(0), descriptor)
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    for col in T.unroll(8):
        output[row, col] = tmem[row, col]


@T.prim_func
def synced_cp_64x128b_cta_group2(
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
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    descriptor: T.uint64
    if row < 64:
        for col in T.unroll(32):
            shared[row, col] = source[cta, row, col]
    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if cta == 0 and warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0, 0]), 1, 64, 3
        )
        T.ptx["tcgen05.cp.cta_group::2.64x128b.warpx2::02_13"](T.uint32(0), descriptor)
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(barrier[0]), T.uint16(3)
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    for col in T.unroll(4):
        output[cta, row, col] = tmem[row, col]
    T.cuda.cluster_sync()


@T.prim_func
def synced_cp_128x256b_swizzle64_logical(
    source: T.Buffer((64, 64), "bfloat16"), output: T.Buffer((2, 128, 8), "uint32")
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    physical_row = T.meta_var(warp * 32 + lane)
    shared = T.alloc_buffer((64, 64), "bfloat16", scope="shared", layout=_BF16_SMEM_64B)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    descriptor: T.uint64

    if warp < 2:
        source_row = warp * 32 + lane
        for col in T.serial(64):
            shared[source_row, col] = source[source_row, col]
    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
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
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    for word in T.unroll(8):
        output[0, physical_row, word] = tmem[physical_row, word]
        output[1, physical_row, word] = tmem[physical_row, 8 + word]


@T.prim_func
def synced_cp_descriptor_crosses_short_pool_alias(
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
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    descriptor: T.uint64

    if warp == 0:
        for byte in T.unroll(16):
            backing[64 + lane * 16 + byte] = source[lane, byte]
    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(descriptor_root[0]), ldo=1, sdo=8, swizzle=0
        )
        T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](T.uint32(0), descriptor)
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.ptx.tcgen05.fence__after_thread_sync()

    for word in T.unroll(4):
        output[physical_row, word] = tmem[physical_row, word]


# -- tests --------------------------------------------------------------------


def _check(legacy_kernel, synced_kernel, inputs, expected, output="output", legacy_races=True):
    if legacy_races:
        _assert_legacy_kernel_races(legacy_kernel, inputs())
    else:
        _assert_clean(legacy_kernel, inputs())
    result = _run(synced_kernel, inputs())
    assert result.status.get("kind") == "completed", result.status
    np.testing.assert_array_equal(result.outputs[output], expected)
    _assert_clean(synced_kernel, inputs())


def test_raw_tcgen_cp_warpx4_replicates_source_into_all_warp_lanes():
    """Delta copy of ``tests/numsim/runtime/test_raw_tcgen_codegen.py::test_raw_tcgen_cp_warpx4_replicates_source_into_all_warp_lanes``
    (un-waited ``tcgen05.cp``; module docstring). Both legacy cases kept:
    ``issue=1`` replicates the 32x128b source into all four warps' lanes,
    ``issue=0`` (predicated off) leaves the zeroed TMEM; with no copy issued
    the legacy kernel is race-free (racecheck clean), and both kernels give
    zeros."""

    source = np.arange(32 * 4, dtype=np.uint32).reshape(32, 4) * np.uint32(11) + 5
    for issue, expected in ((1, np.tile(source, (4, 1))), (0, np.zeros((128, 4), np.uint32))):
        _check(
            raw_tcgen_cp_warpx4,
            synced_cp_warpx4,
            lambda: {"source": source, "issue": issue, "output": np.full((128, 4), 99, np.uint32)},
            expected,
            legacy_races=bool(issue),
        )
    legacy_skipped = _run(
        raw_tcgen_cp_warpx4,
        {"source": source, "issue": 0, "output": np.full((128, 4), 99, np.uint32)},
    )
    np.testing.assert_array_equal(legacy_skipped.outputs["output"], np.zeros((128, 4), np.uint32))


def test_raw_tcgen_cp_128x256b_decodes_matrix_descriptor_swizzle():
    """Delta copy of ``tests/numsim/runtime/test_raw_tcgen_codegen.py::test_raw_tcgen_cp_128x256b_decodes_matrix_descriptor_swizzle``
    (un-waited ``tcgen05.cp``; module docstring). v2 ran the legacy kernel to
    rows 0..95 right and rows 96..127 still zero: the copy landed between the
    reads of different warps."""

    logical = np.arange(128 * 8, dtype=np.uint32).reshape(128, 8) * np.uint32(13) + 7
    source = _swizzle_128_physical(logical)
    _check(
        raw_tcgen_cp_128x256b_swizzle,
        synced_cp_128x256b_swizzle,
        lambda: {"source": source, "output": np.zeros_like(logical)},
        logical,
    )


def test_raw_tcgen_cp_covers_4x256b_and_both_warpx2_pairings():
    """Delta copy of ``tests/numsim/runtime/test_raw_tcgen_codegen.py::test_raw_tcgen_cp_covers_4x256b_and_both_warpx2_pairings``
    (un-waited ``tcgen05.cp``; module docstring). Both kernels kept (4x256b and
    64x128b ``warpx2::01_23``)."""

    logical_4x256 = np.arange(4 * 8, dtype=np.uint32).reshape(4, 8) * np.uint32(29) + np.uint32(7)
    atoms = logical_4x256.view(np.uint8).reshape(4, 2, 16)
    source_4x256 = _tcgen_cp_two_atom_physical(atoms)
    _check(
        raw_tcgen_cp_4x256b,
        synced_cp_4x256b,
        lambda: {"source": source_4x256, "output": np.zeros((4, 8), dtype=np.uint32)},
        logical_4x256,
    )

    source_64 = np.arange(64 * 4, dtype=np.uint32).reshape(64, 4) * np.uint32(13) + 5
    expected_64 = np.concatenate(
        [source_64[:32], source_64[:32], source_64[32:], source_64[32:]], axis=0
    )
    _check(
        raw_tcgen_cp_warpx2_01_23,
        synced_cp_warpx2_01_23,
        lambda: {"source": source_64, "output": np.zeros((128, 4), dtype=np.uint32)},
        expected_64,
    )


@pytest.mark.parametrize(
    ("legacy_kernel", "synced_kernel", "codes", "packer", "shift"),
    [
        (raw_tcgen_cp_decompress_b4, synced_cp_decompress_b4, np.uint8(16), _pack_b4_cp_source, np.uint8(2)),
        (raw_tcgen_cp_decompress_b6, synced_cp_decompress_b6, np.uint8(64), _pack_b6_cp_source, np.uint8(0)),
    ],
    ids=["b4", "b6"],
)
def test_raw_tcgen_cp_decompression_expands_packed_codes(legacy_kernel, synced_kernel, codes, packer, shift):
    """Delta copy of ``tests/numsim/runtime/test_raw_tcgen_codegen.py::test_raw_tcgen_cp_decompression_expands_packed_codes``
    (both params; un-waited ``tcgen05.cp``; module docstring)."""

    logical = (np.arange(4 * 32, dtype=np.uint8).reshape(4, 32) * np.uint8(7) + np.uint8(3)) % codes
    source = packer(logical)
    expected = np.ascontiguousarray(logical << shift).view(np.uint32)
    _check(
        legacy_kernel,
        synced_kernel,
        lambda: {"source": source, "output": np.zeros((4, 8), dtype=np.uint32)},
        expected,
    )


def test_raw_tcgen_cp_128b_base32b_descriptor_uses_32byte_atomic_swizzle():
    """Delta copy of ``tests/numsim/runtime/test_raw_tcgen_codegen.py::test_raw_tcgen_cp_128b_base32b_descriptor_uses_32byte_atomic_swizzle``
    (un-waited ``tcgen05.cp``; module docstring)."""

    logical = np.arange(4 * 32, dtype=np.uint8).reshape(4, 32) * np.uint8(5) + np.uint8(1)
    source = _swizzle_128_base32_physical(logical)
    _check(
        raw_tcgen_cp_128b_base32b,
        synced_cp_128b_base32b,
        lambda: {"source": source, "output": np.zeros((4, 8), dtype=np.uint32)},
        np.ascontiguousarray(logical).view(np.uint32),
    )


def test_raw_tcgen_cp_cta_group2_writes_both_ctas_with_warpx2_routing():
    """Delta copy of ``tests/numsim/runtime/test_raw_tcgen_codegen.py::test_raw_tcgen_cp_cta_group2_writes_both_ctas_with_warpx2_routing``
    (un-waited ``tcgen05.cp``; module docstring). The corrected kernel commits
    with ``cta_group::2 ... multicast::cluster`` to both CTAs' barriers."""

    logical = np.arange(2 * 64 * 4, dtype=np.uint32).reshape(2, 64, 4) * np.uint32(19) + 9
    source = np.stack([_swizzle_128_physical(logical[cta]) for cta in range(2)])
    expected = np.stack(
        [
            np.concatenate(
                [logical[cta, :32], logical[cta, 32:], logical[cta, :32], logical[cta, 32:]], axis=0
            )
            for cta in range(2)
        ]
    )
    _check(
        raw_tcgen_cp_64x128b_cta_group2,
        synced_cp_64x128b_cta_group2,
        lambda: {"source": source, "output": np.zeros((2, 128, 4), dtype=np.uint32)},
        expected,
    )


def test_raw_tcgen_cp_128x256b_swizzle64_routes_both_k_halves_for_every_row():
    """Delta copy of ``tests/numsim/runtime/test_raw_tcgen_codegen.py::test_raw_tcgen_cp_128x256b_swizzle64_routes_both_k_halves_for_every_row``
    (un-waited ``tcgen05.cp``; module docstring)."""

    row = np.arange(64, dtype=np.uint16)[:, None]
    col = np.arange(64, dtype=np.uint16)[None, :]
    source_bits = (np.uint16(0x3C00) + row * np.uint16(67) + col).astype(np.uint16)
    packed = source_bits[:, 0::2].astype(np.uint32) | (
        source_bits[:, 1::2].astype(np.uint32) << np.uint32(16)
    )
    expected = np.empty((2, 128, 8), dtype=np.uint32)
    expected[0, :64] = packed[:, :8]
    expected[0, 64:] = packed[:, 16:24]
    expected[1, :64] = packed[:, 8:16]
    expected[1, 64:] = packed[:, 24:32]
    _check(
        raw_tcgen_cp_128x256b_swizzle64_logical,
        synced_cp_128x256b_swizzle64_logical,
        lambda: {"source": source_bits, "output": np.zeros((2, 128, 8), dtype=np.uint32)},
        expected,
    )


def test_raw_tcgen_descriptor_can_cross_a_short_pool_alias_within_its_backing():
    """Delta copy of ``tests/numsim/runtime/test_raw_tcgen_codegen.py::test_raw_tcgen_descriptor_can_cross_a_short_pool_alias_within_its_backing``
    (un-waited ``tcgen05.cp``; module docstring). The corrected kernel keeps the
    16-byte pool alias as descriptor root, so the copy still reads 512 bytes
    past it within the 640-byte backing."""

    source = (np.arange(32 * 16, dtype=np.uint16).reshape(32, 16) * 13 + 7).astype(np.uint8)
    packed = np.ascontiguousarray(source).view(np.uint32).reshape(32, 4)
    _check(
        raw_tcgen_cp_descriptor_crosses_short_pool_alias,
        synced_cp_descriptor_crosses_short_pool_alias,
        lambda: {"source": source, "output": np.zeros((128, 4), dtype=np.uint32)},
        np.tile(packed, (4, 1)),
    )


def _mxf4_bindings(mode):
    zeros = np.zeros(8192, dtype=np.uint8)
    scale_cells = _pack_e8m0_cells(np.full((128, 4), np.uint8(127), dtype=np.uint8))
    return {
        "mode": mode,
        "a_physical": zeros,
        "b_physical": zeros,
        "scale_a_cells": scale_cells,
        "scale_b_cells": scale_cells,
    }


@pytest.mark.parametrize(
    ("mode", "expected_space"),
    [
        (1, "shared"),
        pytest.param(
            2,
            "tmem",
            marks=v2_gap(
                "tcgen05.mma accumulating into never-written TMEM D is 'clean' (no "
                "uninitialized_read): sched/partition.rs reports async uninit reads after "
                "run_mma has written D, so the RMW D read looks valid"
            ),
        ),
    ],
    ids=["shared", "tmem"],
)
def test_raw_tcgen_mxf4_bulk_smem_and_tmem_reads_require_review(mode, expected_space):
    """Copy of ``tests/numsim/runtime/test_raw_tcgen_codegen.py::test_raw_tcgen_mxf4_bulk_smem_and_tmem_reads_require_review``
    (legacy loops over ``mode`` 1 and 2; split into two params). Assertions
    kept verbatim. ``mode=2`` is a v2 bug (module docstring)."""

    result = _run(raw_tcgen_mxf4_bulk_reads_fail_closed, _mxf4_bindings(mode))

    assert result.verdict == "review"
    assert result.diagnostics
    assert {item["status"] for item in result.diagnostics} == {"review"}
    assert {item["kind"] for item in result.diagnostics} == {"uninitialized_read"}
    assert {item["space"] for item in result.diagnostics} == {expected_space}
