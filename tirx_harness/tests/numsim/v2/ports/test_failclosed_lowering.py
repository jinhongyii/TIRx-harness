"""v2 copies of six tests whose kernels v2 rejects at ``transpile`` by design
(W1, numsim-behaviour-deltas "Lowering fail-closed forms"):

* L1: direct load/store through a replicated TMEM view (contract item 29),
  ``UnsupportedTIRxError`` matching ``tmem_replicated_view``;
* L2: ``tirx.ptx_legacy.*`` builtins, matching ``tirx.ptx_legacy``.

The legacy tests asserted numerical outputs of these forms; the new contract is
the fail-closed rejection. Kernels copied verbatim (two come from the
pure-TVM ``tests/numsim/support/kernels.py``).
"""

from __future__ import annotations

import pytest
from tvm.backend.cuda.tile_primitive.gemm_async.tcgen05 import sf_smem_layout, sf_tmem_layout
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import tmem_datapath_layout

from tests.numsim.support.kernels import tcgen_scale_bitcast_cta_group2, tcgen_scale_bitcast_shared_to_tmem
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine


@T.prim_func
def _tcgen_cp_two_pairs_in_one_cluster(
    source: T.Buffer((4, 128, 4), "uint8"),
    output: T.Buffer((4, 128, 4), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([4])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    shared = T.alloc_buffer(
        (128, 4),
        "uint8",
        scope="shared",
        layout=sf_smem_layout(128, SF_K=4, sf_per_mma=4),
    )
    scale_tmem = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=4),
        allocated_addr=0,
    )
    for col in T.serial(4):
        shared[row, col] = source[cta, row, col]
    T.cuda.cluster_sync()
    if ((cta % 2) == 0) and (warp == 0) and (lane == 0):
        Tx.copy_async(scale_tmem[:, :], shared[:, :], cta_group=2)
    T.cuda.cluster_sync()
    for col in T.serial(4):
        output[cta, row, col] = scale_tmem[row, col]


_PACKED_FP4_MMA_128X32 = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (128, 32))


_PACKED_FP4_MMA_8X32 = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (8, 32))


@T.prim_func
def block_scaled_mxfp4_gemm(
    left_packed: T.Buffer((128, 32), "uint8"),
    right_packed: T.Buffer((8, 32), "uint8"),
    scale_a: T.Buffer((128, 2), "float8_e8m0fnu"),
    scale_b: T.Buffer((8, 2), "float8_e8m0fnu"),
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
        (128, 2),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=2, sf_per_mma=2),
        allocated_addr=16,
    )
    scale_b_tmem = T.decl_buffer(
        (128, 2),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=2, sf_per_mma=2),
        allocated_addr=24,
    )
    if lane == 0:
        Tx.copy(left_shared_packed[:, :], left_packed[:, :])
        Tx.copy(right_shared_packed[:, :], right_packed[:, :])
        for row_index in T.serial(128):
            for scale_index in T.serial(2):
                scale_a_tmem[row_index, scale_index] = scale_a[row_index, scale_index]
        for row_index in T.serial(8):
            for scale_index in T.serial(2):
                scale_b_tmem[row_index, scale_index] = scale_b[row_index, scale_index]
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
        for row_index in T.serial(128):
            for column_index in T.serial(8):
                output[row_index, column_index] = accumulator[row_index, column_index]


@T.prim_func
def ptx_mma_legacy_s8_u8_m16n8k32(
    a: T.Buffer((16, 32), "int8"),
    b: T.Buffer((32, 8), "uint8"),
    c: T.Buffer((16, 8), "int32"),
    output: T.Buffer((16, 8), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((16,), "int8")
    b_regs = T.alloc_local((8,), "uint8")
    accumulator = T.alloc_local((4,), "int32")
    for register in T.unroll(4):
        row = T.meta_var(group + (register % 2) * 8)
        col = T.meta_var(thread * 4 + (register // 2) * 16)
        for packed in T.unroll(4):
            a_regs[register * 4 + packed] = a[row, col + packed]
    for register in T.unroll(2):
        row = T.meta_var(thread * 4 + register * 16)
        for packed in T.unroll(4):
            b_regs[register * 4 + packed] = b[row + packed, group]
    accumulator[0] = c[group, thread * 2]
    accumulator[1] = c[group, thread * 2 + 1]
    accumulator[2] = c[group + 8, thread * 2]
    accumulator[3] = c[group + 8, thread * 2 + 1]
    T.ptx_legacy.mma(
        "m16n8k32",
        "row",
        "col",
        "int8",
        "uint8",
        "int32",
        a_regs.data,
        0,
        b_regs.data,
        0,
        accumulator.data,
        0,
        False,
        dtype="int32",
    )
    output[group, thread * 2] = accumulator[0]
    output[group, thread * 2 + 1] = accumulator[1]
    output[group + 8, thread * 2] = accumulator[2]
    output[group + 8, thread * 2 + 1] = accumulator[3]


@T.prim_func
def legacy_ldmatrix_x1_domain(output: T.Buffer((64,), "uint16")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((256,), "uint16", scope="shared")
    local = T.alloc_buffer((2,), "uint16", scope="local")
    for element in T.unroll(8):
        shared[lane * 8 + element] = T.cast(lane * 8 + element + 1, "uint16")
    T.cuda.warp_sync()
    T.evaluate(
        T.ptx_legacy.ldmatrix(
            False,
            1,
            ".b16",
            local.data,
            0,
            shared.data,
            0,
            dtype="uint16",
        )
    )
    output[lane * 2] = local[0]
    output[lane * 2 + 1] = local[1]


def _rejects(kernel, match: str) -> None:
    with pytest.raises(UnsupportedTIRxError, match=match):
        v2.transpile(kernel)


def test_tcgen_cp_bitcasts_uint8_scale_payload_into_float8_tmem():
    """Port of ``tests/numsim/integration/test_tcgen_transfer_artifact.py::test_tcgen_cp_bitcasts_uint8_scale_payload_into_float8_tmem`` (L1)."""
    _rejects(tcgen_scale_bitcast_shared_to_tmem, "tmem_replicated_view")


def test_tcgen_cp_cta_group2_reads_and_writes_each_cta_scale_backing():
    """Port of ``tests/numsim/integration/test_tcgen_transfer_artifact.py::test_tcgen_cp_cta_group2_reads_and_writes_each_cta_scale_backing`` (L1)."""
    _rejects(tcgen_scale_bitcast_cta_group2, "tmem_replicated_view")


def test_tcgen_cp_cta_group2_routes_each_pair_in_four_cta_cluster():
    """Port of ``tests/numsim/integration/test_tcgen_transfer_artifact.py::test_tcgen_cp_cta_group2_routes_each_pair_in_four_cta_cluster`` (L1)."""
    _rejects(_tcgen_cp_two_pairs_in_one_cluster, "tmem_replicated_view")


def test_mxfp4_uses_ue8m0_scales_over_32_element_vectors():
    """Port of ``tests/numsim/runtime/test_tile_general_semantics.py::test_mxfp4_uses_ue8m0_scales_over_32_element_vectors`` (L1)."""
    _rejects(block_scaled_mxfp4_gemm, "tmem_replicated_view")


def test_legacy_m16n8k32_int8_reuses_dense_form_and_engine():
    """Port of ``tests/numsim/runtime/test_dense_mma_forms.py::test_legacy_m16n8k32_int8_reuses_dense_form_and_engine`` (L2)."""
    _rejects(ptx_mma_legacy_s8_u8_m16n8k32, "tirx.ptx_legacy")


def test_legacy_ldmatrix_x1_domain_matches_independent_fragment_mapping():
    """Port of ``tests/numsim/runtime/test_matrix_memory_domain_oracle.py::test_legacy_ldmatrix_x1_domain_matches_independent_fragment_mapping`` (L2)."""
    _rejects(legacy_ldmatrix_x1_domain, "tirx.ptx_legacy")
