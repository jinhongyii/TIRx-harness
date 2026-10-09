"""v2 expected-error copies of ``tests/numsim/integration/test_raw_versus_typed_cta2_mma.py``
(``test_raw_cta2_mma_matches_the_typed_gemm_async_exactly`` [3 params] and
``test_raw_cta2_ts_m128_selects_the_matching_a_lane_bank``).

Ruling: ``numsim-behaviour-deltas.md`` row **T4** (CONTRACT_REQUESTS.md "Contract
changes for W1" Batch 2 item 17; W2 engine-stops triage "TMEM buffer access outside
the warp's sub-partition"). A buffer-form TMEM ``Load``/``Store`` executes as
``tcgen05.ld/st .32x32b``: warp ``w`` may address only TMEM lanes
``32 * (w % 4) .. +32``. Every kernel here runs one warp (warp 0) whose lane 0
reads the accumulator rows (or writes the A-operand rows) at TMEM lanes >= 32, so
both the raw and the typed kernel stop with a ``bad_address`` execution error.
Legacy modelled TMEM buffers abstractly and compared the two outputs; no param
of either function avoids the rule, so no numerical assertion survives.
Kernels are copied verbatim from the legacy module.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import (
    ComposeLayout,
    S,
    TCol,
    TileLayout,
    TLane,
    tmem_datapath_layout,
)

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

_MMA_F16_32B = ComposeLayout(3, 1, 3, TileLayout(S[(128,)]))
_TMEM_A_CTA2_M128_BANKED_F16 = TileLayout(S[(2, 64, 16) : (64 @ TLane, 1 @ TLane, 1 @ TCol)])


@T.prim_func
def raw_cta2_ss_layout_b_mma(
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((128, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("B", 64, 32),
        allocated_addr=0,
    )
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if lane == 0:
        Tx.copy(left_shared[:, :], left[cta * 64 : (cta + 1) * 64, :])
        Tx.copy(right_shared[:, :], right[cta * 16 : (cta + 1) * 16, :])
    T.cuda.cluster_sync()

    if (cta == 0) and (lane == 0):
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=128,
            N=32,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(left_shared[0, 0]), ldo=16, sdo=16, swizzle=1
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(right_shared[0, 0]), ldo=16, sdo=16, swizzle=1
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cluster_sync()

    if lane == 0:
        for row in T.serial(64):
            for col in T.serial(32):
                output[cta * 64 + row, col] = accumulator[row, col]


@T.prim_func
def typed_cta2_ss_layout_b_gemm_async(
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((128, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("B", 64, 32),
        allocated_addr=0,
    )

    if lane == 0:
        Tx.copy(left_shared[:, :], left[cta * 64 : (cta + 1) * 64, :])
        Tx.copy(right_shared[:, :], right[cta * 16 : (cta + 1) * 16, :])
    T.cuda.cluster_sync()

    if (cta == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=False,
            cta_group=2,
        )
    T.cuda.cluster_sync()

    if lane == 0:
        for row in T.serial(64):
            for col in T.serial(32):
                output[cta * 64 + row, col] = accumulator[row, col]


@T.prim_func
def raw_cta2_ss_layout_a_mma(
    left: T.Buffer((256, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((256, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (128, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("A", 128, 32),
        allocated_addr=0,
    )
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if lane == 0:
        Tx.copy(left_shared[:, :], left[cta * 128 : (cta + 1) * 128, :])
        Tx.copy(right_shared[:, :], right[cta * 16 : (cta + 1) * 16, :])
    T.cuda.cluster_sync()

    if (cta == 0) and (lane == 0):
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=256,
            N=32,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(left_shared[0, 0]), ldo=16, sdo=16, swizzle=1
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(right_shared[0, 0]), ldo=16, sdo=16, swizzle=1
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cluster_sync()

    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(32):
                output[cta * 128 + row, col] = accumulator[row, col]


@T.prim_func
def typed_cta2_ss_layout_a_gemm_async(
    left: T.Buffer((256, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((256, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (128, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("A", 128, 32),
        allocated_addr=0,
    )

    if lane == 0:
        Tx.copy(left_shared[:, :], left[cta * 128 : (cta + 1) * 128, :])
        Tx.copy(right_shared[:, :], right[cta * 16 : (cta + 1) * 16, :])
    T.cuda.cluster_sync()

    if (cta == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=False,
            cta_group=2,
        )
    T.cuda.cluster_sync()

    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(32):
                output[cta * 128 + row, col] = accumulator[row, col]


@T.prim_func
def raw_cta2_ts_layout_b_mma(
    left: T.Buffer((2, 2, 64, 16), "float16"),
    right: T.Buffer((2, 16, 16), "float16"),
    output: T.Buffer((2, 64, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    left_tmem = T.decl_buffer(
        (2, 64, 16),
        "float16",
        scope="tmem",
        layout=_TMEM_A_CTA2_M128_BANKED_F16,
        allocated_addr=0,
    )
    accumulator = T.decl_buffer(
        (64, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("B", 64, 32),
        allocated_addr=8,
    )
    desc_i: T.uint32
    desc_b: T.uint64

    if lane == 0:
        Tx.copy(right_shared[:, :], right[cta, :, :])
        for bank in T.serial(2):
            for row in T.serial(64):
                for col in T.serial(16):
                    left_tmem[bank, row, col] = left[cta, bank, row, col]
    T.cuda.cluster_sync()

    if (cta == 0) and (lane == 0):
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=128,
            N=32,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(right_shared[0, 0]), ldo=16, sdo=16, swizzle=1
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
            T.uint32(8),
            T.uint32(0),
            desc_b,
            desc_i,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cluster_sync()

    if lane == 0:
        for row in T.serial(64):
            for col in T.serial(32):
                output[cta, row, col] = accumulator[row, col]


@T.prim_func
def typed_cta2_ts_layout_b_gemm_async(
    left: T.Buffer((2, 2, 64, 16), "float16"),
    right: T.Buffer((2, 16, 16), "float16"),
    output: T.Buffer((2, 64, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    left_tmem = T.decl_buffer(
        (2, 64, 16),
        "float16",
        scope="tmem",
        layout=_TMEM_A_CTA2_M128_BANKED_F16,
        allocated_addr=0,
    )
    accumulator = T.decl_buffer(
        (64, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("B", 64, 32),
        allocated_addr=8,
    )

    if lane == 0:
        Tx.copy(right_shared[:, :], right[cta, :, :])
        for bank in T.serial(2):
            for row in T.serial(64):
                for col in T.serial(16):
                    left_tmem[bank, row, col] = left[cta, bank, row, col]
    T.cuda.cluster_sync()

    if (cta == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_tmem[:, :, :],
            right_shared[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=2,
        )
    T.cuda.cluster_sync()

    if lane == 0:
        for row in T.serial(64):
            for col in T.serial(32):
                output[cta, row, col] = accumulator[row, col]


@T.prim_func
def raw_cta2_ts_mma(
    left: T.Buffer((256, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((256, 32), "float32"),
):
    """`M=256` CTA-pair MMA whose A operand is each CTA's own TMEM shard."""

    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    left_tmem = T.decl_buffer(
        (128, 16),
        "float16",
        scope="tmem",
        layout=tmem_datapath_layout("A", 128, 16),
        allocated_addr=0,
    )
    accumulator = T.decl_buffer(
        (128, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("A", 128, 32),
        allocated_addr=8,
    )
    desc_i: T.uint32
    desc_b: T.uint64

    if lane == 0:
        Tx.copy(right_shared[:, :], right[cta * 16 : (cta + 1) * 16, :])
        for row in T.serial(128):
            for col in T.serial(16):
                left_tmem[row, col] = left[cta * 128 + row, col]
    T.cuda.cluster_sync()

    if (cta == 0) and (lane == 0):
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=256,
            N=32,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(right_shared[0, 0]), ldo=16, sdo=16, swizzle=1
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
            T.uint32(8),
            T.uint32(0),
            desc_b,
            desc_i,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cluster_sync()

    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(32):
                output[cta * 128 + row, col] = accumulator[row, col]


@T.prim_func
def typed_cta2_ts_gemm_async(
    left: T.Buffer((256, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((256, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    left_tmem = T.decl_buffer(
        (128, 16),
        "float16",
        scope="tmem",
        layout=tmem_datapath_layout("A", 128, 16),
        allocated_addr=0,
    )
    accumulator = T.decl_buffer(
        (128, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("A", 128, 32),
        allocated_addr=8,
    )

    if lane == 0:
        Tx.copy(right_shared[:, :], right[cta * 16 : (cta + 1) * 16, :])
        for row in T.serial(128):
            for col in T.serial(16):
                left_tmem[row, col] = left[cta * 128 + row, col]
    T.cuda.cluster_sync()

    if (cta == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_tmem[:, :],
            right_shared[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=2,
        )
    T.cuda.cluster_sync()

    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(32):
                output[cta * 128 + row, col] = accumulator[row, col]



def _first_stop(error: v2.ExecutionError) -> dict:
    stops = [
        d
        for d in error.diagnostics
        if d.get("status") in ("error", "incomplete") and d.get("reason") != "subset_execution"
    ]
    assert stops, error.diagnostics
    return stops[0]


def _assert_tmem_subpartition_bad_address(kernel, inputs, buffer: str) -> None:
    """Row T4: ``ExecutionError`` stopping on an ``error`` of kind ``bad_address``
    for a TMEM lane outside warp 0's sub-partition."""

    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(v2.transpile(kernel), inputs)
    stop = _first_stop(excinfo.value)
    assert stop["status"] == "error", stop
    assert stop["kind"] == "bad_address", stop
    message = str(stop.get("message", ""))
    assert f"{buffer}[" in message, stop
    assert "outside warp 0's sub-partition" in message, stop


@pytest.mark.parametrize(
    ("raw_kernel", "typed_kernel", "joint_m", "raw_buffer", "typed_buffer"),
    [
        (raw_cta2_ss_layout_b_mma, typed_cta2_ss_layout_b_gemm_async, 128, "accumulator", "accumulator"),
        (raw_cta2_ss_layout_a_mma, typed_cta2_ss_layout_a_gemm_async, 256, "accumulator", "accumulator"),
        (raw_cta2_ts_mma, typed_cta2_ts_gemm_async, 256, "left_tmem", "left_tmem"),
    ],
    ids=[
        "ss_m128_n32_layout_b",
        "ss_m256_n32_layout_a",
        "ts_m256_n32_layout_a",
    ],
)
def test_raw_cta2_mma_matches_the_typed_gemm_async_exactly(
    raw_kernel, typed_kernel, joint_m, raw_buffer, typed_buffer
):
    """Replaces ``tests/numsim/integration/test_raw_versus_typed_cta2_mma.py::test_raw_cta2_mma_matches_the_typed_gemm_async_exactly``
    (all 3 params). Delta row T4: both kernels fail with ``bad_address``."""

    rng = np.random.default_rng(20260807)
    left = rng.integers(-4, 5, size=(joint_m, 16)).astype(np.float16)
    right = rng.integers(-3, 4, size=(32, 16)).astype(np.float16)

    for kernel, buffer in ((raw_kernel, raw_buffer), (typed_kernel, typed_buffer)):
        _assert_tmem_subpartition_bad_address(
            kernel,
            {"left": left, "right": right, "output": np.zeros((joint_m, 32), dtype=np.float32)},
            buffer,
        )


def test_raw_cta2_ts_m128_selects_the_matching_a_lane_bank():
    """Replaces ``tests/numsim/integration/test_raw_versus_typed_cta2_mma.py::test_raw_cta2_ts_m128_selects_the_matching_a_lane_bank``.
    Delta row T4: warp 0 writes ``left_tmem`` bank 1 (TMEM lanes 64..127)."""

    rng = np.random.default_rng(20260811)
    left = rng.integers(-4, 5, size=(2, 2, 64, 16)).astype(np.float16)
    right = rng.integers(-3, 4, size=(2, 16, 16)).astype(np.float16)
    for kernel in (raw_cta2_ts_layout_b_mma, typed_cta2_ts_layout_b_gemm_async):
        _assert_tmem_subpartition_bad_address(
            kernel,
            {"left": left, "right": right, "output": np.zeros((2, 64, 32), dtype=np.float32)},
            "left_tmem",
        )
